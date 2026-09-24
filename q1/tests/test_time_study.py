import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
import torch

from q1.analysis.time_study.anchors import intervals_for
from q1.analysis.time_study.data import merge, intersection_length, gaps, length
from q1.analysis.time_study.review import score_annotations
from q1.analysis.time_study.sampling import grid_indices
from q1.analysis.time_study.soft_model import (DIMENSIONS, MODES, NativeAlignment,
    attention_weights, fit_normalizers, prepare_batch)
from q1.analysis.time_study.validate import check_mapping


class TimeMapTests(unittest.TestCase):
    def fixture(self):
        source={"intervals":np.array([[0.,.5],[.5,1.]]),"quality":np.array([1.,.5]),"x":np.array([[1.],[3.]])}
        z={"timestamps":np.array([[0.,1.]]),"time_valid_mask":np.array([True]),"word_indices":np.array([0]),
           "source_vision_intervals":source["intervals"].copy(),"map_vision_offsets":np.array([0,2]),
           "map_vision_indices":np.array([0,1]),"map_vision_weights":np.array([2/3,1/3])}
        return z,source

    def test_reconstruct_quality_weighted_feature(self):
        z,s=self.fixture(); errors,mu,std=check_mapping(z,"vision",s)
        self.assertEqual(errors,[]); np.testing.assert_allclose(mu,[[5/3]])
        np.testing.assert_allclose(std,[[np.sqrt(8/9)]])

    def test_negative_source_indices_rejected(self):
        z,s=self.fixture(); z["map_vision_indices"][0]=-1
        self.assertIn("source_index_out_of_range",check_mapping(z,"vision",s)[0])

    def test_wrong_weights_and_invalid_time_mapping_rejected(self):
        z,s=self.fixture(); z["map_vision_weights"][:]=.5
        self.assertIn("row_0_incorrect_weights",check_mapping(z,"vision",s)[0])
        z["time_valid_mask"][0]=False
        self.assertIn("row_0_incorrect_source_set",check_mapping(z,"vision",s)[0])

    def test_union_coverage_counts_overlap_once(self):
        a=merge([[0,1],[.5,2],[3,3.5]])
        self.assertEqual(length(a),2.5); self.assertEqual(gaps(a,4),[[2.,3.],[3.5,4]])
        self.assertEqual(intersection_length(a,[[1,3.25]]),1.25)

    def test_nearest_pts_grid_has_no_cumulative_drift(self):
        for native in (23.976,25.,29.97):
            pts=np.arange(240)/native+.037
            for rate in (5,8,10,12):
                idx=grid_indices(pts,rate); grid=pts[0]+np.arange(len(idx))/rate
                self.assertTrue(np.all(np.diff(idx)>0))
                self.assertLessEqual(float(np.max(np.abs(pts[idx]-grid))),.5/native+1e-9)
                effective=(len(idx)-1)/(pts[idx[-1]]-pts[idx[0]])
                self.assertLess(abs(effective-rate),.08)

    def test_sampling_above_native_rate_never_synthesizes_frames(self):
        pts=np.arange(24)/24
        idx=grid_indices(pts,60)
        np.testing.assert_array_equal(idx,np.arange(24))

    def test_fixed_anchors_cover_exact_duration(self):
        s={"media":{"duration":1.13}}
        for mode in ("fixed_0.25s","fixed_0.5s","fixed_1.0s","uniform_50"):
            a,mask=intervals_for(s,mode)
            self.assertEqual(a[0,0],0); self.assertEqual(a[-1,1],1.13)
            np.testing.assert_array_equal(a[1:,0],a[:-1,1]); self.assertTrue(mask.all())
            self.assertTrue((a[:,1]>a[:,0]).all())


class NativeAttentionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(2)

    def test_local_mask_blocks_even_a_large_distant_logit(self):
        logits=torch.tensor([[[0.,0.,1000.]]]); dist=torch.tensor([[[0.,.4,4.]]])
        overlap=torch.tensor([[[.1,0.,0.]]]); q=torch.ones(1,3); am=torch.ones(1,1,dtype=torch.bool)
        for mode in ("time_kernel","local_attention","local_no_penalty"):
            w=attention_weights(mode,logits,dist,overlap,q,am)
            self.assertEqual(w[0,0,2],0); self.assertAlmostEqual(float(w.detach().sum()),1.)
        w=attention_weights("local_attention",logits,dist,overlap,q,am)
        self.assertGreater(w[0,0,0],w[0,0,1])
        self.assertGreater(attention_weights("global_attention",logits,dist,overlap,q,am)[0,0,2],.99)

    def test_all_missing_attention_zero_and_backward_finite(self):
        for mode in MODES:
            logits=torch.ones((1,2,3),requires_grad=True)
            w=attention_weights(mode,logits,torch.zeros_like(logits),torch.ones_like(logits),torch.zeros(1,3),torch.ones(1,2,dtype=torch.bool))
            self.assertTrue(torch.isfinite(w).all()); self.assertEqual(float(w.detach().sum()),0.)
            if w.requires_grad:
                w.sum().backward(); self.assertTrue(torch.isfinite(logits.grad).all())

    def samples(self):
        result=[]
        for j in range(3):
            s={"media":{"duration":1.}}
            for m,d in DIMENSIONS.items():
                s[m]={"x":np.full((2,d),float(j)),"intervals":np.array([[0.,.5],[.5,1.]]),"quality":np.ones(2)}
            s["text"]["content_valid"]=np.ones(2,bool); result.append(s)
        return result

    def test_scaler_does_not_read_heldout_features(self):
        samples=self.samples(); a=fit_normalizers(samples,[0,1])
        samples[2]["vision"]["x"][:]=1e10
        b=fit_normalizers(samples,[0,1])
        np.testing.assert_array_equal(a["vision"]["mean"],b["vision"]["mean"])
        np.testing.assert_allclose(a["vision"]["mean"],.5)

    def test_invalid_text_time_preserves_content_missing_av_stays_masked(self):
        samples=self.samples(); norm=fit_normalizers(samples,[0,1])
        for s in samples:
            for m in DIMENSIONS:s[m]["quality"][:]=0
        batch=prepare_batch(samples,[0,1],norm,"cpu")
        self.assertTrue(batch["text"]["content"].all()); self.assertFalse((batch["text"]["quality"]>0).any())
        for mode in MODES:
            model=NativeAlignment(mode); reg,cls,maps=model(batch,True)
            self.assertTrue(torch.isfinite(reg).all()); self.assertTrue(torch.isfinite(cls).all())
            self.assertTrue(all(float(w.detach().sum())==0 for w in maps.values()))
            (reg.square().sum()+cls.square().sum()).backward()
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))


class HumanAnnotationTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.cfg={"output_dir":self.tmp.name}; self.out=Path(self.tmp.name)/"analysis/time_study/review"; self.out.mkdir(parents=True)
        self.row={"sample_id":"clipA","word_index":0,"auto_start":.1,"auto_end":.8,"auto_valid":True,
                  "manual_start":np.nan,"manual_end":np.nan,"reviewer_code":""}
        pd.DataFrame([self.row]).to_csv(self.out/"word_annotations_template.csv",index=False)
        (self.out/"cases.json").write_text(json.dumps([{"sample_id":"clipA","duration":2.}]))

    def score(self,row):
        path=self.out/"annotations.csv"; pd.DataFrame([row]).to_csv(path,index=False)
        return score_annotations(path,self.cfg)

    def test_blank_annotations_do_not_create_ground_truth(self):
        r=self.score(self.row); self.assertEqual(r["annotated_words"],0); self.assertIsNone(r["boundary_mae_seconds"])

    def test_unknown_partial_and_unattributed_annotations_rejected(self):
        for changes in ({"sample_id":"unknown"},{"manual_start":.2},{"manual_start":.2,"manual_end":.9,"reviewer_code":"   "},
                        {"manual_start":.2,"manual_end":3.,"reviewer_code":"R1"}):
            with self.subTest(changes=changes), self.assertRaises(ValueError): self.score({**self.row,**changes})

    def test_boundary_score_uses_reference_auto_times(self):
        r=self.score({**self.row,"auto_start":99.,"auto_end":100.,"manual_start":.15,"manual_end":.85,"reviewer_code":"R1"})
        self.assertEqual(r["annotated_words"],1); self.assertAlmostEqual(r["boundary_mae_seconds"],.05)


if __name__=="__main__": unittest.main()

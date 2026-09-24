"""Independent checks of experiment artifacts, splits and exported maps."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, f1_score, accuracy_score

from ...common import atomic_json, file_hash, read_json
from .data import default_config, load_samples, output_dir
from .soft_model import NativeAlignment, fit_normalizers, prepare_batch
from .validate import check_mapping


def run(cfg=None):
    cfg=cfg or default_config(); root=output_dir(cfg,"")
    samples=load_samples(cfg); ids=[s["id"] for s in samples]
    source={s["id"]:s for s in samples}; y=np.array([s["label"] for s in samples])
    groups=np.array([s["video_id"] for s in samples])
    checks={}
    for name in ("validation","sampling","anchors","soft","review"):
        r=read_json(root/name/"report.json")
        assert all(file_hash(p)==h for p,h in r["provenance"]["code_files"].items()),name+" stale provenance"
        assert file_hash(Path(cfg["output_dir"])/"manifest.json")==r["provenance"]["manifest_sha256"]
        checks[name+"_current_code_and_manifest"]=True
    # Every alternative anchor view must reconstruct its exported values from native rows.
    views=0
    for p in (root/"anchors"/"features").glob("*/*.npz"):
        s=next(s for s in samples if s["key"]==p.stem)
        with np.load(p) as z:
            reconstructed={}
            for m in ("emotion","acoustic","vision"):
                errors,mu,std=check_mapping(z,m,s[m]); assert not errors,(str(p),m,errors)
                reconstructed[m]=(mu,std)
            e,a,v=(reconstructed[m] for m in ("emotion","acoustic","vision"))
            np.testing.assert_array_equal(np.concatenate([e[0],a[0],a[1]],axis=1).astype(np.float16),z["audio"])
            np.testing.assert_array_equal(np.concatenate(v,axis=1).astype(np.float16),z["vision"])
            if p.parent.name!="word_gaps":
                errors,mu,_=check_mapping(z,"text",s["text"]); assert not errors
                np.testing.assert_array_equal(mu.astype(np.float16),z["text"])
                assert z["timestamps"][0,0]==0 and z["timestamps"][-1,1]==s["media"]["duration"]
        views+=1
    assert views==500; checks["alternative_views_reconstructed"]=views
    # Independent sklearn metrics and explicit group separation for all anchor readouts.
    anchor=read_json(root/"anchors"/"report.json"); pred=pd.read_csv(root/"anchors"/"predictions.csv")
    assert pred.sample_id.tolist()==ids
    anchor_folds=read_json(root/"anchors"/"folds.json")
    for name,r in anchor["results"].items():
        pr=pred[name+"__regression"].to_numpy(); pc=pred[name+"__class"].to_numpy()
        np.testing.assert_allclose(mean_absolute_error(y,pr),r["mae"])
        np.testing.assert_allclose(f1_score(np.sign(y),pc,labels=[-1,0,1],average="macro",zero_division=0),r["macro_f1_3"])
        coverage=np.zeros(len(samples),int)
        for f in anchor_folds[name]:
            tr,te=f["train_indices"],f["test_indices"]; assert not set(groups[tr])&set(groups[te]); coverage[te]+=1
            for inner in f["inner_folds"]:
                a,b=inner["train_indices"],inner["valid_indices"]
                assert set(a+b).issubset(set(tr)) and not set(groups[a])&set(groups[b])
        assert (coverage==1).all()
    checks["anchor_metrics_and_nested_group_splits"]=len(anchor["results"])
    # Native-PTS identities remain exact for all sampling caches.
    sampled=0
    for p in (root/"sampling"/"cache").glob("*/*/vision.npz"):
        s=next(s for s in samples if s["key"]==p.parent.parent.name)
        with np.load(p) as z, np.load(s["directory"]/"frames.npz") as raw:
            np.testing.assert_array_equal(z["pts"],raw["pts"][z["frame_indices"]])
        assert read_json(p.with_name("done.json"))["sha256"]==file_hash(p); sampled+=1
    assert sampled==40; checks["sampling_caches_with_original_pts"]=sampled
    # Independently replay every saved neural checkpoint; no optimization is repeated.
    torch.set_num_threads(2); device="cuda:0" if torch.cuda.is_available() else "cpu"
    soft=read_json(root/"soft"/"report.json"); folds=read_json(root/"soft"/"folds.json")
    table=pd.read_csv(root/"soft"/"predictions.csv"); replayed=0
    for f in folds:
        tr,te=f["train_indices"],f["test_indices"]; assert not set(groups[tr])&set(groups[te])
        norm=fit_normalizers(samples,tr); batch=prepare_batch(samples,te,norm,device)
        for mode in soft["protocol"]["modes"]:
            for seed in soft["protocol"]["seeds"]:
                p=root/"soft"/"checkpoints"/mode/f"fold{f['fold']}_seed{seed}"/"model.pt"
                marker=read_json(p.with_name("done.json")); assert marker["model_sha256"]==file_hash(p)
                assert marker["prediction_sha256"]==file_hash(p.with_name("predictions.npz"))
                saved=torch.load(p,map_location="cpu",weights_only=False)
                np.testing.assert_array_equal(saved["train_indices"],tr); np.testing.assert_array_equal(saved["test_indices"],te)
                for m in norm:
                    for field in ("mean","scale"): np.testing.assert_array_equal(saved["normalizers"][m][field],norm[m][field])
                model=NativeAlignment(mode).to(device); model.load_state_dict(saved["state_dict"]); model.eval()
                with torch.no_grad():
                    pr,pc=model(batch); pr=pr.cpu().numpy()*saved["label_std"]+saved["label_mean"]
                    prob=pc.softmax(-1).cpu().numpy()
                with np.load(p.with_name("predictions.npz")) as z:
                    np.testing.assert_allclose(pr,z["regression"],atol=1e-6,rtol=1e-6)
                    np.testing.assert_allclose(prob,z["probabilities"],atol=1e-6,rtol=1e-6)
                replayed+=1
    for mode,r in soft["results"].items():
        for j,seed in enumerate(soft["protocol"]["seeds"]):
            part=table[(table["mode"]==mode)&(table.seed==seed)].set_index("sample_id").loc[ids]
            np.testing.assert_allclose(mean_absolute_error(y,part.regression),r["mae"]["per_seed"][j])
            np.testing.assert_allclose(accuracy_score(np.sign(y),part["class"]),r["accuracy3"]["per_seed"][j])
            np.testing.assert_allclose(f1_score(np.sign(y),part["class"],labels=[-1,0,1],average="macro",zero_division=0),r["macro_f1_3"]["per_seed"][j])
    checks["neural_checkpoints_replayed"]=replayed
    attention=0
    for p in (root/"soft"/"attention").glob("*/*.npz"):
        meta=read_json(p.with_suffix(".json")); s=source[meta["sample_id"]]
        assert file_hash(p)==meta["weights_sha256"] and meta["held_out_index"] not in meta["training_indices"]
        with np.load(p) as z:
            for m in ("emotion","acoustic","vision"):
                w=z[m+"_weights"]; np.testing.assert_array_equal(z[m+"_source_intervals"],s[m]["intervals"])
                assert np.isfinite(w).all() and (w>=0).all()
                assert (w[:,s[m]["quality"]==0]==0).all()
                sums=w.sum(1); assert np.all(np.isclose(sums,0,atol=1e-6)|np.isclose(sums,1,atol=1e-6))
                if meta["mode"] in ("time_kernel","local_attention","local_no_penalty"):
                    distances=np.abs(z["anchor_times"].mean(1)[:,None]-z[m+"_source_intervals"].mean(1)[None,:])
                    assert (w[distances>.5+1e-6]==0).all()
            np.testing.assert_array_equal(z["vision_original_pts"],s["vision"]["pts"])
        attention+=1
    checks["heldout_attention_maps_checked"]=attention; assert attention==25
    annotations=pd.read_csv(root/"review"/"word_annotations_template.csv")
    assert annotations[["manual_start","manual_end","reviewer_code"]].isna().all().all()
    cases=read_json(root/"review"/"cases.json"); assert len(cases)==18
    excluded=0
    for c in cases:
        with np.load(root/"review"/c["quality_sidecar"]) as z:
            np.testing.assert_array_equal(z["usable_after_review_mask"],(z["original_quality"]>0)&~z["review_excluded_mask"])
            excluded+=int(z["review_excluded_mask"].sum())
    assert excluded==8
    checks["human_words_fabricated"]=0; checks["evidence_flagged_false_face_frames"]=excluded
    result={"passed":True,"checks":checks,"verifier_sha256":file_hash(__file__)}
    atomic_json(root/"verification.json",result); print(json.dumps(result,ensure_ascii=False,indent=2))
    return result


if __name__=="__main__": run()

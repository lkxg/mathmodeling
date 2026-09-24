"""Native-PTS sampling-rate perturbations on eight preselected diagnostic clips."""
from __future__ import annotations
import copy
import json
import logging
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from ...backends import VisionEncoder, box_iou
from ...common import atomic_json, atomic_npz, digest, file_hash, read_json
from ...temporal import interval_pool, sampled_support
from ..probe import aligned_blocks, clean_json, design, CONFIGS
from . import data
from .data import default_config, load_samples, output_dir, provenance

LOG = logging.getLogger(__name__)
RATES = (5., 8., 10., 12., 25.)


def grid_indices(pts, fps):
    """Nearest original frame to a fixed seconds grid, never synthesize frames."""
    pts = np.asarray(pts)
    requested = pts[0] + np.arange(int(np.floor((pts[-1] - pts[0]) * fps)) + 1) / fps
    right = np.minimum(np.searchsorted(pts, requested), len(pts) - 1)
    left = np.maximum(right - 1, 0)
    selected = np.where(abs(pts[left] - requested) <= abs(pts[right] - requested), left, right)
    return np.unique(selected)


def select_samples(samples, vad):
    chosen, used = [], set()
    def add(s, reason):
        if s["id"] not in used:
            chosen.append((s, reason)); used.add(s["id"])
    add(samples[0], "first_manifest_normal_reference")
    for s in samples:
        if s["media"]["silent_audio"]:
            add(s, "digital_silence")
    noface = [s for s in samples if not np.any(s["vision"]["quality"] > 0) and not s["media"]["silent_audio"]]
    add(noface[0], "no_detected_face")
    normal = [s for s in samples if not s["meta"]["summary"]["alignment_suspect"] and np.any(s["vision"]["quality"] > 0)]
    add(min(normal, key=lambda s:s["media"]["duration"]), "short_normal")
    add(max(samples, key=lambda s:s["media"]["duration"]), "longest_clip")
    scored = vad.dropna(subset=["recall"]).sort_values(["recall", "sample_id"])
    lookup = {s["id"]:s for s in samples}
    for sid in scored.sample_id:
        if sid not in used:
            add(lookup[sid], "lowest_vad_recall"); break
    def motion(s):
        v=s["vision"]; x=v["x"][v["quality"]>0, -10:]
        return float(np.median(np.linalg.norm(np.diff(x, axis=0), axis=1))) if len(x)>2 else -1
    for s in sorted(samples, key=motion, reverse=True):
        if s["id"] not in used:
            add(s, "largest_median_landmark_change"); break
    return chosen


def extract(backend, s, fps, cfg):
    import cv2
    import torch
    from ...common import read_json
    with np.load(s["directory"] / "frames.npz", allow_pickle=False) as z:
        pts = z["pts"]
    selected = grid_indices(pts, fps)
    want = set(selected.tolist())
    cap = cv2.VideoCapture(s["video"])
    if not cap.isOpened():
        raise IOError(f"Cannot decode {s['id']}")
    features, qualities, records = [], [], []
    previous, count = None, 0
    try:
        with tempfile.TemporaryDirectory(prefix="q1-sampling-") as d, torch.inference_mode():
            image_path = str(Path(d)/"frame.png")
            while True:
                ok, image = cap.read()
                if not ok: break
                i=count; count+=1
                if i not in want: continue
                h,w=image.shape[:2]; scale=min(1.,cfg["vision"]["max_side"]/max(h,w))
                small=cv2.resize(image,(round(w*scale),round(h*scale))) if scale<1 else image
                if not cv2.imwrite(image_path,small): raise IOError("Cannot write frame")
                dets,_=backend.detector.detect_faces(image_path)
                dets=np.asarray(dets)
                dets=dets[dets[:,4]>=cfg["vision"]["face_threshold"]]
                vector=np.zeros(28,np.float32); quality=0.; flags=[]; bbox_record=None; reset=False
                if len(dets):
                    pick=int(np.argmax(np.maximum(0,dets[:,2:4]-dets[:,:2]).prod(axis=1)))
                    if previous is not None:
                        ious=box_iou(previous,dets[:,:4])
                        if ious.max()>=cfg["vision"]["track_iou_threshold"]: pick=int(ious.argmax())
                        else: reset=True
                    det=dets[pick]; box=det[:4].copy()
                    box[[0,2]]=np.clip(box[[0,2]],0,small.shape[1]); box[[1,3]]=np.clip(box[[1,3]],0,small.shape[0])
                    l,t,r,b=box.astype(int)
                    if r>l and b>t:
                        emotion,gaze,au=backend.predictor.predict(small[t:b,l:r])
                        landmarks=(det[5:15].reshape(5,2)-box[:2])/np.maximum(box[2:]-box[:2],1)
                        vector=np.concatenate([x.detach().float().cpu().numpy().ravel() for x in (emotion,gaze,au)]+[landmarks.ravel()]).astype(np.float32)
                        if np.isfinite(vector).all(): quality=float(det[4])
                        else: flags.append("nonfinite_prediction"); vector[:]=0
                        previous=box; bbox_record=(box/scale).tolist()
                    else: flags.append("invalid_bbox")
                else: flags.append("no_face"); previous=None
                features.append(vector); qualities.append(quality)
                records.append({"frame_index":i,"pts":float(pts[i]),"flags":flags,"bbox":bbox_record,"track_reset":reset})
    finally:
        cap.release()
    if count!=len(pts) or len(features)!=len(selected): raise ValueError("Decoded frame/PTS mismatch")
    return {"features":np.asarray(features),"quality":np.asarray(qualities),
            "pts":pts[selected],"frame_indices":selected,
            "intervals":sampled_support(pts[selected],s["media"]["video_start"],s["media"]["video_end"])}, records


def aligned_vision(s, v):
    target=s["export"]["timestamps"].copy()
    target[~s["export"]["time_valid_mask"],1]=target[~s["export"]["time_valid_mask"],0]
    mean,std,valid,cov,mapping=interval_pool(target,v["intervals"],v["features"],v["quality"])
    features=np.concatenate([mean,std],axis=1).astype(np.float16)
    centroid=np.full(len(target),np.nan)
    for k,row in enumerate(mapping):
        if row["indices"]:
            centroid[k]=np.sum(v["pts"][row["indices"]]*np.asarray(row["weights"]))
    z={**s["export"],"vision":features,"component_mask":s["export"]["component_mask"].copy()}
    z["component_mask"][:,3]=valid
    return features,valid,cov,centroid,aligned_blocks(z)


def predictor(samples, cfg):
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler, FunctionTransformer
    from sklearn.pipeline import make_pipeline
    from sklearn.linear_model import Ridge, RidgeClassifier
    root=Path(cfg["output_dir"])/"analysis"/"probe_v2"
    with np.load(root/"blocks.npz",allow_pickle=False) as z: blocks={k:z[k] for k in z.files}
    assert read_json(root/"blocks.json")["sample_ids"]==[s["id"] for s in samples]
    x,weights=design(blocks,CONFIGS["fusion_aligned"])
    y=np.asarray([s["label"] for s in samples])
    folds=read_json(root/"folds.json")["fusion_aligned"]
    models={}
    def predict(s, block):
        if s["video_id"] not in models:
            f=next(f for f in folds if f["test_group"]==s["video_id"]); tr=f["train_indices"]
            heads=[Ridge(alpha=f["regression_alpha"],solver="svd"),
                   RidgeClassifier(alpha=f["classification_alpha"],solver="svd")]
            fitted=[]
            for head,target in zip(heads,(y,np.sign(y))):
                model=make_pipeline(SimpleImputer(keep_empty_features=True),StandardScaler(),
                                    FunctionTransformer(lambda z:z*weights),head)
                fitted.append(model.fit(x[tr],target[tr]))
            models[s["video_id"]]=fitted
        xx,_=design({k:np.asarray(v)[None,:] for k,v in block.items()},CONFIGS["fusion_aligned"])
        a,b=models[s["video_id"]]
        return float(a.predict(xx)[0]),int(b.predict(xx)[0])
    return predict


def run(cfg=None):
    import cv2, torch
    cfg=cfg or default_config()
    torch.set_num_threads(2); cv2.setNumThreads(1)
    out=output_dir(cfg,"sampling")
    samples=load_samples(cfg)
    vad=pd.read_csv(Path(cfg["output_dir"])/"analysis"/"alignment_vad.csv")
    selected=select_samples(samples,vad)
    protocol={"rates":RATES,"selection":[{"sample_id":s["id"],"reason":reason} for s,reason in selected],
              "selection_uses_sentiment_labels":False,"sampler":"fixed seconds grid -> nearest original PTS; deduplicate",
              "legacy":"original minimum-spacing 10Hz cache is an additional comparator",
              "prediction":"fixed outer-training-fold Ridge heads from probe_v2; no perturbation-based tuning"}
    atomic_json(out/"protocol.json",protocol)
    backend=None; rows=[]
    predict=predictor(samples,cfg)
    for s,reason in selected:
        variants={"legacy10":{k:s["vision"][{"features":"x"}.get(k,k)] for k in
                              ("features","quality","pts","frame_indices","intervals")}}
        with np.load(s["directory"]/"frames.npz",allow_pickle=False) as f: raw_pts=f["pts"]
        for rate in RATES:
            d=out/"cache"/s["key"]/str(int(rate)); d.mkdir(parents=True,exist_ok=True)
            signature=digest({"video":s["video_sha256"],"rate":rate,"vision":cfg["vision"],
                              "face":cfg["models"]["face"],"code":file_hash(__file__),
                              "frames":file_hash(s["directory"]/"frames.npz")})
            marker=d/"done.json"; path=d/"vision.npz"
            saved=read_json(marker) if marker.exists() else {}
            if path.exists() and saved.get("signature")==signature and saved.get("sha256")==file_hash(path):
                with np.load(path,allow_pickle=False) as z: v={k:z[k] for k in z.files}
            else:
                if backend is None: backend=VisionEncoder(cfg)
                LOG.info("%s: grid %.0f Hz (%s)",s["id"],rate,reason)
                v,records=extract(backend,s,rate,cfg)
                atomic_npz(path,**v); atomic_json(d/"frames.json",records,compact=True)
                atomic_json(marker,{"signature":signature,"sha256":file_hash(path)})
            variants[str(int(rate))]=v
        base=aligned_vision(s,variants["10"]); bp,bc=predict(s,base[4])
        for name,v in variants.items():
            f,mask,cov,centroid,block=aligned_vision(s,v)
            both=mask&base[1]&s["export"]["time_valid_mask"]
            finite=np.isfinite(centroid)&np.isfinite(base[3])
            relative=float(np.linalg.norm(f[both].astype(float)-base[0][both])/max(np.linalg.norm(base[0][both].astype(float)),1e-12)) if np.any(both) else None
            pr,pc=predict(s,block)
            pts_error=float(np.max(abs(v["pts"]-raw_pts[v["frame_indices"]])))
            if pts_error>1e-10: raise AssertionError("Original PTS identity lost")
            rows.append({"sample_id":s["id"],"reason":reason,"rate":name,"frames":len(v["pts"]),
                         "effective_hz":(len(v["pts"])-1)/(v["pts"][-1]-v["pts"][0]) if len(v["pts"])>1 else 0.,
                         "pts_identity_error_seconds":pts_error,"face_valid_ratio":float(np.mean(v["quality"]>0)),
                         "valid_target_ratio":float(mask.mean()),"mask_changes_vs_grid10":int((mask!=base[1]).sum()),
                         "relative_feature_change":relative,
                         "mean_abs_centroid_shift_seconds":float(np.mean(abs(centroid[finite]-base[3][finite]))) if finite.any() else None,
                         "mean_signed_centroid_shift_seconds":float(np.mean(centroid[finite]-base[3][finite])) if finite.any() else None,
                         "mean_coverage":float(cov.mean()),"regression_delta_vs_grid10":pr-bp,"class_flip_vs_grid10":pc!=bc})
    table=pd.DataFrame(rows); table.to_csv(out/"samples.csv",index=False,encoding="utf-8-sig")
    numeric=["effective_hz","relative_feature_change","mean_abs_centroid_shift_seconds","mean_coverage"]
    summary={str(k):{**{c:float(g[c].median()) for c in numeric},"max_pts_error_seconds":float(g.pts_identity_error_seconds.max()),
                         "mean_abs_prediction_change":float(g.regression_delta_vs_grid10.abs().mean()),
                         "class_flips":int(g.class_flip_vs_grid10.sum())}
             for k,g in table.groupby("rate",sort=False)}
    report={"protocol":protocol,"samples":len(selected),"passed_pts_identity":bool((table.pts_identity_error_seconds<=1e-10).all()),
            "summary":summary,"provenance":provenance(cfg,[__file__,data.__file__]),
            "limitations":["仅8条预先选择的诊断样本；不能据此宣称100条全部稳健。",
                           "频率高于原视频帧率时只保留可用原帧，不生成插值帧。",
                           "不同采样率改变人脸跟踪历史和窗口近似，特征不要求逐值相同。",
                           "秒级坐标身份应保持精确；邻近帧代表时间的变化不是全局时钟漂移。"]}
    atomic_json(out/"report.json",clean_json(report))
    print(json.dumps(clean_json(report),ensure_ascii=False,indent=2))
    return report


if __name__=="__main__":
    logging.basicConfig(level=logging.INFO,format="%(asctime)s %(levelname)s %(message)s")
    run()

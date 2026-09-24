"""Native feature pools under word, fixed-duration and fixed-count anchors."""
from __future__ import annotations
import logging
from pathlib import Path
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from ...common import atomic_json, atomic_npz
from ...pipeline import csr
from ...temporal import interval_pool
from ..probe import (ALPHAS, CONFIGS, baseline_predictions, bootstrap_scores, clean_json,
                      design, interval)
from ..probe_cv import (batch_metrics, cluster_bootstrap_indices, holm, nested_predict,
                         within_group_targets)
from . import data
from .data import default_config, load_samples, mean_valid, output_dir, provenance

LOG=logging.getLogger(__name__)
MODES=("word_gaps","fixed_0.25s","fixed_0.5s","fixed_1.0s","uniform_50")


def intervals_for(s, mode):
    duration=s["media"]["duration"]
    if mode=="word_gaps": return s["export"]["timestamps"].copy(),s["export"]["time_valid_mask"].copy()
    if mode=="uniform_50": edges=np.linspace(0,duration,51)
    else:
        step=float(mode.removeprefix("fixed_").removesuffix("s"))
        edges=np.r_[np.arange(0,duration,step),duration]
    return np.column_stack([edges[:-1],edges[1:]]),np.ones(len(edges)-1,bool)


def pool_view(s, mode):
    if mode=="word_gaps": return {k:v.copy() for k,v in s["export"].items()}
    times,valid=intervals_for(s,mode); pools={}
    for m in data.MODALITIES:
        src=s[m]
        pools[m]=interval_pool(times,src["intervals"],src["x"],src["quality"])
    t,e,a,v=(pools[m] for m in data.MODALITIES)
    result={"text":t[0].astype(np.float16),"audio":np.concatenate([e[0],a[0],a[1]],axis=1).astype(np.float16),
            "vision":np.concatenate([v[0],v[1]],axis=1).astype(np.float16),"timestamps":times,
            "time_valid_mask":valid,"valid_mask":valid.copy(),"word_indices":np.full(len(times),-1,np.int32),
            "length":np.asarray(len(times),np.int32),
            "component_mask":np.column_stack([pools[m][2] for m in data.MODALITIES]),
            "modality_mask":np.column_stack([t[2],e[2]|a[2],v[2]]),
            "coverage":np.column_stack([e[3],a[3],v[3]])}
    for m in data.MODALITIES:
        off,idx,w=csr(pools[m][4])
        result.update({f"source_{m}_intervals":s[m]["intervals"],
                       f"map_{m}_offsets":off,f"map_{m}_indices":idx,f"map_{m}_weights":w})
    return result


def readout(s, view, duration_weighted=False):
    # Hold the full original text content fixed across temporal-anchor comparisons.
    weights=np.maximum(0,np.diff(view["timestamps"],axis=1).ravel()) if duration_weighted else np.ones(len(view["text"]))
    def avg(x,mask):
        w=weights*mask
        return np.sum(x.astype(float)*w[:,None],axis=0)/w.sum() if w.sum()>0 else np.full(x.shape[1],np.nan)
    c,a,v=view["component_mask"],view["audio"],view["vision"]
    return {"text":mean_valid(s["export"]["text"],(s["export"]["word_indices"]>=0)&s["export"]["component_mask"][:,0]),
            "e2v_aligned":avg(a[:,:768],c[:,1]),
            "lld_mean_aligned":avg(a[:,768:793],c[:,2]),"lld_std_aligned":avg(a[:,793:],c[:,2]),
            "vision_mean_aligned":avg(v[:,:28],c[:,3]),"vision_std_aligned":avg(v[:,28:],c[:,3])}


def run(cfg=None):
    cfg=cfg or default_config(); samples=load_samples(cfg); out=output_dir(cfg,"anchors")
    y=np.asarray([s["label"] for s in samples]); groups=np.asarray([s["video_id"] for s in samples])
    designs={}; quality=[]
    for mode in MODES:
        blocks={pool:[] for pool in ("uniform","duration")}
        d=out/"features"/mode; d.mkdir(parents=True,exist_ok=True)
        for s in samples:
            view=pool_view(s,mode)
            atomic_npz(d/(s["key"]+".npz"),**view)
            quality.append({"sample_id":s["id"],"mode":mode,"positions":len(view["text"]),
                            "median_interval_seconds":float(np.median(np.maximum(0,np.diff(view["timestamps"],axis=1)))),
                            "text_valid_ratio":float(view["component_mask"][:,0].mean()),
                            "audio_valid_ratio":float(view["modality_mask"][:,1].mean()),
                            "vision_valid_ratio":float(view["component_mask"][:,3].mean())})
            for pool in blocks: blocks[pool].append(readout(s,view,pool=="duration"))
        for pool,rows in blocks.items():
            b={k:np.asarray([r[k] for r in rows]) for k in rows[0]}
            designs[mode+"__"+pool]=design(b,CONFIGS["fusion_aligned"])
    pd.DataFrame(quality).to_csv(out/"anchor_quality.csv",index=False,encoding="utf-8-sig")
    protocol={"modes":MODES,"clip_readouts":["uniform anchor mean","duration-weighted anchor mean"],
              "full_text_held_constant":True,"input_dimensions":[768,818,56],
              "cv":"37 video-group outer folds, 5 video-group inner folds",
              "permutations":499,"bootstrap":2000,"seed":2026,
              "note":"Different anchors change pooling/statistical scale; this is not a manual boundary-accuracy test."}
    atomic_json(out/"protocol.json",protocol)
    targets=within_group_targets(y,groups,499,np.random.default_rng(2026))
    boot=cluster_bootstrap_indices(groups,2000,np.random.default_rng(2027))
    scores={}; preds={}; draws={}; folds={}
    with threadpool_limits(limits=2):
        for name,(x,w) in designs.items():
            LOG.info("anchor experiment %s",name)
            pr,pc,f=nested_predict(x,targets,groups,w,ALPHAS,5)
            metrics=batch_metrics(targets,pr,pc); r={k:float(v[0]) for k,v in metrics.items()}
            for metric,higher in (("mae",False),("pearson",True),("macro_f1_3",True)):
                v=metrics[metric]; count=np.sum(v[1:]>=v[0]) if higher else np.sum(v[1:]<=v[0])
                r["p_"+metric]=float((1+count)/500)
            draws[name]=bootstrap_scores(y,pr[:,0],pc[:,0],boot)
            r["ci95"]={k:interval(v) for k,v in draws[name].items()}
            scores[name]=r; preds[name]=(pr[:,0],pc[:,0]); folds[name]=f
            atomic_npz(out/(name+"_all_targets.npz"),regression=pr,classification=pc)
            LOG.info("%s MAE=%.4f Macro-F1=%.4f",name,r["mae"],r["macro_f1_3"])
    for metric in ("mae","pearson","macro_f1_3"):
        for name,p in zip(scores,holm([r["p_"+metric] for r in scores.values()])):
            scores[name]["p_"+metric+"_holm"]=float(p)
    comparisons={}
    for pool in ("uniform","duration"):
        base="word_gaps__"+pool
        for mode in MODES[1:]:
            name=mode+"__"+pool
            comparisons[name+" - "+base]={k:v for metric in ("mae","macro_f1_3") for k,v in
                [(metric+"_diff",scores[name][metric]-scores[base][metric]),
                 (metric+"_ci95",interval(draws[name][metric]-draws[base][metric]))]}
    table={"sample_id":[s["id"] for s in samples],"video_id":groups,"label":y}
    for name,(pr,pc) in preds.items(): table[name+"__regression"]=pr; table[name+"__class"]=pc
    pd.DataFrame(table).to_csv(out/"predictions.csv",index=False,encoding="utf-8-sig")
    pd.DataFrame([{"experiment":n,**{k:v for k,v in r.items() if isinstance(v,(int,float))}} for n,r in scores.items()]).to_csv(out/"summary.csv",index=False,encoding="utf-8-sig")
    report={"samples":len(samples),"protocol":protocol,"results":scores,"paired_bootstrap":comparisons,
            "provenance":provenance(cfg,[__file__,data.__file__]),
            "limitations":["All source views are retained; mean pooling is an auxiliary linear readout and discards sequence order.",
                           "Text content is held fixed, so changes are attributable to audio/vision aggregation and their interaction.",
                           "Intervals for invalid word timestamps remain invalid; fixed grids do not repair forced alignment.",
                           "Paired bootstrap intervals are exploratory, without multiple-comparison correction."]}
    atomic_json(out/"report.json",clean_json(report)); atomic_json(out/"folds.json",folds)
    return report


if __name__=="__main__":
    logging.basicConfig(level=logging.INFO,format="%(asctime)s %(levelname)s %(message)s")
    run()

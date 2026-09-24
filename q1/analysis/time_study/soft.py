"""Prespecified five-fold, three-seed exploratory local-alignment experiment."""
from __future__ import annotations
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import logging
import random
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupKFold

from ...common import atomic_json, atomic_npz, digest, file_hash, read_json
from ..probe import bootstrap_scores, clean_json, interval
from ..probe_cv import batch_metrics, cluster_bootstrap_indices
from . import data, soft_model
from .data import default_config, load_samples, output_dir, provenance
from .soft_model import MODES, NativeAlignment, fit_normalizers, prepare_batch

LOG = logging.getLogger(__name__)
PROTOCOL = {"outer_folds":5, "seeds":[2026,2027,2028], "modes":MODES,
            "epochs":80, "optimizer":"AdamW", "learning_rate":.001, "weight_decay":.01,
            "batch":"all training clips in one batch", "latent_dimension":16,
            "anchor_seconds":.5, "local_window_seconds":.5, "time_penalty_sigma_seconds":.25,
            "dropout":.1, "gradient_clip":5.,
            "loss":"train-standardized label MSE + training-class-balanced cross entropy",
            "normalization":"native valid frames, train clips only; text includes time-invalid words",
            "selection":"Fixed before the run; no early stopping, inner selection or test-based tuning",
            "readout":"mean/std of native-encoded attended contexts, full-content text mean, availability flags",
            "bootstrap":2000, "bootstrap_seed":2030}


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def export_attention(path, batch, maps, local_index, sample):
    n = int(batch["anchor_mask"][local_index].sum())
    arrays = {"anchor_times":batch["anchor_times"][local_index,:n].cpu().numpy()}
    for m,w in maps.items():
        size = len(sample[m]["x"]); matrix = w[local_index,:n,:size].cpu().numpy()
        # Dense storage retains every weight and its original native row identity.
        arrays[m+"_weights"] = matrix
        arrays[m+"_source_indices"] = np.arange(size)
        arrays[m+"_source_intervals"] = sample[m]["intervals"]
        arrays[m+"_quality"] = sample[m]["quality"]
        if m == "vision":
            arrays["vision_original_frame_indices"] = sample[m]["frame_indices"]
            arrays["vision_original_pts"] = sample[m]["pts"]
    atomic_npz(path,**arrays)


def run(cfg=None):
    cfg = cfg or default_config(); out = output_dir(cfg,"soft")
    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(2); torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    samples = load_samples(cfg)
    y = np.asarray([s["label"] for s in samples]); groups = np.asarray([s["video_id"] for s in samples])
    folds = list(GroupKFold(n_splits=5).split(y,groups=groups))
    split_rows = [{"fold":j,"train_indices":tr.tolist(),"test_indices":te.tolist(),
                   "train_groups":sorted(set(groups[tr])),"test_groups":sorted(set(groups[te]))} for j,(tr,te) in enumerate(folds)]
    for tr,te in folds:
        assert not set(groups[tr])&set(groups[te])
    prov = provenance(cfg,[__file__,soft_model.__file__,data.__file__])
    signature = digest({"provenance":prov,"protocol":PROTOCOL,"splits":split_rows})
    atomic_json(out/"protocol.json",{**PROTOCOL,"device":str(device),"signature":signature})
    atomic_json(out/"folds.json",split_rows)
    seeds = PROTOCOL["seeds"]; predictions = {m:np.full((len(samples),len(seeds)),np.nan) for m in MODES}
    probabilities = {m:np.full((len(samples),len(seeds),3),np.nan) for m in MODES}
    histories=[]; start_time=time.monotonic()
    for fold,(train,test) in enumerate(folds):
        normalizers = fit_normalizers(samples,train)
        train_batch = test_batch = None
        mean,std = float(y[train].mean()),max(float(y[train].std()),1e-6)
        target = torch.tensor((y[train]-mean)/std,dtype=torch.float32,device=device)
        labels = torch.tensor(np.sign(y[train])+1,dtype=torch.long,device=device)
        counts = np.bincount(labels.cpu().numpy(),minlength=3)
        class_weights = torch.tensor(len(train)/(3*np.maximum(counts,1)),dtype=torch.float32,device=device)
        for j,seed in enumerate(seeds):
            for mode in MODES:
                d = out/"checkpoints"/mode/f"fold{fold}_seed{seed}"; d.mkdir(parents=True,exist_ok=True)
                marker = d/"done.json"; checkpoint = d/"model.pt"; prediction_file=d/"predictions.npz"
                state = read_json(marker) if marker.exists() else {}
                if state.get("signature")==signature and checkpoint.exists() and prediction_file.exists() and state.get("model_sha256")==file_hash(checkpoint) and state.get("prediction_sha256")==file_hash(prediction_file):
                    with np.load(prediction_file) as z:
                        predictions[mode][test,j]=z["regression"]; probabilities[mode][test,j]=z["probabilities"]
                    histories.append(state["training"]); continue
                if train_batch is None:
                    train_batch=prepare_batch(samples,train,normalizers,device)
                    test_batch=prepare_batch(samples,test,normalizers,device)
                seed_all(seed+fold*100)
                model=NativeAlignment(mode).to(device)
                optimizer=torch.optim.AdamW(model.parameters(),lr=PROTOCOL["learning_rate"],weight_decay=PROTOCOL["weight_decay"])
                losses=[]; model.train()
                for epoch in range(PROTOCOL["epochs"]):
                    optimizer.zero_grad(set_to_none=True)
                    reg,cls=model(train_batch)
                    loss=torch.nn.functional.mse_loss(reg,target)+torch.nn.functional.cross_entropy(cls,labels,weight=class_weights)
                    if not torch.isfinite(loss): raise ValueError(f"Nonfinite training loss: {mode}/{fold}/{seed}")
                    loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),PROTOCOL["gradient_clip"]); optimizer.step()
                    losses.append(float(loss.detach()))
                model.eval()
                with torch.no_grad():
                    reg,cls,maps=model(test_batch,return_attention=True)
                    pr=reg.cpu().numpy()*std+mean; pc=cls.softmax(-1).cpu().numpy()
                    if j==0:
                        # One held-out example per fold, first manifest index; no cherry-picking.
                        s=samples[test[0]]; ad=out/"attention"/mode; ad.mkdir(parents=True,exist_ok=True)
                        ap=ad/(s["key"]+".npz")
                        export_attention(ap,test_batch,maps,0,s)
                        atomic_json(ap.with_suffix(".json"),{"sample_id":s["id"],"fold":fold,"seed":seed,"mode":mode,
                                    "training_indices":train.tolist(),"held_out_index":int(test[0]),
                                    "source":"native cache rows; original physical CSR maps remain unchanged",
                                    "signature":signature,"weights_sha256":file_hash(ap)})
                predictions[mode][test,j]=pr; probabilities[mode][test,j]=pc
                atomic_npz(prediction_file,regression=pr,probabilities=pc,test_indices=test)
                torch.save({"state_dict":model.state_dict(),"normalizers":normalizers,"label_mean":mean,"label_std":std,
                            "protocol":PROTOCOL,"mode":mode,"fold":fold,"seed":seed,"train_indices":train,"test_indices":test,
                            "signature":signature},checkpoint)
                h={"mode":mode,"fold":fold,"seed":seed,"initial_loss":losses[0],"final_loss":losses[-1],"losses":losses,
                   "parameters":sum(p.numel() for p in model.parameters())}
                histories.append(h)
                atomic_json(marker,{"signature":signature,"model_sha256":file_hash(checkpoint),
                                   "prediction_sha256":file_hash(prediction_file),"training":h})
                LOG.info("fold=%d seed=%d mode=%s loss=%.4f -> %.4f elapsed=%.1fs",fold,seed,mode,losses[0],losses[-1],time.monotonic()-start_time)
                del optimizer,model,maps
        del train_batch,test_batch
        if device.type=="cuda": torch.cuda.empty_cache()
    rows=[]; report_results={}; draws={}
    boot=cluster_bootstrap_indices(groups,PROTOCOL["bootstrap"],np.random.default_rng(PROTOCOL["bootstrap_seed"]))
    for mode in MODES:
        pr=predictions[mode]; cls=probabilities[mode].argmax(-1)-1
        if not np.isfinite(pr).all() or not np.isfinite(probabilities[mode]).all(): raise ValueError("Incomplete OOF predictions")
        metrics=batch_metrics(np.broadcast_to(y[:,None],pr.shape),pr,cls)
        summaries={k:{"mean":float(v.mean()),"seed_std":float(v.std(ddof=1)),"per_seed":v.tolist()} for k,v in metrics.items()}
        report_results[mode]=summaries
        # Average the score, not the predictions, across the three fixed seeds.
        per_seed=[bootstrap_scores(y,pr[:,j],cls[:,j],boot) for j in range(len(seeds))]
        draws[mode]={k:np.mean([d[k] for d in per_seed],axis=0) for k in per_seed[0]}
        for i,s in enumerate(samples):
            for j,seed in enumerate(seeds):
                rows.append({"sample_id":s["id"],"video_id":s["video_id"],"fold":next(f for f,(_,te) in enumerate(folds) if i in te),
                             "label":y[i],"mode":mode,"seed":seed,"regression":pr[i,j],"class":cls[i,j],
                             **{f"prob_{c}":probabilities[mode][i,j,c+1] for c in (-1,0,1)}})
    baseline_reg=np.empty_like(y); baseline_cls=np.empty_like(y)
    for tr,te in folds:
        baseline_reg[te]=y[tr].mean(); baseline_cls[te]=np.bincount((np.sign(y[tr])+1).astype(int),minlength=3).argmax()-1
    comparisons={}
    for reference in ("hard_overlap","time_kernel","local_no_penalty","global_attention"):
        comparisons["local_attention - "+reference]={metric:{"difference":report_results["local_attention"][metric]["mean"]-report_results[reference][metric]["mean"],
                         "ci95":interval(draws["local_attention"][metric]-draws[reference][metric])} for metric in ("mae","macro_f1_3")}
    pd.DataFrame(rows).to_csv(out/"predictions.csv",index=False,encoding="utf-8-sig")
    pd.DataFrame([{"mode":m,**{k+"_"+stat:v[stat] for k,v in r.items() for stat in ("mean","seed_std")}} for m,r in report_results.items()]).to_csv(out/"summary.csv",index=False,encoding="utf-8-sig")
    report={"samples":len(samples),"groups":len(set(groups)),"protocol":PROTOCOL,"results":report_results,
            "training_baseline":{k:float(v[0]) for k,v in batch_metrics(y,baseline_reg,baseline_cls).items()},
            "paired_bootstrap":comparisons,"provenance":prov,"signature":signature,
            "elapsed_seconds":time.monotonic()-start_time,"training_runs":len(histories),
            "limitations":["Attachment 1 only, 100 clips / 37 source videos; exploratory, not Q2/3 training.",
                           "No hyperparameter search or held-out-label early stopping. Same folds/seeds/encoders/head across modes.",
                           "Hard overlap and time kernel do not use learned query/key logits; those unused parameters receive no gradient.",
                           "Bootstrap resamples video groups on fixed OOF predictions and averages seed metrics; no refitting or multiple-test adjustment.",
                           "Attention weights are associations, not corrected word boundaries or human ground truth.",
                           "Comparison with earlier 37-fold Ridge scores is descriptive, not a matched controlled experiment."]}
    atomic_json(out/"training.json",histories); atomic_json(out/"report.json",clean_json(report))
    LOG.info("Complete: %d training runs, %.1fs",len(histories),report["elapsed_seconds"])
    return report


if __name__=="__main__":
    logging.basicConfig(level=logging.INFO,format="%(asctime)s %(levelname)s %(message)s")
    run()

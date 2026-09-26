"""Figures, error analysis, and explanation cards for the selected model."""
from __future__ import annotations

import csv
import html
import json
import subprocess
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import confusion_matrix

from .context_data import attach_text
from .core import DATA, OUT, metrics, pack_split, read_pkl, save_json
from .selected_infer import predict_batches
from .selected_runtime import load_selected


def read_rows(path):
    with path.open(encoding="utf-8-sig",newline="") as f:return list(csv.DictReader(f))


def main():
    root=OUT/"experiments"
    device="cuda" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(8)
    model,_=load_selected(device)
    with np.load(OUT/"normalization.npz") as z:stats={k:z[k] for k in z.files}
    raw=read_pkl(DATA/"附件2-数据集特征文件/aligned_50.pkl")
    valid,_=pack_split(raw["valid"],device,stats)
    attach_text(raw["valid"],valid,device)
    logits,reg=predict_batches(model,valid)
    y=valid["class"].cpu().numpy()
    pred=logits.argmax(-1)
    matrix=confusion_matrix(y,pred,labels=[0,1,2]).tolist()
    save_json(root/"selected_error_analysis.json",{
        "confusion_rows_actual_cols_predicted":matrix,
        "neutral_recall":float(np.mean(pred[y==1]==1)),
        "complete_metrics":metrics(logits,reg,valid)})

    scenarios=read_rows(root/"selected_robustness.csv")
    fig,ax=plt.subplots(figsize=(7,4))
    for mode in ("text","audio","vision"):
        rates=(.1,.3,.5)
        scores=[np.mean([float(r["accuracy"]) for r in scenarios if r["modality"]==mode and float(r["rate"])==rate]) for rate in rates]
        ax.plot(rates,scores,marker="o",label=mode)
    ax.axhline(float(scenarios[0]["accuracy"]),linestyle="--",color="black",label="complete")
    ax.set(xlabel="Contiguous missing fraction",ylabel="Validation accuracy",title="Selected model: local missingness")
    ax.legend()
    fig.tight_layout()
    fig.savefig(root/"selected_robustness.png",dpi=180)
    plt.close(fig)

    details=json.loads((root/"attachment4_explanations_selected.json").read_text())
    shares=np.array([[d["modality_share"][m] for m in ("text","audio","vision")] for d in details])
    fig,ax=plt.subplots(figsize=(7,4))
    bottom=np.zeros(len(details))
    for j,name in enumerate(("text","audio","vision")):
        ax.bar(np.arange(len(details)),shares[:,j],bottom=bottom,label=name)
        bottom+=shares[:,j]
    ax.set(xlabel="Attachment 4 sample",ylabel="Share of absolute class-logit contribution",
           title="Selected model: modality effects",xticks=np.arange(len(details)),
           xticklabels=[d["sample_id"] for d in details])
    ax.legend()
    fig.tight_layout()
    fig.savefig(root/"selected_modality_effects.png",dpi=180)
    plt.close(fig)

    timed=read_rows(root/"attachment4_evidence_times_selected.csv")
    by_sample=defaultdict(list)
    for row in timed:by_sample[row["sample_id"]].append(row)
    audit={x["sample_id"]:x for x in json.loads((root/"attachment4_input_audit_selected.json").read_text())}
    frames=root/"keyframes_selected"
    frames.mkdir(exist_ok=True)
    videos=DATA/"附件4-可解释专项视频样本与特征文件/附件4-可解释专项视频样本与特征文件/对齐版本/videos"
    cards=["<html><head><meta charset='utf-8'><style>body{font:16px sans-serif;max-width:1000px;margin:auto;color:#222}article{border:1px solid #ddd;padding:18px;margin:20px 0}img{max-width:400px}table{border-collapse:collapse}td,th{border:1px solid #ddd;padding:5px}</style></head><body><h1>Selected model: attachment 4 explanation cards</h1><p>Exact three-modality Shapley contributions explain the predicted class logit. Local windows are ranked by the logit decrease when removed. Time is estimated from forced alignment.</p>"]
    for item in details:
        sid=item["sample_id"]
        visual=[r for r in by_sample[sid] if r["evidence_modality"]=="vision" and r["time_status"]=="aligned"]
        if visual:
            best=max(visual,key=lambda r:float(r["target_logit_drop"]))
            middle=(float(best["start_seconds"])+float(best["end_seconds"]))/2
            subprocess.run(["ffmpeg","-v","error","-y","-ss",str(middle),"-i",str(videos/f"{sid}.mp4"),
                            "-frames:v","1","-q:v","4",str(frames/f"{sid}.jpg")],check=True)
        parts=[f"<article><h2>Sample {html.escape(sid)}: {html.escape(item['predicted_label'])}, intensity {item['reported_intensity']:.3f}</h2>",
               f"<p>{html.escape(item['raw_text'])}</p>",
               f"<p>Main modality: {item['main_modality']}; class-logit contributions: "+
               ", ".join(f"{m} {item['modality_contribution'][m]:+.3f}" for m in ("text","audio","vision"))+"</p>"]
        if not audit[sid]["modality_available"]["vision"]:
            parts.append("<p><b>Visual feature unavailable:</b> the supplied matrix contains only zeros.</p>")
        parts.append("<table><tr><th>Modality</th><th>Evidence text</th><th>Time (s)</th><th>Logit drop</th></tr>")
        for row in sorted(by_sample[sid],key=lambda r:float(r["target_logit_drop"]),reverse=True)[:5]:
            timing=f"{row['start_seconds']}–{row['end_seconds']}" if row["time_status"]=="aligned" else "unresolved"
            parts.append(f"<tr><td>{row['evidence_modality']}</td><td>{html.escape(row['text_span'])}</td><td>{timing}</td><td>{float(row['target_logit_drop']):+.3f}</td></tr>")
        parts.append("</table>")
        if (frames/f"{sid}.jpg").exists():parts.append(f"<p><img src='keyframes_selected/{sid}.jpg'></p>")
        parts.append("</article>")
        cards.extend(parts)
    cards.append("</body></html>")
    (root/"attachment4_cards_selected.html").write_text("\n".join(cards),encoding="utf-8")
    print("selected report complete",matrix,flush=True)


if __name__=="__main__":
    main()

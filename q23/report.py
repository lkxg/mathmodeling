from __future__ import annotations

import csv
import html
import json
import subprocess
from collections import Counter, defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import confusion_matrix

from .core import DATA, OUT, metrics, pack_split, predict, read_pkl, save_json
from .evaluate import load_model


def rows(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, stats, _ = load_model(device)
    raw = read_pkl(DATA / "附件2-数据集特征文件/aligned_50.pkl")
    valid, _ = pack_split(raw["valid"], device, stats)
    logits, reg = predict(model, valid)
    y = valid["class"].cpu().numpy()
    yhat = logits.argmax(-1)
    cm = confusion_matrix(y, yhat, labels=[0,1,2])
    report = {"validation_confusion_matrix_rows_actual_cols_predicted": cm.tolist(),
              "validation_by_class": {str(c): {"count": int((y == c).sum()),
                                            "accuracy": float(np.mean(yhat[y == c] == c)),
                                            "mae": float(np.mean(abs(reg[y == c] - valid["reg"].cpu().numpy()[y == c])))}
                                      for c in range(3)},
              "validation_complete": metrics(logits, reg, valid)}
    scenarios = rows(OUT / "validation_robustness.csv")
    if (OUT / "validation_robustness_baseline.csv").exists():
        baseline = rows(OUT / "validation_robustness_baseline.csv")
        report["mean_masked_f1_baseline"] = float(np.mean([float(r["macro_f1"]) for r in baseline[1:]]))
        report["mean_masked_f1_robust"] = float(np.mean([float(r["macro_f1"]) for r in scenarios[1:]]))
    save_json(OUT / "analysis.json", report)

    fig, ax = plt.subplots(figsize=(7,4))
    for mode in ("text", "audio", "vision"):
        rates = sorted({float(r["rate"]) for r in scenarios[1:] if r["modality"] == mode})
        ys = [np.mean([float(r["macro_f1"]) for r in scenarios[1:] if r["modality"] == mode and float(r["rate"]) == rate]) for rate in rates]
        ax.plot(rates, ys, marker="o", label=mode)
    ax.axhline(float(scenarios[0]["macro_f1"]), color="black", linestyle="--", label="complete")
    ax.set(xlabel="Contiguous missing fraction", ylabel="Validation macro-F1", title="Effect of local modality gaps")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "q2_robustness.png", dpi=180)
    plt.close(fig)

    details = json.loads((OUT / "attachment4_explanations.json").read_text())
    input_audit = {x["sample_id"]: x for x in json.loads((OUT / "attachment4_input_audit.json").read_text())}
    shares = np.array([[d["modality_share"][m] for m in ("text", "audio", "vision")] for d in details])
    fig, ax = plt.subplots(figsize=(7,4))
    x = np.arange(len(details))
    bottom = np.zeros(len(details))
    for i, name in enumerate(("text", "audio", "vision")):
        ax.bar(x, shares[:,i], bottom=bottom, label=name)
        bottom += shares[:,i]
    ax.set(xlabel="Attachment 4 sample", ylabel="Share of absolute class-logit contribution",
           title="Sample-level modality effects", xticks=x, xticklabels=[d["sample_id"] for d in details])
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(OUT / "q3_modality_effects.png", dpi=180)
    plt.close(fig)

    timed = rows(OUT / "attachment4_evidence_times.csv")
    by_sample = defaultdict(list)
    for row in timed:
        by_sample[row["sample_id"]].append(row)
    frames = OUT / "keyframes"
    frames.mkdir(exist_ok=True)
    folder = DATA / "附件4-可解释专项视频样本与特征文件/附件4-可解释专项视频样本与特征文件/对齐版本/videos"
    for item in details:
        sid = item["sample_id"]
        visual = [r for r in by_sample[sid] if r["evidence_modality"] == "vision" and r["time_status"] == "aligned"]
        if not visual:
            continue
        best = max(visual, key=lambda r: float(r["target_logit_drop"]))
        middle = (float(best["start_seconds"]) + float(best["end_seconds"])) / 2
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", str(middle), "-i", str(folder / f"{sid}.mp4"),
                        "-frames:v", "1", "-q:v", "4", str(frames / f"{sid}.jpg")], check=True)

    cards = ["<html><head><meta charset='utf-8'><style>body{font:16px sans-serif;max-width:1000px;margin:auto;color:#222}article{border:1px solid #ddd;padding:18px;margin:20px 0}img{max-width:400px}table{border-collapse:collapse}td,th{border:1px solid #ddd;padding:5px}</style></head><body><h1>Attachment 4 explanation cards</h1><p>Class scores use exact three-modality Shapley contributions. Local evidence uses score change after masking a three-position window. Word times are forced-alignment estimates.</p>"]
    for item in details:
        sid = item["sample_id"]
        card = [f"<article><h2>Sample {html.escape(sid)}: {html.escape(item['predicted_label'])}, intensity {item['intensity']:.3f}</h2>",
                f"<p>{html.escape(item['raw_text'])}</p>",
                f"<p>Main modality: <b>{html.escape(item['main_modality'])}</b>. Class-logit contributions: " +
                ", ".join(f"{m} {item['modality_contribution'][m]:+.3f}" for m in ("text", "audio", "vision")) + "</p>",
                "<table><tr><th>Modality</th><th>Evidence text</th><th>Time (s)</th><th>Score drop</th></tr>"]
        if not input_audit[sid]["modality_available"]["vision"]:
            card.insert(2, "<p><b>Visual feature unavailable:</b> supplied vision matrix contains only zeros.</p>")
        for r in sorted(by_sample[sid], key=lambda r: float(r["target_logit_drop"]), reverse=True)[:5]:
            timing = f"{r['start_seconds']}–{r['end_seconds']}" if r["time_status"] == "aligned" else "unresolved"
            card.append(f"<tr><td>{r['evidence_modality']}</td><td>{html.escape(r['text_span'])}</td><td>{timing}</td><td>{float(r['target_logit_drop']):+.3f}</td></tr>")
        card.append("</table>")
        if (frames / f"{sid}.jpg").exists():
            card.append(f"<p><img src='keyframes/{sid}.jpg' alt='Frame near strongest visual window'></p>")
        card.append("</article>")
        cards.extend(card)
    cards.append("</body></html>")
    (OUT / "attachment4_cards.html").write_text("\n".join(cards), encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()

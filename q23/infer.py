from __future__ import annotations

import itertools
import math
from pathlib import Path

import numpy as np
import torch

from .core import DATA, OUT, pack_split, predict, read_pkl, save_csv, save_json
from .evaluate import load_model

NAMES = ("text", "audio", "vision")
LABELS = ("Negative", "Neutral", "Positive")


def subset_item(item, keep):
    out = {k: v.clone() for k, v in item.items()}
    for m in range(3):
        if m not in keep:
            out["mask"][..., m] = False
    out["ids"] = torch.where(out["mask"][..., 0], out["ids"], 0)
    out["audio"] *= out["mask"][..., 1, None]
    out["vision"] *= out["mask"][..., 2, None]
    return out


@torch.no_grad()
def explain_one(model, item):
    keys = [tuple(k for k in range(3) if bits & (1 << k)) for bits in range(8)]
    values = {}
    for keep in keys:
        p = model(subset_item(item, keep))
        values[frozenset(keep)] = (p["logits"][0].detach().cpu().numpy(), float(p["reg"][0]))
    full = values[frozenset((0,1,2))]
    target = int(np.argmax(full[0]))
    contributions, reg_contributions = [], []
    for m in range(3):
        cls, reg = 0., 0.
        others = [j for j in range(3) if j != m]
        for n in range(3):
            for s in itertools.combinations(others, n):
                weight = math.factorial(n) * math.factorial(2-n) / math.factorial(3)
                a, b = values[frozenset(s)], values[frozenset((*s,m))]
                cls += weight * (b[0][target] - a[0][target])
                reg += weight * (b[1] - a[1])
        contributions.append(float(cls))
        reg_contributions.append(float(reg))
    magnitudes = np.abs(contributions)
    share = magnitudes / max(float(magnitudes.sum()), 1e-8)
    main = int(np.argmax(contributions if max(contributions) > 0 else magnitudes))
    return {"predicted_class": target, "predicted_label": LABELS[target],
            "intensity": full[1], "class_logits": full[0].tolist(),
            "class_probabilities": torch.softmax(torch.tensor(full[0]), -1).tolist(),
            "modality_contribution": dict(zip(NAMES, contributions)),
            "intensity_contribution": dict(zip(NAMES, reg_contributions)),
            "modality_share": dict(zip(NAMES, share.tolist())),
            "main_modality": NAMES[main], "baseline_class_logit": float(values[frozenset()][0][target])}


@torch.no_grad()
def evidence_windows(model, item, target, width=3, top_k=3):
    full = model(item)
    base = float(full["logits"][0, target])
    out = []
    mask = item["mask"][0]
    for m, name in enumerate(NAMES):
        pos = torch.nonzero(mask[:, m], as_tuple=False).flatten().tolist()
        if not pos:
            continue
        lo, hi = min(pos), max(pos) + 1
        candidates = []
        for start in range(lo, hi):
            end = min(start+width, hi)
            if not bool(mask[start:end, m].any()):
                continue
            changed = {k: v.clone() for k, v in item.items()}
            changed["mask"][:, start:end, m] = False
            changed["ids"] = torch.where(changed["mask"][..., 0], changed["ids"], 0)
            changed["audio"] *= changed["mask"][..., 1, None]
            changed["vision"] *= changed["mask"][..., 2, None]
            p = model(changed)
            candidates.append({"modality": name, "start_position": start,
                               "end_position_exclusive": end,
                               "target_logit_drop": base-float(p["logits"][0, target]),
                               "intensity_change": float(full["reg"][0]-p["reg"][0])})
        candidates.sort(key=lambda c: c["target_logit_drop"], reverse=True)
        selected = []
        for c in candidates:
            if all(c["end_position_exclusive"] <= s["start_position"] or
                   s["end_position_exclusive"] <= c["start_position"] for s in selected):
                selected.append(c)
            if len(selected) == top_k:
                break
        out.extend(selected)
    return out


def infer_q2(model, stats, device):
    folder = DATA / "附件3-模态缺失特征样本/对齐版本"
    rows = []
    details = []
    for path in sorted(folder.glob("*.pkl")):
        raw = read_pkl(path)["test"]
        item, _ = pack_split(raw, device, stats)
        logits, reg = predict(model, item)
        probs = torch.softmax(torch.as_tensor(logits[0]), -1).numpy()
        observed = item["mask"][0].cpu().numpy()
        valid = item["valid"][0].cpu().numpy()
        missing = {name: int(np.sum(valid & ~observed[:, m])) for m, name in enumerate(NAMES)}
        cls = int(logits[0].argmax())
        raw_strength = float(reg[0])
        strength = 0. if cls == 1 else min(raw_strength, -1e-6) if cls == 0 else max(raw_strength, 1e-6)
        rows.append({"sample_id": path.stem, "polarity": LABELS[cls], "class_id": cls,
                     "intensity": round(strength, 6), "raw_intensity": round(raw_strength, 6),
                     "prob_negative": round(float(probs[0]), 6),
                     "prob_neutral": round(float(probs[1]), 6),
                     "prob_positive": round(float(probs[2]), 6)})
        details.append({"sample_id": path.stem, "observed_valid_positions": int(valid.sum()),
                        "missing_positions": missing})
    save_csv(OUT / "attachment3_predictions.csv", rows)
    save_json(OUT / "attachment3_input_audit.json", details)
    return len(rows)


def infer_q3(model, stats, device):
    folder = DATA / "附件4-可解释专项视频样本与特征文件/附件4-可解释专项视频样本与特征文件/对齐版本"
    rows, detailed, audit = [], [], []
    for path in sorted(folder.glob("*.pkl")):
        raw = read_pkl(path)
        shaped = {k: np.asarray(v)[None] if k in ("audio", "vision", "text_bert") else v for k,v in raw.items()}
        item, _ = pack_split(shaped, device, stats)
        available = item["mask"][0].any(dim=0).cpu().tolist()
        exp = explain_one(model, item)
        windows = evidence_windows(model, item, exp["predicted_class"])
        exp.update({"sample_id": path.stem, "raw_text": str(raw["raw_text"]),
                    "token_ids": item["ids"][0].cpu().tolist(), "windows": windows})
        detailed.append(exp)
        audit.append({"sample_id": path.stem,
                      "nonzero_positions": {m: int(item["mask"][0,:,j].sum().item()) for j,m in enumerate(NAMES)},
                      "modality_available": dict(zip(NAMES, available))})
        rows.append({"sample_id": path.stem, "polarity": exp["predicted_label"],
                     "class_id": exp["predicted_class"], "intensity": round(exp["intensity"], 6),
                     "main_modality": exp["main_modality"],
                     "vision_available": available[2],
                     **{f"contribution_{m}": round(exp["modality_contribution"][m], 6) for m in NAMES},
                     **{f"share_{m}": round(exp["modality_share"][m], 6) for m in NAMES},
                     "evidence_positions": ";".join(f"{w['modality']}:{w['start_position']}-{w['end_position_exclusive']-1}" for w in windows)})
    save_csv(OUT / "attachment4_predictions_explanations.csv", rows)
    save_json(OUT / "attachment4_explanations.json", detailed)
    save_json(OUT / "attachment4_input_audit.json", audit)
    return len(rows)


def main():
    torch.set_num_threads(8)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, stats, _ = load_model(device)
    n2 = infer_q2(model, stats, device)
    n3 = infer_q3(model, stats, device)
    print(f"attachment3={n2}, attachment4={n3}")


if __name__ == "__main__":
    main()

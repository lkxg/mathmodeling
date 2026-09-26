"""Validation-only missing-modality comparison for all new architectures."""
from __future__ import annotations

import numpy as np
import torch

from .context_data import attach_text
from .core import DATA, MODEL_ID, MODEL_REVISION, OUT, metrics, pack_split, read_pkl, save_csv, save_json
from .selected_evaluate import mask_scenario
from .selected_infer import predict_batches
from .selected_runtime import SelectedEnsemble


MODELS = {
    "existing_pair": ["shared_private", "reliability_proxy"],
    "dual_query": ["dual_query"],
    "enhance_balance": ["enhance_balance"],
    "dual_query_distilled": ["dual_query_distilled"],
}


def main():
    torch.set_num_threads(8)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    raw = read_pkl(DATA / "附件2-数据集特征文件/aligned_50.pkl")
    with np.load(OUT / "normalization.npz") as saved:
        stats = {key: saved[key] for key in saved.files}
    valid, _ = pack_split(raw["valid"], device, stats)
    attach_text(raw["valid"], valid, device)
    models = {name: SelectedEnsemble(members, device).eval() for name, members in MODELS.items()}
    from transformers import BertModel
    bert = BertModel.from_pretrained(MODEL_ID, revision=MODEL_REVISION).to(device).eval()
    cases = [("complete", (), 0, "none")]
    for modality in range(3):
        for rate in (.1, .3, .5):
            for location in ("start", "middle", "end"):
                cases.append(("single_contiguous", (modality,), rate, location))
    for modalities in ((0, 1), (0, 2), (1, 2), (0, 1, 2)):
        cases.append(("multiple_contiguous", modalities, .3, "middle"))
    rows = []
    for kind, modalities, rate, location in cases:
        masked = (mask_scenario(valid, raw["valid"]["text_bert"], modalities,
                                rate, location, bert, device) if modalities else valid)
        for name, model in models.items():
            result = metrics(*predict_batches(model, masked), valid)
            rows.append({"model": name, "scenario": kind,
                         "modality": "+".join(("text", "audio", "vision")[m] for m in modalities) or "all",
                         "rate": rate, "location": location, **result})
        print(kind, modalities, rate, location, flush=True)
    root = OUT / "experiments"
    save_csv(root / "new_architecture_robustness.csv", rows)
    summary = {}
    for name in MODELS:
        summary[name] = {}
        for kind in ("complete", "single_contiguous", "multiple_contiguous"):
            subset = [row for row in rows if row["model"] == name and row["scenario"] == kind]
            summary[name][kind] = {
                "mean_accuracy": float(np.mean([row["accuracy"] for row in subset])),
                "mean_macro_f1": float(np.mean([row["macro_f1"] for row in subset])),
                "mean_mae": float(np.mean([row["mae"] for row in subset])),
            }
    save_json(root / "new_architecture_robustness_summary.json", summary)
    print(summary, flush=True)


if __name__ == "__main__":
    main()

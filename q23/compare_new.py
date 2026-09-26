"""Fixed validation comparison of new architectures and small ensembles.

All candidates are declared before reading predictions; the test split is never
used to choose a model. The old selection remains available for reproduction.
"""
from __future__ import annotations

import numpy as np

from .core import DATA, OUT, metrics, pack_split, read_pkl, save_csv, save_json


CANDIDATES = [
    ("shared_private", "reliability_proxy"),
    ("dual_query",),
    ("enhance_balance",),
    ("dual_query_distilled",),
    ("shared_private", "reliability_proxy", "dual_query"),
    ("shared_private", "reliability_proxy", "enhance_balance"),
    ("shared_private", "reliability_proxy", "dual_query_distilled"),
    ("shared_private", "reliability_proxy", "dual_query", "enhance_balance"),
    ("shared_private", "reliability_proxy", "enhance_balance", "dual_query_distilled"),
]


def main():
    raw = read_pkl(DATA / "附件2-数据集特征文件/aligned_50.pkl")
    _, stats = pack_split(raw["train"], "cpu")
    valid, _ = pack_split(raw["valid"], "cpu", stats)
    root = OUT / "experiments"
    names = sorted({name for combo in CANDIDATES for name in combo})
    predictions = {}
    for name in names:
        with np.load(root / f"{name}_valid_predictions.npz") as saved:
            predictions[name] = (saved["logits"], saved["reg"])
    rows = []
    for combo in CANDIDATES:
        logits = np.mean([predictions[name][0] for name in combo], axis=0)
        reg = np.mean([predictions[name][1] for name in combo], axis=0)
        rows.append({"models": "+".join(combo), "count": len(combo),
                     **metrics(logits, reg, valid)})
    rows.sort(key=lambda row: (row["accuracy"], row["macro_f1"], -row["mae"]), reverse=True)
    save_csv(root / "new_architecture_comparison.csv", rows)
    winner = rows[0]
    save_json(root / "new_architecture_selection.json", {
        "selected_models": winner["models"].split("+"),
        "rule": "max validation Accuracy, then macro-F1, then lower MAE",
        "validation": winner,
        "candidates": rows,
    })
    for row in rows:
        print(row["models"], f"Acc={row['accuracy']:.4f}",
              f"F1={row['macro_f1']:.4f}", f"MAE={row['mae']:.4f}")


if __name__ == "__main__":
    main()

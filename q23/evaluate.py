from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from .core import DATA, OUT, Predictor, block_mask, load_embedding, metrics, pack_split, predict, read_pkl, save_csv, save_json, seed_all


def load_model(device, baseline=False):
    obj = torch.load(OUT / ("model_baseline.pt" if baseline else "model.pt"), map_location="cpu", weights_only=True)
    model = Predictor(load_embedding(device)).to(device)
    status = model.load_state_dict(obj["state_dict"], strict=False)
    if status.missing_keys != ["embedding.weight"] or status.unexpected_keys:
        raise RuntimeError(f"Checkpoint mismatch: {status}")
    model.eval()
    with np.load(OUT / "normalization.npz") as z:
        stats = {k: z[k] for k in z.files}
    return model, stats, obj


def measure_scenarios(model, data):
    rows = []
    clean = metrics(*predict(model, data), data)
    rows.append({"scenario": "完整输入", "modality": "all", "rate": 0, **clean})
    for mode, name in enumerate(("text", "audio", "vision")):
        for rate in (.1, .3, .5):
            for location in ("start", "middle", "end"):
                masked = {k: v.clone() for k, v in data.items()}
                m = masked["mask"]
                for i in range(len(m)):
                    pos = torch.nonzero(masked["valid"][i], as_tuple=False).flatten().cpu().numpy()
                    if not len(pos):
                        continue
                    length = max(1, round(len(pos)*rate))
                    start = 0 if location == "start" else (len(pos)-length)//2 if location == "middle" else len(pos)-length
                    m[i, pos[start:start+length], mode] = False
                masked["ids"] = torch.where(m[..., 0], masked["ids"], 0)
                masked["audio"] *= m[..., 1, None]
                masked["vision"] *= m[..., 2, None]
                result = metrics(*predict(model, masked), data)
                rows.append({"scenario": "单模态连续缺失", "modality": name,
                             "rate": rate, "location": location, **result})
    for modes in ((0,1), (0,2), (1,2), (0,1,2)):
        masked = {k: v.clone() for k, v in data.items()}
        m = masked["mask"]
        for i in range(len(m)):
            pos = torch.nonzero(masked["valid"][i], as_tuple=False).flatten().cpu().numpy()
            if not len(pos):
                continue
            length = max(1, round(len(pos)*.3))
            start = (len(pos)-length)//2
            for j in modes:
                m[i, pos[start:start+length], j] = False
        masked["ids"] = torch.where(m[..., 0], masked["ids"], 0)
        masked["audio"] *= m[..., 1, None]
        masked["vision"] *= m[..., 2, None]
        result = metrics(*predict(model, masked), data)
        rows.append({"scenario": "多模态同步连续缺失", "modality": "+".join(("text", "audio", "vision")[j] for j in modes),
                     "rate": .3, "location": "middle", **result})
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--include-test", action="store_true", help="Report held-out labeled test once after model selection")
    parser.add_argument("--baseline", action="store_true")
    args = parser.parse_args()
    seed_all(2026)
    torch.set_num_threads(8)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, stats, checkpoint = load_model(device, args.baseline)
    raw = read_pkl(DATA / "附件2-数据集特征文件/aligned_50.pkl")
    valid, _ = pack_split(raw["valid"], device, stats)
    rows = measure_scenarios(model, valid)
    suffix = "_baseline" if args.baseline else ""
    save_csv(OUT / f"validation_robustness{suffix}.csv", rows)
    def average(metric, scenario=None):
        selected = [r[metric] for r in rows[1:] if scenario is None or r["scenario"] == scenario]
        return float(np.mean(selected))
    report = {"selected_epoch": checkpoint["epoch"], "validation_complete": rows[0],
              "mean_masked_macro_f1": average("macro_f1"),
              "mean_masked_mae": average("mae"),
              "mean_single_macro_f1": average("macro_f1", "单模态连续缺失"),
              "mean_single_mae": average("mae", "单模态连续缺失"),
              "mean_multiple_macro_f1": average("macro_f1", "多模态同步连续缺失"),
              "mean_multiple_mae": average("mae", "多模态同步连续缺失")}
    if args.include_test:
        test, _ = pack_split(raw["test"], device, stats)
        report["held_out_test_complete"] = metrics(*predict(model, test), test)
    save_json(OUT / f"metrics{suffix}.json", report)
    print(report)


if __name__ == "__main__":
    main()

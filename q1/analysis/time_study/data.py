from __future__ import annotations

import importlib.metadata
from pathlib import Path

import numpy as np

from ...common import digest, file_hash, load_config, read_json
from ...pipeline import Cache

MODALITIES = ("text", "emotion", "acoustic", "vision")


def default_config():
    return load_config(Path(__file__).parents[2] / "config.yaml")


def provenance(cfg, modules):
    root = Path(cfg["output_dir"])
    code = {str(Path(p).resolve()): file_hash(p) for p in modules}
    return {"code_sha256": digest(code), "code_files": code,
            "manifest_sha256": file_hash(root / "manifest.json"),
            "run_config_sha256": file_hash(root / "run_config.json"),
            "versions": {p: importlib.metadata.version(p) for p in
                         ("numpy", "scipy", "scikit-learn", "torch")}}


def load_samples(cfg, verify=True):
    root = Path(cfg["output_dir"])
    manifest = read_json(root / "manifest.json")["samples"]
    cache = Cache(cfg)
    result = []
    for sample in manifest:
        if verify and not cache.valid(sample, "aggregate"):
            raise ValueError(f"主流程缓存不完整或过期：{sample['id']}")
        d = root / "cache" / sample["key"]
        s = {**sample, "directory": d, "media": read_json(d / "media.json")}
        s["words"] = read_json(d / "align.json")["words"]
        s["meta"] = read_json(root / "features" / (sample["key"] + ".json"))
        with np.load(root / "features" / (sample["key"] + ".npz"), allow_pickle=False) as z:
            s["export"] = {k: z[k] for k in z.files}
        with np.load(d / "text.npz", allow_pickle=False) as z:
            valid = z["valid"].astype(bool)
            timed = np.asarray([w["time_valid"] for w in s["words"]])
            s["text"] = {"x": z["features"], "intervals": np.asarray([[w["start"], w["end"]] for w in s["words"]]),
                         "quality": (valid & timed).astype(float), "content_valid": valid}
        with np.load(d / "audio_features.npz", allow_pickle=False) as z:
            for m in ("emotion", "acoustic"):
                s[m] = {"x": z[m], "intervals": z[m + "_intervals"],
                        "quality": z[m + "_valid"].astype(float)}
        with np.load(d / "vision.npz", allow_pickle=False) as z:
            s["vision"] = {"x": z["features"], "intervals": z["intervals"],
                           "quality": z["quality"], "frame_indices": z["frame_indices"], "pts": z["pts"]}
        result.append(s)
    return result


def merge(intervals):
    out = []
    for a, b in sorted(np.asarray(intervals).reshape(-1, 2).tolist()):
        if b <= a:
            continue
        if out and a <= out[-1][1]:
            out[-1][1] = max(b, out[-1][1])
        else:
            out.append([float(a), float(b)])
    return out


def length(intervals):
    return sum(b - a for a, b in intervals)


def intersection_length(a, b):
    i = j = 0
    total = 0.
    while i < len(a) and j < len(b):
        total += max(0., min(a[i][1], b[j][1]) - max(a[i][0], b[j][0]))
        if a[i][1] <= b[j][1]:
            i += 1
        else:
            j += 1
    return total


def gaps(intervals, duration):
    out, cursor = [], 0.
    for a, b in merge(intervals):
        if a > cursor:
            out.append([cursor, a])
        cursor = max(cursor, b)
    if cursor < duration:
        out.append([cursor, duration])
    return out


def mean_valid(x, mask):
    return x[mask].astype(float).mean(axis=0) if np.any(mask) else np.full(x.shape[1], np.nan)


def output_dir(cfg, name):
    p = Path(cfg["output_dir"]) / "analysis" / "time_study" / name
    p.mkdir(parents=True, exist_ok=True)
    return p

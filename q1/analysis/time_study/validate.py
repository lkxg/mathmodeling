"""Failing validation for time identities, maps, coverage and reconstruction."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

from ...common import atomic_json
from ...pipeline import validate as validate_features
from ..probe import clean_json
from . import data
from .data import (default_config, load_samples, merge, length, intersection_length,
                   gaps, output_dir, provenance)


def check_mapping(z, name, source, tolerance=2e-6):
    times = z["timestamps"]
    windows, quality, values = source["intervals"], source["quality"], source["x"]
    offsets, indices, weights = (z[f"map_{name}_{k}"] for k in ("offsets", "indices", "weights"))
    issues = []
    if not np.array_equal(z[f"source_{name}_intervals"], windows):
        issues.append("source_intervals_changed")
    if (offsets.shape != (len(times) + 1,) or offsets[0] != 0 or np.any(np.diff(offsets) < 0)
            or offsets[-1] != len(indices) or len(weights) != len(indices)):
        return ["invalid_csr"], None, None
    if np.any(indices < 0) or np.any(indices >= len(windows)):
        return ["source_index_out_of_range"], None, None
    if not np.isfinite(weights).all() or np.any(weights < 0):
        return ["invalid_weights"], None, None
    good = ((quality > 0) & np.isfinite(quality) & np.isfinite(values).all(axis=1)
            & np.isfinite(windows).all(axis=1) & (windows[:, 1] > windows[:, 0]))
    means = np.zeros((len(times), values.shape[1]), np.float32)
    stds = means.copy()
    centers, ranges = {}, {}
    for k, (start, end) in enumerate(times):
        overlap = np.maximum(0., np.minimum(end, windows[:, 1]) - np.maximum(start, windows[:, 0]))
        expected = np.flatnonzero(good & (overlap > 0)) if z["time_valid_mask"][k] and end > start else np.array([], int)
        lo, hi = offsets[k:k+2]
        ix, w = indices[lo:hi], weights[lo:hi]
        if not np.array_equal(expected, ix):
            issues.append(f"row_{k}_incorrect_source_set")
            continue
        if len(ix):
            ew = overlap[ix] * quality[ix]
            ew /= ew.sum()
            if not np.allclose(w, ew, atol=tolerance, rtol=tolerance) or not np.isclose(w.sum(), 1, atol=tolerance):
                issues.append(f"row_{k}_incorrect_weights")
            x = values[ix].astype(float)
            mu = np.sum(x * ew[:, None], axis=0)
            means[k] = mu
            stds[k] = np.sqrt(np.maximum(0., np.sum((x - mu) ** 2 * ew[:, None], axis=0)))
            centers[k] = float(np.sum(windows[ix].mean(axis=1) * ew))
            ranges[k] = (int(ix.min()), int(ix.max()))
    word_rows = np.flatnonzero((z["word_indices"] >= 0) & z["time_valid_mask"])
    word_rows = word_rows[np.argsort(z["word_indices"][word_rows])]
    previous = None
    for k in word_rows:
        if k not in centers:
            continue
        current = (*ranges[k], centers[k])
        if previous is not None and (current[0] < previous[0] or current[1] < previous[1]
                                     or current[2] < previous[2] - 1e-6):
            issues.append(f"row_{k}_nonmonotonic_word_mapping")
        previous = current
    return issues, means, stds


def audit(samples, cfg):
    rows, failures = [], []
    totals = {m: {"valid_seconds": 0., "mapped_seconds": 0.} for m in ("emotion", "acoustic", "vision")}
    for s in samples:
        z, duration = s["export"], s["media"]["duration"]
        issues = []
        target = merge(z["timestamps"][z["time_valid_mask"]])
        missing = gaps(target, duration)
        max_gap = max([b - a for a, b in missing], default=0.)
        if cfg["aggregation"]["retain_gaps"] and max_gap > cfg["aggregation"]["min_gap_seconds"] + 1e-6:
            issues.append("unexpected_long_unassigned_gap")
        if np.any(z["timestamps"] < -1e-6) or np.any(z["timestamps"] > duration + 1e-6):
            issues.append("target_out_of_clip")
        word_ids = z["word_indices"][z["word_indices"] >= 0]
        if sorted(word_ids.tolist()) != list(range(len(s["words"]))):
            issues.append("word_identity_not_one_to_one")
        reconstructed = {}
        for m in totals:
            local, mean, std = check_mapping(z, m, s[m])
            issues.extend([m + ":" + e for e in local])
            reconstructed[m] = (mean, std)
            union = merge(s[m]["intervals"][s[m]["quality"] > 0])
            totals[m]["valid_seconds"] += length(union)
            totals[m]["mapped_seconds"] += intersection_length(union, target)
        if all(v[0] is not None for v in reconstructed.values()):
            e, a, v = (reconstructed[m] for m in ("emotion", "acoustic", "vision"))
            audio, vision = np.concatenate([e[0], a[0], a[1]], axis=1), np.concatenate(v, axis=1)
            for name, x in (("audio", audio), ("vision", vision)):
                if not np.array_equal(x.astype(z[name].dtype), z[name]):
                    issues.append(name + ":reconstruction_mismatch")
        with np.load(s["directory"] / "frames.npz", allow_pickle=False) as frames:
            idx = s["vision"]["frame_indices"]
            if np.any(idx < 0) or np.any(idx >= len(frames["pts"])) or not np.array_equal(frames["pts"][idx], s["vision"]["pts"]):
                issues.append("original_frame_pts_mismatch")
        for w in s["words"]:
            a, b = w["char_span"]
            if not 0 <= a < b <= len(s["raw_text"]):
                issues.append("invalid_original_text_span")
        row = {"sample_id": s["id"], "duration": duration, "mapped_seconds": length(target),
               "unmapped_seconds": length(missing), "max_gap_seconds": max_gap,
               "word_count": len(word_ids), "time_valid_word_count": int(sum(w["time_valid"] for w in s["words"])),
               "passed": not issues, "issues": ";".join(issues)}
        rows.append(row)
        failures += [{"sample_id": s["id"], "issue": e} for e in issues]
    for v in totals.values():
        v["retained_ratio"] = v["mapped_seconds"] / v["valid_seconds"] if v["valid_seconds"] else None
    return {"samples": len(samples), "passed": not failures, "issues": failures,
            "clip_seconds": sum(r["duration"] for r in rows),
            "mapped_seconds": sum(r["mapped_seconds"] for r in rows),
            "unmapped_seconds": sum(r["unmapped_seconds"] for r in rows),
            "longest_gap_seconds": max(r["max_gap_seconds"] for r in rows),
            "source_support": totals, "rows": rows,
            "note": "时间并集覆盖含未分配区间；不等于词边界准确率。单调检查排除无效词时间。"}


def run(cfg=None):
    cfg = cfg or default_config()
    samples = load_samples(cfg)
    basic = validate_features(cfg, [{k: s[k] for k in
              ("key", "id", "video_id", "clip_id", "video", "video_sha256", "raw_text", "label", "annotation")} for s in samples])
    report = audit(samples, cfg)
    report["basic_complete"] = basic["complete"]
    report["provenance"] = provenance(cfg, [__file__, data.__file__])
    report["passed"] &= basic["complete"] == len(samples)
    out = output_dir(cfg, "validation")
    atomic_json(out / "report.json", clean_json(report))
    pd.DataFrame(report["rows"]).to_csv(out / "samples.csv", index=False, encoding="utf-8-sig")
    print(json.dumps({k: v for k, v in report.items() if k not in ("rows", "provenance")}, ensure_ascii=False, indent=2))
    if not report["passed"]:
        raise SystemExit(1)
    return report


if __name__ == "__main__":
    run()

"""Independent checks of identities, source maps and exported numerical values."""
from pathlib import Path

import numpy as np

from .common import digest, read_json
from .temporal import union_length, word_spans


REASON_LABELS = {
    "silent_audio": "数字静音",
    "out_of_audio_bounds": "边界越界",
    "overlap_or_nonmonotonic": "非单调或重叠",
    "non_positive_duration": "非正时长",
    "other": "其他",
}
# This is an explicit display priority, not an inference about root causes.
REASON_PRIORITY = tuple(REASON_LABELS)


def invalid_reason(word):
    return next((r for r in REASON_PRIORITY if r in word.get("flags", [])), "other")


def merged_intervals(intervals):
    merged = []
    for a, b in sorted(np.asarray(intervals, float).reshape(-1, 2).tolist()):
        if not np.isfinite([a, b]).all() or b <= a:
            raise ValueError("集合统计只接受有限的正时长区间")
        if merged and a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    return merged


def intersect_intervals(left, right):
    left, right = merged_intervals(left), merged_intervals(right)
    result, i, j = [], 0, 0
    while i < len(left) and j < len(right):
        a, b = max(left[i][0], right[j][0]), min(left[i][1], right[j][1])
        if b > a:
            result.append([a, b])
        if left[i][1] <= right[j][1]:
            i += 1
        else:
            j += 1
    return result


def temporal_support(targets, acoustic, visual):
    """I is the target union; A/V are valid native supports clipped to I."""
    target = merged_intervals(targets)
    a = intersect_intervals(target, acoustic)
    v = intersect_intervals(target, visual)
    return {"timeline_seconds": union_length(target),
            "audio_support_seconds": union_length(a),
            "vision_support_seconds": union_length(v),
            "union_support_seconds": union_length(a + v),
            "intersection_support_seconds": union_length(intersect_intervals(a, v))}


def valid_support(windows, values, quality):
    good = (np.isfinite(values).all(1) & np.isfinite(windows).all(1)
            & (windows[:, 1] > windows[:, 0]) & np.isfinite(quality) & (quality > 0))
    return windows[good]


def validation_metrics(rows):
    rows = [r for r in rows if r["status"] == "ok"]
    def total(key):
        return sum(r.get(key, 0) for r in rows)
    def ratio(a, b):
        return a / b if b else None
    target, duration = total("timeline_seconds"), total("duration")
    coverage = {
        "aggregation": "sum of interval-union seconds across validated clips, not mean of clip percentages",
        "target_definition": "I = union of time-valid word intervals and retained gaps; independent of modality masks",
        "source_definition": "J_m = union of finite native windows with valid features and positive effective quality",
        "timeline_retention": {"seconds": target, "denominator_seconds": duration, "ratio": ratio(target, duration)},
    }
    for name in ("audio", "vision", "union", "intersection"):
        seconds = total(name + "_support_seconds")
        coverage[name] = {"seconds": seconds, "denominator_seconds": target, "ratio": ratio(seconds, target)}
    words, positions = total("words"), total("positions")
    alignment = {"total_words": words, "time_valid_words": total("time_valid_words"),
                 "structural_valid_ratio": ratio(total("time_valid_words"), words),
                 "invalid_words": total("invalid_words"),
                 "samples_with_invalid_words": sum(r["invalid_words"] > 0 for r in rows),
                 "nonexclusive_reasons": {k: total("flag_" + k) for k in REASON_LABELS},
                 "exclusive_reasons": {k: total("primary_" + k) for k in REASON_LABELS},
                 "exclusive_priority": list(REASON_PRIORITY),
                 "unreturned_words_in_validated_samples": 0,
                 "unreturned_note": "Strict original-text lexical coverage is required; a missing word fails the sample."}
    if sum(alignment["exclusive_reasons"].values()) != alignment["invalid_words"]:
        raise ValueError("互斥原因合计不等于无效词数")
    availability = {"positions": positions, "gap_positions": total("gap_positions"),
                    "definition": "available positions / all word-plus-gap positions; not extraction success"}
    for name in ("text", "audio", "vision"):
        count = total(name + "_available_positions")
        availability[name] = {"count": count, "ratio": ratio(count, positions)}
    availability["text_word_extraction"] = {"count": total("text_available_words"),
                                             "denominator_words": words,
                                             "ratio": ratio(total("text_available_words"), words)}
    return {"metrics_version": 2, "temporal_coverage": coverage,
            "word_alignment": alignment, "modality_availability": availability}


def check_mapping(z, name, windows, values, quality, output, column):
    times = z["timestamps"]
    offsets, indices, weights = (z[f"map_{name}_{k}"] for k in ("offsets", "indices", "weights"))
    if not np.array_equal(z[f"source_{name}_intervals"], windows):
        raise ValueError(f"{name}源时间窗改变")
    if (offsets.shape != (len(times) + 1,) or offsets[0] != 0 or np.any(np.diff(offsets) < 0)
            or offsets[-1] != len(indices) or len(weights) != len(indices)
            or np.any(indices < 0) or np.any(indices >= len(windows))
            or not np.isfinite(weights).all() or np.any(weights < 0)):
        raise ValueError(f"{name}来源索引或权重非法")
    good = (np.isfinite(values).all(1) & np.isfinite(windows).all(1)
            & (windows[:, 1] > windows[:, 0]) & np.isfinite(quality) & (quality > 0))
    means = np.zeros((len(times), values.shape[1]), np.float32)
    stds, mask, coverage = means.copy(), np.zeros(len(times), bool), np.zeros(len(times))
    centers = {}
    for k, (start, end) in enumerate(times):
        overlap = np.maximum(0, np.minimum(end, windows[:, 1]) - np.maximum(start, windows[:, 0]))
        expected = np.flatnonzero(good & (overlap > 0)) if z["time_valid_mask"][k] and end > start else np.array([], int)
        lo, hi = offsets[k:k + 2]
        ix, w = indices[lo:hi], weights[lo:hi]
        if not np.array_equal(ix, expected):
            raise ValueError(f"{name}位置{k}映射了错误源帧")
        if len(ix):
            ew = overlap[ix] * quality[ix]
            ew /= ew.sum()
            if not np.allclose(w, ew, rtol=2e-6, atol=2e-6):
                raise ValueError(f"{name}位置{k}权重不能重建")
            x = values[ix].astype(float)
            mu = (x * ew[:, None]).sum(0)
            means[k] = mu
            stds[k] = np.sqrt(np.maximum(0, ((x - mu) ** 2 * ew[:, None]).sum(0)))
            mask[k] = True
            coverage[k] = union_length([(max(start, windows[j, 0]), min(end, windows[j, 1])) for j in ix]) / (end - start)
            centers[k] = (int(ix.min()), int(ix.max()), float((windows[ix].mean(1) * ew).sum()))
    reconstructed = np.concatenate([means, stds], axis=1).astype(z[output].dtype)
    if not np.array_equal(reconstructed, z[output]):
        raise ValueError(f"{name}源帧无法重建导出均值和标准差")
    if not np.array_equal(mask, z["modality_mask"][:, column]):
        raise ValueError(f"{name}可用掩码不符")
    if not np.allclose(coverage, z["coverage"][:, column - 1], atol=2e-6, rtol=2e-6):
        raise ValueError(f"{name}覆盖率不符")
    words = np.flatnonzero((z["word_indices"] >= 0) & z["time_valid_mask"])
    previous = None
    for k in words[np.argsort(z["word_indices"][words])]:
        if k in centers:
            current = centers[k]
            if previous is not None and any(a < b - 1e-6 for a, b in zip(current, previous)):
                raise ValueError(f"{name}词序与来源时间顺序不一致")
            previous = current


def check_sample(sample, directory, npz, meta, cfg):
    from .pipeline import reviewed_quality
    directory = Path(directory)
    media = read_json(directory / "media.json")
    words = read_json(directory / "align.json")["words"]
    if word_spans(sample["raw_text"], [w["text"] for w in words]) != [w["char_span"] for w in words]:
        raise ValueError("原文没有被完整、准确地映射到词记录")
    with np.load(npz, allow_pickle=False) as source:
        z = {k: source[k] for k in source.files}
    n = int(z["length"])
    if meta["schema_version"] != 3 or int(z["schema_version"]) != 3 or meta["config_sha256"] != digest(cfg):
        raise ValueError("特征协议或配置版本不符")
    if n < 1 or len(meta["timeline"]) != n or z["timestamps"].shape != (n, 2):
        raise ValueError("时间轴长度不符")
    if z["modality_mask"].shape != (n, 3) or z["coverage"].shape != (n, 2):
        raise ValueError("模态掩码或覆盖率形状不符")
    for key in ("valid_mask", "time_valid_mask", "modality_mask"):
        if z[key].dtype != np.bool_:
            raise ValueError("掩码必须为布尔数组")
    if z["valid_mask"].shape != (n,) or z["time_valid_mask"].shape != (n,) or not z["valid_mask"].all():
        raise ValueError("单样本长度或padding掩码错误")
    for col, (name, dimension) in enumerate((("text", 768), ("audio", 50), ("vision", 56))):
        x = z[name]
        if x.shape != (n, dimension) or meta["dimensions"][name] != dimension or not np.isfinite(x).all():
            raise ValueError(f"{name}维度或数值错误")
        if np.any(x[~z["modality_mask"][:, col]] != 0):
            raise ValueError(f"{name}缺失位置必须为零")
    t = z["timestamps"]
    if not np.isfinite(t).all() or np.any(t < -1e-6) or np.any(t > media["duration"] + 1e-6):
        raise ValueError("目标时间超出原视频范围")
    ids = z["word_indices"]
    if ids.shape != (n,) or np.any(ids < -1) or sorted(ids[ids >= 0].tolist()) != list(range(len(words))):
        raise ValueError("原文词身份没有一一保留")
    with np.load(directory / "text.npz", allow_pickle=False) as raw:
        expected = np.zeros_like(z["text"])
        mask = np.zeros(n, bool)
        for k, index in enumerate(ids):
            if index >= 0:
                expected[k] = raw["features"][index]
                mask[k] = raw["valid"][index]
                word = words[index]
                if z["time_valid_mask"][k] != word["time_valid"] or not np.array_equal(t[k], [word["start"], word["end"]]):
                    raise ValueError("输出位置与原词时间不符")
                a, b = word["char_span"]
                if not 0 <= a < b <= len(sample["raw_text"]):
                    raise ValueError("原文字串范围非法")
        if not np.array_equal(expected, z["text"]) or not np.array_equal(mask, z["modality_mask"][:, 0]):
            raise ValueError("词级文本与输出对应不符")
    with np.load(directory / "audio.npz", allow_pickle=False) as raw:
        if raw["acoustic"].shape[1] != 25 or not np.array_equal(raw["acoustic_valid"], z["source_acoustic_valid"]):
            raise ValueError("原生声学特征或有效性记录不符")
        check_mapping(z, "acoustic", raw["acoustic_intervals"], raw["acoustic"], raw["acoustic_valid"].astype(float), "audio", 1)
        acoustic_support = valid_support(raw["acoustic_intervals"], raw["acoustic"], raw["acoustic_valid"])
        if media["silent_audio"] and (raw["acoustic_valid"].any() or z["modality_mask"][:, 1].any() or any(w["time_valid"] for w in words)):
            raise ValueError("全静音样本错误地拥有有效音频或词时间")
    with np.load(directory / "vision.npz", allow_pickle=False) as raw:
        quality, excluded, _ = reviewed_quality(sample, raw, cfg)
        if not np.array_equal(quality, z["source_vision_quality"]) or not np.array_equal(excluded, z["source_vision_excluded"]):
            raise ValueError("视觉质量复核记录不符")
        check_mapping(z, "vision", raw["intervals"], raw["features"], quality, "vision", 2)
        visual_support = valid_support(raw["intervals"], raw["features"], quality)
        with np.load(directory / "frames.npz", allow_pickle=False) as frames:
            ix = raw["frame_indices"]
            if np.any(ix < 0) or np.any(ix >= len(frames["pts"])) or not np.array_equal(frames["pts"][ix], raw["pts"]):
                raise ValueError("原视频帧号与PTS不一致")
    cursor, longest = 0., 0.
    intervals = t[z["time_valid_mask"]]
    for a, b in sorted(intervals.tolist()):
        if b <= a:
            raise ValueError("有效区间持续时间必须为正")
        longest = max(longest, a - cursor)
        cursor = max(cursor, b)
    longest = max(longest, media["duration"] - cursor)
    if cfg["aggregation"]["retain_gaps"] and longest > cfg["aggregation"]["min_gap_seconds"] + 1e-6:
        raise ValueError("存在不应丢失的长时间间隙")
    support = temporal_support(intervals, acoustic_support, visual_support)
    invalid = [w for w in words if not w["time_valid"]]
    counts = {"invalid_words": len(invalid), "gap_positions": int((ids < 0).sum()),
              "text_available_words": int(z["modality_mask"][ids >= 0, 0].sum())}
    for k in REASON_LABELS:
        counts["flag_" + k] = sum(k in w.get("flags", []) for w in invalid) if k != "other" else sum(invalid_reason(w) == "other" for w in invalid)
        counts["primary_" + k] = sum(invalid_reason(w) == k for w in invalid)
    for column, name in enumerate(("text", "audio", "vision")):
        counts[name + "_available_positions"] = int(z["modality_mask"][:, column].sum())
    return {"mapped_seconds": support["timeline_seconds"], "longest_unmapped_seconds": longest,
            "time_valid_words": sum(w["time_valid"] for w in words), **support, **counts}

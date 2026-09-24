"""Pure numerical operations; no model imports or uniform-time guesses."""
from __future__ import annotations

import unicodedata

import numpy as np

# Time comparison tolerance in seconds: far below one 16 kHz sample (62.5 µs), yet absorbs
# float subtraction error so that e.g. 3.28 - 3.2 still counts as an 80 ms gap.
TIME_EPS = 1e-6


def _kept(c):
    return c == "'" or unicodedata.category(c)[0] in "LN"


def word_spans(raw_text, words):
    """Map Qwen's punctuation-stripped English words to original character spans.

    Require full lexical coverage, preserving repeated words and punctuation.
    Never use fuzzy matches to fabricate provenance.
    """
    chars = [(c, i) for i, c in enumerate(raw_text) if _kept(c)]
    canonical = "".join(c for c, _ in chars)
    cursor, spans = 0, []
    for word in words:
        clean = "".join(c for c in word if _kept(c))
        if not clean or canonical[cursor:cursor + len(clean)] != clean:
            raise ValueError(f"对齐词无法严格映射回原文：{word!r}, lexical offset={cursor}")
        spans.append([chars[cursor][1], chars[cursor + len(clean) - 1][1] + 1])
        cursor += len(clean)
    if cursor != len(chars):
        raise ValueError("对齐结果没有覆盖原文全部字母/数字，保留错误以供核查")
    return spans


def pool_tokens(hidden, offsets, spans, attention):
    hidden = np.asarray(hidden)
    offsets = np.asarray(offsets)
    rows, mapping, valid = [], [], []
    for start, end in spans:
        idx = np.flatnonzero((offsets[:, 0] < end) & (offsets[:, 1] > start)
                             & (offsets[:, 1] > offsets[:, 0]) & np.asarray(attention, bool))
        mapping.append(idx.tolist())
        ok = bool(len(idx)) and bool(np.isfinite(hidden[idx]).all())
        valid.append(ok)
        rows.append(hidden[idx].mean(0) if ok else np.zeros(hidden.shape[1]))
    return np.asarray(rows, np.float32), np.asarray(valid, bool), mapping


def union_length(intervals):
    total, end = 0.0, -float("inf")
    for a, b in sorted(intervals):
        if b > max(a, end):
            total += b - max(a, end)
        end = max(end, b)
    return total


def interval_pool(targets, intervals, features, quality):
    """Overlap × quality pooling; coverage is the union, never summed overlap."""
    features = np.asarray(features, np.float32)
    intervals = np.asarray(intervals, np.float64).reshape(-1, 2)
    quality = np.asarray(quality, np.float64)
    if features.ndim != 2 or len(features) != len(intervals) or len(quality) != len(intervals):
        raise ValueError("特征、时间窗与质量向量长度不一致")
    good = (np.isfinite(features).all(1) & np.isfinite(intervals).all(1)
            & (intervals[:, 1] > intervals[:, 0]) & np.isfinite(quality) & (quality > 0))
    means = np.zeros((len(targets), features.shape[1]), np.float32)
    stds, mask = np.zeros_like(means), np.zeros(len(targets), bool)
    coverage = np.zeros(len(targets), np.float32)
    mapping = []
    for k, (start, end) in enumerate(targets):
        overlap = np.maximum(0, np.minimum(end, intervals[:, 1]) - np.maximum(start, intervals[:, 0]))
        idx = np.flatnonzero(good & (overlap > 0)) if end > start else np.array([], int)
        if not len(idx):
            mapping.append({"indices": [], "weights": []})
            continue
        weight = overlap[idx] * quality[idx]
        weight /= weight.sum()
        values = features[idx].astype(np.float64)
        mean = np.sum(values * weight[:, None], 0)
        means[k] = mean
        stds[k] = np.sqrt(np.maximum(0, np.sum((values - mean) ** 2 * weight[:, None], 0)))
        mask[k] = True
        coverage[k] = union_length([(max(start, intervals[j, 0]), min(end, intervals[j, 1]))
                                    for j in idx]) / (end - start)
        mapping.append({"indices": idx.tolist(), "weights": weight.tolist()})
    return means, stds, mask, coverage, mapping


def make_timeline(words, duration, retain_gaps=True, min_gap=0.08):
    entries = [{"kind": "word", "word_index": i, "text": w["text"],
                "start": w["start"], "end": w["end"], "time_valid": w["time_valid"]}
               for i, w in enumerate(words)]
    if retain_gaps:
        cursor = 0.0
        for start, end in sorted((w["start"], w["end"]) for w in words if w["time_valid"]):
            if start - cursor >= min_gap - TIME_EPS:
                entries.append({"kind": "unassigned", "word_index": -1, "text": "",
                                "start": cursor, "end": start, "time_valid": True})
            cursor = max(cursor, end)
        if duration - cursor >= min_gap - TIME_EPS:
            entries.append({"kind": "unassigned", "word_index": -1, "text": "",
                            "start": cursor, "end": duration, "time_valid": True})
    return sorted(entries, key=lambda x: (x["start"], x["end"], x["word_index"]))


def sampled_support(pts, video_start, video_end):
    """Nearest-sampled-frame piecewise-constant support, not original frame duration."""
    pts = np.asarray(pts, np.float64)
    if not len(pts):
        return np.zeros((0, 2))
    if np.any(np.diff(pts) <= 0):
        raise ValueError("采样帧时间戳必须严格递增")
    bounds = np.r_[video_start, (pts[:-1] + pts[1:]) / 2, video_end]
    return np.column_stack([bounds[:-1], bounds[1:]])

"""B: annotation-free alignment check — aligned words vs. an independent VAD.

Silero VAD never sees the transcript, so agreement between its speech frames and the
Qwen3 word intervals is evidence the timestamps land on speech. A circular-shift null
(same word layout, wrong position) calibrates what chance agreement looks like.
"""
from __future__ import annotations

import argparse
import importlib.metadata
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

from ..common import atomic_json, load_config, read_json

GRID = 0.01          # evaluation grid in seconds
WINDOW = 512         # silero frame at 16 kHz = 32 ms
THRESHOLD = 0.5      # silero default speech threshold
MIN_SHIFT = 0.5      # null shifts closer than this to the true position are excluded
RECALL_MIN = 0.5     # words cover under half of detected speech -> likely compressed alignment
P_MAX = 0.05         # reported only: the shift test loses power when speech fills the clip


def speech_probability(model, wave):
    import torch
    model.reset_states()
    pad = (-len(wave)) % WINDOW
    frames = np.pad(wave, (0, pad)).reshape(-1, WINDOW)
    with torch.inference_mode():
        return np.asarray([float(model(torch.from_numpy(f), 16000)) for f in frames])


def grid_masks(words, probs, media):
    n = int(np.ceil(media["duration"] / GRID))
    centers = (np.arange(n) + .5) * GRID
    frame = np.floor((centers - media["audio_offset"]) * 16000 / WINDOW).astype(int)
    inside = (frame >= 0) & (frame < len(probs))
    speech = np.zeros(n, bool)
    speech[inside] = probs[frame[inside]] >= THRESHOLD
    word = np.zeros(n, bool)
    per_word = []
    for w in words:
        if not w["time_valid"]:
            continue
        span = (centers >= w["start"]) & (centers < w["end"])
        word |= span
        if span.any():
            per_word.append(float(speech[span].mean()))
    return word, speech, per_word


def agreement(word, speech):
    both = np.count_nonzero(word & speech)
    union = np.count_nonzero(word | speech)
    return (both / max(np.count_nonzero(word), 1), both / max(np.count_nonzero(speech), 1),
            both / union if union else np.nan)


def shift_null(word, speech):
    """IoU for every circular shift at least MIN_SHIFT away from the observed layout."""
    n, k0 = len(word), int(round(MIN_SHIFT / GRID))
    return np.asarray([agreement(np.roll(word, k), speech)[2] for k in range(k0, n - k0 + 1)])


def evaluate(cfg):
    from silero_vad import load_silero_vad
    model = load_silero_vad()
    root = Path(cfg["output_dir"])
    samples = read_json(root / "manifest.json")["samples"]
    summary = pd.read_csv(root / "summary.csv").set_index("sample_id")
    rows = []
    for s in samples:
        d = root / "cache" / s["key"]
        media = read_json(d / "media.json")
        words = read_json(d / "align.json")["words"]
        wave, sr = sf.read(d / "audio.wav", dtype="float32")
        if sr != 16000:
            raise ValueError("VAD需要16 kHz输入")
        word, speech, per_word = grid_masks(words, speech_probability(model, wave), media)
        row = {"sample_id": s["id"], "duration": media["duration"],
               "valid_words": len(per_word), "word_seconds": word.sum() * GRID,
               "speech_seconds": speech.sum() * GRID, "silent_audio": media["silent_audio"],
               "alignment_suspect": bool(summary.loc[s["id"], "alignment_suspect"])}
        if speech.any() and word.any() and not media["silent_audio"]:
            precision, recall, iou = agreement(word, speech)
            null = shift_null(word, speech)
            row.update(precision=precision, recall=recall, iou=iou,
                       words_on_speech=float(np.mean(np.asarray(per_word) >= .5)),
                       null_iou_mean=float(np.nanmean(null)) if len(null) else np.nan,
                       null_shifts=len(null),
                       p_value=float((1 + np.sum(null >= iou)) / (1 + len(null))) if len(null) else np.nan)
        rows.append(row)
    table = pd.DataFrame(rows)
    reasons = []
    for (_, r), s in zip(table.iterrows(), samples):
        found = [x for x in str(summary.loc[s["id"], "suspect_reasons"]).split(";") if x and x != "nan"]
        if np.isnan(r.get("iou", np.nan)):
            found += [] if r.silent_audio else ["vad_unscored"]
        else:
            found += ["vad_low_recall"] if r.recall < RECALL_MIN else []
        reasons.append(";".join(found))
    table["review_reasons"] = reasons
    table["needs_review"] = table.review_reasons != ""
    out = root / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "alignment_vad.csv", index=False, encoding="utf-8-sig")
    scored = table.dropna(subset=["iou"])
    normal = scored[~scored.alignment_suspect]
    suspect = scored[scored.alignment_suspect]

    def describe(part):
        return {"n": len(part), **{f"{c}_median": float(part[c].median()) for c in
                ("precision", "recall", "iou", "words_on_speech", "null_iou_mean")},
                f"p_below_{P_MAX}": int((part.p_value < P_MAX).sum())}

    report = {"method": {"vad": "silero-vad " + importlib.metadata.version("silero-vad"),
                         "frame_seconds": WINDOW / 16000, "threshold": THRESHOLD, "grid_seconds": GRID,
                         "null": f"循环平移词掩码，排除|shift|<{MIN_SHIFT}s；p=(1+#null≥obs)/(1+#null)",
                         "precision": "词时间中被VAD判为语音的比例", "recall": "VAD语音中被词覆盖的比例",
                         "review_rule": f"结构标记 ∪ recall<{RECALL_MIN} ∪ 非静音却无法评分；"
                                        f"p值仅报告：语音占满片段时平移检验无区分力"},
              "scored": len(scored), "unscored_ids": table[table.iou.isna()].sample_id.tolist(),
              "all": describe(scored), "not_suspect": describe(normal), "suspect": describe(suspect),
              "review_ids": table[table.needs_review].sample_id.tolist()}
    atomic_json(out / "alignment_vad.json", report)
    return report, table


def main():
    parser = argparse.ArgumentParser(description="对齐结果与独立VAD的一致性检验")
    parser.add_argument("--config", default=str(Path(__file__).parents[1] / "config.yaml"))
    report, _ = evaluate(load_config(parser.parse_args().config))
    import json
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

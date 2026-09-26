"""Map attachment 4 evidence positions to original words and clip timestamps.

Feature rows are aligned to BERT token positions; word times come from the
same Qwen forced aligner used in question 1. Alignment is audited explicitly.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

from q1.backends import Aligner
from q1.media import prepare_media

from .core import DATA, OUT, MODEL_ID, MODEL_REVISION, read_pkl, save_csv, save_json


def load_q1_config():
    import yaml
    cfg = yaml.safe_load((OUT.parent.parent / "q1/config.yaml").read_text())
    cfg["device"] = "cuda:0"
    return cfg


def map_one(path, tokenizer, aligner, cfg):
    raw = read_pkl(path)
    sid = path.stem
    video = path.parent / "videos" / f"{sid}.mp4"
    directory = OUT / "align_cache" / sid
    directory.mkdir(parents=True, exist_ok=True)
    if not (directory / "media.json").exists():
        prepare_media({"video": str(video)}, directory, cfg)
    if not (directory / "align.json").exists():
        aligner.process({"raw_text": str(raw["raw_text"])}, directory, cfg)
    media = json.loads((directory / "media.json").read_text())
    words = json.loads((directory / "align.json").read_text())["words"]
    text = str(raw["raw_text"])
    tok = tokenizer(text, max_length=50, padding="max_length", truncation=True,
                    return_offsets_mapping=True)
    expected = np.asarray(raw["text_bert"])[0].astype(int)
    actual = np.asarray(tok["input_ids"]).astype(int)
    id_match = bool(np.array_equal(expected, actual))
    mapping = []
    for i, (a,b) in enumerate(tok["offset_mapping"]):
        candidates = [w for w in words if w["time_valid"] and
                      max(0, min(b,w["char_span"][1])-max(a,w["char_span"][0])) > 0]
        if a == b or not candidates:
            mapping.append(None)
        else:
            w = candidates[0]
            mapping.append({"token": text[a:b], "word": w["text"],
                            "start_seconds": w["start"], "end_seconds": w["end"]})
    return {"sample_id": sid, "video": str(video), "video_duration_seconds": media["duration"],
            "token_id_match": id_match, "matched_positions": sum(x is not None for x in mapping),
            "positions": mapping, "invalid_word_count": sum(not w["time_valid"] for w in words)}


def main():
    cfg = load_q1_config()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION, use_fast=True)
    folder = DATA / "附件4-可解释专项视频样本与特征文件/附件4-可解释专项视频样本与特征文件/对齐版本"
    aligner = Aligner(cfg)
    mappings = []
    for path in sorted(folder.glob("*.pkl")):
        result = map_one(path, tokenizer, aligner, cfg)
        mappings.append(result)
        print(path.stem, result["token_id_match"], result["matched_positions"], flush=True)
    save_json(OUT / "attachment4_position_to_time.json", mappings)
    details_path = OUT / "attachment4_explanations.json"
    details = json.loads(details_path.read_text())
    by_id = {x["sample_id"]: x for x in mappings}
    rows = []
    for exp in details:
        timing = by_id[exp["sample_id"]]
        for w in exp["windows"]:
            positions = timing["positions"][w["start_position"]:w["end_position_exclusive"]]
            observed = [p for p in positions if p is not None]
            row = {"sample_id": exp["sample_id"], "polarity": exp["predicted_label"],
                   "intensity": round(exp["intensity"], 6), "main_modality": exp["main_modality"],
                   "evidence_modality": w["modality"], "start_position": w["start_position"],
                   "end_position_exclusive": w["end_position_exclusive"],
                   "target_logit_drop": round(w["target_logit_drop"], 6),
                   "token_id_match": timing["token_id_match"],
                   "text_span": " ".join(p["token"] for p in observed),
                   "start_seconds": round(min(p["start_seconds"] for p in observed), 4) if observed else "",
                   "end_seconds": round(max(p["end_seconds"] for p in observed), 4) if observed else "",
                   "time_status": "aligned" if observed and timing["token_id_match"] else "unresolved"}
            rows.append(row)
    save_csv(OUT / "attachment4_evidence_times.csv", rows)
    with (OUT / "attachment4_predictions_explanations.csv").open(encoding="utf-8-sig", newline="") as f:
        summary = list(csv.DictReader(f))
    for sample in summary:
        candidates = [r for r in rows if r["sample_id"] == sample["sample_id"]
                      and r["evidence_modality"] == sample["main_modality"]
                      and r["time_status"] == "aligned"]
        if candidates:
            strongest = max(candidates, key=lambda r: r["target_logit_drop"])
            sample.update({"main_evidence_text": strongest["text_span"],
                           "main_evidence_start_s": strongest["start_seconds"],
                           "main_evidence_end_s": strongest["end_seconds"],
                           "main_evidence_logit_drop": strongest["target_logit_drop"]})
        else:
            sample.update({"main_evidence_text": "", "main_evidence_start_s": "",
                           "main_evidence_end_s": "", "main_evidence_logit_drop": ""})
    save_csv(OUT / "attachment4_predictions_explanations.csv", summary)
    save_json(OUT / "attachment4_alignment_audit.json", {
        "samples": len(mappings), "token_id_matches": sum(x["token_id_match"] for x in mappings),
        "timed_windows": sum(r["time_status"] == "aligned" for r in rows),
        "windows": len(rows),
        "note": "Word times are forced alignment estimates; correspondence from provided feature rows to BERT token positions is checked by exact token ID match."})


if __name__ == "__main__":
    main()

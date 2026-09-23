from __future__ import annotations

import csv
import gc
import json
import logging
import time
from pathlib import Path

import numpy as np

from .common import atomic_json, atomic_npz, code_hash, digest, environment, file_hash, read_json, seed_runtime
from .media import prepare_media
from .temporal import interval_pool, make_timeline

LOG = logging.getLogger("q1")
STAGES = ("media", "align", "text", "audio", "vision", "aggregate")
DEPS = {"media": (), "align": ("media",), "text": ("align",), "audio": ("media",),
        "vision": ("media",), "aggregate": ("media", "align", "text", "audio", "vision")}
FILES = {"media": ("audio.wav", "frames.npz", "media.json"), "align": ("align.json",),
         "text": ("text.npz", "text.json"), "audio": ("audio_features.npz", "audio_features.json"),
         "vision": ("vision.npz", "vision.json")}


class Cache:
    def __init__(self, cfg):
        self.root = Path(cfg["output_dir"])
        self.base = {"config": cfg, "code_sha256": code_hash(), "environment": environment()}

    def directory(self, sample):
        return self.root / "cache" / sample["key"]

    def marker(self, sample, stage):
        return self.directory(sample) / (stage + ".meta.json")

    def files(self, sample, stage):
        if stage == "aggregate":
            return [self.root / "features" / (sample["key"] + suffix) for suffix in (".npz", ".json")]
        return [self.directory(sample) / name for name in FILES[stage]]

    def signature(self, sample, stage):
        deps = {}
        for dep in DEPS[stage]:
            if not self.valid(sample, dep):
                raise ValueError(f"{sample['id']} 的 {dep} 缺失、损坏或已过期；先重跑该阶段")
            deps[dep] = read_json(self.marker(sample, dep))
        return digest({**self.base, "sample": sample, "stage": stage, "dependencies": deps})

    def valid(self, sample, stage):
        try:
            mark = read_json(self.marker(sample, stage))
            if mark["signature"] != self.signature(sample, stage):
                return False
            files = self.files(sample, stage)
            return all(p.is_file() and mark["files"][str(p.relative_to(self.root))] == file_hash(p) for p in files)
        except (OSError, ValueError, KeyError, TypeError):
            return False

    def commit(self, sample, stage, signature, result):
        atomic_json(self.marker(sample, stage), {"signature": signature, "result": result,
            "files": {str(p.relative_to(self.root)): file_hash(p) for p in self.files(sample, stage)}})


def csr(mapping):
    """Ragged per-position source lists -> offsets/indices/weights arrays."""
    offsets = np.cumsum([0] + [len(m["indices"]) for m in mapping]).astype(np.int64)
    indices = np.asarray([i for m in mapping for i in m["indices"]], np.int32)
    weights = np.asarray([w for m in mapping for w in m["weights"]], np.float32)
    return offsets, indices, weights


def aggregate(sample, directory, cfg):
    media = read_json(directory / "media.json")
    alignment = read_json(directory / "align.json")
    text_meta = read_json(directory / "text.json")
    audio_meta = read_json(directory / "audio_features.json")
    vision_meta = read_json(directory / "vision.json")
    timeline = make_timeline(alignment["words"], media["duration"],
                             cfg["aggregation"]["retain_gaps"], cfg["aggregation"]["min_gap_seconds"])
    times = np.asarray([[r["start"], r["end"]] for r in timeline], np.float64)
    time_valid = np.asarray([r["time_valid"] for r in timeline], bool)
    # Invalid timestamps retain their row, but never contribute audio/visual provenance.
    pool_times = times.copy()
    pool_times[~time_valid, 1] = pool_times[~time_valid, 0]
    word_ids = np.asarray([r["word_index"] for r in timeline], np.int32)
    with np.load(directory / "text.npz", allow_pickle=False) as raw:
        text = np.zeros((len(timeline), raw["features"].shape[1]), np.float32)
        text_valid = np.zeros(len(timeline), bool)
        for k, index in enumerate(word_ids):
            if index >= 0:
                text[k] = raw["features"][index]
                text_valid[k] = raw["valid"][index]
    with np.load(directory / "audio_features.npz", allow_pickle=False) as raw:
        emo, _, emask, ecov, emap = interval_pool(pool_times, raw["emotion_intervals"],
                                                  raw["emotion"], raw["emotion_valid"])
        lld, lstd, lmask, lcov, lmap = interval_pool(pool_times, raw["acoustic_intervals"],
                                                   raw["acoustic"], raw["acoustic_valid"])
        source_intervals = {"emotion": raw["emotion_intervals"], "acoustic": raw["acoustic_intervals"]}
    with np.load(directory / "vision.npz", allow_pickle=False) as raw:
        vis, vstd, vmask, vcov, vmap = interval_pool(pool_times, raw["intervals"],
                                                    raw["features"], raw["quality"])
        source_intervals["vision"] = raw["intervals"]
    audio = np.concatenate([emo, lld, lstd], axis=1)
    vision = np.concatenate([vis, vstd], axis=1)
    component_mask = np.column_stack([text_valid, emask, lmask, vmask])
    # Audio is usable if at least one audio feature family is available; component_mask
    # is mandatory when consuming its 768+25+25 feature blocks.
    modality_mask = np.column_stack([text_valid, emask | lmask, vmask])
    coverage = np.column_stack([ecov, lcov, vcov])
    dtype = np.dtype(cfg["aggregation"]["export_dtype"])
    converted = {}
    for name, values in (("text", text), ("audio", audio), ("vision", vision)):
        if not np.isfinite(values).all() or np.any(np.abs(values) > np.finfo(dtype).max):
            raise ValueError(f"{name}含非有限值或超出{dtype}范围，请检查或使用float32")
        converted[name] = values.astype(dtype)
    provenance = {}
    for name, mapping in (("emotion", emap), ("acoustic", lmap), ("vision", vmap)):
        offsets, indices, weights = csr(mapping)
        provenance.update({f"source_{name}_intervals": np.asarray(source_intervals[name], np.float64),
                           f"map_{name}_offsets": offsets, f"map_{name}_indices": indices,
                           f"map_{name}_weights": weights})
    out = Path(cfg["output_dir"]) / "features" / sample["key"]
    atomic_npz(Path(str(out) + ".npz"), **converted, timestamps=times,
               valid_mask=np.ones(len(timeline), bool), time_valid_mask=time_valid,
               modality_mask=modality_mask, component_mask=component_mask,
               coverage=coverage, word_indices=word_ids, length=np.asarray(len(timeline), np.int32),
               **provenance)
    alignment_ratio = sum(w["time_valid"] for w in alignment["words"]) / len(alignment["words"])
    reasons = (["silent_audio"] if media["silent_audio"] else []) + (
        ["low_valid_alignment"] if alignment_ratio < cfg["aggregation"]["alignment_suspect_ratio"] else [])
    result = {"sample_id": sample["id"], "duration": media["duration"], "positions": len(timeline),
        "words": len(alignment["words"]), "text_dim": text.shape[1], "audio_dim": audio.shape[1],
        "vision_dim": vision.shape[1], "valid_text_ratio": float(text_valid.mean()),
        "valid_audio_ratio": float((emask | lmask).mean()), "valid_vision_ratio": float(vmask.mean()),
        "valid_alignment_ratio": alignment_ratio, "silent_audio": media["silent_audio"],
        "alignment_suspect": bool(reasons), "suspect_reasons": ";".join(reasons)}
    atomic_json(Path(str(out) + ".json"), {"schema_version": 2, "summary": result,
        "source": {k: sample[k] for k in ("id", "video_id", "clip_id", "video_sha256", "raw_text")},
        "time_origin_pts": media["origin_pts"], "timeline": timeline, "words": alignment["words"],
        "token_metadata": text_meta, "audio_metadata": audio_meta, "vision_metadata": vision_meta,
        "provenance": "NPZ中map_<m>_offsets/indices/weights为CSR：位置k的源索引为indices[offsets[k]:offsets[k+1]]，"
                      "对应归一化权重同位置；源索引指向source_<m>_intervals及cache中的源特征行。"
                      "文本位置到token的映射见token_metadata.word_to_tokens[word_indices[k]]",
        "dimensions": {"text": text.shape[1], "audio": audio.shape[1], "vision": vision.shape[1]},
        "audio_layout": {"emotion_mean": [0, emo.shape[1]], "lld_mean": [emo.shape[1], emo.shape[1] + lld.shape[1]],
                         "lld_std": [emo.shape[1] + lld.shape[1], audio.shape[1]]},
        "mask_columns": {"modality_mask": ["text", "audio", "vision"],
                         "component_mask": ["text", "emotion", "acoustic", "vision"],
                         "coverage": ["emotion", "acoustic", "vision"]},
        "padding": "变长保存，无预填充；batch padding见q1.dataset.collate",
        "gap_meaning": "未分配到有效词的区间；不自动推断为静音或笑声",
        "config_sha256": digest(cfg), "run_record": "../run_config.json"}, compact=True)
    return result


def default_backend(stage, cfg):
    from .backends import Aligner, AudioEncoder, TextEncoder, VisionEncoder
    return {"align": Aligner, "text": TextEncoder, "audio": AudioEncoder, "vision": VisionEncoder}[stage](cfg)


def execute(cfg, samples, stages=STAGES, force=False, fail_fast=False, factory=default_backend):
    seed_runtime(cfg["seed"])
    cache = Cache(cfg)
    cache.root.mkdir(parents=True, exist_ok=True)
    atomic_json(cache.root / "run_config.json", cache.base)
    failed = 0
    for stage in stages:
        pending = [s for s in samples if force or not cache.valid(s, stage)]
        LOG.info("%s: %d pending, %d cached", stage, len(pending), len(samples) - len(pending))
        if not pending:
            continue
        for sample in pending:
            cache.marker(sample, stage).unlink(missing_ok=True)
        backend, init_error = None, None
        if stage not in ("media", "aggregate"):
            try:
                backend = factory(stage, cfg)
            except Exception as exc:
                init_error = exc
                LOG.exception("%s backend initialization failed", stage)
        for sample in pending:
            began = time.monotonic()
            event = {"time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                     "stage": stage, "sample_id": sample["id"]}
            try:
                if init_error:
                    raise RuntimeError(f"{stage}后端初始化失败：{init_error}") from init_error
                signature = cache.signature(sample, stage)
                directory = cache.directory(sample)
                if stage == "media":
                    result = prepare_media(sample, directory, cfg)
                elif stage == "aggregate":
                    result = aggregate(sample, directory, cfg)
                else:
                    result = backend.process(sample, directory, cfg)
                cache.commit(sample, stage, signature, result)
                event.update(status="ok", seconds=time.monotonic() - began)
                LOG.info("%s %s: ok (%.2fs)", stage, sample["id"], event["seconds"])
            except Exception as exc:
                failed += 1
                event.update(status="failed", error=f"{type(exc).__name__}: {exc}")
                LOG.exception("%s %s: failed", stage, sample["id"])
                if fail_fast:
                    raise
            finally:
                with (cache.root / "events.jsonl").open("a", encoding="utf-8") as f:
                    f.write(json.dumps(event, ensure_ascii=False) + "\n")
        del backend
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
    return failed


def validate(cfg, samples):
    cache = Cache(cfg)
    selected = len(samples) != cfg.get("expected_samples", len(samples))
    suffix = ".selected" if selected else ""
    rows, failures = [], []
    for sample in samples:
        row = {"sample_id": sample["id"], "status": "pending"}
        if cache.valid(sample, "aggregate"):
            try:
                paths = cache.files(sample, "aggregate")
                meta = read_json(paths[1])
                with np.load(paths[0], allow_pickle=False) as data:
                    n = int(data["length"])
                    if n < 1 or n != len(meta["timeline"]):
                        raise ValueError("时间轴和有效长度不一致")
                    if data["timestamps"].shape != (n, 2) or data["modality_mask"].shape != (n, 3):
                        raise ValueError("时间戳或模态掩码形状错误")
                    for modality, column in (("text", 0), ("audio", 1), ("vision", 2)):
                        x = data[modality]
                        if x.shape != (n, meta["dimensions"][modality]) or not np.isfinite(x).all():
                            raise ValueError(f"{modality}的维数或数值异常")
                        if np.any(x[~data["modality_mask"][:, column]] != 0):
                            raise ValueError(f"{modality}不可用位置没有置零")
                    if np.any(data["coverage"] < 0) or np.any(data["coverage"] > 1 + 1e-6):
                        raise ValueError("覆盖率不在[0,1]")
                    if not data["valid_mask"].all():
                        raise ValueError("单样本存储不应含padding")
                    for name in ("emotion", "acoustic", "vision"):
                        offsets = data[f"map_{name}_offsets"]
                        indices = data[f"map_{name}_indices"]
                        if (offsets.shape != (n + 1,) or offsets[0] != 0 or np.any(np.diff(offsets) < 0)
                                or offsets[-1] != len(indices) or len(data[f"map_{name}_weights"]) != len(indices)
                                or (len(indices) and indices.max() >= len(data[f"source_{name}_intervals"]))):
                            raise ValueError(f"{name}来源映射不一致")
                row.update(meta["summary"], status="ok", bytes=sum(p.stat().st_size for p in paths))
            except Exception as exc:
                row.update(status="invalid", error=str(exc))
                failures.append(sample["id"])
        else:
            row["stages"] = ",".join(s for s in STAGES if cache.valid(sample, s))
            failures.append(sample["id"])
        rows.append(row)
    fields = list(dict.fromkeys(k for row in rows for k in row))
    path = cache.root / ("summary" + suffix + ".csv")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    report = {"scope": "selected" if selected else "all", "expected": len(samples),
              "complete": len(samples) - len(failures), "incomplete_ids": failures,
              "feature_bytes": sum(r.get("bytes", 0) for r in rows),
              "submission_limit_bytes": 50_000_000,
              "silent_audio_ids": [r["sample_id"] for r in rows if r.get("silent_audio")],
              "alignment_suspect_ids": [r["sample_id"] for r in rows if r.get("alignment_suspect")],
              "note": "大小仅包括选中样本的特征及映射，50MB还需预留代码和问题2/3材料；结构验证不代表时间精度验证"}
    atomic_json(cache.root / ("validation" + suffix + ".json"), report)
    return report

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

from .common import atomic_json, atomic_npz, file_hash, run_command


def build_manifest(cfg):
    root = Path(cfg["video_root"])
    df = pd.read_excel(cfg["labels"], sheet_name="label", dtype={"video_id": str, "clip_id": str})
    required = {"video_id", "clip_id", "text", "label", "annotation"}
    if not required.issubset(df.columns) or df[list(required)].isna().any().any():
        raise ValueError("标注表缺少必需字段或含空值")
    rows, keys = [], set()
    for r in df.to_dict("records"):
        video_id, clip_id = r["video_id"], r["clip_id"]
        if any(c in video_id + clip_id for c in "/\\") or video_id in (".", ".."):
            raise ValueError("非法样本ID")
        key = video_id + "__" + clip_id
        if key in keys:
            raise ValueError(f"重复样本：{key}")
        keys.add(key)
        video = (root / video_id / (clip_id + ".mp4")).resolve()
        if not video.is_file():
            raise FileNotFoundError(video)
        rows.append({"key": key, "id": video_id + "$_$" + clip_id, "video_id": video_id,
                     "clip_id": clip_id, "video": str(video), "video_sha256": file_hash(video),
                     "raw_text": r["text"], "label": float(r["label"]), "annotation": r["annotation"]})
    actual = {str(p.resolve()) for p in root.rglob("*.mp4")}
    expected = {r["video"] for r in rows}
    if actual != expected:
        raise ValueError(f"视频/标签不一一对应：未标注视频={sorted(actual - expected)}")
    if len(rows) != cfg["expected_samples"]:
        raise ValueError(f"期待{cfg['expected_samples']}条，实际{len(rows)}条")
    manifest = {"samples": rows, "label_sha256": file_hash(cfg["labels"]), "count": len(rows)}
    atomic_json(Path(cfg["output_dir"]) / "manifest.json", manifest)
    return manifest


def prepare_media(sample, directory, cfg):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = sample["video"]
    probe = json.loads(run_command(["ffprobe", "-v", "error", "-show_streams", "-show_format",
                                    "-of", "json", path]))
    videos = [s for s in probe["streams"] if s["codec_type"] == "video"]
    audios = [s for s in probe["streams"] if s["codec_type"] == "audio"]
    if not videos or not audios:
        raise ValueError("缺少音频或视频流；保留样本并记录失败，不生成伪特征")
    frames = json.loads(run_command(["ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_frames", "-show_entries", "frame=best_effort_timestamp_time,duration_time,pkt_duration_time",
        "-of", "json", path]))["frames"]
    pts = np.array([float(f["best_effort_timestamp_time"]) for f in frames], np.float64)
    if not len(pts) or np.any(np.diff(pts) <= 0):
        raise ValueError("视频PTS缺失、重复或非递增，需人工检查")
    audio_frames = json.loads(run_command(["ffprobe", "-v", "error", "-select_streams", "a:0",
        "-show_frames", "-show_entries", "frame=best_effort_timestamp_time,nb_samples",
        "-of", "json", path]))["frames"]
    if not audio_frames:
        raise ValueError("没有可解码的音频帧")
    audio_pts = np.asarray([float(f["best_effort_timestamp_time"]) for f in audio_frames])
    source_sr = int(audios[0]["sample_rate"])
    source_counts = np.asarray([int(f["nb_samples"]) for f in audio_frames])
    if len(audio_pts) > 1 and np.any(abs(np.diff(audio_pts) - source_counts[:-1] / source_sr) > 2 / source_sr):
        raise ValueError("音频PTS有不连续区间，不能用连续采样下标冒充时间；需明确补隙规则后重跑")
    audio_start = float(audio_pts[0])
    origin = min(float(pts[0]), audio_start)
    pts -= origin
    audio_offset = audio_start - origin
    fd, temporary = tempfile.mkstemp(dir=directory, suffix=".wav")
    os.close(fd)
    try:
        run_command(["ffmpeg", "-v", "error", "-nostdin", "-y", "-i", path,
                     "-map", "0:a:0", "-vn", "-ac", "1", "-ar", cfg["media"]["sample_rate"],
                     "-c:a", "pcm_f32le", temporary])
        wave, sr = sf.read(temporary, dtype="float32")
        if wave.ndim != 1 or not len(wave) or not np.isfinite(wave).all():
            raise ValueError("解码音频无效")
        os.replace(temporary, directory / "audio.wav")
    finally:
        Path(temporary).unlink(missing_ok=True)
    tail = frames[-1].get("duration_time", frames[-1].get("pkt_duration_time"))
    frame_step = float(tail) if tail is not None else (float(np.median(np.diff(pts))) if len(pts) > 1 else 0)
    if frame_step <= 0:
        raise ValueError("无法确定末帧持续时间")
    end = float(pts[-1] + frame_step)
    intervals = np.column_stack([pts, np.r_[pts[1:], end]])
    atomic_npz(directory / "frames.npz", pts=pts, intervals=intervals)
    peak = float(np.abs(wave).max())
    info = {"origin_pts": origin, "audio_offset": audio_offset, "audio_samples": len(wave),
            "sample_rate": sr, "audio_duration": len(wave) / sr,
            "duration": max(end, audio_offset + len(wave) / sr),
            "video_start": float(pts[0]), "video_end": end, "video_frames": len(pts),
            "width": videos[0]["width"], "height": videos[0]["height"],
            "audio_rms": float(np.sqrt(np.mean(wave.astype(np.float64) ** 2))),
            "audio_peak": peak, "silent_audio": peak <= cfg["media"]["silence_peak"],
            "source_streams": probe["streams"],
            "ffmpeg_version": run_command(["ffmpeg", "-version"]).splitlines()[0],
            "ffprobe_version": run_command(["ffprobe", "-version"]).splitlines()[0],
            "time_rule": "original video PTS minus origin; decoded audio index / sample_rate + audio_offset"}
    atomic_json(directory / "media.json", info)
    return info

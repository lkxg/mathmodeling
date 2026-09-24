from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import random
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import yaml


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def atomic_json(path, data, compact=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, allow_nan=False,
                      **({"separators": (",", ":")} if compact else {"indent": 2}))
            f.write("\n")
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def atomic_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            np.savez_compressed(f, **arrays)
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def digest(data):
    return hashlib.sha256(json.dumps(data, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def run_command(args):
    proc = subprocess.run([str(x) for x in args], capture_output=True, text=True)
    if proc.returncode:
        raise RuntimeError(f"{args[0]} failed ({proc.returncode}): {proc.stderr[-4000:]}")
    return proc.stdout


def load_config(path):
    path = Path(path).resolve()
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    root = (path.parent / cfg.pop("root_dir", "..")).resolve()
    for key in ("video_root", "labels", "output_dir"):
        cfg[key] = str((root / cfg[key]).resolve())
    if cfg.get("quality_review"):
        cfg["quality_review"] = str((root / cfg["quality_review"]).resolve())
    if cfg["media"]["sample_rate"] != 16000:
        raise ValueError("强制对齐与声学提取使用16 kHz输入；sample_rate必须为16000")
    if cfg["vision"]["sample_fps"] <= 0 or cfg["vision"]["max_side"] < 32:
        raise ValueError("视觉采样率必须为正，max_side必须至少为32")
    if cfg["aggregation"]["export_dtype"] not in ("float16", "float32"):
        raise ValueError("export_dtype仅支持float16 / float32")
    for name, model in cfg["models"].items():
        if len(model.get("revision", "")) != 40:
            raise ValueError(f"{name}必须固定40位模型提交号，不能使用浮动的main")
    return cfg


def environment():
    versions = {}
    for pkg in ("torch", "torchvision", "torchaudio", "transformers", "numpy", "pandas",
                "opensmile", "openface-test", "timm", "huggingface_hub"):
        try:
            versions[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            versions[pkg] = None
    return versions


def code_hash():
    # Presentation and validation edits do not change extracted feature values.
    names = ("backends.py", "common.py", "media.py", "pipeline.py", "temporal.py")
    return digest({name: file_hash(Path(__file__).with_name(name)) for name in names})


def seed_runtime(seed):
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False


def snapshot(spec, patterns):
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(repo_id=spec["id"], revision=spec["revision"],
                                  allow_patterns=patterns))

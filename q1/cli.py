from __future__ import annotations

import argparse
import fcntl
import importlib
import json
import logging
import shutil
import sys
from pathlib import Path

from .common import environment, load_config, snapshot
from .media import build_manifest
from .pipeline import STAGES, execute, validate


def select_samples(manifest, limit=None, ids=None):
    samples = manifest["samples"]
    if ids:
        ids = set(ids)
        samples = [s for s in samples if s["id"] in ids or s["key"] in ids]
        found = {name for s in samples for name in (s["id"], s["key"])}
        if ids - found:
            raise ValueError(f"未知样本ID：{sorted(ids - found)}")
    if limit is not None:
        if limit < 1:
            raise ValueError("limit必须大于0")
        samples = samples[:limit]
    return samples


def doctor(cfg):
    status = {"packages": environment(), "commands": {p: shutil.which(p) for p in ("ffmpeg", "ffprobe")},
              "imports": {}}
    for name in ("opensmile", "funasr", "openface.face_detection", "openface.multitask_model",
                 "transformers.models.qwen3_asr.processing_qwen3_asr"):
        try:
            importlib.import_module(name)
            status["imports"][name] = "ok"
        except Exception as exc:
            status["imports"][name] = f"{type(exc).__name__}: {exc}"
    try:
        import torch
        status["cuda"] = {"available": torch.cuda.is_available(), "requested": cfg["device"],
                          "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
    except ImportError:
        status["cuda"] = {"available": False}
    print(json.dumps(status, ensure_ascii=False, indent=2))
    return int(not all(status["commands"].values()) or any(v != "ok" for v in status["imports"].values())
               or (cfg["device"].startswith("cuda") and not status["cuda"]["available"]))


def fetch(cfg):
    patterns = {"align": ["*.json", "*.jinja", "*.safetensors"],
                "text": ["*.json", "model.safetensors"],
                "emotion": ["*.yaml", "*.json", "*.pt"],
                "face": ["Alignment_RetinaFace.pth", "MTL_backbone.pth"]}
    for name, spec in cfg["models"].items():
        print(f"Fetching {name}: {spec['id']} @ {spec['revision']}", flush=True)
        print(snapshot(spec, patterns[name]), flush=True)


def print_report(report):
    # The complete list stays in validation.json; keep the console readable.
    brief = {k: v for k, v in report.items() if k != "incomplete_ids"}
    brief["incomplete_count"] = len(report["incomplete_ids"])
    brief["incomplete_examples"] = report["incomplete_ids"][:5]
    print(json.dumps(brief, ensure_ascii=False, indent=2))


def main(argv=None):
    parser = argparse.ArgumentParser(description="E题第一问：三模态特征与词级时间对齐")
    parser.add_argument("--config", default=str(Path(__file__).with_name("config.yaml")))
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("doctor", help="检查运行依赖，不下载模型")
    sub.add_parser("fetch-models", help="下载配置中固定版本的四组权重")
    sub.add_parser("manifest", help="核验全部视频/标签，输出样本清单")
    run = sub.add_parser("run", help="执行全部或指定阶段，自动复用有效缓存")
    run.add_argument("--stage", choices=("all", *STAGES), default="all")
    run.add_argument("--force", action="store_true", help="重建指定阶段；下游缓存自动失效")
    run.add_argument("--fail-fast", action="store_true")
    check = sub.add_parser("validate", help="检查完整性、形状、掩码及缓存；不评判时间戳精度")
    for p in (run, check):
        p.add_argument("--limit", type=int)
        p.add_argument("--id", dest="ids", action="append", help="按sample ID筛选，可重复；负号开头请用--id=...")
    preview = sub.add_parser("preview", help="生成一个样本的时间对齐与原视频帧图")
    preview.add_argument("--id", required=True)
    preview.add_argument("--start", type=float, default=0)
    preview.add_argument("--end", type=float)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    try:
        cfg = load_config(args.config)
        if args.command == "doctor":
            return doctor(cfg)
        if args.command == "fetch-models":
            fetch(cfg)
            return 0
        output = Path(cfg["output_dir"])
        output.mkdir(parents=True, exist_ok=True)
        with (output / ".run.lock").open("w") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError("同一输出目录已有运行中的任务，请等待或使用另一配置目录") from exc
            manifest = build_manifest(cfg)
            if args.command == "manifest":
                print(f"Verified {manifest['count']} samples: {output / 'manifest.json'}")
                return 0
            if args.command == "preview":
                from .preview import render
                sample = select_samples(manifest, ids=[args.id])[0]
                print(render(cfg, sample, args.start, args.end))
                return 0
            samples = select_samples(manifest, args.limit, args.ids)
            if args.command == "run":
                stages = STAGES if args.stage == "all" else (args.stage,)
                failures = execute(cfg, samples, stages, args.force, args.fail_fast)
                report = validate(cfg, manifest["samples"])
                print_report(report)
                return int(failures > 0)
            report = validate(cfg, samples)
            print_report(report)
            return int(report["complete"] != report["expected"])
    except Exception as exc:
        logging.exception("命令失败：%s", exc)
        return 1

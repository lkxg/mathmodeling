"""Compare standard utterance-level 88 Functionals with two 50-D controls.

This module does not modify the primary feature pipeline or probe_v2 results.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import opensmile
import pandas as pd
import soundfile as sf
from sklearn.metrics import confusion_matrix
from threadpoolctl import threadpool_limits

from ...common import atomic_json, atomic_npz, digest, file_hash, load_config, read_json
from ...pipeline import Cache
from ..probe import (ALPHAS, analysis_hash, baseline_predictions, bootstrap_scores,
                     clean_json, interval)
from ..probe_cv import (CLASSES, batch_metrics, cluster_bootstrap_indices, holm,
                        nested_predict, within_group_targets)

LOG = logging.getLogger(__name__)
NAMES = {
    "egemaps_88_functionals": "标准88维 Functionals（整段）",
    "egemaps_50_clip": "自定义50维均值/标准差（整段）",
    "egemaps_50_aligned": "已有50维均值/标准差（词区间汇总）",
}
PROJECT = Path(__file__).resolve().parents[3]


def code_files():
    paths = list(Path(__file__).parent.glob("*.py"))
    paths += [Path(__file__).parents[1] / name for name in ("probe.py", "probe_cv.py")]
    return {str(p.relative_to(PROJECT)): file_hash(p) for p in sorted(paths)}


def load_reference(cfg, samples):
    root = Path(cfg["output_dir"])
    directory = root / "analysis/probe_v2"
    meta, report = (read_json(directory / name) for name in ("blocks.json", "report.json"))
    if meta["sample_ids"] != [s["id"] for s in samples]:
        raise ValueError("既有实验的样本顺序不一致")
    if meta["analysis_code_sha256"] != analysis_hash():
        raise ValueError("既有探针代码已变化，不能作为同协议对照")
    if file_hash(directory / "blocks.npz") != meta["npz_sha256"]:
        raise ValueError("既有特征块校验失败")
    if meta["signature"] != report["input_signature"]:
        raise ValueError("既有报告与特征块不属于同次实验")
    for name, expected in meta["input_files"].items():
        if file_hash(root / name) != expected:
            raise ValueError(f"既有实验输入已变化：{name}")
    cache = Cache(cfg)
    for sample in samples:
        if not cache.valid(sample, "aggregate"):
            raise ValueError(f"主产物或源缓存过期：{sample['id']}")
    with np.load(directory / "blocks.npz", allow_pickle=False) as z:
        blocks = {k: z[k] for k in z.files if k.startswith("lld_")}
    return blocks, report


def extract(cfg, samples, out):
    root = Path(cfg["output_dir"])
    smile = opensmile.Smile(feature_set=opensmile.FeatureSet.eGeMAPSv02,
                            feature_level=opensmile.FeatureLevel.Functionals,
                            num_workers=1)
    assert smile.num_features == 88 and len(smile.feature_names) == 88
    configs = Path(opensmile.__file__).parent / "core/config"
    config_hashes = {str(p.relative_to(configs)): file_hash(p)
                     for p in sorted(configs.rglob("*")) if p.is_file()}
    inputs = {str(p.relative_to(root)): file_hash(p) for s in samples
              for p in [root / "cache" / s["key"] / name for name in ("audio.wav", "media.json")]}
    spec = {
        "feature_set": "eGeMAPSv02", "feature_level": "Functionals",
        "opensmile_version": importlib.metadata.version("opensmile"),
        "sample_ids": [s["id"] for s in samples], "sample_rate": 16000,
        "normalization_before_extraction": "none; use cached mono waveform unchanged",
        "segment": "entire decoded audio; one row per clip; no word cropping",
        "silence_rule": "same as primary pipeline: peak <= 1e-4; all 88 coordinates missing",
        "nonfinite_rule": "retain raw output for audit; non-finite coordinates become NaN for train-only imputation",
        "feature_names": smile.feature_names, "input_files": inputs,
        "opensmile_config_files": config_hashes,
    }
    signature = digest({"spec": spec, "code": code_files()})
    feature_path, meta_path = out / "features.npz", out / "features.json"
    old = read_json(meta_path) if meta_path.exists() else {}
    if feature_path.exists() and old.get("signature") == signature and old.get("sha256") == file_hash(feature_path):
        with np.load(feature_path, allow_pickle=False) as z:
            arrays = {k: z[k] for k in z.files}
        LOG.info("复用已校验的100条标准Functionals")
        return arrays, old
    rows, silent, times = [], [], []
    for i, sample in enumerate(samples, 1):
        directory = root / "cache" / sample["key"]
        media = read_json(directory / "media.json")
        wave, sr = sf.read(directory / "audio.wav", dtype="float32")
        assert sr == 16000 and wave.ndim == 1 and len(wave) == media["audio_samples"]
        is_silent = bool(np.max(np.abs(wave)) <= cfg["media"]["silence_peak"])
        assert is_silent == media["silent_audio"]
        frame = smile.process_signal(wave, sr)
        if frame.shape != (1, 88) or frame.columns.tolist() != smile.feature_names:
            raise ValueError(f"Functionals输出不是标准1×88：{sample['id']}")
        rows.append(frame.to_numpy()[0]); silent.append(is_silent)
        times.append([media["audio_offset"], media["audio_offset"] + len(wave) / sr])
        if i % 10 == 0:
            LOG.info("标准Functionals提取 %d/%d", i, len(samples))
    raw = np.asarray(rows, np.float32)
    silent = np.asarray(silent, bool)
    x = raw.astype(np.float64)
    x[~np.isfinite(x)] = np.nan
    x[silent] = np.nan
    arrays = {"raw": raw, "features": x, "silent_mask": silent,
              "intervals": np.asarray(times), "sample_ids": np.asarray(spec["sample_ids"])}
    atomic_npz(feature_path, **arrays)
    meta = {**spec, "signature": signature, "sha256": file_hash(feature_path),
            "raw_nonfinite_coordinates": int((~np.isfinite(raw)).sum()),
            "missing_rows": int(np.isnan(x).all(axis=1).sum()),
            "silent_samples": [s["id"] for s, flag in zip(samples, silent) if flag]}
    atomic_json(meta_path, meta)
    pd.DataFrame(x, index=pd.Index(spec["sample_ids"], name="sample_id"),
                 columns=smile.feature_names).to_csv(out / "features.csv", encoding="utf-8-sig")
    return arrays, meta


def evaluate(cfg, threads=2):
    started = time.monotonic()
    root = Path(cfg["output_dir"])
    out = root / "analysis/functionals88"
    out.mkdir(parents=True, exist_ok=True)
    samples = read_json(root / "manifest.json")["samples"]
    assert len(samples) == 100
    LOG.info("核对主产物、既有探针输入与协议")
    blocks, previous = load_reference(cfg, samples)
    arrays, features_meta = extract(cfg, samples, out)
    old_protocol = previous["protocol"]
    assert old_protocol["seed"] == cfg["seed"]
    assert old_protocol["permutations"] == 499 and old_protocol["bootstrap"] == 2000
    assert old_protocol["alphas"] == ALPHAS.tolist()
    y = np.asarray([s["label"] for s in samples], float)
    groups = np.asarray([s["video_id"] for s in samples])
    designs = {
        "egemaps_88_functionals": arrays["features"],
        "egemaps_50_clip": np.c_[blocks["lld_mean_clip"], blocks["lld_std_clip"]],
        "egemaps_50_aligned": np.c_[blocks["lld_mean_aligned"], blocks["lld_std_aligned"]],
    }
    for x in designs.values():
        assert x.shape[0] == 100 and not np.isinf(x).any()
        assert np.isnan(x[arrays["silent_mask"]]).all()
    atomic_npz(out / "designs.npz", **designs)
    versions = {p: importlib.metadata.version(p) for p in
                ("opensmile", "numpy", "scipy", "scikit-learn", "soundfile", "threadpoolctl")}
    protocol = {
        "outer_cv": "LeaveOneGroupOut by video_id (37 folds)",
        "inner_cv": "GroupKFold by video_id (5 folds)",
        "preprocessing": old_protocol["preprocessing"],
        "weights": "after train-only standardization, divide each coordinate by sqrt(D); D=88 or 50",
        "regression": old_protocol["regression"], "classification": old_protocol["classification"],
        "alphas": ALPHAS.tolist(), "tie_break": "larger alpha", "seed": cfg["seed"],
        "permutations": 499, "permutation_rule": old_protocol["permutation_rule"],
        "bootstrap": 2000, "bootstrap_unit": old_protocol["bootstrap_unit"],
        "multiplicity": "19 configurations: previous 17 + standard 88 + whole-clip 50; Holm separately per metric",
        "paired_intervals": "95% percentile cluster bootstrap on fixed outer predictions; exploratory, no multiplicity adjustment",
        "configuration_names": NAMES, "threads": threads,
    }
    sources = {str(p.relative_to(root)): file_hash(p) for p in
               (root / "manifest.json", root / "run_config.json", root / "analysis/probe_v2/report.json",
                root / "analysis/probe_v2/blocks.json", root / "analysis/probe_v2/blocks.npz")}
    signature = digest({"features": features_meta["signature"], "designs": file_hash(out / "designs.npz"),
                        "protocol": protocol, "versions": versions, "sources": sources, "code": code_files()})
    targets = within_group_targets(y, groups, 499, np.random.default_rng(cfg["seed"]))
    boot_indices = cluster_bootstrap_indices(groups, 2000, np.random.default_rng(cfg["seed"] + 1))
    predictions = baseline_predictions(y, groups)
    baselines = {name: {k: float(v[0]) for k, v in batch_metrics(y, pr, pc).items()}
                 for name, (pr, pc) in predictions.items()}
    results, boots, folds_all = {}, {}, {}
    checkpoints = out / "checkpoints"
    checkpoints.mkdir(exist_ok=True)
    for name, x in designs.items():
        LOG.info("验证 %s (%d维)", name, x.shape[1])
        weights = np.full(x.shape[1], x.shape[1] ** -.5)
        npz, meta_path = checkpoints / (name + ".npz"), checkpoints / (name + ".json")
        old = read_json(meta_path) if meta_path.exists() else {}
        if npz.exists() and old.get("signature") == signature and old.get("sha256") == file_hash(npz):
            with np.load(npz, allow_pickle=False) as z:
                pr, pc = z["regression"], z["classification"]
            folds = old["folds"]
        else:
            def progress(fold, total):
                if fold % 10 == 0 or fold == total:
                    LOG.info("  外层 %d/%d，含499次组内置换", fold, total)
            pr, pc, folds = nested_predict(x, targets, groups, weights, ALPHAS, 5, progress)
            atomic_npz(npz, regression=pr, classification=pc)
            atomic_json(meta_path, {"signature": signature, "sha256": file_hash(npz), "folds": folds})
        scores = batch_metrics(targets, pr, pc)
        r = {k: float(v[0]) for k, v in scores.items()}
        for metric, higher in (("mae", False), ("pearson", True), ("macro_f1_3", True)):
            values = scores[metric]
            extreme = values[1:] >= values[0] if higher else values[1:] <= values[0]
            r["p_" + metric] = float((1 + extreme.sum()) / 500)
        predictions[name] = (pr[:, 0], pc[:, 0])
        boots[name] = bootstrap_scores(y, pr[:, 0], pc[:, 0], boot_indices)
        r.update(dims=x.shape[1], missing_rows=int(np.isnan(x).all(axis=1).sum()),
                 ci95={k: interval(v) for k, v in boots[name].items()},
                 confusion_matrix=confusion_matrix(np.sign(y), pc[:, 0], labels=CLASSES).tolist(),
                 regression_alpha_median=float(np.median([f["regression_alpha"] for f in folds])),
                 classification_alpha_median=float(np.median([f["classification_alpha"] for f in folds])))
        results[name], folds_all[name] = r, folds
        LOG.info("  MAE=%.4f Pearson=%.4f Accuracy=%.4f Macro-F1=%.4f",
                 r["mae"], r["pearson"], r["accuracy3"], r["macro_f1_3"])
        atomic_json(out / "progress.json", {"complete": len(results), "total": 3, "last": name})
    family = {**previous["results"], **results}
    assert len(family) == 19
    adjusted = {}
    for metric in ("mae", "pearson", "macro_f1_3"):
        pvalues = holm([r["p_" + metric] for r in family.values()])
        adjusted[metric] = {name: float(p) for name, p in zip(family, pvalues)}
        for name in results:
            results[name]["p_" + metric + "_holm19"] = adjusted[metric][name]
    comparisons = {}
    for a, b in [("egemaps_88_functionals", "egemaps_50_clip"),
                 ("egemaps_88_functionals", "egemaps_50_aligned"),
                 ("egemaps_50_clip", "egemaps_50_aligned")]:
        comparisons[a + " - " + b] = {
            k: value for metric in ("mae", "pearson", "accuracy3", "macro_f1_3")
            for k, value in [(metric + "_diff", results[a][metric] - results[b][metric]),
                             (metric + "_diff_ci95", interval(boots[a][metric] - boots[b][metric]))]}
    table = {"sample_id": [s["id"] for s in samples], "video_id": groups, "label": y,
             "class_label": np.sign(y).astype(int)}
    for name, (pr, pc) in predictions.items():
        table[name + "__regression"], table[name + "__class"] = pr, pc
    pd.DataFrame(table).to_csv(out / "predictions.csv", index=False, encoding="utf-8-sig")
    report = clean_json({
        "completed_utc": datetime.now(timezone.utc).isoformat(), "elapsed_seconds": time.monotonic() - started,
        "n": len(y), "groups": len(np.unique(groups)), "protocol": protocol, "versions": versions,
        "experiment_signature": signature, "code_files": code_files(), "source_files": sources,
        "features_sha256": file_hash(out / "features.npz"), "designs_sha256": file_hash(out / "designs.npz"),
        "predictions_sha256": file_hash(out / "predictions.csv"),
        "extraction": {k: features_meta[k] for k in ("raw_nonfinite_coordinates", "missing_rows", "silent_samples")},
        "results": results, "baselines": baselines, "paired_bootstrap": comparisons,
        "holm19": adjusted, "previous_report_sha256": file_hash(root / "analysis/probe_v2/report.json"),
        "limitations": [
            "100条片段、37个原视频；本次是既有实验之后的探索性补充，不是独立测试集。",
            "标准88维使用整段音频；与词区间50维比较时同时改变时间覆盖、加权和统计量。整段50维用于补充控制。",
            "标准88维与自定义50维的统计定义不同；包括归一化标准差、分位数、浊音条件统计和时间结构，不能只归因于增加38维。",
            "2条数字静音在全部方案中均视为缺失，但仍保留在100条评估中，仅在训练折内均值填补。",
            "组内置换保留视频间标签分布；成对Bootstrap区间基于固定预测，未作多重比较校正。",
            "句级预测结果不能证明词级时间对齐质量；本实验不修改主流程的768/818/56输出。",
        ]})
    atomic_json(out / "report.json", report)
    atomic_json(out / "folds.json", folds_all)
    write_report(out, report)
    LOG.info("完成：%s", out / "report.md")
    return report


def write_report(out, report):
    rows = []
    lines = ["# 标准 eGeMAPSv02 88维 Functionals 补充实验", "",
             "100条样本、37个原视频；外层留一视频，内层5折分组选择Ridge参数。499次组内置换、2000次视频级成对Bootstrap。",
             "整段88维与整段50维覆盖相同音频；词区间50维复现原有探针。2条全静音保留为缺失观测。", "",
             "|方案|维度|MAE↓|Pearson↑|Accuracy3↑|Macro-F1↑|F1校正p（19项）|",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for name, r in report["results"].items():
        row = {"configuration": name, "name": NAMES[name], **{k: r[k] for k in
               ("dims", "mae", "pearson", "accuracy3", "macro_f1_3", "p_macro_f1_3_holm19")}}
        rows.append(row)
        lines.append(f"|{NAMES[name]}|{r['dims']}|{r['mae']:.4f}|{r['pearson']:.4f}|{r['accuracy3']:.4f}|{r['macro_f1_3']:.4f}|{r['p_macro_f1_3_holm19']:.3f}|")
    lines += ["", "训练折均值回归基线MAE：" + f"{report['baselines']['train_mean']['mae']:.4f}。",
              "", "## 成对差值（前者减后者）", "",
              "MAE为负表示前者误差较低；Macro-F1为正表示前者分类较好。95%区间为探索性Bootstrap区间。", "",
              "|比较|MAE差值［95%区间］|Macro-F1差值［95%区间］|", "|---|---:|---:|"]
    for pair, values in report["paired_bootstrap"].items():
        a, b = pair.split(" - ")
        entries = []
        for key in ("mae", "macro_f1_3"):
            lo, hi = values[key + "_diff_ci95"]
            entries.append(f"{values[key + '_diff']:+.4f} [{lo:+.4f}, {hi:+.4f}]")
        lines.append(f"|{NAMES[a]} − {NAMES[b]}|{'|'.join(entries)}|")
    lines += ["", "## 解释边界", ""] + ["- " + s for s in report["limitations"]]
    lines += ["", "## 复现与产物", "", "```bash",
              ".venv-q1/bin/python -m q1.analysis.functionals88 --threads 2", "```", "",
              "features.npz保留原始88维和掩码后特征；features.json保存全部字段名、输入与官方配置哈希；",
              "designs.npz保存三套实际实验输入；predictions.csv为逐样本留出预测；folds.json及checkpoints/保留调参与置换结果。",
              "标准定义：https://audeering.github.io/opensmile-python/#feature-sets", ""]
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")
    pd.DataFrame(rows).to_csv(out / "summary.csv", index=False, encoding="utf-8-sig")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(Path(__file__).parents[2] / "config.yaml"))
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("threads must be positive")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    with threadpool_limits(limits=args.threads):
        evaluate(load_config(args.config), args.threads)


if __name__ == "__main__":
    main()

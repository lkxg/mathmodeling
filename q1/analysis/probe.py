"""Q1 auxiliary probes: complete features, nested video groups and three classes.

Corrected reports go to analysis/probe_v2; the previous experiment is retained.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from threadpoolctl import threadpool_limits

from ..common import atomic_json, atomic_npz, digest, file_hash, load_config, read_json
from ..pipeline import Cache
from ..temporal import pool_tokens
from .probe_cv import (CLASSES, batch_metrics, cluster_bootstrap_indices, holm,
                       nested_predict, nested_splits, within_group_targets)

LOG = logging.getLogger(__name__)
ALPHAS = np.logspace(-2, 6, 17)
BERT = {"id": "google-bert/bert-base-uncased", "revision": "86b5e0934494bd15c9632b12f734a8a67f723594"}
TEXT = ["text"]
A768 = ["e2v_aligned"]
A793 = A768 + ["lld_mean_aligned"]
A818 = A793 + ["lld_std_aligned"]
V28 = ["vision_mean_aligned"]
V56 = V28 + ["vision_std_aligned"]
AC = ["e2v_clip", "lld_mean_clip", "lld_std_clip"]
VC = ["vision_mean_clip", "vision_std_clip"]
CONFIGS = {
    "text_modernbert": TEXT, "text_bert": ["bert"],
    "audio_768_aligned": A768, "audio_793_aligned": A793, "audio_818_aligned": A818,
    "audio_768_clip": ["e2v_clip"], "audio_818_clip": AC,
    "egemaps_50_aligned": A818[1:],
    "vision_28_aligned": V28, "vision_56_aligned": V56, "vision_56_clip": VC,
    "audio_visual_aligned": A818 + V56,
    "text_audio_aligned": TEXT + A818, "text_visual_aligned": TEXT + V56,
    "fusion_aligned": TEXT + A818 + V56,
    "fusion_no_std": TEXT + A793 + V28, "fusion_clip": TEXT + AC + VC,
}
PAIRS = [
    ("audio_793_aligned", "audio_768_aligned"),
    ("audio_818_aligned", "audio_793_aligned"),
    ("audio_818_aligned", "audio_768_aligned"),
    ("vision_56_aligned", "vision_28_aligned"),
    ("fusion_aligned", "fusion_no_std"), ("fusion_aligned", "fusion_clip"),
    ("audio_818_aligned", "audio_818_clip"), ("vision_56_aligned", "vision_56_clip"),
    ("text_modernbert", "text_bert"), ("fusion_aligned", "text_modernbert"),
    ("text_audio_aligned", "text_modernbert"), ("text_visual_aligned", "text_modernbert"),
    ("fusion_aligned", "train_mean"), ("fusion_aligned", "train_median"),
    ("text_modernbert", "train_median"),
]


def masked_stats(x, keep, with_std=False):
    width = x.shape[1] * (2 if with_std else 1)
    if not np.any(keep):
        return np.full(width, np.nan)
    values = x[keep].astype(np.float64)
    return np.r_[values.mean(0), values.std(0)] if with_std else values.mean(0)


def aligned_blocks(z):
    """Average actual exported coordinates, including within-word standard deviations."""
    words = z["word_indices"] >= 0
    timed = words & z["time_valid_mask"]
    c, a, v = z["component_mask"], z["audio"], z["vision"]
    if a.shape[1] != 818 or v.shape[1] != 56 or z["text"].shape[1] != 768:
        raise ValueError("当前消融定义要求768/818/56维输出")
    return {
        "text": masked_stats(z["text"], words & c[:, 0]),
        "e2v_aligned": masked_stats(a[:, :768], timed & c[:, 1]),
        "lld_mean_aligned": masked_stats(a[:, 768:793], timed & c[:, 2]),
        "lld_std_aligned": masked_stats(a[:, 793:818], timed & c[:, 2]),
        "vision_mean_aligned": masked_stats(v[:, :28], timed & c[:, 3]),
        "vision_std_aligned": masked_stats(v[:, 28:56], timed & c[:, 3]),
    }


def bert_sentences(samples, root, device):
    import torch
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(BERT["id"], revision=BERT["revision"], use_fast=True)
    model = AutoModel.from_pretrained(BERT["id"], revision=BERT["revision"]).to(device).eval()
    rows = []
    with torch.inference_mode():
        for sample in samples:
            enc = tok(sample["raw_text"], return_tensors="pt", return_offsets_mapping=True,
                      truncation=False)
            if enc["input_ids"].shape[1] > model.config.max_position_embeddings:
                raise ValueError("BERT文本超长；禁止静默截断")
            offsets = enc.pop("offset_mapping")[0].numpy()
            attention = enc["attention_mask"][0].numpy()
            hidden = model(**enc.to(device)).last_hidden_state[0].float().cpu().numpy()
            words = read_json(root / "cache" / sample["key"] / "align.json")["words"]
            vectors, valid, _ = pool_tokens(hidden, offsets, [w["char_span"] for w in words], attention)
            # Same subword -> word -> utterance rule and export precision as ModernBERT.
            rows.append(masked_stats(vectors.astype(np.float16), valid))
    del model
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return np.asarray(rows)


def analysis_hash():
    return digest({p.name: file_hash(p) for p in sorted(Path(__file__).parent.glob("*.py"))})


def get_blocks(cfg, samples, out):
    root, cache = Path(cfg["output_dir"]), Cache(cfg)
    records = {}
    LOG.info("验证全部特征及源缓存，建立实验输入校验和")
    for s in samples:
        if not cache.valid(s, "aggregate"):
            raise ValueError(f"{s['id']} 特征或缓存过期，请先完成q1 validate")
        files = [root / "features" / (s["key"] + ext) for ext in (".npz", ".json")]
        files += [root / "cache" / s["key"] / n for n in
                  ("audio_features.npz", "vision.npz", "align.json")]
        records.update({str(p.relative_to(root)): file_hash(p) for p in files})
    records.update({n: file_hash(root / n) for n in ("manifest.json", "run_config.json")})
    signature = digest({"files": records, "analysis_code": analysis_hash(), "bert": BERT})
    npz, meta = out / "blocks.npz", out / "blocks.json"
    if meta.exists() and npz.exists():
        saved = read_json(meta)
        if saved.get("signature") == signature and saved.get("npz_sha256") == file_hash(npz):
            LOG.info("复用经过校验的句级特征缓存")
            with np.load(npz, allow_pickle=False) as z:
                return {k: z[k] for k in z.files}, signature
    rows = []
    for s in samples:
        with np.load(root / "features" / (s["key"] + ".npz"), allow_pickle=False) as z:
            row = aligned_blocks(z)
        d = root / "cache" / s["key"]
        with np.load(d / "audio_features.npz", allow_pickle=False) as z:
            row["e2v_clip"] = masked_stats(z["emotion"], z["emotion_valid"])
            acoustic = masked_stats(z["acoustic"], z["acoustic_valid"], True)
            row["lld_mean_clip"], row["lld_std_clip"] = acoustic[:25], acoustic[25:]
        with np.load(d / "vision.npz", allow_pickle=False) as z:
            vision = masked_stats(z["features"], z["quality"] > 0, True)
            row["vision_mean_clip"], row["vision_std_clip"] = vision[:28], vision[28:]
        rows.append(row)
    blocks = {k: np.asarray([r[k] for r in rows]) for k in rows[0]}
    LOG.info("计算BERT对照，使用与ModernBERT相同的词聚合规则")
    blocks["bert"] = bert_sentences(samples, root, cfg["device"])
    atomic_npz(npz, **blocks)
    atomic_json(meta, {"signature": signature, "npz_sha256": file_hash(npz),
                       "input_files": records, "analysis_code_sha256": analysis_hash(),
                       "bert": BERT, "sample_ids": [s["id"] for s in samples]})
    return blocks, signature


def reference_dimension(name):
    # Fixed family denominators preserve coordinate weights during ablation.
    return 50 if name.startswith("lld_") else 56 if name.startswith("vision_") else 768


def design(blocks, parts):
    x = np.concatenate([blocks[p] for p in parts], axis=1)
    weights = np.concatenate([np.full(blocks[p].shape[1], reference_dimension(p) ** -.5) for p in parts])
    return x, weights


def clean_json(value):
    if isinstance(value, dict):
        return {k: clean_json(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [clean_json(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean_json(value.tolist())
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def bootstrap_scores(y, pred, cls, indices):
    rows = [batch_metrics(y[i], pred[i], cls[i]) for i in indices]
    return {k: np.asarray([r[k][0] for r in rows]) for k in rows[0]}


def interval(values):
    finite = np.asarray(values)[np.isfinite(values)]
    return np.percentile(finite, [2.5, 97.5]).tolist() if len(finite) else [None, None]


def baseline_predictions(y, groups):
    mean, median = np.empty_like(y), np.empty_like(y)
    majority = np.empty(len(y), np.int8)
    for train, test, _ in nested_splits(groups):
        mean[test], median[test] = y[train].mean(), np.median(y[train])
        majority[test] = CLASSES[np.asarray([(np.sign(y[train]) == c).sum() for c in CLASSES]).argmax()]
    return {"train_mean": (mean, majority), "train_median": (median, majority)}



def evaluate(cfg, permutations=499, bootstrap=2000, inner_folds=5, threads=2, output_name="probe_v2"):
    if permutations < 1 or bootstrap < 2 or inner_folds < 2 or threads < 1:
        raise ValueError("permutations>=1，bootstrap>=2，inner-folds>=2，threads>=1")
    if Path(output_name).name != output_name:
        raise ValueError("output-name必须是单个目录名")
    started = time.monotonic()
    root = Path(cfg["output_dir"])
    out = root / "analysis" / output_name
    out.mkdir(parents=True, exist_ok=True)
    samples = read_json(root / "manifest.json")["samples"]
    y = np.asarray([s["label"] for s in samples], dtype=float)
    groups = np.asarray([s["video_id"] for s in samples])
    annotations = {"Negative": -1, "Neutral": 0, "Positive": 1}
    if not np.array_equal(np.sign(y), [annotations[s["annotation"]] for s in samples]):
        raise ValueError("连续标签符号与原始三分类标注不一致")
    blocks, input_signature = get_blocks(cfg, samples, out)
    versions = {p: importlib.metadata.version(p) for p in
                ("numpy", "scipy", "scikit-learn", "torch", "transformers", "threadpoolctl")}
    protocol = {
        "outer_cv": f"LeaveOneGroupOut by video_id ({len(np.unique(groups))} folds)",
        "inner_cv": f"GroupKFold by video_id ({inner_folds} folds)",
        "preprocessing": "每个内/外训练折独立拟合均值填补与StandardScaler；不使用留出折统计量",
        "weights": "固定家族参考维度：text/e2v=768，LLD=50，vision=56；每维乘reference_dim^(-1/2)，消融不改变保留坐标权重",
        "regression": "Ridge with intercept; alpha minimizes pooled inner OOF MAE",
        "classification": "one-vs-rest Ridge on one-hot labels; argmax; alpha maximizes pooled inner OOF macro-F1",
        "class_order": [-1, 0, 1], "class_names": ["Negative", "Neutral", "Positive"],
        "alphas": ALPHAS.tolist(), "tie_break": "larger alpha",
        "permutations": permutations, "permutation_rule": "仅在同一video_id内置换标签；每次重做内层调参及外层预测",
        "permutation_adjustment": f"Holm分别校正每个指标的{len(CONFIGS)}项模型检验",
        "bootstrap": bootstrap, "bootstrap_unit": "video_id；抽中视频的全部片段一起保留，成对抽样",
        "seed": cfg["seed"], "threads": threads,
        "aligned": "有效词位置的完整导出坐标均值；词内标准差直接读取，未改成词间标准差",
        "clip": "有效源帧全片段均值/标准差；保留全片段信息",
        "bert": BERT, "bert_pooling": "与ModernBERT一致：原文跨度子词均值→词均值，FP16后读取",
        "secondary_acc2": "仅非零标签，回归预测按符号二分类；不替代三分类",
    }
    signature = digest({"inputs": input_signature, "protocol": protocol,
                        "versions": versions, "analysis_code": analysis_hash()})
    targets = within_group_targets(y, groups, permutations, np.random.default_rng(cfg["seed"]))
    boot_indices = cluster_bootstrap_indices(groups, bootstrap, np.random.default_rng(cfg["seed"] + 1))
    predictions = baseline_predictions(y, groups)
    results, baselines, boot_results, folds_by_model = {}, {}, {}, {}
    for name, (pr, pc) in predictions.items():
        baselines[name] = {k: float(v[0]) for k, v in batch_metrics(y, pr, pc).items()}
        boot_results[name] = bootstrap_scores(y, pr, pc, boot_indices)
        baselines[name]["ci95"] = {k: interval(v) for k, v in boot_results[name].items()}
    checkpoint_dir = out / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    with threadpool_limits(limits=threads):
        for number, (name, parts) in enumerate(CONFIGS.items(), 1):
            LOG.info("[%d/%d] %s", number, len(CONFIGS), name)
            x, weights = design(blocks, parts)
            path, meta = checkpoint_dir / (name + ".npz"), checkpoint_dir / (name + ".json")
            saved = read_json(meta) if meta.exists() else {}
            if path.exists() and saved.get("signature") == signature and saved.get("sha256") == file_hash(path):
                with np.load(path, allow_pickle=False) as z:
                    pr, pc = z["regression"], z["classification"]
                folds = saved["folds"]
                LOG.info("  复用已校验的完整嵌套验证结果")
            else:
                def progress(fold, total):
                    if fold % 10 == 0 or fold == total:
                        LOG.info("  外层 %d/%d（含全部置换目标）", fold, total)
                pr, pc, folds = nested_predict(x, targets, groups, weights, ALPHAS, inner_folds, progress)
                atomic_npz(path, regression=pr, classification=pc)
                atomic_json(meta, {"signature": signature, "sha256": file_hash(path), "folds": folds})
            all_scores = batch_metrics(targets, pr, pc)
            r = {k: float(v[0]) for k, v in all_scores.items()}
            for metric, higher in (("mae", False), ("pearson", True), ("macro_f1_3", True)):
                values = all_scores[metric]
                extreme = values[1:] >= values[0] if higher else values[1:] <= values[0]
                r["p_" + metric] = float((1 + extreme.sum()) / (permutations + 1))
            r.update(dims=x.shape[1], parts=parts, missing_all_features=int(np.isnan(x).all(axis=1).sum()),
                     missing_by_part={p: int(np.isnan(blocks[p]).all(axis=1).sum()) for p in parts},
                     regression_alpha_median=float(np.median([f["regression_alpha"] for f in folds])),
                     classification_alpha_median=float(np.median([f["classification_alpha"] for f in folds])),
                     confusion_matrix=confusion_matrix(np.sign(y), pc[:, 0], labels=CLASSES).tolist())
            predictions[name] = (pr[:, 0], pc[:, 0])
            boot_results[name] = bootstrap_scores(y, pr[:, 0], pc[:, 0], boot_indices)
            r["ci95"] = {k: interval(v) for k, v in boot_results[name].items()}
            results[name], folds_by_model[name] = r, folds
            LOG.info("  MAE=%.4f, r=%.4f, Acc3=%.3f, Macro-F1=%.3f",
                     r["mae"], r["pearson"], r["accuracy3"], r["macro_f1_3"])
            atomic_json(out / "progress.json", {"complete": number, "total": len(CONFIGS),
                                                "last": name, "signature": signature})
    for metric in ("mae", "pearson", "macro_f1_3"):
        for name, value in zip(results, holm([r["p_" + metric] for r in results.values()])):
            results[name]["p_" + metric + "_holm"] = float(value)
    comparisons, combined = {}, {**baselines, **results}
    for a, b in PAIRS:
        comparisons[a + " - " + b] = {
            k: v for metric in ("mae", "pearson", "accuracy3", "macro_f1_3")
            for k, v in [(metric + "_diff", combined[a][metric] - combined[b][metric]),
                         (metric + "_diff_ci95", interval(boot_results[a][metric] - boot_results[b][metric]))]}
    table = {"sample_id": [s["id"] for s in samples], "video_id": groups, "label": y,
             "class_label": np.sign(y).astype(int)}
    for name, (pr, pc) in predictions.items():
        table[name + "__regression"] = pr
        table[name + "__class"] = pc
    pd.DataFrame(table).to_csv(out / "predictions.csv", index=False, encoding="utf-8-sig")
    variable_groups = [g for g in np.unique(groups) if len(np.unique(y[groups == g])) > 1]
    report = {
        "schema_version": 2, "completed_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.monotonic() - started, "n": len(y), "groups": len(np.unique(groups)),
        "class_counts": {str(c): int((np.sign(y) == c).sum()) for c in CLASSES},
        "protocol": protocol, "versions": versions, "analysis_code_sha256": analysis_hash(),
        "input_signature": input_signature, "experiment_signature": signature,
        "permutation_variable_groups": len(variable_groups),
        "baselines": baselines, "results": results, "paired_bootstrap": comparisons,
        "limitations": [
            "仅100条片段、37个原视频；冻结特征的线性可读性评估不能证明某一维度最优，也不是问题2/3的最终模型结果。",
            "对齐与整段池化同时改变保留区间、样本权重和统计尺度；差异不能单独归因于时间对齐精度。",
            "对齐探针只汇总有效词，未使用额外未分配区间；静音及未观测分量只用训练折均值填补。",
            "组内置换检验以同视频内标签可交换为假设，保留视频间标签分布；不检验完全消除视频间关系的零假设。",
            "成对Bootstrap以原视频为单位，区间基于固定外层预测，未重新拟合全部模型；区间未作多重比较校正，解释为探索性证据。",
            "17项模型的置换p值按指标分别作Holm校正；不应挑选未经校正的单项p值宣称显著。",
            "二分类辅助指标排除25条中性样本；主分类指标在全部100条样本上计算三分类Accuracy和Macro-F1。",
            "VAD只能验证语音活动区间一致性，人工词边界精度仍未测量。",
        ],
        "method_sources": [
            "https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.permutation_test_score.html",
            "https://scikit-learn.org/stable/modules/cross_validation.html",
        ],
    }
    report = clean_json(report)
    atomic_json(out / "report.json", report)
    atomic_json(out / "folds.json", folds_by_model)
    from .probe_report import write_presentation
    write_presentation(out, report)
    LOG.info("全部完成：%s", out / "report.md")
    return report


def main():
    parser = argparse.ArgumentParser(description="Q1完整特征消融、嵌套分组验证与三分类探针")
    parser.add_argument("--config", default=str(Path(__file__).parents[1] / "config.yaml"))
    parser.add_argument("--permutations", type=int, default=499)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--inner-folds", type=int, default=5)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--output-name", default="probe_v2")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config(args.config)
    import torch
    torch.set_num_threads(args.threads)
    report = evaluate(cfg, args.permutations, args.bootstrap, args.inner_folds, args.threads, args.output_name)
    print(json.dumps({"complete": len(report["results"]), "samples": report["n"],
                      "elapsed_seconds": report["elapsed_seconds"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()

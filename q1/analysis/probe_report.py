"""Tables and exportable figures for Q1's auxiliary experiments."""
from pathlib import Path
import pandas as pd


def write_presentation(out, report):
    out = Path(out)
    rows = []
    for name, r in {**report["baselines"], **report["results"]}.items():
        rows.append({"experiment": name, **{k: v for k, v in r.items() if
                    k in ("dims", "mae", "pearson", "accuracy3", "macro_f1_3",
                          "acc2_regression_sign", "p_mae", "p_mae_holm",
                          "p_macro_f1_3", "p_macro_f1_3_holm")}})
    pd.DataFrame(rows).to_csv(out / "summary.csv", index=False, encoding="utf-8-sig")
    lines = [
        "# 第一问修正版辅助实验", "",
        f"冻结特征；{report['n']}条片段、{report['groups']}个原视频。结果用于特征诊断，不属于问题2/3的最终模型评测。", "",
        "外层按原视频留一，内层5折按原视频选参数。回归使用Ridge，三分类使用独热标签Ridge读出及argmax，分别选择正则化参数。", "",
        "|实验|维度|MAE↓|Pearson↑|三分类Accuracy↑|三分类Macro-F1↑|",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        corr = "—" if r["pearson"] is None else f"{r['pearson']:.4f}"
        lines.append(f"|{r['experiment']}|{r.get('dims', '—')}|{r['mae']:.4f}|{corr}|{r['accuracy3']:.4f}|{r['macro_f1_3']:.4f}|")
    lines += ["", "## 成对消融差值及按视频Bootstrap区间", "",
              "差值为左项减右项；MAE负值有利，Macro-F1正值有利。区间跨0表示方向尚不明确。", "",
              "|比较|MAE差值 [95%区间]|Macro-F1差值 [95%区间]|", "|---|---:|---:|"]
    for name, r in report["paired_bootstrap"].items():
        cells = []
        for metric in ("mae", "macro_f1_3"):
            a, b = r[metric + "_diff_ci95"]
            cells.append(f"{r[metric + '_diff']:+.4f} [{a:+.4f}, {b:+.4f}]")
        lines.append("|" + name + "|" + "|".join(cells) + "|")
    lines += ["", "## 组内置换检验", "",
              f"{report['protocol']['permutations']}次组内置换，每次重做内层调参。p值按每个指标的17项模型检验作Holm校正。", "",
              "|实验|MAE原始p|MAE校正p|Macro-F1原始p|Macro-F1校正p|", "|---|---:|---:|---:|---:|"]
    for name, r in report["results"].items():
        lines.append("|" + name + "|" + "|".join(f"{r[k]:.4f}" for k in
                     ("p_mae", "p_mae_holm", "p_macro_f1_3", "p_macro_f1_3_holm")) + "|")
    lines += ["", "## 解释边界", ""] + ["- " + v for v in report["limitations"]]
    lines += ["", "完整协议、版本和校验和见 report.json；逐样本预测见 predictions.csv；分组与选中参数见 folds.json。"]
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = list(report["results"])
    fig, axes = plt.subplots(1, 2, figsize=(13, 9), sharey=True, layout="constrained")
    for ax, metric, title in zip(axes, ("mae", "macro_f1_3"),
                                ("MAE (lower is better)", "3-class macro-F1 (higher is better)")):
        for i, name in enumerate(names):
            r = report["results"][name]
            low, high = r["ci95"][metric]
            ax.plot([low, high], [i, i], color="#517d9a", lw=1.5)
            ax.scatter(r[metric], i, color="#143d59", s=25, zorder=3)
        baseline = report["baselines"]["train_median"][metric]
        ax.axvline(baseline, ls="--", color="#c35d36", label="Training median / majority baseline")
        ax.set_title(title)
        ax.grid(axis="x", alpha=.2)
        ax.legend(fontsize=8, loc="lower right")
    axes[0].set_yticks(range(len(names)), names, fontsize=9)
    axes[0].invert_yaxis()
    fig.suptitle("Q1 frozen-feature probes | nested video-group CV | cluster bootstrap 95% intervals")
    for ext in ("png", "svg", "pdf"):
        fig.savefig(out / ("probe_comparison." + ext), dpi=170)
    plt.close(fig)

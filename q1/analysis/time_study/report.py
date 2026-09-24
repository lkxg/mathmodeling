"""Generate the study narrative and standalone scientific figures."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

from ...common import atomic_json, file_hash, read_json
from .data import default_config, output_dir


def run(cfg=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["text.parse_math"]=False
    cfg=cfg or default_config(); root=output_dir(cfg,"")
    reports={n:read_json(root/n/"report.json") for n in ("validation","sampling","anchors","soft","review")}
    verification=read_json(root/"verification.json"); assert verification["passed"]
    v,p,a,s,r=(reports[n] for n in ("validation","sampling","anchors","soft","review"))
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout="constrained")
    rates=["5","8","10","12","25"]
    axes[0,0].plot([int(n) for n in rates],[1000*p["summary"][n]["mean_abs_centroid_shift_seconds"] for n in rates],"o-",color="#287d88")
    axes[0,0].set(xlabel="Requested sampling rate (Hz)",ylabel="Median centroid shift vs grid10 (ms)",title="A. Temporal perturbation: 8 diagnostic clips")
    modes=list(a["protocol"]["modes"]); yy=np.arange(len(modes))
    for j,(pool,color) in enumerate((("uniform","#287d88"),("duration","#dd8f4a"))):
        axes[0,1].barh(yy+(j-.5)*.35,[a["results"][m+"__"+pool]["macro_f1_3"] for m in modes],height=.33,color=color,label=pool+" readout")
    axes[0,1].set(yticks=yy,yticklabels=modes,xlabel="OOF Macro-F1",title="B. Anchors: 37 video-group outer folds")
    axes[0,1].legend(fontsize=8)
    axes[0,1].set_ylim(-1.2,len(modes)-.4)
    names=list(s["results"]); xx=np.arange(len(names))
    for ax,metric,title in ((axes[1,0],"mae","C. Neural controls: MAE (lower is better)"),(axes[1,1],"macro_f1_3","D. Neural controls: Macro-F1")):
        ax.bar(xx,[s["results"][n][metric]["mean"] for n in names],yerr=[s["results"][n][metric]["seed_std"] for n in names],capsize=3,color=["#287d88","#78a8a5","#dd8f4a","#ac7b47","#a8aeb4"])
        ax.set(xticks=xx,xticklabels=[n.replace("_","\n") for n in names],title=title,ylabel="Mean across 3 fixed seeds")
        ax.tick_params(axis="x",labelsize=8)
    for ax in axes.ravel(): ax.grid(axis="y",alpha=.18); ax.set_axisbelow(True)
    fig.suptitle("Q1 time-coordinate study | error bars = seed SD, not confidence intervals",fontsize=12)
    for extension in ("png","svg","pdf"): fig.savefig(root/("study_overview."+extension),dpi=160)
    plt.close(fig)
    # Same prespecified held-out example under hard and learned local weights.
    examples=[(p,read_json(p.with_suffix(".json"))) for p in (root/"soft/attention/local_attention").glob("*.npz")]
    example,meta=next((p,m) for p,m in examples if m["fold"]==0)
    fig,axes=plt.subplots(2,3,figsize=(12,7),layout="constrained")
    for i,mode in enumerate(("hard_overlap","local_attention")):
        with np.load(root/"soft/attention"/mode/example.name) as z:
            for j,m in enumerate(("emotion","acoustic","vision")):
                ax=axes[i,j]; mesh=ax.pcolormesh(z[m+"_source_intervals"].mean(1),z["anchor_times"].mean(1),z[m+"_weights"],shading="nearest",cmap="viridis")
                ax.set(xlabel="Native source time (s)",ylabel="Anchor time (s)",title=mode+" / "+m)
                fig.colorbar(mesh,ax=ax,shrink=.8)
    fig.suptitle(meta["sample_id"]+" | held-out fold 0, seed 2026; weights are not corrected timestamps")
    for ext in ("png","svg","pdf"): fig.savefig(root/("attention_example."+ext),dpi=150)
    plt.close(fig)
    lines=["# 第一问：时间坐标与局部软对齐补充实验", "",
           "自动实验和产物核查已完成。当前证据支持继续使用可解释的词区间＋间隙映射作为主方案；固定50段和局部软对齐保留为可复现实验选项。人工词边界真值尚未标注。", "",
           "## 1. 统一时间身份核验", "",
           f"全部 {v['samples']} 条通过主流程结构验证与新增深度核验：源帧索引/PTS一致、词身份保留、有效词映射单调、CSR源集合及重叠权重正确，且导出音视频特征可从原生源帧独立重建。",
           f"视频总时长 {v['clip_seconds']:.6f} 秒；目标区间并集覆盖 {v['mapped_seconds']:.6f} 秒（{100*v['mapped_seconds']/v['clip_seconds']:.4f}%）；未覆盖共 {v['unmapped_seconds']:.3f} 秒，最长单段 {1000*v['longest_gap_seconds']:.1f} ms，小于保留间隙的80 ms阈值。",
           "覆盖率包含未分配区间，不能用来替代词边界准确率。检测误报仍可能通过结构核验。详见 [validation/report.json](validation/report.json)。", "",
           "## 2. 采样率扰动", "",
           "预先按静音、无脸、时长、低VAD覆盖和视觉变化选择8条诊断样本，不按情感标签选取。以原始PTS最近邻对应固定秒级网格，在5/8/10/12/25 Hz重新检测并提取视觉特征。每种采样率都用固定外层训练折的读出器预测，不重新调参。",
           "旧的最小帧间距规则会受原帧量化影响，例如约24 fps输入的名义10 Hz采样实际接近8 Hz；新增网格采样只用于实验，未替换主特征。高于原帧率时不插值造帧。", "",
           "|目标 Hz|实际 Hz 中位数|特征相对变化中位数|代表帧时间偏移中位数 (ms)|分类变化 / 8|",
           "|---:|---:|---:|---:|---:|"]
    for rate in rates:
        z=p["summary"][rate]; lines.append(f"|{rate}|{z['effective_hz']:.3f}|{z['relative_feature_change']:.3f}|{1000*z['mean_abs_centroid_shift_seconds']:.2f}|{z['class_flips']}|")
    lines += ["", "所有40份新采样缓存的源PTS误差均为0；但12 Hz和25 Hz各有1条分类变化。说明坐标身份稳定，特征及预测并非完全不变。代表帧偏移统计仅计算有可比较有效观测的目标位置；不能把8条诊断样本外推为全量稳健性保证。", "",
              "## 3. 锚定区间对照", "",
              "全部100条；37个原视频留一外层、5折分组内层；499次视频组内标签置换、2000次按视频Bootstrap；10项检验按指标分别Holm校正。保持完整文本内容不变，改变音视频时间池化及统计尺度。此处读出包含间隙，不与旧词位置探针混为同一实验。", "",
              "|锚点|等权读出 MAE|等权 Macro-F1|时长加权 MAE|时长加权 Macro-F1|", "|---|---:|---:|---:|---:|"]
    for m in modes:
        u,d=(a["results"][m+"__"+pool] for pool in ("uniform","duration"))
        lines.append(f"|{m}|{u['mae']:.4f}|{u['macro_f1_3']:.4f}|{d['mae']:.4f}|{d['macro_f1_3']:.4f}|")
    lines += ["", "固定50段相对词区间的MAE和Macro-F1成对差值区间均跨0；没有替换依据。固定0.25秒的时长加权Macro-F1差值区间略低于0，是未校正的探索性结果。各配置的置换检验经Holm校正均未通过0.05阈值。均值读出会丢失顺序，不能据此否定固定锚点在其他时序模型中的价值。500份替代视图均通过源映射重建核验。", "",
              "## 4. 原生编码＋局部软对齐", "",
              "文本、emotion2vec、声学LLD和视觉各自保持原生时间轴，分别投影到16维潜在空间。每0.5秒的锚点从时间重叠的文本潜变量形成查询；局部注意力只允许源帧中心在锚点中心±0.5秒内，以0.25秒尺度施加平方时间距离惩罚。完整文本语义分支保留时间无效词。全缺失模态产生零上下文和无效标记。",
              "5个固定视频分组外层折 × 3个种子 × 5种方法，共75次训练。训练折内拟合原生特征标准化；固定80轮AdamW，无测试集早停、调参或挑选种子。基础编码器和读出器相同；硬重叠与时间核不使用查询/键内容分数。", "",
              "|方法|MAE 均值 ± 种子标准差|Macro-F1 均值 ± 种子标准差|", "|---|---:|---:|"]
    for n,z in s["results"].items(): lines.append(f"|{n}|{z['mae']['mean']:.4f} ± {z['mae']['seed_std']:.4f}|{z['macro_f1_3']['mean']:.4f} ± {z['macro_f1_3']['seed_std']:.4f}|")
    diff=s["paired_bootstrap"]["local_attention - hard_overlap"]
    lines += ["", f"局部注意力相对硬重叠的MAE差为 {diff['mae']['difference']:+.4f}，95%区间 {diff['mae']['ci95'][0]:+.4f}～{diff['mae']['ci95'][1]:+.4f}；Macro-F1差为 {diff['macro_f1_3']['difference']:+.4f}，95%区间 {diff['macro_f1_3']['ci95'][0]:+.4f}～{diff['macro_f1_3']['ci95'][1]:+.4f}。均无明确改进。",
              f"局部注意力优于无约束全局注意力的差值区间未跨0，但区间未作多重比较校正，只是探索性证据。同折训练均值回归基线MAE为 {s['training_baseline']['mae']:.4f}，提示本数据量下神经回归并未获益。",
              "已保存全部模型、训练折归一化参数、逐样本预测和25份预先确定的留出样本注意力映射。验证重新加载75个检查点重放预测，并检查时间窗外权重为0。注意力记录独立保存，不改写物理时间身份。", "",
              "## 5. 18条可疑样本的复核", "",
              f"[打开复核页面](review/index.html)：原始转写、缓存音频、原视频、波形/VAD/词区间、54张原帧截图、逐词标注与CSV导出。共 {r['words_for_annotation']} 个词的人工边界栏保持空白。模板导入评分命令见代码README，拒绝未知索引、缺半个边界、越界区间和缺失复核者代号。",
              "AI已查看每段3张原帧。另核查 `-NFrJFQijFE__2` 的全部8个有效人脸检测框：均位于鸟羽，属于误报。证据为 [检测框图](review/face_detection_audit.jpg)，原帧索引45、48、60、63、66、69、96、117。",
              "[画面观察与原视频哈希](../../../../q1/analysis/time_study/scene_review.json)记录复核范围。`review/quality/`保存独立排除掩码，只排除这8个有证据的误报，未逐帧认证其余观测。本轮统计实验使用原特征，未套用事后排除掩码；主产物仍含这8个误报，后续训练应显式应用复核掩码并重新聚合/验证。",
              "两条数字全静音样本由波形确认；动画/器材画面、场景切换、低光及侧脸也分别记录。未进行人工听音词边界标注，人工MAE/P90/100 ms命中率均不可报告。", "",
              "## 6. 建议与复现", "",
              "第一问主线应强调统一可追溯时间坐标、原生源序列、质量掩码和可重建映射。现阶段维持词区间＋间隙导出；将固定锚点和局部软对齐作为有对照、有效果边界的扩展。优先完成词边界人工真值及视觉误检处理，再判断是否引入更复杂模型。",
              "所有实验仅用附件1，不代替问题2/3在附件2上的训练和验证。", "",
              "[图表](study_overview.svg) · [注意力示例](attention_example.svg) · [独立核验记录](verification.json) · [代码与运行说明](../../../../q1/analysis/time_study/README.md)", ""]
    (root/"report.md").write_text("\n".join(lines),encoding="utf-8")
    atomic_json(root/"summary.json",{"automated_experiments_complete":True,"human_word_boundary_truth_complete":False,
                "main_alignment_replacement_supported":False,"known_original_false_face_frames":8,
                "source_reports":{name:file_hash(root/name/"report.json") for name in reports},
                "verification_sha256":file_hash(root/"verification.json"),"report_code_sha256":file_hash(__file__)})
    return root/"report.md"


if __name__=="__main__": print(run())

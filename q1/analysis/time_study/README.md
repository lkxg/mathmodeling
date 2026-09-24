# 时间身份与局部软对齐实验

本目录是第一问的扩展实验，使用附件1的100条片段及冻结的原生特征。不修改主提取流水线或旧版探针，不替代第二、三问的训练任务。

## 复现

从项目根目录运行，先确认主流程缓存和 `probe_v2` 完整：

```bash
.venv-q1/bin/python -m q1.analysis.time_study.validate
.venv-q1/bin/python -m q1.analysis.time_study.review
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 .venv-q1/bin/python -m q1.analysis.time_study.sampling
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 .venv-q1/bin/python -m q1.analysis.time_study.anchors
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 .venv-q1/bin/python -m q1.analysis.time_study.soft
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 .venv-q1/bin/python -m q1.analysis.time_study.verify_experiments
.venv-q1/bin/python -m q1.analysis.time_study.report
.venv-q1/bin/python -m unittest discover -s q1/tests -v
```

采样扰动依赖 `analysis/alignment_vad.csv` 和 `analysis/probe_v2/{blocks.npz,blocks.json,folds.json}`。缺少时先运行父目录README中的VAD与探针命令。采样需要加载视觉模型，软对齐使用CUDA训练；其余主要是CPU计算。首次运行较慢的阶段会输出逐样本或逐模型进度。不要在实验运行中修改对应源文件。

产物位于 `outputs/q1/analysis/time_study/`：

|位置|内容|
|---|---|
|`report.md`、`summary.json`|中文结果及结论边界|
|`validation/`|源身份、时间单调、覆盖率、映射权重及FP16导出重建|
|`sampling/`|8条预选样本、40份原帧采样缓存、原始10 Hz对照、固定读出器扰动结果|
|`anchors/`|500份替代视图；5种锚点×2种片段汇总；预测、分组划分及统计检验|
|`soft/`|75个训练检查点、OOF预测、训练轨迹、25份留出注意力映射|
|`review/`|18条样本、263词空白人工模板、可播放标注页面、原帧与质量排除掩码|
|`verification.json`|独立重建、检查点重放、分组隔离和指标重算记录|
|`study_overview.*`、`attention_example.*`|PNG/SVG/PDF独立图表|

## 实验口径

主表示维度仍为768/818/56，原始时间坐标、原文及模态源序列保留。`validate` 调用主流程结构验证，再检查各时间映射的源集合与 `overlap × quality` 权重，并从源向量重建最终导出值。异常以非零状态退出。覆盖统计取时间并集，包含未分配间隙，不解释为词级准确率。

采样实验按固定秒网格选最近的真实PTS帧，不插值。它与主流水线的最小间隔采样不同；同时报告实际Hz，避免名义采样率掩盖约24 fps视频中10 Hz退化为约8 Hz的量化现象。原始帧号/PTS应完全不变；代表帧、特征和预测允许变化并需要量化。8条诊断样本不代表总体分布。

锚点比较词区间＋间隙、0.25/0.5/1秒固定区间、全片均匀50段；后两者不修复无效词边界。固定网格保存文本/音频/视觉各自的CSR源索引和权重；源数组与主缓存一一对应。读出同时比较等权与时长加权，文本全片内容保持一致。外层按37个原视频留一，内层5折视频分组；499次组内置换重新选择正则化，2000次按视频Bootstrap。10配置按指标分别进行Holm校正，成对Bootstrap区间未校正。读出包含间隙且会丢失顺序，不能直接套用旧词位置探针的结论。

软对齐先对原生文本768维、emotion2vec768维、LLD25维和视觉28维分别做可训练Linear→LayerNorm→GELU编码到16维。0.5秒查询锚点使用时间重叠的文本潜变量；音视频保持自己的原生采样密度。局部窗口为锚点中心±0.5秒，惩罚为 `-0.5 * (Δt / 0.25)^2`，检测质量作为权重先验。全缺失位置返回零权重，完整文本分支继续保留时间无效词。

对照包括硬重叠、固定高斯时间核、局部注意力、局部无惩罚、全局注意力。编码器与读出器相同，硬重叠和时间核不使用内容查询/键分数。5折GroupKFold×3固定种子×5方法，80轮全训练集批次、AdamW学习率0.001、权重衰减0.01。所有标准化和类别权重仅使用训练折；没有早停、测试集调参或选种子。报告种子均值/标准差；成对区间对固定OOF预测按原视频Bootstrap并平均种子指标，不等同于多次重新训练的置信保证。

注意力的每一列保留原生源行索引/时间，视觉另保留原始帧号和PTS；与物理CSR映射分开存放，不将学习到的关联写成修正后的真实词时间。

## 复核与人工评分

直接打开 `review/index.html`。若浏览器限制本地媒体访问，可在项目根目录启动仅本机服务：

```bash
.venv-q1/bin/python -m http.server 8765 --bind 127.0.0.1
```

然后访问 `http://127.0.0.1:8765/outputs/q1/analysis/time_study/review/index.html`。人工标注以缓存音频播放秒数＋`audio_offset`为时钟；原视频仅辅助观看，浏览器可能重置媒体时钟。输入匿名复核者代号，逐词听音填写边界并导出CSV。标注保存在当前浏览器，但应导出留存。自动边界不会预填到人工边界栏。

```bash
.venv-q1/bin/python -m q1.analysis.time_study.review --annotations /path/to/q1_word_annotations.csv
```

只对人工完整标注、且自动时间有效的词计算边界MAE、P90及100 ms命中率；另外报告人工已标注但自动无效的词。拒绝重复/未知词、半个边界、越界、非数值和没有复核者的标注。空模板不会产生人工真值。

`scene_review.json` 是AI对每条3帧及一个鸟类片段额外8帧的观察记录，带原视频哈希，不是人工词边界标注。其中 `-NFrJFQijFE__2` 原帧45、48、60、63、66、69、96、117的检测框均为鸟羽误报。`review/quality/<key>.npz` 保留：

- `original_frame_indices`、`original_pts`、`original_quality`：原检测身份与质量。
- `review_excluded_mask`：有图像证据的误报排除标记。
- `usable_after_review_mask`：`(original_quality > 0) & ~review_excluded_mask`。

该掩码只纠正已查明误报，不证明剩余全部帧合格，也没有自动覆盖主产物或事后改变本轮实验数据。后续消费方需在聚合前显式屏蔽排除行、重算均值/标准差/CSR并重新验证，不能仅将片段标签改为“无脸”。

## 当前结果与限制

100/100主样本通过正式时间核验；500份替代视图可重建；75模型预测可重放；35项测试通过。固定50段没有明确超过词区间。局部注意力MAE为0.6002、Macro-F1为0.4565；硬重叠为0.5950/0.4666，成对差值区间跨0。局部方法相对全局注意力有探索性优势，但不足以支持替换主方案。人工词边界真值仍未标注。

实验代码放在嵌套目录，避免改变主提取代码哈希及旧探针的分析哈希。实验输出较大，提交时按题目总大小限制挑选报告、汇总及必要示例，不应将全部缓存和检查点打包。

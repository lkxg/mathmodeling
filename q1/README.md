# 第一问：可追溯的三模态特征与词级时间对齐

固定组合：**Qwen3 ForcedAligner + ModernBERT + openSMILE + OpenFace 3.0**。输出维度为 **文本768／语音50／视觉56**。只读取附件1的100条视频和给定英文转写。

## SAQW-TA：语义锚点质量加权时序对齐方法

本文将现有方案记为 **SAQW-TA（Semantic-Anchor Quality-Weighted Temporal Alignment）**：在共同PTS时间坐标下，以给定转写的有效词区间为语义锚点，补充满足阈值的未分配gap，再将原生音视频观测按时间重叠和观测质量映射至这些目标区间。方法名用于指代本方案；预训练模型和加权统计公式沿用已有工具与数学定义，不据命名宣称首创或效果优于其他方法。

文本通过字符跨度把上下文子词表示映射到原词；音频和视觉保留各自原生时间窗。此处的异步对齐处理不同采样率与观测区间的对应关系，权重由时间重叠决定，未包含可学习的情绪表达时滞模型。

设目标区间为 $I_i=[s_i,e_i)$，模态 $m\in\{A,V\}$ 的有效原生观测为 $(J_k^{(m)},x_k^{(m)},q_k^{(m)})$，其中 $J_k^{(m)}=[a_k,b_k)$。仅对时间有效且分母为正的位置定义：

$$
\ell_{ik}^{(m)}=\max\{0,\min(e_i,b_k)-\max(s_i,a_k)\},\qquad
w_{ik}^{(m)}=\frac{\ell_{ik}^{(m)}q_k^{(m)}}{\sum_j\ell_{ij}^{(m)}q_j^{(m)}}.
$$

$$
\mu_i^{(m)}=\sum_k w_{ik}^{(m)}x_k^{(m)},\qquad
\sigma_i^{(m)}=\sqrt{\sum_k w_{ik}^{(m)}(x_k^{(m)}-\mu_i^{(m)})^2}.
$$

标准差逐特征维计算。音频 $A_i=[\mu_i^{(A)},\sigma_i^{(A)}]\in\mathbb R^{50}$，视觉 $V_i=[\mu_i^{(V)},\sigma_i^{(V)}]\in\mathbb R^{56}$，文本 $T_i\in\mathbb R^{768}$ 独立按原词字符跨度获得。无效词时间或无源支持的位置将对应音视频写零并置模态掩码为false；文本仍可保留。

统一位置表示为

$$
X_i=(T_i,A_i,V_i,I_i,t_i,M_i,C_i,P_i),
$$

其中 $t_i$ 是时间有效标记，$M_i\in\{0,1\}^3$ 是模态掩码，$C_i\in[0,1]^2$ 是逐位置音频/视觉覆盖，$P_i$ 为原词、token、源音频窗口和原视频帧的来源记录。真实位置与batch填充位置另用 `valid_mask` 区分。

可在论文中作为**已实现的方案特点**描述：词锚点与gap共同组织目标时间轴；重叠与质量共同确定原生观测的聚合权重；CSR映射、显式掩码和覆盖集合支持回溯与复算。这些有代码与产物依据；是否提高下游性能需要单独的对照实验，不能由完整性验证推出。

## 运行

从项目根目录执行。配置与固定模型修订号集中在 `q1/config.yaml`。

```bash
bash q1/install.sh
.venv-q1/bin/python -m q1 doctor
.venv-q1/bin/python -m q1 fetch-models
.venv-q1/bin/python -m q1 run
.venv-q1/bin/python -m q1 validate
.venv-q1/bin/python -m q1 preview --id=-3g5yACwYnA__13
.venv-q1/bin/python -m q1 report
.venv-q1/bin/python -m unittest discover -s q1/tests -v
```

系统需要Python 3.12和ffmpeg/ffprobe。默认CUDA，当前验证设备V100 32 GB。Qwen3使用FP16，ModernBERT使用FP32，均采用eager attention。openSMILE在CPU提取声学量。

`install.sh`建立继承现有系统包的独立 `.venv-q1`，从本目录唯一的 `requirements.txt` 补齐依赖；OpenFace单独用 `--no-deps` 安装。其官方包 `openface-test==0.1.26` 声明了旧版依赖，实际验证版本见 `environment-tested.txt`，不直接套用这些旧版约束。图表建议安装Noto CJK字体；Word可编辑，PDF可用LibreOffice转换。

`run --limit 1 --fail-fast`先执行首条。`run --stage audio`等命令只执行指定阶段；阶段顺序为 `media → align → text`，`media → audio/vision`，最后 `aggregate`。指定阶段前，上游结果必须有效。`--force`强制重建所选阶段。

样本筛选支持 `--id=视频ID__片段ID` 或Excel原始ID；原始ID包含 `$` 时须用单引号。运行失败会记录日志并返回非零状态，同一输出目录由文件锁保护。

## 特征定义

|模态|原生特征与处理|输出维度|
|---|---|---:|
|文本|保留题目转写；ModernBERT编码完整文本，按原文字符跨度把子词表示平均为词表示|768|
|语音|openSMILE eGeMAPSv02的25项LowLevelDescriptors；每个目标区间计算加权均值25＋加权总体标准差25|50|
|视觉|OpenFace单帧28项观测，在目标区间计算加权均值28＋加权总体标准差28|56|

Qwen3 ForcedAligner接收音频和给定转写，补充词起止时间。原文不被重新识别的文本替换。模型保持冻结，主流程不训练情感分类器。

openSMILE使用工具返回的原生时间窗，25项LLD包括响度、基频、谱形、MFCC、jitter/shimmer、谐噪比和共振峰等；完整顺序保存在 `cache/<ID>/audio.json`。50维是区间内加权均值与加权总体标准差的组合，不是整句88维Functionals。二者使用相同的“重叠长度×有效质量”归一化权重；标准差为 `sqrt(sum(w * (x - mean)**2))`，不作n−1修正，也不除以均值。

OpenFace单帧28维由表情logits 8维、视线2维、AU原始输出8维、RetinaFace五个人脸点相对坐标10维组成。AU索引不冒充FACS编号。初始选择最大人脸，随后按边框IoU跟踪；这不等于说话人识别。未检测到脸、非法边框与非有限输出均记为不可用。

`quality_review.json`保留已核查的原视频哈希和原帧排除规则。聚合和预览应用规则，原始检测结果仍留作来源记录。鸟类片段已确认的8帧羽毛误报不会进入正式视觉特征；视频哈希不符时拒绝应用排除规则。

### 观测质量权重的具体定义

视觉的有效质量可写为 $q_k^{(V)}=c_k d_k b_k f_k(1-r_k)$。其中 $c_k$ 为选中人脸框的有限检测得分，无可用框时取0；其余量均是0/1门控：

|量|对应当前实现|
|---|---|
|$d_k$|存在检测得分≥0.8的选中人脸框；初始最大脸，后续优先最大IoU|
|$b_k$|框裁到图像范围并转换为整数裁剪坐标后，`right>left`且`bottom>top`|
|$f_k$|清零处理之前的28维模型观测全部有限；非有限预测整帧不可用|
|$r_k$|视频哈希匹配且原帧号命中复核排除规则时为1|

实现上，`VisionEncoder`先把不合格观测的质量设为0，否则保存检测得分；`reviewed_quality`再将排除帧置0。聚合还要求源时间窗有限且为正时长、特征与质量有限。IoU用于选脸，不乘入质量权重；当前没有独立的遮挡度、清晰度或说话人置信度项。检测得分是质量代理量，未经情感或遮挡可靠性校准。

音频质量为 `acoustic_valid` 的0/1值，即“整段非数字静音”与“当前25维LLD全部有限”的合取。这不是逐帧语音活动检测；不能用它把gap命名为静音。

## 时间组织与掩码

共同原点来自实际解码的音视频PTS。视频保留原帧号和PTS；音频时间为“原始音频偏移＋重采样后下标/16000”。音频PTS不连续或OpenCV与ffprobe帧数不符时明确失败。

有效词起止时间作为目标区间，额外保留至少80 ms的未分配间隙。阈值带1 µs浮点容差。各模态在原生时间轴编码，再按重叠长度×观测质量归一化聚合；覆盖率用时间并集计算。

这里采用 **WORD + GAP**：gap仅表示“没有分配给结构有效词的区间”，可能包含停顿、非语言声音或未通过时间检查的词。gap不带停顿或静音标签，不能直接写成WORD + PAUSE。小于阈值的间隙可能不保留，因此用目标时间轴保留率量化完整程度，不宣称绝对完整。

无效词时间保留文本，时间掩码为false，不强行分配音视频。整段波形峰值≤1e-4时视为数字静音：全部音频观测和词时间无效，文本保留，视觉独立处理。零值可能是合法观测，必须使用显式掩码判断缺失。

## 目录与接口

```text
q1/
  config.yaml, requirements.txt, install.sh
  backends.py                  四种工具适配器
  media.py, temporal.py         媒体解析、时间与数值操作
  pipeline.py, validation.py    执行、缓存、聚合、独立核验
  dataset.py                   读取与batch填充
  cli.py, preview.py, report.py 命令行、典型样本图、Word报告
  quality_review.json          有来源身份的视觉排除规则
  tests/                       主流程与覆盖集合回归测试

outputs/q1/
  manifest.json                样本、标签、原视频SHA256
  run_config.json              实际配置、代码SHA256、依赖版本
  events.jsonl                 当前处理及已核验缓存复用日志
  summary.csv, validation.json 全量汇总与独立核验结果
  invalid_words.csv            每个无效词的全部原因标签与互斥分类
  cleanup.json                 此次精简与已验证缓存复用记录
  cache/<ID>/                  原生序列与阶段校验记录
  features/<ID>.npz/.json       正式三模态特征与来源映射
  report/                      当前Word/PDF、架构图与典型样本图
```

`cache/<ID>/`包含 `audio.wav`、`media.json`、`frames.npz`、`align.json`、`text.npz/.json`、`audio.npz/.json`、`vision.npz/.json` 和六个阶段的 `.meta.json`。缓存校验原视频、原文、配置、提取代码、主要依赖、上游及输出哈希；修改文档、预览或验证器不会重跑模型。

正式接口为 **schema_version=3**，NPZ使用 `allow_pickle=False` 读取。

|字段|形状|含义|
|---|---|---|
|`text / audio / vision`|`[L,768] / [L,50] / [L,56]`|默认FP16；音频前25列均值、后25列标准差|
|`timestamps`|`[L,2]`|片段相对时间，FP64|
|`valid_mask / time_valid_mask`|`[L]`|实际序列位置／有效时间|
|`modality_mask`|`[L,3]`|文本、音频、视觉观测有效性|
|`coverage`|`[L,2]`|逐位置声学、视觉有效源支持覆盖率，分母为该目标区间长度|
|`word_indices`|`[L]`|原文词索引，未分配区间为−1|
|`source_<m>_intervals`|`[S_m,2]`|原生时间窗，m为acoustic/vision|
|`map_<m>_offsets`|`[L+1]`|CSR偏移|
|`map_<m>_indices / weights`|`[nnz]`|参与聚合的源行号及归一化权重|
|`source_acoustic_valid`|`[S_a]`|原生声学有效性|
|`source_vision_quality / excluded`|`[S_v]`|复核后的视觉质量／排除标记|

样本JSON给出原文、字符跨度、词到token映射、原帧号/PTS、边框和质量记录。通过CSR索引可回查每个输出位置的音频窗口与视频帧。

```python
from q1.dataset import load_sample, collate
sample = load_sample("outputs/q1/features/-3g5yACwYnA__13.npz")
batch = collate([sample])
# audio: [B,L_max,50]；padding位置及各模态掩码补false。
```

## 验证与提交

`validate`核查100条样本的固定维度、数值与掩码、原词身份、时间覆盖、CSR索引范围、权重、原视频帧PTS，并独立从原生特征重建每个位置的音频/视觉均值和标准差。任何失败均返回非零状态。

当前验证覆盖工程完整性、时间结构和数值可重建性，不提供独立的词边界准确率结论。可疑样本列在 `validation.json`。典型样本图展示词、波形、声学曲线、AU曲线与原视频帧的共同时间轴。

## 指标口径（metrics_version=2）

每条样本s定义 `I_s` 为 `time_valid_mask=true` 的目标区间并集，**包含有效词区间和保留gap**；`J_A,s`、`J_V,s` 为特征与时间窗有限、时长为正、有效质量>0的原生音频/视觉支持区间并集。所有集合均先在单条样本内求并集去重，全量统计累加秒数后相除。

NPZ中的逐位置指标正式定义为

$$
C_{si}^{(m)}=\frac{|I_{si}\cap\mathcal J_s^{(m)}|}{|I_{si}|}\in[0,1],\qquad m\in\{A,V\},
$$

只在时间有效且 $|I_{si}|>0$ 时使用上述比例，否则存0。接近1表示有效源支持几乎覆盖整个目标区间，接近0表示支持很少或不存在；它不度量词边界是否准确，也不将检测质量分数作为时长折扣。

|指标|分子|分母|当前结果|
|---|---|---|---:|
|目标时间轴保留率|Σ长度(I_s)|Σ视频时长D_s|99.9706%|
|音频有效源覆盖率|Σ长度(I_s∩J_A,s)|Σ长度(I_s)|97.3381%|
|视觉有效源覆盖率|Σ长度(I_s∩J_V,s)|Σ长度(I_s)|95.4629%|
|至少一种模态有效|Σ长度(I_s∩(J_A,s∪J_V,s))|Σ长度(I_s)|99.1210%|
|音视频共同有效|Σ长度(I_s∩J_A,s∩J_V,s)|Σ长度(I_s)|93.6800%|

目标时间轴保留率为786.047211/786.278211秒，完全不检查源模态是否有效。其余四项分母都是786.047211秒。这些都不是词边界准确率。最长35 ms是未纳入目标时间轴的间隙，不能解释为最长音频或视觉缺失。`mapped_seconds`保留为历史兼容字段，明确定义为目标区间并集长度；新汇总使用 `temporal_coverage`。

1926词中1707词结构时间有效（88.6293%），219词无效。按“静音→越界→非单调/重叠→非正时长→其他”的展示优先级互斥归类：静音24、越界8、非正时长187，其余0。原始多标签计数为静音24、越界14、非正时长201，存在重叠，不能直接相加。完整原文覆盖检查未发现未返回词；缺词会使样本失败。59条样本含至少一个无效词，11条满足整段可疑阈值（静音或结构有效词比例<60%）。

附录T/A/V是**位置模态可用率**，分母为全部2476个词＋gap位置；对应比例77.79%/91.07%/89.10%。550个gap没有文本。另计的**文本词级抽取成功率**为1926/1926=100%，两者不能混用。

典型样本图同时显示有效/无效词时间、独立gap带、原始波形、有效声学曲线、AU输出、逐位置音视频源覆盖以及带原帧号、PTS和有效质量q的画面，便于按同一时间坐标复核。

`features/`、全量汇总、当前报告和必要代码是交付材料；`cache/`用于复算和断点续跑。题目的50 MB总附件限制还需容纳问题2/3。第一问的768/50/56特征空间与附件2并非同一接口，不能直接替换后者。

官方来源：[Qwen3 ForcedAligner](https://huggingface.co/Qwen/Qwen3-ForcedAligner-0.6B-hf)、[ModernBERT](https://huggingface.co/answerdotai/ModernBERT-base)、[openSMILE](https://audeering.github.io/opensmile-python/)、[OpenFace 3.0](https://github.com/CMU-MultiComp-Lab/OpenFace-3.0)。固定修订号见配置，实际环境版本见运行记录。

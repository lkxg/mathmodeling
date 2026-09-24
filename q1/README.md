# E题第一问：从100条视频到可追溯的词级三模态特征

实现流程：已有英文转写 → Qwen3强制对齐 → ModernBERT文本表示、emotion2vec帧级表示、openSMILE低层声学描述量、OpenFace 3.0面部特征 → 时间重叠与观测质量加权聚合 → 特征、掩码、质量记录、原始位置映射。

程序只进行预训练模型推理，不训练情感分类器，不修改原视频、转写或情感标签，不读取附件2/3/4。输出属于第一问自主提取的特征空间，不能因维度接近而直接代入附件2训练的模型。

## 快速运行

以下命令均从项目根目录 `/home/riftuser/mathmodeling` 执行。本工作区已创建 `.venv-q1`，并下载配置中固定版本的模型权重。权重缓存在用户的 Hugging Face 缓存目录，约3.5 GB，不属于提交附件。

```bash
# 新环境：以本工作区已有的 requirements-ml.txt 基础环境创建独立虚拟环境。
bash q1/install.sh

# 检查包、模型接口、ffmpeg、GPU；不会下载模型。
.venv-q1/bin/python -m q1 doctor

# 核查全部100条视频与Excel的精确对应关系。
.venv-q1/bin/python -m q1 manifest

# 首次使用可预下载权重；run也会按需下载。
.venv-q1/bin/python -m q1 fetch-models

# 先验证一条真实样本。
.venv-q1/bin/python -m q1 run --limit 1 --fail-fast
.venv-q1/bin/python -m q1 validate --limit 1

# 执行全部100条；已有的有效结果会复用。
.venv-q1/bin/python -m q1 run
.venv-q1/bin/python -m q1 validate

# 生成典型样本图：词边界、波形、声学曲线、AU曲线、原视频帧。
.venv-q1/bin/python -m q1 preview --id=-3g5yACwYnA__13

# 验证聚合、掩码、时间锚点、原文映射、缓存损坏等逻辑。
.venv-q1/bin/python -m unittest discover -s q1/tests -v
```

当前验证范围：35项自动测试（13项主流程、8项探针协议、14项时间映射/软对齐/人工评分测试），以及全部100条真实视频的全模型流程。新增时间深度核验、500份替代锚定视图重建及75个实验模型预测重放。结构性时间检查不等于人工核验后的对齐精度。

`install.sh` 继承已有基础环境，所有新增包安装在 `.venv-q1` 中。另一台机器应先在合适的Python环境中安装根目录 `requirements-ml.txt` 的基础依赖及系统 `ffmpeg`。已验证Python 3.12、PyTorch 2.13.0+cu126、torchvision 0.28.0+cu126、V100 32GB；详细环境记录见 `environment-tested.txt`。

OpenFace官方PyPI包名是 `openface-test`，Python导入名是 `openface`。其0.1.26版元数据严格锁定旧版NumPy等依赖，所以单独使用 `--no-deps` 安装，并以实际接口测试核验兼容性。不要直接让它的依赖解析器降级整个工作区。`pip check` 会报告这些已知的上游版本声明冲突；本项目不声称符合这些旧版声明。

## 配置和阶段

配置在 `q1/config.yaml`。`root_dir` 相对配置文件，其余文件路径相对 `root_dir`。模型使用完整提交哈希，避免同名模型随时间变化。GPU默认 `cuda:0`。Qwen3采用FP16，ModernBERT采用FP32；均使用eager attention，适配V100，不依赖FlashAttention 2或BF16。

可按阶段执行，例如：

```bash
.venv-q1/bin/python -m q1 run --stage media
.venv-q1/bin/python -m q1 run --stage align
.venv-q1/bin/python -m q1 run --stage text
.venv-q1/bin/python -m q1 run --stage audio
.venv-q1/bin/python -m q1 run --stage vision
.venv-q1/bin/python -m q1 run --stage aggregate
```

依赖关系为 `media → align → text`、`media → audio`、`media → vision`，`aggregate`要求前面所有阶段完成。指定阶段不会自动执行其上游；缺少依赖时会明确报错。`--id=-3g5yACwYnA__13` 可限定样本，参数可重复。ID也可以使用Excel对应的 `video_id$_$clip_id`，在shell中务必用单引号。`--limit N` 使用标注表顺序的前N条。

默认对单样本错误记录日志并继续，命令最终返回非零退出码。`--fail-fast` 在首次错误时停止。后端初始化失败不会用随机向量或全零“成功结果”替代。每阶段只加载本阶段模型，结束后释放GPU内存。

缓存核验同时考虑视频SHA256、原文、配置、代码内容、主要依赖版本、上游产物和输出文件校验和。损坏、配置变更和上游变化会使相应结果失效。修改代码后可能需要重跑；这是有意采用的保守规则。`--force` 强制重建指定阶段。同一输出目录用文件锁避免并发写入。

## 特征和维度

| 输出 | 定义 | 默认维度 |
|---|---|---:|
| `text` | ModernBERT子词按原文字符跨度聚合成词级上下文表示 | 768 |
| `audio` | emotion2vec均值768维 + eGeMAPSv02 LLD均值25维 + LLD标准差25维 | 818 |
| `vision` | OpenFace特征均值28维 + 标准差28维 | 56 |

OpenFace单帧28维由8维表情logits、2维视线输出、8维AU原始输出、RetinaFace的5个人脸点相对边框坐标组成。未启用STAR的68/98点模型，也未实现身体姿态提取。AU原始索引没有擅自标注为FACS编号；表情输出没有转成赛题情感标签。

视觉先选择最大人脸，后续优先选择与前帧边框IoU最大的脸；低于阈值时重新选择，并记录 `track_reset`。这是主体跟踪规则，不保证找到了实际说话人。未检出脸、异常边框和非有限预测均记录标记；不可用位置不参与聚合。RetinaFace检测分数只作为观测质量权重，不解释为情感预测置信度。

OpenFace适配器覆盖其RetinaFace加载方法：关闭相对当前目录读取预训练backbone的行为，严格加载包含backbone的完整官方权重。它不修改已安装包的源码。

## 时间定义

1. 从音视频实际解码时间戳确定共同原点。视频使用ffprobe的帧PTS，音频使用首个解码音频帧PTS及重采样后的采样下标。若音频PTS不连续，明确报错，避免将不连续录音误当作连续波形。
2. Qwen3输入原始转写，返回词级时间。保存模型原始时间及映射至片段的时间；非正时长、重叠和越界分别标记。对齐没有返回校准置信度，因此 `confidence` 为null。
3. 对齐词严格映射回原文字符跨度，包括重复词、缩写和内部标点。映射不完整时失败，不使用模糊匹配编造对应关系。ModernBERT使用原始完整句子编码，通过token字符offset聚合，排除特殊token。
4. emotion2vec位置锚点由官方卷积配置推算。默认感受野400个采样点、步长320个采样点，即16 kHz下25 ms窗口、20 ms步长，并核验帧数。它的Transformer表示仍含整段上下文，不能把这个锚点误读为模型只看了该25 ms。
5. openSMILE直接使用其返回的start/end索引。选择的是25维LLD，不是整句88维统计量。
6. 视觉默认按实际PTS约10 Hz抽样，保存原帧索引与PTS。聚合时间窗采用相邻采样帧之间的中点划分，即分段常值近似；这不是声称每一帧都已检测。原视频帧数与OpenCV解码帧数不一致时，拒绝继续错配。
7. 按词区间与源特征时间窗的重叠长度乘以质量权重，归一化后求均值/标准差。覆盖率计算区间并集，不重复累计重叠的卷积窗口。

Qwen3输出无效时间的词仍保留文本特征，但时间掩码为false，不把音视频特征强行分配给它。默认额外保存长度至少80 ms的未分配区间，`kind=unassigned`、`word_index=-1`、文本不可用。这类区间可能含停顿、笑声、背景声或对齐失败，代码不擅自给出声音类别。间隙长度比较带1 µs容差，避免浮点舍入使同为80 ms的间隙有的保留、有的遗漏。

整段音频峰值不超过 `media.silence_peak`（默认1e-4，约−80 dBFS）视为数字静音：`media.json` 记 `silent_audio=true`，所有词加 `silent_audio` 标记且时间无效，emotion2vec与openSMILE整段不可用；文本特征保留。对齐器在静音上仍会输出时间，这些时间不作为证据。

对齐有效词比例低于 `aggregation.alignment_suspect_ratio`（默认0.6）或为静音的样本，在汇总中标记 `alignment_suspect` 及 `suspect_reasons`，并列入 `validation.json` 的 `alignment_suspect_ids`，供人工抽检。程序不删除这些样本，也不改用均分时长补救；其中形式上有效的词时间可信度同样较低。

没有使用情感标签确定对齐边界，没有线性均分时长来伪造时间戳，也没有把原始样本统一截断到50个位置。

## 输出目录

```text
outputs/q1/
  manifest.json                  全部100条清单、标签、视频校验和
  run_config.json                配置、代码校验和、依赖版本
  events.jsonl                   每阶段/每样本成功或失败日志
  summary.csv                    全量汇总；未处理样本也有行
  validation.json                结构验证、未完成ID、产物大小
  cache/<video_id>__<clip_id>/
    audio.wav                    16 kHz单声道浮点音频
    media.json, frames.npz        原始流信息、音视频偏移、帧PTS
    align.json                   词、原文跨度、时间、异常标记
    text.npz, text.json           词向量、token与原文映射
    audio_features.npz/.json     原始帧级语音特征及时间锚点
    vision.npz/.json              采样帧特征、边框、检测质量
    <stage>.meta.json             阶段提交标记及文件校验和
  features/<video_id>__<clip_id>.npz
  features/<video_id>__<clip_id>.json
  previews/<video_id>__<clip_id>.png
```

NPZ文件不使用pickle对象，读取时可固定 `allow_pickle=False`。每条样本是变长序列，`length=L`。

| 字段 | 形状 | 含义 |
|---|---|---|
| `text / audio / vision` | `[L, d]` | 默认FP16保存；源缓存为FP32 |
| `timestamps` | `[L, 2]` | 相对片段原点的秒数，FP64 |
| `valid_mask` | `[L]` | 实际序列位置，单样本保存时全true |
| `time_valid_mask` | `[L]` | 此位置时间是否可用于聚合 |
| `modality_mask` | `[L, 3]` | 文本、语音、视觉是否至少有可用特征 |
| `component_mask` | `[L, 4]` | 文本、emotion2vec、openSMILE、视觉分别是否可用 |
| `coverage` | `[L, 3]` | emotion2vec、openSMILE、视觉在目标区间的有效覆盖比例 |
| `word_indices` | `[L]` | `align.json`词索引；额外区间为-1 |
| `source_<m>_intervals` | `[S_m, 2]` | 源特征时间窗，`m`为emotion/acoustic/vision |
| `map_<m>_offsets` | `[L+1]` | CSR偏移：位置k的源为 `indices[offsets[k]:offsets[k+1]]` |
| `map_<m>_indices / map_<m>_weights` | `[nnz]` | 参与聚合的源索引及归一化权重（FP32） |

语音模态只要两组特征之一可用，`modality_mask[:,1]` 就为true；下游必须结合 `component_mask` 区分768维和25+25维是否可用。零值可能是合法观测，不能用“全零”替代显式掩码。标准差反映源特征在窗口内的变化，不是预测不确定度。

配套JSON（紧凑格式，schema_version 2）记录时间轴、词及原文跨度、词到token映射、帧索引、PTS、边框和质量信息；配置与环境只在 `run_config.json` 保存一次，样本JSON用 `config_sha256` 引用。结合NPZ中的CSR映射可按索引回查音频、原文和视频帧。

读取和batch填充：

```python
from q1.dataset import load_sample, collate

sample = load_sample("outputs/q1/features/-3g5yACwYnA__13.npz")
batch = collate([sample])
# batch['text']: [B, L_max, 768]
# batch['valid_mask']: padding位置为false；其余掩码也同时补false。
```

## 验证及提交

`validate`核验当前配置与代码对应的产物、维度、数值、掩码和覆盖率，报告未完成ID，失败时返回非零退出码。`run --limit 1`结束时的汇总仍覆盖100条，因此看到“1/100”是正常的；使用 `validate --limit 1` 可只核验这条，输出 `summary.selected.csv` 与 `validation.selected.json`，不会覆盖全量汇总。

`summary.csv`的模态有效比例以全部序列位置为分母，包含无文本的额外区间；不要把它直接解读为词识别准确率。对齐有效比例也只是结构检查通过率。真正的时间精度需要在部分原始视频上人工标记词边界后计算，程序没有伪造这个评估结果。

`features/`是全部100条自主生成、经过定义的词级特征及映射。`cache/`包含工作用波形和高频特征，会明显更大。提交前应按题目要求选择必要文件、压缩并检查总大小；50MB还要留给问题2/3核心材料。不要直接打包虚拟环境、公共预训练权重或原视频。公共权重能否仅提供下载与校验信息，应以竞赛正式提交规定为准。

## 补充实验

新增的VAD一致性检查与特征消融说明见 [analysis/README.md](analysis/README.md)。修正版探针直接使用导出的完整768/818/56维特征，增加音频768→793→818、视觉28→56及融合标准差消融，采用嵌套视频分组验证，同时报告回归与完整三分类指标。

时间身份核验、采样率扰动、锚点对照、局部软对齐与18条样本的复核页面见 [analysis/time_study/README.md](analysis/time_study/README.md)。目前没有证据支持用固定50段或局部注意力替换词区间主方案；人工词边界真值仍为空。视觉复核发现一个鸟类片段的8帧人脸误报，已单独导出带源身份的排除掩码，尚未覆盖主特征。

```bash
.venv-q1/bin/python -m pip install -r q1/analysis/requirements.txt
.venv-q1/bin/python -m q1.analysis.alignment_quality
.venv-q1/bin/python -m q1.analysis.probe --threads 2
```

修正版结果在 `outputs/q1/analysis/probe_v2/`，其中 `report.md` 为可读汇总，`report.json` 保存协议、版本、输入校验和及统计检验，`predictions.csv` 保存逐样本预测，`probe_comparison.svg/.pdf/.png` 可用于论文绘图。原 `analysis/probe.json` 是旧协议结果，不能与新结果混写。

这些是附件1上的辅助特征诊断，不属于问题2/3的训练或专项测试结果。VAD不提供人工词边界真值；探针结果不能证明某一维度最优。

## 模型来源

- Qwen3原生Transformers强制对齐：https://huggingface.co/Qwen/Qwen3-ForcedAligner-0.6B-hf
- ModernBERT：https://huggingface.co/answerdotai/ModernBERT-base
- emotion2vec：https://github.com/ddlBoJack/emotion2vec
- openSMILE：https://audeering.github.io/opensmile-python/
- OpenFace 3.0：https://github.com/CMU-MultiComp-Lab/OpenFace-3.0
- OpenFace官方仓库链接的权重：https://huggingface.co/nutPace/openface_weights

所有模型的确切提交号写在配置中；模型功能说明应与实际使用的分支、输出维度和预训练来源一起写入论文。

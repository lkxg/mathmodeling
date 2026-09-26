# E题问题2和问题3：局部缺失鲁棒预测与可复核解释

后续新增架构实验见 [NEW_ARCHITECTURES.md](NEW_ARCHITECTURES.md)；上一轮优化方案见 [IMPROVED.md](IMPROVED.md)。

本目录提供已训练的模型代码。训练数据只使用赛题附件2 `aligned_50.pkl` 的 `train` 划分；`valid` 选择训练轮次；`test` 仅在模型选定后报告一次基础性能。附件3、4均只做推理。两问共用一套预测参数；问题3在预测器之外计算模态贡献和证据窗口。

## 数据接口与模型

附件3没有预计算 `text` 字段，因此三类数据统一读取 `text_bert`。冻结 `google-bert/bert-base-uncased`（修订号写在 `core.py`）的词向量，不进行情感微调。三个输入分别为词元向量768维、声学74维、视觉35维；每路投影到96维。用逐位置可用性掩码约束三模态门控，将融合序列送入两层、四头的时序Transformer。注意力池化得到样本表示；另外保留三路单模态专家及辅助分类头。主头同时预测三类极性和[-3,3]强度。总损失为交叉熵＋0.5倍Huber回归损失＋0.1倍单模态交叉熵。

音视频从训练集有效非零行估计每维均值和标准差，并把变换后特征截在[-8,8]。文本的0号、[CLS]与[SEP]不作为局部情绪证据；零行、遮挡和填充分开处理。连续缺失训练在每个样本的有效位置内随机选择1到3个模态和10%至60%长度的连续段；20%训练样本保持完整。文本缺失直接把词元ID设为0，再编码，防止完整句子的预计算上下文表示泄漏缺失位置的信息。

这套模型是轻量专家与掩码门控的实现，并未复现任何单篇论文的全部模块。训练没有加入附件2以外的情感数据。

## 运行

从仓库根目录执行；具体实验参数汇总在 `q23/config.yaml`，命令行可覆盖轮数、随机种子和batch大小。实际验证环境见 `q23/environment-tested.txt`。首次使用会从 Hugging Face 下载固定修订号的 BERT 权重；证据定位调用第一问相同的 Qwen3 ForcedAligner。

```bash
.venv-q1/bin/python -m q23.train --epochs 35
.venv-q1/bin/python -m q23.evaluate --include-test
.venv-q1/bin/python -m q23.train --epochs 35 --baseline
.venv-q1/bin/python -m q23.evaluate --baseline
.venv-q1/bin/python -m q23.infer
.venv-q1/bin/python -m q23.align_evidence
.venv-q1/bin/python -m q23.report
.venv-q1/bin/python -m q23.check
.venv-q1/bin/python -m q23.package
```

训练固定随机种子2026、batch size 128、AdamW学习率5e-4、余弦退火；每轮在验证集上计算 `macro-F1 - 0.08×MAE`，保存最高分轮次。`--baseline`仅取消训练时的连续区间遮挡，用于量化其作用；测试集和专项无标签集不参与模型选择。由于GPU计算可能有少量非确定性，复跑结果可在小数末位变化。

分类与回归是两个独立输出头。附件3结果文件中，若最终极性为中性，提交强度写0；正负类别则保持相应的强度符号，极少数符号相反时裁到接近零。`raw_intensity`保留未经这一步处理的回归输出，供检查。模型选择和上表基础回归指标使用原始回归输出；`metrics.json`额外提供最终一致化输出的MAE与相关系数，避免两种口径混淆。

## 当前结果

|模型／输入|Accuracy|Macro-F1|MAE|Pearson|
|---|---:|---:|---:|---:|
|鲁棒模型，验证集完整输入|0.5865|0.5454|0.6662|0.5304|
|鲁棒模型，附件2测试划分完整输入|0.6231|0.5527|0.7331|0.5519|
|无区间遮挡基线，验证集完整输入|0.5632|0.5399|0.6880|0.5055|

验证集单模态连续缺失：文本、音频、视觉各按10%／30%／50%，分别落在有效序列前／中／后，共27种场景。鲁棒模型的场景平均Macro-F1为0.5340、MAE为0.6784；同架构无区间遮挡基线分别为0.5024、0.7591。另检验文本＋音频、文本＋视觉、音频＋视觉和三模态同步在中部缺失30%的4种场景，鲁棒模型平均Macro-F1为0.5228、MAE为0.6883；基线分别为0.4816、0.7973。该缺失率按有效序列位置计，不代表视频秒数。完整逐场景数据见 `outputs/q23/validation_robustness*.csv`。

第三问的模态作用程度用三模态全部8个保留组合计算精确Shapley值，作用目标为最终预测类别的**logit**；保留正负号，同时将绝对值归一化为展示份额。主要参考模态取最大正贡献，若三者都非正则取绝对贡献最大者。各模态中将连续三个位置遮挡，记录预测类别logit下降，选择不重叠的关键窗口。遮挡结果只是对**当前模型**的干预解释，不是情绪的真实因果原因；门控权重不作为贡献真值。

附件4的原文用固定BERT tokenizer映射到给定 `text_bert`；20条样本词元ID全部精确一致。第一问的强制对齐器对原视频转写给出词时间，再把证据窗口映射到时段。177个窗口中173个可给出词级估计时段；其余4个明确标为 `unresolved`。视频帧在视觉窗口中点附近抽取。对齐时间为模型估计，附件原始特征没有独立逐行真实时间戳，不能把它写成已验证的特征生成时间。题面称该集模态完整，但实际对齐版 `13.pkl` 的视觉特征全为零；本实现将其标为视觉不可用，不输出该样本的视觉关键帧。该例外记录在 `attachment4_input_audit.json`。

## 输出

- `outputs/q23/model.pt`：鲁棒模型可训练参数，约1.3 MB；冻结BERT词向量依赖指定公开模型修订号。
- `outputs/q23/model_baseline.pt`：无区间遮挡对照模型。
- `outputs/q23/normalization.npz`：仅从训练集估计的标准化参数。
- `outputs/q23/metrics.json`、`validation_robustness.csv`、`analysis.json`：基础性能、缺失规律和错误分析。
- `outputs/q23/attachment3_predictions.csv`：附件3全部30条极性与强度预测；`attachment3_input_audit.json`记录输入缺失位置数。
- `outputs/q23/attachment4_predictions_explanations.csv`：附件4全部20条预测、三模态贡献和证据位置；`attachment4_explanations.json`保留每条窗口的干预量，`attachment4_input_audit.json`记录模态可用性。
- `outputs/q23/attachment4_evidence_times.csv`：关键窗口对应文本、时段及时间状态；`attachment4_position_to_time.json`和`attachment4_alignment_audit.json`保留对齐依据。
- `outputs/q23/attachment4_cards.html`、`keyframes/`、`q2_robustness.png`、`q3_modality_effects.png`：论文展示材料。

`outputs/q23/align_cache/` 是对齐时的临时音频与来源数据，提交时无需纳入。若最终压缩包限制50 MB，模型、标准化参数、代码、CSV、JSON和必要图表可纳入，预训练BERT与Qwen模型通过固定修订号下载。

论文中须说明三分类标签0/1/2分别为Negative/Neutral/Positive。专项集无标签，不能报告附件3或附件4上的准确率。验证集的混淆矩阵及分组误差见 `analysis.json`，典型解释卡可直接用浏览器打开。
附件3这30条对齐版输入检查中未发现文本内部连续零段，实际缺口落在音频或视觉；文本缺失与多模态同步缺失结论来自附件2验证集的模拟实验。验证集中性类召回率约29.9%，是当前模型的主要薄弱环节。附件4的20条预测没有中性结果；这是模型输出分布，专项集无标签，不能据此判断是否正确。

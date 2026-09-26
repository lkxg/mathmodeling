# E题第二、三问：提高准确率的架构对照与最终模型

新增查询融合、跨模态增强和蒸馏的进一步实验及结论见 [NEW_ARCHITECTURES.md](NEW_ARCHITECTURES.md)。

本轮在原有可运行方案上增加三种小型架构，并用附件2固定的训练集3395条训练、验证集728条选择模型。附件2保留测试划分727条只在选定组合后评估一次。附件3、4始终只用于最终推理。

## 统一输入

三种架构都读取 `aligned_50.pkl` 的文本768维、音频74维、视觉35维时序特征。附件2给出的 `text` 与固定修订号 `google-bert/bert-base-uncased` 对 `text_bert` 的输出在抽样检查中逐位置吻合；附件3仅有 `text_bert`，因此推理时由同一冻结BERT现场编码。附件4已有文本特征，直接使用。音视频标准化参数仍仅由附件2训练集的有效非零观测估计，显式掩码区分零行、局部缺失和填充。

训练时文本缺失发生在词元输入处：先将一段词元ID和注意力掩码清零，再由冻结BERT编码。为节省计算，每条训练样本预生成一个确定的随机连续缺口版本；音频和视觉缺口每轮重新抽样。文本缺口缓存保存在 `outputs/q23/experiment_cache/`，不纳入交付包。这个设计避免从完整文本的上下文表示向被遮挡词泄漏信息。

## 三种方案

|代码名称|结构|借鉴点|
|---|---|
|`text_anchor`|将文本作为锚点，三模态在对齐位置动态门控，经过两层时序Transformer预测；文本缺失时由可用音视频形成代理表示|[EMOE，CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Fang_EMOE_Modality-Specific_Enhanced_Dynamic_Emotion_Experts_CVPR_2025_paper.html) 的动态模态专家思路|
|`shared_private`|每模态投影到共享128维和私有48维；共享通道按可用模态平均，私有通道保留各自信息，随后时序建模|[FUSE-Net，CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/papers/Yang_Factorize_Reconstruct_Enhance_A_Unified_Framework_for_Multimodal_Sentiment_Analysis_CVPR_2026_paper.pdf) 的表示分解思路|
|`reliability_proxy`|每模态给出潜在表示与学习到的标量方差，依据逆方差融合为代理表示，再加入各模态私有信息|[P-RMF，ACL 2025](https://aclanthology.org/2025.acl-long.1075/) 的不确定性代理融合思路|

这些是面向赛题规模的简化实现，不声称逐模块复现原论文。三个模型均用交叉熵、Huber强度损失和单模态辅助分类损失训练。固定随机种子2026、batch size 128、AdamW学习率4e-4、权重衰减0.03、25轮余弦退火。各模型轮次只按验证集 `Accuracy + 0.05×Macro-F1 − 0.02×MAE` 选择。随后仅比较三个单模型和四个等权组合，以验证集Accuracy优先、Macro-F1次之、MAE再次之选择最终模型。

## 实际结果

|模型|验证Accuracy|Macro-F1|MAE|
|---|---:|---:|---:|
|前一版时序门控模型|0.5865|0.5454|0.6662|
|文本锚点|0.6319|0.5902|0.6073|
|共享／私有表示|0.6374|0.5928|0.5967|
|可靠性代理|0.6415|0.6104|0.6034|
|**选定：共享／私有＋可靠性代理等权组合**|**0.6484**|**0.6137**|**0.5947**|

最终组合在附件2保留测试划分上的Accuracy为**0.6602**、Macro-F1为**0.5829**、MAE为**0.6472**、Pearson为**0.6655**。这些数值不是附件3或附件4的准确率；两套专项数据无标签。验证集混淆矩阵和错误分析见 `selected_error_analysis.json`，其中中性类召回率为70/184≈38.0%，仍是主要弱项。

局部连续缺失验证采用文本／音频／视觉分别缺失10%、30%、50%，位于有效序列前／中／后的27种场景，以及中间30%位置的4种多模态同步缺失。最终模型单模态场景的平均Accuracy为**0.6287**、Macro-F1为**0.5898**；多模态场景分别为**0.6113**、**0.5694**。完整表格和曲线在 `selected_robustness.csv` 与 `selected_robustness.png`。缺失率按有效位置计，不直接等同于秒数。

## 第三问解释

最终组合的类别logit用于计算三模态全部8种保留组合的精确Shapley贡献；连续3个位置的遮挡实验定位各模态关键证据。CSV同时提供极性、强度、主要模态、三模态带符号贡献及归一化作用份额。强度输出按极性保持符号一致，中性写0；`raw_intensity`保留模型原始回归值。贡献计算针对原始模型输出，不能解释为情绪的真实因果原因。

附件4的20条文本词元均与同一BERT tokenizer精确匹配。沿用第一问强制对齐得到的词级时间，177个证据窗口中173个可映射到估计时段，4个标记为 `unresolved`。附件4的 `13.pkl` 视觉特征全为零，按真实输入标为视觉不可用，也没有视觉关键帧。附件原特征没有逐行真实时间戳，词级时段属于可复核的估计对应。

## 重现与交付

从仓库根目录执行，环境版本见 `q23/environment-tested.txt`：

```bash
.venv-q1/bin/python -m q23.compare_models --epochs 25
.venv-q1/bin/python -m q23.select_models
.venv-q1/bin/python -m q23.selected_evaluate --include-test
.venv-q1/bin/python -m q23.selected_infer
.venv-q1/bin/python -m q23.selected_evidence
.venv-q1/bin/python -m q23.selected_report
.venv-q1/bin/python -m q23.selected_check
```

**最终专项结果**在 `outputs/q23/experiments/`：

- `attachment3_predictions_selected.csv`：30条第二问极性和强度预测。
- `attachment4_predictions_explanations_selected.csv`：20条第三问预测、模态贡献和主要证据。
- `attachment4_evidence_times_selected.csv`：全部关键证据的文字及估计时间。
- `attachment4_cards_selected.html`、`keyframes_selected/`：可视化解释卡与视频帧。
- `selected_metrics.json`、`selected_robustness.csv`、`selected_error_analysis.json`：验证、测试和错误分析。
- `shared_private.pt`、`reliability_proxy.pt`、`selection.json`：最终组合参数及选择记录。`text_anchor.pt`、其他实验历史和原模型仍保留作对照。

代码和结果可使用 `python -m q23.package_improved` 生成压缩交付包。预训练BERT权重通过文档固定的公开修订号下载，不放入压缩包。

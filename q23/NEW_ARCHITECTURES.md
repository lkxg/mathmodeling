# E题第二、三问：新增架构实验（2026-09-25）

在上一轮的三种上下文融合模型基础上，继续测试了两种新架构和一种新训练方法。三者都是结合本题 50 个对齐位置和已有特征做的**简化改造**，不是论文原模型的完整复现。训练、验证分别固定使用附件2的 3395、728 条；先前保留的 727 条测试数据没有参与本轮模型选择。附件3、4没有情绪真值，不能计算 Accuracy。

另设一个结构差异较大的核方法对照：每个模态按有效位置提取均值、标准差和可用比例，仅在训练集拟合标准化与 PCA，再用 RBF-SVM 分类、RBF-SVR 回归。代码在 `kernel_baseline.py`。它没有时序交互，也没有进行缺失增强，因此只比较完整输入。

## 方法与代码

|代码名|实现思路|文献启发|
|---|---|---|
|`dual_query`|三个可学习单模态查询分别汇聚有效位置，再由跨模态查询汇聚可用模态；局部门控融合与全局查询特征共同送入时序 Transformer|[Miyoshi 等，WACV 2026，Dual-Query Fusion](https://openaccess.thecvf.com/content/WACV2026/html/Miyoshi_Robust_Multimodal_Emotion_Recognition_from_Incomplete_Modalities_via_Query-Based_Unimodal_WACV_2026_paper.html)|
|`enhance_balance`|对每个模态做跨模态补偿，以样本和位置相关的可信度门控融合；原有单模态辅助分类头保持一致|[He 等，CVPR 2026，Enhance-then-Balance](https://openaccess.thecvf.com/content/CVPR2026/html/He_Enhance-then-Balance_Modality_Collaboration_for_Robust_Multimodal_Sentiment_Analysis_CVPR_2026_paper.html)|
|`dual_query_distilled`|用上一轮的完整输入最终组合作教师，缺失增强后的查询融合模型作学生；联合类别 KL、强度 Huber 与批内表示关系损失|[Zhuang 等，ICCV 2025，CMAD](https://openaccess.thecvf.com/content/ICCV2025/html/Zhuang_CMAD_Correlation-Aware_and_Modalities-Aware_Distillation_for_Multimodal_Sentiment_Analysis_with_ICCV_2025_paper.html)|

代码在 `fusion_variants.py`、`train_distill.py`。其中 `enhance_balance` **没有**实现原文的能量协调优化和教师可信度蒸馏；`dual_query_distilled` **没有**实现 CMAD 完整的多层关系蒸馏与模态难度自适应正则。因此以下数值只代表本仓库的简化实现。

两个新神经架构与蒸馏学生使用相同的冻结 BERT 文本上下文表示、训练集音视频标准化参数、数据增强、随机种子 2026、25 轮最大训练、batch size 128 和 AdamW。每轮模型选择分数为 `Accuracy + 0.05×Macro-F1 − 0.02×MAE`。模型之间和预先列出的九个候选组合只在验证集比较；组合等权平均类别 logits 和强度输出。核方法只作为独立基线，不参与组合选择。

## 实测结果

|方法|最佳轮次|完整输入 Accuracy|Macro-F1|强度 MAE|27种单模态缺口平均 Accuracy|4种多模态缺口平均 Accuracy|
|---|---:|---:|---:|---:|---:|---:|
|上一轮最终组合 `shared_private + reliability_proxy`|—|**0.6484**|**0.6137**|0.5947|0.6287|0.6113|
|查询式融合 `dual_query`|3|0.6415|0.6126|0.6184|0.6245|**0.6147**|
|增强平衡 `enhance_balance`|5|0.6470|0.6053|**0.5945**|**0.6301**|0.6137|
|查询式融合 + 蒸馏 `dual_query_distilled`|6|0.6332|0.5895|0.5972|0.6167|0.6006|
|核方法对照 `kernel_baseline`|—|0.6058|0.5538|0.6308|未评估|未评估|

单模态缺口指文本、音频、视觉各自缺失 10%／30%／50%，位置在有效序列前／中／后，共 27 种；多模态缺口指中间 30% 的四种同步缺失组合。表中缺口指标是场景 Accuracy 的算术平均，**不是**对样本数或未知真实缺失分布加权。单模型的微小差异尤其是 0.0014 约等于验证集一个样本，不足以声称稳定提升。

预先比较的九个候选中，加入增强平衡的三模型组合 Accuracy 为 0.6456、MAE 为 0.5904；加入蒸馏模型的三模型组合 Accuracy 为 0.6442、MAE 为 0.5890。两者强度 MAE 有所降低，分类 Accuracy 下降。**按预设的 Accuracy 优先规则，最终仍保留上一轮的两模型组合**，附件3的 30 条预测和附件4的 20 条解释也保持由它生成。该组合此前在保留测试集上的 Accuracy 0.6602、Macro-F1 0.5829；没有因本轮实验反复用测试集挑模型。

完整结果分别见 `outputs/q23/experiments/comparison_dual_query_enhance_balance.csv`、`dual_query_distilled_summary.json`、`new_architecture_comparison.csv` 和 `new_architecture_robustness.csv`；模型权重为同目录下的 `dual_query.pt`、`enhance_balance.pt`、`dual_query_distilled.pt`。`new_architecture_selection.json` 记录比较规则及胜出组合，`new_architecture_robustness_summary.json` 汇总缺口指标。
核方法的数值、预测和序列化参数分别在 `kernel_baseline_summary.json`、`kernel_baseline_valid_predictions.npz`、`kernel_baseline.joblib`。

## 复现

从仓库根目录、使用 `q23/environment-tested.txt` 中的环境运行：

```bash
.venv-q1/bin/python -m q23.compare_models --epochs 25 --models dual_query enhance_balance
.venv-q1/bin/python -m q23.train_distill --student dual_query --epochs 25
.venv-q1/bin/python -m q23.kernel_baseline
.venv-q1/bin/python -m q23.compare_new
.venv-q1/bin/python -m q23.new_robustness
.venv-q1/bin/python -m q23.selected_check
.venv-q1/bin/python -m q23.package_new
```

这些命令依赖上一轮已经训练好的 `shared_private.pt`、`reliability_proxy.pt`、`selection.json`、音视频标准化文件，以及固定版本的 BERT。若从头重现，先按 `IMPROVED.md` 的命令训练并选择上一轮组合。训练用遮挡文本 BERT 缓存可重建，不放进压缩包。

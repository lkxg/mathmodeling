# 第一问的补充验证

仅使用附件1的100条原始样本、原标注和冻结特征。这些实验检查对齐与特征设计，不替代问题2/3在附件2上的训练验证。

## 运行

先完成主流程并通过全量结构验证：

```bash
.venv-q1/bin/python -m q1 validate
.venv-q1/bin/python -m pip install -r q1/analysis/requirements.txt
.venv-q1/bin/python -m q1.analysis.alignment_quality
.venv-q1/bin/python -m q1.analysis.probe --threads 2
.venv-q1/bin/python -m unittest discover -s q1/tests -v
```

探针默认做499次组内标签置换、2000次按视频Bootstrap、17组配置。`--permutations`、`--bootstrap`、`--inner-folds`可调整，全部设置写入报告。`--output-name`指定analysis下的独立结果目录；默认`probe_v2`。旧版`analysis/probe.json`与`probe_predictions.csv`保留，仅供审计。

## VAD检查的含义

Silero VAD不读取转写，在10 ms网格比较语音活动与有效词区间。Precision衡量词区间中语音所占比例，Recall衡量检测出的语音被词区间覆盖的比例。循环平移对照保留词区间布局，用于观察真实位置的一致性是否更高。

待检查集合包括主流程的可疑样本、Recall低于0.5的样本和非静音但无法评分的样本。VAD判断的是语音活动，不能确认词身份或提供人工边界误差。语音几乎占满片段时平移检验区分力有限，p值不用于剔除样本。

## 完整特征与消融

将有效词位置的各导出坐标取均值，形成固定长度的片段向量。词时间无效时保留文本；音频和视觉使用各自component_mask。完全不可观测的分量记NaN，在每个训练折内填补。未分配区间不进入词级探针。

|组件|取值|维度|
|---|---|---:|
|文本|有效词向量均值|768|
|emotion2vec|有效词音频向量前768维的均值|768|
|LLD均值|导出audio的768:793列的均值|25|
|LLD词内标准差|导出audio的793:818列的均值|25|
|视觉均值|导出vision的0:28列的均值|28|
|视觉词内标准差|导出vision的28:56列的均值|28|

标准差坐标直接读取导出结果，不再改算各词均值之间的标准差。音频比较768、793、818维；视觉比较28、56维；融合比较有无两类标准差。另有单模态、双模态、完整融合及整段池化对照，共17组。BERT与ModernBERT采用相同的原文跨度映射、子词→词→片段池化及FP16读取精度，模型版本固定。

整段对照从有效源帧求全片段均值和标准差。它与词级汇总同时改变了保留区间、权重和统计尺度，因此不能把所有性能差异单独解释成对齐的作用。

各坐标在训练折内标准化，再除以所属完整特征家族参考维度的平方根：文本与emotion2vec为768，LLD为50，视觉为56。消融删除坐标时，其余坐标的权重不变。音频包含两个家族，不宣称三种模态严格等权。

## 分组验证与指标

- 外层按video_id留一，共37折。同一原视频的全部片段同时进入留出集。
- 每个外层训练集内使用5折GroupKFold选择正则化参数。每个内层训练折重新拟合缺失填补和StandardScaler。
- 回归使用带截距的Ridge，按内层汇总MAE选择参数。
- 三分类使用独热标签的一对多Ridge读出，以argmax给出Negative、Neutral、Positive，按内层汇总Macro-F1独立选择参数。不把回归符号当作三分类。
- 两个任务均使用事先固定的17档alpha（0.01至1000000），同分时选择更大的alpha。
- 主指标为全部100条上的MAE、Pearson、三分类Accuracy和Macro-F1。回归符号二分类只作为排除中性样本的辅助指标。
- 均值、中位数与多数类基线仅从外层训练集计算。

Ridge的标签线性性质允许复用训练折的矩阵分解，同时批量计算置换目标。每个置换仍独立完成参数选择。测试将优化实现与scikit-learn逐折、逐参数、逐置换重拟合的预测作比较。

## 统计检验

标签只在相同video_id内置换，保留每个视频的标签多重集。该检验以同视频内可交换为假设，检验组内特征与标签的关系，不检验完全抹除视频间关联的零假设。每次置换重做完整嵌套选择。默认499次，最小p值为0.002。每个指标的17项模型检验分别用Holm方法校正。

Bootstrap以原视频为抽样单位，被抽中视频的全部片段一起进入重采样，各方案共享抽样索引。成对差值的95%区间用于消融分析。该区间基于固定的外层预测，未重新拟合模型，也未作多重比较校正；只作为探索性证据。

方法依据：[scikit-learn分组交叉验证](https://scikit-learn.org/stable/modules/cross_validation.html)、[组内置换约束](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.permutation_test_score.html)。

## 输出与复现

`probe_v2/`包含汇总报告、逐样本预测、全部内外层索引及选中参数、图表，以及特征块和逐实验断点缓存。每次运行核验主流程产物，记录输入文件与分析代码的SHA256、库版本和随机种子。损坏或过期的分析缓存不复用。

分析代码与主流程位于不同模块，不需要重新提取原始视频特征。模型推理仅用于固定BERT对照；后续小样本线性代数主要在CPU执行。

论文可使用`report.md`、`summary.csv`、`predictions.csv`和`probe_comparison.svg/.pdf`；工作用`blocks.npz`和`checkpoints/`不必整体打包，提交时仍须核对全部题目的50 MB总限额。

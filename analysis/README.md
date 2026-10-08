# 实验结论与图表

当前研究结论以两轮固定90轮正式对照为主。每组seeds42/43/44，报告均值±样本标准差；
Public用于候选选择，Private用于最终报告。协议之间的旧结果不混合排名。

## 正式实验

| 实验 | 结论与方法 | 图表与聚合附件 |
|---|---|---|
| MicroResNet结构S0–S3：12次完整90轮，采用S1 | [结论](architecture_ce_v1/RESULTS.md)、[设计与预设门槛](../configs/plans/architecture_ce_v1.md) | [质量/成本图](architecture_ce_v1/architecture_comparison.png)、[PDF](architecture_ce_v1/architecture_comparison.pdf)、[训练曲线](architecture_ce_v1/training_curves.png)、[混淆矩阵](architecture_ce_v1/public_confusion_matrices.png)、[聚合指标](architecture_ce_v1/aggregate_metrics.json) |
| 三模型损失/采样A–D：36次完整90轮 | [结论](fixed_abcd_v3/RESULTS.md) | [Public对照](fixed_abcd_v3/public_comparison.png)、[PDF](fixed_abcd_v3/public_comparison.pdf)、[聚合指标](fixed_abcd_v3/aggregate_metrics.json) |

[数据说明](../data/README.md)披露类别不平衡、跨划分重复和评价边界；
[聚合审计](data_audit.json)保留统计方法与数据SHA。推理默认模型及来源见
[项目说明](../README.md#5-导出与推理应用)和[导出清单](../inference/saved_models/export_manifest.json)。

<details>
<summary>历史实验与效率附件</summary>

| 附件 | 口径与边界 |
|---|---|
| [CE第一阶段](ce_stage1/RESULTS.md)、[聚合指标](ce_stage1/aggregate_metrics.json) | 9次正式run，允许早停；不是后续完整90轮基线 |
| [AMP配对基准](benchmark_amp/aggregate_metrics.json) | 历史短程效率测量；不能线性外推90轮训练收益 |
| [旧效率基准](efficiency_results.json) | 指定环境、随机初始化模型；与本轮结构成本分别报告 |
| [全数据类别分布](class_distribution.png)、[样例图](sample_visualization.png) | 历史EDA背景图；源码统计全部Usage，不能冒充Training专属分布或正式实验图 |

</details>

## 来源与复算

版本化的`aggregate_metrics.json`包含组级均值/样本std、类别支持数、来源SHA和流程验收摘要。
逐样本标签/概率、完整运行记录、训练历史、原始计时和权重均保留本地，不随Git提供。

本地正式归档分别位于`analysis/architecture_ce_v1/`、`analysis/fixed_abcd_v3/`与`training/runs/`。
对外结果依靠原提交、冻结协议和数据指纹追溯；逐样本复算需要对应归档，
新实验运行方式见[复现说明](../REPRODUCIBILITY.md)。

历史开发图与备用训练Notebook保留本地且被忽略；当前索引仅链接随Git提供的材料。

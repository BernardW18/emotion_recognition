# 实验结论与图表索引

Git只保存聚合结论、必要来源指纹、图表与分析源代码；原始实验产物保留本地。

| 内容 | 可随仓库获得 | 本地原始记录（Git忽略） |
|---|---|---|
| 固定90轮A–D正式实验 | [结论](fixed_abcd_v3/RESULTS.md)、[聚合指标](fixed_abcd_v3/aggregate_metrics.json)、[图](fixed_abcd_v3/public_comparison.png)、[PDF](fixed_abcd_v3/public_comparison.pdf) | fixed_abcd_v3/results.json、runs.csv、evaluations/；training/runs/ |
| 历史允许早停CE | [原结论](ce_stage1/RESULTS.md)、[聚合指标](ce_stage1/aggregate_metrics.json) | ce_stage1/results.json、runs.csv、evaluations/ |
| 历史AMP配对基准 | [聚合指标](benchmark_amp/aggregate_metrics.json)及根README的结果说明 | benchmark_amp/results.json（含逐配对记录） |
| 数据质量审计 | [聚合审计](data_audit.json)；不含原始像素 | data/fer2013.csv、data/cache/ |
| 历史效率基准 | [参数/MACs/延迟聚合](efficiency_results.json)；不含逐次计时序列 | 运行时原始计时与训练输出 |
| 下一轮结构对照 | [12-run设计方案](../configs/plans/architecture_ce_v1.md) | 尚未生成新结构实验数据 |

`aggregate_metrics.json`仅存各组均值/样本std、类别支持数、计划/来源SHA及整体流程验收。
不存逐样本标签/预测/概率、每run完整元数据、训练历史或checkpoint。
源码/独立配置/冻结清单仍纳入Git，结论依靠原提交与数据指纹追溯；原始文件SHA允许核对本地归档。

历史根目录图表和分析Notebook保留为历史材料，不能将其与新的冻结协议结果混合排名。
旧CE结论中的results.json/runs.csv等引用是本地归档路径，新检出请使用本索引的聚合指标。
`docs/`继续整目录忽略，课程报告/PPT保留原样；取消跟踪不删除本地文件，不清除旧提交历史。
白名单只作用于生成的analysis目录；data_audit.json与efficiency_results.json为已有小型聚合报告，明确保留。

# 历史CE第一阶段正式实验结果（允许早停）

完成日期：2026-10-07。协议：`comparison-ce-v1`。执行提交：`cc195bd99360b1243fb7f031fc5b1adc2d10742f`。

三个现有模型，各seeds=42/43/44，共9次独立正式训练；全部正式准入通过。
普通CE、普通随机采样、常规增强开启、类别专属增强关闭；LR=3e-4、batch=128、最多90轮、允许冻结早停。
训练使用Windows项目.venv、CUDA AMP；best由PublicTest val_acc选择。最终评估统一CPU float32/batch64，每个best在PrivateTest上评估一次。
最终CPU float32的PublicTest准确率与训练AMP日志有少量差异，最大约0.084个百分点（准确率计数净差约3张）；保留原best选择，不重新挑选epoch。

## 汇总

各项为均值 ± 样本标准差（n=3），以百分数表示；±后的值是百分点。

| 模型 | PublicTest accuracy | PrivateTest accuracy | PrivateTest macro-F1 | PrivateTest balanced accuracy | Disgust recall |
|---|---:|---:|---:|---:|---:|
| mini_cnn | 60.86 ± 2.04 | 61.98 ± 2.17 | 52.57 ± 2.89 | 52.66 ± 2.23 | 5.45 ± 4.81 |
| vgg_lite | 61.54 ± 3.03 | 62.24 ± 3.29 | 50.61 ± 3.70 | 51.96 ± 2.91 | 0.00 ± 0.00 |
| micro_resnet | 64.25 ± 0.22 | 64.76 ± 0.38 | 58.76 ± 1.15 | 58.34 ± 0.87 | 29.09 ± 5.45 |

## 每次运行

| 模型 | seed | run_id | 完成轮数 | best轮数 | 结束原因 | PrivateTest accuracy | PrivateTest macro-F1 |
|---|---:|---|---:|---:|---|---:|---:|
| mini_cnn | 42 | `20261007_070919_seed42` | 85 | 75 | val_acc | 63.69% | 53.91% |
| mini_cnn | 43 | `20261007_071225_seed43` | 38 | 28 | val_acc | 59.54% | 49.26% |
| mini_cnn | 44 | `20261007_071358_seed44` | 89 | 79 | val_acc | 62.69% | 54.55% |
| vgg_lite | 42 | `20261007_071712_seed42` | 36 | 26 | val_acc | 60.18% | 48.29% |
| vgg_lite | 43 | `20261007_072010_seed43` | 89 | 79 | val_acc | 66.04% | 54.88% |
| vgg_lite | 44 | `20261007_072711_seed44` | 34 | 24 | val_acc | 60.49% | 48.66% |
| micro_resnet | 42 | `20261007_073002_seed42` | 38 | 28 | val_acc | 65.14% | 59.20% |
| micro_resnet | 43 | `20261007_073211_seed43` | 40 | 30 | val_acc | 64.75% | 59.64% |
| micro_resnet | 44 | `20261007_073427_seed44` | 38 | 28 | val_acc | 64.39% | 57.46% |

## 验收与文件

9/9正式准入通过；18份完整评估，逐份3,589条预测可复算accuracy/macro-F1/balanced accuracy。
原有122个CSV/缓存/权重/runs/分析/报告文件SHA-256保持一致，冻结文件未改变。

- `results.json`：计划指纹、run目录、权重SHA、全部指标、各类recall/support、均值和样本标准差、验收结果。
- `runs.csv`：逐run结果与完整轮数/结束原因。
- `evaluations/<model>_seed<seed>/<split>/summary.json`及`predictions.csv`：统一评估产物。

## 解读限制

官方划分仍存在完全相同像素的跨划分重叠：PublicTest 280条、PrivateTest 288条与Training重叠；本轮不属于去重训练。
PrivateTest的Disgust仅55张，一张约对应1.82个百分点recall；不能把三种子描述统计当成显著性证据。
现有模型activation/dropout不同，结果比较的是固定模型定义及共同训练预算。
本轮只完成CE基线，不能推断Focal/加权采样/CB-Focal的收益，也不与历史不同训练配置的权重作受控排名。
耗时来自last完整轮断点，不含进程启动和最终评估；各run可能早停，不能直接视为相同训练工作量下的速度排名。

## 后续正式实验

以上记录属于允许早停的历史CE阶段。后续已完成[固定90轮A–D对照](../fixed_abcd_v3/RESULTS.md)，
以及[MicroResNet结构对照](../architecture_ce_v1/RESULTS.md)，当前推理采用S1。
本阶段与后续实验具有独立协议，不将它们合并为同一组对照。

机器可核验的组级结果与来源见[聚合指标](aggregate_metrics.json)。
上面的results.json、runs.csv和逐样本预测路径均为本地归档，不随Git提供。
冻结清单的权威归档为[comparison-ce-v1](../../configs/protocols/comparison-ce-v1.json)，
验证历史run时显式指定该归档；方法见[复现说明](../../REPRODUCIBILITY.md)。

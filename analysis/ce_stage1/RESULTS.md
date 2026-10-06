# CE第一阶段正式实验结果

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

## 当前结果暴露的问题、解决方案与验收

### CE01：早停与学习率重启可能耦合，部分种子结果不稳定

**证据：** 6/9个run在第34–40轮结束，均由val_acc连续10轮无改善触发；第一次cosine_warm重启发生在第30轮之后。
另外三个run运行85/89/89轮。MiniCNN与VGGLite的PrivateTest accuracy样本标准差分别为2.17/3.29个百分点。
当前曲线只能支持关联，不能认定重启必然导致低分；本轮结果符合冻结规则，90轮是上限而非强制完成轮数。

**可能的解决方案：** 在单独的新协议中只比较CE训练策略，先采用共同90轮预算，
将patience=0、val_loss_patience=0以关闭两种早停，其余模型/损失/采样/增强/LR曲线保持一致。
仅依据Training/PublicTest审查曲线与选择best；与本轮早停策略分开命名，不把不同预算结果混为同一组。
该方案尚未执行，不能据已经取得的PrivateTest成绩挑选参数。

**验收标准：** 新协议的三个模型×三种子9次均完整到90轮并通过正式准入，或明确记录失败；
有效配置证明仅早停策略改变。报告PublicTest均值±样本标准差、每run best轮数、实际时间，
确认第30轮重启后的恢复过程被完整保留；不以必须涨分作为流程验收条件。
每份最终best仅做一次PrivateTest评估，保留本轮全部原run及指纹。

### CE02：少样本类别识别不足，尤其VGGLite的Disgust召回为零

**证据：** PrivateTest Disgust支持数为55，三个种子的平均召回分别为MiniCNN 5.45%、
VGGLite 0%、MicroResNet 29.09%。总体accuracy不能代替各类别效果；PublicTest也需按保存的各类指标审查。

**可能的解决方案：** 按已有A→B→C→D草案分别评估Focal、加权采样、CB-Focal，
独立配置仅改变该步规定的一项，关闭类别专属增强；先定稿预算再冻结对照。
使用PublicTest的macro-F1、balanced accuracy和Disgust recall判断候选，不根据PrivateTest调参。

**验收标准：** 每个模型/臂至少seeds=42/43/44，所有实际criterion/采样器与冻结清单相符，
逐run保留七类recall/support和逐样本预测，汇总均值及样本标准差。
候选只有在PublicTest平均Disgust recall高于同协议CE且macro-F1均值不下降时，才记为该问题的有效改善；
否则如实记录未改善。最终PrivateTest仅用于一次完整报告，三种子描述统计不冒充显著性证明。

### CE03：Git换行转换会破坏冻结文件的字节绑定（已修复）

**问题：** 原提交的冻结JSON被仓库自动转为LF，Git中SHA-256为
`58c97fda654427a6f9bb0a990d20c6787dc0f553b047e9717df1ad9038a8ee66`，
与九份run绑定的本地实际文件`82089f0cc62e2bc40ffceeee826f72d68d862f7277bfe9ca2733e567d3619ee5`不同。
重新检出可能导致本来正确的run无法通过协议文件SHA校验。

**已实施修复：** `.gitattributes`为冻结清单设置`-text`，按原字节保存；
本地冻结文件、实验配置及结果数值不变。新评估JSON/CSV/Markdown统一LF，刷新其产物哈希。

**验收：** Git内容过滤前后冻结文件字节一致；在系统临时Git仓库实际add/commit/删除副本/checkout后，
清单SHA仍为`82089f…`；临时仓库已清理。主仓库这条保存规则与冻结文件应一起提交，
更新主仓库存储时需显式执行`git add --renormalize -- docs/comparison_protocol_frozen.json`，
再将保存规则与清单一起提交；本地文件字节未修改。

# 模型比较协议（comparison-v2）— CE 第一阶段已冻结

**当前状态：CE 第一阶段已冻结并完成。** 用户确认第一阶段仅执行三个模型的
A 臂（普通 CE 基线），实际清单为 [comparison_protocol_frozen.json](comparison_protocol_frozen.json)，
协议 id 为 `comparison-ce-v1`，冻结/完成日期 2026-10-07，代码基准
`485a2782b3eed934c5c61ca68b978c8d370986a5`，执行提交 `cc195bd`。每模型 seeds=42/43/44，
共9次独立训练，9/9正式准入通过，18份统一评估完成；结果见
[CE第一阶段结果](../analysis/ce_stage1/RESULTS.md)。
B/C/D/E 与 AMP ON/OFF 比较仍是后续草案，不属于本次冻结范围；不能据第一阶段结果宣称其已完成。
冻结后不得根据 PrivateTest 结果调整方案。

**版本记录**：v1（初稿）→ v2（2026-10-07，按第二轮审核 R08 更新：新增 CE 基线臂、
选定"固定共同方案"预算口径并写入统一超参数值）→ v2.1（2026-10-07，性能轮 PB01–PB05
完成后更新：增强实现统一为 batched；optimizer_fused 保持默认 false）。
预算表数值已选定；CE 第一阶段已获用户确认并冻结，其余比较臂尚未冻结。

## 1. 范围与目标

- 比较对象：MiniCNN / VGGLite / MicroResNet（现有三架构，不改结构、不换数据集）。
- 比较问题：
  - **F11**：三个模型在同一训练配置下的性能差异（≥3 seeds）；
  - **F07**：不平衡策略的逐项分解——普通 CE 基线、Focal、加权采样、
    Class-Balanced Focal（新增显式 CE 基线臂，经同一入口
    `--config configs/baseline_config.yaml` 启动）；
  - **F12（训练层面）**：AMP ON/OFF 对训练速度与最终精度的影响
    （速度部分已有配对基准 `analysis/benchmark_amp/results.json`）。
- 不在本协议内：新架构、分类头改造、其他数据集、论文迁移。

## 2. 数据与划分

- 使用官方划分（Training 28,709 / PublicTest 3,589 / PrivateTest 3,589）；
  重叠披露见 [`docs/data_audit.md`](data_audit.md)。
- **模型选择只用 PublicTest**；PrivateTest 仅在方案冻结后、对最终 `best.pth`
  评估一次（通过统一评估入口），不用于任何中间决策。
- 统计口径：每个配置 ≥3 seeds（42 / 43 / 44）；报告 mean ± std（样本标准差），
  并给出每个 run 的 run_id（完整追溯：run_meta + config_effective + history）。

## 3. 预注册的训练设置

### 3.1 预算口径：固定共同方案（已选定）

三个模型采用共同的优化/训练预算，模型定义保留现有 activation/dropout 差异，
以冻结清单中的 `model_spec` 为准，因此不能声称仅改变了网络拓扑。
CE 第一阶段使用 `configs/baseline_config.yaml`；共同训练设置为：

| 维度 | 统一值 |
|---|---|
| 优化器 / 学习率 | adam / **3e-4**（各模型段统一） |
| batch size | **128**（各模型段统一） |
| epochs 上限 | **90**（早停可提前停止；轮数口径一致） |
| 调度器 | cosine_warm（t0=30, T_mult=2，全局值） |
| 早停 | patience=10 / val_loss_patience=15 / threshold=1.05（全局值） |
| 权重衰减 | 1e-4 |
| 精度 | AMP 自动（GPU）；cudnn_deterministic=false |
| seeds | 42 / 43 / 44（每配置 3 个独立 run） |
| 增强 | 常规图像增强开启（统一）；类别专属增强与采样开关按"臂"定义（见 3.2） |

> 执行说明：**已于 2026-10-07 应用**——`configs/training_config.yaml` 与
> `configs/baseline_config.yaml` 的三个模型段已统一为上述预算值（防漂移测试覆盖）。
> CE 第一阶段的冻结日期和 commit 已登记于实际清单；冻结后不得改动绑定源码/配置。
> 历史各模型的差异化超参（mini 5e-4/256/30 等）不再用于正式比较。
> **增强实现（2026-10-07 追加）**：正式比较统一使用 `augmentation.impl="batched"`
> （batched-v1；部分参数范围/门控沿用，输出分布与 legacy 不同；批种子规则
> `sha256(train_seed|epoch|batch_idx)` 随训练协议快照记录）。
> 第三轮独立审核更正：插值由 nearest 改为双线性，旋转/平移合并重采样，不能称为同分布。
> 正式冻结须写明具体变换规则；实际实现/模型/臂保持统一，切换实现需新建 run（T05）。
> 第四轮独立审核（6610651）：主/基线已默认 workers=0、persistent_workers=false，T02 通过。
> U01/U02已通过v3状态与事务回滚验收，PB06批级保护取数已实施。
> U03准入校验已补齐实际清单和真实产物；CE 第一阶段已完成9次正式训练及准入，其他臂仍未执行。
> **fused Adam**：`optimizer_fused` 保持默认 false——PB03 实测完整流程中位改善
> ~4.9–5.2%（两轮 9 组配对），未稳定达到 ≥5% 判据，保留为可选项。

### 3.2 对照臂设计（每臂相对基线只改变一项）

| 臂 | 损失 | 采样 | 类别专属增强 | 起点配置 |
|---|---|---|---|---|
| A · 基线 | cross_entropy | 普通随机 | 关 | `configs/baseline_config.yaml` |
| B | focal (γ=2.0) | 普通随机 | 关 | A + `loss_type=focal` |
| C | focal (γ=2.0) | 加权采样 | 关 | A + `loss_type=focal` + `class_balanced_sampling=true` |
| D | cb_focal (β=0.999) | 加权采样 | 关 | A + `loss_type=cb_focal` + `class_balanced_sampling=true` |

- 每臂 3 个 seeds；A→B→C→D 每步只改变一项，收益可归因。
- 类别专属增强如需评估，追加为独立臂 E（A + `class_specific.enabled=true`），
  不与主表混排。
- CE 基线经同一入口启动：`python training/train.py --config configs/baseline_config.yaml`。

### 3.3 其余约束

- 每个 run 独立 `run_id`；`run_meta.json` 记录 git commit + 脏标记 + 数据指纹。
- 恢复训练前经训练协议比对 v2（完整生效配置 + 数据指纹；S02）与精确恢复条件
  （S01：`workers=0` 或 `workers≥1` 且 `persistent_workers=false`；恢复端数据管线
  规格必须与断点一致，`persistent_workers=true` 明确拒绝）。
- 早停与调度器超参数在所有比较臂中完全一致；不使用 PrivateTest 做任何中间决策。
- **AMP 对比**：同 seed、同配置的 OFF/ON 两个 run 配对；报告完整流程时间
  与最终指标差异；加速比以配对基准或完整流程时间为准，不用单步时间外推。

## 4. 评估口径

- 唯一允许的评估方式：`utils/evaluation.evaluate_checkpoint`（显式 checkpoint + split）。
- 每个 run 评估其 `best.pth` 一次：报告 accuracy / macro-F1 / balanced accuracy /
  各类 recall 与支持数；同时保存逐样本预测（可复算）。
- Disgust 支持数（PrivateTest 55 张）在报告中注明：**一张样本约 1.82 个百分点**
  recall 变化，解读时注意粒度。

## 5. 判定与报告规则

- A–D 臂比较：以 **macro-F1 为主指标**（类别不平衡场景），accuracy 为次；
  同时报告 Disgust recall 与 CE 基线对照。
- 模型对比：accuracy 为主、macro-F1 并报；**不计算**"每千参数贡献准确率"类指标。
- 不预设"必须提升"：结论允许是"无显著差异"；如实报告全部结果，包括反例。
- 若实施附加去重协议（`dedup-v1`），其重训结果**单独命名与报告**，
  不与官方协议结果混排（见 docs/data_audit.md §3）。

## 6. 冻结流程（U03 实现已验收，CE 第一阶段已冻结）

1. 评审第3节的预算与各臂，将实际实现/配置定稿提交；运行新增的
   tools/freeze_comparison.py --plan MODEL ARM CONFIG（可重复）--seeds 42 43 44
   --protocol-id 协议名 --output docs/comparison_protocol_frozen.json。
   工具要求可追溯的干净代码基准，读取真实模型/配置/数据管线，先验证再写文件。
   至少冻结3个不同seeds；每个模型/臂组合唯一，不能用同一实际方案重复命名不同臂。
2. 冻结文件schema_version=1，包含protocol_id、frozen_at、git_commit、code_sha256及plans。
   每个plan记录model_name/arm/model_spec/seeds、完整training_protocol、max_epochs、
   allow_early_stop与lr_floor=1e-7。源码和配置字节指纹、实际参数、数据/划分指纹、
   loader/增强版本/损失/调度器/精度均绑定；seed是清单允许的唯一计划变量。
   只含id/日期/commit的旧标记JSON不再可用。清单变更或源码/配置变化须新协议/run。
3. 使用 training/train.py --purpose formal --config 对应配置 --model 对应模型
   --seed 清单seed --epochs 本会话轮数。启动、续训和准入均核对实际清单；
   超预算拒绝。未完成预算且无允许的早停时是session_completed，不进入正式汇总；
   达到预算或可复算的合规早停后才finished，且禁止继续追加该正式run。
4. 汇总前验证v3 last/best真实权重与模型结构/状态、run/数据/协议、history、
   config_effective和meta；partial、坏/外国断点、伪造完成标记或漂移设置均拒绝。
   各模型/臂所有seeds按第4/5节报告，缺run时不宣称完整比较。

示例（仅三个模型CE基线A；其他已定义臂需在同一次冻结命令追加实际配置）：

    .\.venv\Scripts\python.exe tools/freeze_comparison.py --protocol-id comparison-v2-frozen1 --output docs/comparison_protocol_frozen.json --seeds 42 43 44 --plan mini_cnn A configs/baseline_config.yaml --plan vgg_lite A configs/baseline_config.yaml --plan micro_resnet A configs/baseline_config.yaml

上面的命令为生成流程示例，不要重新执行或覆盖已创建的 CE 清单。当前第一阶段的
实际冻结与启动方法见下节；其他臂仍是草案，后续需独立定稿并处理新的协议绑定。

## 7. CE 第一阶段启动清单（2026-10-07）

实际清单 SHA-256：`82089f0cc62e2bc40ffceeee826f72d68d862f7277bfe9ca2733e567d3619ee5`。
源代码/配置 SHA-256：`a9b05cc1e4abf55561513216f09df491b37c5fdd0700fe7fbc95fa32990dce71`。
三个模型各执行 seeds=42、43、44，顺序建议 MiniCNN → VGGLite → MicroResNet。
每次独立 run；loss=cross_entropy、普通随机采样、类别专属增强关闭、常规增强开启。
统一 LR=3e-4、batch=128、最多90轮，允许现有冻结早停，GPU AMP开启、workers=0。
`best.pth` 按 PublicTest val_acc 选择；最终并报 accuracy/macro-F1/balanced accuracy/各类 recall，
按每模型3个seed报告均值及样本标准差。保留官方划分及其重复披露，不宣称去重训练。

冻结时的准备检查已通过：Windows `.venv`、PyTorch 2.14.1+cu130、RTX 5070 Ti Laptop；
九组真实配置均通过正式初始化与计划匹配，并以真实128张训练样本完成GPU AMP前向和反向，
loss/梯度有限；没有optimizer更新、没有正式epoch、没有创建持久run，临时目录已删除。
这只验证启动与单批计算，不代表完成正式实验或新增精度结论。

建议把本次冻结清单与同步文档提交一次，以保持正式run的Git记录干净。
这些文档/清单的提交不会改变已绑定的源码/配置指纹。运行前接通电源并关闭自动睡眠。
从项目根目录用PowerShell启动第一项：

```powershell
Set-Location 'D:\Document\Unniversity\emotion_recognition'
.\.venv\Scripts\python.exe -X utf8 training/train.py --config configs/baseline_config.yaml --model mini_cnn --seed 42 --epochs 90 --purpose formal
```

结束后依次将seed改为43/44，再对vgg_lite、micro_resnet执行三种seed，共九次。
首次从头训练不传 `--resume`。不要直接省略 `--config` 或 `--purpose formal`，
默认主配置不是本阶段CE基线，默认用途smoke也不属于正式实验。

中断时按一次Ctrl+C，等待保存退出；续训显式指定同run的 `checkpoints/last.pth`，
仍带同一config/model/seed/purpose，`--epochs` 是本会话追加轮数，不能使累计轮数超过90。
例如完整训练到第20轮时最多追加70轮；已完成预算或合规早停的finished run不再续训。
尽量不用 `--resume auto`，以免选择到其他seed/配置的run。

第一项结束后检查 run_meta 为finished/experiment_completed=true、history完整，
last/best可读且正式准入通过，然后继续其他八项。PublicTest用于既定模型选择；
每个完成run的best在PrivateTest上只做一次最终评估，保存逐样本预测，不据此调整配置。

## 8. 本阶段完成记录

执行提交 `cc195bd99360b1243fb7f031fc5b1adc2d10742f`；九次训练均从头开始，创建独立run，
运行时Git干净且源码/配置/实际冻结文件未改变。九次均按冻结的val_acc早停结束，
完整轮数分别为MiniCNN 85/38/89、VGGLite 36/89/34、MicroResNet 38/40/38。
每份best统一以CPU float32/batch64评估PublicTest与PrivateTest各一次；保存的预测可以复算主要指标。
CPU与训练AMP的PublicTest准确率最多差约0.084个百分点，best仍按原训练日志选择。

PrivateTest accuracy均值±样本标准差（n=3）：MiniCNN 61.98±2.17%、
VGGLite 62.24±3.29%、MicroResNet 64.76±0.38%；macro-F1分别52.57±2.89%、
50.61±3.70%、58.76±1.15%。完整逐run/各类指标与指纹见 `analysis/ce_stage1`。
本轮仍是官方划分CE基线。早停/重启耦合和少样本类别问题的待验证方案、验收标准见结果文档；
相关新方案尚未冻结/执行，当前源码和配置保持原样。

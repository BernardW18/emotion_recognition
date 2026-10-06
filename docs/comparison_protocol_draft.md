# 模型比较协议（草案 comparison-v2）— 未冻结

**状态：草案。** 正式重训前必须评审并**冻结**（记录冻结日期与 git commit）；冻结后
不得根据 PrivateTest 结果调整方案。本文件为"先冻结方案、再启动重训"提供预注册口径。

**版本记录**：v1（初稿）→ v2（2026-10-07，按第二轮审核 R08 更新：新增 CE 基线臂、
选定"固定共同方案"预算口径并写入统一超参数值）→ v2.1（2026-10-07，性能轮 PB01–PB05
完成后更新：增强实现统一为 batched；optimizer_fused 保持默认 false）。
预算表数值已选定，冻结仍需用户评审。

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

三个模型使用**完全相同的训练超参**，差异只来自架构本身。正式比较执行时，
把 `configs/training_config.yaml` 的三个模型段统一为：

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
> `configs/baseline_config.yaml` 的三个模型段已统一为上述值（差异只来自架构本身；
> 防漂移测试覆盖）。冻结日期 + commit 待正式重训提交前登记；冻结后不得改动。
> 历史各模型的差异化超参（mini 5e-4/256/30 等）不再用于正式比较。
> **增强实现（2026-10-07 追加）**：正式比较统一使用 `augmentation.impl="batched"`
> （batched-v1；部分参数范围/门控沿用，输出分布与 legacy 不同；批种子规则
> `sha256(train_seed|epoch|batch_idx)` 随训练协议快照记录）。
> 第三轮独立审核更正：插值由 nearest 改为双线性，旋转/平移合并重采样，不能称为同分布。
> 正式冻结须写明具体变换规则；实际实现/模型/臂保持统一，切换实现需新建 run（T05）。
> 当前主/基线配置仍默认 persistent_workers=true，须先修复 T02 才能支持精确续训。
> 来源绑定成功不等于正式实验准入；正式汇总还需用途、冻结和完成状态校验（T06）。
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

## 6. 冻结流程（T06 机制已实施）

1. 本草案评审（预算表 3.1 数值、臂设计 3.2、规则取舍）；
2. 冻结：创建 `docs/comparison_protocol_frozen.json`，至少包含：

   ```json
   {"protocol_id": "comparison-v2-frozen1", "frozen_at": "YYYY-MM-DD",
    "git_commit": "<草案定稿 commit>", "notes": "可选说明"}
   ```

   冻结后不得改动：run_meta 会记录该文件字节 SHA-256，正式准入判定时复核一致性；
   事后更换冻结内容会使既有 run 失去正式资格（`check_formal_eligibility`）；
3. 启动重训：`python training/train.py --purpose formal ...`
   （按 3.2 的臂 × 3 seeds；执行前确认修复轮次全部复核）。
   未冻结 / 未声明 formal 的 run 一律判定为"非正式（流程验证/诊断）"，
   可独立查看但不进入正式汇总；
4. 按本协议第 4/5 节报告；正式比较前本协议状态保持"待冻结/实验未执行"。

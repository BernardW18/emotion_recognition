# 模型比较协议（草案 comparison-v2）— 未冻结

**状态：草案。** 正式重训前必须评审并**冻结**（记录冻结日期与 git commit）；冻结后
不得根据 PrivateTest 结果调整方案。本文件为"先冻结方案、再启动重训"提供预注册口径。

**版本记录**：v1（初稿）→ v2（2026-10-07，按第二轮审核 R08 更新：新增 CE 基线臂、
选定"固定共同方案"预算口径并写入统一超参数值）。预算表数值已选定，冻结仍需用户评审。

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

> 执行说明：冻结时（记录日期 + commit）一次性把主配置模型段改为统一值并提交；
> 运行期间不得改动。历史各模型的差异化超参（mini 5e-4/256/30 等）不再用于正式比较。

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
- 恢复训练前经训练协议比对（R02）与精确恢复条件（R01，`num_workers=0`）。
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

## 6. 冻结流程

1. 本草案评审（预算表 3.1 数值、臂设计 3.2、规则取舍）；
2. 冻结：记录冻结日期 + git commit，之后不再改动；
3. 启动重训（按 3.2 的臂 × 3 seeds；执行前确认 R01–R07 修复已复核）；
4. 按本协议第 4/5 节报告；正式比较前本协议状态保持"待冻结/实验未执行"。

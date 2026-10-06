# 模型比较协议（草案 comparison-v1）— 未冻结

**状态：草案。** 正式重训前必须评审并**冻结**（记录冻结日期与 git commit）；冻结后
不得根据 PrivateTest 结果调整方案。本文件为"先冻结方案、再启动重训"提供预注册口径。

## 1. 范围与目标

- 比较对象：MiniCNN / VGGLite / MicroResNet（现有三架构，不改结构、不换数据集）。
- 比较问题：
  - **F11**：三个模型在同一训练配置下的性能差异（≥3 seeds）；
  - **F07**：Focal Loss vs CB Focal Loss 的差异；
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

- 基座配置：`configs/training_config.yaml`（以冻结 commit 为准）。
- 每个 run 独立 `run_id`；`run_meta.json` 记录 git commit + 脏标记 + 数据指纹。
- 早停与调度器超参数在所有比较臂中完全一致；不使用 PrivateTest 做任何中间决策。
- **Focal vs CB-Focal**：仅切换 `loss_type`（及其 `cb_focal_beta`），
  增强、采样、LR、epochs、seed 全部保持一致。
- **AMP 对比**：同 seed、同配置的 OFF/ON 两个 run 配对；报告完整流程时间
  （训练日志）与最终指标差异；speedup 以配对基准或完整流程时间为准，
  不用单步时间外推。

## 4. 评估口径

- 唯一允许的评估方式：`utils/evaluation.evaluate_checkpoint`（显式 checkpoint + split）。
- 每个 run 评估其 `best.pth` 一次：报告 accuracy / macro-F1 / balanced accuracy /
  各类 recall 与支持数；同时保存逐样本预测（可复算）。
- Disgust 支持数（PrivateTest 55 张）在报告中注明：**一张样本约 1.82 个百分点**
  recall 变化，解读时注意粒度。

## 5. 判定与报告规则

- Focal vs CB-Focal：以 **macro-F1 为主指标**（类别不平衡场景），accuracy 为次；
  同时报告 Disgust recall。
- 模型对比：accuracy 为主、macro-F1 并报；**不计算**"每千参数贡献准确率"类指标。
- 不预设"必须提升"：结论允许是"无显著差异"；如实报告全部结果，包括反例。
- 若实施附加去重协议（`dedup-v1`），其重训结果**单独命名与报告**，
  不与官方协议结果混排（见 docs/data_audit.md §3）。

## 6. 冻结流程

1. 本草案评审（含规则 2/3 的取舍）；
2. 冻结：记录冻结日期 + git commit，之后不再改动；
3. 启动重训（3 模型 × 3 seeds；按需追加 Focal/CB-Focal 与 AMP 臂）；
4. 按本协议第 4/5 节报告。

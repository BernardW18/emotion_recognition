# FER2013 人脸表情识别

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

基于 PyTorch 的人脸表情识别系统：三个 CNN 模型（MiniCNN / VGGLite / MicroResNet）的训练、
统一评估与对比，以及 Streamlit 推理演示应用。

仓库：<https://github.com/BernardW18/emotion_recognition>（实际 remote）

**核心栈**: PyTorch + Streamlit + Jupyter + scikit-learn

---

## 特性概览

- **统一模型构造（model_spec）**：训练保存、离线评估、Notebook、推理应用共用同一套
  版本化模型规格与加载逻辑；结构参数（激活函数 / dropout / SE）不再因默认值退化而不一致。
- **run 隔离**：每次训练写入独立目录 `training/runs/<模型>/<run_id>/`（生效配置、
  run 元数据、history、last/best 断点），不同 seed/配置的运行完全分开、可反查。
- **完整续训**：`last.pth` 保存 RNG / AMP scaler / 早停计数 / best 状态；恢复后与
  连续训练逐批次一致（回归测试验证）；中断断点明确标记 `partial`。
- **早停**：val_acc 无改善（`training.patience`）与 val_loss 恶化（高于历史最优的倍率，
  `val_loss_patience` + `val_loss_threshold`）双监控，触发时记录原因；`=0` 为禁用。
- **统一评估入口**：显式 checkpoint + split，输出完整指标、逐样本预测与概率、
  权重/划分哈希（全部可复算，含 ECE/NLL）。
- **显式导出 + 来源清单**：`tools/export_model.py` 导出到推理目录并登记
  `export_manifest.json`（run、SHA-256、model_spec、指标），默认不覆盖同名文件。
- **恢复保护（S01/S02）**：训练协议为**完整生效配置快照 + 运行时有效值 + 数据指纹**，
  恢复前整表比对，任何影响训练状态的配置/数据变化都会被拒绝（换策略请新建 run）；
  **精确恢复（与连续训练逐批一致）支持 `workers=0` 或 `workers≥1` 且
  `persistent_workers=false`**（独立 generator 复算批次序列与 worker 种子），
  `persistent_workers=true` 明确拒绝；中断后同实例继续会自动回滚到完整 last。
- **性能（PB01–PB05）**：批级张量增强（`augmentation.impl`，同分布实测提速
  88.4%（workers=0）/ 39.0%（workers=4））；uint8 像素缓存（工厂加载 8.7→0.24s，逐位一致）；
  评估快速路径（完整评估 -96.0%）；Grad-CAM 按需 + 缓存（命中 -92.6%~-95.5%）；
  fused Adam 为可选项（默认关，实测 ~4.9–5.2%）。
- **质量门槛**：pytest（197 项）+ ruff + mypy（全项目 42 个源文件 0 错误）全部通过。

---

## 项目结构

```
emotion_recognition/
├── data/                        # 数据集（需手动下载）
│   ├── fer2013.csv              # FER2013 CSV（不纳入版本控制）
│   ├── cache/                   # uint8 像素缓存（可重建派生物，不纳入版本控制）
│   └── README.md
├── models/                      # CNN 模型定义（可配置激活函数）
│   ├── mini_cnn.py              # 基线模型（1,274,823 参数）
│   ├── vgg_lite.py              # VGG 变体（5,407,687 参数）
│   └── micro_resnet.py          # 微型残差网络（753,991 参数，可选 SE）
├── training/                    # 训练子系统
│   ├── train.py                 # CLI 训练脚本（推荐使用）
│   ├── trainer.py               # 共享训练引擎（Trainer 类）
│   ├── checkpoint.py            # run 目录 / Checkpoint 管理 + ipywidgets 选择器
│   ├── notebooks/               # 01_eda / 02_03_04 训练 Notebook（与 CLI 同一接口）
│   ├── runs/                    # 训练运行目录（不纳入版本控制）
│   ├── checkpoints/  logs/      # legacy 历史产物（只读保留，信息不全）
├── inference/                   # 推理演示应用
│   ├── app.py                   # Streamlit 应用
│   ├── service.py               # 缓存化分析服务（PB04：按需 Grad-CAM + LRU 缓存）
│   ├── infer_utils.py           # 推理工具（统一加载、预处理、Grad-CAM）
│   └── saved_models/            # 推理权重 + export_manifest.json
├── utils/                       # 公共工具模块
│   ├── model_spec.py            # 模型规格与统一构造/加载（F01）
│   ├── evaluation.py            # 统一评估入口（F10）
│   ├── config_validation.py     # 配置集中校验（F13/R07）
│   ├── stdio.py                 # 控制台/日志 UTF-8 统一（R09）
│   ├── activations.py           # 激活函数工厂
│   ├── losses.py                # Focal Loss + CB Focal Loss
│   └── constants.py             # 类别名称 / Emoji
├── configs/
│   ├── training_config.yaml     # 集中式训练配置（YAML）
│   └── baseline_config.yaml     # CE 基线配置（R08：普通采样 + 关类别增强）
├── tools/                       # 工具脚本
│   ├── export_model.py          # 导出权重到推理目录（附来源清单）
│   ├── evaluate_checkpoint.py   # 统一评估入口的 CLI 包装
│   ├── data_audit.py            # 数据审计（官方披露 + 敏感性分析）
│   ├── benchmark_efficiency.py  # 效率基准（参数/MACs/延迟/显存）
│   └── run_amp_comparison.py    # AMP 配对基准
├── analysis/                    # 数据分析与评估结果
│   ├── data_audit.json          # 数据审计输出
│   ├── efficiency_results.json  # 效率基准输出
│   ├── evaluations/             # 统一评估结果（指标 + 逐样本预测）
│   ├── benchmark_amp/           # AMP 配对基准输出
│   ├── comparison_report.ipynb  # 三模型对比分析
│   └── confusion_matrices/  roc_curves/  training_curves/
├── docs/
│   └── data_audit.md            # 数据重复披露与去重协议草案
├── tests/                       # 测试（197 项：核心/训练管线/推理/应用/配置/像素缓存/批级增强/推理服务/fused）
├── pyproject.toml               # 项目配置 + ruff + mypy
├── requirements.txt             # 依赖安装入口（CUDA 组合）
└── README.md
```

---

## 快速开始

### 1. 安装依赖（Windows + CUDA 13.0）

> 已验证组合：`torch==2.14.1+cu130` / `torchvision==0.29.1+cu130`（RTX 5070 Ti Laptop，驱动 596.49）。
> 注意：**先升级 pip**（新创建的虚拟环境中曾遇到旧 pip 解析 CUDA 组合失败的情况）。

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt    # 内含 CUDA 轮子源 + 项目本体 + dev 依赖
```

或手工分两步（与 `requirements.txt` 等价）：

```bash
pip install torch==2.14.1+cu130 torchvision==0.29.1+cu130 --index-url https://download.pytorch.org/whl/cu130
pip install -e ".[dev]"
```

### 2. 下载数据集

从 [Kaggle FER2013](https://www.kaggle.com/datasets/msambare/fer2013) 下载 `fer2013.csv`，
放入 `data/` 目录（官方划分：Training 28,709 / PublicTest 3,589 / PrivateTest 3,589；
已知重复问题见「数据说明」）。

### 3. 训练模型

**CLI 方式**（推荐；每次运行创建独立 run 目录）：

```bash
# 从头训练 MicroResNet（训练到第 60 轮）
python training/train.py --model micro_resnet --epochs 60

# 恢复训练（自动选择该模型最新的 last.pth，沿用原 run 目录）
python training/train.py --model mini_cnn --epochs 10 --resume auto

# 启用 AMP / 禁用 AMP（覆盖配置）
python training/train.py --model vgg_lite --epochs 50 --amp
python training/train.py --model vgg_lite --epochs 50 --no-amp

# 仅运行性能诊断（不改变正式产物）
python training/train.py --model micro_resnet --diagnose --steps 5
```

**Jupyter 方式**：`cd training/notebooks && jupyter notebook`，
按 `01_eda → 02/03/04 训练` 顺序运行；训练 Notebook 使用与 CLI 相同的 Trainer /
Checkpoint 选择器（列出 `training/runs/` 下的可续训断点）。

**训练控制与产物**：

- **开始**：运行 CLI 命令或训练 Notebook Cell。
- **暂停**：`Ctrl+C` / Jupyter 停止按钮——保存 `interrupted_*` 断点（`partial` 标记）。
- **继续**：`--resume auto`（最新完整 `last.pth`）或指定断点路径。
- **停止**：早停自动触发（原因记录在 run 元数据与输出）。
- **产物**：`training/runs/<模型>/<run_id>/`：`config_effective.yaml`、`run_meta.json`
  （CLI 参数 / seed / git / 环境 / 数据指纹 / 状态 / 恢复事件）、`history.json`、
  `checkpoints/{last,best,epoch_*,interrupted_*}.pth`。
- **精确续训的支持范围（S01 实测）**：`num_workers=0`，或 `workers≥1` 且
  `persistent_workers=false`（训练 sampler / DataLoader 使用独立 generator，
  其状态随断点保存/恢复，批次序列与 worker 种子可复算——回归覆盖普通/增强/
  加权采样组合）。`persistent_workers=true` 不提供精确恢复（启动时提示、加载时拒绝）。
  恢复端的数据管线规格（workers/persistent/sampler/batch）必须与断点一致；
  训练协议（完整生效配置 + 数据指纹）任一变化都会被拒绝。

### 4. 评估

```bash
# 统一评估入口（显式 checkpoint + split；输出指标、逐样本预测与权重/划分哈希）
python tools/evaluate_checkpoint.py --checkpoint <权重路径> --split PrivateTest

# 三模型对比分析（Notebook 内使用同一评估入口）
jupyter notebook analysis/comparison_report.ipynb
```

### 5. 导出与推理应用

```bash
# 显式导出（写 export_manifest.json；默认不覆盖同名文件）
python tools/export_model.py --list
python tools/export_model.py --checkpoint training/runs/<模型>/<run_id>/checkpoints/best.pth

# 启动推理应用
streamlit run inference/app.py    # 或 python run_app.py
```

推理应用的范围（明确限定）：**已裁剪的单张人脸**表情类别预测（接近 48×48 灰度训练域）；
展示未校准的模型概率与 Grad-CAM 辅助热力图；不包含人脸检测/对齐，不作为真实心理状态测量。

---

## 模型对比

当前三个权重的统一评估结果（CPU float32，官方划分；权重为历史 legacy 产物，
来源与 SHA-256 见 `inference/saved_models/export_manifest.json`）：

| 模型 | 参数量（实测） | PublicTest acc | PrivateTest acc | PrivateTest Macro-F1 | PrivateTest balanced acc |
|------|:-----:|:-----:|:-----:|:-----:|:-----:|
| MiniCNN | 1,274,823 | 61.13% | 62.41% | 57.58% | 56.67% |
| VGGLite | 5,407,687 | 65.03% | 65.51% | 63.65% | 63.96% |
| MicroResNet | 753,991 | 67.29% | 66.82% | 63.73% | 63.97% |

> 历史参考（有日期，仅作背景）：Khaireddin & Chen (2021) 报告 VGGNet 在 FER2013 上
> 73.28%；其训练/评估协议与本表不同，**不能与上表直接相减或排名**。
> 上表结果全部可由 `analysis/evaluations/` 下的保存预测复算。

---

## 数据说明（重要）

FER2013 官方划分存在**跨划分完全重复**：PublicTest 中 280 条、PrivateTest 中 288 条
与 Training 像素完全相同（另有 1516 个重复组、57 个标签冲突组）。

- 完整披露、方法与敏感性分析（排除 PrivateTest∩Training 288 条后
  acc 61.04%/63.95%/65.07%、Macro-F1 53.95%/60.30%/60.42%）见
  [`docs/data_audit.md`](docs/data_audit.md)；可复跑：`python tools/data_audit.py`。
- 引用本项目分数时请一并披露上述重叠，且**不得声称"数据泄漏已消除"**——
  本仓库未实施附加去重协议（草案见 docs/data_audit.md §3，未冻结、未使用）。
- 本审计只覆盖完全相同像素；不主张人员互斥或近重复已排除。

---

## 校准说明（概率未经校准）

应用与评估输出的「概率」为模型 softmax 输出，**未经校准**，不等同于「预测正确的概率」。
15 等宽区间的 ECE / NLL（PrivateTest，legacy 模型，可由保存预测复算）：

| 模型 | ECE (15 bins) | NLL |
|------|:-----:|:-----:|
| MiniCNN | 0.1123 | 1.0381 |
| VGGLite | 0.2144 | 1.0538 |
| MicroResNet | 0.0645 | 0.9025 |

> 这些数字只描述本次测量（指定权重、划分与区间）；应用内不做校准，界面文案明确标注「未校准」。

---

## 效率与 AMP 实测

效率基准（`analysis/efficiency_results.json`，batch=1 float32，预热 10 + 重复 100，
GPU 计时同步；输入为已裁剪 48×48 灰度）：

| 模型 | MACs (batch=1) | GPU 前向 median/P95 | CPU 前向 median | 训练峰值显存（按配置 batch） |
|------|:-----:|:-----:|:-----:|:-----:|
| MiniCNN | 23,078,656 | 0.43 / 0.52 ms | 0.52 ms | 458 MB（batch 256） |
| VGGLite | 260,982,528 | 0.76 / 0.89 ms | 2.00 ms | 732 MB（batch 128） |
| MicroResNet | 47,334,272 | 1.48 / 1.76 ms | 1.47 ms | 490 MB（batch 128） |

AMP 配对基准（替代旧「1.4–1.8 倍」的外推数字；协议：同一初始权重、同一数据顺序、
交替顺序、3 组配对、1 epoch + eval 完整流程）：

| 模型 | 完整流程 OFF → ON | 流程加速比 | 稳态 step OFF → ON | step 比值 | 峰值显存 OFF → ON |
|------|:-----:|:-----:|:-----:|:-----:|:-----:|
| MiniCNN | 20.7s → 20.7s | 1.00x | 12.3 → 8.6 ms | 1.41x | 638 → 344 MB |
| VGGLite | 25.0s → 22.6s | 1.10x | 27.4 → 16.6 ms | 1.65x | 985 → 577 MB |
| MicroResNet | 21.0s → 20.9s | 1.00x | 10.6 → 10.4 ms | 1.00x | 666 → 312 MB |

> 结论（如实报告）：在本基准口径（`num_workers=0`，数据加载在主进程内，因此完整流程
> 被数据管线主导）下，AMP 的收益主要体现在**稳态 step 时间（1.4–1.7x）与显存
> （约 -30%～-45%）**；端到端完整流程（1 epoch + eval）加速为 1.00–1.10x。
> **不做单步时间到完整训练的线性外推**（旧文档中「1.4–1.8 倍加速」的说法在此口径下不成立）。
> 全部 9 组配对的输入批次顺序经哈希验证一致；基准前后正式权重与日志 SHA-256 未变。
> 完整协议与逐配对原始数据：`analysis/benchmark_amp/results.json`。

---

## 配置

所有超参数集中管理在 `configs/training_config.yaml`（启动前经 `utils/config_validation.py`
集中校验，未知键与非法值会明确报错）：

| 配置项 | 说明 |
|--------|------|
| 学习率 / 优化器 / 调度器 | adam / adamw / sgd + cosine / cosine_warm / step / plateau |
| 激活函数 | relu / leaky_relu / elu / gelu（各模型可单独覆盖，显式声明） |
| 数据增强 | RandomFlip / Rotation / Affine / ColorJitter / RandomErasing（总开关） |
| 类别特定增强 | 对 Disgust 类 80% 概率额外旋转+平移+擦除 |
| MixUp | Beta 分布批内混合，默认关闭 |
| 类别平衡 | WeightedRandomSampler（与 CB Focal 共用同一份划分统计） |
| 损失函数 | Focal / CB Focal / **CrossEntropy（CE 基线，`configs/baseline_config.yaml`）** |
| 混合精度 | AMP ON/OFF（GPU 自动启用，由配置与 CLI 覆盖） |
| 早停 | val_acc 无改善 + val_loss 恶化双监控（=0 禁用） |
| Checkpoint | save_best / monitor_metric(val_acc/val_loss) / 定期断点清理 |
| 确定性 | cudnn_deterministic（由配置决定，代码不再强制覆盖） |
| 增强实现 | `augmentation.impl`：legacy（逐样本）/ **batched（批级张量，默认）**；同分布，随机序列不同 |
| fused Adam | `training.optimizer_fused`（仅 CUDA + Adam；默认关，为可选项） |

---

## 技术栈

- **深度学习**: PyTorch（AMP / Focal Loss / CosineAnnealingWarmRestarts）
- **UI**: Streamlit（推理）+ ipywidgets（训练控制）
- **数据分析**: Jupyter Notebook + Pandas + Matplotlib + Seaborn
- **评估**: scikit-learn（混淆矩阵 / ROC / 分类报告）+ 自实现 ECE/NLL
- **可解释性**: Grad-CAM 热力图（纯手写，无第三方依赖）
- **质量保证**: pytest（197 项通过）+ ruff（通过）+ mypy（全项目 42 源文件 0 错误）

## 项目状态与限制

- 本轮（2026-10）为**修复、整理与性能优化轮**：统一评估 + 工程修复 + 续训可靠性
  返修（S01–S05）与性能专项（PB01–PB05）均已实施并验收，实测数字与边界见
  [PROJECT_REVIEW.md](PROJECT_REVIEW.md)（如批级增强为同分布实现、非逐位等价）；
  **未进行正式重训**，上表数字来自历史权重（legacy，信息不全，已有明确标注与迁移说明）。
- 性能摘要：工厂加载 8.7→0.24s、批级增强 88.4%/39.0%（w0/w4）、完整评估 -96.0%、
  Grad-CAM 命中 -92.6%~-95.5%；fused Adam 未达稳定 ≥5% 目标，保留为可选项（默认 false）。
- `training/checkpoints/`、`training/logs/` 为 legacy 历史产物（只读保留），
  新训练一律写入 `training/runs/`。
- 2026-10-07 完成三次**流程验证短训练**（mini_cnn：`20261007_022921_seed42`、
  `20261007_031251_seed42`、`20261007_041818_seed42`；第三个在统一预算下验证
  **4 workers（non-persistent）精确恢复**全链路：训练 → `--resume auto`（未降级、
  协议校验通过）→ 导出 → PublicTest 评估）。三者仅验证保存/恢复/导出/评估闭环可用，
  其评估分数（34.63% / 36.89% / 44.13%）**不代表正式结果**，已在 export_manifest 中标注来源。
- 本项目由作者个人独立完成，无组员分工，不另列个人贡献清单；课程报告 DOCX / PPT 不再修改。
- 当前独立审核与性能分析见 [PROJECT_REVIEW.md](PROJECT_REVIEW.md)，未完成项按该文档验收。

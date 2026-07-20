# FER2013 人脸情感识别系统

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![CI](https://github.com/BernardW18/emotion-recognition/actions/workflows/ci.yml/badge.svg)](https://github.com/BernardW18/emotion-recognition/actions/workflows/ci.yml)

基于 PyTorch 的人脸表情识别系统，包含三个 CNN 模型的训练、评估与对比，以及 Streamlit 推理演示应用。

**核心栈**: PyTorch + Streamlit + Jupyter + scikit-learn

---

## 项目结构

```
emotion_recognition/
├── data/                        # 数据集（需手动下载）
│   ├── fer2013.csv              # FER2013 CSV（不纳入版本控制）
│   └── README.md
├── models/                      # CNN 模型定义（可配置激活函数）
│   ├── mini_cnn.py              # 基线模型 (~50K 参数)
│   ├── vgg_lite.py              # VGG 变体 (~1.5M 参数)
│   └── micro_resnet.py          # 微型残差网络 (~400K 参数，可选 SE 模块)
├── training/                    # 训练子系统
│   ├── train.py                 # CLI 训练脚本（推荐使用）
│   ├── trainer.py               # 共享训练引擎（Trainer 类）
│   ├── checkpoint.py            # Checkpoint 管理 + ipywidgets UI
│   ├── notebooks/
│   │   ├── 01_eda.ipynb         # 数据探索与分析
│   │   ├── 02_train_baseline.ipynb  # MiniCNN 训练
│   │   ├── 03_train_vgg.ipynb       # VGGLite 训练
│   │   └── 04_train_resnet.ipynb    # MicroResNet 训练
│   ├── checkpoints/             # 模型断点（不纳入版本控制）
│   └── logs/                    # 训练历史 JSON
├── inference/                   # 推理演示应用
│   ├── app.py                   # Streamlit 应用
│   ├── infer_utils.py            # 推理工具（预处理、Grad-CAM）
│   └── saved_models/            # 推理用最优模型（不纳入版本控制）
├── utils/                       # 公共工具模块
│   ├── activations.py           # 激活函数工厂
│   ├── losses.py                # Focal Loss + CB Focal Loss
│   └── constants.py             # 类别名称/Emoji/样本数常量
├── configs/
│   └── training_config.yaml     # 集中式训练配置（YAML）
├── analysis/                    # 数据分析与评估结果
│   ├── class_distribution.png   # 类别分布图
│   ├── sample_visualization.png # 样本可视化
│   ├── comparison_report.ipynb  # 三模型对比分析
│   ├── confusion_matrices/      # 混淆矩阵 PNG
│   ├── roc_curves/              # ROC 曲线 PNG
│   ├── training_curves/         # 训练曲线 PNG
│   └── amp_comparison_results.json  # AMP 实验数据
├── tests/
│   └── test_core.py             # 34 个单元测试
├── .github/
│   └── workflows/
│       └── ci.yml               # GitHub Actions CI 配置
├── .gitattributes               # Git 属性配置
├── LICENSE                      # MIT 许可证
├── pyproject.toml               # 项目配置 + ruff + mypy
├── requirements.txt
└── README.md
```

---

## 快速开始

### 1. 安装依赖

```bash
pip install -e .
```

> 所有依赖定义在 `pyproject.toml` 中，`requirements.txt` 仅为引导文件。

### 2. 下载数据集

从 [Kaggle FER2013](https://www.kaggle.com/datasets/msambare/fer2013) 下载 `fer2013.csv`，放入 `data/` 目录。

### 3. 训练模型

**推荐方式 — CLI 脚本**（支持参数覆盖、断点恢复、性能诊断）：

```bash
# 从头训练 MicroResNet
python training/train.py --model micro_resnet --epochs 60

# 恢复训练 MiniCNN
python training/train.py --model mini_cnn --epochs 30 --resume auto

# 启用 AMP 训练 VGGLite
python training/train.py --model vgg_lite --epochs 50 --amp

# 仅运行性能诊断
python training/train.py --model micro_resnet --diagnose --steps 5

# 覆盖学习率和 batch size
python training/train.py --model micro_resnet --epochs 30 --lr 0.0005 --batch-size 256
```

**传统方式 — Jupyter Notebook**：

```bash
cd training/notebooks
jupyter notebook

# 运行顺序：
# 1. 01_eda.ipynb         -- 数据探索与分析
# 2. 02/03/04 训练 Notebook -- 模型训练
```

**训练控制**：
- **开始**: 运行 CLI 命令或训练 Notebook Cell
- **暂停**: 按 `Ctrl+C` 或 Jupyter 停止按钮(■)，断点自动保存
- **继续**: 通过 `--resume auto` 或 Checkpoint 选择 Widget
- **停止**: Early Stopping 自动触发（val_acc + val_loss 双监控）

每次训练的最佳模型自动导出到 `inference/saved_models/`。

### 4. 模型评估与对比

```bash
# 三模型对比分析
jupyter notebook analysis/comparison_report.ipynb

# 单元测试
pytest tests/ -v
```

### 5. 启动推理应用

```bash
streamlit run inference/app.py
```

推理应用支持：
- 模型选择（MiniCNN / VGGLite / MicroResNet）
- 各类别概率分布条形图
- Grad-CAM 热力图可视化（展示模型关注区域）

---

## 模型对比

| 模型 | 参数量 | 最佳 val_acc | 架构特点 | 适用场景 |
|------|:-----:|:-----------:|----------|----------|
| MiniCNN | ~50K | **61.24%** | 3层卷积 + FC | 快速基线，资源受限环境 |
| VGGLite | ~1.5M | **65.00%** | 5层小核堆叠 + 3层FC | 参数量大但精度有限 |
| MicroResNet | ~400K | **67.32%** | 4残差块 + GAP + SE可选 | 参数效率最优 |

> FER2013 单网络 SOTA: **73.28%** (VGGNet, Khaireddin & Chen 2021)

## 配置

所有超参数集中管理在 `configs/training_config.yaml`，包括：

| 配置项 | 说明 |
|--------|------|
| 学习率 / 优化器 / 调度器 | adam / adamw / sgd + cosine / cosine_warm / step |
| 激活函数 | relu / leaky_relu / elu / gelu（各模型可单独覆盖） |
| 数据增强 | RandomFlip / Rotation / Affine / ColorJitter / RandomErasing |
| 类别特定增强 | 对 Disgust 类 80% 概率额外旋转+平移+擦除 |
| MixUp 混合增强 | Beta(0.2, 0.2) 批内混合，默认关闭 |
| 类别平衡 | WeightedRandomSampler（Disgust 16.5x 采样权重） |
| 损失函数 | Focal Loss / CB Focal Loss（归一化后可切换） |
| 混合精度 | AMP ON/OFF（RTX 5070 Ti 上 1.4-1.8x 加速） |
| Early Stopping | val_acc + val_loss 双指标监控，可配置 patience |
| Checkpoint | 定期/最优/断点三机制自动保存 |

## 技术栈

- **深度学习**: PyTorch (AMP / Focal Loss / CosineAnnealingWarmRestarts)
- **UI**: Streamlit（推理）+ ipywidgets（训练控制）
- **数据分析**: Jupyter Notebook + Pandas + Matplotlib + Seaborn
- **评估**: scikit-learn（混淆矩阵 / ROC / 分类报告）
- **可解释性**: Grad-CAM 热力图（纯手写，无第三方依赖）
- **质量保证**: pytest (34 tests) + ruff + mypy
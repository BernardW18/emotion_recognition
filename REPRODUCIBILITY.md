# 环境、复现与来源验证

## Windows项目虚拟环境

开发与实验统一使用项目内`.venv\Scripts\python.exe`。
已验证的实验环境为Python3.10.11、PyTorch2.14.1+cu130、torchvision0.29.1+cu130，
GPU为RTX5070Ti Laptop。CUDA训练、TF32/AMP及硬件差异意味着不保证跨设备逐位重现。

新机器先创建Windows虚拟环境，已有环境直接使用：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip check
```

[requirements.txt](requirements.txt)安装CUDA组合与项目开发依赖；
[requirements-lock.txt](requirements-lock.txt)记录2026-10-07核对的完整环境快照。
需要按快照安装时，用`-r requirements-lock.txt`替代上述requirements安装，并补充
`.\.venv\Scripts\python.exe -m pip install -e ".[dev]"`。
快照不是历史legacy权重训练环境的证明，也不保证其他硬件完全复现计时。

## 数据与权重

按[数据说明](data/README.md)准备`data/fer2013.csv`，核对SHA、七类映射与官方Usage。
仓库不包含CSV、像素缓存、checkpoint或逐样本预测；图表和聚合结论可以直接查看。

[导出清单](inference/saved_models/export_manifest.json)记录默认S1的唯一来源与SHA。
在有训练归档的机器上按[README](README.md#5-导出与推理应用)导出并启动演示；
仅克隆仓库不会获得该文件。复核已有checkpoint时，以SHA和model_spec为准，不按文件修改时间挑选。

## 当前配置与执行入口

| 内容 | 入口 |
|---|---|
| 当前S1配方 | configs/architecture_s1_config.yaml |
| S0–S3设计和预设判断 | [结构方案](configs/plans/architecture_ce_v1.md) |
| 单模型训练与续训 | training/train.py |
| 实际协议冻结 | tools/freeze_comparison.py |
| 结构实验执行/统计 | tools/run_architecture_comparison.py、tools/summarize_architecture.py |
| 损失/采样统计 | tools/summarize_ablation.py |
| 指定checkpoint评估 | tools/evaluate_checkpoint.py |
| 导出与默认权重 | tools/export_model.py |

`configs/training_config.yaml`是历史通用配置，默认CLI并不自动采用S1。
S1训练必须显式传`--config configs/architecture_s1_config.yaml`；smoke只验证流程，不进入正式研究汇总。

## 冻结记录与历史资格

| 协议 | 用途 | 对外结果 |
|---|---|---|
| [comparison-fixed-abcd-v3](configs/protocols/comparison-fixed-abcd-v3.json) | 三模型×四臂×三seed，完整90轮 | [损失/采样结论](analysis/fixed_abcd_v3/RESULTS.md) |
| [comparison-architecture-ce-v1](configs/protocols/comparison-architecture-ce-v1.json) | MicroResNet四臂×三seed，完整90轮 | [结构结论](analysis/architecture_ce_v1/RESULTS.md) |
| [comparison-ce-v1](configs/protocols/comparison-ce-v1.json) | 历史三模型CE，允许早停 | [历史CE结论](analysis/ce_stage1/RESULTS.md) |
| [comparison-fixed-abcd-v2](configs/protocols/comparison-fixed-abcd-v2.json) | 历史冻结版本 | 不用于当前正式排名 |

冻结文件绑定实际配置、模型、随机种子、预算、运行时、数据SHA和原提交。
已完成run按其原提交的Git源码验证；源码修改后不会抹去已完成结果，但新训练/续训会检查当前代码指纹。
本仓库完成S1采用后代码已变化，不能直接拿旧协议在当前源码上启动新的正式训练。
历史协议字节与Git历史都需保留，不能改写原协议来让新代码通过。

历史CE run保留了旧的本地协议路径。验证它们时显式传
`frozen_protocol_path=Path("configs/protocols/comparison-ce-v1.json")`，按相同SHA的归档副本核对。
`utils.comparison_check.check_formal_eligibility`确认用途、实际last/best、完成状态、数据及来源；
仅run-bound或一次训练会话结束不等于正式实验完成。

## 重复结构实验

下面命令用于当前代码重新执行同一S0–S3设计；它们创建新的协议和输出，
不会恢复/替代已发布12个run。先提交实现和配置，保持Git工作区干净。

```powershell
$env:OMP_NUM_THREADS="4"
$env:MKL_NUM_THREADS="4"
$env:OPENBLAS_NUM_THREADS="4"
.\.venv\Scripts\python.exe tools/freeze_comparison.py --protocol-id comparison-architecture-reproduction-v1 --output configs/protocols/comparison-architecture-reproduction-v1.json --seeds 42 43 44 --plan micro_resnet S0 configs/architecture_s0_config.yaml --plan micro_resnet S1 configs/architecture_s1_config.yaml --plan micro_resnet S2 configs/architecture_s2_config.yaml --plan micro_resnet S3 configs/architecture_s3_config.yaml
.\.venv\Scripts\python.exe tools/run_architecture_comparison.py --protocol configs/protocols/comparison-architecture-reproduction-v1.json --output analysis/architecture_reproduction_v1
```

已有同名新协议时另选唯一名称；不要覆盖。中断后，在源码/环境/数据仍与该协议一致时，
给同一执行命令增加`--resume`，继续记录中的run。完成run不追加epoch、不重选seed。
若要精确复现原实现，应使用冻结JSON记录的原提交，并按原环境准备数据；不能把当前HEAD当成原实现。
新输出保持本地忽略；检查完成后，显式选择结论与图表并更新发布白名单。

## 评估与指标

在具有对应checkpoint的机器上，可以只复核PublicTest，另用新目录保留原归档：

```powershell
.\.venv\Scripts\python.exe tools/evaluate_checkpoint.py --checkpoint inference/saved_models/micro_resnet_s1_20261008_182922_seed43.pth --split PublicTest --device cpu --batch-size 64 --output-dir analysis/local_s1_public_check
```

CPU float32、48×48灰度、x/255、无增强、固定划分。预测CSV保存标签、预测与七类概率，
`utils.evaluation.recompute_metrics_from_predictions`可从归档重算accuracy、macro-F1等指标。
有完整归档时优先复算，避免把新的重复Private评估当作新的独立证据。
Public用于模型/候选判断；Private用于已经固定方案的最终报告，不能据其结果再改选。

结构效率基准使用同轮环境、CPU线程数4、batch1、预热20次、正式100次、三轮轮换顺序；
GPU计时同步。预处理与Grad-CAM不混入模型前向，部署成本取三轮P95中位数。
GPU float32计时记录TF32开关；正式质量指标使用CPU float32。

## 实现验证与Git边界

```powershell
.\.venv\Scripts\python.exe -m pytest tests/ -q
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m mypy models data training inference utils tools
```

部分真实数据/权重集成测试依赖本地归档，缺失时可能跳过；结果报告必须区分通过与跳过。
源码、配置/协议、测试、小型聚合摘要、结论与明确发布的图表纳入Git。
内部审核、旧图/备用训练Notebook、原始预测/计时、run/checkpoint/日志和缓存保持本地。
冻结协议按原字节保存，清理文件树不重写历史。

# FER2013 轻量人脸表情识别

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

本项目围绕FER2013七类表情识别，比较损失/采样策略和轻量网络结构对识别质量与计算成本的影响，
并实现可追溯的训练、评估和Streamlit推理演示。当前采用 **MicroResNet S1：延后第一组平均池化**。

## 1. 问题与方法

FER2013输入为48×48灰度人脸，存在类别不平衡和跨划分重复。总体准确率可能掩盖少样本类别效果；
增加网络深度也会增加成本，因此同时考察accuracy、macro-F1、各类召回、参数量与推理延迟。

实现MiniCNN、VGGLite与MicroResNet三个模型，共用训练引擎、模型规格与评估流程。
分两轮完成正式对照：

| 实验 | 受控变量 | 预算与目的 | 结论入口 |
|---|---|---|---|
| 损失/采样A–D | CE、Focal、加权采样、CB-Focal组合 | 三模型×四臂×三seed，36次完整90轮；检验少样本收益与总体质量取舍 | [正式结果](analysis/fixed_abcd_v3/RESULTS.md) |
| 结构S0–S3 | 池化位置、残差块数量、SE | MicroResNet四臂×三seed，12次完整90轮；比较质量与成本 | [正式结果](analysis/architecture_ce_v1/RESULTS.md)、[设计与门槛](configs/plans/architecture_ce_v1.md) |

每轮使用seeds42/43/44，best由训练中的PublicTest val_acc选择。
最终指标统一使用CPU float32、batch64、无增强；报告三seed均值±样本标准差。
不同协议的历史基线分别报告，不混合排名。完整导航见[实验索引](analysis/README.md)。

## 2. 损失与采样实验

Focal和加权采样提高了部分模型的Disgust召回，但伴随总体准确率取舍。
MicroResNet的B/C/D均未通过该轮PublicTest的“Disgust召回提高且macro-F1不下降”双条件门槛，
因此保留CE和普通随机采样进入结构对照。

这不表示某种损失普遍无效；结论限于本项目固定配方、三seed和官方划分。
[全部七类结果与取舍](analysis/fixed_abcd_v3/RESULTS.md)、[聚合指标](analysis/fixed_abcd_v3/aggregate_metrics.json)。

## 3. 结构结果与S1选择

S0在同轮重新训练；S1只将第一组平均池化移至stage1之后，让该阶段在24×24特征上工作，
保持通道、残差块、分类器与参数量不变。S2增加深度和容量，S3沿用已有SE模块。

下表为PublicTest；准确率和宏F1以百分数表示，±为样本标准差（百分点）。
CPU P95是同轮三轮测量P95的中位数，单位ms，仅计模型前向。

| 臂 | 改动 | 参数量 | Public accuracy | Public macro-F1 | CPU P95(ms) |
|---|---|---:|---:|---:|---:|
| S0 | 原池化位置 | 753,991 | 67.07 ± 0.21 | 63.59 ± 0.44 | 1.438 |
| S1 | 延后第一组平均池化 | 753,991 | 68.71 ± 0.20 | 66.25 ± 0.19 | 1.835 |
| S2 | 增加残差块 | 1,493,575 | 67.93 ± 0.27 | 65.12 ± 1.03 | 2.561 |
| S3 | 加入SE | 764,663 | 67.04 ± 0.74 | 63.43 ± 1.18 | 1.802 |

S1相对同轮S0，accuracy提高 **1.64个百分点**、macro-F1提高 **2.66个百分点**，
三个配对seed的macro-F1均改善；参数量相同，CPU P95约为S0的 **1.28倍**。
S2有质量收益，但CPU P95约1.78倍，未过预设部署门槛；S3未过质量门槛。
本轮结果支持保留延后池化，不能据此断言“网络越深越差”或已经证明了具体特征机制。

候选只按PublicTest与效率门槛选择，并在本轮Private评估前固定。
S1的PrivateTest accuracy为 **69.17 ± 0.39%**，macro-F1为 **66.96 ± 0.62%**。
Private未用于改变本轮候选；该测试集此前已经使用，不作为首次未见或外部独立测试集。

![结构质量与成本比较](analysis/architecture_ce_v1/architecture_comparison.png)

更多证据：[训练曲线](analysis/architecture_ce_v1/training_curves.png)、
[Public混淆矩阵](analysis/architecture_ce_v1/public_confusion_matrices.png)、
[完整结论](analysis/architecture_ce_v1/RESULTS.md)、[聚合指标](analysis/architecture_ce_v1/aggregate_metrics.json)。

## 4. 数据与结论边界

- 官方划分：Training28,709、PublicTest3,589、PrivateTest3,589；输入48×48灰度，归一化为x/255。
- PublicTest有280条、PrivateTest有288条记录与Training像素完全相同；重复尚未消除。
- 三seed是描述性比较，不宣称统计显著；没有建立人员互斥或跨人员泛化结论。
- S2同时增加深度与容量，不能将其效果单独归因于深度。
- 延迟来自指定硬件、软件与线程配置，不能跨设备直接使用毫秒排名。
- 推理仅支持已裁剪单张人脸；softmax未经校准，Grad-CAM为辅助可视化，不测量真实心理状态。

数据格式、类别统计、重复检查方法与来源校验见[数据说明](data/README.md)。

## 5. 导出与推理应用

使用项目目录内的Windows虚拟环境：

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run inference/app.py
```

本地默认权重为 **S1、seed43、best epoch82**，文件名
`micro_resnet_s1_20261008_182922_seed43.pth`。按S1三个run的Public宏F1选择，
该单checkpoint的Public accuracy68.49%、macro-F1 66.46%，与上面的三seed均值分别标注。
默认只显示推荐模型，勾选“显示历史与验证权重”可查看其他本地导出。

权重文件不随Git提供；新克隆只有源码、结论与图表。已有该run归档时，可将它显式导出：

```powershell
.\.venv\Scripts\python.exe tools/export_model.py --checkpoint training/runs/micro_resnet/20261008_182922_seed43/checkpoints/best.pth --name micro_resnet_s1_20261008_182922_seed43 --default
```

导出默认拒绝覆盖同名文件；已有默认权重时直接启动应用。
[导出清单](inference/saved_models/export_manifest.json)记录来源、模型规格、SHA与验收摘要。
全部3,589张Public图像导出前后CPU概率逐位一致，28张覆盖七类的CPU/GPU单图类别一致，
Grad-CAM与缓存通过部署验收；这些检查不代替完整实验的三seed评估。

## 6. 复现与代码入口

[复现说明](REPRODUCIBILITY.md)说明环境、数据/权重准备、训练、评估与冻结协议。
已有完成实验依靠原提交和数据指纹追溯；源码变更后的新正式训练需重新冻结，不能覆盖已有协议。

| 目录/文件 | 用途 |
|---|---|
| [models/](models/) | 三个模型定义，S1仍保持四个残差块 |
| [training/](training/) | 共享Trainer、CLI、checkpoint与精确恢复 |
| [data/](data/) | 数据加载、批级增强、像素缓存与数据说明 |
| [inference/](inference/) | 单图预测、按需Grad-CAM、缓存及演示界面 |
| [utils/](utils/) | 版本化模型规格、指标与来源/配置校验 |
| [configs/](configs/) | 训练配方、结构设计与冻结协议 |
| [tools/](tools/) | 实验执行、统计、评估和导出 |
| [tests/](tests/) | 训练完整性、模型兼容与推理回归 |
| [analysis/](analysis/README.md) | 正式结论、聚合指标与发布图表 |

Git保留源码、配置、冻结清单、结论和图表；原始CSV、checkpoint、逐样本预测、日志、缓存与内部审核资料保留本地。

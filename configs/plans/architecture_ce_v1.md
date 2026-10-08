# MicroResNet 结构对照实验方案 v1

日期：2026-10-08。状态：comparison-architecture-ce-v1的12/12训练与36份评估均已完成。
实际执行状态保存在本地analysis/architecture_ce_v1/experiment_state.json；本设计文档不等于实验结果。
适用项目：emotion_recognition，Windows `.venv\Scripts\python.exe`。研究范围仍为 FER2013 七分类。

## 1. 研究问题与已有证据

问题：有限计算量下，当前表情识别更需要增加网络容量，还是在早期保留局部细节？
不预设加深、延后池化或 SE 必然有效。依据赵凯的公开研究介绍，将质量与计算效率一起评价：
[赵凯研究介绍](https://kaizhao.net/research)。本方案是现有项目的结构消融，不宣称为原创网络方法。

已完成的 `comparison-fixed-abcd-v3` 是损失/采样对照，不是深度消融；36/36 run 完整90轮。
MicroResNet 的 CE PublicTest accuracy=67.12±0.22%、macro-F1=63.94±0.26%（三seed、样本标准差）。
其三个 CE best 在本对话补充的无增强、CPU float32/batch64、eval 模式 Training 诊断中，
accuracy 均值78.01%、macro-F1均值77.27%。约10.9个百分点的准确率差距提示泛化问题，
不能单凭此排除容量不足或断言已定位过拟合。诊断没有重训或再次访问 PrivateTest。
当前4个残差块、753,991参数；单层3×3 stem 后连续两次池化，进入 stage1 前已是12×12。

## 2. 第一轮固定矩阵：4臂×3seed，共12个run

所有臂使用同一训练配方，唯一变化为本表结构因素。S0也在新源码/协议下重新从头训练3次，
不混用历史 v3 的 A run；历史数据完整保留，仅作为背景。

| 臂 | 结构定义 | 唯一改变 | 参数量（实测） | MACs（batch1实测） |
|---|---|---|---:|---:|
| S0 | blocks=[2,2]，channels=[64,128]，pool=before_stage1，SE=false | 新协议基线 | 753,991 | 47.33M |
| S1 | blocks=[2,2]，channels=[64,128]，pool=after_stage1，SE=false | 第一组平均池化的位置 | 753,991 | 111.04M |
| S2 | blocks=[4,4]，channels=[64,128]，pool=before_stage1，SE=false | 残差块数量 | 1,493,575 | 89.80M |
| S3 | blocks=[2,2]，channels=[64,128]，pool=before_stage1，SE=true | 现有 SE 模块 | 764,663 | 约47.34M |

2026-10-08真实构造核算：S0/S1/S2/S3分别为753,991/753,991/1,493,575/764,663参数，
MACs分别为47,334,272/111,035,264/89,801,600/47,344,512。使用torch flop_counter，
一个乘加记1 MAC、FLOPs=2×MACs；计数器未覆盖的池化/激活不算在MACs内，其代价包含在实测延迟中。
这些数字是结构成本，不是准确率结果。

S1严格只移动 `downsample1` 中的 AvgPool：1×1卷积仍在24×24执行；stage1在24×24执行后
再池化到12×12，随后 channel_expand2 和 downsample2 将特征送到6×6的 stage2。
保持最大池化、残差块、通道数、参数及最终分类器不变。S1计算量预计约为S0的2.35倍，
因此必须实测代价，不能称为无成本改进。

S2只增加重复残差块，保持池化位置、stage数、通道数和分类器不变。容量随深度增加，
该对照只能说明“加深并增加容量”的效果。若以后要归因于深度本身，再单独设计参数量
近似匹配的加宽对照；该后续实验不属于当前12-run计划。
S3沿用当前实现的SE位置与reduction=8，不同时修改注意力位置或其他增强。

seeds=42/43/44。执行顺序按seed分块轮换：
42：S0→S1→S2→S3；43：S1→S2→S3→S0；44：S2→S3→S0→S1。
同一模型初始化的消耗会随结构改变；同seed不代表不同结构具有相同初始权重或Dropout轨迹。
相同外部数据/增强规格和随机种子用于配对统计，不能承诺不同架构逐次RNG完全一致。

## 3. 共同训练配方

- FER2013 原官方划分：Training 28,709，PublicTest 3,589，PrivateTest 3,589；48×48灰度，x/255。
- CSV SHA-256：3b8d9617d1017f34733c8f2474d7784c563ce86c40a86ac12c2d37cc968f871b。
- 从 `configs/ce_fixed_config.yaml` 的 MicroResNet 生效配置派生独立臂配置；CE、普通随机采样。
- 90完整epoch；patience=0、val_loss_patience=0；禁止低学习率熔断提前结束。
- Adam，LR=3e-4，weight_decay=1e-4，gradient accumulation=1，max_grad_norm=1.0。
- CosineAnnealingWarmRestarts：T0=30、T_mult=2；调度节奏与原CE相同，禁止按臂临时调参。
- GELU；分类头Dropout=0.3，残差块Dropout2d=0.1；SE内部激活沿用现有实现。
- batch128；GPU AMP开启；torch_compile=false，fused Adam=false；workers=0，persistent_workers=false。
- 保留相同batched-v1普通增强，类别专属增强与MixUp关闭；增强的epoch/batch种子规则相同。
- CPU线程固定4，OMP/MKL/OPENBLAS线程固定4；同一Windows虚拟环境、GPU和依赖版本。
- cudnn benchmark=true、deterministic=false，明确不保证CUDA逐位重现。
- best始终按训练中的PublicTest val_acc选择，严格改善才替换，平分保留较早best；last用于续训。

这是固定训练配方下的结构比较，不代表各架构经过独立调参后的绝对性能上限。

## 4. 开始训练前的实现与冻结条件

原ModelSpec v1未记录 blocks/channels/pool_order。本轮已新增MicroResNet v2结构规格与配置校验；
旧v1按明确的原结构构造，旧结构配置继续生成v1，以保持历史绑定兼容。完整显式结构字段的
新配置生成v2。以下实现与冻结验收条件继续有效：

1. 让真实模型构造显式接收并验证 blocks、channels、pool_order、use_se；保存到新版本规格。
   已知v1原结构按明确兼容规则加载，不静默套用新变体默认值；原权重预测保持一致。
2. 新建S0–S3独立YAML；除结构字段外的实际训练协议完全一致。实际S0参数/前向与原结构相符。
3. CPU/CUDA确认1×1×48×48及128×1×48×48→7类输出；每臂真实AMP训练批次有效更新、状态有限。
   记录AMP跳过更新数；不能把optimizer尝试数当作有效更新数。
4. 续训加载严格验证结构、完整配置、scheduler/scaler/RNG与数据；错误结构与partial状态拒绝或
   按既有规则回退到完整last；原完成run不追加训练、不覆盖权重。
5. 用于结构对照的汇总器不能直接调用现有仅支持A–D损失/采样的summarize_ablation臂检查；
   新入口须验证S0–S3实际结构、每臂完整三seed、无重复/挑选run、完整90轮和正式资格。
6. 完成必要回归后提交源码和配置，工作区干净时运行freeze_comparison，生成新且唯一的
   `configs/protocols/comparison-architecture-ce-v1.json`；冻结4臂实际plan及各3个seed（共12项执行）、数据/代码指纹。
   本设计稿不是可执行冻结清单；清单未生成前不启动formal训练。

流程验收：12个run各90轮、结束原因budget、真实last/best和有效更新记录均通过正式准入。
失败保留并明确报告；只从同run完整last显式续训剩余轮数，不重新挑选seed或重复run。
Windows训练期间监控只读追加日志；不频繁打开正在原子替换的history/meta文件。

## 5. 统一评估与预先固定的判断

训练后每个best以CPU float32/batch64/eval/no augmentation评估完整Training和PublicTest。
保存全部七类指标、混淆矩阵及逐样本预测到本地原始目录；主要指标从保存预测复算。
Training诊断只解释拟合/泛化差距，不能代替PublicTest选模型。

对S1/S2/S3分别与同新协议S0做相同seed的差值，报告均值±样本std(ddof=1)和全部3个差值。
**质量通过**须同时满足：

- PublicTest平均macro-F1提升至少0.010（1.0个百分点）；
- PublicTest平均accuracy差值至少-0.005（下降不超过0.5个百分点）；
- 3/3配对seed的macro-F1差值均大于0。

balanced accuracy、完整七类recall/support（尤其Fear/Sad/Disgust）、混淆方向与训练曲线全部报告。
若只提高Training表现，就记录“未解决验证泛化”；质量未过则如实保留负结果，不放宽门槛。
三seed与三个干预为描述性比较，不据此宣称统计显著、纯深度因果或所有设置均更优。

**轻量部署通过**另外要求：参数量≤1.6M、batch128可完成训练且无OOM、
CPU float32/batch1的P95模型前向延迟≤同轮S0的1.5倍。
质量通过但效率未过，记录“质量收益存在，未达到轻量部署门槛”；不能抹去这个科学结果。
在质量通过的臂中，优先满足部署门槛者，再按PublicTest平均macro-F1、CPU P95、参数量依次排序。
没有臂质量通过就保留S0；仅有质量通过而部署未过的臂则研究结论保留、默认演示继续用S0。
候选排序与是否部署只用PublicTest/效率记录，在访问本轮PrivateTest前写入带时间与SHA的本地记录。

PrivateTest最终为12个best各评估一次，统一CPU float32/batch64，不据结果调参或改选候选。
该官方测试集此前已被使用，不宣称是首次未见或外部独立测试集。官方划分跨集像素重复
继续披露；不宣称跨人员泛化，不混入删重复样本后的成绩。

## 6. 效率测量口径

同设备、同版本、同CPU线程数；原历史效率数字仅作背景，S0–S3必须在同轮重新测量。
CPU float32/batch1与GPU float32/batch1模型前向：预热20次，正式100次，完整做3轮，
轮换测量顺序；GPU计时前后synchronize。报告每轮median/P95与跨轮统计；
部署门槛比较3轮P95的中位数。预处理单独计时；Grad-CAM、UI、人脸检测不混入模型前向。
参数量与MACs通过真实模型计数；训练峰值显存用相同batch128/AMP/Adam测量，并记录范围。
另外报告完整90轮累计训练耗时、AMP尝试/有效更新/跳过数；中断失败消耗也如实标注。
不从单batch时间线性外推完整训练加速，不跨硬件使用毫秒排名。

## 7. 成果与Git边界

Git保留：本设计稿、源码/配置/冻结清单、`analysis/<实验>/RESULTS.md`、
`aggregate_metrics.json`（小型聚合结果及来源SHA）、图表PNG/PDF/SVG、分析源Notebook。
本地保留但不跟踪：CSV、逐样本预测、每run完整JSON/历史、checkpoint、缓存、原始基准重复记录。
`docs/`全部继续忽略；课程报告/PPT不修改；历史文件不因取消跟踪而删除，也不重写Git历史。

预计发布图：七类召回对照、Public宏F1/accuracy与CPU P95的权衡、训练/验证曲线与泛化差距。
没有实际数据前不绘制性能结果图，也不把表中估算成本当成训练效果。

## 8. 代码入口与文档归属

四臂配置为 `configs/architecture_s0_config.yaml` 至 `architecture_s3_config.yaml`。
执行入口 `tools/run_architecture_comparison.py` 按固定顺序训练，显式 `--resume` 才能继续同run，
不挑选新seed；完成后调用 `tools/summarize_architecture.py` 与 `tools/benchmark_architecture.py`。
汇总器验证四臂真实结构、共同配方、完整90轮、正式准入、完整三seed与保存预测复算；
候选保存时间和SHA后才访问PrivateTest。原始数据在 `analysis/architecture_ce_v1/` 本地保留，
Git仅追踪RESULTS、聚合摘要和图表。工具运行不依赖docs目录。

`docs/`是用户要求整目录忽略的本地审核区；本方案需要随源码保留和追溯，
因此唯一版本化方案放在 `configs/plans/`，本地docs只记录审核和执行状态。

## 9. 本轮冻结与实现验收

实现提交：`8aaf41b75afd90aff0d4e50d10411e28a7ec931c`。
源码/配置指纹：`0d57433b7b71dd8b6e0579e6c45d31aed491ebf2d1721d6d9b5a0602d8021690`。
冻结清单：`configs/protocols/comparison-architecture-ce-v1.json`，4臂×3seed、每run90轮。
冻结文件SHA-256：`ad03786c6153a87e04da046ddee0240c07637b46c4937f1270026513e22b6245`。旧A–D冻结文件不变。

完整pytest336项通过；随后新增/强化结构逻辑的25项针对性回归通过，合计339个唯一用例。
Ruff、mypy40源文件、pip check通过；S0与8aaf41b之前原模型的初始化权重/CPU前向逐位一致，
旧v1加载一致；四组真实模型连续与恢复训练一致，错误池化协议拒绝且不改参数；
CPU/CUDA batch1与batch128形状正确，CUDA batch128 AMP有有效更新且参数有限。
实测参数/MACs已列在第二节。251个历史产物SHA一致，报告/PPT未修改。

## 10. 执行结果（2026-10-08 19:21，Asia/Shanghai）

12个run各完成90轮并通过正式准入；Training/Public/Private各12份评估，候选先于Private固定。
S1通过质量与部署门槛，Public accuracy相对S0+1.64pp、macro-F1+2.66pp，三个seed均改善，
参数仍753,991，CPU P95约1.28倍。S2质量通过但CPU P95约1.78倍，未过部署门槛；
S3未过质量门槛。三seed为描述性证据，保留负结果。
结论、七类统计和图表见[RESULTS](../../analysis/architecture_ce_v1/RESULTS.md)。
总执行约74.6分钟；历史归档在执行结束时SHA验证一致。演示权重未自动替换。

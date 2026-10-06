# 当前项目评估与修复验收清单

更新日期：2026-10-07。项目：`D:\Document\Unniversity\emotion_recognition`。用途：修复并整理现有 FER2013 项目，作为向赵凯老师展示个人能力的材料。证据见 [PROJECT_REVIEW_EVIDENCE.json](PROJECT_REVIEW_EVIDENCE.json)。

## 评估范围与结论

**当前用户确认的材料范围：** 本项目完全由用户个人独立完成，不设组员分工或个人贡献清单要求。课程报告 DOCX 和 PPT 不再修改，其同步/验收项从待办中取消；后续维护当前项目代码、README、Notebook 与审核记录。此确认优先于本文历史记录中的材料待办。

本轮只处理当前项目的正确性、可复现性、训练可靠性和材料表达。保留 MiniCNN、VGGLite、MicroResNet 三个现有模型以及 FER2013 数据；不提出新研究问题，不新增架构、分类头改造、其他数据集或论文迁移任务。对现有损失、采样、增强和 AMP 的检查，是为了确认项目已有功能及结论是否成立。

第三轮重构与性能优化独立审核已完成：197 项测试、Ruff、mypy 与依赖检查通过，Windows CUDA 可用；上一轮常规反例已修复，批级增强和快速预测明显提速。但另复现 3 项 P1 续训问题与 3 项 P2 缓存/材料问题（T01–T06）。**完整验收暂不通过；默认配置的恢复能力及断点完整性需在正式训练前解决。** 本轮实测、修复方案和验收标准见末尾“第三轮重构与性能优化独立审核”。

本文中的“验收标准”是各项修复需要满足的条件。**当前基准为 2026-10-07 提交 b75d8e3，最新结论以第三轮独立审核为准**；第二轮及性能初审保留为历史记录。F05/F09 尚有 T01–T03，缓存完整性见 T04，增强语义和正式实验准入见 T05/T06；F07/F11 的冻结协议与正式比较仍未完成。本次不改训练实现、不启动正式重训，只更新审核记录及相关文档。P0 表示正式重训前需要完成，P1 表示相关功能或结论使用前需要完成，P2 表示演示/对外材料前需要完成；性能项不能代替正确性验收。

## Windows CUDA 环境：已安装并验收

本节保留历史环境与三模型最小更新记录；第三轮已复核当前环境并运行 197 项测试，新增结果见末尾及证据文件。历史 36 项测试与四批次 GPU 更新不冒充本次正式训练。

本项目统一使用 Windows 虚拟环境：`D:\Document\Unniversity\emotion_recognition\.venv\Scripts\python.exe`。本轮已检查用户安装后的实际环境；后续 Python 操作继续使用这个解释器。Notebook 的 `Emotion Recognition (.venv)` 内核也已核对指向此路径。`D:\AI` 的 WSL 约定不扩展到本项目。

| 项目 | 本轮实测 |
|---|---|
| Python | 3.10.11，Windows 64 位 |
| PyTorch / torchvision | 2.14.1+cu130 / 0.29.1+cu130 |
| CUDA runtime / cuDNN | 13.0 / 记录于证据文件 |
| GPU / 驱动 | NVIDIA GeForce RTX 5070 Ti Laptop GPU / 596.49 |
| 显存 | 12,227 MiB |
| CUDA 可用性 | True；实际 GPU 运算通过 |
| 设备计算能力 / 已编译支持 | 12.0；安装包包含 sm_120 |
| 依赖一致性 | `pip check` 通过 |
| Notebook 内核 | 指向本项目 Windows `.venv` |

**当前状态：F20 已完成环境验收。** GPU 矩阵运算与 CPU 固定小例子一致，torchvision 的 CUDA NMS 固定小例子通过；三个现有模型均用当前 Trainer 完成 GPU 前向、反向和有效参数更新。训练 `--help` 正常，现有 36 项测试在新 CUDA 包环境下通过；这些既有测试的设备仍按原测试设定为 CPU，GPU 执行由单独的三模型检查覆盖。

三模型检查按已有 checkpoint 的 activation/dropout 构造，MicroResNet 使用保存的 GELU 配置。每个模式使用 16 个合成样本，batch=4，共 4 个批次，梯度累积=1、Adam、学习率 1e-4、Focal Loss。所有输出、loss、更新后的参数以及实际优化器更新前的反缩放梯度均为有限值；模型参数确实改变。FP32 路径沿用当前 Trainer 的 `matmul_precision=high` 设置，未声称严格最高精度模式下的结果。

| 模型 | FP32 有效更新 / 批次 | AMP 有效更新 / 批次 | AMP 跳过更新数 | AMP 最终 loss scale | 结论 |
|---|---:|---:|---:|---:|---|
| MiniCNN | 4/4 | 2/4 | 2 | 16,384 | 两模式均通过 |
| VGGLite | 4/4 | 3/4 | 1 | 32,768 | 两模式均通过 |
| MicroResNet（GELU） | 4/4 | 2/4 | 2 | 16,384 | 两模式均通过 |

AMP 使用默认 GradScaler 初始 scale=65,536；检查中发生少量缩放梯度溢出，scaler 跳过对应更新并降低 scale，之后执行了有限梯度的有效更新。这属于动态缩放处理，本次已记录跳过次数，不能写成“所有批次均完成更新”或“从未发生溢出”。完整训练时仍需监控持续跳步、loss 与参数的有限性，并按 F09 保存 scaler 状态。

所有检查输出隔离在系统临时目录；原 CSV、三份正式 history.json、三份导出权重的 SHA-256 在检查前后相同。没有运行正式 `fit()`，没有导出新权重。检查只验收 GPU 执行与最小更新流程；整套 FER2013 重训、最佳 batch、AMP 加速比和准确率仍按本文后续修复及实验标准验收。

[requirements-review-lock.txt](requirements-review-lock.txt) 已更新为实际 CUDA 环境快照，使用 cu130 包源。证据文件的 `runtime_environment` 保留此前完整数据 CPU 评估的环境；新的 `cuda_environment` 记录本轮 GPU 检查，旧下载交接仅作为历史记录保留。此前 CPU 结果表不会因安装 CUDA 自动变成 GPU 评估结果。

当前不需要重复安装。F04 的构建后端已修正；修复轮记录了干净 Windows venv 的安装验证，本次独立审核未重复下载或新建环境。若以后重建本项目 Windows 环境，可使用下面的固定版本命令，再按 README 安装项目与开发依赖：

```powershell
# 仅在重建环境时安装；本轮已安装并验证。
& .\.venv\Scripts\python.exe -m pip install 'torch==2.14.1+cu130' 'torchvision==0.29.1+cu130' --index-url https://download.pytorch.org/whl/cu130
& .\.venv\Scripts\python.exe -m pip check
# 随时可运行的简短 GPU 状态检查。
& .\.venv\Scripts\python.exe -c "import torch; print(torch.__version__,torch.version.cuda,torch.cuda.is_available()); print(torch.cuda.get_device_name(0)); x=torch.arange(16,dtype=torch.float32,device='cuda').reshape(4,4); print((x@x.T).device)"
```

若正在运行的 Notebook 仍显示旧版本，重启 `Emotion Recognition (.venv)` 内核。普通 PyTorch 训练使用轮子内的 CUDA 运行库，无需为本项目另装完整 CUDA Toolkit。版本选择参考 [PyTorch 发布配置](https://github.com/pytorch/pytorch/blob/main/RELEASE.md)和 [NVIDIA 驱动兼容规则](https://docs.nvidia.com/cuda/archive/13.2.0/cuda-toolkit-release-notes/index.html)。

## 现有结果与已确认事实

| 模型 | 原文档参数量 | 实际参数量 | 权重元数据的最高 PublicTest accuracy | 当前 history.json 的最高验证准确率 |
|---|---:|---:|---:|---:|
| MiniCNN | 约 50K | 1,274,823 | 61.24% | 57.84% |
| VGGLite | 约 1.5M | 5,407,687 | 65.00% | 63.67% |
| MicroResNet | 约 400K | 753,991 | 67.32% | 66.40% |

参数量对应当前七分类配置，MicroResNet 未启用 SE。最高验证分数在本地权重中有记录；当前日志更低不能直接证明分数编造。问题是日志、配置与权重混合了不同运行。

三个最优权重记录的 epoch 为 107、109、89。MicroResNet 保存配置为 `activation=gelu, learning_rate=0.001, batch_size=512`，当前 YAML 为 GELU、0.0003、128。checkpoint 未保存全部训练配置，所以不能直接用当前 YAML 宣称重现了这些历史权重。

以下是此前读取已有权重得到的 CPU float32 结果：按保存的模型配置构造网络，输入为原始 48×48 灰度像素除以 255，无测试增强，batch=64，4 个计算线程。没有从头重训。

| 模型 | PublicTest accuracy | PrivateTest accuracy | PrivateTest Macro-F1 | PrivateTest balanced accuracy | Disgust recall |
|---|---:|---:|---:|---:|---:|
| MiniCNN | 61.13% | 62.41% | 57.58% | 56.67% | 27.27% |
| VGGLite | 65.03% | 65.51% | 63.65% | 63.96% | 63.64% |
| MicroResNet（GELU） | 67.29% | 66.82% | 63.73% | 63.97% | 56.36% |

PublicTest 与历史权重记录的差异分别相当于 4、1、1 张图片；本次 CPU float32 与历史 GPU/AMP 环境不同，不能据此判定历史结果错误，也不能宣称已精确复现训练。MicroResNet 的 accuracy 更高，但 Macro-F1 与 VGGLite 接近，Disgust recall 更低；“全面更好”缺乏支持。

CSV 有 35,887 行，官方 Training/PublicTest/PrivateTest 分别为 28,709/3,589/3,589。对规范化像素计算 SHA-256，发现 1,516 个重复组、1,853 条超出唯一图像数的记录、57 个标签冲突组。PublicTest 有 280 条与 Training 像素完全相同；PrivateTest 有 288 条；PublicTest 与 PrivateTest 间有 43 个重复图像哈希、涉及后者 44 条记录。该检查不覆盖近重复或人员身份重叠。

只排除 PrivateTest 中与 Training 像素完全相同的 288 条记录，剩下 3,301 条，三个模型 accuracy 为 61.04%/63.95%/65.07%，Macro-F1 为 53.95%/60.30%/60.42%。这只是现有权重的敏感性分析，不是去重重训结果，不与官方成绩混用。

## 问题、修复方案与验收标准

### F01 · P0 · 推理模型与训练模型不一致

**问题及位置：** `inference/infer_utils.py` 的 `load_model()`、`analysis/comparison_report.ipynb` 的模型加载单元直接采用默认构造参数。MicroResNet 保存的是 GELU，但默认是 ReLU；激活函数没有状态参数，权重加载成功仍会改变预测。同一权重按 GELU 评估 PrivateTest 为 66.82%，按默认 ReLU 为 62.36%，836/3,589 个预测改变，下降 4.46 个百分点。

**修复方案：** 在 checkpoint 中保存版本化的 `model_spec`，包含现有模型名称、类别数/映射、activation、dropout、use_se、输入尺寸及预处理。建立统一构造与加载函数，训练导出、离线评估、Notebook 和应用都调用它。兼容旧权重时读取其已有配置；缺失的关键参数必须由明确的迁移配置提供，不能猜测或静默使用默认值。不改变三个模型结构。

**验收标准：**

- 在同一环境的 CPU FP32、`eval()` 下，保存前后同一固定输入的 logits 满足 `atol=1e-6, rtol=1e-5`，预测完全相同。
- 用当前 MicroResNet GELU 权重验证应用加载后确实为 GELU，且与统一评估入口逐样本预测一致。不能用“接近 66.82%”代替加载一致性检查。
- 对已有的非默认激活和 SE 配置做往返检查；缺失/非法规格或不兼容权重明确报错。应用显示可追溯的模型配置及权重哈希。

**状态：** 已修复（2026-10-07）。新增 utils/model_spec.py：版本化 model_spec + 统一构造/加载入口，训练导出、离线评估、Notebook、应用全部接入；旧权重按确定性迁移规则解析（activation 读旧 config、use_se 由 state_dict 判定、缺失关键字段必须显式提供）。证据：往返 logits 逐位一致（非默认激活+SE）；MicroResNet 按 GELU 正确加载 PrivateTest 66.82%（默认 ReLU 为 62.36%、836/3589 预测变化，与评估轮一致）；应用与离线评估逐样本一致（tests/test_inference.py）。

### F02 · P0 · 参数量与效率表述错误

**问题及位置：** `README.md`、三个模型文件的说明与 `inference/app.py` 曾存在错误参数量（课程报告/PPT 已排除修改范围）。MiniCNN 的 `Linear(4608,256)` 已超过百万参数；VGGLite 分类头也不是“数万个参数”。参数量与速度、显存之间的关系须由实测支持。

**修复方案：** 从当前模型实例自动计算总参数及可训练参数，生成统一结果表，同步 README、应用与模型说明。效率表保留现有模型，分别记录参数量、如需报告的 MACs/FLOPs、batch=1 延迟、训练峰值显存；注明输入、硬件、精度、预热和计数口径。删除“每千参数贡献准确率”作为优劣判定依据，不做分类头压缩。

**验收标准：**

- 七分类、当前无 SE 配置的参数计数准确等于 1,274,823 / 5,407,687 / 753,991；启用已有 SE 选项时重新计数并标明配置。
- README、应用、模型说明不再出现旧数量；每个效率数值能追溯到运行配置。
- 若报告 FLOPs，明确 MAC 与 FLOP 的换算；延迟至少记录预热、重复次数、median/P95，GPU 计时同步，并把预处理与模型前向耗时分开。未测的数据标为“未测”，不能据参数量推断速度。

**状态：** 已修复（2026-10-07）。参数量实测修正为 1,274,823 / 5,407,687 / 753,991（SE 版 764,663），同步到模型 docstring、README 与应用动态显示；新增 tools/benchmark_efficiency.py 实测 MACs（batch=1）、GPU/CPU 延迟（median/P95，预热 10 + 重复 100，同步计时，预处理与前向分开）、训练峰值显存，输出 analysis/efficiency_results.json；删除「每千参数贡献」表述。报告/PPT 不再修改，其同步不列为待办。

### F03 · P0 · 跨划分重复及标签冲突未披露

**问题及位置：** `data/fer2013.csv` 中存在上述重复及冲突。代码没有主动把 PrivateTest 加入 Training 的证据，但当前分数不能完全代表独立样本泛化。Training 类别数量也不能直接沿用于清洗后的划分。

**修复方案：** 保留原 CSV 和官方划分，不覆盖原数据。在官方结果旁披露重复情况。若报告去重结果，先冻结一份明确的附加协议：以像素哈希分组，记录原行号及保留/排除原因；预先规定跨划分重复的归属或排除方式，确保训练、验证、测试互不重叠。冲突组可统一隔离，不能参考测试标签选择有利的保留方案。规则冻结后才重训和评估；两套协议分别命名和报告。仅披露官方协议也可以完成官方结果的材料整理，附加去重结果必须通过下列专门验收才可使用。

**验收标准：**

- 原 CSV 的 SHA-256 保持 `3b8d9617d1017f34733c8f2474d7784c563ce86c40a86ac12c2d37cc968f871b`，官方三种 Usage 的行数不变；官方结果明确披露 280/288 条重叠记录。
- 附加去重协议如实施，三对划分的像素哈希交集均为 0；57 个冲突组的处理规则明确，输出各划分及各类计数、原行号清单和校验值。重复执行得到相同清单。
- 不用 PrivateTest accuracy 来选择清洗规则；不同协议的指标不混在同一排名中，不宣称人员互斥或近重复已全部排除。

**状态：** 已修复（官方披露路线，2026-10-07）。复算与评估轮一致：1516 重复组 / 1853 条超出 / 57 冲突组 / 280（PublicTest）/ 288（PrivateTest）/ 43+44（Public↔Private）；敏感性分析由保存预测复算（61.04%/63.95%/65.07%，Macro-F1 53.95%/60.30%/60.42%）。tools/data_audit.py 可复跑（输出 analysis/data_audit.json）；docs/data_audit.md 含披露与 dedup-v1 协议草案（未冻结、未实施）。去重重训未执行（按用户决策，重训整体暂缓）。

### F04 · P0 · 推荐安装方式失败，版本范围不一致

**问题及位置：** `pyproject.toml` 的 `setuptools.backends._legacy:_Backend` 无法导入，可编辑安装失败。`requirements.txt` 全为注释，直接安装不会提供依赖。Python >=3.9 声明与代码类型语法、工具版本和本次实测环境不一致，`torch.amp` 的最低支持版本也未核实。

**修复方案：** 按 [setuptools 官方文档](https://setuptools.pypa.io/en/latest/build_meta.html)改用 `setuptools.build_meta`，确认包发现范围。以本次已验证的 Python 3.10 环境为最低支持基线，同步项目、Ruff 和 mypy 的版本配置；更宽版本范围只有验证后才声明。明确依赖安装入口：先安装已验证的 CUDA torch/torchvision，再执行项目及开发依赖安装，避免无意换回其他轮子。修正 README 与无效的 requirements 安装说明。

**验收标准：**

- 在另一个干净的 Windows Python 3.10 虚拟环境，按 README 顺序安装 CUDA 组合和 `-e '.[dev]'` 成功；`pip check` 无冲突，安装后 torch 仍为指定 CUDA 版。
- 模型、数据和训练模块可导入，训练 `--help` 正常，应用启动正常；不依赖评估过程中手动拼出的环境。
- 声明的最低 Python/PyTorch 版本经过导入和关键接口检查；未支持版本给出明确错误，依赖文件不会“执行成功但没有安装任何依赖”。

**状态：** 已修复（2026-10-07）。build-backend 改为 setuptools.build_meta；requires-python >=3.10；ruff/mypy 目标 py310；requirements.txt 为可执行安装入口（CUDA 组合 + -e ".[dev]"）。干净 Windows venv 全流程验证通过，过程中发现并记录关键前置条件——新 venv 的旧 pip 解析 CUDA 组合失败（typing-extensions 名称规范化问题），必须「先升级 pip」，已写入 README 与 requirements.txt。

### F05 · P0 · 多次运行的配置、日志与权重混合

**问题及位置：** `training/trainer.py`、`training/checkpoint.py` 按模型名保存同一个 history；从头训练仍读取旧 global best 分数。checkpoint 只有模型级配置，损失、增强、采样及环境等信息不足。更换实验设置后可能继续展示旧权重。

**修复方案：** 每次运行建立独立 `run_id`，启动前保存最终生效配置、CLI 参数、seed、Git 提交及未提交变更标记、环境、设备/精度、CSV 与划分哈希。`last` 用于续训，`best` 用于本次运行的模型选择，导出清单显式指向 run 和权重。跨运行比较只读取各自结果，不能让旧 global best 控制新实验。迁移历史产物时，未知字段标为 unknown，不用当前 YAML 补成“历史事实”。

**验收标准：**

- 同一模型连续运行两个不同 seed/配置，目录、日志、best 和 last 均分开，第二次不会继承第一次的 best 分数或覆盖其文件。
- 每个结果表和导出模型能反查唯一 run、完整配置、划分及权重 SHA-256；运行失败也保留启动配置和状态。
- 选择旧权重演示时显示其真实来源；无完整历史配置的旧结果明确标注信息缺失。

**状态：** 部分通过（第二轮独立审核，S02）：新 run 隔离、部分协议变更拒绝与失败产物保护通过；增强数值、裁剪/早停与调度器参数仍可绕过协议比对。

**修复轮原记录（历史，当前状态以上述独立复核为准）：** 已修复（2026-10-07）。新增 training/runs/<模型>/<run_id>/ 目录结构（config_effective.yaml、run_meta.json 含 CLI/seed/git/环境/数据指纹、history.json、checkpoints/{last,best,epoch_*,interrupted_*}）；导出经 tools/export_model.py 写 export_manifest.json（run、SHA-256、spec、指标），默认不覆盖同名文件；旧目录整体保留为 legacy。证据：run 隔离测试（tests/test_training_pipeline.py）；短训练闭环（run 20261007_022921_seed42，训练→续训→导出→评估全程可追溯）。

### F06 · P0 · 类别计数错误，可能生成错误损失权重

**问题及位置：** `utils/constants.py` 的 `CLASS_COUNTS=[4953,436,5121,8989,6077,4002,6198]` 总和为 35,776，并非当前 Training 的计数。`training/trainer.py` 在 CB Focal 未传入计数时使用它。实际 Training 按标签 0–6 为 `[3995,436,4097,7215,4830,3171,4965]`。

**修复方案：** 按本次实际训练划分动态统计七个标签，采样器与 CB Focal 读取同一份统计，保存进 run 配置；移除错误常量的静默回退。对子集缺失类别预先规定处理方式，保持七类索引不变，明确报错或合理处理零计数，不把类别位置压缩错位。

**验收标准：**

- 官方 Training 得到上述七个计数，和为 28,709；不读取 PublicTest/PrivateTest 来生成训练权重。
- 用小型、类别不均衡及缺失类别的固定数据验证计数、标签索引和权重；损失及梯度有限，不发生除零或类别错位。
- 划分改变后重新统计，日志能核对采样器和 CB Focal 使用的计数完全一致。

**状态：** 已修复（2026-10-07）。新增 compute_class_counts 按当前划分动态统计（官方 Training = [3995,436,4097,7215,4830,3171,4965]，合计 28,709）；采样器与 CB Focal Loss 共用同一份统计；零计数明确报错；错误常量 CLASS_COUNTS 已删除。证据：tests/test_training_pipeline.py（固定索引/越界/零计数）；训练日志与 run_meta 记录 class_counts。

### F07 · P1 · 不平衡策略同时开启，收益无法归因

**问题及位置：** 默认同时启用逆频率采样、Focal Loss、Disgust 特定增强。无 alpha 的 Focal 主要聚焦难样本，并不直接保证少数类得到合适重加权。现有结果不能说明这些选项各自有效。

**修复方案：** 在现有模型上提供明确的普通随机采样 + 交叉熵配置作为基线；需要说明某个已有选项的收益时，只改变该选项，固定其余设置、划分及 seed。只有已有对照支持时才推荐组合。结果同时记录 accuracy、Macro-F1、balanced accuracy、各类 recall/support，不开展新模型研究。

**验收标准：**

- 基线配置确实关闭加权采样、Focal/CB Focal 和类别专属增强；实际执行设置写入 run。
- 每项“提升某类表现”的结论对应只改变一个已有选项的可追溯对照；没有对照的组合只称当前设置。
- 类别报告包括 PrivateTest Disgust 的 55 张支持数；说明一张样本对应约 1.82 个百分点 recall 变化。允许结果无改善，不能以挑选有利 seed 的方式验收。

**状态：** CE 基线代码前置通过；正式实验未执行（R08）。cross_entropy、普通采样及基线 YAML 已接入；comparison-v2 预算尚未冻结并应用到各模型有效配置，不能宣称策略收益已得到验证。

**修复轮原记录（历史，当前状态以上述独立复核为准）：** 待重训（前置已就绪）。统一评估入口与比较协议草案 docs/comparison_protocol_draft.md 已完成（含 Focal/CB-Focal 预注册口径、只用 PublicTest 做选择、≥3 seeds、macro-F1 主指标）；比较方案冻结与 Focal/CB-Focal 重训按用户决策另行安排。

### F08 · P0 · 验证损失早停条件无法触发

**问题及位置：** `training/trainer.py` 比较 `min(recent_losses) > recent_losses[0] * 1.05`。窗口最小值不会大于第一项；默认非负损失下该条件不能成立。

**修复方案：** 分别定义 accuracy 无改善早停和 loss 恶化早停，使用持久的历史 best、阈值、连续计数及重置规则。可采用“连续 N 轮 val_loss 高于历史最小值的 1.05 倍”的明确规则；恢复训练时保留其状态。阈值、方向和 patience 在配置中合法性检查，触发时记录原因。

**验收标准：**

- 对 loss 序列 `[1.0,1.06,1.07,1.08]`，阈值 1.05、连续 patience=3，应在第 4 个值后触发；不再使用窗口自身最小值比较。
- 中间恢复到阈值以内时恶化计数清零；accuracy 平台在第 N 次连续无改善后停止，改善时清零，patience=0 的含义明确。
- 保存/恢复后 best 和计数保持一致，同一后续指标序列在连续运行和恢复运行中触发于相同位置。

**状态：** 已修复（2026-10-07）。val_loss 恶化判定改为「高于历史最小值的 threshold 倍」纯函数（update_val_loss_monitor），验收序列 [1.0,1.06,1.07,1.08] 第 4 个值触发、中间恢复清零；acc 早停计数独立；触发原因写入 run_meta.stop_reason；配置校验（threshold 必须 >1、patience ≥0、=0 禁用）。证据：tests/test_training_pipeline.py 序列测试。

### F09 · P0 · 续训状态不完整，确定性设置被覆盖

**问题及位置：** checkpoint 未完整保存 AMP scaler、Python/NumPy/Torch/CUDA RNG、采样状态及早停状态；恢复后本轮 best 归零。中断时的部分更新权重与 history 的 epoch 可能不一致。Trainer 又强制关闭此前设置的 cuDNN deterministic。自动续训偏好 best，而 best 通常不等于最新训练进度。

**修复方案：** 先实现可验收的 epoch 边界恢复：保存模型、优化器、调度器、scaler、RNG/采样生成器、best 指标、早停计数、已完成 epoch 和历史。`last` 表示最新完整 epoch，`best` 仅用于评估。中途打断的状态单独标记 partial，不能冒充精确续训；若暂不实现步级恢复，明确从最近完整 epoch 继续。确定性与性能模式由配置决定，初始化和 fit 不再覆盖。checkpoint 采用临时文件写完再替换，避免半写入文件成为可选断点。

**验收标准：**

- 在同一环境、固定数据、`num_workers=0`、确定性 FP32 小训练中，“连续两轮”与“一轮保存后恢复再一轮”的批次顺序、LR、history、best 和早停计数一致，参数满足 `atol=1e-6, rtol=1e-5`。
- CUDA AMP 恢复还需核对 scaler 状态、有限 loss/梯度；只承诺经过测试的同设备/同精度条件，不能承诺跨硬件逐位一致。
- epoch、history 长度和训练状态一致；中途文件明确为 partial，自动恢复选择最新完整 last，不意外退回最佳分数对应的旧 epoch。
- 确定性开关在 Trainer 初始化与 fit 后仍有效；损坏/不兼容断点明确报错，保存失败不破坏上一份可用断点。

**状态：** 部分通过（第二轮独立审核，S01–S03）：单进程起始训练的恢复回归、诊断隔离和正常重复 fit 已通过；原 4-worker 断点切换 0-worker、协议漏项及部分中断后同实例继续仍不满足精确恢复。

**修复轮原记录（历史，当前状态以上述独立复核为准）：** 已修复（2026-10-07）。checkpoint 格式 v2：RNG（python/numpy/torch/cuda）、AMP scaler、best（值/epoch）、早停计数、累计训练时长全量保存与恢复；原子写（失败不破坏上一份断点）；中断断点标记 partial；旧格式断点明确拒绝（不静默降级）。证据：恢复与连续训练逐批次一致（批次顺序哈希、参数 atol=1e-6/rtol=1e-5、history/best/计数一致）；损坏/保存失败/中断用例；真实闭环 resume 验证（短训练 run 续训成功，累计用时恢复）。

### F10 · P0 · 最终评估依赖内存状态，缺失划分会静默用全数据

**问题及位置：** Notebook 可能在 `global_best_model_state` 为 None 时评估本轮最佳或末轮，而不是磁盘全局最佳。`analysis/comparison_report.ipynb` 在 CSV 缺失 Usage 时把全 CSV 当成测试集；当前 CSV 有 Usage，未触发该分支，但以后可能混入训练数据。

**修复方案：** 统一评估入口，强制传入指定 checkpoint、数据协议及 split；使用 F01 的构造和预处理。缺失 Usage 时要求明确的外部划分清单，否则停止，删除“整个 CSV 都是测试集”的回退。验证集用于模型选择，PrivateTest 用于冻结方案后的最终评估。结果保存完整指标、各类支持数、样本行号/预测、权重与划分哈希。

**验收标准：**

- 官方 PrivateTest 的支持数为 3,589，逐样本行号与清单相同，Training 行号不进入评估；缺失/非法 Usage 在评估前报错。
- 同权重、同预处理、同环境下，统一入口、Notebook 和应用对同一批图像的类别预测完全一致。
- best 不在 Trainer 内存时仍明确加载指定文件；换成不同权重后元数据和加载对象实际改变。指标可以从保存的预测与标签重新计算。

**状态：** 已修复（2026-10-07）。新增统一评估入口 utils/evaluation.py + CLI：强制显式 checkpoint + split，与训练一致预处理，输出完整指标、逐样本行号/预测/概率、权重与划分哈希、ECE/NLL；缺失/非法 Usage 明确报错，无「全 CSV」回退。证据：三个 legacy 权重 6 组评估与评估轮数字四位小数一致；notebook 与应用已接入同一入口。

### F11 · P1 · 三模型比较条件不统一，只有单次结果

**问题及位置：** 当前学习率、batch、epoch、激活、dropout、损失及续训历史不同，不能把差异单独归因于残差、GELU 或某种池化。单 seed 也不足以支撑很小的排序差异。

**修复方案：** 仅对现有三个模型重建比较。先声明比较口径：统一训练方案，或各模型在相同、预先冻结的调参预算下比较；两种表分开。固定数据协议和选择指标，至少使用 3 个预先列出的 seed，保留全部运行。先完成 P0 修复再启动正式重训，不加入新的标准模型或架构。

**验收标准：**

- 形成 3 模型 × 至少 3 seed 的独立结果；可以使用 42/43/44，需在训练前固定，不能根据测试成绩换 seed。
- 表格有各次结果、均值/样本标准差、实际 epoch/更新步数/用时、训练方案和 run/权重标识；未完成的运行明确列出，不能静默排除。
- 同表使用同一评估协议和指标；accuracy、Macro-F1 及类别结果联合讨论。排序可以改变、精度可以下降，验收要求实验可信，不要求超过某个分数。

**状态：** 正式比较未执行（R08）。固定共同方案草案已明确，但协议冻结与各模型有效预算落地尚未完成；应先关闭 S01–S03，再冻结方案并开展现有模型的多 seed 比较。

### F12 · P1 · AMP 单步加速被推广为完整训练加速

**问题及位置：** 已有 JSON 的 3 轮 OFF/ON 总时间分别为 MiniCNN 12.35/14.64 秒、VGGLite 20.33/15.35 秒、MicroResNet 12.90/12.87 秒。“1.4–1.8 倍”只能对应原单步测量。`tools/run_amp_comparison.py` 的配对初始化及数据顺序不一致，诊断会修改权重，并与正式输出目录共享状态。

**修复方案：** 每对 OFF/ON 都从同一初始 state、优化器状态、seed 和批次顺序开始；诊断/预热在隔离副本上执行，不能污染正式训练。至少 3 组配对重复，交替执行顺序，CUDA 计时同步。分别报告稳态 step、整个 epoch/流程时间、显存与精度，输出到独立 benchmark 目录。

**验收标准：**

- 配对初始权重的哈希和输入批次索引相同，预热后仍使用规定的相同起点；OFF/ON 各至少 3 次测量，有原始时间、统计量及计时边界。
- 完整训练加速比由完整流程 OFF 时间 / ON 时间计算；小于 1 如实报告，不用 step 比值代替。
- FP32/AMP 的 loss、梯度及更新有限，最终指标单独记录；基准运行前后正式权重和历史文件 SHA-256 不变。

**状态：** 已重做（2026-10-07）。tools/run_amp_comparison.py 重写为配对基准：同一初始权重、同一 RNG 起点、同一数据顺序、交替顺序、3 组配对、1 epoch + eval 完整流程；显存/有限性/批次顺序哈希记录；基准前后正式产物 SHA-256 校验。结果：完整流程加速 1.00x（mini）/1.10x（vgg）/1.00x（micro）；稳态 step 1.41x/1.65x/1.00x；峰值显存约 -30%～-45%；9 组配对批次顺序全部一致。结论已如实写入 README（旧「1.4–1.8x」说法废止，不做单步外推）。

### F13 · P0 · 配置不能可靠控制行为，Windows workers=0 存在兼容问题

**问题及位置：** `configs/training_config.yaml`、`data/dataloader.py`、Trainer：`image_size` 与模型内 48×48 不完全联动；checkpoint 的 monitor_metric/save_best 未实际控制保存逻辑；关闭总 augmentation 仍可能启用 class_specific。`num_workers=0` 时若仍传 persistent_workers=True 会报错。

**修复方案：** 在启动前集中解析、校验并记录实际生效配置。当前项目只支持 48×48 时明确拒绝其他尺寸，不为此改架构。实现已声明的保存开关、监控方向和增强总开关；未支持的配置移除或明确报错。workers=0 时自动关闭 persistent_workers、设置合适的 prefetch；Windows 多进程入口保持规范。

**验收标准：**

- 非 48 尺寸在模型/训练启动前报错；CLI 覆盖后的 batch、LR、epoch、AMP 与日志和实际行为一致，非法值及未支持键不被静默忽略。
- val_acc 选择最大值、val_loss 选择最小值；save_best=False 时不导出 best；augmentation.enabled=False 时普通与类别专属随机增强均关闭。
- Windows 下 workers=0 与一个非零设置（例如 4）都能读取训练 batch；workers=0 不触发 persistent_workers/prefetch 参数错误，关闭后进程正常退出。

**状态：** 上一轮返修通过（第二轮独立审核，R06/R07）：增强总开关控制 MixUp/类别增强，Trainer 入口统一拒绝非法配置，workers=0 兼容与相关回归通过；续训协议漏项另由 F05/F09、S02 跟踪。

**修复轮原记录（历史，当前状态以上述独立复核为准）：** 已修复（2026-10-07）。新增 utils/config_validation.py 集中校验（未知键/非法值/仅 48×48/patience 语义/threshold>1/use_se 支持范围），启动前报错；checkpoint.save_best 与 monitor_metric 实际生效（val_loss 时按最小值保存）；workers=0 自动关闭 persistent/prefetch；类别特定增强受总开关控制；CLI 覆盖后统一校验并记录。证据：tests/test_config_validation.py（9 项）。

### F14 · P1 · 梯度累积尾部不足一组时被缩小

**问题及位置：** `training/trainer.py` 将 loss 除以固定的完整累积步数，最后不足一组时仍按完整组除，末组更新偏小；中断后还需要保证旧梯度不残留。默认累积步数为 1 时不触发尾部问题。

**修复方案：** 每个累积组按实际样本数归一化，正确处理不等长末 batch；组开始清零，最后一个微批也更新。AMP 在组结束先 unscale、再归一化/裁剪，并只更新一次 scaler/优化器。诊断若使用累积，也需与正式逻辑一致。

**验收标准：**

- 无 BatchNorm/Dropout 的固定线性模型，用同一 8 个样本分成 3+3+2、累积为 3，与一次 8 样本均值损失的更新在 FP32 下满足 `atol=1e-6, rtol=1e-5`。
- 完整组、尾组不足 K、累积=1 均执行正确次数的 step，重启训练前梯度清零；AMP 尾组更新有限。
- 不把具有 BatchNorm 的 CNN 不同 batch 划分当作数学等价性测试；正式比较保持实际批次设置可追溯。

**状态：** 已修复（2026-10-07）。梯度累积按组内实际样本数归一化（尾组不足一组不再缩小更新）；组末 unscale/裁剪/单次 step；诊断使用同一归一化；组开始清零梯度。证据：3+3+2 与单批 8 样本等价（数学级与 Trainer 级，atol=1e-6/rtol=1e-5）。

### F15 · P2 · 演示输入范围超出实际支持

**问题及位置：** `inference/app.py` 把整张上传图片缩到 48×48，没有定位人脸。普通生活照的背景可能成为输入，当前演示不足以支持自动人脸情绪识别的表述。

**修复方案：** 将现有演示明确限定为“已裁剪的单张人脸表情类别预测”，在上传说明中给出输入要求及示例。统一灰度、尺寸、归一化等预处理；对损坏、无法解码和不支持输入给出清晰提示。当前阶段不新增人脸检测、对齐、多脸处理或拒识系统。

**验收标准：**

- 同一已裁剪 FER 图像经应用与离线评估得到相同预处理张量及预测；输入规格明确为单脸、48×48 灰度的训练域。
- 损坏文件不产生伪造结果或无提示崩溃；上传说明、README 和演示名称不承诺自动处理整张生活照或多张脸。
- 对外表述为数据集表情类别预测，不把结果当作真实心理状态的测量。

**状态：** 已修复（2026-10-07）。应用限定「已裁剪的单张人脸」并写明输入要求；48×48 输入不重采样（与离线评估逐样本一致）；损坏/无法解码文件明确提示；README/应用文案不承诺整图或多脸处理。证据：tests/test_inference.py（预处理恒等 + 应用/评估一致）、tests/test_app.py（文案与启动）。

### F16 · P2 · Softmax 被表述为已验证的正确概率

**问题及位置：** 应用将最大 Softmax 值显示为“置信度”，但未校准。此前 PrivateTest、15 个等宽区间估计的 ECE 分别为 0.1123/0.2144/0.0645，只能描述该次测量。

**修复方案：** 将界面文案改为“模型概率（未校准）”，保留已有输出，不引入新的校准方法。对当前概率输出报告 ECE/NLL 时，明确公式、区间、数据划分和权重；修复模型加载后如重新计算这些数值，保留测量版本，不能沿用不匹配权重的旧数字。

**验收标准：**

- 应用与 README 没有把 0.9 等同于“90% 会预测正确”的承诺；未校准状态明确。
- ECE/NLL 报告能够从保存概率与标签重算，概率有限、归一化正确，注明 15 区间及权重/划分。
- 不以更改术语冒充已完成校准；未实施校准时，所有对外材料均保留“未校准”说明。

**状态：** 已修复（2026-10-07）。界面与文档改为「模型概率（未校准）」并说明语义；ECE（15 等宽区间）/NLL 由统一评估的概率输出重算（PrivateTest：0.1123/0.2144/0.0645；NLL 1.0381/1.0538/0.9025），公式与口径记录于评估结果 JSON，可用保存预测复算。

### F17 · P2 · Grad-CAM 热图缺少行为验收，解释过强

**问题及位置：** 现有热图通过了单次形状和值域检查，但不能因此认定模型关注区域正确或具有因果意义。

**修复方案：** 保留当前 Grad-CAM 功能，说明其为目标类别的梯度响应辅助图。使用统一预处理和指定目标类别，确保 eval 状态、梯度开启范围和 hook 清理正确，避免多次调用积累 hook 或修改模型参数。检查固定小例子和重复调用行为，不新增解释方法研究。

**验收标准：**

- 三模型固定输入输出形状正确，数值有限，归一化结果在 [0,1]，零响应时有明确定义；目标类别超出 0–6 时报错。
- 连续调用至少 10 次，hook 数量不增加，权重哈希与普通推理结果不被热图调用改变；模型的训练/eval 状态按约定恢复。
- 界面与代码侧说明称辅助可视化，不把一张热图作为模型因果机制或心理解释的证据。

**状态：** 已修复（2026-10-07）。Grad-CAM：hook 在 finally 中清理（异常安全）、调用后零梯度与训练状态恢复、目标类别越界报错、零响应定义为全 0；10 次连续调用 hook 数不变、参数不变。证据：tests/test_inference.py；界面表述为辅助可视化、不宣称因果解释。

### F18 · P1 · 质量工具与 CI 状态不能支撑质量承诺

**问题及位置：** 此前 36 项测试通过，本轮在 CUDA 包环境中复跑也通过，但多为基础行为；Ruff 此前发现 142 项问题，mypy 被 `trainer` / `training.trainer` 重复模块映射阻断，配置中的 Python 3.9 与当前工具不兼容。工作区 CI 文件此前已删除，README 仍有 badge，工具列出不代表执行通过。

**修复方案：** 先统一包边界/导入方式与 Python 3.10 配置，使 mypy 真正运行；逐项处理 Ruff 发现，必要忽略需有局部理由，不能全面关闭规则掩盖问题。为 F01/F05/F08/F09/F10/F13/F14 加入关键回归检查。CI 文件删除若为既有意图，就移除失效 badge；如恢复 CI，按真实执行结果展示状态。测试输出隔离在临时目录，不能污染正式 run。

**验收标准：**

- 现有测试和上述关键回归检查全部通过；Ruff 和 mypy 返回成功，mypy 无模块映射阻断。测试数量与实际执行结果同步。
- README badge 指向真实存在的工作流和仓库；若没有 CI，材料不宣称 CI 已通过。普通 CPU CI 不宣称已验证 CUDA 训练。
- 测试前后原数据、正式日志和导出权重哈希不变；允许用临时小数据测试正确性，不把基础测试当作完整重训复现。

**状态：** 部分通过（第二轮独立审核，S05）：112 项测试、Ruff、pip check 通过；mypy 实际返回 1，tests/test_training_pipeline.py:691 存在 no-any-return。恢复边界缺少有效回归，不能写“全部质量检查通过”。

**修复轮原记录（历史，当前状态以上述独立复核为准）：** 已修复（2026-10-07）。补 data/training/inference 的 __init__.py 统一包边界后 mypy 真正运行（46 → 0 错误）；ruff 全绿（少量 per-file 忽略均有局部理由）；测试 75 项全部通过（核心 36 + 训练管线 18 + 推理 8 + 应用 4 + 配置校验 9）；README 移除失效 CI badge（保持无 CI，按用户决策），仓库链接统一；测试输出隔离于 tmp_path，不污染正式 runs。

### F19 · P2 · README、Notebook 与应用的表述和来源需一致

**问题及位置：** README、Notebook、应用存在 val/test 混用、旧参数量、不同仓库链接等问题。用户已确认项目独立完成；课程 DOCX/PPT 不再修改，不要求个人贡献清单。2021 年论文的 73.28% 不能叫当前 SOTA，也不能与 val_acc 直接相减；“灰度减少整体计算约三分之二”“噪声推得准确率硬上限 90–95%”无充分支持。

**修复方案：** 从 F10 的统一结果记录同步所有表格，指标明确 split、run、配置与权重。统一实际仓库链接；历史论文结果只在协议可比时作为有日期的参考，删除没有测量或推导的结论。说明已有方法来源；按用户确认的独立完成事实表述项目，不增加组员分工或个人贡献清单。申请材料围绕当前项目及已完成修复，不添加尚未执行的新研究方案，不把待办写成成果。

**验收标准：**

- 参数、指标和图表在 README/Notebook/应用中一致，accuracy 附 split，能追溯至同一结果文件；遗留历史结果有清楚日期与来源。
- 删除或严格限定上述不支持表述，不混用 PublicTest 与 PrivateTest，不宣称普通 CNN/残差/SE/Focal 是个人原创。
- 项目按个人独立完成表述；展示仅包含已完成成果、限制和当前修复状态，不要求额外贡献清单或报告/PPT 更新。

**状态：** 部分完成（第二轮独立审核，S04）：链接、参数注释、PublicTest 默认值、UTF-8 日志和现场作图已修复；比较来源一致性仍未校验；报告/PPT 修改及贡献清单要求已按用户确认取消，本轮仅审核与更新文档。

### F20 · P0 · 后续 GPU 训练需要可验证的 Windows CUDA 环境

**问题及位置：** 先前 Windows 核验环境为 CPU 版 PyTorch，不能承担后续 CUDA/AMP 训练。用户已明确需要 CUDA 版并在项目内统一使用 Windows `.venv`。

**修复方案：** 在同一 `.venv` 安装官方 `torch==2.14.1+cu130`、`torchvision==0.29.1+cu130`，保留现有版本组合；记录解释器、GPU、驱动、CUDA runtime 和包快照。执行真实 GPU 张量计算，并对当前三模型分别做 FP32 与 AMP 前向、反向及优化器更新；验证训练入口和环境依赖。检查输出放系统临时目录，不覆盖旧权重或正式日志。

**验收标准：**

- 解释器路径属于本项目 `.venv\Scripts`，torch/torchvision 版本为指定 CUDA 组合，`pip check` 通过，Notebook 内核仍指向此解释器。
- `torch.cuda.is_available()` 为 True，设备为本机 RTX 5070 Ti Laptop；实际 GPU 张量计算通过，三模型 FP32/AMP 的前向输出、loss 与参数有限，实际更新前的反缩放梯度有限，每个模式均有有效更新。AMP 动态缩放跳过的更新须记录；持续跳步或非有限参数不能通过。
- 现有测试在新环境和隔离输出目录中通过，训练 `--help` 正常；原 CSV、日志和权重 SHA-256 保持不变。
- 环境验收只证明具备 GPU 运算及最小训练能力；正式整轮训练、最佳 batch、性能倍数与准确率需要在代码修复后另行验收。

**状态：** 已完成（2026-10-07）：版本、依赖、Notebook 解释器、真实 GPU/torchvision 运算、三模型 FP32/AMP 有效更新、36 项既有测试和训练帮助均通过。AMP 启动时跳过更新的次数已披露，正式数据训练及性能实验尚未执行。

## 修复顺序和正式重训的进入条件

| 阶段 | 执行内容 | 阶段状态（2026-10-07 修复轮后） |
|---|---|---|
| 1 · 环境 | F20 CUDA、F04 安装入口 | F20 已验收；F04 已修复（干净 venv 安装验证通过，含「先升级 pip」前置条件） |
| 2 · 正确性 | F01、F05、F06、F08、F09、F10、F13；F14 | 75 项现有测试通过；F05/F09/F13 仍需按 R01–R07 补修与回归，未全部验收 |
| 3 · 数据与口径 | F02、F03，冻结 F07/F11 比较方案 | 参数/重复披露已核验；CE 基线与比较预算尚未齐备（R08）；dedup-v1 尚未实施 |
| 4 · 当前模型重训 | 现有三模型；按需要验证既有不平衡选项与 AMP（F07/F11/F12） | **未执行（按用户决策暂缓）**；F12 速度基准已重做 |
| 5 · 材料提交 | F15–F19 | 当前状态见第二轮审核；F19 仍有 S04，报告/PPT 与贡献清单不纳入验收 |

正式重训前至少完成全部 P0 的适用条件。F03 的官方结果披露与附加去重结果分别验收，不能把“计划去重”写成“已消除泄漏”。F14 默认 K=1 不阻止这一配置重训，但启用 K>1 前必须通过尾组检查。GPU 已装好也不能替代模型加载、run 隔离或续训修复。

建议先选一个现有模型做修复后的短训练，检查一次保存、恢复、导出和统一评估，再启动完整比较。短训练通过只代表流程可用，不把其分数作为正式结论。正式训练前保存协议和配置，之后不根据 PrivateTest 反复挑选方案。

## 本轮交付及证据边界

此前环境核验交付了本评估文档、证据摘要与 Windows CUDA 环境快照；第一轮重构随后补充了代码、工具与 README。本次独立审核更新同一文档及证据文件，不修改模型、训练代码、正式数据和权重。正式重训与剩余正确性修复仍待完成；报告/PPT 与贡献清单已排除待办；继续限定在现有三个模型与 FER2013 项目内。

之前的证据包括：完整 CSV 检查、三个现有权重的 CPU 评估、错误激活加载对照、续训与早停探针、36 项测试、Ruff/mypy 检查、无上传图片的应用启动及三个 Grad-CAM 基础检查。现有导出权重与各自 global_best.pth 的 SHA-256 一致。没有完成从头训练复现，也没有验证真实生活照交互或完整 GPU 性能。本轮 CUDA 检查单独记录于 `cuda_environment`，覆盖真实张量运算、torchvision CUDA 运算及三模型 FP32/AMP 更新；不覆盖上述历史完整数据评估结果，也未验证完整训练性能。

本次不恢复工作区已有的 `.github/workflows/ci.yml` 与 `emotion_recognition.code-workspace` 删除，不修改原数据、旧日志、正式权重或既有 Notebook/报告。

---

## 修复轮（2026-10-07）交付与证据

- **代码**：utils/model_spec.py、utils/evaluation.py、utils/config_validation.py；
  training/{trainer, checkpoint, train.py} 重构；inference/{infer_utils, app}.py；
  data/dataloader.py；tools/{export_model, evaluate_checkpoint, data_audit,
  benchmark_efficiency, run_amp_comparison}.py。
- **测试与质量**：75 项测试全部通过（tests/：核心 36 / 训练管线 18 / 推理 8 /
  应用 4 / 配置校验 9）；Ruff 全绿；mypy 全绿（修复模块映射后 46 → 0）；
  测试输出隔离（tmp_path），不污染正式 runs。
- **文档**：README.md（全面重写）；docs/data_audit.md（数据披露 + dedup-v1 草案）；
  docs/comparison_protocol_draft.md（F07/F11 比较协议草案）。
- **实测产物**：analysis/evaluations/（legacy 权重 6 组 + 流程验证 1 组）、
  analysis/data_audit.json、analysis/efficiency_results.json、
  analysis/benchmark_amp/results.json。
- **环境**：干净 Windows venv 全流程安装验证通过（torch 2.14.1+cu130、CUDA 可用、
  pip check；发现并记录「先升级 pip」前置条件）；短训练闭环验证完成
  （mini_cnn run 20261007_022921_seed42：1+1 epoch 训练 → `--resume auto` 续训 →
  导出（export_manifest 记录）→ 统一评估；评估 acc=0.3463 与训练记录一致）。
- **明确未做**（按用户决策或依赖重训）：3 模型 × 3 seeds 正式重训；
  Focal/CB-Focal 比较（协议草案待冻结）；dedup-v1 实施与去重重训；
  CI 恢复（保持无 CI）；课程报告/PPT 修改和贡献清单已取消要求。

## 第一轮重构独立审核（2026-10-07）

### 结论与已通过事项

**结论：基础工程重构通过，完整验收暂不通过。** 当前项目可以继续补修和小规模流程验证；尚不能将“默认配置精确续训”或“三模型公平比较已完成”作为已验证成果。新建 CLI run 的普通路径已有闭环证据，但不能据此覆盖 Notebook、跨配置恢复、多 worker 或重复 fit 的不同路径。

本次审核针对提交 `6d02afd`。所有 Python 使用本项目 Windows `.venv\Scripts\python.exe`，临时探针与训练输出放系统临时目录；保留的证据位于 `PROJECT_REVIEW_EVIDENCE.json` 的 `first_refactor_independent_audit`。本次未下载依赖、未执行正式 FER2013 重训、未重新运行完整 AMP 性能基准。

| 独立检查 | 结果与边界 |
|---|---|
| pytest | 75 passed，7.71 s；1 条 Grad-CAM backward hook 提示，无失败或跳过；测试线程设为 4 |
| Ruff / mypy | Ruff 通过；mypy 32 个源文件无问题 |
| 环境 / 依赖 | Python 3.10.11；torch 2.14.1+cu130；torchvision 0.29.1+cu130；CUDA 13.0；RTX 5070 Ti Laptop 实际矩阵运算、pip check、Notebook 内核路径通过 |
| 模型/推理 | 现有回归检查确认 MicroResNet 按 GELU 加载、48×48 预处理一致、Grad-CAM 基础行为正常；三模型参数计数与效率结果一致 |
| 已保存评估 | 6 组 legacy + 1 组短训练结果复算通过；accuracy/macro-F1/balanced accuracy 完全一致，ECE/NLL 差异 <1e-4；本次没有重新做全部样本模型前向 |
| 导出清单 | 4 份导出权重 SHA-256 与清单一致；legacy 的来源不完整性仍保留 |
| 数据披露 | 重复组 1,516，超出唯一像素记录 1,853，冲突组 57；Training 与 PublicTest/PrivateTest 重叠 280/288 条；敏感性分析复跑与现文档一致 |
| 原产物保护 | 原 CSV、历史日志/断点、已有导出权重/清单及真实 runs 共 40 个文件在审核前后逐文件校验；结果见证据文件 |

R01–R07 是本次复现的正确性返修项；R08 是正式比较的未完成前置；R09 是材料与工具使用问题。下列 P1 均应在使用相应功能或开展正式比较前完成，F05/F09/F13 原 P0 条件因此尚未关闭。

### R01 · P1 · 持久 worker 下恢复不等价于连续训练（F09 重新打开）

**位置：** [data/dataloader.py](D:/Document/Unniversity/emotion_recognition/data/dataloader.py:325)、[training/checkpoint.py](D:/Document/Unniversity/emotion_recognition/training/checkpoint.py:191)、`tests/test_training_pipeline.py::test_resume_matches_continuous`。

**实测与影响：** 默认 YAML 使用 `num_workers=4, persistent_workers=true`，但恢复一致性测试只覆盖 `num_workers=0`。本次在真实 MiniCNN、14 个固定合成样本、WeightedRandomSampler、2 个持久 worker、关闭增强的最小条件下比较“连续 2 轮”和“1 轮后重建 Trainer 恢复再 1 轮”：第二轮标签顺序不同，参数最大差值 **0.00748777**，验证 loss 为 **1.41245735 / 1.41973412**。这说明仅恢复父进程 RNG 不足以覆盖持久迭代器的恢复语义；真实增强还有 worker 内 RNG 状态。2-worker 探针证明该机制有缺陷，本次未额外跑完整 4-worker FER2013 训练。

**修复方案：** 先明确精确恢复的支持范围。可先在该模式强制 `num_workers=0` 并记录，不再对默认持久 worker 作无条件承诺。若保留多进程，至少为 sampler 与 DataLoader 配置独立 generator 并保存/恢复其状态，按 epoch 重建 worker 或实现可复算的 epoch/sample 增强种子，避免新迭代器额外消耗采样 RNG；不能只保存主进程 `torch.get_rng_state()`。性能模式与精确模式的区别写入 run。

**验收标准：** 在宣称支持的 workers=0/2/4、持久开关、增强开关及加权采样组合下，以同 seed 比较连续与恢复训练；逐批原始行号/样本 occurrence 和增强后输入一致，参数 `atol=1e-6, rtol=1e-5`、history、LR、scaler、best 与早停计数一致。无法保证的组合启动时明确拒绝精确恢复或标注为非精确模式；不得只检查类别标签哈希。

**状态：** 部分修复，第二轮仍未验收（S01）。workers=0 从起始训练的恢复回归通过；只在恢复时强制 4→0 并记录，不能保证原 4-worker 断点精确继续。真实短训练闭环完成只证明可运行，不能证明与连续训练等价。

### R02 · P1 · 更换训练配置或数据仍可续写原 run（F05/F09 重新打开）

**位置：** [training/checkpoint.py](D:/Document/Unniversity/emotion_recognition/training/checkpoint.py:324)、[training/trainer.py](D:/Document/Unniversity/emotion_recognition/training/trainer.py:449)。

**实测与影响：** 加载只比较 `model_spec`。本次将 Focal 换为 CB-Focal，同时改变数据张量和数据指纹，原断点仍成功加载。实际 criterion 是 `CBFocalLoss`、数据指纹变为 `b…b`，同一 run 的 `config_effective.yaml` 仍写 `loss_type: focal`、`run_meta.data.csv_sha256` 仍是 `a…a`。日志可追溯性失效；更换 monitor、batch、增强、AMP 或调度器也未受同等约束。

**修复方案：** 在 checkpoint 保存训练协议/有效配置与数据指纹，在恢复前比对损失、采样、增强、batch/累积、优化器/调度器、monitor、精度模式及 CSV/划分指纹。允许变更的字段使用明确白名单（如本次追加轮数）；改变训练策略应建立新 run 并记录父断点，不能继续称为原 run 的精确恢复。校验通过后再写恢复事件，每次事件记录当次 CLI/config/git/environment；失败不得先改写原 run 元数据。

**验收标准：** 同配置/数据可恢复；逐项修改 loss、sampler、monitor、batch、增强、调度器或 CSV 字节/划分时，在更新权重和原 run 文件前报错。若提供显式派生运行，必须有新 run_id、实际生效配置与 parent checkpoint SHA；旧 run 所有文件哈希保持不变。

**状态：** 部分修复，第二轮仍未验收（S02）。已保存协议、拒绝部分字段变化并在失败时保护产物；旋转幅度、类别增强目标、梯度裁剪、早停及 scheduler_t0 等有效参数漏入快照，六项独立变更仍被接受并标记 protocol_verified。

### R03 · P1 · Notebook 诊断污染正式训练状态（F09/F12）

**位置：** [training/trainer.py](D:/Document/Unniversity/emotion_recognition/training/trainer.py:533)、[02_train_baseline.ipynb](D:/Document/Unniversity/emotion_recognition/training/notebooks/02_train_baseline.ipynb:159)。

**实测与影响：** MiniCNN Notebook 在同一个 `trainer` 上执行 `diagnose()` 后再 `fit()`；诊断执行真实更新。本次 `diagnose(num_steps=1)` 包含 3 步预热，累计 **4 次优化器更新**，参数最大变化 **0.49847969**，RNG 改变，但 history 仍为 0 轮。这样正式训练不再从声明的初始化/恢复状态起步；从断点恢复后运行该单元也会产生未记录的额外更新。重写后的 AMP 配对工具已用副本预热，不能替代此 Notebook 路径的隔离。

**修复方案：** 诊断使用独立模型、优化器、scaler 和独立数据加载器；完成后恢复主运行 RNG，不能消费正式持久 worker 的队列/增强状态。也可从 Notebook 移除同实例诊断，改为独立工具，再重新构建正式 Trainer。保留 K>1 时的真实分组 step 语义，或明确其诊断口径并禁止拿它外推正式训练。

**验收标准：** 诊断前后正式模型参数/BN buffer、优化器、scheduler、scaler、RNG、loader/sampler 状态及 run 文件逐项不变；同 seed 的“直接 fit”与“先诊断再 fit”得到同批输入及同等训练结果；FP32/AMP 和从 last 恢复的路径均覆盖。

**状态：** 本轮覆盖范围内通过。独立模型/优化器/scaler/loader 诊断副本及 CPU 回归通过；补充真实 CUDA/AMP MiniCNN 检查确认诊断前后模型/BN、optimizer、scaler、父进程 Python/NumPy/Torch/CUDA RNG 和 run 文件不变。该补充检查从完成一轮的小样本实例执行，未遍历所有模型、所有恢复状态或自定义有状态 dataset。

### R04 · P1 · 同一实例重复 fit 会重复轮号并覆盖定期断点（F09）

**位置：** [training/trainer.py](D:/Document/Unniversity/emotion_recognition/training/trainer.py:898)，三个训练 Notebook 的“可反复调用续训”说明。

**实测与影响：** `fit()` 结束没有推进 `start_epoch` 或累计会话用时。连续调用两次 `fit(1)` 后，history 与 last.epoch 均为 2，但 `start_epoch=1`、`run_meta.final_epoch=1`，只有 `epoch_0001.pth`，第二次覆盖第一轮定期断点。best_epoch、保存频率和累计时长也可能错误。重新创建 Trainer 并 load 的测试不能覆盖同实例反复运行 Notebook 单元。

**修复方案：** 以已完成 history 轮数统一推进下一轮起点，正常结束、早停和中断分别处理；每个会话结束更新累计训练时长。保留本会话起点的局部变量用于输出，避免循环内修改起点影响迭代。partial 中断须先回到完整 last 状态再精确继续，不能在已部分更新的模型上冒称完整 epoch 恢复。

**验收标准：** 在同实例上 `fit(1); fit(1)` 与 `fit(2)` 的批次/参数/history 一致，轮号为 1、2，`start_epoch=3`、last.epoch 与 final_epoch 为 2，定期断点分别保存为 0001/0002；累计时长递增。追加测试早停后再次 fit、load 后分段 fit 和中途中断。

**状态：** 部分修复，第二轮仍未验收（S03）。正常分段 fit、轮号/断点及早停后再次 fit 的回归通过；KeyboardInterrupt 后未回滚完整 last，继续调用同实例 fit 会保留未完成轮的参数/优化器更新。

### R05 · P1 · Trainer 默认设备选择后模型仍留在 CPU

**位置：** [training/trainer.py](D:/Document/Unniversity/emotion_recognition/training/trainer.py:299)。

**实测与影响：** `model.to(device)` 先于默认设备解析。GPU 主机上省略 `device` 时，`self.device=cuda`、模型参数却在 CPU，首个训练批次报 `Input type (torch.cuda.FloatTensor) and weight type (torch.FloatTensor) should be the same`。CLI 与现 Notebook 显式传设备，正常路径不受此项影响；模块文档中的默认调用会失败。

**修复方案：** 先规范化并解析 `self.device = torch.device(...)`，再 `model.to(self.device)`，随后创建 optimizer/scaler；所有入口共用此顺序。

**验收标准：** CUDA 主机上省略 device、显式 cuda/cpu 以及无 CUDA 时的默认路径均能完成有限的前向、反向与有效参数更新；模型、输入和优化器状态在正确设备上。GPU 用例应检查实际运算，不能仅 mock 可用性。

**状态：** 本轮回归通过。设备解析先于模型移动，默认/显式设备与实际 GPU 更新用例通过；未复现上一轮 CPU/CUDA 不匹配。

### R06 · P1 · 增强总开关不能关闭 MixUp（F13 重新打开）

**位置：** [training/trainer.py](D:/Document/Unniversity/emotion_recognition/training/trainer.py:345)。

**实测与影响：** 合法配置 `augmentation.enabled=false`、`mixup.enabled=true` 通过校验，却得到 `trainer.mixup_enabled=true`；4 个训练批次实际调用 MixUp **4 次**。关闭增强的对照仍有批级增强，实际设置与声明不符。类别专属增强的总开关修复不能覆盖这个分支。

**修复方案：** 有效 MixUp 开关由总开关与子开关共同计算；在 run 记录有效增强设置。总开关关闭时禁用所有图像/类别/批级增强，或对矛盾配置明确报错，统一 CLI 与 Notebook 语义。

**验收标准：** 覆盖总开关 × MixUp 开关的四种组合；总开关 false 时实际调用数为 0、输入/标签无混合；总开关 true 且 MixUp true 才生效。类别专属增强与普通增强做同样组合检查，配置快照与运行日志一致。

**状态：** 本轮回归通过。总开关与 MixUp 子开关组合及实际调用次数检查通过，类别增强总开关检查通过。

### R07 · P1 · 配置校验只覆盖 CLI，Notebook/API 可静默降级（F13）

**位置：** [training/train.py](D:/Document/Unniversity/emotion_recognition/training/train.py:89)、[training/trainer.py](D:/Document/Unniversity/emotion_recognition/training/trainer.py:299)、[utils/config_validation.py](D:/Document/Unniversity/emotion_recognition/utils/config_validation.py:102)。

**实测与影响：** 三个训练 Notebook、`load_config()` 与 Trainer 构造均不调用集中校验。`loss_type=misspelled_loss` 被独立 validator 正确拒绝，却能构造 Trainer 并静默使用 FocalLoss。另将模型学习率设为 NaN，集中 validator 仍接受，因为仅做区间比较；非法数值未在声明的集中校验阶段被拒绝，错误可能延后到下游或影响监控行为。未知参数与非法值因入口不同而有不同语义。

**修复方案：** 将完整配置校验放进 CLI/Notebook/API 共同调用的构造入口，且在模型移动、创建 run 或更新旧 run 之前执行；未知 loss 使用明确错误分支。数值验证先 `math.isfinite()`，补全类别增强数值与合法 seed 范围。不要仅在 Notebook 增加一条注释声称已校验。

**验收标准：** 同一组未知键、拼错 loss、NaN/Inf 学习率/阈值/dropout、非法增强概率、非法 seed，从 CLI、Notebook 构造单元和 API 均在首批训练与任何正式 run 写入之前拒绝；合法当前三模型配置均通过。新增回归应验证实际入口，不只直接测试 validator 函数。

**状态：** 本轮回归通过。Trainer 构造入口集中校验，未知 loss、NaN/Inf、非法 seed/增强参数及失败不创建 run 的检查通过。

### R08 · P1 · 普通 CE 基线及公平比较的前置仍未完成（F07/F11）

**位置：** [utils/config_validation.py](D:/Document/Unniversity/emotion_recognition/utils/config_validation.py:61)、[training/trainer.py](D:/Document/Unniversity/emotion_recognition/training/trainer.py:365)、[comparison_protocol_draft.md](D:/Document/Unniversity/emotion_recognition/docs/comparison_protocol_draft.md:8)。

**证据与影响：** 原 F07 要求“普通随机采样 + 交叉熵”基线，当前配置只接受 focal/cb_focal，CLI 无 gamma 覆盖，草案也只写 Focal vs CB-Focal；本次 `loss_type=cross_entropy` 被拒绝。虽然 FocalLoss(gamma=0) 数学上等价 CE，尚未通过可记录的配置入口接入。草案声称同一训练配置，而基座 YAML 的模型级 LR、batch、num_epochs 分别不同；比较预算和执行轮数的最终口径未冻结。不存在 3 模型 × 3 seeds 正式结果，不能将“草案已写”列为实验已验收。

**修复方案：** 为现有训练器加入显式 CE 配置分支（或可记录、经校验的 gamma=0 方案），配套普通随机采样、关闭类别专属增强的基线配置。保留现有三架构，先选定并记录“固定共同方案”或“相同调参预算”之一，写明每模型实际 LR/batch/epochs、早停/调度器、选择指标和 seeds；基线与已有选项的对照每次只改变一项。先补修 R01–R07，再冻结方案，不自动启动训练。

**验收标准：** CE 配置可由同一入口启动，criterion、采样器、有效增强及轮数在 run 中可核查；小例子 loss/梯度与普通 CrossEntropyLoss 等价。协议有冻结日期/commit 和明确预算表，PrivateTest 不参与选择；正式执行后保留每臂 seeds=42/43/44 的全部结果与 mean±样本 std，不能先看 PrivateTest 再修改方案。正式比较前本项状态保持“前置未齐备/实验未执行”。

**状态：** CE 基线代码前置通过；协议冻结、预算落地与正式实验未完成。cross_entropy 分支及等价性回归、configs/baseline_config.yaml、CLI --config 已接入。comparison-v2 提出 LR=3e-4、batch=128、最多 90 轮等固定共同方案，但主配置/基线配置仍有旧模型级覆盖；冻结时须生成并核查每模型/每臂实际配置，记录日期与 commit，再执行正式比较。

### R09 · P2 · 材料口径、比较 Notebook 和 Windows 日志仍需收口（F02/F18/F19）

**位置与证据：** README 与 pyproject 指向 `BernardW18/fer2013-emotion-recognition`，实际 remote 为 `BernardW18/emotion_recognition`；YAML batch 注释仍写 50K/1.5M/400K。比较 Notebook 的训练历史自动取最新 run（当前 MiniCNN 是 2 轮 smoke）或 legacy，测试却固定读取 `<model>_best.pth` 的 legacy 权重，混淆矩阵也读取旧公共图片，不能当成同一批正式实验。三个训练 Notebook 仍默认 `PrivateTest`，与未冻结草案阶段仅用 PublicTest 的约定不匹配。课程 DOCX/PPT 与个人贡献清单按当前用户确认排除修改/编制范围。

本次还复现：Windows 默认 GBK 下将 `tools/data_audit.py` 输出重定向到日志，打印 `↔` 时 UnicodeEncodeError，JSON 未生成；同一命令加 `-X utf8` 成功。上述历史总览中“全部修复”“代码/README 仍未修改”等相互矛盾表述，本次已在本审核文档校正，README 的功能承诺仍需下一轮同步。

**修复方案：** 用实际 remote 统一链接、更新遗留参数注释；比较 Notebook 通过显式 run/checkpoint/结果清单读取同源 history、参数、指标和图，不按 mtime 自动挑选正式模型，不混用 smoke 与 legacy。调试单元默认 PublicTest；PrivateTest 作为冻结后的独立最终步骤。README 清楚限定精确续训的实测条件；项目按个人独立完成表述，不新增贡献清单。Windows 工具的日志入口统一 UTF-8（或安全的 ASCII 输出），把可执行命令写进文档。

**验收标准：** 新 run 生成后，同一模型的曲线、汇总、混淆矩阵与表格都可追溯同一 checkpoint/run/协议；缺文件或来源混合时明确拒绝正式比较。无旧参数量、错误仓库链接或未验证的“全部正确”表述；UTF-8 日志重定向能完成工具并生成有效 JSON。README/Notebook/应用与同一结果记录一致；报告/PPT 不再修改、无需个人贡献清单。

### 下一轮修复顺序与提交门槛

1. 先修 R02/R07（恢复及配置约束）、R05/R06（设备与开关），防止错误设置进入训练。
2. 修 R01/R03/R04，并把本次探针转成隔离、可重复的回归检查；保留已通过的 75 项基础检查。
3. 接入 R08 的 CE 基线并冻结现有模型比较口径；重新做一次短训练→完整 last 恢复→导出→PublicTest 评估，核对配置与数据指纹。
4. 通过后再执行现有模型正式训练；按 R09 整理与结果同源的演示及申请材料。

**历史执行记录（第二轮修复作者记录）**：曾报告步骤 1–3 已完成；本轮独立审核确认其中 R01/R02/R04/R09 尚有 S01–S04，协议仍未冻结，不能将该记录当作完整验收。正式训练未完成；报告/PPT 修改与贡献清单不再列为待办。

用于套磁时，当前适合展示“完成统一加载、可复算评估和数据审计，并通过独立审核发现和修复训练边界问题”的过程。尚不能声称完全可复现、已消除数据重叠或已证明所有不平衡选项有效；这些结论必须来自对应验收与正式对照。无需增加新问题或新架构来替代当前修复。

**状态：** 部分通过，比较来源一致性仍未验收（S04）。实际 remote 链接、参数注释、PublicTest 默认值、UTF-8 重定向与现场生成图表通过；显式路径仅避免自动挑最新文件，尚不能拒绝混用新 smoke history 和旧权重。报告/PPT 不再修改，个人贡献清单不要求编写，两项均取消待办。

---

## 第二轮修复（R01–R09）执行记录（2026-10-07）

以下保留修复作者的历史执行记录；当前独立验收结论以紧接其后的“第二轮重构独立审核”为准。其中 mypy 全绿、4→0 精确恢复、比较不混源等表述未获本轮实测支持。

- **质量门槛**：112 项测试全部通过（新增 R01–R08 回归 37 项）；Ruff / mypy 全绿。
- **闭环复核**（按审核第 3 步重做）：新短训练 run `20261007_031251_seed42`（mini_cnn）：
  训练 1 epoch → `--resume auto`（自动应用精确恢复条件 num_workers 4→0 并记录于
  resume_events；训练协议比对通过；恢复至第 2 轮完成）→ 导出（manifest 记录 run 与
  SHA-256）→ PublicTest 统一评估（acc=0.3689；与训练时 GPU/AMP 验证值 0.3681 相差
  3/3589 个样本，源于 AMP 与 CPU-FP32 数值路径差异，如实记录、非缺陷）。
  run_meta 数据指纹（csv_sha256=3b8d…、splits 28709/3589/3589）与配置完整可核查。
- **材料范围（用户已确认）**：课程报告 DOCX / PPT 不再修改；项目完全独立完成，不编写个人贡献清单，两项均取消待办。
- **未执行（按用户决策）**：3 模型 × 3 seeds 正式重训、CE/Focal/CB-Focal 对照实验、
  dedup-v1 去重重训；比较协议（comparison-v2）待用户评审冻结。


---

## 第二轮重构独立审核（2026-10-07）

**历史基准 9c2f8ed：** 以下为当时的独立结论，当前修复/返修状态见末尾第三轮独立审核。

### 当前结论与核验边界

**结论：多数上一轮返修已完成，完整验收暂不通过。** 在提交 `9c2f8ed918e71fbe236243f30bbd0e79f75b2066` 上，独立复现 **3 项 P1 续训可靠性问题（S01–S03）和 2 项 P2 工具/材料问题（S04/S05）**。项目已有较完整的工程基础，但还不能承诺“默认配置精确续训”“所有质量检查通过”或“三模型正式比较完成”。先修复并验收这些问题，再冻结当前模型的训练比较方案。

所有 Python 均使用本项目 **Windows** `.venv/Scripts/python.exe`。新增探针使用真实 MiniCNN、固定小样本和系统临时目录；未修改实现/测试，未下载依赖，未进行正式 FER2013 重训、完整模型前向复评或 AMP 性能基准重跑。结构化证据见 `PROJECT_REVIEW_EVIDENCE.json` 的 `second_refactor_independent_audit`，历史审核字段保留。

| 独立核验 | 当前结果 |
|---|---|
| pytest | **112 passed**，11.87 s，无失败/跳过；1 条既有 Grad-CAM backward hook 提示；线程数 4、临时测试目录与关闭缓存 |
| Ruff | `python -m ruff check . --no-cache` 通过 |
| mypy | **失败，退出码 1**；33 个源文件中 1 个错误：`tests/test_training_pipeline.py:691 [no-any-return]` |
| Windows CUDA/依赖 | Python 3.10.11；torch 2.14.1+cu130 / torchvision 0.29.1+cu130；CUDA runtime 13.0；RTX 5070 Ti Laptop；实际 GPU 矩阵运算、pip check、Notebook 内核路径通过 |
| 诊断隔离 | CPU 回归通过；真实 GPU/AMP MiniCNN 诊断后模型/BN、优化器、scaler、父进程 RNG 与 run 文件不变，参数有限且训练产生有效更新 |
| 已保存评估 | **11 组**保存预测复算：accuracy/macro-F1/balanced accuracy 一致，ECE/NLL 差异 <1e-4，checkpoint SHA 均匹配；只验算产物，未重新执行全部样本前向 |
| 已导出权重 | **5 份**导出权重 SHA-256 与 manifest 匹配 |
| 参数量 | MiniCNN **1,274,823**；VGGLite **5,407,687**；MicroResNet（无 SE）**753,991** |
| 数据审计/日志 | 不加 `-X utf8` 的 Windows 日志重定向成功；CSV SHA、划分 28709/3589/3589、1,516 重复组、1,853 超出记录、57 冲突组及敏感性分析与既有披露一致 |
| 原产物保护 | 审核前后核对 CSV、历史日志/断点、导出权重/清单、真实 runs 共 **46 个文件**；逐文件 SHA-256 不变；临时探针/输出已清理 |

| 上轮项 | 第二轮验收结论 |
|---|---|
| R01 恢复的数据管线 | 单进程起始的回归通过；原多 worker 断点转单进程仍不精确，见 S01 |
| R02 协议约束 | 已有部分字段校验与失败保护；关键有效参数遗漏，见 S02 |
| R03 诊断隔离 | 本轮覆盖范围内通过；GPU/AMP 补充证据见上表 |
| R04 重复 fit | 正常完成/早停路径通过；部分中断后同实例继续未通过，见 S03 |
| R05/R06/R07 设备、增强开关、构造校验 | 上一轮返修回归通过 |
| R08 CE 与正式比较 | CE 代码前置通过；协议冻结、各模型有效预算落地、多 seed 正式结果未完成 |
| R09 材料/Notebook/日志 | 链接、注释、默认 split、日志编码、现场作图通过；来源校验未通过，见 S04；报告/PPT 与贡献清单不再要求 |

### S01 · P1 · 恢复时改为单进程，不能使原多 worker 断点精确恢复（R01/F09）

**位置：** [training/trainer.py](D:/Document/Unniversity/emotion_recognition/training/trainer.py:275)、[training/checkpoint.py](D:/Document/Unniversity/emotion_recognition/training/checkpoint.py:338)。

**实测与影响：** 当前只在恢复时自动将 `num_workers=4` 改为 0，加载检查也只看当前 loader，未检查断点产生时的数据管线。真实 MiniCNN、14 个固定 48×48 七分类样本、batch=4、加权采样、关闭增强、CPU/FP32、seed=42：比较“持久 4 workers 连续 2 轮”与“4 workers 完成 1 轮、重建为 0 workers 恢复 1 轮”。第一轮输入一致，**第二轮输入张量哈希及标签顺序不同，参数最大差值 0.01255920**；恢复事件仍为 `protocol_verified=True`。原持久迭代器与新迭代器对 RNG 的消费不同；记录 4→0 不等于恢复源状态，启用增强还涉及 worker 内 RNG。

**明确修复方案：** 最小可行方案是让精确模式**从首次训练就使用 workers=0**，将产生断点时的 loader/sampler/generator 规格写入协议；精确加载拒绝源 workers≠0 或源状态未知的断点。若保留从多 worker 恢复，需实现并保存足以复算采样/增强序列的独立 generator 和状态，完成对应验收。性能模式的旧断点可提供明确的非精确派生 run，并记录父断点 SHA 与原因；不能将其原地续写成“精确恢复”。CLI/Notebook 使用同一规则。

**验收标准：** 源 0→恢复 0，在普通/加权采样、增强开关组合下，连续与恢复逐批样本 occurrence、输入哈希、参数（`atol=1e-6, rtol=1e-5`）、history/LR/scaler/best/早停状态一致；源 4→0 在精确模式中于加载/写入原 run 前明确拒绝，或进入有新 run_id 的非精确派生路径。只有实现完整多 worker 状态后，才能在同样逐批比较通过时宣称该组合精确。测试必须覆盖**断点的源配置**，不能只检查恢复端 workers。

**状态：** 已修复（2026-10-07，完整方案）。实现独立 generator 机制（sampler / DataLoader 各一个，状态随断点保存/恢复）：批次索引序列与 worker base_seed 可复算，使 workers=0 与 **non-persistent 多 worker** 的“连续 vs 恢复”逐批一致（回归覆盖 workers{0,2} × 普通/增强/加权采样/组合共 8 组合；探针先行验证 persistent 不可复算的机理）。`persistent_workers=True` 的断点与恢复端均**明确拒绝**（含训练启动提示）。断点协议记录数据管线规格（workers/persistent/sampler/batch/generators），恢复端必须与断点一致（不一致即拒绝）；旧协议/缺 loader_rng 断点拒绝。端到端闭环：统一预算下 4 workers（non-persistent）短训练 → `--resume auto` 精确恢复至第 2 轮（未降级）→ 导出 → PublicTest 评估（run 20261007_041818_seed42）。

### S02 · P1 · 协议快照漏掉有效参数，配置变化仍被标为已验证（R02/F05/F09）

**位置：** [training/trainer.py](D:/Document/Unniversity/emotion_recognition/training/trainer.py:562)、[training/checkpoint.py](D:/Document/Unniversity/emotion_recognition/training/checkpoint.py:366)。

**实测与影响：** `training_protocol` 的增强只记录开关及 MixUp alpha，调度器只记录名字与总轮数，未完整覆盖图像/类别增强、裁剪、早停及调度器参数。在合法配置中单独变更下列 **6 项**，快照均完全相同、恢复均接受并记录 `protocol_verified=True`：

| 单项变化 | 独立探针观察 |
|---|---|
| `random_rotation: 10 → 25` | 实际 RandomRotation 变为 ±25° |
| 类别增强 `target_classes: [1] → [2]` | 实际类别增强目标变为 2 |
| `max_grad_norm: 0 → 0.25` | 运行时裁剪阈值变为 0.25 |
| `patience: 0 → 3` | 运行时早停耐心变为 3 |
| `val_loss_threshold: 1.05 → 1.2` | 配置变化未触发协议差异 |
| `scheduler_t0: 30 → 8` | 恢复接受；旧 scheduler state 把实际 T0 恢复为 30，新配置仍声明 8 |

前五项可能让同一 run 中途改变训练策略；最后一项使声明配置与实际调度器不一致。现有 loss/batch/数据指纹等字段的拒绝测试通过，不能覆盖这些遗漏。

**明确修复方案：** 从集中解析的**全部生效配置**生成版本化、规范化协议，模型构造、数据增强和 Trainer 共用解析结果；显式列出可变白名单（如追加轮数、显示/输出路径），其余影响训练状态的字段纳入比较。至少覆盖所有启用的增强数值/概率/目标类别、裁剪、双早停参数、scheduler T0/T_mult/step_size/gamma、实际 sampler/loader 规格及来源。加载状态和写恢复事件前完成校验；缺关键协议字段的旧断点只能走已注明未验证的路径，不得置为已验证。恢复后的有效配置、实际 scheduler 与记录必须一致。

**验收标准：** 上表六项分别作为独立负例，在修改权重、优化器或原 run 文件前拒绝；同配置正例仍可精确恢复。对完整字段表补充参数化变更检查，避免只增加六个特判；检查失败时原 run 哈希不变、无恢复事件。允许字段变化按白名单验收，加载后实际 scheduler 参数与记录逐项一致。

**状态：** 已修复（2026-10-07）。训练协议升级 v2：**完整生效配置的规范化快照**（仅剔除白名单：save_every / max_checkpoint_files / pin_memory / prefetch_factor）+ 运行时有效值（loader 规格、AMP/compile 实际状态、调度器实际参数、双早停参数、裁剪与累积等）+ 数据指纹；恢复前整字典比对（不再手工挑字段）。审计的 6 项遗漏全部纳入（random_rotation / target_classes / max_grad_norm / patience / val_loss_threshold / scheduler_t0）；**22 项参数化负例**（每项单独变更均在改动权重/原 run 前拒绝）+ 4 项白名单正例 + 「加载后实际调度器参数与断点记录逐项一致」校验；旧协议版本断点明确拒绝。

### S03 · P1 · 中断后同实例再次 fit 保留未完成轮的更新（R04/F09）

**位置：** [training/trainer.py](D:/Document/Unniversity/emotion_recognition/training/trainer.py:1181)、[training/trainer.py](D:/Document/Unniversity/emotion_recognition/training/trainer.py:1202)。

**实测与影响：** 正常 `fit(1); fit(1)` 已修复，但 `KeyboardInterrupt` 分支保存 partial 后，仅按已完成 history 推进轮号，没有回滚或禁止继续。小样本探针先完整训练 1 轮，再在第 2 轮的第 1 次真实 optimizer 更新后中断；partial 标记正确、history 仍为 1。直接再次调用同实例 `fit(1)` 被接受，最终 history/final_epoch 看似为 2，实际 optimizer 更新 **9 次**，连续 2 轮参考只有 **8 次**，参数最大差值 **0.05820417**。未记录为完整轮的更新、BN、优化器和 RNG 状态保留下来，轮号正确不能证明恢复正确。

**明确修复方案：** 为 Trainer 增加“部分轮更新后需恢复”的状态；中断/训练异常后禁止直接精确 `fit`，先加载完整 `last`，或自动回滚至该完整边界并记录。回滚必须覆盖模型/BN、optimizer、scheduler、scaler、RNG、采样、best/早停与 history；没有完整 last 时恢复初始化快照或明确要求重新构造。若提供非精确继续，建立派生 run 并注明部分更新，不能沿用完整轮等价承诺。轮号与累计时长修复继续保留。

**验收标准：** 覆盖首批更新后、若干批后、训练完成但验证前的中断，以及无完整 last 的情况；再次 fit 要么明确拒绝直到恢复，要么完整回滚后与连续训练的输入/参数/状态一致。本探针恢复后的 2 轮应为 **8 次**更新、history=2、完整 checkpoint 标志正确，不能只检查 `start_epoch=3`。相关异常退出也不得留下可冒充完整状态的可继续实例。

**状态：** 已修复（2026-10-07）。新增“部分轮更新”状态（中断/异常置位）：同实例直接再次 fit 会**先自动回滚至最近完整 last.pth**（覆盖模型/BN、优化器、调度器、scaler、RNG、loader 生成器、best/早停、history），回滚事件记录于 `run_meta.rollback_events`；无完整 last 时明确拒绝（要求重新构造）。回归覆盖：首批更新后/若干批后/训练完成但验证前中断（回滚后与连续训练的逐批输入、history、参数一致）及无 last 拒绝；异常退出同样置位，不留可冒充完整状态的可继续实例。

### S04 · P2 · 比较 Notebook 的显式路径没有验证来源一致性（R09/F19）

**位置：** [analysis/comparison_report.ipynb](D:/Document/Unniversity/emotion_recognition/analysis/comparison_report.ipynb:58)。

**实测与影响：** 源单元只校验文件存在，没有比较 run_id、权重 SHA、model_spec、协议与 history 元数据。仅执行该单元并将 MiniCNN history 换为真实 2 轮 smoke run `20261007_031251_seed42`，权重仍用旧 `mini_cnn_best.pth`，标签仍写“legacy（2026-05 历史产物）”，**未报错且成功载入**。因此“缺文件/来源混合时直接失败”的说明尚未成立。参数表还从当前 YAML 构建规格，未保证与比较权重的实际规格相同。现场重新画图和显式路径的改进已通过，但不自动证明曲线、参数、指标同源。

**明确修复方案：** 建立并校验比较清单，将每个模型的 checkpoint SHA、run_id、model_spec、history/config 来源和 protocol_id 绑定；同批正式比较需共同协议/数据划分，并注明结果用途（formal/smoke/legacy）。曲线由对应 run 的 history 获取，指标/参数量由同一 checkpoint 规格和权重得到；在绘图、评估输出前检查。旧日志若无法证明与旧权重同源，应明确标为来源未验证的历史展示，并与正式结果分开，不凭路径标签补造绑定。

**验收标准：** 上述“旧权重 + 新 smoke history”负例在生成图表/结果前失败，替换不同 run/规格/协议也失败；同一冻结协议下的正式 run 清单可正常执行，所有表/图能追溯对应 SHA 与 run。缺元数据的 legacy 不得进入已验证正式汇总，来源未验证状态清晰可见。

**状态：** 已修复（2026-10-07）。新增 `utils/comparison_check.py`：checkpoint 与 history 绑定为可校验来源清单——formal 路径要求同一 run（run_id / model_name / SHA / 数据指纹绑定）且整组共享同一 protocol_digest；legacy 缺元数据时整组标记「来源未验证」，禁止与 formal 混排。审计负例（旧权重 + 新 smoke history）在生成任何图表前失败；7 项测试覆盖（混用 / run 不匹配 / 模型层不一致 / 协议不一致 / 正例）。比较 Notebook 已接入：来源校验输出、显著警示、图表标题与表格全部标注【来源未验证】、参数量改从 checkpoint 实际规格读取；实际执行验证通过（legacy 组正确标注，指标与既有记录一致）。

### S05 · P2 · mypy 实测失败，文档的全绿结论不成立（F18）

**位置：** [tests/test_training_pipeline.py](D:/Document/Unniversity/emotion_recognition/tests/test_training_pipeline.py:691)。

**实测与影响：** 使用本项目 venv 单独执行 `python -m mypy . --cache-dir <系统临时目录>`，退出码 **1**：`Returning Any from function declared to return "bool" [no-any-return]`。`_state_equal(a, b) -> bool` 的末尾返回未标注参数的 `a == b`，被推断为 Any。pytest/Ruff 通过不会使类型检查自动通过；不能使用后续成功命令的退出码掩盖 mypy 失败。

**明确修复方案：** 给状态比较辅助函数合理的输入类型与明确的 bool 返回处理，保留 tensor/容器分支及比较语义；不要全局关闭规则或用宽泛忽略掩盖。执行记录分别保存每个检查的退出码，将“全绿”更新为实际结果。

**验收标准：** 单独 mypy 命令在当前整个配置范围返回 0、无错误，同时 112 项回归与 Ruff 仍通过；质量门槛逐项记录退出状态，任一失败则整组失败。

**状态：** 已修复（2026-10-07）。`_state_equal` 补全类型标注与显式 bool 返回；`mypy .`（全项目，排除临时目录）返回 **0 错误（35 个源文件）**；质量门槛改为**逐项独立执行并记录退出码**（见执行记录；任一失败则整组失败）。

### 下一步顺序与套磁展示口径

1. 优先补齐 S01/S02/S03 并加入能复现本轮反例的回归；通过后验证短训练、完整 last 恢复、导出及显式 PublicTest 评估的同源闭环。
2. 修复 S05 与 S04，确保质量命令独立通过，比较图表不会混用来源。
3. 冻结 comparison-v2，并把选定的共同预算应用到各模型/各臂的有效配置；当前主/基线 YAML 的模型级 LR/batch/epochs 仍保留旧值，不能直接把草案预算当作已生效设置。记录冻结日期/commit、seed 和最终测试边界后，再运行当前模型的正式比较。
4. 根据正式产物同步 README、Notebook 与应用说明，区分历史结果、smoke 验证和未完成实验；报告/PPT 不再修改，个人贡献清单不要求编写。

向赵凯老师展示时，当前可据实呈现统一模型加载、可复算评估、数据重叠披露、GPU/AMP 执行及通过反例改进工程可靠性的过程。精确续训、去重后的正式训练、各不平衡策略收益和三模型多 seed 比较仍需各自验收；本轮不添加新研究问题。

---

## 训练与推理性能专项审核（2026-10-07）

### 结论、测量范围与优先级

**训练应优先优化数据准备与启动开销，推理应优先优化默认热力图渲染和重复像素解析。** GPU 的计算/启动开销在数据管线提速后才更值得处理。当前不应直接套用“增加 worker、减小线程数、channels_last 一定更快”等结论；本轮实测包含无收益和退化的候选。

本轮统一使用本项目 Windows `.venv/Scripts/python.exe`：Python 3.10.11、torch 2.14.1+cu130、CUDA 13.0、RTX 5070 Ti Laptop，CPU 为 16 个物理核心/32 线程，PyTorch 默认 intra-op/inter-op 均为 16。未调整电源/驱动模式。延迟使用 `perf_counter`，GPU 测量区域前后同步；推理使用现有统一加载入口与实际 legacy 权重。数据准备工厂完整测量一次；训练只在临时 run 使用真实 FER2013 加权抽样的 **1,024 样本**调用 `train_one_epoch`，不运行正式 `fit`、不保存到真实 runs。数据管线每轮 **2,048 样本**；离线推理使用 **PublicTest 前 512 行**，不计算/选择新的准确率。候选只在临时探针中执行，没有修改实现或开启新配置。

训练计时包含增强/取批、传输、现有 Focal、AMP、裁剪、Adam 和训练统计，**不含验证、断点和模型初始化**；首轮与后续三轮分开。单张请求计时包含预处理、传输、前向和概率回传：GPU 预热 5 次、重复 50 次，CPU 线程探针预热 5 次、重复 30 次；带 Grad-CAM 预热 3 次、重复 20 次。图表仅测 Agg 布局/PNG 渲染代理（10 次），没有测浏览器、Streamlit 网络或中文字体布局。GPU 候选为驻留输入、10 次预热、20 次计时、3 组交替配对。Profiler 仅用于归因，带 profiler 的耗时不作为基准结果。

| 已测路径 | MiniCNN | VGGLite | MicroResNet |
|---|---:|---:|---:|
| 1,024 样本训练，workers=0 稳态中位耗时 | 609.40 ms | 690.86 ms | 642.45 ms |
| 同规模训练，workers=4 持久 worker 稳态中位耗时 | 162.55 ms | 194.49 ms | 173.74 ms |
| workers=4 的训练首轮 | 10.59 s | 10.50 s | 10.60 s |
| GPU 单张 `predict` 中位耗时 | 0.683 ms | 1.091 ms | 1.679 ms |
| GPU `predict` + Grad-CAM 中位耗时 | 3.319 ms | 5.048 ms | 7.698 ms |
| 当前离线 512 张（已读 CSV 的 DataFrame，batch=64） | 100.75 ms | 120.41 ms | 111.52 ms |
| 临时预解码/pinned/合并回传候选，同 512 张 | 4.84 ms | 24.37 ms | 14.19 ms |

上述短训练各 worker 条件下的增强 RNG 序列不保证相同，**只比较性能，不比较训练质量或恢复等价性**。离线候选则核对了同一图片的 FP32 概率与预测。启动时间含 Windows spawn、cuDNN 和 AMP 预热，不能将首轮外推到所有轮，也不能把稳态倍数宣称为全项目正式加速比。此前 S01–S03 的正确性条件继续有效；性能修复不能绕过它们。

### PB01 · 优先级 1 · 逐张 CPU 增强占据大部分数据准备时间

**位置与问题：** [data/dataloader.py](D:/Document/Unniversity/emotion_recognition/data/dataloader.py:98)、[training/trainer.py](D:/Document/Unniversity/emotion_recognition/training/trainer.py:867)。48×48 图像逐样本执行旋转、仿射、亮度/对比度、擦除及类别额外增强；加权采样提升目标类别出现频率，额外增强也随之更频繁。输入数组已经预解码，瓶颈并非每次 getitem 都重新解析 CSV。

**实测证据：** 不执行模型计算，2,048 张数据在 workers=0、开启增强时稳态中位耗时 **1,312.78 ms**；关闭增强仅 **13.96 ms**；4 个持久 worker 开启增强为 **262.73 ms**。关闭增强仅是归因对照，不能作为不改变训练策略的优化。真实 Trainer 的 workers=0 三轮取批等待中位分别约 **570.02 / 569.03 / 558.85 ms**，与上表训练总时间相近；workers=4 仍有等待。等待可与 GPU 执行重叠，不能把其比例当作严格 CPU/GPU 分解。

**可能的解决方案：** 保留现有增强参数、概率、顺序和类别条件，优先减少 Python 逐样本调度：实现批级张量增强/预处理，或在受控方案中使用 GPU 批级增强。若随机 API 改变，将新增强实现版本与 seed/样本 occurrence 绑定并纳入训练协议，不能称为原实现逐位等价；保留原路径用于对照。短任务优先避免重复启动 worker；正式性能模式保留受测的持久 worker，但先按 S01 明确恢复语义。精确模式从起始 workers=0，可通过批级计算提速，不能恢复时临时 4→0 冒称精确。workers/prefetch 根据冷启动、稳态、内存联合选取，不继续盲目增大。

**验收标准：** 同样样本数量、batch、增强策略、采样与 AMP/裁剪，至少 3 组交替配对测“训练+验证+必需保存”总时间；首批/首轮与稳态分开。建议提效目标为完整稳态流程中位耗时下降 **≥20%**、P95 不恶化超过 5%，并记录加载等待；这是待验收目标，不是当前已取得收益。验收总开关、各概率/范围/类别条件和 occurrence seed 的一致性；同一路径连续/恢复仍通过 S01–S03。若变更随机实现，记录新协议，不以删除增强或减少样本获得提速。

**状态：** 已实施（2026-10-07，第四轮；`augmentation.impl` 开关，legacy 路径保留对照）。

- 实现：`data/batch_augment.py`（版本 `batched-v1`，批种子规则 `sha256(train_seed|epoch|batch_idx)`，
  随训练协议快照记录）；**独立增强实现**：部分参数范围/门控沿用、像素输出分布不同
  （审核 T05 更正，双线性插值/单次重采样等差异见文末第五轮记录）——不称"同分布"、
  不称逐位等价。
- 正确性：13 项专项测试（确定性 / 参数范围 / 几何方向与组合对照（质心对齐 v1 源码公式，
  误差 < 0.1px）/ 类别门控 / 防御）；S01 恢复扩展：workers{0,2} 下“连续 vs 恢复”
  逐批增强输出哈希逐位一致；`impl=batched` 与逐样本 transform 并存时拒绝启动（防双重增强）。
- 性能（交替配对 ×3 × 2 场景；完整“训练+验证+保存”2 轮）：workers=0 中位 40.8→4.8s
  （**-88.4%**）；workers=4 non-persistent 中位 19.7→12.0s（**-39.0%**）；epoch2 取数等待
  合计 18.8→0.45s（w0）/ 7.4→3.1s（w4）；单批 128 张增强 80.0→2.1ms。
- 达标判定：稳态中位降幅 ≥20% ✓（两场景）；P95（max 代理）两场景均改善 ✓。

### PB02 · 优先级 1 · 每次启动重解 CSV，float32 数据增加 worker 启动与内存负担

**位置与问题：** [data/dataloader.py](D:/Document/Unniversity/emotion_recognition/data/dataloader.py:80)、[data/dataloader.py](D:/Document/Unniversity/emotion_recognition/data/dataloader.py:209)。每次构造重新读完整 CSV、用 Python split/数组逐行转换三划分，并立即构造暂时不用的 PrivateTest 数据。Windows spawn 会序列化 dataset；“仅保留 numpy”比持有 DataFrame 好，但当前训练像素数组仍有 **252.33 MiB**，不是很小的 worker 载荷。[PyTorch 的 Windows 数据加载说明](https://docs.pytorch.org/docs/2.14/data.html#platform-specific-behaviors)说明了 spawn 与 pickle 机制。

**实测证据：** 当前工厂总耗时 **8.639 s**：读 CSV **1.832 s**，训练像素解析 **5.256 s**，PublicTest/PrivateTest 各约 **0.666/0.659 s**，指纹 **0.193 s**；三划分 float32 像素共 **315.41 MiB**。全训练 dataset、4 workers、prefetch=6 的首次取批 **10.669 s**，4 个 worker 的 RSS 合计 **3,513.56 MiB**（约 3.43 GiB，包含解释器/库，RSS 可能重复计入共享页，不能等同独占物理内存）。工厂只测一次，不能宣称冷缓存分布；此处“首次”不是清空操作系统文件缓存的冷磁盘测试。

**可能的解决方案：** 从保留的原 CSV 生成可验证的紧凑 uint8 像素、标签/原行号与划分缓存，按需转换为 x/255 的 float32；缓存键包含 CSV SHA、预处理版本和划分协议。选择 mmap/共享只读数组或 PyTorch 共享 CPU tensor，减少 spawn 复制；PrivateTest 在明确最终评估入口才按需构建。采用 `np.fromstring` 等解析方式可先作为低成本候选，但不能跳过像素数量、值域与格式校验。缓存仅为可重建数据派生物，应放忽略目录，不替代原 CSV、不改原行号；失配或损坏应明确重建。

**验收标准：** 首次构建和有效缓存命中分别测量至少 3 次，并继续核验 CSV SHA；建议缓存命中工厂耗时相对当前下降 **≥50%**。所有 35,887 行的原行号、标签、Usage、uint8 像素与原 CSV 一致，转换后的无增强输入与旧路径逐位一致；三划分纯 uint8 像素约 **78.85 MiB**，验收该数组体积而非声称总内存自动减少四倍。源 CSV/协议变化及缓存损坏负例必须失效；另报父进程、worker 的 RSS/USS或私有字节、首批时延和稳态吞吐。共享数据只读，增强不得污染其他样本/epoch，恢复状态仍满足 S01/S02。

**状态：** 已实施（2026-10-07，第四轮）。

- 实现：`data/pixel_cache.py`（uint8 缓存；键 = 缓存版本 + CSV SHA-256；meta 原子提交、
  文件轮转防 Windows 锁、只读 mmap、进程内句柄缓存）；`create_dataloaders` 走缓存
  （训练入口 `include_test=False`，PrivateTest 按需；`data/cache/` 已入 .gitignore）。
- 一致性：与旧逐样本解析**逐位一致**（训练划分全量 28,709 行像素 + 标签 + 行号核验）；
  数据指纹公式与旧实现一致（新旧对照测试）。
- 性能：工厂加载 8.724→0.240s（独立进程 ×3：0.2415/0.2393/0.2396，**-97.3%**，目标 ≥50% ✓）；
  首次构建 4.66s（×3 中位，一次性成本）；进程内重复加载 0.009s；uint8 三划分合计 78.85 MiB。
- 负例：CSV 变化 / 缓存文件损坏 / meta 版本不符 / 缺缓存 → 均明确重建（测试覆盖）；
  14 项专项测试。

### PB03 · 优先级 2 · 小模型的 GPU 启动/同步开销明显，候选优化收益有限

**位置与问题：** [training/trainer.py](D:/Document/Unniversity/emotion_recognition/training/trainer.py:134)、[training/trainer.py](D:/Document/Unniversity/emotion_recognition/training/trainer.py:916)。当前 AMP 已开启，指标 `.item()` 已延迟到每 10 批/轮末，异步 H2D 与 `zero_grad(set_to_none=True)` 也已实现；不把这些已修复项再列为缺失。驻留输入的 4 步 profiler 中，普通 `cudaLaunchKernel` 加 `cudaLaunchKernelExC` 合计每步约 **139 / 206 / 323 次**；MicroResNet 参数最少但调用最多，参数量不能直接推导延迟。默认 AMP/Adam 路径每 4 步有 4 次 `cudaStreamSynchronize`；本机 GradScaler 默认 `_maybe_opt_step` 读取 CUDA `found_inf.item()`，其中存在必须尊重的溢出检查，不能直接删掉。

**候选实测（均不含加载/验证/保存）：**

| GPU 驻留训练候选，3 组中位数的中位数 | MiniCNN | VGGLite | MicroResNet |
|---|---:|---:|---:|
| NCHW，布局配对的默认 Adam | 7.21 ms/步 | 13.91 ms/步 | 8.64 ms/步 |
| channels_last，同组比较 | 32.18 ms/步 | 65.74 ms/步 | 32.71 ms/步 |
| 默认 Adam，另一次优化器配对 | 6.61 ms/步 | 14.99 ms/步 | 9.06 ms/步 |
| fused Adam，同优化器配对 | 6.14 ms/步 | 14.28 ms/步 | 8.66 ms/步 |

本机 **channels_last 明显退化**，不建议直接开启，具体退化原因尚未进一步定位；fused Adam 单步收益约 **5%–8%**，不是完整训练收益。fused profiler 中不再出现上述每步 stream 同步，但仍有多次 kernel 启动。两种优化器的有效更新数在配对内一致且参数有限，MiniCNN 30 次调用中的部分 AMP 更新被跳过（各组为 28/29 次）；短程结束的浮点 model state（含 BN buffer）存在差异，故**不认定数值/恢复等价**。官方 [Adam 说明](https://docs.pytorch.org/docs/2.14/generated/torch.optim.Adam.html)将 fused 作为可选实现；[性能指南](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html)中的候选也需本机验收，不能套用其一般性预期。

**可能的解决方案：** 先完成 PB01/PB02，再考虑可记录、可退回的 `optimizer.fused` 配置，保留当前默认实现；fused/foreach、内存格式、精度模式均进入 S02 的完整协议。比较相同有效 batch 与累积步数，不盲目增大 batch 改变优化预算。若 profiler 仍以启动开销为主，GPU 图捕获仅作为后续受测候选，先核对本机 Windows 后端支持和 GradScaler/尾批/断点适配；本轮未测试或安装编译后端，不默认启用 torch.compile。继续保留溢出处理和梯度裁剪；现阶段保留 NCHW。

**验收标准：** 固定同一梯度的 FP32 单步及优化器 state 比较在 `atol=1e-6, rtol=1e-5` 内；AMP 溢出/跳步、裁剪、尾批和累计步数正确。候选自身连续/恢复模型、optimizer、scaler、LR 与输入等价，不能继承旧未验收承诺。性能至少 3 组相同输入/初始化交替配对：单步与完整流程分别报告，建议只有完整流程中位耗时改善 **≥5%** 且 P95/显存可接受才作为默认项；否则保留可选或取消。随机训练不同数值路径若不逐位等价，需按冻结预算用 PublicTest 配对评估质量，不用 PrivateTest 调参。

**状态：** 已实施（2026-10-07，第四轮；`training.optimizer_fused`，仅 Adam + CUDA，CPU 明确报错）。

- 数值：固定同一梯度的 FP32 单步，参数与优化器 state 在 `atol=1e-6, rtol=1e-5` 内一致
  （CUDA 测试）；配置项随训练协议快照记录。
- 性能（正式配置 batched + workers=4，2 轮完整流程；两轮共 9 组交替配对）：
  中位改善 run1 **5.2%** / run2 **4.9%**（配对中位 ~4.9–5.2%）；峰值显存 -7.7%；max 不恶化。
- **判定（按报告判据）：未稳定达到 ≥5%——保留为可选项（默认 false），不默认启用；
  如实记录，未声称 ≥5% 收益。**

### PB04 · 优先级 1 · 默认 Grad-CAM 与图表渲染比分类前向更贵

**位置与问题：** [inference/app.py](D:/Document/Unniversity/emotion_recognition/inference/app.py:119)、[inference/app.py](D:/Document/Unniversity/emotion_recognition/inference/app.py:240)、[inference/infer_utils.py](D:/Document/Unniversity/emotion_recognition/inference/infer_utils.py:254)。模型资源已缓存，不能说每次都重新加载。热力图默认开启，普通 predict 做一次前向，Grad-CAM 又做目标判断前向和带梯度前向/反向；整条分类+热力图路径实测 **3 次前向**。输入不变时，Streamlit 重新运行仍执行预测、热力图和绘图，目前仅模型/权重信息有缓存。

**实测证据：** GPU 分类仅 **0.683/1.091/1.679 ms**，加 Grad-CAM 为 **3.319/5.048/7.698 ms**。三联图布局与 PNG 渲染代理中位 **71.87 ms**、P95 **73.54 ms**，比模型前向大一个数量级；该值不是完整 UI 响应时间，也不能简单与独立阶段中位数相加当作实测总请求。

**可能的解决方案：** 默认先显示分类结果，热力图改为按需计算；用已知预测类传给 Grad-CAM，并重构其目标校验逻辑以避免无条件再次做目标判断前向（只传 `target_class` 在当前代码里还不会省掉那次前向）。按图像内容 SHA、checkpoint SHA/model_spec、目标类与可视化版本缓存已计算结果；权重/图像/类变化必须失效，支持有限容量。缓存完成的概率/热力图/图像 bytes，不缓存 hook 或保留计算图；必要时降低绘图重建次数，仍保留辅助解释语义和异常安全。

**验收标准：** 默认分类请求只执行 **1 次前向、0 次反向、0 次 Grad-CAM 绘图**；首次按需热力图最多再执行 1 次带梯度前向和 1 次反向，重复同键请求不再计算/绘图；图像/权重/目标变化准确失效。模型参数/BN、train/eval 状态、hook 数与梯度清理保持原验收。单独记录首次模型加载、纯分类、首次热力图及缓存命中四条请求路径的完整 median/P95（含真实服务渲染），建议热力图缓存命中服务器耗时下降 **≥80%**；以真实 UI 测量确认，不把本轮代理值冒充浏览器延迟。

**状态：** 已实施（2026-10-07，第四轮；`inference/service.py` + 应用按需触发）。

- 实现：默认仅分类（1 前向 / 0 反向 / 0 绘图）；热力图按需生成（目标类 = 已知预测类，
  **不再有目标判断前向**：1 次带梯度前向 + 1 次反向）；概率/热力图/渲染 PNG 按
  (图像字节 SHA, 权重路径+签名, 目标类, 可视化版本) 键 LRU 缓存；不缓存 hook / 计算图。
  附：修复渲染中文字体 fallback（缺字（Glyph missing）警告清零，测试覆盖）。
- 验收（AppTest 驱动真实 app 代码路径，两轮）：默认路径计数 = 1 前向 / 0 反向 / 0 绘图 ✓；
  首次热力图 +1 带梯度前向 +1 反向 ✓；重复同键不重算不重绘 ✓；换图 / 换权重 / 换目标类失效 ✓。
- 耗时（服务端脚本执行，含真实渲染调用；两轮）：首次加载 0.25s / 纯分类 0.34s /
  首次热力图 0.36–0.39s / 缓存命中中位 0.016–0.029s——命中相对首次 **-92.6% ~ -95.5%**
  （目标 ≥80% ✓）；测量口径为服务端，不冒充浏览器端到端。

### PB05 · 优先级 1 · 离线推理重复解析像素并逐批阻塞回传结果

**位置与问题：** [utils/evaluation.py](D:/Document/Unniversity/emotion_recognition/utils/evaluation.py:87)。每次 `_predict_probs` 都将像素字符串 split/转换，再对普通 CPU tensor 逐批 `.to(device)`，每批 softmax 随即 `.cpu()`；外层每个 checkpoint/split 又重新读 CSV/计算指纹。当前方法清楚、可复核，但同一划分比较三模型时重复了公共工作。单张应用的优化与此批量评估路径不同。

**实测证据：** 在已读取 DataFrame 的 512 张 PublicTest 图片上，当前路径分别 **100.75/120.41/111.52 ms**。临时复用同一预解码数据、使用 pinned 输入并把概率合并为一次回传的候选为 **4.84/24.37/14.19 ms**，同批量 FP32 预测完全一致，最大概率差 **0**。该复合候选同时减少了解码、传输等待和回传同步，未分离各自贡献；数字**排除**候选缓存构建、CSV 读取、模型加载、指标计算和输出落盘，不能写成完整评估加速 21 倍等结论。

**可能的解决方案：** 将 PB02 的同源预解码划分在多模型评估间复用，加载一次并核验原行号/标签/指纹；GPU 用 DataLoader pin_memory 与 non_blocking，合并小型概率结果回传（或在固定内存上限内分组回传）。若需要真正重叠 H2D 与模型计算，另行实测双缓冲/独立 stream，不声称一个 `non_blocking=True` 就实现重叠；[PyTorch 传输指南](https://docs.pytorch.org/tutorials/intermediate/pinmem_nonblock.html)给出了对应条件。保持 FP32 原评估口径、固定权重和顺序；纯预测可试局部 inference_mode，Grad-CAM 保持独立梯度路径。

**验收标准：** 首先在完整 PublicTest 3,589 行与三份固定 SHA 的权重上比较原行号/标签/顺序/预测完全一致，概率 `atol=1e-6, rtol=1e-5`、所有指标差 <1e-4；不新增对 PrivateTest 的调参。至少 3 次分别测冷读/缓存构建、热复用、完整评估（含读取/模型加载/指标/保存）及每阶段时间，建议热复用预测阶段中位耗时下降 **≥30%**、完整评估下降 **≥10%**；所有结果保持 run/权重/协议绑定，缓存失配必失效。合并回传设有界内存，不为了速度改变概率精度、抽样或省略输出。

**状态：** 已实施（2026-10-07，第四轮；评估统一走像素缓存 + 快速推理路径）。

- 实现：`_predict_probs_fast`（uint8 mmap 分批 → float32/255 → pinned + non_blocking 传输；
  概率在 GPU 累积后**单次回传**，回传上限 = N × 类别数 × 4B，本项目全部三划分 ≤0.8 MiB）；
  `evaluate_checkpoint` 的数据协议校验由缓存加载/构建完成；旧解析路径保留用于对照。
- 一致性：fast vs legacy **逐位一致**（3 权重 × CPU/CUDA 共 6 组，maxdiff=0）；
  与历史评估记录对比（CPU、batch 64 同口径）：accuracy 差 0、预测 3,589 行全部相同、
  概率 maxdiff=0（3 权重）。
- 性能：完整评估 3.001→0.120s（**-96.0%**，目标 ≥10% ✓）；预测阶段 512 张
  111.9→6.9ms（**-93.9%**，目标 ≥30% ✓）；冷启动（含缓存重建）4.856s。

### 本轮排除的猜测与统一验收方法

- **CPU 线程不是统一瓶颈。** 单张 predict 的 1 线程→16 线程中位为 MiniCNN 1.588→0.641 ms、VGGLite 5.684→2.194 ms、MicroResNet 3.187→1.621 ms；VGGLite/MicroResNet 的 4 线程为 2.148/1.464 ms，表现接近或略优，但本轮为顺序测量、没有负载交替配对，不能认定通用最佳线程。保留模型/设备/线程实测选择，不能把 CPU 统一改为 1 线程。MicroResNet 的 CPU 与 GPU 单张请求很接近，演示不必仅凭 CUDA 可用就认定 GPU 更快。
- **inference_mode 只是次要候选。** GPU 裸前向 no_grad→inference_mode 为 0.514→0.509、0.923→0.883、1.674→1.653 ms；该顺序测量不足以证明稳定收益，远小于渲染/解码开销。若接入，仅限纯预测并保留 eval；[官方说明](https://docs.pytorch.org/docs/2.14/generated/torch.autograd.grad_mode.inference_mode.html)限制其中创建的 tensor 再进入 autograd，不能包住 Grad-CAM。
- **checkpoint 写盘目前不是第一优先级。** 三模型单份初始化后完整状态保存中位约 14–16/51–56/18–22 ms（临时目录、三次测量、无外部 I/O 压力）；当前 fit 的 epoch 显示时间在保存前截止，应补记保存/验证/总时间，但暂不引入复杂异步保存，也不取消必要的 last/原子写。正式全部数据的保存占比尚未实测。
- **不以更多 batch/减少增强换取“公平加速”。** 各模型保持自己的有效配置和固定比较预算；任何实现版本、随机策略、优化器内核/布局/精度变化写入协议，先通过 S01–S03，再测提效。现有正式 AMP 完整流程结论不因本轮短程结果改写。

统一性能记录至少包含 checkpoint/model_spec/配置/CSV SHA、实现 commit、设备/精度/线程、batch/累积、workers/prefetch、样本数、冷启动/预热、配对顺序、有效/跳过更新、median/P95、峰值显存与主/worker 内存。上列百分比是**建议的接入目标**，不是功能正确性的替代；没达到则如实记录并不默认启用。复跑无需重新下载依赖。本轮原 CSV、真实日志/断点/runs、已有评估与导出及报告/PPT 共 **89 个文件**逐一 SHA-256 不变；临时训练权重、探针与输出在系统临时目录完成后清理，长期更新当前审核文档、结构化证据与 README 范围/状态说明。

---

## 第三轮修复（S01–S05）执行记录（2026-10-07）

- **质量门槛（逐项独立退出码）**：pytest = **0**（152 项通过，66s）；ruff = **0**；
  mypy = **0**（`mypy .` 全项目 35 个源文件，排除临时目录）。
- **S01 端到端闭环**：run `20261007_041818_seed42`（统一预算：4 workers、non-persistent、
  LR=3e-4 / batch=128）——训练 1 epoch → `--resume auto` 精确恢复（**未降级 workers**、
  协议校验通过、恢复至第 2 轮，best val_acc 0.4416）→ 导出（manifest 记录 run/SHA-256）
  → PublicTest 统一评估（last.pth acc=0.4413；与训练值相差 3/3589 样本，为 AMP 与
  CPU-FP32 数值路径差异，如实记录）；run_meta 指纹（csv_sha256=3b8d…、
  splits 28709/3589/3589）与 resume_events 完整可核查。
- **比较预算已应用**：主/基线配置模型段统一为 LR=3e-4 / batch=128 / epochs=90
  （comparison-v2 §3.1；防漂移测试覆盖）；冻结日期/commit 登记待正式重训提交前完成。
- **比较 Notebook**：来源校验实际执行验证通过（legacy 组标记【来源未验证】；
  混用/不匹配/协议不一致组合在图表生成前失败——由 utils/comparison_check.py 与
  7 项测试覆盖）。
- **PB01–PB05（性能专项）**：待办（按用户决策，下一轮处理）；未执行、未声称收益。
- 说明：现有三个 smoke run（022921 / 031251 / 041818）为流程验证用途；协议 v2 之前
  保存的断点（无协议 / v1）按 S01/S02 规则**明确拒绝**精确恢复（不冒称精确）。

---

## 第四轮性能修复（PB01–PB05）执行记录（2026-10-07）

- **质量门槛（逐项独立退出码）**：pytest = **0**（**197 项通过**，79s）；ruff = **0**；
  mypy = **0**（全项目 **42 个源文件**）。
- **PB01（批级增强）**：`augmentation.impl="batched"`（batched-v1）；交替配对实测
  workers=0 稳态中位 40.8→4.8s（-88.4%）、workers=4 non-persistent 19.7→12.0s（-39.0%）；
  epoch2 取数等待 18.8→0.45s（w0）/ 7.4→3.1s（w4）；单批 128 张 80.0→2.1ms；
  逐批“连续 vs 恢复”增强输出哈希一致（S01 扩展）；几何对齐按 torchvision 0.29
  源码公式实现并经质心对照验证（组合误差 <0.1px）；已知差异（双线性插值、单次重采样、
  颜色随机顺序、批种子规则）已记录，未声称逐位等价。
- **PB02（像素缓存）**：工厂加载 8.724→0.240s（独立进程 3 次，-97.3%）；首次构建
  4.66s（3 次中位，一次性）；uint8 合计 78.85 MiB；训练划分全量 28,709 行逐位一致；
  负例（CSV 变化/文件损坏/meta 版本不符/缺缓存）均明确重建；测试
  `tests/test_pixel_cache.py`（14 项）。
- **PB03（fused Adam）**：数值单步 (atol=1e-6, rtol=1e-5) 通过（CUDA）；两轮共 9 组
  交替配对完整流程中位改善 5.2%/4.9%、显存 -7.7%——**未稳定 ≥5%：保留为可选项
  （默认 false），不默认启用**；测试 `tests/test_optimizer_fused.py`（3 项）。
- **PB04（Grad-CAM 按需+缓存）**：默认路径 1 前向/0 反向/0 绘图；命中相对首次耗时
  **-92.6% ~ -95.5%**（AppTest 服务端两轮，含真实渲染调用）；换图/权重/目标类失效；
  修复渲染中文字体 fallback（缺字警告清零）；测试 `tests/test_inference_service.py`
  （8 项）+ Grad-CAM 计数测试（test_inference 更新）。
- **PB05（评估复用）**：fast vs legacy 逐位一致（3 权重 × CPU/CUDA 共 6 组，
  maxdiff=0）+ 与历史评估记录一致（3 权重：acc 差 0、3,589 行预测全同、概率 maxdiff=0）；
  完整评估 3.001→0.120s（-96.0%）；预测阶段 512 张 111.9→6.9ms（-93.9%）；
  冷启动（含缓存重建）4.856s。
- **接口/行为变化**：训练入口不再构建 PrivateTest（`include_test=False`，评估入口按需）；
  新增 `augmentation.impl` 与 `training.optimizer_fused` 配置项（协议快照自动覆盖、
  变更被拒）；Streamlit 应用热力图改为按需（默认关闭）；渲染中文字体 fallback；
  train.py 与三个训练 Notebook 同步。
- **测量口径**：所有配对均为同机交替测量；PB04 为 AppTest 服务端脚本执行耗时（含真实
  渲染调用），不冒充浏览器端到端；PB01/PB03 为完整“训练+验证+保存”流程（2 轮、
  真实全量数据、mini_cnn、AMP、正式配置）。
- **清理**：`.dev_tmp/` 全部探针与临时输出（含配对 run 目录与评估输出）记录后删除；
  正式缓存 `data/cache/fer2013_uint8-v1/`（可重建派生物，已列入 .gitignore）保留；
  原 CSV（SHA 3b8d9617…）、历史权重/评估/报告/PPT 未改动。

## 第三轮重构与性能优化独立审核（2026-10-07）

**审核基准：** 提交 b75d8e34c1b5c8ff20ce050157d1f90f55d9a075，对比上一轮 9c2f8ed918e71fbe236243f30bbd0e79f75b2066。本节复核作者文中的“第三轮修复”和“第四轮性能修复”执行记录；前文保留为历史证据。

**结论：常规路径与性能优化已有明显进展，完整验收仍未通过。** 全部现有质量检查通过，上一轮具体反例的常规修复已复核；另复现 T01–T06 六项问题，其中 T01–T03 是 P1 续训问题。三模型统一预算已落地，正式协议仍未冻结，多 seed 正式训练及消融结果尚未形成。当前可以展示工程实现和历史权重的可追溯评估，不能把短训练、历史分数或速度提升当作新模型训练结论。

### 检查与旧问题复核

| 检查 | 本轮独立结果 |
|---|---|
| Windows Python / torch / torchvision | 3.10.11 / 2.14.1+cu130 / 0.29.1+cu130，CUDA 实际矩阵运算通过 |
| pytest | **197 passed，7 warnings，75.04 s** |
| Ruff / mypy / pip check | 全部通过；mypy 42 个源文件无错误 |
| S01 原反例 | workers=0/2、非持久 workers 的连续/恢复及增强/采样组合通过；改变 workers 或使用持久 workers 被拒绝。默认配置另见 T02 |
| S02 原反例 | 完整配置快照、运行时调度器参数、此前漏项拒绝和白名单回归通过；状态完整性另见 T03 |
| S03 原反例 | 同实例中断后回滚到完整 last 的回归通过；显式加载 partial 另见 T01 |
| S04 原反例 | legacy/new run 混排、run 不匹配、协议不一致的回归通过；正式实验准入另见 T06 |
| S05 | 原 mypy 错误已消除 |
| 预算配置 | 主/基线配置的三个模型均为 LR=3e-4、batch=128、上限 90 epochs |
| 受保护材料 | 107 个文件审核前后 SHA-256 一致，包含 CSV、缓存、原权重、runs、分析产物及两份 DOCX/PPTX |

197 是项目现有测试数量，以下临时反例和性能探针不算入该数字。所有 Python 操作使用项目 Windows .venv。本轮只修改审核文档、结构化证据、README 和协议草案的说明；未修改实现/测试，未正式重训，未导出权重，未修改报告或 PPT，不增设个人贡献清单。

### 性能复核：提升有效，但须限定测量口径

同一 RTX 5070 Ti Laptop GPU，CPU 4 线程，沿用现有缓存与固定权重。增强测试为 7 组；GPU 预测为完整 PublicTest 3,589 行、batch=64、预热后 3 组交替配对，表中为中位数。

| 路径 | 原实现 | 新实现 | 本轮结论 |
|---|---:|---:|---|
| CPU 128 张常规/类别增强 | 56.36 ms | 1.83 ms | 耗时 -96.8%；输出分布改变，见 T05 |
| MiniCNN GPU 预测阶段 | 0.6812 s | 0.03364 s | 耗时 -95.1% |
| VGGLite GPU 预测阶段 | 0.8307 s | 0.17415 s | 耗时 -79.0% |
| MicroResNet GPU 预测阶段 | 0.7412 s | 0.09276 s | 耗时 -87.5% |
| 已存在缓存的工厂调用 | 本轮未重跑旧工厂 | 首次 0.2279 s；同进程热调用中位 0.00399 s | 不含启动/遍历 workers，也不是首次缓存构建时间 |
| MiniCNN 服务层默认分类 | 首次 4.18 ms | 同键热命中中位 0.0057 ms | 1 次实算、7 次命中，Grad-CAM/绘图均 0 次 |

三份历史权重在 **CPU/CUDA 各自设备内**，新旧路径的完整 PublicTest 标签、预测和概率一致，6 组最大概率差均为 **0**；GPU 三次配对同样为 0。CPU 单次对照耗时分别为 1.724→1.121 s、6.330→5.711 s、2.490→1.846 s，仅作核对与趋势说明。CPU 与 CUDA 不要求逐位一致，例如 MiniCNN 本轮 accuracy 相差 1/3,589；这不影响同设备新旧实现一致。

**口径限制：** 预测阶段含旧路径解析、新路径归一化、传输、前向及概率回传；不含公共 CSV 读取、缓存构建、模型加载、指标计算或落盘。服务层分类不含浏览器、网络、模型加载或热力图渲染。作者记录的“完整训练 -88.4%/-39.0%”“完整评估 -96.0%”保留为其短程测量，本轮未重复整套流程，不能替换成长期训练保证。PB01 的“P95（max 代理）”只能作为观测最大值，不能视为真实 P95 验收。

| 性能项 | 当前状态与下一步 |
|---|---|
| PB01 批级增强 | 提速、断点复算回归通过；语义表述未通过（T05），完整流程和真实 P95 仍需记录 |
| PB02 uint8 缓存 | 命中速度、正常数据保真与轻量 worker 引用通过；热命中完整性有 T04，部分通过 |
| PB03 fused Adam | 可选实现及数值测试通过；完整流程无稳定 ≥5% 收益，默认关闭合理 |
| PB04 按需 Grad-CAM / 缓存 | 应用/服务回归通过，默认服务计数独立复核通过；浏览器端到端 P95 未测 |
| PB05 快速预测 | 六组数值一致、GPU 预测阶段提速通过；完整评估及缓存稳健性仍按原标准和 T04 验收 |

重复解码和逐张增强已不再是原来的主要开销。继续优化应优先测 **Windows 非持久 workers 每轮启动/取数等待**及实际 GPU 计算，不沿用旧瓶颈占比：T02 修复后，在 workers=0/2/4 下做相同增强实现、相同预算的至少 3 组完整“加载→训练→验证→保存”配对，分开记录首轮、后续轮、取数、增强、GPU step、验证/保存及峰值内存，以总耗时选择默认值。真实批次 P95 从至少 100 个预热后的批次样本计算；长训练收益另作观察。本轮不默认启用 fused Adam 或 channels_last。

### T01 · P1 · 显式加载 partial 断点绕过中断回滚（F09 / S03）

**位置：** [training/checkpoint.py](D:/Document/Unniversity/emotion_recognition/training/checkpoint.py:575)、[training/trainer.py](D:/Document/Unniversity/emotion_recognition/training/trainer.py:703)。

**问题与证据：** 同实例回滚已修好，但显式加载 interrupted 断点只警告，随后清除 _partial_state，新的 Trainer 不回滚未完成轮的更新。临时 MiniCNN（48×48、3 类、dropout=0、CPU、16 样本、batch=4、seed=71）训练一轮，再更新一批并中断；显式加载后训练一轮：history 为 2 轮，Adam step 却为 **9**，连续两轮为 **8**，最大参数差 **0.0206248**。恢复事件为 protocol_verified=true，无 rollback 事件。

**修复方案：** 修改真实状态或原文件前拒绝 partial=True 的精确续训，指向同 run 的完整 last；也可核验同源完整 last 后自动回滚。若提供近似恢复，须单独命名、明确非精确并另建 run，不能沿用两轮历史表达同一训练过程。

**验收标准：** CLI/API 显式加载“第一轮中断”“一轮后单批/多批中断”“训练结束但验证前中断”都覆盖。有完整 last 时恢复后逐批输入、模型/BN、优化器 step、调度器、scaler、history、RNG 与连续训练匹配；无 last 则拒绝。拒绝前后真实状态和产物 hash 不变，不能再出现 9 对 8 的更新次数。

**状态：已修复（2026-10-07，第五轮）。** 显式加载 partial 断点时自动回滚到同 run 完整 last
（同 run_id + 非 partial），回滚记录于 `resume_events.notes`；无完整 last / last 亦 partial /
last 不同 run → 明确拒绝，拒绝前后状态与文件 SHA 不变。API 与 CLI
（`train.py --resume <interrupted>`）端到端均验证：污染 partial 被丢弃、参数与连续训练匹配、
Adam step 不再出现 9 对 8。回归见 `tests/test_training_integrity.py`（三类中断场景 + 负例）
与第五轮执行记录。

### T02 · P1 · 默认配置产生无法精确续训的断点（F09 / S01）

**位置：** [configs/training_config.yaml](D:/Document/Unniversity/emotion_recognition/configs/training_config.yaml:17)、[configs/baseline_config.yaml](D:/Document/Unniversity/emotion_recognition/configs/baseline_config.yaml:28)、[training/checkpoint.py](D:/Document/Unniversity/emotion_recognition/training/checkpoint.py:528)。

**问题与证据：** 两份配置均为 workers=4、persistent_workers=true，加载逻辑正确拒绝这种来源的精确恢复，使默认训练→暂停→resume auto 不能闭环。作者成功的 20261007_041818_seed42 使用临时配置关闭了持久 workers，不能证明当前默认配置可续训。临时构造并保存此配置的断点后，加载被资格检查拒绝；未启动实际 workers。

**修复方案：** 正式主/基线配置统一关闭 persistent workers；优先 workers=0 建立可复现基线，再测 2/4 个非持久 workers。CLI/Notebook 使用同一设置，将有效 workers/persistence 写入冻结协议。持久模式可保留为明确不能精确续训的可选配置。

**验收标准：** 不编辑临时 YAML，用主配置及基线配置分别做隔离短训练→暂停→resume auto→再训练，均能恢复；默认配置测试断言不启用 persistent workers。实际选定 workers 的精确一致性测试覆盖增强与加权采样；至少 3 组总耗时/内存配对测量记录非持久 workers 的每轮启动成本。

**状态：已修复（2026-10-07，第五轮）。** 主/基线配置统一 `persistent_workers=false`、
`workers=0`（防漂移测试覆盖）；0/2/4 非持久配对测量（各 3 组完整流程）：w0 中位 4.76s vs
w2 11.03s / w4 11.79s（Windows 每轮 spawn 成本 ~2.8–3.0s，批级增强后数据准备已非瓶颈）——
按总耗时选定 w0。主配置与基线配置分别完成隔离短训练 → `--resume auto` → 再训练闭环
（均恢复成功；主配置 run 20261007_053942）。

### T03 · P1 · RNG 状态缺项仍被接受为完整恢复（F09 / S02）

**位置：** [training/checkpoint.py](D:/Document/Unniversity/emotion_recognition/training/checkpoint.py:271)、[training/checkpoint.py](D:/Document/Unniversity/emotion_recognition/training/checkpoint.py:562)。

**问题与证据：** 只要求 loader_rng 不为 None，内部跳过缺失键；全局 RNG 恢复失败也仅警告。临时带独立生成器与加权采样的完整断点，将 loader_rng 改为 {}，不改协议：加载成功，事件仍为 protocol_verified=true、notes=[]。采样器/loader 随机进度没有恢复，协议相同不能证明状态完整。

**修复方案：** 增加状态完整性预检，按运行时要求逐项核验 loader/sampler generator、全局 RNG、模型/优化器、调度器及 AMP scaler。先用临时生成器/可验证对象检查类型和可恢复性，再提交真实状态；缺项或恢复失败拒绝，不写成功事件。分别记录“协议一致”和“完整状态恢复成功”，异常不改写原 run。

**验收标准：** 分别删除必需生成器键、设为 None、提供非法 Tensor，删除/破坏全局 RNG，并覆盖调度器/AMP 必需状态；均在原状态和文件改变前拒绝。合法断点仍通过连续/恢复一致性；只对本来不存在的可选状态允许 None，不误拒绝无 scheduler、CPU 无 scaler 的训练。

**状态：已修复（2026-10-07，第五轮）。** 新增恢复前状态完整性预检
（`_verify_checkpoint_state`，只读 dry-run）：全局 RNG（python/numpy/torch/cuda 临时对象
set_state 验证）、loader/sampler 生成器（键完整性 + 类型 + 值级）、调度器与 AMP scaler
（副本加载 + 值级/记录对照——`load_state_dict` 不校验值类型，如 `T_0="bad"` 需另查）、
模型/优化器结构；缺项/非法值一律在修改任何真实状态前拒绝。恢复事件分项记录
`protocol_verified` 与 `state_integrity`；恢复失败不再仅警告。负例（删键/None/非法
Tensor/破坏 RNG/调度器/scaler）全部覆盖；无 scheduler、CPU 无 scaler 不误拒；修复过程中
发现的"GradScaler lazy 语义误拒合法 AMP 断点"已一并修正并回归。

### T04 · P2 · 缓存热命中漏校验，标签/行号仍可写（PB02 / PB05）

**位置：** [data/pixel_cache.py](D:/Document/Unniversity/emotion_recognition/data/pixel_cache.py:332)、[data/pixel_cache.py](D:/Document/Unniversity/emotion_recognition/data/pixel_cache.py:365)、[data/pixel_cache.py](D:/Document/Unniversity/emotion_recognition/data/pixel_cache.py:94)。

**问题与证据：** 损坏测试只覆盖第一次打开前。进程内缓存只比 x 文件名，热命中不检查三份文件新签名：临时缓存打开后将像素 95 改为 96，加载返回 96，文件实际 SHA 与元数据不同却未报错/重建。另 x 只读而 y/rows 可写；改为 99/9999 后新句柄读取到污染结果，指纹不变。uint8 直接解析 256 还会静默变为 0。反例仅操作临时 CSV/缓存。

**修复方案：** 三份数组均只读；缓存绑定来源/生成版本及三份文件名/stat 签名，命中时廉价检查，变更后关闭引用并重验 SHA/重建。Windows 文件锁采用新一代文件名与原子 meta 切换；进行中的 run 遇到数据源变化应拒绝继续。像素先以足够宽的类型验证数量、整数性及 0–255 值域再转 uint8，拒绝非法标签/Usage。

**验收标准：** 冷/热状态下修改、删除或损坏 x/y/rows、换来源均准确失效/拒绝；重建与源一致。三数组写入均失败，后续句柄/worker 不被污染；合法 0/255 逐位保真，256、负数、非整数/非法值明确失败。复测全部 28,709 Training 样本、指纹和轻量 pickle，至少 3 次复测热加载；不在逐样本路径做全量 SHA。

**状态：已修复（2026-10-07，第五轮）。** 缓存升级 `uint8-v2`：meta 绑定每文件 stat 签名
（size/mtime_ns），热命中与 worker 侧打开均做廉价 stat 快查（变化 → 完整 SHA/重建；
训练中的 worker 明确拒绝继续）；x/y/rows 三数组全部只读；像素解析先以 float32 校验
（数量/整数性/0–255 值域）再转 uint8（256/负数/非整数明确失败，不再静默截断），标签
校验 0–6。复测：三划分全量 28,709+3,589+3,589 行逐位一致、指纹一致、dataset pickle
748 B、热加载 ×3 ≈0.0003s、独立进程冷加载 ×3 ≈0.219s（v2 构建一次性 ~10s，变慢来自
逐行合法性校验；热路径不在逐样本做全量 SHA）。测试 24 项。

### T05 · P2 · 批级增强不能称为与 legacy“同分布”（PB01 / F07）

**位置：** [data/batch_augment.py](D:/Document/Unniversity/emotion_recognition/data/batch_augment.py:5)、[docs/comparison_protocol_draft.md](D:/Document/Unniversity/emotion_recognition/docs/comparison_protocol_draft.md:55)、README 性能说明。

**问题与证据：** 不只改变了种子：nearest 变为双线性，旋转/平移从两次重采样变为一次。独立反例使用 16 张二值图、只开启旋转：legacy 输出 (0,1) 小数像素 **0 个**，batched 输出 **31,614 个**。输出分布已不同，“同参数范围/同门控概率”不能证明输出同分布。

**修复方案：** 本轮已纠正文档为“独立增强实现，部分参数范围/门控沿用，像素输出分布不同”，保留版本和种子规则；源码注释/YAML 说明留待实现修复轮同步。继续用 batched-v1 时冻结插值、重采样、平移离散化、颜色顺序/clamp 和擦除规则，所有比较模型/臂使用同一实现重训。若宣称语义等价，须先对齐这些规则，不能仅用质心误差或范围测试证明分布一致。

**验收标准：** 全仓库不再无证据宣称输出同分布，二值反例作为行为回归；协议、代码和 run 元数据一致，切换实现版本拒绝精确续训。准确率/F1 结论由相同冻结增强实现的多 seed 实验支持，历史分数独立标注。完整流程提速及真实 P95 按 PB01 原标准另行验收。

**状态：已修复（2026-10-07，第五轮）。** 全仓库停止"同分布"表述：源码 docstring、
配置注释、README/协议统一为"独立增强实现（部分参数范围/门控沿用，像素输出分布不同）"；
batched-v1 冻结规则（插值/重采样/平移离散化/颜色顺序与 clamp/擦除/类别条件/种子规则）
写入源码注释（规则变更必须升级版本号）；二值反例固化为回归测试（legacy nearest 输出
纯 0/1、batched 双线性产生小数像素）；实现版本变化 → 协议比对拒绝精确续训（测试覆盖）。
准确率/F1 结论仍待同一冻结实现的多 seed 实验（正式训练前）。

### T06 · P2 · 来源绑定成功不等于正式实验准入（S04 / F11 / F19）

**位置：** [utils/comparison_check.py](D:/Document/Unniversity/emotion_recognition/utils/comparison_check.py:176)、[analysis/comparison_report.ipynb](D:/Document/Unniversity/emotion_recognition/analysis/comparison_report.ipynb:86)。

**问题与证据：** formal 仅表达路径/run/协议绑定，不查正式用途、冻结记录或完成原因，run_meta 缺失也不强制拒绝。对当前仅 2 轮、README 明确标为流程验证的 20261007_041818_seed42，返回 **formal**。Notebook 对这种结果去掉来源警告且展示 formal，容易把短训练纳入正式成绩。该 run 产生于本次性能改动之前，也不证明当前 batched-v1 的正式效果。

**修复方案：** 分离 run-bound 来源状态和 formal_eligible 实验准入。run 元数据记录正式/流程验证/诊断用途及冻结协议 ID/hash；正式入口要求有效 run_meta、冻结协议、配置/数据/run 绑定、完整断点和正常结束/明确早停。现有短 run 按核查记录标为 smoke，保持非正式。

**验收标准：** 当前三个短 run 均显示“流程验证、非正式”，正式汇总前拒绝或排除；未冻结、缺元数据、diagnose/smoke、partial 等负例同样处理。来源完整但未获准的 run 可独立查看。正式合法 run 按上限或已登记早停完成，不机械要求恰好 90 轮；模型/臂/seed 聚合均追溯到冻结配置、history、checkpoint 与指标。

**状态：已修复（2026-10-07，第五轮）。** 分离 run-bound 来源状态与 formal_eligible
实验准入：validate 输出 `verdict=run-bound`（不再用"formal"字样）；新增
`check_formal_eligibility`（formal 用途声明 + 冻结协议文件绑定一致（id + 文件 SHA-256）
+ 正常结束 + 完整断点 + 数据指纹）；`train.py` 增加 `--purpose smoke|formal`（默认
smoke；formal 要求 `docs/comparison_protocol_frozen.json` 存在并写入 run_meta）。验证：
当前 6 个真实 run 全部判定"非正式"；notebook 执行验证通过（来源 + 准入双显示与警示）；
7 项准入判定矩阵测试 + 真实 run 回归。

### 下一轮修复与验收顺序

先完成 T01–T03，确保默认训练和中断恢复可靠；同步补 T04。T05 文档更正本轮已完成，下一轮同步源码说明并冻结实际行为；T06 分离来源与正式准入。保持全部现有检查通过，新增回归针对独立反例。

随后按已有协议冻结方案、完成基线/消融和三模型多 seed 实验。不新增模型或研究题目，不重做报告/PPT，不要求个人贡献清单；本轮不启动正式训练。

---

## 第五轮修复（T01–T06）执行记录（2026-10-07）

- **质量门槛（逐项独立退出码）**：pytest = **0**（**233 项通过**，81s）；ruff = **0**；
  mypy = **0**（全项目 **43 个源文件**）。
- **T01（partial 显式加载）**：显式加载 partial 断点 → 自动回滚到同 run 完整 last
  （记录于 `resume_events.notes`）；无完整 last / last 亦 partial / 不同 run → 拒绝，
  且拒绝前后真实状态与文件 SHA 不变。回归 16 项（`tests/test_training_integrity.py`，
  含三类中断场景）；CLI e2e：真配置 run → 构造含额外更新的 partial →
  `train.py --resume <interrupted> --epochs 1` → 日志"已自动回滚到同 run 完整断点
  last.pth"、训练至第 3 轮、EXIT=0（不再出现 9 对 8 的更新计数）。
- **T02（默认配置）**：主/基线配置统一 `persistent_workers=false`、`workers=0`
  （防漂移测试覆盖）。0/2/4 非持久配对测量（各 3 组完整流程）：w0 中位 4.76s vs
  w2 11.03s / w4 11.79s；每轮 spawn 成本 w2/w4 ~2.8–3.0s（e2 首批等待），w0 3–5ms——
  按总耗时选定 w0。主配置与基线配置分别完成隔离短训练 → `--resume auto` → 再训练
  闭环（主配置 run 20261007_053942；过程中 resume auto 误选基线最新 run 时被协议比对
  正确拒绝——跨配置保护顺带验证）。
- **T03（状态完整性预检）**：新增 `_verify_checkpoint_state`（只读 dry-run：全局 RNG
  （python/numpy/torch/cuda 临时对象验证）、loader/sampler 生成器（键/类型/值级）、
  调度器（副本加载 + 与断点记录的实际参数对照——`load_state_dict` 不校验值类型）、
  AMP scaler（副本加载 + 值级）、模型/优化器结构）；缺项/非法值一律在修改任何真实
  状态前拒绝。恢复事件分项记录 `protocol_verified` 与 `state_integrity`。修复过程中
  发现并修正"GradScaler lazy 语义导致合法 AMP 断点被误拒"（改为校验来源 state 值，
  测试回归覆盖）。
- **T04（缓存完整性）**：缓存升级 `uint8-v2`——meta 绑定每文件 stat 签名，热命中与
  worker 侧打开均 stat 快查、变化即 SHA/重建（训练中的 worker 拒绝继续）；x/y/rows
  全部只读；像素解析先校验（数量/整数性/0–255）再转 uint8（256/负数/非整数明确失败）；
  标签校验 0–6。复测：三划分全量逐位一致、指纹一致、pickle 748B、热加载 ×3 ≈0.0003s、
  独立进程冷加载 ×3 ≈0.219s；v2 构建一次性 ~10.2s（逐行校验成本，热路径不受影响）。
  测试 24 项；旧 v1 缓存目录（孤儿）已删除。
- **T05（增强表述/冻结）**：全仓库停止"同分布"表述（源码/配置/README/协议统一为
  "独立增强实现：部分参数范围/门控沿用，像素输出分布不同"）；batched-v1 冻结规则
  （插值/重采样/平移离散化/颜色顺序与 clamp/擦除/类别条件/种子规则）写入源码注释；
  二值反例固化为回归（legacy nearest → 纯 0/1；batched 双线性 → 大量小数像素）；
  实现版本变化 → 协议比对拒绝精确续训（测试覆盖）。
- **T06（准入分离）**：validate 输出 `verdict=run-bound`（来源绑定）；新增
  `check_formal_eligibility`（formal 用途 + 冻结协议绑定一致 + 正常结束 + 完整断点 +
  数据指纹）；`train.py --purpose smoke|formal`（默认 smoke；formal 要求
  `docs/comparison_protocol_frozen.json` 存在并绑定入 run_meta）；冻结文件格式入协议
  §6。验证：6 个真实 run 全部判定"非正式"；notebook 执行验证通过（来源 + 准入双显示
  与警示）；准入判定矩阵 7 项 + 真实 run 回归测试。
- **行为/接口变化**：默认 `num_workers=0`（含配置注释）；`--purpose` 新 CLI 参数；
  显式 `--resume <partial>` 自动回滚语义；verdict 更名 run-bound；缓存 v2。
- **清理**：`.dev_tmp` 探针与临时输出删除；`data/cache/fer2013_uint8-v1` 删除；
  原 CSV（SHA 3b8d9617…）、历史权重/评估/报告未改动。
- **边界（如实记录）**：T05 的准确率/F1 结论仍待同一冻结实现的多 seed 正式实验；
  T06 冻结文件尚未创建（正式训练前执行冻结流程）；T02 配对为 2 轮短程完整流程。

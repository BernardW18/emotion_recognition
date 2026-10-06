# 当前项目评估与修复验收清单

更新日期：2026-10-07。项目：`D:\Document\Unniversity\emotion_recognition`。用途：修复并整理现有 FER2013 项目，作为向赵凯老师展示个人能力的材料。证据见 [PROJECT_REVIEW_EVIDENCE.json](PROJECT_REVIEW_EVIDENCE.json)。

## 评估范围与结论

本轮只处理当前项目的正确性、可复现性、训练可靠性和材料表达。保留 MiniCNN、VGGLite、MicroResNet 三个现有模型以及 FER2013 数据；不提出新研究问题，不新增架构、分类头改造、其他数据集或论文迁移任务。对现有损失、采样、增强和 AMP 的检查，是为了确认项目已有功能及结论是否成立。

第一轮重构已建立统一模型规格、独立 run、显式评估和导出入口，修正了 MicroResNet 的激活加载、参数量与早停基础逻辑。本次独立审核确认 75 项测试、Ruff、mypy、依赖检查通过，Windows CUDA 可用；保存的评估与导出结果可复核。但默认持久 worker 下的精确续训、续训配置/数据约束、诊断隔离、重复 fit 与配置一致性仍有可复现问题，尚不满足“全部正确性问题已关闭”的结论。完整结果见末尾“第一轮重构独立审核”。

本文中的“验收标准”是各项修复需要满足的条件。**2026-10-07 独立审核，提交 `6d02afd117183b4ded3896a32ddbf07d0c8fa7cc`**：F05/F09/F13 部分通过并重新打开，F07/F11 的正式比较尚未执行，F07 的 CE 基线前置也未完成，F19 部分完成。**不启动正式重训、不修改模型或训练实现**；本次只更新审核文档与结构化证据。P0 表示正式重训前需要完成，P1 表示相关功能或实验结论使用前需要完成，P2 表示演示和对外材料提交前需要完成；不能把测试数量或短训练闭环代替各项验收。

## Windows CUDA 环境：已安装并验收

本节保留上一轮环境与三模型最小更新的核验记录；本次独立审核已复核环境并运行 75 项测试，新增结果见末尾与证据文件的新字段。历史 36 项测试和四批次 GPU 更新结果不冒充本次正式训练。

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

**问题及位置：** `README.md`、三个模型文件的说明、`inference/app.py` 的模型信息、课程报告及 PPT 的参数量不正确。MiniCNN 的 `Linear(4608,256)` 已超过百万参数；VGGLite 分类头也不是“数万个参数”。参数量与速度、显存之间的关系没有实测支持。

**修复方案：** 从当前模型实例自动计算总参数及可训练参数，生成统一结果表，同步所有材料。效率表保留现有模型，分别记录参数量、如需报告的 MACs/FLOPs、batch=1 延迟、训练峰值显存；注明输入、硬件、精度、预热和计数口径。删除“每千参数贡献准确率”作为优劣判定依据，不做分类头压缩。

**验收标准：**

- 七分类、当前无 SE 配置的参数计数准确等于 1,274,823 / 5,407,687 / 753,991；启用已有 SE 选项时重新计数并标明配置。
- README、应用、模型说明、报告和 PPT 不再出现旧数量；每个效率数值能追溯到运行配置。
- 若报告 FLOPs，明确 MAC 与 FLOP 的换算；延迟至少记录预热、重复次数、median/P95，GPU 计时同步，并把预处理与模型前向耗时分开。未测的数据标为“未测”，不能据参数量推断速度。

**状态：** 已修复（2026-10-07）。参数量实测修正为 1,274,823 / 5,407,687 / 753,991（SE 版 764,663），同步到模型 docstring、README 与应用动态显示；新增 tools/benchmark_efficiency.py 实测 MACs（batch=1）、GPU/CPU 延迟（median/P95，预热 10 + 重复 100，同步计时，预处理与前向分开）、训练峰值显存，输出 analysis/efficiency_results.json；删除「每千参数贡献」表述。报告/PPT 本轮未同步（见 F19）。

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

**状态：** 部分通过，独立审核重新打开（R02）：新建 run 的隔离测试通过，但不同配置/数据仍可续写原 run，恢复来源记录不完整。

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

**状态：** 待补齐前置与正式实验（R08）：尚无可由 CLI/YAML 启动的普通 CE 基线，比较预算未冻结；不能标记为前置已全部就绪。

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

**状态：** 部分通过，独立审核重新打开（R01/R03/R04）：单进程恢复回归通过，持久 worker 恢复不等价、诊断会修改正式状态，同实例重复 fit 的轮号/断点错误。

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

**状态：** 正式比较未执行，前置仍需补齐（R08，同 F07）。统一评估和协议草案已存在；实际训练预算尚未冻结，应先补修 R01–R07、确定各模型有效配置与执行轮数，再开展 3 模型 × 3 seeds 比较。

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

**状态：** 部分通过，独立审核重新打开（R06/R07）：CLI 校验和 workers=0 兼容已通过；MixUp 不受总开关控制，Notebook/API 绕过校验，NaN 数值未拒绝。

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

- 应用、README 和报告没有把 0.9 等同于“90% 会预测正确”的承诺；未校准状态明确。
- ECE/NLL 报告能够从保存概率与标签重算，概率有限、归一化正确，注明 15 区间及权重/划分。
- 不以更改术语冒充已完成校准；未实施校准时，所有对外材料均保留“未校准”说明。

**状态：** 已修复（2026-10-07）。界面与文档改为「模型概率（未校准）」并说明语义；ECE（15 等宽区间）/NLL 由统一评估的概率输出重算（PrivateTest：0.1123/0.2144/0.0645；NLL 1.0381/1.0538/0.9025），公式与口径记录于评估结果 JSON，可用保存预测复算。

### F17 · P2 · Grad-CAM 热图缺少行为验收，解释过强

**问题及位置：** 现有热图通过了单次形状和值域检查，但不能因此认定模型关注区域正确或具有因果意义。

**修复方案：** 保留当前 Grad-CAM 功能，说明其为目标类别的梯度响应辅助图。使用统一预处理和指定目标类别，确保 eval 状态、梯度开启范围和 hook 清理正确，避免多次调用积累 hook 或修改模型参数。检查固定小例子和重复调用行为，不新增解释方法研究。

**验收标准：**

- 三模型固定输入输出形状正确，数值有限，归一化结果在 [0,1]，零响应时有明确定义；目标类别超出 0–6 时报错。
- 连续调用至少 10 次，hook 数量不增加，权重哈希与普通推理结果不被热图调用改变；模型的训练/eval 状态按约定恢复。
- 界面和报告称辅助可视化，不把一张热图作为模型因果机制或心理解释的证据。

**状态：** 已修复（2026-10-07）。Grad-CAM：hook 在 finally 中清理（异常安全）、调用后零梯度与训练状态恢复、目标类别越界报错、零响应定义为全 0；10 次连续调用 hook 数不变、参数不变。证据：tests/test_inference.py；界面表述为辅助可视化、不宣称因果解释。

### F18 · P1 · 质量工具与 CI 状态不能支撑质量承诺

**问题及位置：** 此前 36 项测试通过，本轮在 CUDA 包环境中复跑也通过，但多为基础行为；Ruff 此前发现 142 项问题，mypy 被 `trainer` / `training.trainer` 重复模块映射阻断，配置中的 Python 3.9 与当前工具不兼容。工作区 CI 文件此前已删除，README 仍有 badge，工具列出不代表执行通过。

**修复方案：** 先统一包边界/导入方式与 Python 3.10 配置，使 mypy 真正运行；逐项处理 Ruff 发现，必要忽略需有局部理由，不能全面关闭规则掩盖问题。为 F01/F05/F08/F09/F10/F13/F14 加入关键回归检查。CI 文件删除若为既有意图，就移除失效 badge；如恢复 CI，按真实执行结果展示状态。测试输出隔离在临时目录，不能污染正式 run。

**验收标准：**

- 现有测试和上述关键回归检查全部通过；Ruff 和 mypy 返回成功，mypy 无模块映射阻断。测试数量与实际执行结果同步。
- README badge 指向真实存在的工作流和仓库；若没有 CI，材料不宣称 CI 已通过。普通 CPU CI 不宣称已验证 CUDA 训练。
- 测试前后原数据、正式日志和导出权重哈希不变；允许用临时小数据测试正确性，不把基础测试当作完整重训复现。

**状态：** 现有质量检查通过，覆盖仍须补齐：独立复跑 75 项测试、Ruff、mypy 全部通过；R01–R07 的实际入口/状态边界尚无对应有效回归，不能据测试数量宣布全部正确。

**修复轮原记录（历史，当前状态以上述独立复核为准）：** 已修复（2026-10-07）。补 data/training/inference 的 __init__.py 统一包边界后 mypy 真正运行（46 → 0 错误）；ruff 全绿（少量 per-file 忽略均有局部理由）；测试 75 项全部通过（核心 36 + 训练管线 18 + 推理 8 + 应用 4 + 配置校验 9）；README 移除失效 CI badge（保持无 CI，按用户决策），仓库链接统一；测试输出隔离于 tmp_path，不污染正式 runs。

### F19 · P2 · 对外材料有不可比或无证据表述，个人贡献不明确

**问题及位置：** README、Notebook、课程 DOCX/PPT、应用存在 val/test 混用、旧参数量、不同仓库链接等问题。课程报告写“小组”，个人负责部分不明确。2021 年论文的 73.28% 不能叫当前 SOTA，也不能与 val_acc 直接相减；“灰度减少整体计算约三分之二”“噪声推得准确率硬上限 90–95%”无充分支持。

**修复方案：** 从 F10 的统一结果记录同步所有表格，指标明确 split、run、配置与权重。统一实际仓库链接；历史论文结果只在协议可比时作为有日期的参考，删除没有测量或推导的结论。列出本人负责的数据处理、实现、实验、分析与组员部分，说明使用方法来源。申请材料围绕当前项目及已完成修复，不添加尚未执行的新研究方案，不把待办写成成果。

**验收标准：**

- 参数、指标和图表在 README/Notebook/应用/报告/PPT 中一致，accuracy 附 split，能追溯至同一结果文件；遗留历史结果有清楚日期与来源。
- 删除或严格限定上述不支持表述，不混用 PublicTest 与 PrivateTest，不宣称普通 CNN/残差/SE/Focal 是个人原创。
- 有一份具体的个人贡献清单和可核查工作记录；一页摘要及短演示仅包含已完成成果、限制和当前修复状态。进入课题组不以单一分数作为保证。

**状态：** 部分完成（2026-10-07，按用户决策调整范围）。README 全面重写：参数量/指标口径与 split 统一、数据重叠披露、未校准说明、安装与 runs/导出流程、无 CI 表述；模型 docstring 与 data/README 同步。课程报告 DOCX / PPT 本轮不修改（用户决策）；个人贡献清单未单独成文。

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
| 5 · 材料提交 | F15–F19 | F15–F18 已修复；F19 代码侧材料完成，报告/PPT 未同步（用户决策） |

正式重训前至少完成全部 P0 的适用条件。F03 的官方结果披露与附加去重结果分别验收，不能把“计划去重”写成“已消除泄漏”。F14 默认 K=1 不阻止这一配置重训，但启用 K>1 前必须通过尾组检查。GPU 已装好也不能替代模型加载、run 隔离或续训修复。

建议先选一个现有模型做修复后的短训练，检查一次保存、恢复、导出和统一评估，再启动完整比较。短训练通过只代表流程可用，不把其分数作为正式结论。正式训练前保存协议和配置，之后不根据 PrivateTest 反复挑选方案。

## 本轮交付及证据边界

此前环境核验交付了本评估文档、证据摘要与 Windows CUDA 环境快照；第一轮重构随后补充了代码、工具与 README。本次独立审核更新同一文档及证据文件，不修改模型、训练代码、正式数据和权重。正式重训、剩余正确性修复及课程报告/PPT/个人贡献材料仍待完成；继续限定在现有三个模型与 FER2013 项目内。

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
  课程报告 DOCX / PPT 同步；CI 恢复（保持无 CI）。

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

**状态：** 已修复（2026-10-07）。明确「精确恢复」实测支持范围 = `num_workers=0`：恢复时自动将数据加载调整为单进程并记录于 `run_meta.resume_events`（`resume_conditions` 含原值/调整值与范围说明）；`load_checkpoint` 对多进程 loader 明确拒绝。回归测试升级为逐批「增强后输入张量哈希 + 标签序列」比对，覆盖 普通/增强/加权采样/二者组合 四种组合（连续 vs 恢复逐批一致）；真实 CLI 闭环复核（run 20261007_031251_seed42：workers 4→0 强制并记录、恢复至第 2 轮）。多进程数据管线下的恢复一致性明确不作承诺。

### R02 · P1 · 更换训练配置或数据仍可续写原 run（F05/F09 重新打开）

**位置：** [training/checkpoint.py](D:/Document/Unniversity/emotion_recognition/training/checkpoint.py:324)、[training/trainer.py](D:/Document/Unniversity/emotion_recognition/training/trainer.py:449)。

**实测与影响：** 加载只比较 `model_spec`。本次将 Focal 换为 CB-Focal，同时改变数据张量和数据指纹，原断点仍成功加载。实际 criterion 是 `CBFocalLoss`、数据指纹变为 `b…b`，同一 run 的 `config_effective.yaml` 仍写 `loss_type: focal`、`run_meta.data.csv_sha256` 仍是 `a…a`。日志可追溯性失效；更换 monitor、batch、增强、AMP 或调度器也未受同等约束。

**修复方案：** 在 checkpoint 保存训练协议/有效配置与数据指纹，在恢复前比对损失、采样、增强、batch/累积、优化器/调度器、monitor、精度模式及 CSV/划分指纹。允许变更的字段使用明确白名单（如本次追加轮数）；改变训练策略应建立新 run 并记录父断点，不能继续称为原 run 的精确恢复。校验通过后再写恢复事件，每次事件记录当次 CLI/config/git/environment；失败不得先改写原 run 元数据。

**验收标准：** 同配置/数据可恢复；逐项修改 loss、sampler、monitor、batch、增强、调度器或 CSV 字节/划分时，在更新权重和原 run 文件前报错。若提供显式派生运行，必须有新 run_id、实际生效配置与 parent checkpoint SHA；旧 run 所有文件哈希保持不变。

**状态：** 已修复（2026-10-07）。checkpoint 保存 `training_protocol`（损失/采样/增强/批/累积/优化器/调度器/monitor/精度/数据指纹）；恢复前逐项比对（在加载权重之前），差异逐条列出并拒绝；恢复事件仅在全部校验通过后写入（含当次 CLI/git/环境/config_effective SHA-256）；失败时原 run 文件逐字节不变、模型未被加载、无事件写入（回归测试覆盖 9 个变异场景 + 产物不变性）。旧断点（无协议记录）标记「协议未验证」并警告，不冒称精确。

### R03 · P1 · Notebook 诊断污染正式训练状态（F09/F12）

**位置：** [training/trainer.py](D:/Document/Unniversity/emotion_recognition/training/trainer.py:533)、[02_train_baseline.ipynb](D:/Document/Unniversity/emotion_recognition/training/notebooks/02_train_baseline.ipynb:159)。

**实测与影响：** MiniCNN Notebook 在同一个 `trainer` 上执行 `diagnose()` 后再 `fit()`；诊断执行真实更新。本次 `diagnose(num_steps=1)` 包含 3 步预热，累计 **4 次优化器更新**，参数最大变化 **0.49847969**，RNG 改变，但 history 仍为 0 轮。这样正式训练不再从声明的初始化/恢复状态起步；从断点恢复后运行该单元也会产生未记录的额外更新。重写后的 AMP 配对工具已用副本预热，不能替代此 Notebook 路径的隔离。

**修复方案：** 诊断使用独立模型、优化器、scaler 和独立数据加载器；完成后恢复主运行 RNG，不能消费正式持久 worker 的队列/增强状态。也可从 Notebook 移除同实例诊断，改为独立工具，再重新构建正式 Trainer。保留 K>1 时的真实分组 step 语义，或明确其诊断口径并禁止拿它外推正式训练。

**验收标准：** 诊断前后正式模型参数/BN buffer、优化器、scheduler、scaler、RNG、loader/sampler 状态及 run 文件逐项不变；同 seed 的“直接 fit”与“先诊断再 fit”得到同批输入及同等训练结果；FP32/AMP 和从 last 恢复的路径均覆盖。

**状态：** 已修复（2026-10-07）。`diagnose()` 完全隔离：独立模型/优化器/GradScaler/数据加载器副本，结束后恢复 RNG（torch/numpy/python）；正式模型参数、BN buffer、优化器、history、run 文件逐项不变（回归测试）；「先诊断再 fit」与「直接 fit」逐批输入与结果一致（回归测试）；`--diagnose` CLI 改为独立临时目录运行（不创建/触碰正式 runs）；K>1 保留组末 step 语义。

### R04 · P1 · 同一实例重复 fit 会重复轮号并覆盖定期断点（F09）

**位置：** [training/trainer.py](D:/Document/Unniversity/emotion_recognition/training/trainer.py:898)，三个训练 Notebook 的“可反复调用续训”说明。

**实测与影响：** `fit()` 结束没有推进 `start_epoch` 或累计会话用时。连续调用两次 `fit(1)` 后，history 与 last.epoch 均为 2，但 `start_epoch=1`、`run_meta.final_epoch=1`，只有 `epoch_0001.pth`，第二次覆盖第一轮定期断点。best_epoch、保存频率和累计时长也可能错误。重新创建 Trainer 并 load 的测试不能覆盖同实例反复运行 Notebook 单元。

**修复方案：** 以已完成 history 轮数统一推进下一轮起点，正常结束、早停和中断分别处理；每个会话结束更新累计训练时长。保留本会话起点的局部变量用于输出，避免循环内修改起点影响迭代。partial 中断须先回到完整 last 状态再精确继续，不能在已部分更新的模型上冒称完整 epoch 恢复。

**验收标准：** 在同实例上 `fit(1); fit(1)` 与 `fit(2)` 的批次/参数/history 一致，轮号为 1、2，`start_epoch=3`、last.epoch 与 final_epoch 为 2，定期断点分别保存为 0001/0002；累计时长递增。追加测试早停后再次 fit、load 后分段 fit 和中途中断。

**状态：** 已修复（2026-10-07）。fit 收尾按 history 长度推进 `start_epoch` 并累计会话时长；同实例 `fit(1);fit(1)` 与 `fit(2)` 的逐批输入/参数/history 一致、轮号 1/2、`start_epoch=3`、定期断点 0001/0002 分别保存（各自 epoch 正确）、`final_epoch=2`（回归测试）；早停后再次 fit 从正确轮号继续（回归测试）。

### R05 · P1 · Trainer 默认设备选择后模型仍留在 CPU

**位置：** [training/trainer.py](D:/Document/Unniversity/emotion_recognition/training/trainer.py:299)。

**实测与影响：** `model.to(device)` 先于默认设备解析。GPU 主机上省略 `device` 时，`self.device=cuda`、模型参数却在 CPU，首个训练批次报 `Input type (torch.cuda.FloatTensor) and weight type (torch.FloatTensor) should be the same`。CLI 与现 Notebook 显式传设备，正常路径不受此项影响；模块文档中的默认调用会失败。

**修复方案：** 先规范化并解析 `self.device = torch.device(...)`，再 `model.to(self.device)`，随后创建 optimizer/scaler；所有入口共用此顺序。

**验收标准：** CUDA 主机上省略 device、显式 cuda/cpu 以及无 CUDA 时的默认路径均能完成有限的前向、反向与有效参数更新；模型、输入和优化器状态在正确设备上。GPU 用例应检查实际运算，不能仅 mock 可用性。

**状态：** 已修复（2026-10-07）。设备解析先于 `model.to(device)`；省略 device（CUDA 主机）、显式 cuda/cpu、无 CUDA 默认路径均有回归覆盖（GPU 用例验证真实前向/反向与有效参数更新）。

### R06 · P1 · 增强总开关不能关闭 MixUp（F13 重新打开）

**位置：** [training/trainer.py](D:/Document/Unniversity/emotion_recognition/training/trainer.py:345)。

**实测与影响：** 合法配置 `augmentation.enabled=false`、`mixup.enabled=true` 通过校验，却得到 `trainer.mixup_enabled=true`；4 个训练批次实际调用 MixUp **4 次**。关闭增强的对照仍有批级增强，实际设置与声明不符。类别专属增强的总开关修复不能覆盖这个分支。

**修复方案：** 有效 MixUp 开关由总开关与子开关共同计算；在 run 记录有效增强设置。总开关关闭时禁用所有图像/类别/批级增强，或对矛盾配置明确报错，统一 CLI 与 Notebook 语义。

**验收标准：** 覆盖总开关 × MixUp 开关的四种组合；总开关 false 时实际调用数为 0、输入/标签无混合；总开关 true 且 MixUp true 才生效。类别专属增强与普通增强做同样组合检查，配置快照与运行日志一致。

**状态：** 已修复（2026-10-07）。有效 MixUp 开关 = 总开关 AND 子开关；run_meta 记录 `effective_augmentation`；四种组合的「开关值 + 实际调用次数」回归测试通过；类别专属增强总开关回归测试通过。

### R07 · P1 · 配置校验只覆盖 CLI，Notebook/API 可静默降级（F13）

**位置：** [training/train.py](D:/Document/Unniversity/emotion_recognition/training/train.py:89)、[training/trainer.py](D:/Document/Unniversity/emotion_recognition/training/trainer.py:299)、[utils/config_validation.py](D:/Document/Unniversity/emotion_recognition/utils/config_validation.py:102)。

**实测与影响：** 三个训练 Notebook、`load_config()` 与 Trainer 构造均不调用集中校验。`loss_type=misspelled_loss` 被独立 validator 正确拒绝，却能构造 Trainer 并静默使用 FocalLoss。另将模型学习率设为 NaN，集中 validator 仍接受，因为仅做区间比较；非法数值未在声明的集中校验阶段被拒绝，错误可能延后到下游或影响监控行为。未知参数与非法值因入口不同而有不同语义。

**修复方案：** 将完整配置校验放进 CLI/Notebook/API 共同调用的构造入口，且在模型移动、创建 run 或更新旧 run 之前执行；未知 loss 使用明确错误分支。数值验证先 `math.isfinite()`，补全类别增强数值与合法 seed 范围。不要仅在 Notebook 增加一条注释声称已校验。

**验收标准：** 同一组未知键、拼错 loss、NaN/Inf 学习率/阈值/dropout、非法增强概率、非法 seed，从 CLI、Notebook 构造单元和 API 均在首批训练与任何正式 run 写入之前拒绝；合法当前三模型配置均通过。新增回归应验证实际入口，不只直接测试 validator 函数。

**状态：** 已修复（2026-10-07）。完整配置校验移入 Trainer 构造入口（在任何模型移动 / 创建 run 之前）；NaN/Inf 数值、seed 范围、类别增强数值补全、未知 loss 明确报错（Trainer 内另有防御分支）；回归覆盖「构造入口实际拒绝」（失败不创建 run 目录）与 validator 级 8 个数值用例。

### R08 · P1 · 普通 CE 基线及公平比较的前置仍未完成（F07/F11）

**位置：** [utils/config_validation.py](D:/Document/Unniversity/emotion_recognition/utils/config_validation.py:61)、[training/trainer.py](D:/Document/Unniversity/emotion_recognition/training/trainer.py:365)、[comparison_protocol_draft.md](D:/Document/Unniversity/emotion_recognition/docs/comparison_protocol_draft.md:8)。

**证据与影响：** 原 F07 要求“普通随机采样 + 交叉熵”基线，当前配置只接受 focal/cb_focal，CLI 无 gamma 覆盖，草案也只写 Focal vs CB-Focal；本次 `loss_type=cross_entropy` 被拒绝。虽然 FocalLoss(gamma=0) 数学上等价 CE，尚未通过可记录的配置入口接入。草案声称同一训练配置，而基座 YAML 的模型级 LR、batch、num_epochs 分别不同；比较预算和执行轮数的最终口径未冻结。不存在 3 模型 × 3 seeds 正式结果，不能将“草案已写”列为实验已验收。

**修复方案：** 为现有训练器加入显式 CE 配置分支（或可记录、经校验的 gamma=0 方案），配套普通随机采样、关闭类别专属增强的基线配置。保留现有三架构，先选定并记录“固定共同方案”或“相同调参预算”之一，写明每模型实际 LR/batch/epochs、早停/调度器、选择指标和 seeds；基线与已有选项的对照每次只改变一项。先补修 R01–R07，再冻结方案，不自动启动训练。

**验收标准：** CE 配置可由同一入口启动，criterion、采样器、有效增强及轮数在 run 中可核查；小例子 loss/梯度与普通 CrossEntropyLoss 等价。协议有冻结日期/commit 和明确预算表，PrivateTest 不参与选择；正式执行后保留每臂 seeds=42/43/44 的全部结果与 mean±样本 std，不能先看 PrivateTest 再修改方案。正式比较前本项状态保持“前置未齐备/实验未执行”。

**状态：** 代码侧已就绪（2026-10-07）；正式比较维持「前置未齐备 / 实验未执行」。新增 `cross_entropy` 损失分支（小例子与 torch CE 等价，回归测试）；`configs/baseline_config.yaml`（CE + 普通采样 + 关类别增强；与主配置防漂移测试）；`--config` CLI 入口（同一入口启动基线）。比较协议升级 comparison-v2：预算口径选定「固定共同方案」（LR=3e-4 / batch=128 / epochs≤90 / cosine_warm / 早停一致；docs/comparison_protocol_draft.md §3.1）与 A–D 对照臂（§3.2）。协议冻结仍需用户评审（冻结时记录日期与 commit）。

### R09 · P2 · 材料口径、比较 Notebook 和 Windows 日志仍需收口（F02/F18/F19）

**位置与证据：** README 与 pyproject 指向 `BernardW18/fer2013-emotion-recognition`，实际 remote 为 `BernardW18/emotion_recognition`；YAML batch 注释仍写 50K/1.5M/400K。比较 Notebook 的训练历史自动取最新 run（当前 MiniCNN 是 2 轮 smoke）或 legacy，测试却固定读取 `<model>_best.pth` 的 legacy 权重，混淆矩阵也读取旧公共图片，不能当成同一批正式实验。三个训练 Notebook 仍默认 `PrivateTest`，与未冻结草案阶段仅用 PublicTest 的约定不匹配。课程 DOCX/PPT 未同步、个人贡献尚未单独成文。

本次还复现：Windows 默认 GBK 下将 `tools/data_audit.py` 输出重定向到日志，打印 `↔` 时 UnicodeEncodeError，JSON 未生成；同一命令加 `-X utf8` 成功。上述历史总览中“全部修复”“代码/README 仍未修改”等相互矛盾表述，本次已在本审核文档校正，README 的功能承诺仍需下一轮同步。

**修复方案：** 用实际 remote 统一链接、更新遗留参数注释；比较 Notebook 通过显式 run/checkpoint/结果清单读取同源 history、参数、指标和图，不按 mtime 自动挑选正式模型，不混用 smoke 与 legacy。调试单元默认 PublicTest；PrivateTest 作为冻结后的独立最终步骤。README 清楚限定精确续训的实测条件；贡献材料只由可核查本人工作组成。Windows 工具的日志入口统一 UTF-8（或安全的 ASCII 输出），把可执行命令写进文档。

**验收标准：** 新 run 生成后，同一模型的曲线、汇总、混淆矩阵与表格都可追溯同一 checkpoint/run/协议；缺文件或来源混合时明确拒绝正式比较。无旧参数量、错误仓库链接或未验证的“全部正确”表述；UTF-8 日志重定向能完成工具并生成有效 JSON。对外提交前报告/PPT/README 与同一结果记录一致，有具体个人贡献清单；未完成材料明确保留待办。

### 下一轮修复顺序与提交门槛

1. 先修 R02/R07（恢复及配置约束）、R05/R06（设备与开关），防止错误设置进入训练。
2. 修 R01/R03/R04，并把本次探针转成隔离、可重复的回归检查；保留已通过的 75 项基础检查。
3. 接入 R08 的 CE 基线并冻结现有模型比较口径；重新做一次短训练→完整 last 恢复→导出→PublicTest 评估，核对配置与数据指纹。
4. 通过后再执行现有模型正式训练；按 R09 整理与结果同源的演示及申请材料。

**执行记录（2026-10-07）**：步骤 1–3 已完成（R01–R08 修复与回归转测试、闭环复核、
协议预算选定，见下节）；第 4 步的正式训练按用户决策未启动，待协议冻结后执行；
R09 中的报告/PPT 与个人贡献清单按用户决定保留待办。

用于套磁时，当前适合展示“完成统一加载、可复算评估和数据审计，并通过独立审核发现和修复训练边界问题”的过程。尚不能声称完全可复现、已消除数据重叠或已证明所有不平衡选项有效；这些结论必须来自对应验收与正式对照。无需增加新问题或新架构来替代当前修复。

**状态：** 已修复（2026-10-07，报告/PPT 除外，按用户决定）。仓库链接统一为实际 remote（BernardW18/emotion_recognition）；YAML 遗留参数量注释更新；比较 Notebook 重构为显式来源（`COMPARISON_SOURCES`，缺失即失败、不按 mtime 挑选、不混用 smoke/legacy），混淆矩阵/recall/散点现场生成（不再读取历史图片），实际执行验证通过、重复评估与既有记录完全一致；训练 Notebook 评估默认改为 PublicTest（冻结后最终评估再改 PrivateTest）；工具输出统一 UTF-8（utils/stdio.py；GBK 重定向场景实测通过）。报告 DOCX/PPT 未修改（用户决定拒绝修改 word/ppt，可询问用户核实）；个人贡献清单不做（用户决定，保持待办）。

---

## 第二轮修复（R01–R09）执行记录（2026-10-07）

- **质量门槛**：112 项测试全部通过（新增 R01–R08 回归 37 项）；Ruff / mypy 全绿。
- **闭环复核**（按审核第 3 步重做）：新短训练 run `20261007_031251_seed42`（mini_cnn）：
  训练 1 epoch → `--resume auto`（自动应用精确恢复条件 num_workers 4→0 并记录于
  resume_events；训练协议比对通过；恢复至第 2 轮完成）→ 导出（manifest 记录 run 与
  SHA-256）→ PublicTest 统一评估（acc=0.3689；与训练时 GPU/AMP 验证值 0.3681 相差
  3/3589 个样本，源于 AMP 与 CPU-FP32 数值路径差异，如实记录、非缺陷）。
  run_meta 数据指纹（csv_sha256=3b8d…、splits 28709/3589/3589）与配置完整可核查。
- **材料决定（记录在案）**：课程报告 DOCX / PPT 不修改（用户决定拒绝修改 word/ppt，
  可询问用户核实）；个人贡献清单不做（用户决定，保持待办）。
- **未执行（按用户决策）**：3 模型 × 3 seeds 正式重训、CE/Focal/CB-Focal 对照实验、
  dedup-v1 去重重训；比较协议（comparison-v2）待用户评审冻结。

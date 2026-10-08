# FER2013 数据与评价说明

数据集背景与标签定义参考[原始FER2013挑战页](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data)。
本项目实际使用含完整标签和Usage的`data/fer2013.csv`归档，不随Git提供。
其他打包版本的图片目录或竞赛train/test文件不能仅改名替代此CSV；复现先核对字段、划分及文件SHA。

## 格式与固定身份

| 字段 | 内容 |
|---|---|
| emotion | 七类标签0–6 |
| pixels | 48×48灰度图，共2,304个空格分隔像素值，按行排列 |
| Usage | Training、PublicTest、PrivateTest官方划分 |

实际使用文件SHA-256：
`3b8d9617d1017f34733c8f2474d7784c563ce86c40a86ac12c2d37cc968f871b`。
训练前不合并/重划分Usage。输入归一化为x/255；训练增强与无增强评估分别执行。

## 划分与类别统计

| ID | 类别 | Training | PublicTest | PrivateTest |
|---|---|---:|---:|---:|
| 0 | Angry | 3,995 | 467 | 491 |
| 1 | Disgust | 436 | 56 | 55 |
| 2 | Fear | 4,097 | 496 | 528 |
| 3 | Happy | 7,215 | 895 | 879 |
| 4 | Sad | 4,830 | 653 | 594 |
| 5 | Surprise | 3,171 | 415 | 416 |
| 6 | Neutral | 4,965 | 607 | 626 |
| — | 合计 | 28,709 | 3,589 | 3,589 |

Training中Disgust仅436张，与Happy的7,215张差距明显。
accuracy之外同时报告macro-F1、balanced accuracy及七类召回，避免只由大类决定结论。
历史EDA分布图合并了全部Usage，本表区分各划分。

## 完全重复像素审计

将pixels字符串按空白拆分后以单空格连接，计算SHA-256，并按哈希分组。
这是规范化像素完全相同的检查，不覆盖近重复或人员身份重叠。

| 指标 | 数量 |
|---|---:|
| 重复组（同一像素出现至少两次） | 1,516 |
| 超出唯一图像数的记录 | 1,853 |
| 标签冲突组 | 57 |
| PublicTest中与Training像素相同的记录 | 280 |
| PrivateTest中与Training像素相同的记录 | 288 |
| PrivateTest中与PublicTest像素相同的记录 | 44 |

统计与方法见[数据审计摘要](../analysis/data_audit.json)。当前正式结果仍采用官方划分，
没有实施去重重训，不宣称数据泄漏已消除、人员互斥或跨人员泛化。
PublicTest模型选择和PrivateTest最终报告均应同时披露此重叠。

审计摘要中的敏感性分析仅针对legacy历史权重：排除PrivateTest中与Training重复的288条，
对剩余3,301条重新计算指标。它不是去重重训，也不是当前S1的额外实验，不参与当前模型排名。
没有对应历史预测归档时，重跑审计仍可生成重复统计，但不会恢复该历史敏感性表。

## 本地校验

在项目目录核对数据文件：

```powershell
Get-FileHash -LiteralPath data/fer2013.csv -Algorithm SHA256
.\.venv\Scripts\python.exe tools/data_audit.py --output analysis/local_data_audit.json
```

新的审计输出默认保持本地，既有发布摘要不自动覆盖。缓存、CSV与原始逐样本预测由Git忽略。
环境与正式评估操作见[复现说明](../REPRODUCIBILITY.md)。

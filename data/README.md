# 数据目录

## FER2013 数据集

- **来源**: [Kaggle - FER2013](https://www.kaggle.com/datasets/msambare/fer2013)
- **文件**: `fer2013.csv`
- **格式**: CSV，包含三列：
  - `emotion`: 情感标签 (0-6)
  - `pixels`: 48x48 灰度图像的像素值（空格分隔的字符串）
  - `Usage`: 数据划分标记 (Training / PublicTest / PrivateTest)

## 情感类别映射

| ID | 类别 |
|----|------|
| 0  | Angry |
| 1  | Disgust |
| 2  | Fear |
| 3  | Happy |
| 4  | Sad |
| 5  | Surprise |
| 6  | Neutral |

## 注意事项

- `fer2013.csv` 不纳入 Git 版本控制（文件较大）
- 下载后放置于本目录即可
- 数据集存在类别不平衡问题，Disgust 类样本极少
- 数据存在跨划分的完全重复像素（PublicTest 280 条、PrivateTest 288 条与 Training 重复），
  详细披露与敏感性分析见 [docs/data_audit.md](../docs/data_audit.md)；
  可复跑审计：`python tools/data_audit.py`

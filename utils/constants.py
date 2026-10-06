"""
公共常量模块 — 情感类别名称、Emoji 映射等
所有模块从此导入，避免三处拷贝导致的不一致风险

用法:
    from utils.constants import CLASS_NAMES, CLASS_EMOJIS, EMOTION_COUNT
"""

__all__ = [
    "CLASS_NAMES",
    "CLASS_EMOJIS",
    "EMOTION_COUNT",
    "EMOTION_ID_MAP",
]

CLASS_NAMES = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

EMOTION_COUNT = len(CLASS_NAMES)

# 情感名称到索引的映射 (O(1) 查询)
EMOTION_ID_MAP = {name: idx for idx, name in enumerate(CLASS_NAMES)}

# 注意：类别计数不再硬编码。历史上此处曾保存一份与官方 Training 划分不符的
# 固定计数（总和 35,776 ≠ 28,709），导致 CB Focal Loss 使用错误权重。
# 训练时一律由 data.dataloader.compute_class_counts 按当前实际划分统计
# （官方 Training 划分的七个计数为 [3995, 436, 4097, 7215, 4830, 3171, 4965]）。

# 情感类别对应的 emoji（用于推理 UI 展示）
CLASS_EMOJIS = {
    "Angry": "😠",
    "Disgust": "🤢",
    "Fear": "😨",
    "Happy": "😊",
    "Sad": "😢",
    "Surprise": "😲",
    "Neutral": "😐",
}

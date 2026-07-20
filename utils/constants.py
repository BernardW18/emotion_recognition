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
    "CLASS_COUNTS",
]

CLASS_NAMES = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

EMOTION_COUNT = len(CLASS_NAMES)

# 情感名称到索引的映射 (O(1) 查询)
EMOTION_ID_MAP = {name: idx for idx, name in enumerate(CLASS_NAMES)}

# FER2013 各类别训练集样本数（基于 CSV Usage=Training 统计）
# 用于 CB Focal Loss 的 class_counts 参数
CLASS_COUNTS = [4953, 436, 5121, 8989, 6077, 4002, 6198]

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

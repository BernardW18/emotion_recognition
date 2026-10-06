"""
数据加载模块 — FER2013 Dataset, 数据增强, DataLoader 工厂
从 training/trainer.py 拆分而来，实现数据层与训练层分离。

职责:
  - FER2013Dataset: 从 CSV 按需加载图像并支持 transform
  - build_train_transform: 根据配置组合数据增强管道
  - create_dataloaders: 读取 CSV → 划分 → 构建 DataLoader
  - compute_class_counts: 按固定索引统计类别样本数（采样器与损失共用）
  - compute_split_fingerprint: CSV + 各划分指纹（供 run 元数据追溯）
  - compute_class_weights: 计算逆频率类别权重（已弃用工具）

用法:
    from data.dataloader import create_dataloaders, FER2013Dataset
    train_loader, val_loader, test_loader, classes = create_dataloaders(config)
"""

import hashlib
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# ============================================================
# 项目根目录（动态计算）
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent


__all__ = [
    "FER2013Dataset",
    "build_train_transform",
    "build_class_aug_transform",
    "compute_class_counts",
    "compute_split_fingerprint",
    "create_dataloaders",
    "compute_class_weights",
    "PROJECT_ROOT",
]


# ============================================================
# FER2013 数据集类
# ============================================================
class FER2013Dataset(Dataset):
    """
    FER2013 数据集类（CPU 路径，多进程安全）
    支持配置文件中所有增强选项，包括 RandomErasing 和类别特定增强。

    性能优化：
    - 首次构建时将 CSV 中像素字符串批量解析为 float32 numpy 数组
    - 仅保留 labels 数组，不持有 DataFrame 引用
    - 多进程安全：worker 只需 pickle 轻量 numpy 数组，不会拷贝巨型 DataFrame
    - pin_memory 实现异步 CPU→GPU 传输
    """

    def __init__(self, dataframe, transform=None, class_aug_map: dict | None = None):
        """
        Args:
            dataframe: pandas DataFrame，含 emotion 和 pixels 列
            transform: 基础数据增强管道
            class_aug_map: 类别特定增强映射 {class_index: (transform, probability)}
                例如 {1: (extra_transforms, 0.8)} 表示 Disgust 类有 80% 概率应用额外变换
        """
        self.transform = transform
        self.class_aug_map = class_aug_map or {}
        # 训练元数据（由 create_dataloaders 填充；默认 None）
        self.class_counts: list | None = None
        self.split_fingerprint: dict | None = None
        # 预计算：一次性解析所有像素字符串为 float32 数组，后续取值 O(1)
        self._pixels = self._precompute_pixels(dataframe)
        # 仅提取 labels，不持有 DataFrame 引用（避免多进程 pickle 巨型对象）
        self._labels = dataframe["emotion"].values

    def _precompute_pixels(self, dataframe):
        """批量解析像素字符串为 (N, 48, 48) float32 数组"""
        n = len(dataframe)
        pixels_all = np.empty((n, 48, 48), dtype=np.float32)
        pixel_data = dataframe["pixels"].values
        for i in range(n):
            pixels_all[i] = np.array(
                pixel_data[i].split(), dtype=np.float32
            ).reshape(48, 48)
        pixels_all /= 255.0
        return pixels_all

    def __len__(self):
        return len(self._labels)

    def __getitem__(self, idx):
        image = self._pixels[idx]
        image = np.expand_dims(image, axis=0)  # (1, 48, 48)
        label = self._labels[idx]

        if self.transform:
            image = self.transform(torch.FloatTensor(image))
        else:
            image = torch.FloatTensor(image)

        # 类别特定增强：对指定类别应用额外变换
        if label in self.class_aug_map:
            extra_transform, prob = self.class_aug_map[label]
            if torch.rand(1).item() < prob:
                image = extra_transform(image)

        return image, label


# ============================================================
# 数据增强管道
# ============================================================
def build_train_transform(aug_config: dict) -> transforms.Compose:
    """
    根据配置构建训练数据增强管道

    Args:
        aug_config: configs中的 augmentation section

    Returns:
        transforms.Compose 实例，若增强未启用返回 None
    """
    if not aug_config.get("enabled", False):
        return None

    transform_list = []

    if aug_config.get("random_horizontal_flip"):
        transform_list.append(
            transforms.RandomHorizontalFlip(p=aug_config["random_horizontal_flip"])
        )

    if aug_config.get("random_rotation"):
        transform_list.append(
            transforms.RandomRotation(degrees=aug_config["random_rotation"])
        )

    # 微小平移
    translate = aug_config.get("random_affine_translate", 0)
    if translate > 0:
        transform_list.append(
            transforms.RandomAffine(degrees=0, translate=(translate, translate))
        )

    # 亮度/对比度抖动
    brightness = aug_config.get("color_jitter_brightness", 0)
    contrast = aug_config.get("color_jitter_contrast", 0)
    if brightness > 0 or contrast > 0:
        transform_list.append(
            transforms.ColorJitter(brightness=brightness, contrast=contrast)
        )

    # P1-6: 支持 RandomErasing
    if aug_config.get("random_erase", False):
        transform_list.append(transforms.RandomErasing(p=0.1))

    return transforms.Compose(transform_list) if transform_list else None


def build_class_aug_transform(aug_config: dict, *, master_enabled: bool = True) -> dict:
    """
    根据配置构建类别特定增强映射。

    Args:
        aug_config: configs 中 augmentation.class_specific 部分
        master_enabled: augmentation.enabled 总开关；关闭时整体停用类别增强

    Returns:
        {class_index: (transform_compose, probability)} 字典
    """
    if not master_enabled or not aug_config.get("enabled", False):
        return {}

    target_classes = aug_config.get("target_classes", [])
    if not target_classes:
        return {}

    prob = aug_config.get("augment_prob", 0.8)
    extra_list = []

    rotation = aug_config.get("extra_rotation", 0)
    if rotation > 0:
        extra_list.append(transforms.RandomRotation(degrees=rotation))

    translate = aug_config.get("extra_translate", 0)
    if translate > 0:
        extra_list.append(
            transforms.RandomAffine(degrees=0, translate=(translate, translate))
        )

    erase_prob = aug_config.get("extra_erase_prob", 0)
    if erase_prob > 0:
        extra_list.append(transforms.RandomErasing(p=erase_prob))

    if not extra_list:
        return {}

    extra_transform = transforms.Compose(extra_list)
    return {cls: (extra_transform, prob) for cls in target_classes}


# ============================================================
# DataLoader 工厂
# ============================================================
def create_dataloaders(config: dict, model_name: str | None = None):
    """
    创建训练/验证/测试 DataLoader

    注意：类别平衡由训练器中的 Focal Loss 处理，无需显式计算 class_weights。

    Args:
        config: 完整训练配置
        model_name: 模型名称，用于读取模型特定的 batch_size

    Returns:
        train_loader, val_loader, test_loader, class_names
    """
    class_names = config["data"]["class_names"]
    use_cuda = torch.cuda.is_available()

    # 加载数据
    dataset_path = PROJECT_ROOT / config["data"]["dataset_path"]
    df = pd.read_csv(dataset_path)

    # 按官方 Usage 划分（官方协议）；缺失 Usage 时明确报错，不做静默回退
    if "Usage" not in df.columns:
        raise ValueError(
            "数据集缺少 Usage 列：本项目使用官方 Training/PublicTest/PrivateTest 划分；"
            "自定义划分清单尚未支持，请提供含 Usage 的 FER2013 CSV"
        )
    train_df = df[df["Usage"] == "Training"].reset_index(drop=True)
    val_df = df[df["Usage"] == "PublicTest"].reset_index(drop=True)
    test_df = df[df["Usage"] == "PrivateTest"].reset_index(drop=True)
    if min(len(train_df), len(val_df), len(test_df)) == 0:
        raise ValueError(
            f"Usage 划分不完整: Training={len(train_df)}, "
            f"PublicTest={len(val_df)}, PrivateTest={len(test_df)}"
        )
    if len(train_df) + len(val_df) + len(test_df) != len(df):
        raise ValueError("存在未归入官方划分的行（Usage 取值异常），请检查数据文件")

    # 获取 batch_size
    if model_name and model_name in config.get("models", {}):
        batch_size = config["models"][model_name].get(
            "batch_size", config["training"]["batch_size"]
        )
    else:
        batch_size = config["training"]["batch_size"]

    # 数据增强
    train_transform = build_train_transform(config["augmentation"])

    # 类别特定增强（如 Disgust 类额外增强）；总开关关闭时整体停用
    class_aug_map = build_class_aug_transform(
        config.get("augmentation", {}).get("class_specific", {}),
        master_enabled=config.get("augmentation", {}).get("enabled", False),
    )

    # 统一使用 CPU 路径：多进程 worker 并行增强 + pin_memory 异步传输
    train_dataset = FER2013Dataset(
        train_df, transform=train_transform, class_aug_map=class_aug_map)
    val_dataset = FER2013Dataset(val_df)
    test_dataset = FER2013Dataset(test_df)

    # 训练集类别计数（固定 0..num_classes-1 索引）：采样器与 CB Focal Loss 共用同一份统计
    num_classes = config["data"]["num_classes"]
    train_class_counts = compute_class_counts(train_df["emotion"].values, num_classes)
    train_dataset.class_counts = train_class_counts
    print(
        f"训练集类别计数 (索引 0-{num_classes - 1}): "
        f"{train_class_counts} | 合计 {sum(train_class_counts)}"
    )

    # 数据指纹（CSV SHA-256 + 各划分行号/标签哈希）：随 run_meta 记录，可复核追溯
    train_dataset.split_fingerprint = compute_split_fingerprint(dataset_path, df)

    # DataLoader 配置
    dl_config = config.get("dataloader", {})
    train_workers = dl_config.get("num_workers", 0)
    pin_memory = dl_config.get("pin_memory", use_cuda)
    persistent_workers = dl_config.get("persistent_workers", False)
    prefetch_factor = dl_config.get("prefetch_factor", 2)

    # num_workers=0 时 DataLoader 不接受 persistent_workers=True / prefetch_factor：
    # 自动关闭并明确提示（Windows spawn 场景常见）
    if train_workers <= 0:
        if persistent_workers:
            print("  num_workers=0：persistent_workers 自动关闭（DataLoader 不允许）")
        persistent_workers = False

    # 训练 DataLoader：多进程并行增强 + 大 prefetch 缓冲吸收变换抖动
    # 更大的 prefetch_factor 让 DataLoader 预取更多批次到队列，
    # 当某批变换耗时高时主线程可以直接消费缓存，避免等待
    train_loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": train_workers,
        "pin_memory": pin_memory,
        "persistent_workers": persistent_workers,
        "prefetch_factor": prefetch_factor if train_workers > 0 else None,
    }

    # 验证/测试 DataLoader：无数据增强，不需要多进程
    # Windows 的多进程 spawn 开销大（~2-14秒首次），单进程反而更快
    eval_loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": 0,
        "pin_memory": pin_memory,
    }

    nw_info = f"train_workers={train_workers}(prefetch={prefetch_factor}), eval_workers=0"
    print(f"DataLoader: batch_size={batch_size}, {nw_info}, pin_memory={pin_memory}")

    # Class-balanced sampling：用 WeightedRandomSampler 过采样少样本类
    # 与 Focal Loss 互补：sampler 负责数据分布，Focal Loss 负责梯度权重
    class_balanced = dl_config.get("class_balanced_sampling", False)
    train_sampler = None
    if class_balanced:
        weights = _compute_sampler_weights(
            train_df, num_classes=num_classes, class_counts=train_class_counts
        )
        train_sampler = torch.utils.data.WeightedRandomSampler(
            weights, num_samples=len(weights), replacement=True,
        )
        print(f"  class_balanced_sampling: enabled (min_weight={min(weights):.4f}, "
              f"max_weight={max(weights):.4f}, ratio={max(weights)/min(weights):.1f}x)")

    train_loader = DataLoader(
        train_dataset, shuffle=(train_sampler is None),
        sampler=train_sampler, **train_loader_kwargs,
    )
    val_loader = DataLoader(val_dataset, shuffle=False, **eval_loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **eval_loader_kwargs)

    return train_loader, val_loader, test_loader, class_names


# ============================================================
# Class-balanced sampler 权重计算
# ============================================================
def compute_split_fingerprint(csv_path, df: pd.DataFrame) -> dict:
    """
    计算数据指纹：CSV 文件 SHA-256 + 官方各划分的（行号‖标签）哈希。

    划分哈希算法：sha256( 行号 int64-LE 字节 ‖ 标签 int64-LE 字节 )；
    行号 = CSV 数据行顺序（表头后第 i 行，从 0 计）。
    结果用于 run 元数据记录，可复核训练所用划分与数据文件未变。
    """
    h = hashlib.sha256()
    with open(csv_path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    fingerprint: dict[str, Any] = {
        "protocol": "official-usage",
        "csv_path": str(csv_path),
        "csv_sha256": h.hexdigest(),
        "splits": {},
    }
    for split in ("Training", "PublicTest", "PrivateTest"):
        mask = (df["Usage"] == split).to_numpy()
        idx = np.nonzero(mask)[0].astype(np.int64)
        labels = df.loc[mask, "emotion"].to_numpy(dtype=np.int64)
        sh = hashlib.sha256(idx.tobytes() + labels.tobytes()).hexdigest()
        fingerprint["splits"][split] = {
            "rows": int(mask.sum()),
            "row_label_hash_sha256": sh,
        }
    return fingerprint


def compute_class_counts(labels, num_classes: int) -> list:
    """
    按 0..num_classes-1 的固定索引统计各类别样本数。

    - 保持类别索引不变：某类缺失记为 0，不压缩位置（防止类别错位）
    - 标签越界时明确报错

    Args:
        labels: 标签数组（numpy 或 list）
        num_classes: 类别总数（本项目固定为 7）

    Returns:
        长度 = num_classes 的计数列表
    """
    labels = np.asarray(labels)
    if labels.size:
        lo, hi = int(labels.min()), int(labels.max())
        if lo < 0 or hi >= num_classes:
            raise ValueError(f"标签值超出 [0, {num_classes}) 范围: min={lo}, max={hi}")
    return [int((labels == i).sum()) for i in range(num_classes)]


def _compute_sampler_weights(train_df: pd.DataFrame, *, num_classes: int,
                             class_counts: list) -> list:
    """
    为 WeightedRandomSampler 计算每个样本的采样权重（逆频率加权）。

    与 CB Focal Loss 使用同一份 class_counts（compute_class_counts 统计，
    经 train_loader.dataset.class_counts 传入 Trainer），保证两种机制口径一致。
    零计数类别明确报错（不压缩类别索引、不静默跳过）。

    Args:
        train_df: 训练集 DataFrame（含 'emotion' 列）
        num_classes: 类别总数（固定索引 0..num_classes-1）
        class_counts: compute_class_counts 的统计结果

    Returns:
        每个样本的权重列表，顺序与 train_df 一致
    """
    zero_classes = [i for i, c in enumerate(class_counts) if c <= 0]
    if zero_classes:
        raise ValueError(
            f"训练集类别索引 {zero_classes} 样本数为 0，无法计算逆频率采样权重；"
            "请检查数据划分或关闭 class_balanced_sampling"
        )
    labels = train_df["emotion"].values
    counts = np.asarray(class_counts, dtype=np.float64)
    class_weights = counts.sum() / (num_classes * counts)
    # 每个样本的权重 = 其类别的逆频率权重
    per_sample = class_weights[labels]
    return [float(w) for w in per_sample]


# ============================================================
# 类别权重计算（保留为工具函数）
# ============================================================
def compute_class_weights(train_df, class_names, device):
    """根据训练集计算逆频率类别权重

    ⚠️ 已弃用: 训练器改用 Focal Loss 处理类别不平衡，不再需要此函数。
       保留作为工具函数供手动分析使用。"""
    warnings.warn(
        "compute_class_weights 已弃用，Trainer 使用 Focal Loss 处理类别不平衡",
        DeprecationWarning, stacklevel=2,
    )
    emotion_counts = train_df["emotion"].value_counts().sort_index()
    total = len(train_df)
    class_weights = total / (len(class_names) * emotion_counts.values)
    return torch.as_tensor(class_weights, dtype=torch.float32).to(device)

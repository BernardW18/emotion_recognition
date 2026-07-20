"""
数据加载模块 — FER2013 Dataset, 数据增强, DataLoader 工厂
从 training/trainer.py 拆分而来，实现数据层与训练层分离。

职责:
  - FER2013Dataset: 从 CSV 按需加载图像并支持 transform
  - build_train_transform: 根据配置组合数据增强管道
  - create_dataloaders: 读取 CSV → 划分 → 构建 DataLoader
  - compute_class_weights: 计算逆频率类别权重

用法:
    from data.dataloader import create_dataloaders, FER2013Dataset
    train_loader, val_loader, test_loader, classes = create_dataloaders(config)
"""

import warnings
from pathlib import Path

import numpy as np
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
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

    def __init__(self, dataframe, transform=None, class_aug_map: dict = None):
        """
        Args:
            dataframe: pandas DataFrame，含 emotion 和 pixels 列
            transform: 基础数据增强管道
            class_aug_map: 类别特定增强映射 {class_index: (transform, probability)}
                例如 {1: (extra_transforms, 0.8)} 表示 Disgust 类有 80% 概率应用额外变换
        """
        self.transform = transform
        self.class_aug_map = class_aug_map or {}
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


def build_class_aug_transform(aug_config: dict) -> dict:
    """
    根据配置构建类别特定增强映射。

    Args:
        aug_config: configs 中 augmentation.class_specific 部分

    Returns:
        {class_index: (transform_compose, probability)} 字典
    """
    if not aug_config.get("enabled", False):
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
def create_dataloaders(config: dict, model_name: str = None):
    """
    创建训练/验证/测试 DataLoader

    注意：类别平衡由训练器中的 Focal Loss 处理，无需显式计算 class_weights。

    Args:
        config: 完整训练配置
        model_name: 模型名称，用于读取模型特定的 batch_size

    Returns:
        train_loader, val_loader, test_loader, CLASS_NAMES
    """
    CLASS_NAMES = config["data"]["class_names"]
    use_cuda = torch.cuda.is_available()

    # 加载数据
    dataset_path = PROJECT_ROOT / config["data"]["dataset_path"]
    df = pd.read_csv(dataset_path)

    # 按Usage划分或手动划分
    if "Usage" in df.columns:
        train_df = df[df["Usage"] == "Training"].reset_index(drop=True)
        val_df = df[df["Usage"] == "PublicTest"].reset_index(drop=True)
        test_df = df[df["Usage"] == "PrivateTest"].reset_index(drop=True)
    else:
        train_df, temp_df = train_test_split(
            df, test_size=0.2, random_state=42, stratify=df["emotion"]
        )
        val_df, test_df = train_test_split(
            temp_df, test_size=0.5, random_state=42, stratify=temp_df["emotion"]
        )

    # 获取 batch_size
    if model_name and model_name in config.get("models", {}):
        batch_size = config["models"][model_name].get("batch_size", config["training"]["batch_size"])
    else:
        batch_size = config["training"]["batch_size"]

    # 数据增强
    train_transform = build_train_transform(config["augmentation"])

    # 类别特定增强（如 Disgust 类额外增强）
    class_aug_map = build_class_aug_transform(
        config.get("augmentation", {}).get("class_specific", {})
    )

    # 统一使用 CPU 路径：多进程 worker 并行增强 + pin_memory 异步传输
    train_dataset = FER2013Dataset(
        train_df, transform=train_transform, class_aug_map=class_aug_map)
    val_dataset = FER2013Dataset(val_df)
    test_dataset = FER2013Dataset(test_df)

    # DataLoader 配置
    dl_config = config.get("dataloader", {})
    train_workers = dl_config.get("num_workers", 0)
    pin_memory = dl_config.get("pin_memory", use_cuda)
    persistent_workers = dl_config.get("persistent_workers", False)
    prefetch_factor = dl_config.get("prefetch_factor", 2)

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
        weights = _compute_sampler_weights(train_df)
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

    return train_loader, val_loader, test_loader, CLASS_NAMES


# ============================================================
# Class-balanced sampler 权重计算
# ============================================================
def _compute_sampler_weights(train_df: pd.DataFrame) -> list:
    """
    为 WeightedRandomSampler 计算每个样本的采样权重。
    使用逆频率加权：少样本类别获得更高采样概率。

    Args:
        train_df: 训练集 DataFrame（含 'emotion' 列）

    Returns:
        每个样本的权重列表，顺序与 train_df 一致

    与 Focal Loss 的关系：
        - sampler 修正数据分布 → 每个 epoch 看见更多少样本
        - Focal Loss 修正梯度分布 → 少样本的 loss 贡献更大
        - 两者互补，可同时启用
    """
    class_counts = train_df["emotion"].value_counts().sort_index()
    n_total = len(train_df)
    n_classes = len(class_counts)
    class_weights = n_total / (n_classes * class_counts.values)
    # 每个样本的权重 = 其类别的逆频率权重
    weights = class_weights[train_df["emotion"].values].tolist()
    return weights


# ============================================================
# 类别权重计算（保留为工具函数）
# ============================================================
def compute_class_weights(train_df, CLASS_NAMES, device):
    """根据训练集计算逆频率类别权重

    ⚠️ 已弃用: 训练器改用 Focal Loss 处理类别不平衡，不再需要此函数。
       保留作为工具函数供手动分析使用。"""
    warnings.warn(
        "compute_class_weights 已弃用，Trainer 使用 Focal Loss 处理类别不平衡",
        DeprecationWarning, stacklevel=2,
    )
    emotion_counts = train_df["emotion"].value_counts().sort_index()
    total = len(train_df)
    class_weights = total / (len(CLASS_NAMES) * emotion_counts.values)
    return torch.FloatTensor(class_weights).to(device)

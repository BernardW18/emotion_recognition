"""
数据加载模块 — FER2013 Dataset, 数据增强, DataLoader 工厂
从 training/trainer.py 拆分而来，实现数据层与训练层分离。

职责:
  - FER2013Dataset: 从像素缓存 / DataFrame 按需加载图像并支持 transform
  - build_train_transform: 根据配置组合逐样本增强管道（legacy 实现）
  - create_dataloaders: 像素缓存（PB02）→ 划分 → 构建 DataLoader
  - compute_class_counts: 按固定索引统计类别样本数（采样器与损失共用）
  - compute_split_fingerprint: CSV + 各划分指纹（供 run 元数据追溯）
  - compute_split_fingerprint_cached: 同公式，从像素缓存元数据计算（不触碰像素文件）
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

from data.pixel_cache import (
    check_cache_ref,
    file_sha256_cached,
    get_memmap_split,
    load_or_build,
)

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
    "compute_split_fingerprint_cached",
    "create_dataloaders",
    "compute_class_weights",
    "PROJECT_ROOT",
]


# ============================================================
# FER2013 数据集类
# ============================================================
class FER2013Dataset(Dataset):
    """
    FER2013 数据集类（多进程安全）

    三种像素来源（三选一）：
    - cache_ref=(cache_dir, split, entry)：PB02 像素缓存路径。dataset 仅持轻量元数据，
      像素经进程内只读 mmap 惰性获取；spawn 时 pickle 不携带像素数据本体。
    - pixels=(N,48,48) 数组 + labels：直接提供（uint8 按 x/255 归一化；float32 视为
      已归一化）。用于测试与对照。
    - dataframe：旧路径（构建时逐行解析为 float32/255）。保留用于对照与兼容测试。

    增强：legacy 实现经 transform（逐样本 torchvision 管道）与 class_aug_map；
    batched 实现（PB01）时两者为 None/空 dict，批级增强由 Trainer 训练循环执行。
    """

    def __init__(self, dataframe=None, transform=None, class_aug_map: dict | None = None, *,
                 pixels=None, labels=None, cache_ref=None):
        """
        Args:
            dataframe: 旧路径的 pandas DataFrame（含 emotion 和 pixels 列）
            transform: 逐样本增强管道（legacy 实现）
            class_aug_map: 类别特定增强映射 {class_index: (transform, probability)}
                例如 {1: (extra_transforms, 0.8)} 表示 Disgust 类有 80% 概率应用额外变换
            pixels: (N,48,48) 像素数组（需与 labels 配对）
            labels: (N,) 标签数组
            cache_ref: (cache_dir, split, entry) 像素缓存引用（PB02 路径）
        """
        self.transform = transform
        self.class_aug_map = class_aug_map or {}
        # 训练元数据（由 create_dataloaders 填充；默认 None）
        self.class_counts: list | None = None
        self.split_fingerprint: dict | None = None

        sources = sum(s is not None for s in (dataframe, pixels, cache_ref))
        if sources != 1:
            raise ValueError(
                "FER2013Dataset 需要恰好一种像素来源：dataframe / pixels / cache_ref"
            )

        if cache_ref is not None:
            cache_dir, split, entry = cache_ref
            # 仅保留轻量引用（字符串 + 小 dict）：多进程 spawn 不复制像素数据
            self._cache_ref: tuple | None = (str(cache_dir), str(split), dict(entry))
            self._pixels = None
            self._labels = None
            self._n = int(entry["n"])
        elif pixels is not None:
            if labels is None:
                raise ValueError("pixels 模式需要同时提供 labels")
            self._cache_ref = None
            self._pixels = pixels
            self._labels = labels
            self._n = len(pixels)
        else:
            # 旧路径：一次性解析所有像素字符串为 float32/255，仅保留 numpy（不持有 DataFrame）
            self._cache_ref = None
            self._pixels = self._precompute_pixels(dataframe)
            # 仅提取 labels，不持有 DataFrame 引用（避免多进程 pickle 巨型对象）
            self._labels = dataframe["emotion"].values
            self._n = len(self._labels)

    def _precompute_pixels(self, dataframe):
        """批量解析像素字符串为 (N, 48, 48) float32 数组（已除 255；对照/兼容路径）"""
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
        return self._n

    def __getitem__(self, idx):
        arrays = get_memmap_split(self._cache_ref) if self._cache_ref is not None else None
        return self._get_sample(idx, arrays)

    def __getitems__(self, indices):
        """Preserve scalar index/RNG order while validating once before and after a batch."""
        arrays = get_memmap_split(self._cache_ref) if self._cache_ref is not None else None
        samples = [self._get_sample(idx, arrays) for idx in indices]
        self.validate_cache()
        return samples

    def validate_cache(self):
        if self._cache_ref is not None:
            check_cache_ref(self._cache_ref)

    def _get_sample(self, idx, arrays=None):
        if arrays is not None:
            # 缓存路径：只读 mmap 取单张（uint8 slice → float32 /255）
            x_mm, y_mm, _rows = arrays
            image = np.asarray(x_mm[idx], dtype=np.float32)
            image /= 255.0
            label = int(y_mm[idx])
        elif self._pixels is not None and getattr(self._pixels, "dtype", None) == np.uint8:
            image = np.asarray(self._pixels[idx], dtype=np.float32)
            image /= 255.0
            label = int(self._labels[idx])
        else:
            image = np.asarray(self._pixels[idx], dtype=np.float32)
            label = self._labels[idx]

        image = np.expand_dims(image, axis=0)  # (1, 48, 48)

        if self.transform:
            image = self.transform(torch.FloatTensor(image))
        else:
            image = torch.FloatTensor(image)

        # 类别特定增强：对指定类别应用额外变换（legacy 实现）
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
def create_dataloaders(config: dict, model_name: str | None = None, *,
                       include_test: bool = True):
    """
    创建训练/验证/测试 DataLoader

    - 像素来源为 data/pixel_cache 的 uint8 只读缓存（PB02）：加载时核验 CSV SHA-256，
      缓存缺失/失配/损坏时明确重建；PrivateTest 仅在 include_test=True 时构建
    - 增强实现由 augmentation.impl 选择：'legacy' 使用逐样本 torchvision 管道
      （dataset.transform / class_aug_map）；'batched' 时 dataset 不带增强，
      批级张量增强由 Trainer 训练循环执行（PB01）
    - 类别平衡由训练器中的 Focal Loss 处理，无需显式计算 class_weights

    Args:
        config: 完整训练配置
        model_name: 模型名称，用于读取模型特定的 batch_size
        include_test: 是否构建 PrivateTest loader（训练入口可传 False 按需构建）

    Returns:
        train_loader, val_loader, test_loader（include_test=False 时为 None）, class_names
    """
    class_names = config["data"]["class_names"]
    use_cuda = torch.cuda.is_available()

    # ---- 数据来源：像素缓存（PB02）uint8 mmap + CSV SHA 校验（失配/损坏→重建）----
    dataset_path = PROJECT_ROOT / config["data"]["dataset_path"]
    pixel_cache = load_or_build(dataset_path)
    _x_tr, y_tr, _rows_tr = pixel_cache.split_arrays("Training")   # 触发文件校验/打开
    _x_va, y_va, _rows_va = pixel_cache.split_arrays("PublicTest")

    # 获取 batch_size
    if model_name and model_name in config.get("models", {}):
        batch_size = config["models"][model_name].get(
            "batch_size", config["training"]["batch_size"]
        )
    else:
        batch_size = config["training"]["batch_size"]

    # ---- 增强实现（PB01）：augmentation.impl ∈ {legacy, batched} ----
    aug_config = config.get("augmentation", {})
    aug_impl = aug_config.get("impl", "legacy")
    if aug_impl not in ("legacy", "batched"):
        raise ValueError(
            f"augmentation.impl 不支持: {aug_impl!r}（应为 'legacy' / 'batched'）"
        )
    train_transform = None
    class_aug_map: dict = {}
    if aug_impl == "legacy":
        # 逐样本增强管道（旧实现，保留用于对照）
        train_transform = build_train_transform(aug_config)
        # 类别特定增强（如 Disgust 类额外增强）；总开关关闭时整体停用
        class_aug_map = build_class_aug_transform(
            aug_config.get("class_specific", {}),
            master_enabled=aug_config.get("enabled", False),
        )
    else:
        print("  增强实现: batched（批级张量增强，由 Trainer 训练循环执行）")

    # ---- 数据集（仅持轻量缓存引用；spawn 时 pickle 不携带像素数据本体）----
    train_dataset = FER2013Dataset(
        transform=train_transform, class_aug_map=class_aug_map,
        cache_ref=(pixel_cache.cache_dir, "Training",
                   pixel_cache.meta["splits"]["Training"]),
    )
    val_dataset = FER2013Dataset(
        cache_ref=(pixel_cache.cache_dir, "PublicTest",
                   pixel_cache.meta["splits"]["PublicTest"]),
    )

    # 训练集类别计数（固定 0..num_classes-1 索引）：采样器与 CB Focal Loss 共用同一份统计
    num_classes = config["data"]["num_classes"]
    train_class_counts = compute_class_counts(y_tr, num_classes)
    train_dataset.class_counts = train_class_counts
    print(
        f"训练集类别计数 (索引 0-{num_classes - 1}): "
        f"{train_class_counts} | 合计 {sum(train_class_counts)}"
    )

    # 数据指纹（CSV SHA-256 + 各划分行号/标签哈希）：随 run_meta 记录，可复核追溯
    train_dataset.split_fingerprint = compute_split_fingerprint_cached(dataset_path, pixel_cache)

    # DataLoader 配置
    source_signatures = []
    for source in (Path(dataset_path), pixel_cache.cache_dir / "meta.json"):
        stat = source.stat()
        source_signatures.append({
            "path": str(source.resolve()), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns,
        })
    for dataset in (train_dataset, val_dataset):
        if dataset is not None and dataset._cache_ref is not None:
            dataset._cache_ref[2]["source_signatures"] = source_signatures

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

    # 训练 DataLoader：多进程取数 + 大 prefetch 缓冲吸收变换抖动
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

    # ---- 精确恢复（S01）：训练 sampler / DataLoader 使用独立 generator ----
    # 批次索引序列（sampler 抽取）与 worker 基础种子（DataLoader base_seed）
    # 由这两个 generator 驱动；其状态随 checkpoint 保存/恢复，使 workers=0 与
    # non-persistent 多 worker 的“连续 vs 恢复”逐批一致成为可复算行为。
    # persistent_workers=True 不提供精确恢复能力（worker 内部 RNG 进度跨会话不可见）。
    train_seed = config["seed"]
    g_sampler = torch.Generator().manual_seed(train_seed)
    g_loader = torch.Generator().manual_seed(train_seed + 12345)

    # Class-balanced sampling：用 WeightedRandomSampler 过采样少样本类
    # 与 Focal Loss 互补：sampler 负责数据分布，Focal Loss 负责梯度权重
    class_balanced = dl_config.get("class_balanced_sampling", False)
    train_sampler = None
    if class_balanced:
        weights = _compute_sampler_weights(
            y_tr, num_classes=num_classes, class_counts=train_class_counts
        )
        train_sampler = torch.utils.data.WeightedRandomSampler(
            weights, num_samples=len(weights), replacement=True, generator=g_sampler,
        )
        print(f"  class_balanced_sampling: enabled (min_weight={min(weights):.4f}, "
              f"max_weight={max(weights):.4f}, ratio={max(weights)/min(weights):.1f}x)")

    if train_workers > 0 and persistent_workers:
        print(
            "  ⚠️  persistent_workers=True：该配置不提供精确恢复能力"
            "（中断后续训无法逐批一致；如需精确恢复请关闭 persistent_workers）"
        )

    train_loader = DataLoader(
        train_dataset, shuffle=(train_sampler is None),
        sampler=train_sampler, generator=g_loader, **train_loader_kwargs,
    )
    val_loader = DataLoader(val_dataset, shuffle=False, **eval_loader_kwargs)

    test_loader = None
    if include_test:
        test_dataset = FER2013Dataset(
            cache_ref=(pixel_cache.cache_dir, "PrivateTest",
                       pixel_cache.meta["splits"]["PrivateTest"]),
        )
        if test_dataset._cache_ref is not None:
            test_dataset._cache_ref[2]["source_signatures"] = source_signatures
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


def compute_split_fingerprint_cached(csv_path, pixel_cache) -> dict:
    """
    数据指纹（像素缓存版）：CSV 文件 SHA-256 + 官方各划分的（行号‖标签）哈希。

    与 compute_split_fingerprint 公式完全一致（sha256(行号 int64-LE ‖ 标签 int64-LE)；
    行号 = CSV 数据行顺序，从 0 计），但直接读取缓存中的 labels/行号文件：
    不触碰 26 MiB 级像素文件、不打断 PrivateTest 的按需构建语义。
    """
    fingerprint: dict[str, Any] = {
        "protocol": "official-usage",
        "csv_path": str(csv_path),
        "csv_sha256": file_sha256_cached(csv_path),
        "splits": {},
    }
    for split in ("Training", "PublicTest", "PrivateTest"):
        entry = pixel_cache.meta["splits"][split]
        rows = np.load(pixel_cache.cache_dir / entry["rows"]["file"])
        labels = np.load(pixel_cache.cache_dir / entry["y"]["file"])
        sh = hashlib.sha256(
            np.asarray(rows, dtype=np.int64).tobytes()
            + np.asarray(labels, dtype=np.int64).tobytes()
        ).hexdigest()
        fingerprint["splits"][split] = {
            "rows": int(len(labels)),
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


def _compute_sampler_weights(labels, *, num_classes: int,
                             class_counts: list) -> list:
    """
    为 WeightedRandomSampler 计算每个样本的采样权重（逆频率加权）。

    与 CB Focal Loss 使用同一份 class_counts（compute_class_counts 统计，
    经 train_loader.dataset.class_counts 传入 Trainer），保证两种机制口径一致。
    零计数类别明确报错（不压缩类别索引、不静默跳过）。

    Args:
        labels: 训练集标签数组（顺序与数据集一致）
        num_classes: 类别总数（固定索引 0..num_classes-1）
        class_counts: compute_class_counts 的统计结果

    Returns:
        每个样本的权重列表，顺序与 labels 一致
    """
    zero_classes = [i for i, c in enumerate(class_counts) if c <= 0]
    if zero_classes:
        raise ValueError(
            f"训练集类别索引 {zero_classes} 样本数为 0，无法计算逆频率采样权重；"
            "请检查数据划分或关闭 class_balanced_sampling"
        )
    labels = np.asarray(labels)
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

"""
批级张量增强（PB01）— 训练循环内按批执行的向量化增强（CPU，float32）。

与 legacy 逐样本增强（data.dataloader.build_train_transform，torchvision 管道）的关系
（T05 更正）：**独立增强实现**——部分参数范围与门控概率沿用 legacy，但像素输出
分布不同（见下「batched-v1 冻结规则」），**不能称为「同分布」**，也不逐位等价。
随机性由每批确定性种子派生：seed_batch = sha256(train_seed|epoch|batch_idx)，
批内逐样本独立采样；同一 (epoch, batch_idx) 在连续训练与断点恢复中复算出同一批
增强参数（批次序列本身由 S01 的 sampler/loader generator 状态恢复保证）。

batched-v1 冻结规则（版本号对应以下行为；任何规则变更必须升级版本号——版本随训练
协议快照记录，切换实现版本会拒绝精确续训）：
  - 几何：flip → rotate → translate 合并为单次**双线性**仿射（affine_grid + grid_sample，
    align_corners=False、padding=0）；legacy v1 为 nearest 插值且旋转/平移各重采样一次。
  - 平移离散化：比例先 round 到整像素（对齐 v1 RandomAffine 的 int(round(...))）。
  - 颜色：brightness 乘性无 clamp；contrast 为逐样本均值混合并 clamp 到 [0,1]；
    两者先后顺序按 50/50 随机（对齐 v1 ColorJitter 的 randperm 行为）。
  - 擦除：基础 p=0.1 / 类别增强 p=extra；面积比、宽高比与最多 10 次尝试对齐
    v1 RandomErasing 语义。
  - 类别条件：目标类别以 augment_prob 门控，增强链 rot → translate → erase。

顺序（对齐 legacy Compose 顺序）：
  flip → rotate → translate → brightness/contrast（随机先后）→ erase(p=0.1)
  → 类别条件增强（rot → translate → erase(p=extra)，以 augment_prob 门控）。

输入输出约定：images 为 CPU 上的 (B,1,H,W) float32 张量、值域 [0,1]；
增强后亮度可轻微超出 [0,1]；调用应发生在 .to(device) 之前。
"""

from __future__ import annotations

import hashlib
import math

import torch
from torch import Tensor

__all__ = [
    "BATCH_AUG_VERSION",
    "BATCH_AUG_SEED_RULE",
    "BatchAugmenter",
    "build_batch_augmenter",
    "affine_batch",
]

BATCH_AUG_VERSION = "batched-v1"
BATCH_AUG_SEED_RULE = "sha256(train_seed|epoch|batch_idx)"

# 与 legacy RandomErasing(p=0.1) / v1 默认参数一致
_ERASE_P = 0.1
_ERASE_AREA = (0.02, 0.33)
_ERASE_ASPECT = (0.3, 3.3)
_ERASE_MAX_TRIES = 10


def _batch_seed(train_seed: int, epoch: int, batch_idx: int) -> int:
    """批级确定性种子（见 BATCH_AUG_SEED_RULE）。"""
    key = f"{int(train_seed)}|{int(epoch)}|{int(batch_idx)}".encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:8], "big")


def _contrast_blend(x: Tensor, factor: Tensor) -> Tensor:
    """
    单通道对比度混合（对齐 v1 adjust_contrast 的 _blend 语义）：
    (f * x + (1 - f) * mean) clamp 到 [0, 1]，mean 为逐样本全图均值。
    factor: (B,)
    """
    mean = x.mean(dim=(1, 2, 3), keepdim=True)
    f = factor[:, None, None, None]
    return (f * x + (1.0 - f) * mean).clamp_(0.0, 1.0)


def affine_batch(x: Tensor, angle_deg: Tensor, tx: Tensor, ty: Tensor) -> Tensor:
    """
    对 (B,1,H,W) 施加逐样本仿射（旋转 + 平移），边界填 0（与 v1 fill=0 一致）。

    与 legacy 的数学对齐（读 torchvision 0.29 源码后按 v1 公式实现）：
    - 旋转：对齐 F.rotate(+angle) 的采样矩阵（RandomRotation 路径；
      已知其与 F.affine(+angle) 的角度约定相反，F.rotate 内部传 -angle）；
      非方图交叉项按 v1 的 h/w 缩放行为复制。
    - 平移：对齐 RandomAffine(degrees=0, translate) 路径——比例先 round 到整像素
      （v1: int(round(uniform(-max_dx, max_dx)))），归一化坐标偏移 = -2 * 比例。
    - 组合（rotate 后 translate 的采样链 S_rot∘S_tr）：平移分量经旋转矩阵变换
      （A = [R | R @ b1]），对齐"先 RandomRotation 后 RandomAffine"的复合效果。

    Args:
        angle_deg: (B,) 旋转角度（度）
        tx, ty: (B,) 平移（图像宽/高的比例）
    """
    b, _c, h, w = x.shape
    if (
        float(angle_deg.abs().max()) < 1e-12
        and float(tx.abs().max()) < 1e-12
        and float(ty.abs().max()) < 1e-12
    ):
        return x

    rad = angle_deg * (math.pi / 180.0)
    cos, sin = torch.cos(rad), torch.sin(rad)

    # 旋转采样矩阵（对齐 F.rotate(+angle)；方图时 = [[cos,-sin],[sin,cos]]）
    a00 = cos
    a01 = -sin * (h / w)
    a10 = sin * (w / h)
    a11 = cos

    # 平移分量（对齐 F.affine(angle=0, translate)；先 round 到整像素）
    tq_x = torch.round(tx * w) / w
    tq_y = torch.round(ty * h) / h
    b1x = -2.0 * tq_x
    b1y = -2.0 * tq_y

    # 组合：平移分量经旋转矩阵变换
    bx = a00 * b1x + a01 * b1y
    by = a10 * b1x + a11 * b1y

    theta = torch.zeros(b, 2, 3, dtype=x.dtype)
    theta[:, 0, 0] = a00
    theta[:, 0, 1] = a01
    theta[:, 1, 0] = a10
    theta[:, 1, 1] = a11
    theta[:, 0, 2] = bx
    theta[:, 1, 2] = by

    grid = torch.nn.functional.affine_grid(theta, list(x.shape), align_corners=False)
    return torch.nn.functional.grid_sample(
        x, grid, mode="bilinear", padding_mode="zeros", align_corners=False
    )


def _apply_erase(x: Tensor, mask: Tensor, g: torch.Generator) -> Tensor:
    """
    对 mask=True 的样本擦除随机矩形（v1 RandomErasing 语义：面积比/宽高比/最多 10 次尝试）。

    不修改输入：有擦除发生时返回克隆后的新张量；无擦除时原样返回。
    """
    idxs = torch.nonzero(mask).flatten().tolist()
    if not idxs:
        return x
    out = x.clone()
    _b, _c, h, w = out.shape
    for i in idxs:
        for _ in range(_ERASE_MAX_TRIES):
            area = float(torch.empty(1).uniform_(_ERASE_AREA[0], _ERASE_AREA[1], generator=g))
            # 宽高比对数均匀（等价于 uniform(log(1/3.3), log(3.3))）
            log_ratio = float(torch.empty(1).uniform_(
                math.log(1.0 / _ERASE_ASPECT[1]), math.log(_ERASE_ASPECT[1]), generator=g
            ))
            aspect = math.exp(log_ratio)
            eh = int(round(math.sqrt(area * h * w * aspect)))
            ew = int(round(math.sqrt(area * h * w / aspect)))
            if eh < h and ew < w:
                top = int(torch.randint(0, h - eh + 1, (1,), generator=g))
                left = int(torch.randint(0, w - ew + 1, (1,), generator=g))
                out[i, :, top: top + eh, left: left + ew] = 0.0
                break
    return out


class BatchAugmenter:
    """
    批级增强器（详见模块 docstring）。

    调用约定: augmenter(images, labels, epoch=e, batch_idx=i) → 增强后的 images。
    labels 用于类别条件增强（与 legacy class_aug_map 相同的目标类别与概率）。
    """

    def __init__(self, aug_config: dict, seed: int):
        aug = aug_config or {}
        self.seed = int(seed)
        self.flip_p = float(aug.get("random_horizontal_flip", 0) or 0)
        self.rot_deg = float(aug.get("random_rotation", 0) or 0)
        self.translate = float(aug.get("random_affine_translate", 0) or 0)
        self.brightness = float(aug.get("color_jitter_brightness", 0) or 0)
        self.contrast = float(aug.get("color_jitter_contrast", 0) or 0)
        self.erase = bool(aug.get("random_erase", False))

        cls = (aug.get("class_specific", {}) or {})
        self.cls_enabled = bool(cls.get("enabled", False))
        self.cls_targets = [int(c) for c in (cls.get("target_classes", []) or [])]
        self.cls_prob = float(cls.get("augment_prob", 0.8))
        self.cls_rot_deg = float(cls.get("extra_rotation", 0) or 0)
        self.cls_translate = float(cls.get("extra_translate", 0) or 0)
        self.cls_erase_p = float(cls.get("extra_erase_prob", 0) or 0)

    # ---- 随机采样（独立方法便于测试；消耗顺序恒定即为确定性来源）----
    def _batch_generator(self, epoch: int, batch_idx: int) -> torch.Generator:
        return torch.Generator().manual_seed(_batch_seed(self.seed, epoch, batch_idx))

    def _sample_params(self, batch: int, g: torch.Generator) -> dict:
        """批内逐样本独立采样基础增强参数（消耗序固定）。"""
        params: dict = {}
        params["flip"] = (
            torch.rand(batch, generator=g) < self.flip_p if self.flip_p > 0 else None
        )
        params["angle"] = (
            torch.empty(batch).uniform_(-self.rot_deg, self.rot_deg, generator=g)
            if self.rot_deg > 0 else None
        )
        if self.translate > 0:
            params["tx"] = torch.empty(batch).uniform_(-self.translate, self.translate, generator=g)
            params["ty"] = torch.empty(batch).uniform_(-self.translate, self.translate, generator=g)
        else:
            params["tx"] = params["ty"] = None
        params["bright"] = (
            torch.empty(batch).uniform_(1.0 - self.brightness, 1.0 + self.brightness, generator=g)
            if self.brightness > 0 else None
        )
        params["contrast"] = (
            torch.empty(batch).uniform_(1.0 - self.contrast, 1.0 + self.contrast, generator=g)
            if self.contrast > 0 else None
        )
        params["erase"] = (
            torch.rand(batch, generator=g) < _ERASE_P if self.erase else None
        )
        params["color_order"] = (
            torch.rand(batch, generator=g) < 0.5
            if (params["bright"] is not None and params["contrast"] is not None) else None
        )
        return params

    def __call__(self, images: Tensor, labels: Tensor, *, epoch: int, batch_idx: int) -> Tensor:
        if images.ndim != 4 or images.shape[1] != 1:
            raise ValueError(f"BatchAugmenter 期望 (B,1,H,W) 张量，得到 {tuple(images.shape)}")
        if images.device.type != "cpu":
            raise ValueError("批级增强在 CPU 上执行（应在 .to(device) 之前调用）")

        g = self._batch_generator(epoch, batch_idx)
        batch = images.shape[0]
        x = images
        p = self._sample_params(batch, g)

        # ---- 应用（顺序对齐 legacy Compose）----
        # 1) 水平翻转
        if p["flip"] is not None:
            x = torch.where(p["flip"][:, None, None, None], x.flip(-1), x)

        # 2) 旋转 + 3) 平移（合成单次仿射）
        if p["angle"] is not None or p["tx"] is not None:
            zeros = torch.zeros(batch)
            x = affine_batch(
                x,
                p["angle"] if p["angle"] is not None else zeros,
                p["tx"] if p["tx"] is not None else zeros,
                p["ty"] if p["ty"] is not None else zeros,
            )

        # 4) 亮度/对比度（50/50 随机先后，对齐 v1 ColorJitter randperm）
        if p["bright"] is not None and p["contrast"] is not None:
            b4 = p["bright"][:, None, None, None]
            x_bc = _contrast_blend(x * b4, p["contrast"])            # brightness → contrast
            x_cb = _contrast_blend(x, p["contrast"]) * b4            # contrast → brightness
            x = torch.where(p["color_order"][:, None, None, None], x_bc, x_cb)
        elif p["bright"] is not None:
            x = x * p["bright"][:, None, None, None]
        elif p["contrast"] is not None:
            x = _contrast_blend(x, p["contrast"])

        # 5) 随机擦除
        if p["erase"] is not None:
            x = _apply_erase(x, p["erase"], g)

        # 6) 类别条件增强（rot → translate → erase；augment_prob 门控）
        if self.cls_enabled and self.cls_targets:
            cls_gate = torch.rand(batch, generator=g) < self.cls_prob
            c_angle = (
                torch.empty(batch).uniform_(-self.cls_rot_deg, self.cls_rot_deg, generator=g)
                if self.cls_rot_deg > 0 else None
            )
            c_tx: Tensor | None = None
            c_ty: Tensor | None = None
            if self.cls_translate > 0:
                c_tx = torch.empty(batch).uniform_(
                    -self.cls_translate, self.cls_translate, generator=g
                )
                c_ty = torch.empty(batch).uniform_(
                    -self.cls_translate, self.cls_translate, generator=g
                )
            c_erase_rand = (
                torch.rand(batch, generator=g) if self.cls_erase_p > 0 else None
            )

            in_targets = torch.isin(labels, torch.tensor(self.cls_targets, dtype=labels.dtype))
            cls_mask = cls_gate & in_targets
            if bool(cls_mask.any()):
                if c_angle is not None or c_tx is not None:
                    zeros = torch.zeros(batch)
                    x2 = affine_batch(
                        x,
                        c_angle if c_angle is not None else zeros,
                        c_tx if c_tx is not None else zeros,
                        c_ty if c_ty is not None else zeros,
                    )
                    x = torch.where(cls_mask[:, None, None, None], x2, x)
                if c_erase_rand is not None:
                    x = _apply_erase(x, cls_mask & (c_erase_rand < self.cls_erase_p), g)

        return x


def build_batch_augmenter(aug_config: dict, seed: int) -> BatchAugmenter | None:
    """按配置构建批级增强器；augmentation.enabled=False 时返回 None（不做任何增强）。"""
    if not (aug_config or {}).get("enabled", False):
        return None
    return BatchAugmenter(aug_config, seed=seed)

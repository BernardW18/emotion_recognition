"""
损失函数模块 — Focal Loss + Class-Balanced Focal Loss

理论: FL(p_t) = -(1 - p_t)^gamma * log(p_t)
  - gamma > 0 降低易分类样本的 loss 贡献
  - 模型自然聚焦于"难的"样本，无需显式类别加权
  - gamma=2.0 对 FER2013 类长尾分布效果较好

Class-Balanced Focal Loss（CVPR 2019）：
  - 基于有效样本数理论 E_n = (1 - beta^n) / (1 - beta)
  - CB 权重 = (1 - beta) / (1 - beta^class_count)
  - beta=0.9 / 0.99 / 0.999 控制"有效样本数"的衰减速率
  - beta=0.999 对 FER2013 的极度长尾（Disgust 436 样本）效果较好

用法:
    from utils.losses import FocalLoss, CBFocalLoss
    criterion = FocalLoss(gamma=2.0)
    cb_criterion = CBFocalLoss(gamma=2.0, beta=0.999, class_counts=[4953, 436, ...])
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["FocalLoss", "CBFocalLoss"]


class FocalLoss(nn.Module):
    """
    Focal Loss

    相比 CrossEntropyLoss(weight=...) 的优势：
    - 无硬性类别权重缩放，不产生极端梯度
    - 动态调整：已学会的样本 loss 自动衰减，模型聚焦于困难样本
    - gamma 参数控制衰减速率，简洁易调

    Args:
        gamma: 聚焦参数，>= 0。默认 2.0。
               gamma=0 退化为标准 CrossEntropyLoss。
               gamma 越大，易分类样本的 loss 贡献越低。
        reduction: 'mean' | 'sum' | 'none'。默认 'mean'。
    """

    def __init__(self, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction

    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            inputs: (N, C) logits，未经 softmax
            targets: (N,) 类别索引

        Returns:
            loss: scalar 或 (N,) 取决于 reduction
        """
        # 标准交叉熵（逐样本，reduction='none'）
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")

        if self.gamma == 0.0:
            # 退化为标准 CE
            loss = ce_loss
        else:
            # p_t = exp(-CE) 即 softmax 对 target 类的概率
            pt = torch.exp(-ce_loss)
            # FL = (1 - p_t)^gamma * CE
            loss = (1 - pt) ** self.gamma * ce_loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class CBFocalLoss(nn.Module):
    """
    Class-Balanced Focal Loss（CVPR 2019）

    在 Focal Loss 的基础上，结合 Class-Balanced 理论：
      CB_FL = (1 - beta) / (1 - beta^n_j) * FL(p_t)

    其中 n_j 是类别 j 的样本数。少样本类获得更高的 loss 权重，
    但同时 Focal Loss 的动态调节机制仍在工作。

    Args:
        gamma: Focal Loss 聚焦参数。默认 2.0。
        beta: 有效样本数衰减因子，0 <= beta < 1。
              越接近 1，"有效样本数"越接近真实样本数。
              FER2013 建议 beta=0.999（Disgust 436 样本的影响显著）。
        class_counts: 各类别的样本数列表，顺序与 CLASS_NAMES 一致。
                      例如 [4953, 436, 5121, 8989, 6077, 4002, 6198]。
        reduction: 'mean' | 'sum' | 'none'。默认 'mean'。
    """

    def __init__(
        self,
        gamma: float = 2.0,
        beta: float = 0.999,
        class_counts: list = None,
        reduction: str = "mean",
    ):
        super().__init__()
        if beta < 0 or beta >= 1:
            raise ValueError(f"beta 必须在 [0, 1) 范围内，当前 beta={beta}")
        if class_counts is None or len(class_counts) == 0:
            raise ValueError("class_counts 不能为空，请提供各类别的样本数列表")

        self.gamma = gamma
        self.beta = beta
        self.reduction = reduction

        # 预计算各类别的 CB 权重
        # CB_weight[j] = (1 - beta) / (1 - beta^n_j)
        n_j = torch.tensor(class_counts, dtype=torch.float32)
        raw_weights = (1 - beta) / (1 - beta ** n_j)
        # 归一化：让权重的频率加权均值为 1.0
        # 避免原始 CB 权重均值仅 ~0.001 导致 loss 量级缩小千倍、有效 LR 骤降
        freq = n_j / n_j.sum()
        mean_weight = (raw_weights * freq).sum()
        normalized_weights = raw_weights / mean_weight
        self.register_buffer("_cb_weights", normalized_weights)

    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            inputs: (N, C) logits，未经 softmax
            targets: (N,) 类别索引

        Returns:
            loss: scalar 或 (N,) 取决于 reduction
        """
        # 标准交叉熵（逐样本）
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")

        # Focal Loss 调制
        if self.gamma > 0:
            pt = torch.exp(-ce_loss)
            focal_weight = (1 - pt) ** self.gamma
        else:
            focal_weight = 1.0

        # CB 权重：每个样本根据其类别获取对应的 CB 权重
        # 将 _cb_weights 移到与 targets 相同的设备（CUDA/CPU）
        cb_weights = self._cb_weights.to(targets.device)
        cb_weight = cb_weights[targets]

        # 组合：CB_FL = cb_weight * focal_weight * CE
        loss = cb_weight * focal_weight * ce_loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss

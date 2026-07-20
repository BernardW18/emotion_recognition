"""
VGGLite - VGG变体
5层卷积（2-2-1分组）+ 全连接，参数量约 1.5M
核心理念：小卷积核堆叠替代大卷积核，增加非线性表达能力
输入：1 x 48 x 48 灰度图
输出：7 类情感
"""

import torch
import torch.nn as nn
from utils.activations import get_activation


class VGGLite(nn.Module):
    def __init__(self, num_classes: int = 7, dropout: float = 0.5, activation: str = "relu"):
        """
        Args:
            num_classes: 分类类别数
            dropout: Dropout 概率
            activation: 激活函数名称，支持 relu / leaky_relu / elu / gelu
        """
        super().__init__()
        act = get_activation(activation)

        self.features = nn.Sequential(
            # Block 1: 48 -> 24, 2 x Conv3-64
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            act,
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            act,
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.1),

            # Block 2: 24 -> 12, 2 x Conv3-128
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            act,
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            act,
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.1),

            # Block 3: 12 -> 6, 1 x Conv3-256
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            act,
            nn.MaxPool2d(2, 2),
        )

        # 动态计算展平维度
        with torch.no_grad():
            dummy = torch.zeros(1, 1, 48, 48)
            feat_dim = self.features(dummy).numel()

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feat_dim, 512),
            act,
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            act,
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x

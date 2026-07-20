"""
MicroResNet - 微型残差网络
4个残差块 + 全局平均池化，参数量约 400K
核心理念：残差连接解决梯度消失，全局平均池化减少参数
输入：1 x 48 x 48 灰度图
输出：7 类情感
"""

import torch
import torch.nn as nn
from utils.activations import get_activation


class SEBlock(nn.Module):
    """Squeeze-and-Excitation 通道注意力模块

    对每个通道计算重要性权重（0~1），与输入逐通道相乘。
    参数量: 2 * C * C/r（约 2K/block @ C=64, r=8）
    """

    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        reduced = max(channels // reduction, 4)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, reduced),
            nn.ReLU(inplace=True),
            nn.Linear(reduced, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.se(x).view(x.size(0), -1, 1, 1)
        return x * scale


class ResidualBlock(nn.Module):
    """基础残差块：两个3x3卷积 + skip connection + 可选 SE"""

    def __init__(self, channels: int, dropout: float = 0.1,
                 activation: str = "relu", use_se: bool = False):
        """
        Args:
            channels: 输入/输出通道数
            dropout: Dropout2d 概率
            activation: 激活函数名称
            use_se: 是否在残差连接后添加 SE 通道注意力
        """
        super().__init__()
        act = get_activation(activation)

        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.act = act
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.dropout = nn.Dropout2d(dropout)
        self.se = SEBlock(channels) if use_se else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.dropout(out)
        out += residual
        out = self.act(out)
        if self.se is not None:
            out = self.se(out)
        return out


class MicroResNet(nn.Module):
    def __init__(self, num_classes: int = 7, dropout: float = 0.3,
                 activation: str = "relu", use_se: bool = False):
        """
        Args:
            num_classes: 分类类别数
            dropout: 分类器 Dropout 概率
            activation: 激活函数名称，支持 relu / leaky_relu / elu / gelu
        """
        super().__init__()
        act = get_activation(activation)

        # Stem: 48 -> 24
        self.stem = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            act,
            nn.MaxPool2d(2, 2),
        )

        # Stage 1: 24 -> 12, 64 channels
        self.downsample1 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1),
            nn.AvgPool2d(2, 2),
        )
        self.stage1 = nn.Sequential(
            ResidualBlock(64, activation=activation, use_se=use_se),
            ResidualBlock(64, activation=activation, use_se=use_se),
        )

        # Stage 2: 12 -> 6, 128 channels
        self.channel_expand2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=1),
            nn.BatchNorm2d(128),
        )
        self.downsample2 = nn.AvgPool2d(2, 2)
        self.stage2 = nn.Sequential(
            ResidualBlock(128, activation=activation, use_se=use_se),
            ResidualBlock(128, activation=activation, use_se=use_se),
        )

        # Global Average Pooling + Classifier
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)

        # Stage 1: downsample then residual blocks
        x = self.downsample1(x)
        x = self.stage1(x)

        # Stage 2: expand channels, downsample, then residual blocks
        x = self.channel_expand2(x)
        x = self.downsample2(x)
        x = self.stage2(x)

        x = self.gap(x)
        x = self.classifier(x)
        return x

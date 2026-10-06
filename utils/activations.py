"""
公共工具模块 — 激活函数工厂

从 training/trainer.py 抽取，消除 models → training 的循环依赖。
所有模型层、训练层、推理层均可安全引用此模块。
"""

__all__ = ["ACTIVATION_REGISTRY", "get_activation"]

import torch.nn as nn

ACTIVATION_REGISTRY: dict[str, type[nn.Module]] = {
    "relu": nn.ReLU,
    "leaky_relu": nn.LeakyReLU,
    "elu": nn.ELU,
    "gelu": nn.GELU,
}


def get_activation(name: str = "relu", **kwargs) -> nn.Module:
    """
    获取激活函数模块

    Args:
        name: 激活函数名称，支持 relu / leaky_relu / elu / gelu
        **kwargs: 传递给激活函数的额外参数（如 negative_slope=0.01 for LeakyReLU）

    Returns:
        nn.Module 实例

    Raises:
        ValueError: 不支持的激活函数名称
    """
    name = name.lower()
    if name not in ACTIVATION_REGISTRY:
        supported = list(ACTIVATION_REGISTRY.keys())
        raise ValueError(f"不支持的激活函数: {name}，可选: {supported}")
    return ACTIVATION_REGISTRY[name](**kwargs)

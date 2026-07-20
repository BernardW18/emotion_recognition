"""
推理工具函数：图像预处理、模型加载、推理、Grad-CAM热力图
供 Streamlit app.py 调用
"""

__all__ = [
    "MODEL_REGISTRY", "CLASS_NAMES", "CLASS_EMOJIS",
    "get_model_class", "load_model", "preprocess_image",
    "predict", "get_available_models", "generate_gradcam",
    "get_model_info",
]

import sys
from pathlib import Path

# 确保项目根目录在 sys.path 中（自包含，适用于任何工作目录）
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image

# 从公共常量模块导入，确保单一定义
from utils.constants import CLASS_NAMES, CLASS_EMOJIS, EMOTION_COUNT

# 项目根目录（与 sys.path 设置共用）
PROJECT_ROOT = _PROJECT_ROOT

# 模型名称到类的映射
MODEL_REGISTRY = {
    "mini_cnn": "models.mini_cnn:MiniCNN",
    "vgg_lite": "models.vgg_lite:VGGLite",
    "micro_resnet": "models.micro_resnet:MicroResNet",
}


def get_model_class(model_name: str):
    """根据模型名称动态导入模型类"""
    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"未知模型: {model_name}，可选: {list(MODEL_REGISTRY.keys())}"
        )
    module_path, class_name = MODEL_REGISTRY[model_name].rsplit(":", 1)
    module = __import__(module_path, fromlist=[class_name])
    return getattr(module, class_name)


def load_model(model_name: str, checkpoint_path: str = None, device: str = None):
    """
    加载训练好的模型

    Args:
        model_name: 模型名称 (mini_cnn / vgg_lite / micro_resnet)
        checkpoint_path: checkpoint路径，默认使用saved_models/{model_name}_best.pth
        device: 计算设备

    Returns:
        model: 加载好权重的模型（eval模式）
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if checkpoint_path is None:
        checkpoint_path = (
            PROJECT_ROOT / "inference" / "saved_models" / f"{model_name}_best.pth"
        )
    else:
        checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint未找到: {checkpoint_path}\n"
            f"请先运行训练Notebook生成模型文件"
        )

    # 构建模型
    ModelClass = get_model_class(model_name)
    model = ModelClass(num_classes=len(CLASS_NAMES))

    # 加载权重
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model, device


def preprocess_image(image: Image.Image) -> torch.Tensor:
    """
    将PIL图像预处理为模型输入张量

    Args:
        image: PIL Image对象（任意尺寸、任意模式）

    Returns:
        tensor: 形状为 (1, 1, 48, 48) 的归一化张量
    """
    # 转灰度
    if image.mode != "L":
        image = image.convert("L")

    # 缩放到48x48
    image = image.resize((48, 48), Image.Resampling.BILINEAR)

    # 转numpy -> 归一化 -> 转tensor
    img_array = np.array(image, dtype=np.float32) / 255.0
    tensor = torch.FloatTensor(img_array).unsqueeze(0).unsqueeze(0)  # (1, 1, 48, 48)

    return tensor


def predict(model: nn.Module, image: Image.Image, device: str = None) -> dict:
    """
    对单张图像进行情感预测

    Args:
        model: 加载好权重的模型
        image: PIL Image对象
        device: 计算设备

    Returns:
        result: {
            "emotion": 预测类别名,
            "emoji": 对应emoji,
            "confidence": 预测置信度,
            "probabilities": {类别名: 概率}
        }
    """
    if device is None:
        device = next(model.parameters()).device

    tensor = preprocess_image(image).to(device)

    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1).squeeze(0).cpu().numpy()

    predicted_idx = int(np.argmax(probs))
    predicted_class = CLASS_NAMES[predicted_idx]
    confidence = float(probs[predicted_idx])

    probabilities = {name: float(prob) for name, prob in zip(CLASS_NAMES, probs)}

    return {
        "emotion": predicted_class,
        "emoji": CLASS_EMOJIS[predicted_class],
        "confidence": confidence,
        "probabilities": probabilities,
    }


def get_available_models() -> list:
    """扫描saved_models目录，返回可用的模型列表"""
    saved_dir = PROJECT_ROOT / "inference" / "saved_models"
    available = []
    if saved_dir.exists():
        for path in sorted(saved_dir.glob("*.pth")):
            # 从文件名提取模型名: xxx_best.pth -> xxx
            model_name = path.stem.replace("_best", "")
            if model_name in MODEL_REGISTRY:
                available.append(model_name)
    return available


def get_model_info(model_name: str) -> dict | None:
    """
    从 checkpoint 读取模型元数据（精度、参数量等），不加载完整权重。

    Args:
        model_name: 模型名称 (mini_cnn / vgg_lite / micro_resnet)

    Returns:
        info dict 或 None（checkpoint 不存在时）
        {
            "val_acc": float,           # 保存时的验证准确率
            "global_best_val_acc": float, # 跨会话全局最佳
            "trained_epochs": int,      # 已训练轮数
            "params_str": str,          # 参数量描述 (~50K / ~1.5M / ~400K)
            "training_duration": str,   # 训练耗时 (如 1h23m)
        }
    """
    checkpoint_path = (
        PROJECT_ROOT / "inference" / "saved_models" / f"{model_name}_best.pth"
    )
    if not checkpoint_path.exists():
        return None

    # 轻量读取 checkpoint 元数据（不加载完整模型权重到 GPU）
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    history = ckpt.get("history", {})

    trained_epochs = len(history.get("val_acc", [])) if history else 0
    global_best = ckpt.get("global_best_val_acc", 0.0)
    val_acc = ckpt.get("val_acc", 0.0)
    duration_s = ckpt.get("training_duration_seconds", 0.0)

    # 参数量（与训练时一致的模型架构对应）
    param_info = {
        "mini_cnn": "~50K",
        "vgg_lite": "~1.5M",
        "micro_resnet": "~400K",
    }

    # 格式化训练耗时
    if duration_s > 3600:
        dur = f"{int(duration_s // 3600)}h{int((duration_s % 3600) // 60):02d}m"
    elif duration_s > 60:
        dur = f"{int(duration_s // 60)}m{int(duration_s % 60):02d}s"
    else:
        dur = f"{int(duration_s)}s"

    return {
        "val_acc": val_acc,
        "global_best_val_acc": global_best,
        "trained_epochs": trained_epochs,
        "params_str": param_info.get(model_name, "?"),
        "training_duration": dur,
    }


# ============================================================
# Grad-CAM 热力图（纯手写实现，无第三方依赖）
# ============================================================

def _find_last_conv_layer(model: nn.Module) -> nn.Module:
    """
    自动找到模型中最后一个卷积层

    Args:
        model: PyTorch 模型

    Returns:
        最后一个 Conv2d 模块
    """
    last_conv = None
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            last_conv = module
    if last_conv is None:
        raise ValueError("模型中未找到卷积层")
    return last_conv


def generate_gradcam(
    model: nn.Module,
    image: Image.Image,
    device: str = None,
    target_class: int = None,
    target_layer: nn.Module = None,
) -> np.ndarray:
    """
    生成 Grad-CAM 热力图

    Args:
        model: 加载好权重的模型（eval模式）
        image: PIL Image对象
        device: 计算设备
        target_class: 目标类别索引，None则使用模型预测的类别
        target_layer: 目标卷积层，None则自动选择最后一个卷积层

    Returns:
        heatmap: numpy array (H, W)，值域 [0, 1]，可直接用 matplotlib 展示
    """
    if device is None:
        device = next(model.parameters()).device

    # 自动选择目标卷积层
    if target_layer is None:
        target_layer = _find_last_conv_layer(model)

    # 前向传播获取目标类别
    tensor = preprocess_image(image).to(device)
    model.eval()

    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1)
    predicted_class = int(torch.argmax(probs))

    if target_class is None:
        target_class = predicted_class

    # Grad-CAM: 注册 hook 捕获梯度和特征图
    gradients = []
    activations = []

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    def forward_hook(module, input, output):
        activations.append(output)

    handles = [
        target_layer.register_forward_hook(forward_hook),
        target_layer.register_full_backward_hook(backward_hook),
    ]

    # 前向 + 反向
    model.zero_grad()
    outputs = model(tensor)
    loss = outputs[0, target_class]
    loss.backward()

    # 移除 hooks
    for h in handles:
        h.remove()

    # 计算 Grad-CAM
    grads = gradients[0]  # (1, C, H', W')
    acts = activations[0]  # (1, C, H', W')

    # 全局平均池化梯度 → 每个通道的权重
    weights = grads.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)

    # 加权求和
    cam = (weights * acts).sum(dim=1, keepdim=True)  # (1, 1, H', W')
    cam = F.relu(cam)

    # 归一化到 [0, 1]
    cam = cam.squeeze().cpu().detach().numpy()
    cam_min, cam_max = cam.min(), cam.max()
    if cam_max - cam_min > 1e-8:
        cam = (cam - cam_min) / (cam_max - cam_min)
    else:
        cam = np.zeros_like(cam)

    # 上采样到 48x48
    heatmap = np.array(Image.fromarray((cam * 255).astype(np.uint8)).resize((48, 48), Image.Resampling.BILINEAR))
    heatmap = heatmap.astype(np.float32) / 255.0

    return heatmap

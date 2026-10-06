"""
推理工具：统一模型加载（F01）、预处理、单张推理、Grad-CAM 热力图（F17）

模型加载一律经由 utils/model_spec.load_model_from_checkpoint —— 与训练导出、
离线评估共用同一构造逻辑，保证「应用加载的模型结构 == 训练时保存的结构」。
旧格式权重按确定性迁移规则解析（见 utils/model_spec.py 的迁移说明）。
"""

__all__ = [
    "CLASS_NAMES", "CLASS_EMOJIS", "SAVED_MODELS_DIR",
    "load_model", "preprocess_image", "predict", "generate_gradcam",
    "list_available_checkpoints", "describe_checkpoint", "read_export_manifest",
]

import json
import logging
import sys
from pathlib import Path

# 确保项目根目录在 sys.path 中（自包含，适用于任何工作目录）
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

# 从公共常量模块导入，确保单一定义
from utils.constants import CLASS_EMOJIS, CLASS_NAMES
from utils.model_spec import (
    build_model_from_spec,
    count_parameters,
    file_sha256,
    load_model_from_checkpoint,
    resolve_spec_from_checkpoint,
)

PROJECT_ROOT = _PROJECT_ROOT
SAVED_MODELS_DIR = PROJECT_ROOT / "inference" / "saved_models"
MANIFEST_PATH = SAVED_MODELS_DIR / "export_manifest.json"

logger = logging.getLogger("infer_utils")


# ============================================================
# 权重清单与信息
# ============================================================
def read_export_manifest() -> dict:
    """读取导出清单（记录每个导出权重的来源 run / SHA-256 / 指标 / spec）。"""
    if MANIFEST_PATH.exists():
        try:
            with open(MANIFEST_PATH, encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict) and isinstance(data.get("exports"), list):
                return data
        except Exception as e:
            logger.warning("export_manifest.json 读取失败: %s", e)
    return {"exports": []}


def list_available_checkpoints() -> list[dict]:
    """
    列出推理目录下全部权重文件（含导出清单来源信息与 legacy 标记）。

    Returns:
        [{"file", "path", "label", "model_name", "legacy", "source"}]
    """
    manifest = read_export_manifest()
    by_file = {e.get("file"): e for e in manifest["exports"]}
    items: list[dict] = []
    if not SAVED_MODELS_DIR.exists():
        return items
    for path in sorted(SAVED_MODELS_DIR.glob("*.pth")):
        entry = by_file.get(path.name)
        legacy = entry is None or bool(entry.get("legacy"))
        model_name = (entry or {}).get("model_name")
        if not model_name:
            for cand in ("mini_cnn", "vgg_lite", "micro_resnet"):
                if path.stem.startswith(cand):
                    model_name = cand
                    break
            model_name = model_name or path.stem

        if entry:
            m = entry.get("metrics") or {}
            acc = m.get("best_val_acc") or m.get("val_acc") or 0.0
            label = f"{model_name} | {path.stem} | run={entry.get('run_id', '?')}"
            if acc:
                label += f" | val_acc={acc * 100:.2f}%"
            if legacy:
                label += " | [legacy 旧产物]"
        else:
            label = f"{model_name} | {path.stem} | [无来源记录·旧产物]"

        items.append({
            "file": path.name,
            "path": str(path),
            "label": label,
            "model_name": model_name,
            "legacy": legacy,
            "source": entry,
        })
    return items


def describe_checkpoint(path) -> dict:
    """
    读取任意权重文件的可追溯信息：model_spec（含迁移 notes）、参数量、指标、SHA-256。
    仅读取元数据并构造模型计数参数（不加载权重）。
    """
    path = Path(path)
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    spec, notes = resolve_spec_from_checkpoint(ckpt)
    probe_model = build_model_from_spec(spec, allow_unknown_dropout=True)
    total, trainable = count_parameters(probe_model)

    metrics = {"epoch": ckpt.get("epoch"), "val_acc": ckpt.get("val_acc")}
    if isinstance(ckpt.get("best"), dict):
        metrics["best_val_acc"] = ckpt["best"].get("val_acc")
        metrics["monitor_metric"] = ckpt["best"].get("monitor_metric")
    elif "best_val_acc" in ckpt:
        metrics["best_val_acc"] = ckpt["best_val_acc"]

    return {
        "path": str(path),
        "sha256": file_sha256(path),
        "model_spec": spec.to_dict(),
        "spec_notes": notes,
        "params_total": total,
        "params_trainable": trainable,
        "metrics": metrics,
        "format_version": ckpt.get("format_version"),
        "legacy": "model_spec" not in ckpt,
        "run_id": ckpt.get("run_id"),
    }


# ============================================================
# 模型加载（统一入口）
# ============================================================
def load_model(checkpoint_path, device: str | None = None):
    """
    加载模型（统一构造入口，与训练/离线评估一致）。

    Args:
        checkpoint_path: 权重文件路径
        device: 计算设备；None 时自动选择

    Returns:
        (model[eval], device, meta) —— meta 含 model_spec / SHA-256 / 迁移 notes
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model, spec, meta = load_model_from_checkpoint(checkpoint_path, device=device)
    return model, device, meta


# ============================================================
# 预处理与推理
# ============================================================
def preprocess_image(image: Image.Image) -> torch.Tensor:
    """
    将 PIL 图像预处理为模型输入张量（与训练 / 离线评估一致）：
    灰度 → 48×48（已是 48×48 时不重采样，保持逐像素不变）→ x/255 → (1, 1, 48, 48)

    输入要求：已裁剪的单张人脸图像（训练域为 48×48 灰度；整张生活照不适合）。
    """
    if image.mode != "L":
        image = image.convert("L")
    if image.size != (48, 48):
        image = image.resize((48, 48), Image.Resampling.BILINEAR)
    img_array = np.array(image, dtype=np.float32) / 255.0
    return torch.FloatTensor(img_array).unsqueeze(0).unsqueeze(0)


def predict(model: nn.Module, image: Image.Image,
            device: str | torch.device | None = None) -> dict:
    """
    对单张（已裁剪人脸）图像进行表情类别预测。

    Returns:
        {
            "emotion": 预测类别名,
            "emoji": 对应 emoji,
            "probability": float,        # 预测类别的 softmax 输出（未校准概率）
            "probabilities": {类别名: 概率},  # 未校准
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

    return {
        "emotion": predicted_class,
        "emoji": CLASS_EMOJIS[predicted_class],
        "probability": float(probs[predicted_idx]),
        "probabilities": {
            name: float(prob) for name, prob in zip(CLASS_NAMES, probs, strict=True)
        },
    }


# ============================================================
# Grad-CAM 热力图（纯手写实现，无第三方依赖）
# ============================================================
def _find_last_conv_layer(model: nn.Module) -> nn.Module:
    """自动找到模型中最后一个卷积层"""
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
    device: str | torch.device | None = None,
    target_class: int | None = None,
    target_layer: nn.Module | None = None,
) -> np.ndarray:
    """
    生成 Grad-CAM 热力图（目标类别的梯度响应辅助图，非模型因果机制的证据）。

    - hook 在 finally 中移除（异常安全，多次调用不积累 hook）
    - 调用前后的训练/eval 状态恢复；梯度在结束后清零，不修改模型参数
    - 目标类别必须在 [0, 类别数) 范围内，否则明确报错
    - PB04：target_class 已提供时：仅 1 次带梯度前向 + 1 次反向（0 次目标判断前向）；
      target_class=None 时才需 1 次无梯度前向判断目标类

    Returns:
        heatmap: numpy array (48, 48)，值域 [0, 1]；零响应时全 0
    """
    if device is None:
        device = next(model.parameters()).device
    device = torch.device(device)

    was_training = model.training
    model.eval()

    if target_layer is None:
        target_layer = _find_last_conv_layer(model)
    assert target_layer is not None  # _find_last_conv_layer 未找到会直接报错

    tensor = preprocess_image(image).to(device)

    # PB04：目标类别已知时跳过「目标判断前向」（仅执行带梯度前向 + 反向）；
    # 范围校验延迟到带梯度前向之后（不额外前向）
    if target_class is None:
        with torch.no_grad():
            logits = model(tensor)
        target_class = int(torch.argmax(logits, dim=1)[0])
    target_class = int(target_class)

    gradients: list[torch.Tensor] = []
    activations: list[torch.Tensor] = []

    def forward_hook(_module, _input, output):
        activations.append(output)

    def backward_hook(_module, _grad_input, grad_output):
        gradients.append(grad_output[0])

    handles = [
        target_layer.register_forward_hook(forward_hook),
        target_layer.register_full_backward_hook(backward_hook),
    ]
    try:
        model.zero_grad(set_to_none=True)
        outputs = model(tensor)
        if not 0 <= target_class < outputs.size(1):
            raise ValueError(
                f"target_class={target_class} 超出 [0, {outputs.size(1)}) 范围"
            )
        score = outputs[0, target_class]
        score.backward()
    finally:
        for h in handles:
            h.remove()
        model.zero_grad(set_to_none=True)
        model.train(was_training)

    grads = gradients[0]   # (1, C, H', W')
    acts = activations[0]  # (1, C, H', W')

    # 全局平均池化梯度 → 每个通道的权重；加权求和 → ReLU
    weights = grads.mean(dim=(2, 3), keepdim=True)
    cam_tensor = F.relu((weights * acts).sum(dim=1, keepdim=True))

    # 归一化到 [0, 1]；零响应定义为全 0
    cam: np.ndarray = cam_tensor.squeeze().cpu().detach().numpy()
    cam_min, cam_max = cam.min(), cam.max()
    span = cam_max - cam_min
    cam = (cam - cam_min) / span if span > 1e-8 else np.zeros_like(cam)

    # 上采样到 48×48
    heatmap = np.array(
        Image.fromarray((cam * 255).astype(np.uint8)).resize((48, 48), Image.Resampling.BILINEAR)
    )
    return heatmap.astype(np.float32) / 255.0

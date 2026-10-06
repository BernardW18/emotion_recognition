"""
推理服务层（PB04）：缓存化的分析流程（分类 → 可选 Grad-CAM → 可视化渲染）。

默认路径（want_gradcam=False）：仅分类（1 次前向、0 次反向、0 次绘图）。
Grad-CAM 按需触发：使用已知的预测类别作为目标类，执行 1 次带梯度前向 + 1 次反向
（不在 infer_utils.generate_gradcam 内额外做「目标判断前向」）。

缓存（有限容量 LRU，键 = 图像字节 SHA + 权重文件路径/mtime + 目标类 + 可视化版本）：
  - 分类结果（predict 输出）
  - Grad-CAM 热力图（float32 ndarray）
  - 渲染完成的图 PNG（bytes，供 st.image 直接使用）
命中路径不重算、不重绘。图像/权重/目标类变化 → 键变化 → 自动失效。
不缓存 hook、不保留计算图（Grad-CAM 的梯度路径与异常安全由 infer_utils 保证）。

STATS 计数供验收测量与测试（各路径的实算/命中次数）。
"""

from __future__ import annotations

import hashlib
from collections import OrderedDict
from io import BytesIO

import numpy as np
from PIL import Image

from inference.infer_utils import CLASS_NAMES, generate_gradcam, predict

__all__ = [
    "VIZ_VERSION",
    "STATS",
    "analyze",
    "image_sha256",
    "reset_stats",
    "clear_caches",
]

VIZ_VERSION = "viz-v1"


def image_sha256(image_bytes: bytes) -> str:
    """图像字节内容的 SHA-256（缓存键的一部分；内容变化 → 键变化）。"""
    return hashlib.sha256(image_bytes).hexdigest()


class _LRU:
    """小型 LRU（容量固定；超出时淘汰最久未用项）。"""

    def __init__(self, capacity: int):
        self.capacity = max(1, int(capacity))
        self._data: OrderedDict = OrderedDict()

    def get(self, key, default=None):
        if key in self._data:
            self._data.move_to_end(key)
            return self._data[key]
        return default

    def put(self, key, value) -> None:
        self._data[key] = value
        self._data.move_to_end(key)
        while len(self._data) > self.capacity:
            self._data.popitem(last=False)

    def __len__(self) -> int:
        return len(self._data)


_PREDICT_CACHE = _LRU(64)
_GRADCAM_CACHE = _LRU(32)
_FIGURE_CACHE = _LRU(32)

# 路径计数（验收/测试）：compute = 实算次数；hits = 缓存命中次数
STATS: dict[str, int] = {
    "predict_compute": 0, "predict_hits": 0,
    "gradcam_compute": 0, "gradcam_hits": 0,
    "figure_compute": 0, "figure_hits": 0,
}


def reset_stats() -> None:
    for key in STATS:
        STATS[key] = 0


def clear_caches() -> None:
    _PREDICT_CACHE._data.clear()
    _GRADCAM_CACHE._data.clear()
    _FIGURE_CACHE._data.clear()


def _render_triple_png(gray_img: Image.Image, heatmap: np.ndarray, title: str) -> bytes:
    """三联图（模型输入 / 热力图 / 叠加）渲染为 PNG bytes（Agg，无交互环境）。

    中文字体 fallback：Windows（YaHei/SimHei）→ Linux（Noto Sans SC）→ DejaVu；
    局部 rc_context，不污染全局配置。缺字会以警告暴露（测试覆盖）。
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rc = {
        "font.sans-serif": ["Microsoft YaHei", "SimHei", "Noto Sans SC", "DejaVu Sans"],
        "axes.unicode_minus": False,
    }
    with matplotlib.rc_context(rc):
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(gray_img, cmap="gray")
        axes[0].set_title("模型输入（48×48 灰度）")
        axes[0].axis("off")

        axes[1].imshow(heatmap, cmap="jet", vmin=0, vmax=1)
        axes[1].set_title("Grad-CAM 热力图")
        axes[1].axis("off")

        axes[2].imshow(gray_img, cmap="gray")
        axes[2].imshow(heatmap, cmap="jet", alpha=0.5, vmin=0, vmax=1)
        axes[2].set_title(f"叠加: {title}")
        axes[2].axis("off")

        plt.tight_layout()
        buf = BytesIO()
        fig.savefig(buf, format="png", dpi=100)
        plt.close(fig)
    return buf.getvalue()


def analyze(
    model,
    image: Image.Image,
    image_digest: str,
    ckpt_key: str,
    device,
    *,
    want_gradcam: bool = False,
    target_class: int | None = None,
) -> dict:
    """
    执行（或从缓存取得）分类结果与可选 Grad-CAM 可视化。

    Args:
        model: 已加载的 eval 模型
        image: 上传图像（PIL）
        image_digest: image_sha256(...) 的结果（缓存键）
        ckpt_key: 权重标识（路径 + mtime；权重变化 → 失效）
        device: 计算设备
        want_gradcam: 是否生成热力图（默认 False = 仅分类）
        target_class: 热力图目标类别；None 时使用预测类别（已知，无需额外前向判断）

    Returns:
        {"result": predict 结果, "heatmap": ndarray | None, "figure_png": bytes | None}
    """
    predict_key = ("predict", image_digest, ckpt_key)
    result = _PREDICT_CACHE.get(predict_key)
    if result is None:
        result = predict(model, image, device)
        STATS["predict_compute"] += 1
        _PREDICT_CACHE.put(predict_key, result)
    else:
        STATS["predict_hits"] += 1

    heatmap = None
    figure_png = None
    if want_gradcam:
        target_idx = (
            int(target_class) if target_class is not None
            else int(CLASS_NAMES.index(result["emotion"]))
        )
        gradcam_key = ("gradcam", image_digest, ckpt_key, target_idx, VIZ_VERSION)
        heatmap = _GRADCAM_CACHE.get(gradcam_key)
        if heatmap is None:
            heatmap = generate_gradcam(model, image, device, target_class=target_idx)
            STATS["gradcam_compute"] += 1
            _GRADCAM_CACHE.put(gradcam_key, heatmap)
        else:
            STATS["gradcam_hits"] += 1

        figure_key = ("figure", image_digest, ckpt_key, target_idx, VIZ_VERSION)
        figure_png = _FIGURE_CACHE.get(figure_key)
        if figure_png is None:
            gray = image.convert("L")
            if gray.size != (48, 48):
                gray = gray.resize((48, 48), Image.Resampling.BILINEAR)
            figure_png = _render_triple_png(gray, heatmap, result["emotion"])
            STATS["figure_compute"] += 1
            _FIGURE_CACHE.put(figure_key, figure_png)
        else:
            STATS["figure_hits"] += 1

    return {"result": result, "heatmap": heatmap, "figure_png": figure_png}

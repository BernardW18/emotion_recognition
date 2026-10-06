"""PB04 推理服务层测试：默认 1 前向 / 按需热力图计数 / 缓存命中不重算 / 失效 / PNG。"""

import io

import numpy as np
import pytest
import torch
import torch.nn as nn
from PIL import Image

from inference import service
from inference.infer_utils import CLASS_NAMES
from inference.service import STATS, analyze, clear_caches, image_sha256, reset_stats


def _face_bytes(seed=0):
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 256, size=(48, 48), dtype=np.uint8)
    img = Image.fromarray(arr, mode="L")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue(), img


class _CountingModel(nn.Module):
    """带卷积的 7 类小模型 + forward/backward 计数（backward 经输出 hook）。"""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 4, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(4, 7)
        self.forward_calls = 0
        self.backward_calls = 0

    def forward(self, x):
        self.forward_calls += 1
        x = torch.relu(self.conv(x))
        x = self.pool(x).flatten(1)
        out = self.fc(x)
        if torch.is_grad_enabled() and out.requires_grad:
            out.register_hook(self._on_backward)
        return out

    def _on_backward(self, _grad):
        self.backward_calls += 1


@pytest.fixture(autouse=True)
def _fresh_caches():
    reset_stats()
    clear_caches()
    yield
    reset_stats()
    clear_caches()


@pytest.fixture
def model():
    torch.manual_seed(0)
    return _CountingModel()


CPU = torch.device("cpu")


# ============================================================
# 默认路径：1 前向 / 0 反向 / 0 绘图
# ============================================================
def test_default_path_single_forward_no_gradcam(model):
    raw, img = _face_bytes(1)
    out = analyze(model, img, image_sha256(raw), "ckptA", CPU, want_gradcam=False)
    assert model.forward_calls == 1, "默认分类路径应只执行 1 次前向"
    assert model.backward_calls == 0
    assert STATS["figure_compute"] == 0
    assert STATS["gradcam_compute"] == 0
    assert out["heatmap"] is None and out["figure_png"] is None
    assert out["result"]["emotion"] in CLASS_NAMES
    assert 0.0 <= out["result"]["probability"] <= 1.0


# ============================================================
# 按需热力图：+1 带梯度前向 +1 反向（无目标判断前向）
# ============================================================
def test_gradcam_on_demand_forward_backward_counts(model):
    raw, img = _face_bytes(2)
    d = image_sha256(raw)
    analyze(model, img, d, "ckptB", CPU, want_gradcam=False)   # 分类：1 前向
    out = analyze(model, img, d, "ckptB", CPU, want_gradcam=True)

    assert model.forward_calls == 2, "分类 + 带梯度前向（不再有目标判断前向）"
    assert model.backward_calls == 1
    assert STATS["gradcam_compute"] == 1
    assert STATS["figure_compute"] == 1
    assert out["heatmap"] is not None and out["heatmap"].shape == (48, 48)
    assert out["figure_png"][:4] == b"\x89PNG"


def test_gradcam_direct_call_counts(model):
    """直接 want_gradcam=True（无前置分类）：分类 1 前向 + 热力图 1 带梯度前向 + 1 反向。"""
    raw, img = _face_bytes(8)
    out = analyze(model, img, image_sha256(raw), "ckptG", CPU, want_gradcam=True)
    assert model.forward_calls == 2
    assert model.backward_calls == 1
    assert out["figure_png"] is not None


# ============================================================
# 缓存命中：不重算、不重绘
# ============================================================
def test_cache_hit_no_recompute(model):
    raw, img = _face_bytes(3)
    d = image_sha256(raw)
    analyze(model, img, d, "ckptC", CPU, want_gradcam=True)
    f1, b1 = model.forward_calls, model.backward_calls
    out = analyze(model, img, d, "ckptC", CPU, want_gradcam=True)

    assert model.forward_calls == f1 and model.backward_calls == b1
    assert STATS["predict_hits"] == 1
    assert STATS["gradcam_hits"] == 1
    assert STATS["figure_hits"] == 1
    assert out["figure_png"][:4] == b"\x89PNG"


# ============================================================
# 失效：图像 / 权重 / 目标类变化
# ============================================================
def test_invalidation_image_and_weight(model):
    raw1, img1 = _face_bytes(4)
    raw2, img2 = _face_bytes(5)
    d1, d2 = image_sha256(raw1), image_sha256(raw2)

    analyze(model, img1, d1, "ckptD", CPU, want_gradcam=False)
    n = model.forward_calls

    # 图像变化 → 重算
    analyze(model, img2, d2, "ckptD", CPU, want_gradcam=False)
    assert model.forward_calls == n + 1

    # 权重变化（ckpt_key 变）→ 重算
    m = model.forward_calls
    analyze(model, img2, d2, "ckptD_mtime2", CPU, want_gradcam=False)
    assert model.forward_calls == m + 1


def test_invalidation_target_class(model):
    raw, img = _face_bytes(6)
    d = image_sha256(raw)
    base = analyze(model, img, d, "ckptE", CPU, want_gradcam=False)
    pred = int(CLASS_NAMES.index(base["result"]["emotion"]))
    t_a = (pred + 1) % 7
    t_b = (pred + 2) % 7

    analyze(model, img, d, "ckptE", CPU, want_gradcam=True, target_class=t_a)
    n = model.forward_calls
    analyze(model, img, d, "ckptE", CPU, want_gradcam=True, target_class=t_a)  # 命中
    assert model.forward_calls == n
    analyze(model, img, d, "ckptE", CPU, want_gradcam=True, target_class=t_b)  # 失效
    assert model.forward_calls == n + 1


# ============================================================
# LRU 容量
# ============================================================
def test_render_triple_png_no_missing_glyphs():
    """渲染使用中文字体 fallback：不应出现缺字（Glyph missing）警告。"""
    import warnings as _warnings

    gray = Image.fromarray(np.zeros((48, 48), dtype=np.uint8), mode="L")
    heatmap = np.zeros((48, 48), dtype=np.float32)
    with _warnings.catch_warnings(record=True) as wlist:
        _warnings.simplefilter("always")
        png = service._render_triple_png(gray, heatmap, "Happy")
    glyph_warns = [w for w in wlist if "missing from font" in str(w.message)]
    assert not glyph_warns, f"渲染存在缺字警告: {[str(w.message) for w in glyph_warns[:3]]}"
    assert png[:4] == b"\x89PNG"


def test_lru_capacity_eviction():
    lru = service._LRU(2)
    lru.put("a", 1)
    lru.put("b", 2)
    lru.put("c", 3)
    assert lru.get("a") is None
    assert lru.get("b") == 2 and lru.get("c") == 3 and len(lru) == 2
    lru.get("b")           # b 最近使用
    lru.put("d", 4)
    assert lru.get("b") == 2 and lru.get("c") is None and lru.get("d") == 4

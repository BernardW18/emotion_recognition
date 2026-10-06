"""
推理层测试 — F01（统一加载一致性）/ F15（预处理一致性）/ F17（Grad-CAM 行为）

依赖 inference/saved_models/ 下的权重文件（legacy 权重）；缺失时跳过。
"""

import sys
from functools import lru_cache
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pytest
import torch
from PIL import Image

from inference.infer_utils import (
    describe_checkpoint,
    generate_gradcam,
    list_available_checkpoints,
    load_model,
    predict,
    preprocess_image,
)

LEGACY_CKPT = PROJECT_ROOT / "inference" / "saved_models" / "micro_resnet_best.pth"
_skip_no_ckpt = pytest.mark.skipif(not LEGACY_CKPT.exists(), reason="缺少微调权重文件")


@lru_cache(maxsize=1)
def _private_test_frame():
    """缓存 PrivateTest 子帧（避免重复读取 300MB CSV）。"""
    import pandas as pd

    df = pd.read_csv(PROJECT_ROOT / "data" / "fer2013.csv")
    return df[df["Usage"] == "PrivateTest"].reset_index(drop=True)


def _sample_face(index: int = 0) -> Image.Image:
    """从 FER2013 PrivateTest 取一张已裁剪人脸（uint8 灰度 48×48）。"""
    sub = _private_test_frame()
    pixels = np.array(sub.loc[index, "pixels"].split(), dtype=np.uint8).reshape(48, 48)
    return Image.fromarray(pixels, mode="L")


# ============================================================
# F15 · 预处理一致性
# ============================================================
def test_preprocess_identity_for_48x48():
    """已是 48×48 的输入不重采样，逐像素等于 x/255。"""
    rng = np.random.RandomState(0)
    arr = (rng.rand(48, 48) * 255).astype(np.uint8)
    tensor = preprocess_image(Image.fromarray(arr, mode="L"))
    assert tensor.shape == (1, 1, 48, 48)
    assert np.allclose(tensor.numpy()[0, 0], arr.astype(np.float32) / 255.0, atol=1e-6)


def test_preprocess_resizes_other_sizes():
    """非 48×48 输入缩放到 48×48。"""
    img = Image.fromarray(np.zeros((64, 96), dtype=np.uint8), mode="L")
    assert preprocess_image(img).shape == (1, 1, 48, 48)


# ============================================================
# F01 · 应用加载 == 统一构造；并与离线评估流水线逐样本一致
# ============================================================
@_skip_no_ckpt
def test_app_load_is_gelu_and_matches_offline():
    """应用加载 MicroResNet 确实为 GELU，且应用与离线评估对同批图像预测一致。"""
    model, device, meta = load_model(str(LEGACY_CKPT), device="cpu")
    assert meta["model_spec"]["activation"] == "gelu"
    assert meta["model_spec"]["use_se"] is False
    assert len(meta["checkpoint_sha256"]) == 64

    from utils.evaluation import _predict_probs

    sub = _private_test_frame().iloc[:5]

    # 离线评估流水线
    probs_offline, _ = _predict_probs(model, sub, batch_size=5, device=torch.device("cpu"))

    # 应用流水线（PIL → preprocess → predict）
    for i in range(len(sub)):
        pixels = np.array(sub.iloc[i]["pixels"].split(), dtype=np.uint8).reshape(48, 48)
        image = Image.fromarray(pixels, mode="L")
        result = predict(model, image, device)
        assert result["emotion"] == model_spec_class(probs_offline[i].argmax()), i
        assert result["probability"] == pytest.approx(float(probs_offline[i].max()), abs=1e-6)
        assert sum(result["probabilities"].values()) == pytest.approx(1.0, abs=1e-5)


def model_spec_class(idx) -> str:
    from utils.constants import CLASS_NAMES

    return CLASS_NAMES[int(idx)]


# ============================================================
# F17 · Grad-CAM 行为
# ============================================================
@_skip_no_ckpt
def test_gradcam_shape_range_and_target_bounds():
    model, device, _ = load_model(str(LEGACY_CKPT), device="cpu")
    image = _sample_face(0)

    heatmap = generate_gradcam(model, image, device)
    assert heatmap.shape == (48, 48)
    assert np.isfinite(heatmap).all()
    assert heatmap.min() >= 0.0 and heatmap.max() <= 1.0

    with pytest.raises(ValueError, match="超出"):
        generate_gradcam(model, image, device, target_class=7)
    with pytest.raises(ValueError, match="超出"):
        generate_gradcam(model, image, device, target_class=-1)

    # 指定合法目标类别
    heatmap2 = generate_gradcam(model, image, device, target_class=0)
    assert heatmap2.shape == (48, 48)


@_skip_no_ckpt
def test_gradcam_repeated_calls_do_not_accumulate_hooks_or_change_weights():
    """连续调用 ≥10 次：hook 不积累、参数不变、训练状态恢复。"""
    model, device, _ = load_model(str(LEGACY_CKPT), device="cpu")
    image = _sample_face(1)

    def hook_count(m):
        total = 0
        for mod in m.modules():
            total += len(getattr(mod, "_forward_hooks", {}))
            total += len(getattr(mod, "_backward_hooks", {}))
            total += len(getattr(mod, "_full_backward_hooks", {}))
        return total

    before_hooks = hook_count(model)
    before_params = [p.detach().clone() for p in model.parameters()]

    for _ in range(10):
        generate_gradcam(model, image, device)

    assert hook_count(model) == before_hooks, "hook 数量增加"
    for p_before, p_after in zip(before_params, model.parameters(), strict=True):
        assert torch.equal(p_before, p_after), "Grad-CAM 调用改变了模型参数"

    # eval 状态默认保持
    assert model.training is False
    # 从 train 状态进入 → 恢复 train
    model.train()
    generate_gradcam(model, image, device)
    assert model.training is True
    model.eval()


@_skip_no_ckpt
def test_gradcam_zero_response_defined():
    """零响应（全零梯度/激活）时热图为全 0 而非 NaN。"""
    import torch.nn as nn

    class ZeroCam(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(1, 4, 3, padding=1)
            self.fc = nn.Linear(4 * 48 * 48, 3)
            with torch.no_grad():
                self.conv.weight.zero_()
                self.conv.bias.zero_()

        def forward(self, x):
            f = self.conv(x)  # 恒为 0 的激活（参数梯度仍存在）
            return self.fc(f.flatten(1))

    model = ZeroCam()
    image = Image.fromarray(np.zeros((48, 48), dtype=np.uint8), mode="L")
    heatmap = generate_gradcam(model, image, "cpu")
    assert heatmap.shape == (48, 48)
    assert np.all(heatmap == 0.0)


# ============================================================
# 权重清单
# ============================================================
def test_list_available_checkpoints_finds_legacy():
    items = list_available_checkpoints()
    if not items:
        pytest.skip("saved_models 为空")
    names = {it["file"] for it in items}
    assert "micro_resnet_best.pth" in names
    # 无 manifest 时应标记为 legacy/无来源
    micro = next(it for it in items if it["file"] == "micro_resnet_best.pth")
    assert isinstance(micro["label"], str) and micro["label"]


@_skip_no_ckpt
def test_describe_checkpoint_reports_spec_and_params():
    info = describe_checkpoint(LEGACY_CKPT)
    assert info["model_spec"]["activation"] == "gelu"
    assert info["params_total"] == 753991
    assert info["legacy"] is True  # 旧格式权重
    assert len(info["sha256"]) == 64

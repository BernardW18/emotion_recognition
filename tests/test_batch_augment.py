"""PB01 批级张量增强测试：确定性 / 参数范围 / 几何方向对齐 / 类别门控 / 防御。"""

import pytest
import torch

from data.batch_augment import (
    BATCH_AUG_VERSION,
    BatchAugmenter,
    affine_batch,
    build_batch_augmenter,
)

_AUG_CFG = {
    "enabled": True,
    "random_horizontal_flip": 0.5,
    "random_rotation": 10,
    "random_affine_translate": 0.1,
    "color_jitter_brightness": 0.2,
    "color_jitter_contrast": 0.2,
    "random_erase": True,
    "class_specific": {
        "enabled": True, "target_classes": [1], "augment_prob": 0.8,
        "extra_rotation": 20, "extra_translate": 0.15, "extra_erase_prob": 0.2,
    },
}

def _aug(**overrides):
    cfg = dict(_AUG_CFG)
    cfg.update(overrides)
    return BatchAugmenter(cfg, seed=42)


def _aug_off(**overrides):
    """除 overrides 外全部关闭的增强器（用于单操作行为测试）。"""
    cfg = {
        "random_horizontal_flip": 0, "random_rotation": 0, "random_affine_translate": 0,
        "color_jitter_brightness": 0, "color_jitter_contrast": 0, "random_erase": False,
        "class_specific": {"enabled": False},
    }
    cfg.update(overrides)
    return BatchAugmenter(cfg, seed=42)


def _centroid(x):
    w = x[0, 0]
    tot = float(w.sum())
    ys = torch.arange(w.shape[0]).float()
    return float((w.sum(1) * ys).sum() / tot), float((w.sum(0) * ys).sum() / tot)


# ============================================================
# 构建与版本
# ============================================================
def test_disabled_build_returns_none():
    assert build_batch_augmenter({"enabled": False}, seed=1) is None
    assert build_batch_augmenter({}, seed=1) is None
    assert build_batch_augmenter(_AUG_CFG, seed=1) is not None


def test_version_constant():
    assert BATCH_AUG_VERSION == "batched-v1"


# ============================================================
# 确定性：种子 = f(seed, epoch, batch_idx)
# ============================================================
def test_deterministic_per_epoch_batch_key():
    aug = _aug()
    imgs = torch.rand(32, 1, 8, 8)
    labels = torch.randint(0, 3, (32,))
    a1 = aug(imgs, labels, epoch=2, batch_idx=5)
    a2 = aug(imgs, labels, epoch=2, batch_idx=5)
    a3 = aug(imgs, labels, epoch=2, batch_idx=6)
    a4 = aug(imgs, labels, epoch=3, batch_idx=5)
    assert torch.equal(a1, a2), "同 (epoch, batch_idx) 必须逐位一致"
    assert not torch.equal(a1, a3), "不同 batch_idx 应产生不同增强"
    assert not torch.equal(a1, a4), "不同 epoch 应产生不同增强"


def test_param_ranges_and_rates():
    aug = _aug()
    g = aug._batch_generator(0, 0)
    p = aug._sample_params(8192, g)
    assert 0.47 < float(p["flip"].float().mean()) < 0.53
    assert abs(float(p["angle"].min())) <= 10.0 and abs(float(p["angle"].max())) <= 10.0
    assert abs(float(p["tx"].min())) <= 0.1 and abs(float(p["tx"].max())) <= 0.1
    assert abs(float(p["ty"].min())) <= 0.1 and abs(float(p["ty"].max())) <= 0.1
    assert float(p["bright"].min()) >= 0.8 and float(p["bright"].max()) <= 1.2
    assert float(p["contrast"].min()) >= 0.8 and float(p["contrast"].max()) <= 1.2
    assert 0.08 < float(p["erase"].float().mean()) < 0.12
    assert 0.45 < float(p["color_order"].float().mean()) < 0.55


# ============================================================
# 几何方向与 v1（legacy torchvision）对齐
# ============================================================
def test_rotate_translate_direction_matches_v1():
    from torchvision.transforms.functional import affine as tf_affine
    from torchvision.transforms.functional import rotate as tf_rotate

    x = torch.zeros(1, 1, 48, 48)
    x[0, 0, 10:20, 10:20] = 1.0
    c0 = _centroid(x)

    # 纯旋转：对齐 F.rotate(+20°)
    y = affine_batch(x, torch.tensor([20.0]), torch.tensor([0.0]), torch.tensor([0.0]))
    c = _centroid(y)
    cref = _centroid(tf_rotate(x, 20.0))
    assert abs(c[0] - cref[0]) < 0.7 and abs(c[1] - cref[1]) < 0.7

    # 纯平移：tx=+0.1 → 内容右移约 5px（48px 的 0.1，round 到整像素）
    y = affine_batch(x, torch.tensor([0.0]), torch.tensor([0.1]), torch.tensor([0.0]))
    c = _centroid(y)
    assert abs((c[1] - c0[1]) - 5.0) < 0.5, f"tx=+0.1 位移异常: {c[1] - c0[1]:.2f}px"

    # 组合：rotate(20°) 后 translate 5px（legacy 链：RandomRotation → RandomAffine）
    y = affine_batch(x, torch.tensor([20.0]), torch.tensor([0.1]), torch.tensor([0.0]))
    c = _centroid(y)
    cref = _centroid(tf_affine(tf_rotate(x, 20.0), 0.0, (5.0, 0.0), 1.0, 0.0))
    assert abs(c[0] - cref[0]) < 0.7 and abs(c[1] - cref[1]) < 0.7


# ============================================================
# 单操作行为
# ============================================================
def test_flip_only_matches_expected_transform():
    aug = _aug_off(random_horizontal_flip=0.5)
    x = torch.rand(128, 1, 8, 8)
    labels = torch.zeros(128, dtype=torch.long)
    out = aug(x, labels, epoch=0, batch_idx=0)
    same = (out == x).flatten(1).all(1)
    flipped = (out == x.flip(-1)).flatten(1).all(1)
    assert bool((same | flipped).all()), "只开 flip 时输出必须等于原图或左右翻转"
    rate = float(flipped.float().mean())
    assert 0.3 < rate < 0.7


def test_erase_only_rate_and_zero_fill():
    aug = _aug_off(random_erase=True)
    x = torch.rand(1024, 1, 8, 8) + 0.01   # 避免原值为 0 干扰判断
    labels = torch.zeros(1024, dtype=torch.long)
    out = aug(x, labels, epoch=0, batch_idx=0)
    has_zero = (out == 0).flatten(1).any(1)
    rate = float(has_zero.float().mean())
    assert 0.06 < rate < 0.15, f"erase 比例异常: {rate}"
    not_erased = ~has_zero
    assert torch.equal(out[not_erased], x[not_erased]), "未擦除样本不得被改动"


def test_noop_config_returns_input_unchanged():
    aug = _aug_off()
    x = torch.rand(8, 1, 8, 8)
    labels = torch.zeros(8, dtype=torch.long)
    out = aug(x, labels, epoch=0, batch_idx=0)
    assert torch.equal(out, x)


# ============================================================
# 类别条件增强（对齐 legacy class_aug_map 语义）
# ============================================================
def test_class_specific_gating_prob_1():
    aug = _aug_off(class_specific={
        "enabled": True, "target_classes": [1], "augment_prob": 1.0,
        "extra_rotation": 20, "extra_translate": 0, "extra_erase_prob": 0,
    })
    x = torch.rand(64, 1, 8, 8)
    labels = torch.zeros(64, dtype=torch.long)
    labels[32:] = 1
    out = aug(x, labels, epoch=0, batch_idx=0)
    assert torch.equal(out[:32], x[:32]), "非目标类别不得被增强"
    changed = (out[32:] != x[32:]).flatten(1).any(1)
    assert bool(changed.all()), "augment_prob=1.0 时目标类别应全部被旋转"


def test_class_specific_gating_rate():
    aug = _aug_off(class_specific={
        "enabled": True, "target_classes": [1], "augment_prob": 0.2,
        "extra_rotation": 20, "extra_translate": 0, "extra_erase_prob": 0,
    })
    x = torch.rand(2048, 1, 8, 8)
    labels = torch.ones(2048, dtype=torch.long)
    out = aug(x, labels, epoch=0, batch_idx=0)
    rate = float((out != x).flatten(1).any(1).float().mean())
    assert 0.15 < rate < 0.25, f"类别增强比例异常: {rate}"


def test_extra_erase_applies_only_to_gated_targets():
    aug = _aug_off(class_specific={
        "enabled": True, "target_classes": [1], "augment_prob": 1.0,
        "extra_rotation": 0, "extra_translate": 0, "extra_erase_prob": 0.5,
    })
    x = torch.rand(1024, 1, 8, 8) + 0.01
    labels = torch.ones(1024, dtype=torch.long)
    out = aug(x, labels, epoch=0, batch_idx=0)
    has_zero = (out == 0).flatten(1).any(1)
    rate = float(has_zero.float().mean())
    assert 0.4 < rate < 0.6, f"类别额外擦除比例异常: {rate}"


# ============================================================
# 防御：输入校验 / 不改输入 / 有限值
# ============================================================
def test_input_guards():
    aug = _aug()
    with pytest.raises(ValueError, match="B,1,H,W"):
        aug(torch.rand(8, 3, 8, 8), torch.zeros(8, dtype=torch.long), epoch=0, batch_idx=0)
    if torch.cuda.is_available():
        with pytest.raises(ValueError, match="CPU"):
            aug(torch.rand(8, 1, 8, 8, device="cuda"),
                torch.zeros(8, dtype=torch.long, device="cuda"), epoch=0, batch_idx=0)


def test_does_not_mutate_input_and_stays_finite():
    aug = _aug()
    x = torch.rand(64, 1, 16, 16)
    x_copy = x.clone()
    labels = torch.randint(0, 7, (64,))
    out = aug(x, labels, epoch=0, batch_idx=0)
    assert torch.equal(x, x_copy), "增强不得就地修改输入张量"
    assert out.shape == x.shape
    assert torch.isfinite(out).all()

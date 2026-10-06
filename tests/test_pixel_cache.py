"""PB02 像素缓存专项测试：构建/校验/命中/失效重建/逐位一致/轻量 pickle/只读性。

使用临时迷你 CSV 与独立缓存目录，不触碰正式 data/cache 与 data/fer2013.csv。
"""

import json
import pickle

import numpy as np
import pandas as pd
import pytest

from data import pixel_cache
from data.dataloader import (
    FER2013Dataset,
    compute_split_fingerprint,
    compute_split_fingerprint_cached,
)


# ============================================================
# 迷你 CSV fixture
# ============================================================
def _write_mini_csv(path, *, n_train=14, n_val=6, n_test=6, seed=0):
    """写迷你 FER2013 风格 CSV：每行 2304 个 0-255 整数；emotion 循环 0..6。"""
    rng = np.random.default_rng(seed)
    rows = []
    usages = ["Training"] * n_train + ["PublicTest"] * n_val + ["PrivateTest"] * n_test
    for i, usage in enumerate(usages):
        pixels = rng.integers(0, 256, size=48 * 48, dtype=np.uint8)
        rows.append({
            "emotion": i % 7,
            "pixels": " ".join(str(int(v)) for v in pixels),
            "Usage": usage,
        })
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    return df


@pytest.fixture
def mini_csv(tmp_path):
    csv_path = tmp_path / "mini_fer2013.csv"
    df = _write_mini_csv(csv_path)
    return csv_path, df


@pytest.fixture
def mini_cache(tmp_path, mini_csv):
    csv_path, df = mini_csv
    cache_dir = tmp_path / "cache"
    pc = pixel_cache.load_or_build(csv_path, cache_dir, quiet=True)
    return pc, csv_path, df, cache_dir


# ============================================================
# 构建与往返一致性
# ============================================================
def test_build_and_roundtrip(mini_csv, tmp_path):
    csv_path, df = mini_csv
    cache_dir = tmp_path / "cache"
    pc = pixel_cache.load_or_build(csv_path, cache_dir, quiet=True)

    # meta 内容
    assert pc.meta["cache_version"] == pixel_cache.CACHE_VERSION
    assert pc.meta["csv_sha256"] == pixel_cache.file_sha256_cached(csv_path)
    assert pc.meta["total_rows"] == len(df)
    for split, n in (("Training", 14), ("PublicTest", 6), ("PrivateTest", 6)):
        assert pc.meta["splits"][split]["n"] == n

    # 数据内容：像素/标签/行号与源 DataFrame 一致
    for split in ("Training", "PublicTest", "PrivateTest"):
        x, y, rows = pc.split_arrays(split)
        mask = (df["Usage"] == split).to_numpy()
        sub = df.loc[mask]
        expected_px = np.stack([
            np.fromstring(s, dtype=np.float32, sep=" ").reshape(48, 48)
            for s in sub["pixels"]
        ])
        got_px = np.asarray(x[:], dtype=np.float32)
        assert np.array_equal(got_px, expected_px), f"{split} 像素不一致"
        assert np.array_equal(y, sub["emotion"].to_numpy(dtype=np.int64))
        assert np.array_equal(rows, sub.index.to_numpy(dtype=np.int64))


def test_pixels_match_legacy_float_path(mini_csv, tmp_path):
    """无增强输入与旧路径（DataFrame→float32/255）逐位一致。"""
    csv_path, df = mini_csv
    cache_dir = tmp_path / "cache"
    pc = pixel_cache.load_or_build(csv_path, cache_dir, quiet=True)

    df_tr = df[df["Usage"] == "Training"].reset_index(drop=True)
    ds_old = FER2013Dataset(df_tr)
    x_mm, y_tr, _ = pc.split_arrays("Training")
    new_px = np.asarray(x_mm[:], dtype=np.float32) / 255.0
    assert np.array_equal(new_px, ds_old._pixels)  # 逐位一致（含 /255 舍入）
    assert np.array_equal(y_tr, ds_old._labels)

    # 单样本 __getitem__ 一致（新 cache_ref 路径 vs 旧路径）
    ds_new = FER2013Dataset(cache_ref=(cache_dir, "Training", pc.meta["splits"]["Training"]))
    for i in range(3):
        img_new, lab_new = ds_new[i]
        img_old, lab_old = ds_old[i]
        assert img_new.equal(img_old) and img_new.shape == (1, 48, 48)
        assert int(lab_new) == int(lab_old)


# ============================================================
# 命中不重建 / 失效重建
# ============================================================
def test_cache_hit_does_not_rebuild(mini_cache):
    pc, csv_path, _df, cache_dir = mini_cache
    meta_path = cache_dir / "meta.json"
    created = json.loads(meta_path.read_text(encoding="utf-8"))["created_at"]

    pc2 = pixel_cache.load_or_build(csv_path, cache_dir, quiet=True)
    assert pc2.meta["created_at"] == created, "命中时不应重建"


def test_csv_change_triggers_rebuild(mini_cache):
    pc, csv_path, df, cache_dir = mini_cache
    old_sha = pc.meta["csv_sha256"]

    # 修改一个像素值（同划分结构）
    df2 = df.copy()
    df2.loc[0, "pixels"] = " ".join(["7"] * 48 * 48)
    df2.to_csv(csv_path, index=False)

    pc2 = pixel_cache.load_or_build(csv_path, cache_dir, quiet=True)
    assert pc2.meta["csv_sha256"] != old_sha
    x, _y, _rows = pc2.split_arrays("Training")
    assert int(np.asarray(x[0]).reshape(-1)[0]) == 7, "重建后应反映新 CSV 内容"


def test_corrupted_npy_triggers_rebuild(mini_cache):
    pc, csv_path, _df, cache_dir = mini_cache
    fname = pc.meta["splits"]["Training"]["x"]["file"]
    target = cache_dir / fname
    size = target.stat().st_size

    # 同长度破坏数据区字节（保持 shape 可读，靠 SHA-256 检出）
    raw = bytearray(target.read_bytes())
    raw[size // 2] ^= 0xFF
    target.write_bytes(bytes(raw))

    # 新句柄触发校验 → 重建 → 数据恢复正确
    pc2 = pixel_cache.load_or_build(csv_path, cache_dir, quiet=True)
    x, _y, _rows = pc2.split_arrays("Training")
    ref = pixel_cache.load_or_build(csv_path, cache_dir, quiet=True)
    x_ref, _y, _rows = ref.split_arrays("Training")
    assert np.array_equal(np.asarray(x[:]), np.asarray(x_ref[:])), "重建后数据应与源一致"
    # meta 中的 SHA 应更新为当前文件
    assert pixel_cache.file_sha256_cached(target) == pc2.meta["splits"]["Training"]["x"]["sha256"]


def test_meta_version_mismatch_triggers_rebuild(mini_cache):
    pc, csv_path, _df, cache_dir = mini_cache
    meta_path = cache_dir / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["cache_version"] = "uint8-v0-obsolete"
    meta_path.write_text(json.dumps(meta), encoding="utf-8")

    pc2 = pixel_cache.load_or_build(csv_path, cache_dir, quiet=True)
    assert pc2.meta["cache_version"] == pixel_cache.CACHE_VERSION


def test_missing_cache_builds(mini_csv, tmp_path):
    csv_path, _df = mini_csv
    cache_dir = tmp_path / "nonexistent" / "cache"
    pc = pixel_cache.load_or_build(csv_path, cache_dir, quiet=True)
    assert (cache_dir / "meta.json").exists()
    assert pc.total_rows == 26


# ============================================================
# dataset 轻量化 / 只读性 / 恢复调用
# ============================================================
def test_dataset_cache_ref_pickle_is_lightweight(mini_cache):
    pc, _csv_path, _df, cache_dir = mini_cache
    ds = FER2013Dataset(cache_ref=(cache_dir, "Training", pc.meta["splits"]["Training"]))
    assert ds._pixels is None, "缓存路径不应持有像素数组"

    blob = pickle.dumps(ds)
    assert len(blob) < 4096, f"pickle 载荷应轻量（实际 {len(blob)} 字节）"

    ds2 = pickle.loads(blob)
    assert len(ds2) == len(ds)
    img, label = ds2[0]
    assert img.shape == (1, 48, 48)
    img_ref, label_ref = ds[0]
    assert img.equal(img_ref) and int(label) == int(label_ref)


def test_mmap_readonly_and_no_pollution(mini_cache):
    pc, _csv_path, _df, cache_dir = mini_cache
    x, _y, _rows = pc.split_arrays("Training")
    assert not x.flags.writeable, "缓存数组应为只读 mmap"

    ds = FER2013Dataset(cache_ref=(cache_dir, "Training", pc.meta["splits"]["Training"]))
    img, _ = ds[0]
    img += 100.0  # 修改返回的张量
    img2, _ = ds[0]
    assert not np.array_equal(img.numpy(), img2.numpy()), "张量应为副本（修改不影响数据集）"
    assert img2.min() >= 0.0 and img2.max() <= 1.0, "重新取值应仍为原数据"


def test_split_arrays_unknown_split(mini_cache):
    pc, _csv_path, _df, _cache_dir = mini_cache
    with pytest.raises(KeyError):
        pc.split_arrays("NoSuchSplit")


# ============================================================
# 指纹：缓存版与旧版公式一致
# ============================================================
def test_fingerprint_cached_equals_legacy(mini_cache):
    pc, csv_path, df, _cache_dir = mini_cache
    fp_old = compute_split_fingerprint(csv_path, df)
    fp_new = compute_split_fingerprint_cached(csv_path, pc)
    assert fp_new["csv_sha256"] == fp_old["csv_sha256"]
    assert fp_new["protocol"] == fp_old["protocol"]
    for split in ("Training", "PublicTest", "PrivateTest"):
        assert fp_new["splits"][split] == fp_old["splits"][split], f"{split} 指纹不一致"


# ============================================================
# create_dataloaders：include_test 与 mini config
# ============================================================
def _mini_config(csv_path):
    return {
        "data": {
            "dataset_path": str(csv_path),
            "image_size": 48,
            "num_classes": 7,
            "class_names": ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"],
        },
        "dataloader": {
            "num_workers": 0,
            "pin_memory": False,
            "persistent_workers": False,
            "prefetch_factor": 2,
            "class_balanced_sampling": False,
        },
        "augmentation": {"enabled": False},
        "training": {"batch_size": 4},
        "models": {"mini_cnn": {"batch_size": 4}},
        "checkpoint": {},
        "seed": 42,
    }


def test_create_dataloaders_include_test_flag(mini_csv, tmp_path, monkeypatch):
    """include_test=False 时 PrivateTest 不构建；True 时构建且长度正确。"""
    csv_path, _df = mini_csv
    monkeypatch.setenv("PYTEST_MINICACHE", "1")
    from data import pixel_cache as pc_mod

    # 为测试用独立缓存目录：直接 monkeypatch 默认缓存目录
    monkeypatch.setattr(pc_mod, "default_cache_dir", lambda: tmp_path / "dataloader_cache")

    from data.dataloader import create_dataloaders

    config = _mini_config(csv_path)
    train_loader, val_loader, test_loader, class_names = create_dataloaders(
        config, "mini_cnn", include_test=False
    )
    assert test_loader is None
    assert len(train_loader.dataset) == 14
    assert len(val_loader.dataset) == 6
    assert class_names[0] == "Angry"
    assert train_loader.dataset.split_fingerprint is not None
    assert train_loader.dataset.class_counts == [2, 2, 2, 2, 2, 2, 2]

    # 训练一批（无增强）：形状与值域
    images, labels = next(iter(train_loader))
    assert images.shape[1:] == (1, 48, 48)
    assert images.min() >= 0.0 and images.max() <= 1.0

    train_loader2, val_loader2, test_loader2, _ = create_dataloaders(
        config, "mini_cnn", include_test=True
    )
    assert test_loader2 is not None
    assert len(test_loader2.dataset) == 6


def test_create_dataloaders_rejects_unknown_aug_impl(mini_csv, tmp_path, monkeypatch):
    csv_path, _df = mini_csv
    from data import pixel_cache as pc_mod

    monkeypatch.setattr(pc_mod, "default_cache_dir", lambda: tmp_path / "dataloader_cache2")
    from data.dataloader import create_dataloaders

    config = _mini_config(csv_path)
    config["augmentation"] = {"enabled": True, "impl": "cloud-9d"}
    with pytest.raises(ValueError, match="impl"):
        create_dataloaders(config, "mini_cnn", include_test=False)


# ============================================================
# cache_dir 与 CSV SHA 校验的独立性（来源不同 CSV 不串缓存）
# ============================================================
def test_different_csv_same_cache_dir_rebuilds(tmp_path):
    csv_a = tmp_path / "a.csv"
    csv_b = tmp_path / "b.csv"
    _write_mini_csv(csv_a, seed=1)
    _write_mini_csv(csv_b, seed=2)
    cache_dir = tmp_path / "shared_cache"

    pc_a = pixel_cache.load_or_build(csv_a, cache_dir, quiet=True)
    sha_a = pc_a.meta["csv_sha256"]
    pc_b = pixel_cache.load_or_build(csv_b, cache_dir, quiet=True)
    assert pc_b.meta["csv_sha256"] != sha_a, "不同 CSV 应触发重建而非复用"
    assert pc_b.meta["csv_sha256"] == pixel_cache.file_sha256_cached(csv_b)

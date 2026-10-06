"""
训练管线回归测试 — F05 / F08 / F09 / F13 / F14

覆盖:
  - 梯度累积按组内实际样本数归一化（3+3+2 与单批 8 样本等价；数学 + Trainer 两级）
  - val_loss 恶化监控序列与清零（F08 验收序列 [1.0, 1.06, 1.07, 1.08]）
  - 续训一致性：连续两轮 == 一轮保存后恢复再一轮（批次顺序/参数/history/best/计数）
  - 两个 run 目录与产物互相隔离（不同 seed 不继承 best、不覆盖文件）
  - 断点损坏/旧格式明确报错；保存失败不破坏上一份可用断点
  - partial 中断标记与 run_meta 状态
  - 早停计数/best 往返；monitor_metric / save_best 开关
"""

import copy
import sys
from pathlib import Path

# 确保项目根目录在 sys.path 中
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import hashlib
import random

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from training.checkpoint import CHECKPOINT_FORMAT_VERSION, read_run_meta
from training.trainer import (
    Trainer,
    compute_batch_sizes,
    compute_group_totals,
    update_val_loss_monitor,
)
from utils.losses import FocalLoss
from utils.model_spec import file_sha256


# ============================================================
# 辅助
# ============================================================
def _tiny_data(n=16, dims=4, classes=3, seed=0):
    """固定小数据（独立 generator，不消耗全局 RNG）"""
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, dims, generator=g)
    y = torch.randint(0, classes, (n,), generator=g)
    return TensorDataset(x, y)


def _make_config(seed=42):
    return {
        "data": {
            "dataset_path": "data/fer2013.csv", "image_size": 48,
            "num_classes": 3, "class_names": ["c0", "c1", "c2"],
        },
        "dataloader": {
            "num_workers": 0, "pin_memory": False, "persistent_workers": False,
            "prefetch_factor": 2, "class_balanced_sampling": False,
        },
        "augmentation": {
            "enabled": False, "class_specific": {"enabled": False},
            "mixup": {"enabled": False},
        },
        "training": {
            "batch_size": 4, "num_epochs": 2, "learning_rate": 0.01, "weight_decay": 0.0,
            "optimizer": "adam", "scheduler": "none", "patience": 0,
            "val_loss_patience": 0, "val_loss_threshold": 1.05,
            "loss_type": "focal", "amp": False, "gradient_accumulation_steps": 1,
            "max_grad_norm": 0.0, "cudnn_deterministic": False,
        },
        "models": {
            "mini_cnn": {
                "learning_rate": 0.01, "batch_size": 4, "num_epochs": 2,
                "dropout": 0.0, "activation": "relu",
            },
        },
        "checkpoint": {
            "save_best": True, "save_every_n_epochs": 1,
            "monitor_metric": "val_acc", "max_checkpoint_files": 5,
        },
        "seed": seed,
    }


def _make_trainer(config, model=None, loader=None, val_loader=None, run_dir=None,
                  focal_gamma=2.0):
    if model is None:
        torch.manual_seed(config["seed"])
        model = nn.Linear(4, 3)
    if loader is None:
        loader = DataLoader(_tiny_data(), batch_size=4, shuffle=False)
    if val_loader is None:
        val_loader = DataLoader(_tiny_data(seed=1), batch_size=4, shuffle=False)
    return Trainer(
        model=model,
        train_loader=loader,
        val_loader=val_loader,
        test_loader=val_loader,
        config=config,
        model_name="mini_cnn",
        device=torch.device("cpu"),
        run_dir=run_dir,
        focal_gamma=focal_gamma,
    )


class SpyLoader:
    """包装 DataLoader，记录逐批（输入内容哈希 + 标签序列）。

    R01 要求比对"逐批原始样本 occurrence 与增强后输入"，而非只比标签哈希：
    增强后的输入张量内容哈希能唯一反映实际喂给模型的样本与变换结果。
    """

    def __init__(self, loader):
        self._loader = loader
        self.batches = []  # [(x_sha256[:16], (labels...)), ...]
        self.seen = []     # 标签序列（兼容旧断言/可读性）

    def __iter__(self):
        for xb, yb in self._loader:
            arr = np.ascontiguousarray(xb.numpy())
            x_hash = hashlib.sha256(arr.tobytes()).hexdigest()[:16]
            labels = tuple(int(v) for v in yb.tolist())
            self.batches.append((x_hash, labels))
            self.seen.append(labels)
            yield xb, yb

    def __len__(self):
        return len(self._loader)

    def __getattr__(self, name):
        return getattr(self._loader, name)


class _TransformDataset(torch.utils.data.Dataset):
    """带数据增强的包装集（用于增强开启下的恢复一致性测试）。"""

    def __init__(self, base, transform):
        self.base = base
        self.transform = transform

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        x, y = self.base[index]
        return self.transform(x), y


# ============================================================
# F14 · 梯度累积
# ============================================================
def test_grad_accum_helpers():
    loader = DataLoader(_tiny_data(n=8), batch_size=3, shuffle=False)  # 3+3+2
    assert compute_batch_sizes(loader) == [3, 3, 2]
    assert compute_group_totals([3, 3, 2], 3) == [8, 8, 8]
    assert compute_group_totals([4, 4, 4, 4, 4], 2) == [8, 8, 8, 8, 4]


def test_grad_accum_equivalence_math():
    """3+3+2 分组累积与一次 8 样本均值损失的更新等价（FP32）。"""
    torch.manual_seed(0)
    lin = nn.Linear(4, 3)
    x = torch.randn(8, 4)
    y = torch.randint(0, 3, (8,))
    crit = FocalLoss(gamma=2.0)

    # 参考：一次 8 样本
    ref = copy.deepcopy(lin)
    opt_ref = torch.optim.Adam(ref.parameters(), lr=0.1)
    crit(ref(x), y).backward()
    opt_ref.step()

    # 分组：3+3+2，组内实际样本数归一化
    m = copy.deepcopy(lin)
    opt = torch.optim.Adam(m.parameters(), lr=0.1)
    sizes = [3, 3, 2]
    totals = compute_group_totals(sizes, 3)
    start = 0
    for s, g in zip(sizes, totals, strict=True):
        loss = crit(m(x[start:start + s]), y[start:start + s])
        (loss * s / g).backward()
        start += s
    opt.step()

    assert torch.allclose(ref.weight, m.weight, atol=1e-6, rtol=1e-5)
    assert torch.allclose(ref.bias, m.bias, atol=1e-6, rtol=1e-5)


def test_train_one_epoch_accum_matches_single_batch(tmp_path):
    """Trainer 的 train_one_epoch 在 K=3（3+3+2）下与单批 8 样本的更新一致。"""
    torch.manual_seed(1)
    lin = nn.Linear(4, 3)
    dataset = _tiny_data(n=8, seed=2)
    loader = DataLoader(dataset, batch_size=3, shuffle=False)

    config = _make_config()
    config["training"].update({"gradient_accumulation_steps": 3, "learning_rate": 0.1})
    config["models"]["mini_cnn"]["learning_rate"] = 0.1

    trainer = Trainer(
        model=copy.deepcopy(lin), train_loader=loader, val_loader=loader,
        test_loader=loader, config=config, model_name="mini_cnn",
        device=torch.device("cpu"), run_dir=tmp_path / "run",
    )
    trainer._current_epoch = 1
    trainer.train_one_epoch()

    # 参考：一次 8 样本，Adam step 一次
    x = torch.stack([dataset[i][0] for i in range(8)])
    y = torch.stack([dataset[i][1] for i in range(8)])
    ref = copy.deepcopy(lin)
    opt = torch.optim.Adam(ref.parameters(), lr=0.1)
    FocalLoss(gamma=2.0)(ref(x), y).backward()
    opt.step()

    assert torch.allclose(trainer.model.weight, ref.weight, atol=1e-6, rtol=1e-5)
    assert torch.allclose(trainer.model.bias, ref.bias, atol=1e-6, rtol=1e-5)


# ============================================================
# F08 · val_loss 恶化监控
# ============================================================
def test_val_loss_monitor_sequence():
    """验收序列 [1.0, 1.06, 1.07, 1.08]，阈值 1.05、patience=3 → 第 4 个值后触发。"""
    min_loss, counter = None, 0
    triggered_at = None
    for i, v in enumerate([1.0, 1.06, 1.07, 1.08], start=1):
        min_loss, counter = update_val_loss_monitor(min_loss, counter, v, 1.05)
        if counter >= 3 and triggered_at is None:
            triggered_at = i
    assert triggered_at == 4, f"应第 4 个值后触发，实际 {triggered_at}"
    assert min_loss == 1.0


def test_val_loss_monitor_resets_on_recovery():
    """中间恢复到阈值以内时恶化计数清零。"""
    min_loss, counter = None, 0
    for v in [1.0, 1.06, 1.04, 1.06, 1.07]:
        min_loss, counter = update_val_loss_monitor(min_loss, counter, v, 1.05)
    assert counter == 2, f"1.04 处应清零，之后 1.06/1.07 计数=2，实际 {counter}"


# ============================================================
# F09 · 续训一致性
# ============================================================
def _tiny_image_data(n=16, seed=0):
    """固定小图像数据 (n, 1, 8, 8)（独立 generator；满足 torchvision 增强的形状要求）。"""
    g = torch.Generator().manual_seed(seed)
    x = torch.rand(n, 1, 8, 8, generator=g)
    y = torch.randint(0, 3, (n,), generator=g)
    return TensorDataset(x, y)


def _resume_env(tmp_path, run_name, *, augmentation=False, balanced_sampling=False, seed=11):
    """构造恢复一致性测试环境（R01：支持增强 / 加权采样组合）。"""
    config = _make_config(seed=seed)
    config["training"]["scheduler"] = "cosine"
    config["models"]["mini_cnn"]["num_epochs"] = 2
    torch.manual_seed(seed)
    model = nn.Sequential(nn.Flatten(), nn.Linear(64, 3))
    base = _tiny_image_data(n=16, seed=5)
    if augmentation:
        from torchvision import transforms

        dataset = _TransformDataset(base, transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
        ]))
    else:
        dataset = base
    if balanced_sampling:
        sampler = torch.utils.data.WeightedRandomSampler(
            weights=[1.0] * len(dataset), num_samples=len(dataset), replacement=True,
        )
        loader = SpyLoader(DataLoader(dataset, batch_size=4, sampler=sampler))
    else:
        loader = SpyLoader(DataLoader(dataset, batch_size=4, shuffle=True))
    val_loader = DataLoader(_tiny_image_data(n=16, seed=6), batch_size=4)
    trainer = Trainer(
        model=model, train_loader=loader, val_loader=val_loader,
        test_loader=val_loader, config=config, model_name="mini_cnn",
        device=torch.device("cpu"), run_dir=tmp_path / run_name,
    )
    return trainer, loader


@pytest.mark.parametrize(
    "augmentation,balanced_sampling",
    [(False, False), (True, False), (False, True), (True, True)],
    ids=["plain", "augmentation", "weighted_sampler", "aug_and_sampler"],
)
def test_resume_matches_continuous(tmp_path, augmentation, balanced_sampling):
    """连续两轮 == 一轮保存后恢复再一轮（R01：逐批输入哈希 + 参数 + history + best + 计数）。"""
    kwargs = {"augmentation": augmentation, "balanced_sampling": balanced_sampling}

    # A：连续两轮
    ta, la = _resume_env(tmp_path, "continuous", **kwargs)
    ta.fit(2)

    # B：一轮 → 保存 → 重建 Trainer 恢复 → 再一轮
    tb, _ = _resume_env(tmp_path, "resumed", **kwargs)
    tb.fit(1)
    last_ckpt = tb.checkpoints_dir / "last.pth"
    assert last_ckpt.exists()

    tb2, lb2 = _resume_env(tmp_path, "resumed", **kwargs)
    tb2.load_checkpoint(last_ckpt)
    assert tb2.start_epoch == 2
    tb2.fit(1)

    # R01：逐批实际输入（增强后张量哈希 + 标签序列）必须一致——不能只比标签
    assert lb2.batches == la.batches[4:], "恢复后的逐批输入与连续训练不一致"

    # history / LR / best / 早停计数
    for key in ["train_loss", "train_acc", "val_loss", "val_acc", "lr"]:
        assert len(ta.history[key]) == len(tb2.history[key]) == 2
        assert ta.history[key] == pytest.approx(tb2.history[key], rel=1e-6, abs=1e-8), key
    assert ta.best_val_acc == pytest.approx(tb2.best_val_acc, rel=1e-9)
    assert ta.best_epoch == tb2.best_epoch
    assert ta.acc_patience_counter == tb2.acc_patience_counter
    assert ta.loss_worse_counter == tb2.loss_worse_counter
    assert ta.hist_min_val_loss == pytest.approx(tb2.hist_min_val_loss, rel=1e-9)

    # 参数一致
    for p_a, p_b in zip(ta.model.parameters(), tb2.model.parameters(), strict=True):
        assert torch.allclose(p_a, p_b, atol=1e-6, rtol=1e-5)


def test_early_stop_state_roundtrip(tmp_path):
    """best / 早停计数在保存-恢复后保持一致。"""
    config = _make_config()
    t = _make_trainer(config, run_dir=tmp_path / "r1")
    t.best_val_acc = 0.7
    t.best_epoch = 5
    t.best_monitor_value = 0.7
    t.acc_patience_counter = 2
    t.hist_min_val_loss = 0.5
    t.loss_worse_counter = 3
    path = t.checkpoints_dir / "state.pth"
    t.save_checkpoint(path)

    t2 = _make_trainer(config, run_dir=tmp_path / "r2")
    t2.load_checkpoint(path)
    assert t2.best_val_acc == pytest.approx(0.7)
    assert t2.best_epoch == 5
    assert t2.acc_patience_counter == 2
    assert t2.hist_min_val_loss == pytest.approx(0.5)
    assert t2.loss_worse_counter == 3


# ============================================================
# F05 · run 隔离
# ============================================================
def test_two_runs_isolated(tmp_path, monkeypatch):
    """同一模型两个不同 seed 的运行：目录、best、last 完全分开，互不覆盖。"""
    import training.trainer as trainer_mod
    monkeypatch.setattr(trainer_mod, "RUNS_ROOT", tmp_path / "runs")

    config1 = _make_config(seed=1)
    config2 = _make_config(seed=2)
    t1 = _make_trainer(config1)
    t2 = _make_trainer(config2)
    assert t1.run_dir != t2.run_dir

    t1.fit(1)
    best1 = t1.checkpoints_dir / "best.pth"
    last1 = t1.checkpoints_dir / "last.pth"
    assert best1.exists() and last1.exists()
    sha_best1 = file_sha256(best1)
    sha_last1 = file_sha256(last1)

    t2.fit(1)
    # 第二次运行不覆盖第一次的文件
    assert file_sha256(best1) == sha_best1
    assert file_sha256(last1) == sha_last1
    assert (t2.checkpoints_dir / "best.pth").exists()

    # run_meta 各自独立，且记录 seed
    meta1 = read_run_meta(t1.run_dir)
    meta2 = read_run_meta(t2.run_dir)
    assert meta1["seed"] == 1 and meta2["seed"] == 2
    assert meta1["run_id"] != meta2["run_id"]


# ============================================================
# F09 · 断点健壮性
# ============================================================
def test_corrupted_checkpoint_raises(tmp_path):
    config = _make_config()
    t = _make_trainer(config, run_dir=tmp_path / "run")

    bad = tmp_path / "bad.pth"
    bad.write_bytes(b"this is definitely not a torch checkpoint")
    with pytest.raises(RuntimeError, match="损坏"):
        t.load_checkpoint(bad)

    invalid = tmp_path / "invalid.pth"
    torch.save({"some": "dict"}, invalid)
    with pytest.raises(RuntimeError, match="model_state_dict"):
        t.load_checkpoint(invalid)

    legacy = tmp_path / "legacy.pth"
    torch.save({"model_name": "mini_cnn", "model_state_dict": t.model.state_dict()}, legacy)
    with pytest.raises(RuntimeError, match="格式版本"):
        t.load_checkpoint(legacy)


def test_failed_save_preserves_previous(tmp_path, monkeypatch):
    """保存失败不破坏上一份可用断点，且不残留临时文件。"""
    config = _make_config()
    t = _make_trainer(config, run_dir=tmp_path / "run")
    ckpt_path = t.checkpoints_dir / "last.pth"
    t.save_checkpoint(ckpt_path)
    sha_before = file_sha256(ckpt_path)

    def broken_save(*args, **kwargs):
        raise RuntimeError("simulated disk failure")

    monkeypatch.setattr(torch, "save", broken_save)
    with pytest.raises(RuntimeError, match="simulated"):
        t.save_checkpoint(ckpt_path)

    assert file_sha256(ckpt_path) == sha_before
    leftovers = list(t.checkpoints_dir.glob("*.tmp"))
    assert leftovers == [], f"残留临时文件: {leftovers}"


def test_partial_checkpoint_marked(tmp_path):
    config = _make_config()
    t = _make_trainer(config, run_dir=tmp_path / "run")
    path = t.checkpoints_dir / "interrupted.pth"
    t.save_checkpoint(path, partial=True)
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    assert ckpt["partial"] is True
    assert ckpt["format_version"] == CHECKPOINT_FORMAT_VERSION


def test_keyboard_interrupt_saves_partial(tmp_path, monkeypatch):
    """fit 中 KeyboardInterrupt：保存 partial 断点 + run_meta 标记 interrupted。"""
    config = _make_config()
    t = _make_trainer(config, run_dir=tmp_path / "run")

    def boom(self, *args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(Trainer, "evaluate", boom)
    t.fit(1)

    ints = list(t.checkpoints_dir.glob("interrupted_*.pth"))
    assert len(ints) == 1
    ckpt = torch.load(ints[0], map_location="cpu", weights_only=False)
    assert ckpt["partial"] is True
    meta = read_run_meta(t.run_dir)
    assert meta["status"] == "interrupted"


# ============================================================
# F13 · 保存开关与监控指标
# ============================================================
def test_monitor_metric_val_loss(tmp_path):
    config = _make_config()
    config["checkpoint"]["monitor_metric"] = "val_loss"
    t = _make_trainer(config, run_dir=tmp_path / "run")
    t.fit(1)
    best = t.checkpoints_dir / "best.pth"
    assert best.exists()
    ckpt = torch.load(best, map_location="cpu", weights_only=False)
    assert ckpt["best"]["monitor_metric"] == "val_loss"


def test_save_best_false(tmp_path):
    config = _make_config()
    config["checkpoint"]["save_best"] = False
    t = _make_trainer(config, run_dir=tmp_path / "run")
    t.fit(1)
    assert not (t.checkpoints_dir / "best.pth").exists()
    assert (t.checkpoints_dir / "last.pth").exists()


def test_fit_writes_run_meta_and_history(tmp_path):
    config = _make_config()
    t = _make_trainer(config, run_dir=tmp_path / "run")
    t.fit(1)
    meta = read_run_meta(t.run_dir)
    assert meta["status"] == "finished"
    assert meta["final_epoch"] == 1
    assert (t.run_dir / "history.json").exists()
    assert (t.run_dir / "config_effective.yaml").exists()


# ============================================================
# F06 · 类别计数（固定索引、零计数校验、采样权重同源）
# ============================================================
def test_compute_class_counts_fixed_index():
    from data.dataloader import compute_class_counts

    counts = compute_class_counts(np.array([0, 1, 1, 3]), 7)
    assert counts == [1, 2, 0, 1, 0, 0, 0]  # 缺失类别保持固定索引位置
    with pytest.raises(ValueError, match="超出"):
        compute_class_counts(np.array([0, 7]), 7)


def test_sampler_weights_use_class_counts():
    from data.dataloader import _compute_sampler_weights

    df = pd.DataFrame({"emotion": [0, 0, 0, 1]})
    weights = _compute_sampler_weights(df, num_classes=2, class_counts=[3, 1])
    assert weights == pytest.approx([4 / 6, 4 / 6, 4 / 6, 4 / 2])
    with pytest.raises(ValueError, match="样本数为 0"):
        _compute_sampler_weights(df, num_classes=2, class_counts=[4, 0])


def test_cb_focal_rejects_zero_count():
    from utils.losses import CBFocalLoss

    with pytest.raises(ValueError, match="非正数"):
        CBFocalLoss(gamma=2.0, beta=0.999, class_counts=[100, 0, 50])

# ============================================================
# R01 · 精确恢复支持范围（审计第二轮）
# ============================================================
def test_enforce_exact_resume_conditions():
    from training.trainer import enforce_exact_resume_conditions

    cfg = {"dataloader": {"num_workers": 4, "persistent_workers": True}}
    info = enforce_exact_resume_conditions(cfg)
    assert cfg["dataloader"]["num_workers"] == 0
    assert cfg["dataloader"]["persistent_workers"] is False
    assert info["adjusted"] is True and info["original_num_workers"] == 4

    cfg2 = {"dataloader": {"num_workers": 0}}
    info2 = enforce_exact_resume_conditions(cfg2)
    assert info2["adjusted"] is False


def test_load_checkpoint_rejects_multi_worker_loader(tmp_path):
    config = _make_config()
    t = _make_trainer(config, run_dir=tmp_path / "base")
    path = t.checkpoints_dir / "last.pth"
    t.save_checkpoint(path)

    t2 = _make_trainer(config, run_dir=tmp_path / "resume")
    t2.train_loader.num_workers = 2  # 模拟多进程数据管线
    with pytest.raises(RuntimeError, match="num_workers=0"):
        t2.load_checkpoint(path)


def test_resume_success_records_event(tmp_path):
    config = _make_config()
    t = _make_trainer(config, run_dir=tmp_path / "run")
    path = t.checkpoints_dir / "last.pth"
    t.save_checkpoint(path)

    t2 = _make_trainer(config, run_dir=t.run_dir)
    t2.load_checkpoint(path)

    meta = read_run_meta(t.run_dir)
    events = meta.get("resume_events", [])
    assert len(events) == 1
    ev = events[0]
    assert ev["protocol_verified"] is True
    assert ev["checkpoint_sha256"] == file_sha256(path)
    assert "git_commit" in ev and "environment" in ev and "config_effective_sha256" in ev


# ============================================================
# R02 · 恢复前训练协议比对（换配置不得续写原 run）
# ============================================================
@pytest.mark.parametrize(
    "mutate,match",
    [
        (lambda c: c["training"].update(scheduler="cosine"), "scheduler"),
        (lambda c: c["training"].update(cudnn_deterministic=True), "cudnn_deterministic"),
        (lambda c: c["checkpoint"].update(monitor_metric="val_loss"), "monitor_metric"),
        (lambda c: c["dataloader"].update(class_balanced_sampling=True),
         "class_balanced_sampling"),
        (lambda c: c["augmentation"].update(enabled=True), "augmentation"),
    ],
)
def test_resume_rejects_protocol_change(tmp_path, mutate, match):
    config = _make_config()
    t = _make_trainer(config, run_dir=tmp_path / "base")
    path = t.checkpoints_dir / "last.pth"
    t.save_checkpoint(path)

    changed = copy.deepcopy(config)
    mutate(changed)
    t2 = _make_trainer(changed, run_dir=tmp_path / "resume")
    with pytest.raises(RuntimeError, match=match):
        t2.load_checkpoint(path)


def test_resume_rejects_batch_size_change(tmp_path):
    config = _make_config()
    t = _make_trainer(config, run_dir=tmp_path / "base")
    path = t.checkpoints_dir / "last.pth"
    t.save_checkpoint(path)

    loader = DataLoader(_tiny_data(), batch_size=3, shuffle=False)
    t2 = _make_trainer(config, loader=loader, run_dir=tmp_path / "resume")
    with pytest.raises(RuntimeError, match="train_batch_size"):
        t2.load_checkpoint(path)


def test_resume_rejects_focal_gamma_change(tmp_path):
    config = _make_config()
    t = _make_trainer(config, run_dir=tmp_path / "base")
    path = t.checkpoints_dir / "last.pth"
    t.save_checkpoint(path)

    t2 = _make_trainer(config, run_dir=tmp_path / "resume", focal_gamma=3.0)
    with pytest.raises(RuntimeError, match="focal_gamma"):
        t2.load_checkpoint(path)


def test_resume_rejects_data_fingerprint_change(tmp_path):
    config = _make_config()
    t = _make_trainer(config, run_dir=tmp_path / "base")
    t.train_loader.dataset.split_fingerprint = {"version": 1}
    path = t.checkpoints_dir / "last.pth"
    t.save_checkpoint(path)

    t2 = _make_trainer(config, run_dir=tmp_path / "resume")
    t2.train_loader.dataset.split_fingerprint = {"version": 2}
    with pytest.raises(RuntimeError, match="data_fingerprint"):
        t2.load_checkpoint(path)


def test_resume_failure_leaves_artifacts_untouched(tmp_path):
    """校验失败时：模型未被加载、原 run 文件逐字节不变、无恢复事件写入。"""
    config = _make_config()
    t = _make_trainer(config, run_dir=tmp_path / "run")
    path = t.checkpoints_dir / "last.pth"
    t.save_checkpoint(path)
    run_dir = t.run_dir
    meta_path = run_dir / "run_meta.json"
    meta_sha = file_sha256(meta_path)
    effective_path = run_dir / "config_effective.yaml"
    effective_sha = file_sha256(effective_path)

    changed = copy.deepcopy(config)
    changed["training"]["scheduler"] = "cosine"
    t2 = _make_trainer(changed, run_dir=run_dir)  # resume 场景：复用原 run 目录
    params_before = [p.detach().clone() for p in t2.model.parameters()]

    with pytest.raises(RuntimeError, match="scheduler"):
        t2.load_checkpoint(path)

    assert file_sha256(meta_path) == meta_sha, "失败时改写了 run_meta.json"
    assert file_sha256(effective_path) == effective_sha, "失败时改写了 config_effective.yaml"
    meta = read_run_meta(run_dir)
    assert meta.get("resume_events", []) == [], "校验失败却写入了恢复事件"
    for before, after in zip(params_before, t2.model.parameters(), strict=True):
        assert torch.equal(before, after), "校验失败前已改动模型权重"


# ============================================================
# R03 · 诊断完全隔离
# ============================================================
class _TinyBNNet(nn.Module):
    """带 BatchNorm 的小模型（覆盖 BN buffer 的状态不变性检查）。"""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 8, 3, padding=1)
        self.bn = nn.BatchNorm2d(8)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(8, 3)

    def forward(self, x):
        x = torch.relu(self.bn(self.conv(x)))
        x = self.pool(x).flatten(1)
        return self.fc(x)


def _state_equal(a, b) -> bool:
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(_state_equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(
            _state_equal(x, y) for x, y in zip(a, b, strict=True)
        )
    if isinstance(a, torch.Tensor):
        return torch.equal(a, b)
    return a == b


def test_diagnose_does_not_touch_training_state(tmp_path):
    """诊断前后：参数/BN buffer、优化器、RNG、run 文件逐项不变。"""
    config = _make_config()
    torch.manual_seed(config["seed"])
    model = _TinyBNNet()
    loader = DataLoader(_tiny_image_data(n=16, seed=3), batch_size=4, shuffle=True)
    val_loader = DataLoader(_tiny_image_data(n=16, seed=4), batch_size=4)
    t = Trainer(
        model=model, train_loader=loader, val_loader=val_loader,
        test_loader=val_loader, config=config, model_name="mini_cnn",
        device=torch.device("cpu"), run_dir=tmp_path / "run",
    )
    t.fit(1)  # 使 optimizer 有状态、BN 有 running stats、history 非空

    params_before = {k: v.detach().clone() for k, v in t.model.state_dict().items()}
    opt_before = copy.deepcopy(t.optimizer.state_dict())
    rng_torch = torch.get_rng_state()
    rng_numpy = np.random.get_state()
    rng_python = random.getstate()
    files_before = {
        str(p.relative_to(t.run_dir)): file_sha256(p)
        for p in sorted(t.run_dir.rglob("*")) if p.is_file()
    }

    t.diagnose(num_steps=2)

    for key, value in t.model.state_dict().items():
        assert torch.equal(params_before[key], value), f"诊断改动了 {key}"
    assert _state_equal(opt_before, t.optimizer.state_dict()), "诊断改动了优化器状态"
    assert torch.equal(rng_torch, torch.get_rng_state()), "torch RNG 未恢复"
    np_after = np.random.get_state()
    assert rng_numpy[0] == np_after[0]
    assert np.array_equal(rng_numpy[1], np_after[1])
    assert rng_numpy[2] == np_after[2] and rng_numpy[3] == np_after[3]
    assert rng_python == random.getstate(), "python RNG 未恢复"
    assert len(t.history["train_loss"]) == 1, "诊断改动了训练历史"

    files_after = {
        str(p.relative_to(t.run_dir)): file_sha256(p)
        for p in sorted(t.run_dir.rglob("*")) if p.is_file()
    }
    assert files_after == files_before, "诊断改动了 run 文件"


def test_diagnose_then_fit_equals_direct_fit(tmp_path):
    """同 seed 下：'先诊断再 fit' 与 '直接 fit' 同批输入、同等结果。"""

    def build(name):
        config = _make_config(seed=7)
        torch.manual_seed(7)
        model = nn.Linear(4, 3)
        loader = SpyLoader(DataLoader(_tiny_data(n=16, seed=8), batch_size=4, shuffle=True))
        val_loader = DataLoader(_tiny_data(n=16, seed=9), batch_size=4)
        trainer = Trainer(
            model=model, train_loader=loader, val_loader=val_loader,
            test_loader=val_loader, config=config, model_name="mini_cnn",
            device=torch.device("cpu"), run_dir=tmp_path / name,
        )
        return trainer, loader

    ta, la = build("direct")
    ta.fit(1)

    tb, lb = build("diagnosed")
    tb.diagnose(num_steps=1)
    tb.fit(1)

    assert lb.batches == la.batches, "先诊断改变了正式训练的逐批输入"
    assert ta.history["val_acc"] == pytest.approx(tb.history["val_acc"])
    for p_a, p_b in zip(ta.model.parameters(), tb.model.parameters(), strict=True):
        assert torch.allclose(p_a, p_b, atol=1e-6, rtol=1e-5)


# ============================================================
# R04 · 同一实例重复 fit（轮号 / 定期断点 / 累计时长）
# ============================================================
def _r04_env(tmp_path, run_name, *, patience=None, seed=13):
    config = _make_config(seed=seed)
    config["training"]["scheduler"] = "cosine"
    config["models"]["mini_cnn"]["num_epochs"] = 4
    if patience is not None:
        config["training"]["patience"] = patience
    torch.manual_seed(13)
    model = nn.Linear(4, 3)
    loader = SpyLoader(DataLoader(_tiny_data(n=16, seed=10), batch_size=4, shuffle=True))
    val_loader = DataLoader(_tiny_data(n=16, seed=11), batch_size=4)
    trainer = Trainer(
        model=model, train_loader=loader, val_loader=val_loader,
        test_loader=val_loader, config=config, model_name="mini_cnn",
        device=torch.device("cpu"), run_dir=tmp_path / run_name,
    )
    return trainer, loader


def test_fit_twice_same_instance_equals_fit_2(tmp_path):
    ta, la = _r04_env(tmp_path, "twice")
    ta.fit(1)
    dur_once = ta._accumulated_train_time
    ta.fit(1)

    tb, lb = _r04_env(tmp_path, "once")
    tb.fit(2)

    # 轮号推进与定期断点
    assert ta.start_epoch == 3
    assert la.batches == lb.batches, "分两次 fit 的逐批输入与一次 fit(2) 不一致"
    for key in ("train_loss", "val_acc", "lr"):
        assert ta.history[key] == pytest.approx(tb.history[key], rel=1e-6), key
    for p_a, p_b in zip(ta.model.parameters(), tb.model.parameters(), strict=True):
        assert torch.allclose(p_a, p_b, atol=1e-6, rtol=1e-5)

    assert (ta.checkpoints_dir / "epoch_0001.pth").exists()
    assert (ta.checkpoints_dir / "epoch_0002.pth").exists()
    e1 = torch.load(ta.checkpoints_dir / "epoch_0001.pth",
                    map_location="cpu", weights_only=False)
    e2 = torch.load(ta.checkpoints_dir / "epoch_0002.pth",
                    map_location="cpu", weights_only=False)
    assert e1["epoch"] == 1 and e2["epoch"] == 2

    last = torch.load(ta.checkpoints_dir / "last.pth", map_location="cpu", weights_only=False)
    assert last["epoch"] == 2
    meta = read_run_meta(ta.run_dir)
    assert meta["final_epoch"] == 2

    # 累计时长递增且被写入断点
    assert ta._accumulated_train_time >= dur_once > 0
    assert last["training_duration_seconds"] > 0


def test_fit_after_early_stop_continues(tmp_path, monkeypatch):
    t, _ = _r04_env(tmp_path, "es", patience=1, seed=21)
    t2model = t.model

    calls = {"n": 0}

    def fake_eval(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            return 0.5, 0.5, 0.9   # epoch 1：best 改善
        return 0.9, 0.01, 0.05     # 之后：不改善 → 立即触发 acc 早停

    monkeypatch.setattr(t, "evaluate", fake_eval)
    t.fit(3)
    assert len(t.history["train_loss"]) == 2, "应在第 2 轮触发早停"
    assert t.start_epoch == 3

    # 早停后再 fit：从第 3 轮继续且再次触发（epoch 轮号不重复）
    t.fit(1)
    assert len(t.history["train_loss"]) == 3
    meta = read_run_meta(t.run_dir)
    assert meta["final_epoch"] == 3
    assert t2model is t.model


# ============================================================
# R05 · Trainer 默认设备（省略 device 时模型必须移动到正确设备）
# ============================================================
@pytest.mark.skipif(not torch.cuda.is_available(), reason="需要 CUDA")
def test_trainer_default_device_cuda(tmp_path):
    config = _make_config(seed=1)
    config["training"]["amp"] = True
    torch.manual_seed(1)
    model = nn.Linear(4, 3)
    loader = DataLoader(_tiny_data(n=16), batch_size=4)
    t = Trainer(
        model=model, train_loader=loader, val_loader=loader, test_loader=loader,
        config=config, model_name="mini_cnn", run_dir=tmp_path / "cuda_default",
    )  # 不传 device
    assert t.device.type == "cuda"
    assert next(t.model.parameters()).device.type == "cuda"
    assert t.use_amp is True

    before = [p.detach().clone() for p in t.model.parameters()]
    t._current_epoch = 1
    t.train_one_epoch()
    for a, b in zip(before, t.model.parameters(), strict=True):
        assert torch.isfinite(b).all()
        assert not torch.equal(a, b), "GPU 路径未发生有效参数更新"

    # 显式 cpu 路径
    t_cpu = Trainer(
        model=nn.Linear(4, 3), train_loader=loader, val_loader=loader, test_loader=loader,
        config=_make_config(seed=1), model_name="mini_cnn", device="cpu",
        run_dir=tmp_path / "cpu_explicit",
    )
    assert t_cpu.device.type == "cpu"
    assert next(t_cpu.model.parameters()).device.type == "cpu"


# ============================================================
# R06 · 增强总开关覆盖 MixUp / 类别专属增强
# ============================================================
@pytest.mark.parametrize(
    "master,sub_mixup,expected",
    [(False, False, False), (False, True, False), (True, False, False), (True, True, True)],
)
def test_mixup_effective_switch(tmp_path, monkeypatch, master, sub_mixup, expected):
    import training.trainer as trainer_mod

    config = _make_config()
    config["augmentation"]["enabled"] = master
    config["augmentation"]["mixup"] = {"enabled": sub_mixup, "alpha": 0.2}
    t = _make_trainer(config, run_dir=tmp_path / f"mix_{int(master)}{int(sub_mixup)}")
    assert t.mixup_enabled is expected

    calls = []
    real_mixup = trainer_mod.mixup_data

    def spy(*args, **kwargs):
        calls.append(1)
        return real_mixup(*args, **kwargs)

    monkeypatch.setattr(trainer_mod, "mixup_data", spy)
    t._current_epoch = 1
    t.train_one_epoch()  # 共 4 批
    assert len(calls) == (4 if expected else 0), "有效 MixUp 开关与实际调用不一致"


def test_class_aug_respects_master_switch():
    from data.dataloader import build_class_aug_transform

    aug = {"enabled": True, "target_classes": [1], "extra_rotation": 20, "augment_prob": 0.8}
    assert build_class_aug_transform(aug, master_enabled=False) == {}
    assert build_class_aug_transform(aug, master_enabled=True) != {}


# ============================================================
# R07 · 校验进入 Trainer 构造入口（Notebook/API 与 CLI 同语义）
# ============================================================
def test_trainer_construction_rejects_invalid_config(tmp_path):
    from utils.config_validation import ConfigValidationError

    mutations = [
        lambda c: c["training"].update(loss_type="misspelled_loss"),
        lambda c: c["training"].update(learning_rate=float("nan")),
        lambda c: c["training"].update(batch_size=0),
        lambda c: c.update(seed=-5),
    ]
    for i, mutate in enumerate(mutations):
        cfg = _make_config()
        mutate(cfg)
        run_dir = tmp_path / f"invalid_{i}"
        with pytest.raises(ConfigValidationError):
            _make_trainer(cfg, run_dir=run_dir)
        assert not run_dir.exists(), "配置校验失败前不应创建 run 目录"

# ============================================================
# R08 · CE 基线分支与基线配置
# ============================================================
def test_cross_entropy_branch_equals_torch_ce(tmp_path):
    config = _make_config()
    config["training"]["loss_type"] = "cross_entropy"
    t = _make_trainer(config, run_dir=tmp_path / "ce")
    assert isinstance(t.criterion, nn.CrossEntropyLoss)
    x = torch.randn(8, 3)
    y = torch.randint(0, 3, (8,))
    assert torch.allclose(t.criterion(x, y), torch.nn.functional.cross_entropy(x, y))


def test_baseline_config_is_valid():
    """configs/baseline_config.yaml 通过集中校验（CE 基线可由同一入口启动）。"""
    from training.trainer import load_config as _load_config
    from utils.config_validation import validate_config

    cfg = _load_config(str(PROJECT_ROOT / "configs" / "baseline_config.yaml"))
    for model in ("mini_cnn", "vgg_lite", "micro_resnet"):
        validate_config(cfg, model_name=model)
    assert cfg["training"]["loss_type"] == "cross_entropy"
    assert cfg["dataloader"]["class_balanced_sampling"] is False
    assert cfg["augmentation"]["class_specific"]["enabled"] is False


def test_baseline_config_differs_only_in_three_fields():
    """防漂移：baseline 与主配置除三处基线差异外逐项相同。"""
    from training.trainer import load_config as _load_config

    main = _load_config()
    base = _load_config(str(PROJECT_ROOT / "configs" / "baseline_config.yaml"))
    m, b = copy.deepcopy(main), copy.deepcopy(base)
    m["training"]["loss_type"] = "X"
    b["training"]["loss_type"] = "X"
    m["dataloader"]["class_balanced_sampling"] = "X"
    b["dataloader"]["class_balanced_sampling"] = "X"
    m["augmentation"]["class_specific"]["enabled"] = "X"
    b["augmentation"]["class_specific"]["enabled"] = "X"
    assert m == b, "baseline_config.yaml 与主配置出现了三处以外的差异（需同步）"

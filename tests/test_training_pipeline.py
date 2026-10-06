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


def _make_trainer(config, model=None, loader=None, val_loader=None, run_dir=None):
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
    )


class SpyLoader:
    """包装 DataLoader，记录每批标签顺序（用于验证续训的批次顺序一致）。"""

    def __init__(self, loader):
        self._loader = loader
        self.seen = []

    def __iter__(self):
        for xb, yb in self._loader:
            self.seen.append(tuple(int(v) for v in yb.tolist()))
            yield xb, yb

    def __len__(self):
        return len(self._loader)

    def __getattr__(self, name):
        return getattr(self._loader, name)


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
def test_resume_matches_continuous(tmp_path):
    """连续两轮 == 一轮保存后恢复再一轮（含批次顺序、参数、history、best、计数）。"""

    def build(run_name):
        config = _make_config(seed=11)
        config["training"]["scheduler"] = "cosine"
        config["models"]["mini_cnn"]["num_epochs"] = 2
        torch.manual_seed(11)
        model = nn.Linear(4, 3)
        loader = SpyLoader(DataLoader(_tiny_data(n=16, seed=5), batch_size=4, shuffle=True))
        trainer = Trainer(
            model=model, train_loader=loader,
            val_loader=DataLoader(_tiny_data(n=16, seed=6), batch_size=4),
            test_loader=DataLoader(_tiny_data(n=16, seed=6), batch_size=4),
            config=config, model_name="mini_cnn", device=torch.device("cpu"),
            run_dir=tmp_path / run_name,
        )
        return trainer, loader

    # A：连续两轮
    ta, la = build("continuous")
    ta.fit(2)

    # B：一轮 → 保存 → 恢复 → 再一轮
    tb, _ = build("resumed")
    tb.fit(1)
    last_ckpt = tb.checkpoints_dir / "last.pth"
    assert last_ckpt.exists()

    tb2, lb2 = build("resumed")
    tb2.load_checkpoint(last_ckpt)
    assert tb2.start_epoch == 2
    tb2.fit(1)

    # 批次顺序：B 恢复后第一轮 == A 的第二轮
    assert lb2.seen == la.seen[4:], "恢复后的批次顺序与连续训练不一致"

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

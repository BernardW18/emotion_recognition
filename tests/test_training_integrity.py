"""T01/T03 专项测试：partial 显式加载回滚 + 状态完整性预检。

- T01：显式加载 partial 断点：同 run 有完整 last → 自动回滚（丢弃额外更新）；
       无 last / last 亦 partial / last 不同 run → 拒绝；拒绝前后状态与文件不变。
- T03：状态完整性预检：删除/破坏必需状态（loader_rng 键、全局 RNG、调度器、scaler）
       在原状态与文件改变前拒绝；合法断点通过且事件分开记录协议/状态。
"""

import shutil

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from training.checkpoint import read_run_meta
from training.trainer import Trainer
from utils.model_spec import file_sha256


# ============================================================
# 环境 helpers
# ============================================================
def _tiny_image_data(n=16, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.rand(n, 1, 8, 8, generator=g)
    y = torch.randint(0, 3, (n,), generator=g)
    return TensorDataset(x, y)


def _make_config(*, scheduler="cosine", amp=False, seed=11):
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
            "optimizer": "adam", "scheduler": scheduler, "patience": 0,
            "val_loss_patience": 0, "val_loss_threshold": 1.05,
            "loss_type": "focal", "amp": amp, "gradient_accumulation_steps": 1,
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


def _env(tmp_path, name, *, scheduler="cosine", amp=False, seed=11, workers=0):
    """恢复一致性测试环境（独立生成器；与 create_dataloaders 语义一致）。"""
    config = _make_config(scheduler=scheduler, amp=amp, seed=seed)
    torch.manual_seed(seed)
    model = nn.Sequential(nn.Flatten(), nn.Linear(64, 3))
    g_sampler = torch.Generator().manual_seed(seed)
    g_loader = torch.Generator().manual_seed(seed + 12345)
    sampler = torch.utils.data.RandomSampler(
        _tiny_image_data(n=16, seed=5), generator=g_sampler
    )
    loader_kwargs = dict(batch_size=4, sampler=sampler, generator=g_loader)
    if workers > 0:
        loader_kwargs.update(num_workers=workers, persistent_workers=False, prefetch_factor=2)
    loader = DataLoader(_tiny_image_data(n=16, seed=5), **loader_kwargs)
    val_loader = DataLoader(_tiny_image_data(n=16, seed=6), batch_size=4)
    device = torch.device("cuda" if amp else "cpu")
    trainer = Trainer(
        model=model, train_loader=loader, val_loader=val_loader,
        test_loader=val_loader, config=config, model_name="mini_cnn",
        device=device, run_dir=tmp_path / name,
    )
    return trainer


def _params(trainer):
    return [p.detach().clone() for p in trainer.model.parameters()]


def _same_params(before, after):
    return all(torch.equal(a, b) for a, b in zip(before, after, strict=True))


# ============================================================
# T01 · partial 显式加载
# ============================================================
def test_explicit_partial_without_last_rejected(tmp_path):
    """无完整 last：拒绝，且拒绝前后文件 SHA 与真实参数不变、无恢复事件。"""
    t = _env(tmp_path, "a")
    partial_path = t.checkpoints_dir / "interrupted_epoch001_x.pth"
    t.save_checkpoint(partial_path, partial=True)

    t2 = _env(tmp_path, "b")
    params_before = _params(t2)
    sha_before = file_sha256(partial_path)
    meta_before = read_run_meta(t2.run_dir)

    with pytest.raises(RuntimeError, match="last"):
        t2.load_checkpoint(partial_path)

    assert file_sha256(partial_path) == sha_before, "拒绝时修改了断点文件"
    assert _same_params(params_before, _params(t2)), "拒绝时改动了模型参数"
    meta_after = read_run_meta(t2.run_dir)
    assert meta_after.get("resume_events", []) == meta_before.get("resume_events", []), \
        "拒绝却写入了恢复事件"


def test_explicit_partial_rolls_back_to_complete_last(tmp_path):
    """核心场景：partial 含额外更新 → 显式加载自动回滚到同 run 完整 last，
    额外更新被丢弃；继续训练与连续训练一致。"""
    # 干净的 epoch1 状态（ta）
    ta = _env(tmp_path, "clean")
    ta.fit(1)
    ta_last = ta.checkpoints_dir / "last.pth"
    assert ta_last.exists()

    # 同 run 的“中断”环境：读入 epoch1 → 保存自己的完整 last → 污染 → 保存 partial
    tb = _env(tmp_path, "interrupted")
    tb.load_checkpoint(ta_last)                     # tb 状态 = 干净 epoch1
    tb.save_checkpoint(tb.checkpoints_dir / "last.pth")   # tb 自己的完整 last
    with torch.no_grad():
        for p in tb.model.parameters():
            p.add_(0.25)                            # 模拟未完成轮的额外更新
    partial_path = tb.checkpoints_dir / "interrupted_epoch002_x.pth"
    tb.save_checkpoint(partial_path, partial=True)

    # 显式加载 partial → 自动回滚到同 run 完整 last
    tc = _env(tmp_path, "restored")
    tc.load_checkpoint(partial_path)
    assert tc.start_epoch == 2

    clean_sd = torch.load(tb.checkpoints_dir / "last.pth", map_location="cpu",
                          weights_only=False)["model_state_dict"]
    for key, value in tc.model.state_dict().items():
        assert torch.equal(value, clean_sd[key]), f"{key} 未回滚到完整断点（额外更新残留）"

    meta = read_run_meta(tc.run_dir)
    event = meta["resume_events"][-1]
    assert event["protocol_verified"] is True
    assert any("自动回滚" in n for n in event["notes"]), event["notes"]
    assert event["checkpoint"].endswith("last.pth")

    # 继续训练：恢复的第 2 轮 == 连续训练的第 2 轮
    ta.fit(1)
    tc.fit(1)
    for p_a, p_c in zip(ta.model.parameters(), tc.model.parameters(), strict=True):
        assert torch.allclose(p_a, p_c, atol=1e-6, rtol=1e-5)
    assert ta.history["val_acc"][-1] == pytest.approx(tc.history["val_acc"][-1], abs=1e-9)


def test_explicit_partial_last_also_partial_rejected(tmp_path):
    t = _env(tmp_path, "a")
    t.save_checkpoint(t.checkpoints_dir / "last.pth", partial=True)
    partial_path = t.checkpoints_dir / "interrupted_epoch001_y.pth"
    t.save_checkpoint(partial_path, partial=True)

    t2 = _env(tmp_path, "b")
    with pytest.raises(RuntimeError, match="last.pth 也是 partial"):
        t2.load_checkpoint(partial_path)


def test_explicit_partial_foreign_last_rejected(tmp_path):
    """last.pth 属于另一个 run（run_id 不同）→ 拒绝回滚。"""
    ta = _env(tmp_path, "A")
    ta.fit(1)

    tb = _env(tmp_path, "B")
    partial_path = tb.checkpoints_dir / "interrupted_epoch001_z.pth"
    tb.save_checkpoint(partial_path, partial=True)
    shutil.copy(ta.checkpoints_dir / "last.pth", tb.checkpoints_dir / "last.pth")

    tc = _env(tmp_path, "C")
    with pytest.raises(RuntimeError, match="同一 run"):
        tc.load_checkpoint(partial_path)


# ============================================================
# T03 · 状态完整性预检
# ============================================================
def _save_with_mutation(tmp_path, name, mutate):
    """训练一步保存合法断点，再按 mutate 篡改后存为独立文件（不改原 last）。"""
    t = _env(tmp_path, name)
    t.fit(1)
    src = t.checkpoints_dir / "last.pth"
    ckpt = torch.load(src, map_location="cpu", weights_only=False)
    mutate(ckpt)
    tampered = t.checkpoints_dir / "tampered.pth"
    torch.save(ckpt, tampered)
    return tampered


_LOADER_RNG_MUTATIONS = {
    "empty_dict": lambda ck: ck.__setitem__("loader_rng", {}),
    "none_values": lambda ck: ck.__setitem__(
        "loader_rng", {"sampler_generator": None, "loader_generator": None}
    ),
    "bad_tensor": lambda ck: ck.__setitem__(
        "loader_rng", {"sampler_generator": torch.zeros(3), "loader_generator": torch.zeros(3)}
    ),
}


@pytest.mark.parametrize(
    "mutation", sorted(_LOADER_RNG_MUTATIONS), ids=sorted(_LOADER_RNG_MUTATIONS)
)
def test_precheck_rejects_broken_loader_rng(tmp_path, mutation):
    tampered = _save_with_mutation(tmp_path, "a", _LOADER_RNG_MUTATIONS[mutation])
    t2 = _env(tmp_path, "b")
    params_before = _params(t2)
    sha_before = file_sha256(tampered)

    with pytest.raises(RuntimeError, match="sampler_generator|loader_generator|loader_rng"):
        t2.load_checkpoint(tampered)

    assert file_sha256(tampered) == sha_before
    assert _same_params(params_before, _params(t2)), "预检拒绝前已改动模型参数"
    assert read_run_meta(t2.run_dir).get("resume_events", []) == [], "拒绝却写入了恢复事件"


_RNG_MUTATIONS = {
    "missing_rng": lambda ck: ck.pop("rng"),
    "bad_torch": lambda ck: ck["rng"].__setitem__("torch", torch.zeros(3)),
    "bad_numpy": lambda ck: ck["rng"].__setitem__("numpy", ("bad",)),
    "bad_python": lambda ck: ck["rng"].__setitem__("python", ("bad",)),
}


@pytest.mark.parametrize("mutation", sorted(_RNG_MUTATIONS), ids=sorted(_RNG_MUTATIONS))
def test_precheck_rejects_broken_global_rng(tmp_path, mutation):
    tampered = _save_with_mutation(tmp_path, "a", _RNG_MUTATIONS[mutation])
    t2 = _env(tmp_path, "b")
    with pytest.raises(RuntimeError, match="rng|RNG"):
        t2.load_checkpoint(tampered)


_SCHED_MUTATIONS = {
    "missing_scheduler_state": lambda ck: ck.pop("scheduler_state_dict"),
    "broken_scheduler_state": lambda ck: ck.__setitem__("scheduler_state_dict", {"T_0": "bad"}),
}


@pytest.mark.parametrize("mutation", sorted(_SCHED_MUTATIONS), ids=sorted(_SCHED_MUTATIONS))
def test_precheck_rejects_missing_or_broken_scheduler_state(tmp_path, mutation):
    tampered = _save_with_mutation(tmp_path, "a", _SCHED_MUTATIONS[mutation])
    t2 = _env(tmp_path, "b")   # 当前训练带 cosine 调度器 → 断点必须携带其状态
    with pytest.raises(RuntimeError, match="scheduler"):
        t2.load_checkpoint(tampered)


def test_precheck_allows_optional_absent_states(tmp_path):
    """无调度器（scheduler=none）/ CPU 无 scaler：不误拒，且事件分项记录。"""
    t = _env(tmp_path, "a", scheduler="none")
    t.fit(1)
    last = t.checkpoints_dir / "last.pth"

    t2 = _env(tmp_path, "b", scheduler="none")
    t2.load_checkpoint(last)

    event = read_run_meta(t2.run_dir)["resume_events"][-1]
    assert event["protocol_verified"] is True
    integrity = event["state_integrity"]
    assert integrity["scheduler"] == "not-used"
    assert integrity["amp_scaler"] == "not-used"
    assert integrity["global_rng"] == "ok"
    assert integrity["loader_rng"] == "ok"


def test_precheck_passes_with_workers2(tmp_path):
    """多 worker（non-persistent）合法断点：预检通过、恢复成功。"""
    t = _env(tmp_path, "a", workers=2)
    t.fit(1)
    last = t.checkpoints_dir / "last.pth"

    t2 = _env(tmp_path, "b", workers=2)
    t2.load_checkpoint(last)
    assert t2.start_epoch == 2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="需要 CUDA（AMP scaler）")
def test_precheck_rejects_missing_or_broken_scaler_state(tmp_path):
    def run_case(name, mutate, match):
        t = _env(tmp_path, name, amp=True)
        t.fit(1)   # 真实训练一轮：AMP 断点含已初始化的 scaler 状态
        src = t.checkpoints_dir / "last.pth"
        ckpt = torch.load(src, map_location="cpu", weights_only=False)
        mutate(ckpt)
        tampered = t.checkpoints_dir / "tampered.pth"
        torch.save(ckpt, tampered)
        t2 = _env(tmp_path, name + "_r", amp=True)
        with pytest.raises(RuntimeError, match=match):
            t2.load_checkpoint(tampered)

    run_case("amp1", lambda ck: ck.pop("scaler_state_dict"), "scaler")
    run_case("amp2", lambda ck: ck.__setitem__("scaler_state_dict", {"scale": "bad"}), "scaler")

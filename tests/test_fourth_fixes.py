"""Independent audit counterexamples and valid full-state/formal/cache regressions."""
import copy
import json

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms

import data.pixel_cache as pc
import training.checkpoint as ck
from data.dataloader import FER2013Dataset, create_dataloaders
from models import MiniCNN
from tests.test_pixel_cache import _write_mini_csv
from tests.test_training_integrity import _env
from training.trainer import Trainer, load_config
from utils.comparison_check import check_formal_eligibility
from utils.formal_protocol import code_fingerprint, load_frozen_protocol
from utils.model_spec import build_model_from_spec, file_sha256, make_spec_from_config


def _same(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b)
    elif isinstance(a, np.ndarray):
        assert np.array_equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            _same(a[key], b[key])
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b)
        for left, right in zip(a, b, strict=True):
            _same(left, right)
    else:
        assert a == b


def _live(t):
    return copy.deepcopy({
        "model": t.model.state_dict(), "optimizer": t.optimizer.state_dict(),
        "scheduler": t.scheduler.state_dict() if t.scheduler else None,
        "scaler": t.scaler.state_dict() if t.scaler else None,
        "rng": ck._capture_rng_state(), "loader": ck._capture_loader_rng(t.train_loader),
        "history": t.history, "best": (t.best_epoch, t.best_val_acc, t.best_monitor_value),
        "early": (t.acc_patience_counter, t.loss_worse_counter, t.hist_min_val_loss),
        "progress": (t.start_epoch, t._optimizer_updates, t._optimizer_attempts),
        "meta": t.run_meta,
    })


def _mutate(checkpoint, case):
    if case.startswith("missing_"):
        checkpoint.pop(case.removeprefix("missing_"))
    elif case == "empty_adam":
        checkpoint["optimizer_state_dict"]["state"] = {}
    elif case == "group_truncated":
        checkpoint["optimizer_state_dict"]["param_groups"][0]["params"].pop()
    elif case == "unknown_id":
        states = checkpoint["optimizer_state_dict"]["state"]
        states[10000] = states.pop(next(iter(states)))
    elif case == "bad_model_value":
        state = checkpoint["model_state_dict"]
        state[list(state)[-1]] = "bad"
    elif case == "bad_scheduler":
        checkpoint["scheduler_state_dict"]["last_epoch"] = "invalid"
    elif case == "bad_optimizer_eps":
        checkpoint["optimizer_state_dict"]["param_groups"][0]["eps"] = -1
    elif case == "wrong_epoch":
        checkpoint["epoch"] += 1
    elif case == "wrong_best":
        checkpoint["best"]["epoch"] += 10
    elif case == "negative_second_moment":
        state = checkpoint["optimizer_state_dict"]["state"]
        state[next(iter(state))]["exp_avg_sq"].fill_(-1)
    elif case == "bad_cycle":
        checkpoint["scheduler_state_dict"]["T_i"] = 0


@pytest.mark.parametrize("case", [
    "missing_history", "missing_best", "missing_early_stop_state", "missing_optimizer_updates",
    "empty_adam", "group_truncated", "unknown_id", "bad_model_value", "bad_scheduler",
    "bad_optimizer_eps", "wrong_epoch", "wrong_best", "negative_second_moment", "bad_cycle",
])
def test_invalid_state_rejected_atomically(tmp_path, case):
    scheduler = "cosine_warm" if case == "bad_cycle" else "cosine"
    source = _env(tmp_path, "source", scheduler=scheduler)
    source.fit(1)
    path = source.checkpoints_dir / "last.pth"
    payload = torch.load(path, weights_only=False, map_location="cpu")
    _mutate(payload, case)
    tampered = tmp_path / "tampered.pth"
    torch.save(payload, tampered)
    target = _env(tmp_path, "target", scheduler=scheduler)
    before = _live(target)
    files = {p: file_sha256(p) for p in target.run_dir.rglob("*") if p.is_file()}
    with pytest.raises(RuntimeError):
        target.load_checkpoint(tampered)
    _same(before, _live(target))
    assert all(file_sha256(p) == sha for p, sha in files.items())


def test_unexpected_late_load_error_rolls_back_all_state(tmp_path, monkeypatch):
    source = _env(tmp_path, "source")
    source.fit(1)
    target = _env(tmp_path, "target")
    target.fit(1)
    before = _live(target)
    original = ck._apply_training_state

    def fail_after_real_restore(trainer, payload):
        original(trainer, payload)
        raise RuntimeError("injected late failure")

    monkeypatch.setattr(ck, "_apply_training_state", fail_after_real_restore)
    with pytest.raises(RuntimeError, match="injected"):
        target.load_checkpoint(source.checkpoints_dir / "last.pth")
    _same(before, _live(target))


def test_zero_update_checkpoint_remains_valid(tmp_path):
    source = _env(tmp_path, "zero", scheduler="none")
    path = source.save_checkpoint(source.checkpoints_dir / "initial.pth")
    target = _env(tmp_path, "target", scheduler="none")
    target.load_checkpoint(path)
    target.fit(1)
    assert target._optimizer_updates == 4


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("amp", [False, True])
def test_actual_minicnn_cuda_resume_and_rng_rejection(tmp_path, amp):
    from tests.test_training_integrity import _make_config

    config = _make_config(scheduler="none", amp=amp, seed=71)
    config["models"]["mini_cnn"]["dropout"] = 0.3
    config["training"]["cudnn_deterministic"] = True
    generator = torch.Generator().manual_seed(17)
    data = TensorDataset(
        torch.rand(16, 1, 48, 48, generator=generator),
        torch.randint(0, 3, (16,), generator=generator),
    )

    def make(name):
        torch.manual_seed(71)
        torch.cuda.manual_seed_all(71)
        return Trainer(
            MiniCNN(num_classes=3, dropout=0.3), DataLoader(data, batch_size=4),
            DataLoader(data, batch_size=4), None, copy.deepcopy(config), "mini_cnn",
            device=torch.device("cuda"), run_dir=tmp_path / name,
        )

    source = make("source")
    source.fit(1)
    path = source.checkpoints_dir / "last.pth"
    payload = torch.load(path, weights_only=False, map_location="cpu")
    for bad in ("missing", None, [], [torch.zeros(3)], payload["rng"]["torch_cuda"] * 2):
        target = make("target")
        changed = copy.deepcopy(payload)
        if isinstance(bad, str) and bad == "missing":
            changed["rng"].pop("torch_cuda")
        else:
            changed["rng"]["torch_cuda"] = bad
        tampered = tmp_path / "rng_bad.pth"
        torch.save(changed, tampered)
        before = _live(target)
        with pytest.raises(RuntimeError):
            target.load_checkpoint(tampered)
        _same(before, _live(target))
    if amp:
        for key, value in (("scale", 0), ("scale", float("nan")), ("growth_factor", 1),
                           ("backoff_factor", 2), ("growth_interval", 0),
                           ("_growth_tracker", -1)):
            target = make("target")
            changed = copy.deepcopy(payload)
            changed["scaler_state_dict"][key] = value
            tampered = tmp_path / "scaler_bad.pth"
            torch.save(changed, tampered)
            with pytest.raises(RuntimeError):
                target.load_checkpoint(tampered)
    resumed = make("resume")
    resumed.load_checkpoint(path)
    resumed.fit(1)
    resumed_rng = ck._capture_rng_state()
    continuous = make("continuous")
    continuous.fit(2)
    _same(resumed.model.state_dict(), continuous.model.state_dict())
    _same(resumed.optimizer.state_dict(), continuous.optimizer.state_dict())
    if amp:
        _same(resumed.scaler.state_dict(), continuous.scaler.state_dict())
    _same(resumed_rng, ck._capture_rng_state())
    assert resumed.history == continuous.history


@pytest.mark.parametrize("bad", [1.5, -0.5, float("nan"), float("inf"), -1, 7])
def test_labels_not_truncated(tmp_path, bad):
    csv = tmp_path / "mini.csv"
    _write_mini_csv(csv)
    import pandas as pd
    frame = pd.read_csv(csv)
    frame["emotion"] = frame["emotion"].astype(float)
    frame.loc[0, "emotion"] = bad
    frame.to_csv(csv, index=False)
    with pytest.raises(ValueError):
        pc.build_cache(csv, tmp_path / "cache", quiet=True)
    assert not (tmp_path / "cache" / "meta.json").exists()


@pytest.mark.parametrize("tail", [" junk", " 1junk", " nan", " inf"])
def test_pixel_parser_consumes_entire_text(tail):
    with pytest.raises(ValueError):
        pc.parse_pixels_column([" ".join(["1"] * 2304) + tail])


def test_batch_fetch_matches_scalar_rng_and_reduces_stats(tmp_path, monkeypatch):
    csv = tmp_path / "mini.csv"
    _write_mini_csv(csv)
    cache = pc.load_or_build(csv, tmp_path / "cache", quiet=True)
    entry = cache.meta["splits"]["Training"]
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(), transforms.RandomRotation(15),
    ])
    dataset = FER2013Dataset(
        cache_ref=(cache.cache_dir, "Training", entry), transform=transform,
    )
    cache.split_arrays("Training")
    original = pc._stat_signature_matches
    calls = []

    def count(path, record):
        calls.append(str(path))
        return original(path, record)

    monkeypatch.setattr(pc, "_stat_signature_matches", count)
    indices = [1, 0, 1, 3, 5]
    torch.manual_seed(11)
    scalar = [dataset[i] for i in indices]
    calls.clear()
    torch.manual_seed(11)
    batched = dataset.__getitems__(indices)
    _same(scalar, batched)
    assert len(calls) == 6
    cache.close()


@pytest.mark.parametrize("kind", ["x", "y", "rows"])
def test_batch_detects_each_changed_file(tmp_path, kind):
    csv = tmp_path / "mini.csv"
    _write_mini_csv(csv)
    cache = pc.load_or_build(csv, tmp_path / "cache", quiet=True)
    entry = cache.meta["splits"]["Training"]
    dataset = FER2013Dataset(cache_ref=(cache.cache_dir, "Training", entry))
    dataset.__getitems__([0, 1])
    path = cache.cache_dir / entry[kind]["file"]
    import os
    stat = path.stat()
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
    with pytest.raises(pc.PixelCacheCorruptedError):
        dataset.__getitems__([0, 1])
    cache.close()


def _formal_fixture(tmp_path, monkeypatch, *, patience=0, epochs=2):
    csv = tmp_path / "mini.csv"
    _write_mini_csv(csv)
    monkeypatch.setattr(pc, "default_cache_dir", lambda: tmp_path / "cache")
    config = load_config("configs/baseline_config.yaml")
    config["data"]["dataset_path"] = str(csv)
    config["dataloader"].update(num_workers=0, pin_memory=False, persistent_workers=False)
    config["augmentation"]["enabled"] = False
    config["training"].update(amp=False, patience=patience, val_loss_patience=0)
    config["models"]["mini_cnn"].update(batch_size=4, num_epochs=epochs)
    spec = make_spec_from_config(config, "mini_cnn")

    def make(name, purpose="smoke", record=None, cfg=None):
        cfg = copy.deepcopy(cfg or config)
        train, val, _, _ = create_dataloaders(cfg, "mini_cnn", include_test=False)
        return Trainer(
            build_model_from_spec(spec), train, val, None, cfg, "mini_cnn",
            device=torch.device("cpu"), run_dir=tmp_path / name,
            run_purpose=purpose, frozen_protocol=record,
        )

    template = make("template")
    plan = {
        "model_name": "mini_cnn", "arm": "A", "seeds": [42, 43, 44],
        "model_spec": spec.to_dict(), "training_protocol": template.get_training_protocol(),
        "max_epochs": epochs, "allow_early_stop": True, "lr_floor": 1e-7,
    }
    frozen = tmp_path / "frozen.json"
    frozen.write_text(json.dumps({
        "schema_version": 1, "protocol_id": "test-plan", "frozen_at": "2026-10-07",
        "git_commit": "fixture", "code_sha256": code_fingerprint(), "plans": [plan],
    }), encoding="utf-8")
    return config, frozen, load_frozen_protocol(frozen), make


def test_formal_full_artifacts_and_split_sessions(tmp_path, monkeypatch):
    _, frozen, record, make = _formal_fixture(tmp_path, monkeypatch)
    t = make("formal", "formal", record)
    t.fit(1)
    assert t.run_meta["status"] == "session_completed"
    assert not check_formal_eligibility(t.run_dir, frozen_protocol_path=frozen)["formal_eligible"]
    resumed = make("formal", "formal", record)
    resumed.load_checkpoint(t.checkpoints_dir / "last.pth")
    resumed.fit(1)
    result = check_formal_eligibility(resumed.run_dir, frozen_protocol_path=frozen)
    assert result["formal_eligible"], result["reasons"]
    with pytest.raises(ValueError):
        resumed.fit(1)
    torch.save({"partial": False}, resumed.checkpoints_dir / "last.pth")
    result = check_formal_eligibility(resumed.run_dir, frozen_protocol_path=frozen)
    assert not result["formal_eligible"]


@pytest.mark.parametrize("field", ["lr", "batch", "seed", "data", "id", "over_budget"])
def test_formal_drift_rejected_before_training(tmp_path, monkeypatch, field):
    config, frozen, record, make = _formal_fixture(tmp_path, monkeypatch)
    cfg = copy.deepcopy(config)
    if field == "lr":
        cfg["models"]["mini_cnn"]["learning_rate"] = 0.01
    elif field == "batch":
        cfg["models"]["mini_cnn"]["batch_size"] = 8
    elif field == "seed":
        cfg["seed"] = 11
    elif field == "data":
        import pandas as pd
        csv = tmp_path / "mini.csv"
        frame = pd.read_csv(csv)
        frame.loc[0, "emotion"] = (frame.loc[0, "emotion"] + 1) % 7
        frame.to_csv(csv, index=False)
    elif field == "id":
        record["protocol_id"] = "wrong"
    if field == "over_budget":
        t = make("invalid", "formal", record)
        before = _live(t)
        with pytest.raises(ValueError):
            t.fit(3)
        _same(before, _live(t))
    else:
        with pytest.raises(ValueError):
            make("invalid", "formal", record, cfg)
        assert not (tmp_path / "invalid").exists()


def test_formal_early_stop_is_eligible(tmp_path, monkeypatch):
    _, frozen, record, make = _formal_fixture(tmp_path, monkeypatch, patience=1, epochs=4)
    t = make("early", "formal", record)
    metrics = iter([(1.0, 0.7, 1.0), (1.0, 0.6, 1.0)])
    t.evaluate = lambda: next(metrics)
    t.fit(4)
    assert t.run_meta["final_epoch"] == 2
    assert t.run_meta["completion_reason"] == "val_acc"
    result = check_formal_eligibility(t.run_dir, frozen_protocol_path=frozen)
    assert result["formal_eligible"], result["reasons"]


@pytest.mark.parametrize("part", ["run_id", "best", "history", "config", "faked_finish"])
def test_formal_artifact_binding_and_completion_checked(tmp_path, monkeypatch, part):
    _, frozen, record, make = _formal_fixture(tmp_path, monkeypatch)
    t = make("formal", "formal", record)
    t.fit(1 if part == "faked_finish" else 2)
    if part == "run_id":
        path = t.checkpoints_dir / "last.pth"
        payload = torch.load(path, weights_only=False, map_location="cpu")
        payload["run_id"] = "foreign"
        torch.save(payload, path)
    elif part == "best":
        torch.save({"partial": False}, t.checkpoints_dir / "best.pth")
    elif part == "history":
        (t.run_dir / "history.json").write_text("{}", encoding="utf-8")
    elif part == "config":
        (t.run_dir / "config_effective.yaml").write_text("seed: 11", encoding="utf-8")
    else:
        t._persist_run_meta(
            status="finished", experiment_completed=True, completion_reason="budget",
        )
    result = check_formal_eligibility(t.run_dir, frozen_protocol_path=frozen)
    assert not result["formal_eligible"], result


def test_prefetched_batch_guard_rejects_before_optimizer(tmp_path, monkeypatch):
    _, _, _, make = _formal_fixture(tmp_path, monkeypatch)
    t = make("smoke")
    entry = t.train_loader.dataset._cache_ref[2]
    path = tmp_path / "cache" / entry["rows"]["file"]
    original_forward = t.model.forward

    def corrupt_during_forward(images):
        import os
        stat = path.stat()
        os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
        return original_forward(images)

    monkeypatch.setattr(t.model, "forward", corrupt_during_forward)
    with pytest.raises(pc.PixelCacheCorruptedError):
        t.fit(1)
    assert t._optimizer_updates == 0
    assert t._optimizer_attempts == 0


@pytest.mark.parametrize("source", ["csv", "meta"])
def test_dataset_rejects_source_generation_change(tmp_path, monkeypatch, source):
    _, _, _, make = _formal_fixture(tmp_path, monkeypatch)
    t = make("smoke")
    t.train_loader.dataset.__getitems__([0])
    path = tmp_path / ("mini.csv" if source == "csv" else "cache/meta.json")
    import os
    stat = path.stat()
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
    with pytest.raises(pc.PixelCacheCorruptedError):
        t.train_loader.dataset.__getitems__([0])


def test_invalid_build_preserves_existing_meta(tmp_path):
    import pandas as pd
    csv = tmp_path / "mini.csv"
    _write_mini_csv(csv)
    cache = pc.load_or_build(csv, tmp_path / "cache", quiet=True)
    meta = file_sha256(cache.cache_dir / "meta.json")
    frame = pd.read_csv(csv)
    frame["emotion"] = frame["emotion"].astype(float)
    frame.loc[0, "emotion"] = 1.5
    frame.to_csv(csv, index=False)
    with pytest.raises(ValueError):
        pc.build_cache(csv, cache.cache_dir, quiet=True)
    assert file_sha256(cache.cache_dir / "meta.json") == meta
    cache.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_amp_all_skipped_epoch_valid_and_zero_updates_resume(tmp_path):
    t = _env(tmp_path, "all_skip", amp=True, scheduler="none")
    # Simulate a valid AMP skip by forcing overflow detection in every scaled gradient.
    for param in t.model.parameters():
        param.register_hook(lambda gradient: gradient * float("inf"))
    t.fit(1)
    assert t._optimizer_updates == 0
    target = _env(tmp_path, "target", amp=True, scheduler="none")
    target.load_checkpoint(t.checkpoints_dir / "last.pth")
    assert target._optimizer_updates == 0
    assert target._optimizer_attempts == 4


def test_failed_resume_event_commit_restores_metadata_bytes(tmp_path, monkeypatch):
    source = _env(tmp_path, "source")
    source.fit(1)
    target = _env(tmp_path, "target")
    before = _live(target)
    meta_path = target.run_dir / "run_meta.json"
    meta_bytes = meta_path.read_bytes()
    original = target._record_resume_event

    def fail_after_commit(*args, **kwargs):
        original(*args, **kwargs)
        raise RuntimeError("injected commit failure")

    monkeypatch.setattr(target, "_record_resume_event", fail_after_commit)
    with pytest.raises(RuntimeError, match="commit failure"):
        target.load_checkpoint(source.checkpoints_dir / "last.pth")
    _same(before, _live(target))
    assert meta_path.read_bytes() == meta_bytes


def test_frozen_code_digest_is_verified(tmp_path, monkeypatch):
    _, frozen, _, _ = _formal_fixture(tmp_path, monkeypatch)
    body = json.loads(frozen.read_text(encoding="utf-8"))
    body["code_sha256"] = "0" * 64
    frozen.write_text(json.dumps(body), encoding="utf-8")
    with pytest.raises(ValueError, match="代码指纹"):
        load_frozen_protocol(frozen)




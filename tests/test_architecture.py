"""Architecture identity, old-weight compatibility, recovery and paired selection regressions."""

import copy
import subprocess
import types
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from models.micro_resnet import MicroResNet
from training.trainer import Trainer, load_config, set_seed
from utils.architecture_comparison import (
    ARCHITECTURES,
    collect_architecture_runs,
    select_architecture,
    validate_architecture_manifest,
)
from utils.config_validation import ConfigValidationError, validate_config
from utils.constants import CLASS_NAMES
from utils.model_spec import ModelSpec, build_model_from_spec, make_spec_from_config

ROOT = Path(__file__).resolve().parents[1]


def config(arm):
    return load_config(ROOT / f"configs/architecture_{arm.lower()}_config.yaml")


def manifest():
    plans = []
    for arm in ARCHITECTURES:
        cfg = config(arm)
        plans.append(
            {
                "model_name": "micro_resnet",
                "arm": arm,
                "seeds": [42, 43, 44],
                "max_epochs": 90,
                "allow_early_stop": False,
                "model_spec": make_spec_from_config(cfg, "micro_resnet").to_dict(),
                "training_protocol": {
                    "config": cfg,
                    "runtime": {"amp": True},
                    "data_fingerprint": {"csv_sha256": "same"},
                },
            }
        )
    return {"manifest": {"plans": plans}}


@pytest.mark.parametrize(
    "arm,params,spatial",
    [
        ("S0", 753991, 12),
        ("S1", 753991, 24),
        ("S2", 1493575, 12),
        ("S3", 764663, 12),
    ],
)
def test_structure_shapes_parameters_and_roundtrip(arm, params, spatial):
    cfg = config(arm)
    validate_config(cfg, model_name="micro_resnet")
    spec = make_spec_from_config(cfg, "micro_resnet")
    assert spec.spec_version == 2 and ModelSpec.from_dict(spec.to_dict()) == spec
    model = build_model_from_spec(spec).eval()
    assert sum(p.numel() for p in model.parameters()) == params
    shapes = []
    h1 = model.stage1.register_forward_pre_hook(lambda m, xs: shapes.append(xs[0].shape[-1]))
    h2 = model.stage2.register_forward_pre_hook(lambda m, xs: shapes.append(xs[0].shape[-1]))
    with torch.no_grad():
        for batch in (1, 128):
            assert model(torch.rand(batch, 1, 48, 48)).shape == (batch, 7)
    h1.remove()
    h2.remove()
    assert shapes == [spatial, 6, spatial, 6]


def test_s0_and_v1_match_archived_original_exactly():
    source = subprocess.run(
        ["git", "show", "71520bd:models/micro_resnet.py"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    ).stdout.decode("utf-8")
    original = types.ModuleType("archived_micro_resnet")
    exec(compile(source, "archived_micro_resnet", "exec"), original.__dict__)
    torch.manual_seed(11)
    old = original.MicroResNet(activation="gelu").eval()
    torch.manual_seed(11)
    s0 = build_model_from_spec(make_spec_from_config(config("S0"), "micro_resnet")).eval()
    assert all(torch.equal(v, s0.state_dict()[k]) for k, v in old.state_dict().items())
    v1 = make_spec_from_config(load_config(ROOT / "configs/ce_fixed_config.yaml"), "micro_resnet")
    assert v1.spec_version == 1 and "blocks" not in v1.to_dict()
    restored = build_model_from_spec(ModelSpec.from_dict(v1.to_dict())).eval()
    restored.load_state_dict(old.state_dict(), strict=True)
    x = torch.rand(3, 1, 48, 48)
    with torch.no_grad():
        assert torch.equal(old(x), s0(x))
        assert torch.equal(old(x), restored(x))


@pytest.mark.parametrize(
    "key,value",
    [
        ("blocks", [True, 2]),
        ("blocks", [0, 2]),
        ("blocks", [2]),
        ("channels", [64.0, 128]),
        ("channels", [-1, 128]),
        ("pool_order", "guess"),
    ],
)
def test_invalid_structure_rejected_everywhere(key, value):
    cfg = config("S0")
    cfg["models"]["micro_resnet"][key] = value
    with pytest.raises(ConfigValidationError):
        validate_config(cfg, model_name="micro_resnet")
    with pytest.raises(ValueError):
        make_spec_from_config(cfg, "micro_resnet")
    kwargs = dict(blocks=[2, 2], channels=[64, 128], pool_order="before_stage1")
    kwargs[key] = value
    with pytest.raises(ValueError):
        MicroResNet(**kwargs)


def test_missing_structure_and_v1_override_rejected():
    cfg = config("S0")
    del cfg["models"]["micro_resnet"]["blocks"]
    with pytest.raises(ConfigValidationError):
        validate_config(cfg, model_name="micro_resnet")
    with pytest.raises(ValueError):
        make_spec_from_config(cfg, "micro_resnet")
    spec = make_spec_from_config(config("S0"), "micro_resnet").to_dict()
    spec["spec_version"] = 1
    with pytest.raises(ValueError, match="v1"):
        ModelSpec.from_dict(spec)


@pytest.mark.parametrize("arm", list(ARCHITECTURES))
def test_real_architecture_resume_matches_continuous_and_rejects_pool_change(tmp_path, arm):
    cfg = config(arm)
    cfg["training"].update(amp=False, cudnn_deterministic=True)
    cfg["models"]["micro_resnet"].update(batch_size=4, num_epochs=2)
    x = torch.rand(8, 1, 48, 48, generator=torch.Generator().manual_seed(8))
    dataset = TensorDataset(x, torch.arange(8) % 7)

    def make(directory, settings=cfg):
        set_seed(42, deterministic=True)
        spec = make_spec_from_config(settings, "micro_resnet")
        loader = DataLoader(dataset, batch_size=4, generator=torch.Generator().manual_seed(101))
        return Trainer(
            build_model_from_spec(spec),
            loader,
            loader,
            None,
            settings,
            "micro_resnet",
            device=torch.device("cpu"),
            run_dir=directory,
        )

    continuous = make(tmp_path / "continuous")
    continuous.fit(2)
    first = make(tmp_path / "split")
    first.fit(1)
    restored = make(first.run_dir)
    restored.load_checkpoint(first.checkpoints_dir / "last.pth")
    restored.fit(1)
    assert continuous.history == restored.history
    assert all(
        torch.equal(v, restored.model.state_dict()[k])
        for k, v in continuous.model.state_dict().items()
    )
    wrong = copy.deepcopy(cfg)
    wrong["models"]["micro_resnet"]["pool_order"] = (
        "before_stage1" if arm == "S1" else "after_stage1"
    )
    rejected = make(first.run_dir, wrong)
    before = {k: v.clone() for k, v in rejected.model.state_dict().items()}
    with pytest.raises(RuntimeError, match="协议|配置|不匹配"):
        rejected.load_checkpoint(first.checkpoints_dir / "last.pth")
    assert all(torch.equal(v, rejected.model.state_dict()[k]) for k, v in before.items())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("arm", list(ARCHITECTURES))
def test_cuda_batch128_amp_has_finite_effective_updates(arm):
    model = build_model_from_spec(make_spec_from_config(config(arm), "micro_resnet")).cuda()
    assert model(torch.rand(1, 1, 48, 48, device="cuda")).shape == (1, 7)
    opt = torch.optim.Adam(model.parameters(), lr=3e-4)
    scaler = torch.amp.GradScaler("cuda")
    before = model.classifier[-1].weight.detach().clone()
    updates = 0
    for _ in range(8):
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda"):
            out = model(torch.rand(128, 1, 48, 48, device="cuda"))
            loss = torch.nn.functional.cross_entropy(out, torch.arange(128, device="cuda") % 7)
        assert out.shape == (128, 7) and torch.isfinite(loss)
        scale = scaler.get_scale()
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        updates += scaler.get_scale() >= scale
    assert updates > 0 and not torch.equal(before, model.classifier[-1].weight)
    assert all(torch.isfinite(p).all() for p in model.parameters())


def test_manifest_checks_single_factor_and_full_plan():
    record = manifest()
    validate_architecture_manifest(record)
    with pytest.raises(ValueError, match="不完整"):
        collect_architecture_runs(record, [])
    record["manifest"]["plans"][1]["training_protocol"]["config"]["training"]["weight_decay"] = 0.01
    with pytest.raises(ValueError, match="共享|配方"):
        validate_architecture_manifest(record)


def public_rows(gain=0.011):
    rows = []
    for arm in ARCHITECTURES:
        for seed in (42, 43, 44):
            value = 0.6 + (gain if arm == "S1" else 0)
            metrics = dict(
                accuracy=0.67,
                macro_f1=value,
                balanced_accuracy=value,
                ece=0.05,
                nll=1.0,
                per_class={n: dict(support=10, recall=value) for n in CLASS_NAMES},
            )
            rows.append(dict(arm=arm, seed=seed, split="PublicTest", metrics=metrics))
    return rows


def costs():
    return {
        a: dict(params_total=753991, cpu_p95_median_ms=1.0, batch128_no_oom=True)
        for a in ARCHITECTURES
    }


def test_quality_and_deployment_gates_are_separate_and_public_only():
    rows = public_rows()
    assert select_architecture(rows, costs())["candidate"] == "S1"
    expensive = costs()
    expensive["S1"]["cpu_p95_median_ms"] = 1.6
    result = select_architecture(rows, expensive)
    assert result["candidate"] == "S0"
    assert result["gates"]["S1"]["quality_pass"] and not result["gates"]["S1"]["deployment_pass"]
    assert select_architecture(public_rows(0.009), costs())["candidate"] == "S0"
    rows[3]["metrics"]["macro_f1"] = 0.599
    assert not select_architecture(rows, costs())["gates"]["S1"]["quality_pass"]
    rows[0]["split"] = "PrivateTest"
    with pytest.raises(ValueError):
        select_architecture(rows, costs())


def test_all_arms_changing_recipe_and_duplicate_summary_rejected():
    record = manifest()
    for plan in record["manifest"]["plans"]:
        plan["training_protocol"]["config"]["models"]["micro_resnet"]["activation"] = "relu"
    with pytest.raises(ValueError, match="配方"):
        validate_architecture_manifest(record)
    rows = public_rows()
    rows[-1] = copy.deepcopy(rows[0])
    with pytest.raises(ValueError):
        select_architecture(rows, costs())


def test_accuracy_regression_blocks_quality_even_if_macro_f1_improves():
    rows = public_rows()
    for row in rows:
        if row["arm"] == "S1":
            row["metrics"]["accuracy"] = 0.664
    assert not select_architecture(rows, costs())["gates"]["S1"]["quality_pass"]


def test_finalize_fixes_public_selection_before_private_and_keeps_it(tmp_path, monkeypatch):
    import tools.summarize_architecture as report

    rows = public_rows()
    run_rows = []
    for row in rows:
        run_rows.append(
            {
                "arm": row["arm"],
                "seed": row["seed"],
                "run_dir": str(tmp_path),
                "training_duration_seconds": 10.0,
                "optimizer_attempts": 100,
                "optimizer_updates": 99,
                "resume_events": 0,
            }
        )
    record = manifest()
    record.update(protocol_id="test", file_sha256="bound")
    record["manifest"].update(code_sha256="source", git_commit="commit")
    monkeypatch.setattr(report, "collect_architecture_runs", lambda *a: run_rows)
    monkeypatch.setattr(
        report,
        "benchmark_architectures",
        lambda *a: {
            "aggregate": costs(),
            "raw": {},
            "method": {},
            "preprocessing": {"median_ms": 1.0, "p95_ms": 1.0, "raw_ms": [1.0]},
        },
    )
    monkeypatch.setattr(report, "write_figures", lambda *a: None)
    visited = []
    fixed_sha = []

    def fake_evaluate(row, split, binding, output):
        if split == "PrivateTest":
            selection = output / "candidate_selection.json"
            assert selection.exists()
            selected = report.file_sha256(selection)
            if not fixed_sha:
                fixed_sha.append(selected)
            assert selected == fixed_sha[0]
        visited.append(split)
        metrics = next(
            r["metrics"] for r in rows if (r["arm"], r["seed"]) == (row["arm"], row["seed"])
        )
        return {**row, "split": split, "metrics": metrics, "predictions_sha256": "pred"}

    monkeypatch.setattr(report, "evaluate_once", fake_evaluate)
    aggregate = report.finalize(record, [], tmp_path)
    assert visited == ["Training"] * 12 + ["PublicTest"] * 12 + ["PrivateTest"] * 12
    assert aggregate["selection"]["candidate"] == "S1"
    assert "raw_ms" not in aggregate["preprocessing"]
    assert "runs" not in aggregate and (tmp_path / "results.json").exists()
    visited.clear()
    report.finalize(record, [], tmp_path)
    assert report.file_sha256(tmp_path / "candidate_selection.json") == fixed_sha[0]

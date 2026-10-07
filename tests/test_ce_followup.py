"""Fixed budgets and byte-preserving historical protocol audits."""
import copy
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
import torch
import yaml
from torch.utils.data import WeightedRandomSampler

import tests.test_fourth_fixes as fixtures
import tools.freeze_comparison as freeze
import tools.summarize_ablation as ablation
import utils.comparison_check as comparison
import utils.formal_protocol as formal
from tests.test_fourth_fixes import _formal_fixture
from tools.summarize_ablation import collect_runs, summarize_rows
from training.trainer import load_config
from utils.losses import CBFocalLoss, FocalLoss

ARM_CONFIGS = (
    ("ce_fixed_config.yaml", "cross_entropy", False, torch.nn.CrossEntropyLoss),
    ("focal_config.yaml", "focal", False, FocalLoss),
    ("focal_weighted_config.yaml", "focal", True, FocalLoss),
    ("cb_focal_weighted_config.yaml", "cb_focal", True, CBFocalLoss),
)


def test_followup_configs_change_only_the_declared_factors():
    baseline = load_config("configs/baseline_config.yaml")
    for name, loss, weighted, _ in ARM_CONFIGS:
        cfg = load_config(f"configs/{name}")
        assert cfg["training"]["patience"] == cfg["training"]["val_loss_patience"] == 0
        assert all(m["num_epochs"] == 90 for m in cfg["models"].values())
        assert cfg["training"]["loss_type"] == loss
        assert cfg["dataloader"]["class_balanced_sampling"] is weighted
        restored = copy.deepcopy(cfg)
        restored["training"].update(
            patience=baseline["training"]["patience"],
            val_loss_patience=baseline["training"]["val_loss_patience"],
            loss_type=baseline["training"]["loss_type"],
        )
        restored["dataloader"]["class_balanced_sampling"] = False
        assert restored == baseline


@pytest.mark.parametrize("name,loss,weighted,criterion", ARM_CONFIGS)
def test_actual_arm_components_and_finite_training(tmp_path, monkeypatch, name,
                                                   loss, weighted, criterion):
    cfg, _, _, make = _formal_fixture(tmp_path, monkeypatch)
    arm = load_config(f"configs/{name}")
    cfg["training"].update(loss_type=arm["training"]["loss_type"])
    cfg["dataloader"]["class_balanced_sampling"] = weighted
    trainer = make("arm", cfg=cfg)
    assert isinstance(trainer.criterion, criterion)
    assert isinstance(trainer.train_loader.sampler, WeightedRandomSampler) is weighted
    assert not trainer.class_specific_enabled
    trainer.fit(1)
    assert trainer._optimizer_updates > 0
    assert all(torch.isfinite(p).all() for p in trainer.model.parameters())
    assert trainer.get_training_protocol()["config"]["training"]["loss_type"] == loss


def test_fixed_budget_survives_low_lr_and_validation_regression(tmp_path, monkeypatch):
    _, path, _, make = _formal_fixture(tmp_path, monkeypatch, epochs=4)
    body = json.loads(path.read_text(encoding="utf-8"))
    body["plans"][0]["allow_early_stop"] = False
    path.write_text(json.dumps(body), encoding="utf-8")
    record = formal.load_frozen_protocol(path)
    trainer = make("fixed", "formal", record)
    # Exercise the previous unconditional LR fuse after a scheduler drop/restart.
    train = trainer.train_one_epoch

    def low_lr_epoch():
        trainer.optimizer.param_groups[0]["lr"] = 1e-8
        return train()

    monkeypatch.setattr(trainer, "train_one_epoch", low_lr_epoch)
    scores = iter([0.8, 0.6, 0.5, 0.4])
    monkeypatch.setattr(trainer, "evaluate", lambda: (2.0, next(scores), 1.0))
    trainer.fit(4)
    assert trainer.run_meta["final_epoch"] == 4
    assert trainer.run_meta["completion_reason"] == "budget"
    result = comparison.check_formal_eligibility(trainer.run_dir, frozen_protocol_path=path)
    assert result["formal_eligible"], result["reasons"]


def test_fixed_plan_rejects_enabled_early_stopping(tmp_path, monkeypatch):
    _, path, _, _ = _formal_fixture(tmp_path, monkeypatch, patience=1)
    body = json.loads(path.read_text(encoding="utf-8"))
    body["plans"][0]["allow_early_stop"] = False
    path.write_text(json.dumps(body), encoding="utf-8")
    with pytest.raises(ValueError, match="固定预算"):
        formal.load_frozen_protocol(path)


@pytest.mark.parametrize("patience", [0, 1])
def test_freeze_derives_completion_policy_from_actual_configuration(
    tmp_path, monkeypatch, patience,
):
    cfg, _, _, _ = _formal_fixture(tmp_path, monkeypatch, patience=patience)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    out = tmp_path / "output.json"
    monkeypatch.setattr(freeze, "collect_git_info", lambda: {"commit": "fixture", "dirty": False})
    monkeypatch.setattr(sys, "argv", [
        "freeze", "--protocol-id", "fixed-test", "--output", str(out),
        "--plan", "mini_cnn", "A", str(config_path),
    ])
    freeze.main()
    result = formal.load_frozen_protocol(out)
    assert result["manifest"]["plans"][0]["allow_early_stop"] is (patience > 0)


def test_finished_run_survives_protocol_relocation(tmp_path, monkeypatch):
    _, path, record, make = _formal_fixture(tmp_path, monkeypatch)
    trainer = make("formal", "formal", record)
    trainer.fit(2)
    archived = tmp_path / "configs" / "protocols" / "test-plan.json"
    archived.parent.mkdir(parents=True)
    archived.write_bytes(path.read_bytes())
    path.unlink()
    monkeypatch.setattr(comparison, "PROJECT_ROOT", tmp_path)
    result = comparison.check_formal_eligibility(trainer.run_dir)
    assert result["formal_eligible"], result["reasons"]
    archived.write_bytes(archived.read_bytes() + b"\n")
    result = comparison.check_formal_eligibility(trainer.run_dir, frozen_protocol_path=archived)
    assert not result["formal_eligible"]
    assert any("file_sha256" in reason for reason in result["reasons"])


def _git(root, *args):
    return subprocess.run(["git", "-C", str(root), *args], check=True,
                          capture_output=True, text=True).stdout.strip()


@pytest.mark.parametrize("invalid_path", [None, 0, []])
def test_malformed_bound_path_is_ineligible_without_crashing(tmp_path, monkeypatch, invalid_path):
    _, path, record, make = _formal_fixture(tmp_path, monkeypatch)
    trainer = make("formal", "formal", record)
    trainer.fit(2)
    archived = tmp_path / "configs" / "protocols" / "test-plan.json"
    archived.parent.mkdir(parents=True)
    archived.write_bytes(path.read_bytes())
    monkeypatch.setattr(comparison, "PROJECT_ROOT", tmp_path)
    meta_path = trainer.run_dir / "run_meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["frozen_protocol"]["path"] = invalid_path
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    result = comparison.check_formal_eligibility(trainer.run_dir)
    assert not result["formal_eligible"]
    assert result["reasons"]


def test_historical_audit_uses_git_snapshot_but_training_requires_current_code(
    tmp_path, monkeypatch,
):
    (tmp_path / "fixture").mkdir()
    _, path, _, _ = _formal_fixture(tmp_path / "fixture", monkeypatch)
    repo = tmp_path / "git-source"
    repo.mkdir()
    _git(repo, "init")
    (repo / "models").mkdir()
    (repo / "models" / "model.py").write_text("# original\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "-c", "user.name=Test", "-c", "user.email=test@example.invalid",
         "commit", "-m", "original source")
    commit = _git(repo, "rev-parse", "HEAD")
    fingerprint = formal.archived_code_fingerprint(commit, root=repo)
    actual_archive = formal.archived_code_fingerprint
    monkeypatch.setattr(formal, "archived_code_fingerprint",
                        lambda ref: actual_archive(ref, root=repo))
    body = json.loads(path.read_text(encoding="utf-8"))
    body.update(git_commit=commit, code_sha256=fingerprint)
    path.write_text(json.dumps(body), encoding="utf-8")
    # Working tree may change; the exact committed code is independently verified.
    (repo / "models" / "model.py").write_text("# changed\n", encoding="utf-8")
    record = formal.load_frozen_protocol(path, historical=True)
    assert record["manifest"]["code_sha256"] == fingerprint
    with pytest.raises(ValueError, match="代码指纹"):
        formal.load_frozen_protocol(path)
    body["code_sha256"] = "0" * 64
    path.write_text(json.dumps(body), encoding="utf-8")
    with pytest.raises(ValueError, match="代码指纹"):
        formal.load_frozen_protocol(path, historical=True)
    for invalid in ("fixture", "0" * 40):
        with pytest.raises(ValueError):
            actual_archive(invalid, root=repo)


def test_archived_manifest_git_checkout_preserves_exact_bytes(tmp_path):
    root = Path(__file__).resolve().parents[1]
    repo = tmp_path / "storage"
    repo.mkdir()
    _git(repo, "init")
    (repo / ".gitattributes").write_bytes((root / ".gitattributes").read_bytes())
    path = repo / "configs" / "protocols" / "comparison-ce-v1.json"
    path.parent.mkdir(parents=True)
    source = (root / "configs" / "protocols" / "comparison-ce-v1.json").read_bytes()
    path.write_bytes(source)
    _git(repo, "add", ".")
    _git(repo, "-c", "user.name=Test", "-c", "user.email=test@example.invalid",
         "commit", "-m", "archive exact bytes")
    path.unlink()
    _git(repo, "checkout", "--", "configs/protocols/comparison-ce-v1.json")
    assert path.read_bytes() == source


def _rows(arm, *, macro=0.5, recall=0.2):
    names = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
    return [
        {"model_name": "mini_cnn", "arm": arm, "seed": seed, "split": "PublicTest",
         "metrics": {"accuracy": 0.6, "macro_f1": macro, "balanced_accuracy": 0.5,
                     "per_class": {n: {"support": 56, "recall": recall} for n in names}}}
        for seed in (42, 43, 44)
    ]


def test_public_acceptance_requires_recall_gain_without_macro_regression():
    rows = _rows("A") + _rows("B", recall=0.4, macro=0.49)
    rows += _rows("C", recall=0.4, macro=0.5) + _rows("D", recall=0.2, macro=0.7)
    result = summarize_rows(rows)["mini_cnn"]
    assert not result["A"]["improved_vs_ce"]
    assert not result["B"]["improved_vs_ce"]
    assert result["C"]["improved_vs_ce"]
    assert not result["D"]["improved_vs_ce"]
    assert result["C"]["per_class"]["Disgust"]["support"] == 56
    assert len(result["C"]["per_class"]) == 7


@pytest.mark.parametrize("bad", ["private", "missing_seed", "duplicate", "support", "classes"])
def test_invalid_ablation_summary_rejected(bad):
    rows = _rows("A")
    if bad == "private":
        rows[0]["split"] = "PrivateTest"
    elif bad == "missing_seed":
        rows.pop()
    elif bad == "duplicate":
        rows[0]["seed"] = rows[1]["seed"]
    elif bad == "support":
        rows[0]["metrics"]["per_class"]["Disgust"]["support"] = 55
    else:
        rows[0]["metrics"]["per_class"].pop("Disgust")
    with pytest.raises(ValueError):
        summarize_rows(rows)


def test_ablation_collect_rejects_missing_and_duplicate_frozen_runs(tmp_path, monkeypatch):
    _, path, record, make = _formal_fixture(tmp_path, monkeypatch)
    trainer = make("formal", "formal", record)
    trainer.fit(2)
    with pytest.raises(ValueError, match="尚不完整"):
        collect_runs(record, [trainer.run_dir])
    with pytest.raises(ValueError, match="多个 run"):
        collect_runs(record, [trainer.run_dir, trainer.run_dir])
    trainers = [trainer]
    for seed in (43, 44):
        cfg = copy.deepcopy(trainer.config)
        cfg["seed"] = seed
        t = make(f"formal{seed}", "formal", record, cfg=cfg)
        t.fit(2)
        trainers.append(t)
    rows = collect_runs(record, [t.run_dir for t in trainers])
    assert [r["seed"] for r in rows] == [42, 43, 44]


@pytest.mark.parametrize("bad", ["loss", "sampler", "augmentation", "budget"])
def test_ablation_plan_labels_and_single_factor_binding_checked(tmp_path, monkeypatch, bad):
    _, _, record, _ = _formal_fixture(tmp_path, monkeypatch)
    plan = copy.deepcopy(record["manifest"]["plans"][0])
    plan["arm"] = "B"
    plan["training_protocol"]["config"]["training"]["loss_type"] = "focal"
    if bad == "loss":
        plan["training_protocol"]["config"]["training"]["loss_type"] = "cross_entropy"
    elif bad == "sampler":
        plan["training_protocol"]["config"]["dataloader"]["class_balanced_sampling"] = True
    elif bad == "augmentation":
        plan["training_protocol"]["config"]["augmentation"]["class_specific"]["enabled"] = True
    else:
        plan["max_epochs"] += 1
    record["manifest"]["plans"].append(plan)
    with pytest.raises(ValueError, match="臂定义|共享配置"):
        collect_runs(record, [])


def test_ablation_cli_saves_only_public_predictions_and_rejects_wrong_data(
    tmp_path, monkeypatch,
):
    original_writer = fixtures._write_mini_csv

    def complete_public_csv(path):
        original_writer(path)
        frame = pd.read_csv(path)
        # Original cache fixture deliberately has only six validation labels.
        extra = frame.loc[frame["Usage"] == "PublicTest"].iloc[[0]].copy()
        extra["emotion"] = 6
        pd.concat([frame, extra], ignore_index=True).to_csv(path, index=False)

    monkeypatch.setattr(fixtures, "_write_mini_csv", complete_public_csv)
    cfg, path, record, make = _formal_fixture(tmp_path, monkeypatch)
    trainers = []
    for seed in (42, 43, 44):
        config = copy.deepcopy(cfg)
        config["seed"] = seed
        trainer = make(f"formal{seed}", "formal", record, cfg=config)
        trainer.fit(2)
        trainers.append(trainer)
    out = tmp_path / "public-results"
    args = ["summarize_ablation", "--protocol", str(path), "--output", str(out),
            "--csv", cfg["data"]["dataset_path"]]
    for trainer in trainers:
        args.extend(["--run", str(trainer.run_dir)])
    monkeypatch.setattr(sys, "argv", args)
    ablation.main()
    result = json.loads((out / "results.json").read_text(encoding="utf-8"))
    assert result["selection_split"] == "PublicTest"
    assert len(result["runs"]) == 3
    assert all(row["split"] == "PublicTest" for row in result["runs"])
    assert len(list(out.rglob("predictions.csv"))) == 3
    assert len(result["summary"]["mini_cnn"]["A"]["per_class"]) == 7
    assert not result["summary"]["mini_cnn"]["A"]["improved_vs_ce"]
    wrong = tmp_path / "wrong.csv"
    wrong.write_text("wrong data", encoding="utf-8")
    args[args.index("--csv") + 1] = str(wrong)
    failed_out = tmp_path / "wrong-output"
    args[args.index("--output") + 1] = str(failed_out)
    with pytest.raises(ValueError, match="评估数据"):
        ablation.main()
    assert not failed_out.exists()

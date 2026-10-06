"""
比较来源一致性校验测试 — S04

覆盖审计负例：旧权重 + 新 run history 必须失败；及 run/协议/模型组一致性。
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest

from tests.test_training_pipeline import _make_config, _make_trainer
from utils.comparison_check import (
    ComparisonSourceError,
    check_comparison_source,
    validate_comparison_set,
)

LEGACY_CKPTS = {
    name: PROJECT_ROOT / "inference" / "saved_models" / f"{name}_best.pth"
    for name in ("mini_cnn", "vgg_lite", "micro_resnet")
}
LEGACY_HISTORIES = {
    name: PROJECT_ROOT / "training" / "logs" / name / "history.json"
    for name in ("mini_cnn", "vgg_lite", "micro_resnet")
}


def _make_run(tmp_path, run_name, *, config=None, model="mini_cnn"):
    """在 tmp_path/runs/<model>/<run_name>/ 下造一个真 run（meta+history+last.pth）。"""
    t = _make_trainer(
        config or _make_config(),
        run_dir=tmp_path / "runs" / model / run_name,
        model_name=model,
    )
    t.fit(1)
    return t


def test_legacy_set_marked_unverified():
    """全部 legacy 来源：允许运行但整组标记「来源未验证」。"""
    sources = {
        name: {"checkpoint": LEGACY_CKPTS[name], "history": LEGACY_HISTORIES[name]}
        for name in ("mini_cnn", "vgg_lite", "micro_resnet")
    }
    reports = validate_comparison_set(sources)
    assert all(r["verdict"] == "legacy-unverified" for r in reports)


def test_legacy_ckpt_with_run_history_rejected(tmp_path):
    """审计负例：旧权重 + 新 run（smoke）history 必须在图表前失败。"""
    t = _make_run(tmp_path, "smoke_a")
    with pytest.raises(ComparisonSourceError, match="混用"):
        check_comparison_source(
            "mini_cnn", LEGACY_CKPTS["mini_cnn"], t.run_dir / "history.json",
            runs_dir=tmp_path / "runs",
        )


def test_formal_ckpt_with_legacy_history_rejected(tmp_path):
    """新权重 + 旧日志：来源不一致，失败。"""
    t = _make_run(tmp_path, "smoke_b")
    with pytest.raises(ComparisonSourceError, match="来源不一致"):
        check_comparison_source(
            "mini_cnn", t.checkpoints_dir / "last.pth", LEGACY_HISTORIES["mini_cnn"],
            runs_dir=tmp_path / "runs",
        )


def test_run_mismatch_rejected(tmp_path):
    """history 与 checkpoint 的 run 不匹配：失败。"""
    t1 = _make_run(tmp_path, "run_c")
    t2 = _make_run(tmp_path, "run_d")
    with pytest.raises(ComparisonSourceError, match="不匹配"):
        check_comparison_source(
            "mini_cnn", t1.checkpoints_dir / "last.pth", t2.run_dir / "history.json",
            runs_dir=tmp_path / "runs",
        )


def test_same_run_passes_formal(tmp_path):
    """同一 run 的 checkpoint + history：formal 通过。"""
    t = _make_run(tmp_path, "run_e")
    report = check_comparison_source(
        "mini_cnn", t.checkpoints_dir / "last.pth", t.run_dir / "history.json",
        runs_dir=tmp_path / "runs",
    )
    assert report["verdict"] == "formal"
    assert report["run_id"] == t.run_id
    assert report["protocol_digest"]


def test_mixed_formal_and_legacy_set_rejected(tmp_path):
    """formal 与 legacy 混组：失败。"""
    t = _make_run(tmp_path, "run_f")
    sources = {
        "mini_cnn": {
            "checkpoint": t.checkpoints_dir / "last.pth",
            "history": t.run_dir / "history.json",
        },
        "vgg_lite": {
            "checkpoint": LEGACY_CKPTS["vgg_lite"],
            "history": LEGACY_HISTORIES["vgg_lite"],
        },
    }
    with pytest.raises(ComparisonSourceError, match="混用"):
        validate_comparison_set(sources, runs_dir=tmp_path / "runs")


def test_different_protocol_rejected(tmp_path):
    """formal 组内协议不一致（如 scheduler 不同）：失败。"""
    t1 = _make_run(tmp_path, "run_g")
    cfg2 = _make_config()
    cfg2["training"]["scheduler"] = "cosine"
    cfg2["models"]["vgg_lite"] = dict(cfg2["models"]["mini_cnn"])
    t2 = _make_run(tmp_path, "run_h", config=cfg2, model="vgg_lite")
    sources = {
        "mini_cnn": {
            "checkpoint": t1.checkpoints_dir / "last.pth",
            "history": t1.run_dir / "history.json",
        },
        "vgg_lite": {
            "checkpoint": t2.checkpoints_dir / "last.pth",
            "history": t2.run_dir / "history.json",
        },
    }
    with pytest.raises(ComparisonSourceError, match="协议"):
        validate_comparison_set(sources, runs_dir=tmp_path / "runs")

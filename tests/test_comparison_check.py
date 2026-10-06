"""
比较来源一致性校验测试 — S04；正式实验准入测试 — T06

覆盖审计负例：旧权重 + 新 run history 必须失败；及 run/协议/模型组一致性；
T06：来源绑定（run-bound）与正式实验准入（formal_eligible）分离的判定矩阵。
"""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest
import torch

from tests.test_training_pipeline import _make_config, _make_trainer
from utils.comparison_check import (
    ComparisonSourceError,
    check_comparison_source,
    check_formal_eligibility,
    validate_comparison_set,
)
from utils.model_spec import file_sha256

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


def test_run_bound_ckpt_with_legacy_history_rejected(tmp_path):
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


def test_same_run_passes_run_bound(tmp_path):
    """同一 run 的 checkpoint + history：run-bound（来源绑定）通过。"""
    t = _make_run(tmp_path, "run_e")
    report = check_comparison_source(
        "mini_cnn", t.checkpoints_dir / "last.pth", t.run_dir / "history.json",
        runs_dir=tmp_path / "runs",
    )
    assert report["verdict"] == "run-bound"
    assert report["run_id"] == t.run_id
    assert report["protocol_digest"]


def test_mixed_run_bound_and_legacy_set_rejected(tmp_path):
    """run-bound 与 legacy 混组：失败。"""
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
    """run-bound 组内协议不一致（如 scheduler 不同）：失败。"""
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

# ============================================================
# T06 · 正式实验准入判定（来源绑定 ≠ 正式资格）
# ============================================================
def _write_case(tmp_path, name, meta_overrides=None, *, write_frozen=None,
                last_partial=False, with_meta=True):
    """伪造 run 目录：run_meta + checkpoints/last.pth（可选冻结文件）。"""
    run_dir = tmp_path / "runs" / "mini_cnn" / name
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    if with_meta:
        meta = {
            "run_purpose": "smoke", "status": "finished",
            "data": {"csv_sha256": "a" * 64},
        }
        meta.update(meta_overrides or {})
        (run_dir / "run_meta.json").write_text(
            json.dumps(meta, ensure_ascii=False), encoding="utf-8"
        )
    torch.save({"partial": last_partial}, run_dir / "checkpoints" / "last.pth")
    frozen_path = tmp_path / "frozen.json"
    if write_frozen is not None:
        frozen_path.write_text(write_frozen, encoding="utf-8")
    return run_dir, frozen_path


_FROZEN_BODY = '{"protocol_id": "fz-1", "frozen_at": "2026-01-01", "git_commit": "c0ffee"}'


def _frozen_record(fp: Path) -> dict:
    return {"protocol_id": "fz-1", "file_sha256": file_sha256(fp)}


def test_eligibility_smoke_and_unspecified_marked_informal(tmp_path):
    for name, purpose in (("e_smoke", "smoke"), ("e_unspec", "unspecified")):
        run_dir, fp = _write_case(tmp_path, name, {"run_purpose": purpose})
        e = check_formal_eligibility(run_dir, frozen_protocol_path=fp)
        assert not e["formal_eligible"]
        assert any("run_purpose" in r for r in e["reasons"])


def test_eligibility_formal_complete_passes(tmp_path):
    fp = tmp_path / "frozen.json"
    fp.write_text(_FROZEN_BODY, encoding="utf-8")
    run_dir, _ = _write_case(
        tmp_path, "e_formal",
        {"run_purpose": "formal", "frozen_protocol": _frozen_record(fp)},
        write_frozen=_FROZEN_BODY,
    )
    e = check_formal_eligibility(run_dir, frozen_protocol_path=fp)
    assert e["formal_eligible"], e["reasons"]
    assert e["run_purpose"] == "formal"


def test_eligibility_formal_without_frozen_record(tmp_path):
    run_dir, fp = _write_case(tmp_path, "e_nofrozen", {"run_purpose": "formal"})
    e = check_formal_eligibility(run_dir, frozen_protocol_path=fp)
    assert not e["formal_eligible"]
    assert any("冻结协议绑定" in r for r in e["reasons"])


def test_eligibility_frozen_file_missing(tmp_path):
    ghost = tmp_path / "ghost_frozen.json"       # 记录指向的文件不存在
    ghost_record = {"protocol_id": "fz-1", "file_sha256": "0" * 64}
    run_dir, _ = _write_case(
        tmp_path, "e_ghost",
        {"run_purpose": "formal", "frozen_protocol": ghost_record},
    )
    e = check_formal_eligibility(run_dir, frozen_protocol_path=ghost)
    assert not e["formal_eligible"]
    assert any("不存在" in r for r in e["reasons"])


def test_eligibility_frozen_content_changed(tmp_path):
    fp = tmp_path / "frozen.json"
    fp.write_text(_FROZEN_BODY, encoding="utf-8")
    record = _frozen_record(fp)
    # 记录后冻结内容被更换（SHA 不符）→ 拒绝
    fp.write_text(_FROZEN_BODY.replace("c0ffee", "beefee"), encoding="utf-8")
    run_dir, _ = _write_case(
        tmp_path, "e_changed",
        {"run_purpose": "formal", "frozen_protocol": record},
        write_frozen=None,
    )
    e = check_formal_eligibility(run_dir, frozen_protocol_path=fp)
    assert not e["formal_eligible"]
    assert any("不一致" in r for r in e["reasons"])


def test_eligibility_not_finished_or_partial_or_no_data(tmp_path):
    fp = tmp_path / "frozen.json"
    fp.write_text(_FROZEN_BODY, encoding="utf-8")
    base = {"run_purpose": "formal", "frozen_protocol": _frozen_record(fp)}

    r1, _ = _write_case(tmp_path, "e_running", {**base, "status": "running"})
    assert not check_formal_eligibility(r1, frozen_protocol_path=fp)["formal_eligible"]

    r2, _ = _write_case(tmp_path, "e_partial", base, last_partial=True)
    assert not check_formal_eligibility(r2, frozen_protocol_path=fp)["formal_eligible"]

    r3, _ = _write_case(tmp_path, "e_nodata",
                        {**base, "data": {}})
    e3 = check_formal_eligibility(r3, frozen_protocol_path=fp)
    assert not e3["formal_eligible"]
    assert any("数据指纹" in r for r in e3["reasons"])

    r4, _ = _write_case(tmp_path, "e_nometa", with_meta=False)
    e4 = check_formal_eligibility(r4, frozen_protocol_path=fp)
    assert not e4["formal_eligible"]
    assert any("run_meta" in r for r in e4["reasons"])


def test_real_historical_runs_are_not_eligible():
    """当前项目中未声明用途的历史/短 run：全部判定为非正式（T06 验收）。"""
    metas = sorted((PROJECT_ROOT / "training" / "runs").glob("*/*/run_meta.json"))
    if not metas:
        pytest.skip("training/runs 下无 run_meta")
    checked = 0
    for m in metas:
        meta = json.loads(m.read_text(encoding="utf-8"))
        if meta.get("run_purpose") is not None:
            continue  # 已显式声明用途的 run 不属于本回归对象
        e = check_formal_eligibility(m.parent)
        assert not e["formal_eligible"], f"{m.parent} 不应通过正式准入"
        checked += 1
    assert checked >= 3, f"应至少覆盖 3 个历史 run（实际 {checked}）"

"""
比较来源一致性校验 — S04

比较 Notebook（analysis/comparison_report.ipynb）与正式比较流程共用本模块，
把每个模型的 checkpoint 与 history 绑定为可校验的“来源清单”，区分两种模式：

  - **formal**：checkpoint 为新格式 run（含 run_id / model_spec / training_protocol），
    history 必须来自**同一 run**（training/runs/<model>/<run_id>/history.json），
    且整组必须共享同一训练协议（protocol_digest 一致）。
  - **legacy-unverified**：checkpoint 为 legacy 产物（无 run 元数据），history 必须
    来自 legacy 日志（training/logs/）；整组明确标注“来源未验证”，不得混入正式汇总。

“新权重 + 旧日志”“旧权重 + 新 run 日志”“run 不匹配”“协议不一致”等组合
都会在校验阶段失败（在生成任何图表/表格/评估输出之前）。

用法:
    from utils.comparison_check import validate_comparison_set
    reports = validate_comparison_set({name: {"checkpoint": ..., "history": ...}})
"""

__all__ = ["ComparisonSourceError", "check_comparison_source", "validate_comparison_set"]

import hashlib
import json
from pathlib import Path

import torch

from utils.model_spec import file_sha256

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = PROJECT_ROOT / "training" / "runs"
LEGACY_LOGS_DIR = PROJECT_ROOT / "training" / "logs"


class ComparisonSourceError(RuntimeError):
    """比较来源不一致 / 不完整：在生成图表或结果前抛出。"""


def _resolve(path) -> Path:
    """相对路径按项目根解析（notebook 在 analysis/ 下运行等场景）。"""
    p = Path(path)
    return p if p.is_absolute() else (PROJECT_ROOT / p)


def _load_checkpoint_info(checkpoint_path: Path) -> dict:
    """读取 checkpoint 的来源信息（run_id / legacy 判定 / 协议摘要 / SHA-256）。"""
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    protocol = ckpt.get("training_protocol")
    digest = None
    if isinstance(protocol, dict):
        digest = hashlib.sha256(
            json.dumps(protocol, sort_keys=True, default=str).encode()
        ).hexdigest()[:16]
    return {
        "path": str(checkpoint_path),
        "sha256": file_sha256(checkpoint_path),
        "legacy": "training_protocol" not in ckpt or "model_spec" not in ckpt,
        "run_id": ckpt.get("run_id") or None,
        "model_name": ckpt.get("model_name") or None,
        "has_spec": "model_spec" in ckpt,
        "protocol_digest": digest,
        "data_fingerprint": (protocol or {}).get("data_fingerprint"),
    }


def _classify_history_path(history_path: Path, runs_dir: Path, legacy_logs_dir: Path):
    """返回 (kind, model_name, run_id)；kind ∈ {"run", "legacy", "other"}。"""
    resolved = history_path.resolve()
    try:
        rel = resolved.relative_to(Path(runs_dir).resolve())
        if len(rel.parts) == 3 and rel.parts[-1] == "history.json":
            return "run", rel.parts[0], rel.parts[1]
    except ValueError:
        pass
    try:
        rel = resolved.relative_to(Path(legacy_logs_dir).resolve())
        if len(rel.parts) == 2 and rel.parts[-1] == "history.json":
            return "legacy", rel.parts[0], None
    except ValueError:
        pass
    return "other", None, None


def check_comparison_source(
    model_name: str,
    checkpoint_path,
    history_path,
    *,
    runs_dir: Path = RUNS_DIR,
    legacy_logs_dir: Path = LEGACY_LOGS_DIR,
) -> dict:
    """
    校验单个模型来源（checkpoint + history 的绑定关系）。

    Returns:
        {"model_name", "checkpoint", "checkpoint_sha256", "checkpoint_spec",
         "history", "run_id", "protocol_digest", "verdict", "problems"}
      - verdict == "formal"：来源绑定齐全（同一 run + 协议）。
      - verdict == "legacy-unverified"：legacy 展示（缺元数据，明确标注）。

    Raises:
        ComparisonSourceError: 文件缺失 / 来源混用 / run 不匹配 / 指纹不一致。
    """
    ckpt_path = _resolve(checkpoint_path)
    hist_path = _resolve(history_path)
    if not ckpt_path.exists():
        raise ComparisonSourceError(f"{model_name}: checkpoint 不存在: {ckpt_path}")
    if not hist_path.exists():
        raise ComparisonSourceError(f"{model_name}: history 不存在: {hist_path}")

    info = _load_checkpoint_info(ckpt_path)
    kind, hist_model, hist_run_id = _classify_history_path(
        hist_path, runs_dir, legacy_logs_dir
    )

    report = {
        "model_name": model_name,
        "checkpoint": info["path"],
        "checkpoint_sha256": info["sha256"],
        "history": str(hist_path),
        "run_id": None,
        "protocol_digest": None,
        "verdict": None,
        "problems": [],
    }

    if info["legacy"]:
        if kind != "legacy":
            raise ComparisonSourceError(
                f"{model_name}: legacy 权重与来自 '{kind}' 的 history 混用"
                f"（{hist_path}）。legacy 权重只能配 training/logs/ 下的 legacy 日志，"
                "整组标记「来源未验证」；禁止与 runs/ 运行记录混排。"
            )
        report["verdict"] = "legacy-unverified"
        return report

    # ---- 新格式权重（formal 路径）----
    ckpt_model_name = info.get("model_name")
    if ckpt_model_name and ckpt_model_name != model_name:
        raise ComparisonSourceError(
            f"{model_name}: checkpoint 的 model_name（{ckpt_model_name}）"
            "与目标模型不一致。"
        )
    ckpt_run_id = info["run_id"]
    if not ckpt_run_id:
        raise ComparisonSourceError(
            f"{model_name}: 新格式权重缺少 run_id（{ckpt_path}）——来源不完整。"
        )
    if kind != "run":
        raise ComparisonSourceError(
            f"{model_name}: 新格式权重（run={ckpt_run_id}）必须配合来自 "
            "training/runs/<model>/<run_id>/history.json 的 history；"
            f"当前为 '{kind}' 来源（{hist_path}）——来源不一致。"
        )
    if hist_model is not None and hist_model != model_name:
        raise ComparisonSourceError(
            f"{model_name}: history 所属模型（{hist_model}）与目标模型不一致。"
        )
    if hist_run_id != ckpt_run_id:
        raise ComparisonSourceError(
            f"{model_name}: history 所属 run（{hist_run_id}）与 checkpoint 的 run"
            f"（{ckpt_run_id}）不匹配。"
        )

    # 数据指纹一致性（run_meta 与 checkpoint 协议均可读时）
    run_meta_path = hist_path.parent / "run_meta.json"
    if run_meta_path.exists() and info["data_fingerprint"]:
        try:
            run_meta = json.loads(run_meta_path.read_text(encoding="utf-8"))
        except Exception:
            run_meta = {}
        meta_fp = run_meta.get("data") if isinstance(run_meta.get("data"), dict) else None
        if meta_fp and info["data_fingerprint"].get("csv_sha256") != meta_fp.get("csv_sha256"):
            raise ComparisonSourceError(
                f"{model_name}: run 元数据的数据指纹与 checkpoint 协议不一致"
                "（CSV SHA-256 不同）——请核对 run 产物。"
            )

    report["run_id"] = ckpt_run_id
    report["protocol_digest"] = info["protocol_digest"]
    report["verdict"] = "formal"
    return report


def validate_comparison_set(
    sources: dict,
    *,
    runs_dir: Path = RUNS_DIR,
    legacy_logs_dir: Path = LEGACY_LOGS_DIR,
) -> list[dict]:
    """
    校验整组比较来源（每个模型一个 checkpoint + history 绑定）。

    Args:
        sources: {model_name: {"checkpoint": path, "history": path}}

    Returns:
        每项 report（见 check_comparison_source）。

    Raises:
        ComparisonSourceError: 任一项失败；formal 与 legacy 混用；
        或 formal 组内协议不一致（protocol_digest 不唯一）。
    """
    reports = [
        check_comparison_source(
            name, src["checkpoint"], src["history"],
            runs_dir=runs_dir, legacy_logs_dir=legacy_logs_dir,
        )
        for name, src in sources.items()
    ]
    verdicts = {r["verdict"] for r in reports}
    if "formal" in verdicts and "legacy-unverified" in verdicts:
        raise ComparisonSourceError(
            "禁止混用：formal 与 legacy-unverified 来源出现在同一比较集合；"
            "请统一为同一批冻结协议的 runs，或全部标注 legacy。"
        )
    if verdicts == {"formal"}:
        digests = {r["protocol_digest"] for r in reports}
        if len(digests) != 1 or None in digests:
            raise ComparisonSourceError(
                f"正式比较要求共享同一训练协议；当前 protocol_digest={sorted(map(str, digests))}——"
                "请确认所有 run 使用同一冻结协议（docs/comparison_protocol_draft.md）。"
            )
    return reports

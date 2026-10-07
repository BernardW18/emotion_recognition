"""
比较来源一致性校验（S04）与正式实验准入（T06）

比较 Notebook（analysis/comparison_report.ipynb）与正式比较流程共用本模块：

  1) **来源绑定**（S04，validate_comparison_set）：把每个模型的 checkpoint 与 history
     绑定为可校验的“来源清单”，区分两种状态：
       - **run-bound**：checkpoint 为新格式 run（含 run_id / model_spec /
         training_protocol），history 必须来自**同一 run**，且整组共享同一训练协议
         （protocol_digest 一致）。注意：run-bound 仅代表**来源绑定齐全**，
         **不代表正式实验资格**。
       - **legacy-unverified**：legacy 产物（无 run 元数据）+ legacy 日志；
         整组标注“来源未验证”，不得混入正式汇总。
  2) **正式实验准入**（T06，check_formal_eligibility）：在来源绑定之外，另行判定单个
     run 是否满足正式实验要求（formal 用途声明 + 冻结协议绑定一致 + 正常结束 +
     完整断点 + 数据指纹）。来源完整但未获准的 run 可独立查看，但不进入正式汇总。

“新权重 + 旧日志”“旧权重 + 新 run 日志”“run 不匹配”“协议不一致”等组合
都会在校验阶段失败（在生成任何图表/表格/评估输出之前）。

用法:
    from utils.comparison_check import validate_comparison_set, check_formal_eligibility
    reports = validate_comparison_set({name: {"checkpoint": ..., "history": ...}})
    elig = check_formal_eligibility("training/runs/mini_cnn/<run_id>")
"""

__all__ = [
    "FROZEN_PROTOCOL_PATH",
    "ComparisonSourceError",
    "check_comparison_source",
    "validate_comparison_set",
    "check_formal_eligibility",
]

import hashlib
import json
from pathlib import Path

import torch
import yaml

from utils.formal_protocol import completion_reason, load_frozen_protocol, validate_formal_plan
from utils.model_spec import ModelSpec, build_model_from_spec, file_sha256

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = PROJECT_ROOT / "training" / "runs"
LEGACY_LOGS_DIR = PROJECT_ROOT / "training" / "logs"

# T06：冻结协议文件（正式训练 --purpose formal 时须存在；冻结流程见
# docs/comparison_protocol_draft.md §6。格式：{"protocol_id", "frozen_at",
# "git_commit", ...}；run_meta 记录其 id + 文件字节 SHA-256，判定时复核一致性）
FROZEN_PROTOCOL_PATH = PROJECT_ROOT / "configs" / "protocols" / "comparison_protocol_frozen.json"


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
      - verdict == "run-bound"：来源绑定齐全（同一 run + 协议）；
        **不等于正式实验资格**（准入见 check_formal_eligibility）。
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

    # ---- 新格式权重（run-bound 路径）----
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
    report["verdict"] = "run-bound"
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
        ComparisonSourceError: 任一项失败；run-bound 与 legacy 混用；
        或 run-bound 组内协议不一致（protocol_digest 不唯一）。
    """
    reports = [
        check_comparison_source(
            name, src["checkpoint"], src["history"],
            runs_dir=runs_dir, legacy_logs_dir=legacy_logs_dir,
        )
        for name, src in sources.items()
    ]
    verdicts = {r["verdict"] for r in reports}
    if "run-bound" in verdicts and "legacy-unverified" in verdicts:
        raise ComparisonSourceError(
            "禁止混用：run-bound 与 legacy-unverified 来源出现在同一比较集合；"
            "请统一为同一批冻结协议的 runs，或全部标注 legacy。"
        )
    if verdicts == {"run-bound"}:
        digests = {r["protocol_digest"] for r in reports}
        if len(digests) != 1 or None in digests:
            raise ComparisonSourceError(
                "来源绑定组要求共享同一训练协议；"
                f"当前 protocol_digest={sorted(map(str, digests))}——"
                "请确认所有 run 使用同一冻结协议（docs/comparison_protocol_draft.md）；"
                "正式实验准入另行判定（check_formal_eligibility）。"
            )
    return reports


def check_formal_eligibility(
    run_dir,
    *,
    frozen_protocol_path: Path | None = None,
) -> dict:
    """
    T06：正式实验准入判定（**与 run-bound 来源状态分离**）。

    来源绑定齐全（run-bound）不代表正式实验资格；正式合法 run 需同时满足：

      1) run_meta.json 存在且含数据指纹（data.csv_sha256）；
      2) ``run_purpose == "formal"``（训练入口 ``--purpose formal`` 显式声明；
         缺省 / smoke / diagnose 均视为非正式）；
      3) 可执行冻结清单与实际配置/seed/数据/模型/臂逐项相符；
      4) 达到预算或可复算的合规早停；仅结束会话的 session_completed 不准入；
      5) v3 last/best 的真实结构、完整进度与 history/config/meta/来源全部绑定。

    Returns:
        {"run_dir", "run_purpose", "status", "frozen_protocol",
         "formal_eligible": bool, "reasons": [...]}
    """
    from training.state_validation import validate_checkpoint_payload

    run_dir = Path(run_dir)
    result: dict = {
        "run_dir": str(run_dir),
        "run_purpose": None,
        "status": None,
        "frozen_protocol": None,
        "formal_eligible": False,
        "reasons": [],
    }
    if not run_dir.exists():
        result["reasons"].append(f"run 目录不存在: {run_dir}")
        return result

    meta_path = run_dir / "run_meta.json"
    if not meta_path.exists():
        result["reasons"].append("缺少 run_meta.json（非新格式 run）")
        return result
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception as e:
        result["reasons"].append(f"run_meta.json 不可解析: {e}")
        return result

    if not isinstance(meta, dict):
        result["reasons"].append("run_meta 必须为字典")
        return result
    result["run_purpose"] = meta.get("run_purpose")
    result["status"] = meta.get("status")
    result["frozen_protocol"] = meta.get("frozen_protocol")

    if meta.get("run_purpose") != "formal":
        result["reasons"].append(
            f"run_purpose={meta.get('run_purpose')!r}"
            "（非 formal；流程验证/诊断/未声明均非正式）"
        )

    frozen = meta.get("frozen_protocol")
    record = None
    if not isinstance(frozen, dict) or not frozen.get("protocol_id"):
        result["reasons"].append("缺少冻结协议绑定（frozen_protocol 记录）")
    else:
        if frozen_protocol_path is not None:
            fp = Path(frozen_protocol_path)
        else:
            bound_path = frozen.get("path")
            fp = (
                Path(bound_path)
                if isinstance(bound_path, str) and bound_path else FROZEN_PROTOCOL_PATH
            )
            if not fp.is_file():
                # Match exact bytes, not user-controlled protocol IDs as path components.
                matches = [p for p in (PROJECT_ROOT / "configs" / "protocols").glob("*.json")
                           if file_sha256(p) == frozen.get("file_sha256")]
                fp = matches[0] if len(matches) == 1 else FROZEN_PROTOCOL_PATH
        if not fp.exists():
            result["reasons"].append(f"冻结协议文件不存在: {fp}")
        elif frozen.get("file_sha256") != file_sha256(fp):
            result["reasons"].append(
                "冻结协议文件与 run 记录不一致（file_sha256 不符）"
                "——冻结内容不得事后变更"
            )
        else:
            try:
                record = load_frozen_protocol(fp, historical=True)
                if any(record[k] != frozen.get(k) for k in (
                    "protocol_id", "file_sha256", "manifest",
                )):
                    raise ValueError("冻结协议 id/实际清单与 run 绑定不一致")
            except (ValueError, KeyError, TypeError) as e:
                result["reasons"].append(f"冻结协议非法: {e}")

    data = meta.get("data")
    if not (isinstance(data, dict) and data.get("csv_sha256")):
        result["reasons"].append("run_meta 缺数据指纹（csv_sha256）")

    if meta.get("status") != "finished":
        result["reasons"].append(f"训练未正常结束（status={meta.get('status')!r}）")

    last_path = run_dir / "checkpoints" / "last.pth"
    if not last_path.exists():
        result["reasons"].append("缺少完整断点 checkpoints/last.pth")
    else:
        try:
            last = torch.load(last_path, map_location="cpu", weights_only=False)
            # Building a model for shape/schema checks must not consume caller RNG.
            with torch.random.fork_rng(devices=[]):
                model = build_model_from_spec(ModelSpec.from_dict(last["model_spec"]))
            validate_checkpoint_payload(last, model)
            if last["partial"]:
                raise ValueError("last.pth 为 partial")
            if (last["run_id"] != run_dir.name or meta.get("run_id") != run_dir.name
                    or last["model_name"] != meta.get("model_name")
                    or last["model_spec"] != meta.get("model_spec")
                    or last.get("run_purpose") != "formal"
                    or last.get("frozen_protocol") != frozen):
                raise ValueError("last 与模型/run/用途/冻结绑定不匹配")
            protocol = last["training_protocol"]
            if protocol["data_fingerprint"] != data:
                raise ValueError("last 数据指纹与 run_meta 不匹配")
            history = json.loads((run_dir / "history.json").read_text(encoding="utf-8"))
            if history != last["history"] or meta.get("final_epoch") != last["epoch"]:
                raise ValueError("last/history/meta 进度不匹配")
            # Reconcile saved effective config with the checkpoint's normalized snapshot.
            from training.trainer import _PROTOCOL_WHITELIST_PATHS, _delete_config_path

            config = yaml.safe_load((run_dir / "config_effective.yaml").read_text("utf-8"))
            for path in _PROTOCOL_WHITELIST_PATHS:
                _delete_config_path(config, path)
            if config != protocol["config"] or meta.get("seed") != config["seed"]:
                raise ValueError("生效配置/seed 与断点不匹配")
            if record is None:
                raise ValueError("缺少合法可执行冻结清单")
            plan = validate_formal_plan(
                record, last["model_name"], last["model_spec"], protocol, historical=True
            )
            reason = completion_reason(plan, last)
            if (reason is None or not meta.get("experiment_completed")
                    or meta.get("completion_reason") != reason):
                raise ValueError("未达到冻结预算且无合规早停，或完成标记不一致")
            best = torch.load(
                run_dir / "checkpoints" / "best.pth", map_location="cpu", weights_only=False
            )
            validate_checkpoint_payload(best, model)
            if (best["partial"] or best["epoch"] != last["best"]["epoch"]
                    or any(best.get(k) != last.get(k) for k in (
                        "run_id", "model_name", "model_spec", "training_protocol",
                        "run_purpose", "frozen_protocol",
                    ))
                    or any(best["history"][k] != v[:best["epoch"]]
                           for k, v in last["history"].items())):
                raise ValueError("best 与当前 run/历史/协议不匹配")
        except (OSError, RuntimeError, ValueError, KeyError, TypeError, AttributeError) as e:
            result["reasons"].append(f"正式产物不完整/不合规: {e}")

    result["formal_eligible"] = not result["reasons"]
    return result

"""
Checkpoint 与 run 管理模块 — run 隔离（F05）/ 完整续训状态（F09）

职责:
  - run 目录与元数据：training/runs/<model_name>/<run_id>/
  - 保存 / 加载 checkpoint（模型、优化器、调度器、AMP scaler、RNG、早停状态、history）
  - 原子写入：先写临时文件再替换；保存失败不破坏上一份可用断点
  - 定期断点清理
  - 续训断点自动发现（最新 run 的最新完整 last.pth）

run 目录结构:
    training/runs/<model_name>/<run_id>/
        config_effective.yaml      # 启动时最终生效配置（CLI 覆盖后）
        run_meta.json              # CLI 参数 / seed / git / 环境 / 数据指纹 / 状态
        history.json               # 训练历史（每 epoch 更新）
        checkpoints/
            last.pth               # 最新完整 epoch（续训入口）
            best.pth               # 本 run 最优（受 checkpoint.save_best 控制）
            epoch_XXXX.pth         # 定期断点
            interrupted_*.pth      # 中断断点（partial=True）

checkpoint 格式版本: 3，新增实际更新/尝试数及完整进度校验。版本 1/2 缺少
可验证更新记录，不支持精确续训；历史权重仍可推理/评估，不静默迁移。

用法:
    from training.checkpoint import save_checkpoint, load_checkpoint
    save_checkpoint(trainer, run_dir / "checkpoints" / "last.pth")
    load_checkpoint(trainer, "training/runs/mini_cnn/<run_id>/checkpoints/last.pth")
"""

__all__ = [
    "CHECKPOINT_FORMAT_VERSION",
    "TRAINING_PROTOCOL_VERSION",
    "scheduler_effective_dict",
    "collect_git_info",
    "collect_environment_info",
    "write_json_atomic",
    "write_run_meta",
    "read_run_meta",
    "save_checkpoint",
    "load_checkpoint",
    "load_checkpoint_metadata",
    "cleanup_old_checkpoints",
    "find_resume_checkpoint",
    "select_checkpoint_widget",
    "_format_duration",
]

import copy
import json
import logging
import os
import platform
import random as pyrandom
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from training.state_validation import FORMAT_VERSION, validate_checkpoint_payload

# 项目根目录（直接计算，避免循环依赖）
PROJECT_ROOT = Path(__file__).resolve().parent.parent

logger = logging.getLogger("checkpoint")

CHECKPOINT_FORMAT_VERSION = FORMAT_VERSION

# 训练协议快照版本（S01/S02：完整生效配置 + 运行时有效值 + 数据管线规格）
TRAINING_PROTOCOL_VERSION = 2

# run 根目录：training/runs/
RUNS_ROOT = PROJECT_ROOT / "training" / "runs"
# 旧产物目录（只读保留）：training/checkpoints/
LEGACY_CHECKPOINT_ROOT = PROJECT_ROOT / "training" / "checkpoints"

_EPOCH_CKPT_GLOB = "epoch_*.pth"


# ============================================================
# 格式化辅助函数
# ============================================================
def _format_duration(seconds: float) -> str:
    """将秒数格式化为人类可读时长 (如 12m35s / 1h23m)"""
    if seconds <= 0:
        return ""
    if seconds < 60:
        return f"{int(seconds)}s"
    elif seconds < 3600:
        return f"{int(seconds // 60)}m{int(seconds % 60):02d}s"
    else:
        return f"{int(seconds // 3600)}h{int((seconds % 3600) // 60):02d}m"


def _relative_time(ts: str) -> str:
    """将 ISO 时间戳转换为相对时间 (如 5分钟前 / 1小时前 / 昨天)"""
    try:
        dt = datetime.fromisoformat(ts)
        diff = (datetime.now() - dt).total_seconds()
        if diff < 60:
            return "刚刚"
        elif diff < 3600:
            return f"{int(diff / 60)}分钟前"
        elif diff < 86400:
            return f"{int(diff / 3600)}小时前"
        elif diff < 172800:
            return "昨天"
        else:
            return dt.strftime("%m-%d %H:%M")
    except Exception:
        return ""


# ============================================================
# run 元数据收集
# ============================================================
def collect_git_info(root: Path | None = None) -> dict:
    """收集 git 提交与工作区状态；非 git 环境或失败时返回 unknown 标记。"""
    root = root or PROJECT_ROOT
    info: dict[str, object] = {"commit": "unknown", "dirty": None, "dirty_files": None}
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, timeout=10
        )
        if commit.returncode == 0:
            info["commit"] = commit.stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"], cwd=root, capture_output=True, text=True, timeout=10
        )
        if status.returncode == 0:
            files = [
                line.split(maxsplit=1)[-1]
                for line in status.stdout.splitlines()
                if line.strip()
            ]
            info["dirty"] = bool(files)
            info["dirty_files"] = files[:50]
    except Exception as e:  # git 不可用不应阻断训练
        logger.warning("git 信息收集失败: %s", e)
    return info


def collect_environment_info() -> dict:
    """收集 Python / torch / CUDA / 设备环境信息。"""
    info = {
        "python": platform.python_version(),
        "platform": sys.platform,
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        info["device_name"] = torch.cuda.get_device_name(0)
    return info


def write_json_atomic(path: Path, obj) -> Path:
    """通用原子 JSON 写入（先写临时文件再替换，失败不破坏旧文件）。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}_", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
        os.replace(tmp_name, path)
    except BaseException:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)
        raise
    return path


def write_run_meta(run_dir: Path, meta: dict) -> Path:
    """原子写入 run_meta.json。"""
    return write_json_atomic(Path(run_dir) / "run_meta.json", meta)


def read_run_meta(run_dir: Path) -> dict | None:
    """读取 run_meta.json；不存在或损坏返回 None。"""
    path = Path(run_dir) / "run_meta.json"
    if not path.exists():
        return None
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    except Exception as e:
        logger.warning("run_meta.json 读取失败: %s", e)
        return None


# ============================================================
# Checkpoint 核心操作
# ============================================================
def _capture_rng_state() -> dict:
    """捕获 Python / NumPy / Torch / CUDA 随机数状态（F09 续训一致性）。"""
    state = {
        "python": pyrandom.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: dict) -> None:
    """恢复全局 RNG 状态（T03：调用前必须已通过状态完整性预检；失败即抛出，不静默降级）。"""
    try:
        pyrandom.setstate(state["python"])
        np.random.set_state(state["numpy"])
        torch.set_rng_state(state["torch"])
        if torch.cuda.is_available() and state.get("torch_cuda") is not None:
            torch.cuda.set_rng_state_all(state["torch_cuda"])
    except Exception as e:
        raise RuntimeError(f"全局 RNG 状态恢复失败（状态完整性预检后意外失败）: {e}") from e


def _protocol_diff(reference: dict, current: dict, path: str = "") -> list[str]:
    """递归比较两个训练协议快照，返回差异描述列表（R02）。"""
    diffs: list[str] = []
    for key in sorted(set(reference) | set(current)):
        full = f"{path}{key}"
        if key not in reference:
            diffs.append(f"{full}: 断点中缺失（当前={current[key]!r}）")
        elif key not in current:
            diffs.append(f"{full}: 当前缺失（断点={reference[key]!r}）")
        else:
            ref_value, cur_value = reference[key], current[key]
            if isinstance(ref_value, dict) and isinstance(cur_value, dict):
                diffs.extend(_protocol_diff(ref_value, cur_value, path=f"{full}."))
            elif ref_value != cur_value:
                diffs.append(f"{full}: 断点={ref_value!r} ≠ 当前={cur_value!r}")
    return diffs


def scheduler_effective_dict(scheduler) -> dict | None:
    """调度器“实际构建参数”快照（S02：声明配置与实际行为的一致性校验）。"""
    if scheduler is None:
        return {"name": "none"}
    out: dict[str, object] = {"name": type(scheduler).__name__}
    for attr in ("T_0", "T_mult", "T_max", "step_size", "gamma", "patience"):
        if hasattr(scheduler, attr):
            value = getattr(scheduler, attr)
            if isinstance(value, (int, float, str, bool)) or value is None:
                out[attr] = value
            else:
                out[attr] = str(value)
    return out


def _capture_loader_rng(loader) -> dict:
    """捕获训练 loader 的独立生成器状态（S01：批次序列与 worker 种子复算）。"""
    sampler = getattr(loader, "sampler", None)
    g_sampler = getattr(sampler, "generator", None)
    g_loader = getattr(loader, "generator", None)
    return {
        "sampler_generator": (
            g_sampler.get_state() if isinstance(g_sampler, torch.Generator) else None
        ),
        "loader_generator": (
            g_loader.get_state() if isinstance(g_loader, torch.Generator) else None
        ),
    }


def _restore_loader_rng(loader, state: dict | None) -> None:
    """恢复 loader 生成器状态（T03：预检已保证键完整且可恢复；失败即抛出）。"""
    if state is None:
        raise RuntimeError("断点缺少 loader_rng 状态（状态完整性预检后意外失败）")
    sampler = getattr(loader, "sampler", None)
    g_sampler = getattr(sampler, "generator", None)
    if state.get("sampler_generator") is not None:
        if not isinstance(g_sampler, torch.Generator):
            raise RuntimeError("当前 sampler 无独立 generator，无法恢复其状态")
        g_sampler.set_state(state["sampler_generator"])
    g_loader = getattr(loader, "generator", None)
    if state.get("loader_generator") is not None:
        if not isinstance(g_loader, torch.Generator):
            raise RuntimeError("当前 DataLoader 无 generator，无法恢复 base_seed 序列")
        g_loader.set_state(state["loader_generator"])


def _verify_checkpoint_state(trainer, checkpoint: dict) -> dict:
    """
    T03：恢复前的状态完整性预检（**只读验证，不修改任何真实状态**）。

    按运行时要求逐项核验断点状态：全局 RNG（python/numpy/torch/cuda）、
    loader/sampler 生成器、调度器、AMP scaler、模型与优化器结构。
    所有"可恢复性"用临时生成器/副本（dry-run）验证，不触碰真实对象；
    任何缺项/非法值/不可恢复项 → 抛出 RuntimeError（在修改真实状态或文件前拒绝）。

    允许的可选缺省：本来不存在对应组件的训练（无 scheduler、CPU 无 scaler）
    不要求断点携带其状态；其余"当前需要而断点缺失"一律拒绝。

    Returns:
        dict：各组件校验摘要（写入恢复事件的 state_integrity）
    """
    report: dict[str, str] = {}
    problems: list[str] = []
    validate_checkpoint_payload(
        checkpoint, getattr(trainer.model, "_orig_mod", trainer.model)
    )
    if checkpoint["rng_cuda_device_count"] != torch.cuda.device_count():
        raise RuntimeError("CUDA RNG 设备数量/映射与当前环境不一致，拒绝精确恢复")

    # ---- 1) 全局 RNG（python / numpy / torch / cuda）----
    rng = checkpoint.get("rng")
    if not isinstance(rng, dict):
        problems.append("缺少全局 RNG 状态（rng 字段缺失或非法）")
    else:
        for key in ("python", "numpy", "torch"):
            if key not in rng or rng[key] is None:
                problems.append(f"rng 缺少必需键 {key!r}")
        if rng.get("python") is not None:
            try:
                pyrandom.Random().setstate(rng["python"])
            except Exception as e:
                problems.append(f"rng.python 非法（无法恢复）: {e}")
        if rng.get("numpy") is not None:
            try:
                np.random.RandomState().set_state(rng["numpy"])
            except Exception as e:
                problems.append(f"rng.numpy 非法（无法恢复）: {e}")
        if rng.get("torch") is not None:
            try:
                torch.Generator().set_state(rng["torch"])
            except Exception as e:
                problems.append(f"rng.torch 非法（无法恢复）: {e}")
        if rng.get("torch_cuda") is not None:
            if not torch.cuda.is_available():
                problems.append("断点含 CUDA RNG 状态，但当前环境无 CUDA：无法完整恢复")
            else:
                try:
                    for index, state_i in enumerate(rng["torch_cuda"]):
                        torch.Generator(device=f"cuda:{index}").set_state(state_i)
                except Exception as e:
                    problems.append(f"rng.torch_cuda 非法（无法恢复）: {e}")
    report["global_rng"] = "ok"

    # ---- 2) loader / sampler 生成器 ----
    loader_rng = checkpoint.get("loader_rng")
    if not isinstance(loader_rng, dict):
        problems.append("缺少 loader_rng（loader/sampler 生成器状态）")
    else:
        sampler = getattr(trainer.train_loader, "sampler", None)
        g_sampler = getattr(sampler, "generator", None)
        g_loader = getattr(trainer.train_loader, "generator", None)
        for key, gen, who in (
            ("sampler_generator", g_sampler, "sampler"),
            ("loader_generator", g_loader, "DataLoader"),
        ):
            state_i = loader_rng.get(key)
            if isinstance(gen, torch.Generator):
                if state_i is None:
                    problems.append(
                        f"断点缺少 {key}：当前 {who} 有独立生成器，无法复算其随机进度"
                    )
                else:
                    try:
                        torch.Generator().set_state(state_i)
                    except Exception as e:
                        problems.append(f"{key} 非法（无法恢复）: {e}")
            elif state_i is not None:
                problems.append(f"断点含 {key} 状态，但当前 {who} 无独立生成器")
    report["loader_rng"] = "ok"

    # ---- 3) 调度器（有 scheduler 训练时要求断点携带其状态）----
    # 注意：torch 的 LRScheduler.load_state_dict 会静默接受非法值（如 T_0="bad"），
    # 因此除副本加载外，还需对照断点记录的“实际调度器参数”（scheduler_effective）
    if trainer.scheduler is not None:
        sched_state = checkpoint.get("scheduler_state_dict")
        if sched_state is None:
            problems.append(
                "当前训练使用调度器，但断点缺少 scheduler_state_dict：无法精确恢复 LR 周期"
            )
        else:
            try:
                probe = copy.deepcopy(trainer.scheduler)
                probe.load_state_dict(sched_state)
                expected_sched = (
                    (checkpoint.get("training_protocol") or {}).get("runtime") or {}
                ).get("scheduler_effective")
                actual_sched = scheduler_effective_dict(probe)
                if expected_sched is not None and actual_sched != expected_sched:
                    problems.append(
                        "scheduler_state_dict 与断点记录的调度器参数不一致: "
                        f"{actual_sched} vs {expected_sched}"
                    )
                else:
                    report["scheduler"] = "checked"
            except Exception as e:
                problems.append(f"scheduler_state_dict 非法（无法恢复）: {e}")
    else:
        report["scheduler"] = "not-used"

    # ---- 4) AMP scaler（启用 AMP 时要求断点携带其状态）----
    # 同样在副本上加载并对关键值做数值化检查（load_state_dict 不校验坏值）
    scaler = getattr(trainer, "scaler", None)
    if scaler is not None:
        scaler_state = checkpoint.get("scaler_state_dict")
        if scaler_state is None:
            problems.append(
                "当前训练启用 AMP，但断点缺少 scaler_state_dict：无法精确恢复缩放状态"
            )
        else:
            try:
                probe_scaler = copy.deepcopy(scaler)
                probe_scaler.load_state_dict(scaler_state)
                # GradScaler.load_state_dict 为 lazy 语义：加载后 _scale 等实例属性
                # 仍可能是 None（等首次 update 才初始化）——因此校验来源 state 的值，
                # 而不是加载后实例的属性。
                required = ("scale", "growth_factor", "backoff_factor", "growth_interval")
                missing = [k for k in required if scaler_state.get(k) is None]
                if missing:
                    raise ValueError(f"缺少必需数值项 {missing}")
                float(scaler_state["scale"])
                float(scaler_state["growth_factor"])
                float(scaler_state["backoff_factor"])
                int(scaler_state["growth_interval"])
                tracker = scaler_state.get("_growth_tracker")
                if tracker is not None:   # None = 未初始化 tracker（合法，恢复后 lazy 继续）
                    int(tracker)
                report["amp_scaler"] = "checked"
            except Exception as e:
                problems.append(f"scaler_state_dict 非法（无法恢复）: {e}")
    else:
        report["amp_scaler"] = "not-used"

    # ---- 5) 模型结构（键 + 逐键形状；不复制权重）----
    model_sd = checkpoint.get("model_state_dict")
    if not isinstance(model_sd, dict):
        problems.append("model_state_dict 缺失或非 dict")
    else:
        current_sd = getattr(trainer.model, "_orig_mod", trainer.model).state_dict()
        missing = sorted(set(model_sd) - set(current_sd))
        extra = sorted(set(current_sd) - set(model_sd))
        if missing or extra:
            problems.append(
                f"模型状态键不匹配（断点缺失 {missing[:5]} / 当前多余 {extra[:5]}）"
            )
        else:
            for key in current_sd:
                cur_v, ck_v = current_sd[key], model_sd[key]
                if (
                    isinstance(cur_v, torch.Tensor) and isinstance(ck_v, torch.Tensor)
                    and tuple(cur_v.shape) != tuple(ck_v.shape)
                ):
                    problems.append(
                        f"模型参数 {key} 形状不匹配: "
                        f"断点 {tuple(ck_v.shape)} vs 当前 {tuple(cur_v.shape)}"
                    )
                    break
    report["model"] = "checked"

    # ---- 6) 优化器结构（param_groups 数量 + state 张量形状对照参数）----
    opt_state = checkpoint.get("optimizer_state_dict")
    if (
        not isinstance(opt_state, dict)
        or "param_groups" not in opt_state or "state" not in opt_state
    ):
        problems.append("optimizer_state_dict 缺失或结构非法")
    else:
        n_cur = len(trainer.optimizer.param_groups)
        n_ck = len(opt_state["param_groups"])
        if n_cur != n_ck:
            problems.append(f"优化器 param_groups 数量不匹配: 断点 {n_ck} vs 当前 {n_cur}")
        else:
            for current_group, saved_group in zip(
                trainer.optimizer.param_groups, opt_state["param_groups"], strict=True
            ):
                if len(current_group["params"]) != len(saved_group["params"]):
                    problems.append("优化器每组参数数目不匹配")
                for key, value in current_group.items():
                    if key not in ("params", "lr") and saved_group.get(key) != value:
                        problems.append(f"优化器静态参数 {key} 与实际配置不一致")
            flat_params = [p for g in trainer.optimizer.param_groups for p in g["params"]]
            for pid, st in opt_state["state"].items():
                if not isinstance(st, dict):
                    problems.append(f"优化器 state[{pid}] 非 dict")
                    break
                bad = False
                for name, val in st.items():
                    if (
                        isinstance(val, torch.Tensor) and val.numel() > 1
                        and int(pid) < len(flat_params)
                        and tuple(val.shape) != tuple(flat_params[int(pid)].shape)
                    ):
                        problems.append(
                            f"优化器 state[{pid}].{name} 形状 {tuple(val.shape)} "
                            f"与参数 {tuple(flat_params[int(pid)].shape)} 不符"
                        )
                        bad = True
                        break
                if bad:
                    break
        report["optimizer"] = "checked"

    if problems:
        raise RuntimeError(
            "断点状态完整性预检未通过，恢复被拒绝（未修改任何真实状态）：\n  - "
            + "\n  - ".join(problems)
        )
    return report


def _apply_training_state(trainer, checkpoint: dict) -> dict:
    """
    状态恢复的共享实现（load_checkpoint 与 S03 自动回滚共用）：

    覆盖 模型/BN、优化器、调度器、AMP scaler、history、训练进度、累计时长、
    best、早停计数、全局 RNG、loader 生成器；并校验恢复后实际调度器参数
    与断点记录一致（S02）。调用前必须已通过 _verify_checkpoint_state 预检（T03），
    本函数内任何恢复失败都会直接抛出（不静默降级）。

    Returns:
        {"saved_epoch": int}
    """
    # 恢复模型权重（torch.compile 场景同样展开到原始模块）
    try:
        getattr(trainer.model, "_orig_mod", trainer.model).load_state_dict(
            checkpoint["model_state_dict"]
        )
    except RuntimeError as e:
        raise RuntimeError(f"模型结构不匹配，无法加载 checkpoint: {e}") from e

    # 优化器
    if "optimizer_state_dict" in checkpoint:
        trainer.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    # 调度器
    if "scheduler_state_dict" in checkpoint:
        if trainer.scheduler is not None:
            trainer.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            logger.info("  - 调度器状态已恢复（LR 周期连续）")
    elif trainer.scheduler is not None:
        logger.warning("  - 断点无调度器状态，调度器保持初始化状态")

    # AMP scaler
    if "scaler_state_dict" in checkpoint and getattr(trainer, "scaler", None) is not None:
        trainer.scaler.load_state_dict(checkpoint["scaler_state_dict"])
        logger.info("  - AMP GradScaler 状态已恢复")

    # 训练历史
    if "history" in checkpoint:
        trainer.history = checkpoint["history"]

    # 训练进度
    history_epochs = len(trainer.history.get("val_acc", []))
    saved_epoch = checkpoint.get("epoch", 0)
    if saved_epoch != history_epochs and history_epochs > 0:
        logger.warning(
            "Checkpoint epoch (%d) 与 history 长度 (%d) 不一致，使用 history 长度",
            saved_epoch, history_epochs,
        )
        saved_epoch = history_epochs
    trainer.start_epoch = saved_epoch + 1

    # 累计训练时间
    trainer._accumulated_train_time = checkpoint.get("training_duration_seconds", 0.0)
    trainer._total_train_time = trainer._accumulated_train_time
    trainer._optimizer_updates = checkpoint["optimizer_updates"]
    trainer._optimizer_attempts = checkpoint["optimizer_attempts"]

    # 本轮 best（值/epoch 恢复，保证“同一后续指标序列触发于相同位置”）
    best = checkpoint.get("best") or {}
    trainer.best_val_acc = float(best.get("val_acc", 0.0))
    trainer.best_epoch = int(best.get("epoch", 0))
    trainer.best_monitor_value = best.get("monitor_value")

    # 早停计数
    es = checkpoint.get("early_stop_state") or {}
    trainer.acc_patience_counter = int(es.get("acc_patience_counter", 0))
    trainer.loss_worse_counter = int(es.get("loss_worse_counter", 0))
    trainer.hist_min_val_loss = es.get("hist_min_val_loss")

    # 全局 RNG（python / numpy / torch / cuda）：T03 预检已通过，失败即抛出
    _restore_rng_state(checkpoint["rng"])

    # S01/T03：训练 loader 的独立生成器（批次序列与 worker 种子的复算前提）
    _restore_loader_rng(trainer.train_loader, checkpoint["loader_rng"])

    # S02：恢复后实际调度器参数必须与断点记录一致
    expected_scheduler = (
        (checkpoint.get("training_protocol") or {}).get("runtime") or {}
    ).get("scheduler_effective")
    if expected_scheduler is not None and trainer.scheduler is not None:
        actual_scheduler = scheduler_effective_dict(trainer.scheduler)
        if actual_scheduler != expected_scheduler:
            raise RuntimeError(
                "恢复后实际调度器参数与断点记录不一致："
                f"{actual_scheduler} vs {expected_scheduler}"
            )

    return {"saved_epoch": saved_epoch}


def _restore_training_state(trainer, checkpoint: dict, *, on_commit=None) -> dict:
    """Commit all live state atomically; unexpected late errors restore every component."""
    model = getattr(trainer.model, "_orig_mod", trainer.model)
    model_state = copy.deepcopy(model.state_dict())
    optimizer_state = copy.deepcopy(trainer.optimizer.state_dict())
    scheduler_state = copy.deepcopy(trainer.scheduler.state_dict()) if trainer.scheduler else None
    scaler_state = copy.deepcopy(trainer.scaler.state_dict()) if trainer.scaler else None
    rng = _capture_rng_state()
    loader_rng = _capture_loader_rng(trainer.train_loader)
    names = (
        "history", "start_epoch", "_accumulated_train_time", "_total_train_time",
        "best_val_acc", "best_epoch", "best_monitor_value", "acc_patience_counter",
        "loss_worse_counter", "hist_min_val_loss", "_optimizer_updates",
        "_optimizer_attempts", "run_meta", "_partial_state",
    )
    saved = {name: copy.deepcopy(getattr(trainer, name)) for name in names}
    meta_path = trainer.run_dir / "run_meta.json"
    meta_bytes = meta_path.read_bytes() if meta_path.exists() else None
    try:
        result = _apply_training_state(trainer, checkpoint)
        if on_commit is not None:
            on_commit()
        return result
    except BaseException:
        model.load_state_dict(model_state)
        trainer.optimizer.load_state_dict(optimizer_state)
        if trainer.scheduler is not None:
            trainer.scheduler.load_state_dict(scheduler_state)
        if trainer.scaler is not None:
            trainer.scaler.load_state_dict(scaler_state)
        for name, value in saved.items():
            setattr(trainer, name, value)
        if meta_bytes is None:
            meta_path.unlink(missing_ok=True)
        elif not meta_path.exists() or meta_path.read_bytes() != meta_bytes:
            fd, tmp = tempfile.mkstemp(dir=meta_path.parent, prefix=".rollback_", suffix=".tmp")
            try:
                with os.fdopen(fd, "wb") as stream:
                    stream.write(meta_bytes)
                os.replace(tmp, meta_path)
            finally:
                if os.path.exists(tmp):
                    os.unlink(tmp)
        _restore_rng_state(rng)
        _restore_loader_rng(trainer.train_loader, loader_rng)
        raise


def save_checkpoint(trainer, path, *, history: dict | None = None, partial: bool = False):
    """
    保存完整训练状态（原子写入）。

    Args:
        trainer: Trainer 实例
        path: 保存路径
        history: 训练历史；None 则使用 trainer.history
        partial: True 表示 epoch 中途的中断保存（权重含未完成 epoch 的部分更新）

    Returns:
        path: 保存的文件路径
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    hist = history if history is not None else trainer.history
    # 以 history 长度作为已完成轮数的唯一来源
    epoch_num = len(hist.get("val_acc", []))

    checkpoint = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "run_id": getattr(trainer, "run_id", ""),
        "model_name": trainer.model_name,
        "model_spec": trainer.model_spec.to_dict(),
        # R02：训练协议快照（恢复前逐项比对，防止换配置续写原 run）
        "training_protocol": trainer.get_training_protocol(),
        # torch.compile 的 OptimizedModule 会加 "_orig_mod." 前缀，保存前展开
        "model_state_dict": getattr(trainer.model, "_orig_mod", trainer.model).state_dict(),
        "optimizer_state_dict": trainer.optimizer.state_dict(),
        "optimizer_updates": trainer._optimizer_updates,
        "optimizer_attempts": trainer._optimizer_attempts,
        "training_device": str(trainer.device),
        "rng_cuda_device_count": torch.cuda.device_count(),
        "run_purpose": trainer.run_purpose,
        "frozen_protocol": trainer.frozen_protocol,
        "epoch": epoch_num,
        "val_acc": hist["val_acc"][-1] if hist.get("val_acc") else None,
        "val_loss": hist["val_loss"][-1] if hist.get("val_loss") else None,
        "history": hist,
        "best": {
            "monitor_metric": trainer.monitor_metric,
            "monitor_value": trainer.best_monitor_value,
            "val_acc": trainer.best_val_acc,
            "epoch": trainer.best_epoch,
        },
        "early_stop_state": {
            "acc_patience_counter": trainer.acc_patience_counter,
            "loss_worse_counter": trainer.loss_worse_counter,
            "hist_min_val_loss": trainer.hist_min_val_loss,
        },
        "rng": _capture_rng_state(),
        # S01：loader/sampler 独立生成器状态（批次序列与 worker 种子复算）
        "loader_rng": _capture_loader_rng(trainer.train_loader),
        "training_duration_seconds": getattr(trainer, "_total_train_time", 0.0),
        "class_counts": getattr(trainer, "class_counts", None),
        "partial": partial,
        "timestamp": datetime.now().isoformat(),
    }
    if trainer.scheduler is not None:
        checkpoint["scheduler_state_dict"] = trainer.scheduler.state_dict()
    if getattr(trainer, "scaler", None) is not None:
        checkpoint["scaler_state_dict"] = trainer.scaler.state_dict()

    # 原子写：临时文件 + 替换（保存失败不破坏上一份可用断点）
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=".ckpt_", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            torch.save(checkpoint, f)
        os.replace(tmp_name, path)
    except BaseException:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)
        raise

    return path


def load_checkpoint(trainer, checkpoint_path) -> dict:
    """
    从 checkpoint 恢复完整训练状态（F09 / S01 / S02）。

    校验顺序（全部通过后才加载权重/状态；失败不触碰原 run 文件）：
      格式版本 → [T01: partial 断点自动回滚到同 run 完整 last] →
      model_name/model_spec → 训练协议（完整生效配置 + 数据管线资格）→
      [T03: 状态完整性预检（RNG/生成器/调度器/scaler/模型与优化器结构，
      只读 dry-run，缺项或非法值在修改真实状态前拒绝）]；
      随后恢复模型/优化器/调度器/scaler/history/best/早停/RNG/loader 生成器，
      并校验恢复后实际调度器参数与断点记录一致。

    T01：显式加载 partial（epoch 中途）断点时：若同 run 存在完整 last.pth
    （同 run_id、非 partial）→ 自动回滚到该完整断点并在恢复事件中记录；
    无完整 last（或 last 亦为 partial / 不同 run）→ 拒绝。

    精确恢复支持范围（S01 实测）：
      - workers=0；或 workers>=1 且 persistent_workers=False（独立 generator 复算）
      - persistent_workers=True 的断点/恢复端均明确拒绝

    Raises:
        FileNotFoundError: 文件不存在
        RuntimeError: 文件损坏 / 旧格式 / 规格或协议不一致 / 不满足精确恢复条件
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint 未找到: {checkpoint_path}")

    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except Exception as e:
        raise RuntimeError(f"断点文件损坏或无法读取: {checkpoint_path}\n{e}") from e

    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise RuntimeError(f"文件不是有效的 checkpoint（缺少 model_state_dict）: {checkpoint_path}")

    fmt = checkpoint.get("format_version")
    if fmt != CHECKPOINT_FORMAT_VERSION:
        raise RuntimeError(
            f"断点格式版本 {fmt!r} 不受支持（当前支持 {CHECKPOINT_FORMAT_VERSION}）。\n"
            "旧格式断点缺少 RNG / scaler / 早停状态，无法精确续训（不静默降级）。\n"
            f"如确需使用: {checkpoint_path}，请先迁移或从头训练。"
        )

    # ---- T01：显式加载 partial（epoch 中途）断点 → 自动回滚到同 run 完整 last ----
    fallback_note = None
    if checkpoint.get("partial"):
        requested_path = checkpoint_path
        last_path = checkpoint_path.parent / "last.pth"
        if not last_path.exists():
            raise RuntimeError(
                "该断点为 partial（epoch 中途保存，权重含未完成 epoch 的部分更新）：\n"
                "  精确恢复被拒绝；且同 run 未找到完整断点 last.pth。\n"
                f"  请求的断点: {requested_path}\n"
                "请从头训练，或改用完整 epoch 边界的断点（last.pth / epoch_XXXX.pth）。"
            )
        fallback_ckpt = torch.load(last_path, map_location="cpu", weights_only=False)
        if not isinstance(fallback_ckpt, dict) or "model_state_dict" not in fallback_ckpt:
            raise RuntimeError(f"同 run 的 last.pth 不是有效 checkpoint: {last_path}")
        if fallback_ckpt.get("partial"):
            raise RuntimeError(
                f"同 run 的 last.pth 也是 partial 断点（{last_path}）：无法自动回滚到完整边界。"
            )
        if not checkpoint.get("run_id") or fallback_ckpt.get("run_id") != checkpoint.get("run_id"):
            raise RuntimeError(
                "拒绝 partial 回滚：last.pth 与请求断点不是同一 run "
                f"（{fallback_ckpt.get('run_id')!r} != {checkpoint.get('run_id')!r}）。"
            )
        fallback_note = (
            f"显式加载的 partial 断点（{requested_path.name}，含未完成 epoch 的部分更新）"
            f"已自动回滚到同 run 完整断点 {last_path.name}"
        )
        logger.warning("  - %s", fallback_note)
        checkpoint_path = last_path
        checkpoint = fallback_ckpt

    # 模型名与规格校验
    ckpt_model_name = checkpoint.get("model_name", "")
    if ckpt_model_name and ckpt_model_name != trainer.model_name:
        raise RuntimeError(
            f"Checkpoint 模型名不匹配: checkpoint='{ckpt_model_name}', "
            f"当前='{trainer.model_name}'。请确认 checkpoint 路径正确。"
        )
    ckpt_spec = checkpoint.get("model_spec")
    if ckpt_spec is None:
        raise RuntimeError(f"Checkpoint 缺少 model_spec 字段: {checkpoint_path}")
    current_spec = trainer.model_spec.to_dict()
    if ckpt_spec != current_spec:
        raise RuntimeError(
            "Checkpoint 的 model_spec 与当前训练配置不一致，拒绝续训：\n"
            f"  checkpoint: {ckpt_spec}\n"
            f"  当前配置 : {current_spec}\n"
            "请使用与断点一致的数据/模型配置，或从头训练。"
        )

    # ---- S02：训练协议校验（完整生效配置 + 运行时有效值 + 数据管线 + 指纹）----
    ckpt_protocol = checkpoint.get("training_protocol")
    if ckpt_protocol is None:
        raise RuntimeError(
            "该断点缺少 training_protocol 记录（早于协议比对引入的版本）：\n"
            "无法验证配置/数据/数据管线一致性，精确恢复被拒绝（不冒称精确）。\n"
            "如需继续训练，请以当前配置从头新建 run。"
        )
    protocol_version = ckpt_protocol.get("protocol_version")
    if protocol_version != TRAINING_PROTOCOL_VERSION:
        raise RuntimeError(
            f"断点训练协议版本 {protocol_version!r} 不受支持"
            f"（当前 {TRAINING_PROTOCOL_VERSION}）：\n"
            "旧版本协议缺少完整字段（loader/增强/调度器参数等），精确恢复被拒绝。"
        )

    # ---- S01：数据管线资格检查 ----
    src_loader_spec = (ckpt_protocol.get("runtime") or {}).get("loader") or {}
    if src_loader_spec.get("persistent_workers"):
        raise RuntimeError(
            "该断点产生于 persistent_workers=True 的数据管线：worker 内部 RNG 进度"
            "跨会话不可复算，精确恢复不支持。\n"
            "可选：关闭 persistent_workers（或使用 workers=0）后从头训练以获得精确恢复能力。"
        )
    src_workers = int(src_loader_spec.get("num_workers") or 0)
    if src_workers > 0 and not src_loader_spec.get("has_generators"):
        raise RuntimeError(
            f"该断点（num_workers={src_workers}）缺少独立生成器记录：无法复算批次"
            "序列与 worker 种子，精确恢复不支持。"
        )
    resume_loader = trainer.train_loader
    if bool(getattr(resume_loader, "persistent_workers", False)):
        raise RuntimeError(
            "恢复端 train_loader 使用 persistent_workers=True：不支持精确恢复。\n"
            "请关闭 persistent_workers 后重建 DataLoader，或新建 run。"
        )
    resume_workers = int(getattr(resume_loader, "num_workers", 0) or 0)
    if resume_workers > 0 and getattr(resume_loader, "generator", None) is None:
        raise RuntimeError(
            f"恢复端 train_loader（num_workers={resume_workers}）缺少独立 generator："
            "无法复算批次/增强序列，精确恢复不支持。"
        )

    # 完整协议比对（含 loader 规格：num_workers / sampler / batch；恢复端必须与断点一致）
    protocol_diffs = _protocol_diff(ckpt_protocol, trainer.get_training_protocol())
    if protocol_diffs:
        raise RuntimeError(
            "恢复被拒绝：训练协议与断点不一致（改动训练策略/数据管线请新建 run）：\n  - "
            + "\n  - ".join(protocol_diffs)
        )

    if checkpoint.get("run_purpose") == "formal" or trainer.run_purpose == "formal":
        from utils.formal_protocol import validate_formal_plan

        if checkpoint.get("run_purpose") != trainer.run_purpose:
            raise RuntimeError("正式用途不一致，不能将 smoke 与 formal 相互转换续写")
        if checkpoint.get("frozen_protocol") != trainer.frozen_protocol:
            raise RuntimeError("正式断点的冻结协议绑定不一致")
        if checkpoint.get("run_id") != trainer.run_id:
            raise RuntimeError("正式断点不能恢复到其他 run")
        validate_formal_plan(
            trainer.frozen_protocol, trainer.model_name, current_spec,
            trainer.get_training_protocol(),
        )

    # S01：loader 生成器状态必须随断点保存（v2 协议断点必含）
    if checkpoint.get("loader_rng") is None:
        raise RuntimeError(
            f"断点缺少 loader_rng（loader/sampler 生成器状态），无法复算批次序列: "
            f"{checkpoint_path}"
        )

    # ---- T03：状态完整性预检（只读；缺项/非法值在修改任何真实状态前拒绝）----
    state_integrity = _verify_checkpoint_state(trainer, checkpoint)

    # ---- 全部校验通过：恢复训练状态（与 S03 自动回滚共享实现）----
    notes = [fallback_note] if fallback_note else []
    _restore_training_state(
        trainer, checkpoint,
        on_commit=lambda: trainer._record_resume_event(
            checkpoint_path, protocol_verified=True, notes=notes,
            state_integrity=state_integrity,
        ),
    )

    logger.info("已从 %s 恢复训练:", checkpoint_path)
    logger.info("  - 上次训练到 epoch %s", checkpoint.get("epoch", "?"))
    best = checkpoint.get("best") or {}
    logger.info("  - 本轮 best: %s=%.4f (epoch %d)",
                best.get("monitor_metric", "val_acc"),
                float(best.get("monitor_value", best.get("val_acc", 0.0)) or 0.0),
                trainer.best_epoch)
    logger.info(
        "  - 早停计数: acc=%d, loss_worse=%d",
        trainer.acc_patience_counter, trainer.loss_worse_counter,
    )
    logger.info("  - 将从 epoch %d 继续训练", trainer.start_epoch)
    # 恢复成功后才写恢复事件（此前任何失败都不会改写原 run 元数据）；
    # T01/T03：协议一致（protocol_verified）与完整状态恢复（state_integrity）分开记录
    return checkpoint


def load_checkpoint_metadata(path: Path) -> dict | None:
    """
    轻量读取 checkpoint 元数据（不加载完整模型权重）。
    对新旧格式都尽力读取；读取失败返回 None。
    """
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        history = ckpt.get("history", {})
        best = ckpt.get("best") or {}
        return {
            "format_version": ckpt.get("format_version", 1),
            "run_id": ckpt.get("run_id", ""),
            "epoch": ckpt.get("epoch", 0),
            "val_acc": ckpt.get("val_acc", 0.0),
            "best_val_acc": best.get("val_acc", ckpt.get("best_val_acc", 0.0)),
            "monitor_metric": best.get("monitor_metric", "val_acc"),
            "monitor_value": best.get("monitor_value"),
            "model_name": ckpt.get("model_name", ""),
            "timestamp": ckpt.get("timestamp", ""),
            "trained_epochs": len(history.get("val_acc", [])) if history else 0,
            "training_duration_seconds": ckpt.get("training_duration_seconds", 0.0),
            "partial": bool(ckpt.get("partial", False)),
        }
    except Exception:
        return None


def cleanup_old_checkpoints(checkpoints_dir: Path, config: dict):
    """
    自动清理最旧的定期断点（epoch_*.pth），保留最新的 max_checkpoint_files 份。
    不影响 last.pth / best.pth / interrupted_*.pth。
    """
    max_files = config["checkpoint"].get("max_checkpoint_files", 5)
    if max_files <= 0:
        return

    epoch_ckpts = sorted(
        Path(checkpoints_dir).glob(_EPOCH_CKPT_GLOB),
        key=lambda p: p.stat().st_mtime,
    )
    while len(epoch_ckpts) > max_files:
        oldest = epoch_ckpts.pop(0)
        oldest.unlink()
        logger.info("已清理旧断点: %s", oldest.name)


# ============================================================
# 续训断点自动发现
# ============================================================
def find_resume_checkpoint(model_name: str):
    """
    返回该模型全部 run 中最新的 last.pth（按修改时间倒序取第一个）。
    只返回新格式 run 断点；旧 training/checkpoints/ 目录产物不可用于精确续训。

    Returns:
        checkpoint 路径，无可用断点时返回 None
    """
    model_runs = RUNS_ROOT / model_name
    if not model_runs.exists():
        return None
    candidates = sorted(
        model_runs.glob("*/checkpoints/last.pth"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


# ============================================================
# ipywidgets Checkpoint 选择弹窗（扫描新格式 runs 目录）
# ============================================================
def _build_checkpoint_label(path: Path, meta: dict | None) -> str:
    """构建单条 checkpoint 的下拉菜单显示标签。"""
    name = path.name
    if name == "last.pth":
        tag = "\u23ea Last(完整)"
    elif name == "best.pth":
        tag = "\u2605 Best(run 最优)"
    elif name.startswith("interrupted_"):
        tag = "\u26a0 中断(partial)"
    elif name.startswith("epoch_"):
        tag = "  定期"
    else:
        tag = "  ?"

    if not meta:
        return f"{tag} | {path.parent.parent.name} | (元数据不可读)"

    time_str = _relative_time(meta.get("timestamp", ""))
    acc_str = f"val_acc={meta['val_acc'] * 100:.2f}%"
    run_id = meta.get("run_id", path.parent.parent.name)
    trained = meta.get("trained_epochs", meta.get("epoch", 0))
    return f"{tag} | {run_id} | {time_str} | {acc_str} | 已训练{trained}轮"


def select_checkpoint_widget(model_name: str):
    """
    扫描 training/runs/<model_name>/*/checkpoints/ 下的可用断点，
    通过 ipywidgets 弹窗选择续训断点；确认后写入 notebook 全局变量 RESUME_FROM。

    注意：只有新格式 run 断点（last.pth / interrupted / 定期）可续训；
    旧 training/checkpoints/ 产物不支持精确续训，不在此列出。
    """
    try:
        import ipywidgets as widgets
        from IPython import get_ipython
        from IPython.display import display
    except ImportError:
        print("ipywidgets 未安装，将从头开始训练。安装方式: pip install ipywidgets")
        return None

    ip = get_ipython()
    model_runs = RUNS_ROOT / model_name
    ip.user_ns["RESUME_FROM"] = None

    entries = []
    if model_runs.exists():
        for path in model_runs.glob("*/checkpoints/*.pth"):
            meta = load_checkpoint_metadata(path)
            if meta is not None and meta.get("format_version") == CHECKPOINT_FORMAT_VERSION:
                entries.append((path, meta, path.stat().st_mtime))

    if not entries:
        print(
            f"未找到 {model_name} 的可续训断点（training/runs/ 下无新格式 run），"
            "将从头开始训练。"
        )
        return None

    entries.sort(key=lambda e: e[2], reverse=True)
    label_entries = [(p, _build_checkpoint_label(p, m), m) for p, m, _ in entries]

    options = ["\u2795 [从头开始训练]"] + [item[1] for item in label_entries]
    dropdown = widgets.Dropdown(
        options=options,
        value=options[0],
        layout=widgets.Layout(width="760px"),
    )
    confirm_btn = widgets.Button(
        description="确认选择",
        button_style="primary",
        layout=widgets.Layout(width="100px", margin="4px 0 0 0"),
    )
    result_box = widgets.Output(layout=widgets.Layout(margin="4px 0 0 0"))

    def on_confirm(_btn):
        selected = dropdown.value
        with result_box:
            result_box.clear_output()
            if selected.startswith("\u2795"):
                ip.user_ns["RESUME_FROM"] = None
                print(">>> 从头开始训练")
            else:
                for path, label, meta in label_entries:
                    if label == selected:
                        ip.user_ns["RESUME_FROM"] = path
                        print(f">>> 恢复训练: {path}")
                        print(f"    run={meta.get('run_id')} epoch={meta.get('epoch')} "
                              f"val_acc={meta['val_acc']:.4f} partial={meta.get('partial')}")
                        break

    confirm_btn.on_click(on_confirm)
    display(widgets.VBox([
        widgets.HTML("<b>选择 Checkpoint（新格式 runs）</b> "
                     "<span style='font-weight:normal; font-size:12px; color:#666;'>"
                     "点击「确认选择」生效</span>"),
        widgets.HBox([dropdown, confirm_btn]),
        result_box,
    ]))

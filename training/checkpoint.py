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

checkpoint 格式版本: 2。旧格式（1，training/checkpoints/ 下历史产物）不包含
RNG / scaler / 早停状态，不支持精确续训，load 时明确报错（不静默降级）。

用法:
    from training.checkpoint import save_checkpoint, load_checkpoint
    save_checkpoint(trainer, run_dir / "checkpoints" / "last.pth")
    load_checkpoint(trainer, "training/runs/mini_cnn/<run_id>/checkpoints/last.pth")
"""

__all__ = [
    "CHECKPOINT_FORMAT_VERSION",
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

# 项目根目录（直接计算，避免循环依赖）
PROJECT_ROOT = Path(__file__).resolve().parent.parent

logger = logging.getLogger("checkpoint")

CHECKPOINT_FORMAT_VERSION = 2

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


def _restore_rng_state(state: dict) -> list[str]:
    """恢复 RNG 状态；返回未能恢复的项列表。"""
    failed = []
    try:
        pyrandom.setstate(state["python"])
        np.random.set_state(state["numpy"])
        torch.set_rng_state(state["torch"])
        if torch.cuda.is_available() and "torch_cuda" in state:
            torch.cuda.set_rng_state_all(state["torch_cuda"])
    except Exception as e:
        failed.append(str(e))
        logger.warning("RNG 状态恢复失败: %s", e)
    return failed


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
        # torch.compile 的 OptimizedModule 会加 "_orig_mod." 前缀，保存前展开
        "model_state_dict": getattr(trainer.model, "_orig_mod", trainer.model).state_dict(),
        "optimizer_state_dict": trainer.optimizer.state_dict(),
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
    从 checkpoint 恢复完整训练状态（F09）。

    恢复内容：模型权重、优化器、调度器、AMP scaler、RNG、history、
    本轮 best（值/epoch）、早停计数、累计训练时间。

    Raises:
        FileNotFoundError: 文件不存在
        RuntimeError: 文件损坏 / 旧格式 / 模型名或规格不匹配
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
        logger.warning("  - 断点无调度器状态（残留旧格式？），调度器保持初始化状态")

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

    # 本轮 best（值/epoch 恢复，保证"同一后续指标序列触发于相同位置"）
    best = checkpoint.get("best") or {}
    trainer.best_val_acc = float(best.get("val_acc", 0.0))
    trainer.best_epoch = int(best.get("epoch", 0))
    trainer.best_monitor_value = best.get("monitor_value")

    # 早停计数
    es = checkpoint.get("early_stop_state") or {}
    trainer.acc_patience_counter = int(es.get("acc_patience_counter", 0))
    trainer.loss_worse_counter = int(es.get("loss_worse_counter", 0))
    trainer.hist_min_val_loss = es.get("hist_min_val_loss")

    # RNG（批次顺序一致性）
    rng_failed = []
    if "rng" in checkpoint:
        rng_failed = _restore_rng_state(checkpoint["rng"])
    else:
        rng_failed = ["断点缺少 RNG 状态"]

    # partial 标注
    if checkpoint.get("partial"):
        logger.warning(
            "  - 该断点为 epoch 中途的 partial 保存：权重含未完成 epoch 的部分更新，"
            "history 从最近完整 epoch(=%d) 继续；如需严格一致请从 last.pth 恢复",
            saved_epoch,
        )

    logger.info("已从 %s 恢复训练:", checkpoint_path)
    logger.info("  - 上次训练到 epoch %s", checkpoint.get("epoch", "?"))
    logger.info("  - 本轮 best: %s=%.4f (epoch %d)",
                best.get("monitor_metric", "val_acc"),
                float(best.get("monitor_value", best.get("val_acc", 0.0)) or 0.0),
                trainer.best_epoch)
    logger.info(
        "  - 早停计数: acc=%d, loss_worse=%d",
        trainer.acc_patience_counter, trainer.loss_worse_counter,
    )
    logger.info("  - 将从 epoch %d 继续训练", trainer.start_epoch)
    if rng_failed:
        logger.warning("  - RNG 恢复不完整: %s（批次顺序可能不一致）", rng_failed)

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

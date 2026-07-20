"""
Checkpoint 管理模块 — 从 trainer.py 拆分

职责:
  - 保存/加载 checkpoint（模型权重 + 优化器状态 + 训练历史）
  - 轻量元数据读取（不加载完整权重）
  - 旧 checkpoint 自动清理
  - Checkpoint 选择 UI 组件（ipywidgets）

用法:
    from training.checkpoint import save_checkpoint, load_checkpoint
    load_checkpoint(trainer, "path/to/checkpoint.pth")
"""

import json
import copy
import logging
from pathlib import Path
from datetime import datetime

import torch

# 项目根目录（直接计算，避免循环依赖）
PROJECT_ROOT = Path(__file__).resolve().parent.parent

logger = logging.getLogger("checkpoint")


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


def _progress_bar(current: int, total: int, width: int = 10) -> str:
    """生成文本进度条 (如 ████░░░░░░)"""
    if total <= 0:
        return ""
    ratio = min(current / total, 1.0)
    filled = int(width * ratio)
    return "\u2588" * filled + "\u2591" * (width - filled)


# ============================================================
# Checkpoint 核心操作
# ============================================================

def save_checkpoint(trainer, path: Path, is_best: bool = False, history: dict = None):
    """
    保存完整训练状态

    Args:
        trainer: Trainer 实例（提供 model, optimizer, history 等属性）
        path: 保存路径
        is_best: 是否为最佳模型（触发到推理目录的自动导出）
        history: 训练历史，None 则使用 trainer.history

    Returns:
        path: 保存的文件路径
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    hist = history or trainer.history
    # 以 history 长度作为已完成轮数的唯一来源
    epoch_num = len(hist.get("val_acc", []))

    checkpoint = {
        "epoch": epoch_num,
        "model_state_dict": trainer.model.state_dict(),
        "optimizer_state_dict": trainer.optimizer.state_dict(),
        "val_acc": hist["val_acc"][-1] if hist["val_acc"] else 0.0,
        "val_loss": hist["val_loss"][-1] if hist["val_loss"] else 0.0,
        "best_val_acc": trainer.best_val_acc,
        "global_best_val_acc": trainer.global_best_val_acc,
        "history": hist,
        "config": trainer.model_config,
        "model_name": trainer.model_name,
        "timestamp": datetime.now().isoformat(),
        "training_duration_seconds": getattr(trainer, "_total_train_time", 0.0),
    }
    # 保存调度器状态（修复断点恢复时 LR 周期重置问题）
    if trainer.scheduler is not None:
        checkpoint["scheduler_state_dict"] = trainer.scheduler.state_dict()
    torch.save(checkpoint, path)

    # P0-1: 最佳模型自动导出到推理目录
    if is_best:
        import shutil
        inference_dir = PROJECT_ROOT / "inference" / "saved_models"
        inference_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(path, inference_dir / f"{trainer.model_name}_best.pth")

    return path


def load_checkpoint(trainer, checkpoint_path: str) -> dict:
    """
    从 checkpoint 恢复训练状态

    Args:
        trainer: Trainer 实例（提供 model, optimizer, model_name 等属性）
        checkpoint_path: checkpoint 文件路径

    Returns:
        checkpoint 信息字典

    Raises:
        FileNotFoundError: 文件不存在
        RuntimeError: checkpoint 不兼容
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint未找到: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=trainer.device, weights_only=False)

    # 版本校验：检查模型名和配置是否匹配
    ckpt_model_name = checkpoint.get("model_name", "")
    if ckpt_model_name and ckpt_model_name != trainer.model_name:
        raise RuntimeError(
            f"Checkpoint 模型名不匹配: checkpoint='{ckpt_model_name}', "
            f"当前='{trainer.model_name}'。请确认 checkpoint 路径正确。"
        )

    # 恢复模型权重
    try:
        trainer.model.load_state_dict(checkpoint["model_state_dict"])
    except RuntimeError as e:
        raise RuntimeError(
            f"模型结构不匹配，无法加载 checkpoint: {e}\n"
            f"请确认 checkpoint 是用相同的模型架构训练的。"
        ) from e

    # 恢复优化器状态
    if "optimizer_state_dict" in checkpoint:
        trainer.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    # 恢复调度器状态（修复断点恢复时 LR 周期重置问题）
    # 兼容旧 checkpoint（无 scheduler_state_dict 字段）：仅在字段存在且调度器可用时恢复
    if "scheduler_state_dict" in checkpoint and trainer.scheduler is not None:
        trainer.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        logger.info("  - 调度器状态已恢复（LR 周期连续）")

    # 恢复训练历史（必须在计算 start_epoch 之前）
    if "history" in checkpoint:
        trainer.history = checkpoint["history"]

    # 恢复训练进度
    history_epochs = len(trainer.history.get("val_acc", []))
    saved_epoch = checkpoint.get("epoch", 0)
    if saved_epoch != history_epochs and history_epochs > 0:
        logger.warning(
            "Checkpoint epoch (%d) 与 history 长度 (%d) 不一致，使用 history 长度",
            saved_epoch, history_epochs,
        )
        saved_epoch = history_epochs
    trainer.start_epoch = saved_epoch + 1

    # 恢复累积训练时间
    trainer._accumulated_train_time = checkpoint.get("training_duration_seconds", 0.0)

    # 恢复最优验证准确率
    # ⚠️ best_val_acc 不恢复：让新 session 从 0 开始累计"本轮最佳"
    #    global_best_val_acc 恢复：它是跨会话的"全局最佳"记录
    trainer.best_val_acc = 0.0
    trainer.best_model_state = None
    trainer.global_best_val_acc = 0.0
    if "global_best_val_acc" in checkpoint:
        trainer.global_best_val_acc = checkpoint["global_best_val_acc"]

    logger.info("已从 %s 恢复训练:", checkpoint_path)
    logger.info("  - 上次训练到 epoch %s", checkpoint.get('epoch', '?'))
    logger.info("  - 全局最佳 val_acc: %.4f (跨会话)", trainer.global_best_val_acc)
    logger.info("  - 本轮 session 最佳 val_acc 从 0 开始累计")
    logger.info("  - 将从 epoch %d 继续训练", trainer.start_epoch)

    return checkpoint


def load_checkpoint_metadata(path: Path) -> dict | None:
    """
    轻量读取 checkpoint 元数据（不加载完整模型权重）。
    无需 GPU，适合在 Jupyter widget 或文件扫描场景中使用。

    Args:
        path: checkpoint 文件路径

    Returns:
        包含 epoch / val_acc / best_val_acc / model_name / timestamp /
        trained_epochs / training_duration_seconds 的字典，
        读取失败时返回 None
    """
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        history = ckpt.get("history", {})
        return {
            "epoch": ckpt.get("epoch", 0),
            "val_acc": ckpt.get("val_acc", 0.0),
            "best_val_acc": ckpt.get("best_val_acc", 0.0),
            "global_best_val_acc": ckpt.get("global_best_val_acc", 0.0),
            "model_name": ckpt.get("model_name", ""),
            "timestamp": ckpt.get("timestamp", ""),
            "trained_epochs": len(history.get("val_acc", [])) if history else 0,
            "training_duration_seconds": ckpt.get("training_duration_seconds", 0.0),
        }
    except Exception:
        return None


def cleanup_old_checkpoints(save_dir: Path, config: dict):
    """
    自动清理最旧的定期断点文件，保留最新的 max_checkpoint_files 份。
    仅删除 checkpoint_epoch*.pth，不影响 best_model.pth 和 interrupted_*.pth。
    """
    max_files = config["checkpoint"].get("max_checkpoint_files", 5)
    if max_files <= 0:
        return

    epoch_ckpts = sorted(
        save_dir.glob("checkpoint_epoch*.pth"),
        key=lambda p: p.stat().st_mtime,
    )

    while len(epoch_ckpts) > max_files:
        oldest = epoch_ckpts.pop(0)
        oldest.unlink()
        logger.info("已清理旧断点: %s", oldest.name)


# ============================================================
# Checkpoint 自动检测
# ============================================================
def find_resume_checkpoint(model_name: str, prefer: str = "best"):
    """
    自动检测并返回最适合恢复训练的 checkpoint 路径。

    优先级:
      1. global_best.pth（若有且 prefer='best'）
      2. local_best.pth（若有且 prefer='best'）
      3. 最新的 interrupted_*.pth（若有且 prefer='interrupted'）
      4. 最新的 checkpoint_epoch*.pth（按修改时间）
      5. 旧版兼容: best_model.pth（若有）

    Args:
        model_name: 模型名称
        prefer: 'best' 优先选最佳模型, 'latest' 优先选最新断点

    Returns:
        checkpoint 路径，无可用 checkpoint 时返回 None
    """
    checkpoint_dir = PROJECT_ROOT / "training" / "checkpoints" / model_name
    if not checkpoint_dir.exists():
        return None

    if prefer == "best":
        global_best = checkpoint_dir / "global_best.pth"
        if global_best.exists():
            return global_best
        local_best = checkpoint_dir / "local_best.pth"
        if local_best.exists():
            return local_best
        old_best = checkpoint_dir / "best_model.pth"
        if old_best.exists():
            return old_best

    interrupted = sorted(
        checkpoint_dir.glob("interrupted_*.pth"),
        key=lambda p: p.stat().st_mtime, reverse=True,
    )
    if interrupted:
        return interrupted[0]

    periodic = sorted(
        checkpoint_dir.glob("checkpoint_epoch*.pth"),
        key=lambda p: p.stat().st_mtime, reverse=True,
    )
    if periodic:
        return periodic[0]

    return None


# ============================================================
# ipywidgets Checkpoint 选择弹窗
# ============================================================
def _build_checkpoint_label(path: Path, meta: dict | None, size_mb: float, best_val_acc: float) -> str:
    """构建单条 checkpoint 的下拉菜单显示标签"""
    name = path.name
    if name == "global_best.pth":
        tag = "\u2605 Global Best"
    elif name == "local_best.pth":
        tag = "\u2605 Local Best"
    elif name == "best_model.pth":
        tag = "\u2605 Best"
    elif name.startswith("interrupted_"):
        tag = "\u26a0 中断"
    else:
        tag = "  定期"

    if not meta:
        return f"{tag} | (元数据不可读) | {size_mb:.1f}MB"

    time_str = _relative_time(meta.get("timestamp", ""))
    acc_pct = meta["val_acc"] * 100
    acc_str = f"{acc_pct:.2f}%"

    is_best_tag = name in ("global_best.pth", "local_best.pth", "best_model.pth")
    trained = meta.get("trained_epochs", meta.get("epoch", 0))
    saved_epoch = meta.get("epoch", 0)
    if saved_epoch != trained and trained > 0:
        epoch_str = f"已训练{trained}轮(权重第{saved_epoch}轮)"
    else:
        epoch_str = f"已训练{trained}轮"

    dur = meta.get("training_duration_seconds", 0.0)
    dur_str = _format_duration(dur) if dur > 0 else ""

    delta_str = ""
    if not is_best_tag and best_val_acc > 0 and meta["val_acc"] > 0:
        delta = (meta["val_acc"] - best_val_acc) * 100
        if abs(delta) > 0.01:
            delta_str = f"({delta:+.2f}%)"

    parts = [tag, time_str, acc_str, epoch_str]
    if dur_str:
        parts.append(dur_str)
    if delta_str:
        parts.append(delta_str)
    if not is_best_tag:
        parts.append(f"{size_mb:.1f}MB")

    return " | ".join(parts)


def _build_summary_card(best_entry, model_name: str) -> str:
    """构建最佳模型摘要 HTML 卡片"""
    if not best_entry:
        return '<div style="color:#888; padding:4px 0;">暂无已训练的 checkpoint</div>'

    _, meta = best_entry
    if not meta:
        return f'<div style="color:#888; padding:4px 0;">{best_entry[0].name} (元数据不可读)</div>'

    acc_pct = meta["val_acc"] * 100
    global_best_pct = meta.get("global_best_val_acc", meta["val_acc"]) * 100
    time_str = _relative_time(meta.get("timestamp", ""))
    trained = meta.get("trained_epochs", meta.get("epoch", 0))
    dur = meta.get("training_duration_seconds", 0.0)
    dur_str = _format_duration(dur) if dur > 0 else "--"
    source = best_entry[0].name if best_entry else ""

    return (
        f'<div style="background:#f0f8ff; border-left:3px solid #4a90d9; padding:6px 10px; '
        f'border-radius:4px; font-size:13px; margin-bottom:6px;">'
        f'<b>全局最佳:</b> {model_name} | '
        f'val_acc=<b>{global_best_pct:.2f}%</b> | '
        f'已训练 <b>{trained}</b> 轮 | '
        f'用时 {dur_str} | '
        f'{time_str}'
        f'</div>'
    )


def select_checkpoint_widget(model_name: str):
    """
    扫描可用 checkpoint，通过 ipywidgets 弹窗让用户选择是否恢复训练。
    确认后直接将选择结果写入 notebook 全局变量 RESUME_FROM。

    下拉菜单每条显示: 类型 | 相对时间 | 准确率 | 已训练轮数 | 训练时长 | 与best差距
    顶部显示最佳模型摘要卡片。

    Args:
        model_name: 模型名称
    """
    try:
        import ipywidgets as widgets
        from IPython.display import display
        from IPython import get_ipython
    except ImportError:
        print("ipywidgets 未安装，将从头开始训练。安装方式: pip install ipywidgets")
        return None

    ip = get_ipython()
    checkpoint_dir = PROJECT_ROOT / "training" / "checkpoints" / model_name

    best_entry = None
    global_best_val_acc = 0.0
    all_entries = []

    if checkpoint_dir.exists():
        for p in checkpoint_dir.glob("*.pth"):
            meta = load_checkpoint_metadata(p)
            size_mb = p.stat().st_size / (1024 * 1024)
            mtime = p.stat().st_mtime
            entry = (p, meta, size_mb, mtime)

            if p.name == "global_best.pth":
                best_entry = entry
                if meta:
                    global_best_val_acc = meta.get("global_best_val_acc", meta["val_acc"])
            elif p.name == "local_best.pth" or p.name == "best_model.pth":
                all_entries.append(entry)
                if meta and meta["val_acc"] > global_best_val_acc:
                    global_best_val_acc = meta["val_acc"]
            else:
                all_entries.append(entry)

    all_entries.sort(key=lambda e: e[3], reverse=True)

    if not best_entry and not all_entries:
        print(f"未找到 {model_name} 的任何 checkpoint，将从头开始训练。")
        ip.user_ns["RESUME_FROM"] = None
        return None

    best_val_acc = max(
        global_best_val_acc,
        best_entry[1]["val_acc"] if best_entry and best_entry[1] else 0.0,
    )

    label_entries = []
    if best_entry:
        path, meta, size_mb, _ = best_entry
        label = _build_checkpoint_label(path, meta, size_mb, best_val_acc)
        label_entries.append((path, label, meta))

    for path, meta, size_mb, _ in all_entries:
        label = _build_checkpoint_label(path, meta, size_mb, best_val_acc)
        label_entries.append((path, label, meta))

    options = ["\u2795 [从头开始训练]"] + [item[1] for item in label_entries]

    summary_html = _build_summary_card(
        (best_entry[0], best_entry[1]) if best_entry else None,
        model_name,
    )
    summary_html += f'<div style="font-size:12px; color:#666; margin-bottom:4px;">共 {len(label_entries)} 个 checkpoint 可选</div>'

    dropdown = widgets.Dropdown(
        options=options,
        value=options[0],
        description="",
        layout=widgets.Layout(width="680px"),
        style={"description_width": "0px"},
    )

    confirm_btn = widgets.Button(
        description="确认选择",
        button_style="primary",
        layout=widgets.Layout(width="100px", margin="4px 0 0 0"),
    )
    result_box = widgets.Output(layout=widgets.Layout(margin="4px 0 0 0"))

    default_path = None
    ip.user_ns["RESUME_FROM"] = default_path

    def on_confirm(btn):
        btn.disabled = True
        confirm_btn.button_style = "success"
        confirm_btn.description = "已确认"
        with result_box:
            result_box.clear_output()
            selected = dropdown.value
            if selected.startswith("\u2795 [从头开始训练]"):
                ip.user_ns["RESUME_FROM"] = None
                print(">>> 从头开始训练")
            else:
                for path, label, meta in label_entries:
                    if label == selected:
                        ip.user_ns["RESUME_FROM"] = path
                        if meta:
                            trained = meta.get("trained_epochs", meta.get("epoch", 0))
                            dur = _format_duration(meta.get("training_duration_seconds", 0))
                            print(f">>> 恢复训练: {path.name}")
                            print(f"    val_acc={meta['val_acc']:.4f}  "
                                  f"best_val_acc={meta['best_val_acc']:.4f}  "
                                  f"已训练 {trained} 轮  用时 {dur or '--'}")
                        else:
                            print(f">>> 恢复训练: {path.name}")
                        break

    confirm_btn.on_click(on_confirm)

    display(widgets.VBox([
        widgets.HTML(f"<b>选择 Checkpoint</b> <span style='font-weight:normal; font-size:12px; color:#666;'>"
                     f"点击「确认选择」生效</span>"),
        widgets.HTML(summary_html),
        widgets.HBox([dropdown, confirm_btn]),
        result_box,
    ]))

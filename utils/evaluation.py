"""
统一评估入口 — 修复 F10

强制显式传入：checkpoint 路径、数据协议与 split；使用 F01 的统一模型构造
（含旧权重迁移）与与训练一致的预处理（48×48 灰度、x/255、无测试增强）。

- 缺失 / 非法 Usage：评估前明确报错（不支持"整个 CSV 当测试集"的回退）
- 输出：完整指标（accuracy / macro-F1 / balanced accuracy / 各类 recall/support）、
  逐样本行号与预测、概率（可重算 ECE/NLL）、权重与划分指纹
- 验证集用于模型选择；PrivateTest 用于冻结方案后的最终评估

用法:
    from utils.evaluation import evaluate_checkpoint
    result = evaluate_checkpoint("training/runs/.../best.pth", split="PrivateTest")
    # 或命令行: python -c "..."（或通过 Notebook/工具调用）
"""

__all__ = [
    "OFFICIAL_SPLITS",
    "DEFAULT_CSV",
    "load_split_dataframe",
    "evaluate_checkpoint",
    "compute_ece_nll",
    "recompute_metrics_from_predictions",
]

import csv
import hashlib
import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
)

from utils.model_spec import file_sha256, load_model_from_checkpoint

logger = logging.getLogger("evaluation")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CSV = PROJECT_ROOT / "data" / "fer2013.csv"
OFFICIAL_SPLITS = ("Training", "PublicTest", "PrivateTest")


def load_split_dataframe(csv_path, split: str):
    """
    按官方 Usage 读取指定划分。

    Returns:
        (子 DataFrame, 原行号 np.ndarray) —— 行号为 CSV 数据行顺序（0 基）

    Raises:
        ValueError: 缺失 Usage / 未知 Usage 取值 / split 非法或被评估集为空
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"数据文件未找到: {csv_path}")
    if split not in OFFICIAL_SPLITS:
        raise ValueError(f"split 必须是 {OFFICIAL_SPLITS} 之一，得到 {split!r}")

    df = pd.read_csv(csv_path)
    if "Usage" not in df.columns:
        raise ValueError(
            "数据缺少 Usage 列：评估需要明确的划分清单；"
            "不支持把整个 CSV 当作评估集（请提供含官方划分的文件）"
        )
    unknown = sorted(set(df["Usage"].unique()) - set(OFFICIAL_SPLITS))
    if unknown:
        raise ValueError(f"Usage 含未知取值 {unknown}，无法确定划分清单")

    mask = (df["Usage"] == split).to_numpy()
    if not mask.any():
        raise ValueError(f"划分 {split} 为空")
    sub = df[mask]
    return sub, sub.index.to_numpy()


def _predict_probs(
    model, df: pd.DataFrame, batch_size: int, device
) -> tuple[np.ndarray, np.ndarray]:
    """对划分内全部样本推理，返回 (probs, labels)。预处理与训练一致：x/255、无增强。"""
    pixels = np.stack(
        [np.array(s.split(), dtype=np.float32).reshape(48, 48) for s in df["pixels"]]
    )
    pixels /= 255.0
    x_all = torch.from_numpy(pixels).unsqueeze(1)

    probs_list = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(x_all), batch_size):
            logits = model(x_all[i : i + batch_size].to(device))
            probs_list.append(torch.softmax(logits, dim=1).cpu())
    probs = torch.cat(probs_list).numpy()
    labels = df["emotion"].to_numpy()
    return probs, labels


def compute_ece_nll(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> dict:
    """
    计算 ECE 与 NLL（公式明确，可复算）。

    - ECE: 15 个等宽区间 [0,1]，ECE = Σ_b (n_b / N) * |acc_b - conf_b|，
      conf = 样本的预测类别 softmax 概率
    - NLL: -mean(log p_true)

    Returns:
        {"ece": float, "nll": float, "n_bins": int, "bins": [...]}
    """
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels)
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == labels).astype(np.float64)
    n_total = len(labels)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    bin_stats = []
    for b in range(n_bins):
        lo, hi = float(bins[b]), float(bins[b + 1])
        m = (conf >= lo) & (conf < hi) if b < n_bins - 1 else (conf >= lo) & (conf <= hi)
        n_b = int(m.sum())
        if n_b > 0:
            bin_acc = float(correct[m].mean())
            bin_conf = float(conf[m].mean())
            ece += n_b / n_total * abs(bin_acc - bin_conf)
            bin_stats.append({"bin": [lo, hi], "n": n_b, "avg_conf": bin_conf, "avg_acc": bin_acc})

    true_probs = probs[np.arange(n_total), labels]
    nll = float(-np.mean(np.log(np.clip(true_probs, 1e-12, 1.0))))
    return {"ece": float(ece), "nll": nll, "n_bins": n_bins, "bins": bin_stats}


def _metrics_from_arrays(
    probs: np.ndarray, labels: np.ndarray, class_names: list, n_bins: int = 15
) -> dict:
    preds = probs.argmax(axis=1)
    n_classes = len(class_names)

    per_class = {}
    for c, name in enumerate(class_names):
        m = labels == c
        support = int(m.sum())
        recall = float((preds[m] == c).mean()) if support else None
        prec_m = preds == c
        precision = float((labels[prec_m] == c).mean()) if prec_m.sum() else None
        per_class[name] = {"support": support, "recall": recall, "precision": precision}

    cal = compute_ece_nll(probs, labels, n_bins=n_bins)
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "macro_f1": float(f1_score(labels, preds, average="macro")),
        "balanced_accuracy": float(balanced_accuracy_score(labels, preds)),
        "per_class": per_class,
        "confusion_matrix": confusion_matrix(labels, preds, labels=list(range(n_classes))).tolist(),
        "ece": cal["ece"],
        "nll": cal["nll"],
        "ece_bins": cal["bins"],
    }


def evaluate_checkpoint(
    checkpoint_path,
    *,
    split: str,
    csv_path=None,
    device: str | torch.device = "cpu",
    batch_size: int = 64,
    output_dir=None,
    spec_override: dict | None = None,
) -> dict:
    """
    统一评估入口：指定 checkpoint + 数据协议与 split → 完整结果（并落盘）。

    Args:
        checkpoint_path: 权重路径（必填；显式指定，无内存状态依赖）
        split: "Training" / "PublicTest" / "PrivateTest"
        csv_path: 数据集路径（默认 data/fer2013.csv）
        device: 计算设备（默认 CPU float32，与历史评估口径一致）
        batch_size: 推理批大小
        output_dir: 结果目录（默认 analysis/evaluations/<时间>_<模型>_<split>/）
        spec_override: 旧权重迁移的显式覆盖（见 utils/model_spec.py）

    Returns:
        结果 dict（含 metrics 与逐样本 predictions）
    """
    csv_path = Path(csv_path) if csv_path is not None else DEFAULT_CSV
    device = torch.device(device)

    # 先做数据校验（缺失/非法 Usage 在加载模型前报错）
    sub, row_indices = load_split_dataframe(csv_path, split)

    model, spec, meta = load_model_from_checkpoint(
        checkpoint_path, device=device, spec_override=spec_override
    )
    probs, labels = _predict_probs(model, sub, batch_size, device)
    metrics = _metrics_from_arrays(probs, labels, list(spec.class_names), n_bins=15)

    result = {
        "evaluated_at": datetime.now().isoformat(),
        "checkpoint": {
            "path": meta["checkpoint_path"],
            "sha256": meta["checkpoint_sha256"],
            "model_spec": meta["model_spec"],
            "spec_notes": meta["spec_notes"],
            "epoch": meta.get("epoch"),
            "val_acc_recorded": meta.get("val_acc"),
        },
        "protocol": {
            "split": split,
            "csv_path": str(csv_path),
            "csv_sha256": file_sha256(csv_path),
            "n_samples": int(len(sub)),
            "row_index_sha256": hashlib.sha256(row_indices.astype(np.int64).tobytes()).hexdigest(),
            "row_index_first": int(row_indices[0]),
            "row_index_last": int(row_indices[-1]),
        },
        "preprocessing": {
            "normalize": spec.normalize,
            "image_size": spec.image_size,
            "augmentation": "none",
            "batch_size": batch_size,
            "device": str(device),
            "dtype": "float32",
        },
        "metrics": metrics,
        "predictions": {
            "row_indices": [int(i) for i in row_indices],
            "labels": labels.tolist(),
            "preds": probs.argmax(axis=1).tolist(),
            "probs": [[round(float(p), 6) for p in row] for row in probs],
            "confidences": [round(float(c), 6) for c in probs.max(axis=1)],
        },
    }

    # 落盘
    out_dir = Path(output_dir) if output_dir is not None else (
        PROJECT_ROOT / "analysis" / "evaluations"
        / f"{datetime.now():%Y%m%d_%H%M%S}_{spec.model_name}_{split}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {k: v for k, v in result.items() if k != "predictions"}
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    header = ["row_index", "label", "pred", "confidence"] + [f"p_{n}" for n in spec.class_names]
    confidences = probs.max(axis=1)
    preds_for_csv = probs.argmax(axis=1)
    with open(out_dir / "predictions.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for i in range(len(labels)):
            writer.writerow(
                [int(row_indices[i]), int(labels[i]), int(preds_for_csv[i]),
                 round(float(confidences[i]), 6)]
                + [round(float(p), 6) for p in probs[i]]
            )
    result["output_dir"] = str(out_dir)

    m = metrics
    print(f"[评估] {spec.model_name} | {split} | n={len(sub)}")
    print(f"  checkpoint: {meta['checkpoint_path']} (sha256={meta['checkpoint_sha256'][:16]}...)")
    print(f"  activation={spec.activation} use_se={spec.use_se} dropout={spec.dropout}")
    print(f"  accuracy={m['accuracy']:.4f} | macro_f1={m['macro_f1']:.4f} | "
          f"balanced_acc={m['balanced_accuracy']:.4f}")
    print(f"  ECE(15bin)={m['ece']:.4f} | NLL={m['nll']:.4f}")
    print(f"  结果目录: {out_dir}")
    return result


def recompute_metrics_from_predictions(predictions_csv, class_names: list | None = None) -> dict:
    """
    从保存的 predictions.csv 重算指标（验证"指标可从保存的预测与标签重新计算"）。
    """
    df = pd.read_csv(predictions_csv)
    labels = df["label"].to_numpy()
    probs = df[[c for c in df.columns if c.startswith("p_")]].to_numpy()
    if class_names is None:
        class_names = [c[2:] for c in df.columns if c.startswith("p_")]
    return _metrics_from_arrays(probs, labels, class_names, n_bins=15)

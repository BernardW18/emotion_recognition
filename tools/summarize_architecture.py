"""Evaluate a complete architecture protocol, fix Public candidate, then report PrivateTest."""

import argparse
import csv
import json
import statistics
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
import numpy as np
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.benchmark_architecture import benchmark_architectures
from training.checkpoint import write_json_atomic
from utils.architecture_comparison import (
    ARCHITECTURES,
    collect_architecture_runs,
    select_architecture,
    summarize_architecture,
)
from utils.constants import CLASS_NAMES
from utils.evaluation import evaluate_checkpoint
from utils.formal_protocol import load_frozen_protocol
from utils.model_spec import file_sha256
from utils.stdio import ensure_utf8_stdio


def evaluate_once(row, split, record, output):
    checkpoint = Path(row["run_dir"]) / "checkpoints/best.pth"
    directory = output / "evaluations" / row["arm"] / str(row["seed"]) / split
    summary = directory / "summary.json"
    predictions = directory / "predictions.csv"
    if summary.exists() or predictions.exists():
        if not summary.exists() or not predictions.exists():
            raise ValueError(f"不完整评估需先核验，不覆盖原始记录: {directory}")
        result = json.loads(summary.read_text(encoding="utf-8"))
    else:
        result = evaluate_checkpoint(
            checkpoint, split=split, device="cpu", batch_size=64, output_dir=directory
        )
    data_sha = record["manifest"]["plans"][0]["training_protocol"]["data_fingerprint"]["csv_sha256"]
    if (
        result["checkpoint"]["sha256"] != file_sha256(checkpoint)
        or result["protocol"]["csv_sha256"] != data_sha
        or result["protocol"]["split"] != split
        or result["preprocessing"]["device"] != "cpu"
        or result["preprocessing"]["dtype"] != "float32"
        or result["preprocessing"]["batch_size"] != 64
    ):
        raise ValueError("评估权重/划分/数据/精度不一致")
    with predictions.open(encoding="utf-8", newline="") as handle:
        saved = list(csv.DictReader(handle))
    y = np.asarray([int(r["label"]) for r in saved])
    pred = np.asarray([int(r["pred"]) for r in saved])
    if len(saved) != result["protocol"]["n_samples"]:
        raise ValueError("保存预测不完整")
    recomputed = {
        "accuracy": accuracy_score(y, pred),
        "macro_f1": f1_score(y, pred, average="macro"),
        "balanced_accuracy": balanced_accuracy_score(y, pred),
    }
    if any(abs(v - result["metrics"][k]) > 1e-12 for k, v in recomputed.items()):
        raise ValueError("保存的标签/预测不能复算主要指标")
    for index, name in enumerate(CLASS_NAMES):
        support = int((y == index).sum())
        recall = float(((y == index) & (pred == index)).sum() / support)
        expected = result["metrics"]["per_class"][name]
        if support != expected["support"] or abs(recall - expected["recall"]) > 1e-12:
            raise ValueError("七类支持数/召回不能从保存预测复算")
    return {
        **row,
        "split": split,
        "metrics": result["metrics"],
        "checkpoint_sha256": result["checkpoint"]["sha256"],
        "predictions_sha256": file_sha256(predictions),
        "summary_sha256": file_sha256(summary),
        "output_dir": str(directory),
    }


def write_figures(output, summaries, costs, rows):
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(ARCHITECTURES)
    public = summaries["PublicTest"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    x = np.arange(7)
    for i, arm in enumerate(names):
        recalls = public[arm]["per_class"]
        axes[0].errorbar(
            x + i * 0.04,
            [recalls[n]["mean"] * 100 for n in CLASS_NAMES],
            yerr=[recalls[n]["std"] * 100 for n in CLASS_NAMES],
            marker="o",
            label=arm,
            capsize=3,
        )
        f1 = public[arm]["metrics"]["macro_f1"]
        axes[1].errorbar(
            costs[arm]["cpu_p95_median_ms"],
            f1["mean"] * 100,
            yerr=f1["std"] * 100,
            fmt="o",
            capsize=3,
        )
        axes[1].annotate(arm, (costs[arm]["cpu_p95_median_ms"], f1["mean"] * 100))
    axes[0].set(xticks=x, xticklabels=CLASS_NAMES, ylabel="PublicTest recall (%)")
    axes[0].tick_params(axis="x", rotation=30)
    axes[0].legend()
    axes[1].set(xlabel="CPU batch1 P95 median of 3 rounds (ms)", ylabel="PublicTest macro-F1 (%)")
    fig.suptitle("90 epochs CE; 3 seeds; mean +/- sample SD (descriptive)")
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"architecture_comparison.{suffix}", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    for arm, ax in zip(names, axes.flat, strict=True):
        histories = [
            json.loads((Path(r["run_dir"]) / "history.json").read_text("utf-8"))
            for r in rows
            if r["arm"] == arm and r["split"] == "PublicTest"
        ]
        for key, label in (("train_acc", "augmented Training"), ("val_acc", "PublicTest (AMP)")):
            values = np.asarray([h[key] for h in histories]) * 100
            mean = values.mean(axis=0)
            std = values.std(axis=0, ddof=1)
            ax.plot(np.arange(1, 91), mean, label=label)
            ax.fill_between(np.arange(1, 91), mean - std, mean + std, alpha=0.15)
        ax.set(title=arm, xlabel="epoch", ylabel="accuracy (%)")
        ax.legend()
    fig.suptitle("Training trajectories: three seeds, mean +/- sample SD")
    fig.savefig(output / "training_curves.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 4, figsize=(16, 4), constrained_layout=True)
    for arm, ax in zip(names, axes, strict=True):
        matrices = [
            r["metrics"]["confusion_matrix"]
            for r in rows
            if r["arm"] == arm and r["split"] == "PublicTest"
        ]
        matrix = np.asarray(matrices).mean(axis=0)
        matrix = matrix / matrix.sum(axis=1, keepdims=True)
        ax.imshow(matrix, vmin=0, vmax=1, cmap="Blues")
        ax.set(
            title=arm,
            xticks=range(7),
            yticks=range(7),
            xticklabels=CLASS_NAMES,
            yticklabels=CLASS_NAMES,
        )
        ax.tick_params(axis="x", rotation=90)
    fig.suptitle("PublicTest mean row-normalized confusion; 3 seeds")
    fig.savefig(output / "public_confusion_matrices.png", dpi=160)
    plt.close(fig)


def finalize(record, run_dirs, output):
    output = Path(output)
    rows = collect_architecture_runs(record, run_dirs)
    output.mkdir(parents=True, exist_ok=True)
    cost_path = output / "efficiency_raw.json"
    if cost_path.exists():
        cost = json.loads(cost_path.read_text("utf-8"))
        if cost.get("protocol_sha256") != record["file_sha256"]:
            raise ValueError("效率基准与冻结协议不匹配")
    else:
        torch.manual_seed(42)
        cost = benchmark_architectures(record)
        cost["protocol_sha256"] = record["file_sha256"]
        write_json_atomic(cost_path, cost)
    evaluations = []
    summaries = {}
    for split in ("Training", "PublicTest"):
        split_rows = [evaluate_once(r, split, record, output) for r in rows]
        summaries[split] = summarize_architecture(split_rows, split)
        evaluations.extend(split_rows)
    public = [r for r in evaluations if r["split"] == "PublicTest"]
    selection = select_architecture(public, cost["aggregate"])
    selection.update(
        protocol_sha256=record["file_sha256"],
        efficiency_raw_sha256=file_sha256(cost_path),
        public_prediction_sha256=[r["predictions_sha256"] for r in public],
    )
    selected_path = output / "candidate_selection.json"
    if selected_path.exists():
        fixed = json.loads(selected_path.read_text("utf-8"))
        if {k: v for k, v in fixed.items() if k != "fixed_at"} != selection:
            raise ValueError("已固定的Public候选与当前结果不符，禁止在Private之后改选")
        selection = fixed
    else:
        selection["fixed_at"] = datetime.now().isoformat()
        write_json_atomic(selected_path, selection)
    selection_sha = file_sha256(selected_path)
    print(f"ARCH candidate fixed before PrivateTest: {selection['candidate']}", flush=True)
    private = [evaluate_once(r, "PrivateTest", record, output) for r in rows]
    assert file_sha256(selected_path) == selection_sha
    summaries["PrivateTest"] = summarize_architecture(private, "PrivateTest")
    evaluations.extend(private)
    raw = {
        "protocol_id": record["protocol_id"],
        "protocol_sha256": record["file_sha256"],
        "runs": rows,
        "evaluations": evaluations,
        "summaries": summaries,
        "selection": selection,
        "candidate_selection_sha256": selection_sha,
    }
    write_json_atomic(output / "results.json", raw)
    duration = {}
    for arm in ARCHITECTURES:
        group = [r for r in rows if r["arm"] == arm]
        times = [r["training_duration_seconds"] for r in group]
        duration[arm] = {
            "duration_seconds_mean": statistics.mean(times),
            "duration_seconds_std": statistics.stdev(times),
            "duration_seconds_total": sum(times),
            "optimizer_attempts_total": sum(r["optimizer_attempts"] for r in group),
            "optimizer_updates_total": sum(r["optimizer_updates"] for r in group),
            "resume_events_total": sum(r["resume_events"] for r in group),
        }
    aggregate = {
        "schema_version": 1,
        "protocol_id": record["protocol_id"],
        "protocol_sha256": record["file_sha256"],
        "code_sha256": record["manifest"]["code_sha256"],
        "source_git_commit": record["manifest"]["git_commit"],
        "source_results_sha256": file_sha256(output / "results.json"),
        "completed_at": datetime.now().isoformat(),
        "run_count": 12,
        "epochs_per_run": 90,
        "sample_std_ddof": 1,
        "evaluation": {"device": "cpu", "dtype": "float32", "batch_size": 64},
        "summaries": summaries,
        "selection": {k: v for k, v in selection.items() if k != "public_prediction_sha256"},
        "candidate_selection_sha256": selection_sha,
        "efficiency": cost["aggregate"],
        "efficiency_method": cost["method"],
        "preprocessing": {k: v for k, v in cost["preprocessing"].items() if k != "raw_ms"},
        "training": duration,
        "limitations": [
            "n=3 descriptive, not statistical significance",
            "official cross-split pixel duplicates remain",
            "PrivateTest previously used; never used to change candidate",
            "GPU float32 timing records TF32 backend flags",
        ],
    }
    write_json_atomic(output / "aggregate_metrics.json", aggregate)
    text = [
        "# MicroResNet结构对照结论",
        "",
        f"协议 `{record['protocol_id']}`：12/12 run完整90轮，CE，seeds42/43/44。",
        "三seed均值±样本标准差；百分数，std为百分点。",
        "",
    ]
    for split in ("PublicTest", "PrivateTest"):
        text += [
            f"## {split}",
            "",
            "| 臂 | accuracy | macro-F1 | balanced accuracy | Disgust recall |",
            "|---|---:|---:|---:|---:|",
        ]
        for arm, stats in summaries[split].items():
            values = [stats["metrics"][k] for k in ("accuracy", "macro_f1", "balanced_accuracy")]
            values.append(stats["per_class"]["Disgust"])
            text.append(
                f"| {arm} | "
                + " | ".join(f"{v['mean'] * 100:.2f} ± {v['std'] * 100:.2f}" for v in values)
                + " |"
            )
        text.append("")
    text += [
        "## 预先固定门槛与成本",
        "",
        "质量：Public宏F1至少+1个百分点、accuracy下降≤0.5个百分点、3/3配对seed均改善。",
        "部署：参数≤1.6M、batch128无OOM、CPU P95≤同轮S0的1.5倍。质量与部署分别判定。",
        "",
        "| 臂 | 宏F1差值(pp) | 三个seed差值(pp) | 质量通过 | 部署通过 | 参数 | CPU P95(ms) |",
        "|---|---:|---|---|---|---:|---:|",
    ]
    for arm, gate in selection["gates"].items():
        paired = ", ".join(f"{v * 100:+.2f}" for v in gate["paired_macro_f1_deltas"])
        quality = "是" if gate["quality_pass"] else ("基线" if arm == "S0" else "否")
        deploy = "是" if gate["deployment_pass"] else "否"
        c = cost["aggregate"][arm]
        text.append(
            f"| {arm} | {gate['macro_f1_delta'] * 100:+.2f} | {paired} | {quality} | {deploy} | "
            f"{c['params_total']:,} | {c['cpu_p95_median_ms']:.3f} |"
        )
    text += [
        "",
        f"候选：**{selection['candidate']}**；在PrivateTest前于{selection['fixed_at']}固定。",
        "Private只作最终报告，没有据其调参或改变候选；此处候选不等于已更新推理演示权重。",
        "",
        "## 无增强Training诊断",
        "",
        "| 臂 | Training acc | Public acc | 差距(pp) |",
        "|---|---:|---:|---:|",
    ]
    for arm in ARCHITECTURES:
        tr = summaries["Training"][arm]["metrics"]["accuracy"]["mean"]
        val = summaries["PublicTest"][arm]["metrics"]["accuracy"]["mean"]
        text.append(f"| {arm} | {tr * 100:.2f} | {val * 100:.2f} | {(tr - val) * 100:.2f} |")
    text += [
        "",
        "差距用于解释拟合/泛化；不能单独定位过拟合成因。S2同时增加深度与容量，不能归因于纯深度。",
        "n=3为描述性比较，不宣称统计显著。官方划分像素重复仍存在，不宣称跨人员泛化。",
        "",
        "![质量与成本](architecture_comparison.png)",
        "![训练曲线](training_curves.png)",
        "![混淆矩阵](public_confusion_matrices.png)",
        "",
        "完整七类统计、成本、更新数与来源指纹见[聚合摘要](aggregate_metrics.json)。",
        "[方案](../../configs/plans/architecture_ce_v1.md)。原始预测、逐run记录、重复计时与权重仅保留本地。",
    ]
    (output / "RESULTS.md").write_text("\n".join(text) + "\n", encoding="utf-8")
    write_figures(output, summaries, cost["aggregate"], evaluations)
    print(f"ARCH complete: {output}", flush=True)
    return aggregate


def main():
    ensure_utf8_stdio()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--run", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    finalize(load_frozen_protocol(args.protocol, historical=True), args.run, args.output)


if __name__ == "__main__":
    main()

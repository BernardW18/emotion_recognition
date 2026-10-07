"""Audit every frozen run and compare minority-class interventions on PublicTest."""
import argparse
import copy
import json
import statistics
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.checkpoint import write_json_atomic
from utils.comparison_check import check_formal_eligibility
from utils.evaluation import evaluate_checkpoint
from utils.formal_protocol import load_frozen_protocol, validate_formal_plan
from utils.model_spec import file_sha256
from utils.stdio import ensure_utf8_stdio


def summarize_rows(rows: list[dict]) -> dict:
    """Keep all seeds and all seven recalls; choose using PublicTest exclusively."""
    groups: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        if row["split"] != "PublicTest":
            raise ValueError("候选判定仅允许 PublicTest")
        key = (row["model_name"], row["arm"])
        groups.setdefault(key, []).append(row)
    output: dict = {}
    for (model, arm), group in sorted(groups.items()):
        seeds = [row["seed"] for row in group]
        if len(seeds) < 3 or len(set(seeds)) != len(seeds):
            raise ValueError("每个模型/臂至少需要3个不同 seeds，重复/缺失不可汇总")
        class_names = list(group[0]["metrics"]["per_class"])
        if len(class_names) != 7 or any(
            list(row["metrics"]["per_class"]) != class_names for row in group
        ):
            raise ValueError("必须保存相同顺序的完整七类指标")
        stats = {}
        for metric in ("accuracy", "macro_f1", "balanced_accuracy"):
            values = [row["metrics"][metric] for row in group]
            stats[metric] = {"mean": statistics.mean(values), "std": statistics.stdev(values)}
        classes = {}
        for name in class_names:
            supports = [row["metrics"]["per_class"][name]["support"] for row in group]
            if len(set(supports)) != 1 or supports[0] <= 0:
                raise ValueError("各 run 的类别支持数必须一致且为正")
            values = [row["metrics"]["per_class"][name]["recall"] for row in group]
            classes[name] = {
                "support": supports[0], "mean": statistics.mean(values),
                "std": statistics.stdev(values),
            }
        output.setdefault(model, {})[arm] = {
            "n": len(group), "seeds": sorted(seeds), "metrics": stats, "per_class": classes,
        }
    for model, arms in output.items():
        if "A" not in arms:
            raise ValueError(f"{model} 缺少同协议 CE 基线 A")
        baseline = arms["A"]
        for arm, group_summary in arms.items():
            if group_summary["seeds"] != baseline["seeds"]:
                raise ValueError(f"{model}/{arm} 与 CE 基线 seeds 不一致")
            recall_improved = (
                group_summary["per_class"]["Disgust"]["mean"]
                > baseline["per_class"]["Disgust"]["mean"]
            )
            macro_preserved = (
                group_summary["metrics"]["macro_f1"]["mean"]
                >= baseline["metrics"]["macro_f1"]["mean"]
            )
            group_summary["improved_vs_ce"] = arm != "A" and recall_improved and macro_preserved
    return output


def collect_runs(record: dict, run_dirs: list[Path]) -> list[dict]:
    """Reject mixed protocols, incomplete plans or duplicate seeds before any evaluation."""
    plans = record["manifest"]["plans"]
    baselines = {p["model_name"]: p for p in plans if p["arm"] == "A"}
    definitions = {"A": ("cross_entropy", False), "B": ("focal", False),
                   "C": ("focal", True), "D": ("cb_focal", True)}
    for plan in plans:
        config = plan["training_protocol"]["config"]
        if plan["arm"] not in definitions or plan["model_name"] not in baselines:
            raise ValueError("消融清单须使用A–D臂且每个模型有CE基线A")
        loss, weighted = definitions[plan["arm"]]
        if (config["training"]["loss_type"] != loss
                or config["dataloader"]["class_balanced_sampling"] is not weighted
                or config["augmentation"]["class_specific"]["enabled"]):
            raise ValueError(f"{plan['model_name']}/{plan['arm']} 实际损失/采样/增强不符合臂定义")
        baseline = baselines[plan["model_name"]]

        def without_arm_factors(cfg):
            normalized = copy.deepcopy(cfg)
            normalized.pop("seed", None)
            normalized["training"].pop("loss_type")
            normalized["dataloader"].pop("class_balanced_sampling")
            return normalized

        if (without_arm_factors(config)
                != without_arm_factors(baseline["training_protocol"]["config"])
                or any(plan[k] != baseline[k] for k in (
                    "seeds", "max_epochs", "allow_early_stop", "lr_floor", "model_spec",
                ))):
            raise ValueError("同模型各臂只能改变损失/采样，须共享配置、种子、预算和模型")
    expected = {
        (plan["model_name"], plan["arm"], seed)
        for plan in record["manifest"]["plans"] for seed in plan["seeds"]
    }
    data_hashes = {plan["training_protocol"]["data_fingerprint"]["csv_sha256"]
                   for plan in record["manifest"]["plans"]}
    if len(data_hashes) != 1:
        raise ValueError("同一次消融比较必须使用相同数据指纹")
    collected = {}
    for run_dir in run_dirs:
        eligibility = check_formal_eligibility(run_dir, frozen_protocol_path=Path(record["path"]))
        if not eligibility["formal_eligible"]:
            raise ValueError(f"{run_dir}: {eligibility['reasons']}")
        meta = json.loads((run_dir / "run_meta.json").read_text(encoding="utf-8"))
        checkpoint = torch.load(run_dir / "checkpoints" / "last.pth",
                                map_location="cpu", weights_only=False)
        plan = validate_formal_plan(
            record, meta["model_name"], meta["model_spec"],
            checkpoint["training_protocol"], historical=True,
        )
        key = (plan["model_name"], plan["arm"], meta["seed"])
        if key in collected:
            raise ValueError(f"同一个模型/臂/seed 不能挑选或混合多个 run: {key}")
        collected[key] = {"model_name": key[0], "arm": key[1], "seed": key[2],
                          "run_dir": str(run_dir.resolve()), "run_id": meta["run_id"],
                          "final_epoch": meta["final_epoch"],
                          "best_epoch": checkpoint["best"]["epoch"],
                          "completion_reason": meta["completion_reason"],
                          "training_duration_seconds": checkpoint["training_duration_seconds"]}
    if set(collected) != expected:
        raise ValueError(f"冻结计划尚不完整: 缺少 {sorted(expected - set(collected))}")
    return [collected[key] for key in sorted(collected)]


def main():
    ensure_utf8_stdio()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--run", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--csv", type=Path, default=PROJECT_ROOT / "data" / "fer2013.csv")
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("输出已存在；请指定新目录，以保留原评估")
    record = load_frozen_protocol(args.protocol, historical=True)
    rows = collect_runs(record, args.run)
    expected_sha = record["manifest"]["plans"][0]["training_protocol"]["data_fingerprint"][
        "csv_sha256"
    ]
    if file_sha256(args.csv) != expected_sha:
        raise ValueError("评估数据与冻结训练数据不匹配")
    for row in rows:
        result = evaluate_checkpoint(
            Path(row["run_dir"]) / "checkpoints" / "best.pth", split="PublicTest",
            device="cpu", batch_size=64, csv_path=args.csv,
            output_dir=args.output / row["model_name"] / row["arm"] / str(row["seed"]),
        )
        if result["protocol"]["csv_sha256"] != expected_sha:
            raise ValueError("评估数据与冻结训练数据不匹配")
        row.update(split="PublicTest", metrics=result["metrics"],
                   checkpoint=result["checkpoint"], protocol=result["protocol"],
                   predictions_sha256=file_sha256(Path(result["output_dir"]) / "predictions.csv"),
                   summary_sha256=file_sha256(Path(result["output_dir"]) / "summary.json"))
    body = {"protocol_id": record["protocol_id"], "protocol_sha256": record["file_sha256"],
            "selection_split": "PublicTest", "sample_std_ddof": 1, "runs": rows,
            "summary": summarize_rows(rows),
            "acceptance": "mean Disgust recall > A and mean macro-F1 >= A; descriptive n>=3"}
    write_json_atomic(args.output / "results.json", body)
    print(f"已保存 {len(rows)} 个 run 的完整验证集比较：{args.output}")


if __name__ == "__main__":
    main()

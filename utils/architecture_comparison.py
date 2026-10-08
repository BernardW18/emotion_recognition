"""Frozen four-arm architecture checks and prespecified paired PublicTest gates."""

import copy
import json
import statistics
from pathlib import Path

import torch

from utils.comparison_check import check_formal_eligibility
from utils.constants import CLASS_NAMES
from utils.formal_protocol import validate_formal_plan
from utils.model_spec import ModelSpec

ARCHITECTURES = {
    "S0": ([2, 2], [64, 128], "before_stage1", False),
    "S1": ([2, 2], [64, 128], "after_stage1", False),
    "S2": ([4, 4], [64, 128], "before_stage1", False),
    "S3": ([2, 2], [64, 128], "before_stage1", True),
}
SEEDS = (42, 43, 44)


def validate_architecture_manifest(record):
    """Require the actual four structures, identical recipe/runtime/data and full budget."""
    plans = record["manifest"]["plans"]
    if len(plans) != 4 or {p["arm"] for p in plans} != set(ARCHITECTURES):
        raise ValueError("结构清单必须恰好包含S0–S3四臂")
    common = None
    for plan in plans:
        if (
            plan["model_name"] != "micro_resnet"
            or plan["seeds"] != list(SEEDS)
            or plan["max_epochs"] != 90
            or plan["allow_early_stop"] is not False
        ):
            raise ValueError("结构清单要求MicroResNet、seeds42/43/44、完整90轮")
        spec = ModelSpec.from_dict(plan["model_spec"])
        expected = ARCHITECTURES[plan["arm"]]
        actual = (list(spec.blocks or ()), list(spec.channels or ()), spec.pool_order, spec.use_se)
        if spec.spec_version != 2 or actual != expected:
            raise ValueError(f"{plan['arm']} 实际模型结构与臂定义不符")
        protocol = copy.deepcopy(plan["training_protocol"])
        config = protocol["config"]
        mcfg = config["models"]["micro_resnet"]
        recipe = {
            "learning_rate": 3e-4,
            "batch_size": 128,
            "num_epochs": 90,
            "dropout": 0.3,
            "activation": "gelu",
        }
        training_recipe = {
            "optimizer": "adam",
            "weight_decay": 1e-4,
            "scheduler": "cosine_warm",
            "scheduler_t0": 30,
            "amp": True,
            "torch_compile": False,
            "optimizer_fused": False,
            "gradient_accumulation_steps": 1,
            "max_grad_norm": 1.0,
            "patience": 0,
            "val_loss_patience": 0,
            "cudnn_deterministic": False,
        }
        if (
            any(mcfg.get(k) != v for k, v in recipe.items())
            or any(config["training"].get(k) != v for k, v in training_recipe.items())
            or config["dataloader"]["num_workers"] != 0
            or config["dataloader"]["persistent_workers"]
            or config["augmentation"]["impl"] != "batched"
            or not config["augmentation"]["enabled"]
        ):
            raise ValueError("结构协议必须使用已定稿的CE/AMP/Adam/90轮训练配方")
        if (
            tuple(mcfg[k] for k in ("blocks", "channels", "pool_order", "use_se")) != expected
            or config["training"]["loss_type"] != "cross_entropy"
            or config["dataloader"]["class_balanced_sampling"]
            or config["augmentation"]["class_specific"]["enabled"]
            or config["augmentation"]["mixup"]["enabled"]
        ):
            raise ValueError("实际配置结构/CE/采样/增强与实验定义不符")
        config.pop("seed", None)
        for key in ("blocks", "channels", "pool_order", "use_se"):
            mcfg.pop(key)
        if common is None:
            common = protocol
        elif protocol != common:
            raise ValueError("结构各臂除声明的结构字段外必须共享完整配置/运行时/数据")
    return plans


def collect_architecture_runs(record, run_dirs):
    """Audit real checkpoints, reject missing/duplicate/mixed runs before evaluation."""
    plans = validate_architecture_manifest(record)
    expected = {(p["arm"], seed) for p in plans for seed in p["seeds"]}
    collected = {}
    for directory in run_dirs:
        directory = Path(directory)
        eligibility = check_formal_eligibility(
            directory,
            frozen_protocol_path=Path(record["path"]),
        )
        if not eligibility["formal_eligible"]:
            raise ValueError(f"{directory}: {eligibility['reasons']}")
        meta = json.loads((directory / "run_meta.json").read_text(encoding="utf-8"))
        last = torch.load(
            directory / "checkpoints/last.pth", map_location="cpu", weights_only=False
        )
        plan = validate_formal_plan(
            record,
            meta["model_name"],
            meta["model_spec"],
            last["training_protocol"],
            historical=True,
        )
        key = (plan["arm"], meta["seed"])
        if key in collected:
            raise ValueError(f"重复的结构臂/seed，不允许挑选run: {key}")
        collected[key] = {
            "arm": key[0],
            "seed": key[1],
            "run_dir": str(directory.resolve()),
            "run_id": directory.name,
            "final_epoch": last["epoch"],
            "best_epoch": last["best"]["epoch"],
            "training_duration_seconds": last["training_duration_seconds"],
            "optimizer_attempts": last["optimizer_attempts"],
            "optimizer_updates": last["optimizer_updates"],
            "resume_events": len(meta.get("resume_events", [])),
        }
    if set(collected) != expected:
        raise ValueError(f"冻结结构计划不完整: 缺少{sorted(expected - set(collected))}")
    return [collected[key] for key in sorted(collected)]


def summarize_architecture(rows, split):
    """Summarize exactly twelve evaluations, preserving all class supports and seeds."""
    if split not in ("Training", "PublicTest", "PrivateTest"):
        raise ValueError("未知评估划分")
    expected = {(arm, seed) for arm in ARCHITECTURES for seed in SEEDS}
    actual = [(r["arm"], r["seed"]) for r in rows]
    if len(actual) != 12 or set(actual) != expected or any(r["split"] != split for r in rows):
        raise ValueError("汇总要求相同划分、S0–S3完整三seed且无重复")
    output = {}
    for arm in ARCHITECTURES:
        group = [r for r in rows if r["arm"] == arm]
        stats = {}
        for name in ("accuracy", "macro_f1", "balanced_accuracy", "ece", "nll"):
            values = [r["metrics"][name] for r in group]
            stats[name] = {"mean": statistics.mean(values), "std": statistics.stdev(values)}
        classes = {}
        for row in group:
            if list(row["metrics"]["per_class"]) != list(CLASS_NAMES):
                raise ValueError("必须报告标准顺序的完整七类")
        for name in CLASS_NAMES:
            supports = [r["metrics"]["per_class"][name]["support"] for r in group]
            if len(set(supports)) != 1 or supports[0] <= 0:
                raise ValueError("七类支持数必须一致且为正")
            values = [r["metrics"]["per_class"][name]["recall"] for r in group]
            classes[name] = {
                "support": supports[0],
                "mean": statistics.mean(values),
                "std": statistics.stdev(values),
            }
        output[arm] = {"n": 3, "seeds": list(SEEDS), "metrics": stats, "per_class": classes}
    return output


def select_architecture(public_rows, efficiency):
    """Use PublicTest plus same-round cost only; PrivateTest is never an input."""
    summary = summarize_architecture(public_rows, "PublicTest")
    baseline = summary["S0"]["metrics"]
    by_key = {(r["arm"], r["seed"]): r["metrics"] for r in public_rows}
    gates = {}
    for arm in ARCHITECTURES:
        f1_delta = summary[arm]["metrics"]["macro_f1"]["mean"] - baseline["macro_f1"]["mean"]
        acc_delta = summary[arm]["metrics"]["accuracy"]["mean"] - baseline["accuracy"]["mean"]
        paired = [by_key[arm, s]["macro_f1"] - by_key["S0", s]["macro_f1"] for s in SEEDS]
        cost = efficiency[arm]
        ratio = cost["cpu_p95_median_ms"] / efficiency["S0"]["cpu_p95_median_ms"]
        quality = arm != "S0" and f1_delta >= 0.010 - 1e-12 and acc_delta >= -0.005 - 1e-12
        quality = quality and all(d > 0 for d in paired)
        deploy = cost["params_total"] <= 1600000 and cost["batch128_no_oom"] and ratio <= 1.5
        gates[arm] = {
            "macro_f1_delta": f1_delta,
            "accuracy_delta": acc_delta,
            "paired_macro_f1_deltas": paired,
            "quality_pass": quality,
            "deployment_pass": bool(deploy),
            "cpu_p95_ratio_vs_s0": ratio,
        }
    candidates = [
        a for a in ARCHITECTURES if gates[a]["quality_pass"] and gates[a]["deployment_pass"]
    ]
    candidates.sort(
        key=lambda a: (
            -summary[a]["metrics"]["macro_f1"]["mean"],
            efficiency[a]["cpu_p95_median_ms"],
            efficiency[a]["params_total"],
        )
    )
    return {
        "selection_split": "PublicTest",
        "gates": gates,
        "candidate": candidates[0] if candidates else "S0",
        "ranked_deployable_candidates": candidates,
    }

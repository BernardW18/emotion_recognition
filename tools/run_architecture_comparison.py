"""Execute the frozen rotated twelve-run CE plan, then evaluate and publish conclusions."""

import argparse
import gc
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.dataloader import create_dataloaders
from tools.summarize_architecture import finalize
from training.checkpoint import write_json_atomic
from training.trainer import Trainer, load_config, set_seed
from utils.architecture_comparison import ARCHITECTURES, validate_architecture_manifest
from utils.comparison_check import check_formal_eligibility
from utils.formal_protocol import load_frozen_protocol
from utils.model_spec import build_model_from_spec, file_sha256, make_spec_from_config
from utils.stdio import ensure_utf8_stdio


def execution_order():
    arms = list(ARCHITECTURES)
    return [
        (arm, seed)
        for index, seed in enumerate((42, 43, 44))
        for arm in arms[index:] + arms[:index]
    ]


def snapshot_previous(output):
    """Protect local prior experiments and course artifacts without depending on docs/."""
    files = {ROOT / "data/fer2013.csv"}
    for directory in (
        "analysis",
        "training/runs",
        "training/checkpoints",
        "training/logs",
        "inference/saved_models",
        "report&ppt",
        "configs/protocols",
    ):
        files.update(
            p for p in (ROOT / directory).rglob("*") if p.is_file() and output not in p.parents
        )
    return [
        {"path": str(p.relative_to(ROOT)), "sha256": file_sha256(p)}
        for p in sorted(files)
        if p.exists()
    ]


def verify_previous(records):
    for record in records:
        if file_sha256(ROOT / record["path"]) != record["sha256"]:
            raise ValueError(f"历史产物发生改变: {record['path']}")


def main():
    ensure_utf8_stdio()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "analysis/architecture_ce_v1")
    parser.add_argument(
        "--resume", action="store_true", help="continue recorded runs, never pick new seeds/runs"
    )
    args = parser.parse_args()
    if sys.platform != "win32" or Path(sys.prefix).resolve() != ROOT / ".venv":
        raise SystemExit("本实验必须使用项目Windows .venv")
    if not torch.cuda.is_available():
        raise SystemExit("本冻结实验要求CUDA AMP，不降级为CPU训练")
    if any(
        os.environ.get(k) != "4"
        for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
    ):
        raise SystemExit("启动前请将OMP/MKL/OPENBLAS_NUM_THREADS全部设为4")
    torch.set_num_threads(4)
    record = load_frozen_protocol(args.protocol)
    validate_architecture_manifest(record)
    output = args.output.resolve()
    state_path = output / "experiment_state.json"
    if state_path.exists():
        if not args.resume:
            raise SystemExit("执行记录已存在；明确使用--resume继续同run，不覆盖历史")
        state = json.loads(state_path.read_text("utf-8"))
        if state["protocol_sha256"] != record["file_sha256"]:
            raise ValueError("恢复执行记录的冻结协议不一致")
        if any(
            state["environment"][k] != value
            for k, value in (
                ("torch", torch.__version__),
                ("cuda", torch.version.cuda),
                ("gpu", torch.cuda.get_device_name(0)),
            )
        ):
            raise ValueError("恢复环境与原CUDA/torch/GPU不一致")
        if state["status"] == "complete":
            print("ARCH already complete; no training/evaluation repeated")
            return
        verify_previous(state["protected_files"])
    else:
        if args.resume:
            raise SystemExit("没有可恢复的执行记录")
        output.mkdir(parents=True, exist_ok=True)
        state = {
            "protocol_id": record["protocol_id"],
            "protocol_sha256": record["file_sha256"],
            "started_at": datetime.now().isoformat(),
            "completed": [],
            "current": None,
            "protected_files": snapshot_previous(output),
            "environment": {
                "python": sys.executable,
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(0),
                "cpu_threads": 4,
                "omp_mkl_openblas_threads": 4,
            },
        }
    state.update(status="running", pid=os.getpid())
    write_json_atomic(state_path, state)
    import ctypes

    ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    try:
        for arm, seed in execution_order():
            completed = [r for r in state["completed"] if (r["arm"], r["seed"]) == (arm, seed)]
            if completed:
                if len(completed) != 1:
                    raise ValueError("执行记录包含重复run")
                continue
            load_frozen_protocol(args.protocol)
            current = state["current"]
            if current is not None and (current["arm"], current["seed"]) != (arm, seed):
                raise ValueError("恢复顺序与已记录run不一致")
            cfg = load_config(ROOT / f"configs/architecture_{arm.lower()}_config.yaml")
            cfg["seed"] = seed
            set_seed(seed, deterministic=False)
            train, val, _, _ = create_dataloaders(cfg, "micro_resnet", include_test=False)
            spec = make_spec_from_config(cfg, "micro_resnet")
            trainer = Trainer(
                build_model_from_spec(spec),
                train,
                val,
                None,
                cfg,
                "micro_resnet",
                device=torch.device("cuda"),
                class_counts=train.dataset.class_counts,
                run_dir=Path(current["run_dir"]) if current else None,
                run_meta_extra={"architecture_arm": arm, "execution_output": str(output)},
                run_purpose="formal",
                frozen_protocol=record,
            )
            state["current"] = {"arm": arm, "seed": seed, "run_dir": str(trainer.run_dir)}
            write_json_atomic(state_path, state)
            if current:
                trainer.load_checkpoint(trainer.checkpoints_dir / "last.pth")
            remaining = 90 - (trainer.start_epoch - 1)
            print(
                f"ARCH start {arm} seed{seed}, remaining={remaining}, run={trainer.run_dir}",
                flush=True,
            )
            if remaining:
                trainer.fit(remaining)
            eligible = check_formal_eligibility(trainer.run_dir, frozen_protocol_path=args.protocol)
            if not eligible["formal_eligible"]:
                raise ValueError(eligible["reasons"])
            state["completed"].append(state["current"])
            state["current"] = None
            write_json_atomic(state_path, state)
            print(f"ARCH completed {len(state['completed'])}/12: {arm} seed{seed}", flush=True)
            del trainer, train, val
            gc.collect()
            torch.cuda.empty_cache()
        state["status"] = "evaluating"
        write_json_atomic(state_path, state)
        aggregate = finalize(record, [r["run_dir"] for r in state["completed"]], output)
        verify_previous(state["protected_files"])
        state.update(
            status="complete",
            finished_at=datetime.now().isoformat(),
            candidate=aggregate["selection"]["candidate"],
            protected_files_unchanged=True,
        )
        write_json_atomic(state_path, state)
        print(
            f"ARCH all complete; historical files preserved={len(state['protected_files'])}",
            flush=True,
        )
    except BaseException as exc:
        state.update(status="failed", error=repr(exc), failed_at=datetime.now().isoformat())
        write_json_atomic(state_path, state)
        raise
    finally:
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


if __name__ == "__main__":
    main()

"""Same-round CPU/CUDA latency and AMP batch128 cost for the four frozen structures."""

import gc
import statistics
import sys
import time
from pathlib import Path

import torch
from PIL import Image
from torch.utils.flop_counter import FlopCounterMode

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from inference.infer_utils import preprocess_image
from utils.architecture_comparison import ARCHITECTURES, validate_architecture_manifest
from utils.model_spec import ModelSpec, build_model_from_spec, count_parameters


def timings(call, sync=lambda: None):
    """Keep individual measurements locally; aggregate median/P95 by round."""
    for _ in range(20):
        call()
    sync()
    values = []
    for _ in range(100):
        sync()
        start = time.perf_counter()
        call()
        sync()
        values.append((time.perf_counter() - start) * 1000)
    return {"median_ms": statistics.median(values), "p95_ms": sorted(values)[94], "raw_ms": values}


def benchmark_architectures(record):
    plans = validate_architecture_manifest(record)
    models = {
        p["arm"]: build_model_from_spec(ModelSpec.from_dict(p["model_spec"])).eval() for p in plans
    }
    output = {}
    for arm, model in models.items():
        with torch.inference_mode(), FlopCounterMode(display=False) as counter:
            model(torch.rand(1, 1, 48, 48))
        total, _ = count_parameters(model)
        flops = int(counter.get_total_flops())
        output[arm] = {
            "params_total": total,
            "macs_batch1": flops // 2,
            "flops_batch1": flops,
            "cpu_rounds": [],
            "gpu_rounds": [],
        }
    arms = list(ARCHITECTURES)
    for round_index in range(3):
        order = arms[round_index:] + arms[:round_index]
        for arm in order:
            model = models[arm]
            for device, key in (("cpu", "cpu_rounds"), ("cuda", "gpu_rounds")):
                model.to(device).eval()
                x = torch.rand(1, 1, 48, 48, device=device)
                sync = torch.cuda.synchronize if device == "cuda" else lambda: None
                with torch.inference_mode():
                    measured = timings(lambda model=model, x=x: model(x), sync)
                output[arm][key].append(measured)
                model.cpu()
                del x
        print(f"ARCH cost round {round_index + 1}/3 complete", flush=True)
    for arm, model in models.items():
        gc.collect()
        torch.cuda.empty_cache()
        model.cuda().train()
        opt = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-4)
        scaler = torch.amp.GradScaler("cuda")
        torch.cuda.reset_peak_memory_stats()
        updates = 0
        for _ in range(8):
            opt.zero_grad(set_to_none=True)
            x = torch.rand(128, 1, 48, 48, device="cuda")
            y = torch.arange(128, device="cuda") % 7
            with torch.autocast("cuda"):
                loss = torch.nn.functional.cross_entropy(model(x), y)
            if not torch.isfinite(loss):
                raise ValueError(f"{arm}: AMP loss 非有限")
            scale = scaler.get_scale()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            updates += scaler.get_scale() >= scale
        if not updates or not all(torch.isfinite(p).all() for p in model.parameters()):
            raise ValueError(f"{arm}: AMP未产生有限有效更新")
        output[arm].update(
            batch128_no_oom=True,
            amp_attempts=8,
            amp_updates=updates,
            train_peak_vram_mb=torch.cuda.max_memory_allocated() / 1024**2,
        )
        model.cpu()
        del opt, scaler, x, y, loss
        print(f"ARCH {arm} batch128 AMP cost complete", flush=True)
    image = Image.new("L", (48, 48), color=128)
    preprocess = timings(lambda: preprocess_image(image))
    aggregate = {}
    for arm, measurements in output.items():
        aggregate[arm] = {
            k: v for k, v in measurements.items() if k not in ("cpu_rounds", "gpu_rounds")
        }
        for device in ("cpu", "gpu"):
            rounds = [
                {k: v for k, v in r.items() if k != "raw_ms"}
                for r in measurements[f"{device}_rounds"]
            ]
            aggregate[arm][f"{device}_rounds"] = rounds
            aggregate[arm][f"{device}_p95_median_ms"] = statistics.median(
                r["p95_ms"] for r in rounds
            )
    return {
        "raw": output,
        "aggregate": aggregate,
        "preprocessing": preprocess,
        "method": {
            "warmup": 20,
            "repeats": 100,
            "rounds": 3,
            "cpu_threads": torch.get_num_threads(),
            "batch": 1,
            "dtype": "float32",
            "gpu_synchronize": True,
            "macs_scope": "torch flop_counter registered ops; pooling/activations excluded",
            "amp_training_cost": "8 synthetic CE Adam steps, batch128, clip1.0",
            "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
            "tf32_cudnn": torch.backends.cudnn.allow_tf32,
        },
    }

"""
效率基准 — F02

指标与口径（全部记录到输出 JSON）:
  - 参数量（总数 / 可训练数）
  - MACs / FLOPs（batch=1；torch.utils.flop_counter 计数；FLOPs = 2 × MACs）
  - 推理延迟（batch=1）：预热 10 次 + 重复 100 次，报告 median/P95；
    GPU 计时前后 synchronize；「图像预处理」与「模型前向」分开计时
  - 训练峰值显存（仅 GPU）：按各模型配置 batch_size 的随机张量训练 5 步，
    记录 torch.cuda.max_memory_allocated（不含数据加载管线）

说明: 模型为随机初始化（效率只取决于架构与配置，与权重无关）；
      延迟数字只在标注的硬件与软件环境下有效，不能跨环境直接比较。

用法: python tools/benchmark_efficiency.py
输出: analysis/efficiency_results.json
"""
import json
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from inference.infer_utils import preprocess_image
from training.trainer import load_config
from utils.model_spec import build_model_from_spec, count_parameters, make_spec_from_config
from utils.stdio import ensure_utf8_stdio

WARMUP = 10
REPEAT = 100
OUTPUT_PATH = PROJECT_ROOT / "analysis" / "efficiency_results.json"


def _p95(values):
    values = sorted(values)
    return values[min(int(0.95 * (len(values) - 1)), len(values) - 1)]


def _time_forward_ms(model, x, device):
    sync = device.type == "cuda"

    def _sync():
        if sync:
            torch.cuda.synchronize()

    model.eval()
    with torch.no_grad():
        for _ in range(WARMUP):
            model(x)
        _sync()
        times = []
        for _ in range(REPEAT):
            _sync()
            t0 = time.perf_counter()
            model(x)
            _sync()
            times.append((time.perf_counter() - t0) * 1000)
    return {
        "median_ms": statistics.median(times),
        "p95_ms": _p95(times),
        "min_ms": min(times),
        "warmup": WARMUP,
        "repeat": REPEAT,
    }


def _time_preprocess_ms(image):
    for _ in range(WARMUP):
        preprocess_image(image)
    times = []
    for _ in range(REPEAT):
        t0 = time.perf_counter()
        preprocess_image(image)
        times.append((time.perf_counter() - t0) * 1000)
    return {
        "median_ms": statistics.median(times),
        "p95_ms": _p95(times),
        "warmup": WARMUP,
        "repeat": REPEAT,
    }


def _count_macs(model, device):
    from torch.utils.flop_counter import FlopCounterMode

    x = torch.randn(1, 1, 48, 48, device=device)
    model.eval()
    with torch.no_grad(), FlopCounterMode(display=False) as fcm:
        model(x)
    flops = int(fcm.get_total_flops())
    return flops, flops // 2


def _train_peak_vram_mb(model, batch_size, device, steps=5):
    if device.type != "cuda":
        return None
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    model.train()
    torch.cuda.reset_peak_memory_stats()
    for _ in range(steps):
        x = torch.randn(batch_size, 1, 48, 48, device=device)
        y = torch.randint(0, 7, (batch_size,), device=device)
        loss = nn.functional.cross_entropy(model(x), y)
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
    peak = torch.cuda.max_memory_allocated() / (1024 ** 2)
    model.eval()
    return peak


def main():
    ensure_utf8_stdio()
    config = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    devices = [device]
    if device.type == "cuda":
        devices.append(torch.device("cpu"))

    rng = np.random.RandomState(0)
    image = Image.fromarray((rng.rand(48, 48) * 255).astype(np.uint8), mode="L")

    out = {
        "generated_at": datetime.now().isoformat(),
        "environment": {
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "platform": sys.platform,
            "torch_threads": torch.get_num_threads(),
        },
        "method": {
            "input": "1×1×48×48 float32（灰度，x/255，无增强）",
            "warmup": WARMUP,
            "repeat": REPEAT,
            "timer": "time.perf_counter；GPU 前后 torch.cuda.synchronize()",
            "macs": "torch.utils.flop_counter；FLOPs = 2 × MACs；batch=1",
            "vram": "随机张量训练 5 步，torch.cuda.max_memory_allocated（不含数据加载）",
            "notes": "随机初始化权重；结果仅在本环境有效",
        },
        "models": {},
    }

    print(f"设备: {device} | torch {torch.__version__} | CUDA {torch.version.cuda}")
    print()
    header = (
        f"{'模型':<14} {'参数量':>12} {'MACs(1)':>12} "
        f"{'前向中位(设备)':>14} {'预处理中位':>10} {'峰值显存':>9}"
    )
    print(header)
    print("-" * len(header))

    for name in ("mini_cnn", "vgg_lite", "micro_resnet"):
        spec = make_spec_from_config(config, name)
        model = build_model_from_spec(spec)
        total, trainable = count_parameters(model)

        entry = {
            "model_spec": spec.to_dict(),
            "params_total": total,
            "params_trainable": trainable,
            "latency": {},
            "preprocess_cpu": None,
            "macs_batch1": None,
            "flops_batch1": None,
            "train_peak_vram_mb": None,
        }

        for dev in devices:
            m = model.to(dev)
            flops, macs = _count_macs(m, dev)
            if dev == device:
                entry["macs_batch1"] = macs
                entry["flops_batch1"] = flops
            x1 = torch.randn(1, 1, 48, 48, device=dev)
            entry["latency"][dev.type] = _time_forward_ms(m, x1, dev)

        entry["preprocess_cpu"] = _time_preprocess_ms(image)

        bs = config["models"][name].get("batch_size", config["training"]["batch_size"])
        model.to(device)
        entry["train_peak_vram_mb"] = _train_peak_vram_mb(model, bs, device)
        entry["train_vram_batch_size"] = bs

        out["models"][name] = entry

        fwd = entry["latency"][device.type]["median_ms"]
        pre = entry["preprocess_cpu"]["median_ms"]
        peak = entry["train_peak_vram_mb"]
        peak_str = f"{peak:.0f}MB" if peak is not None else "N/A(CPU)"
        print(f"{name:<14} {total:>12,} {entry['macs_batch1']:>12,} "
              f"{fwd:>12.2f}ms {pre:>8.3f}ms {peak_str:>9}")

    out_path = OUTPUT_PATH
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print()
    print(f"输出: {out_path}")


if __name__ == "__main__":
    main()

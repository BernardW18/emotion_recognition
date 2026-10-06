"""
AMP 配对基准 — F12

目的：在配对条件下测量 AMP ON/OFF 的训练速度差异（替代原「单步测量 + 完整训练外推」）。
原证据的问题：配对初始化与数据顺序不一致、诊断会修改权重、与正式输出目录共享状态。

配对协议:
  - 每对 OFF/ON 从同一初始 state_dict 副本、同一 RNG 起点、同一数据顺序开始
  - 每个模型 3 组配对，交替执行顺序（OFF→ON / ON→OFF / OFF→ON）
  - 预热在一次性副本上进行（预热后丢弃，不影响正式测量起点）
  - 逐批次记录输入标签哈希，验证两模式的输入批次索引完全相同
测量:
  - 稳态 step 时间：每 run 前 STEP_MEASURE_STEPS 步（median/P95），
    口径 = 单步训练循环体（含数据加载，前向+反向+更新），每步 CUDA 同步
  - 完整流程时间：1 个 epoch 训练 + 1 次验证评估（端到端 wall time）
    → 加速比 = 流程 OFF 时间 / 流程 ON 时间（小于 1 如实报告）
  - 显存：每 run 峰值（torch.cuda.max_memory_allocated）
  - 精度：1 epoch 后的验证指标（单独记录，不作结论）
  - 有限性：loss 逐步检查 nan/inf；梯度抽样检查（unscale 后），统计异常次数
隔离:
  - 不写入 training/runs、training/logs、inference/saved_models
  - 基准前后校验正式权重与历史文件 SHA-256 不变
  - num_workers=0（保证配对数据顺序可复现；与正式训练多 worker 吞吐不同，仅用于 OFF/ON 对比）

输出: analysis/benchmark_amp/results.json
用法: python tools/run_amp_comparison.py
"""
import copy
import hashlib
import json
import math
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from data.dataloader import create_dataloaders
from training.checkpoint import _capture_rng_state, _restore_rng_state
from training.trainer import build_optimizer, load_config, set_seed
from utils.losses import FocalLoss
from utils.model_spec import build_model_from_spec, file_sha256, make_spec_from_config
from utils.stdio import ensure_utf8_stdio

PAIRS_PER_MODEL = 3
STEP_MEASURE_STEPS = 20
OUTPUT_PATH = PROJECT_ROOT / "analysis" / "benchmark_amp" / "results.json"


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def _batch_id(labels: torch.Tensor) -> str:
    return hashlib.sha256(labels.detach().cpu().numpy().tobytes()).hexdigest()[:12]


def _warmup_copy(model, device, amp_enabled):
    """在一次性副本上预热（触发 cuDNN 算法选择），副本随后丢弃。"""
    warm = copy.deepcopy(model)
    warm.train()
    opt = torch.optim.Adam(warm.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler("cuda") if amp_enabled else None
    bs = 64
    for _ in range(3):
        x = torch.randn(bs, 1, 48, 48, device=device)
        y = torch.randint(0, 7, (bs,), device=device)
        if amp_enabled:
            with torch.amp.autocast("cuda"):
                loss = torch.nn.functional.cross_entropy(warm(x), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        else:
            loss = torch.nn.functional.cross_entropy(warm(x), y)
            loss.backward()
            opt.step()
        opt.zero_grad(set_to_none=True)
    del warm


def _run_once(config, model_name, amp_enabled, base_state, rng_state,
              train_loader, val_loader, device):
    """单次完整流程测量（1 epoch 训练 + 1 次验证）。"""
    spec = make_spec_from_config(config, model_name)
    model = build_model_from_spec(spec).to(device)
    model.load_state_dict(base_state)

    merged = dict(config["training"])
    merged["learning_rate"] = config["models"][model_name].get(
        "learning_rate", config["training"]["learning_rate"]
    )
    optimizer = build_optimizer(model, merged)
    criterion = FocalLoss(gamma=2.0)
    scaler = torch.amp.GradScaler("cuda") if amp_enabled else None

    _warmup_copy(model, device, amp_enabled)

    # 恢复 RNG 起点（数据顺序 + 增强随机性一致）
    _restore_rng_state(rng_state)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    model.train()
    step_times = []
    batch_ids = []
    loss_nan_inf = 0
    grad_nan_inf_checks = 0
    grad_nan_inf_found = 0
    last_loss = None

    _sync(device)
    t_epoch0 = time.perf_counter()
    num_steps = len(train_loader)
    for step, (xb, yb) in enumerate(train_loader):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        if amp_enabled:
            with torch.amp.autocast("cuda"):
                loss = criterion(model(xb), yb)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

        if device.type == "cuda":
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        if step < STEP_MEASURE_STEPS:
            step_times.append(dt * 1000)

        # 有限性与输入顺序记录
        loss_v = float(loss.detach())
        last_loss = loss_v
        if not math.isfinite(loss_v):
            loss_nan_inf += 1
        batch_ids.append(_batch_id(yb))
        if step % 10 == 0 or step == num_steps - 1:
            grad_nan_inf_checks += 1
            any_bad = False
            for p in model.parameters():
                if p.grad is not None and not torch.isfinite(p.grad).all():
                    any_bad = True
                    break
            if any_bad:
                grad_nan_inf_found += 1

        optimizer.zero_grad(set_to_none=True)
    _sync(device)
    epoch_wall = time.perf_counter() - t_epoch0

    peak_vram = None
    if device.type == "cuda":
        peak_vram = torch.cuda.max_memory_allocated() / (1024 ** 2)

    # 验证（评估时间单独记录）
    model.eval()
    val_loss_sum = 0.0
    correct = 0
    total = 0
    _sync(device)
    t_eval0 = time.perf_counter()
    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            if amp_enabled:
                with torch.amp.autocast("cuda"):
                    out = model(xb)
                    loss_batch = criterion(out, yb)
            else:
                out = model(xb)
                loss_batch = criterion(out, yb)
            val_loss_sum += loss_batch.detach().float().item() * yb.size(0)
            correct += int((out.argmax(1) == yb).sum().item())
            total += yb.size(0)
    _sync(device)
    eval_wall = time.perf_counter() - t_eval0

    return {
        "amp": amp_enabled,
        "steps": num_steps,
        "step_median_ms": statistics.median(step_times) if step_times else None,
        "step_p95_ms": (
            sorted(step_times)[min(int(0.95 * (len(step_times) - 1)), len(step_times) - 1)]
            if step_times else None
        ),
        "epoch_seconds": epoch_wall,
        "eval_seconds": eval_wall,
        "process_seconds": epoch_wall + eval_wall,
        "peak_vram_mb": peak_vram,
        "final_train_loss": last_loss,
        "val_loss": val_loss_sum / max(total, 1),
        "val_acc": correct / max(total, 1),
        "loss_nan_inf_count": loss_nan_inf,
        "grad_checks": grad_nan_inf_checks,
        "grad_nan_inf_found": grad_nan_inf_found,
        "batch_ids_hash": hashlib.sha256("".join(batch_ids).encode()).hexdigest(),
    }


def _median_of(pairs: list, key: str, label: str) -> float:
    """从配对结果列表中取指定侧（OFF/ON）某一指标的中位数。"""
    return float(statistics.median([p[label][key] for p in pairs]))


def _formal_hashes() -> dict:
    hashes = {}
    saved = PROJECT_ROOT / "inference" / "saved_models"
    if saved.exists():
        for p in sorted(saved.glob("*.pth")):
            hashes[str(p)] = file_sha256(p)
        manifest = saved / "export_manifest.json"
        if manifest.exists():
            hashes[str(manifest)] = file_sha256(manifest)
    logs = PROJECT_ROOT / "training" / "logs"
    if logs.exists():
        for p in sorted(logs.glob("*/*.json")):
            hashes[str(p)] = file_sha256(p)
    return hashes


def main():
    ensure_utf8_stdio()
    config = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("未检测到 CUDA GPU：AMP 基准需要 GPU（AMP 仅在 CUDA 上启用），跳过。")
        return

    print(f"设备: {torch.cuda.get_device_name(0)} | CUDA {torch.version.cuda} | "
          f"PyTorch {torch.__version__}")
    print(f"协议: {PAIRS_PER_MODEL} 组配对 / 每 run 1 epoch + eval / "
          f"前 {STEP_MEASURE_STEPS} 步计时 / num_workers=0")
    print()

    before_hashes = _formal_hashes()

    results = {
        "generated_at": datetime.now().isoformat(),
        "environment": {
            "device": torch.cuda.get_device_name(0),
            "cuda": torch.version.cuda,
            "torch": torch.__version__,
        },
        "protocol": {
            "pairs_per_model": PAIRS_PER_MODEL,
            "setup": "每对 OFF/ON 同一初始 state_dict、同一 RNG 起点、同一数据顺序",
            "order": "交替（OFF→ON / ON→OFF / OFF→ON）",
            "warmup": "一次性副本预热后丢弃",
            "epochs_per_run": 1,
            "step_measure_steps": STEP_MEASURE_STEPS,
            "num_workers": 0,
            "loss": "focal",
            "random_seed": config["seed"],
        },
        "models": {},
    }

    for model_name in ("mini_cnn", "vgg_lite", "micro_resnet"):
        print(f"{'=' * 64}")
        print(f"模型: {model_name}")
        print(f"{'=' * 64}")

        # 数据（num_workers=0 保证顺序可复现）
        cfg = copy.deepcopy(config)
        cfg["dataloader"]["num_workers"] = 0
        cfg["dataloader"]["persistent_workers"] = False
        train_loader, val_loader, _test_loader, _ = create_dataloaders(cfg, model_name)

        # 统一起点
        set_seed(config["seed"])
        spec = make_spec_from_config(config, model_name)
        base_model = build_model_from_spec(spec)
        base_state = copy.deepcopy(base_model.state_dict())
        base_rng = _capture_rng_state()
        del base_model

        pairs = []
        for pair_idx in range(PAIRS_PER_MODEL):
            order = [False, True] if pair_idx % 2 == 0 else [True, False]
            order_str = "OFF->ON" if pair_idx % 2 == 0 else "ON->OFF"
            print(f"  配对 {pair_idx + 1}/{PAIRS_PER_MODEL}（顺序 {order_str}）...")
            runs = {}
            for amp in order:
                label = "ON " if amp else "OFF"
                t_start = time.time()
                runs[amp] = _run_once(
                    cfg, model_name, amp, base_state, base_rng,
                    train_loader, val_loader, device,
                )
                r = runs[amp]
                print(f"    {label}: step中位 {r['step_median_ms']:.1f}ms | "
                      f"epoch {r['epoch_seconds']:.1f}s | eval {r['eval_seconds']:.1f}s | "
                      f"峰值 {r['peak_vram_mb']:.0f}MB | val_acc {r['val_acc']:.4f} | "
                      f"用时 {time.time() - t_start:.0f}s")
            off, on = runs[False], runs[True]
            pair = {
                "order": order_str,
                "OFF": off,
                "ON": on,
                "process_speedup": off["process_seconds"] / on["process_seconds"],
                "step_ratio": (off["step_median_ms"] / on["step_median_ms"])
                if on["step_median_ms"] else None,
                "batch_order_match": off["batch_ids_hash"] == on["batch_ids_hash"],
            }
            pairs.append(pair)
            print(f"    配对结果: 流程加速 {pair['process_speedup']:.2f}x | "
                  f"批次顺序一致: {pair['batch_order_match']}")
            print()

        summary = {
            "off_process_median_s": _median_of(pairs, "process_seconds", "OFF"),
            "on_process_median_s": _median_of(pairs, "process_seconds", "ON"),
            "process_speedup_median": statistics.median(
                [p["process_speedup"] for p in pairs]
            ),
            "off_step_median_ms": _median_of(pairs, "step_median_ms", "OFF"),
            "on_step_median_ms": _median_of(pairs, "step_median_ms", "ON"),
            "step_ratio_median": statistics.median(
                [p["step_ratio"] for p in pairs if p["step_ratio"]]
            ),
            "all_batch_orders_match": all(p["batch_order_match"] for p in pairs),
            "any_loss_nan_inf": any(
                p[label]["loss_nan_inf_count"] > 0 for p in pairs for label in ("OFF", "ON")
            ),
            "any_grad_nan_inf": any(
                p[label]["grad_nan_inf_found"] > 0 for p in pairs for label in ("OFF", "ON")
            ),
        }
        results["models"][model_name] = {"pairs": pairs, "summary": summary}
        print(f"  === {model_name} 汇总 ===")
        print(f"  完整流程: OFF {summary['off_process_median_s']:.1f}s → "
              f"ON {summary['on_process_median_s']:.1f}s | "
              f"加速比 {summary['process_speedup_median']:.2f}x （<1 表示变慢）")
        print(f"  稳态 step: OFF {summary['off_step_median_ms']:.1f}ms → "
              f"ON {summary['on_step_median_ms']:.1f}ms | "
              f"比值 {summary['step_ratio_median']:.2f}x")
        print()

    after_hashes = _formal_hashes()
    unchanged = before_hashes == after_hashes
    results["formal_artifacts_unchanged"] = unchanged
    results["formal_artifacts_checked"] = len(before_hashes)

    # 报告
    print(f"{'=' * 64}")
    print("AMP 配对基准报告")
    print(f"{'=' * 64}")
    print(f"{'模型':<14} {'流程 OFF':>10} {'流程 ON':>10} {'加速比':>8} "
          f"{'step OFF':>9} {'step ON':>9} {'顺序一致':>8}")
    for name, r in results["models"].items():
        s = r["summary"]
        print(f"{name:<14} {s['off_process_median_s']:>9.1f}s {s['on_process_median_s']:>9.1f}s "
              f"{s['process_speedup_median']:>7.2f}x "
              f"{s['off_step_median_ms']:>7.1f}ms {s['on_step_median_ms']:>7.1f}ms "
              f"{str(s['all_batch_orders_match']):>8}")
    print()
    print(f"正式产物 SHA-256 全部不变: {unchanged}（检查 {len(before_hashes)} 个文件）")
    if not unchanged:
        print("警告: 正式产物发生变化！")
        for k in set(before_hashes) | set(after_hashes):
            if before_hashes.get(k) != after_hashes.get(k):
                print(f"  变化: {k}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存: {OUTPUT_PATH}")


if __name__ == "__main__":
    # Windows multiprocessing spawn 模式需要 __main__ 保护
    main()

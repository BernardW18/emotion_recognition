"""
AMP 性能对比实验脚本
在 3 个模型上分别以 AMP ON/OFF 运行诊断，对比训练速度和精度。

用法:
    python tools/run_amp_comparison.py
"""
import sys
from pathlib import Path

# 项目根目录 = tools/../
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import json
import time
import copy
from training.trainer import (
    load_config, set_seed, Trainer,
)
from data.dataloader import create_dataloaders
from models.mini_cnn import MiniCNN
from models.vgg_lite import VGGLite
from models.micro_resnet import MicroResNet


def run_experiment():
    config = load_config()
    set_seed(config["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device.type != "cuda":
        print("未检测到 CUDA GPU，AMP 实验跳过")
        return

    print(f"设备: {torch.cuda.get_device_name(0)}")
    print(f"CUDA 版本: {torch.version.cuda}")
    print(f"PyTorch 版本: {torch.__version__}")
    print()

    MODELS = {
        "mini_cnn": (MiniCNN, config["models"]["mini_cnn"]),
        "vgg_lite": (VGGLite, config["models"]["vgg_lite"]),
        "micro_resnet": (MicroResNet, config["models"]["micro_resnet"]),
    }

    results = []

    for model_name, (ModelClass, mcfg) in MODELS.items():
        for amp_enabled in [False, True]:
            cfg_copy = copy.deepcopy(config)
            cfg_copy["training"]["amp"] = amp_enabled

            print(f"{'='*50}")
            print(f"模型: {model_name} | AMP: {'ON' if amp_enabled else 'OFF'}")
            print(f"{'='*50}")

            train_loader, val_loader, test_loader, _ = create_dataloaders(cfg_copy, model_name)

            model = ModelClass(
                num_classes=config["data"]["num_classes"],
                dropout=mcfg.get("dropout", 0.3),
                activation=mcfg.get("activation", config["training"]["activation"]),
            )

            trainer = Trainer(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                config=cfg_copy,
                model_name=model_name,
                device=device,
            )

            # 诊断 10 步（含 3 步预热）
            timings = trainer.diagnose(num_steps=10)

            # 训练 3 轮看精度
            print(f"\n训练 3 轮...")
            start = time.time()
            history = trainer.fit(3)
            elapsed = time.time() - start

            results.append({
                "model": model_name,
                "amp": amp_enabled,
                "avg_step_ms": (
                    timings["data_load"] + timings["forward"]
                    + timings["backward"] + timings["optimizer"]
                ),
                "data_load_ms": timings["data_load"],
                "forward_ms": timings["forward"],
                "backward_ms": timings["backward"],
                "optimizer_ms": timings["optimizer"],
                "epochs_trained": len(history["train_loss"]),
                "final_val_acc": history["val_acc"][-1] if history["val_acc"] else 0,
                "final_val_loss": history["val_loss"][-1] if history["val_loss"] else 0,
                "total_time_3_epochs": elapsed,
            })

            print()

    # 报告
    print(f"\n{'='*60}")
    print("AMP 对比实验报告")
    print(f"{'='*60}")
    print(f"{'模型':<14} {'AMP':<6} {'it/s':<8} {'3轮用时':<10} {'val_acc':<10} {'val_loss':<10}")
    print(f"{'-'*60}")
    for r in results:
        it_per_sec = 1000.0 / r["avg_step_ms"] if r["avg_step_ms"] > 0 else 0
        print(
            f"{r['model']:<14} "
            f"{'ON' if r['amp'] else 'OFF':<6} "
            f"{it_per_sec:<8.0f} "
            f"{r['total_time_3_epochs']:<10.1f}s "
            f"{r['final_val_acc']:<10.4f} "
            f"{r['final_val_loss']:<10.4f}"
        )

    # 速度提升倍率
    print(f"\n速度提升 (AMP ON vs OFF):")
    for model_name in MODELS:
        off = [r for r in results if r["model"] == model_name and not r["amp"]]
        on = [r for r in results if r["model"] == model_name and r["amp"]]
        if off and on:
            speedup = off[0]["avg_step_ms"] / on[0]["avg_step_ms"]
            print(
                f"  {model_name}: {speedup:.2f}x "
                f"({1000/off[0]['avg_step_ms']:.0f} -> {1000/on[0]['avg_step_ms']:.0f} it/s)"
            )

    # 保存结果
    out_path = PROJECT_ROOT / "analysis" / "amp_comparison_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n结果已保存: {out_path}")


if __name__ == "__main__":
    # Windows multiprocessing spawn 模式需要 __main__ 保护
    run_experiment()

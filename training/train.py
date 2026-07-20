"""
FER2013 模型训练 CLI
替代 Jupyter Notebook 训练 Cell，支持命令行参数驱动训练。
保留 Notebook 仅做评估/可视化。

用法:
    python training/train.py --model micro_resnet --epochs 30
    python training/train.py --model mini_cnn --epochs 20 --amp
    python training/train.py --model vgg_lite --epochs 10 --resume auto
    python training/train.py --model micro_resnet --diagnose --steps 5
"""
import sys
import argparse
import time
from pathlib import Path

# 项目根目录（兼容从 project root 和 training/ 下运行）
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.trainer import (
    load_config, set_seed, Trainer,
)
from data.dataloader import create_dataloaders
from training.checkpoint import find_resume_checkpoint
from models.mini_cnn import MiniCNN
from models.vgg_lite import VGGLite
from models.micro_resnet import MicroResNet


MODEL_REGISTRY = {
    "mini_cnn": MiniCNN,
    "vgg_lite": VGGLite,
    "micro_resnet": MicroResNet,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="FER2013 情感识别模型训练",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "示例:\n"
            "  python training/train.py --model micro_resnet --epochs 30\n"
            "  python training/train.py --model mini_cnn --epochs 20 --amp\n"
            "  python training/train.py --model vgg_lite --epochs 10 --resume auto\n"
        ),
    )
    parser.add_argument(
        "--model", type=str, default="micro_resnet",
        choices=list(MODEL_REGISTRY.keys()),
        help="模型名称 (默认: micro_resnet)",
    )
    parser.add_argument(
        "--epochs", type=int, default=30,
        help="本次训练轮数 (默认: 30)",
    )
    parser.add_argument(
        "--amp", action="store_true", default=None,
        help="启用 AMP 混合精度训练（覆盖 config 中的 amp 设置）",
    )
    parser.add_argument(
        "--no-amp", action="store_true", default=None,
        help="禁用 AMP（覆盖 config 中的 amp 设置）",
    )
    parser.add_argument(
        "--resume", type=str, nargs="?", const="auto", default=None,
        help=(
            "恢复训练。可指定 checkpoint 路径，或 'auto' 自动选择最佳 checkpoint。"
            "不指定则从头训练。"
        ),
    )
    parser.add_argument(
        "--diagnose", action="store_true",
        help="仅运行性能诊断（不训练）",
    )
    parser.add_argument(
        "--steps", type=int, default=10,
        help="诊断步数 (默认: 10，含 3 步预热)",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="随机种子（覆盖 config 中的 seed）",
    )
    parser.add_argument(
        "--lr", type=float, default=None,
        help="学习率覆盖",
    )
    parser.add_argument(
        "--batch-size", type=int, default=None,
        help="Batch size 覆盖",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    model_name = args.model

    # 加载配置
    config = load_config()
    seed = args.seed if args.seed is not None else config["seed"]
    set_seed(seed)

    # AMP 覆盖
    if args.amp is True:
        config["training"]["amp"] = True
    elif args.no_amp is True:
        config["training"]["amp"] = False

    # lr/batch_size 覆盖
    if args.lr is not None:
        config["models"][model_name]["learning_rate"] = args.lr
    if args.batch_size is not None:
        config["models"][model_name]["batch_size"] = args.batch_size

    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_config = config["models"][model_name]
    activation = model_config.get("activation", config["training"]["activation"])

    print(f"\n{'='*60}")
    print(f"FER2013 训练 | 模型: {model_name} | 设备: {device}")
    print(f"  AMP: {config['training']['amp']} | 激活: {activation}")
    print(f"  LR: {model_config.get('learning_rate', '?')} | "
          f"Batch: {model_config.get('batch_size', '?')}")
    print(f"  种子: {seed}")
    print(f"{'='*60}\n")

    # 创建 DataLoaders
    train_loader, val_loader, test_loader, _ = create_dataloaders(config, model_name)
    print(f"  训练集: {len(train_loader.dataset)} | "
          f"验证集: {len(val_loader.dataset)} | "
          f"测试集: {len(test_loader.dataset)}")

    # 构建模型
    ModelClass = MODEL_REGISTRY[model_name]
    model = ModelClass(
        num_classes=config["data"]["num_classes"],
        dropout=model_config.get("dropout", 0.3),
        activation=activation,
    )

    # 构建训练器
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        config=config,
        model_name=model_name,
        device=device,
    )

    print(f"  参数量: {trainer.total_params:,}\n")

    # 恢复训练
    resume_path = None
    if args.resume:
        if args.resume == "auto":
            resume_path = find_resume_checkpoint(model_name, prefer="best")
            if resume_path is None:
                print("  auto: 未找到可用 checkpoint，从头训练")
            else:
                print(f"  auto: 选择 {resume_path.name}")
        else:
            resume_path = Path(args.resume)
            if not resume_path.exists():
                print(f"  checkpoint 不存在: {resume_path}，从头训练")
                resume_path = None

    if resume_path is not None:
        trainer.load_checkpoint(resume_path)

    # 诊断模式
    if args.diagnose:
        print(f"\n  诊断模式: {args.steps} 步（含 3 步预热）\n")
        timings = trainer.diagnose(num_steps=max(args.steps - 3, 5))
        return

    # 训练
    fit_start = time.time()
    history = trainer.fit(args.epochs)
    total = time.time() - fit_start

    # 输出摘要
    if history and history["val_acc"]:
        best_epoch = max(range(len(history["val_acc"])),
                        key=lambda i: history["val_acc"][i])
        print(f"\n{'='*60}")
        print(f"训练完成")
        print(f"  最佳 val_acc: {history['val_acc'][best_epoch]:.4f} "
              f"(第 {best_epoch + 1} 轮)")
        print(f"  总用时: {total:.1f}s")
        print(f"  日志: {trainer.log_dir / 'history.json'}")
        print(f"  最优模型: {trainer.save_dir / 'global_best.pth'}")
        print(f"{'='*60}\n")


if __name__ == "__main__":
    main()

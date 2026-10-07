"""
FER2013 模型训练 CLI

- 每次运行创建独立 run 目录：training/runs/<模型>/<run_id>/
- last.pth 为最新完整 epoch（--resume auto 自动选择）
- 启动前集中校验配置（非法值/未知键明确报错）
- 完成后用 tools/export_model.py 显式导出到推理目录

用法:
    python training/train.py --model micro_resnet --epochs 30
    python training/train.py --model mini_cnn --epochs 20 --amp
    python training/train.py --model vgg_lite --epochs 10 --resume auto
    python training/train.py --model micro_resnet --diagnose --steps 5
"""
import argparse
import sys
import time
from pathlib import Path

# 项目根目录（兼容从 project root 和 training/ 下运行）
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from data.dataloader import create_dataloaders
from training.checkpoint import find_resume_checkpoint
from training.trainer import (
    Trainer,
    load_config,
    set_seed,
)
from utils.comparison_check import FROZEN_PROTOCOL_PATH
from utils.config_validation import validate_config
from utils.formal_protocol import load_frozen_protocol
from utils.model_spec import build_model_from_spec, make_spec_from_config
from utils.stdio import ensure_utf8_stdio


def _load_frozen_protocol(path=FROZEN_PROTOCOL_PATH) -> dict:
    """T06：正式训练（--purpose formal）要求冻结协议文件存在且字段齐全。

    返回记录（含文件字节 SHA-256），写入 run_meta.frozen_protocol；
    正式实验准入判定时复核该 SHA 与当前文件一致（防止事后更换冻结内容）。
    """
    try:
        return load_frozen_protocol(path)
    except (ValueError, KeyError, TypeError) as e:
        raise SystemExit(f"正式训练需要完整冻结方案：{e}；见协议 §6") from e


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
        choices=["mini_cnn", "vgg_lite", "micro_resnet"],
        help="模型名称 (默认: micro_resnet)",
    )
    parser.add_argument("--epochs", type=int, default=30, help="本次训练轮数 (默认: 30)")
    parser.add_argument("--amp", action="store_true", default=None,
                        help="启用 AMP 混合精度训练（覆盖 config 中的 amp 设置）")
    parser.add_argument("--no-amp", action="store_true", default=None,
                        help="禁用 AMP（覆盖 config 中的 amp 设置）")
    parser.add_argument(
        "--resume", type=str, nargs="?", const="auto", default=None,
        help=("恢复训练：'auto' 自动选择最新 run 的 last.pth，或指定断点路径。"
              "不指定则从头训练（创建新 run）"),
    )
    parser.add_argument("--diagnose", action="store_true", help="仅运行性能诊断（不训练）")
    parser.add_argument(
        "--purpose", choices=["smoke", "formal"], default="smoke",
        help="run 用途声明（T06：默认 smoke=流程验证、非正式；formal 需冻结协议文件存在）",
    )
    parser.add_argument("--steps", type=int, default=10, help="诊断步数 (默认: 10，含 3 步预热)")
    parser.add_argument(
        "--config", type=str, default=None,
        help="训练配置文件路径（默认 configs/training_config.yaml；"
             "CE 基线用 configs/baseline_config.yaml）",
    )
    parser.add_argument("--seed", type=int, default=None, help="随机种子（覆盖 config 中的 seed）")
    parser.add_argument("--protocol", type=Path, default=FROZEN_PROTOCOL_PATH,
                        help="正式实验的冻结清单（configs/protocols/ 下的 JSON）")
    parser.add_argument("--lr", type=float, default=None, help="学习率覆盖")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size 覆盖")
    return parser.parse_args()


def main():
    ensure_utf8_stdio()
    args = parse_args()
    model_name = args.model

    config = load_config(args.config)

    # ---- CLI 覆盖（应用后再集中校验，保证日志与"实际生效配置"一致）----
    if args.amp and args.no_amp:
        raise SystemExit("错误: --amp 与 --no-amp 不能同时使用")
    if args.amp:
        config["training"]["amp"] = True
    elif args.no_amp:
        config["training"]["amp"] = False
    if args.seed is not None:
        config["seed"] = args.seed
    if args.lr is not None:
        config["models"][model_name]["learning_rate"] = args.lr
    if args.batch_size is not None:
        config["models"][model_name]["batch_size"] = args.batch_size

    # ---- 启动前集中校验（F13）----
    validate_config(config, model_name=model_name)

    # ---- T06：用途声明与冻结协议绑定（--purpose formal 需冻结文件存在）----
    frozen_protocol = _load_frozen_protocol(args.protocol) if args.purpose == "formal" else None

    seed = config["seed"]
    deterministic = bool(config["training"].get("cudnn_deterministic", False))
    set_seed(seed, deterministic=deterministic)

    # ---- 恢复解析（先确定断点，再按同一配置构建 DataLoader）----
    # 注：精确恢复要求恢复端的数据管线规格（workers/persistent/sampler）与断点
    # 完全一致；加载时自动校验，不满足会在加载前明确拒绝（S01）
    resume_path = None
    if args.resume:
        if args.resume == "auto":
            resume_path = find_resume_checkpoint(model_name)
            if resume_path is None:
                print("  auto: 未找到可续训断点（training/runs/ 下无 last.pth），从头训练")
            else:
                print(f"  auto: 选择 {resume_path}")
        else:
            resume_path = Path(args.resume)
            if not resume_path.exists():
                raise SystemExit(f"错误: checkpoint 不存在: {resume_path}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_config = config["models"][model_name]
    activation = model_config.get("activation", config["training"]["activation"])

    print(f"\n{'=' * 60}")
    print(f"FER2013 训练 | 模型: {model_name} | 设备: {device}")
    print(f"  AMP: {config['training']['amp']} | 激活: {activation} | "
          f"cudnn_deterministic: {deterministic}")
    print(f"  LR: {model_config.get('learning_rate', '?')} | "
          f"Batch: {model_config.get('batch_size', '?')}")
    print(f"  种子: {seed}")
    print(f"{'=' * 60}\n")

    # ---- 数据 ----
    # PrivateTest 不在训练启动时构建（PB02）：训练不使用 test_loader，
    # 统一评估入口按需加载（tools/evaluate_checkpoint.py / utils.evaluation）
    train_loader, val_loader, _, _ = create_dataloaders(config, model_name, include_test=False)
    print(f"  训练集: {len(train_loader.dataset)} | "
          f"验证集: {len(val_loader.dataset)}")
    print("  PrivateTest: 未构建（评估时按需加载）")

    # ---- 模型（统一构造入口 F01）----
    spec = make_spec_from_config(config, model_name)
    model = build_model_from_spec(spec)

    # 续训时沿用原 run 目录；新训练由 Trainer 生成新 run_id
    run_dir = resume_path.parent.parent if resume_path is not None else None

    # ---- 诊断模式（R03：独立 Trainer + 临时目录；不创建/触碰正式 runs）----
    if args.diagnose:
        import tempfile

        diagnose_dir = Path(tempfile.mkdtemp(prefix="fer2013_diagnose_"))
        print(f"\n  诊断模式: {args.steps} 步（含 3 步预热）")
        print(f"  隔离输出目录: {diagnose_dir}（不属于正式 runs）\n")
        diag_trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=None,
            config=config,
            model_name=model_name,
            device=device,
            class_counts=train_loader.dataset.class_counts,
            run_dir=diagnose_dir / "diagnose",
            run_meta_extra={"cli_args": vars(args), "purpose": "diagnose"},
            run_purpose="diagnose",
        )
        diag_trainer.diagnose(num_steps=max(args.steps - 3, 5))
        diag_trainer._persist_run_meta(status="diagnose_completed")
        return

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=None,
        config=config,
        model_name=model_name,
        device=device,
        class_counts=train_loader.dataset.class_counts,
        run_dir=run_dir,
        run_meta_extra={"cli_args": vars(args)},
        run_purpose=args.purpose,
        frozen_protocol=frozen_protocol,
    )

    print(f"  参数量: {trainer.total_params:,}\n")

    if resume_path is not None:
        trainer.load_checkpoint(resume_path)

    # ---- 训练 ----
    fit_start = time.time()
    history = trainer.fit(args.epochs)
    total = time.time() - fit_start

    if history and history["val_acc"]:
        best_epoch = max(range(len(history["val_acc"])), key=lambda i: history["val_acc"][i])
        print(f"\n{'=' * 60}")
        print("训练完成")
        print(f"  最佳 val_acc: {history['val_acc'][best_epoch]:.4f} (第 {best_epoch + 1} 轮)")
        print(f"  总用时: {total:.1f}s")
        print(f"  Run 目录: {trainer.run_dir}")
        print(f"  日志: {trainer.run_dir / 'history.json'}")
        if trainer.save_best:
            print(f"  最优模型: {trainer.checkpoints_dir / 'best.pth'}")
        print("  提示: 导出到推理目录请运行  python tools/export_model.py --help")
        print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()

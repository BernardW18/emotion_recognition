"""
统一评估入口的命令行包装（F10）

用法:
    python tools/evaluate_checkpoint.py --checkpoint <path> --split PrivateTest
    python tools/evaluate_checkpoint.py --checkpoint <path> --split PublicTest --device cuda
"""
import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.evaluation import evaluate_checkpoint  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="统一评估入口（指定 checkpoint + split）")
    parser.add_argument("--checkpoint", required=True, help="权重路径（必填，无内存状态依赖）")
    parser.add_argument("--split", default="PrivateTest",
                        choices=["Training", "PublicTest", "PrivateTest"], help="评估划分")
    parser.add_argument("--csv", default=None, help="数据集路径（默认 data/fer2013.csv）")
    parser.add_argument("--device", default="cpu", help="计算设备（默认 cpu，与历史评估口径一致）")
    parser.add_argument("--batch-size", type=int, default=64, help="推理批大小")
    parser.add_argument(
        "--output-dir", default=None,
        help="结果目录（默认 analysis/evaluations/<时间戳>_...）",
    )
    args = parser.parse_args()

    evaluate_checkpoint(
        args.checkpoint,
        split=args.split,
        csv_path=args.csv,
        device=args.device,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()

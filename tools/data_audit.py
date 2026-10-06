"""
数据审计工具 — 官方协议披露（F03）

计算并输出:
  - CSV 文件 SHA-256（应与冻结值一致）
  - 官方三划分行数与各自指纹
  - 像素完全相同的重复组统计（组数 / 超出唯一图像数的记录数 / 标签冲突组数）
  - 跨划分重叠（Training↔PublicTest / Training↔PrivateTest / PublicTest↔PrivateTest）
  - 敏感性分析：排除 PrivateTest∩Training 的完全重复记录后，基于现有权重
    预测（analysis/evaluations 下的 predictions.csv）重算指标

规范化规则: pixels 字符串按空白拆分后以单空格连接（去除多余空白），再做 SHA-256；
与评估轮的检查口径一致。本脚本不修改原始 CSV。

输出: analysis/data_audit.json（+ 控制台摘要）
用法: python tools/data_audit.py
"""
import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from utils.model_spec import file_sha256

OFFICIAL_SPLITS = ("Training", "PublicTest", "PrivateTest")
CROSS_PAIRS = (
    ("Training", "PublicTest"),
    ("Training", "PrivateTest"),
    ("PublicTest", "PrivateTest"),
)
DEFAULT_CSV = PROJECT_ROOT / "data" / "fer2013.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "analysis" / "data_audit.json"
EXPECTED_CSV_SHA256 = "3b8d9617d1017f34733c8f2474d7784c563ce86c40a86ac12c2d37cc968f871b"


def _norm_hash(pixel_string: str) -> str:
    """规范化像素字符串（去多余空白）后的 SHA-256。"""
    return hashlib.sha256(" ".join(pixel_string.split()).encode()).hexdigest()


def _sensitivity_analysis(df: pd.DataFrame) -> dict | None:
    """排除 PrivateTest∩Training 完全重复记录后重算指标（用已有预测 CSV）。"""
    training_hashes = set(df.loc[df["Usage"] == "Training", "pix_hash"])
    pt = df[df["Usage"] == "PrivateTest"]
    dup_global_rows = set(pt.index[pt["pix_hash"].isin(training_hashes)])
    if not dup_global_rows:
        return None

    eval_dir = PROJECT_ROOT / "analysis" / "evaluations"
    results = {}
    for model in ("mini_cnn", "vgg_lite", "micro_resnet"):
        pred_path = eval_dir / f"legacy_{model}_PrivateTest" / "predictions.csv"
        if not pred_path.exists():
            continue
        pred_df = pd.read_csv(pred_path)
        keep = ~pred_df["row_index"].isin(dup_global_rows)
        labels = pred_df.loc[keep, "label"]
        preds = pred_df.loc[keep, "pred"]
        results[model] = {
            "n_after_exclusion": int(keep.sum()),
            "accuracy": float(accuracy_score(labels, preds)),
            "macro_f1": float(f1_score(labels, preds, average="macro")),
        }
    if not results:
        return None
    return {
        "note": (
            "仅作敏感性分析：不改变官方划分结果；本表不代表去重重训结果，"
            "不能与官方成绩混用"
        ),
        "excluded_metric": "PrivateTest 中与 Training 像素完全相同的记录",
        "excluded_rows_in_private_test": len(dup_global_rows),
        "results": results,
    }


def main():
    parser = argparse.ArgumentParser(description="FER2013 官方协议数据审计（F03）")
    parser.add_argument("--csv", type=str, default=str(DEFAULT_CSV))
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()

    csv_path = Path(args.csv)
    df = pd.read_csv(csv_path)

    print(f"读取 {csv_path} ({len(df)} 行)")
    csv_sha = file_sha256(csv_path)

    split_counts = {s: int((df["Usage"] == s).sum()) for s in OFFICIAL_SPLITS}
    print(f"官方划分行数: {split_counts}")

    # 像素规范化哈希（35887 行，一次性）
    df["pix_hash"] = [_norm_hash(s) for s in df["pixels"]]
    groups = df.groupby("pix_hash")
    sizes = groups.size()
    dup_groups = sizes[sizes > 1]
    n_dup_groups = int(len(dup_groups))
    n_excess = int((dup_groups - 1).sum())
    conflict_counts = groups["emotion"].nunique()
    n_conflict_groups = int((conflict_counts > 1).sum())
    print(f"重复组: {n_dup_groups} 组；超出唯一图像数的记录: {n_excess} 条；"
          f"标签冲突组: {n_conflict_groups} 组")

    cross = {}
    for a, b in CROSS_PAIRS:
        hashes_a = set(df.loc[df["Usage"] == a, "pix_hash"])
        mask_b = df["Usage"] == b
        hashes_b = df.loc[mask_b, "pix_hash"]
        inter = set(hashes_b) & hashes_a
        cross[f"{a}__{b}"] = {
            "unique_image_hashes": len(inter),
            "rows_in_second_split": int(hashes_b.isin(inter).sum()),
        }
        print(f"跨划分重叠 {a}↔{b}: {len(inter)} 个图像哈希，"
              f"{b} 中 {cross[f'{a}__{b}']['rows_in_second_split']} 条记录")

    sensitivity = _sensitivity_analysis(df)

    result = {
        "checked_on": datetime.now().isoformat(),
        "csv_path": str(csv_path),
        "csv_sha256": csv_sha,
        "csv_sha256_matches_frozen": csv_sha == EXPECTED_CSV_SHA256,
        "split_counts": split_counts,
        "duplicate_pixel_stats": {
            "duplicate_groups": n_dup_groups,
            "excess_rows_beyond_unique": n_excess,
            "label_conflict_groups": n_conflict_groups,
        },
        "cross_split_overlap": cross,
        "sensitivity_analysis": sensitivity,
        "method": {
            "pixel_hash": "sha256(规范化像素字符串：空白拆分后单空格连接)",
            "duplicate_definition": "规范化像素字符串完全相同",
            "scope_note": "仅检查完全重复；不覆盖近重复或人员身份重叠",
        },
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print()
    print(f"SHA-256 与冻结值一致: {result['csv_sha256_matches_frozen']}")
    if sensitivity:
        print(
            "敏感性分析（排除 PrivateTest∩Training "
            f"{sensitivity['excluded_rows_in_private_test']} 条）:"
        )
        for model, m in sensitivity["results"].items():
            print(f"  {model}: n={m['n_after_exclusion']} "
                  f"acc={m['accuracy']:.4f} macro_f1={m['macro_f1']:.4f}")
    print(f"输出: {out_path}")


if __name__ == "__main__":
    main()

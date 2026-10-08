"""
模型导出工具 — 显式导出断点到推理目录并登记来源清单（F05）

导出行为：
  - 复制指定 checkpoint 到 inference/saved_models/<name>.pth
  - 更新 inference/saved_models/export_manifest.json：记录 source checkpoint、
    run_id、SHA-256、model_spec、指标（导出的模型都能反查唯一 run 与权重）
  - 默认不覆盖已存在的同名文件（需 force=True 显式覆盖，避免意外换权重）

函数用法:
    from tools.export_model import export_checkpoint
    entry = export_checkpoint("training/runs/micro_resnet/<run_id>/checkpoints/best.pth")

命令行用法:
    python tools/export_model.py --checkpoint <path> [--name NAME] [--force] [--default]
    python tools/export_model.py --list
"""
import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from training.checkpoint import write_json_atomic
from utils.model_spec import file_sha256, resolve_spec_from_checkpoint
from utils.stdio import ensure_utf8_stdio

SAVED_DIR = PROJECT_ROOT / "inference" / "saved_models"
MANIFEST_PATH = SAVED_DIR / "export_manifest.json"


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def _load_manifest() -> dict:
    if MANIFEST_PATH.exists():
        try:
            with open(MANIFEST_PATH, encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict) and isinstance(data.get("exports"), list):
                return data
        except Exception:
            pass
    return {"exports": []}


def export_checkpoint(
    checkpoint_path, *, name: str | None = None, force: bool = False, make_default: bool = False
) -> dict:
    """
    导出单个断点到推理目录并登记清单。

    Args:
        checkpoint_path: 源断点路径
        name: 导出名称（不含 .pth；默认 <模型名>_<run_id>）
        force: 覆盖已存在的同名文件
        make_default: 将本次导出设为推理应用默认权重（其他导出不改变默认项）

    Returns:
        manifest 中的 entry dict

    Raises:
        FileNotFoundError / ValueError / FileExistsError / RuntimeError
    """
    src = Path(checkpoint_path)
    if not src.exists():
        raise FileNotFoundError(f"断点不存在: {src}")

    try:
        ckpt = torch.load(src, map_location="cpu", weights_only=False)
    except Exception as e:
        raise ValueError(f"断点无法读取: {src}\n{e}") from e
    if not isinstance(ckpt, dict) or "model_state_dict" not in ckpt:
        raise ValueError(f"文件不是有效断点（缺少 model_state_dict）: {src}")

    spec, notes = resolve_spec_from_checkpoint(ckpt)
    model_name = ckpt.get("model_name") or spec.model_name
    run_id = ckpt.get("run_id") or "legacy"
    is_legacy = "model_spec" not in ckpt
    out_name = name or f"{model_name}_{run_id}"

    SAVED_DIR.mkdir(parents=True, exist_ok=True)
    dst = SAVED_DIR / f"{out_name}.pth"
    if dst.exists() and not force:
        raise FileExistsError(
            f"{dst} 已存在（不自动覆盖）。使用 force=True 显式覆盖，或指定其他名称。"
        )

    shutil.copy2(src, dst)
    exported_sha = file_sha256(dst)
    source_sha = file_sha256(src)
    if exported_sha != source_sha:
        raise RuntimeError("复制后 SHA-256 不一致，导出中止（请检查磁盘）")

    metrics = {
        "epoch": ckpt.get("epoch"),
        "val_acc": ckpt.get("val_acc"),
        "best_val_acc": (ckpt.get("best") or {}).get("val_acc", ckpt.get("best_val_acc")),
    }

    manifest = _load_manifest()
    entry = {
        "file": dst.name,
        "model_name": model_name,
        "run_id": run_id,
        "legacy": is_legacy,
        "format_version": ckpt.get("format_version"),
        "source_checkpoint": _rel(src),
        "source_sha256": source_sha,
        "exported_sha256": exported_sha,
        "exported_at": datetime.now().isoformat(),
        "model_spec": spec.to_dict(),
        "spec_notes": notes,
        "metrics": metrics,
    }
    exports = [e for e in manifest["exports"] if e.get("file") != dst.name]
    exports.append(entry)
    manifest["exports"] = exports
    if make_default:
        manifest["default_checkpoint"] = dst.name
    manifest["updated_at"] = datetime.now().isoformat()
    write_json_atomic(MANIFEST_PATH, manifest)
    return entry


def list_exports() -> None:
    manifest = _load_manifest()
    exports = manifest["exports"]
    if not exports:
        print("export_manifest.json 为空（尚无导出记录）")
        return
    print(f"{'文件':<44} {'run_id':<24} {'val_acc':>9} {'导出时间':<20} 来源")
    print("-" * 130)
    for e in exports:
        metrics = e.get("metrics") or {}
        acc = metrics.get("best_val_acc") or metrics.get("val_acc") or 0.0
        legacy = " [legacy]" if e.get("legacy") else ""
        print(f"{e.get('file', '?'):<44} {str(e.get('run_id', '?')):<24} "
              f"{acc * 100:>8.2f}% {str(e.get('exported_at', ''))[:19]:<20} "
              f"{e.get('source_checkpoint', '?')}{legacy}")


def main():
    ensure_utf8_stdio()
    parser = argparse.ArgumentParser(description="导出模型断点到推理目录（附来源清单）")
    parser.add_argument("--checkpoint", type=str, default=None, help="源断点路径")
    parser.add_argument("--name", type=str, default=None,
                        help="导出名称（不含 .pth；默认 <模型名>_<run_id>）")
    parser.add_argument("--force", action="store_true", help="覆盖已存在的同名导出")
    parser.add_argument("--default", action="store_true", help="将本次导出设为推理应用默认权重")
    parser.add_argument("--list", action="store_true", help="列出现有导出与来源")
    args = parser.parse_args()

    if args.list:
        list_exports()
        return

    if not args.checkpoint:
        parser.error("需要 --checkpoint <路径>（或使用 --list）")

    try:
        entry = export_checkpoint(
            args.checkpoint, name=args.name, force=args.force, make_default=args.default
        )
    except (FileNotFoundError, ValueError, FileExistsError, RuntimeError) as e:
        raise SystemExit(f"错误: {e}") from e

    print("导出完成：")
    print(f"  文件: {SAVED_DIR / entry['file']}")
    print(f"  来源: {entry['source_checkpoint']} "
          f"(run={entry['run_id']}, legacy={entry['legacy']})")
    print(f"  SHA-256: {entry['exported_sha256']}")
    m = entry["metrics"]
    print(f"  指标: epoch={m['epoch']} val_acc={m['val_acc']} best_val_acc={m['best_val_acc']}")
    print(f"  清单: {MANIFEST_PATH}")


if __name__ == "__main__":
    main()

"""
FER2013 像素缓存（PB02）— 从冻结 CSV 生成可验证的紧凑 uint8 缓存。

设计要点：
  - 内容：三官方划分的 (N,48,48) uint8 像素、int64 标签、int64 原行号（CSV 数据行序 0 基）。
    加载时按需转换为 float32（x/255），与旧逐行解析路径逐位一致（uint8→float32 精确）。
  - 校验（v2 / T04）：缓存键 = CACHE_VERSION + CSV SHA-256；每个数据文件记录
    SHA-256、shape、dtype 与 stat 签名（size/mtime_ns）。meta 是唯一权威代际标记
    （最后原子提交）；文件写入用 tmp + os.replace，目标被占用（Windows mmap 锁）
    时自动轮转文件名并在 meta 中记录实际文件名。
  - 加载：三份数组均为只读（x 为只读 mmap，y/rows 加载后置 write=False）；
    进程内按 (cache_dir, split) 惰性打开，多进程共享 page cache、不随 pickle 传递。
    热命中与 worker 侧打开均做 stat 签名快查（数据源变化 → 完整校验/重建，
    训练中的 worker 侧明确拒绝继续）；SHA-256 仅在非热命中路径全量计算。
  - 失配/损坏：明确重建（损坏文件触发重建并在重建后重新校验）；损坏负例必失效。
  - 解析（T04）：像素先以 float64 校验（数量 / 整数性 / 0–255 值域）再转 uint8；
    非法值明确失败（不做静默截断）；标签校验合法域 0–6。
  - 该缓存只是可重建派生物，不替代原 CSV、不改原行号；位于忽略目录（data/cache/）。

用法：
    from data.pixel_cache import load_or_build, default_cache_dir
    cache = load_or_build("data/fer2013.csv")
    x_u8, y, rows = cache.split_arrays("Training")   # x_u8 为只读 memmap (N,48,48) uint8
"""

from __future__ import annotations

import contextlib
import gc
import hashlib
import json
import os
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import pandas as pd

__all__ = [
    "CACHE_VERSION",
    "OFFICIAL_SPLITS",
    "PixelCacheError",
    "PixelCacheCorruptedError",
    "default_cache_dir",
    "parse_pixels_column",
    "build_cache",
    "load_or_build",
    "PixelCache",
]

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CACHE_VERSION = "uint8-v2"   # v2（T04）：meta 绑定每文件 stat 签名（size/mtime_ns）
OFFICIAL_SPLITS = ("Training", "PublicTest", "PrivateTest")
PIXEL_SHAPE = (48, 48)
_NUM_CLASSES = 7   # FER2013 官方 7 类（标签合法值 0–6）
PIXELS_PER_IMAGE = PIXEL_SHAPE[0] * PIXEL_SHAPE[1]


class PixelCacheError(RuntimeError):
    """缓存使用错误（缺失且不允许重建等）。"""


class PixelCacheCorruptedError(PixelCacheError):
    """缓存文件缺失/SHA 不符/shape 不符等，需要重建。"""


def default_cache_dir() -> Path:
    """默认缓存目录：data/cache/fer2013_<version>/（已加入 .gitignore）。"""
    return PROJECT_ROOT / "data" / "cache" / f"fer2013_{CACHE_VERSION}"


# ============================================================
# SHA-256（进程内按 (size, mtime_ns) 缓存；文件变化自动重算）
# ============================================================
_SHA_CACHE: dict[tuple[str, int, int], str] = {}


def file_sha256_cached(path) -> str:
    path = Path(path)
    st = path.stat()
    key = (str(path), int(st.st_size), int(st.st_mtime_ns))
    cached = _SHA_CACHE.get(key)
    if cached is not None:
        return cached
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    digest = h.hexdigest()
    _SHA_CACHE[key] = digest
    return digest


# ============================================================
# 像素解析（构建缓存时一次性执行）
# ============================================================
def parse_pixels_column(pixel_strings) -> np.ndarray:
    """
    将像素字符串序列解析为 (N, 48, 48) uint8 数组。

    T04：先以 float32 解析并按行校验（数量 / 整数性 / 0–255 值域），任何非法值
    明确报错（带行位置），再转换为 uint8——避免"256 静默截断为 0"类错误。
    与旧 float32 路径的关系：合法值 0–255 → uint8 → float32 精确表示，
    除以 255.0 后逐位一致。
    """
    n = len(pixel_strings)
    out = np.empty((n, *PIXEL_SHAPE), dtype=np.uint8)
    for i, s in enumerate(pixel_strings):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                vals = np.fromstring(s, dtype=np.float64, sep=" ")
        except (ValueError, Warning) as e:
            raise ValueError(f"像素解析失败：第 {i} 行含非法词元/尾部垃圾") from e
        if not bool(np.isfinite(vals).all()):
            raise ValueError(f"像素解析失败：第 {i} 行存在非有限值")
        if vals.shape != (PIXELS_PER_IMAGE,):
            raise ValueError(
                f"像素解析失败：第 {i} 行得到 {vals.shape[0] if vals.ndim else '?'} 个值"
                f"（期望 {PIXELS_PER_IMAGE}）"
            )
        non_int = vals != np.floor(vals)
        if bool(np.any(non_int)):
            bad = int(np.argmax(non_int))
            raise ValueError(
                f"像素解析失败：第 {i} 行第 {bad} 个值不是整数（{vals[bad]!r}）"
            )
        vmin, vmax = float(vals.min()), float(vals.max())
        if vmin < 0.0 or vmax > 255.0:
            raise ValueError(
                f"像素解析失败：第 {i} 行存在超出 0–255 的值（min={vmin}, max={vmax}）"
            )
        out[i] = vals.reshape(PIXEL_SHAPE).astype(np.uint8)
    return out


def _parse_split_dataframe(df: pd.DataFrame, split: str) -> dict:
    """按官方 Usage 提取单个划分并解析。行号 = CSV 数据行顺序（0 基，等于 df 索引）。"""
    mask = (df["Usage"] == split).to_numpy()
    if not mask.any():
        raise ValueError(f"划分 {split} 为空，无法构建缓存")
    sub = df.loc[mask]
    x = parse_pixels_column(sub["pixels"].values)
    raw_labels = pd.to_numeric(sub["emotion"], errors="raise").to_numpy(dtype=np.float64)
    valid = (
        np.isfinite(raw_labels) & (raw_labels == np.floor(raw_labels))
        & (raw_labels >= 0) & (raw_labels < _NUM_CLASSES)
    )
    if not bool(valid.all()):
        row = sub.index[int(np.flatnonzero(~valid)[0])]
        raise ValueError(f"划分 {split} 原行 {row} 含非法标签，须为 0–6 的有限整数")
    y = raw_labels.astype(np.int64)
    if y.size and (int(y.min()) < 0 or int(y.max()) >= _NUM_CLASSES):
        raise ValueError(
            f"划分 {split} 含非法标签（期望 0–{_NUM_CLASSES - 1}，"
            f"实际 {int(y.min())}–{int(y.max())}），拒绝构建缓存"
        )
    rows = sub.index.to_numpy(dtype=np.int64)
    return {"x": x, "y": y, "rows": rows}


# ============================================================
# 构建
# ============================================================
def _atomic_write_json(path: Path, obj: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def _replace_or_rotate(tmp_path: Path, final_path: Path) -> str:
    """os.replace；目标被占用（如其他进程 mmap 中，Windows 下为 PermissionError）
    时轮转为带时间戳的新文件名；其他 OSError（真错误）直接抛出。"""
    try:
        os.replace(tmp_path, final_path)
        return final_path.name
    except PermissionError:
        rotated = final_path.with_name(
            f"{final_path.stem}.{int(time.time() * 1000)}{final_path.suffix}"
        )
        os.replace(tmp_path, rotated)
        return rotated.name


def _write_split_files(cache_dir: Path, split: str, data: dict) -> dict:
    """写出单个划分的 npy 文件（tmp+replace），返回 meta 片段。"""
    out = {}
    for kind in ("x", "y", "rows"):
        arr = data[kind]
        # A new generation never overwrites files held by an active dataset/worker.
        final = cache_dir / f"{split}_{kind}_{uuid4().hex}.npy"
        tmp = final.with_name(final.name + ".tmp")
        # 注意：np.save 对不以 .npy 结尾的路径会自动追加后缀，需用 file object 写入
        with open(tmp, "wb") as f:
            np.save(f, arr)
        actual_name = _replace_or_rotate(tmp, final)
        st = (cache_dir / actual_name).stat()
        out[kind] = {
            "file": actual_name,
            "sha256": file_sha256_cached(cache_dir / actual_name),
            # T04：stat 签名（热命中的廉价完整性检查；替换/改写会改变 size/mtime_ns）
            "size": int(st.st_size),
            "mtime_ns": int(st.st_mtime_ns),
            "shape": list(arr.shape),
            "dtype": str(arr.dtype),
        }
    return out


def _cleanup_stale_files(cache_dir: Path, meta: dict) -> None:
    """best-effort 清理不被当前 meta 引用的旧数据文件（被锁的文件跳过）。"""
    referenced = {"meta.json"}
    for sinfo in meta.get("splits", {}).values():
        for kind in ("x", "y", "rows"):
            referenced.add(sinfo[kind]["file"])
    for p in cache_dir.glob("*.npy*"):
        if p.name in referenced:
            continue
        with contextlib.suppress(OSError):
            p.unlink()


def build_cache(csv_path, cache_dir=None, *, quiet: bool = False) -> dict:
    """
    从 CSV 构建缓存（读 CSV → 三划分解析 → 写 npy → 原子提交 meta.json）。

    Returns:
        meta dict（已写入 cache_dir/meta.json）
    """
    csv_path = Path(csv_path)
    cache_dir = Path(cache_dir) if cache_dir is not None else default_cache_dir()
    cache_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    df = pd.read_csv(csv_path)
    if "Usage" not in df.columns or "emotion" not in df.columns or "pixels" not in df.columns:
        raise ValueError("数据集缺少 Usage/emotion/pixels 列，无法构建像素缓存")
    unknown = sorted(set(df["Usage"].unique()) - set(OFFICIAL_SPLITS))
    if unknown:
        raise ValueError(f"Usage 含未知取值 {unknown}，无法确定划分清单")

    splits_data = {}
    for split in OFFICIAL_SPLITS:
        splits_data[split] = _parse_split_dataframe(df, split)

    total_n = sum(len(v["y"]) for v in splits_data.values())
    if total_n != len(df):
        raise ValueError(
            f"存在未归入官方划分的行（划分合计 {total_n} != CSV {len(df)} 行），拒绝构建缓存"
        )

    meta: dict[str, Any] = {
        "cache_version": CACHE_VERSION,
        "created_at": datetime.now().isoformat(),
        "csv_path": str(csv_path),
        "csv_sha256": file_sha256_cached(csv_path),
        "total_rows": int(len(df)),
        "layout": {
            "pixel_shape": list(PIXEL_SHAPE),
            "pixel_dtype": "uint8",
            "label_dtype": "int64",
            "row_dtype": "int64",
        },
        "splits": {},
    }
    for split, data in splits_data.items():
        entry = _write_split_files(cache_dir, split, data)
        entry["n"] = int(len(data["y"]))
        entry["counts"] = [int((data["y"] == i).sum()) for i in range(7)]
        meta["splits"][split] = entry

    _atomic_write_json(cache_dir / "meta.json", meta)
    _cleanup_stale_files(cache_dir, meta)

    if not quiet:
        dt = time.perf_counter() - t0
        sizes_mib = int(meta["total_rows"]) * PIXELS_PER_IMAGE / 1024 ** 2
        print(
            f"[pixel_cache] 缓存已构建（{dt:.2f}s）：{cache_dir} | "
            f"uint8 像素合计约 {sizes_mib:.2f} MiB（{meta['total_rows']} 行）"
        )
    return meta


# ============================================================
# 加载与校验
# ============================================================
_MMAP_CACHE: dict[tuple[str, str], dict] = {}


def _stat_signature_matches(path: Path, entry_kind: dict) -> bool:
    """
    T04 廉价完整性检查：文件 size/mtime_ns 与 meta 记录一致。

    - v2 缓存必含签名；缺签名记录（异常情形）视作不匹配（转完整 SHA/重建路径）
    - 文件缺失/不可 stat → False
    """
    try:
        st = path.stat()
    except OSError:
        return False
    if "size" not in entry_kind or "mtime_ns" not in entry_kind:
        return False
    return bool(
        st.st_size == entry_kind["size"] and st.st_mtime_ns == entry_kind["mtime_ns"]
    )


def _close_cached_mmaps(cache_dir: Path) -> None:
    """关闭并移除该缓存目录的进程内 mmap 引用（重建前调用；Windows 文件锁友好）。"""
    keys = [k for k in _MMAP_CACHE if k[0] == str(cache_dir)]
    for k in keys:
        _MMAP_CACHE.pop(k, None)
    if keys:
        gc.collect()


def _meta_invalid_reason(meta: dict, cache_dir: Path, csv_path: Path) -> str | None:
    """meta 级校验（不含数据文件 SHA）：不合格返回原因字符串，合格返回 None。"""
    if not isinstance(meta, dict):
        return "meta.json 不可解析"
    if meta.get("cache_version") != CACHE_VERSION:
        return f"缓存版本不符（{meta.get('cache_version')!r} != {CACHE_VERSION!r}）"
    for split in OFFICIAL_SPLITS:
        entry = meta.get("splits", {}).get(split)
        if not isinstance(entry, dict):
            return f"meta 缺少划分 {split}"
        for kind in ("x", "y", "rows"):
            if not isinstance(entry.get(kind), dict) or not entry[kind].get("file"):
                return f"meta 划分 {split} 缺少 {kind} 文件记录"
    try:
        csv_sha = file_sha256_cached(csv_path)
    except OSError as e:
        return f"CSV 不可读: {e}"
    if meta.get("csv_sha256") != csv_sha:
        return "CSV SHA-256 不匹配（数据文件已变化或缓存来源不同）"
    return None


class PixelCache:
    """已校验的像素缓存句柄；split_arrays 为只读 mmap 数组（惰性打开，进程内共享）。"""

    def __init__(self, csv_path, cache_dir, meta: dict, *, quiet: bool = False):
        self.csv_path = Path(csv_path)
        self.cache_dir = Path(cache_dir)
        self.meta = meta
        self._quiet = quiet

    # ---- 概览 ----
    @property
    def total_rows(self) -> int:
        return int(self.meta["total_rows"])

    def split_n(self, split: str) -> int:
        return int(self.meta["splits"][split]["n"])

    # ---- 数据访问 ----
    def _load_split_checked(self, split: str) -> tuple:
        entry = self.meta["splits"][split]
        paths = {}
        for kind in ("x", "y", "rows"):
            p = self.cache_dir / entry[kind]["file"]
            if not p.exists():
                raise PixelCacheCorruptedError(f"缓存文件缺失: {p.name}")
            # T04：先做 stat 签名快查（已被替换/改写 → 直接重建，不必先算 SHA）
            if not _stat_signature_matches(p, entry[kind]):
                raise PixelCacheCorruptedError(
                    f"缓存文件 stat 签名不符（已被替换或修改）: {p.name}"
                )
            if file_sha256_cached(p) != entry[kind]["sha256"]:
                raise PixelCacheCorruptedError(f"缓存文件 SHA-256 不符（已损坏）: {p.name}")
            paths[kind] = p

        x = np.load(paths["x"], mmap_mode="r")
        y = np.load(paths["y"])
        rows = np.load(paths["rows"])
        # T04：y/rows 一律只读（防止就地写入污染进程内共享缓存）
        y.setflags(write=False)
        rows.setflags(write=False)
        exp = {k: entry[k] for k in ("x", "y", "rows")}
        if tuple(x.shape) != tuple(exp["x"]["shape"]) or str(x.dtype) != exp["x"]["dtype"]:
            raise PixelCacheCorruptedError(
                f"划分 {split} 像素 shape/dtype 不符: {x.shape}/{x.dtype}"
            )
        if tuple(y.shape) != tuple(exp["y"]["shape"]) or str(y.dtype) != exp["y"]["dtype"]:
            raise PixelCacheCorruptedError(
                f"划分 {split} 标签 shape/dtype 不符: {y.shape}/{y.dtype}"
            )
        if (
            tuple(rows.shape) != tuple(exp["rows"]["shape"])
            or str(rows.dtype) != exp["rows"]["dtype"]
        ):
            raise PixelCacheCorruptedError(
                f"划分 {split} 行号 shape/dtype 不符: {rows.shape}/{rows.dtype}"
            )
        return x, y, rows

    def split_arrays(self, split: str) -> tuple:
        """
        返回 (x_uint8_mmap, y_int64, rows_int64)。

        - x 为只读 memmap（进程内按 (cache_dir, split) 缓存，多进程共享 page cache）
        - 首次访问校验文件 SHA-256/shape/dtype；损坏时明确重建后重新校验
        - 不返回可写视图：只读数组的 slice 为只读，增强不得污染缓存
        """
        if split not in self.meta.get("splits", {}):
            raise KeyError(f"未知划分 {split!r}（应为 {OFFICIAL_SPLITS}）")
        key = (str(self.cache_dir), split)
        cached = _MMAP_CACHE.get(key)
        if cached is not None and cached["file_x"] == self.meta["splits"][split]["x"]["file"]:
            # T04：热命中同样做三份文件的 stat 签名快查；变化则放弃引用并走完整校验/重建
            entry = self.meta["splits"][split]
            if all(
                _stat_signature_matches(self.cache_dir / entry[k]["file"], entry[k])
                for k in ("x", "y", "rows")
            ):
                return cached["x"], cached["y"], cached["rows"]
            _MMAP_CACHE.pop(key, None)
            gc.collect()
        try:
            x, y, rows = self._load_split_checked(split)
        except PixelCacheCorruptedError as e:
            if not self._quiet:
                print(f"[pixel_cache] {e}；自动重建缓存...")
            self._rebuild()
            x, y, rows = self._load_split_checked(split)
        _MMAP_CACHE[key] = {
            "file_x": self.meta["splits"][split]["x"]["file"],
            "x": x, "y": y, "rows": rows,
        }
        return x, y, rows

    def _rebuild(self) -> None:
        _close_cached_mmaps(self.cache_dir)
        self.meta = build_cache(self.csv_path, self.cache_dir, quiet=self._quiet)

    def close(self) -> None:
        _close_cached_mmaps(self.cache_dir)


def check_cache_ref(cache_ref) -> None:
    """Cheap batch-boundary protection for immutable files, CSV and meta generation."""
    cache_dir, _split, entry = cache_ref
    for kind in ("x", "y", "rows"):
        if not _stat_signature_matches(Path(cache_dir) / entry[kind]["file"], entry[kind]):
            raise PixelCacheCorruptedError(f"缓存 {kind} 已变化，拒绝继续使用")
    for record in entry.get("source_signatures", []):
        if not _stat_signature_matches(Path(record["path"]), record):
            raise PixelCacheCorruptedError("CSV/meta 代际已变化，拒绝继续使用")


def get_memmap_split(cache_ref) -> tuple:
    """
    worker/dataset 侧按索引访问所需的轻量打开：（cache_dir, split, entry）→ (x, y, rows)。

    - 进程内惰性打开（每进程一份只读 mmap；不重复主进程已做过的文件 SHA 校验）
    - 打开后仍做 shape/dtype 轻量检查；文件缺失/损坏会抛出 PixelCacheCorruptedError
    """
    cache_dir, split, entry = cache_ref
    check_cache_ref(cache_ref)
    key = (str(cache_dir), str(split))
    hit = _MMAP_CACHE.get(key)
    if hit is not None and hit["file_x"] == entry["x"]["file"]:
        # T04：命中同样做 stat 签名快查（数据源变化 → 拒绝继续，避免读到被替换的数据）
        return hit["x"], hit["y"], hit["rows"]

    p = Path(cache_dir)
    for kind in ("x", "y", "rows"):
        fp = p / entry[kind]["file"]
        if not fp.exists():
            raise PixelCacheCorruptedError(f"缓存文件缺失（worker 侧）: {fp.name}")
        if not _stat_signature_matches(fp, entry[kind]):
            raise PixelCacheCorruptedError(
                f"缓存文件 stat 签名不符（数据源已变化，拒绝继续使用）: {fp.name}"
            )
        if file_sha256_cached(fp) != entry[kind]["sha256"]:
            raise PixelCacheCorruptedError(f"缓存文件 SHA-256 不符（worker 侧）: {fp.name}")
    try:
        x = np.load(p / entry["x"]["file"], mmap_mode="r")
        y = np.load(p / entry["y"]["file"])
        rows = np.load(p / entry["rows"]["file"])
    except OSError as e:
        raise PixelCacheCorruptedError(f"打开缓存文件失败（{cache_dir} / {split}）: {e}") from e
    if tuple(x.shape) != tuple(entry["x"]["shape"]) or str(x.dtype) != entry["x"]["dtype"]:
        raise PixelCacheCorruptedError(
            f"缓存像素 shape/dtype 异常: {x.shape}/{x.dtype} "
            f"!= {entry['x']['shape']}/{entry['x']['dtype']}"
        )
    if tuple(y.shape) != tuple(entry["y"]["shape"]) or str(y.dtype) != entry["y"]["dtype"]:
        raise PixelCacheCorruptedError(f"缓存标签 shape/dtype 异常: {y.shape}/{y.dtype}")
    if (tuple(rows.shape) != tuple(entry["rows"]["shape"])
            or str(rows.dtype) != entry["rows"]["dtype"]):
        raise PixelCacheCorruptedError(f"缓存行号 shape/dtype 异常: {rows.shape}/{rows.dtype}")
    # T04：y/rows 一律只读
    y.setflags(write=False)
    rows.setflags(write=False)
    _MMAP_CACHE[key] = {
        "file_x": entry["x"]["file"], "x": x, "y": y, "rows": rows,
    }
    return x, y, rows


def load_or_build(
    csv_path,
    cache_dir=None,
    *,
    rebuild_on_mismatch: bool = True,
    quiet: bool = False,
) -> PixelCache:
    """
    加载像素缓存；缺失/失配/损坏时（rebuild_on_mismatch=True）明确重建。

    - meta 级校验：缓存版本 + CSV SHA-256（每次调用核验，读取有缓存不重复计算）
    - 数据文件级校验：在 split_arrays 首次访问时按文件 SHA-256 校验
    """
    csv_path = Path(csv_path)
    cache_dir = Path(cache_dir) if cache_dir is not None else default_cache_dir()

    meta = None
    reason = None
    meta_path = cache_dir / "meta.json"
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as e:
            meta = None
            reason = f"meta.json 读取失败: {e}"
        if meta is not None:
            reason = _meta_invalid_reason(meta, cache_dir, csv_path)
            if reason is not None:
                meta = None
    else:
        reason = f"缓存不存在: {cache_dir}"

    if meta is None:
        if not rebuild_on_mismatch:
            raise PixelCacheError(f"像素缓存不可用（{reason}），且未允许重建")
        if not quiet:
            print(f"[pixel_cache] 缓存无效（{reason}），重建...")
        _close_cached_mmaps(cache_dir)
        meta = build_cache(csv_path, cache_dir, quiet=quiet)

    return PixelCache(csv_path, cache_dir, meta, quiet=quiet)

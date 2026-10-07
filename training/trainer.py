"""
共享训练模块 — run 隔离 / 早停 / 完整续训 / 梯度累积修复（F05 / F08 / F09 / F13 / F14）

与旧版的关键差异:
  - 每次训练建立独立 run 目录: training/runs/<model_name>/<run_id>/
    （内含 config_effective.yaml、run_meta.json、history.json、checkpoints/）
  - last.pth = 最新完整 epoch 的完整状态（续训入口，每个 epoch 结束保存）
  - best.pth 仅用于评估，受 checkpoint.save_best / monitor_metric 控制
  - 早停（F08）: val_acc 连续无改善（training.patience 轮）；
    或 val_loss 连续高于历史最小值的 threshold 倍（val_loss_patience 轮）。
    中间恢复时计数清零；取值为 0 表示禁用该项。触发时记录原因。
  - 续训（F09）: 恢复 RNG / AMP scaler / best / 早停计数 / 累计时长；
    中断断点标记 partial=True（epoch 中途状态，不冒充精确续训）
  - 梯度累积（F14）: 按组内实际样本数归一化，尾组不足一组时不缩小更新
  - 确定性与性能模式（cudnn.deterministic / benchmark / matmul precision）
    由配置决定；Trainer 初始化按配置应用一次，fit() 不再强制覆盖

用法:
    from training.trainer import Trainer
    trainer = Trainer(model, train_loader, val_loader, test_loader, config, model_name)
    # test_loader 允许为 None（PB02：PrivateTest 由统一评估入口按需构建）
    trainer.fit(30)
"""

__all__ = [
    "load_config", "set_seed", "get_activation", "ACTIVATION_REGISTRY",
    "build_optimizer", "build_scheduler",
    "mixup_data", "mixup_criterion",
    "compute_batch_sizes", "compute_group_totals", "update_val_loss_monitor",
    "Trainer",
]

import logging
import random
import sys
import time
from collections.abc import Sized
from datetime import datetime
from pathlib import Path
from typing import cast

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.batch_augment import (
    BATCH_AUG_SEED_RULE,
    BATCH_AUG_VERSION,
    BatchAugmenter,
    build_batch_augmenter,
)
from training.checkpoint import (
    RUNS_ROOT,
    TRAINING_PROTOCOL_VERSION,
    _capture_rng_state,
    _format_duration,
    _restore_rng_state,
    _restore_training_state,
    _verify_checkpoint_state,
    collect_environment_info,
    collect_git_info,
    read_run_meta,
    scheduler_effective_dict,
    write_json_atomic,
    write_run_meta,
)
from training.checkpoint import (
    cleanup_old_checkpoints as _cleanup_old_checkpoints,
)
from training.checkpoint import (
    load_checkpoint as _load_checkpoint,
)
from training.checkpoint import (
    load_checkpoint_metadata as _load_checkpoint_metadata,
)
from training.checkpoint import (
    save_checkpoint as _save_checkpoint,
)
from utils.activations import ACTIVATION_REGISTRY, get_activation
from utils.config_validation import validate_config
from utils.formal_protocol import completion_reason, validate_formal_plan
from utils.losses import CBFocalLoss, FocalLoss
from utils.model_spec import count_parameters, file_sha256, make_spec_from_config

# 日志
logger = logging.getLogger("trainer")

# ============================================================
# 项目根目录（动态计算）
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# 配置加载
# ============================================================
def load_config(config_path: str | None = None) -> dict:
    """加载训练配置文件"""
    if config_path is None:
        path = PROJECT_ROOT / "configs" / "training_config.yaml"
    else:
        path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"配置文件未找到: {path}")
    with open(path, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        raise ValueError(f"配置文件格式错误（应为 YAML 映射）: {path}")
    return config


# ============================================================
# 随机种子
# ============================================================
def set_seed(seed: int = 42, deterministic: bool = False):
    """
    固定所有随机种子，保证可复现性。

    Args:
        seed: 随机种子值
        deterministic: 是否启用 cuDNN 确定性模式。
            - False（默认）：允许 cuDNN 使用快速非确定性算法，训练速度 2-10x 更快
            - True：严格可复现，但会显著降低 GPU 训练速度

    注意：cudnn.benchmark 由 deterministic 自动控制：
        - deterministic=False → benchmark=True（自动选择最快卷积算法）
        - deterministic=True  → benchmark=False（使用确定性算法）
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic


# ============================================================
# 优化器工厂
# ============================================================
def build_optimizer(model: nn.Module, config: dict) -> optim.Optimizer:
    """根据配置构建优化器（config 为 training section 或合并后的配置）。

    PB03：optimizer_fused（默认 False，仅 Adam）启用 fused 实现；要求 CUDA 参数，
    否则明确报错（不静默回退——协议记录的是实际生效配置）。
    """
    lr = config["learning_rate"]
    wd = config["weight_decay"]
    name = config.get("optimizer", "adam").lower()

    if name == "adam":
        fused = bool(config.get("optimizer_fused", False))
        if fused:
            param_device = next(model.parameters()).device
            if param_device.type != "cuda":
                raise ValueError(
                    "training.optimizer_fused=True 仅支持 CUDA 参数"
                    f"（当前设备: {param_device.type}）；请关闭该选项或改用 CUDA 训练"
                )
            return optim.Adam(model.parameters(), lr=lr, weight_decay=wd, fused=True)
        return optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    elif name == "adamw":
        return optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    elif name == "sgd":
        return optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=wd)
    else:
        raise ValueError(f"不支持的优化器: {name}")


# ============================================================
# 学习率调度器工厂
# ============================================================
def build_scheduler(optimizer: optim.Optimizer, config: dict, num_epochs: int,
                    model_config: dict | None = None):
    """根据配置构建学习率调度器；none 返回 None"""
    name = config.get("scheduler", "none").lower()

    if name == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    elif name == "cosine_warm":
        # CosineAnnealingWarmRestarts: 每 T_0 轮重启 LR，逐步收敛
        # T_mult=2 表示每个新周期长度为前一个的 2 倍
        # T_0 优先级: 模型级 scheduler_t0 > 全局 scheduler_t0 > num_epochs 推导
        t0 = (model_config or {}).get("scheduler_t0")
        if t0 is None:
            t0 = config.get("scheduler_t0")
        if t0 is None:
            t0 = max(num_epochs // 3, 5)
        return optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=t0, T_mult=2
        )
    elif name == "step":
        return optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config.get("scheduler_step_size", 10),
            gamma=config.get("scheduler_gamma", 0.5),
        )
    elif name == "plateau":
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=3
        )
    elif name == "none":
        return None
    else:
        raise ValueError(f"不支持的调度器: {name}")


# ============================================================
# MixUp 工具函数
# ============================================================
def mixup_data(x: torch.Tensor, y: torch.Tensor, alpha: float = 1.0, device=None):
    """MixUp: 混合批内随机两张图像及其标签。Returns: mixed_x, y_a, y_b, lambda"""
    if alpha <= 0:
        return x, y, y, torch.tensor(1.0)
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(device or x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred: torch.Tensor, y_a: torch.Tensor,
                    y_b: torch.Tensor, lam: float):
    """MixUp 损失: λ * CE(pred, y_a) + (1-λ) * CE(pred, y_b)"""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# ============================================================
# 梯度累积辅助（F14）
# ============================================================
def compute_batch_sizes(loader: DataLoader) -> list[int]:
    """
    预计算 DataLoader 各 micro-batch 的样本数。

    仅在最后一个 batch 不满时与 batch_size 不同；不支持 batch_sampler。
    """
    n = loader.batch_size
    if n is None:
        raise ValueError("梯度累积要求 DataLoader 使用普通 batch_size（不支持 batch_sampler）")
    num_batches = len(loader)
    if num_batches == 0:
        return []
    if getattr(loader, "drop_last", False):
        return [n] * num_batches
    total = len(cast(Sized, loader.dataset))
    last = total - n * (num_batches - 1)
    if last == n:
        return [n] * num_batches
    sizes = [n] * num_batches
    sizes[-1] = int(last)
    return sizes


def compute_group_totals(sizes: list[int], grad_accum_steps: int) -> list[int]:
    """
    返回每个 micro-batch 所属累积组的实际总样本数。

    组定义：每 grad_accum_steps 个连续 micro-batch 为一组（最后一组可为不足满员）。
    归一化系数 = 本 micro 样本数 / 组总样本数 → 与「整组一次性求均值 loss」等价。
    """
    totals = []
    for start in range(0, len(sizes), grad_accum_steps):
        end = min(start + grad_accum_steps, len(sizes))
        group_total = sum(sizes[start:end])
        totals.extend([group_total] * (end - start))
    return totals


def update_val_loss_monitor(
    hist_min_val_loss: float | None,
    loss_worse_counter: int,
    val_loss: float,
    threshold: float,
) -> tuple[float, int]:
    """
    val_loss 恶化监控（F08）。

    - 更新历史最小 val_loss
    - 当前 val_loss 高于「历史最小值 × threshold」时恶化计数 +1，否则清零（中间恢复清零）

    Returns:
        (新的 hist_min_val_loss, 新的 loss_worse_counter)
    """
    if hist_min_val_loss is None or val_loss < hist_min_val_loss:
        hist_min_val_loss = val_loss
    if hist_min_val_loss > 0 and val_loss > hist_min_val_loss * threshold:
        loss_worse_counter += 1
    else:
        loss_worse_counter = 0
    return hist_min_val_loss, loss_worse_counter


# 训练协议快照中允许变化、不参与比对的白名单字段（S02）：
# 仅限不影响训练状态的显示/输出/内存选项；轮数以 fit 参数给出（追加轮数自然合法）。
_PROTOCOL_WHITELIST_PATHS = (
    ("checkpoint", "save_every_n_epochs"),
    ("checkpoint", "max_checkpoint_files"),
    ("dataloader", "pin_memory"),
    ("dataloader", "prefetch_factor"),
)


def _delete_config_path(config: dict, path: tuple) -> None:
    """从配置快照中删除白名单字段（缺失时静默）。"""
    node: object = config
    for key in path[:-1]:
        if not isinstance(node, dict):
            return
        node = node.get(key, {})
    if isinstance(node, dict):
        node.pop(path[-1], None)


# ============================================================
# Trainer 核心训练器
# ============================================================
class _UpdateCounter:
    """A deepcopy-safe optimizer hook that does not retain an entire Trainer."""

    def __init__(self) -> None:
        self.value = 0

    def __call__(self, optimizer, args, kwargs):
        found_inf = getattr(optimizer, "found_inf", None)
        if found_inf is None or found_inf.item() == 0:
            self.value += 1


class Trainer:
    """
    统一训练器，集成:
    - run 隔离（独立 run 目录 + run_meta + history）
    - 断点续训（RNG / scaler / best / 早停计数完整恢复）
    - 暂停自动保存（partial 标记）
    - Early stopping（val_acc + val_loss 双监控，独立开关）
    - 最优模型保存（monitor_metric / save_best 由配置控制）
    - 梯度累积（按组内实际样本数归一化）
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader | None,
        config: dict,
        model_name: str,
        device: torch.device | None = None,
        focal_gamma: float = 2.0,
        class_counts: list | None = None,
        run_dir: str | Path | None = None,
        run_meta_extra: dict | None = None,
        run_purpose: str = "unspecified",
        frozen_protocol: dict | None = None,
    ):
        # ---- 配置集中校验（R07）：在任何模型移动 / run 写入之前执行 ----
        validate_config(config, model_name=model_name)

        # ---- 设备解析先于模型移动（R05）----
        if device is not None:
            self.device = device if isinstance(device, torch.device) else torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.config = config
        self.model_name = model_name

        # ---- 模型规格（F01）：与 checkpoint / 推理共用同一来源 ----
        self.model_spec = make_spec_from_config(config, model_name)

        # ---- 确定性与性能模式：由配置决定（F09），fit 不再覆盖 ----
        self.deterministic = bool(config["training"].get("cudnn_deterministic", False))
        if self.device.type == "cuda":
            torch.backends.cudnn.deterministic = self.deterministic
            torch.backends.cudnn.benchmark = not self.deterministic
            # deterministic=False 时允许 TF32（matmul_precision=high，Tensor Core 收益）
            # deterministic=True 时使用 highest 保证严格一致的浮点行为
            torch.set_float32_matmul_precision("highest" if self.deterministic else "high")

        # ---- torch.compile（Windows 无 Triton 时自动跳过） ----
        self.use_compile = False
        if self.device.type == "cuda" and config["training"].get("torch_compile", False):
            try:
                import triton  # noqa: F401
                compile_mode = config["training"].get("torch_compile_mode", "reduce-overhead")
                print(f"torch.compile({compile_mode}) 编译中...")
                compiled = torch.compile(self.model, mode=compile_mode)
                # torch.compile 返回 OptimizedModule（stub 未精确建模为 nn.Module）
                self.model = compiled  # type: ignore[assignment]
                self.use_compile = True
                print("  torch.compile 启用")
            except ImportError:
                print("  torch.compile 跳过：Triton 未安装（Windows 原生不支持），使用 eager 模式")

        # ---- 模型特定配置 ----
        self.model_config = config["models"].get(model_name, {})
        self.scheduler_num_epochs = self.model_config.get(
            "num_epochs", config["training"]["num_epochs"]
        )
        # 早停（F08）：=0 表示禁用；触发条件在 fit 中检查
        self.patience = config["training"].get("patience", 7)
        self.val_loss_patience = config["training"].get("val_loss_patience", 0)
        self.val_loss_threshold = config["training"].get("val_loss_threshold", 1.05)

        # ---- 增强（R06）：批级（MixUp）与类别专属增强均受总开关约束 ----
        self.augmentation_master = bool(config.get("augmentation", {}).get("enabled", False))
        mixup_cfg = config.get("augmentation", {}).get("mixup", {})
        self.mixup_enabled = self.augmentation_master and bool(mixup_cfg.get("enabled", False))
        self.mixup_alpha = mixup_cfg.get("alpha", 0.2)
        self.class_specific_enabled = (
            self.augmentation_master
            and bool(config.get("augmentation", {}).get("class_specific", {}).get("enabled", False))
        )

        # ---- 增强实现（PB01）：augmentation.impl ∈ {legacy, batched} ----
        # legacy：逐样本 torchvision 管道（dataset.transform，由 create_dataloaders 绑定）
        # batched：批级张量增强，在 train_one_epoch 内于 .to(device) 之前执行
        aug_cfg = config.get("augmentation", {})
        self._aug_impl = aug_cfg.get("impl", "legacy")
        if self._aug_impl not in ("legacy", "batched"):
            raise ValueError(
                f"augmentation.impl 不支持: {self._aug_impl!r}（应为 'legacy' / 'batched'）"
            )
        self.batch_augmenter: BatchAugmenter | None = None
        if self._aug_impl == "batched":
            self.batch_augmenter = build_batch_augmenter(aug_cfg, seed=config["seed"])
            ds_transform = getattr(train_loader.dataset, "transform", None)
            if ds_transform is not None:
                raise ValueError(
                    "augmentation.impl=batched 但 train_loader.dataset 仍带逐样本 transform"
                    "（会双重增强）；请经 create_dataloaders 构建训练集"
                )

        # ---- 类别计数（F06）：采样器与 CB Focal Loss 必须同源 ----
        ds_counts = getattr(train_loader.dataset, "class_counts", None)
        if (
            class_counts is not None and ds_counts is not None
            and list(class_counts) != list(ds_counts)
        ):
            raise ValueError(
                "传入的 class_counts 与 train_loader.dataset.class_counts 不一致："
                f"{list(class_counts)} vs {list(ds_counts)}（采样器与损失必须使用同一份统计）"
            )
        resolved_counts = class_counts if class_counts is not None else ds_counts
        self.class_counts = list(resolved_counts) if resolved_counts is not None else None

        # ---- 损失函数（R08：显式 cross_entropy 分支；未知类型明确报错）----
        loss_type = config["training"].get("loss_type", "focal")
        self.loss_type = loss_type
        self.focal_gamma = focal_gamma
        self.cb_focal_beta: float | None = None
        self.class_balanced_sampling = bool(
            config.get("dataloader", {}).get("class_balanced_sampling", False)
        )
        self.criterion: nn.Module
        if loss_type == "cb_focal":
            if self.class_counts is None:
                raise ValueError(
                    "loss_type=cb_focal 需要 class_counts：请通过 create_dataloaders 构建训练集"
                    "或显式传入（不再回退硬编码常量）"
                )
            beta = float(config["training"].get("cb_focal_beta", 0.999))
            self.cb_focal_beta = beta
            self.criterion = CBFocalLoss(
                gamma=focal_gamma, beta=beta, class_counts=self.class_counts
            )
            print(
                f"  loss: CBFocalLoss(gamma={focal_gamma}, beta={beta}) | "
                f"class_counts={self.class_counts}"
            )
        elif loss_type == "cross_entropy":
            self.criterion = nn.CrossEntropyLoss()
            print("  loss: CrossEntropyLoss（普通 CE 基线）")
        elif loss_type == "focal":
            self.criterion = FocalLoss(gamma=focal_gamma)
            if self.class_counts is not None:
                print(
                    f"  loss: FocalLoss(gamma={focal_gamma}) | "
                    f"class_counts(采样器参考)={self.class_counts}"
                )
        else:
            raise ValueError(
                f"未知 loss_type: {loss_type!r}"
                "（合法: focal / cb_focal / cross_entropy；应由配置校验提前拦截）"
            )

        # ---- 优化器 & 调度器 ----
        merged_training = dict(config["training"])
        merged_training["learning_rate"] = self.model_config.get(
            "learning_rate", config["training"]["learning_rate"]
        )
        self.optimizer = build_optimizer(self.model, merged_training)
        # A post-step hook counts real updates, including GradScaler skips, without GPU sync.
        self._update_counter = _UpdateCounter()
        self._optimizer_attempts = 0
        self.optimizer.register_step_post_hook(self._update_counter)
        # PB03：实际生效的 fused 状态（fused=true 时 build_optimizer 已校验 CUDA 并生效）
        self._optimizer_fused = bool(merged_training.get("optimizer_fused", False))
        self.scheduler = build_scheduler(
            self.optimizer, config["training"], self.scheduler_num_epochs,
            model_config=self.model_config,
        )

        # ---- best / 早停计数状态 ----
        self.monitor_metric = config["checkpoint"].get("monitor_metric", "val_acc")
        self.save_best = config["checkpoint"].get("save_best", True)
        self.best_val_acc = 0.0            # 始终跟踪 val_acc 峰值（报告与 acc 早停用）
        self.best_epoch = 0                # monitor 指标的峰值 epoch
        self.best_monitor_value: float | None = None     # monitor 指标峰值
        self.acc_patience_counter = 0
        self.loss_worse_counter = 0
        self.hist_min_val_loss: float | None = None

        # ---- 训练状态 ----
        self.start_epoch = 1
        self.history: dict[str, list[float]] = {
            "train_loss": [], "train_acc": [], "val_loss": [],
            "val_acc": [], "val_top5_acc": [], "lr": [],
            "optimizer_updates": [], "optimizer_attempts": [],
        }
        self.best_model_state = None  # 内存快照仅在 fit 会话内有效；权威 best 在 best.pth
        # S03：True 表示上一会话以“部分轮更新”结束（中断/异常）；
        # 此时直接 fit 会先自动回滚到最近完整 last.pth（rollback_events 记录）
        self._partial_state = False

        # ---- 参数量 ----
        self.total_params, self.trainable_params = count_parameters(
            getattr(self.model, "_orig_mod", self.model)
        )

        # ---- AMP 混合精度（GPU 自动启用） ----
        use_amp = self.device.type == "cuda" and config["training"].get("amp", True)
        self.use_amp = use_amp
        self.scaler = torch.amp.GradScaler("cuda") if use_amp else None

        # ---- 梯度累积 / 裁剪 ----
        self.grad_accum_steps = config["training"].get("gradient_accumulation_steps", 1)
        self.max_grad_norm = config["training"].get("max_grad_norm", 0.0)

        # ---- 训练时间追踪（跨恢复会话累积） ----
        self._accumulated_train_time = 0.0
        self._total_train_time = 0.0

        # ---- run_meta（F05/R02）：仅新 run 在构造时落盘；已存在的 run 在恢复校验
        #      通过之前不改写任何文件（失败不得先改写原 run 元数据） ----
        self._run_meta_extra = run_meta_extra or {}
        # T06：run 用途声明与冻结协议绑定（train.py --purpose formal 时提供；
        # 默认 unspecified/smoke 均为非正式，正式实验准入由 check_formal_eligibility 判定）
        self.run_purpose = str(run_purpose or "unspecified")
        self.frozen_protocol = dict(frozen_protocol) if frozen_protocol else None
        if self.run_purpose == "formal":
            validate_formal_plan(
                self.frozen_protocol, self.model_name, self.model_spec.to_dict(),
                self.get_training_protocol(),
            )
        # ---- run 目录（F05）----
        if run_dir is None:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base = f"{stamp}_seed{config['seed']}"
            run_dir = RUNS_ROOT / model_name / base
            suffix = 1
            while run_dir.exists():
                suffix += 1
                run_dir = RUNS_ROOT / model_name / f"{base}_{suffix}"
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.run_id = self.run_dir.name
        self.checkpoints_dir = self.run_dir / "checkpoints"
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)

        existing_meta = read_run_meta(self.run_dir)
        if existing_meta is None:
            self.run_meta = self._build_run_meta(run_meta_extra)
            self._persist_run_meta(status="initialized")
        else:
            self.run_meta = existing_meta

        # 生效配置落盘（若调用方尚未写入）
        config_path = self.run_dir / "config_effective.yaml"
        if not config_path.exists():
            with open(config_path, "w", encoding="utf-8") as f:
                yaml.safe_dump(config, f, allow_unicode=True, sort_keys=False)

    # ========================================================
    # run 元数据
    # ========================================================
    def _build_run_meta(self, extra: dict | None) -> dict:
        dataset = getattr(self.train_loader, "dataset", None)
        split_fp = getattr(dataset, "split_fingerprint", None)
        return {
            "run_id": self.run_id,
            "model_name": self.model_name,
            "status": "initialized",
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "seed": self.config["seed"],
            "run_purpose": self.run_purpose,
            "frozen_protocol": self.frozen_protocol,
            "model_spec": self.model_spec.to_dict(),
            "class_counts": self.class_counts,
            "git": collect_git_info(),
            "environment": collect_environment_info(),
            "training_device": str(self.device),
            "amp": self.use_amp,
            "cudnn_deterministic": self.deterministic,
            "torch_compile": self.use_compile,
            "effective_augmentation": {
                "master": self.augmentation_master,
                "impl": self._aug_impl,
                "batch_augmenter_version": (
                    BATCH_AUG_VERSION if self.batch_augmenter is not None else None
                ),
                "batch_augmenter_seed_rule": (
                    BATCH_AUG_SEED_RULE if self.batch_augmenter is not None else None
                ),
                "mixup": self.mixup_enabled,
                "class_specific": self.class_specific_enabled,
            },
            "data": split_fp if split_fp is not None else {
                "note": "split_fingerprint 未提供（train_loader 非 create_dataloaders 构建）"
            },
            "cli_args": (extra or {}).get("cli_args"),
            "resume_events": [],
        }

    def _persist_run_meta(self, **updates) -> None:
        self.run_meta.update(updates)
        self.run_meta["updated_at"] = datetime.now().isoformat()
        write_run_meta(self.run_dir, self.run_meta)

    def get_training_protocol(self) -> dict:
        """
        训练协议快照 v2（S01/S02）：**完整生效配置**的规范化副本（仅剔除白名单
        字段）+ 运行时有效值（数据管线规格、AMP/compile 实际状态、调度器实际参数、
        双早停参数等）+ 数据指纹。恢复前整字典比对，任何影响训练状态的差异都会被
        拒绝（避免手工挑字段遗漏）。

        字段覆盖：全部启用的增强数值/概率/目标类别、裁剪、累积、双早停参数、
        scheduler（T0/T_mult/step_size/gamma 的声明与实际）、sampler/loader 规格、
        损失、优化器、monitor、精度与数据指纹。
        """
        import copy as _copy

        snapshot = _copy.deepcopy(self.config)
        for path in _PROTOCOL_WHITELIST_PATHS:
            _delete_config_path(snapshot, path)

        loader = self.train_loader
        sampler = getattr(loader, "sampler", None)
        has_generators = (
            isinstance(getattr(loader, "generator", None), torch.Generator)
            and (sampler is None or isinstance(getattr(sampler, "generator", None),
                                               torch.Generator))
        )
        runtime = {
            "loader": {
                "num_workers": int(getattr(loader, "num_workers", 0) or 0),
                "persistent_workers": bool(getattr(loader, "persistent_workers", False)),
                "sampler": type(sampler).__name__ if sampler is not None else None,
                "batch_size": loader.batch_size,
                "has_generators": has_generators,
            },
            "amp_effective": bool(self.use_amp),
            "torch_compile_effective": bool(self.use_compile),
            "loss_effective": self.loss_type,
            "focal_gamma": float(self.focal_gamma),
            "cb_focal_beta_effective": self.cb_focal_beta,
            "scheduler_effective": scheduler_effective_dict(self.scheduler),
            "max_grad_norm": float(self.max_grad_norm),
            "gradient_accumulation_steps": int(self.grad_accum_steps),
            "patience": int(self.patience),
            "val_loss_patience": int(self.val_loss_patience),
            "val_loss_threshold": float(self.val_loss_threshold),
            "monitor_metric": self.monitor_metric,
            "save_best": bool(self.save_best),
            "cudnn_deterministic": bool(self.deterministic),
            "batch_augmentation_effective": {
                "impl": self._aug_impl,
                "active": self.batch_augmenter is not None,
                "version": (
                    BATCH_AUG_VERSION if self.batch_augmenter is not None else None
                ),
                "seed_rule": (
                    BATCH_AUG_SEED_RULE if self.batch_augmenter is not None else None
                ),
            },
            "optimizer_fused_effective": bool(getattr(self, "_optimizer_fused", False)),
        }
        dataset = getattr(loader, "dataset", None)
        return {
            "protocol_version": TRAINING_PROTOCOL_VERSION,
            "config": snapshot,
            "runtime": runtime,
            "data_fingerprint": getattr(dataset, "split_fingerprint", None),
        }

    def _record_resume_event(self, checkpoint_path, *, protocol_verified: bool,
                             notes: list | None = None,
                             state_integrity: dict | None = None) -> None:
        """恢复成功后记录事件（R02/T01/T03）：含当次 CLI/git/环境与生效配置摘要；
        协议一致（protocol_verified）与完整状态恢复（state_integrity 预检摘要）分开记录。"""
        event = {
            "resumed_at": datetime.now().isoformat(),
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": file_sha256(checkpoint_path),
            "protocol_verified": protocol_verified,
            "state_integrity": dict(state_integrity or {}),
            "notes": list(notes or []),
            "cli_args": self._run_meta_extra.get("cli_args"),
            "resume_conditions": self._run_meta_extra.get("resume_conditions"),
            "git_commit": collect_git_info().get("commit"),
            "environment": collect_environment_info(),
            "config_effective_sha256": file_sha256(self.run_dir / "config_effective.yaml"),
        }
        self.run_meta.setdefault("resume_events", []).append(event)
        self._persist_run_meta()

    def _write_history(self) -> None:
        write_json_atomic(self.run_dir / "history.json", self.history)

    @property
    def _optimizer_updates(self) -> int:
        return self._update_counter.value

    @_optimizer_updates.setter
    def _optimizer_updates(self, value: int) -> None:
        self._update_counter.value = value

    @staticmethod
    def _validate_loader_cache(loader) -> None:
        check = getattr(loader.dataset, "validate_cache", None)
        if check is not None:
            check()

    # ========================================================
    # checkpoint 委托（training/checkpoint.py）
    # ========================================================
    def save_checkpoint(self, path, *, partial: bool = False, history: dict | None = None):
        """保存完整训练状态（原子写入）"""
        return _save_checkpoint(self, path, partial=partial, history=history)

    def load_checkpoint(self, checkpoint_path) -> dict:
        """
        从 checkpoint 恢复完整训练状态（校验/拒绝逻辑见 training.checkpoint）；
        恢复后清空梯度、清除“部分轮更新”标记（状态回到已知完整边界）。
        """
        ckpt = _load_checkpoint(self, checkpoint_path)
        self.optimizer.zero_grad(set_to_none=True)
        self._partial_state = False
        return ckpt

    @staticmethod
    def load_checkpoint_metadata(path: Path) -> dict | None:
        """轻量读取 checkpoint 元数据"""
        return _load_checkpoint_metadata(path)

    def _cleanup_old_checkpoints(self):
        """按配置清理最旧的定期断点"""
        _cleanup_old_checkpoints(self.checkpoints_dir, self.config)

    def _grad_scaler(self) -> torch.amp.GradScaler:
        """AMP GradScaler（use_amp=True 时必存在；helper 用于类型收窄）"""
        assert self.scaler is not None, "GradScaler 仅在 use_amp=True 时可用"
        return self.scaler

    def _rollback_to_last_checkpoint(self) -> None:
        """
        S03：会话以“部分轮更新”结束（中断/异常）后，恢复到最近完整 last.pth
        再允许继续训练；覆盖模型/BN、优化器、调度器、scaler、RNG、loader 生成器、
        best/早停计数与 history。没有完整 last.pth 时明确拒绝（要求重新构造）。
        回滚事件记录于 run_meta.rollback_events。
        """
        last_path = self.checkpoints_dir / "last.pth"
        if not last_path.exists():
            raise RuntimeError(
                "上一会话以部分轮更新结束（键盘中断/异常），且不存在完整 last.pth：\n"
                "无法自动回滚到完整边界。请重新构造 Trainer 从头训练，"
                "不要沿用含部分更新的实例。"
            )
        checkpoint = torch.load(last_path, map_location="cpu", weights_only=False)
        # T03：与显式恢复同一完整性门槛（预检只读；失败即拒绝，不进入半恢复状态）
        _verify_checkpoint_state(self, checkpoint)
        if (
            checkpoint.get("partial") or checkpoint.get("run_id") != self.run_id
            or checkpoint.get("model_spec") != self.model_spec.to_dict()
            or checkpoint.get("training_protocol") != self.get_training_protocol()
        ):
            raise RuntimeError("自动回滚的 last 不是当前 run/协议的完整边界")

        def record_rollback():
            self._partial_state = False
            self.run_meta.setdefault("rollback_events", []).append({
                "rolled_back_at": datetime.now().isoformat(),
                "reason": "partial-update recovery",
                "checkpoint": str(last_path),
                "checkpoint_sha256": file_sha256(last_path),
                "history_epochs": len(self.history["train_loss"]),
            })
            self._persist_run_meta()

        _restore_training_state(self, checkpoint, on_commit=record_rollback)
        self.optimizer.zero_grad(set_to_none=True)
        print(
            f"检测到部分轮更新：已自动回滚至最近完整断点"
            f"（epoch {self.start_epoch - 1}）并记录 rollback_events"
        )

    # ========================================================
    # 性能诊断
    # ========================================================
    def diagnose(self, num_steps: int = 10) -> dict:
        """
        性能诊断：在**独立副本**上逐步计时，定位训练瓶颈（含 3 步预热）。

        R03 隔离保证:
          - 独立模型 / 优化器 / GradScaler / 数据加载器副本；不消费正式 loader 的
            持久 worker / 增强队列状态
          - 结束后恢复主运行 RNG；正式模型参数、BN buffer、优化器、scheduler、
            scaler、loader/sampler 状态与 run 文件均不受影响（"先诊断再 fit" 与
            "直接 fit" 得到同批输入与同等结果）
          - K>1 时按真实分组语义在组末执行 optimizer step
        """
        import copy as _copy
        import statistics

        warmup_steps = 3
        total_steps = warmup_steps + num_steps
        device = self.device
        rng_state = _capture_rng_state()

        # ---- 独立副本：模型 / 优化器 / scaler ----
        base_model = getattr(self.model, "_orig_mod", self.model)
        probe_model = _copy.deepcopy(base_model).to(device)
        probe_model.train()
        merged = dict(self.config["training"])
        merged["learning_rate"] = self.model_config.get(
            "learning_rate", self.config["training"]["learning_rate"]
        )
        probe_optimizer = build_optimizer(probe_model, merged)
        probe_scaler = torch.amp.GradScaler("cuda") if self.use_amp else None

        # ---- 独立数据加载器（不触碰正式 loader 的持久 worker / 队列状态）----
        probe_loader = DataLoader(
            self.train_loader.dataset,
            batch_size=self.train_loader.batch_size,
            shuffle=False,
            num_workers=0,
        )
        sizes = compute_batch_sizes(probe_loader)
        group_totals = compute_group_totals(sizes, self.grad_accum_steps)

        print(f"\n{'=' * 50}")
        print(f"性能诊断 | {warmup_steps} 步预热 + {num_steps} 步采样 | 设备: {device}")
        print(f"   AMP: {self.use_amp} | Compile: {self.use_compile}")
        print(f"   cudnn.benchmark: {torch.backends.cudnn.benchmark}")
        print(f"   cudnn.deterministic: {torch.backends.cudnn.deterministic}")
        if torch.cuda.is_available():
            print(f"   float32_matmul_precision: {torch.get_float32_matmul_precision()}")
        print(f"   batch_size: {probe_loader.batch_size}")
        print(f"   GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
        print("   隔离模式：诊断运行在独立副本上，主训练状态与 run 文件不受影响")

        def _sync():
            if torch.cuda.is_available():
                torch.cuda.synchronize()

        warmup_timings: dict[str, list[float]] = {
            "data_load": [], "forward": [], "backward": [], "optimizer": [],
        }
        timings: dict[str, list[float]] = {
            "data_load": [], "forward": [], "backward": [], "optimizer": [],
        }

        data_iter = iter(probe_loader)
        probe_optimizer.zero_grad(set_to_none=True)

        try:
            for step_idx in range(total_steps):
                _sync()
                t0 = time.perf_counter()
                try:
                    images, labels = next(data_iter)
                except StopIteration:
                    data_iter = iter(probe_loader)
                    images, labels = next(data_iter)
                if images.device != device:
                    images = images.to(device, non_blocking=True)
                    labels = labels.to(device, non_blocking=True)
                _sync()
                t1 = time.perf_counter()

                batch_size = images.size(0)
                group_total = (
                    group_totals[step_idx % len(group_totals)] if group_totals else batch_size
                )

                if self.use_amp:
                    assert probe_scaler is not None
                    with torch.amp.autocast("cuda"):
                        outputs = probe_model(images)
                        loss = self.criterion(outputs, labels)
                else:
                    outputs = probe_model(images)
                    loss = self.criterion(outputs, labels)
                _sync()
                t2 = time.perf_counter()

                scaled_loss = loss * batch_size / group_total
                if self.use_amp:
                    assert probe_scaler is not None
                    probe_scaler.scale(scaled_loss).backward()
                else:
                    scaled_loss.backward()
                _sync()
                t3 = time.perf_counter()

                # 组末 step（保留 K>1 的真实分组语义）
                if (step_idx + 1) % self.grad_accum_steps == 0 or (step_idx + 1) == total_steps:
                    if self.max_grad_norm > 0:
                        if self.use_amp:
                            assert probe_scaler is not None
                            probe_scaler.unscale_(probe_optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            probe_model.parameters(), self.max_grad_norm
                        )
                    if self.use_amp:
                        assert probe_scaler is not None
                        probe_scaler.step(probe_optimizer)
                        probe_scaler.update()
                    else:
                        probe_optimizer.step()
                    probe_optimizer.zero_grad(set_to_none=True)
                _sync()
                t4 = time.perf_counter()

                phase_times = {
                    "data_load": (t1 - t0) * 1000,
                    "forward": (t2 - t1) * 1000,
                    "backward": (t3 - t2) * 1000,
                    "optimizer": (t4 - t3) * 1000,
                }

                if step_idx < warmup_steps:
                    for k, v in phase_times.items():
                        warmup_timings[k].append(v)
                    phase_total = sum(phase_times.values())
                    print(f"   预热 {step_idx + 1}/{warmup_steps}: {phase_total:.0f}ms")
                else:
                    for k, v in phase_times.items():
                        timings[k].append(v)

            print(f"\n{'阶段':<15} {'平均(ms)':<12} {'最小(ms)':<12} {'最大(ms)':<12}")
            print(f"{'-' * 51}")
            total_avg = 0.0
            for stage, values in timings.items():
                avg = statistics.mean(values)
                total_avg += avg
                print(f"{stage:<15} {avg:>10.2f}  {min(values):>10.2f}  {max(values):>10.2f}")

            _sync()
            probe_model.eval()
            eval_start = time.perf_counter()
            with torch.no_grad():
                for xb, yb in self.val_loader:
                    xb = xb.to(device, non_blocking=True)
                    yb = yb.to(device, non_blocking=True)
                    if self.use_amp:
                        with torch.amp.autocast("cuda"):
                            probe_model(xb)
                    else:
                        probe_model(xb)
            _sync()
            eval_time = (time.perf_counter() - eval_start) * 1000
            print(f"{'eval (全部)':<15} {eval_time:>10.2f}  {'':<12} {'':<12}")

            if warmup_timings["forward"]:
                warmup_total = sum(statistics.mean(v) for v in warmup_timings.values())
                print(f"\n   预热阶段平均: {warmup_total:.0f}ms/步 → "
                      f"采样阶段平均: {total_avg:.0f}ms/步")

            if torch.cuda.is_available():
                print(f"   VRAM: {torch.cuda.memory_allocated() / 1024**2:.0f}MB / "
                      f"{torch.cuda.max_memory_allocated() / 1024**2:.0f}MB peak")

            est_it_per_sec = 1000.0 / total_avg if total_avg > 0 else 0
            print(f"{'-' * 51}")
            print(f"{'单步总计':<15} {total_avg:>10.2f} ms")
            print(f"{'预估稳态速度':<15} ~{est_it_per_sec:.0f} it/s")

            print("\n💡 诊断建议:")
            if torch.backends.cudnn.deterministic:
                print("   ⚠️  cudnn.deterministic=True：强制使用慢速确定性算法，建议在配置中关闭")
            max_stage = max(timings, key=lambda k: statistics.mean(timings[k]))
            max_pct = (
                statistics.mean(timings[max_stage]) / total_avg * 100 if total_avg > 0 else 0
            )
            if est_it_per_sec < 10:
                print(f"   ⚠️  速度异常低（{est_it_per_sec:.0f} it/s），"
                      f"主要瓶颈: {max_stage} ({max_pct:.0f}%)")
                if max_stage == "data_load":
                    print("       → 数据加载瓶颈，检查 num_workers 是否足够")
                elif max_stage == "forward":
                    print("       → 前向瓶颈，检查模型大小")
                elif max_stage == "backward":
                    print("       → 反向瓶颈，检查 AMP 是否正常工作")
            elif est_it_per_sec < 50:
                print(f"   ⚡ 速度正常偏低（{est_it_per_sec:.0f} it/s），"
                      f"瓶颈: {max_stage} ({max_pct:.0f}%)")
                if max_stage == "backward":
                    print("       → Windows WDDM 驱动有额外开销，属正常范围")
            else:
                print(f"   ✅ 速度正常（{est_it_per_sec:.0f} it/s）")

            print(f"{'=' * 50}\n")

            return {k: statistics.mean(v) for k, v in timings.items()}
        finally:
            # R03：恢复主运行 RNG（诊断消耗的随机数不泄漏到正式训练）
            _restore_rng_state(rng_state)
            del probe_loader, probe_model, probe_optimizer

    # ========================================================
    # 训练一个 epoch（梯度累积按组内实际样本数归一化）
    # ========================================================
    def train_one_epoch(self) -> tuple:
        """训练一个 epoch（含 AMP、梯度累积、梯度裁剪、tqdm 进度条）"""
        self.model.train()
        running_loss = torch.zeros((), device=self.device)
        correct = torch.zeros((), dtype=torch.long, device=self.device)
        total = 0

        num_batches = len(self.train_loader)
        sizes = compute_batch_sizes(self.train_loader)
        group_totals = compute_group_totals(sizes, self.grad_accum_steps)
        self.optimizer.zero_grad(set_to_none=True)

        epoch_idx = getattr(self, "_current_epoch", 0)
        pbar = tqdm(self.train_loader, desc=f"[Epoch {epoch_idx}]", leave=False)
        for batch_idx, (images, labels) in enumerate(pbar):
            # Main-process guard also covers stale batches prefetched by workers.
            self._validate_loader_cache(self.train_loader)
            # PB01：批级张量增强（CPU、.to(device) 之前；批种子 = f(seed, epoch, batch_idx)，
            # 连续与恢复训练在第 e 轮第 i 批复算出同一批增强参数）
            if self.batch_augmenter is not None:
                images = self.batch_augmenter(
                    images, labels, epoch=epoch_idx, batch_idx=batch_idx
                )
            if images.device != self.device:
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

            batch_size = images.size(0)

            # 组归一化系数：本 micro 样本数 / 组内实际总样本数（F14，尾组不缩小）
            group_total = group_totals[batch_idx] if group_totals else batch_size

            # MixUp：混合批内图像（仅训练，不用于验证）
            if self.mixup_enabled:
                images, labels_a, labels_b, lam = mixup_data(
                    images, labels, alpha=self.mixup_alpha, device=self.device,
                )

            if self.use_amp:
                with torch.amp.autocast("cuda"):
                    outputs = self.model(images)
                    if self.mixup_enabled:
                        loss = mixup_criterion(self.criterion, outputs, labels_a, labels_b, lam)
                    else:
                        loss = self.criterion(outputs, labels)
                scaled_loss = loss * batch_size / group_total
                self._grad_scaler().scale(scaled_loss).backward()
            else:
                outputs = self.model(images)
                if self.mixup_enabled:
                    loss = mixup_criterion(self.criterion, outputs, labels_a, labels_b, lam)
                else:
                    loss = self.criterion(outputs, labels)
                scaled_loss = loss * batch_size / group_total
                scaled_loss.backward()

            # 组末更新：最后一组（可为不足 K 个 micro）在末尾 micro 后同样触发
            group_start = (batch_idx // self.grad_accum_steps) * self.grad_accum_steps
            group_end = min(group_start + self.grad_accum_steps, num_batches)
            if (batch_idx + 1) == group_end:
                if self.max_grad_norm > 0:
                    if self.use_amp:
                        self._grad_scaler().unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

                self._validate_loader_cache(self.train_loader)
                self._optimizer_attempts += 1
                if self.use_amp:
                    self._grad_scaler().step(self.optimizer)
                    self._grad_scaler().update()
                else:
                    self.optimizer.step()

                self.optimizer.zero_grad(set_to_none=True)

            # 统计（延迟 .item() 调用；.float() 防 AMP 下 float16 累积溢出）
            running_loss += loss.detach().float() * batch_size
            if self.mixup_enabled:
                total += batch_size
            else:
                _, predicted = outputs.max(1)
                total += batch_size
                correct += predicted.eq(labels).sum()

            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == num_batches:
                batch_loss = (running_loss / total).item()
                if self.mixup_enabled:
                    pbar.set_postfix(loss=f"{batch_loss:.4f}")
                else:
                    pbar.set_postfix(
                        loss=f"{batch_loss:.4f}", acc=f"{(correct / total).item():.4f}"
                    )

        if self.mixup_enabled:
            return (running_loss / total).item(), 0.0
        return (running_loss / total).item(), (correct / total).item()

    @torch.no_grad()
    def evaluate(self, loader: DataLoader | None = None) -> tuple:
        """
        评估模型

        Returns:
            (loss, top1_acc, top5_acc) 三元组
        """
        self.model.eval()
        running_loss = torch.tensor(0.0, device=self.device)
        top1_correct = torch.tensor(0, dtype=torch.long, device=self.device)
        top5_correct = torch.tensor(0, dtype=torch.long, device=self.device)
        total = 0

        eval_loader = loader or self.val_loader

        for images, labels in eval_loader:
            self._validate_loader_cache(eval_loader)
            if images.device != self.device:
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

            if self.use_amp:
                with torch.amp.autocast("cuda"):
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

            batch_size = images.size(0)
            running_loss += loss.detach() * batch_size

            _, predicted = outputs.max(1)
            top1_correct += predicted.eq(labels).sum()

            num_classes = outputs.size(1)
            k = min(5, num_classes)
            _, topk_pred = outputs.topk(k, dim=1)
            top5_correct += topk_pred.eq(labels.view(-1, 1).expand_as(topk_pred)).sum()

            total += batch_size

        top1_acc = (top1_correct / total).item()
        top5_acc = (top5_correct / total).item()
        return (running_loss / total).item(), top1_acc, top5_acc

    # ========================================================
    # 主训练循环
    # ========================================================
    def fit(self, additional_epochs: int = 0) -> dict:
        """
        主训练循环（开放式，无 epoch 上限）。

        - 早停（val_acc 无改善 / val_loss 恶化，独立开关，触发记录原因）
        - 每个 epoch 结束保存 last.pth + history.json
        - best.pth 受 checkpoint.save_best / monitor_metric 控制
        - KeyboardInterrupt 保存 partial 断点；异常时 run_meta 标记 failed
        """
        save_every = self.config["checkpoint"].get("save_every_n_epochs", 5)

        if additional_epochs <= 0:
            print(f"\n{'=' * 60}")
            print(f"⚠️  需要指定训练轮数，例如: trainer.fit({30 if self.start_epoch <= 1 else 10})")
            print(f"   当前权重已训练至第 {self.start_epoch - 1} 轮")
            if self._accumulated_train_time > 0:
                print(f"   累计用时: {_format_duration(self._accumulated_train_time)}")
            print(f"{'=' * 60}")
            return self.history

        # S03：若上一会话以部分轮更新结束（中断/异常），先回滚到最近完整断点
        if self._partial_state:
            self._rollback_to_last_checkpoint()

        formal_plan = None
        if self.run_purpose == "formal":
            formal_plan = validate_formal_plan(
                self.frozen_protocol, self.model_name, self.model_spec.to_dict(),
                self.get_training_protocol(),
            )
            if self.start_epoch + additional_epochs - 1 > formal_plan["max_epochs"]:
                raise ValueError("本次会话将超过冻结预算，拒绝训练")
            if self.run_meta.get("experiment_completed"):
                raise ValueError("正式实验已完成，不能追加训练；请新建方案/run")

        total_epochs = self.start_epoch + additional_epochs - 1

        print(f"\n{'=' * 60}")
        print(f"模型: {self.model_name} | 参数量: {self.total_params:,}")
        print(f"Run: {self.run_id} | 目录: {self.run_dir}")
        print(f"设备: {self.device} | AMP: {self.use_amp} | Compile: {self.use_compile}")
        if self.device.type == "cuda":
            print(f"   matmul_precision: {torch.get_float32_matmul_precision()}")
            print(f"   cudnn: benchmark={torch.backends.cudnn.benchmark} "
                  f"deterministic={torch.backends.cudnn.deterministic}")
        if self.class_counts is not None:
            print(f"   class_counts(训练权重来源): {self.class_counts}")

        already_trained = self.start_epoch - 1
        if already_trained > 0:
            print(
                f"权重已训练: {already_trained} 轮 | "
                f"本次续训: {additional_epochs} 轮 → 训练至第 {total_epochs} 轮"
            )
            prev_dur = _format_duration(self._accumulated_train_time)
            if prev_dur:
                print(f"之前累计用时: {prev_dur}")
        else:
            print(f"从头训练: 共 {additional_epochs} 轮 → 训练至第 {total_epochs} 轮")

        print(f"{'=' * 60}\n")

        fit_start = time.perf_counter()
        session_start_epoch = self.start_epoch
        self._persist_run_meta(
            status="running",
            last_fit_started_at=datetime.now().isoformat(),
            last_fit_requested_epochs=additional_epochs,
            fit_start_epoch=session_start_epoch,
        )

        stop_reason = None
        interrupted = False

        try:
            for epoch in range(session_start_epoch, session_start_epoch + additional_epochs):
                self._current_epoch = epoch
                session_epoch = epoch - session_start_epoch + 1
                epoch_start = time.perf_counter()

                # 训练 + 验证
                train_loss, train_acc = self.train_one_epoch()
                val_loss, val_acc, val_top5 = self.evaluate()
                current_lr = self.optimizer.param_groups[0]["lr"]

                # 计时统计
                epoch_elapsed = time.perf_counter() - epoch_start
                self._total_train_time = self._accumulated_train_time + (
                    time.perf_counter() - fit_start
                )
                epochs_this_session = epoch - session_start_epoch + 1

                total_history_epochs = len(self.history["train_loss"])
                if total_history_epochs > 0 and self._accumulated_train_time > 0:
                    avg_epoch_time = self._total_train_time / total_history_epochs
                else:
                    avg_epoch_time = (time.perf_counter() - fit_start) / epochs_this_session

                remaining_epochs = session_start_epoch + additional_epochs - epoch - 1
                eta_seconds = avg_epoch_time * remaining_epochs

                # 记录历史
                self.history["train_loss"].append(train_loss)
                self.history["train_acc"].append(train_acc)
                self.history["val_loss"].append(val_loss)
                self.history["val_acc"].append(val_acc)
                self.history["val_top5_acc"].append(val_top5)
                self.history["lr"].append(current_lr)
                self.history["optimizer_updates"].append(self._optimizer_updates)
                self.history["optimizer_attempts"].append(self._optimizer_attempts)

                # 学习率调度
                if self.scheduler:
                    if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                        self.scheduler.step(val_acc)
                    else:
                        self.scheduler.step()

                # ---- best 判定（monitor_metric，F13）----
                acc_improved = val_acc > self.best_val_acc
                if acc_improved:
                    self.best_val_acc = val_acc

                if self.monitor_metric == "val_acc":
                    current_metric = val_acc
                    improved = self.best_monitor_value is None or val_acc > self.best_monitor_value
                else:  # val_loss（越小越好）
                    current_metric = val_loss
                    improved = self.best_monitor_value is None or val_loss < self.best_monitor_value

                marker = ""
                if improved:
                    self.best_monitor_value = current_metric
                    self.best_epoch = epoch
                    marker = " *"

                # ---- 早停计数（F08）----
                if self.patience > 0:
                    self.acc_patience_counter = 0 if acc_improved else self.acc_patience_counter + 1

                if self.val_loss_patience > 0:
                    self.hist_min_val_loss, self.loss_worse_counter = update_val_loss_monitor(
                        self.hist_min_val_loss, self.loss_worse_counter,
                        val_loss, self.val_loss_threshold,
                    )

                # Save after updating all progress, including early-stop counters.
                if improved and self.save_best:
                    self.save_checkpoint(self.checkpoints_dir / "best.pth")

                # ---- 定期保存 ----
                if epoch % save_every == 0:
                    self.save_checkpoint(
                        self.checkpoints_dir / f"epoch_{epoch:04d}.pth"
                    )
                    self._cleanup_old_checkpoints()

                # ---- last.pth（最新完整 epoch，续训入口）----
                self.save_checkpoint(self.checkpoints_dir / "last.pth")

                # ---- history 落盘 ----
                self._write_history()

                epoch_str = _format_duration(epoch_elapsed)
                eta_str = _format_duration(eta_seconds) if remaining_epochs > 0 else "--"

                print(
                    f"[Epoch {epoch} | 本轮 {session_epoch}/{additional_epochs}] "
                    f"Train: {train_loss:.4f}/{train_acc:.4f} | "
                    f"Val: {val_loss:.4f}/{val_acc:.4f} (T5:{val_top5:.4f}) | "
                    f"LR: {current_lr:.6f} | "
                    f"{epoch_str} | ETA {eta_str}{marker}"
                )

                # ---- 早停判定（触发时记录原因）----
                if self.patience > 0 and self.acc_patience_counter >= self.patience:
                    stop_reason = (
                        f"val_acc 早停：连续 {self.patience} 轮无改善"
                        f"（best val_acc={self.best_val_acc:.4f}）"
                    )
                if (
                    self.val_loss_patience > 0
                    and self.loss_worse_counter >= self.val_loss_patience
                ):
                    reason_loss = (
                        f"val_loss 早停：连续 {self.val_loss_patience} 轮高于历史最小值"
                        f"（{self.hist_min_val_loss:.4f}）的 {self.val_loss_threshold} 倍"
                    )
                    stop_reason = stop_reason + "；" + reason_loss if stop_reason else reason_loss

                if stop_reason:
                    print(f"\nEarly stopping at [Epoch {epoch}]: {stop_reason}")
                    break

                # LR 下界熔断
                if current_lr < 1e-7 and (
                    formal_plan is None or formal_plan["allow_early_stop"]
                ):
                    stop_reason = f"LR 已降至 {current_lr:.2e}，自动停止训练"
                    print(f"\n{stop_reason} (第 {epoch} 轮)")
                    break

        except KeyboardInterrupt:
            interrupted = True
            self._total_train_time = self._accumulated_train_time + (
                time.perf_counter() - fit_start
            )
            interrupted_epoch = getattr(self, "_current_epoch", session_start_epoch)
            print(f"\n⚠️  训练被用户中断 (第 {interrupted_epoch} 轮)")
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            interrupt_path = (
                self.checkpoints_dir / f"interrupted_epoch{interrupted_epoch:03d}_{ts}.pth"
            )
            self.save_checkpoint(interrupt_path, partial=True)
            self._partial_state = True  # S03：禁止未经回滚的直接继续
            print(f"断点已保存至: {interrupt_path}（partial：权重含未完成 epoch 的部分更新）")
            print("   可通过 --resume auto 从最近完整 epoch 继续；"
                  "同一实例再次 fit 将先自动回滚到最近完整断点（rollback_events 记录）")
        except Exception as e:
            # 失败会话的耗时同样计入累计（时间实际已消耗；状态由 run_meta 标记 failed）
            self._accumulated_train_time += time.perf_counter() - fit_start
            # S03：异常退出可能留下部分轮更新；禁止未经回滚的直接继续
            self._partial_state = True
            self._persist_run_meta(status="failed", error=repr(e))
            raise

        # ---- 收尾（R04：推进会话起点与累计时长，支持同实例重复 fit）----
        self._accumulated_train_time += time.perf_counter() - fit_start
        self._total_train_time = self._accumulated_train_time
        self.start_epoch = len(self.history["train_loss"]) + 1
        self._write_history()
        final_epoch = getattr(self, "_current_epoch", session_start_epoch - 1)

        if interrupted:
            self._persist_run_meta(
                status="interrupted",
                interrupted_at=datetime.now().isoformat(),
                final_epoch=final_epoch,
            )
        else:
            reason = None
            if formal_plan is not None:
                last = torch.load(
                    self.checkpoints_dir / "last.pth", map_location="cpu", weights_only=False
                )
                reason = completion_reason(formal_plan, last)
            self._persist_run_meta(
                status="finished" if formal_plan is None or reason else "session_completed",
                experiment_completed=bool(reason) if formal_plan is not None else False,
                completion_reason=reason,
                finished_at=datetime.now().isoformat(),
                final_epoch=final_epoch,
                stop_reason=stop_reason,
                best_val_acc=self.best_val_acc,
                best_monitor_value=self.best_monitor_value,
                best_monitor_metric=self.monitor_metric,
                best_epoch=self.best_epoch,
            )

        total_dur = _format_duration(self._total_train_time)
        print(f"\n{'=' * 60}")
        print(f"训练结束 | 训练至第 {final_epoch} 轮 | "
              f"best val_acc: {self.best_val_acc:.4f} | 总用时: {total_dur}")
        if stop_reason:
            print(f"停止原因: {stop_reason}")
        print(f"Run 目录: {self.run_dir}")
        if self.save_best:
            print(f"最优模型: {self.checkpoints_dir / 'best.pth'}")
        print(f"续训断点: {self.checkpoints_dir / 'last.pth'}")
        print(f"{'=' * 60}")

        return self.history

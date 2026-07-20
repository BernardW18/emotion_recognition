"""
共享训练模块 - 所有训练 Notebook 统一使用
解决: P2-8(代码重复) P0-2(resume) P0-3(暂停保存) P1-6(RandomErasing) P2-9(路径) P2-11(tqdm)
"""

import sys
import json
import copy
import logging
import random
import shutil
import time
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.activations import get_activation, ACTIVATION_REGISTRY

# ============================================================
# 数据加载模块（Trainer 内部不使用，Notebook 应直接导入 data.dataloader）
# 保留注释提示迁移方向；实际导入已在下方移除
# ============================================================

# ============================================================
# Checkpoint 管理（仅内部使用，不重导出）
# ============================================================
from training.checkpoint import (
    save_checkpoint as _save_checkpoint,
    load_checkpoint as _load_checkpoint,
    load_checkpoint_metadata as _load_checkpoint_metadata,
    cleanup_old_checkpoints as _cleanup_old_checkpoints,
    _format_duration,
)

# ============================================================
# 损失函数（Focal Loss 替代 CrossEntropy + class_weights）
# ============================================================
from utils.losses import FocalLoss, CBFocalLoss

# 日志
logger = logging.getLogger("trainer")

# 导出控制（不再重导出 data.dataloader 和 checkpoint 的符号）
__all__ = [
    "load_config", "set_seed", "get_activation", "ACTIVATION_REGISTRY",
    "build_optimizer", "build_scheduler",
    "Trainer",
]

# ============================================================
# 项目根目录（动态计算，解决 P2-9 脆弱路径问题）
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# 配置加载
# ============================================================
def load_config(config_path: str = None) -> dict:
    """加载训练配置文件"""
    if config_path is None:
        config_path = PROJECT_ROOT / "configs" / "training_config.yaml"
    else:
        config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件未找到: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


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
    """
    根据配置构建优化器

    Args:
        model: 待优化的模型
        config: 训练配置（training section）

    Returns:
        优化器实例
    """
    lr = config["learning_rate"]
    wd = config["weight_decay"]
    name = config.get("optimizer", "adam").lower()

    if name == "adam":
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
                    model_config: dict = None):
    """
    根据配置构建学习率调度器

    Args:
        optimizer: 优化器实例
        config: 训练配置（training section）
        num_epochs: 总训练轮数（模型级 num_epochs，在 cosine 中用作 T_max，
                    在 cosine_warm 中用于推导 T_0）
        model_config: 模型特定配置（可选），用于模型级 scheduler_t0 覆盖

    Returns:
        调度器实例，若配置为 none 则返回 None
    """
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
            # 未显式设置时，从 num_epochs 推导：使训练期间约经历 3 个余弦周期
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
# Trainer 核心训练器
# ============================================================
# MixUp 工具函数
# ============================================================
def mixup_data(x: torch.Tensor, y: torch.Tensor, alpha: float = 1.0, device: torch.device = None):
    """
    MixUp: 混合批内随机两张图像及其标签。

    Returns:
        mixed_x, y_a, y_b, lambda
    """
    if alpha <= 0:
        return x, y, y, torch.tensor(1.0)
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(device or x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred: torch.Tensor, y_a: torch.Tensor, y_b: torch.Tensor, lam: float):
    """
    MixUp 损失: λ * CE(pred, y_a) + (1-λ) * CE(pred, y_b)
    """
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# ============================================================
class Trainer:
    """
    统一训练器，集成:
    - 断点恢复（P0-2）
    - 暂停自动保存（P0-3）
    - tqdm进度条（P2-11）
    - Early stopping
    - 最优模型 + 定期 checkpoint 保存
    - 训练历史 JSON 记录
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        config: dict,
        model_name: str,
        device: torch.device = None,
        focal_gamma: float = 2.0,
        class_counts: list = None,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.config = config
        self.model_name = model_name
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # GPU 性能优化：开启 cudnn.benchmark（自动选择最优卷积算法）
        # 注意：必须同时关闭 deterministic，否则 cuDNN 只在确定性算法间 benchmark，速度极慢
        # set_seed() 可能已设置 deterministic=True，这里必须显式重置
        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            # 允许 Tensor Core 加速 float32 矩阵乘法（FC 层受益显著）
            # 'high' = 使用 Tensor Core，略微牺牲精度；'highest' = 禁用 Tensor Core（默认）
            torch.set_float32_matmul_precision('high')

        # torch.compile：编译模型以获得 kernel 融合和算子优化
        # 需要检查 Triton 是否可用（Windows 原生不支持 Triton）
        self.use_compile = False
        if self.device.type == "cuda" and config["training"].get("torch_compile", False):
            try:
                import triton  # noqa: F401
                compile_mode = config["training"].get("torch_compile_mode", "reduce-overhead")
                print(f"torch.compile({compile_mode}) 编译中...")
                self.model = torch.compile(self.model, mode=compile_mode)
                self.use_compile = True
                print("  ✅ torch.compile 启用")
            except ImportError:
                print("  ⚠️  torch.compile 跳过：Triton 未安装（Windows 原生不支持），使用 eager 模式")

        # 模型特定配置覆盖全局配置
        self.model_config = config["models"].get(model_name, {})
        # num_epochs 仅用于 CosineAnnealingLR 的 T_max，不限制训练轮数
        self.scheduler_num_epochs = self.model_config.get(
            "num_epochs", config["training"]["num_epochs"]
        )
        self.patience = config["training"].get("patience", 7)
        # val_loss 辅助监控配置（与 patience 整合，共享同一个 patience 计数器）
        self.val_loss_patience = config["training"].get("val_loss_patience", 0)
        self.val_loss_threshold = config["training"].get("val_loss_threshold", 1.05)

        # MixUp 配置（batch 级别混合增强）
        self.mixup_enabled = config.get("augmentation", {}).get("mixup", {}).get("enabled", False)
        self.mixup_alpha = config.get("augmentation", {}).get("mixup", {}).get("alpha", 0.2)

        # 损失函数：Focal Loss 或 CB Focal Loss（配置驱动）
        # loss_type: "focal" / "cb_focal"
        loss_type = config["training"].get("loss_type", "focal")
        if loss_type == "cb_focal":
            beta = config["training"].get("cb_focal_beta", 0.999)
            from utils.constants import CLASS_COUNTS
            self.criterion = CBFocalLoss(
                gamma=focal_gamma, beta=beta,
                class_counts=class_counts or CLASS_COUNTS,
            )
            print(f"  loss: CBFocalLoss(gamma={focal_gamma}, beta={beta})")
        else:
            self.criterion = FocalLoss(gamma=focal_gamma)

        # 优化器 & 调度器
        # 合并全局训练配置和模型特定配置
        merged_training = dict(config["training"])
        merged_training["learning_rate"] = self.model_config.get("learning_rate", config["training"]["learning_rate"])
        self.optimizer = build_optimizer(self.model, merged_training)
        self.scheduler = build_scheduler(
            self.optimizer, config["training"], self.scheduler_num_epochs,
            model_config=self.model_config,
        )

        # 目录
        self.save_dir = PROJECT_ROOT / "training" / "checkpoints" / model_name
        self.log_dir = PROJECT_ROOT / "training" / "logs" / model_name
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # 训练状态
        self.start_epoch = 1
        self.history = {
            "train_loss": [], "train_acc": [], "val_loss": [],
            "val_acc": [], "val_top5_acc": [], "lr": [],
        }
        self.best_val_acc = 0.0       # 本轮训练峰值 val_acc
        self.best_model_state = None   # 本轮训练峰值权重
        self.global_best_val_acc = 0.0 # 跨会话全局最佳 val_acc
        self.global_best_model_state = None  # 全局最佳权重

        # 即使从头训练，也要加载磁盘上已有的 global_best.pth 记录
        # 防止新 session 的低精度覆盖之前跨会话的最高记录
        global_best_path = self.save_dir / "global_best.pth"
        if global_best_path.exists():
            meta = _load_checkpoint_metadata(global_best_path)
            if meta and meta["global_best_val_acc"] > 0:
                self.global_best_val_acc = meta["global_best_val_acc"]
                logger.info(
                    "已从磁盘加载全局最佳记录: val_acc=%.4f (新 session 将与之比较)",
                    self.global_best_val_acc,
                )

        # 统计参数量
        self.total_params = sum(p.numel() for p in self.model.parameters())

        # AMP 混合精度训练（GPU 自动启用）
        use_amp = self.device.type == "cuda" and config["training"].get("amp", True)
        self.use_amp = use_amp
        self.scaler = torch.amp.GradScaler("cuda") if use_amp else None

        # 梯度累积步数
        self.grad_accum_steps = config["training"].get("gradient_accumulation_steps", 1)

        # 梯度裁剪阈值（0 表示不裁剪）
        self.max_grad_norm = config["training"].get("max_grad_norm", 0.0)

        # 训练时间追踪（跨恢复会话累积）
        self._accumulated_train_time = 0.0  # 之前会话累积的秒数
        self._total_train_time = 0.0        # 当前总训练时间（累积 + 当前会话）

    def diagnose(self, num_steps: int = 10) -> dict:
        """
        性能诊断：逐步计时，定位训练瓶颈

        测量以下阶段的耗时：
        - data_load: DataLoader 取出一个 batch 的时间
        - forward: 前向传播
        - backward: 反向传播
        - optimizer: 优化器更新 + zero_grad
        - eval: 验证集完整评估

        注意：此方法会执行真实训练步骤（含 forward + backward），会修改模型权重。
        如需无损诊断，请在诊断后从 checkpoint 恢复权重。

        Args:
            num_steps: 采样步数（不含预热步），默认 10

        Returns:
            包含各阶段平均耗时(ms)的字典（已排除预热）
        """
        import statistics

        # 预热步数：cuDNN benchmark 首次运行会测试多种卷积算法，耗时数秒
        # 预热后 cuDNN 会缓存最优算法，后续步骤反映真实稳态性能
        warmup_steps = 3
        total_steps = warmup_steps + num_steps

        print(f"\n{'=' * 50}")
        print(f"🔍 性能诊断 | {warmup_steps} 步预热 + {num_steps} 步采样 | 设备: {self.device}")
        print(f"   AMP: {self.use_amp} | Compile: {self.use_compile}")
        print(f"   cudnn.benchmark: {torch.backends.cudnn.benchmark}")
        print(f"   cudnn.deterministic: {torch.backends.cudnn.deterministic}")
        if torch.cuda.is_available():
            mp = torch.get_float32_matmul_precision()
            print(f"   float32_matmul_precision: {mp}")
        print(f"   batch_size: {self.train_loader.batch_size}")
        print(f"   GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        print(f"{'=' * 50}")

        warmup_timings = {"data_load": [], "forward": [], "backward": [], "optimizer": []}
        timings = {"data_load": [], "forward": [], "backward": [], "optimizer": []}

        self.model.train()
        data_iter = iter(self.train_loader)

        for step_idx in range(total_steps):
            # 1. 数据加载
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            t0 = time.perf_counter()
            images, labels = next(data_iter)
            if images.device != self.device:
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            t1 = time.perf_counter()

            # 2. 前向传播
            if self.use_amp:
                with torch.amp.autocast("cuda"):
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            t2 = time.perf_counter()

            # 3. 反向传播
            scaled_loss = loss / self.grad_accum_steps
            if self.use_amp:
                self.scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            t3 = time.perf_counter()

            # 4. 优化器
            if self.max_grad_norm > 0:
                if self.use_amp:
                    self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            if self.use_amp:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            t4 = time.perf_counter()

            phase_times = {
                "data_load": (t1 - t0) * 1000,
                "forward": (t2 - t1) * 1000,
                "backward": (t3 - t2) * 1000,
                "optimizer": (t4 - t3) * 1000,
            }

            # 预热阶段不记录
            if step_idx < warmup_steps:
                for k, v in phase_times.items():
                    warmup_timings[k].append(v)
                phase_total = sum(phase_times.values())
                print(f"   预热 {step_idx + 1}/{warmup_steps}: {phase_total:.0f}ms")
            else:
                for k, v in phase_times.items():
                    timings[k].append(v)

        # 统计（仅采样阶段）
        print(f"\n{'阶段':<15} {'平均(ms)':<12} {'最小(ms)':<12} {'最大(ms)':<12}")
        print(f"{'-' * 51}")
        total_avg = 0
        for stage, values in timings.items():
            avg = statistics.mean(values)
            total_avg += avg
            mn = min(values)
            mx = max(values)
            print(f"{stage:<15} {avg:>10.2f}  {mn:>10.2f}  {mx:>10.2f}")

        # 评估耗时
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        eval_start = time.perf_counter()
        self.evaluate()
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        eval_time = (time.perf_counter() - eval_start) * 1000
        print(f"{'eval (全部)':<15} {eval_time:>10.2f}  {'':<12} {'':<12}")

        # 预热对比
        if warmup_timings["forward"]:
            warmup_total = sum(statistics.mean(v) for v in warmup_timings.values())
            print(f"\n   预热阶段平均: {warmup_total:.0f}ms/步 → 采样阶段平均: {total_avg:.0f}ms/步")

        # VRAM
        if torch.cuda.is_available():
            print(f"   VRAM: {torch.cuda.memory_allocated() / 1024**2:.0f}MB / "
                  f"{torch.cuda.max_memory_allocated() / 1024**2:.0f}MB peak")

        est_it_per_sec = 1000.0 / total_avg if total_avg > 0 else 0
        print(f"{'-' * 51}")
        print(f"{'单步总计':<15} {total_avg:>10.2f} ms")
        print(f"{'预估稳态速度':<15} ~{est_it_per_sec:.0f} it/s")

        # 瓶颈诊断
        print(f"\n💡 诊断建议:")
        if torch.backends.cudnn.deterministic:
            print(f"   ⚠️  cudnn.deterministic=True：强制使用慢速确定性算法，建议关闭")
        max_stage = max(timings, key=lambda k: statistics.mean(timings[k]))
        max_pct = statistics.mean(timings[max_stage]) / total_avg * 100 if total_avg > 0 else 0
        if est_it_per_sec < 10:
            print(f"   ⚠️  速度异常低（{est_it_per_sec:.0f} it/s），主要瓶颈: {max_stage} ({max_pct:.0f}%)")
            if max_stage == "data_load":
                print(f"       → 数据加载瓶颈，检查 num_workers 是否足够")
            elif max_stage == "forward":
                print(f"       → 前向瓶颈，检查模型大小")
            elif max_stage == "backward":
                print(f"       → 反向瓶颈，检查 AMP 是否正常工作")
        elif est_it_per_sec < 50:
            print(f"   ⚡ 速度正常偏低（{est_it_per_sec:.0f} it/s），瓶颈: {max_stage} ({max_pct:.0f}%)")
            if max_stage == "backward":
                print(f"       → Windows WDDM 驱动有额外开销，属正常范围")
        else:
            print(f"   ✅ 速度正常（{est_it_per_sec:.0f} it/s）")

        print(f"{'=' * 50}\n")

        return {k: statistics.mean(v) for k, v in timings.items()}

    def load_checkpoint(self, checkpoint_path: str) -> dict:
        """从 checkpoint 恢复训练状态（委托给 checkpoint.py）"""
        return _load_checkpoint(self, checkpoint_path)

    @staticmethod
    def load_checkpoint_metadata(path: Path) -> dict | None:
        """轻量读取 checkpoint 元数据（委托给 checkpoint.py）"""
        return _load_checkpoint_metadata(path)

    def _cleanup_old_checkpoints(self):
        """自动清理最旧的定期断点文件（委托给 checkpoint.py）"""
        return _cleanup_old_checkpoints(self.save_dir, self.config)

    def save_checkpoint(self, path: Path, is_best: bool = False, history: dict = None):
        """保存完整训练状态（委托给 checkpoint.py）"""
        return _save_checkpoint(self, path, is_best, history)

    def train_one_epoch(self) -> tuple:
        """训练一个epoch（含 AMP、梯度累积、梯度裁剪、tqdm 进度条）"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(self.train_loader, desc=f"[Epoch {self._current_epoch}]", leave=False)
        for batch_idx, (images, labels) in enumerate(pbar):
            # 数据可能已在 GPU（preload_to_gpu 模式），仅在需要时传输
            if images.device != self.device:
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

            # MixUp：混合批内图像（仅训练，不用于验证）
            if self.mixup_enabled:
                images, labels_a, labels_b, lam = mixup_data(
                    images, labels, alpha=self.mixup_alpha, device=self.device,
                )

            # AMP 自动混合精度
            if self.use_amp:
                with torch.amp.autocast("cuda"):
                    outputs = self.model(images)
                    if self.mixup_enabled:
                        loss = mixup_criterion(self.criterion, outputs, labels_a, labels_b, lam)
                    else:
                        loss = self.criterion(outputs, labels)
                # 按梯度累积步数缩放 loss
                scaled_loss = loss / self.grad_accum_steps
                self.scaler.scale(scaled_loss).backward()
            else:
                outputs = self.model(images)
                if self.mixup_enabled:
                    loss = mixup_criterion(self.criterion, outputs, labels_a, labels_b, lam)
                else:
                    loss = self.criterion(outputs, labels)
                scaled_loss = loss / self.grad_accum_steps
                scaled_loss.backward()

            # 梯度累积：每 accum_steps 步更新一次
            if (batch_idx + 1) % self.grad_accum_steps == 0 or (batch_idx + 1) == len(self.train_loader):
                # 梯度裁剪
                if self.max_grad_norm > 0:
                    if self.use_amp:
                        self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

                if self.use_amp:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()

                # set_to_none=True 跳过梯度 memset，比填充 0 更快
                self.optimizer.zero_grad(set_to_none=True)

            # 统计（延迟 .item() 调用，减少 CUDA 同步）
            # .float() 防止 AMP float16 累积溢出（float16 最大值 65504）
            batch_size = images.size(0)
            running_loss += loss.detach().float() * batch_size
            if self.mixup_enabled:
                # MixUp 下标签是混合的，无法计算 top-1 准确率
                total += batch_size
                correct = correct  # 保持原值，不累加
            else:
                _, predicted = outputs.max(1)
                total += batch_size
                correct += predicted.eq(labels).sum()

            # 每 10 步更新一次进度条（减少同步开销）
            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(self.train_loader):
                batch_loss = (running_loss / total).item()
                if self.mixup_enabled:
                    # MixUp 下标签为混合，无法计算准确率
                    pbar.set_postfix(loss=f"{batch_loss:.4f}")
                else:
                    pbar.set_postfix(loss=f"{batch_loss:.4f}", acc=f"{(correct / total).item():.4f}")

        if self.mixup_enabled:
            # MixUp 下不计算准确率，返回 0.0 占位
            return (running_loss / total).item(), 0.0
        return (running_loss / total).item(), (correct / total).item()

    @torch.no_grad()
    def evaluate(self, loader: DataLoader = None) -> tuple:
        """
        评估模型

        Returns:
            (val_loss, top1_acc, top5_acc) 三元组
        """
        self.model.eval()
        running_loss = torch.tensor(0.0, device=self.device)
        top1_correct = torch.tensor(0, dtype=torch.long, device=self.device)
        top5_correct = torch.tensor(0, dtype=torch.long, device=self.device)
        total = 0

        eval_loader = loader or self.val_loader

        for images, labels in eval_loader:
            # 数据可能已在 GPU（preload_to_gpu 模式），仅在需要时传输
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

            # Top-1 和 Top-5 准确率（类别数不足 5 时取 min）
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

    def fit(self, additional_epochs: int = 0) -> dict:
        """
        主训练循环（开放式，无 epoch 上限）

        Args:
            additional_epochs: 本次会话要额外训练的轮数。
                - 从头训练时默认 0 → 需要 notebook 显式传入（如 fit(30)）
                - 从断点恢复时默认 0 → 需要显式传入（如 fit(10)）

        - KeyboardInterrupt 捕获 + 自动保存（P0-3: 暂停保存）
        - Early stopping
        - Per-epoch 计时 & ETA 预估
        """
        patience_counter = 0
        save_every = self.config["checkpoint"].get("save_every_n_epochs", 5)

        if additional_epochs <= 0:
            print(f"\n{'=' * 60}")
            print(f"⚠️  需要指定训练轮数，例如: trainer.fit({30 if self.start_epoch <= 1 else 10})")
            print(f"   当前权重已训练至第 {self.start_epoch - 1} 轮")
            if self._accumulated_train_time > 0:
                print(f"   累计用时: {_format_duration(self._accumulated_train_time)}")
            print(f"{'=' * 60}")
            return self.history

        total_epochs = self.start_epoch + additional_epochs - 1

        print(f"\n{'=' * 60}")
        print(f"模型: {self.model_name} | 参数量: {self.total_params:,}")
        print(f"设备: {self.device} | AMP: {self.use_amp} | Compile: {self.use_compile}")
        if self.device.type == "cuda":
            print(f"   matmul_precision: {torch.get_float32_matmul_precision()}")

        already_trained = self.start_epoch - 1
        if already_trained > 0:
            print(f"权重已训练: {already_trained} 轮 | 本次续训: {additional_epochs} 轮 → 训练至第 {total_epochs} 轮")
            prev_dur = _format_duration(self._accumulated_train_time)
            if prev_dur:
                print(f"之前累计用时: {prev_dur}")
        else:
            print(f"从头训练: 共 {additional_epochs} 轮 → 训练至第 {total_epochs} 轮")

        print(f"{'=' * 60}\n")

        fit_start = time.time()

        # 确保 cuDNN 使用最优性能设置（set_seed 可能在 Trainer 创建后再次调用）
        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False

        try:
            for epoch in range(self.start_epoch, self.start_epoch + additional_epochs):
                self._current_epoch = epoch
                session_epoch = epoch - self.start_epoch + 1  # 本轮第几轮
                epoch_start = time.time()

                # 训练
                train_loss, train_acc = self.train_one_epoch()

                # 验证
                val_loss, val_acc, val_top5 = self.evaluate()
                current_lr = self.optimizer.param_groups[0]["lr"]

                # 计时统计
                epoch_elapsed = time.time() - epoch_start
                self._total_train_time = self._accumulated_train_time + (time.time() - fit_start)
                epochs_this_session = epoch - self.start_epoch + 1

                # 计算平均每轮耗时（综合历史和当前会话）
                total_history_epochs = len(self.history["train_loss"])  # 之前累积的 epoch 数
                if total_history_epochs > 0 and self._accumulated_train_time > 0:
                    avg_epoch_time = self._total_train_time / total_history_epochs
                else:
                    avg_epoch_time = (time.time() - fit_start) / epochs_this_session

                remaining_epochs = self.start_epoch + additional_epochs - epoch - 1
                eta_seconds = avg_epoch_time * remaining_epochs

                # 记录历史
                self.history["train_loss"].append(train_loss)
                self.history["train_acc"].append(train_acc)
                self.history["val_loss"].append(val_loss)
                self.history["val_acc"].append(val_acc)
                self.history["val_top5_acc"].append(val_top5)
                self.history["lr"].append(current_lr)

                # 学习率调度
                if self.scheduler:
                    if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                        self.scheduler.step(val_acc)
                    else:
                        self.scheduler.step()

                # 最优模型保存
                marker = ""
                if val_acc > self.best_val_acc:
                    self.best_val_acc = val_acc
                    self.best_model_state = copy.deepcopy(self.model.state_dict())
                    patience_counter = 0
                    self.save_checkpoint(self.save_dir / "local_best.pth", is_best=False)
                    # 全局最佳：仅当超过跨会话历史记录时才保存
                    if val_acc > self.global_best_val_acc:
                        self.global_best_val_acc = val_acc
                        self.global_best_model_state = copy.deepcopy(self.model.state_dict())
                        self.save_checkpoint(self.save_dir / "global_best.pth", is_best=True)
                    marker = " *"
                else:
                    patience_counter += 1

                # 定期保存（带时间戳，区分不同运行）
                if epoch % save_every == 0:
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    self.save_checkpoint(
                        self.save_dir / f"checkpoint_epoch{epoch:03d}_{ts}.pth",
                        history=copy.deepcopy(self.history),
                    )
                    # 自动清理最旧的定期断点
                    self._cleanup_old_checkpoints()

                epoch_str = _format_duration(epoch_elapsed)
                eta_str = _format_duration(eta_seconds) if remaining_epochs > 0 else "--"

                print(
                    f"[Epoch {epoch} | 本轮 {session_epoch}/{additional_epochs}] "
                    f"Train: {train_loss:.4f}/{train_acc:.4f} | "
                    f"Val: {val_loss:.4f}/{val_acc:.4f} (T5:{val_top5:.4f}) | "
                    f"LR: {current_lr:.6f} | "
                    f"{epoch_str} | ETA {eta_str}{marker}"
                )

                # Early stopping（兼顾 val_acc 和 val_loss）
                if patience_counter >= self.patience:
                    print(f"\nEarly stopping at [Epoch {epoch} | 本轮 {session_epoch}/{additional_epochs}], "
                          f"best val_acc: {self.best_val_acc:.4f}")
                    break

                # val_loss 辅助监控：与 patience_counter 整合
                # 当 val_loss 连续 val_loss_patience 轮持续上升（超过阈值）时触发
                if self.val_loss_patience > 0 and len(self.history["val_loss"]) >= self.val_loss_patience:
                    recent_losses = self.history["val_loss"][-self.val_loss_patience:]
                    if min(recent_losses) > recent_losses[0] * self.val_loss_threshold:
                        print(f"\nVal loss 持续上升（最后 {self.val_loss_patience} 轮），自动停止 (第 {epoch} 轮)")
                        break

                # LR 下界熔断：LR 降至极低 (< 1e-7) 时自动停止
                if current_lr < 1e-7:
                    print(f"\nLR 已降至 {current_lr:.2e}，自动停止训练 (第 {epoch} 轮)")
                    print(f"  best val_acc: {self.best_val_acc:.4f}")
                    break

        except KeyboardInterrupt:
            # P0-3: 暂停时自动保存断点（带时间戳）
            self._total_train_time = self._accumulated_train_time + (time.time() - fit_start)
            interrupted_epoch = getattr(self, '_current_epoch', self.start_epoch)
            print(f"\n⚠️  训练被用户中断 (第 {interrupted_epoch} 轮)")
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            interrupt_path = self.save_dir / f"interrupted_epoch{interrupted_epoch:03d}_{ts}.pth"
            self.save_checkpoint(
                interrupt_path,
                history=copy.deepcopy(self.history),
            )
            print(f"✅ 断点已自动保存至: {interrupt_path}")
            print(f"   可通过 load_checkpoint('{interrupt_path}') 恢复训练\n")

        # 保存训练历史
        history_path = self.log_dir / "history.json"
        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(self.history, f, indent=2)

        total_dur = _format_duration(self._total_train_time)
        final_epoch = getattr(self, '_current_epoch', self.start_epoch - 1)
        print(f"\n{'=' * 60}")
        print(f"训练完成 | 训练至第 {final_epoch} 轮 | 最优 val_acc: {self.best_val_acc:.4f} | 总用时: {total_dur}")
        print(f"训练历史: {history_path}")
        print(f"{'=' * 60}")

        return self.history


# ============================================================
# 以下函数已迁移至 training/checkpoint.py
# find_resume_checkpoint, select_checkpoint_widget,
# _build_checkpoint_label, _build_summary_card
#
# 通过 import 从 checkpoint.py 重导出，保持向后兼容。
# 查看实现请打开 training/checkpoint.py
# ============================================================

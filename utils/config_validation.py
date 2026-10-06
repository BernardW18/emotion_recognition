"""
训练配置集中校验 — 修复 F13

在训练启动前解析并校验"实际生效配置"：
  - 未知配置键、非法取值一律明确报错（不静默忽略、不静默用默认值）
  - 当前三个模型仅支持 48×48 输入，其他尺寸在模型/训练启动前报错
  - checkpoint.save_best / monitor_metric、augmentation 总开关等开关项取值校验
  - patience=0 的语义：禁用对应早停（acc 用 training.patience，loss 用 val_loss_patience）

用法:
    from utils.config_validation import ConfigValidationError, validate_config
    validate_config(config, model_name="micro_resnet")
"""

__all__ = ["ConfigValidationError", "validate_config", "SUPPORTED_MODELS"]

import logging
import math
from typing import Any, NoReturn

from utils.activations import ACTIVATION_REGISTRY

logger = logging.getLogger("config_validation")

SUPPORTED_MODELS = ("mini_cnn", "vgg_lite", "micro_resnet")

_ALLOWED_ROOT_KEYS = {
    "data", "dataloader", "augmentation", "training", "models", "checkpoint", "seed",
}
_ALLOWED_DATA_KEYS = {"dataset_path", "image_size", "num_classes", "class_names"}
_ALLOWED_DATALOADER_KEYS = {
    "num_workers", "pin_memory", "persistent_workers", "prefetch_factor",
    "class_balanced_sampling",
}
_ALLOWED_AUG_KEYS = {
    "enabled", "random_horizontal_flip", "random_rotation", "random_affine_translate",
    "color_jitter_brightness", "color_jitter_contrast", "random_erase",
    "class_specific", "mixup",
}
_ALLOWED_CLASS_AUG_KEYS = {
    "enabled", "target_classes", "augment_prob", "extra_rotation",
    "extra_translate", "extra_erase_prob",
}
_ALLOWED_MIXUP_KEYS = {"enabled", "alpha"}
_ALLOWED_TRAINING_KEYS = {
    "batch_size", "num_epochs", "learning_rate", "weight_decay", "optimizer",
    "activation", "scheduler", "scheduler_step_size", "scheduler_gamma", "scheduler_t0",
    "patience", "val_loss_patience", "val_loss_threshold", "loss_type", "cb_focal_beta",
    "amp", "torch_compile", "torch_compile_mode", "gradient_accumulation_steps",
    "max_grad_norm", "cudnn_deterministic",
}
_ALLOWED_MODEL_KEYS = {
    "learning_rate", "batch_size", "num_epochs", "dropout", "activation",
    "scheduler_t0", "use_se",
}
_ALLOWED_CHECKPOINT_KEYS = {
    "save_best", "save_every_n_epochs", "monitor_metric", "max_checkpoint_files",
}

_OPTIMIZERS = {"adam", "adamw", "sgd"}
_SCHEDULERS = {"cosine", "cosine_warm", "step", "plateau", "none"}
_LOSS_TYPES = {"focal", "cb_focal", "cross_entropy"}
_MONITOR_METRICS = {"val_acc", "val_loss"}
_TORCH_COMPILE_MODES = {"default", "reduce-overhead", "max-autotune"}


class ConfigValidationError(ValueError):
    """配置校验失败：启动前明确报错，不静默使用默认值或忽略非法值。"""


def _fail(path: str, msg: str) -> NoReturn:
    raise ConfigValidationError(f"[{path}] {msg}")


def _check_unknown(section: dict, allowed: set, path: str) -> None:
    unknown = sorted(set(section) - allowed)
    if unknown:
        _fail(path, f"存在未被支持的配置键: {unknown}（允许的键: {sorted(allowed)}）")


def _ensure_section(config: dict, key: str, allowed: set) -> dict:
    value = config.get(key)
    if not isinstance(value, dict):
        _fail(key, f"必须是配置段（dict），得到 {type(value).__name__}")
    _check_unknown(value, allowed, key)
    return value


def _optional_section(config: dict, key: str, allowed: set) -> dict:
    """可缺省段：缺省时按空段处理（各调用点使用默认值）；提供了就必须合法。"""
    value = config.get(key, {})
    if not isinstance(value, dict):
        _fail(key, f"必须是配置段（dict），得到 {type(value).__name__}")
    _check_unknown(value, allowed, key)
    return value


def _ensure_bool(value: Any, path: str) -> bool:
    if not isinstance(value, bool):
        _fail(path, f"必须是布尔值 true/false，得到 {value!r}")
    return value


def _ensure_int(value: Any, path: str, *, min_value: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        _fail(path, f"必须是整数，得到 {value!r}")
    if min_value is not None and value < min_value:
        _fail(path, f"必须 >= {min_value}，得到 {value}")
    return value


def _ensure_number(value: Any, path: str, *, min_value: float | None = None,
                   exclusive_min: bool = False, max_value: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(path, f"必须是数值，得到 {value!r}")
    if not math.isfinite(float(value)):
        _fail(path, f"必须是有限数值（不允许 NaN/Inf），得到 {value!r}")
    value = float(value)
    if min_value is not None:
        if exclusive_min and value <= min_value:
            _fail(path, f"必须 > {min_value}，得到 {value}")
        if not exclusive_min and value < min_value:
            _fail(path, f"必须 >= {min_value}，得到 {value}")
    if max_value is not None and value > max_value:
        _fail(path, f"必须 <= {max_value}，得到 {value}")
    return float(value)


def _ensure_choice(value: Any, path: str, choices: set) -> Any:
    if value not in choices:
        _fail(path, f"取值 {value!r} 不被支持，可选: {sorted(choices)}")
    return value


def validate_config(config: dict, *, model_name: str | None = None) -> None:
    """
    集中校验完整训练配置；任何非法值/未知键抛出 ConfigValidationError。

    Args:
        config: 已合并 CLI 覆盖的"实际生效"配置
        model_name: 若指定，额外校验该模型段的完整性与合法性
    """
    if not isinstance(config, dict):
        _fail("config", f"必须是 dict，得到 {type(config).__name__}")
    _check_unknown(config, _ALLOWED_ROOT_KEYS, "config")

    # ---------------- data ----------------
    data = _ensure_section(config, "data", _ALLOWED_DATA_KEYS)
    image_size = data.get("image_size", 48)
    if image_size != 48:
        _fail(
            "data.image_size",
            f"={image_size} 不被支持：当前三个模型仅支持 48×48 输入（F13；不为其他尺寸改架构）",
        )
    class_names = data.get("class_names")
    if not isinstance(class_names, list) or not class_names or not all(
        isinstance(n, str) for n in class_names
    ):
        _fail("data.class_names", f"必须是非空字符串列表，得到 {class_names!r}")
    num_classes = data.get("num_classes")
    _ensure_int(num_classes, "data.num_classes", min_value=1)
    if num_classes != len(class_names):
        _fail("data.num_classes", f"={num_classes} 与 class_names 长度 {len(class_names)} 不一致")
    if not isinstance(data.get("dataset_path"), str) or not data["dataset_path"]:
        _fail("data.dataset_path", f"必须是非空字符串，得到 {data.get('dataset_path')!r}")

    # ---------------- dataloader ----------------
    dl = _optional_section(config, "dataloader", _ALLOWED_DATALOADER_KEYS)
    num_workers = _ensure_int(dl.get("num_workers", 0), "dataloader.num_workers", min_value=0)
    _ensure_bool(dl.get("pin_memory", True), "dataloader.pin_memory")
    _ensure_bool(dl.get("persistent_workers", False), "dataloader.persistent_workers")
    _ensure_bool(dl.get("class_balanced_sampling", False), "dataloader.class_balanced_sampling")
    prefetch = dl.get("prefetch_factor", 2)
    if num_workers > 0:
        _ensure_int(prefetch, "dataloader.prefetch_factor", min_value=1)
    # num_workers=0 时 persistent_workers/prefetch_factor 由 create_dataloaders 自动关闭并提示

    # ---------------- augmentation ----------------
    aug = _optional_section(config, "augmentation", _ALLOWED_AUG_KEYS)
    _ensure_bool(aug.get("enabled", False), "augmentation.enabled")
    for key in ("random_horizontal_flip", "random_affine_translate", "color_jitter_brightness",
                "color_jitter_contrast"):
        if key in aug:
            _ensure_number(aug[key], f"augmentation.{key}", min_value=0.0, max_value=1.0)
    if "random_rotation" in aug:
        _ensure_number(aug["random_rotation"], "augmentation.random_rotation", min_value=0.0)
    if "random_erase" in aug:
        _ensure_bool(aug["random_erase"], "augmentation.random_erase")

    class_spec = aug.get("class_specific", {})
    if not isinstance(class_spec, dict):
        _fail("augmentation.class_specific", f"必须是 dict，得到 {type(class_spec).__name__}")
    _check_unknown(class_spec, _ALLOWED_CLASS_AUG_KEYS, "augmentation.class_specific")
    _ensure_bool(class_spec.get("enabled", False), "augmentation.class_specific.enabled")
    targets = class_spec.get("target_classes", [])
    if not isinstance(targets, list):
        _fail("augmentation.class_specific.target_classes", "必须是列表")
    for t in targets:
        _ensure_int(t, "augmentation.class_specific.target_classes[]", min_value=0)
        if t >= num_classes:
            _fail(
                "augmentation.class_specific.target_classes",
                f"类别索引 {t} 超出 [0, {num_classes}) 范围",
            )
    if "augment_prob" in class_spec:
        _ensure_number(class_spec["augment_prob"], "augmentation.class_specific.augment_prob",
                       min_value=0.0, max_value=1.0)
    if "extra_rotation" in class_spec:
        _ensure_number(class_spec["extra_rotation"],
                       "augmentation.class_specific.extra_rotation", min_value=0.0)
    for key in ("extra_translate", "extra_erase_prob"):
        if key in class_spec:
            _ensure_number(class_spec[key], f"augmentation.class_specific.{key}",
                           min_value=0.0, max_value=1.0)

    mixup = aug.get("mixup", {})
    if not isinstance(mixup, dict):
        _fail("augmentation.mixup", f"必须是 dict，得到 {type(mixup).__name__}")
    _check_unknown(mixup, _ALLOWED_MIXUP_KEYS, "augmentation.mixup")
    _ensure_bool(mixup.get("enabled", False), "augmentation.mixup.enabled")
    if "alpha" in mixup:
        _ensure_number(
            mixup["alpha"], "augmentation.mixup.alpha", min_value=0.0, exclusive_min=True
        )

    # ---------------- training ----------------
    tr = _ensure_section(config, "training", _ALLOWED_TRAINING_KEYS)
    _ensure_int(tr.get("batch_size"), "training.batch_size", min_value=1)
    _ensure_int(tr.get("num_epochs"), "training.num_epochs", min_value=1)
    _ensure_number(
        tr.get("learning_rate"), "training.learning_rate", min_value=0.0, exclusive_min=True
    )
    _ensure_number(tr.get("weight_decay", 0.0), "training.weight_decay", min_value=0.0)
    _ensure_choice(tr.get("optimizer", "adam"), "training.optimizer", _OPTIMIZERS)
    _ensure_choice(tr.get("activation", "relu"), "training.activation", set(ACTIVATION_REGISTRY))
    _ensure_choice(tr.get("scheduler", "none"), "training.scheduler", _SCHEDULERS)
    if "scheduler_step_size" in tr:
        _ensure_int(tr["scheduler_step_size"], "training.scheduler_step_size", min_value=1)
    if "scheduler_gamma" in tr:
        _ensure_number(
            tr["scheduler_gamma"], "training.scheduler_gamma", min_value=0.0, max_value=1.0
        )
    if "scheduler_t0" in tr:
        _ensure_int(tr["scheduler_t0"], "training.scheduler_t0", min_value=1)
    # patience=0 语义：禁用对应早停（>=1 才生效）
    _ensure_int(tr.get("patience", 7), "training.patience", min_value=0)
    _ensure_int(tr.get("val_loss_patience", 0), "training.val_loss_patience", min_value=0)
    # val_loss 恶化判定阈值必须严格 > 1（否则"上升 5%"之类的语义无意义）
    _ensure_number(tr.get("val_loss_threshold", 1.05), "training.val_loss_threshold",
                   min_value=1.0, exclusive_min=True)
    _ensure_choice(tr.get("loss_type", "focal"), "training.loss_type", _LOSS_TYPES)
    if "cb_focal_beta" in tr:
        _ensure_number(tr["cb_focal_beta"], "training.cb_focal_beta", min_value=0.0, max_value=1.0)
        if float(tr["cb_focal_beta"]) >= 1.0:
            _fail("training.cb_focal_beta", f"必须 < 1，得到 {tr['cb_focal_beta']}")
    _ensure_bool(tr.get("amp", True), "training.amp")
    _ensure_bool(tr.get("torch_compile", False), "training.torch_compile")
    if "torch_compile_mode" in tr:
        _ensure_choice(
            tr["torch_compile_mode"], "training.torch_compile_mode", _TORCH_COMPILE_MODES
        )
    _ensure_int(tr.get("gradient_accumulation_steps", 1),
                "training.gradient_accumulation_steps", min_value=1)
    _ensure_number(tr.get("max_grad_norm", 0.0), "training.max_grad_norm", min_value=0.0)
    _ensure_bool(tr.get("cudnn_deterministic", False), "training.cudnn_deterministic")

    # ---------------- models ----------------
    models = config.get("models")
    if not isinstance(models, dict) or not models:
        _fail("models", "必须是包含模型配置的 dict")
    _check_unknown(models, set(SUPPORTED_MODELS), "models")
    if model_name is not None and model_name not in models:
        _fail("models", f"缺少指定模型 {model_name!r} 的配置段")
    for name, mcfg in models.items():
        if not isinstance(mcfg, dict):
            _fail(f"models.{name}", f"必须是 dict，得到 {type(mcfg).__name__}")
        _check_unknown(mcfg, _ALLOWED_MODEL_KEYS, f"models.{name}")
        if name == model_name:
            required = {"learning_rate", "dropout", "activation"}
            missing = sorted(required - set(mcfg))
            if missing:
                _fail(f"models.{name}", f"缺少必需字段: {missing}（不静默使用模型默认值）")
        if "learning_rate" in mcfg:
            _ensure_number(mcfg["learning_rate"], f"models.{name}.learning_rate",
                           min_value=0.0, exclusive_min=True)
        if "batch_size" in mcfg:
            _ensure_int(mcfg["batch_size"], f"models.{name}.batch_size", min_value=1)
        if "num_epochs" in mcfg:
            _ensure_int(mcfg["num_epochs"], f"models.{name}.num_epochs", min_value=1)
        if "dropout" in mcfg:
            _ensure_number(mcfg["dropout"], f"models.{name}.dropout", min_value=0.0, max_value=1.0)
            if float(mcfg["dropout"]) >= 1.0:
                _fail(f"models.{name}.dropout", f"必须 < 1，得到 {mcfg['dropout']}")
        if "activation" in mcfg:
            _ensure_choice(
                mcfg["activation"], f"models.{name}.activation", set(ACTIVATION_REGISTRY)
            )
        if "scheduler_t0" in mcfg:
            _ensure_int(mcfg["scheduler_t0"], f"models.{name}.scheduler_t0", min_value=1)
        if "use_se" in mcfg:
            use_se = _ensure_bool(mcfg["use_se"], f"models.{name}.use_se")
            if use_se and name != "micro_resnet":
                _fail(f"models.{name}.use_se", f"{name} 不支持 SE 模块")

    # ---------------- checkpoint ----------------
    ckpt = _optional_section(config, "checkpoint", _ALLOWED_CHECKPOINT_KEYS)
    _ensure_bool(ckpt.get("save_best", True), "checkpoint.save_best")
    _ensure_int(ckpt.get("save_every_n_epochs", 5), "checkpoint.save_every_n_epochs", min_value=1)
    _ensure_choice(
        ckpt.get("monitor_metric", "val_acc"), "checkpoint.monitor_metric", _MONITOR_METRICS
    )
    _ensure_int(ckpt.get("max_checkpoint_files", 5), "checkpoint.max_checkpoint_files", min_value=0)

    # ---------------- seed ----------------
    seed = config.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int):
        _fail("seed", f"必须是整数，得到 {seed!r}")
    if not 0 <= seed < 2**32:
        _fail("seed", f"必须在 [0, 2^32) 范围内（torch 手动种子要求），得到 {seed}")

    logger.debug("配置校验通过（model_name=%s）", model_name)

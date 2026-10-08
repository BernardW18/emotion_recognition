"""
模型规格（model_spec）与统一构造 / 加载工厂 — 修复 F01

背景：推理与评估曾直接用模型默认参数构造网络。MicroResNet 保存配置为 GELU、
默认构造为 ReLU；激活函数不含可学习参数，因此 load_state_dict() 即使
strict=True 也不会报错，但预测会改变。本模块是唯一的模型构造与 checkpoint
加载入口，训练导出、离线评估、Notebook 与应用都必须经由这里，保证
「实际构造的模型结构 == 训练时保存的结构」。

旧 checkpoint 迁移规则（确定性，不允许静默猜默认值）：
  - activation : 读取旧 checkpoint 的 config.activation；缺失必须由 spec_override 显式提供
  - use_se     : MicroResNet 由 state_dict 是否含 ".se." 参数键判定；其余模型无 SE
  - dropout    : 读取旧 config.dropout；缺失记 None（对 eval 无影响，仅推理可用）
  - num_classes / class_names : 由分类头输出维度推断；等于标准七类时使用项目
    标准映射，否则必须由 spec_override 提供 class_names

用法:
    from utils.model_spec import load_model_from_checkpoint, build_model_from_spec

    model, spec, meta = load_model_from_checkpoint("path/to/best.pth", device="cpu")
    print(meta["checkpoint_sha256"], spec.activation)
"""

__all__ = [
    "MODEL_SPEC_VERSION",
    "SUPPORTED_MODELS",
    "SUPPORTED_IMAGE_SIZE",
    "ModelSpec",
    "build_model_from_spec",
    "make_spec_from_config",
    "resolve_spec_from_checkpoint",
    "load_model_from_checkpoint",
    "infer_num_classes_from_state_dict",
    "count_parameters",
    "file_sha256",
]

import hashlib
import logging
from dataclasses import dataclass, field
from dataclasses import fields as dataclass_fields
from pathlib import Path
from typing import Any

import torch

from utils.activations import ACTIVATION_REGISTRY
from utils.constants import CLASS_NAMES
from utils.model_structure import (
    DEFAULT_BLOCKS,
    DEFAULT_CHANNELS,
    DEFAULT_POOL_ORDER,
    STRUCTURE_FIELDS,
    validate_micro_structure,
)

logger = logging.getLogger("model_spec")

MODEL_SPEC_VERSION = 2
SUPPORTED_MODELS = ("mini_cnn", "vgg_lite", "micro_resnet")
SUPPORTED_IMAGE_SIZE = 48
SUPPORTED_INPUT_CHANNELS = 1
ALLOWED_NORMALIZE = ("x/255.0",)
SE_SUPPORTED_MODELS = ("micro_resnet",)

# 各模型分类头权重键（用于从 state_dict 推断类别数）
# 索引为输出层在各模型 Sequential 中的实际位置（Flatten/Dropout 等模块同样占位）
_HEAD_KEYS = {
    "mini_cnn": "classifier.4.weight",
    "vgg_lite": "classifier.7.weight",
    "micro_resnet": "classifier.2.weight",
}


def file_sha256(path: str | Path, chunk_size: int = 1 << 20) -> str:
    """计算文件 SHA-256 十六进制摘要。"""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


@dataclass(frozen=True)
class ModelSpec:
    """版本化模型规格：描述「权重是如何被训练/构造的」的全部结构信息。"""

    spec_version: int
    model_name: str
    class_names: tuple[str, ...]
    activation: str
    dropout: float | None
    use_se: bool
    input_channels: int
    image_size: int
    normalize: str
    spec_source: str = "training"
    migration_notes: tuple[str, ...] = field(default_factory=tuple)
    blocks: tuple[int, ...] | None = None
    channels: tuple[int, ...] | None = None
    pool_order: str | None = None

    @property
    def num_classes(self) -> int:
        return len(self.class_names)

    def to_dict(self) -> dict[str, Any]:
        result = {
            "spec_version": self.spec_version,
            "model_name": self.model_name,
            "class_names": list(self.class_names),
            "activation": self.activation,
            "dropout": self.dropout,
            "use_se": self.use_se,
            "input_channels": self.input_channels,
            "image_size": self.image_size,
            "normalize": self.normalize,
            "spec_source": self.spec_source,
            "migration_notes": list(self.migration_notes),
        }
        if self.spec_version == 2:
            result.update(blocks=list(self.blocks or ()), channels=list(self.channels or ()),
                          pool_order=self.pool_order)
        return result

    def display_summary(self) -> str:
        """人类可读摘要（应用/日志展示用）。"""
        dropout = "unknown" if self.dropout is None else f"{self.dropout:g}"
        return (
            f"{self.model_name} | classes={self.num_classes} | "
            f"activation={self.activation} | dropout={dropout} | use_se={self.use_se}"
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModelSpec":
        """严格解析并校验 model_spec；任何非法字段都会明确报错。"""
        if not isinstance(data, dict):
            raise ValueError(f"model_spec 必须是 dict，得到 {type(data).__name__}")

        version = data.get("spec_version")
        if type(version) is not int or version not in (1, 2):
            raise ValueError(
                f"不支持的 model_spec 版本: {version!r}（当前支持 1/2）"
            )

        model_name = data.get("model_name")
        if model_name not in SUPPORTED_MODELS:
            raise ValueError(
                f"未知模型名: {model_name!r}，可选: {list(SUPPORTED_MODELS)}"
            )

        class_names = data.get("class_names")
        if not class_names or not all(isinstance(n, str) for n in class_names):
            raise ValueError(f"class_names 必须是非空字符串列表，得到: {class_names!r}")
        class_names = tuple(class_names)
        if len(set(class_names)) != len(class_names):
            raise ValueError(f"class_names 存在重复: {class_names}")

        activation = data.get("activation")
        if activation not in ACTIVATION_REGISTRY:
            raise ValueError(
                f"不支持的激活函数: {activation!r}，可选: {list(ACTIVATION_REGISTRY)}"
            )

        dropout = data.get("dropout")
        if dropout is not None:
            dropout = float(dropout)
            if not 0.0 <= dropout < 1.0:
                raise ValueError(f"dropout 必须在 [0, 1) 范围内，得到 {dropout}")

        if "use_se" not in data:
            raise ValueError("model_spec 缺少 use_se 字段（必须显式提供）")
        if type(data["use_se"]) is not bool:
            raise ValueError("model_spec.use_se 必须为布尔值")
        use_se = data["use_se"]
        if use_se and model_name not in SE_SUPPORTED_MODELS:
            raise ValueError(f"{model_name} 不支持 use_se")

        input_channels = data.get("input_channels")
        image_size = data.get("image_size")
        if input_channels != SUPPORTED_INPUT_CHANNELS or image_size != SUPPORTED_IMAGE_SIZE:
            raise ValueError(
                f"当前项目仅支持 {SUPPORTED_INPUT_CHANNELS}x{SUPPORTED_IMAGE_SIZE}x"
                f"{SUPPORTED_IMAGE_SIZE} 输入（F13），得到 "
                f"{input_channels}x{image_size}x{image_size}"
            )

        normalize = data.get("normalize")
        if normalize not in ALLOWED_NORMALIZE:
            raise ValueError(f"不支持的预处理: {normalize!r}，可选: {list(ALLOWED_NORMALIZE)}")

        migration_notes = data.get("migration_notes") or ()
        if not isinstance(migration_notes, (list, tuple)):
            raise ValueError("migration_notes 必须是字符串列表")
        migration_notes = tuple(str(n) for n in migration_notes)

        blocks: tuple[int, ...] | None = None
        channels: tuple[int, ...] | None = None
        pool_order: str | None = None
        if version == 1:
            if STRUCTURE_FIELDS.intersection(data):
                raise ValueError("model_spec v1 固定原结构，结构字段必须使用 v2")
        else:
            if model_name != "micro_resnet":
                raise ValueError("model_spec v2 结构字段仅支持 micro_resnet")
            if not STRUCTURE_FIELDS.issubset(data):
                raise ValueError("model_spec v2 缺少 blocks/channels/pool_order")
            blocks, channels, pool_order = validate_micro_structure(
                data["blocks"], data["channels"], data["pool_order"],
            )

        return cls(
            spec_version=version,
            model_name=model_name,
            class_names=class_names,
            activation=activation,
            dropout=dropout,
            use_se=use_se,
            input_channels=input_channels,
            image_size=image_size,
            normalize=normalize,
            spec_source=str(data.get("spec_source", "unknown")),
            migration_notes=migration_notes,
            blocks=blocks, channels=channels, pool_order=pool_order,
        )


def build_model_from_spec(
    spec: ModelSpec, *, allow_unknown_dropout: bool = False
) -> torch.nn.Module:
    """
    唯一模型构造入口：按 ModelSpec 构造模型（不做任何默认参数替换）。

    Args:
        spec: 模型规格（须已通过校验）
        allow_unknown_dropout: spec.dropout 为 None 时是否允许构造。
            None 只可能来自旧权重迁移；此时以 p=0 构造，仅保证 eval 数值一致，
            不能用于继续训练。

    Raises:
        ValueError: dropout 未知且不允许（训练场景），或模型名/参数非法
    """
    if spec.dropout is None and not allow_unknown_dropout:
        raise ValueError(
            "spec.dropout 未知（None）：该规格只允许用于推理评估；"
            "继续训练需要完整配置（请提供 spec_override）"
        )
    dropout = 0.0 if spec.dropout is None else spec.dropout

    if spec.model_name == "mini_cnn":
        from models.mini_cnn import MiniCNN

        return MiniCNN(num_classes=spec.num_classes, dropout=dropout, activation=spec.activation)
    if spec.model_name == "vgg_lite":
        from models.vgg_lite import VGGLite

        return VGGLite(num_classes=spec.num_classes, dropout=dropout, activation=spec.activation)
    if spec.model_name == "micro_resnet":
        from models.micro_resnet import MicroResNet

        return MicroResNet(
            num_classes=spec.num_classes,
            dropout=dropout,
            activation=spec.activation,
            use_se=spec.use_se,
            blocks=DEFAULT_BLOCKS if spec.spec_version == 1 else spec.blocks,
            channels=DEFAULT_CHANNELS if spec.spec_version == 1 else spec.channels,
            pool_order=DEFAULT_POOL_ORDER if spec.spec_version == 1 else spec.pool_order,
        )
    raise ValueError(f"未知模型名: {spec.model_name!r}")


def make_spec_from_config(config: dict[str, Any], model_name: str) -> ModelSpec:
    """
    从训练配置生成 ModelSpec（训练侧入口）。

    要求配置显式提供关键字段（activation / dropout），缺失时明确报错，
    不静默采用模型默认参数。
    """
    if model_name not in SUPPORTED_MODELS:
        raise ValueError(f"未知模型名: {model_name!r}，可选: {list(SUPPORTED_MODELS)}")

    model_cfg = config.get("models", {}).get(model_name)
    if model_cfg is None:
        raise ValueError(f"配置缺少 models.{model_name} 段")

    data_cfg = config.get("data", {})
    training_cfg = config.get("training", {})

    activation = model_cfg.get("activation", training_cfg.get("activation"))
    if activation is None:
        raise ValueError(
            f"配置缺少 models.{model_name}.activation 与 training.activation，"
            "无法确定激活函数（不静默使用默认值）"
        )

    if "dropout" not in model_cfg:
        raise ValueError(
            f"配置缺少 models.{model_name}.dropout（不静默使用默认值）"
        )
    dropout = float(model_cfg["dropout"])

    use_se = bool(model_cfg.get("use_se", False))

    class_names = data_cfg.get("class_names")
    if not class_names:
        raise ValueError("配置缺少 data.class_names")
    num_classes = data_cfg.get("num_classes", len(class_names))
    if num_classes != len(class_names):
        raise ValueError(
            f"data.num_classes={num_classes} 与 class_names 长度 {len(class_names)} 不一致"
        )

    image_size = data_cfg.get("image_size", SUPPORTED_IMAGE_SIZE)
    if image_size != SUPPORTED_IMAGE_SIZE:
        raise ValueError(
            f"当前项目仅支持 {SUPPORTED_IMAGE_SIZE}x{SUPPORTED_IMAGE_SIZE} 输入"
            f"（data.image_size={image_size} 不被支持）"
        )

    structure = STRUCTURE_FIELDS.intersection(model_cfg)
    if structure and (model_name != "micro_resnet" or structure != STRUCTURE_FIELDS):
        raise ValueError("结构字段仅支持 MicroResNet，且 blocks/channels/pool_order 必须全部提供")
    return ModelSpec.from_dict(
        {
            "spec_version": 2 if structure else 1,
            "model_name": model_name,
            "class_names": list(class_names),
            "activation": str(activation),
            "dropout": dropout,
            "use_se": use_se,
            "input_channels": SUPPORTED_INPUT_CHANNELS,
            "image_size": image_size,
            "normalize": ALLOWED_NORMALIZE[0],
            "spec_source": "training",
            **({key: model_cfg[key] for key in STRUCTURE_FIELDS} if structure else {}),
        }
    )


def infer_num_classes_from_state_dict(state_dict: dict[str, Any], model_name: str) -> int:
    """从分类头权重形状推断类别数。"""
    key = _HEAD_KEYS.get(model_name)
    if key is None:
        raise ValueError(f"未知模型名: {model_name!r}")
    if key not in state_dict:
        raise ValueError(f"state_dict 缺少分类头参数 {key!r}，无法推断类别数")
    return int(state_dict[key].shape[0])


def resolve_spec_from_checkpoint(
    checkpoint: dict[str, Any],
    *,
    spec_override: dict[str, Any] | None = None,
) -> tuple[ModelSpec, list[str]]:
    """
    从 checkpoint 解析 ModelSpec。

    - 新格式：直接读取并校验 "model_spec" 字段
    - 旧格式（无 model_spec）：按确定性迁移规则重建，并返回 notes 记录依据；
      无法确定的关键字段必须由 spec_override 显式提供，否则报错
    - spec_override 为 dict，字段优先于迁移结果（显式迁移配置入口）
    """
    notes: list[str] = []
    override = dict(spec_override) if spec_override else {}

    if override:
        valid_keys = {f.name for f in dataclass_fields(ModelSpec)}
        unknown = sorted(set(override) - valid_keys)
        if unknown:
            raise ValueError(
                f"spec_override 含未知字段: {unknown}（合法字段: {sorted(valid_keys)}）"
            )

    if "model_spec" in checkpoint:
        raw = dict(checkpoint["model_spec"])
        if override:
            raw.update(override)
            notes.append(f"spec_override 显式覆盖字段: {sorted(override.keys())}")
        spec = ModelSpec.from_dict(raw)
        notes.append(f"model_spec v{spec.spec_version}（来源: {spec.spec_source}）")
        return spec, notes

    # ---- 旧格式迁移 ----
    model_name = checkpoint.get("model_name")
    if model_name not in SUPPORTED_MODELS:
        raise ValueError(
            f"checkpoint 缺少有效的 model_name（得到 {model_name!r}），无法迁移"
        )

    state_dict = checkpoint.get("model_state_dict")
    if state_dict is None:
        raise ValueError("checkpoint 缺少 model_state_dict，无法迁移")

    old_cfg = checkpoint.get("config") or {}

    # activation：override > 旧 config；都没有则报错
    activation = old_cfg.get("activation")
    if "activation" in override:
        activation = override["activation"]
        notes.append("activation 由 spec_override 提供")
    elif activation is None:
        raise ValueError(
            f"旧 checkpoint（{model_name}）未记录 activation，无法确定性迁移。"
            "请通过 spec_override 显式提供，例如 {'activation': 'gelu'}"
        )
    else:
        notes.append("activation 来自旧 checkpoint 的 config 字段")

    # use_se：MicroResNet 由 state_dict 键判定；其余模型无 SE
    if model_name == "micro_resnet":
        use_se = any(".se." in k for k in state_dict)
        notes.append(f"use_se={use_se} 由 state_dict 参数键判定（旧 config 未记录该字段）")
    else:
        use_se = False

    # dropout：override > 旧 config；缺失记 None（仅推理）
    dropout = old_cfg.get("dropout")
    if "dropout" in override:
        dropout = override["dropout"]
        notes.append("dropout 由 spec_override 提供")
    elif dropout is None:
        notes.append("dropout 未记录：对 eval 无影响；该规格仅用于推理，不用于续训")

    # class_names：override > 标准七类（与分类头维度一致时）；否则报错
    num_classes = infer_num_classes_from_state_dict(state_dict, model_name)
    if "class_names" in override:
        class_names: list[str] = list(override["class_names"])
        notes.append("class_names 由 spec_override 提供")
    elif num_classes == len(CLASS_NAMES):
        class_names = list(CLASS_NAMES)
        notes.append("class_names 使用项目标准七类映射（与分类头输出维度一致）")
    else:
        raise ValueError(
            f"分类头输出维度为 {num_classes}（非 7），无法确定类别映射；"
            "请通过 spec_override 提供 class_names"
        )

    partial: dict[str, Any] = {
        "spec_version": 1,
        "model_name": model_name,
        "class_names": class_names,
        "activation": activation,
        "dropout": float(dropout) if dropout is not None else None,
        "use_se": use_se,
        "input_channels": SUPPORTED_INPUT_CHANNELS,
        "image_size": SUPPORTED_IMAGE_SIZE,
        "normalize": ALLOWED_NORMALIZE[0],
        "spec_source": "legacy-migration",
    }

    if override:
        partial.update(override)
        notes.append(f"spec_override 显式覆盖字段: {sorted(override.keys())}")

    partial["migration_notes"] = tuple(notes)
    spec = ModelSpec.from_dict(partial)
    return spec, notes


def load_model_from_checkpoint(
    checkpoint_path: str | Path,
    *,
    device: str | torch.device = "cpu",
    spec_override: dict[str, Any] | None = None,
    strict: bool = True,
) -> tuple[torch.nn.Module, ModelSpec, dict[str, Any]]:
    """
    统一加载入口：checkpoint -> (model[eval], spec, meta)。

    与训练导出、离线评估、Notebook、应用共用同一构造逻辑；
    返回的 meta 含 checkpoint SHA-256、spec、迁移 notes 等可追溯信息。
    """
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint 未找到: {path}")

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if "model_state_dict" not in checkpoint:
        raise ValueError(f"文件不是有效的 checkpoint（缺少 model_state_dict）: {path}")

    spec, notes = resolve_spec_from_checkpoint(checkpoint, spec_override=spec_override)
    model = build_model_from_spec(spec, allow_unknown_dropout=True)

    try:
        model.load_state_dict(checkpoint["model_state_dict"], strict=strict)
    except RuntimeError as e:
        raise RuntimeError(
            f"模型结构与 checkpoint 不匹配（迁移判定可能不完整）: {path}\n{e}"
        ) from e

    model.to(device)
    model.eval()

    meta: dict[str, Any] = {
        "checkpoint_path": str(path),
        "checkpoint_sha256": file_sha256(path),
        "model_spec": spec.to_dict(),
        "spec_notes": notes,
        "eval_only": spec.dropout is None,
    }
    for key in ("epoch", "val_acc", "val_loss", "best_val_acc", "global_best_val_acc", "timestamp"):
        if key in checkpoint:
            value = checkpoint[key]
            meta[key] = float(value) if isinstance(value, float) else value

    return model, spec, meta


def count_parameters(model: torch.nn.Module) -> tuple[int, int]:
    """返回 (总参数量, 可训练参数量)。"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

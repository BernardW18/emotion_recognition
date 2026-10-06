"""
配置校验测试 — F13

验证：现有配置通过；未知键 / 非法值 / 缺失必需字段明确报错（不静默忽略）。
"""

import copy
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest

from training.trainer import load_config
from utils.config_validation import ConfigValidationError, validate_config


@pytest.fixture
def config():
    return load_config()


def test_valid_config_passes(config):
    for model in ("mini_cnn", "vgg_lite", "micro_resnet"):
        validate_config(config, model_name=model)


@pytest.mark.parametrize(
    "mutate,expect_msg",
    [
        (lambda c: c["data"].update(image_size=64), "48"),
        (lambda c: c["training"].update(unknown_key=1), "未被支持的配置键"),
        (lambda c: c["training"].update(val_loss_threshold=1.0), "必须 > 1.0"),
        (lambda c: c["checkpoint"].update(monitor_metric="accuracy"), "不被支持"),
        (lambda c: c["models"]["micro_resnet"].pop("dropout"), "dropout"),
        (lambda c: c["augmentation"]["class_specific"].update(target_classes=[7]), "超出"),
        (lambda c: c["training"].update(patience=-1), "必须 >= 0"),
        (lambda c: c["models"]["mini_cnn"].update(use_se=True), "不支持 SE"),
    ],
)
def test_invalid_configs_raise(config, mutate, expect_msg):
    cfg = copy.deepcopy(config)
    mutate(cfg)
    with pytest.raises(ConfigValidationError, match=f".*{expect_msg}.*"):
        validate_config(cfg, model_name="micro_resnet")

@pytest.mark.parametrize(
    "mutate,expect_msg",
    [
        (lambda c: c["training"].update(learning_rate=float("nan")), "有限数值"),
        (lambda c: c["training"].update(weight_decay=float("inf")), "有限数值"),
        (lambda c: c["training"].update(val_loss_threshold=float("nan")), "有限数值"),
        (lambda c: c["models"]["micro_resnet"].update(learning_rate=float("nan")), "有限数值"),
        (lambda c: c.update(seed=-1), "范围内"),
        (lambda c: c.update(seed=2**32), "范围内"),
        (lambda c: c["augmentation"]["class_specific"].update(extra_translate=float("nan")),
         "有限数值"),
        (lambda c: c["augmentation"]["class_specific"].update(extra_erase_prob=1.5),
         "必须 <= 1.0"),
    ],
)
def test_invalid_numeric_configs_raise(config, mutate, expect_msg):
    cfg = copy.deepcopy(config)
    mutate(cfg)
    with pytest.raises(ConfigValidationError, match=f".*{expect_msg}.*"):
        validate_config(cfg, model_name="micro_resnet")

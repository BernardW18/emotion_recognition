"""Read-only checkpoint v3 validation shared by resume and formal admission."""
import math
from numbers import Real

import torch

FORMAT_VERSION = 3
HISTORY_KEYS = (
    "train_loss", "train_acc", "val_loss", "val_acc", "val_top5_acc", "lr",
    "optimizer_updates", "optimizer_attempts",
)


def _integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} 必须为 >= {minimum} 的整数")
    return value


def _finite(value, name, minimum=None):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"{name} 必须为有限数")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} 必须 >= {minimum}")
    return float(value)


def validate_scaler_state(state):
    if not isinstance(state, dict):
        raise ValueError("scaler_state_dict 缺失或非法")
    if _finite(state.get("scale"), "scale", 0) == 0:
        raise ValueError("scale 必须 > 0")
    if _finite(state.get("growth_factor"), "growth_factor") <= 1:
        raise ValueError("growth_factor 必须 > 1")
    if not 0 < _finite(state.get("backoff_factor"), "backoff_factor") < 1:
        raise ValueError("backoff_factor 必须在 (0, 1)")
    interval = _integer(state.get("growth_interval"), "growth_interval", 1)
    tracker = _integer(state.get("_growth_tracker"), "_growth_tracker")
    if tracker >= interval:
        raise ValueError("_growth_tracker 必须小于 growth_interval")


def validate_optimizer_state(state, model, updates, optimizer_name):
    if not isinstance(state, dict) or not isinstance(state.get("state"), dict):
        raise ValueError("optimizer_state_dict/state 缺失或非法")
    groups = state.get("param_groups")
    if not isinstance(groups, list) or not groups:
        raise ValueError("optimizer param_groups 缺失")
    parameters = list(model.parameters())
    ids = [pid for group in groups for pid in group.get("params", [])]
    if len(ids) != len(parameters) or len(set(ids)) != len(ids):
        raise ValueError("optimizer 每组参数数量/ID 与模型不匹配")
    if any(type(pid) is not int or pid < 0 for pid in ids):
        raise ValueError("optimizer 参数 ID 非法")
    mapping = dict(zip(ids, parameters, strict=True))
    states = state["state"]
    if set(states) - set(ids):
        raise ValueError("optimizer state 含未知参数 ID")
    if updates > 0 and not states:
        raise ValueError("已有有效更新却缺少 optimizer 历史状态")
    for pid, values in states.items():
        if not isinstance(values, dict):
            raise ValueError(f"optimizer state[{pid}] 非法")
        required = ("step", "exp_avg", "exp_avg_sq") if optimizer_name in (
            "adam", "adamw"
        ) else ("momentum_buffer",)
        if any(key not in values for key in required):
            raise ValueError(f"optimizer state[{pid}] 缺少必需状态")
        for key, value in values.items():
            if key == "step":
                if not isinstance(value, torch.Tensor) or value.numel() != 1:
                    raise ValueError("optimizer step 必须为标量 Tensor")
                step = _finite(value.item(), "optimizer step", 0)
                if step != int(step) or step > updates or (updates > 0 and step == 0):
                    raise ValueError("optimizer step 与实际更新次数不一致")
            elif (
                not isinstance(value, torch.Tensor)
                or value.shape != mapping[pid].shape
                or not bool(torch.isfinite(value).all())
            ):
                raise ValueError(f"optimizer state[{pid}].{key} 形状/类型/数值非法")
            elif key in ("exp_avg_sq", "max_exp_avg_sq") and bool((value < 0).any()):
                raise ValueError("Adam 二阶矩必须非负")
    for group in groups:
        _finite(group.get("lr"), "optimizer lr", 0)
        _finite(group.get("weight_decay"), "optimizer weight_decay", 0)


def validate_checkpoint_payload(checkpoint, model=None):
    """Validate complete state and cross-check epoch/history/best/early-stop progress."""
    try:
        if not isinstance(checkpoint, dict) or checkpoint.get("format_version") != FORMAT_VERSION:
            raise ValueError("旧格式缺少实际更新记录；不支持精确恢复/正式准入")
        if type(checkpoint.get("partial")) is not bool:
            raise ValueError("partial 必须为 bool")
        epoch = _integer(checkpoint.get("epoch"), "epoch")
        updates = _integer(checkpoint.get("optimizer_updates"), "optimizer_updates")
        attempts = _integer(checkpoint.get("optimizer_attempts"), "optimizer_attempts")
        if updates > attempts:
            raise ValueError("实际更新次数超过尝试次数")
        history = checkpoint.get("history")
        if not isinstance(history, dict):
            raise ValueError("缺少 history")
        for key in HISTORY_KEYS:
            values = history.get(key)
            if not isinstance(values, list) or len(values) != epoch:
                raise ValueError(f"history.{key} 长度必须与 epoch 一致")
            for value in values:
                _finite(value, f"history.{key}")
        for key in ("train_acc", "val_acc", "val_top5_acc"):
            if any(not 0 <= v <= 1 for v in history[key]):
                raise ValueError(f"history.{key} 超出 [0, 1]")
        for key, total in (("optimizer_updates", updates), ("optimizer_attempts", attempts)):
            values = history[key]
            if any(type(v) is not int or v < 0 for v in values):
                raise ValueError(f"history.{key} 非法")
            if values != sorted(values) or (values and values[-1] > total):
                raise ValueError(f"history.{key} 非单调/超过总计")
            if not checkpoint["partial"] and total != (values[-1] if values else 0):
                raise ValueError(f"完整 epoch 的 {key} 与 history 不匹配")
        _finite(checkpoint.get("training_duration_seconds"), "training_duration_seconds", 0)
        protocol = checkpoint.get("training_protocol")
        if not isinstance(protocol, dict):
            raise ValueError("缺少 training_protocol")
        runtime = protocol["runtime"]
        config = protocol["config"]
        best = checkpoint.get("best")
        es = checkpoint.get("early_stop_state")
        if not isinstance(best, dict) or not isinstance(es, dict):
            raise ValueError("缺少 best / early_stop_state")
        best_epoch = _integer(best.get("epoch"), "best.epoch")
        if best_epoch > epoch:
            raise ValueError("best.epoch 超过已完成轮数")
        metric = runtime["monitor_metric"]
        if best.get("monitor_metric") != metric:
            raise ValueError("best.monitor_metric 与协议不符")
        expected_acc = max([0.0, *history["val_acc"]])
        if _finite(best.get("val_acc"), "best.val_acc") != expected_acc:
            raise ValueError("best.val_acc 与 history 不符")
        if epoch:
            vals = history[metric]
            expected = max(vals) if metric == "val_acc" else min(vals)
            if best.get("monitor_value") != expected or best_epoch != vals.index(expected) + 1:
                raise ValueError("best 指标/轮数与 history 不符")
        elif best.get("monitor_value") is not None or best_epoch != 0:
            raise ValueError("零轮数不能有 best 进度")
        acc_counter = _integer(es.get("acc_patience_counter"), "acc_patience_counter")
        loss_counter = _integer(es.get("loss_worse_counter"), "loss_worse_counter")
        acc_best, expected_counter = 0.0, 0
        for value in history["val_acc"]:
            expected_counter = 0 if value > acc_best else expected_counter + 1
            acc_best = max(acc_best, value)
        if acc_counter != (expected_counter if runtime["patience"] > 0 else 0):
            raise ValueError("acc 早停计数与 history 不符")
        loss_best, expected_loss_counter = None, 0
        if runtime["val_loss_patience"] > 0:
            for value in history["val_loss"]:
                loss_best = value if loss_best is None else min(loss_best, value)
                expected_loss_counter = (
                    expected_loss_counter + 1
                    if loss_best > 0 and value > loss_best * runtime["val_loss_threshold"] else 0
                )
        if loss_counter != expected_loss_counter or es.get("hist_min_val_loss") != loss_best:
            raise ValueError("loss 早停进度与 history 不符")
        count = _integer(checkpoint.get("rng_cuda_device_count"), "rng_cuda_device_count")
        rng = checkpoint.get("rng")
        if not isinstance(rng, dict) or any(rng.get(k) is None for k in (
            "python", "numpy", "torch"
        )):
            raise ValueError("缺少全局 RNG 状态")
        if count:
            states = rng.get("torch_cuda")
            if not isinstance(states, list) or len(states) != count:
                raise ValueError("CUDA RNG 数量/设备映射不完整")
            if any(not isinstance(s, torch.Tensor) or s.dtype != torch.uint8 or s.ndim != 1
                   for s in states):
                raise ValueError("CUDA RNG Tensor 非法")
        if str(checkpoint.get("training_device", "")).startswith("cuda") and count == 0:
            raise ValueError("CUDA 来源缺少 RNG 设备映射")
        model_state = checkpoint.get("model_state_dict")
        if not isinstance(model_state, dict) or not model_state:
            raise ValueError("model_state_dict 缺失")
        if any(not isinstance(v, torch.Tensor) or not bool(torch.isfinite(v).all())
               for v in model_state.values()):
            raise ValueError("模型状态类型/数值非法")
        if runtime["amp_effective"]:
            try:
                validate_scaler_state(checkpoint.get("scaler_state_dict"))
            except ValueError as e:
                raise ValueError(f"scaler_state_dict 非法: {e}") from e
        if runtime["scheduler_effective"]["name"] != "none":
            scheduler = checkpoint.get("scheduler_state_dict")
            if not isinstance(scheduler, dict):
                raise ValueError("scheduler_state_dict 缺失")
            if _integer(scheduler.get("last_epoch"), "scheduler.last_epoch", -1) != epoch:
                raise ValueError("scheduler.last_epoch 与 epoch 不符")
            for key in ("T_cur", "T_i", "_step_count", "num_bad_epochs", "cooldown_counter"):
                if key in scheduler:
                    _integer(scheduler[key], f"scheduler.{key}")
            for key in ("_last_lr", "base_lrs"):
                if key in scheduler:
                    if not isinstance(scheduler[key], list):
                        raise ValueError(f"scheduler.{key} 非法")
                    for v in scheduler[key]:
                        _finite(v, f"scheduler.{key}", 0)
            effective = runtime["scheduler_effective"]
            if effective["name"] == "CosineAnnealingWarmRestarts":
                length = _integer(effective["T_0"], "scheduler.T_0", 1)
                multiplier = _integer(effective["T_mult"], "scheduler.T_mult", 1)
                position = epoch
                while position >= length:
                    position -= length
                    length *= multiplier
                if scheduler.get("T_i") != length or scheduler.get("T_cur") != position:
                    raise ValueError("scheduler 重启周期进度与 epoch 不一致")
        if model is not None:
            expected_state = model.state_dict()
            if set(expected_state) != set(model_state):
                raise ValueError("模型状态键不匹配")
            for key, value in model_state.items():
                if (value.shape != expected_state[key].shape
                        or value.dtype != expected_state[key].dtype):
                    raise ValueError(f"模型状态 {key} 形状/dtype 不符")
            validate_optimizer_state(
                checkpoint.get("optimizer_state_dict"), model, updates,
                config["training"].get("optimizer", "adam").lower(),
            )
    except (KeyError, TypeError, ValueError, AttributeError) as e:
        raise RuntimeError(f"断点状态完整性预检未通过：{e}") from e


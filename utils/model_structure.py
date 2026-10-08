"""Explicit MicroResNet structure validation shared by config, specs and models."""

DEFAULT_BLOCKS = (2, 2)
DEFAULT_CHANNELS = (64, 128)
DEFAULT_POOL_ORDER = "before_stage1"
STRUCTURE_FIELDS = frozenset({"blocks", "channels", "pool_order"})


def validate_micro_structure(blocks, channels, pool_order):
    """Return canonical two-stage structure; never coerce booleans or partial lists."""
    for name, values in (("blocks", blocks), ("channels", channels)):
        if (
            not isinstance(values, (list, tuple))
            or len(values) != 2
            or any(type(v) is not int or v < 1 for v in values)
        ):
            raise ValueError(f"{name} 必须为两个正整数的列表")
    if pool_order not in ("before_stage1", "after_stage1"):
        raise ValueError("pool_order 必须为 before_stage1 或 after_stage1")
    return tuple(blocks), tuple(channels), pool_order

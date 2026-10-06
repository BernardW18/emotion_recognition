"""控制台/日志编码工具：Windows 默认 GBK 下将输出统一为 UTF-8（R09）。

背景：Windows 默认 GBK 代码页下，将工具输出重定向到日志文件时，特殊字符
（如 `↔`）会触发 UnicodeEncodeError，并导致部分工具在写出结果文件前中断。

用法：在 CLI / 工具入口的 main() 开头调用 ensure_utf8_stdio()，
使 stdout/stderr 以 UTF-8 编码写入（包含重定向到文件的情形）。
"""

__all__ = ["ensure_utf8_stdio"]

import contextlib
import sys


def ensure_utf8_stdio() -> None:
    """将 stdout/stderr 重新配置为 UTF-8（不可重配置时静默跳过）。"""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        # 非常规流（如被包装）无法重配置时跳过
        with contextlib.suppress(Exception):
            reconfigure(encoding="utf-8", errors="replace")

"""
FER2013 人脸情感识别 - Streamlit 应用启动脚本
用法:
    python run_app.py
等价于:
    streamlit run inference/app.py
"""

import subprocess
import sys
from pathlib import Path

APP_FILE = Path(__file__).parent / "inference" / "app.py"

if __name__ == "__main__":
    # 通过子进程启动 Streamlit（仅此处创建子进程，无裸模式污染）
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", str(APP_FILE)],
        cwd=Path(__file__).parent,
    )

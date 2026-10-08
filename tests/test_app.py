"""
应用启动与文案冒烟测试 — F15 / F16 / F17

用 Streamlit 官方 AppTest 框架无头运行 inference/app.py，
验证：脚本无异常、无上传时的引导文案、未校准概率表述、权重来源展示。
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest
from streamlit.testing.v1 import AppTest

APP_PATH = PROJECT_ROOT / "inference" / "app.py"


def _run_app():
    at = AppTest.from_file(str(APP_PATH), default_timeout=60)
    at.run()
    return at


def test_app_starts_without_exception():
    at = _run_app()
    assert not at.exception, f"应用启动异常: {[str(e) for e in at.exception]}"


def test_app_mentions_uncalibrated_and_scope():
    """F15/F16：明确未校准概率与输入范围（已裁剪单张人脸）。"""
    at = _run_app()
    all_text = "\n".join(m.value for m in at.markdown) + "\n".join(
        c.value for c in at.caption
    )
    assert "未校准" in all_text, "缺少「未校准」表述"
    assert "已裁剪" in all_text, "缺少输入范围（已裁剪单张人脸）说明"
    assert "不是对真实心理状态的测量" in all_text, "缺少范围限定表述"


def test_app_sidebar_lists_checkpoint_with_source():
    """F01/F05：侧栏权重列表应含来源信息（run 或 legacy 标注）。"""
    at = _run_app()
    selectboxes = at.selectbox
    if not selectboxes:
        pytest.skip("saved_models 为空，跳过权重列表检查")
    options = selectboxes[0].options
    assert any("legacy" in opt or "run=" in opt for opt in options), (
        f"权重列表缺少来源标注: {options}"
    )


def test_app_selects_manifest_default_instead_of_first_filename():
    from inference.infer_utils import list_available_checkpoints

    items = list_available_checkpoints()
    if not items:
        pytest.skip("saved_models 为空")
    expected = next((item for item in items if item["is_default"]), items[0])
    at = _run_app()
    assert not at.exception
    assert at.selectbox[0].value == expected["label"]


def test_app_gradcam_toggle_mentions_auxiliary():
    """F17：Grad-CAM 的表述为辅助可视化，不宣称因果解释。"""
    at = _run_app()
    toggles = at.toggle
    if not toggles:
        pytest.skip("无 toggle（可能无可用权重）")
    help_text = toggles[0].help or ""
    assert "辅助" in help_text or "不构成" in help_text


def test_app_focuses_recommended_model_and_can_reveal_history():
    from inference.infer_utils import list_available_checkpoints

    items = list_available_checkpoints()
    recommended = [item for item in items if item["is_default"]]
    if not recommended or len(items) == len(recommended):
        pytest.skip("本地没有推荐权重与历史权重的组合")
    at = _run_app()
    assert not at.exception
    assert at.selectbox[0].options == [item["label"] for item in recommended]
    at.checkbox[0].check().run()
    assert not at.exception
    assert at.selectbox[0].options == [item["label"] for item in items]
    assert at.selectbox[0].value == recommended[0]["label"]


def test_app_without_local_weights_still_explains_scope(tmp_path, monkeypatch):
    from inference import infer_utils

    monkeypatch.setattr(infer_utils, "SAVED_MODELS_DIR", tmp_path)
    monkeypatch.setattr(infer_utils, "MANIFEST_PATH", tmp_path / "export_manifest.json")
    at = _run_app()
    assert not at.exception and not at.selectbox
    text = "\n".join(m.value for m in at.markdown) + "\n".join(c.value for c in at.caption)
    assert "已裁剪" in text and "未校准" in text and "不是对真实心理状态的测量" in text
    assert any("暂无可用权重" in info.value for info in at.info)

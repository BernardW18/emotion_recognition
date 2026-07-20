"""
FER2013 人脸情感识别 - Streamlit 推理应用
仅负责推理演示，不包含训练功能

启动方式（任选其一）：
    VSCode:  按 F5 或点击侧栏「运行和调试」→「Streamlit: FER2013 推理应用」
    终端:    streamlit run inference/app.py
    快捷:    python run_app.py
"""

import sys
from pathlib import Path

# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import numpy as np
from PIL import Image

from inference.infer_utils import (
    load_model,
    predict,
    generate_gradcam,
    get_available_models,
    get_model_info,
    CLASS_NAMES,
    CLASS_EMOJIS,
)


@st.cache_resource
def load_model_cached(model_name: str, checkpoint_path: str = None, device: str = None):
    """带缓存的模型加载，避免每次推理重新加载权重"""
    return load_model(model_name, checkpoint_path, device)

# ==============================
# 页面配置
# ==============================
st.set_page_config(
    page_title="人脸情感识别",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==============================
# 侧边栏：模型选择与设置
# ==============================
with st.sidebar:
    st.header("⚙️ 模型设置")

    # 模型选择
    available_models = get_available_models()

    if not available_models:
        st.warning(
            "未找到训练好的模型文件。\n\n"
            "请先通过 CLI 脚本或 Jupyter Notebook 训练模型：\n"
            "```bash\n"
            "python training/train.py --model mini_cnn --epochs 30\n"
            "```\n"
            "生成 `.pth` 文件到 `inference/saved_models/` 目录。"
        )
        st.info("可用的模型名称：`mini_cnn`, `vgg_lite`, `micro_resnet`")
        selected_model = st.selectbox(
            "选择模型",
            ["mini_cnn", "vgg_lite", "micro_resnet"],
            help="即使checkpoint不存在也可以选择，上传图片时会提示",
        )
    else:
        selected_model = st.selectbox(
            "选择模型",
            available_models,
            help="仅显示已存在checkpoint的模型",
        )

    # Grad-CAM 开关
    st.divider()
    st.subheader("🔬 可视化")
    show_gradcam = st.toggle("Grad-CAM 热力图", value=True,
                              help="展示模型关注的图像区域（解释性分析）")

    # 模型信息
    st.divider()
    st.subheader("📋 模型信息")

    model_info = {
        "mini_cnn": {
            "desc": "基线模型",
            "params": "~50K",
            "architecture": "3层卷积 + FC",
        },
        "vgg_lite": {
            "desc": "VGG变体",
            "params": "~1.5M",
            "architecture": "5层卷积(小核堆叠) + FC",
        },
        "micro_resnet": {
            "desc": "微型残差网络",
            "params": "~400K",
            "architecture": "4个残差块 + 全局平均池化",
        },
    }

    if selected_model in model_info:
        info = model_info[selected_model]
        st.markdown(f"**类型**: {info['desc']}")
        st.markdown(f"**参数量**: {info['params']}")
        st.markdown(f"**架构**: {info['architecture']}")

        # 从 checkpoint 读取训练精度信息
        ckpt_info = get_model_info(selected_model)
        if ckpt_info:
            if ckpt_info["global_best_val_acc"] > 0:
                st.markdown(f"**验证准确率**: {ckpt_info['global_best_val_acc']:.2%}")
            if ckpt_info["trained_epochs"] > 0:
                st.markdown(f"**训练轮数**: {ckpt_info['trained_epochs']}")
            if ckpt_info["training_duration"]:
                st.markdown(f"**训练耗时**: {ckpt_info['training_duration']}")
        else:
            st.caption("(未找到已训练的模型文件)")

    st.divider()
    st.subheader("📊 情感类别")
    for name in CLASS_NAMES:
        st.markdown(f"- {CLASS_EMOJIS[name]} {name}")

# ==============================
# 主区域
# ==============================
st.title("🎭 人脸情感识别系统")
st.markdown("上传一张人脸图像，系统将识别其中的情感状态。")

# 图像上传
col1, col2 = st.columns(2)

with col1:
    st.subheader("📤 上传图像")
    uploaded_file = st.file_uploader(
        "选择一张人脸图像",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        help="支持 JPG/PNG/BMP/WebP 格式",
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        # 控制显示宽度
        img_width = min(image.width, 400)
        st.image(image, caption="上传的图像", width=img_width)

with col2:
    st.subheader("🔍 识别结果")

    # 初始化为 None，防止推理失败时 Grad-CAM 区域 NameError
    model = None
    device = None
    result = None

    if uploaded_file is not None:
        # 加载模型并推理
        try:
            with st.spinner("正在加载模型并识别..."):
                model, device = load_model_cached(selected_model)
                result = predict(model, image, device)

            # 显示主要结果
            st.markdown("---")
            st.markdown(
                f"<div style='text-align: center; font-size: 48px;'>"
                f"{result['emoji']}</div>",
                unsafe_allow_html=True,
            )
            st.markdown(
                f"<div style='text-align: center; font-size: 24px; font-weight: bold;'>"
                f"{result['emotion']}</div>",
                unsafe_allow_html=True,
            )
            st.markdown(
                f"<div style='text-align: center; font-size: 16px; color: #888;'>"
                f"置信度: {result['confidence']:.1%}</div>",
                unsafe_allow_html=True,
            )
            st.markdown("---")

            # 各类别概率条形图
            st.subheader("📈 各类别概率")
            probs = result["probabilities"]
            # 按概率降序排列
            sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)

            for name, prob in sorted_probs:
                emoji = CLASS_EMOJIS[name]
                is_top = name == result["emotion"]
                bar_color = "🔴" if is_top else "⬜"
                st.markdown(
                    f"{emoji} **{name}**: {prob:.1%} "
                    f"{bar_color} {'▓' * int(prob * 30)}{'░' * (30 - int(prob * 30))}"
                )

        except FileNotFoundError as e:
            st.error(f"❌ {str(e)}")
        except Exception as e:
            st.error(f"❌ 推理出错: {str(e)}")
    else:
        st.info("👈 请先上传一张人脸图像")

# ==============================
# Grad-CAM 热力图区域
# ==============================
if uploaded_file is not None and show_gradcam and model is not None and result is not None:
    st.divider()
    st.subheader("🔬 Grad-CAM 热力图分析")

    try:
        with st.spinner("正在生成热力图..."):
            heatmap = generate_gradcam(model, image, device)

        # 使用 matplotlib 渲染叠加图
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))

        # 原图（灰度48x48）
        gray_img = image.convert("L").resize((48, 48), Image.Resampling.BILINEAR)
        axes[0].imshow(gray_img, cmap="gray")
        axes[0].set_title("输入图像")
        axes[0].axis("off")

        # 热力图
        axes[1].imshow(heatmap, cmap="jet", vmin=0, vmax=1)
        axes[1].set_title("Grad-CAM 热力图")
        axes[1].axis("off")

        # 叠加图
        axes[2].imshow(gray_img, cmap="gray")
        axes[2].imshow(heatmap, cmap="jet", alpha=0.5, vmin=0, vmax=1)
        axes[2].set_title(f"叠加: {result['emotion']}")
        axes[2].axis("off")

        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        st.caption("红色区域表示模型做出该预测时关注的高激活区域")

    except Exception as e:
        st.warning(f"热力图生成失败: {str(e)}")

# ==============================
# 底部信息
# ==============================
st.divider()
st.markdown(
    "<div style='text-align: center; color: #888; font-size: 12px;'>"
    "FER2013 人脸情感识别 | PyTorch + Streamlit | "
    "通过 python training/train.py 或 Jupyter Notebook 训练模型，本应用仅做推理演示"
    "</div>",
    unsafe_allow_html=True,
)

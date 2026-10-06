"""
FER2013 人脸表情识别 - Streamlit 推理应用（推理演示，不含训练）

范围说明（F15）: 演示仅支持「已裁剪的单张人脸表情类别预测」。
建议上传接近 FER2013 训练域的人脸照片（正面、居中的单张人脸；模型输入为
48×48 灰度、x/255）。整张生活照 / 多人合影 / 未裁剪图像不属于支持范围，
本应用不做人脸检测与对齐。

概率说明（F16）: 界面显示的概率为模型 softmax 输出，未经校准，
不等同于「预测正确的概率」；校准指标（ECE/NLL）见 analysis/ 下的评估结果。

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
from PIL import Image

from inference.infer_utils import (
    CLASS_EMOJIS,
    CLASS_NAMES,
    describe_checkpoint,
    generate_gradcam,
    list_available_checkpoints,
    load_model,
    predict,
)


@st.cache_resource
def load_model_cached(checkpoint_path: str, mtime: float, device: str | None = None):
    """带缓存的模型加载（mtime 参与缓存键，文件更新后自动重建）"""
    return load_model(checkpoint_path, device)


@st.cache_data
def describe_checkpoint_cached(checkpoint_path: str, mtime: float):
    """带缓存的权重信息读取（mtime 参与缓存键，文件更新后自动重建）"""
    return describe_checkpoint(checkpoint_path)


# ==============================
# 页面配置
# ==============================
st.set_page_config(
    page_title="人脸表情识别（FER2013）",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==============================
# 侧边栏：权重选择与信息
# ==============================
selected_item = None
show_gradcam = True

with st.sidebar:
    st.header("⚙️ 模型设置")

    checkpoints = list_available_checkpoints()

    if not checkpoints:
        st.warning(
            "未找到可用权重。\n\n"
            "训练后请显式导出到推理目录：\n"
            "```bash\n"
            "python training/train.py --model mini_cnn --epochs 30\n"
            "python tools/export_model.py --checkpoint "
            "training/runs/<模型>/<run_id>/checkpoints/best.pth\n"
            "```"
        )
    else:
        labels = [it["label"] for it in checkpoints]
        choice = st.selectbox("选择权重（含来源信息）", labels, index=0)
        selected_item = next(it for it in checkpoints if it["label"] == choice)

        if selected_item["legacy"]:
            st.warning(
                "该权重为 legacy 旧产物（旧格式断点，来源/配置信息不全），"
                "按确定性迁移规则解析，迁移依据见下方「迁移/配置说明」。"
            )

        st.divider()
        st.subheader("📋 权重信息")
        mtime = Path(selected_item["path"]).stat().st_mtime
        info = describe_checkpoint_cached(selected_item["path"], mtime)
        spec = info["model_spec"]
        st.markdown(f"**规格**: {spec['model_name']} | 类别数 {len(spec['class_names'])}")
        st.markdown(
            f"**激活/正则**: activation={spec['activation']} | "
            f"dropout={spec['dropout']} | use_se={spec['use_se']}"
        )
        st.markdown(f"**参数量**: {info['params_total']:,}")
        st.markdown(f"**SHA-256**: `{info['sha256'][:16]}...`")
        if info["metrics"].get("val_acc"):
            m = info["metrics"]
            st.markdown(f"**记录指标**: epoch={m.get('epoch')} val_acc={m.get('val_acc'):.4f}")
        source = selected_item.get("source")
        if source:
            st.markdown(f"**来源**: run `{source.get('run_id')}`")
            st.caption(f"源文件: {source.get('source_checkpoint')}")
        with st.expander("迁移/配置说明", expanded=False):
            for note in info["spec_notes"]:
                st.markdown(f"- {note}")

        st.divider()
        st.subheader("🔬 可视化")
        show_gradcam = st.toggle(
            "Grad-CAM 热力图", value=True,
            help="目标类别的梯度响应辅助图；辅助可视化，不构成模型因果机制或心理解释的证据",
        )

    st.divider()
    st.subheader("📊 情感类别")
    for name in CLASS_NAMES:
        st.markdown(f"- {CLASS_EMOJIS[name]} {name}")

# ==============================
# 主区域
# ==============================
st.title("🎭 人脸表情识别（FER2013 类别预测）")

if selected_item is None:
    st.info(
        "推理目录暂无可用权重。请先训练模型并用 `tools/export_model.py` 显式导出，"
        "然后刷新本页面。"
    )
    st.stop()

st.markdown("上传一张**已裁剪的单张人脸**图像，模型将预测其 FER2013 七类表情类别。")
st.caption(
    "输入要求：接近训练域的正面人脸（居中的单张人脸；越接近 48×48 灰度的证件照构图越可靠）。"
    "整张生活照、多人合影或未裁剪图像不适合本演示；本应用不做人脸检测与对齐。"
    "输出为数据集表情类别预测与未校准概率，不是对真实心理状态的测量。"
)

# 图像上传
col1, col2 = st.columns(2)

model = None
device = None
result = None
image = None
uploaded_file = None

with col1:
    st.subheader("📤 上传图像")
    uploaded_file = st.file_uploader(
        "选择一张已裁剪的人脸图像",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        help="支持 JPG/PNG/BMP/WebP；建议使用已裁剪的正面人脸",
    )

    if uploaded_file is not None:
        try:
            uploaded_file.seek(0)
            image = Image.open(uploaded_file)
            image.load()  # 触发完整解码，捕获损坏文件
        except Exception as e:
            image = None
            st.error(f"❌ 无法解码该文件（可能已损坏或不是有效图像）: {e}")
        else:
            # 控制显示宽度
            img_width = min(image.width, 400)
            st.image(image, caption="上传的图像", width=img_width)

with col2:
    st.subheader("🔍 识别结果")

    if uploaded_file is not None and image is not None:
        # 加载模型并推理
        try:
            with st.spinner("正在加载模型并识别..."):
                mtime = Path(selected_item["path"]).stat().st_mtime
                model, device, _meta = load_model_cached(selected_item["path"], mtime)
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
                f"模型概率（未校准）: {result['probability']:.1%}</div>",
                unsafe_allow_html=True,
            )
            st.caption("概率为 softmax 输出，未经校准，不等同于「预测正确的概率」。")
            st.markdown("---")

            # 各类别概率条形图
            st.subheader("📈 各类别概率（未校准）")
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

        except Exception as e:
            st.error(f"❌ 推理出错: {str(e)}")
    else:
        st.info("👈 请先上传一张已裁剪的人脸图像")

# ==============================
# Grad-CAM 热力图区域
# ==============================
if (
    uploaded_file is not None
    and image is not None
    and show_gradcam
    and model is not None
    and result is not None
):
    st.divider()
    st.subheader("🔬 Grad-CAM 热力图（辅助可视化）")

    try:
        with st.spinner("正在生成热力图..."):
            heatmap = generate_gradcam(model, image, device)

        # 使用 matplotlib 渲染叠加图
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))

        # 模型输入（48×48 灰度）
        gray_img = image.convert("L")
        if gray_img.size != (48, 48):
            gray_img = gray_img.resize((48, 48), Image.Resampling.BILINEAR)
        axes[0].imshow(gray_img, cmap="gray")
        axes[0].set_title("模型输入（48×48 灰度）")
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

        st.caption(
            "Grad-CAM 为目标类别的梯度响应辅助图，用于查看模型的高响应区域；"
            "它不构成模型因果机制或心理解释的证据。"
        )

    except Exception as e:
        st.warning(f"热力图生成失败: {str(e)}")

# ==============================
# 底部信息
# ==============================
st.divider()
st.markdown(
    "<div style='text-align: center; color: #888; font-size: 12px;'>"
    "FER2013 人脸表情识别（类别预测演示） | PyTorch + Streamlit | "
    "推理与训练共用同一模型构造入口（utils/model_spec），权重来源见侧栏"
    "</div>",
    unsafe_allow_html=True,
)

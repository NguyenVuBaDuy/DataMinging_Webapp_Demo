"""
ui_sidebar.py - Sidebar: chọn model, nút chẩn đoán, trạng thái hệ thống.
"""

import streamlit as st


def render_sidebar(resources):
    """Render sidebar và trả về (model_choice, threshold, diagnose_clicked).

    Args:
        resources: dict từ model_loader.load_all_resources()

    Returns:
        tuple: (model_choice: str, threshold: float, diagnose_clicked: bool)
    """
    feature_order = resources["feature_order"]
    shap_background = resources["shap_background"]
    lightgbm_threshold = resources["lightgbm_threshold"]
    tabnet_threshold = resources["tabnet_threshold"]

    with st.sidebar:
        st.markdown("## 🏥 Bảng Điều Khiển")
        st.markdown("---")

        # -- Chọn model --
        st.markdown("### 🤖 Chọn Mô Hình AI")
        model_choice = st.selectbox(
            "Mô hình dự đoán",
            options=["LightGBM", "TabNet"],
            index=0,
            help="LightGBM: Nhanh, chính xác cao. TabNet: Deep Learning, có Attention Map.",
        )

        # -- Hiển thị thông tin model đã chọn --
        if model_choice == "LightGBM":
            threshold = lightgbm_threshold
            st.markdown(f"""
            <div class="model-info-card">
                <div class="label">Ngưỡng phân loại</div>
                <div class="value">{threshold}</div>
            </div>
            """, unsafe_allow_html=True)
            st.caption("🌳 Gradient Boosting Decision Tree")
            st.caption("📊 XAI: SHAP Waterfall Plot")
        else:
            threshold = tabnet_threshold
            st.markdown(f"""
            <div class="model-info-card">
                <div class="label">Ngưỡng phân loại</div>
                <div class="value">{threshold}</div>
            </div>
            """, unsafe_allow_html=True)
            st.caption("🧠 Deep Learning with Attention")
            st.caption("🔥 XAI: Attention Heatmap")

        st.markdown("---")

        # -- Nút chẩn đoán --
        diagnose_clicked = st.button(
            "🔬 Tiến Hành Chẩn Đoán",
            use_container_width=True,
            type="primary",
        )

        st.markdown("---")

        # -- Trạng thái hệ thống --
        st.markdown("### ⚙️ Trạng Thái Hệ Thống")
        st.markdown(
            '<span class="status-dot"></span> Models đã sẵn sàng',
            unsafe_allow_html=True,
        )
        st.caption(f"Features: {len(feature_order)} chỉ số")
        st.caption(f"SHAP data: {shap_background.shape[0]} mẫu")

    return model_choice, threshold, diagnose_clicked

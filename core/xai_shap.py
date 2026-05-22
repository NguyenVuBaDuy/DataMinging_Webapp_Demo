"""
xai_shap.py - SHAP Waterfall Plot cho LightGBM.

Bước 6 trong kế hoạch Phase 4:
    - Khởi tạo shap.TreeExplainer(lgbm_model, background_data)
    - Tính SHAP values cho input hiện tại
    - Vẽ shap.waterfall_plot() bằng st.pyplot()
    - Hiển thị label tiếng Việt thân thiện
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import shap
import streamlit as st

from core.config import FEATURE_LABELS_VI

# Dùng backend non-interactive cho matplotlib
matplotlib.use("Agg")


def compute_shap_values(model, background_data, input_df):
    """Tính SHAP values cho 1 sample dùng TreeExplainer.

    Args:
        model: LightGBM model.
        background_data: pd.DataFrame, 100 mẫu nền.
        input_df: pd.DataFrame, 1 dòng input.

    Returns:
        shap.Explanation object
    """
    explainer = shap.TreeExplainer(model, background_data)
    shap_values = explainer(input_df)

    return shap_values


def render_shap_waterfall(shap_values, input_df):
    """Render SHAP Waterfall Plot với nhãn tiếng Việt.

    Args:
        shap_values: shap.Explanation từ compute_shap_values.
        input_df: pd.DataFrame dùng để hiển thị giá trị feature.
    """
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #1e3a5f 0%, #2d6a9f 100%);
        border-radius: 12px;
        padding: 1rem 1.5rem;
        margin-bottom: 1rem;
        color: white;
    ">
        <h4 style="margin:0; font-weight:700;">📊 SHAP Waterfall Plot — Giải Thích Dự Đoán</h4>
        <p style="margin:0.3rem 0 0 0; opacity:0.85; font-size:0.9rem;">
            Biểu đồ cho thấy mức đóng góp của từng chỉ số vào kết quả dự đoán.
            Thanh <span style="color:#ef4444;font-weight:600;">đỏ</span> đẩy xác suất tăng (nguy cơ cao hơn),
            thanh <span style="color:#3b82f6;font-weight:600;">xanh</span> kéo xác suất giảm (nguy cơ thấp hơn).
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Lấy SHAP values cho class 1 (bệnh tim)
    # TreeExplainer cho binary classification có thể trả về
    # shape (1, n_features) hoặc (1, n_features, 2)
    sv = shap_values[0]

    # Nếu shap_values có 2 output classes, lấy class 1
    if len(sv.values.shape) > 1 and sv.values.shape[-1] == 2:
        sv = shap.Explanation(
            values=sv.values[:, 1],
            base_values=sv.base_values[1] if hasattr(sv.base_values, '__len__') else sv.base_values,
            data=sv.data,
            feature_names=sv.feature_names,
        )

    # Map feature names sang tiếng Việt
    feature_names_vi = [
        FEATURE_LABELS_VI.get(name, name) for name in sv.feature_names
    ]
    sv.feature_names = feature_names_vi

    # Vẽ waterfall plot
    fig, ax = plt.subplots(figsize=(10, 7))
    plt.sca(ax)

    shap.waterfall_plot(sv, max_display=12, show=False)

    plt.title("Đóng góp của từng chỉ số lâm sàng", fontsize=14, fontweight="bold", pad=15)
    plt.tight_layout()

    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    # Bảng chi tiết SHAP values
    with st.expander("📋 Xem chi tiết giá trị SHAP", expanded=False):
        import pandas as pd

        shap_detail = pd.DataFrame({
            "Chỉ số": feature_names_vi,
            "Giá trị đầu vào": sv.data,
            "SHAP Value": np.round(sv.values, 4),
            "Ảnh hưởng": ["↑ Tăng nguy cơ" if v > 0 else "↓ Giảm nguy cơ" for v in sv.values],
        })
        shap_detail = shap_detail.sort_values("SHAP Value", key=abs, ascending=False)
        st.dataframe(shap_detail, hide_index=True, use_container_width=True)

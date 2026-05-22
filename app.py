"""
app.py - Entry point chính của ứng dụng Streamlit.
Chỉ làm nhiệm vụ orchestration: cấu hình trang, load tài nguyên,
gọi các module UI, xử lý sự kiện.
"""

import streamlit as st

# ============================================================
# CẤU HÌNH TRANG (phải là lệnh Streamlit đầu tiên)
# ============================================================
st.set_page_config(
    page_title="❤️ Hệ Thống Chẩn Đoán Bệnh Tim",
    page_icon="❤️",
    layout="wide",
)

# ============================================================
# IMPORTS - Các module từ package core/
# ============================================================
from core import (
    load_all_resources,
    inject_custom_css,
    render_sidebar,
    render_input_form,
    render_input_summary,
)

# ============================================================
# KHỞI TẠO
# ============================================================

# Inject CSS
inject_custom_css()

# Load tất cả models, config, data (cached)
resources = load_all_resources()

# ============================================================
# RENDER UI
# ============================================================

# Sidebar: chọn model + nút chẩn đoán
model_choice, threshold, diagnose_clicked = render_sidebar(resources)

# Main area: form nhập liệu 12 features
input_df, display_values = render_input_form(resources["feature_order"])

# Tóm tắt dữ liệu đã nhập
render_input_summary(input_df, display_values)

# ============================================================
# XỬ LÝ CHẨN ĐOÁN (Bước 4 & 5 sẽ xây dựng ở đây)
# ============================================================
result_container = st.container()

if diagnose_clicked:
    with result_container:
        st.markdown("---")
        st.info("⏳ Chức năng chẩn đoán sẽ được xây dựng ở **Bước 4 & 5**. Dữ liệu đã sẵn sàng!")
        st.markdown(f"**Model đã chọn:** `{model_choice}` | **Ngưỡng:** `{threshold}`")
        st.dataframe(input_df, hide_index=True, use_container_width=True)

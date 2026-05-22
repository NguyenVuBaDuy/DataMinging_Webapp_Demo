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
    run_diagnosis,
    render_diagnosis_result,
    render_xai_section,
    init_session_state_if_needed,
    render_sample_patients_tab,
)

# ============================================================
# KHỞI TẠO
# ============================================================

# Khởi tạo session state cho form
init_session_state_if_needed()

# Inject CSS
inject_custom_css()

# Load tất cả models, config, data (cached)
resources = load_all_resources()

# ============================================================
# RENDER UI
# ============================================================

# Sidebar: chọn model + nút chẩn đoán
model_choice, threshold, diagnose_clicked = render_sidebar(resources)

# Main area: Tabs phân chia tính năng
tab_input, tab_samples = st.tabs([
    "📝 Nhập Chỉ Số Bệnh Nhân",
    "👥 Bệnh Nhân Mẫu (Kho Lưu Trữ)"
])

with tab_input:
    # Form nhập liệu 12 features
    input_df, display_values = render_input_form(resources["feature_order"])

with tab_samples:
    # Kho dữ liệu bệnh nhân mẫu
    render_sample_patients_tab(resources["sample_patients"])

# Tóm tắt dữ liệu đã nhập
render_input_summary(input_df, display_values)

# ============================================================
# XỬ LÝ CHẨN ĐOÁN (Bước 4-7)
# ============================================================

if diagnose_clicked:
    st.markdown("---")

    # Bước 4: Chạy dự đoán
    with st.spinner("🔬 Đang phân tích dữ liệu lâm sàng..."):
        result = run_diagnosis(model_choice, resources, input_df)

    # Bước 5: Hiển thị kết quả chẩn đoán
    render_diagnosis_result(result)

    # Bước 6 & 7: Hiển thị giải thích XAI (SHAP / Attention)
    render_xai_section(model_choice, resources, input_df)


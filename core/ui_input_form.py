"""
ui_input_form.py - Form nhập liệu 12 chỉ số lâm sàng, chia 3 cột.
"""

import pandas as pd
import streamlit as st


def render_input_form(feature_order):
    """Render form nhập liệu và trả về (input_df, display_values).

    Args:
        feature_order: list[str] thứ tự features từ config.

    Returns:
        tuple: (input_df: pd.DataFrame, display_values: dict)
            - input_df: DataFrame 1 dòng theo đúng feature_order (giá trị số)
            - display_values: dict nhãn tiếng Việt → giá trị hiển thị
    """
    # --- Header ---
    st.markdown("""
    <div class="main-header">
        <h1>❤️ Hệ Thống Chẩn Đoán Bệnh Tim</h1>
        <p>Ứng dụng Machine Learning & Explainable AI hỗ trợ sàng lọc nguy cơ bệnh tim mạch</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 📝 Thông Tin Lâm Sàng Bệnh Nhân")
    st.caption("Vui lòng nhập đầy đủ 12 chỉ số lâm sàng bên dưới để hệ thống phân tích.")

    # ============================
    # 3 CỘT NHẬP LIỆU
    # ============================
    col1, col2, col3 = st.columns(3)

    # --- Cột 1: Chỉ số liên tục ---
    with col1:
        st.markdown(
            '<div class="section-card"><h3>📏 Chỉ Số Liên Tục</h3></div>',
            unsafe_allow_html=True,
        )

        age = st.slider(
            "🎂 Tuổi",
            min_value=20, max_value=80, step=1,
            help="Tuổi của bệnh nhân (năm)",
            key="age",
        )
        trestbps = st.slider(
            "💉 Huyết áp lúc nghỉ (mmHg)",
            min_value=90, max_value=200, step=1,
            help="Huyết áp tâm thu lúc nghỉ ngơi",
            key="trestbps",
        )
        chol = st.slider(
            "🧪 Cholesterol huyết thanh (mg/dl)",
            min_value=120, max_value=560, step=1,
            help="Cholesterol trong máu",
            key="chol",
        )
        oldpeak = st.slider(
            "📉 Độ chênh ST khi vận động",
            min_value=0.0, max_value=6.5, step=0.1,
            help="ST depression induced by exercise relative to rest",
            key="oldpeak",
        )

    # --- Cột 2: Chỉ số nhị phân ---
    with col2:
        st.markdown(
            '<div class="section-card"><h3>🔘 Chỉ Số Phân Loại (Nhị phân)</h3></div>',
            unsafe_allow_html=True,
        )

        sex_options = ["Nam", "Nữ"]
        sex = st.selectbox(
            "👤 Giới tính",
            options=sex_options,
            help="Giới tính sinh học",
            key="sex_display",
        )
        sex_val = 1 if sex == "Nam" else 0

        fbs_options = ["≤ 120 mg/dl (Bình thường)", "> 120 mg/dl (Cao)"]
        fbs = st.selectbox(
            "🍬 Đường huyết lúc đói",
            options=fbs_options,
            help="Đường huyết lúc đói có > 120 mg/dl không?",
            key="fbs_display",
        )
        fbs_val = 0 if "Bình thường" in fbs else 1

        exang_options = ["Không", "Có"]
        exang = st.selectbox(
            "🏃 Đau thắt ngực khi gắng sức",
            options=exang_options,
            help="Có đau thắt ngực khi tập thể dục không?",
            key="exang_display",
        )
        exang_val = 1 if exang == "Có" else 0

        cp_options = ["Đau thắt điển hình (cp=0)", "Loại khác (cp≠0)"]
        cp = st.selectbox(
            "💔 Loại đau ngực (cp)",
            options=cp_options,
            help="cp_0.0: 1 = Đau thắt điển hình, 0 = Các loại đau ngực khác",
            key="cp_display",
        )
        cp_val = 1 if "cp=0" in cp else 0

    # --- Cột 3: Chỉ số đa nhãn ---
    with col3:
        st.markdown(
            '<div class="section-card"><h3>📊 Chỉ Số Phân Loại (Đa nhãn)</h3></div>',
            unsafe_allow_html=True,
        )

        restecg_options = [
            "Bình thường (0)",
            "Bất thường sóng ST-T (1)",
            "Phì đại thất trái (2)",
        ]
        restecg = st.selectbox(
            "📈 Kết quả điện tâm đồ (ECG)",
            options=restecg_options,
            help="Kết quả đo điện tâm đồ lúc nghỉ ngơi",
            key="restecg_display",
        )
        restecg_val = int(restecg.split("(")[1].replace(")", ""))

        slope_options = ["Dốc lên (0)", "Phẳng (1)", "Dốc xuống (2)"]
        slope = st.selectbox(
            "📐 Độ dốc đoạn ST",
            options=slope_options,
            help="Độ dốc của đoạn ST đỉnh khi vận động",
            key="slope_display",
        )
        slope_val = int(slope.split("(")[1].replace(")", ""))

        ca_options = ["0", "1", "2", "3", "4"]
        ca = st.selectbox(
            "🫀 Số mạch máu chính (Fluoroscopy)",
            options=ca_options,
            help="Số lượng mạch máu chính được nhuộm bằng Fluoroscopy (0-4)",
            key="ca_display",
        )
        ca_val = int(ca)

        thal_options = ["Bình thường (thal=0)", "Bất thường (thal≠0)"]
        thal = st.selectbox(
            "🩸 Thalassemia (thal)",
            options=thal_options,
            help="thal_0.0: 1 = Bình thường, 0 = Bất thường (Khiếm khuyết)",
            key="thal_display",
        )
        thal_val = 1 if "thal=0" in thal else 0

    # ============================
    # GOM DỮ LIỆU
    # ============================
    input_dict = {
        "age": age,
        "trestbps": trestbps,
        "chol": chol,
        "oldpeak": oldpeak,
        "sex": sex_val,
        "fbs": fbs_val,
        "restecg": restecg_val,
        "exang": exang_val,
        "slope": slope_val,
        "ca": ca_val,
        "cp_0.0": cp_val,
        "thal_0.0": thal_val,
    }
    input_df = pd.DataFrame([input_dict])[feature_order]

    display_values = {
        "Tuổi": age,
        "Huyết áp (mmHg)": trestbps,
        "Cholesterol (mg/dl)": chol,
        "Độ chênh ST": oldpeak,
        "Giới tính": sex,
        "Đường huyết": fbs,
        "ECG": restecg.split("(")[0].strip(),
        "Đau ngực gắng sức": exang,
        "Độ dốc ST": slope.split("(")[0].strip(),
        "Số mạch máu": ca,
        "Loại đau ngực": cp.split("(")[0].strip(),
        "Thalassemia": thal.split("(")[0].strip(),
    }

    return input_df, display_values


def render_input_summary(input_df, display_values):
    """Render expander hiển thị tóm tắt dữ liệu đã nhập."""
    st.markdown("---")
    with st.expander("📋 Xem tóm tắt dữ liệu đã nhập", expanded=False):
        summary_col1, summary_col2 = st.columns(2)
        with summary_col1:
            st.markdown("**Giá trị hiển thị:**")
            summary_data = pd.DataFrame(display_values.items(), columns=["Chỉ số", "Giá trị"])
            summary_data["Giá trị"] = summary_data["Giá trị"].astype(str)
            st.dataframe(
                summary_data,
                hide_index=True,
                use_container_width=True,
            )
        with summary_col2:
            st.markdown("**Vector đầu vào model (số):**")
            st.dataframe(input_df, hide_index=True, use_container_width=True)

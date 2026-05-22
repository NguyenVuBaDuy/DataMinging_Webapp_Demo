"""
ui_samples.py - Giao diện hiển thị danh sách bệnh nhân mẫu từ xai_insights_test.csv
và chức năng tự động điền (auto-fill) vào form nhập liệu.
"""

import streamlit as st
import pandas as pd


def init_session_state_if_needed():
    """Khởi tạo giá trị mặc định cho form nhập liệu lâm sàng nếu chưa tồn tại trong session state."""
    defaults = {
        "age": 50,
        "trestbps": 120,
        "chol": 200,
        "oldpeak": 1.0,
        "sex_display": "Nam",
        "fbs_display": "≤ 120 mg/dl (Bình thường)",
        "exang_display": "Không",
        "cp_display": "Loại khác (cp≠0)",
        "restecg_display": "Bình thường (0)",
        "slope_display": "Phẳng (1)",
        "ca_display": "0",
        "thal_display": "Bình thường (thal=0)"
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def load_patient_into_state(patient_row):
    """Nạp tất cả chỉ số của bệnh nhân mẫu được chọn vào session state.

    Args:
        patient_row: pd.Series or dict, hàng dữ liệu bệnh nhân từ DataFrame.
    """
    st.session_state["age"] = int(patient_row["age"])
    st.session_state["trestbps"] = int(patient_row["trestbps"])
    st.session_state["chol"] = int(patient_row["chol"])
    st.session_state["oldpeak"] = float(patient_row["oldpeak"])

    # Categorical mapping
    sex_val = float(patient_row["sex"])
    st.session_state["sex_display"] = "Nam" if sex_val == 1.0 else "Nữ"

    fbs_val = float(patient_row["fbs"])
    st.session_state["fbs_display"] = "> 120 mg/dl (Cao)" if fbs_val == 1.0 else "≤ 120 mg/dl (Bình thường)"

    exang_val = float(patient_row["exang"])
    st.session_state["exang_display"] = "Có" if exang_val == 1.0 else "Không"

    cp_val = float(patient_row["cp_0.0"])
    st.session_state["cp_display"] = "Đau thắt điển hình (cp=0)" if cp_val == 1.0 else "Loại khác (cp≠0)"

    restecg_val = int(patient_row["restecg"])
    if restecg_val == 0:
        st.session_state["restecg_display"] = "Bình thường (0)"
    elif restecg_val == 1:
        st.session_state["restecg_display"] = "Bất thường sóng ST-T (1)"
    else:
        st.session_state["restecg_display"] = "Phì đại thất trái (2)"

    slope_val = int(patient_row["slope"])
    if slope_val == 0:
        st.session_state["slope_display"] = "Dốc lên (0)"
    elif slope_val == 1:
        st.session_state["slope_display"] = "Phẳng (1)"
    else:
        st.session_state["slope_display"] = "Dốc xuống (2)"

    st.session_state["ca_display"] = str(int(patient_row["ca"]))

    thal_val = float(patient_row["thal_0.0"])
    st.session_state["thal_display"] = "Bình thường (thal=0)" if thal_val == 1.0 else "Bất thường (thal≠0)"


def render_sample_patients_tab(df_patients):
    """Render giao diện Tab Bệnh Nhân Mẫu.

    Args:
        df_patients: pd.DataFrame chứa 184 mẫu bệnh nhân từ xai_insights_test.csv.
    """
    if df_patients is None:
        st.warning("⚠️ Không tìm thấy file dữ liệu bệnh nhân mẫu xai_insights_test.csv.")
        return

    st.markdown("""
    <div style="background: rgba(30, 58, 95, 0.05); border-radius: 12px; padding: 1.2rem; margin-bottom: 1.5rem; border-left: 4px solid #1e3a5f;">
        <h4 style="margin: 0 0 0.5rem 0; font-weight: 700; color: #1e3a5f;">👥 Kho Dữ Liệu Bệnh Nhân Lâm Sàng Mẫu</h4>
        <p style="margin: 0; opacity: 0.85; font-size: 0.95rem;">
            Chọn một bệnh nhân mẫu trong kho lưu trữ để tự động điền các chỉ số lâm sàng phức tạp.
            Bạn có thể đối chiếu dự đoán của AI với <strong>kết quả thực tế (Actual Target)</strong> đã được xác minh y khoa.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Tạo nhãn hiển thị trực quan cho selectbox
    def make_label(row):
        gender = "Nam" if row["sex"] == 1.0 else "Nữ"
        target_str = "Có bệnh" if row["Actual_Target"] == 1 else "Không bệnh"
        return f"Bệnh nhân #{int(row['Patient_Index'])} (Tuổi: {int(row['age'])}, Giới: {gender}, Thực tế: {target_str})"

    # Selectbox chọn bệnh nhân
    patient_options = [make_label(df_patients.iloc[i]) for i in range(len(df_patients))]
    selected_idx = st.selectbox(
        "🔍 Chọn bệnh nhân mẫu từ kho dữ liệu:",
        options=range(len(df_patients)),
        format_func=lambda x: patient_options[x],
        index=0,
        help="Chọn một bệnh nhân mẫu để xem trước thông tin chi tiết."
    )

    selected_row = df_patients.iloc[selected_idx]

    # --- Hiển thị thông tin chi tiết bệnh nhân được chọn dưới dạng Grid đẹp mắt ---
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**📋 Chỉ số lâm sàng cơ bản:**")
        gender_str = "Nam" if selected_row["sex"] == 1.0 else "Nữ"
        st.write(f"- 🎂 **Tuổi:** {int(selected_row['age'])}")
        st.write(f"- 👤 **Giới tính:** {gender_str}")
        st.write(f"- 💉 **Huyết áp lúc nghỉ:** {int(selected_row['trestbps'])} mmHg")
        st.write(f"- 🧪 **Cholesterol:** {int(selected_row['chol'])} mg/dl")

    with col2:
        st.markdown("**🔬 Chỉ số điện tâm đồ & Khác:**")
        exang_str = "Có" if selected_row["exang"] == 1.0 else "Không"
        cp_str = "Đau thắt điển hình" if selected_row["cp_0.0"] == 1.0 else "Loại khác"
        st.write(f"- 💔 **Đau thắt ngực (cp=0):** {cp_str}")
        st.write(f"- 🏃 **Đau ngực khi gắng sức:** {exang_str}")
        st.write(f"- 📉 **ST depression (oldpeak):** {selected_row['oldpeak']}")
        st.write(f"- 🫀 **Số mạch máu chính (ca):** {int(selected_row['ca'])}")

    with col3:
        st.markdown("**🩺 Kết quả chẩn đoán thực tế & AI:**")
        actual_val = "🚨 Có nguy cơ bệnh tim" if selected_row["Actual_Target"] == 1 else "✅ Không phát hiện nguy cơ"
        lgbm_val = "🚨 Có nguy cơ" if selected_row["LightGBM_Pred"] == 1 else "✅ Không nguy cơ"
        tabnet_val = "🚨 Có nguy cơ" if selected_row["TabNet_Pred"] == 1 else "✅ Không nguy cơ"

        st.markdown(f"**Kết quả lâm sàng thực tế:**  \n**{actual_val}**")
        st.markdown(f"- Dự đoán LightGBM: **{lgbm_val}**")
        st.markdown(f"- Dự đoán TabNet: **{tabnet_val}**")

    st.markdown("---")

    # Nút bấm auto-fill dữ liệu vào form
    if st.button("📥 Nạp Dữ Liệu Bệnh Nhân Này Vào Form Nhập Liệu", use_container_width=True, type="primary"):
        load_patient_into_state(selected_row)
        st.success(f"🎉 Đã nạp thành công dữ liệu Bệnh nhân #{int(selected_row['Patient_Index'])} vào form! Hãy nhấn sang Tab '📝 Nhập Chỉ Số Bệnh Nhân' để kiểm tra hoặc bấm 'Tiến Hành Chẩn Đoán' ở sidebar.")
        # Buộc Streamlit reload trang để áp dụng session state mới cho các widget ngay lập tức
        st.rerun()

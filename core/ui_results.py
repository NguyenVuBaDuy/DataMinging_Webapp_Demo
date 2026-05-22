"""
ui_results.py - Hiển thị kết quả chẩn đoán + gọi XAI modules.

Bước 5 trong kế hoạch Phase 4:
    - Tầng 1: Kết quả chẩn đoán (xác suất, nhãn, metric)
    - Tầng 2: Biểu đồ giải thích XAI (SHAP / Attention)
"""

import streamlit as st


def render_diagnosis_result(result):
    """Render kết quả chẩn đoán chính.

    Args:
        result: dict từ diagnosis.run_diagnosis()
    """
    probability = result["probability"]
    prediction = result["prediction"]
    label = result["label"]
    model_name = result["model_name"]
    threshold = result["threshold"]
    pct = probability * 100

    # --- Card kết quả ---
    if prediction == 1:
        # Có nguy cơ → đỏ
        card_bg = "linear-gradient(135deg, #991b1b 0%, #dc2626 50%, #ef4444 100%)"
        icon = "⚠️"
        status_text = "PHÁT HIỆN NGUY CƠ BỆNH TIM"
        border_color = "#ef4444"
    else:
        # Không nguy cơ → xanh
        card_bg = "linear-gradient(135deg, #065f46 0%, #059669 50%, #10b981 100%)"
        icon = "✅"
        status_text = "KHÔNG PHÁT HIỆN NGUY CƠ"
        border_color = "#10b981"

    st.markdown(f"""
    <div style="
        background: {card_bg};
        border-radius: 16px;
        padding: 2rem 2.5rem;
        margin: 1.5rem 0;
        color: white;
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
        border-left: 6px solid {border_color};
    ">
        <div style="display:flex; align-items:center; gap:12px; margin-bottom:1rem;">
            <span style="font-size:2.5rem;">{icon}</span>
            <div>
                <h2 style="margin:0; font-size:1.6rem; font-weight:800; letter-spacing:-0.5px;">
                    {status_text}
                </h2>
                <p style="margin:0.2rem 0 0 0; opacity:0.9; font-size:0.95rem;">
                    Kết quả từ mô hình {model_name}
                </p>
            </div>
        </div>
        <div style="display:flex; gap:2rem; flex-wrap:wrap;">
            <div style="background:rgba(255,255,255,0.12); border-radius:12px; padding:1rem 1.5rem; flex:1; min-width:180px;">
                <div style="font-size:0.75rem; text-transform:uppercase; letter-spacing:1px; opacity:0.8; font-weight:600;">
                    Xác suất nguy cơ
                </div>
                <div style="font-size:2rem; font-weight:800; margin-top:0.3rem;">
                    {pct:.1f}%
                </div>
            </div>
            <div style="background:rgba(255,255,255,0.12); border-radius:12px; padding:1rem 1.5rem; flex:1; min-width:180px;">
                <div style="font-size:0.75rem; text-transform:uppercase; letter-spacing:1px; opacity:0.8; font-weight:600;">
                    Ngưỡng phân loại
                </div>
                <div style="font-size:2rem; font-weight:800; margin-top:0.3rem;">
                    {threshold * 100:.0f}%
                </div>
            </div>
            <div style="background:rgba(255,255,255,0.12); border-radius:12px; padding:1rem 1.5rem; flex:1; min-width:180px;">
                <div style="font-size:0.75rem; text-transform:uppercase; letter-spacing:1px; opacity:0.8; font-weight:600;">
                    Kết luận
                </div>
                <div style="font-size:1.3rem; font-weight:700; margin-top:0.3rem;">
                    {label}
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Progress bar xác suất
    st.progress(min(probability, 1.0))

    # Disclaimer y tế
    st.markdown("""
    <div style="
        background: rgba(251, 191, 36, 0.1);
        border: 1px solid rgba(251, 191, 36, 0.3);
        border-radius: 8px;
        padding: 0.8rem 1.2rem;
        margin: 0.5rem 0 1.5rem 0;
        font-size: 0.85rem;
        color: #d97706;
    ">
        ⚕️ <strong>Lưu ý:</strong> Đây là công cụ hỗ trợ sàng lọc, <u>KHÔNG</u> thay thế chẩn đoán y khoa.
        Vui lòng tham khảo ý kiến bác sĩ chuyên khoa để có kết luận chính xác.
    </div>
    """, unsafe_allow_html=True)


def render_xai_section(model_choice, resources, input_df):
    """Render phần giải thích XAI tùy theo model đã chọn.

    Args:
        model_choice: "LightGBM" hoặc "TabNet"
        resources: dict từ load_all_resources()
        input_df: pd.DataFrame, 1 dòng input
    """
    st.markdown("---")
    st.markdown("### 🔬 Giải Thích Kết Quả (Explainable AI)")

    if model_choice == "LightGBM":
        from core.xai_shap import compute_shap_values, render_shap_waterfall

        with st.spinner("Đang tính SHAP values..."):
            shap_values = compute_shap_values(
                model=resources["lgbm_model"],
                background_data=resources["shap_background"],
                input_df=input_df,
            )
        render_shap_waterfall(shap_values, input_df)

    elif model_choice == "TabNet":
        from core.xai_tabnet import compute_attention_masks, render_attention_heatmap

        with st.spinner("Đang tính Attention Masks..."):
            explain_matrix, masks = compute_attention_masks(
                model=resources["tabnet_model"],
                scaler=resources["tabnet_scaler"],
                input_df=input_df,
            )
        render_attention_heatmap(explain_matrix, masks, input_df)

"""
xai_tabnet.py - Attention Heatmap cho TabNet.
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from core.config import FEATURE_LABELS_VI

matplotlib.use("Agg")


def compute_attention_masks(model, scaler, input_df):
    """Tính Attention Masks từ TabNet."""
    scaled_input = scaler.transform(input_df.values)
    explain_matrix, masks = model.explain(scaled_input)
    return explain_matrix, masks


def render_attention_heatmap(explain_matrix, masks, input_df):
    """Render Attention Heatmap với nhãn tiếng Việt."""
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #7c3aed 0%, #a855f7 100%);
        border-radius: 12px; padding: 1rem 1.5rem;
        margin-bottom: 1rem; color: white;
    ">
        <h4 style="margin:0; font-weight:700;">🧠 TabNet Attention Heatmap</h4>
        <p style="margin:0.3rem 0 0 0; opacity:0.85; font-size:0.9rem;">
            Màu càng <span style="color:#fbbf24;font-weight:600;">sáng</span>
            = mô hình chú ý nhiều hơn vào chỉ số đó.
        </p>
    </div>
    """, unsafe_allow_html=True)

    feature_names = list(input_df.columns)
    feature_names_vi = [FEATURE_LABELS_VI.get(f, f) for f in feature_names]

    # Bar chart - tổng hợp importance
    attention_scores = explain_matrix[0]
    total = attention_scores.sum()
    att_norm = (attention_scores / total * 100) if total > 0 else attention_scores

    idx = np.argsort(att_norm)[::-1]
    sorted_names = [feature_names_vi[i] for i in idx]
    sorted_scores = att_norm[idx]

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.YlOrRd(np.linspace(0.3, 0.95, len(sorted_scores)))
    bars = ax.barh(range(len(sorted_names)), sorted_scores, color=colors[::-1],
                   edgecolor="white", linewidth=0.5, height=0.7)
    ax.set_yticks(range(len(sorted_names)))
    ax.set_yticklabels(sorted_names, fontsize=11)
    ax.set_xlabel("Mức độ chú ý (%)", fontsize=12, fontweight="600")
    ax.set_title("Mức độ chú ý của TabNet", fontsize=14, fontweight="bold", pad=15)
    ax.invert_yaxis()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for bar, score in zip(bars, sorted_scores):
        if score > 0.5:
            ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                    f"{score:.1f}%", va="center", ha="left", fontsize=10, fontweight="600")
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    # Heatmap qua decision steps
    if masks is not None and len(masks) > 0:
        with st.expander("🔍 Attention Masks theo Decision Step", expanded=False):
            # masks có thể là dict {step: ndarray} hoặc list of ndarrays
            if isinstance(masks, dict):
                mask_list = [masks[k] for k in sorted(masks.keys())]
            else:
                mask_list = list(masks)

            # Mỗi mask có thể có shape (1, n_features) hoặc (n_features,)
            processed = []
            for m in mask_list:
                if isinstance(m, np.ndarray):
                    if m.ndim == 2:
                        processed.append(m[0])
                    else:
                        processed.append(m)

            if processed:
                mask_matrix = np.vstack([p.reshape(1, -1) for p in processed])
                n_steps = mask_matrix.shape[0]
                fig2, ax2 = plt.subplots(figsize=(12, max(3, n_steps * 0.8)))
                im = ax2.imshow(mask_matrix, cmap="YlOrRd", aspect="auto")
                ax2.set_xticks(range(len(feature_names_vi)))
                ax2.set_xticklabels(feature_names_vi, rotation=45, ha="right", fontsize=9)
                ax2.set_yticks(range(n_steps))
                ax2.set_yticklabels([f"Step {i+1}" for i in range(n_steps)])
                ax2.set_title("Attention Mask theo Decision Step", fontsize=13, fontweight="bold")
                plt.colorbar(im, ax=ax2, shrink=0.8, label="Mức chú ý")
                plt.tight_layout()
                st.pyplot(fig2, use_container_width=True)
                plt.close(fig2)

    # Bảng chi tiết
    with st.expander("📋 Chi tiết Attention Scores", expanded=False):
        detail = pd.DataFrame({
            "Chỉ số": [feature_names_vi[i] for i in idx],
            "Giá trị": [str(input_df.iloc[0, idx[k]]) for k in range(len(idx))],
            "Attention (%)": np.round(sorted_scores, 2),
        })
        st.dataframe(detail, hide_index=True, use_container_width=True)

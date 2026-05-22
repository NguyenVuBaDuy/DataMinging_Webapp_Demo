import streamlit as st
import json
import joblib
import pandas as pd
import numpy as np
import os

# ============================================================
# CẤU HÌNH TRANG (phải là lệnh Streamlit đầu tiên)
# ============================================================
st.set_page_config(
    page_title="❤️ Hệ Thống Chẩn Đoán Bệnh Tim",
    page_icon="❤️",
    layout="wide",
)

# ============================================================
# BƯỚC 2: LOAD MODEL & CONFIG
# ============================================================

# --- Đường dẫn tới thư mục chứa model và config ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_DIR = os.path.join(BASE_DIR, "models_and_config")


# --- 2.1: Load app_config.json ---
@st.cache_resource
def load_config():
    """Đọc file cấu hình: feature_order, ngưỡng LightGBM, ngưỡng TabNet."""
    config_path = os.path.join(CONFIG_DIR, "app_config.json")
    with open(config_path, "r") as f:
        config = json.load(f)
    return config


# --- 2.2: Load LightGBM model ---
@st.cache_resource
def load_lightgbm_model():
    """Load model LightGBM đã train sẵn bằng joblib."""
    model_path = os.path.join(CONFIG_DIR, "best_lightgbm_model.pkl")
    model = joblib.load(model_path)
    return model


# --- 2.3: Load TabNet model ---
@st.cache_resource
def load_tabnet_model():
    """Load model TabNet từ file .zip bằng TabNetClassifier."""
    from pytorch_tabnet.tab_model import TabNetClassifier

    model_path = os.path.join(CONFIG_DIR, "best_tabnet_model.zip")
    model = TabNetClassifier()
    model.load_model(model_path)
    return model


# --- 2.4: Load TabNet scaler ---
@st.cache_resource
def load_tabnet_scaler():
    """Load scaler (StandardScaler) dùng để chuẩn hóa dữ liệu trước khi đưa vào TabNet.
    
    Note: File pickle chứa tuple (StandardScaler, training_data_array).
    Chỉ cần lấy StandardScaler ở index [0].
    """
    scaler_path = os.path.join(CONFIG_DIR, "tabnet_scaler.pkl")
    scaler_tuple = joblib.load(scaler_path)
    # scaler_tuple = (StandardScaler, ndarray(184, 12))
    scaler = scaler_tuple[0]
    return scaler


# --- 2.5: Load SHAP background data ---
@st.cache_resource
def load_shap_background_data():
    """Load 100 mẫu background data dùng cho SHAP TreeExplainer."""
    data_path = os.path.join(CONFIG_DIR, "shap_background_data.csv")
    data = pd.read_csv(data_path)
    return data


# ============================================================
# KHỞI TẠO: Load tất cả tài nguyên khi app khởi động
# ============================================================

# Load config
config = load_config()
FEATURE_ORDER = config["feature_order"]
LIGHTGBM_THRESHOLD = config["lightgbm_threshold"]
TABNET_THRESHOLD = config["tabnet_threshold"]

# Load models & scaler
lgbm_model = load_lightgbm_model()
tabnet_model = load_tabnet_model()
tabnet_scaler = load_tabnet_scaler()

# Load SHAP background data
shap_background = load_shap_background_data()


# ============================================================
# CUSTOM CSS - Giao diện y tế chuyên nghiệp
# ============================================================
st.markdown("""
<style>
/* === Import Google Font === */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

/* === Global === */
.stApp {
    font-family: 'Inter', sans-serif;
}

/* === Header Section === */
.main-header {
    background: linear-gradient(135deg, #1e3a5f 0%, #2d6a9f 50%, #4a90d9 100%);
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
    color: white;
    box-shadow: 0 8px 32px rgba(30, 58, 95, 0.3);
    position: relative;
    overflow: hidden;
}
.main-header::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -20%;
    width: 300px;
    height: 300px;
    background: radial-gradient(circle, rgba(255,255,255,0.08) 0%, transparent 70%);
    border-radius: 50%;
}
.main-header h1 {
    font-size: 2rem;
    font-weight: 800;
    margin: 0 0 0.3rem 0;
    letter-spacing: -0.5px;
}
.main-header p {
    font-size: 1rem;
    opacity: 0.85;
    margin: 0;
    font-weight: 400;
}

/* === Section Cards === */
.section-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    margin-bottom: 1rem;
}
.section-card h3 {
    font-size: 1rem;
    font-weight: 700;
    margin: 0 0 1rem 0;
    color: #e2e8f0;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

/* === Feature Info Badge === */
.feature-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: linear-gradient(135deg, #1e3a5f, #2d6a9f);
    color: white;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 600;
    margin-bottom: 0.5rem;
}

/* === Sidebar Styling === */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
}
[data-testid="stSidebar"] .stMarkdown h1,
[data-testid="stSidebar"] .stMarkdown h2,
[data-testid="stSidebar"] .stMarkdown h3 {
    color: #e2e8f0;
}

/* === Diagnose Button === */
[data-testid="stSidebar"] .stButton > button {
    background: linear-gradient(135deg, #dc2626 0%, #ef4444 50%, #f87171 100%);
    color: white;
    font-weight: 700;
    font-size: 1.05rem;
    border: none;
    border-radius: 12px;
    padding: 0.8rem 1.5rem;
    width: 100%;
    letter-spacing: 0.3px;
    transition: all 0.3s ease;
    box-shadow: 0 4px 15px rgba(220, 38, 38, 0.4);
}
[data-testid="stSidebar"] .stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 25px rgba(220, 38, 38, 0.5);
}

/* === Model Info Cards in Sidebar === */
.model-info-card {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 10px;
    padding: 0.8rem 1rem;
    margin: 0.5rem 0;
}
.model-info-card .label {
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: #94a3b8;
    font-weight: 600;
}
.model-info-card .value {
    font-size: 1.1rem;
    font-weight: 700;
    color: #38bdf8;
}

/* === Status Indicator === */
.status-dot {
    display: inline-block;
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: #22c55e;
    margin-right: 6px;
    animation: pulse-dot 2s infinite;
}
@keyframes pulse-dot {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.4; }
}
</style>
""", unsafe_allow_html=True)


# ============================================================
# BƯỚC 3: SIDEBAR - Chọn Model & Nút Chẩn Đoán
# ============================================================

with st.sidebar:
    st.markdown("## 🏥 Bảng Điều Khiển")
    st.markdown("---")

    # -- Chọn model --
    st.markdown("### 🤖 Chọn Mô Hình AI")
    model_choice = st.selectbox(
        "Mô hình dự đoán",
        options=["LightGBM", "TabNet"],
        index=0,
        help="LightGBM: Nhanh, chính xác cao. TabNet: Deep Learning, có Attention Map."
    )

    # -- Hiển thị thông tin model đã chọn --
    if model_choice == "LightGBM":
        threshold = LIGHTGBM_THRESHOLD
        st.markdown(f"""
        <div class="model-info-card">
            <div class="label">Ngưỡng phân loại</div>
            <div class="value">{threshold}</div>
        </div>
        """, unsafe_allow_html=True)
        st.caption("🌳 Gradient Boosting Decision Tree")
        st.caption("📊 XAI: SHAP Waterfall Plot")
    else:
        threshold = TABNET_THRESHOLD
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
        type="primary"
    )

    st.markdown("---")

    # -- Trạng thái hệ thống --
    st.markdown("### ⚙️ Trạng Thái Hệ Thống")
    st.markdown(
        '<span class="status-dot"></span> Models đã sẵn sàng',
        unsafe_allow_html=True
    )
    st.caption(f"Features: {len(FEATURE_ORDER)} chỉ số")
    st.caption(f"SHAP data: {shap_background.shape[0]} mẫu")


# ============================================================
# BƯỚC 3: MAIN AREA - Header & Form Nhập Liệu
# ============================================================

# --- Header ---
st.markdown("""
<div class="main-header">
    <h1>❤️ Hệ Thống Chẩn Đoán Bệnh Tim</h1>
    <p>Ứng dụng Machine Learning & Explainable AI hỗ trợ sàng lọc nguy cơ bệnh tim mạch</p>
</div>
""", unsafe_allow_html=True)

# --- Form nhập liệu ---
st.markdown("### 📝 Thông Tin Lâm Sàng Bệnh Nhân")
st.caption("Vui lòng nhập đầy đủ 12 chỉ số lâm sàng bên dưới để hệ thống phân tích.")

# --- Dictionary nhãn tiếng Việt cho features ---
FEATURE_LABELS_VI = {
    "age": "Tuổi",
    "trestbps": "Huyết áp lúc nghỉ (mmHg)",
    "chol": "Cholesterol huyết thanh (mg/dl)",
    "oldpeak": "Độ chênh ST khi vận động",
    "sex": "Giới tính",
    "fbs": "Đường huyết lúc đói",
    "restecg": "Kết quả điện tâm đồ",
    "exang": "Đau thắt ngực khi gắng sức",
    "slope": "Độ dốc đoạn ST",
    "ca": "Số mạch máu chính (Fluoroscopy)",
    "cp_0.0": "Loại đau ngực",
    "thal_0.0": "Thalassemia",
}

# ============================
# CỘT 1: Chỉ số liên tục
# ============================
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown('<div class="section-card"><h3>📏 Chỉ Số Liên Tục</h3></div>', unsafe_allow_html=True)

    age = st.slider(
        "🎂 Tuổi",
        min_value=20, max_value=80, value=50, step=1,
        help="Tuổi của bệnh nhân (năm)"
    )

    trestbps = st.slider(
        "💉 Huyết áp lúc nghỉ (mmHg)",
        min_value=90, max_value=200, value=120, step=1,
        help="Huyết áp tâm thu lúc nghỉ ngơi"
    )

    chol = st.slider(
        "🧪 Cholesterol huyết thanh (mg/dl)",
        min_value=120, max_value=560, value=200, step=1,
        help="Cholesterol trong máu"
    )

    oldpeak = st.slider(
        "📉 Độ chênh ST khi vận động",
        min_value=0.0, max_value=6.5, value=1.0, step=0.1,
        help="ST depression induced by exercise relative to rest"
    )

# ============================
# CỘT 2: Chỉ số nhị phân
# ============================
with col2:
    st.markdown('<div class="section-card"><h3>🔘 Chỉ Số Phân Loại (Nhị phân)</h3></div>', unsafe_allow_html=True)

    sex = st.selectbox(
        "👤 Giới tính",
        options=["Nam", "Nữ"],
        index=0,
        help="Giới tính sinh học"
    )
    sex_val = 1 if sex == "Nam" else 0

    fbs = st.selectbox(
        "🍬 Đường huyết lúc đói",
        options=["≤ 120 mg/dl (Bình thường)", "> 120 mg/dl (Cao)"],
        index=0,
        help="Đường huyết lúc đói có > 120 mg/dl không?"
    )
    fbs_val = 0 if "Bình thường" in fbs else 1

    exang = st.selectbox(
        "🏃 Đau thắt ngực khi gắng sức",
        options=["Không", "Có"],
        index=0,
        help="Có đau thắt ngực khi tập thể dục không?"
    )
    exang_val = 1 if exang == "Có" else 0

    cp = st.selectbox(
        "💔 Loại đau ngực (cp)",
        options=["Đau thắt điển hình (cp=0)", "Loại khác (cp≠0)"],
        index=1,
        help="cp_0.0: 1 = Đau thắt điển hình, 0 = Các loại đau ngực khác"
    )
    cp_val = 1 if "cp=0" in cp else 0

# ============================
# CỘT 3: Chỉ số đa nhãn
# ============================
with col3:
    st.markdown('<div class="section-card"><h3>📊 Chỉ Số Phân Loại (Đa nhãn)</h3></div>', unsafe_allow_html=True)

    restecg = st.selectbox(
        "📈 Kết quả điện tâm đồ (ECG)",
        options=[
            "Bình thường (0)",
            "Bất thường sóng ST-T (1)",
            "Phì đại thất trái (2)"
        ],
        index=0,
        help="Kết quả đo điện tâm đồ lúc nghỉ ngơi"
    )
    restecg_val = int(restecg.split("(")[1].replace(")", ""))

    slope = st.selectbox(
        "📐 Độ dốc đoạn ST",
        options=[
            "Dốc lên (0)",
            "Phẳng (1)",
            "Dốc xuống (2)"
        ],
        index=1,
        help="Độ dốc của đoạn ST đỉnh khi vận động"
    )
    slope_val = int(slope.split("(")[1].replace(")", ""))

    ca = st.selectbox(
        "🫀 Số mạch máu chính (Fluoroscopy)",
        options=["0", "1", "2", "3", "4"],
        index=0,
        help="Số lượng mạch máu chính được nhuộm bằng Fluoroscopy (0-4)"
    )
    ca_val = int(ca)

    thal = st.selectbox(
        "🩸 Thalassemia (thal)",
        options=["Bình thường (thal=0)", "Bất thường (thal≠0)"],
        index=0,
        help="thal_0.0: 1 = Bình thường, 0 = Bất thường (Khiếm khuyết)"
    )
    thal_val = 1 if "thal=0" in thal else 0


# ============================================================
# GOM DỮ LIỆU THEO ĐÚNG feature_order
# ============================================================
# feature_order: ["age","trestbps","chol","oldpeak","sex","fbs","restecg","exang","slope","ca","cp_0.0","thal_0.0"]

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

input_df = pd.DataFrame([input_dict])[FEATURE_ORDER]

# --- Hiển thị bảng tóm tắt dữ liệu đã nhập ---
st.markdown("---")
with st.expander("📋 Xem tóm tắt dữ liệu đã nhập", expanded=False):
    summary_col1, summary_col2 = st.columns(2)
    with summary_col1:
        st.markdown("**Giá trị hiển thị:**")
        display_data = {
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
        st.dataframe(pd.DataFrame(display_data.items(), columns=["Chỉ số", "Giá trị"]),
                      hide_index=True, use_container_width=True)
    with summary_col2:
        st.markdown("**Vector đầu vào model (số):**")
        st.dataframe(input_df, hide_index=True, use_container_width=True)

# --- Placeholder cho kết quả chẩn đoán (Bước 4 & 5 sẽ xây ở đây) ---
result_container = st.container()

if diagnose_clicked:
    with result_container:
        st.markdown("---")
        st.info("⏳ Chức năng chẩn đoán sẽ được xây dựng ở **Bước 4 & 5**. Dữ liệu đã sẵn sàng!")
        st.markdown(f"**Model đã chọn:** `{model_choice}` | **Ngưỡng:** `{threshold}`")
        st.dataframe(input_df, hide_index=True, use_container_width=True)

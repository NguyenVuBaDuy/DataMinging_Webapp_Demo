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
# TRANG CHÍNH (tạm thời - để kiểm tra Bước 2)
# ============================================================

# (st.set_page_config đã được gọi ở đầu file)

st.title("❤️ Hệ Thống Chẩn Đoán Bệnh Tim")
st.caption("Sử dụng Machine Learning & Explainable AI")

st.divider()

# --- Hiển thị thông tin kiểm tra ---
st.subheader("✅ Kiểm Tra Bước 2: Load Model & Config")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**📋 Config đã load:**")
    st.json(config)

with col2:
    st.markdown("**🤖 Models đã load:**")
    st.success(f"LightGBM: {type(lgbm_model).__name__}")
    st.success(f"TabNet: {type(tabnet_model).__name__}")
    st.success(f"Scaler: {type(tabnet_scaler).__name__}")
    st.info(f"Ngưỡng LightGBM: {LIGHTGBM_THRESHOLD}")
    st.info(f"Ngưỡng TabNet: {TABNET_THRESHOLD}")

with col3:
    st.markdown("**📊 SHAP Background Data:**")
    st.write(f"Shape: {shap_background.shape}")
    st.write(f"Columns: {list(shap_background.columns)}")
    st.dataframe(shap_background.head(), width='stretch')

st.divider()
st.info("🎉 Bước 2 hoàn tất! Tất cả model và config đã được load thành công. Sẵn sàng cho Bước 3.")

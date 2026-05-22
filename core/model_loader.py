"""
model_loader.py - Load model, scaler, config và SHAP data.
Tất cả được cache bằng @st.cache_resource.
"""

import json
import os

import joblib
import pandas as pd
import streamlit as st

from core.config import CONFIG_DIR


# --- Load app_config.json ---
@st.cache_resource
def load_config():
    """Đọc file cấu hình: feature_order, ngưỡng LightGBM, ngưỡng TabNet."""
    config_path = os.path.join(CONFIG_DIR, "app_config.json")
    with open(config_path, "r") as f:
        config = json.load(f)
    return config


# --- Load LightGBM model ---
@st.cache_resource
def load_lightgbm_model():
    """Load model LightGBM đã train sẵn bằng joblib."""
    model_path = os.path.join(CONFIG_DIR, "best_lightgbm_model.pkl")
    model = joblib.load(model_path)
    return model


# --- Load TabNet model ---
@st.cache_resource
def load_tabnet_model():
    """Load model TabNet từ file .zip bằng TabNetClassifier."""
    from pytorch_tabnet.tab_model import TabNetClassifier

    model_path = os.path.join(CONFIG_DIR, "best_tabnet_model.zip")
    model = TabNetClassifier()
    model.load_model(model_path)
    return model


# --- Load TabNet scaler ---
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


# --- Load SHAP background data ---
@st.cache_resource
def load_shap_background_data():
    """Load 100 mẫu background data dùng cho SHAP TreeExplainer."""
    data_path = os.path.join(CONFIG_DIR, "shap_background_data.csv")
    data = pd.read_csv(data_path)
    return data


def load_all_resources():
    """Load tất cả tài nguyên và trả về dict chứa mọi thứ cần thiết.

    Returns:
        dict với keys: config, feature_order, lightgbm_threshold,
        tabnet_threshold, lgbm_model, tabnet_model, tabnet_scaler,
        shap_background
    """
    config = load_config()

    return {
        "config": config,
        "feature_order": config["feature_order"],
        "lightgbm_threshold": config["lightgbm_threshold"],
        "tabnet_threshold": config["tabnet_threshold"],
        "lgbm_model": load_lightgbm_model(),
        "tabnet_model": load_tabnet_model(),
        "tabnet_scaler": load_tabnet_scaler(),
        "shap_background": load_shap_background_data(),
    }

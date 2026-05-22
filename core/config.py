"""
config.py - Hằng số, đường dẫn, và nhãn tiếng Việt cho features.
Không phụ thuộc Streamlit, chỉ chứa dữ liệu tĩnh.
"""

import os

# ============================================================
# ĐƯỜNG DẪN
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_DIR = os.path.join(BASE_DIR, "models_and_config")

# ============================================================
# NHÃN TIẾNG VIỆT CHO FEATURES
# ============================================================
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

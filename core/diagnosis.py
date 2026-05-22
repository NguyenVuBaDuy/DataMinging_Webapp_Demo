"""
diagnosis.py - Luồng xử lý backend: dự đoán bệnh tim bằng LightGBM hoặc TabNet.

Bước 4 trong kế hoạch Phase 4:
    - Gom input thành DataFrame theo đúng feature_order
    - LightGBM: dự đoán trực tiếp → áp ngưỡng 0.50
    - TabNet: scaler.transform() → dự đoán → áp ngưỡng 0.24
    - Trả về xác suất + nhãn dự đoán
"""

import numpy as np
import pandas as pd


def predict_lightgbm(model, input_df, threshold):
    """Dự đoán bằng LightGBM.

    Args:
        model: LightGBM model đã load.
        input_df: pd.DataFrame 1 dòng, đúng feature_order.
        threshold: float, ngưỡng phân loại (mặc định 0.50).

    Returns:
        dict: {
            "probability": float (0-1),
            "prediction": int (0 hoặc 1),
            "label": str ("Có nguy cơ" hoặc "Không có nguy cơ"),
            "model_name": "LightGBM",
            "threshold": float,
        }
    """
    # predict_proba trả về array shape (n_samples, 2)
    # cột 0 = P(class=0), cột 1 = P(class=1 = bệnh tim)
    proba = model.predict_proba(input_df)[:, 1]
    probability = float(proba[0])
    prediction = int(probability >= threshold)

    return {
        "probability": probability,
        "prediction": prediction,
        "label": "Có nguy cơ bệnh tim" if prediction == 1 else "Không có nguy cơ bệnh tim",
        "model_name": "LightGBM",
        "threshold": threshold,
    }


def predict_tabnet(model, scaler, input_df, threshold):
    """Dự đoán bằng TabNet (cần scale dữ liệu trước).

    Args:
        model: TabNet model đã load.
        scaler: StandardScaler đã fit trên training data.
        input_df: pd.DataFrame 1 dòng, đúng feature_order.
        threshold: float, ngưỡng phân loại (mặc định 0.24).

    Returns:
        dict: tương tự predict_lightgbm
    """
    # Scale dữ liệu đầu vào
    scaled_input = scaler.transform(input_df.values)

    # predict_proba trả về array shape (n_samples, 2)
    proba = model.predict_proba(scaled_input)[:, 1]
    probability = float(proba[0])
    prediction = int(probability >= threshold)

    return {
        "probability": probability,
        "prediction": prediction,
        "label": "Có nguy cơ bệnh tim" if prediction == 1 else "Không có nguy cơ bệnh tim",
        "model_name": "TabNet",
        "threshold": threshold,
    }


def run_diagnosis(model_choice, resources, input_df):
    """Hàm chính điều phối luồng chẩn đoán.

    Args:
        model_choice: str, "LightGBM" hoặc "TabNet".
        resources: dict từ load_all_resources().
        input_df: pd.DataFrame 1 dòng theo feature_order.

    Returns:
        dict: kết quả dự đoán (xem predict_lightgbm/predict_tabnet)
    """
    if model_choice == "LightGBM":
        result = predict_lightgbm(
            model=resources["lgbm_model"],
            input_df=input_df,
            threshold=resources["lightgbm_threshold"],
        )
    elif model_choice == "TabNet":
        result = predict_tabnet(
            model=resources["tabnet_model"],
            scaler=resources["tabnet_scaler"],
            input_df=input_df,
            threshold=resources["tabnet_threshold"],
        )
    else:
        raise ValueError(f"Model không hợp lệ: {model_choice}")

    return result

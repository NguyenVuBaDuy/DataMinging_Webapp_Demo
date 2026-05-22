# 📋 PHASE 4 - Phân Tích & Danh Sách Công Việc

## ✅ Tài Nguyên Đã Nhận (Kiểm Kê)

Folder `models_and_config/` đầy đủ **6 file** theo đúng bản thiết kế:

| File | Kích thước | Trạng thái |
|------|-----------|------------|
| `best_lightgbm_model.pkl` | 317 KB | ✅ Có |
| `best_tabnet_model.zip` | 1.7 MB | ✅ Có |
| `tabnet_scaler.pkl` | 9.8 KB | ✅ Có |
| `shap_background_data.csv` | 5.4 KB (100 mẫu) | ✅ Có |
| `app_config.json` | 297 B | ✅ Có |
| `xai_insights_test.csv` | 13.7 KB | ✅ Có |

---

## 🚨 PHÁT HIỆN QUAN TRỌNG: Sự Khác Biệt Giữa Thiết Kế & Thực Tế

> [!WARNING]
> Bản thiết kế Phase 4 trong `DATA MINING PROJECT.md` mô tả **13 chỉ số lâm sàng** với các categorical features đa nhãn (cp có 4 giá trị, thal có 3 giá trị, restecg có 3 giá trị...). **Tuy nhiên**, file `app_config.json` thực tế chỉ có **12 features** và đã qua One-Hot Encoding.

### Feature order thực tế trong `app_config.json`:
```json
["age", "trestbps", "chol", "oldpeak", "sex", "fbs", "restecg", "exang", "slope", "ca", "cp_0.0", "thal_0.0"]
```

### Các điểm khác biệt cần lưu ý:

| Vấn đề | Thiết kế UI (doc) | Thực tế (model) |
|--------|-------------------|------------------|
| **Số features** | 13 (age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal) | 12 features |
| **thalach (Nhịp tim tối đa)** | Có trong UI (slider 70-210 bpm) | ❌ **KHÔNG CÓ** trong feature_order → Đã bị loại trong feature selection |
| **cp (Loại đau ngực)** | Selectbox 4 lựa chọn (0,1,2,3) | `cp_0.0` → Binary (0 hoặc 1), chỉ encode 1 giá trị |
| **thal (Chỉ số Thal)** | Selectbox 3 lựa chọn (1,2,3) | `thal_0.0` → Binary (0 hoặc 1), chỉ encode 1 giá trị |

### Giải thích kỹ thuật:

- **`cp_0.0`**: Đây là cột One-Hot cho `cp == 0` (Đau thắt điển hình). Giá trị = 1 nghĩa là "Đau thắt điển hình", giá trị = 0 nghĩa là "Các loại đau ngực khác".
- **`thal_0.0`**: Tương tự, đây là cột One-Hot cho giá trị đầu tiên của `thal`. Cần kiểm tra phase 1 notebook để xác định chính xác mapping.
- **`thalach` bị loại**: Có thể do quá trình feature selection trong phase 2 đã loại bỏ feature này.

> [!IMPORTANT]
> Bạn cần **xác nhận lại với thành viên 1** về ý nghĩa chính xác của `cp_0.0` và `thal_0.0`, và lý do `thalach` bị loại. Điều này ảnh hưởng trực tiếp đến cách thiết kế form nhập liệu.

---

## 📝 DANH SÁCH CÔNG VIỆC CỤ THỂ

### Bước 0: Xác nhận & Chuẩn Bị (⏱️ ~30 phút)
- [ ] **Hỏi thành viên 1** xác nhận:
  - `cp_0.0 = 1` tương ứng với giá trị nào của cp gốc?
  - `thal_0.0 = 1` tương ứng với giá trị nào của thal gốc?
  - Tại sao `thalach` không có trong feature_order?
- [ ] Cài đặt các thư viện cần thiết:
  ```bash
  pip install streamlit joblib shap matplotlib pytorch-tabnet pandas numpy
  ```

### Bước 1: Tạo Cấu Trúc Project Streamlit (⏱️ ~15 phút)
- [ ] Tạo file `app.py` (file chính Streamlit)
- [ ] Copy folder `models_and_config/` vào cùng thư mục

Cấu trúc thư mục mong muốn:
```
app/
├── app.py                    # File chính Streamlit
├── models_and_config/
│   ├── app_config.json
│   ├── best_lightgbm_model.pkl
│   ├── best_tabnet_model.zip
│   ├── tabnet_scaler.pkl
│   ├── shap_background_data.csv
│   └── xai_insights_test.csv
```

### Bước 2: Xây Dựng Phần Load Model & Config (⏱️ ~1 giờ)
- [ ] Đọc `app_config.json` để lấy `feature_order`, `tabnet_threshold`, `lightgbm_threshold`
- [ ] Load LightGBM model bằng `joblib.load()`
- [ ] Load TabNet model bằng `TabNetClassifier().load_model()`
- [ ] Load TabNet scaler bằng `joblib.load()`
- [ ] Load SHAP background data bằng `pd.read_csv()`
- [ ] Cache tất cả bằng `@st.cache_resource` để tăng tốc

### Bước 3: Xây Dựng UI Nhập Liệu (⏱️ ~2 giờ)
- [ ] **Sidebar**: Bộ chọn model (LightGBM / TabNet) + Nút "Tiến hành Chẩn đoán"
- [ ] **Main area**: Form nhập liệu chia 3 cột (`st.columns(3)`)

> [!IMPORTANT]
> Do model thực tế chỉ dùng **12 features** (không có `thalach`), và `cp`/`thal` đã được one-hot encode thành binary, form nhập liệu cần điều chỉnh:

| Feature (model) | Widget đề xuất | Cách hiển thị |
|-----------------|---------------|---------------|
| `age` | `st.slider` | Tuổi: 20-80, mặc định 50 |
| `trestbps` | `st.slider` | Huyết áp: 90-200 mmHg, mặc định 120 |
| `chol` | `st.slider` | Cholesterol: 120-560 mg/dl, mặc định 200 |
| `oldpeak` | `st.slider` | Độ suy giảm ST: 0.0-6.5, bước 0.1 |
| `sex` | `st.selectbox` | Nam/Nữ → 1/0 |
| `fbs` | `st.selectbox` | < 120 mg/dl / > 120 mg/dl → 0/1 |
| `restecg` | `st.selectbox` | Bình thường / Bất thường → 0/1/2 |
| `exang` | `st.selectbox` | Không/Có → 0/1 |
| `slope` | `st.selectbox` | Dốc lên/Phẳng/Dốc xuống → 0/1/2 |
| `ca` | `st.selectbox` | 0/1/2/3/4 |
| `cp_0.0` | `st.selectbox` | "Đau thắt điển hình" / "Loại khác" → 1/0 |
| `thal_0.0` | `st.selectbox` | "Bình thường" / "Bất thường" → 1/0 |

### Bước 4: Xây Dựng Luồng Xử Lý Backend (⏱️ ~2 giờ)
- [ ] Gom input thành DataFrame/array theo đúng `feature_order`
- [ ] **Nếu LightGBM**: Dự đoán trực tiếp → áp ngưỡng 0.50
- [ ] **Nếu TabNet**: Dữ liệu qua `scaler.transform()` → dự đoán → áp ngưỡng 0.24
- [ ] Trả về xác suất + nhãn dự đoán

```python
# Pseudocode luồng xử lý
if model_choice == "LightGBM":
    proba = lgbm_model.predict_proba(input_df)[:, 1]
    prediction = (proba >= 0.50).astype(int)
elif model_choice == "TabNet":
    scaled_input = scaler.transform(input_df)
    proba = tabnet_model.predict_proba(scaled_input)[:, 1]
    prediction = (proba >= 0.24).astype(int)
```

### Bước 5: Xây Dựng Vùng Hiển Thị Kết Quả (⏱️ ~1.5 giờ)
- [ ] **Tầng 1 - Kết quả chẩn đoán**:
  - `st.metric`: "Xác suất nguy cơ: XX.XX%"
  - `st.error` (đỏ) nếu có nguy cơ / `st.success` (xanh) nếu an toàn
- [ ] **Tầng 2 - Biểu đồ giải thích XAI**:
  - LightGBM → SHAP Waterfall Plot
  - TabNet → Attention Heatmap

### Bước 6: SHAP Waterfall cho LightGBM (⏱️ ~2 giờ)
- [ ] Khởi tạo `shap.TreeExplainer(lgbm_model, background_data)`
- [ ] Tính SHAP values cho input hiện tại
- [ ] Vẽ `shap.waterfall_plot()` bằng `st.pyplot()`
- [ ] Hiển thị label tiếng Việt thân thiện trên biểu đồ

### Bước 7: Attention Heatmap cho TabNet (⏱️ ~1.5 giờ)
- [ ] Gọi `tabnet_model.explain(scaled_input)` để lấy Attention Masks
- [ ] Vẽ Heatmap bằng `matplotlib` / `seaborn`
- [ ] Dải màu Vàng → Đỏ đậm thể hiện mức độ chú ý

### Bước 8 (Nâng Cao - Tùy Chọn): Tab Bệnh Nhân Mẫu (⏱️ ~1 giờ)
- [ ] Load `xai_insights_test.csv`
- [ ] Tạo tab "Danh sách bệnh nhân mẫu trong kho lưu trữ"
- [ ] Click chọn bệnh nhân → auto-fill form → tự động chẩn đoán

### Bước 9: Test & Polish (⏱️ ~1 giờ)
- [ ] Test luồng LightGBM end-to-end
- [ ] Test luồng TabNet end-to-end
- [ ] Kiểm tra responsive, CSS styling
- [ ] Chạy `streamlit run app.py` và verify

---

## ⏱️ Ước Tính Tổng Thời Gian

| Giai đoạn | Thời gian |
|-----------|-----------|
| Xác nhận & chuẩn bị | 30 phút |
| Cấu trúc project | 15 phút |
| Load model & config | 1 giờ |
| UI nhập liệu | 2 giờ |
| Backend logic | 2 giờ |
| Hiển thị kết quả | 1.5 giờ |
| SHAP Waterfall | 2 giờ |
| TabNet Heatmap | 1.5 giờ |
| Test & Polish | 1 giờ |
| **Tổng** | **~12 giờ** |

---

## 🔑 Tóm Tắt Các Điểm Cần Hành Động Ngay

1. **🔴 Ưu tiên 1**: Hỏi thành viên 1 xác nhận ý nghĩa `cp_0.0`, `thal_0.0` và lý do loại `thalach`
2. **🟡 Ưu tiên 2**: Cài đặt dependencies và tạo cấu trúc project
3. **🟢 Ưu tiên 3**: Bắt đầu code `app.py` từ phần load model + UI nhập liệu

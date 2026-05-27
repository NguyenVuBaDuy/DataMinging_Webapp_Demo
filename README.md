# ❤️ HƯỚNG DẪN CÀI ĐẶT VÀ VẬN HÀNH
## HỆ THỐNG CHẨN ĐOÁN NGUY CƠ BỆNH TIM (HEART DISEASE DIAGNOSIS SYSTEM)

> [!NOTE]  
> Đây là tài liệu hướng dẫn chuẩn bị môi trường, cài đặt các thư viện phụ thuộc và vận hành ứng dụng **Hệ thống Chẩn đoán Nguy cơ Bệnh tim**. Tài liệu được biên soạn chi tiết để phục vụ cho quá trình kiểm tra, chấm điểm nguồn và chạy thử nghiệm sản phẩm của Giáo viên hướng dẫn.

### 🌐 ĐƯỜNG DẪN TRẢI NGHIỆM TRỰC TUYẾN (DEMO ONLINE)
Để thuận tiện và tiết kiệm thời gian cho Giáo viên, ứng dụng đã được deploy hoàn chỉnh lên môi trường đám mây của Streamlit. Giáo viên có thể truy cập và trải nghiệm trực tiếp hệ thống tại đường link sau:
👉 **[https://demo-data-mining-app.streamlit.app/](https://demo-data-mining-app.streamlit.app/)**

> [!IMPORTANT]  
> **Lưu ý về Server:** Vì hệ thống sử dụng máy chủ đám mây miễn phí (free hosting), nên đôi khi server sẽ tự động rơi vào chế độ ngủ (**Sleep Server**) sau một khoảng thời gian không có người tương tác. 
> *   Nếu gặp giao diện báo ứng dụng đang sleep, Giáo viên chỉ cần nhấn vào nút **"Yes, get this app back up"** (hoặc nút khởi động lại trên màn hình).
> *   Vui lòng chờ khoảng **1 đến 2 phút** để server tự động build lại tài nguyên và khởi chạy lại bình thường.

---


## 📌 1. Giới Thiệu Tổng Quan Hệ Thống

Ứng dụng được xây dựng trên nền tảng **Streamlit** (Python), tích hợp các mô hình Machine Learning & Deep Learning tiên tiến để dự đoán và giải thích nguy cơ mắc bệnh tim mạch của bệnh nhân dựa trên **13 chỉ số lâm sàng**.

### 🌟 Các Tính Năng Nổi Bật:
*   **Chẩn đoán đa mô hình (Multi-model):** Cho phép lựa chọn giữa mô hình **LightGBM** (Machine Learning truyền thống - hiệu năng cao) và **TabNet** (Deep Learning tiên tiến thiết kế riêng cho dữ liệu dạng bảng).
*   **Giải thích mô hình trực quan (Explainable AI - XAI):**
    *   Sử dụng **SHAP Waterfall Plot** cho mô hình LightGBM để thấy rõ mức độ đóng góp (tăng/giảm nguy cơ) của từng chỉ số lâm sàng.
    *   Sử dụng **Attention Heatmap** cho mô hình TabNet giúp trực quan hóa cơ chế chú ý (attention mechanism) của mạng nơ-ron đối với từng đặc trưng của bệnh nhân.
*   **Kho dữ liệu mẫu (Sample Patients Database):** Tích hợp sẵn danh sách bệnh nhân mẫu từ tập kiểm thử để Giáo viên dễ dàng click chọn nhanh, tự động điền form và thực hiện chẩn đoán ngay lập tức mà không cần nhập liệu thủ công.
*   **Giao diện chuyên nghiệp (Medical UI):** Thiết kế tối ưu theo chuẩn Y tế với font chữ `Inter` hiện đại, hệ thống phân màu trực quan (xanh an toàn, đỏ cảnh báo) và hiệu ứng chuyển động mượt mà.

---

## 📂 2. Cấu Trúc Mã Nguồn (Directory Structure)

Thư mục mã nguồn được thiết kế theo mô hình mô-đun hóa (modularized), tách biệt rõ ràng giữa luồng xử lý giao diện (UI) và tính toán logic (Backend):

```text
web_app/
├── app.py                      # File chạy chính (Orchestration & Entry point)
├── requirements.txt            # Danh sách các thư viện Python cần cài đặt
├── phase4_analysis.md          # Phân tích kỹ thuật và nhật ký thiết kế
├── README.md                   # Tài liệu hướng dẫn này (để nộp cho giáo viên)
│
├── core/                       # Package chứa toàn bộ mã nguồn nghiệp vụ của app
│   ├── __init__.py             # Khởi tạo package và export các hàm giao diện/logic
│   ├── config.py               # Chứa cấu hình tĩnh (đường dẫn, nhãn tiếng Việt)
│   ├── model_loader.py         # Module tải mô hình, bộ chuẩn hóa và cấu hình (cached)
│   ├── styles.py               # Chứa mã nguồn Custom CSS tạo giao diện y tế chuyên nghiệp
│   ├── diagnosis.py            # Logic dự đoán lâm sàng & áp ngưỡng quyết định
│   ├── ui_sidebar.py           # Thiết kế bảng điều khiển Sidebar bên trái
│   ├── ui_input_form.py        # Form nhập liệu lâm sàng 3 cột (Responsive)
│   ├── ui_results.py           # Giao diện hiển thị kết quả chẩn đoán và điều hướng XAI
│   ├── ui_samples.py           # Giao diện quản lý & tự động điền từ kho dữ liệu mẫu
│   ├── xai_shap.py             # Xử lý tính toán và vẽ biểu đồ giải thích SHAP
│   └── xai_tabnet.py           # Xử lý tính toán và vẽ Attention Heatmap cho TabNet
│
└── models_and_config_v2/       # Thư mục lưu trữ tài nguyên mô hình đã được huấn luyện
    ├── app_config.json         # Ngưỡng tối ưu của mô hình (LGBM: 0.50, TabNet: 0.24)
    ├── best_lightgbm_model.pkl # File binary mô hình LightGBM
    ├── best_tabnet_model.zip   # Trọng số mô hình Deep Learning TabNet (PyTorch)
    ├── tabnet_scaler.pkl       # Bộ chuẩn hóa StandardScaler cho mô hình TabNet
    ├── shap_background_data.csv# 100 mẫu dữ liệu nền để tính SHAP values
    └── xai_insights_test.csv   # Danh sách cơ sở dữ liệu bệnh nhân mẫu để test nhanh
```

---

## 💻 3. Yêu Cầu Cấu Hình Hệ Thống

Để ứng dụng vận hành trơn tru và mượt mà, cấu hình môi trường khuyến nghị như sau:

### ⚙️ Yêu cầu phần cứng tối thiểu:
*   **CPU:** Intel Core i3 trở lên (hoặc tương đương).
*   **RAM:** Tối thiểu 4GB (Khuyến nghị 8GB để chạy TabNet Deep Learning mượt mà hơn).
*   **Bộ nhớ trống:** ~500MB trống để cài đặt các thư viện Python đi kèm.

### 🐍 Yêu cầu phần mềm:
*   **Hệ điều hành:** Hỗ trợ tốt trên cả **Windows 10/11**, **macOS** và **Linux (Ubuntu/Debian)**.
*   **Phiên bản Python phù hợp:** **Python 3.9 đến Python 3.12** (Khuyến nghị sử dụng Python 3.10 hoặc 3.11 để đảm bảo độ tương thích hoàn hảo của các thư viện như `torch`, `pytorch-tabnet` và `shap`).

> [!WARNING]  
> **Lưu ý đặc biệt về Môi trường & Phiên bản thư viện:**  
> Hệ thống đã được cấu hình và kiểm thử kỹ lưỡng dựa trên đúng các thông số phần mềm và phiên bản thư viện chỉ định. Nếu Giáo viên hoặc người chạy không đảm bảo đúng môi trường yêu cầu (đặc biệt là phiên bản Python phù hợp và cài đặt chuẩn các phiên bản thư viện trong file `requirements.txt`), xin vui lòng tự khắc phục (self-fix) nếu có lỗi không tương thích xảy ra trong quá trình vận hành.

---


## 🚀 4. Hướng Dẫn Cài Đặt Chi Tiết (Từng Bước)

Giáo viên có thể thiết lập môi trường bằng cách làm theo các bước dưới đây. Nên sử dụng **Virtual Environment (Môi trường ảo)** để tránh xung đột với các thư viện hiện có trên máy.

### Bước 1: Mở terminal hoặc Command Prompt (CMD)
Di chuyển thư mục làm việc đến thư mục chứa mã nguồn của đồ án:
```bash
cd "/đường_dẫn_đến_thư_mục/web_app"
```

### Bước 2: Tạo môi trường ảo (Khuyến nghị)
Tạo môi trường ảo có tên là `venv` bằng lệnh sau:
```bash
# Trên Windows/macOS/Linux:
python -m venv venv
```

### Bước 3: Kích hoạt môi trường ảo
Kích hoạt môi trường ảo vừa tạo:
*   **Trên Windows (Command Prompt):**
    ```cmd
    venv\Scripts\activate
    ```
*   **Trên Windows (PowerShell):**
    ```powershell
    .\venv\Scripts\activate
    ```
*   **Trên macOS / Linux:**
    ```bash
    source venv/bin/activate
    ```

*Khi kích hoạt thành công, bạn sẽ thấy chữ `(venv)` xuất hiện ở đầu dòng lệnh.*

### Bước 4: Cập nhật công cụ pip và cài đặt thư viện phụ thuộc
Cài đặt tất cả các thư viện cần thiết được liệt kê trong file `requirements.txt`:
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

> [!IMPORTANT]  
> Các thư viện chính được cài đặt bao gồm:
> *   `streamlit`: Nền tảng phát triển giao diện Web nhanh.
> *   `lightgbm`: Thư viện chạy mô hình Boosting Machine.
> *   `pytorch-tabnet` & `torch`: Thư viện học sâu PyTorch dùng để chạy và suy luận mô hình TabNet.
> *   `shap`: Công cụ toán học giải thích mô hình (XAI).
> *   `scikit-learn` & `joblib`: Xử lý tiền xử lý dữ liệu và load file scaler/model.
> *   `pandas`, `numpy`, `matplotlib`: Thư viện xử lý và hiển thị biểu đồ dữ liệu.

---

## 🏁 5. Hướng Dẫn Chạy Ứng Dụng (Run Application)

### 🌐 Lựa chọn 1: Trải nghiệm Trực tuyến (Không cần cài đặt)
Giáo viên có thể truy cập nhanh bản đã deploy trực tiếp tại:  
👉 **[https://demo-data-mining-app.streamlit.app/](https://demo-data-mining-app.streamlit.app/)** *(Lưu ý: Nếu server đang sleep, vui lòng bấm khởi động lại và đợi 1-2 phút).*

### 💻 Lựa chọn 2: Chạy cục bộ (Local)
Sau khi quá trình cài đặt thư viện hoàn tất, Giáo viên có thể khởi động ứng dụng cục bộ bằng lệnh tiêu chuẩn dưới đây:

```bash
streamlit run app.py
```

> [!TIP]  
> **Lưu ý quan trọng (Dự phòng khi gặp lỗi lệnh không tìm thấy):**  
> Nếu terminal báo lỗi không nhận diện được lệnh `streamlit` (ví dụ: `command not found: streamlit` hoặc `zsh: streamlit: command not found`), Giáo viên hãy chạy ứng dụng bằng cách gọi trực tiếp qua module Python dưới đây:
> ```bash
> python3 -m streamlit run app.py --server.headless true
> ```

### 🌐 Cách truy cập giao diện:
1.  Ngay sau khi gõ lệnh, một tab trình duyệt mới sẽ **tự động mở ra**.
2.  Nếu trình duyệt không tự động mở, Giáo viên có thể copy một trong các địa chỉ sau và dán vào thanh địa chỉ trình duyệt (Chrome, Edge, Firefox, Safari):
    *   Địa chỉ cục bộ: `http://localhost:8501`
    *   Địa chỉ mạng nội bộ: `http://192.168.x.x:8501` (tùy thuộc vào IP máy)

---

## 📖 6. Hướng Dẫn Giáo Viên Chấm Điểm & Trải Nghiệm Ứng Dụng

Để giúp Giáo viên kiểm tra nhanh và đánh giá đầy đủ các tính năng của sản phẩm, chúng tôi khuyến nghị thực hiện quy trình kiểm thử theo các kịch bản sau:

### 🧪 Kịch bản 1: Kiểm thử nhanh bằng Kho bệnh nhân mẫu (Khuyến nghị cho Giáo viên)
1.  Tại giao diện chính, click chọn Tab thứ hai: **`👥 Bệnh Nhân Mẫu (Kho Lưu Trữ)`**.
2.  Tại đây, một bảng danh sách bệnh nhân thực tế với các mã số cụ thể cùng thông tin lâm sàng chuẩn hóa sẽ hiển thị.
3.  Click chọn một bệnh nhân bất kỳ trong danh sách từ menu thả xuống **`👉 Chọn bệnh nhân từ danh sách để kiểm tra:`**.
4.  Ngay lập tức, toàn bộ **13 chỉ số** trong Form nhập liệu lâm sàng ở Tab 1 sẽ được **tự động điền đầy đủ và chính xác**.
5.  Giáo viên chỉ cần nhìn sang **Bảng điều khiển bên trái (Sidebar)**, chọn mô hình mong muốn (ví dụ: `LightGBM` hoặc `TabNet`) và click nút đỏ **`🔍 TIẾN HÀNH CHẨN ĐOÁN`**.
6.  Hệ thống sẽ thực thi thuật toán và xuất kết quả bên dưới ngay lập tức!

### 📝 Kịch bản 2: Tự nhập chỉ số lâm sàng mới
1.  Tại Tab **`📝 Nhập Chỉ Số Bệnh Nhân`**, Giáo viên có thể tùy ý điều chỉnh các thông số:
    *   **Chỉ số liên tục (Cột 1):** Kéo thanh trượt để thay đổi Tuổi, Huyết áp, Cholesterol, Nhịp tim tối đa, Độ chênh ST.
    *   **Chỉ số phân loại (Cột 2 & Cột 3):** Chọn các thông số Giới tính, Đường huyết, Đau thắt ngực khi gắng sức, Điện tâm đồ, Độ dốc ST, Số mạch máu chính bị tắc, Thalassemia,...
2.  Mỗi lần thay đổi chỉ số, Giáo viên có thể mở rộng mục **`📋 Xem tóm tắt dữ liệu đã nhập`** ở phía dưới để kiểm tra xem Vector số thực tế sẽ được đưa vào mô hình như thế nào.
3.  Chọn mô hình phân tích tại Sidebar và bấm nút **`🔍 TIẾN HÀNH CHẨN ĐOÁN`**.

### 📊 Đọc hiểu kết quả hiển thị:
Khi bấm chẩn đoán, ứng dụng sẽ cung cấp cho giáo viên một báo cáo chi tiết bao gồm 3 phần:
1.  **Chỉ số đo lường lâm sàng (Metrics):**
    *   Hiển thị rõ ràng xác suất nguy cơ mắc bệnh tim (ví dụ: `87.42%`).
    *   Màu sắc cảnh báo động: Nhãn **`ĐỎ (NGUY CƠ CAO)`** kèm lời khuyên y tế khẩn cấp nếu vượt ngưỡng quyết định của mô hình, hoặc nhãn **`XANH LÁ (AN TOÀN / NGUY CƠ THẤP)`** kèm khuyến nghị duy trì lối sống lành mạnh.
2.  **Thông tin so sánh ngưỡng:**
    *   Hiển thị trực quan ngưỡng cắt quyết định (Decision Threshold) được tối ưu hóa trong quá trình nghiên cứu (LightGBM: `0.50`, TabNet: `0.24`).
3.  **Vùng giải thích AI (XAI Section):**
    *   **Nếu chọn LightGBM:** Biểu đồ **SHAP Waterfall Plot** sẽ vẽ ra. Các thanh màu đỏ chỉ ra chỉ số nào làm tăng nguy cơ (ví dụ: số mạch máu tắc nghẽn `ca` cao, tuổi cao), các thanh màu xanh chỉ ra chỉ số nào giúp giảm nguy cơ. Giáo viên sẽ hiểu được rõ ràng "Vì sao mô hình lại đưa ra kết luận đó".
    *   **Nếu chọn TabNet:** Biểu đồ **Attention Heatmap** hiển thị mức độ tập trung của mạng nơ-ron sâu. Trục ngang là các chỉ số lâm sàng, màu sắc từ nhạt (không chú ý) đến đỏ đậm (chú ý cực kỳ cao) chỉ ra những chỉ số quyết định trực tiếp tới phán đoán của mạng học sâu.

---

## 🛠️ 7. Xử Lý Sự Cố Thường Gặp (Troubleshooting)

Nếu gặp khó khăn trong quá trình cài đặt hoặc vận hành, dưới đây là cách giải quyết nhanh:

1.  **Lỗi `zsh: command not found: streamlit` hoặc `streamlit: command not found`**
    *   *Nguyên nhân:* Do đường dẫn của thư viện Python (`bin` hoặc `Scripts`) chưa được tự động thêm vào biến môi trường hệ thống (`PATH`).
    *   *Khắc phục:* Giáo viên hãy chạy trực tiếp Streamlit dưới dạng module Python thông qua lệnh sau:
        ```bash
        python3 -m streamlit run app.py --server.headless true
        ```
2.  **Lỗi `ModuleNotFoundError: No module named '...'`**
    *   *Nguyên nhân:* Môi trường ảo chưa được kích hoạt hoặc chưa cài đặt đầy đủ file `requirements.txt`.
    *   *Khắc phục:* Đảm bảo đã chạy lệnh `source venv/bin/activate` (hoặc `venv\Scripts\activate` trên Windows) và chạy lại `pip install -r requirements.txt`.
3.  **Lỗi liên quan đến PyTorch (`torch`) hoặc PyTorch TabNet trên một số máy Windows cũ:**
    *   *Khắc phục:* Nếu xảy ra lỗi liên quan đến PyTorch, vui lòng chạy lệnh cài đặt PyTorch phiên bản CPU để đảm bảo tương thích tốt nhất:
        ```bash
        pip install torch --extra-index-url https://download.pytorch.org/whl/cpu
        ```
4.  **Giao diện bị lệch font hoặc không hiển thị CSS:**
    *   *Khắc phục:* Đảm bảo máy tính có kết nối mạng Internet khi khởi chạy lần đầu tiên để ứng dụng tải font chữ chuyên dụng `Inter` từ máy chủ Google Fonts.

---
☘️ *Kính chúc Quý thầy cô có trải nghiệm tuyệt vời và đánh giá cao sản phẩm đồ án này của chúng em!*

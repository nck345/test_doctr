# HƯỚNG DẪN CHẠY INFERENCE VÀ ĐO METRICS BENCHMARK TRÊN GOOGLE COLAB
### (HỆ THỐNG PHÂN TÁCH 2 PHẦN - TÁI SỬ DỤNG CHO MỌI MODEL & REPO)

---

## I. TỔNG QUAN KIẾN TRÚC HỆ THỐNG

Quy trình đánh giá được phân tách thành **2 phần độc lập hoàn toàn**:

```
[Google Drive: Dữ liệu lớn]               [GitHub: Repo test_doctr]
  ├── data_benchmark (ảnh crop & scan)       ├── inference.py (chạy DocTr)
  ├── model_pretrained (.pth > 100MB)        └── 1_DocTr_Inference_Benchmark.ipynb
  └── predictions (ảnh kết quả)                       │
           │                                          │
           ▼                                          ▼
┌─────────────────────────────────┐        ┌─────────────────────────────────┐
│     PHẦN 1: INFERENCE MODEL     │        │     PHẦN 2: EVALUATION METRICS  │
│  (1_DocTr_Inference_Benchmark)  │───────►│         (metrics.ipynb)         │
│   Chạy GPU T4 nắn phẳng 130 ảnh │        │ Tự chứa 100% code đo MS-SSIM,   │
│   Thời gian: ~20 giây           │        │ CER, ED trên GPU, xuất file CSV │
└─────────────────────────────────┘        └─────────────────────────────────┘
```

* **Phần 1 (Inference - Thuần DocTr):** Nằm trong repo GitHub `test_doctr`, chạy mạng nơ-ron DocTr để nắn phẳng 130 ảnh cong đầu vào, xuất ảnh kết quả vào Google Drive.
* **Phần 2 (Universal Evaluation - Độc lập 100%):** File notebook `metrics.ipynb` lưu độc lập trên Google Drive (`Colab Notebooks/`), tự chứa toàn bộ thuật toán đo kiểm bên trong (RapidOCR tăng tốc GPU/CUDA, Levenshtein, SSIM, cache Ground Truth). Bạn mở file này từ Google Drive lên là đo được cho **DocTr, DewarpNet, DocUNet hoặc bất kỳ mô hình nào**.

---

## II. CẤU TRÚC THƯ MỤC TRÊN GOOGLE DRIVE (CHUẨN HÓA DÙNG CHUNG)

Thư mục gốc `Document_Dewarp_Benchmark` trên **My Drive**:

```text
MyDrive/
├── Colab Notebooks/
│   ├── 1_DocTr_Inference_Benchmark.ipynb
│   └── metrics.ipynb            <- File đo kiểm độc lập, tự chứa 100% code
└── Document_Dewarp_Benchmark/
    ├── data_benchmark/
    │   ├── crop/crop/           <- 130 ảnh cong đầu vào
    │   └── scan/scan/           <- 65 ảnh scan phẳng chuẩn
    ├── model_pretrained/
    │   ├── DocTrP.pth           <- File trọng số GeoTr (103 MB)
    │   └── illtr.pth            <- File trọng số IllTr (51 MB)
    ├── predictions/             <- Nơi chứa ảnh kết quả nắn phẳng của từng mô hình
    │   ├── DocTr/               <- Ảnh nắn phẳng do DocTr sinh ra
    │   ├── DewarpNet/           <- (Tùy chọn) Ảnh nắn phẳng do DewarpNet sinh ra
    │   └── My_Model/            <- (Tùy chọn) Ảnh nắn phẳng do model tự cải tiến sinh ra
    └── eval_reports/            <- Nơi lưu trữ các file CSV báo cáo kết quả đánh giá
```

---

## III. HƯỚNG DẪN THỰC HIỆN TRÊN GOOGLE COLAB

### 1. Chạy sinh ảnh kết quả DocTr (Phần 1):
* Mở [Google Colab](https://colab.research.google.com/) $\rightarrow$ Mở `1_DocTr_Inference_Benchmark.ipynb`.
* Bật GPU: Menu **Runtime** $\rightarrow$ **Change runtime type** $\rightarrow$ Chọn **T4 GPU** $\rightarrow$ **Save**.
* Bấm **Run all** (hoặc `Ctrl + F9`):
  * GPU T4 xử lý nắn phẳng toàn bộ 130 ảnh DocUNet trong **~20 giây**.
  * Toàn bộ ảnh kết quả được tự động lưu vào `Document_Dewarp_Benchmark/predictions/DocTr/`.

### 2. Chạy đo Metrics (Phần 2 - Dùng chung):
* Mở Google Colab $\rightarrow$ Mở file **`metrics.ipynb`** từ thư mục `Colab Notebooks` trên Google Drive.
* Bật GPU: Menu **Runtime** $\rightarrow$ **Change runtime type** $\rightarrow$ Chọn **T4 GPU**.
* Các bước chạy:
  * **Ô 0 (Kiểm tra GPU):** Bấm chạy trong 1 giây để xác nhận GPU và kích hoạt chip GPU ngay lập tức.
  * **Ô 1 (Cài đặt):** Chạy 1 lần duy nhất khi mở máy (nếu mới cài xong thì Restart session 1 lần như hướng dẫn trong ô).
  * **Ô 2 & Ô 3:** Cấu hình tên model và bấm chạy đo kiểm toàn bộ 130 ảnh tự động.
  * Tự động xuất file báo cáo **`eval_reports/DocTr_metrics.csv`** lưu thẳng vào Google Drive.

---

## IV. CÁCH TÁI SỬ DỤNG CHO CÁC MODEL VÀ REPO KHÁC

Khi bạn nghiên cứu một mô hình mới (ví dụ **DewarpNet**, **DocUNet**, hoặc **Model tự cải tiến**):

1. **Sinh ảnh:** Cho mô hình mới chạy ra ảnh phẳng và lưu vào một thư mục trên Google Drive:
   `Document_Dewarp_Benchmark/predictions/<Tên_Model_Mới>/`
2. **Đo metrics:** Mở lại file **`metrics.ipynb`**, tại **Bước 2** chỉ cần đổi 1 dòng:
   ```python
   MODEL_NAME = "Tên_Model_Mới"  # Ví dụ: "DewarpNet" hoặc "DocUNet"
   ```
3. Bấm chạy: Bộ đo sẽ tự động đánh giá và xuất file `eval_reports/<Tên_Model_Mới>_metrics.csv`.

Mọi mô hình đều được đo bằng **cùng một bộ công cụ RapidOCR và Levenshtein chuẩn**, đảm bảo tính công bằng 100% khi so sánh và đưa vào báo cáo/slide thuyết trình.

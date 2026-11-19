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

## II. CẤU TRÚC THƯ MỤC TRÊN GOOGLE DRIVE (CHUẨN HÓA ĐA TẬP BENCHMARK)

Thư mục gốc `Document_Dewarp_Benchmark` trên **My Drive**:

```text
MyDrive/
├── Colab Notebooks/
│   ├── 1_DocTr_Inference_Benchmark.ipynb
│   └── metrics.ipynb                  <- File đo kiểm độc lập, tự chứa 100% code
└── Document_Dewarp_Benchmark/
    ├── benchmarks/                    <- Chứa tất cả các bộ dữ liệu kiểm chuẩn
    │   ├── docunet_paper_subset/      <- 60 ảnh (Setting 1 theo paper 2110.12942v2) để đối chiếu chuẩn
    │   │   ├── crop/                  <- 60 ảnh chụp cong đầu vào
    │   │   └── scan/                  <- 30 ảnh scan chuẩn Ground Truth
    │   ├── docunet_130/               <- Toàn bộ 130 ảnh DocUNet gốc
    │   │   ├── crop/                  <- 130 ảnh cong đầu vào
    │   │   └── scan/                  <- 65 ảnh scan chuẩn
    │   ├── dir300/                    <- Thư mục template cho tập DIR-300 (crop/ & scan/)
    │   └── docreal/                   <- Thư mục template cho tập DocReal (crop/ & scan/)
    ├── predictions/                   <- Chứa ảnh kết quả nắn phẳng của từng mô hình
    │   └── DocTr/
    │       ├── docunet_paper_subset/  <- 60 ảnh dự đoán tương ứng với paper subset
    │       └── docunet_130/           <- 130 ảnh dự đoán toàn tập DocUNet
    ├── gt_cache/                      <- Cache OCR Ground Truth scan vĩnh viễn trên Drive (.json)
    │   ├── docunet_paper_subset_ocr_cache.json
    │   └── docunet_130_ocr_cache.json
    ├── model_pretrained/              <- File trọng số mô hình (.pth)
    │   ├── DocTrP.pth                 <- File trọng số GeoTr (103 MB)
    │   └── illtr.pth                  <- File trọng số IllTr (51 MB)
    └── eval_reports/                  <- Nơi lưu trữ file CSV báo cáo tóm tắt (không lưu chi tiết từng ảnh)
        ├── DocTr_summary.csv          <- Bảng tổng kết các tập benchmark của mô hình DocTr
        └── benchmark_summary.csv      <- Bảng tổng hợp đối chiếu chung tất cả mô hình (Leaderboard)
```

---

## III. HƯỚNG DẪN THỰC HIỆN TRÊN GOOGLE COLAB

### 1. Chạy sinh ảnh kết quả DocTr (Phần 1):
* Mở [Google Colab](https://colab.research.google.com/) $\rightarrow$ Mở `1_DocTr_Inference_Benchmark.ipynb`.
* Bật GPU: Menu **Runtime** $\rightarrow$ **Change runtime type** $\rightarrow$ Chọn **T4 GPU** $\rightarrow$ **Save**.
* Bấm **Run all** (hoặc `Ctrl + F9`):
  * GPU T4 xử lý nắn phẳng các ảnh cong trong **~20 giây**.
  * Toàn bộ ảnh kết quả được tự động lưu vào `Document_Dewarp_Benchmark/predictions/DocTr/`.

### 2. Chạy đo Metrics (Phần 2 - Dùng chung):
* Mở Google Colab $\rightarrow$ Mở file **`metrics.ipynb`** từ thư mục `Colab Notebooks` trên Google Drive.
* Bật GPU: Menu **Runtime** $\rightarrow$ **Change runtime type** $\rightarrow$ Chọn **T4 GPU**.
* Các bước chạy:
  * **Ô 0 (Kiểm tra GPU):** Bấm chạy trong 1 giây để xác nhận GPU và kích hoạt chip GPU ngay lập tức.
  * **Ô 1 (Cài đặt):** Chạy 1 lần duy nhất khi mở máy (nếu mới cài xong thì Restart session 1 lần như hướng dẫn trong ô).
  * **Ô 2 (Cấu hình):** Chọn tập benchmark, tên mô hình và số luồng xử lý:
    ```python
    # Chọn 1 tập, danh sách nhiều tập, hoặc 'all' để chạy toàn bộ
    BENCHMARKS = 'all'  # Hoặc ['docunet_paper_subset', 'dir300', 'docreal'] hoặc 'docunet_paper_subset'
    MODEL_NAME = 'DocTr'
    MS_SSIM_BACKEND = 'torch' # Chạy trên GPU float64, độ lệch < 10^-6
    NUM_WORKERS = 2           # 2 workers song song cho Colab vCPU
    USE_LOCAL_SSD = True      # Đọc ảnh từ SSD cục bộ (/content/) để tránh nghẽn I/O Google Drive
    ```
  * **Ô 3 (Đo kiểm tự động):** Bấm chạy đo kiểm.
    * Tự động quét tuần tự từng tập benchmark trong danh sách.
    * Xuất báo cáo CSV chi tiết cho từng tập vào `eval_reports/<Model>_<Benchmark>_metrics.csv`.
    * Tự động tạo bảng tổng hợp Master Summary gom toàn bộ kết quả để dễ dàng so sánh.
  * **Ô 4:** Hiển thị bảng tổng kết Master Summary bằng Pandas DataFrame.

---

## IV. CÁCH TÁI SỬ DỤNG CHO CÁC MODEL VÀ BENCHMARK KHÁC

### 1. Khi thêm tập Benchmark mới (ví dụ DIR300 hoặc DocReal):
1. Thả ảnh chụp cong vào: `Document_Dewarp_Benchmark/benchmarks/<tên_benchmark>/crop/`
2. Thả ảnh scan chuẩn vào: `Document_Dewarp_Benchmark/benchmarks/<tên_benchmark>/scan/`
3. Trong `metrics.ipynb` (Ô 2), chỉ cần đổi:
   ```python
   BENCHMARK_NAME = "dir300"  # Hoặc "docreal"
   ```

### 2. Khi đo mô hình mới (ví dụ DewarpNet, DocUNet, hoặc Model tự phát triển):
1. Cho mô hình xuất ảnh nắn phẳng vào:
   `Document_Dewarp_Benchmark/predictions/<Tên_Model>/<Tên_Benchmark>/`
2. Trong `metrics.ipynb` (Ô 2), chỉ cần đổi:
   ```python
   MODEL_NAME = "DewarpNet"
   ```
3. Bấm chạy Ô 3: Báo cáo CSV sẽ được tự động lưu vào `eval_reports/<Tên_Model>_<Tên_Benchmark>_metrics.csv`.

Mọi mô hình và bộ benchmark đều được đo bằng **cùng một bộ công cụ RapidOCR và Levenshtein chuẩn**, đảm bảo tính khách quan và nhất quán tuyệt đối.

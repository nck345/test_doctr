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
│     PHẦN 1: INFERENCE MODEL     │        │    PHẦN 2: UNIVERSAL EVALUATION │
│  (1_DocTr_Inference_Benchmark)  │───────►│ (2_Universal_Dewarp_Evaluation) │
│   Chạy GPU T4 nắn phẳng 130 ảnh │        │ Tự chứa 100% code đo MS-SSIM,   │
│   Thời gian: ~20 giây           │        │ CER, ED cho MỌI model, xuất CSV │
└─────────────────────────────────┘        └─────────────────────────────────┘
```

* **Phần 1 (Inference - Thuần DocTr):** Nằm trong repo GitHub `test_doctr`, chạy mạng nơ-ron DocTr để nắn phẳng 130 ảnh cong đầu vào, xuất ảnh kết quả vào Google Drive.
* **Phần 2 (Universal Evaluation - Độc lập 100%):** File notebook tự chứa toàn bộ thuật toán đo kiểm bên trong (RapidOCR, Levenshtein, SSIM), không phụ thuộc vào bất kỳ repo GitHub nào. Bạn mở file này từ Google Drive lên là đo được cho **DocTr, DewarpNet, DocUNet hoặc mô hình bạn tự nghiên cứu**.

---

## II. CẤU TRÚC THƯ MỤC TRÊN GOOGLE DRIVE (CHUẨN HÓA DÙNG CHUNG)

Tạo một thư mục gốc tên là `Document_Dewarp_Benchmark` trên **My Drive** của bạn với cấu trúc như sau:

```text
MyDrive/
└── Document_Dewarp_Benchmark/
    ├── data_benchmark/
    │   ├── crop/crop/           <- 130 ảnh cong đầu vào từ DocTr-Plus/data benchmark/crop/crop/
    │   └── scan/scan/           <- 65 ảnh scan phẳng chuẩn từ DocTr-Plus/data benchmark/scan/scan/
    ├── model_pretrained/
    │   ├── DocTrP.pth           <- File trọng số mô hình GeoTr (103 MB)
    │   └── illtr.pth            <- File trọng số mô hình IllTr (51 MB)
    ├── predictions/             <- Nơi chứa ảnh kết quả nắn phẳng của từng mô hình
    │   ├── DocTr/               <- Ảnh nắn phẳng do DocTr sinh ra
    │   ├── DewarpNet/           <- (Tùy chọn) Ảnh nắn phẳng do DewarpNet sinh ra
    │   └── My_Model/            <- (Tùy chọn) Ảnh nắn phẳng do model tự cải tiến sinh ra
    └── eval_reports/            <- Nơi lưu trữ các file CSV báo cáo kết quả đánh giá
```

> **Tại sao lưu dữ liệu lớn trên Google Drive?**  
> 1. Checkpoint `DocTrP.pth` nặng **103 MB**, vượt quá giới hạn 100 MB của GitHub (nếu push lên GitHub sẽ bị chặn lỗi).  
> 2. Tập ảnh `crop` (286 MB) và `scan` (419 MB) lưu cố định trên Drive thì mỗi lần mở Colab chỉ cần mount Drive là có ngay, không phải upload lại.

---

## III. HƯỚNG DẪN 3 BƯỚC THỰC HIỆN

### BƯỚC 1: Chuẩn bị trên Google Drive (Làm 1 lần duy nhất)

1. Mở [Google Drive](https://drive.google.com/), tạo thư mục `Document_Dewarp_Benchmark`.
2. Tải 2 thư mục sau từ máy tính lên `Document_Dewarp_Benchmark`:
   * Thư mục `model_pretrained` (chứa `DocTrP.pth` và `illtr.pth`).
   * Thư mục `data_benchmark` (chỉ cần chứa `crop/crop` và `scan/scan`).

---

### BƯỚC 2: Đẩy mã nguồn mới lên GitHub

Tại Terminal của repo trên máy tính, chạy lệnh:
```powershell
git add eval_benchmark.py 1_DocTr_Inference_Benchmark.ipynb 2_Universal_Dewarp_Evaluation.ipynb HUONG_DAN_CHAY_BENCHMARK_COLAB.md
git commit -m "Add universal dewarp benchmark and colab notebooks"
git push
```

---

### BƯỚC 3: Mở và chạy trên Google Colab

#### 1. Chạy sinh ảnh kết quả DocTr (Phần 1):
* Mở [Google Colab](https://colab.research.google.com/) $\rightarrow$ Chọn **Upload notebook** $\rightarrow$ Chọn file:
  `1_DocTr_Inference_Benchmark.ipynb`
* Bật GPU: Menu **Runtime** $\rightarrow$ **Change runtime type** $\rightarrow$ Chọn **T4 GPU** $\rightarrow$ **Save**.
* Bấm **Run all** (hoặc `Ctrl + F9`):
  * Colab sẽ kết nối Google Drive.
  * GPU T4 xử lý nắn phẳng toàn bộ 130 ảnh DocUNet trong **~20 giây**.
  * Toàn bộ ảnh kết quả được tự động lưu vào `Document_Dewarp_Benchmark/predictions/DocTr/`.

#### 2. Chạy đo Metrics (Phần 2 - Dùng chung):
* Mở Google Colab $\rightarrow$ Upload file:
  `2_Universal_Dewarp_Evaluation.ipynb`
* Bấm **Run all**:
  * Notebook sẽ tự động đọc ảnh từ thư mục `predictions/DocTr/`.
  * So khớp với ảnh scan chuẩn trong `data_benchmark/scan/scan/`.
  * Đo đồng thời 4 chỉ số: **MS-SSIM (↑), CER (↓), Levenshtein ED (↓), Thời gian (s)**.
  * Tự động xuất file báo cáo **`eval_reports/DocTr_metrics.csv`** lưu thẳng vào Google Drive.

---

## IV. CÁCH TÁI SỬ DỤNG CHO CÁC MODEL VÀ REPO KHÁC

Khi bạn nghiên cứu một mô hình mới (ví dụ **DewarpNet**, **DocUNet**, hoặc **Model tự cải tiến**):

1. **Sinh ảnh:** Cho mô hình mới chạy ra ảnh phẳng và lưu vào một thư mục trên Google Drive:
   `Document_Dewarp_Benchmark/predictions/<Tên_Model_Mới>/`
2. **Đo metrics:** Mở lại file **`2_Universal_Dewarp_Evaluation.ipynb`**, tại **Bước 3** chỉ cần đổi 1 dòng:
   ```python
   MODEL_NAME = "Tên_Model_Mới"  # Ví dụ: "DewarpNet" hoặc "DocUNet"
   ```
3. Bấm chạy: Bộ đo sẽ tự động đánh giá và xuất file `eval_reports/<Tên_Model_Mới>_metrics.csv`.

Mọi mô hình đều được đo bằng **cùng một bộ công cụ RapidOCR và Levenshtein chuẩn**, đảm bảo tính công bằng 100% khi so sánh và đưa vào báo cáo/slide thuyết trình.

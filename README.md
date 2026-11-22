# Dự đoán khách hàng đăng ký tiền gửi kỳ hạn (Bank Marketing)

Đồ án cuối kỳ môn Máy học — Nhóm 14, lớp DCT124C3, Trường Đại học Sài Gòn.
Bài toán phân lớp nhị phân mất cân bằng: dự đoán khách hàng có đăng ký tiền
gửi có kỳ hạn sau chiến dịch telemarketing hay không.

## Dữ liệu

- Nguồn: UCI Machine Learning Repository, Bank Marketing
  (https://archive.ics.uci.edu/dataset/222/bank+marketing),
  file `bank-additional-full.csv`, 41.188 dòng, 20 biến đầu vào + biến mục tiêu
  `y`, phân tách bằng dấu chấm phẩy, giai đoạn 05/2008–11/2010.
- Trích dẫn: Moro, S., Cortez, P., & Rita, P. (2014). *A data-driven approach
  to predict the success of bank telemarketing.* Decision Support Systems, 62, 22–31.
- Sau khi loại 12 dòng trùng: 41.176 dòng. Chia stratified 80/20
  (random_state=42): train 32.940, test 8.236, tỷ lệ yes đều 11,27%.
  Chia theo thời gian (80% đầu / 20% cuối): train yes 6,38%, test yes 30,83%.
- Dữ liệu gốc có trong `data/raw/`; bản đã chia có trong `data/processed/`.
- Lưu ý về repo GitHub: thư mục `Data/` (chữ hoa) là bản đẩy ngày
  04/11/2026, 6 file trong đó trùng hệt nội dung với `data/` (chữ thường);
  `data/` mới là bản code đang dùng. Giữ lại `Data/` chỉ để bảo toàn lịch
  sử (công cụ đẩy không xóa được file trên repo).

## Cách chạy lại toàn bộ thực nghiệm

```bash
pip install -r requirements.txt
python scripts/01_eda.py                  # EDA trên train
python scripts/02_baseline_and_cv.py      # So sánh CV các mô hình
python scripts/03_tuning.py               # Tuning + validation/learning curve
python scripts/04_final_evaluation.py     # Chọn ngưỡng, đánh giá cuối trên test
python scripts/05_analysis.py             # Phân tích lỗi + diễn giải
python scripts/06_extra_experiments.py    # Thí nghiệm bổ sung
streamlit run app/app.py                  # Demo
```

Mọi script dùng chung `SEED = 42`, cùng split, cùng fold (StratifiedKFold 5)
và cùng pipeline trong `src/features.py`. Chạy lại toàn bộ cho ra đúng các số
trong `reports/`.

Chuỗi v2 theo thời gian dùng thêm `scripts/07_time_design.py`,
`08_calibration_threshold.py`, `09_final_time_evaluation.py`,
`10_time_extras.py`, cùng các bản refit80 (`08b_refit80.py`,
`09b_final80.py`) và bản chọn lại (`07d_reselect80.py`,
`08c_refit80_selected.py`, `09c_final80_selected.py`). Nhật ký thực nghiệm
đầy đủ nằm ở `experiments/experiment_log.csv` (66 thí nghiệm + 1 dòng đính chính
= 67 dòng dữ liệu, 68 dòng gồm dòng tiêu đề).

## Triển khai (Docker)

```bash
docker build -t bank-marketing .
docker run -p 8501:8501 bank-marketing   # mở http://localhost:8501
```

Đã chạy thành công trên máy Windows (Acer Nitro 5) ngày 22/11/2026:
`docker build` 11/11 bước thành công, `docker run` khởi động được Streamlit
tại http://localhost:8501 (Uvicorn server started on 0.0.0:8501). Log xem
`../HinhAnh_KiemThu/docker_windows_20261122.png` và
`../NhatKyKiemChung/docker_log_windows_20261122.txt`.

`Dockerfile` đóng gói demo Streamlit (image `python:3.12-slim`, cổng 8501):

```bash
docker build -t bank-marketing .
docker run -p 8501:8501 bank-marketing
```

Trước đó, trong môi trường kiểm chứng (sandbox Linux, 15/11/2026) Docker
daemon không khởi động được (`failed to start daemon: ... failed to register
"bridge" driver: operation not permitted`), nên `docker build` / `docker run`
chưa chạy được ở đó.

## Cấu trúc

```
├── README.md
├── requirements.txt
├── data/raw, data/processed     # dữ liệu gốc + 4 file CSV đã chia + 2 file JSON thông tin chia
├── notebooks/                   # 01_eda, 02_baseline_and_cv, 03_tuning,
│                                # 04_final_evaluation, 05_extra_experiments,
│                                # 06_time_design_tuning, 07_time_final_evaluation
├── src/                         # config, data, features (pipeline), evaluate
├── scripts/                     # 01–11, bản phụ 07b/07c/07d/08b/08c/09b/09c,
│                                # build/make/smoke_test
├── app/app.py                   # demo Streamlit
├── models/                      # model.joblib, model_calibrated.joblib,
│                                # model_metadata.json, threshold.json
├── reports/                     # CSV kết quả, JSON tóm tắt, figures/ (50 hình)
└── experiments/experiment_log.csv
```

## Pipeline tiền xử lý (src/features.py)

- `unknown` giữ làm category riêng, one-hot.
- `pdays = 999` → cờ `contacted_before` + `pdays_clean = 0` khi chưa liên hệ.
- `campaign` cắt trần ở 15 (hằng số cố định).
- `duration` KHÔNG dùng trong kịch bản chính (chỉ biết sau khi gọi — leakage);
  kịch bản có duration chỉ để đối chứng.
- Mất cân bằng xử lý bằng `class_weight='balanced'`, metric chính là PR-AUC,
  ngưỡng chọn bằng dự đoán out-of-fold trên train.

## Kết quả chính

- Mô hình cuối (đã chốt ngày 15/11/2026): **Decision Tree refit80 +
  calibrate sigmoid** (max_depth = 12, min_samples_leaf = 100,
  min_samples_split = 50, criterion = gini, class_weight = balanced;
  calibrate sigmoid, ngưỡng chọn trên OOF train80).
- Đánh giá cuối trên test theo thời gian (20% cuối, 8.236 mẫu, yes-rate
  30,83%): **ROC-AUC 0,6121, PR-AUC 0,3967** (CI 95% bootstrap ROC-AUC
  [0,5985; 0,6256]). Ngưỡng chi phí–lợi ích 0,0225 và ngưỡng F1 0,0228 đều
  suy biến trên test (gọi 100% khách, recall 1,0, TN = 0) nên mô hình cuối
  chỉ còn giá trị xếp hạng: gọi top 20% khách theo điểm mô hình bắt được
  **27,5%** số người đồng ý.
- Con số stratified v1 chỉ là cận trên lạc quan: Random Forest đã tuning
  (400 cây, max_depth = 8, min_samples_leaf = 5, max_features = 0,3) đạt
  test **ROC-AUC 0,8159, PR-AUC 0,4887, F1 0,5255** (ngưỡng F1 0,6145;
  CI95 PR-AUC [0,4544; 0,5247]); top 20% bắt được 66,1% yes. Số Random
  Forest tham chiếu của giảng viên (0,814 / 0,482) được tái lập gần như
  trọn vẹn bằng RF đã tuning; chênh lệch F1/Recall còn lại chủ yếu do
  ngưỡng quyết định khác nhau.
- Kịch bản có duration đạt PR-AUC CV 0,6582 so với 0,4684 khi không dùng
  duration — bằng chứng định lượng về target leakage; chỉ kịch bản không
  duration mới triển khai được.
- Vòng chọn lại mô hình trên CV train80 là kết quả âm: LogReg (C = 0,001)
  được chọn theo quy tắc 1-SE nhưng sau calibrate sigmoid chỉ đạt test
  ROC-AUC 0,5480, tệ hơn phương án Tree.
- Về quy trình đánh giá: tập test theo thời gian được mở nhiều lần trong
  quá trình phân tích và chọn lại (mỗi lần đều ghi log); riêng lần đánh giá
  cuối của mô hình đã chốt được thực hiện trên test và ghi ở dòng
  EXP_T2_FINAL80 trong nhật ký thí nghiệm.

(Chi tiết đầy đủ trong `reports/` và báo cáo `BaoCao_CuoiKy_Nhom14_BankMarketing.docx`.
Mọi con số trong báo cáo đều do các script trên sinh ra.)

## Khai báo sử dụng AI

- Trợ lý AI (Muse) hỗ trợ: viết code khung cho pipeline và demo, kiểm chứng
  lại số liệu, soạn thảo và rà soát báo cáo/slide/poster.
- Nhóm tự quyết định mọi lựa chọn phương pháp (tiền xử lý, metric, ngưỡng,
  mô hình cuối) theo Cẩm nang thực hiện dự án và phản hồi của giảng viên;
  AI không tự ý chọn mô hình hay chỉnh số liệu.
- Mọi con số trong báo cáo đều được chạy lại bằng code trong repo và kiểm
  chứng trước khi đưa vào; nhật ký thực nghiệm ghi đầy đủ cả kết quả âm
  và các lần đính chính.

## Phiên bản

- Python 3.12, scikit-learn 1.9.1 (xem requirements.txt để có danh sách đầy đủ).

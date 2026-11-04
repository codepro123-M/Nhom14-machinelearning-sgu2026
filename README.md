# Bank Marketing - Term Deposit Subscription Prediction

Dự án Machine Learning dự đoán khả năng khách hàng đăng ký tiền gửi có kỳ hạn (Term Deposit) dựa trên dữ liệu tiếp thị trực tiếp của ngân hàng (UCI Bank Marketing Dataset).

## 1. Cấu trúc thư mục dự án
```text
Bank_Marketing_project/
├── Data/
│   ├── raw/                 # Dữ liệu gốc (bank-additional-full.csv)
│   └── processed/           # Dữ liệu sạch đã chia (train.csv, test.csv)
├── docs/                    # Báo cáo (.docx) và Slide (.pptx)
├── experiments/             # Nhật ký thực nghiệm (.csv)
├── figures/                 # Các biểu đồ EDA và kết quả thực nghiệm (.png)
├── notebooks/
│   ├── 01_data_split_and_check.ipynb   # Kiểm tra, lọc trùng, chia Train/Test
│   └── 02_train_eda.ipynb              # Phân tích khám phá dữ liệu trên Train
├── requirements.txt         # Các thư viện phụ thuộc
└── README.md
```

## 2. Dữ liệu và Tiền xử lý
- **Quy mô gốc**: 41,188 mẫu, 20 thuộc tính đầu vào và 1 biến mục tiêu `y`.
- **Lọc trùng**: Đã phát hiện và loại bỏ 12 dòng trùng lặp hoàn toàn -> còn lại 41,176 mẫu.
- **Phân chia dữ liệu**: Phân chia theo phương pháp **Stratified Split** (tỷ lệ 80/20, `random_state=42`), bảo toàn tỷ lệ nhãn thiểu số (11.27% `yes`) trên cả tập Train (32,940 mẫu) và Test (8,236 mẫu).
- **Nguyên tắc chống rò rỉ**: Toàn bộ quá trình EDA và xây dựng Pipeline chỉ được thực hiện trên tập Train.
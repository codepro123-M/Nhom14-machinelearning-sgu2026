# -*- coding: utf-8 -*-
"""Pipeline tiền xử lý + feature engineering (theo Cẩm nang, giai đoạn 3).

Thiết kế:
- Các biến đổi KHÔNG học từ dữ liệu (tạo cờ, cắt trần giá trị cố định)
  đặt trong add_features (FunctionTransformer) -> không gây leakage.
- Các biến đổi HỌC từ dữ liệu (StandardScaler, OneHotEncoder) nằm trong
  ColumnTransformer, chỉ fit trên fold train.
- Mọi thứ gói vào một Pipeline: dữ liệu thô vào, xác suất ra.

Quyết định tiền xử lý (bảng quyết định giai đoạn 2.4):
- unknown: giữ làm category riêng, one-hot (có thể mang thông tin).
- pdays = 999: tạo cờ contacted_before (1 nếu đã từng liên hệ), đặt
  pdays_clean = 0 khi chưa liên hệ, rồi bỏ cột pdays gốc. 999 làm méo
  thang đo của mô hình tuyến tính/khoảng cách; cờ giữ lại thông tin.
- campaign: cắt trần ở 15 (hằng số cố định -> không rò rỉ), giảm ảnh
  hưởng ngoại lệ cho LogReg/KNN; cây không bị ảnh hưởng.
- duration: KHÔNG có trong NUM/CAT nên kịch bản A tự động loại nó.
  Kịch bản B truyền include_duration=True.
- Mất cân bằng: xử lý bằng class_weight='balanced' ở mô hình (không đổi
  dữ liệu), metric chính là PR-AUC, chọn ngưỡng sau.
- Dữ liệu gốc không có NaN ở biến số (đã kiểm chứng), nên không cần imputer;
  'unknown' ở biến phân loại được giữ làm một category.
"""
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

CAT = ["job", "marital", "education", "default", "housing", "loan",
       "contact", "month", "day_of_week", "poutcome"]
NUM = ["age", "campaign", "previous", "pdays_clean", "contacted_before",
       "emp.var.rate", "cons.price.idx", "cons.conf.idx", "euribor3m", "nr.employed"]

# Nhóm biến cho ablation (RQ3)
GROUP_PERSONAL = {"num": ["age"],
                  "cat": ["job", "marital", "education", "default", "housing", "loan"]}
GROUP_CAMPAIGN = {"num": ["campaign", "previous", "pdays_clean", "contacted_before"],
                  "cat": ["contact", "month", "day_of_week", "poutcome"]}
GROUP_MACRO = {"num": ["emp.var.rate", "cons.price.idx", "cons.conf.idx",
                       "euribor3m", "nr.employed"],
               "cat": []}

CAMPAIGN_CAP = 15


def add_features(X):
    """Biến đổi không học từ dữ liệu. Nhận X thô (20 cột), trả X đã tạo đặc trưng."""
    X = X.copy()
    X["contacted_before"] = (X["pdays"] != 999).astype(int)
    X["pdays_clean"] = X["pdays"].where(X["pdays"] != 999, 0)
    X["campaign"] = X["campaign"].clip(upper=CAMPAIGN_CAP)
    return X.drop(columns=["pdays"])


def build_pipeline(clf, num=None, cat=None, include_duration=False):
    """Xây pipeline đầy đủ: add_features -> ColumnTransformer -> clf.

    include_duration=True chỉ dùng cho kịch bản B (đối chứng leakage).
    """
    num = list(NUM if num is None else num)
    cat = list(CAT if cat is None else cat)
    if include_duration:
        num = num + ["duration"]
    prep = ColumnTransformer([
        ("num", StandardScaler(), num),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat),
    ])
    return Pipeline([
        ("fe", FunctionTransformer(add_features)),
        ("prep", prep),
        ("clf", clf),
    ])


def feature_names_out(pipe):
    """Tên đặc trưng sau bước prep (dùng cho hệ số LogReg, cây nông)."""
    return pipe.named_steps["prep"].get_feature_names_out()

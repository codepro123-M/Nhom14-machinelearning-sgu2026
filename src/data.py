# -*- coding: utf-8 -*-
"""Đọc và chia dữ liệu (theo Cẩm nang, giai đoạn 0.5 và 2.1).

- Dữ liệu gốc: bank-additional-full.csv, 41.188 dòng, 20 biến đầu vào + y,
  phân tách bằng ';', sắp xếp theo thời gian (05/2008-11/2010).
- Loại 12 dòng trùng TRƯỚC khi chia: nếu cùng một dòng xuất hiện ở cả
  train và test thì điểm test bị thổi phồng (một dạng rò rỉ).
- Chia stratified 80/20, random_state=42, khóa tập test, chỉ dùng ở bước cuối.
- Chia theo thời gian: 80% đầu (theo thứ tự dòng) làm train, 20% cuối làm test.
"""
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import RAW_PATH, SEED, TEST_SIZE, TARGET


def load_raw(path=RAW_PATH):
    """Đọc dữ liệu gốc (sep=';')."""
    return pd.read_csv(path, sep=";")


def load_clean(path=RAW_PATH):
    """Đọc + loại trùng + mã hóa y thành 0/1. Giữ nguyên thứ tự thời gian."""
    df = load_raw(path)
    df = df.drop_duplicates().reset_index(drop=True)
    df[TARGET] = (df[TARGET] == "yes").astype(int)
    return df


def get_xy(df):
    """Tách X, y từ dataframe đã làm sạch."""
    return df.drop(columns=[TARGET]), df[TARGET]


def stratified_split(df=None, path=RAW_PATH, test_size=TEST_SIZE, seed=SEED):
    """Chia stratified 80/20. Trả về X_train, X_test, y_train, y_test."""
    df = load_clean(path) if df is None else df
    X, y = get_xy(df)
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=seed)


def time_split(df=None, path=RAW_PATH, frac=0.8):
    """Chia theo thời gian: frac đầu làm train, phần còn lại làm test."""
    df = load_clean(path) if df is None else df
    cut = int(frac * len(df))
    train, test = df.iloc[:cut], df.iloc[cut:]
    return get_xy(train)[0], get_xy(test)[0], get_xy(train)[1], get_xy(test)[1]


def time_split_3way(df=None, path=RAW_PATH, frac_train=0.6, frac_val=0.2):
    """Chia theo thời gian thành 3 phần: train / validation / test.

    Thiết kế v2 (sau góp ý 2026-11-14): train = 60% đầu (chọn mô hình,
    tuning bằng TimeSeriesSplit), validation = 20% giữa (chọn mô hình,
    calibrate, chọn ngưỡng), test = 20% cuối (đánh giá trên test; xem mục 4.2.8
    của báo cáo về việc test đã được mở nhiều lần).
    Tỷ lệ yes trôi mạnh qua 3 phần (4,8% -> 11,1% -> 30,8%) nên đây là
    phép đánh giá trung thực hơn stratified split.
    """
    df = load_clean(path) if df is None else df
    n = len(df)
    cut1 = int(frac_train * n)
    cut2 = int((frac_train + frac_val) * n)
    train, val, test = df.iloc[:cut1], df.iloc[cut1:cut2], df.iloc[cut2:]
    X_tr, y_tr = get_xy(train)
    X_va, y_va = get_xy(val)
    X_te, y_te = get_xy(test)
    return X_tr, X_va, X_te, y_tr, y_va, y_te

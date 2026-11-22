# -*- coding: utf-8 -*-
"""Smoke test cho CI: pipeline chạy được end-to-end trên mẫu nhỏ.

Không cần mô hình đã huấn luyện — chỉ kiểm tra dữ liệu thô đi qua
add_features + ColumnTransformer + mô hình và cho ra xác suất hợp lệ.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sklearn.linear_model import LogisticRegression

from src.data import load_clean
from src.features import build_pipeline


def main():
    df = load_clean()
    X, y = df.drop(columns=["y"]).iloc[:2000], df["y"].iloc[:2000]
    pipe = build_pipeline(LogisticRegression(max_iter=1000))
    pipe.fit(X, y)
    proba = pipe.predict_proba(X.iloc[:100])[:, 1]
    assert proba.shape == (100,), proba.shape
    assert np.all((proba >= 0.0) & (proba <= 1.0)), "proba ngoài [0, 1]"
    print(f"smoke test OK: 2000 mẫu train, 100 dự đoán, proba trong [0, 1], "
          f"mean={proba.mean():.4f}")


if __name__ == "__main__":
    main()

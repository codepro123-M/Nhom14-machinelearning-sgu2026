# -*- coding: utf-8 -*-
"""Kiểm thử tự động (B1, thêm sau góp ý 2026-11-14).

Chạy:  pytest -q   (từ thư mục gốc dự án)
Bao phủ tối thiểu theo yêu cầu:
- add_features: pdays=999 -> contacted_before=0, pdays_clean=0;
  campaign bị cắt trần ở 15.
- Pipeline nhận dữ liệu thô, không lỗi với category lạ (handle_unknown).
- Đầu ra predict_proba nằm trong [0, 1] và đúng shape.
- Chống leakage: pipeline mặc định KHÔNG chứa cột duration.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.features import CAMPAIGN_CAP, add_features, build_pipeline


def raw_row(**over):
    row = {"age": 40, "job": "admin.", "marital": "married",
           "education": "university.degree", "default": "no", "housing": "no",
           "loan": "no", "contact": "cellular", "month": "may",
           "day_of_week": "mon", "duration": 200, "campaign": 1, "pdays": 999,
           "previous": 0, "poutcome": "nonexistent", "emp.var.rate": -0.1,
           "cons.price.idx": 93.994, "cons.conf.idx": -36.4, "euribor3m": 1.33,
           "nr.employed": 5191.0}
    row.update(over)
    return pd.DataFrame([row])


def test_pdays_999_creates_flag_and_zero():
    out = add_features(raw_row(pdays=999))
    assert out.loc[0, "contacted_before"] == 0
    assert out.loc[0, "pdays_clean"] == 0
    assert "pdays" not in out.columns


def test_pdays_contacted_keeps_value():
    out = add_features(raw_row(pdays=42))
    assert out.loc[0, "contacted_before"] == 1
    assert out.loc[0, "pdays_clean"] == 42


def test_campaign_clipped_at_15():
    out = add_features(raw_row(campaign=40))
    assert out.loc[0, "campaign"] == CAMPAIGN_CAP == 15
    out2 = add_features(raw_row(campaign=3))
    assert out2.loc[0, "campaign"] == 3


def test_pipeline_handles_unknown_category():
    pipe = build_pipeline(LogisticRegression(max_iter=1000))
    train = pd.concat([raw_row(), raw_row(job="blue-collar", month="jul")], ignore_index=True)
    y = pd.Series([0, 1])
    pipe.fit(train, y)
    # Nghề / tháng chưa từng thấy trong train -> không được lỗi
    new = raw_row(job="nghề-lạ-chưa-từng-thấy", month="jan", education="illiterate")
    proba = pipe.predict_proba(new)
    assert proba.shape == (1, 2)
    assert 0.0 <= proba[0, 1] <= 1.0


def test_predict_proba_range_and_shape():
    pipe = build_pipeline(RandomForestClassifier(n_estimators=10, random_state=42))
    rng = np.random.default_rng(42)
    rows = pd.concat([raw_row(job=j, month=m, campaign=int(c), pdays=int(p))
                      for j, m, c, p in zip(
                          rng.choice(["admin.", "blue-collar", "retired"], 40),
                          rng.choice(["may", "jul", "aug"], 40),
                          rng.integers(1, 5, 40), rng.choice([999, 10, 100], 40))],
                     ignore_index=True)
    y = pd.Series(rng.integers(0, 2, 40))
    pipe.fit(rows, y)
    proba = pipe.predict_proba(rows)
    assert proba.shape == (40, 2)
    assert np.all((proba >= 0.0) & (proba <= 1.0))
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_no_duration_leakage_by_default():
    """Pipeline mặc định không được chứa cột duration (kịch bản A)."""
    pipe = build_pipeline(LogisticRegression(max_iter=1000))
    prep = pipe.named_steps["prep"]
    num_cols = prep.transformers[0][2]
    assert "duration" not in num_cols
    cat_cols = prep.transformers[1][2]
    assert "duration" not in cat_cols
    # Và với dữ liệu có duration, pipeline vẫn chạy (bỏ qua cột đó)
    pipe.fit(pd.concat([raw_row(), raw_row()], ignore_index=True), pd.Series([0, 1]))
    assert pipe.predict_proba(raw_row(duration=9999)).shape == (1, 2)


def test_duration_scenario_is_explicit():
    """Kịch bản B (có duration) chỉ bật khi truyền include_duration=True."""
    pipe_b = build_pipeline(LogisticRegression(max_iter=1000), include_duration=True)
    num_cols = pipe_b.named_steps["prep"].transformers[0][2]
    assert "duration" in num_cols

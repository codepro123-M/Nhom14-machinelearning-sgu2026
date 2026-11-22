# -*- coding: utf-8 -*-
"""Kiểm thử tự động cho calibrate và ngưỡng (bổ sung sau góp ý Lần 2, 15/11/2026).

Chạy:  pytest -q   (từ thư mục gốc dự án)
Bao phủ:
- Ngưỡng trong models/threshold.json phải khớp models/model_metadata.json
  (ngưỡng chi phí–lợi ích, ngưỡng F1, phương pháp calibrate).
- Phát hiện ngưỡng suy biến: ngưỡng chi phí–lợi ích (0,0225) thấp hơn xác
  suất đã calibrate nhỏ nhất trên test theo thời gian (0,0624) nên mô hình
  gọi 100% khách (recall 1,0, TN=0) — test này ghi lại trạng thái suy biến
  đã biết; nếu ngưỡng hoặc mô hình thay đổi, test sẽ báo để cập nhật báo cáo.
- Xác suất sau calibrate nằm trong [0, 1] trên toàn bộ test.
"""
import json
import sys
from pathlib import Path

import joblib
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _load_json(name):
    return json.loads((ROOT / "models" / name).read_text(encoding="utf-8"))


def test_threshold_files_match_metadata():
    """Ngưỡng + phương pháp calibrate trong threshold.json khớp metadata."""
    th = _load_json("threshold.json")
    md = _load_json("model_metadata.json")
    # metadata ghi ngưỡng CB làm tròn 0,0225; file threshold ghi đầy đủ
    assert th["threshold_cost_benefit"] == pytest.approx(
        md["threshold_cost_benefit"], abs=1e-3)
    assert th["threshold_f1"] == pytest.approx(md["threshold_f1_oof"], abs=1e-9)
    assert th["calibration_method"] == md["calibration_method"] == "sigmoid"
    assert th["threshold_source"].startswith("OOF train80")


def test_degenerate_threshold_is_detected():
    """Ngưỡng suy biến: ngưỡng CB thấp hơn xác suất calibrate nhỏ nhất trên test."""
    th = _load_json("threshold.json")
    md = _load_json("model_metadata.json")
    model = joblib.load(ROOT / "models" / "model_calibrated.joblib")
    test = pd.read_csv(ROOT / "data" / "processed" / "time_test.csv", sep=";")
    proba = model.predict_proba(test.drop(columns=["y"]))[:, 1]
    assert ((proba >= 0.0) & (proba <= 1.0)).all()
    min_proba = float(proba.min())
    # Số đã ghi trong báo cáo (mục 5.3.6): min_proba_test = 0,0624
    assert min_proba == pytest.approx(0.0624, abs=5e-3)
    # Ngưỡng CB thấp hơn xác suất nhỏ nhất -> mọi khách đều vượt ngưỡng
    assert th["threshold_cost_benefit"] < min_proba
    assert th["threshold_f1"] < min_proba
    # Metric test đã ghi trong metadata phải nhất quán với suy biến đó
    cb = md["test_metrics_cb_threshold"]
    assert cb["recall"] == 1.0
    assert cb["tn"] == 0
    assert cb["fn"] == 0

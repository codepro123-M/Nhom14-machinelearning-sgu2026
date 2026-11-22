# -*- coding: utf-8 -*-
"""Giai đoạn 7b (v2): Calibrate xác suất + chọn ngưỡng theo chi phí-lợi ích.

Thiết kế theo thời gian (A1/A2, sau góp ý 2026-11-14):
- Mô hình cuối lấy từ scripts/07_time_design.py (time_selection.json).
- Chọn phương pháp calibrate (isotonic vs Platt/sigmoid) bằng Brier trên
  dự đoán out-of-fold của train60 (CalibratedClassifierCV cv=3), rồi
  calibrate mô hình đã fit trên validation (FrozenEstimator, không chạm test).
- Ngưỡng chính chọn theo chi phí-lợi ích (C=1, R=15 — cặp tham chiếu của
  nhóm, không phải chi phí thật) trên xác suất ĐÃ CALIBRATE của validation.
  Ngưỡng tối ưu F1 và ngưỡng giữ recall>=0,7 vẫn tính để tham chiếu.
- Lưu models/model.joblib (mô hình gốc, chưa calibrate),
  models/model_calibrated.joblib (mô hình dùng trong app) và
  models/threshold.json (định dạng mới, có threshold_cost_benefit).
"""
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, precision_recall_curve
from sklearn.model_selection import cross_val_predict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import MODELS_DIR, REPORTS_DIR, SEED
from src.data import load_clean, time_split_3way
from src.evaluate import _single_thread
from src.features import build_pipeline

COST_C, COST_R = 1.0, 15.0  # cặp tham chiếu (giả định của nhóm, không phải chi phí thật)

CLASSES = {"LogReg": LogisticRegression, "Tree": DecisionTreeClassifier,
           "KNN": KNeighborsClassifier, "HGB": HistGradientBoostingClassifier,
           "LightGBM": LGBMClassifier, "RF": RandomForestClassifier}
DEFAULT_KWARGS = {
    "LogReg": {"max_iter": 3000, "class_weight": "balanced", "random_state": SEED},
    "Tree": {"class_weight": "balanced", "random_state": SEED},
    "KNN": {"n_jobs": 1},
    "HGB": {"class_weight": "balanced", "random_state": SEED},
    "LightGBM": {"class_weight": "balanced", "verbosity": -1, "n_jobs": 1, "random_state": SEED},
    "RF": {"class_weight": "balanced_subsample", "n_jobs": 1, "random_state": SEED},
}


def estimator_from_best(name, best_params):
    kwargs = {k.replace("clf__", ""): v for k, v in best_params[name].items()}
    return CLASSES[name](**{**DEFAULT_KWARGS[name], **kwargs})


def fmt(x, nd=4):
    return f"{x:.{nd}f}".replace(".", ",")


def cost_benefit_threshold(y_true, proba, C=COST_C, R=COST_R):
    """Ngưỡng tối đa hóa lợi nhuận kỳ vọng: xếp theo proba, gọi từ trên xuống.
    Trả về (ngưỡng, top_pct, capture_pct, profit_max)."""
    order = np.argsort(-np.asarray(proba))
    y_sorted = np.asarray(y_true)[order]
    p_sorted = np.asarray(proba)[order]
    profit = R * np.cumsum(y_sorted) - C * (np.arange(len(y_sorted)) + 1)
    k = int(np.argmax(profit))
    return float(p_sorted[k]), float((k + 1) / len(y_sorted)), \
        float(y_sorted[:k + 1].sum() / y_sorted.sum()), float(profit[k])


def main():
    df = load_clean()
    X_tr, X_va, X_te, y_tr, y_va, y_te = time_split_3way(df)
    sel = json.loads((REPORTS_DIR / "time_selection.json").read_text(encoding="utf-8"))
    best_params = json.loads((REPORTS_DIR / "best_params_time.json").read_text(encoding="utf-8"))
    final_name = sel["final_model"]
    print("Mô hình cuối (thiết kế theo thời gian):", final_name, "|", sel["selection_reason"])

    base = _single_thread(build_pipeline(estimator_from_best(final_name, best_params)))
    base.fit(X_tr, y_tr)

    # --- Chọn phương pháp calibrate bằng Brier OOF trên train60 ---
    method_scores = {}
    for method in ["isotonic", "sigmoid"]:
        cal_cv = CalibratedClassifierCV(_single_thread(build_pipeline(
            estimator_from_best(final_name, best_params))), method=method, cv=3)
        oof = cross_val_predict(cal_cv, X_tr, y_tr, method="predict_proba", n_jobs=-1)[:, 1]
        method_scores[method] = float(brier_score_loss(y_tr, oof))
        print(f"  calibrate {method}: Brier OOF (train60) = {method_scores[method]:.4f}")
    method = min(method_scores, key=method_scores.get)
    print("Phương pháp calibrate được chọn:", method)

    # --- Calibrate prefit trên validation (sklearn 1.9 đã bỏ cv="prefit";
    # cách thay thế chính thức là bọc mô hình đã fit trong FrozenEstimator) ---
    cal = CalibratedClassifierCV(FrozenEstimator(base), method=method)
    cal.fit(X_va, y_va)
    p_va_raw = base.predict_proba(X_va)[:, 1]
    p_va_cal = cal.predict_proba(X_va)[:, 1]
    brier = {"brier_val_raw": round(float(brier_score_loss(y_va, p_va_raw)), 4),
             "brier_val_calibrated": round(float(brier_score_loss(y_va, p_va_cal)), 4),
             "brier_oof_train60": {m: round(s, 4) for m, s in method_scores.items()},
             "method": method}
    print("Brier trên validation (lưu ý: calibrate fit trên validation nên số này "
          f"lạc quan): gốc {brier['brier_val_raw']:.4f} -> đã calibrate "
          f"{brier['brier_val_calibrated']:.4f}. So sánh trung thực ở test trong script 09.")

    # --- Ngưỡng trên validation (xác suất đã calibrate) ---
    thr_cb, top_pct, cap_pct, profit = cost_benefit_threshold(y_va, p_va_cal)
    prec, rec, thr_arr = precision_recall_curve(y_va, p_va_cal)
    prec, rec = prec[:-1], rec[:-1]
    f1 = 2 * prec * rec / (prec + rec + 1e-12)
    thr_f1 = float(thr_arr[np.argmax(f1)])
    ok = rec >= 0.7
    thr_r70 = float(thr_arr[ok][np.argmax(prec[ok])]) if ok.any() else float(thr_arr[-1])
    print(f"Ngưỡng chi phí-lợi ích (C={COST_C}, R={COST_R}): {thr_cb:.4f} "
          f"(gọi top {top_pct * 100:.1f}%, bắt {cap_pct * 100:.1f}% yes, "
          f"lợi nhuận {profit:.0f} đơn vị trên validation)")
    print(f"Ngưỡng tối ưu F1 (validation): {thr_f1:.4f} | ngưỡng recall>=0,7: {thr_r70:.4f}")

    # --- Lưu mô hình + ngưỡng + metadata calibrate ---
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(base, MODELS_DIR / "model.joblib")
    joblib.dump(cal, MODELS_DIR / "model_calibrated.joblib")
    (MODELS_DIR / "threshold.json").write_text(
        json.dumps({"threshold_cost_benefit": thr_cb, "threshold_f1": thr_f1,
                    "threshold_recall70": thr_r70, "calibration_method": method,
                    "cost_benefit_C": COST_C, "cost_benefit_R": COST_R},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    (REPORTS_DIR / "calibration_comparison.json").write_text(
        json.dumps({"final_model": final_name, "selection_reason": sel["selection_reason"],
                    "best_params": best_params.get(final_name, {}), **brier,
                    "threshold_cost_benefit": round(thr_cb, 4),
                    "threshold_f1_val": round(thr_f1, 4),
                    "threshold_recall70_val": round(thr_r70, 4),
                    "val_top_pct_at_cb_threshold": round(top_pct, 4),
                    "val_capture_pct_at_cb_threshold": round(cap_pct, 4),
                    "val_profit_at_cb_threshold": round(profit, 1)},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    print("Đã lưu models/model.joblib + models/model_calibrated.joblib + models/threshold.json")


if __name__ == "__main__":
    main()

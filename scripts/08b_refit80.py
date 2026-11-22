# -*- coding: utf-8 -*-
"""Giai đoạn 7b' (v2 refit): Refit mô hình cuối trên train+val (train80).

Đính chính thiết kế (2026-11-14): scripts/08 chỉ fit mô hình cuối trên
train60 (24.705 dòng) rồi calibrate trên validation; mô hình chưa bao giờ
thấy cửa sổ validation 60-80% (8.235 dòng, nơi mar/apr xuất hiện lần đầu),
nên headline test 0,4562 phần lớn là hệ quả thiết kế. Quy trình chuẩn:
chọn mô hình/ngưỡng trên validation, rồi REFIT cùng siêu tham số trên
train+val trước khi đánh giá test một lần (script 09b).

Script này (không sửa 08/09 gốc):
- Lưu trữ bản train60 hiện tại thành model_train60.joblib,
  model_calibrated_train60.joblib, threshold_train60.json,
  model_metadata_train60.json TRƯỚC KHI ghi đè.
- Fit Tree (siêu tham số đã chọn, class_weight="balanced" như 07/08) trên
  train80 = train60 + val (32.940 dòng).
- Chọn phương pháp calibrate (isotonic vs sigmoid) bằng Brier trên dự đoán
  out-of-fold của train80 (CalibratedClassifierCV cv=3, cross_val_predict
  cv=3). Calibrate mô hình cuối bằng CalibratedClassifierCV(cv=3) fit trên
  train80 (không còn validation riêng để calibrate prefit).
- Chọn ngưỡng CHỈ trên dự đoán OOF đã calibrate của train80: ngưỡng tối ưu
  F1, ngưỡng giữ recall>=0,7, ngưỡng chi phí-lợi ích (C=1, R=15).
- Test KHÔNG được chạm trong script này.
"""
import hashlib
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import brier_score_loss, precision_recall_curve
from sklearn.model_selection import cross_val_predict
from sklearn.tree import DecisionTreeClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import MODELS_DIR, RAW_PATH, REPORTS_DIR, SEED
from src.data import load_clean, time_split_3way
from src.evaluate import _single_thread
from src.features import build_pipeline

COST_C, COST_R = 1.0, 15.0  # cặp tham chiếu (giả định của nhóm, không phải chi phí thật)

CLASSES = {"Tree": DecisionTreeClassifier}
DEFAULT_KWARGS = {"Tree": {"class_weight": "balanced", "random_state": SEED}}


def estimator_from_best(name, best_params):
    kwargs = {k.replace("clf__", ""): v for k, v in best_params[name].items()}
    return CLASSES[name](**{**DEFAULT_KWARGS[name], **kwargs})


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
    X80 = pd.concat([X_tr, X_va], ignore_index=True)
    y80 = pd.concat([y_tr, y_va], ignore_index=True)
    print(f"train60={len(X_tr)} (yes {y_tr.mean():.4f}) | val={len(X_va)} "
          f"(yes {y_va.mean():.4f}) | train80={len(X80)} (yes {y80.mean():.4f}) "
          f"| test={len(X_te)} (yes {y_te.mean():.4f})")

    sel = json.loads((REPORTS_DIR / "time_selection.json").read_text(encoding="utf-8"))
    best_params = json.loads((REPORTS_DIR / "best_params_time.json").read_text(encoding="utf-8"))
    final_name = sel["final_model"]
    assert final_name == "Tree", f"Mô hình cuối không phải Tree: {final_name}"
    print("Mô hình cuối (đã chọn trên validation):", final_name, "|",
          sel["selection_reason"])
    print("Siêu tham số refit:", best_params[final_name])

    # --- Lưu trữ bản train60 TRƯỚC KHI ghi đè ---
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    archive = {"model.joblib": "model_train60.joblib",
               "model_calibrated.joblib": "model_calibrated_train60.joblib",
               "threshold.json": "threshold_train60.json",
               "model_metadata.json": "model_metadata_train60.json"}
    for src_name, dst_name in archive.items():
        src, dst = MODELS_DIR / src_name, MODELS_DIR / dst_name
        if src.exists():
            shutil.copy2(src, dst)
            print(f"  lưu trữ: {src_name} -> {dst_name}")
        else:
            print(f"  CẢNH BÁO: không thấy {src_name} để lưu trữ")

    # --- Fit mô hình gốc trên train80 ---
    base = _single_thread(build_pipeline(estimator_from_best(final_name, best_params)))
    base.fit(X80, y80)
    print("Đã fit Tree trên train80:",
          base.named_steps["clf"])

    # --- Chọn phương pháp calibrate bằng Brier OOF trên train80 ---
    # cross_val_predict(CalibratedClassifierCV(cv=3), cv=3): mỗi fold ngoài fit
    # một CalibratedClassifierCV 3-fold bên trong -> xác suất OOF đã calibrate.
    method_scores, oof_by_method = {}, {}
    for method in ["isotonic", "sigmoid"]:
        cal_cv = CalibratedClassifierCV(
            _single_thread(build_pipeline(estimator_from_best(final_name, best_params))),
            method=method, cv=3)
        oof = cross_val_predict(cal_cv, X80, y80, method="predict_proba", cv=3)[:, 1]
        oof_by_method[method] = oof
        method_scores[method] = float(brier_score_loss(y80, oof))
        print(f"  calibrate {method}: Brier OOF (train80) = {method_scores[method]:.4f}")
    method = min(method_scores, key=method_scores.get)
    print("Phương pháp calibrate được chọn:", method)
    from sklearn.metrics import average_precision_score, roc_auc_score
    oof_roc = float(roc_auc_score(y80, oof_by_method[method]))
    oof_pr = float(average_precision_score(y80, oof_by_method[method]))
    print(f"OOF train80 ({method}): ROC-AUC {oof_roc:.4f} | PR-AUC {oof_pr:.4f}")

    # --- Calibrate mô hình cuối bằng CV trên train80 ---
    # Không còn validation riêng (đã gộp vào huấn luyện) nên không dùng prefit/
    # FrozenEstimator như 08; CalibratedClassifierCV(cv=3) fit trực tiếp.
    cal = CalibratedClassifierCV(
        _single_thread(build_pipeline(estimator_from_best(final_name, best_params))),
        method=method, cv=3)
    cal.fit(X80, y80)
    print("Đã calibrate (cv=3) trên train80.")

    # --- Ngưỡng CHỈ trên dự đoán OOF đã calibrate của train80 ---
    p_oof = oof_by_method[method]
    thr_cb, top_pct, cap_pct, profit = cost_benefit_threshold(y80, p_oof)
    prec, rec, thr_arr = precision_recall_curve(y80, p_oof)
    prec, rec = prec[:-1], rec[:-1]
    f1 = 2 * prec * rec / (prec + rec + 1e-12)
    thr_f1 = float(thr_arr[np.argmax(f1)])
    ok = rec >= 0.7
    thr_r70 = float(thr_arr[ok][np.argmax(prec[ok])]) if ok.any() else float(thr_arr[-1])
    print(f"Ngưỡng chi phí-lợi ích (C={COST_C}, R={COST_R}) trên OOF train80: "
          f"{thr_cb:.4f} (gọi top {top_pct * 100:.1f}%, bắt {cap_pct * 100:.1f}% yes, "
          f"lợi nhuận {profit:.0f} đơn vị)")
    print(f"Ngưỡng tối ưu F1 (OOF train80): {thr_f1:.4f} | ngưỡng recall>=0,7: {thr_r70:.4f}")

    # --- Lưu mô hình + ngưỡng + so sánh calibrate (ghi đè bản train60) ---
    joblib.dump(base, MODELS_DIR / "model.joblib")
    joblib.dump(cal, MODELS_DIR / "model_calibrated.joblib")
    (MODELS_DIR / "threshold.json").write_text(
        json.dumps({"threshold_cost_benefit": thr_cb, "threshold_f1": thr_f1,
                    "threshold_recall70": thr_r70, "calibration_method": method,
                    "cost_benefit_C": COST_C, "cost_benefit_R": COST_R,
                    "train_window": "train80 (train60+val, 32940 rows)",
                    "threshold_source": "OOF train80 (cross_val_predict cv=3, calibrated)"},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    (REPORTS_DIR / "calibration_comparison80.json").write_text(
        json.dumps({"final_model": final_name, "selection_reason": sel["selection_reason"],
                    "best_params": best_params.get(final_name, {}),
                    "train_window": "train80 (train60+val)",
                    "n_train80": int(len(X80)), "yes_rate_train80": round(float(y80.mean()), 4),
                    "brier_oof_train80": {m: round(s, 4) for m, s in method_scores.items()},
                    "method": method,
                    "roc_auc_oof_train80": round(oof_roc, 4),
                    "pr_auc_oof_train80": round(oof_pr, 4),
                    "threshold_cost_benefit": round(thr_cb, 4),
                    "threshold_f1_oof": round(thr_f1, 4),
                    "threshold_recall70_oof": round(thr_r70, 4),
                    "oof_top_pct_at_cb_threshold": round(top_pct, 4),
                    "oof_capture_pct_at_cb_threshold": round(cap_pct, 4),
                    "oof_profit_at_cb_threshold": round(profit, 1)},
                   ensure_ascii=False, indent=2), encoding="utf-8")

    # --- Metadata (phần huấn luyện; 09b sẽ ghi đè đầy đủ kèm số test) ---
    sha = hashlib.sha256(RAW_PATH.read_bytes()).hexdigest()[:16]
    meta = {"model": final_name, "best_params": best_params.get(final_name, {}),
            "calibration_method": method,
            "threshold_cost_benefit": round(thr_cb, 4),
            "threshold_f1_oof": round(thr_f1, 4),
            "trained_at_utc": datetime.now(timezone.utc).isoformat(),
            "scikit_learn": sklearn.__version__,
            "data_file": RAW_PATH.name, "data_sha256_prefix": sha,
            "n_train": int(len(X80)), "n_val": int(len(X_va)), "n_test": int(len(X_te)),
            "train_window": "train80 (train60+val, refit sau khi chọn trên validation)",
            "test_metrics_cb_threshold": "pending - run scripts/09b_final80.py",
            "note": "Refit trên train+val sau khi chọn mô hình trên validation; "
                    "calibrate + ngưỡng chọn trên OOF của train80 (không còn "
                    "validation riêng). Đính chính 19/11/2026: test đã được đánh giá nhiều lần trong dự án (xem mục 4.2.8 của báo cáo)."}
    (MODELS_DIR / "model_metadata.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Đã lưu models/model.joblib + models/model_calibrated.joblib + "
          "models/threshold.json + reports/calibration_comparison80.json")


if __name__ == "__main__":
    main()

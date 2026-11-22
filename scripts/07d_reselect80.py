# -*- coding: utf-8 -*-
"""Giai đoạn 6c (v2, chọn lại sau đính chính lần 2 — 2026-11-14): Chọn lại mô hình
cuối bằng CV trên train80.

Bối cảnh: lựa chọn mô hình v2 ban đầu (Tree, PR-AUC validation 0,1239, quy
tắc 1-SE) được làm với các mô hình chỉ fit trên train60 — cùng cửa sổ đã
gây ra lỗi thiết kế đầu tiên (mô hình cuối chưa từng thấy validation
60-80%). Sau refit trên train80, Tree balanced chỉ đạt ROC-AUC 0,6121 trên
test, trong khi LogReg balanced 0,7015 và RF tham số v1 0,7079 trên cùng
train80/test. Vì vậy chọn lại mô hình bằng CV trên train80.

Script này (không sửa 07/08/08b/09/09b):
- KHÔNG tuning lại siêu tham số: tái dùng best_params đã tune trong
  reports/best_params_time.json cho cả 6 mô hình, với đúng estimator mặc
  định (class_weight...) như scripts/07_time_design.py.
- Mỗi mô hình: TimeSeriesSplit(3) CV trên train80 (32.940 dòng), metric
  chính PR-AUC (average_precision), kèm ROC-AUC. Lưu bảng đầy đủ vào
  reports/reselection80_cv.csv.
- Quy tắc 1-SE đúng như 07: mô hình tốt nhất theo PR-AUC CV trung bình;
  duyệt các mô hình đơn giản hơn theo thứ tự SIMPLICITY của 07; mô hình
  đơn giản đầu tiên có khoảng cách < 1 độ lệch chuẩn CV của mô hình tốt
  nhất (std của 3 fold, ddof=0 như 07) thì được chọn. Lưu lựa chọn vào
  reports/reselection80.json.
- Test KHÔNG được chạm trong script này (không dùng để chọn mô hình,
  ngưỡng hay phương pháp calibrate).
"""
import csv
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import REPORTS_DIR, SEED
from src.data import load_clean, time_split_3way
from src.evaluate import _single_thread
from src.features import build_pipeline

# Thứ tự đơn giản -> phức tạp (cho quy tắc 1-SE), Y HỆT 07.
SIMPLICITY = {"LogReg": 0, "Tree": 1, "KNN": 2, "HGB": 3, "LightGBM": 4, "RF": 5}

# Estimator mặc định Y HỆT 07 (kể cả class_weight).
BASE = {
    "LogReg": LogisticRegression(max_iter=3000, class_weight="balanced", random_state=SEED),
    "Tree": DecisionTreeClassifier(class_weight="balanced", random_state=SEED),
    "KNN": KNeighborsClassifier(n_jobs=1),
    "HGB": HistGradientBoostingClassifier(class_weight="balanced", random_state=SEED),
    "LightGBM": LGBMClassifier(class_weight="balanced", verbosity=-1, n_jobs=1,
                               random_state=SEED),
    "RF": RandomForestClassifier(class_weight="balanced_subsample", n_jobs=1,
                                 random_state=SEED),
}


def estimator_from_best(name, best_params):
    import copy
    est = copy.deepcopy(BASE[name])
    est.set_params(**{k.replace("clf__", ""): v for k, v in best_params[name].items()})
    return est


def main():
    df = load_clean()
    X_tr, X_va, X_te, y_tr, y_va, y_te = time_split_3way(df)
    X80 = pd.concat([X_tr, X_va], ignore_index=True)
    y80 = pd.concat([y_tr, y_va], ignore_index=True)
    print(f"train80={len(X80)} (yes {y80.mean():.4f}) | test={len(X_te)} "
          f"(yes {y_te.mean():.4f}) — test KHÔNG được chạm trong script này")

    best_params = json.loads((REPORTS_DIR / "best_params_time.json").read_text(encoding="utf-8"))
    tscv = TimeSeriesSplit(n_splits=3)
    rows = {}
    for name in ["LogReg", "Tree", "KNN", "HGB", "LightGBM", "RF"]:
        est = _single_thread(build_pipeline(estimator_from_best(name, best_params)))
        print(f">>> CV {name} (TimeSeriesSplit(3) trên train80, siêu tham số đã tune) ...",
              flush=True)
        cv = cross_validate(est, X80, y80, cv=tscv,
                            scoring={"pr_auc": "average_precision", "roc_auc": "roc_auc"},
                            n_jobs=1, return_train_score=False)
        pr = cv["test_pr_auc"]
        roc = cv["test_roc_auc"]
        rows[name] = {
            "pr_auc_cv_mean": round(float(pr.mean()), 4),
            "pr_auc_cv_std": round(float(np.std(pr)), 4),  # ddof=0 như 07
            "pr_auc_fold0": round(float(pr[0]), 4),
            "pr_auc_fold1": round(float(pr[1]), 4),
            "pr_auc_fold2": round(float(pr[2]), 4),
            "roc_auc_cv_mean": round(float(roc.mean()), 4),
            "roc_auc_cv_std": round(float(np.std(roc)), 4),
        }
        print(f"    {name}: PR-AUC CV = {pr.mean():.4f} ± {np.std(pr):.4f} "
              f"(folds {pr[0]:.4f} / {pr[1]:.4f} / {pr[2]:.4f}) | "
              f"ROC-AUC CV = {roc.mean():.4f}", flush=True)

    compare = pd.DataFrame(rows).T
    compare.to_csv(REPORTS_DIR / "reselection80_cv.csv")
    print("\n=== So sánh CV trên train80 (TimeSeriesSplit(3)) ===")
    print(compare.to_string())

    # Quy tắc 1-SE trên CV train80 (đúng như 07: so PR-AUC trung bình, ngưỡng
    # là std của 3 fold của mô hình tốt nhất).
    best_name = compare["pr_auc_cv_mean"].idxmax()
    best_val = float(compare.loc[best_name, "pr_auc_cv_mean"])
    cv_std = float(compare.loc[best_name, "pr_auc_cv_std"])
    final_name = best_name
    reason = (f"{best_name} có PR-AUC CV train80 cao nhất ({best_val:.4f} ± "
              f"{cv_std:.4f}).")
    for cand in sorted(compare.index, key=lambda n: SIMPLICITY.get(n, 99)):
        if SIMPLICITY.get(cand, 99) < SIMPLICITY.get(best_name, 99):
            gap = best_val - float(compare.loc[cand, "pr_auc_cv_mean"])
            if gap < cv_std:
                final_name = cand
                reason = (f"{cand} kém {best_name} {gap:.4f} < 1 độ lệch chuẩn "
                          f"CV ({cv_std:.4f}) nên chọn mô hình đơn giản hơn "
                          f"theo quy tắc 1-SE.")
                break
    print("Mô hình cuối (chọn lại trên CV train80):", final_name, "|", reason)
    (REPORTS_DIR / "reselection80.json").write_text(
        json.dumps({"best_model_cv": best_name, "best_pr_auc_cv": best_val,
                    "cv_std_best": cv_std, "final_model": final_name,
                    "selection_reason": reason,
                    "best_params": best_params.get(final_name, {}),
                    "cv_table": {k: v for k, v in rows.items()},
                    "note": "Chọn lại bằng TimeSeriesSplit(3) CV trên train80 "
                            "(32.940 dòng); test không dùng để chọn. Lựa chọn "
                            "v2 ban đầu (Tree trên validation, fit train60) "
                            "được thay thế vì cùng cửa sổ train60 đã gây lỗi "
                            "thiết kế đầu tiên."},
                   ensure_ascii=False, indent=2), encoding="utf-8")

    # --- Nhật ký thực nghiệm: một dòng cho mỗi mô hình (CV train80) ---
    log_path = REPORTS_DIR.parent / "experiments" / "experiment_log.csv"
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")
    with open(log_path, "a", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        for name in ["LogReg", "Tree", "KNN", "HGB", "LightGBM", "RF"]:
            r = rows[name]
            w.writerow([f"EXP_T2_RESEL80_{name}", ts,
                        f"{name} (chọn lại, CV train80)", "without_duration",
                        "class_weight_balanced", "3-TimeSeriesSplit(train80)",
                        r["roc_auc_cv_mean"], r["pr_auc_cv_mean"], "", "", "",
                        f"Chọn lại mô hình bằng TimeSeriesSplit(3) CV trên "
                        f"train80 (32.940 dòng), siêu tham số đã tune; "
                        f"PR-AUC CV {r['pr_auc_cv_mean']} ± {r['pr_auc_cv_std']}"])
    print(f"Đã thêm 6 dòng EXP_T2_RESEL80_* vào experiment_log.csv")


if __name__ == "__main__":
    main()

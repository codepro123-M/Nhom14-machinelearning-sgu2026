# -*- coding: utf-8 -*-
"""Hoàn tất phần tổng hợp của 07 (chỉ chạy phần đuôi, không tuning lại).

Dùng best_params_time.json + 6 CSV time_tuning đã có: refit 6 mô hình tốt
nhất trên train60, đánh giá trên validation, viết time_tuning_compare.csv
và time_selection.json theo đúng logic quy tắc 1-SE của 07.
"""
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import REPORTS_DIR, SEED
from src.data import load_clean, time_split_3way
from src.evaluate import _single_thread
from src.features import build_pipeline

SIMPLICITY = {"LogReg": 0, "Tree": 1, "KNN": 2, "HGB": 3, "LightGBM": 4, "RF": 5}
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


def main():
    df = load_clean()
    X_tr, X_va, X_te, y_tr, y_va, y_te = time_split_3way(df)
    best_params = json.loads((REPORTS_DIR / "best_params_time.json").read_text(encoding="utf-8"))
    rows = {}
    for name in ["LogReg", "Tree", "KNN", "HGB", "LightGBM", "RF"]:
        cvr = pd.read_csv(REPORTS_DIR / f"time_tuning_cv_results_{name}.csv")
        best_row = cvr.loc[cvr["rank_test_score"].idxmin()]
        kwargs = {k.replace("clf__", ""): v for k, v in best_params[name].items()}
        est = _single_thread(build_pipeline(CLASSES[name](**{**DEFAULT_KWARGS[name], **kwargs})))
        est.fit(X_tr, y_tr)
        p = est.predict_proba(X_va)[:, 1]
        rows[name] = {"pr_auc_cv_ts": round(float(best_row["mean_test_score"]), 4),
                      "pr_auc_val": round(float(average_precision_score(y_va, p)), 4),
                      "roc_auc_val": round(float(roc_auc_score(y_va, p)), 4)}
        print(f"    {name}: CV PR-AUC = {rows[name]['pr_auc_cv_ts']:.4f} | "
              f"val PR-AUC = {rows[name]['pr_auc_val']:.4f} | ROC val = {rows[name]['roc_auc_val']:.4f}",
              flush=True)
    compare = pd.DataFrame(rows).T
    best_name = compare["pr_auc_val"].idxmax()
    cvr = pd.read_csv(REPORTS_DIR / f"time_tuning_cv_results_{best_name}.csv")
    best_row = cvr.loc[cvr["mean_test_score"].idxmax()]
    cv_std = float(np.std([best_row[f"split{i}_test_score"] for i in range(3)]))
    compare["pr_auc_cv_std"] = np.nan
    compare.loc[best_name, "pr_auc_cv_std"] = round(cv_std, 4)
    compare.to_csv(REPORTS_DIR / "time_tuning_compare.csv")
    print("\n=== So sanh sau tuning (TimeSeriesSplit CV tren train60 + validation) ===")
    print(compare.to_string())
    best_val = float(compare.loc[best_name, "pr_auc_val"])
    final_name, reason = best_name, f"{best_name} co PR-AUC validation cao nhat ({best_val:.4f})."
    for cand in sorted(compare.index, key=lambda n: SIMPLICITY.get(n, 99)):
        if SIMPLICITY.get(cand, 99) < SIMPLICITY.get(best_name, 99):
            gap = best_val - float(compare.loc[cand, "pr_auc_val"])
            if gap < cv_std:
                final_name, reason = cand, (f"{cand} kem {best_name} {gap:.4f} < 1 do lech "
                                            f"chuan CV ({cv_std:.4f}) nen chon mo hinh don "
                                            f"gian hon theo quy tac 1-SE.")
                break
    print("Mo hinh cuoi (thiet ke theo thoi gian):", final_name, "|", reason)
    (REPORTS_DIR / "time_selection.json").write_text(
        json.dumps({"best_model_val": best_name, "best_pr_auc_val": best_val,
                    "cv_std_best": round(cv_std, 4), "final_model": final_name,
                    "selection_reason": reason, "best_params": best_params.get(final_name, {})},
                   ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

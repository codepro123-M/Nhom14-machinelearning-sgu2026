# -*- coding: utf-8 -*-
"""Chạy nốt tuning RF (A3) — phần còn dở của scripts/07_time_design.py.

Lý do: lần chạy 07 tối 2026-11-13 dừng sau LightGBM, chưa kịp RF, nên
time_tuning_cv_results_RF.csv / best_params_time.json / time_selection.json
chưa được tạo. Script này chỉ chạy RF với đúng lưới, n_iter=40,
TimeSeriesSplit(3), seed và pipeline như 07, rồi tái tạo các file tổng hợp
từ 5 CSV đã có + CSV RF, theo đúng logic 07 (so sánh validation + quy tắc 1-SE).
Không chạy lại 5 mô hình đã xong.
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import REPORTS_DIR, SEED
from src.data import load_clean, time_split_3way
from src.evaluate import _single_thread
from src.features import build_pipeline

SIMPLICITY = {"LogReg": 0, "Tree": 1, "KNN": 2, "HGB": 3, "LightGBM": 4, "RF": 5}


def main():
    df = load_clean()
    X_tr, X_va, X_te, y_tr, y_va, y_te = time_split_3way(df)
    tscv = TimeSeriesSplit(n_splits=3)
    space = {"clf__n_estimators": [100, 200, 300, 400, 600],
             "clf__max_depth": [4, 6, 8, 10, 12, 16, None],
             "clf__min_samples_leaf": [1, 2, 5, 10, 20],
             "clf__max_features": [0.2, 0.3, 0.5, 0.7, "sqrt"]}
    clf = RandomForestClassifier(class_weight="balanced_subsample", n_jobs=1,
                                 random_state=SEED)
    search = RandomizedSearchCV(_single_thread(build_pipeline(clf)), space,
                                n_iter=40, scoring="average_precision",
                                cv=tscv, n_jobs=-1, random_state=SEED,
                                return_train_score=True)
    print(">>> Tuning RF (n_iter=40, TimeSeriesSplit(3) tren train60) ...", flush=True)
    search.fit(X_tr, y_tr)
    bp = {k: (v if not isinstance(v, (np.integer, np.floating)) else v.item())
          for k, v in search.best_params_.items()}
    est = search.best_estimator_
    p_va = est.predict_proba(X_va)[:, 1]
    row = {"pr_auc_cv_ts": round(float(search.best_score_), 4),
           "pr_auc_val": round(float(average_precision_score(y_va, p_va)), 4),
           "roc_auc_val": round(float(roc_auc_score(y_va, p_va)), 4)}
    print(f"    RF: CV PR-AUC = {search.best_score_:.4f} | val PR-AUC = {row['pr_auc_val']:.4f} | {bp}", flush=True)
    pd.DataFrame(search.cv_results_).to_csv(
        REPORTS_DIR / "time_tuning_cv_results_RF.csv", index=False)

    # --- Tái tạo file tổng hợp như cuối script 07 ---
    names = ["LogReg", "Tree", "KNN", "HGB", "LightGBM", "RF"]
    # best_params: lấy lại từ các CSV (hàng rank 1) cho 5 mô hình đã xong
    best_params, boundary, rows = {}, {}, {}
    for name in names:
        cvr = pd.read_csv(REPORTS_DIR / f"time_tuning_cv_results_{name}.csv")
        best_row = cvr.loc[cvr["rank_test_score"].idxmin()]
        params = json.loads(best_row["params"].replace("'", '"')) if isinstance(best_row["params"], str) else best_row["params"]
        # params trong CSV lưu dạng dict Python; parse an toàn bằng ast
        import ast
        if isinstance(best_row["params"], str):
            params = ast.literal_eval(best_row["params"])
        best_params[name] = {k: (v.item() if isinstance(v, (np.integer, np.floating)) else v)
                             for k, v in params.items()}
    # boundary check: cần lưới gốc — định nghĩa lại như 07
    spaces = {
        "LogReg": {"clf__C": [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100]},
        "Tree": {"clf__max_depth": [3, 5, 8, 10, 12, 15, 20, None],
                 "clf__min_samples_leaf": [1, 5, 20, 50, 100, 200],
                 "clf__min_samples_split": [2, 10, 50, 100],
                 "clf__criterion": ["gini", "entropy"]},
        "KNN": {"clf__n_neighbors": [5, 10, 15, 25, 50, 75, 100, 150, 200],
                "clf__weights": ["uniform", "distance"], "clf__p": [1, 2]},
        "HGB": {"clf__learning_rate": [0.03, 0.05, 0.1, 0.2], "clf__max_depth": [None, 3, 5, 8],
                "clf__max_leaf_nodes": [15, 31, 63], "clf__min_samples_leaf": [10, 20, 50],
                "clf__l2_regularization": [0.0, 0.1, 1.0], "clf__max_iter": [100, 200, 300]},
        "LightGBM": {"clf__n_estimators": [100, 200, 300], "clf__learning_rate": [0.03, 0.05, 0.1, 0.2],
                     "clf__num_leaves": [15, 31, 63], "clf__max_depth": [-1, 3, 5, 8],
                     "clf__min_child_samples": [10, 20, 50], "clf__reg_lambda": [0.0, 0.1, 1.0]},
        "RF": {"clf__n_estimators": [100, 200, 300, 400, 600],
               "clf__max_depth": [4, 6, 8, 10, 12, 16, None],
               "clf__min_samples_leaf": [1, 2, 5, 10, 20],
               "clf__max_features": [0.2, 0.3, 0.5, 0.7, "sqrt"]},
    }
    for name in names:
        edge = {}
        for k, v in best_params[name].items():
            grid = spaces[name][k]
            edge[k.replace("clf__", "")] = {"value": v, "grid_min": grid[0], "grid_max": grid[-1],
                                            "at_edge": (v == grid[0] or v == grid[-1])}
        boundary[name] = edge
    # rows: val metrics — tính lại cho 5 mô hình đã xong bằng cách refit best estimator
    # (đắt nhưng đảm bảo nhất quán); RF đã có row ở trên.
    from lightgbm import LGBMClassifier
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.tree import DecisionTreeClassifier
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
    for name in names:
        cvr = pd.read_csv(REPORTS_DIR / f"time_tuning_cv_results_{name}.csv")
        best_row = cvr.loc[cvr["rank_test_score"].idxmin()]
        if name == "RF":
            rows[name] = {**row, "pr_auc_cv_ts": round(float(best_row["mean_test_score"]), 4)}
            continue
        kwargs = {k.replace("clf__", ""): v for k, v in best_params[name].items()}
        est2 = _single_thread(build_pipeline(CLASSES[name](**{**DEFAULT_KWARGS[name], **kwargs})))
        est2.fit(X_tr, y_tr)
        p = est2.predict_proba(X_va)[:, 1]
        rows[name] = {"pr_auc_cv_ts": round(float(best_row["mean_test_score"]), 4),
                      "pr_auc_val": round(float(average_precision_score(y_va, p)), 4),
                      "roc_auc_val": round(float(roc_auc_score(y_va, p)), 4)}
        print(f"    refit {name}: val PR-AUC = {rows[name]['pr_auc_val']:.4f}", flush=True)

    (REPORTS_DIR / "best_params_time.json").write_text(
        json.dumps(best_params, ensure_ascii=False, indent=2), encoding="utf-8")
    (REPORTS_DIR / "tuning_boundary_check.json").write_text(
        json.dumps(boundary, ensure_ascii=False, indent=2), encoding="utf-8")
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

# -*- coding: utf-8 -*-
"""Giai đoạn 6b (v2, sau góp ý 2026-11-14): Thiết kế theo thời gian + tuning công bằng.

Thay đổi so với v1 (stratified làm chuẩn):
- Chia theo thời gian thành 3 phần: train 60% đầu / validation 20% giữa /
  test 20% cuối. Tuning bằng TimeSeriesSplit(3) CHỈ trên train; chọn mô hình
  trên validation; test chỉ chạm ở bước đánh giá cuối (script 09).
- Tuning công bằng: cùng ngân sách n_iter=40 cho cả 6 mô hình, kể cả HGB và
  LightGBM (v1 chỉ tuning 4 mô hình với n_iter=12, HGB/LGBM không tuning).
- Lưới tìm kiếm mở rộng để kiểm tra chạm biên (B3): KNN n_neighbors tới 200,
  RF max_depth tới 16, max_features xuống 0,2; HGB/LGBM có lưới riêng.
- Quy tắc 1-SE: chọn mô hình đơn giản hơn nếu PR-AUC validation kém mô hình
  tốt nhất chưa tới 1 độ lệch chuẩn (std của TimeSeriesSplit CV).
"""
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import PROCESSED_DIR, REPORTS_DIR, SEED
from src.data import load_clean, time_split_3way
from src.evaluate import _single_thread
from src.features import build_pipeline

N_ITER = 40

# Thứ tự đơn giản -> phức tạp (cho quy tắc 1-SE). HGB/LGBM xếp trước RF vì
# nhanh hơn và ít overfit hơn khi hiệu năng ngang nhau.
SIMPLICITY = {"LogReg": 0, "Tree": 1, "KNN": 2, "HGB": 3, "LightGBM": 4, "RF": 5}


def main():
    df = load_clean()
    X_tr, X_va, X_te, y_tr, y_va, y_te = time_split_3way(df)
    info = {"n_total": len(df), "n_train": len(X_tr), "n_val": len(X_va), "n_test": len(X_te),
            "yes_rate_train": round(float(y_tr.mean()), 4),
            "yes_rate_val": round(float(y_va.mean()), 4),
            "yes_rate_test": round(float(y_te.mean()), 4),
            "months_train": sorted(X_tr["month"].unique().tolist()),
            "months_val": sorted(X_va["month"].unique().tolist()),
            "months_test": sorted(X_te["month"].unique().tolist())}
    (PROCESSED_DIR / "time_split_info.json").write_text(
        json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Chia theo thời gian 60/20/20:", info)

    tscv = TimeSeriesSplit(n_splits=3)
    spaces = {
        "LogReg": (LogisticRegression(max_iter=3000, class_weight="balanced", random_state=SEED),
                   {"clf__C": [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100]}),
        "Tree": (DecisionTreeClassifier(class_weight="balanced", random_state=SEED),
                 {"clf__max_depth": [3, 5, 8, 10, 12, 15, 20, None],
                  "clf__min_samples_leaf": [1, 5, 20, 50, 100, 200],
                  "clf__min_samples_split": [2, 10, 50, 100],
                  "clf__criterion": ["gini", "entropy"]}),
        "KNN": (KNeighborsClassifier(n_jobs=1),
                {"clf__n_neighbors": [5, 10, 15, 25, 50, 75, 100, 150, 200],
                 "clf__weights": ["uniform", "distance"],
                 "clf__p": [1, 2]}),
        "HGB": (HistGradientBoostingClassifier(class_weight="balanced", random_state=SEED),
                {"clf__learning_rate": [0.03, 0.05, 0.1, 0.2],
                 "clf__max_depth": [None, 3, 5, 8],
                 "clf__max_leaf_nodes": [15, 31, 63],
                 "clf__min_samples_leaf": [10, 20, 50],
                 "clf__l2_regularization": [0.0, 0.1, 1.0],
                 "clf__max_iter": [100, 200, 300]}),
        "LightGBM": (LGBMClassifier(class_weight="balanced", verbosity=-1, n_jobs=1,
                                    random_state=SEED),
                     {"clf__n_estimators": [100, 200, 300],
                      "clf__learning_rate": [0.03, 0.05, 0.1, 0.2],
                      "clf__num_leaves": [15, 31, 63],
                      "clf__max_depth": [-1, 3, 5, 8],
                      "clf__min_child_samples": [10, 20, 50],
                      "clf__reg_lambda": [0.0, 0.1, 1.0]}),
        "RF": (RandomForestClassifier(class_weight="balanced_subsample", n_jobs=1,
                                      random_state=SEED),
               {"clf__n_estimators": [100, 200, 300, 400, 600],
                "clf__max_depth": [4, 6, 8, 10, 12, 16, None],
                "clf__min_samples_leaf": [1, 2, 5, 10, 20],
                "clf__max_features": [0.2, 0.3, 0.5, 0.7, "sqrt"]}),
    }

    best_params, rows, boundary = {}, {}, {}
    for name, (clf, space) in spaces.items():
        search = RandomizedSearchCV(_single_thread(build_pipeline(clf)), space,
                                    n_iter=N_ITER, scoring="average_precision",
                                    cv=tscv, n_jobs=-1, random_state=SEED,
                                    return_train_score=True)
        print(f">>> Tuning {name} (n_iter={N_ITER}, TimeSeriesSplit(3) trên train60) ...",
              flush=True)
        search.fit(X_tr, y_tr)
        bp = {k: (v if not isinstance(v, (np.integer, np.floating)) else v.item())
              for k, v in search.best_params_.items()}
        best_params[name] = bp
        # Kiểm tra chạm biên: tham số tốt nhất có nằm ở đầu/cuối lưới không (B3)
        edge = {}
        for k, v in bp.items():
            grid = space[k]
            edge[k.replace("clf__", "")] = {
                "value": v, "grid_min": grid[0], "grid_max": grid[-1],
                "at_edge": (v == grid[0] or v == grid[-1])}
        boundary[name] = edge
        # Đánh giá best estimator trên validation (fit lại trên toàn bộ train60)
        est = search.best_estimator_
        p_va = est.predict_proba(X_va)[:, 1]
        rows[name] = {
            "pr_auc_cv_ts": round(float(search.best_score_), 4),
            "pr_auc_val": round(float(average_precision_score(y_va, p_va)), 4),
            "roc_auc_val": round(float(roc_auc_score(y_va, p_va)), 4),
        }
        print(f"    {name}: CV PR-AUC = {search.best_score_:.4f} | "
              f"val PR-AUC = {rows[name]['pr_auc_val']:.4f} | {bp}", flush=True)
        pd.DataFrame(search.cv_results_).to_csv(
            REPORTS_DIR / f"time_tuning_cv_results_{name}.csv", index=False)

    (REPORTS_DIR / "best_params_time.json").write_text(
        json.dumps(best_params, ensure_ascii=False, indent=2), encoding="utf-8")
    (REPORTS_DIR / "tuning_boundary_check.json").write_text(
        json.dumps(boundary, ensure_ascii=False, indent=2), encoding="utf-8")
    compare = pd.DataFrame(rows).T
    # Độ lệch chuẩn CV của mô hình tốt nhất (lấy từ cv_results_ của nó)
    best_name = compare["pr_auc_val"].idxmax()
    cvr = pd.read_csv(REPORTS_DIR / f"time_tuning_cv_results_{best_name}.csv")
    best_row = cvr.loc[cvr["mean_test_score"].idxmax()]
    cv_std = float(np.std([best_row[f"split{i}_test_score"] for i in range(3)]))
    compare["pr_auc_cv_std"] = np.nan
    compare.loc[best_name, "pr_auc_cv_std"] = round(cv_std, 4)
    compare.to_csv(REPORTS_DIR / "time_tuning_compare.csv")
    print("\n=== So sánh sau tuning (TimeSeriesSplit CV trên train60 + validation) ===")
    print(compare.to_string())

    # Quy tắc 1-SE trên validation
    best_val = float(compare.loc[best_name, "pr_auc_val"])
    final_name, reason = best_name, f"{best_name} có PR-AUC validation cao nhất ({best_val:.4f})."
    for cand in sorted(compare.index, key=lambda n: SIMPLICITY.get(n, 99)):
        if SIMPLICITY.get(cand, 99) < SIMPLICITY.get(best_name, 99):
            gap = best_val - float(compare.loc[cand, "pr_auc_val"])
            if gap < cv_std:
                final_name, reason = cand, (f"{cand} kém {best_name} {gap:.4f} < 1 độ lệch "
                                            f"chuẩn CV ({cv_std:.4f}) nên chọn mô hình đơn "
                                            f"giản hơn theo quy tắc 1-SE.")
                break
    print("Mô hình cuối (thiết kế theo thời gian):", final_name, "|", reason)
    (REPORTS_DIR / "time_selection.json").write_text(
        json.dumps({"best_model_val": best_name, "best_pr_auc_val": best_val,
                    "cv_std_best": round(cv_std, 4), "final_model": final_name,
                    "selection_reason": reason, "best_params": best_params.get(final_name, {})},
                   ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

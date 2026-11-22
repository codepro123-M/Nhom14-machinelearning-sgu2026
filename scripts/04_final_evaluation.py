# -*- coding: utf-8 -*-
"""Giai đoạn 7: Chọn ngưỡng và đánh giá cuối trên tập test (Cẩm nang giai đoạn 7).

- Chốt mô hình cuối theo quy tắc 4.4 (xếp hạng PR-AUC CV sau tuning; quy tắc
  1-SE: chênh < 1 độ lệch chuẩn thì chọn mô hình đơn giản hơn).
- Chọn ngưỡng bằng dự đoán out-of-fold TRÊN TRAIN (không chạm test):
  ngưỡng tối ưu F1 + ngưỡng giữ recall >= 0.7.
- Đánh giá trên test cho mô hình cuối; bootstrap CI 95%. (Đính chính 19/11/2026: test đã được đánh giá nhiều lần trong dự án (xem mục 4.2.8 của báo cáo).)
- Bảng test_results.csv cho mọi mô hình đã tuning (chỉ để báo cáo).
- Lưu models/model.joblib + models/threshold.json.
"""
import json
import sys
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, PrecisionRecallDisplay,
                             RocCurveDisplay)
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import FIGURES_DIR, MODELS_DIR, N_SPLITS, REPORTS_DIR, SEED
from src.data import load_clean, stratified_split
from src.evaluate import (bootstrap_ci, compute_metrics, pick_threshold)
from src.features import build_pipeline

plt.rcParams.update({"figure.dpi": 150, "font.size": 10})


def _fmt(x, nd=4):
    """Định dạng số với dấu phẩy thập phân (quy ước tiếng Việt) cho chuỗi lý do."""
    return f"{x:.{nd}f}".replace(".", ",")


# Thứ tự đơn giản -> phức tạp (cho quy tắc 1-SE)
SIMPLICITY = {"LogReg": 0, "Tree": 1, "KNN": 2, "RF": 3}
CLASSES = {"LogReg": LogisticRegression, "Tree": DecisionTreeClassifier,
           "RF": RandomForestClassifier, "KNN": KNeighborsClassifier}
DEFAULT_KWARGS = {"LogReg": {"max_iter": 3000, "class_weight": "balanced", "random_state": SEED},
                  "Tree": {"class_weight": "balanced", "random_state": SEED},
                  "RF": {"class_weight": "balanced_subsample", "n_jobs": -1, "random_state": SEED},
                  "KNN": {"n_jobs": -1}}


def estimator_from_best(name, best_params):
    kwargs = {k.replace("clf__", ""): v for k, v in best_params[name].items()}
    merged = {**DEFAULT_KWARGS[name], **kwargs}
    return CLASSES[name](**merged)


def main():
    df = load_clean()
    X_train, X_test, y_train, y_test = stratified_split(df)
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    tuned = pd.read_csv(REPORTS_DIR / "cv_tuned_models.csv", index_col=0)
    best_params = json.loads((REPORTS_DIR / "best_params.json").read_text(encoding="utf-8"))

    # --- Quy tắc chọn mô hình cuối (4.4) ---
    best_name = tuned["pr_auc_cv"].idxmax()
    best_mean = float(tuned.loc[best_name, "pr_auc_cv"])
    best_std = float(tuned.loc[best_name, "pr_auc_std"])
    final_name = best_name
    reason = f"{best_name} có PR-AUC CV cao nhất ({_fmt(best_mean)})."
    for cand in sorted(tuned.index, key=lambda n: SIMPLICITY.get(n, 99)):
        if SIMPLICITY.get(cand, 99) < SIMPLICITY.get(best_name, 99):
            if best_mean - float(tuned.loc[cand, "pr_auc_cv"]) < best_std:
                final_name = cand
                reason = (f"{cand} kém {best_name} "
                          f"{_fmt(best_mean - float(tuned.loc[cand, 'pr_auc_cv']))} "
                          f"< 1 độ lệch chuẩn ({_fmt(best_std)}) nên chọn mô hình đơn giản hơn "
                          f"theo quy tắc 1-SE.")
                break
    print("Mô hình cuối:", final_name, "|", reason)

    final = build_pipeline(estimator_from_best(final_name, best_params))
    thr_f1, oof = pick_threshold(final, X_train, y_train, cv, "f1")
    thr_r70, _ = pick_threshold(final, X_train, y_train, cv, "recall", 0.7)
    print(f"Ngưỡng tối ưu F1 (OOF train): {thr_f1:.3f} | ngưỡng giữ recall>=0.7: {thr_r70:.3f}")

    # --- Đánh giá trên test ---
    final.fit(X_train, y_train)
    p = final.predict_proba(X_test)[:, 1]
    pred = (p >= thr_f1).astype(int)
    pred_r70 = (p >= thr_r70).astype(int)
    m = compute_metrics(y_test, pred, p)
    m_r70 = compute_metrics(y_test, pred_r70, p)
    ci = bootstrap_ci(y_test, p, pred)
    print("Kết quả test (ngưỡng F1):", {k: round(v, 4) for k, v in m.items() if isinstance(v, float)})
    print("Kết quả test (ngưỡng recall>=0.7):", {k: round(v, 4) for k, v in m_r70.items() if isinstance(v, float)})
    print("Bootstrap CI 95%:", ci)

    summary = {"final_model": final_name, "selection_reason": reason,
               "best_params": best_params.get(final_name, {}),
               "threshold_f1": round(thr_f1, 4), "threshold_recall70": round(thr_r70, 4),
               "test_metrics_f1_threshold": {k: (round(float(v), 4) if isinstance(v, float) else v)
                                             for k, v in m.items()},
               "test_metrics_recall70_threshold": {k: (round(float(v), 4) if isinstance(v, float) else v)
                                                   for k, v in m_r70.items()},
               "bootstrap_ci_95": ci}
    (REPORTS_DIR / "final_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    # --- Hình ROC / PR / confusion matrix trên test ---
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    RocCurveDisplay.from_predictions(y_test, p, ax=axes[0])
    axes[0].set_title(f"ROC curve - {final_name} (test)")
    PrecisionRecallDisplay.from_predictions(y_test, p, ax=axes[1])
    axes[1].set_title(f"PR curve - {final_name} (test)")
    ConfusionMatrixDisplay.from_predictions(y_test, pred, display_labels=["no", "yes"], ax=axes[2])
    axes[2].set_title(f"Confusion matrix - {final_name} (test, ngưỡng F1={thr_f1:.3f})")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig10_final_test_curves.png", dpi=150)
    plt.close(fig)
    print("saved fig10_final_test_curves.png")

    # --- Bảng test cho mọi mô hình đã tuning (chỉ để báo cáo) ---
    rows = {}
    for name in tuned.index:
        model = build_pipeline(estimator_from_best(name, best_params))
        th, _ = pick_threshold(model, X_train, y_train, cv, "f1")
        model.fit(X_train, y_train)
        pp = model.predict_proba(X_test)[:, 1]
        pr = (pp >= th).astype(int)
        mm = compute_metrics(y_test, pr, pp)
        rows[name] = {"threshold_f1": round(th, 4),
                      **{k: round(float(v), 4) for k, v in mm.items() if isinstance(v, float)}}
    test_table = pd.DataFrame(rows).T
    test_table.to_csv(REPORTS_DIR / "test_results.csv")
    print(test_table.to_string())

    # --- Lưu mô hình + ngưỡng ---
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(final, MODELS_DIR / "model.joblib")
    (MODELS_DIR / "threshold.json").write_text(
        json.dumps({"threshold_f1": float(thr_f1), "threshold_recall70": float(thr_r70)},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    print("Đã lưu models/model.joblib + models/threshold.json")


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""Giai đoạn 6: Tinh chỉnh siêu tham số (Cẩm nang giai đoạn 6).

- RandomizedSearchCV, tối ưu average_precision (PR-AUC), cùng CV 5-fold,
  CHỈ trên train, không bao giờ nhìn test.
- Validation curve cho Decision Tree theo max_depth + learning curve cho
  mô hình tốt nhất (phân tích overfitting/underfitting).
- So sánh sau tuning và lưu bảng cv_tuned_models.csv + best_params.json.
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (RandomizedSearchCV, StratifiedKFold,
                                     learning_curve, validation_curve)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import FIGURES_DIR, N_SPLITS, REPORTS_DIR, SEED
from src.data import load_clean, stratified_split
from src.evaluate import cv_report
from src.features import build_pipeline

plt.rcParams.update({"figure.dpi": 150, "font.size": 10})


def main():
    df = load_clean()
    X_train, X_test, y_train, y_test = stratified_split(df)
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

    spaces = {
        "LogReg": (LogisticRegression(max_iter=3000, class_weight="balanced", random_state=SEED),
                   {"clf__C": [0.001, 0.01, 0.1, 1, 10, 100]}),
        "Tree": (DecisionTreeClassifier(class_weight="balanced", random_state=SEED),
                 {"clf__max_depth": [3, 4, 5, 6, 8, 10, 12],
                  "clf__min_samples_leaf": [5, 20, 50, 100, 200],
                  "clf__min_samples_split": [10, 50, 100],
                  "clf__criterion": ["gini", "entropy"]}),
        "RF": (RandomForestClassifier(class_weight="balanced_subsample", n_jobs=1, random_state=SEED),
               {"clf__n_estimators": [200, 400],
                "clf__max_depth": [6, 8, 12, None],
                "clf__min_samples_leaf": [5, 10, 30, 60],
                "clf__max_features": ["sqrt", 0.3, 0.5]}),
        "KNN": (KNeighborsClassifier(n_jobs=1),
                {"clf__n_neighbors": [15, 25, 50, 100],
                 "clf__weights": ["uniform", "distance"]}),
    }

    best = {}
    best_params = {}
    for name, (clf, space) in spaces.items():
        n_iter = min(12, int(np.prod([len(v) for v in space.values()])))
        search = RandomizedSearchCV(build_pipeline(clf), space, n_iter=n_iter,
                                    scoring="average_precision", cv=cv, n_jobs=-1,
                                    random_state=SEED, return_train_score=True)
        print(f">>> Tuning {name} (n_iter={n_iter}) ...", flush=True)
        search.fit(X_train, y_train)
        best[name] = search
        best_params[name] = {k: (v if not isinstance(v, (np.integer, np.floating)) else v.item())
                             for k, v in search.best_params_.items()}
        print(f"    {name}: best PR-AUC (CV) = {search.best_score_:.4f} | {search.best_params_}",
              flush=True)
        pd.DataFrame(search.cv_results_).to_csv(
            REPORTS_DIR / f"tuning_cv_results_{name}.csv", index=False)

    (REPORTS_DIR / "best_params.json").write_text(
        json.dumps(best_params, ensure_ascii=False, indent=2), encoding="utf-8")

    compare = cv_report({k: v.best_estimator_ for k, v in best.items()},
                        X_train, y_train, cv)
    compare.to_csv(REPORTS_DIR / "cv_tuned_models.csv")
    print("\n=== So sánh sau tuning (CV trên train) ===")
    print(compare[["roc_auc_cv", "roc_auc_std", "pr_auc_cv", "pr_auc_std",
                   "f1_cv", "recall_cv", "precision_cv"]].to_string())

    # Validation curve: Decision Tree theo max_depth
    depths = [1, 2, 3, 4, 5, 6, 8, 10, 15, 20, 30]
    tr, va = validation_curve(
        build_pipeline(DecisionTreeClassifier(class_weight="balanced", min_samples_leaf=20,
                                              random_state=SEED)),
        X_train, y_train, param_name="clf__max_depth", param_range=depths,
        cv=cv, scoring="average_precision", n_jobs=-1)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(depths, tr.mean(1), marker="o", label="Train")
    ax.plot(depths, va.mean(1), marker="o", label="CV")
    ax.set_xlabel("max_depth")
    ax.set_ylabel("PR-AUC")
    ax.set_title("Validation curve: Decision Tree theo max_depth")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig08_validation_curve_tree.png", dpi=150)
    plt.close(fig)
    print("saved fig08_validation_curve_tree.png")

    # Learning curve của mô hình tốt nhất (theo PR-AUC CV sau tuning)
    best_name = compare["pr_auc_cv"].idxmax()
    best_model = best[best_name].best_estimator_
    print(f"Mô hình tốt nhất sau tuning (PR-AUC CV): {best_name} "
          f"= {compare.loc[best_name, 'pr_auc_cv']:.4f} "
          f"(std {compare.loc[best_name, 'pr_auc_std']:.4f})")
    sizes, tr, va = learning_curve(best_model, X_train, y_train, cv=cv,
                                   scoring="average_precision",
                                   train_sizes=np.linspace(0.1, 1.0, 6), n_jobs=-1)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(sizes, tr.mean(1), marker="o", label="Train")
    ax.plot(sizes, va.mean(1), marker="o", label="CV")
    ax.set_xlabel("Số mẫu train")
    ax.set_ylabel("PR-AUC")
    ax.set_title(f"Learning curve: {best_name}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig09_learning_curve_best.png", dpi=150)
    plt.close(fig)
    print("saved fig09_learning_curve_best.png")
    (REPORTS_DIR / "tuned_best_model.json").write_text(
        json.dumps({"best_model": best_name,
                    "pr_auc_cv": round(float(compare.loc[best_name, "pr_auc_cv"]), 4),
                    "pr_auc_std": round(float(compare.loc[best_name, "pr_auc_std"]), 4)},
                   ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

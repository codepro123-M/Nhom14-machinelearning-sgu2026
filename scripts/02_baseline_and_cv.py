# -*- coding: utf-8 -*-
"""Giai đoạn 5: Xây dựng mô hình và so sánh bằng CV (Cẩm nang giai đoạn 5).

Mọi mô hình dùng CÙNG split, CÙNG fold (StratifiedKFold 5, seed 42) và CÙNG
pipeline tiền xử lý để so sánh công bằng. Metric chính: PR-AUC.
Thêm HistGradientBoosting làm mô hình tham chiếu mạnh (Cẩm nang 9.6) và
XGBoost/LightGBM nếu cài được (chỉ để đối chiếu, không tràn lan).
"""
import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import N_SPLITS, REPORTS_DIR, SEED
from src.data import load_clean, stratified_split
from src.evaluate import cv_report
from src.features import build_pipeline


def main():
    df = load_clean()
    X_train, X_test, y_train, y_test = stratified_split(df)
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

    models = {
        "Dummy": build_pipeline(DummyClassifier(strategy="prior")),
        "LogReg": build_pipeline(LogisticRegression(max_iter=2000, random_state=SEED)),
        "LogReg_balanced": build_pipeline(
            LogisticRegression(max_iter=2000, class_weight="balanced", random_state=SEED)),
        "Tree": build_pipeline(DecisionTreeClassifier(random_state=SEED)),
        "Tree_balanced": build_pipeline(
            DecisionTreeClassifier(class_weight="balanced", random_state=SEED)),
        "RF": build_pipeline(RandomForestClassifier(
            n_estimators=300, n_jobs=1, random_state=SEED)),
        "RF_balanced": build_pipeline(RandomForestClassifier(
            n_estimators=300, class_weight="balanced_subsample", n_jobs=1, random_state=SEED)),
        "KNN": build_pipeline(KNeighborsClassifier(n_neighbors=25, n_jobs=1)),
        "LinearSVC_balanced": build_pipeline(
            LinearSVC(class_weight="balanced", random_state=SEED)),
        "HGB_balanced": build_pipeline(HistGradientBoostingClassifier(
            max_iter=100, class_weight="balanced", random_state=SEED)),
    }
    # Mô hình tham chiếu mạnh, chỉ thêm nếu thư viện cài được.
    try:
        from xgboost import XGBClassifier
        models["XGBoost"] = build_pipeline(XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.1, subsample=0.8,
            colsample_bytree=0.8, eval_metric="logloss", n_jobs=1, random_state=SEED))
        print("XGBoost: có")
    except Exception as e:
        print("XGBoost: không cài được, bỏ qua -", type(e).__name__)
    try:
        from lightgbm import LGBMClassifier
        models["LightGBM"] = build_pipeline(LGBMClassifier(
            n_estimators=300, learning_rate=0.05, num_leaves=31, n_jobs=1,
            random_state=SEED, verbose=-1))
        print("LightGBM: có")
    except Exception as e:
        print("LightGBM: không cài được, bỏ qua -", type(e).__name__)

    results = cv_report(models, X_train, y_train, cv)
    results.to_csv(REPORTS_DIR / "cv_default_models.csv")
    print(results[["roc_auc_cv", "roc_auc_std", "pr_auc_cv", "pr_auc_std",
                   "f1_cv", "recall_cv", "precision_cv", "fit_time_s"]].to_string())
    print("Đã lưu reports/cv_default_models.csv")


if __name__ == "__main__":
    main()

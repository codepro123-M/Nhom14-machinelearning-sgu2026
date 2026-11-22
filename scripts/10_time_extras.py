# -*- coding: utf-8 -*-
"""Giai đoạn 9b (v2): Thí nghiệm bổ sung trên thiết kế theo thời gian.

- Duration A/B (RQ4): TimeSeriesSplit CV trên train60 với mô hình cuối đã tuning.
- Ablation theo nhóm biến (RQ3): LogReg balanced, 7 tổ hợp, TimeSeriesSplit CV.
- B4 — Drift: vẽ PR-AUC/recall theo quý và recall theo tháng trên test theo
  thời gian; đề xuất chiến lược huấn luyện lại định kỳ (ghi trong báo cáo).
- B5 — Chi phí-lợi ích có khoảng bất định: đường lợi nhuận theo % khách được
  gọi cho 5 cặp (C, R), kèm dải tin cậy bootstrap 95%.
"""
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import FIGURES_DIR, MODELS_DIR, REPORTS_DIR, SEED
from src.data import load_clean, time_split_3way
from src.evaluate import cv_report
from src.features import (GROUP_CAMPAIGN, GROUP_MACRO, GROUP_PERSONAL,
                          build_pipeline)

plt.rcParams.update({"figure.dpi": 150, "font.size": 10})
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
PAIRS = [(1.0, 15.0), (1.0, 10.0), (1.0, 25.0), (2.0, 15.0), (0.5, 15.0)]


def estimator_from_best(name, best_params):
    kwargs = {k.replace("clf__", ""): v for k, v in best_params[name].items()}
    return CLASSES[name](**{**DEFAULT_KWARGS[name], **kwargs})


def main():
    df = load_clean()
    X_tr, X_va, X_te, y_tr, y_va, y_te = time_split_3way(df)
    tscv = TimeSeriesSplit(n_splits=3)
    best_params = json.loads((REPORTS_DIR / "best_params_time.json").read_text(encoding="utf-8"))
    cal_info = json.loads((REPORTS_DIR / "calibration_comparison.json").read_text(encoding="utf-8"))
    final_name = cal_info["final_model"]
    out = {}

    # ---------- Duration A/B trên thiết kế theo thời gian ----------
    est = estimator_from_best(final_name, best_params)
    model_a = build_pipeline(type(est)(**est.get_params()))
    model_b = build_pipeline(type(est)(**est.get_params()), include_duration=True)
    ab = cv_report({"A_khong_duration": model_a, "B_co_duration": model_b},
                   X_tr, y_tr, tscv)
    ab.to_csv(REPORTS_DIR / "duration_ab_time_cv.csv")
    out["duration_ab_time_cv"] = ab[["roc_auc_cv", "pr_auc_cv"]].round(4).to_dict("index")
    print("Duration A/B (TimeSeriesSplit CV, train60):", out["duration_ab_time_cv"])

    # ---------- Ablation theo nhóm biến (TimeSeriesSplit CV) ----------
    groups = {"Personal": GROUP_PERSONAL, "Campaign": GROUP_CAMPAIGN, "Macro": GROUP_MACRO}
    combos = {"Personal": ["Personal"], "Campaign": ["Campaign"], "Macro": ["Macro"],
              "Personal+Campaign": ["Personal", "Campaign"],
              "Campaign+Macro": ["Campaign", "Macro"],
              "Personal+Macro": ["Personal", "Macro"],
              "All": ["Personal", "Campaign", "Macro"]}

    def merge(names):
        num = sum([groups[n]["num"] for n in names], [])
        cat = sum([groups[n]["cat"] for n in names], [])
        return num, cat

    abl = {}
    for label, names in combos.items():
        num, cat = merge(names)
        abl[label] = build_pipeline(
            LogisticRegression(max_iter=3000, class_weight="balanced", random_state=SEED),
            num=num, cat=cat)
    abl_res = cv_report(abl, X_tr, y_tr, tscv)
    abl_res.to_csv(REPORTS_DIR / "ablation_time_cv.csv")
    out["ablation_time_cv"] = abl_res[["roc_auc_cv", "pr_auc_cv", "f1_cv",
                                        "recall_cv"]].round(4).to_dict("index")
    print("Ablation (TimeSeriesSplit CV):\n", abl_res[["roc_auc_cv", "pr_auc_cv"]].to_string())

    # ---------- B4: drift theo quý/tháng trên test ----------
    cal = joblib.load(MODELS_DIR / "model_calibrated.joblib")
    p_test = cal.predict_proba(X_te)[:, 1]
    thr = json.loads((MODELS_DIR / "threshold.json").read_text(encoding="utf-8"))
    pred = (p_test >= thr["threshold_cost_benefit"]).astype(int)
    qdf = pd.read_csv(REPORTS_DIR / "drift_by_quarter.csv")
    fig, ax1 = plt.subplots(figsize=(7, 4.5))
    ax1.plot(qdf["quarter"], qdf["yes_rate"], marker="o", label="Tỷ lệ yes thực")
    ax1.plot(qdf["quarter"], qdf["recall"], marker="s", label="Recall")
    ax1.plot(qdf["quarter"], qdf["pr_auc"], marker="^", label="PR-AUC")
    ax1.set_ylim(0, 1.05)
    ax1.set_xlabel("Quý trong tập test theo thời gian (Q1 sớm nhất)")
    ax1.set_ylabel("Tỷ lệ / điểm số")
    ax1.set_title("Độ trôi trên test theo thời gian: yes-rate, recall, PR-AUC theo quý")
    ax1.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig19_drift_quarter.png", dpi=150)
    plt.close(fig)
    print("saved fig19_drift_quarter.png")

    err_month = pd.read_csv(REPORTS_DIR / "error_by_month_time.csv")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(err_month["group"], err_month["recall"].fillna(0))
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Tháng (test theo thời gian)")
    ax.set_ylabel("Recall lớp yes")
    ax.set_title("Recall theo tháng trên test theo thời gian")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig20_drift_month.png", dpi=150)
    plt.close(fig)
    print("saved fig20_drift_month.png")

    # ---------- B5: đường lợi nhuận theo % gọi, kèm bootstrap ----------
    y = np.asarray(y_te)
    order = np.argsort(-p_test)
    y_sorted = y[order]
    grid = np.linspace(0.05, 1.0, 20)  # % khách được gọi
    rng = np.random.default_rng(SEED)
    curves = {}
    for C, R in PAIRS:
        profits = []
        for _ in range(200):
            idx = rng.integers(0, len(y), len(y))
            ys = y_sorted[idx]
            cum = np.cumsum(ys)
            k = np.maximum((grid * len(y)).astype(int), 1)
            profits.append(R * cum[k - 1] - C * k)
        profits = np.array(profits)
        curves[f"C={C},R={R}"] = {
            "called_pct": grid.round(3).tolist(),
            "profit_mean": profits.mean(axis=0).round(1).tolist(),
            "profit_lo": np.percentile(profits, 2.5, axis=0).round(1).tolist(),
            "profit_hi": np.percentile(profits, 97.5, axis=0).round(1).tolist(),
        }
    (REPORTS_DIR / "cost_benefit_curves_time.json").write_text(
        json.dumps(curves, ensure_ascii=False, indent=2), encoding="utf-8")
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, c in curves.items():
        x = np.array(c["called_pct"]) * 100
        ax.plot(x, c["profit_mean"], marker="o", ms=3, label=label)
        ax.fill_between(x, c["profit_lo"], c["profit_hi"], alpha=0.15)
    ax.axhline(0, color="grey", lw=0.8)
    ax.set_xlabel("% khách được gọi (xếp theo điểm)")
    ax.set_ylabel("Lợi nhuận (đơn vị tiền tệ giả định)")
    ax.set_title("Lợi nhuận theo % khách được gọi — dải mờ là CI 95% (bootstrap)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig21_profit_curves.png", dpi=150)
    plt.close(fig)
    print("saved fig21_profit_curves.png")
    # Điểm tối ưu cho từng cặp (trên test, chỉ để tham chiếu)
    opt = {}
    for label, c in curves.items():
        i = int(np.argmax(c["profit_mean"]))
        opt[label] = {"best_called_pct": round(float(c["called_pct"][i]) * 100, 1),
                      "profit_mean": c["profit_mean"][i]}
    out["cost_benefit_optimum_test"] = opt
    print("Điểm tối ưu lợi nhuận trên test:", opt)

    (REPORTS_DIR / "extra_experiments_time.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Đã lưu reports/extra_experiments_time.json")


if __name__ == "__main__":
    main()

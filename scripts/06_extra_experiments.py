# -*- coding: utf-8 -*-
"""Giai đoạn 9: Thí nghiệm bổ sung (Cẩm nang giai đoạn 9).

Làm 5 thí nghiệm (ưu tiên 9.1, 9.2, 9.5 + 9.3, 9.4):
- 9.1 duration A/B (RQ4): cùng mô hình đã tuning, có/không duration.
- 9.2 ablation theo nhóm biến (RQ3): 7 tổ hợp với LogReg balanced.
- 9.3 chia theo thời gian (concept drift): mô hình cuối + LogReg, cả bản
  đầy đủ và bản BỎ nhóm biến vĩ mô (kiểm tra giả thuyết "đồng hồ thời gian").
- 9.4 hiệu chỉnh xác suất (calibration): isotonic, Brier trước/sau.
- 9.5 gain/lift + chi phí-lợi ích (RQ5): decile, top-20% capture, độ nhạy
  với các cặp (C, R).
- 9.6 SMOTE so với class_weight (chỉ lấy mẫu lại trong fold train).
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             roc_auc_score)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import FIGURES_DIR, N_SPLITS, REPORTS_DIR, SEED
from src.data import load_clean, stratified_split, time_split
from src.evaluate import cv_report, gain_lift_table
from src.features import (GROUP_CAMPAIGN, GROUP_MACRO, GROUP_PERSONAL,
                          add_features, build_pipeline)

plt.rcParams.update({"figure.dpi": 150, "font.size": 10})
CLASSES = {"LogReg": LogisticRegression, "RF": RandomForestClassifier}
DEFAULT_KWARGS = {"LogReg": {"max_iter": 3000, "class_weight": "balanced", "random_state": SEED},
                  "RF": {"class_weight": "balanced_subsample", "n_jobs": -1, "random_state": SEED}}


def estimator_from_best(name, best_params):
    kwargs = {k.replace("clf__", ""): v for k, v in best_params[name].items()}
    return CLASSES[name](**{**DEFAULT_KWARGS[name], **kwargs})


def main():
    df = load_clean()
    X_train, X_test, y_train, y_test = stratified_split(df)
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    best_params = json.loads((REPORTS_DIR / "best_params.json").read_text(encoding="utf-8"))
    final_summary = json.loads((REPORTS_DIR / "final_summary.json").read_text(encoding="utf-8"))
    final_name = final_summary["final_model"]
    out = {}

    # ---------- 9.1 duration A/B (dùng RF đã tuning) ----------
    rf = estimator_from_best("RF", best_params)
    model_a = build_pipeline(RandomForestClassifier(**rf.get_params()))
    model_b = build_pipeline(RandomForestClassifier(**rf.get_params()), include_duration=True)
    ab = cv_report({"A_khong_duration": model_a, "B_co_duration": model_b},
                   X_train, y_train, cv)
    ab.to_csv(REPORTS_DIR / "duration_ab_cv.csv")
    out["duration_ab_cv"] = ab[["roc_auc_cv", "pr_auc_cv"]].round(4).to_dict("index")
    print("9.1 duration A/B (CV):", out["duration_ab_cv"])

    # ---------- 9.2 ablation theo nhóm biến ----------
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
    abl_res = cv_report(abl, X_train, y_train, cv)
    abl_res.to_csv(REPORTS_DIR / "ablation_cv.csv")
    out["ablation_cv"] = abl_res[["roc_auc_cv", "pr_auc_cv", "recall_cv", "f1_cv"]].round(4).to_dict("index")
    print("9.2 ablation (CV):")
    print(abl_res[["roc_auc_cv", "pr_auc_cv"]].to_string())

    # ---------- 9.3 chia theo thời gian ----------
    Xt, Xv, yt, yv = time_split(df)
    final_est = estimator_from_best(final_name, best_params)
    m_full = build_pipeline(type(final_est)(**final_est.get_params())).fit(Xt, yt)
    pv = m_full.predict_proba(Xv)[:, 1]
    m_lr = build_pipeline(LogisticRegression(
        max_iter=3000, class_weight="balanced", random_state=SEED)).fit(Xt, yt)
    pv_lr = m_lr.predict_proba(Xv)[:, 1]
    # Bản bỏ nhóm vĩ mô (kiểm tra giả thuyết "đồng hồ thời gian")
    num_nomacro = [c for c in ["age", "campaign", "previous", "pdays_clean", "contacted_before"]]
    cat_nomacro = ["job", "marital", "education", "default", "housing", "loan",
                   "contact", "month", "day_of_week", "poutcome"]
    m_nomacro = build_pipeline(type(final_est)(**final_est.get_params()),
                               num=num_nomacro, cat=cat_nomacro).fit(Xt, yt)
    pv_nm = m_nomacro.predict_proba(Xv)[:, 1]
    out["time_split"] = {
        "val_yes_rate": round(float(yv.mean()), 4),
        final_name + "_full": {"roc_auc": round(float(roc_auc_score(yv, pv)), 4),
                               "pr_auc": round(float(average_precision_score(yv, pv)), 4)},
        "LogReg_full": {"roc_auc": round(float(roc_auc_score(yv, pv_lr)), 4),
                        "pr_auc": round(float(average_precision_score(yv, pv_lr)), 4)},
        final_name + "_no_macro": {"roc_auc": round(float(roc_auc_score(yv, pv_nm)), 4),
                                   "pr_auc": round(float(average_precision_score(yv, pv_nm)), 4)},
    }
    print("9.3 time split:", out["time_split"])

    # ---------- 9.4 calibration ----------
    import joblib
    from src.config import MODELS_DIR
    final = joblib.load(MODELS_DIR / "model.joblib")
    p_test = final.predict_proba(X_test)[:, 1]
    from sklearn.base import clone as _clone
    _cal_base = _clone(final)
    try:
        _cal_base.set_params(clf__n_jobs=1)  # tránh song song lồng nhau với CalibratedClassifierCV
    except Exception:
        pass
    cal = CalibratedClassifierCV(_cal_base, method="isotonic", cv=5).fit(X_train, y_train)
    p_cal = cal.predict_proba(X_test)[:, 1]
    brier_before = float(brier_score_loss(y_test, p_test))
    brier_after = float(brier_score_loss(y_test, p_cal))
    out["calibration"] = {"brier_before": round(brier_before, 4),
                          "brier_after": round(brier_after, 4)}
    fig, ax = plt.subplots(figsize=(6, 5))
    for prob, lab in [(p_test, "Mô hình gốc"), (p_cal, "Sau calibration (isotonic)")]:
        fp, mp = calibration_curve(y_test, prob, n_bins=10, strategy="quantile")
        ax.plot(mp, fp, marker="o", label=lab)
    ax.plot([0, 1], [0, 1], "--", color="grey")
    ax.set_xlabel("Xác suất dự đoán (trung bình bin)")
    ax.set_ylabel("Tỷ lệ yes thực")
    ax.set_title("Calibration curve (test)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig15_calibration_curve.png", dpi=150)
    plt.close(fig)
    print("9.4 calibration: Brier", brier_before, "->", brier_after)

    # ---------- 9.5 gain / lift / chi phí-lợi ích ----------
    res, dec = gain_lift_table(y_test, p_test)
    dec.to_csv(REPORTS_DIR / "decile_table.csv")
    out["decile_table"] = dec.round(4).to_dict("index")
    top20 = res[res["called_pct"] <= 0.20]["captured"].iloc[-1]
    out["top20_capture"] = round(float(top20), 4)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(res["called_pct"], res["captured"], label="Mô hình cuối")
    ax.plot([0, 1], [0, 1], "--", label="Ngẫu nhiên")
    ax.set_xlabel("% khách được gọi (xếp theo điểm)")
    ax.set_ylabel("% khách đồng ý bắt được")
    ax.set_title("Gain chart (test)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig16_gain_chart.png", dpi=150)
    plt.close(fig)
    # Chi phí-lợi ích: C = chi phí 1 cuộc gọi, R = lợi nhuận khi 1 khách đồng ý.
    # GIẢ ĐỊNH của nhóm (không phải số thật của ngân hàng), thử độ nhạy.
    cb = {}
    for (C, R) in [(1.0, 15.0), (1.0, 10.0), (1.0, 25.0), (2.0, 15.0), (0.5, 15.0)]:
        profit = R * res["y"].cumsum() - C * (res.index + 1)
        k = int(profit.idxmax())
        cb[f"C={C},R={R}"] = {"top_pct": round(float(res.loc[k, "called_pct"] * 100), 1),
                              "capture_pct": round(float(res.loc[k, "captured"] * 100), 1),
                              "profit_max": round(float(profit.loc[k]), 1),
                              "profit_call_all": round(float(profit.iloc[-1]), 1)}
    out["cost_benefit"] = cb
    out["top20_capture_note"] = "Gọi top 20% theo điểm mô hình bắt được {:.1f}% số người đồng ý.".format(top20 * 100)
    print("9.5 top-20% capture:", out["top20_capture"], "| cost-benefit:", cb)

    # ---------- 9.6 SMOTE so với class_weight ----------
    try:
        from imblearn.over_sampling import SMOTE
        from imblearn.pipeline import Pipeline as ImbPipeline
        prep = ColumnTransformer([
            ("num", StandardScaler(),
             ["age", "campaign", "previous", "pdays_clean", "contacted_before",
              "emp.var.rate", "cons.price.idx", "cons.conf.idx", "euribor3m", "nr.employed"]),
            ("cat", OneHotEncoder(handle_unknown="ignore"),
             ["job", "marital", "education", "default", "housing", "loan",
              "contact", "month", "day_of_week", "poutcome"]),
        ])
        smote_pipe = ImbPipeline([
            ("fe", FunctionTransformer(add_features)),
            ("prep", prep),
            ("smote", SMOTE(random_state=SEED)),
            ("clf", LogisticRegression(max_iter=3000, random_state=SEED)),
        ])
        cw_pipe = build_pipeline(LogisticRegression(max_iter=3000, class_weight="balanced",
                                                    random_state=SEED))
        sm = cv_report({"LogReg_SMOTE": smote_pipe, "LogReg_class_weight": cw_pipe},
                       X_train, y_train, cv)
        sm.to_csv(REPORTS_DIR / "smote_comparison_cv.csv")
        out["smote_vs_class_weight"] = sm[["roc_auc_cv", "pr_auc_cv", "f1_cv", "recall_cv"]].round(4).to_dict("index")
        print("9.6 SMOTE vs class_weight:", out["smote_vs_class_weight"])
    except Exception as e:
        out["smote_status"] = f"không chạy được ({type(e).__name__}: {e})"
        print("9.6 SMOTE bỏ qua:", e)

    (REPORTS_DIR / "extra_experiments.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Đã lưu reports/extra_experiments.json")


if __name__ == "__main__":
    main()

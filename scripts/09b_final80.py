# -*- coding: utf-8 -*-
"""Giai đoạn 7c' (v2 refit): Đánh giá trên test theo thời gian
cho mô hình đã refit trên train80 (scripts/08b).

- Mô hình đã calibrate trên train80 (scripts/08b) + ngưỡng chọn trên OOF
  của train80, đánh giá trên test20 (20% cuối theo thời gian) — trong lần
  chạy này test không dùng để tuning, calibrate, chọn ngưỡng hay chọn mô
  hình; tuy nhiên test đã được mở nhiều lần trong toàn bộ dự án (đính
  chính 19/11/2026, xem mục 4.2.8 của báo cáo).
- Bootstrap CI 95% cho ROC-AUC / PR-AUC / F1. Brier trước/sau calibrate
  TRÊN TEST (so sánh trung thực).
- Phân tích lỗi theo nhóm + theo quý trên test (B4). Gain/lift trên test.
- B5: đường lợi nhuận theo % khách được gọi cho 5 cặp (C, R), kèm dải tin
  cậy bootstrap 95% (logic như scripts/10).
- B2: permutation importance + SHAP tính trên TRAIN80 (ghi rõ là in-sample
  vì validation đã gộp vào huấn luyện — không còn tập validation riêng).
- Tất cả file ra có hậu tố 80, KHÔNG ghi đè file của bản train60
  (final_summary_time.json, fig17-fig24, drift_by_quarter.csv, ...).
- Cập nhật models/model_metadata.json đầy đủ + thêm dòng vào
  experiments/experiment_log.csv (cả phần 08b và 09b).
"""
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
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
import sklearn
from sklearn.calibration import calibration_curve
from sklearn.inspection import permutation_importance
from sklearn.metrics import (ConfusionMatrixDisplay, PrecisionRecallDisplay,
                             RocCurveDisplay, average_precision_score,
                             brier_score_loss, roc_auc_score)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import (FIGURES_DIR, MODELS_DIR, RAW_PATH, REPORTS_DIR, SEED)
from src.data import load_clean, time_split_3way
from src.evaluate import bootstrap_ci, compute_metrics, gain_lift_table
from src.features import feature_names_out

plt.rcParams.update({"figure.dpi": 150, "font.size": 10})
PAIRS = [(1.0, 15.0), (1.0, 10.0), (1.0, 25.0), (2.0, 15.0), (0.5, 15.0)]


def unwrap(model):
    """Bóc lớp CalibratedClassifierCV để lấy pipeline gốc."""
    if hasattr(model, "calibrated_classifiers_"):
        est = model.calibrated_classifiers_[0].estimator
        # sklearn 1.9: mô hình prefit được bọc trong FrozenEstimator
        if est.__class__.__name__ == "FrozenEstimator":
            est = est.estimator
        return est
    return model


def error_by_group(X, y, proba, pred, col):
    rows = []
    for val, idx in X.groupby(col).groups.items():
        idx = list(idx)
        yy, pp, pr = y.iloc[idx], proba[idx], pred[idx]
        tp = int(((yy == 1) & (pr == 1)).sum())
        fn = int(((yy == 1) & (pr == 0)).sum())
        fp = int(((yy == 0) & (pr == 1)).sum())
        rows.append({"group": str(val), "n": len(idx), "n_yes": int(yy.sum()),
                     "recall": round(tp / (tp + fn), 4) if tp + fn else None,
                     "n_fp": fp, "n_fn": fn})
    return pd.DataFrame(rows).sort_values("recall")


def append_log(rows):
    """Thêm dòng vào experiments/experiment_log.csv (không sửa dòng cũ)."""
    log_path = REPORTS_DIR.parent / "experiments" / "experiment_log.csv"
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")
    with open(log_path, "a", encoding="utf-8", newline="") as f:
        import csv
        w = csv.writer(f)
        for r in rows:
            w.writerow([r["id"], ts, r["model"], "without_duration",
                        "class_weight_balanced", r["cv"],
                        r.get("roc", ""), r.get("pr", ""), r.get("f1", ""),
                        r.get("rec", ""), r.get("prec", ""), r["notes"]])
    print(f"Đã thêm {len(rows)} dòng vào experiment_log.csv")


def main():
    df = load_clean()
    X_tr, X_va, X_te, y_tr, y_va, y_te = time_split_3way(df)
    X80 = pd.concat([X_tr, X_va], ignore_index=True)
    y80 = pd.concat([y_tr, y_va], ignore_index=True)
    cal = joblib.load(MODELS_DIR / "model_calibrated.joblib")
    base = joblib.load(MODELS_DIR / "model.joblib")
    thr = json.loads((MODELS_DIR / "threshold.json").read_text(encoding="utf-8"))
    cal_info = json.loads((REPORTS_DIR / "calibration_comparison80.json").read_text(encoding="utf-8"))
    final_name = cal_info["final_model"]
    thr_cb = thr["threshold_cost_benefit"]
    thr_f1 = thr["threshold_f1"]
    print(f"Mô hình cuối (refit train80): {final_name} | calibrate: "
          f"{thr['calibration_method']} | ngưỡng CB: {thr_cb:.4f} | ngưỡng F1: {thr_f1:.4f}")

    # ---------- Đánh giá trên test ----------
    p_test = cal.predict_proba(X_te)[:, 1]
    p_raw = base.predict_proba(X_te)[:, 1]
    pred = (p_test >= thr_cb).astype(int)
    m = compute_metrics(y_te, pred, p_test)
    ci = bootstrap_ci(y_te, p_test, pred)
    brier_raw = float(brier_score_loss(y_te, p_raw))
    brier_cal = float(brier_score_loss(y_te, p_test))
    print("Kết quả test theo thời gian, mô hình refit train80 (ngưỡng CB):",
          {k: round(v, 4) for k, v in m.items() if isinstance(v, float)})
    print("Bootstrap CI 95%:", ci)
    print(f"Brier trên test: gốc {brier_raw:.4f} -> đã calibrate {brier_cal:.4f}")

    pred_f1 = (p_test >= thr_f1).astype(int)
    m_f1 = compute_metrics(y_te, pred_f1, p_test)
    print("Tham chiếu — test ở ngưỡng F1 (OOF train80):",
          {k: round(v, 4) for k, v in m_f1.items() if isinstance(v, float)})

    summary = {"final_model": final_name, "selection_reason": cal_info["selection_reason"],
               "best_params": cal_info["best_params"],
               "train_window": "train80 (train60+val, refit sau khi chọn trên validation)",
               "n_train80": int(len(X80)),
               "calibration_method": thr["calibration_method"],
               "threshold_cost_benefit": round(thr_cb, 4),
               "threshold_f1_oof": thr_f1,
               "threshold_recall70_oof": thr["threshold_recall70"],
               "threshold_source": "OOF train80 (cross_val_predict cv=3, calibrated)",
               "test_metrics_cb_threshold": {k: (round(float(v), 4) if isinstance(v, float) else v)
                                             for k, v in m.items()},
               "test_metrics_f1_threshold": {k: (round(float(v), 4) if isinstance(v, float) else v)
                                             for k, v in m_f1.items()},
               "bootstrap_ci_95": ci,
               "brier_test_raw": round(brier_raw, 4),
               "brier_test_calibrated": round(brier_cal, 4),
               "note": "Test theo thời gian (20% cuối). Đính chính 19/11/2026: test đã được đánh giá nhiều lần trong dự án (xem mục 4.2.8 của báo cáo); "
                       "mô hình chọn trên validation rồi refit trên train+val; "
                       "calibrate + ngưỡng chọn trên OOF của train80."}
    (REPORTS_DIR / "final_summary_time80.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---------- fig25: ROC / PR trên test (refit) ----------
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    RocCurveDisplay.from_predictions(y_te, p_test, ax=axes[0])
    axes[0].set_title(f"ROC curve - {final_name} refit train80 (test theo thời gian)")
    PrecisionRecallDisplay.from_predictions(y_te, p_test, ax=axes[1])
    axes[1].set_title(f"PR curve - {final_name} refit train80 (test theo thời gian)")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig25_roc_pr_curves80.png", dpi=150)
    plt.close(fig)
    print("saved fig25_roc_pr_curves80.png")

    # ---------- fig26: confusion matrix (ngưỡng CB) ----------
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(y_te, pred, display_labels=["no", "yes"], ax=ax)
    ax.set_title(f"Confusion matrix - {final_name} refit train80 "
                 f"(test, ngưỡng CB={thr_cb:.3f})")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig26_confusion80.png", dpi=150)
    plt.close(fig)
    print("saved fig26_confusion80.png")

    # ---------- fig27: calibration curve trên test ----------
    fig, ax = plt.subplots(figsize=(6, 5))
    for prob, lab in [(p_raw, "Mô hình gốc (refit train80)"),
                      (p_test, f"Sau calibrate ({thr['calibration_method']})")]:
        fp, mp = calibration_curve(y_te, prob, n_bins=10, strategy="quantile")
        ax.plot(mp, fp, marker="o", label=lab)
    ax.plot([0, 1], [0, 1], "--", color="grey")
    ax.set_xlabel("Xác suất dự đoán (trung bình bin)")
    ax.set_ylabel("Tỷ lệ yes thực")
    ax.set_title("Calibration curve — refit train80 (test theo thời gian)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig27_calibration80.png", dpi=150)
    plt.close(fig)
    print("saved fig27_calibration80.png")

    # ---------- Phân tích lỗi theo nhóm + theo quý trên test (B4) ----------
    X_te_i = X_te.reset_index(drop=True)
    y_te_i = y_te.reset_index(drop=True)
    for col in ["month", "job", "poutcome"]:
        t = error_by_group(X_te_i, y_te_i, p_test, pred, col)
        t.to_csv(REPORTS_DIR / f"error_by_{col}_time80.csv", index=False)
        print(f"Lỗi theo {col} (test, refit): tệ nhất {t.iloc[0]['group']} "
              f"recall {t.iloc[0]['recall']}, tốt nhất {t.iloc[-1]['group']} "
              f"recall {t.iloc[-1]['recall']}")
    q = np.array_split(np.arange(len(y_te_i)), 4)
    qrows = []
    for i, idx in enumerate(q):
        yy, pp, pr = y_te_i.iloc[idx], p_test[idx], pred[idx]
        mm = compute_metrics(yy, pr, pp)
        qrows.append({"quarter": f"Q{i + 1}", "n": len(idx),
                      "yes_rate": round(float(yy.mean()), 4),
                      "roc_auc": round(mm["roc_auc"], 4), "pr_auc": round(mm["pr_auc"], 4),
                      "recall": round(mm["recall"], 4), "precision": round(mm["precision"], 4)})
    qdf = pd.DataFrame(qrows)
    qdf.to_csv(REPORTS_DIR / "drift_by_quarter80.csv", index=False)
    print("Drift theo quý trên test (refit):\n", qdf.to_string(index=False))
    err_month = pd.read_csv(REPORTS_DIR / "error_by_month_time80.csv")

    # ---------- fig29: drift theo quý + recall theo tháng ----------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))
    ax1.plot(qdf["quarter"], qdf["yes_rate"], marker="o", label="Tỷ lệ yes thực")
    ax1.plot(qdf["quarter"], qdf["recall"], marker="s", label="Recall")
    ax1.plot(qdf["quarter"], qdf["pr_auc"], marker="^", label="PR-AUC")
    ax1.set_ylim(0, 1.05)
    ax1.set_xlabel("Quý trong tập test theo thời gian (Q1 sớm nhất)")
    ax1.set_ylabel("Tỷ lệ / điểm số")
    ax1.set_title("Độ trôi trên test (refit train80): yes-rate, recall, PR-AUC theo quý")
    ax1.legend()
    ax2.bar(err_month["group"], err_month["recall"].fillna(0))
    ax2.set_ylim(0, 1.05)
    ax2.set_xlabel("Tháng (test theo thời gian)")
    ax2.set_ylabel("Recall lớp yes")
    ax2.set_title("Recall theo tháng trên test (refit train80)")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig29_drift80.png", dpi=150)
    plt.close(fig)
    print("saved fig29_drift80.png")

    # ---------- Gain / lift trên test ----------
    res, dec = gain_lift_table(y_te, p_test)
    dec.to_csv(REPORTS_DIR / "decile_table_time80.csv")
    top20 = float(res[res["called_pct"] <= 0.20]["captured"].iloc[-1])
    print(f"Top 20% bắt được {top20 * 100:.1f}% số người đồng ý (test, refit).")
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(res["called_pct"], res["captured"], label="Mô hình cuối (refit train80)")
    ax.plot([0, 1], [0, 1], "--", label="Ngẫu nhiên")
    ax.set_xlabel("% khách được gọi (xếp theo điểm)")
    ax.set_ylabel("% khách đồng ý bắt được")
    ax.set_title("Gain chart — refit train80 (test theo thời gian)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig30_gain_chart80.png", dpi=150)
    plt.close(fig)
    print("saved fig30_gain_chart80.png")

    # ---------- B5: đường lợi nhuận theo % gọi, kèm bootstrap ----------
    y = np.asarray(y_te)
    order = np.argsort(-p_test)
    y_sorted = y[order]
    grid = np.linspace(0.05, 1.0, 20)
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
    (REPORTS_DIR / "cost_benefit_curves_time80.json").write_text(
        json.dumps(curves, ensure_ascii=False, indent=2), encoding="utf-8")
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, c in curves.items():
        x = np.array(c["called_pct"]) * 100
        ax.plot(x, c["profit_mean"], marker="o", ms=3, label=label)
        ax.fill_between(x, c["profit_lo"], c["profit_hi"], alpha=0.15)
    ax.axhline(0, color="grey", lw=0.8)
    ax.set_xlabel("% khách được gọi (xếp theo điểm)")
    ax.set_ylabel("Lợi nhuận (đơn vị tiền tệ giả định)")
    ax.set_title("Lợi nhuận theo % khách được gọi — refit train80 "
                 "(dải mờ là CI 95% bootstrap)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig28_profit80.png", dpi=150)
    plt.close(fig)
    print("saved fig28_profit80.png")
    opt = {}
    for label, c in curves.items():
        i = int(np.argmax(c["profit_mean"]))
        opt[label] = {"best_called_pct": round(float(c["called_pct"][i]) * 100, 1),
                      "profit_mean": c["profit_mean"][i]}
    print("Điểm tối ưu lợi nhuận trên test (refit):", opt)
    summary["cost_benefit_optimum_test"] = opt
    summary["top20_capture_test"] = round(top20, 4)
    summary["drift_by_quarter"] = qdf.to_dict("records")
    (REPORTS_DIR / "final_summary_time80.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---------- B2: permutation importance + SHAP TRÊN TRAIN80 (in-sample) ----------
    # Lưu ý: validation đã gộp vào huấn luyện nên không còn tập validation
    # riêng; các số này là in-sample, chỉ dùng để tham chiếu, không phải
    # bằng chứng độc lập về khả năng tổng quát hóa.
    print("Permutation importance trên train80 (in-sample, n_repeats=3) ...")
    perm = permutation_importance(cal, X80, y80, n_repeats=3, random_state=SEED,
                                  scoring="average_precision", n_jobs=-1)
    names = list(X80.columns)
    imp = pd.DataFrame({"feature": names,
                        "importance": perm.importances_mean,
                        "std": perm.importances_std}).sort_values(
                            "importance", ascending=False)
    imp.to_csv(REPORTS_DIR / "permutation_importance_train80.csv", index=False)
    fig, ax = plt.subplots(figsize=(8, 5))
    top = imp.head(12).iloc[::-1]
    ax.barh(top["feature"], top["importance"], xerr=top["std"])
    ax.set_xlabel("Giảm PR-AUC khi hoán vị (train80, in-sample)")
    ax.set_title("Permutation importance — top 12 (train80, in-sample)")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig31_permutation_train80.png", dpi=150)
    plt.close(fig)
    print("saved fig31_permutation_train80.png | top 3:", imp.head(3)["feature"].tolist())

    print("SHAP trên train80 (mẫu 500 dòng, in-sample) ...")
    try:
        import shap
        rng2 = np.random.default_rng(SEED)
        sample_idx = rng2.choice(len(X80), size=min(500, len(X80)), replace=False)
        Xs = X80.iloc[sample_idx]
        pipe = base  # mô hình gốc fit trên toàn bộ train80 (không phải fold CV)
        clf = pipe.named_steps["clf"]
        x_prep = pipe.named_steps["prep"].transform(pipe.named_steps["fe"].transform(Xs))
        if clf.__class__.__name__ in ("RandomForestClassifier",
                                      "HistGradientBoostingClassifier",
                                      "LGBMClassifier", "DecisionTreeClassifier",
                                      "XGBClassifier"):
            vals = shap.TreeExplainer(clf).shap_values(x_prep)
            if isinstance(vals, list):
                vals = vals[1]
            elif getattr(vals, "ndim", 2) == 3:
                vals = vals[:, :, 1]
            mean_abs = np.abs(np.asarray(vals)).mean(axis=0)
        else:
            mean_abs = np.abs(clf.coef_[0]) * np.asarray(x_prep).std(axis=0)
        shap_df = pd.DataFrame({"feature": list(feature_names_out(pipe)),
                                "mean_abs_shap": mean_abs}).sort_values(
            "mean_abs_shap", ascending=False)
        shap_df.to_csv(REPORTS_DIR / "shap_importance_train80.csv", index=False)
        fig, ax = plt.subplots(figsize=(8, 5))
        top = shap_df.head(12).iloc[::-1]
        ax.barh(top["feature"], top["mean_abs_shap"])
        ax.set_xlabel("Mean |SHAP value| (train80, in-sample)")
        ax.set_title("SHAP — |SHAP| trung bình, top 12 (train80, in-sample)")
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / "fig32_shap_train80.png", dpi=150)
        plt.close(fig)
        print("saved fig32_shap_train80.png | top 3:", shap_df.head(3)["feature"].tolist())
    except Exception as e:
        print("SHAP bỏ qua:", type(e).__name__, e)

    # ---------- Metadata mô hình (đầy đủ, ghi đè bản tạm của 08b) ----------
    sha = hashlib.sha256(RAW_PATH.read_bytes()).hexdigest()[:16]
    meta = {"model": final_name, "best_params": cal_info["best_params"],
            "calibration_method": thr["calibration_method"],
            "threshold_cost_benefit": round(thr_cb, 4),
            "threshold_f1_oof": thr_f1,
            "trained_at_utc": datetime.now(timezone.utc).isoformat(),
            "scikit_learn": sklearn.__version__,
            "data_file": RAW_PATH.name, "data_sha256_prefix": sha,
            "n_train": int(len(X80)), "n_val": int(len(X_va)), "n_test": int(len(X_te)),
            "train_window": "train80 (train60+val, refit sau khi chọn trên validation)",
            "test_metrics_cb_threshold": summary["test_metrics_cb_threshold"],
            "test_metrics_f1_threshold": summary["test_metrics_f1_threshold"],
            "bootstrap_ci_95": ci,
            "brier_test_raw": round(brier_raw, 4),
            "brier_test_calibrated": round(brier_cal, 4),
            "top20_capture_test": round(top20, 4),
            "cost_benefit_optimum_test": opt,
            "note": "Mô hình chọn trên validation rồi refit trên train+val; "
                    "calibrate + ngưỡng chọn trên OOF của train80; test theo "
                    "thời gian (20% cuối). Đính chính 19/11/2026: test đã được đánh giá nhiều lần trong dự án (xem mục 4.2.8 của báo cáo)."}
    (MODELS_DIR / "model_metadata.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Đã lưu models/model_metadata.json")

    # ---------- Nhật ký thực nghiệm (08b + 09b) ----------
    append_log([
        {"id": "EXP_T2_REFIT80_CAL", "model": "Tree (refit train80, calibrate OOF)",
         "cv": "3-fold (OOF train80)", "roc": cal_info["roc_auc_oof_train80"],
         "pr": cal_info["pr_auc_oof_train80"], "f1": "", "rec": "", "prec": "",
         "notes": f"Refit Tree trên train80 (32.940 dòng); Brier OOF isotonic "
                  f"{cal_info['brier_oof_train80']['isotonic']} vs sigmoid "
                  f"{cal_info['brier_oof_train80']['sigmoid']} -> chọn "
                  f"{cal_info['method']}"},
        {"id": "EXP_T2_REFIT80_THRESH", "model": "Tree (refit train80, ngưỡng OOF)",
         "cv": "3-fold (OOF train80)", "roc": "", "pr": "", "f1": "", "rec": "",
         "prec": "",
         "notes": f"Ngưỡng trên OOF train80: CB={thr_cb:.4f} (C=1,R=15), "
                  f"F1={thr_f1:.4f}, recall>=0,7={thr['threshold_recall70']:.4f}"},
        {"id": "EXP_T2_FINAL80", "model": "Tree (refit train80, test theo thời gian)",
         "cv": "time-test-20%", "roc": m["roc_auc"], "pr": m["pr_auc"],
         "f1": m["f1"], "rec": m["recall"], "prec": m["precision"],
         "notes": f"Đánh giá trên test (ngưỡng CB {thr_cb:.4f}); "
                  f"TN={m['tn']} FP={m['fp']} FN={m['fn']} TP={m['tp']}; "
                  f"Brier {brier_raw:.4f}->{brier_cal:.4f}; top20 bắt {top20 * 100:.1f}% yes"},
        {"id": "EXP_T2_FINAL80_F1", "model": "Tree (refit train80, test, ngưỡng F1)",
         "cv": "time-test-20%", "roc": m_f1["roc_auc"], "pr": m_f1["pr_auc"],
         "f1": m_f1["f1"], "rec": m_f1["recall"], "prec": m_f1["precision"],
         "notes": f"Tham chiếu test ở ngưỡng F1 (OOF train80) {thr_f1:.4f}; "
                  f"TN={m_f1['tn']} FP={m_f1['fp']} FN={m_f1['fn']} TP={m_f1['tp']}"},
    ])


if __name__ == "__main__":
    main()

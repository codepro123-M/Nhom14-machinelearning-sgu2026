# -*- coding: utf-8 -*-
"""Giai đoạn 7c (v2): Đánh giá trên test theo thời gian.

- Mô hình đã calibrate (scripts/08) + ngưỡng chi phí-lợi ích, đánh giá trên
  test20 (20% cuối theo thời gian) — trong lần chạy này test không dùng
  để tuning, calibrate hay chọn ngưỡng; tuy nhiên test đã được mở nhiều
  lần trong toàn bộ dự án (đính chính 19/11/2026, xem báo cáo mục 4.2.8).
- Bootstrap CI 95% cho ROC-AUC / PR-AUC / F1.
- Brier trước/sau calibrate TRÊN TEST (so sánh trung thực).
- Phân tích lỗi theo nhóm + theo quý trên test (B4).
- Gain/lift trên test (RQ5).
- B2: permutation importance và SHAP tính trên VALIDATION (không dùng test).
- Lưu models/model_metadata.json (ngày huấn luyện, phiên bản thư viện,
  hash dữ liệu, metric) cạnh mô hình.
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


def main():
    df = load_clean()
    X_tr, X_va, X_te, y_tr, y_va, y_te = time_split_3way(df)
    cal = joblib.load(MODELS_DIR / "model_calibrated.joblib")
    base = joblib.load(MODELS_DIR / "model.joblib")
    thr = json.loads((MODELS_DIR / "threshold.json").read_text(encoding="utf-8"))
    cal_info = json.loads((REPORTS_DIR / "calibration_comparison.json").read_text(encoding="utf-8"))
    final_name = cal_info["final_model"]
    thr_cb = thr["threshold_cost_benefit"]
    print(f"Mô hình cuối: {final_name} | calibrate: {thr['calibration_method']} | "
          f"ngưỡng chi phí-lợi ích: {thr_cb:.4f}")

    # ---------- Đánh giá trên test ----------
    p_test = cal.predict_proba(X_te)[:, 1]
    p_raw = base.predict_proba(X_te)[:, 1]
    pred = (p_test >= thr_cb).astype(int)
    m = compute_metrics(y_te, pred, p_test)
    ci = bootstrap_ci(y_te, p_test, pred)
    brier_raw = float(brier_score_loss(y_te, p_raw))
    brier_cal = float(brier_score_loss(y_te, p_test))
    print("Kết quả test theo thời gian (ngưỡng chi phí-lợi ích):",
          {k: round(v, 4) for k, v in m.items() if isinstance(v, float)})
    print("Bootstrap CI 95%:", ci)
    print(f"Brier trên test: gốc {brier_raw:.4f} -> đã calibrate {brier_cal:.4f}")

    # Tham chiếu thêm: ngưỡng F1 trên validation (không phải quyết định mới —
    # ngưỡng đã chọn trên validation, chỉ đánh giá thêm). Cần vì ngưỡng CB
    # trên test gọi gần như toàn bộ khách nên không đọc được khả năng phân
    # biệt của mô hình ở một điểm vận hành thực tế.
    thr_f1 = thr["threshold_f1"]
    pred_f1 = (p_test >= thr_f1).astype(int)
    m_f1 = compute_metrics(y_te, pred_f1, p_test)
    print("Tham chiếu — test ở ngưỡng F1:", {k: round(v, 4) for k, v in m_f1.items()
                                              if isinstance(v, float)})

    summary = {"final_model": final_name, "selection_reason": cal_info["selection_reason"],
               "best_params": cal_info["best_params"],
               "calibration_method": thr["calibration_method"],
               "threshold_cost_benefit": round(thr_cb, 4),
               "threshold_f1_val": thr["threshold_f1"],
               "threshold_recall70_val": thr["threshold_recall70"],
               "test_metrics_cb_threshold": {k: (round(float(v), 4) if isinstance(v, float) else v)
                                             for k, v in m.items()},
               "test_metrics_f1_threshold": {k: (round(float(v), 4) if isinstance(v, float) else v)
                                             for k, v in m_f1.items()},
               "bootstrap_ci_95": ci,
               "brier_test_raw": round(brier_raw, 4),
               "brier_test_calibrated": round(brier_cal, 4)}
    (REPORTS_DIR / "final_summary_time.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---------- Hình ROC / PR / confusion trên test theo thời gian ----------
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    RocCurveDisplay.from_predictions(y_te, p_test, ax=axes[0])
    axes[0].set_title(f"ROC curve - {final_name} (test theo thời gian)")
    PrecisionRecallDisplay.from_predictions(y_te, p_test, ax=axes[1])
    axes[1].set_title(f"PR curve - {final_name} (test theo thời gian)")
    ConfusionMatrixDisplay.from_predictions(y_te, pred, display_labels=["no", "yes"], ax=axes[2])
    axes[2].set_title(f"Confusion matrix - {final_name} (test, ngưỡng CB={thr_cb:.3f})")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig17_final_time_curves.png", dpi=150)
    plt.close(fig)
    print("saved fig17_final_time_curves.png")

    # ---------- Calibration curve trên test ----------
    fig, ax = plt.subplots(figsize=(6, 5))
    for prob, lab in [(p_raw, "Mô hình gốc"), (p_test, f"Sau calibrate ({thr['calibration_method']})")]:
        fp, mp = calibration_curve(y_te, prob, n_bins=10, strategy="quantile")
        ax.plot(mp, fp, marker="o", label=lab)
    ax.plot([0, 1], [0, 1], "--", color="grey")
    ax.set_xlabel("Xác suất dự đoán (trung bình bin)")
    ax.set_ylabel("Tỷ lệ yes thực")
    ax.set_title("Calibration curve (test theo thời gian)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig18_calibration_time.png", dpi=150)
    plt.close(fig)
    print("saved fig18_calibration_time.png")

    # ---------- Phân tích lỗi theo nhóm + theo quý trên test (B4) ----------
    X_te_i = X_te.reset_index(drop=True)
    y_te_i = y_te.reset_index(drop=True)
    for col in ["month", "job", "poutcome"]:
        t = error_by_group(X_te_i, y_te_i, p_test, pred, col)
        t.to_csv(REPORTS_DIR / f"error_by_{col}_time.csv", index=False)
        print(f"Lỗi theo {col} (test): tệ nhất {t.iloc[0]['group']} "
              f"recall {t.iloc[0]['recall']}, tốt nhất {t.iloc[-1]['group']} "
              f"recall {t.iloc[-1]['recall']}")
    # Chia test20 thành 4 quý theo thứ tự thời gian
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
    qdf.to_csv(REPORTS_DIR / "drift_by_quarter.csv", index=False)
    print("Drift theo quý trên test:\n", qdf.to_string(index=False))

    # ---------- Gain / lift trên test ----------
    res, dec = gain_lift_table(y_te, p_test)
    dec.to_csv(REPORTS_DIR / "decile_table_time.csv")
    top20 = float(res[res["called_pct"] <= 0.20]["captured"].iloc[-1])
    print(f"Top 20% bắt được {top20 * 100:.1f}% số người đồng ý (test theo thời gian).")
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(res["called_pct"], res["captured"], label="Mô hình cuối")
    ax.plot([0, 1], [0, 1], "--", label="Ngẫu nhiên")
    ax.set_xlabel("% khách được gọi (xếp theo điểm)")
    ax.set_ylabel("% khách đồng ý bắt được")
    ax.set_title("Gain chart (test theo thời gian)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig22_gain_chart_time.png", dpi=150)
    plt.close(fig)
    print("saved fig22_gain_chart_time.png")

    # ---------- B2: permutation importance + SHAP TRÊN VALIDATION ----------
    # Lưu ý: permutation_importance hoán vị các cột THÔ của X_va nên tên đặc
    # trưng phải là cột gốc, không phải tên sau bước prep (số lượng khác nhau).
    print("Permutation importance trên validation (n_repeats=3) ...")
    perm = permutation_importance(cal, X_va, y_va, n_repeats=3, random_state=SEED,
                                  scoring="average_precision", n_jobs=-1)
    names = list(X_va.columns)
    imp = pd.DataFrame({"feature": names,
                        "importance": perm.importances_mean,
                        "std": perm.importances_std}).sort_values(
                            "importance", ascending=False)
    imp.to_csv(REPORTS_DIR / "permutation_importance_val.csv", index=False)
    fig, ax = plt.subplots(figsize=(8, 5))
    top = imp.head(12).iloc[::-1]
    ax.barh(top["feature"], top["importance"], xerr=top["std"])
    ax.set_xlabel("Giảm PR-AUC khi hoán vị (validation)")
    ax.set_title("Permutation importance — top 12 (validation)")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig23_permutation_val.png", dpi=150)
    plt.close(fig)
    print("saved fig23_permutation_val.png | top 3:", imp.head(3)["feature"].tolist())

    print("SHAP trên validation (mẫu 500 dòng) ...")
    try:
        import shap
        rng = np.random.default_rng(SEED)
        sample_idx = rng.choice(len(X_va), size=min(500, len(X_va)), replace=False)
        Xs = X_va.iloc[sample_idx]
        pipe = unwrap(cal)
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
        else:  # tuyến tính: dùng |hệ số| * độ lệch chuẩn đặc trưng
            mean_abs = np.abs(clf.coef_[0]) * np.asarray(x_prep).std(axis=0)
        shap_df = pd.DataFrame({"feature": list(feature_names_out(pipe)),
                                "mean_abs_shap": mean_abs}).sort_values(
            "mean_abs_shap", ascending=False)
        shap_df.to_csv(REPORTS_DIR / "shap_importance_val.csv", index=False)
        fig, ax = plt.subplots(figsize=(8, 5))
        top = shap_df.head(12).iloc[::-1]
        ax.barh(top["feature"], top["mean_abs_shap"])
        ax.set_xlabel("Mean |SHAP value| (validation)")
        ax.set_title("SHAP — |SHAP| trung bình, top 12 (validation)")
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / "fig24_shap_val.png", dpi=150)
        plt.close(fig)
        print("saved fig24_shap_val.png | top 3:", shap_df.head(3)["feature"].tolist())
    except Exception as e:
        print("SHAP bỏ qua:", type(e).__name__, e)

    # ---------- Metadata mô hình ----------
    sha = hashlib.sha256(RAW_PATH.read_bytes()).hexdigest()[:16]
    meta = {"model": final_name, "best_params": cal_info["best_params"],
            "calibration_method": thr["calibration_method"],
            "threshold_cost_benefit": round(thr_cb, 4),
            "threshold_f1_val": thr["threshold_f1"],
            "trained_at_utc": datetime.now(timezone.utc).isoformat(),
            "scikit_learn": sklearn.__version__,
            "data_file": RAW_PATH.name, "data_sha256_prefix": sha,
            "n_train": int(len(X_tr)), "n_val": int(len(X_va)), "n_test": int(len(X_te)),
            "test_metrics_cb_threshold": summary["test_metrics_cb_threshold"],
            "bootstrap_ci_95": ci,
            "brier_test_raw": round(brier_raw, 4),
            "brier_test_calibrated": round(brier_cal, 4),
            "top20_capture_test": round(top20, 4),
            "note": "Test theo thời gian (20% cuối). Đính chính 19/11/2026: test đã được đánh giá nhiều lần trong dự án (xem mục 4.2.8 của báo cáo); "
                    "ngưỡng và calibrate chọn trên validation."}
    (MODELS_DIR / "model_metadata.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Đã lưu models/model_metadata.json")


if __name__ == "__main__":
    main()

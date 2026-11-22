# -*- coding: utf-8 -*-
"""Đánh giá mô hình (theo Cẩm nang, giai đoạn 4.2, 5.2, 7.1, 7.3, 9.5)."""
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (accuracy_score, average_precision_score,
                             confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score,
                             precision_recall_curve)
from sklearn.model_selection import cross_val_predict, cross_validate

SCORING = {"roc_auc": "roc_auc", "pr_auc": "average_precision",
           "f1": "f1", "recall": "recall", "precision": "precision"}


def _single_thread(pipe):
    """Trả về bản sao của pipeline với clf chạy đơn luồng.

    Lý do: cross_validate / RandomizedSearchCV / cross_val_predict đã song song
    hóa theo fold (n_jobs=-1). Nếu estimator bên trong cũng dùng n_jobs=-1 thì
    song song lồng nhau làm quá tải CPU (máy chỉ có 1,5 CPU) và chạy chậm hơn.
    """
    pipe = clone(pipe)
    try:
        pipe.set_params(clf__n_jobs=1)
    except Exception:
        pass
    return pipe


def cv_report(models, X, y, cv, n_jobs=-1):
    """So sánh các mô hình bằng CV: mean/std qua fold + điểm train (phát hiện overfit)."""
    rows = {}
    for name, pipe in models.items():
        res = cross_validate(_single_thread(pipe), X, y, cv=cv, scoring=SCORING,
                             return_train_score=True, n_jobs=n_jobs)
        row = {}
        for m in SCORING:
            row[m + "_cv"] = res["test_" + m].mean()
            row[m + "_std"] = res["test_" + m].std()
            row[m + "_train"] = res["train_" + m].mean()
        row["fit_time_s"] = res["fit_time"].mean()
        rows[name] = row
    return pd.DataFrame(rows).T.round(4)


def compute_metrics(y_true, y_pred, y_proba):
    """Bộ metric đầy đủ cho một tập dự đoán (dùng ở đánh giá cuối)."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "roc_auc": roc_auc_score(y_true, y_proba),
        "pr_auc": average_precision_score(y_true, y_proba),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def pick_threshold(model, X, y, cv, goal="f1", min_recall=0.7):
    """Chọn ngưỡng bằng dự đoán out-of-fold TRÊN TRAIN (không chạm test).

    goal='f1': ngưỡng tối ưu F1. goal='recall': ngưỡng lớn nhất vẫn giữ
    recall >= min_recall (ưu tiên không bỏ sót khách tiềm năng).
    """
    oof = cross_val_predict(_single_thread(model), X, y, cv=cv, method="predict_proba",
                            n_jobs=-1)[:, 1]
    prec, rec, thr = precision_recall_curve(y, oof)
    prec, rec = prec[:-1], rec[:-1]
    if goal == "f1":
        f1 = 2 * prec * rec / (prec + rec + 1e-12)
        return float(thr[np.argmax(f1)]), oof
    ok = rec >= min_recall
    return float(thr[ok][np.argmax(prec[ok])]), oof


def bootstrap_ci(y_true, y_proba, y_pred, n_boot=1000, seed=42):
    """Khoảng tin cậy 95% (bootstrap) cho ROC-AUC, PR-AUC, F1 trên test."""
    rng = np.random.default_rng(seed)
    yt = np.asarray(y_true)
    boot = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(yt), len(yt))
        if yt[idx].sum() == 0:
            continue
        boot.append((roc_auc_score(yt[idx], y_proba[idx]),
                     average_precision_score(yt[idx], y_proba[idx]),
                     f1_score(yt[idx], y_pred[idx], zero_division=0)))
    ci = np.percentile(boot, [2.5, 97.5], axis=0)
    return {"roc_auc": ci[:, 0].round(4).tolist(),
            "pr_auc": ci[:, 1].round(4).tolist(),
            "f1": ci[:, 2].round(4).tolist()}


def gain_lift_table(y_true, y_proba, n_bins=10):
    """Bảng decile + đường gain/lift: xếp khách theo điểm, gọi từ trên xuống."""
    res = pd.DataFrame({"y": np.asarray(y_true), "p": np.asarray(y_proba)})
    res = res.sort_values("p", ascending=False).reset_index(drop=True)
    res["called_pct"] = (res.index + 1) / len(res)
    res["captured"] = res["y"].cumsum() / res["y"].sum()
    res["decile"] = pd.qcut(res.index, n_bins, labels=False) + 1
    dec = res.groupby("decile")["y"].agg(["sum", "count"])
    dec["cum_capture"] = dec["sum"].cumsum() / dec["sum"].sum()
    dec["lift"] = (dec["sum"] / dec["count"]) / res["y"].mean()
    dec = dec.rename(columns={"sum": "n_yes", "count": "n_called"})
    return res, dec.round(4)

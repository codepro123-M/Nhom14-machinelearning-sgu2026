# -*- coding: utf-8 -*-
"""Giai đoạn 7e (đính chính 19/11/2026): Đánh giá đối chứng rolling-origin
so sánh các ứng viên mô hình cuối mà KHÔNG DÙNG TEST.

Bối cảnh (xem báo cáo mục 4.2.8): mô hình cuối (Tree refit80) được giữ vì
thắng ở vòng chọn trên validation, nhưng vòng chọn đó dùng TimeSeriesSplit(3)
trên train60 quá nhiễu (PR-AUC 0,068-0,077 cho cả 6 mô hình) và quy tắc 1-SE
đã chọn LogReg C=0,001 (chỉ 0,5480 trên test). Trong khi đó nhật ký cho thấy
LogReg mặc định (0,7015) và RF tham số v1 (0,7079) đều hơn Tree cuối trên
chính test đó — nhưng đổi mô hình theo điểm test là vi phạm quy định.

Thí nghiệm này trả lời câu hỏi "mô hình nào tốt hơn" bằng một phép đánh giá
có đối chứng, hoàn toàn không chạm test:

- Dữ liệu: train80 = train60 + validation (32.940 dòng, giữ nguyên thứ tự
  thời gian). Test (8.236 dòng) được time_split_3way trả về nhưng script
  KHÔNG dùng tới (gán vào biến _ và không tham chiếu sau đó).
- Phương pháp: TimeSeriesSplit(n_splits=4) rolling-origin trên train80 —
  cửa sổ huấn luyện mở rộng dần, cửa sổ kiểm chứng là khối tiếp theo.
- Ứng viên (cùng cách dựng estimator như scripts/07_time_design.py,
  class_weight balanced / balanced_subsample):
    Tree_tuned      : DecisionTree, best_params_time.json (depth 12, leaf 100, split 50)
    LogReg_default  : LogisticRegression C=1,0 (mặc định)
    LogReg_tuned    : LogisticRegression C=0,001 (best_params_time.json, chọn bởi 1-SE)
    RF_time_tuned   : RandomForest, best_params_time.json (600 cây, depth 4, leaf 10, f 0,3)
    RF_v1           : RandomForest, tham số đã tune ở vòng stratified v1 (400 cây, depth 8, leaf 5, f 0,3)
- Độ đo chính: PR-AUC (average_precision), tham chiếu ROC-AUC, tính trên từng
  gốc kiểm chứng rồi trung bình.
- Quy tắc chọn ĐÃ ĐĂNG KÝ TRƯỚC khi chạy: mô hình thắng là mô hình có PR-AUC
  trung bình cao nhất; chênh lệch chỉ coi là rõ ràng nếu lớn hơn 1 sai số
  chuẩn (SE) của hiệu trung bình theo cặp trên 4 gốc (mọi mô hình dùng chung
  các gốc nên so sánh theo cặp). Nếu không có chênh lệch rõ ràng, kết luận là
  "không phân biệt được" và giữ Tree (kèm lý do đã nêu ở mục 4.2.8).

Đầu ra: reports/rolling_origin_train80.csv (từng gốc) + 5 dòng
EXP_ROLL_* nối vào experiments/experiment_log.csv + tóm tắt in ra console.
"""
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.config import PROCESSED_DIR, REPORTS_DIR, SEED  # noqa: E402
from src.data import load_clean, time_split_3way  # noqa: E402
from src.features import build_pipeline  # noqa: E402

N_SPLITS = 4


def main():
    df = load_clean()
    X_tr, X_va, _X_te, y_tr, y_va, _y_te = time_split_3way(df)
    X80 = pd.concat([X_tr, X_va], ignore_index=True)
    y80 = pd.concat([y_tr, y_va], ignore_index=True)
    assert len(X80) == 32940, len(X80)
    print(f"train80: {len(X80)} dong (train60 {len(X_tr)} + val {len(X_va)}); "
          f"test KHONG duoc dung trong thi nghiem nay.")

    best_time = json.loads((REPORTS_DIR / "best_params_time.json").read_text(encoding="utf-8"))
    tree_p = best_time["Tree"]
    rf_time_p = best_time["RF"]

    candidates = {
        "Tree_tuned": (DecisionTreeClassifier(class_weight="balanced", random_state=SEED),
                       {"clf__max_depth": tree_p["clf__max_depth"],
                        "clf__min_samples_leaf": tree_p["clf__min_samples_leaf"],
                        "clf__min_samples_split": tree_p["clf__min_samples_split"],
                        "clf__criterion": tree_p["clf__criterion"]}),
        "LogReg_default": (LogisticRegression(max_iter=3000, class_weight="balanced",
                                              random_state=SEED),
                           {"clf__C": 1.0}),
        "LogReg_tuned": (LogisticRegression(max_iter=3000, class_weight="balanced",
                                            random_state=SEED),
                         {"clf__C": 0.001}),
        "RF_time_tuned": (RandomForestClassifier(class_weight="balanced_subsample",
                                                 n_jobs=1, random_state=SEED),
                          {"clf__n_estimators": rf_time_p["clf__n_estimators"],
                           "clf__max_depth": rf_time_p["clf__max_depth"],
                           "clf__min_samples_leaf": rf_time_p["clf__min_samples_leaf"],
                           "clf__max_features": rf_time_p["clf__max_features"]}),
        "RF_v1": (RandomForestClassifier(class_weight="balanced_subsample",
                                         n_jobs=1, random_state=SEED),
                  {"clf__n_estimators": 400, "clf__max_depth": 8,
                   "clf__min_samples_leaf": 5, "clf__max_features": 0.3}),
    }

    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    rows = []
    for name, (clf, params) in candidates.items():
        pipe = build_pipeline(clf).set_params(**params)
        for fold, (tr_idx, va_idx) in enumerate(tscv.split(X80), 1):
            pipe.fit(X80.iloc[tr_idx], y80.iloc[tr_idx])
            p = pipe.predict_proba(X80.iloc[va_idx])[:, 1]
            yv = y80.iloc[va_idx]
            roc = roc_auc_score(yv, p)
            pr = average_precision_score(yv, p)
            rows.append({"model": name, "origin": fold,
                         "n_train": len(tr_idx), "n_val": len(va_idx),
                         "yes_rate_val": round(float(yv.mean()), 4),
                         "roc_auc": round(float(roc), 4),
                         "pr_auc": round(float(pr), 4)})
            print(f"  {name:14s} goc {fold}: n_train={len(tr_idx)} n_val={len(va_idx)} "
                  f"yes={yv.mean():.4f} ROC-AUC={roc:.4f} PR-AUC={pr:.4f}", flush=True)

    out = pd.DataFrame(rows)
    (REPORTS_DIR / "rolling_origin_train80.csv").write_text(
        out.to_csv(index=False), encoding="utf-8")

    print("\n=== Tom tat rolling-origin (train80, 4 goc) ===")
    summary = {}
    for name, g in out.groupby("model"):
        pr_mean, pr_std = g["pr_auc"].mean(), g["pr_auc"].std(ddof=1)
        roc_mean = g["roc_auc"].mean()
        se = pr_std / np.sqrt(len(g))
        summary[name] = (pr_mean, se)
        print(f"  {name:14s} PR-AUC {pr_mean:.4f} (std {pr_std:.4f}, SE {se:.4f}) "
              f"| ROC-AUC {roc_mean:.4f}")
    # So sanh theo cap voi mo hinh tot nhat
    best_name = max(summary, key=lambda k: summary[k][0])
    best_pr, best_se = summary[best_name]
    print(f"\nMo hinh thang theo PR-AUC trung binh: {best_name} ({best_pr:.4f})")
    for name, (pr_mean, se) in summary.items():
        if name == best_name:
            continue
        gap = best_pr - pr_mean
        # SE cua hieu theo cap (dung phuong sai cua hieu tren 4 goc)
        diff = (out[out["model"] == best_name].sort_values("origin")["pr_auc"].values
                - out[out["model"] == name].sort_values("origin")["pr_auc"].values)
        se_diff = diff.std(ddof=1) / np.sqrt(len(diff))
        verdict = "RO RANG" if gap > se_diff else "khong ro rang"
        print(f"  {best_name} - {name}: chenh {gap:.4f}, SE hieu {se_diff:.4f} -> {verdict}")

    # Ghi vao nhat ky thi nghiem
    logp = ROOT / "experiments" / "experiment_log.csv"
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    with open(logp, encoding="utf-8", newline="") as f:
        n_before = len(list(csv.reader(f))) - 1
    with open(logp, "a", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        for name, g in out.groupby("model"):
            pr_mean = round(float(g["pr_auc"].mean()), 4)
            roc_mean = round(float(g["roc_auc"].mean()), 4)
            w.writerow([f"EXP_ROLL_{name}", ts, name.replace("_", " ") + " (rolling-origin)",
                        "without_duration", "class_weight_balanced",
                        "4x TimeSeriesSplit rolling-origin (train80)",
                        roc_mean, pr_mean, "", "", "",
                        "Danh gia doi chung 19/11/2026 (xem bao cao muc 4.2.9): so sanh tren "
                        "train80 bang rolling-origin, KHONG dung test. PR-AUC trung binh "
                        f"tren 4 goc: {pr_mean}; ROC-AUC trung binh: {roc_mean}."])
    print(f"\nDa ghi 5 dong EXP_ROLL_* vao experiment_log.csv "
          f"(truoc: {n_before} dong du lieu).")


if __name__ == "__main__":
    main()

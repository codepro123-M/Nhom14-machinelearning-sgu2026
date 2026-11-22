# -*- coding: utf-8 -*-
"""Tổng hợp nhật ký thực nghiệm (experiments/experiment_log.csv) từ các file
kết quả do các script giai đoạn sinh ra. Chạy sau khi hoàn thành giai đoạn 9.
Mỗi dòng là một thực nghiệm có thể tái lập bằng code trong repo (SEED = 42).
"""
import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
OUT = ROOT / "experiments" / "experiment_log.csv"


def load_json(name):
    return json.loads((REPORTS / name).read_text(encoding="utf-8"))


def main():
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    rows = []

    def add(exp_id, model, scenario, strategy, roc=None, pr=None, f1=None,
            recall=None, precision=None, notes="", cv_folds="5-Stratified"):
        rows.append({"experiment_id": exp_id, "timestamp": ts, "model_name": model,
                     "features_strategy": scenario, "sampling_strategy": strategy,
                     "cv_folds": cv_folds,
                     "roc_auc": None if roc is None else round(float(roc), 4),
                     "pr_auc": None if pr is None else round(float(pr), 4),
                     "f1_score": None if f1 is None else round(float(f1), 4),
                     "recall_yes": None if recall is None else round(float(recall), 4),
                     "precision_yes": None if precision is None else round(float(precision), 4),
                     "notes": notes})

    cv0 = pd.read_csv(REPORTS / "cv_default_models.csv", index_col=0)
    for i, (name, r) in enumerate(cv0.sort_values("pr_auc_cv", ascending=False).iterrows(), 1):
        add(f"EXP_{i:03d}", name, "without_duration",
            "class_weight_balanced" if "balanced" in name.lower() else "None",
            r["roc_auc_cv"], r["pr_auc_cv"], r["f1_cv"], r["recall_cv"], r["precision_cv"],
            "So sánh mặc định bằng 5-Fold CV trên train")

    cvt = pd.read_csv(REPORTS / "cv_tuned_models.csv", index_col=0)
    best_params = load_json("best_params.json")
    for i, (name, r) in enumerate(cvt.iterrows(), 1):
        add(f"EXP_T{i}", name + " (tuned)", "without_duration", "class_weight_balanced",
            r["roc_auc_cv"], r["pr_auc_cv"], r["f1_cv"], r["recall_cv"], r["precision_cv"],
            f"RandomizedSearchCV, tối ưu PR-AUC; best_params={best_params.get(name, {})}")

    final = load_json("final_summary.json")
    m = final["test_metrics_f1_threshold"]
    add("EXP_FINAL", final["final_model"] + " (test)", "without_duration",
        "class_weight_balanced", m["roc_auc"], m["pr_auc"], m["f1"], m["recall"],
        m["precision"],
        f"Đánh giá cuối trên test, ngưỡng F1={final['threshold_f1']}; "
        f"CI95 PR-AUC {final['bootstrap_ci_95']['pr_auc']}")

    extra = load_json("extra_experiments.json")
    ab = extra["duration_ab_cv"]
    for key, label in [("A_khong_duration", "EXP_DUR_A"), ("B_co_duration", "EXP_DUR_B")]:
        v = ab[key]
        add(label, "RF (tuned)", "with_duration" if "B_" in key else "without_duration",
            "class_weight_balanced", v["roc_auc_cv"], v["pr_auc_cv"], None, None, None,
            "Thí nghiệm duration (RQ4): " + label)
    for j, (label, v) in enumerate(extra["ablation_cv"].items(), 1):
        add(f"EXP_ABL{j}", f"LogReg_balanced [{label}]", "without_duration",
            "class_weight_balanced", v["roc_auc_cv"], v["pr_auc_cv"], v["f1_cv"],
            v["recall_cv"], None, "Ablation theo nhóm biến (RQ3)")
    tsplit = extra["time_split"]
    for k, v in tsplit.items():
        if isinstance(v, dict):
            add("EXP_TIME", k, "time_split", "class_weight_balanced",
                v["roc_auc"], v["pr_auc"], None, None, None,
                "Chia theo thời gian: train 80% đầu, test 20% cuối")

    # ---------- v2: thiết kế theo thời gian (sau góp ý 2026-11-14) ----------
    if (REPORTS / "time_tuning_compare.csv").exists():
        cmp = pd.read_csv(REPORTS / "time_tuning_compare.csv", index_col=0)
        bp_time = load_json("best_params_time.json")
        sel = load_json("time_selection.json")
        for i, (name, r) in enumerate(cmp.sort_values(
                "pr_auc_val", ascending=False).iterrows(), 1):
            add(f"EXP_T2_{i:02d}", name + " (tuned, time)", "without_duration",
                "class_weight_balanced", r["roc_auc_val"], r["pr_auc_val"],
                None, None, None,
                f"Tuning công bằng n_iter=40, TimeSeriesSplit(3) trên train60; "
                f"PR-AUC CV={r['pr_auc_cv_ts']}; best_params={bp_time.get(name, {})}",
                cv_folds="3-TimeSeriesSplit")
        add("EXP_T2_SEL", sel["final_model"] + " (chọn trên validation)",
            "without_duration", "class_weight_balanced",
            float(cmp.loc[sel["final_model"], "roc_auc_val"]),
            float(cmp.loc[sel["final_model"], "pr_auc_val"]),
            None, None, None, sel["selection_reason"], cv_folds="validation-20%")
    if (REPORTS / "final_summary_time.json").exists():
        ft = load_json("final_summary_time.json")
        mt = ft["test_metrics_cb_threshold"]
        add("EXP_TIME_FINAL", ft["final_model"] + " (test theo thời gian)",
            "without_duration", "class_weight_balanced",
            mt["roc_auc"], mt["pr_auc"], mt["f1"], mt["recall"], mt["precision"],
            f"Đánh giá trên test 20% cuối; ngưỡng chi phí-lợi ích "
            f"{ft['threshold_cost_benefit']}; calibrate {ft['calibration_method']}; "
            f"CI95 PR-AUC {ft['bootstrap_ci_95']['pr_auc']}; "
            f"Brier test {ft['brier_test_raw']}->{ft['brier_test_calibrated']}",
            cv_folds="time-test-20%")
    if (REPORTS / "extra_experiments_time.json").exists():
        ext2 = load_json("extra_experiments_time.json")
        ab2 = ext2["duration_ab_time_cv"]
        for key, label in [("A_khong_duration", "EXP_DUR_TIME_A"),
                           ("B_co_duration", "EXP_DUR_TIME_B")]:
            v = ab2[key]
            add(label, "final (tuned, time)", "with_duration" if "B_" in key else "without_duration",
                "class_weight_balanced", v["roc_auc_cv"], v["pr_auc_cv"], None, None, None,
                "Thí nghiệm duration trên thiết kế theo thời gian (RQ4)",
                cv_folds="3-TimeSeriesSplit")
        for j, (label, v) in enumerate(ext2["ablation_time_cv"].items(), 1):
            add(f"EXP_ABL_TIME{j}", f"LogReg_balanced [{label}] (time)",
                "without_duration", "class_weight_balanced",
                v["roc_auc_cv"], v["pr_auc_cv"], v["f1_cv"], v["recall_cv"], None,
                "Ablation theo nhóm biến trên thiết kế theo thời gian (RQ3)",
                cv_folds="3-TimeSeriesSplit")

    df = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"Đã ghi {len(df)} dòng vào {OUT}")
    print(df[["experiment_id", "model_name", "roc_auc", "pr_auc"]].to_string(index=False))


if __name__ == "__main__":
    main()

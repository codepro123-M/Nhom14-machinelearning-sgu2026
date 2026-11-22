# -*- coding: utf-8 -*-
"""Giai đoạn 8: Phân tích lỗi và diễn giải mô hình (Cẩm nang giai đoạn 8).

- Phân tích lỗi theo nhóm khách hàng (recall của lớp yes + số gọi nhầm).
- Hệ số Logistic Regression (odds ratio) - diễn giải tuyến tính.
- Permutation importance trên test (chỉ để diễn giải, không chọn mô hình).
- Cây nông max_depth=3 để minh họa luật kinh doanh.
- SHAP (nếu cài được): TreeExplainer trên mô hình cuối.
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
from sklearn.inspection import permutation_importance
from sklearn.tree import DecisionTreeClassifier, plot_tree

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import FIGURES_DIR, MODELS_DIR, REPORTS_DIR, SEED
from src.data import load_clean, stratified_split
from src.evaluate import compute_metrics
from src.features import build_pipeline, feature_names_out

plt.rcParams.update({"figure.dpi": 150, "font.size": 10})


def main():
    df = load_clean()
    X_train, X_test, y_train, y_test = stratified_split(df)
    final = joblib.load(MODELS_DIR / "model.joblib")
    try:
        final.set_params(clf__n_jobs=1)  # tránh song song lồng nhau với permutation_importance
    except Exception:
        pass
    thr = json.loads((MODELS_DIR / "threshold.json").read_text(encoding="utf-8"))
    thr_f1 = thr["threshold_f1"]
    p = final.predict_proba(X_test)[:, 1]
    pred = (p >= thr_f1).astype(int)
    summary = json.loads((REPORTS_DIR / "final_summary.json").read_text(encoding="utf-8"))
    out = {"final_model": summary["final_model"], "threshold_f1": thr_f1}

    # --- 8.1 Phân tích lỗi theo nhóm ---
    err = X_test.copy()
    err["y"] = y_test.values
    err["p"] = p
    err["pred"] = pred
    err["contacted_before"] = (err["pdays"] != 999).astype(int)
    err["is_fp"] = ((err["y"] == 0) & (err["pred"] == 1)).astype(int)
    err["is_fn"] = ((err["y"] == 1) & (err["pred"] == 0)).astype(int)
    group_tables = {}
    for c in ["job", "poutcome", "month", "contacted_before", "contact", "education"]:
        pos = err[err.y == 1]
        tbl = pd.DataFrame({
            "n_yes": pos.groupby(c).size(),
            "recall": pos.groupby(c)["pred"].mean(),
            "n_fp": err.groupby(c)["is_fp"].sum(),
            "n_fn": err.groupby(c)["is_fn"].sum(),
        }).sort_values("recall")
        group_tables[c] = tbl.round(3)
        tbl.to_csv(REPORTS_DIR / f"error_by_{c}.csv")
    out["error_by_group"] = {c: t.to_dict("index") for c, t in group_tables.items()}
    print("=== Recall của lớp yes theo nhóm (thấp nhất) ===")
    for c, t in group_tables.items():
        print(f"-- {c} --")
        print(t.head(4).to_string())

    # --- 8.2 Hệ số Logistic Regression (tuned, fit trên train) ---
    from sklearn.linear_model import LogisticRegression
    best_params = json.loads((REPORTS_DIR / "best_params.json").read_text(encoding="utf-8"))
    lr_kwargs = {k.replace("clf__", ""): v for k, v in best_params["LogReg"].items()}
    lr = build_pipeline(LogisticRegression(max_iter=3000, class_weight="balanced",
                                           random_state=SEED, **lr_kwargs))
    lr.fit(X_train, y_train)
    names = feature_names_out(lr)
    coef = pd.Series(lr.named_steps["clf"].coef_[0], index=names)
    odds = np.exp(coef).sort_values(ascending=False)
    out["logreg_odds_top10"] = {k: round(float(v), 3) for k, v in odds.head(10).items()}
    out["logreg_odds_bottom10"] = {k: round(float(v), 3) for k, v in odds.tail(10).items()}
    print("Odds ratio cao nhất:", out["logreg_odds_top10"])
    print("Odds ratio thấp nhất:", out["logreg_odds_bottom10"])
    fig, ax = plt.subplots(figsize=(9, 5))
    pd.concat([odds.head(8), odds.tail(8)]).iloc[::-1].plot(kind="barh", ax=ax, color="#1f77b4")
    ax.set_title("Odds ratio - Logistic Regression (top/bottom 8)")
    ax.set_xlabel("Odds ratio (exp(coef))")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig11_logreg_odds.png", dpi=150)
    plt.close(fig)

    # --- 8.3 Permutation importance trên test ---
    print(">>> Permutation importance (10 repeats, scoring=PR-AUC) ...", flush=True)
    r = permutation_importance(final, X_test, y_test, scoring="average_precision",
                               n_repeats=10, random_state=SEED, n_jobs=-1)
    imp = pd.Series(r.importances_mean, index=X_test.columns).sort_values(ascending=False)
    imp_std = pd.Series(r.importances_std, index=X_test.columns)
    imp_df = pd.DataFrame({"importance": imp, "std": imp_std}).round(4)
    imp_df.to_csv(REPORTS_DIR / "permutation_importance.csv")
    out["permutation_importance_top10"] = {
        k: round(float(v), 4) for k, v in imp.head(10).items()}
    print("Permutation importance top 10:", out["permutation_importance_top10"])
    fig, ax = plt.subplots(figsize=(9, 5))
    imp.head(12).iloc[::-1].plot(kind="barh", ax=ax, color="#1f77b4")
    ax.set_title("Permutation importance (giảm PR-AUC khi hoán vị biến) - test")
    ax.set_xlabel("Giảm PR-AUC trung bình")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig12_permutation_importance.png", dpi=150)
    plt.close(fig)

    # --- 8.4 Cây nông minh họa luật ---
    shallow = build_pipeline(DecisionTreeClassifier(
        max_depth=3, min_samples_leaf=100, class_weight="balanced", random_state=SEED))
    shallow.fit(X_train, y_train)
    fig = plt.figure(figsize=(20, 9))
    plot_tree(shallow.named_steps["clf"],
              feature_names=feature_names_out(shallow),
              class_names=["no", "yes"], filled=True, fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig13_shallow_tree.png", dpi=150)
    plt.close(fig)
    print("saved fig13_shallow_tree.png")

    # --- 8.5 SHAP (tùy chọn) ---
    try:
        import shap
        print(">>> SHAP TreeExplainer ...", flush=True)
        X_prep = final.named_steps["prep"].transform(
            final.named_steps["fe"].transform(X_test))
        feats = feature_names_out(final)
        sample = X_prep[np.random.default_rng(SEED).choice(len(X_prep), 500, replace=False)]
        explainer = shap.TreeExplainer(final.named_steps["clf"])
        sv = explainer.shap_values(sample)
        vals = sv[1] if isinstance(sv, list) else (sv[:, :, 1] if sv.ndim == 3 else sv)
        mean_abs = np.abs(vals).mean(axis=0)
        top = np.argsort(mean_abs)[::-1][:12]
        out["shap_top12"] = {str(feats[i]): round(float(mean_abs[i]), 4) for i in top}
        fig, ax = plt.subplots(figsize=(9, 5))
        pd.Series({str(feats[i]): mean_abs[i] for i in top[::-1]}).plot(
            kind="barh", ax=ax, color="#1f77b4")
        ax.set_title("SHAP - |SHAP value| trung bình (top 12, mẫu 500 dòng test)")
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / "fig14_shap_summary.png", dpi=150)
        plt.close(fig)
        print("SHAP top 12:", out["shap_top12"])
    except Exception as e:
        out["shap_status"] = f"không chạy được ({type(e).__name__}: {e}); permutation importance đã đủ theo cẩm nang."
        print("SHAP bỏ qua:", e)

    (REPORTS_DIR / "analysis_summary.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Đã lưu reports/analysis_summary.json")


if __name__ == "__main__":
    main()

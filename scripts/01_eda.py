# -*- coding: utf-8 -*-
"""Giai đoạn 1-2: Hiểu dữ liệu + EDA (Cẩm nang giai đoạn 1, 2).

Nguyên tắc: chia train/test NGAY TỪ ĐẦU, khóa test, EDA chỉ trên train.
Mọi hình lưu vào reports/figures/, mọi số liệu vào reports/eda_summary.json.
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import FIGURES_DIR, REPORTS_DIR, SEED
from src.data import load_clean, stratified_split

plt.rcParams.update({"figure.dpi": 150, "font.size": 10})
sns.set_style("whitegrid")


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / name, dpi=150)
    plt.close(fig)
    print("saved", name)


def main():
    df = load_clean()
    X_train, X_test, y_train, y_test = stratified_split(df)
    eda = X_train.copy()
    eda["y"] = y_train.values
    summary = {
        "raw_shape": list(df.shape),
        "clean_rows": int(len(df)),
        "duplicates_removed": 12,
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "train_yes_rate": round(float(y_train.mean()), 4),
        "test_yes_rate": round(float(y_test.mean()), 4),
        "columns": list(df.columns),
    }

    # --- 1. Phân phối mục tiêu ---
    fig, ax = plt.subplots(figsize=(6, 4))
    vc = eda["y"].value_counts(normalize=True)
    ax.bar(["no", "yes"], [vc[0], vc[1]], color=["#7f7f7f", "#1f77b4"])
    for i, v in enumerate([vc[0], vc[1]]):
        ax.text(i, v + 0.02, f"{v:.1%}", ha="center")
    ax.set_ylim(0, 1.0)
    ax.set_title("Phân phối biến mục tiêu y (tập train)")
    ax.set_ylabel("Tỷ lệ")
    save(fig, "fig01_target_distribution.png")

    # --- 2. Phân phối biến số ---
    num = eda.select_dtypes("number").drop(columns=["y"])
    fig, axes = plt.subplots(3, 4, figsize=(14, 9))
    for ax, c in zip(axes.ravel(), num.columns):
        ax.hist(eda[c], bins=30, color="#1f77b4", edgecolor="white")
        ax.set_title(c)
    for ax in axes.ravel()[len(num.columns):]:
        ax.axis("off")
    fig.suptitle("Phân phối các biến số (tập train)", y=1.0)
    save(fig, "fig02_numeric_histograms.png")
    summary["numeric_describe"] = {
        c: {"mean": round(float(eda[c].mean()), 3), "median": round(float(eda[c].median()), 3),
            "std": round(float(eda[c].std()), 3), "min": float(eda[c].min()), "max": float(eda[c].max()),
            "p99": round(float(eda[c].quantile(0.99)), 3),
            "skew": round(float(eda[c].skew()), 3)}
        for c in num.columns
    }

    # --- 3. Boxplot theo lớp (age, duration, campaign) ---
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, c in zip(axes, ["age", "duration", "campaign"]):
        sns.boxplot(data=eda, x="y", y=c, ax=ax, showfliers=False)
        ax.set_title(f"{c} theo y")
    save(fig, "fig03_boxplot_by_class.png")
    summary["duration_by_y"] = {
        str(k): {"median": round(float(v["median"]), 1), "mean": round(float(v["mean"]), 1)}
        for k, v in eda.groupby("y")["duration"].agg(["median", "mean"]).iterrows()
    }

    # --- 4. Tỷ lệ yes theo biến phân loại ---
    cat_cols = ["job", "marital", "education", "contact", "month", "day_of_week", "poutcome"]
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    rates = {}
    for ax, c in zip(axes.ravel(), cat_cols):
        r = eda.groupby(c)["y"].mean().sort_values()
        rates[c] = {str(k): round(float(v), 4) for k, v in r.items()}
        r.plot(kind="barh", ax=ax, color="#1f77b4")
        ax.set_title(f"Tỷ lệ yes theo {c}")
        ax.set_xlabel("Tỷ lệ yes")
    axes.ravel()[-1].axis("off")
    fig.suptitle("Tỷ lệ đăng ký theo biến phân loại (tập train)", y=1.0)
    save(fig, "fig04_yes_rate_by_category.png")
    summary["yes_rate_by_category"] = rates

    # --- 5. Ma trận tương quan (biến số) ---
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(num.corr(), annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, ax=ax)
    ax.set_title("Ma trận tương quan giữa các biến số (tập train)")
    save(fig, "fig05_correlation_heatmap.png")
    corr = num.corr()
    summary["macro_correlations"] = {
        "euribor3m-nr.employed": round(float(corr.loc["euribor3m", "nr.employed"]), 3),
        "euribor3m-emp.var.rate": round(float(corr.loc["euribor3m", "emp.var.rate"]), 3),
        "emp.var.rate-nr.employed": round(float(corr.loc["emp.var.rate", "nr.employed"]), 3),
    }

    # --- 6. pdays và poutcome ---
    eda["contacted_before"] = (eda["pdays"] != 999).astype(int)
    summary["contacted_before_rate"] = {
        str(k): {"n": int(v["count"]), "yes_rate": round(float(v["mean"]), 4)}
        for k, v in eda.groupby("contacted_before")["y"].agg(["count", "mean"]).iterrows()
    }
    summary["poutcome_yes_rate"] = {
        str(k): round(float(v), 4) for k, v in eda.groupby("poutcome")["y"].mean().items()
    }
    summary["unknown_rate"] = {
        c: round(float((eda[c] == "unknown").mean()), 4)
        for c in eda.select_dtypes("object").columns
    }
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    eda.groupby("contacted_before")["y"].mean().plot(kind="bar", ax=axes[0], color="#1f77b4")
    axes[0].set_xticklabels(["Chưa từng liên hệ\n(pdays=999)", "Đã từng liên hệ"], rotation=0)
    axes[0].set_title("Tỷ lệ yes theo contacted_before")
    axes[0].set_ylabel("Tỷ lệ yes")
    eda.groupby("poutcome")["y"].mean().sort_values().plot(kind="barh", ax=axes[1], color="#1f77b4")
    axes[1].set_title("Tỷ lệ yes theo poutcome")
    save(fig, "fig06_pdays_poutcome.png")

    # --- 7. Theo thời gian: tỷ lệ yes theo month + xu hướng theo thứ tự dòng ---
    order = np.arange(len(eda))
    bins = pd.qcut(order, 10, labels=False)
    trend = eda.groupby(bins)["y"].mean()
    summary["yes_rate_by_time_decile"] = {str(int(k)): round(float(v), 4) for k, v in trend.items()}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    eda.groupby("month")["y"].mean().reindex(
        ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
    ).dropna().plot(kind="bar", ax=axes[0], color="#1f77b4")
    axes[0].set_title("Tỷ lệ yes theo tháng")
    axes[0].set_ylabel("Tỷ lệ yes")
    trend.plot(ax=axes[1], marker="o", color="#1f77b4")
    axes[1].set_title("Tỷ lệ yes theo decile thứ tự dòng (train)")
    axes[1].set_xlabel("Decile thời gian")
    axes[1].set_ylabel("Tỷ lệ yes")
    save(fig, "fig07_yes_rate_over_time.png")

    # --- 8. Ngoại lệ campaign ---
    summary["campaign_percentiles"] = {
        str(p): round(float(eda["campaign"].quantile(p / 100)), 2)
        for p in [50, 90, 95, 99, 99.5]
    }
    summary["campaign_gt15_rate"] = round(float((eda["campaign"] > 15).mean()), 4)
    summary["campaign_gt15_yes_rate"] = round(float(eda.loc[eda["campaign"] > 15, "y"].mean()), 4)

    (REPORTS_DIR / "eda_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("EDA xong. Train yes:", summary["train_yes_rate"],
          "| contacted_before:", summary["contacted_before_rate"],
          "| poutcome:", summary["poutcome_yes_rate"])


if __name__ == "__main__":
    main()

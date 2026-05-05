"""
Diagnostic plots for presentation.
Outputs PNGs to models/plots/.
"""
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

OUT = Path("/Users/theo/Documents/Statistical Consulting/models")
PLOTS = OUT / "plots"
PLOTS.mkdir(exist_ok=True)
sns.set_theme(style="whitegrid", context="talk")


# ---------- Demand model diagnostics ----------
def demand_plots():
    pred = pd.read_csv(OUT / "demand_predictions_test.csv")
    metrics = json.load(open(OUT / "demand_metrics.json"))

    # 1. Predicted vs actual booking curves (averaged over IDs by WBA)
    curve = (pred.groupby("WeekBeforeArrival")
                  .agg(actual=("HistoricalBookedNights", "mean"),
                       pred=("pred", "mean")).reset_index())
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(curve["WeekBeforeArrival"], curve["actual"], label="Actual", lw=2)
    ax.plot(curve["WeekBeforeArrival"], curve["pred"], label="Predicted", lw=2, ls="--")
    ax.invert_xaxis()
    ax.set_xlabel("Weeks before arrival")
    ax.set_ylabel("Mean cumulative bookings")
    ax.set_title(f"Predicted vs actual booking curve (test set, R²={metrics['test_r2']:.2f})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS / "demand_curve_pred_vs_actual.png", dpi=140)
    plt.close(fig)

    # 2. Calibration: binned mean prediction vs mean actual
    pred["pred_bin"] = pd.qcut(pred["pred"], 20, duplicates="drop")
    cal = pred.groupby("pred_bin", observed=True).agg(
        mean_pred=("pred", "mean"), mean_actual=("HistoricalBookedNights", "mean"),
        n=("pred", "size")).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(cal["mean_pred"], cal["mean_actual"], "o-", lw=2)
    lim = max(cal["mean_pred"].max(), cal["mean_actual"].max())
    ax.plot([0, lim], [0, lim], "k--", alpha=0.5, label="Perfect calibration")
    ax.set_xlabel("Mean predicted bookings (per bin)")
    ax.set_ylabel("Mean actual bookings (per bin)")
    ax.set_title("Demand model calibration")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS / "demand_calibration.png", dpi=140)
    plt.close(fig)

    # 3. Feature importance (top 20) via permutation-free fast path: skip if missing
    fi_path = OUT / "demand_feature_importance.csv"
    if not fi_path.exists():
        # HistGB doesn't dump it; compute from model permutation on a sample
        bundle = joblib.load(OUT / "demand_hgb.joblib")
        # crude: use sklearn's built-in feature_importances_ if available
        m = bundle["model"]
        if hasattr(m, "feature_importances_"):
            fi = pd.DataFrame({"feature": bundle["feature_cols"],
                               "importance": m.feature_importances_})
            fi = fi.sort_values("importance", ascending=False)
            fi.to_csv(fi_path, index=False)
        else:
            fi = None
    else:
        fi = pd.read_csv(fi_path)

    if fi is not None and len(fi):
        col = "gain" if "gain" in fi.columns else "importance"
        top = fi.head(20)
        fig, ax = plt.subplots(figsize=(9, 7))
        ax.barh(top["feature"][::-1], top[col][::-1])
        ax.set_xlabel(col)
        ax.set_title("Top 20 features")
        fig.tight_layout()
        fig.savefig(PLOTS / "demand_feature_importance.png", dpi=140)
        plt.close(fig)


# ---------- Elasticity plots ----------
def elasticity_plots():
    res = pd.read_csv(OUT / "elasticity_results.csv")

    # 1. Lead-time bar chart
    lt = res[res["model"].str.startswith("LeadTime=")].copy()
    lt["bucket"] = lt["model"].str.replace("LeadTime=", "")
    lt = lt.set_index("bucket").loc[["0-4w", "5-12w", "13-26w", "27-52w"]].reset_index()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(lt["bucket"], lt["elasticity"], yerr=1.96 * lt["std_error"], capsize=6,
           color=sns.color_palette("rocket", len(lt)))
    pooled = res[res["model"] == "Pooled_Flow"]["elasticity"].iloc[0]
    ax.axhline(pooled, ls="--", color="black", alpha=0.6, label=f"Pooled = {pooled:.2f}")
    ax.set_ylabel("Elasticity (β on log price)")
    ax.set_title("Price elasticity by lead-time horizon (95% CI)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS / "elasticity_by_leadtime.png", dpi=140)
    plt.close(fig)

    # 2. Seasonal cluster distribution
    sc = res[res["model"].str.startswith("Season=")].copy()
    sc["cluster"] = sc["model"].str.replace("Season=", "")
    sc = sc.sort_values("elasticity")
    fig, ax = plt.subplots(figsize=(10, 9))
    ax.barh(sc["cluster"], sc["elasticity"],
            xerr=1.96 * sc["std_error"], capsize=3,
            color=sns.color_palette("rocket", len(sc)))
    ax.axvline(pooled, ls="--", color="black", alpha=0.6, label=f"Pooled = {pooled:.2f}")
    ax.set_xlabel("Elasticity")
    ax.set_title("Elasticity by seasonal cluster (95% CI)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS / "elasticity_by_cluster.png", dpi=140)
    plt.close(fig)


# ---------- Optimization plots ----------
def opt_plots():
    opt = pd.read_csv(OUT / "optimal_price_ladders.csv")

    # 1. Uplift distribution
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.histplot(opt["uplift_pct"].clip(-50, 200), bins=40, ax=ax, color="steelblue")
    ax.axvline(opt["uplift_pct"].median(), color="red", ls="--",
               label=f"Median = {opt['uplift_pct'].median():.1f}%")
    ax.axvline(opt["uplift_pct"].mean(), color="black", ls="--",
               label=f"Mean = {opt['uplift_pct'].mean():.1f}%")
    ax.set_xlabel("Revenue uplift vs baseline (%)")
    ax.set_title("Optimization uplift distribution (200 held-out IDs)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS / "opt_uplift_distribution.png", dpi=140)
    plt.close(fig)

    # 2. Ladder heatmap: chosen multiplier frequency per lead bucket
    buckets = ["ladder_0_4w", "ladder_5_12w", "ladder_13_26w", "ladder_27_52w"]
    grid = sorted(opt[buckets[0]].unique())
    mat = np.zeros((len(grid), len(buckets)))
    for j, b in enumerate(buckets):
        counts = opt[b].value_counts(normalize=True)
        for i, g in enumerate(grid):
            mat[i, j] = counts.get(g, 0)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(mat, annot=True, fmt=".1%",
                xticklabels=[b.replace("ladder_", "").replace("_", "–") for b in buckets],
                yticklabels=[f"{g:.2f}x" for g in grid],
                cmap="Blues", ax=ax, cbar_kws={"label": "Share of IDs"})
    ax.set_xlabel("Lead-time bucket")
    ax.set_ylabel("Optimal price multiplier")
    ax.set_title("Optimal price ladder by lead-time")
    fig.tight_layout()
    fig.savefig(PLOTS / "opt_ladder_heatmap.png", dpi=140)
    plt.close(fig)

    # 3. Mean optimal multiplier per bucket with CI
    means = opt[buckets].mean()
    sems = opt[buckets].std() / np.sqrt(len(opt))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar([b.replace("ladder_", "").replace("_", "–") for b in buckets],
           means.values, yerr=1.96 * sems.values, capsize=8,
           color=sns.color_palette("crest", len(buckets)))
    ax.axhline(1.0, color="black", ls="--", alpha=0.5, label="Baseline")
    ax.set_ylabel("Mean optimal multiplier")
    ax.set_title("Average optimal price adjustment by lead-time")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS / "opt_mean_multiplier.png", dpi=140)
    plt.close(fig)


def main():
    print("Demand plots ...")
    demand_plots()
    print("Elasticity plots ...")
    elasticity_plots()
    print("Optimization plots ...")
    opt_plots()
    print(f"Done -> {PLOTS}")


if __name__ == "__main__":
    main()

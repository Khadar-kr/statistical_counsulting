"""
Price Elasticity Estimation
===========================
Two-way fixed effects panel regression:
    log(1 + WeeklyBookings) = beta * log(DiscountedPrice)
                              + alpha_id (ReservableOptionMarketGroupId FE)
                              + gamma_w (WeekBeforeArrival FE)
                              + epsilon

We construct WeeklyBookings = first-difference of HistoricalBookedNights along the
booking-horizon panel (so it is the *flow* of new bookings during week-w-before-arrival),
which is the proper outcome for a price-response model. Cumulative HistoricalBookedNights
is also estimated for comparison.

Stratified elasticities reported by lead-time bucket and seasonal cluster.

Outputs:
    models/elasticity_results.csv
    models/elasticity_summary.txt
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from linearmodels.panel import PanelOLS

warnings.filterwarnings("ignore")

DATA_PATH = "/Users/theo/Documents/Statistical Consulting/simulation_output.csv"
OUT_DIR = Path("/Users/theo/Documents/Statistical Consulting/models")
OUT_DIR.mkdir(exist_ok=True)

DROP_COLS = ["DiscountedPriceLastYear", "HistoricalBookedNightsLastYear", "CapacityLastYear"]
ABSENT = ["DeckingType", "Kitchen", "DeckingExtras"]


def load():
    print("Loading ...")
    df = pd.read_csv(DATA_PATH)
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])
    for c in ABSENT:
        df[c] = df[c].fillna("None").replace(0, "None").astype(str)
    df["WeekStartDate"] = pd.to_datetime(df["WeekStartDate"], errors="coerce")
    df = df[df["DiscountedPrice"] > 0].copy()
    return df


def build_weekly_flow(df):
    """Compute new bookings per snapshot via first-difference within each ID,
    ordered by descending WeekBeforeArrival (52 -> 0)."""
    df = df.sort_values(["ReservableOptionMarketGroupId", "WeekBeforeArrival"],
                        ascending=[True, False]).copy()
    df["NewBookings"] = (
        df.groupby("ReservableOptionMarketGroupId")["HistoricalBookedNights"]
          .diff().fillna(df["HistoricalBookedNights"])
    )
    df.loc[df["NewBookings"] < 0, "NewBookings"] = 0
    return df


def fit_twfe(df, dep_col, label):
    print(f"\n--- {label}: dep = {dep_col} ---")
    panel = df.set_index(["ReservableOptionMarketGroupId", "WeekBeforeArrival"]).copy()
    panel["log_price"] = np.log(panel["DiscountedPrice"])
    panel["log_y"] = np.log1p(panel[dep_col])

    mod = PanelOLS(
        panel["log_y"],
        panel[["log_price"]],
        entity_effects=True,
        time_effects=True,
        check_rank=False,
    )
    res = mod.fit(cov_type="clustered", cluster_entity=True)
    beta = res.params["log_price"]
    se = res.std_errors["log_price"]
    t = res.tstats["log_price"]
    p = res.pvalues["log_price"]
    n = int(res.nobs)
    print(f"  elasticity = {beta:.4f}  (SE {se:.4f}, t {t:.2f}, p {p:.3g}, n={n:,})")
    return {"model": label, "dep": dep_col, "elasticity": beta,
            "std_error": se, "t_stat": t, "p_value": p, "n_obs": n,
            "r2_within": float(res.rsquared_within)}


def stratified(df, by, dep_col, label_prefix):
    rows = []
    for key, sub in df.groupby(by, observed=True):
        if sub["ReservableOptionMarketGroupId"].nunique() < 50:
            continue
        try:
            r = fit_twfe(sub, dep_col, f"{label_prefix}={key}")
            r["stratum_var"] = by
            r["stratum_value"] = str(key)
            rows.append(r)
        except Exception as e:
            print(f"  [skip {key}]: {e}")
    return rows


def main():
    df = load()
    df = build_weekly_flow(df)

    # Lead-time bucket for stratification
    df["LeadTimeBucket"] = pd.cut(
        df["WeekBeforeArrival"], bins=[-0.1, 4, 12, 26, 53],
        labels=["0-4w", "5-12w", "13-26w", "27-52w"],
    )

    results = []

    # 1. Pooled flow elasticity (preferred outcome)
    results.append(fit_twfe(df, "NewBookings", "Pooled_Flow"))

    # 2. Pooled cumulative (sanity)
    results.append(fit_twfe(df, "HistoricalBookedNights", "Pooled_Cumulative"))

    # 3. Stratified by lead-time
    results += stratified(df, "LeadTimeBucket", "NewBookings", "LeadTime")

    # 4. Stratified by seasonal cluster
    results += stratified(df, "SeasonalCluster", "NewBookings", "Season")

    out = pd.DataFrame(results)
    out.to_csv(OUT_DIR / "elasticity_results.csv", index=False)

    with open(OUT_DIR / "elasticity_summary.txt", "w") as f:
        f.write("Price Elasticity (Two-Way Fixed Effects)\n")
        f.write("=" * 60 + "\n\n")
        f.write(out.to_string(index=False))
        f.write("\n\nInterpretation: a 1% price rise -> beta% change in (1+bookings).\n")

    print("\nSaved ->", OUT_DIR / "elasticity_results.csv")


if __name__ == "__main__":
    main()

"""
Revenue Optimization via Counterfactual Price Simulation
========================================================
For a held-out set of arrival weeks, search a discrete grid of price multipliers
applied to each lead-time bucket. For each candidate ladder we:
    1. Build counterfactual snapshots (replace DiscountedPrice with multiplier * baseline).
    2. Score them with the trained LightGBM demand model -> predicted cumulative bookings.
    3. Convert cumulative -> incremental new bookings per week.
    4. Apply capacity constraint (cumulative cannot exceed Capacity).
    5. Revenue = sum_w (Price_w * NewBookings_w).
Pick the ladder maximizing expected revenue per ID.

Outputs:
    models/optimal_price_ladders.csv
    models/optimization_summary.txt
"""

import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

DATA_PATH = "/Users/theo/Documents/Statistical Consulting/simulation_output.csv"
OUT_DIR = Path("/Users/theo/Documents/Statistical Consulting/models")
MODEL_PATH = OUT_DIR / "demand_hgb.joblib"

DROP_COLS = ["DiscountedPriceLastYear", "HistoricalBookedNightsLastYear", "CapacityLastYear"]
ABSENT = ["DeckingType", "Kitchen", "DeckingExtras"]

CAT_COLS = [
    "MarketGroupCode", "BrandGroupCode", "CampsiteCode", "AccoKindCode",
    "AccoTypeRangeCode", "SpecialPeriodCode", "SeasonalCluster", "CampsiteCluster",
    "CampsiteCountry", "CampsiteRegion", "CampsiteType", "AccommodationType",
    "AccommodationRange", "DeckingType", "Kitchen", "DeckingExtras",
    "Airco", "HotTub", "Tropical", "Roof", "TV", "ArrivalMonth",
]
NUM_COLS = ["WeekBeforeArrival", "DiscountedPrice", "Bedrooms", "Bathrooms",
            "Sleeps", "Capacity", "latitude", "longitude", "AvgTemperature"]

# Price multipliers searched per lead-time bucket
MULT_GRID = [0.85, 0.95, 1.00, 1.05, 1.15]
LEAD_BUCKETS = [("0-4w", 0, 4), ("5-12w", 5, 12), ("13-26w", 13, 26), ("27-52w", 27, 52)]


def load():
    df = pd.read_csv(DATA_PATH)
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])
    for c in ABSENT:
        df[c] = df[c].fillna("None").replace(0, "None").astype(str)
    df["WeekStartDate"] = pd.to_datetime(df["WeekStartDate"], errors="coerce")
    df["ArrivalYear"] = df["WeekStartDate"].dt.year
    df["ArrivalWeekOfYear"] = df["WeekStartDate"].dt.isocalendar().week.astype(int)
    df = df[df["DiscountedPrice"] > 0].copy()
    return df


def feat_engineer(df):
    df = df.copy()
    df["LogPrice"] = np.log1p(df["DiscountedPrice"].clip(lower=0))
    df["LeadTimeBucket"] = pd.cut(
        df["WeekBeforeArrival"], bins=[-0.1, 4, 12, 26, 53],
        labels=["last_month", "1_3mo", "3_6mo", "6_12mo"],
    ).astype(str)
    cat = [c for c in CAT_COLS + ["LeadTimeBucket"] if c in df.columns]
    for c in cat:
        df[c] = df[c].astype("category")
    return df


def predict_cum(model, panel, feature_cols):
    pred = model.predict(panel[feature_cols])
    return np.clip(pred, 0, None)


def cum_to_flow(cum):
    """Convert cumulative bookings (ordered WBA 52->0) into per-week new bookings."""
    flow = -np.diff(np.concatenate([[cum[0]], cum]))
    flow = np.clip(flow, 0, None)
    flow[0] = max(cum[-1] - flow[1:].sum(), 0) if len(cum) > 1 else cum[0]
    return flow


def simulate_id(panel_id, model, feature_cols, baseline_price_by_wba, capacity):
    """Vectorized: build one big prediction matrix over all (combo x snapshot) rows."""
    from itertools import product
    panel_id = panel_id.sort_values("WeekBeforeArrival", ascending=False).reset_index(drop=True)
    n = len(panel_id)
    wba = panel_id["WeekBeforeArrival"].values

    bucket_idx = np.zeros(n, dtype=int)
    for bi, (_, lo, hi) in enumerate(LEAD_BUCKETS):
        bucket_idx[(wba >= lo) & (wba <= hi)] = bi

    combos = list(product(MULT_GRID, repeat=len(LEAD_BUCKETS)))
    K = len(combos)
    mults_per_row = np.array([[c[bucket_idx[i]] for i in range(n)] for c in combos])  # (K, n)
    new_prices = baseline_price_by_wba[None, :] * mults_per_row  # (K, n)

    # Tile the panel K times, override price-related cols
    big = pd.concat([panel_id] * K, ignore_index=True, copy=False)
    flat_price = new_prices.reshape(-1)
    big["DiscountedPrice"] = flat_price
    big["LogPrice"] = np.log1p(flat_price)

    cum_flat = np.clip(model.predict(big[feature_cols]), 0, None)
    cum = cum_flat.reshape(K, n)  # snapshots in WBA-desc order (52..0)

    # Enforce monotonic cumulative bookings in arrival-time direction (reverse of WBA)
    cum_tf = np.maximum.accumulate(cum[:, ::-1], axis=1)
    cum_tf = np.minimum(cum_tf, capacity)
    flow_tf = np.diff(np.concatenate([np.zeros((K, 1)), cum_tf], axis=1), axis=1)
    flow_tf = np.clip(flow_tf, 0, capacity)
    flow = flow_tf[:, ::-1]  # back to WBA-desc

    revenue = (flow * new_prices).sum(axis=1)
    bookings = flow.sum(axis=1)
    mean_price = new_prices.mean(axis=1)

    k_best = int(np.argmax(revenue))
    best = {
        "revenue": float(revenue[k_best]),
        "ladder": combos[k_best],
        "bookings": float(bookings[k_best]),
        "mean_price": float(mean_price[k_best]),
    }

    return best


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Train demand model first: {MODEL_PATH}")
    print("Loading model ...")
    bundle = joblib.load(MODEL_PATH)
    model = bundle["model"]
    feature_cols = bundle["feature_cols"]

    print("Loading data ...")
    df = load()
    df = feat_engineer(df)

    # Hold-out: most recent 15% of arrival weeks
    cutoff = df["WeekStartDate"].quantile(0.85)
    holdout = df[df["WeekStartDate"] > cutoff].copy()
    print(f"Hold-out arrivals: {len(holdout):,} rows / "
          f"{holdout['ReservableOptionMarketGroupId'].nunique():,} IDs")

    # Sample IDs for tractable demo
    sample_ids = (holdout["ReservableOptionMarketGroupId"]
                  .drop_duplicates().sample(min(200, holdout["ReservableOptionMarketGroupId"].nunique()),
                                            random_state=42))
    holdout = holdout[holdout["ReservableOptionMarketGroupId"].isin(sample_ids)]
    print(f"Optimizing over {sample_ids.size} sampled IDs ...")

    rows = []
    grouped = holdout.groupby("ReservableOptionMarketGroupId", observed=True)
    for i, (gid, panel_id) in enumerate(grouped):
        baseline = panel_id.sort_values("WeekBeforeArrival", ascending=False)["DiscountedPrice"].values
        capacity = float(panel_id["Capacity"].iloc[0]) if panel_id["Capacity"].iloc[0] > 0 else 1.0

        # Baseline (multiplier 1.0 everywhere)
        baseline_panel = panel_id.sort_values("WeekBeforeArrival", ascending=False).reset_index(drop=True)
        cum_base = predict_cum(model, baseline_panel, feature_cols)
        cum_base = np.minimum(np.maximum.accumulate(cum_base[::-1])[::-1], capacity)
        time_forward = cum_base[::-1]
        flow = np.clip(np.diff(np.concatenate([[0], time_forward])), 0, capacity)[::-1]
        rev_base = float((flow * baseline).sum())

        best = simulate_id(panel_id, model, feature_cols, baseline, capacity)
        rows.append({
            "ReservableOptionMarketGroupId": gid,
            "capacity": capacity,
            "baseline_revenue": rev_base,
            "optimal_revenue": best["revenue"],
            "uplift_pct": 100 * (best["revenue"] - rev_base) / max(rev_base, 1e-6),
            "ladder_0_4w": best["ladder"][0],
            "ladder_5_12w": best["ladder"][1],
            "ladder_13_26w": best["ladder"][2],
            "ladder_27_52w": best["ladder"][3],
            "optimal_total_bookings": best["bookings"],
            "optimal_mean_price": best["mean_price"],
        })
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{sample_ids.size} done")

    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / "optimal_price_ladders.csv", index=False)

    summary = [
        "Revenue Optimization Summary",
        "=" * 60,
        f"IDs evaluated: {len(out)}",
        f"Mean baseline revenue: {out['baseline_revenue'].mean():.2f}",
        f"Mean optimal revenue:  {out['optimal_revenue'].mean():.2f}",
        f"Mean uplift:           {out['uplift_pct'].mean():.2f}%",
        f"Median uplift:         {out['uplift_pct'].median():.2f}%",
        "",
        "Most-chosen multiplier per lead bucket:",
    ]
    for col in ["ladder_0_4w", "ladder_5_12w", "ladder_13_26w", "ladder_27_52w"]:
        mode = out[col].mode().iloc[0]
        summary.append(f"  {col}: {mode}  (mean {out[col].mean():.3f})")
    txt = "\n".join(summary)
    print("\n" + txt)
    (OUT_DIR / "optimization_summary.txt").write_text(txt + "\n")


if __name__ == "__main__":
    main()

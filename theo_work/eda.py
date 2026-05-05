"""
Exploratory Data Analysis — simulation_output.csv
Run: python eda.py
Outputs: eda_output/ directory with printed summaries and plots
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# ── Setup ─────────────────────────────────────────────────────────────────────
DATA_PATH = Path("/Users/theo/Documents/Statistical Consulting/simulation_output.csv")
OUT = Path("/Users/theo/Documents/Statistical Consulting/eda_output")
OUT.mkdir(exist_ok=True)

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.05)
SECTION = "\n" + "="*70 + "\n"

# ── Load ──────────────────────────────────────────────────────────────────────
print("Loading data...")
df = pd.read_csv(DATA_PATH, parse_dates=["WeekStartDate"])
print(f"Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")

# ═══════════════════════════════════════════════════════════════════════════════
# 1. BASIC STRUCTURE
# ═══════════════════════════════════════════════════════════════════════════════
print(SECTION + "1. BASIC STRUCTURE")

print("\nColumn dtypes:")
print(df.dtypes.to_string())

print("\nMissing values (count & %):")
miss = df.isnull().sum()
miss_pct = miss / len(df) * 100
miss_df = pd.DataFrame({"count": miss, "pct": miss_pct})
print(miss_df[miss_df["count"] > 0].to_string() if miss_df["count"].sum() > 0 else "  None")

print("\nDuplicate rows:", df.duplicated().sum())

# ═══════════════════════════════════════════════════════════════════════════════
# 2. PANEL DIMENSIONS
# ═══════════════════════════════════════════════════════════════════════════════
print(SECTION + "2. PANEL DIMENSIONS")

dims = {
    "Unique ReservableOptionMarketGroupId": df["ReservableOptionMarketGroupId"].nunique(),
    "Unique WeekStartDate (arrival weeks)": df["WeekStartDate"].nunique(),
    "Unique CampsiteCode": df["CampsiteCode"].nunique(),
    "Unique MarketGroupCode": df["MarketGroupCode"].nunique(),
    "Unique SeasonalCluster": df["SeasonalCluster"].nunique(),
    "Unique CampsiteCluster": df["CampsiteCluster"].nunique(),
    "Unique AccoKindCode": df["AccoKindCode"].nunique(),
    "Unique AccoTypeRangeCode": df["AccoTypeRangeCode"].nunique(),
    "WeekBeforeArrival range": f"{df['WeekBeforeArrival'].min()} – {df['WeekBeforeArrival'].max()}",
}
for k, v in dims.items():
    print(f"  {k}: {v}")

print("\nWeekStartDate range:", df["WeekStartDate"].min().date(), "to", df["WeekStartDate"].max().date())

print("\nRows per ReservableOptionMarketGroupId (lead-time steps):")
counts = df.groupby("ReservableOptionMarketGroupId").size()
print(counts.describe().to_string())

print("\nMarketGroupCode distribution:")
print(df["MarketGroupCode"].value_counts().to_string())

print("\nCampsiteCode distribution (top 20):")
print(df["CampsiteCode"].value_counts().head(20).to_string())

print("\nAccoKindCode distribution:")
print(df["AccoKindCode"].value_counts().to_string())

print("\nSeasonalCluster distribution:")
print(df["SeasonalCluster"].value_counts().to_string())

print("\nArrivalMonth distribution:")
print(df["ArrivalMonth"].value_counts().sort_index().to_string())

# ═══════════════════════════════════════════════════════════════════════════════
# 3. TARGET VARIABLES
# ═══════════════════════════════════════════════════════════════════════════════
print(SECTION + "3. TARGET VARIABLES")

# Occupancy rate at final arrival
final = df[df["WeekBeforeArrival"] == 0].copy()
final["OccupancyRate"] = final["TotalBookedNights"] / final["Capacity"]

print(f"\nFinal snapshot (WeekBeforeArrival == 0): {len(final):,} rows")
print("\nTotalBookedNights:")
print(final["TotalBookedNights"].describe().to_string())

print("\nCapacity:")
print(final["Capacity"].describe().to_string())

print("\nOccupancyRate (TotalBookedNights / Capacity):")
print(final["OccupancyRate"].describe().to_string())

print("\nOccupancyRate by ArrivalMonth:")
print(final.groupby("ArrivalMonth")["OccupancyRate"].mean().round(3).to_string())

print("\nOccupancyRate by SeasonalCluster:")
print(final.groupby("SeasonalCluster")["OccupancyRate"].mean().sort_values(ascending=False).round(3).to_string())

print("\nHistoricalBookedNights (weekly incremental) stats:")
print(df["HistoricalBookedNights"].describe().to_string())
print(f"  Zero weeks: {(df['HistoricalBookedNights'] == 0).mean():.1%}")
print(f"  Overdispersion (var/mean): {df['HistoricalBookedNights'].var() / df['HistoricalBookedNights'].mean():.2f}")

# ═══════════════════════════════════════════════════════════════════════════════
# 4. BOOKING CURVE ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
print(SECTION + "4. BOOKING CURVE ANALYSIS")

curve = df.groupby("WeekBeforeArrival").agg(
    mean_incr=("HistoricalBookedNights", "mean"),
    median_incr=("HistoricalBookedNights", "median"),
    p90_incr=("HistoricalBookedNights", lambda x: x.quantile(0.9)),
).sort_index(ascending=False)

print("\nMean incremental bookings by WeekBeforeArrival (52 → 0):")
print(curve.round(3).to_string())

# Cumulative booking pace as % of final demand
df_sorted = df.sort_values(["ReservableOptionMarketGroupId", "WeekBeforeArrival"], ascending=[True, False])
df_sorted["CumBookings"] = df_sorted.groupby("ReservableOptionMarketGroupId")["HistoricalBookedNights"].cumsum()
df_sorted["PctFinalDemand"] = df_sorted["CumBookings"] / df_sorted["TotalBookedNights"].replace(0, np.nan)

pace = df_sorted.groupby("WeekBeforeArrival")["PctFinalDemand"].mean().sort_index(ascending=False)
print("\nMean cumulative % of final demand by WeekBeforeArrival (52 → 0):")
print(pace.round(3).to_string())

print("\nBooking curve by SeasonalCluster (mean incremental, selected weeks):")
sel_weeks = [52, 40, 30, 20, 10, 5, 0]
bc_seg = df[df["WeekBeforeArrival"].isin(sel_weeks)].groupby(
    ["SeasonalCluster", "WeekBeforeArrival"])["HistoricalBookedNights"].mean().unstack("WeekBeforeArrival")
print(bc_seg.round(2).to_string())

# ═══════════════════════════════════════════════════════════════════════════════
# 5. PRICE ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
print(SECTION + "5. PRICE ANALYSIS")

print("\nDiscountedPrice overall:")
print(df["DiscountedPrice"].describe().to_string())

print("\nDistinct price levels per ReservableOptionMarketGroupId:")
price_levels = df.groupby("ReservableOptionMarketGroupId")["DiscountedPrice"].nunique()
print(price_levels.describe().to_string())

print("\nDiscountedPrice by SeasonalCluster:")
print(df.groupby("SeasonalCluster")["DiscountedPrice"].describe().round(2).to_string())

print("\nDiscountedPrice by ArrivalMonth:")
print(df.groupby("ArrivalMonth")["DiscountedPrice"].mean().round(2).to_string())

print("\nMean price by WeekBeforeArrival (selected weeks, 52→0):")
price_curve = df.groupby("WeekBeforeArrival")["DiscountedPrice"].mean().sort_index(ascending=False)
print(price_curve.round(2).to_string())

print("\nDiscountedPriceLastYear — zero share:", (df["DiscountedPriceLastYear"] == 0).mean())
print(df["DiscountedPriceLastYear"].describe().to_string())

# Price change YoY where available
mask = df["DiscountedPriceLastYear"] > 0
if mask.sum() > 0:
    df.loc[mask, "PriceChangeYoY"] = (df.loc[mask, "DiscountedPrice"] - df.loc[mask, "DiscountedPriceLastYear"]) / df.loc[mask, "DiscountedPriceLastYear"]
    print("\nYoY price change (where last-year data exists):")
    print(df.loc[mask, "PriceChangeYoY"].describe().round(4).to_string())

# ═══════════════════════════════════════════════════════════════════════════════
# 6. PRICE ELASTICITY — RAW SIGNAL
# ═══════════════════════════════════════════════════════════════════════════════
print(SECTION + "6. PRICE ELASTICITY — RAW SIGNAL")

# Log-log correlation: within each ID, does higher price → lower bookings?
df_nz = df[df["HistoricalBookedNights"] > 0].copy()
df_nz["LogPrice"] = np.log(df_nz["DiscountedPrice"])
df_nz["LogBookings"] = np.log(df_nz["HistoricalBookedNights"])

print("\nOverall Pearson corr(log price, log bookings) [unconditional]:")
print(f"  r = {df_nz[['LogPrice','LogBookings']].corr().iloc[0,1]:.4f}")

# Within-ID correlation (demeaned)
df_nz["LogPrice_dm"] = df_nz["LogPrice"] - df_nz.groupby("ReservableOptionMarketGroupId")["LogPrice"].transform("mean")
df_nz["LogBookings_dm"] = df_nz["LogBookings"] - df_nz.groupby("ReservableOptionMarketGroupId")["LogBookings"].transform("mean")
within_corr = df_nz[["LogPrice_dm","LogBookings_dm"]].corr().iloc[0,1]
print(f"\nWithin-ID (demeaned) corr(log price, log bookings):")
print(f"  r = {within_corr:.4f}")

# Simple OLS elasticity by SeasonalCluster
print("\nSimple OLS log-log elasticity by SeasonalCluster:")
from numpy.linalg import lstsq
for seg, grp in df_nz.groupby("SeasonalCluster"):
    X = np.column_stack([np.ones(len(grp)), grp["LogPrice"].values])
    y = grp["LogBookings"].values
    coef, *_ = lstsq(X, y, rcond=None)
    print(f"  {seg}: β = {coef[1]:.3f}")

# ═══════════════════════════════════════════════════════════════════════════════
# 7. YEAR-OVER-YEAR COMPARISON
# ═══════════════════════════════════════════════════════════════════════════════
print(SECTION + "7. YEAR-OVER-YEAR COMPARISON")

print("\nHistoricalBookedNightsLastYear — zero share:", (df["HistoricalBookedNightsLastYear"] == 0).mean())
print(df["HistoricalBookedNightsLastYear"].describe().to_string())

mask_ly = df["HistoricalBookedNightsLastYear"] > 0
print(f"\nRows with last-year booking data: {mask_ly.sum():,} ({mask_ly.mean():.1%})")

if mask_ly.sum() > 0:
    df.loc[mask_ly, "BookingGrowth"] = (
        df.loc[mask_ly, "HistoricalBookedNights"] - df.loc[mask_ly, "HistoricalBookedNightsLastYear"]
    ) / df.loc[mask_ly, "HistoricalBookedNightsLastYear"]
    print("\nYoY booking growth by SeasonalCluster:")
    print(df.loc[mask_ly].groupby("SeasonalCluster")["BookingGrowth"].describe().round(3).to_string())

# ═══════════════════════════════════════════════════════════════════════════════
# 8. ACCOMMODATION & CAMPSITE FEATURES
# ═══════════════════════════════════════════════════════════════════════════════
print(SECTION + "8. ACCOMMODATION & CAMPSITE FEATURES")

cat_cols = ["AccoTypeRangeCode","SpecialPeriodCode","CampsiteCluster","CampsiteCountry",
            "CampsiteRegion","CampsiteType","AccommodationType","AccommodationRange",
            "DeckingType","Roof","Kitchen","DeckingExtras"]
for col in cat_cols:
    vc = df[col].value_counts()
    print(f"\n{col} ({vc.shape[0]} levels):")
    print(vc.head(10).to_string())

num_feats = ["Bedrooms","Bathrooms","Sleeps","Airco","HotTub","Tropical","TV","AvgTemperature"]
print("\nNumeric accommodation features:")
print(df[num_feats].describe().round(2).to_string())

# Occupancy rate by accommodation features
print("\nMean OccupancyRate by AccommodationRange:")
print(final.groupby("AccommodationRange")["OccupancyRate"].mean().sort_values(ascending=False).round(3).to_string())

print("\nMean OccupancyRate by AccommodationType:")
print(final.groupby("AccommodationType")["OccupancyRate"].mean().sort_values(ascending=False).round(3).to_string())

print("\nMean OccupancyRate by CampsiteCode (top 20 by count):")
top_cs = final["CampsiteCode"].value_counts().head(20).index
print(final[final["CampsiteCode"].isin(top_cs)].groupby("CampsiteCode")["OccupancyRate"].mean().sort_values(ascending=False).round(3).to_string())

# ═══════════════════════════════════════════════════════════════════════════════
# 9. CAPACITY ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
print(SECTION + "9. CAPACITY ANALYSIS")

print("\nCapacity:")
print(df["Capacity"].describe().to_string())
print("\nCapacity distribution (value counts, top 15):")
print(df["Capacity"].value_counts().head(15).to_string())

# Capacity vs CapacityLastYear
mask_cy = df["CapacityLastYear"] > 0
print(f"\nRows with last-year capacity: {mask_cy.sum():,} ({mask_cy.mean():.1%})")
if mask_cy.sum() > 0:
    diff = df.loc[mask_cy, "Capacity"] - df.loc[mask_cy, "CapacityLastYear"]
    print("Capacity change YoY:", diff.describe().round(1).to_string())

# ═══════════════════════════════════════════════════════════════════════════════
# 10. GEOSPATIAL OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════════
print(SECTION + "10. GEOSPATIAL OVERVIEW")

geo = final.groupby(["CampsiteCode","CampsiteCountry","CampsiteRegion"]).agg(
    lat=("latitude","first"),
    lon=("longitude","first"),
    mean_occ=("OccupancyRate","mean"),
    n_weeks=("WeekStartDate","count"),
).reset_index()
print(f"\nUnique campsite locations: {geo.shape[0]}")
print(geo.sort_values("mean_occ", ascending=False).head(20).round(3).to_string())

print("\nOccupancyRate by CampsiteCountry:")
print(final.groupby("CampsiteCountry")["OccupancyRate"].mean().sort_values(ascending=False).round(3).to_string())

print("\nOccupancyRate by CampsiteRegion (top 20):")
print(final.groupby("CampsiteRegion")["OccupancyRate"].mean().sort_values(ascending=False).head(20).round(3).to_string())

# ═══════════════════════════════════════════════════════════════════════════════
# 11. ENDOGENEITY CHECK — does price respond to bookings?
# ═══════════════════════════════════════════════════════════════════════════════
print(SECTION + "11. ENDOGENEITY CHECK — price response to booking pace")

df_sorted2 = df.sort_values(["ReservableOptionMarketGroupId","WeekBeforeArrival"], ascending=[True, False])
df_sorted2["CumBookings"] = df_sorted2.groupby("ReservableOptionMarketGroupId")["HistoricalBookedNights"].cumsum()
df_sorted2["OccAtTime"] = df_sorted2["CumBookings"] / df_sorted2["Capacity"].replace(0, np.nan)
df_sorted2["PriceLag"] = df_sorted2.groupby("ReservableOptionMarketGroupId")["DiscountedPrice"].shift(-1)
df_sorted2["PriceChange"] = df_sorted2["DiscountedPrice"] - df_sorted2["PriceLag"]

# Correlation: occ rate at time t → price change from t to t-1 (one week later)
mask_pc = df_sorted2["PriceLag"].notna()
corr_endog = df_sorted2.loc[mask_pc, ["OccAtTime","PriceChange"]].corr().iloc[0,1]
print(f"\nCorr(occupancy at booking time, subsequent price change): {corr_endog:.4f}")
print("  Positive → price raised when bookings are strong (endogeneity present)")
print("  Near zero → price is exogenous / rule-based")

# Price change pattern
print("\nMean price change (current vs next week) by occupancy decile:")
df_sorted2.loc[mask_pc, "OccDecile"] = pd.qcut(df_sorted2.loc[mask_pc, "OccAtTime"], 10, labels=False, duplicates="drop")
print(df_sorted2.loc[mask_pc].groupby("OccDecile")["PriceChange"].mean().round(3).to_string())

# ═══════════════════════════════════════════════════════════════════════════════
# 12. PLOTS
# ═══════════════════════════════════════════════════════════════════════════════
print(SECTION + "12. GENERATING PLOTS → eda_output/")

# --- 12a. Booking curve by seasonal cluster
fig, ax = plt.subplots(figsize=(10, 5))
for seg, grp in df.groupby("SeasonalCluster"):
    c = grp.groupby("WeekBeforeArrival")["HistoricalBookedNights"].mean().sort_index()
    ax.plot(c.index[::-1], c.values[::-1], label=seg, linewidth=1.5)
ax.set_xlabel("Weeks Before Arrival")
ax.set_ylabel("Mean Weekly Bookings")
ax.set_title("Booking Curve by Seasonal Cluster")
ax.invert_xaxis()
ax.legend(fontsize=8, ncol=2)
fig.tight_layout()
fig.savefig(OUT / "booking_curve_by_cluster.png", dpi=150)
plt.close()

# --- 12b. Price curve by seasonal cluster
fig, ax = plt.subplots(figsize=(10, 5))
for seg, grp in df.groupby("SeasonalCluster"):
    pc = grp.groupby("WeekBeforeArrival")["DiscountedPrice"].mean().sort_index()
    ax.plot(pc.index[::-1], pc.values[::-1], label=seg, linewidth=1.5)
ax.set_xlabel("Weeks Before Arrival")
ax.set_ylabel("Mean Discounted Price (€)")
ax.set_title("Price Curve by Seasonal Cluster")
ax.invert_xaxis()
ax.legend(fontsize=8, ncol=2)
fig.tight_layout()
fig.savefig(OUT / "price_curve_by_cluster.png", dpi=150)
plt.close()

# --- 12c. Occupancy rate distribution
fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(final["OccupancyRate"].dropna(), bins=50, edgecolor="white", linewidth=0.4)
ax.set_xlabel("Occupancy Rate (TotalBookedNights / Capacity)")
ax.set_ylabel("# Arrival Weeks")
ax.set_title("Distribution of Final Occupancy Rate")
ax.xaxis.set_major_formatter(mticker.PercentFormatter(1.0))
fig.tight_layout()
fig.savefig(OUT / "occupancy_distribution.png", dpi=150)
plt.close()

# --- 12d. Occupancy by arrival month
fig, ax = plt.subplots(figsize=(8, 4))
mo_occ = final.groupby("ArrivalMonth")["OccupancyRate"].mean()
ax.bar(mo_occ.index, mo_occ.values, color=sns.color_palette("muted")[0])
ax.set_xlabel("Arrival Month")
ax.set_ylabel("Mean Occupancy Rate")
ax.set_title("Mean Occupancy Rate by Arrival Month")
ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
ax.set_xticks(range(1, 13))
fig.tight_layout()
fig.savefig(OUT / "occupancy_by_month.png", dpi=150)
plt.close()

# --- 12e. Price vs weekly bookings scatter (sample)
sample_ids = df["ReservableOptionMarketGroupId"].drop_duplicates().sample(min(300, df["ReservableOptionMarketGroupId"].nunique()), random_state=42)
samp = df[df["ReservableOptionMarketGroupId"].isin(sample_ids) & (df["HistoricalBookedNights"] > 0)]
fig, ax = plt.subplots(figsize=(7, 5))
ax.scatter(samp["DiscountedPrice"], samp["HistoricalBookedNights"], alpha=0.15, s=8, color=sns.color_palette("muted")[1])
ax.set_xlabel("Discounted Price (€)")
ax.set_ylabel("Weekly Bookings")
ax.set_title("Price vs Weekly Bookings (sampled IDs, non-zero weeks)")
fig.tight_layout()
fig.savefig(OUT / "price_vs_bookings_scatter.png", dpi=150)
plt.close()

# --- 12f. Heatmap: mean occupancy by month × seasonal cluster
pivot_occ = final.pivot_table(values="OccupancyRate", index="SeasonalCluster", columns="ArrivalMonth", aggfunc="mean")
fig, ax = plt.subplots(figsize=(12, max(4, len(pivot_occ) * 0.4 + 1)))
sns.heatmap(pivot_occ, annot=True, fmt=".2f", cmap="YlOrRd", ax=ax, linewidths=0.4, cbar_kws={"label":"Occ Rate"})
ax.set_title("Mean Occupancy Rate: Seasonal Cluster × Arrival Month")
ax.set_xlabel("Arrival Month")
ax.set_ylabel("")
fig.tight_layout()
fig.savefig(OUT / "occ_heatmap_cluster_month.png", dpi=150)
plt.close()

# --- 12g. Distribution of HistoricalBookedNights
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].hist(df["HistoricalBookedNights"], bins=30, edgecolor="white", linewidth=0.4)
axes[0].set_title("HistoricalBookedNights (all)")
axes[0].set_xlabel("Weekly Incremental Bookings")
nz = df[df["HistoricalBookedNights"] > 0]["HistoricalBookedNights"]
axes[1].hist(nz, bins=30, edgecolor="white", linewidth=0.4, color=sns.color_palette("muted")[2])
axes[1].set_title("HistoricalBookedNights (> 0 only)")
axes[1].set_xlabel("Weekly Incremental Bookings")
fig.tight_layout()
fig.savefig(OUT / "bookings_distribution.png", dpi=150)
plt.close()

# --- 12h. Cumulative booking pace
fig, ax = plt.subplots(figsize=(10, 5))
for seg, grp in df_sorted.groupby("SeasonalCluster"):
    p = grp.groupby("WeekBeforeArrival")["PctFinalDemand"].mean().sort_index()
    ax.plot(p.index[::-1], p.values[::-1] * 100, label=seg, linewidth=1.5)
ax.set_xlabel("Weeks Before Arrival")
ax.set_ylabel("Cumulative % of Final Demand Booked")
ax.set_title("Booking Pace Curve by Seasonal Cluster")
ax.invert_xaxis()
ax.legend(fontsize=8, ncol=2)
fig.tight_layout()
fig.savefig(OUT / "booking_pace_by_cluster.png", dpi=150)
plt.close()

# ═══════════════════════════════════════════════════════════════════════════════
# 13. PLOTS BY ADDITIONAL FEATURES
# ═══════════════════════════════════════════════════════════════════════════════
print(SECTION + "13. PLOTS BY ADDITIONAL FEATURES")

ADDITIONAL_FEATURES = [
    "CampsiteRegion", "AccoKindCode", "AccommodationRange",
    "Airco", "Bedrooms", "SpecialPeriodCode",
]
MAX_LEVELS = 8  # cap legend entries; bucket the rest as "Other"


def collapse_levels(series, max_levels=MAX_LEVELS):
    """Keep top (max_levels - 1) most frequent values; bucket the rest as 'Other'."""
    s = series.astype(str).fillna("None").replace("nan", "None")
    if s.nunique() <= max_levels:
        return s
    keep = s.value_counts().head(max_levels - 1).index
    return s.where(s.isin(keep), "Other")


def feature_plots(feature):
    print(f"  → {feature}")
    safe = feature.lower()
    df_f = df.copy()
    df_f[feature] = collapse_levels(df_f[feature])
    final_f = final.copy()
    final_f[feature] = collapse_levels(final_f[feature])
    sorted_f = df_sorted.copy()
    sorted_f[feature] = collapse_levels(sorted_f[feature])

    # 13a. Booking curve by feature
    fig, ax = plt.subplots(figsize=(10, 5))
    for seg, grp in df_f.groupby(feature):
        c = grp.groupby("WeekBeforeArrival")["HistoricalBookedNights"].mean().sort_index()
        ax.plot(c.index[::-1], c.values[::-1], label=str(seg), linewidth=1.5)
    ax.set_xlabel("Weeks Before Arrival")
    ax.set_ylabel("Mean Weekly Bookings")
    ax.set_title(f"Booking Curve by {feature}")
    ax.invert_xaxis()
    ax.legend(fontsize=8, ncol=2, title=feature)
    fig.tight_layout()
    fig.savefig(OUT / f"booking_curve_by_{safe}.png", dpi=150)
    plt.close()

    # 13b. Price curve by feature
    fig, ax = plt.subplots(figsize=(10, 5))
    for seg, grp in df_f.groupby(feature):
        pc = grp.groupby("WeekBeforeArrival")["DiscountedPrice"].mean().sort_index()
        ax.plot(pc.index[::-1], pc.values[::-1], label=str(seg), linewidth=1.5)
    ax.set_xlabel("Weeks Before Arrival")
    ax.set_ylabel("Mean Discounted Price (€)")
    ax.set_title(f"Price Curve by {feature}")
    ax.invert_xaxis()
    ax.legend(fontsize=8, ncol=2, title=feature)
    fig.tight_layout()
    fig.savefig(OUT / f"price_curve_by_{safe}.png", dpi=150)
    plt.close()

    # 13c. Mean occupancy by feature
    occ = (final_f.groupby(feature)["OccupancyRate"].mean()
                  .sort_values(ascending=False))
    fig, ax = plt.subplots(figsize=(max(7, 0.55 * len(occ) + 4), 4.5))
    ax.bar(occ.index.astype(str), occ.values,
           color=sns.color_palette("muted", len(occ)))
    ax.set_xlabel(feature)
    ax.set_ylabel("Mean Occupancy Rate")
    ax.set_title(f"Mean Final Occupancy Rate by {feature}")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(OUT / f"occupancy_by_{safe}.png", dpi=150)
    plt.close()

    # 13d. Booking pace by feature
    fig, ax = plt.subplots(figsize=(10, 5))
    for seg, grp in sorted_f.groupby(feature):
        p = grp.groupby("WeekBeforeArrival")["PctFinalDemand"].mean().sort_index()
        ax.plot(p.index[::-1], p.values[::-1] * 100, label=str(seg), linewidth=1.5)
    ax.set_xlabel("Weeks Before Arrival")
    ax.set_ylabel("Cumulative % of Final Demand Booked")
    ax.set_title(f"Booking Pace by {feature}")
    ax.invert_xaxis()
    ax.legend(fontsize=8, ncol=2, title=feature)
    fig.tight_layout()
    fig.savefig(OUT / f"booking_pace_by_{safe}.png", dpi=150)
    plt.close()

    # 13e. Heatmap: mean occupancy by feature × ArrivalMonth
    pivot = final_f.pivot_table(values="OccupancyRate", index=feature,
                                columns="ArrivalMonth", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(12, max(3.5, len(pivot) * 0.4 + 1)))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="YlOrRd",
                ax=ax, linewidths=0.4, cbar_kws={"label": "Occ Rate"})
    ax.set_title(f"Mean Occupancy Rate: {feature} × Arrival Month")
    ax.set_xlabel("Arrival Month")
    ax.set_ylabel(feature)
    fig.tight_layout()
    fig.savefig(OUT / f"occ_heatmap_{safe}_by_month.png", dpi=150)
    plt.close()


for feat in ADDITIONAL_FEATURES:
    feature_plots(feat)

print(f"\nAll plots saved to {OUT}/")
print("\nEDA complete.")

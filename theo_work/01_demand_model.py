"""
Demand Forecasting Model
========================
Target: HistoricalBookedNights (cumulative bookings at WeekBeforeArrival snapshot)
Approach: LightGBM with Tweedie objective (handles zero-inflation + overdispersion).
Baseline: Poisson GLM via statsmodels for comparison.

Outputs:
    models/demand_lgbm.txt
    models/demand_metrics.json
    models/demand_feature_importance.csv
    models/demand_predictions_test.csv
"""

import json
import os
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold

warnings.filterwarnings("ignore")

DATA_PATH = "/Users/theo/Documents/Statistical Consulting/simulation_output.csv"
OUT_DIR = Path("/Users/theo/Documents/Statistical Consulting/models")
OUT_DIR.mkdir(exist_ok=True)

DROP_COLS = ["DiscountedPriceLastYear", "HistoricalBookedNightsLastYear", "CapacityLastYear"]
ABSENT_FEATURE_COLS = ["DeckingType", "Kitchen", "DeckingExtras"]
TARGET = "HistoricalBookedNights"

CAT_COLS = [
    "MarketGroupCode", "BrandGroupCode", "CampsiteCode", "AccoKindCode",
    "AccoTypeRangeCode", "SpecialPeriodCode", "SeasonalCluster", "CampsiteCluster",
    "CampsiteCountry", "CampsiteRegion", "CampsiteType", "AccommodationType",
    "AccommodationRange", "DeckingType", "Kitchen", "DeckingExtras",
    "Airco", "HotTub", "Tropical", "Roof", "TV", "ArrivalMonth",
]

NUM_COLS = [
    "WeekBeforeArrival", "DiscountedPrice", "Bedrooms", "Bathrooms",
    "Sleeps", "Capacity", "latitude", "longitude", "AvgTemperature",
]


def load_and_prepare():
    print("Loading data ...")
    df = pd.read_csv(DATA_PATH)
    print(f"  shape: {df.shape}")

    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])

    # Per user: 0/NaN in these columns means the feature is absent in the accommodation.
    for col in ABSENT_FEATURE_COLS:
        if col in df.columns:
            df[col] = df[col].fillna("None").replace(0, "None").astype(str)

    df["WeekStartDate"] = pd.to_datetime(df["WeekStartDate"], errors="coerce")
    df["ArrivalYear"] = df["WeekStartDate"].dt.year
    df["ArrivalWeekOfYear"] = df["WeekStartDate"].dt.isocalendar().week.astype(int)

    # Engineered features
    df["LogPrice"] = np.log1p(df["DiscountedPrice"].clip(lower=0))
    df["LeadTimeBucket"] = pd.cut(
        df["WeekBeforeArrival"],
        bins=[-0.1, 4, 12, 26, 53],
        labels=["last_month", "1_3mo", "3_6mo", "6_12mo"],
    ).astype(str)

    for col in CAT_COLS + ["LeadTimeBucket"]:
        if col in df.columns:
            df[col] = df[col].astype("category")

    return df


def time_based_split(df):
    """Hold out the latest arrival weeks for testing."""
    cutoff = df["WeekStartDate"].quantile(0.85)
    train = df[df["WeekStartDate"] <= cutoff].copy()
    test = df[df["WeekStartDate"] > cutoff].copy()
    print(f"  train: {len(train):,} | test: {len(test):,} | cutoff: {cutoff.date()}")
    return train, test


def train_hgb(train, test, feature_cols, cat_features):
    X_tr, y_tr = train[feature_cols].copy(), train[TARGET]
    X_te, y_te = test[feature_cols].copy(), test[TARGET]

    cat_mask = [c in cat_features for c in feature_cols]

    print("Training HistGradientBoostingRegressor (Poisson) ...")
    model = HistGradientBoostingRegressor(
        loss="poisson",
        learning_rate=0.05,
        max_iter=1000,
        max_leaf_nodes=127,
        min_samples_leaf=200,
        l2_regularization=1.0,
        categorical_features=cat_mask,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=30,
        random_state=42,
        verbose=1,
    )
    model.fit(X_tr, y_tr)

    pred_te = np.clip(model.predict(X_te), 0, None)

    metrics = {
        "n_iter": int(model.n_iter_),
        "test_mae": float(mean_absolute_error(y_te, pred_te)),
        "test_rmse": float(np.sqrt(mean_squared_error(y_te, pred_te))),
        "test_r2": float(r2_score(y_te, pred_te)),
        "test_mean_actual": float(y_te.mean()),
        "test_mean_pred": float(pred_te.mean()),
        "test_zero_rate_actual": float((y_te == 0).mean()),
        "test_zero_rate_pred": float((pred_te < 0.5).mean()),
    }
    print("Test metrics:", json.dumps(metrics, indent=2))

    joblib.dump({"model": model, "feature_cols": feature_cols,
                 "cat_features": cat_features}, OUT_DIR / "demand_hgb.joblib")

    test_out = test[["ReservableOptionMarketGroupId", "WeekBeforeArrival",
                     "WeekStartDate", "DiscountedPrice", TARGET]].copy()
    test_out["pred"] = pred_te
    test_out.to_csv(OUT_DIR / "demand_predictions_test.csv", index=False)

    with open(OUT_DIR / "demand_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    return model, metrics


def main():
    df = load_and_prepare()
    train, test = time_based_split(df)

    feature_cols = [c for c in CAT_COLS + NUM_COLS + ["LogPrice", "LeadTimeBucket",
                                                      "ArrivalYear", "ArrivalWeekOfYear"]
                    if c in df.columns]
    cat_features = [c for c in CAT_COLS + ["LeadTimeBucket"] if c in df.columns]

    print(f"Features: {len(feature_cols)} ({len(cat_features)} categorical)")
    train_hgb(train, test, feature_cols, cat_features)
    print("Done. Artifacts ->", OUT_DIR)


if __name__ == "__main__":
    main()

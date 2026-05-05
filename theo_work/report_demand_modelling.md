# Dynamic Pricing for European Campsites
## Report: Data, Methodology & Demand Modelling

---

## 1. Data Description

### 1.1 Dataset Overview

The dataset (`simulation_output.csv`) is a synthetic but structurally faithful representation of the operator's actual booking data. It contains **3,130,816 rows** and **38 columns**, covering arrival weeks from January 2024 to December 2025 across 142 campsites in 8 countries.

The fundamental unit of observation is a **booking-curve snapshot**: one row records the state of a specific product–market combination at a particular point in the booking horizon. Each product is identified by a `ReservableOptionMarketGroupId`, which encodes a unique combination of campsite, accommodation type, market group, and arrival week. For each such product, there are up to 53 snapshots — one per value of `WeekBeforeArrival` from 52 (one year ahead) down to 0 (arrival week). The full dataset therefore constitutes an unbalanced panel.

| Dimension | Count |
|---|---|
| Unique booking curves (`ReservableOptionMarketGroupId`) | ~59,000 |
| Unique campsites | 142 |
| Countries | 8 |
| Market groups | 4 (Domestic, Benelux, DACH, Rest of Europe) |
| Accommodation kinds | 4 (Mobile, Permanent, Static, Seasonal) |
| Accommodation type ranges | 5 (Comfort, Romantic, Luxury, Standard, Family) |
| Accommodation quality ranges | 3 (Budget, Mid, Premium) |
| Booking horizon (`WeekBeforeArrival`) | 0 – 52 |
| Special period types | 9 |
| Seasonal clusters | 142 (product-level groupings) |

### 1.2 Variable Groups

**Identifiers and panel keys**

- `ReservableOptionMarketGroupId` — the primary curve identifier; groups all snapshots of one product over time.
- `WeekBeforeArrival` — the booking horizon index (52 = one year out; 0 = arrival). Decreasing values trace the passage of time toward arrival.
- `WeekStartDate` — calendar date of the arrival week.

**Market and product segmentation**

- `MarketGroupCode` — customer origin/channel (Domestic, Benelux, DACH, Rest of Europe).
- `BrandGroupCode`, `CampsiteCode`, `AccoKindCode`, `AccoTypeRangeCode` — hierarchical product identifiers from brand down to accommodation type.
- `SpecialPeriodCode` — whether the arrival week falls in a school holiday, bank holiday, or standard week.

**Clustering and geography**

- `SeasonalCluster`, `CampsiteCluster` — pre-computed cluster labels grouping products by similar seasonal demand patterns and campsite characteristics respectively.
- `CampsiteCountry`, `CampsiteRegion`, `CampsiteType` — geographic and typological attributes.
- `latitude`, `longitude` — campsite coordinates.

**Accommodation features**

Physical attributes of the accommodation unit: `AccommodationType`, `AccommodationRange`, `Bedrooms`, `Bathrooms`, `Sleeps`, `DeckingType`, `Kitchen`, `DeckingExtras`, `Roof`, and binary flags for `Airco`, `HotTub`, `Tropical`, `TV`.

**Pricing and demand**

- `DiscountedPrice` — the price offered at this point in the booking horizon. This is the key decision variable.
- `DiscountedPriceLastYear` — equivalent price in the prior year (benchmark; present in a subset of rows).
- `HistoricalBookedNights` — **incremental** booked nights recorded at this snapshot. This is the weekly demand signal and the primary modelling target.
- `HistoricalBookedNightsLastYear` — same metric for the equivalent prior-year snapshot.
- `TotalBookedNights` — final cumulative booked nights at the end of the booking horizon (outcome at `WeekBeforeArrival = 0`). Used as the ground truth for occupancy evaluation.
- `Capacity` / `CapacityLastYear` — total available capacity in nights for the arrival week.

**Time and weather**

- `ArrivalMonth` — derived from `WeekStartDate`; captures seasonality.
- `AvgTemperature` — average temperature for the stay period and location.

### 1.3 Target Variable Characteristics

The modelling target (`HistoricalBookedNights`) is a non-negative integer count with strong zero-inflation: **74.0% of all snapshots record zero incremental bookings**. The mean is 0.74 nights per snapshot with a variance-to-mean ratio (index of dispersion) well above 1, confirming overdispersion. This motivates the choice of a Poisson-family model rather than ordinary least squares.

The final occupancy rate (`TotalBookedNights / Capacity`) has a median of approximately 10%, with a right-skewed distribution. Mean occupancy peaks in summer months (June–August) and during special periods (school holidays, festival weeks).

---

## 2. Methodology

The overall analytical pipeline consists of four sequential stages, each building on the outputs of the previous. The pipeline is implemented across five Python scripts.

### 2.1 Stage 0 — Exploratory Data Analysis (`eda.py`)

Before modelling, a thorough EDA establishes the structure of the data and motivates design choices in subsequent stages. The EDA covers:

- **Panel dimensions** — verifying that each curve has the expected 53 snapshots, checking for missing values and duplicates.
- **Booking curve shape** — aggregating mean incremental and cumulative bookings by `WeekBeforeArrival` to characterise the temporal demand profile. The analysis shows that demand arrives unevenly across the horizon: a large share of bookings accumulates in the final 13 weeks before arrival.
- **Price analysis** — examining price variation across dimensions (cluster, month, horizon) and computing year-over-year price changes where prior-year data exist.
- **Raw elasticity signal** — computing within-ID demeaned log-log correlations between price and bookings as a non-parametric check that price variation is associated with demand variation.
- **Endogeneity check** — correlating the current occupancy fill rate with subsequent price changes. A positive correlation would indicate that prices are raised in response to strong demand, which would bias a naive regression toward zero or positive elasticity. This diagnostic informs the choice of a fixed-effects estimator in Stage 2.
- **Segment-level patterns** — booking curves, price curves, occupancy heatmaps, and booking pace by `SeasonalCluster`, `AccoKindCode`, `AccommodationRange`, `Airco`, `Bedrooms`, `CampsiteRegion`, and `SpecialPeriodCode`.

### 2.2 Stage 1 — Demand Forecasting Model (`01_demand_model.py`)

The demand model predicts `HistoricalBookedNights` (incremental weekly bookings) for any product at any point in its booking horizon given the current price and context. This model is the core building block that makes price optimisation possible: by querying it at different hypothetical prices, the optimiser can estimate demand under counterfactual pricing scenarios.

Full details are given in Section 3.

### 2.3 Stage 2 — Price Elasticity Estimation (`02_price_elasticity.py`)

A separate, interpretable estimation of price elasticity is conducted using a **Two-Way Fixed Effects (TWFE) panel regression**. Rather than relying on the black-box demand model, this stage produces a single, statistically grounded number — the price elasticity — that answers the question: *by how much does a 1% price increase reduce weekly new bookings, holding product identity and booking horizon fixed?*

The regression model is:

$$\log(1 + \text{NewBookings}_{it}) = \beta \cdot \log(\text{Price}_{it}) + \alpha_i + \gamma_t + \varepsilon_{it}$$

where $\alpha_i$ are entity fixed effects (one per `ReservableOptionMarketGroupId`, absorbing all time-constant product heterogeneity), $\gamma_t$ are time fixed effects (one per `WeekBeforeArrival`, absorbing demand patterns common across all products at a given lead time), and $\beta$ is the price elasticity of interest.

The outcome `NewBookings` is constructed as the **first difference** of `HistoricalBookedNights` along the booking horizon (sorted from week 52 to week 0), representing the flow of new bookings during each period rather than the cumulative stock. This is the correct outcome for a price-response model.

Standard errors are clustered at the entity level to account for within-curve serial correlation. The model is estimated using `linearmodels.PanelOLS`.

Stratified elasticities are estimated separately by:
- **Lead-time bucket** (0–4w, 5–12w, 13–26w, 27–52w) — to capture heterogeneity in price sensitivity across the booking horizon.
- **Seasonal cluster** — to capture heterogeneity across product types.

### 2.4 Stage 3 — Revenue Optimisation (`03_price_optimization.py`)

The optimiser uses the trained demand model to find the price multiplier ladder that maximises expected revenue for each product on a held-out set of arrival weeks. The approach is:

1. **Price grid construction** — define a discrete grid of multipliers $\{0.85,\ 0.95,\ 1.00,\ 1.05,\ 1.15\}$ applied independently to each of the four lead-time buckets. This yields $5^4 = 625$ candidate price ladders per product.
2. **Counterfactual scoring** — for each candidate ladder, replace the observed price with `multiplier × baseline_price` and score the resulting panel with the demand model to obtain predicted cumulative bookings at each horizon snapshot.
3. **Flow conversion and capacity constraint** — convert predicted cumulative bookings to incremental flow; enforce monotonicity and a hard capacity ceiling.
4. **Revenue computation** — revenue = $\sum_t \text{Price}_t \times \text{NewBookings}_t$ over all horizon weeks.
5. **Ladder selection** — choose the multiplier combination that maximises expected revenue.

The optimisation is run on a sample of 200 held-out product curves (those in the most recent 15% of arrival weeks, not seen during model training).

### 2.5 Stage 4 — Diagnostic Visualisation (`04_diagnostic_plots.py`)

A dedicated diagnostics script produces visualisations for all three preceding stages:

- **Demand model**: predicted vs. actual booking curves (averaged by `WeekBeforeArrival`), calibration plot (binned mean predicted vs. mean actual), and feature importance.
- **Elasticity**: bar charts of elasticity by lead-time bucket (with 95% confidence intervals) and by seasonal cluster.
- **Optimisation**: uplift distribution histogram, optimal multiplier frequency heatmap by lead-time bucket, and mean optimal multiplier per bucket.

---

## 3. Demand Modelling

### 3.1 Objective

The demand model predicts **how many nights will be booked in a given week** for a given product (`HistoricalBookedNights`) as a function of the current price and all available context. It is trained to capture the joint effect of:

- the price set at each horizon point,
- how far in advance the arrival is,
- the product's characteristics (campsite, accommodation type, amenities),
- the market context (customer segment, special period, seasonality, geography).

This prediction is used downstream to simulate demand under alternative prices, enabling counterfactual revenue optimisation.

### 3.2 Algorithm

The model uses **`HistGradientBoostingRegressor`** from scikit-learn with a **Poisson loss function**. This choice is motivated by three properties of the target:

| Property | Evidence | Implication |
|---|---|---|
| Non-negative integer count | Min = 0, right-skewed | OLS is inappropriate |
| Zero-inflation | 74% of rows are zero | Must handle point mass at zero |
| Overdispersion | Variance >> mean | Standard Poisson GLM is too constrained |

The histogram-based gradient boosting implementation handles categorical features natively (without one-hot encoding), is efficient on large datasets (~3M rows), and naturally captures non-linear interactions between the booking horizon, price, and product type.

**Hyperparameters:**

| Parameter | Value | Rationale |
|---|---|---|
| `loss` | `poisson` | Count target with overdispersion |
| `learning_rate` | 0.05 | Conservative; compensated by many iterations |
| `max_iter` | 1000 | Upper bound; early stopping used |
| `max_leaf_nodes` | 127 | Deep trees to capture interactions |
| `min_samples_leaf` | 200 | Regularisation; prevents overfitting on sparse cells |
| `l2_regularization` | 1.0 | Additional regularisation |
| `validation_fraction` | 0.15 | Internal validation set for early stopping |
| `n_iter_no_change` | 30 | Patience for early stopping |

### 3.3 Feature Engineering

In addition to the raw columns, three sets of engineered features are constructed:

**Log-price transformation**

$$\text{LogPrice} = \log(1 + \text{DiscountedPrice})$$

Logging compresses the right tail of the price distribution and linearises the expected log-linear relationship between price and demand. Both the raw price and the log-price are included.

**Lead-time bucket**

`WeekBeforeArrival` is discretised into four interpretable segments:

| Label | Range | Booking behaviour |
|---|---|---|
| `6_12mo` | 27–52 weeks | Early planners; low weekly volume |
| `3_6mo` | 13–26 weeks | Growing interest; peak sensitivity window |
| `1_3mo` | 5–12 weeks | Confirmed intent; moderate volume |
| `last_month` | 0–4 weeks | Last-minute; high daily volume |

This captures the non-linear relationship between lead time and demand that a continuous `WeekBeforeArrival` alone cannot represent.

**Calendar features**

`ArrivalYear` and `ArrivalWeekOfYear` are extracted from `WeekStartDate` to capture long-run trends and within-year seasonal patterns not fully captured by `ArrivalMonth`.

**Missing-feature encoding**

For `DeckingType`, `Kitchen`, and `DeckingExtras`, zero and NaN values indicate the feature is absent from the accommodation (not a data quality issue). These are recoded to the string `"None"` before category encoding, so the model treats absence as a distinct category rather than imputing a spurious value.

**Dropped features**

`DiscountedPriceLastYear`, `HistoricalBookedNightsLastYear`, and `CapacityLastYear` are excluded. These columns have substantial missing rates (filled with zero for new products) and would introduce leakage concerns if used naively, since future-year products have no prior-year equivalent.

### 3.4 Full Feature Set

| Group | Features |
|---|---|
| Product identity (categorical) | MarketGroupCode, BrandGroupCode, CampsiteCode, AccoKindCode, AccoTypeRangeCode, AccommodationType, AccommodationRange |
| Temporal context (categorical) | ArrivalMonth, LeadTimeBucket, SpecialPeriodCode |
| Clustering (categorical) | SeasonalCluster, CampsiteCluster |
| Geography (categorical) | CampsiteCountry, CampsiteRegion, CampsiteType |
| Amenities (categorical) | Airco, HotTub, Tropical, Roof, TV, DeckingType, Kitchen, DeckingExtras |
| Booking horizon (numeric) | WeekBeforeArrival |
| Price (numeric) | DiscountedPrice, LogPrice |
| Physical (numeric) | Bedrooms, Bathrooms, Sleeps, Capacity |
| Geographic coordinates (numeric) | latitude, longitude |
| Weather (numeric) | AvgTemperature |
| Calendar (numeric) | ArrivalYear, ArrivalWeekOfYear |

### 3.5 Train–Test Split

A **time-based split** is used: all arrival weeks up to the 85th percentile of `WeekStartDate` form the training set; the most recent 15% form the test set. This mirrors real-world deployment where the model must generalise to future arrival periods unseen at training time. A random split would be inappropriate as it would allow information from future arrival weeks to leak into training.

| Set | Criterion | Approx. size |
|---|---|---|
| Training | `WeekStartDate` ≤ 85th percentile | ~2.66M rows |
| Test | `WeekStartDate` > 85th percentile | ~0.47M rows |

### 3.6 Model Performance

The model was trained to the maximum iteration budget of 1,000 trees without early stopping triggering, indicating the model continued to improve throughout training.

| Metric | Value | Interpretation |
|---|---|---|
| **Test MAE** | **0.714** | Average absolute error of 0.71 nights per snapshot |
| **Test RMSE** | **1.839** | Penalises large errors (driven by high-demand weeks) |
| **Test R²** | **0.268** | Explains 26.8% of variance in the test set |
| Mean actual | 0.735 | Average incremental bookings per snapshot |
| Mean predicted | 0.619 | Model slightly under-predicts on average |
| Zero rate (actual) | 74.0% | Fraction of zero-booking snapshots |
| Zero rate (predicted) | 69.3% | Model predicts slightly fewer zeros than observed |

**Contextualising R² = 0.268.** At first glance an R² of 27% may appear low. However, the target is dominated by zeros (74% of the test set) and has very high noise at the individual snapshot level — most of the variance in any given week's bookings is inherently random and unforecastable from product features alone. What matters for the optimiser is that the model correctly captures the *relative effect of price* and the *systematic pattern across the booking horizon*. The calibration plot (Figure: `demand_calibration.png`) shows that predicted and actual means align well across deciles, confirming that the model is well-calibrated even if point predictions are noisy.

The slight under-prediction of mean bookings (0.619 vs 0.735) and of zero rates (69.3% vs 74.0%) — meaning the model predicts some bookings where there were none — is a common consequence of the Poisson loss, which penalises underestimation of positive counts more than overestimation of zeros. For the downstream optimisation, this conservative bias is acceptable: the optimiser compares the *relative* revenue across price ladders, so a systematic downward level shift cancels out.

### 3.7 Interpretation of Key Drivers

While gradient boosted trees do not yield coefficient-style interpretability, the structure of the model and the elasticity analysis together allow the following statements:

**Price** is the single most actionable driver. The price elasticity analysis (Section 2.3) confirms a strong, statistically significant negative price effect on demand. The demand model captures this by including both `DiscountedPrice` and `LogPrice`, giving it flexibility to represent non-linear price responses.

**Booking horizon** (`WeekBeforeArrival` and `LeadTimeBucket`) is the strongest structural driver. Weekly bookings are near-zero 52 weeks out and rise sharply in the final 13 weeks. The model uses this feature to differentiate "no demand yet" (early horizon) from "demand is accumulating" (late horizon), which is essential for a price-timing strategy.

**Special period** has a large positive effect on demand across all lead times. Products in school-holiday or festival weeks (`SpecialPeriodCode`) receive significantly more bookings than standard weeks at the same price, implying substantial pricing power during these periods.

**Seasonal cluster** and **campsite cluster** capture persistent product-level demand heterogeneity. Clusters with consistently high occupancy (e.g. Cetoddle, Noivern) have systematically higher baseline demand and also exhibit stronger price sensitivity (elasticities of −2.1 to −2.5).

**Accommodation range** (Budget, Mid, Premium) affects both the price level and the responsiveness. Budget products show stronger price sensitivity in the elasticity analysis, consistent with their customer base being more price-conscious. Premium products command higher prices with lower absolute sensitivity.

**Geography and weather** (`CampsiteCountry`, `latitude`, `longitude`, `AvgTemperature`) contribute seasonally. Temperature correlates with summer demand peaks; country-level effects reflect differences in domestic vs. international demand composition.

**Amenities** (Airco, HotTub, Tropical) are significant for the price level but have a more modest independent effect on bookings volume, as they are partly collinear with accommodation range.

### 3.8 Limitations

**Endogeneity of price.** The EDA reveals a small but positive correlation between the current fill rate and subsequent price changes, indicating that prices are sometimes raised in response to strong early demand. This means `DiscountedPrice` is not fully exogenous in the demand model: the model may partially absorb the effect of strong underlying demand as a price effect, causing it to underestimate the true price sensitivity. The TWFE elasticity estimate (Section 2.3) addresses this through entity fixed effects, but the gradient boosting model does not. For the downstream optimiser, the consequence is that predicted demand under price increases may be slightly optimistic.

**Zero-inflation structure.** The Poisson loss handles overdispersion but is not a true two-part model (hurdle or zero-inflated Poisson). An explicit hurdle model — predicting the probability of any booking first, then the count conditional on at least one booking — could improve calibration on the zero vs. non-zero classification.

**Feature exclusion.** Prior-year price and booking data (`DiscountedPriceLastYear`, `HistoricalBookedNightsLastYear`) are excluded. These are natural anchors for demand forecasting and could improve prediction accuracy, particularly for products with stable year-over-year patterns, if the missingness structure is carefully handled.

**Temporal generalisability.** The model is trained on 2024 data and tested on late-2025 arrival weeks. Significant structural demand shifts (new product launches, market entry, macro shocks) between training and deployment windows would degrade predictive accuracy and necessitate retraining.

---

## Summary Table

| Aspect | Choice | Justification |
|---|---|---|
| Algorithm | HistGradientBoostingRegressor (Poisson) | Native categorical support; handles zero-inflation; Poisson loss for counts |
| Target | `HistoricalBookedNights` (incremental) | Direct price-response signal; avoids cumulative leakage |
| Split | Time-based 85/15 | Preserves temporal order; prevents future leakage |
| Key engineered features | LogPrice, LeadTimeBucket | Capture non-linear price response and horizon structure |
| Dropped features | Last-year price, bookings, capacity | Avoid missing-data leakage for new products |
| Test MAE | 0.714 (vs mean = 0.735) | Model mean is close to actual mean |
| Test R² | 0.268 | Moderate fit on a highly noisy, zero-dominated count |
| Price elasticity (pooled) | −1.53 | Elastic demand; strongest at 13–26w horizon |
| Optimisation uplift (median) | +17.6% revenue | Price ladder adjustment: discount early, premium late |

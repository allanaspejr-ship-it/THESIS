# PROJECT CONTEXT
## AI-Based Short-Term Generation Forecasting of Cascaded Hydroelectric Power Plants

This file defines the thesis/project context, methodology, constraints, modeling priorities, and implementation rules for the repository. Any AI assistant, Codex session, or developer modifying this repository must follow this context to keep the project consistent with the thesis objective.

---

## 1. Thesis Title

**AI-Based Short-Term Generation Forecasting of Cascaded Hydroelectric Power Plants**

---

## 2. Project Purpose

This repository implements an AI-based short-term generation forecasting system for the **Agus cascaded hydroelectric power plants**, specifically:

- Agus 1
- Agus 2
- Agus 4
- Agus 5
- Agus 6
- Agus 7

The main purpose is to forecast the **next 24 hours of hourly generation** for each Agus plant and, where applicable, the total cascade generation. The system is designed to support short-term planning, day-ahead capacity nomination, and operational decision-making for cascaded hydropower generation.

The project is not simply a generic machine-learning forecast. It must respect the operational reality of the Agus cascade, where upstream plant behavior affects downstream plants.

---

## 3. Main Thesis Objective

The main objective is to develop an **AI-based short-term, one-day-ahead generation forecasting model** for the cascaded Agus hydroelectric power plants using a **Radial Basis Function Neural Network (RBFNN)** as the primary model.

The model must learn the relationship between historical hydropower operating conditions and future generation using variables such as:

- Plant generation
- Unit generation
- Unit outage or availability status
- Forebay/reservoir elevation
- Inflow and outflow
- Spillway gate condition
- Total gate opening
- Rainfall
- Time-based behavior
- Upstream/downstream cascade behavior

---

## 4. Core Research Direction

The thesis focuses on **short-term generation forecasting**, not long-term hydrological planning and not full physical simulation.

The intended output is a practical, data-driven forecasting workflow that can:

1. Clean raw Agus generation and operational data.
2. Convert the data into an hourly modeling dataset.
3. Train one forecasting model per plant.
4. Evaluate validation and testing performance using accepted regression metrics.
5. Generate a 24-hour day-ahead forecast.
6. Adjust the forecast based on planned unit outages.
7. Compare the primary RBFNN model against benchmark models.

---

## 5. Primary Model

The **RBFNN model is the main model of the thesis**.

The RBFNN must remain the central model in the repository and must be treated as the model that represents the proposed methodology.

The repository currently uses an RBFNN implementation with:

- TensorFlow/Keras
- Custom radial basis function layer
- KMeans-based center initialization
- MinMax scaling
- Chronological train/validation/test split
- Per-plant model training
- Per-plant model saving
- Validation and testing metrics
- 24-hour recursive forecast generation
- Planned outage adjustment

The RBFNN must not be replaced as the main model by Random Forest, XGBoost, LSTM, GRU, or any other model unless explicitly required for a separate experiment. Other models may only serve as benchmarks.

---

## 6. Benchmark Models

The benchmark models are:

1. **Random Forest Regressor**
2. **XGBoost Regressor**

These benchmark models are used only for comparison. They should help prove that the RBFNN method is effective.

Important rule:

> The RBFNN model must be tuned and improved so that it performs better than both Random Forest and XGBoost in the final thesis results.

Benchmark models should use fair and comparable input features, but the final methodology and thesis emphasis must remain on RBFNN.

---

## 7. Required Model Performance Targets

The model must be evaluated using:

- MAPE
- MAE
- RMSE
- R²

### Required target limits

| Metric | Required / Preferred Target |
|---|---|
| MAPE | Must be below 10%; preferred target is 7% or lower |
| R² | Should be at least 0.80; preferred target is 0.90 or closer to 1.00 |
| RMSE | Must be low relative to each plant's MW scale and must not show large unstable errors |
| MAE | Must be low relative to each plant's MW scale |

### Strict result rule

If Random Forest or XGBoost outperforms RBFNN, the RBFNN must be improved by:

- Better feature engineering
- Better lag selection
- Better RBF center count
- Better gamma/spread tuning
- Better target transformation
- Better outage-aware training behavior
- Better chronological validation design
- Better forecast continuity logic

Do not simply weaken the benchmark models to make RBFNN look better. The RBFNN should be improved honestly.

---

## 8. Scope of the Study

This project is limited to:

- Agus 1, Agus 2, Agus 4, Agus 5, Agus 6, and Agus 7
- Short-term generation forecasting
- One-day-ahead / 24-hour hourly forecast
- Historical offline data
- RBFNN as the proposed model
- Random Forest and XGBoost as benchmarks
- Planned outage-aware forecast adjustment

This project does not include:

- Real-time SCADA integration
- WESM bidding simulation
- Full hydrodynamic modeling
- Complete physical reservoir simulation
- Economic dispatch optimization
- New sensor installation
- Long-term seasonal forecasting

---

## 9. Cascade Logic

The Agus hydroelectric system must be treated as a cascade, not as independent plants.

The cascade relationship is:

```text
Agus 1 → Agus 2 → Agus 4 → Agus 5 → Agus 6 → Agus 7
```

Modeling must consider that upstream generation, outflow, and operational conditions influence downstream generation.

Where possible, the feature set should include:

- Upstream generation lags
- Upstream flow/outflow lags
- Local plant lags
- Reservoir/elevation lags
- Gate/spillage lags
- Rainfall-related features
- Rolling averages and rolling variation

---

## 10. Data Pipeline Context

### Cell 1: Cleaning and Preparation

The first script loads the Excel dataset, standardizes columns, cleans unit generation values, handles missing values, resamples to hourly resolution, saves cleaned outputs, and creates the planned outage template.

Expected outputs:

```text
data/outputs/01_runtime_outputs/cleaned_hourly_data.xlsx
data/outputs/01_runtime_outputs/cleaned_hourly_data.parquet
data/outputs/01_runtime_outputs/Planned_Outages_Input.xlsx
data/outputs/03_metadata/cell1_metadata.json
```

Cell 1 must be run before model training and forecasting.

---

### Cell 2: RBFNN Training and Forecasting

The second script trains or loads the RBFNN model for each plant.

Expected behavior:

- Load cleaned hourly data.
- Load planned outage input.
- Create lag features and rolling features.
- Add time features.
- Add cascade-related upstream features.
- Split data chronologically.
- Train one RBFNN model per plant.
- Save model, scalers, metadata, and metrics.
- Generate a 24-hour day-ahead forecast.
- Apply planned outage adjustment.

Expected outputs:

```text
data/outputs/02_models/rbfnn_<plant>.keras
data/outputs/02_models/x_scaler_<plant>.pkl
data/outputs/02_models/y_scaler_<plant>.pkl
data/outputs/02_models/meta_<plant>.json
data/outputs/03_metadata/validation_testing_metrics/rbfnn_validation_testing_metrics.xlsx
data/outputs/01_runtime_outputs/Day_Ahead_24H_RBFNN_Forecast.xlsx
data/outputs/01_runtime_outputs/Day_Ahead_24H_RBFNN_Forecast.csv
```

---

### Cell 3: Benchmark Training and Forecasting

The third script trains or loads Random Forest and XGBoost benchmark models.

Expected behavior:

- Load cleaned hourly data.
- Load planned outage input.
- Create comparable lag, rolling, and time features.
- Split chronologically.
- Train or load Random Forest and XGBoost models.
- Save validation and testing metrics.
- Generate benchmark forecasts with outage adjustment.

Expected outputs:

```text
data/outputs/04_benchmark_outputs/Day_Ahead_24H_RANDOM_FOREST.xlsx
data/outputs/04_benchmark_outputs/Day_Ahead_24H_XGBOOST.xlsx
data/outputs/04_benchmark_outputs/validation_metrics/benchmark_validation_testing_metrics.xlsx
```

---

## 11. Planned Outage Logic

Planned outage status is represented as binary values per unit:

```text
1 = unit is ON / available
0 = unit is OFF / unavailable
```

The planned outage file contains the next 24 forecast hours and unit-level binary values.

### Baseline vs Planned Status Rule

The forecast must compare:

- Baseline unit status from the latest historical row
- Planned outage unit status from the outage template

The adjustment logic must follow:

| Baseline Status | Planned Status | Forecast Action |
|---|---|---|
| 1 | 1 | No change |
| 0 | 0 | No change |
| 1 | 0 | Subtract / reduce the corresponding unit contribution |
| 0 | 1 | Add / restore the corresponding unit contribution |

The implementation may apply this using unit capacity weights and availability ratio.

If all planned units for a plant are zero, the adjusted generation for that plant must be zero.

---

## 12. Unit Capacity Weights

Use the following plant/unit capacity assumptions for outage-aware adjustment unless better official values are provided.

### Agus 1

- Unit 1 = 40 MW
- Unit 2 = 40 MW
- Total = 80 MW

### Agus 2

- Unit 1 = 60 MW
- Unit 2 = 60 MW
- Unit 3 = 60 MW
- Total = 180 MW

### Agus 4

- Unit 1 = 52.7 MW
- Unit 2 = 52.7 MW
- Unit 3 = 52.7 MW
- Total = 158.1 MW

### Agus 5

- Unit 1 = 27.5 MW
- Unit 2 = 27.5 MW
- Total = 55 MW

### Agus 6

- Unit 1 = 34.5 MW
- Unit 2 = 34.5 MW
- Unit 3 = 50 MW
- Unit 4 = 50 MW
- Unit 5 = 50 MW
- Total = 219 MW

### Agus 7

- Unit 1 = 27 MW
- Unit 2 = 27 MW
- Total = 54 MW

---

## 13. Train/Validation/Test Rule

The dataset must always be split chronologically.

Never use random splitting for this thesis because random splitting can leak future information into training.

Required split:

```text
Training   = earliest 70%
Validation = next 15%
Testing    = latest 15%
```

Validation is used for tuning. Testing is used only for final unbiased evaluation.

---

## 14. Forecast Horizon

The required forecast horizon is:

```text
24 hours ahead
Hourly resolution
```

The forecast must start immediately after the last timestamp of the cleaned historical dataset.

Example:

If the latest historical timestamp is:

```text
2025-06-30 23:00:00
```

Then the forecast must start at:

```text
2025-07-01 00:00:00
```

The forecast must not jump to incorrect future months or dates.

---

## 15. Feature Engineering Rules

The model should prioritize features that are physically and operationally meaningful.

Recommended features:

- Hour of day using sine/cosine encoding
- Day of week
- Month
- Weekend flag
- Plant generation lags
- Plant generation rolling mean, std, min, max
- Short-term difference features
- Unit availability status
- Number of running units
- Plant availability flag
- Rainfall
- Outflow
- Elevation
- Spillway gate features
- Total gate opening features
- Upstream generation lags
- Upstream flow lags

Recommended lag values:

```text
1, 2, 3, 6, 12, 24, 48 hours
```

Rolling window:

```text
24 hours
```

---

## 16. Data Cleaning Rules

Cleaning should preserve real hydropower operating behavior.

Do not remove true zero generation caused by outages.

Do not treat all zeros as errors.

Important cleaning principles:

- Short accidental zero gaps may be interpolated.
- Long zero runs may represent real outage/shutdown and should remain zero.
- Negative generation must be corrected or clipped.
- Negative rainfall must be corrected or clipped.
- Unrealistic spikes should be smoothed only when clearly erroneous.
- Missing values should be handled before modeling.
- KNN imputation may be used after simpler interpolation.
- Outage binary values must remain 0 or 1 after hourly resampling.

---

## 17. Evaluation Rules

Metrics must be computed for:

- Validation set
- Testing set
- Daily validation metrics
- Daily testing metrics
- Per-plant results
- Overall comparison between RBFNN, Random Forest, and XGBoost

Required metrics:

```text
MAPE
MAE
RMSE
R²
```

MAPE must use safe division to avoid division by zero.

When actual generation is zero or near-zero, MAPE may become unstable. The implementation must handle this safely and honestly.

---

## 18. Current Main Problem

The current issue is that validation and testing metrics are too far from acceptable values.

Reported problems include:

- High MAPE
- Large RMSE/MAE
- Negative or poor R²
- Forecast drift
- Forecasts not continuing smoothly from the last actual data
- Benchmark models sometimes outperforming RBFNN
- Planned outage adjustment not always reflecting expected generation changes

The repository must be improved to solve these issues while preserving the thesis methodology.

---

## 19. Required Optimization Direction

When optimizing the project, focus on improving the RBFNN without breaking the methodology.

Priority improvements:

1. Improve feature engineering.
2. Improve cascade features.
3. Improve lag selection.
4. Tune RBF centers per plant.
5. Tune gamma/spread per plant.
6. Tune learning rate and batch size.
7. Compare absolute target vs delta target per plant.
8. Improve outage adjustment logic.
9. Improve forecast continuity from latest historical values.
10. Preserve chronological validation/testing.
11. Save every metric and output clearly.

---

## 20. Codex/AI Agent Rules

When using Codex or any AI coding agent, it must follow these rules:

1. Do not overwrite working scripts without creating a backup or Git commit first.
2. Do not remove the RBFNN as the primary model.
3. Do not make Random Forest or XGBoost the main thesis model.
4. Do not use random train/test splitting.
5. Do not fake metrics.
6. Do not hard-code results only to make metrics look good.
7. Do not delete outage logic.
8. Do not ignore unit-level outage binary values.
9. Do not change the thesis scope into another type of forecasting project.
10. Every code change must improve or preserve thesis consistency.

---

## 21. Repository Goal

The final repository should produce:

- A cleaned hourly Agus dataset
- A planned outage input file
- Trained RBFNN models
- Trained benchmark models
- Validation metrics
- Testing metrics
- Daily metrics
- Forecast outputs
- Outage-adjusted forecasts
- Plots suitable for thesis discussion
- Clear metadata for reproducibility

---

## 22. Final Thesis Identity

This project is a:

> **Cascade-aware, outage-aware, RBFNN-based short-term hydropower generation forecasting system.**

The final implementation must always support this identity.

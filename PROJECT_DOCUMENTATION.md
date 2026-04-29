# HYDRO_FORECASTING Documentation

## Overview

HYDRO_FORECASTING is a cascade-aware, outage-aware forecasting project for the Agus hydroelectric power plants. The project supports short-term, one-day-ahead hourly generation forecasting for:

- Agus 1
- Agus 2
- Agus 4
- Agus 5
- Agus 6
- Agus 7

The primary thesis model is a Radial Basis Function Neural Network (RBFNN). Random Forest and XGBoost are included only as benchmark models for comparison.

## Repository Layout

```text
.
|-- README.md                         project quick-start
|-- PROJECT_CONTEXT.md                thesis context and modeling rules
|-- PROJECT_DOCUMENTATION.md          project documentation
|-- requirements.txt                  Python dependencies
|-- optimized_v2/                     main runnable workflow
|   |-- README.md                     optimized v2 workflow notes
|   `-- scripts/
|       |-- cell1_clean_data.py       data cleaning and outage template creation
|       |-- cell2_rbfnn.py            RBFNN training, evaluation, and forecasting
|       |-- cell3_benchmark.py        Random Forest and XGBoost benchmarks
|       `-- cell4_testing_plots.py    testing-split comparison plots
`-- archive/                          older scripts, dashboard, and prior workflow docs
```

Generated data, model files, outputs, benchmark files, and metadata are intentionally ignored by Git.

## Environment Setup

Create a Python environment and install dependencies from the repository root:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The project depends on common data science and machine learning packages including pandas, scikit-learn, TensorFlow/Keras, XGBoost, matplotlib, Plotly, and Streamlit.

## Input Data

Use the optimized v2 workflow for current runs. Place the source Excel workbook at:

```text
optimized_v2/data/DATA(JAN2024-JUNE2025).xlsx
```

If that exact file is not present, Cell 1 uses the first non-temporary `.xlsx` file found in `optimized_v2/data/`.

## Main Pipeline

Run the scripts from the repository root in this order:

```powershell
python optimized_v2/scripts/cell1_clean_data.py
python optimized_v2/scripts/cell2_rbfnn.py
python optimized_v2/scripts/cell3_benchmark.py
python optimized_v2/scripts/cell4_testing_plots.py
```

### Cell 1: Data Cleaning

Cell 1 loads the raw workbook, standardizes the data, prepares hourly records, saves cleaned data, and creates the planned outage input file.

Expected outputs include:

```text
optimized_v2/outputs/cleaned_data/cleaned_hourly_data.xlsx
optimized_v2/outputs/cleaned_data/cleaned_hourly_data.parquet
optimized_v2/outputs/cleaned_data/cell1_metadata.json
optimized_v2/outputs/outages_planning/Planned_Outages_Input.xlsx
```

### Cell 2: RBFNN Forecasting

Cell 2 is the main thesis workflow. It evaluates or trains one RBFNN model per plant, creates validation and testing metrics, saves testing predictions, and generates a 24-hour day-ahead forecast.

Run with saved models when available:

```powershell
python optimized_v2/scripts/cell2_rbfnn.py
```

Retrain intentionally:

```powershell
python optimized_v2/scripts/cell2_rbfnn.py --train
```

Expected outputs include:

```text
optimized_v2/metadata/overall_metrics/rbfnn_validation_testing_metrics.xlsx
optimized_v2/metadata/overall_metrics/rbfnn_testing_predictions.xlsx
optimized_v2/outputs/rbfnn_forecast/Day_Ahead_24H_RBFNN_Forecast.xlsx
optimized_v2/outputs/rbfnn_forecast/Day_Ahead_24H_RBFNN_Forecast.csv
```

### Cell 3: Benchmark Forecasting

Cell 3 evaluates or trains Random Forest and XGBoost benchmark models. These models are not the thesis model; they are used to compare against the RBFNN.

Run with saved models when available:

```powershell
python optimized_v2/scripts/cell3_benchmark.py
```

Retrain intentionally:

```powershell
python optimized_v2/scripts/cell3_benchmark.py --train
```

Expected outputs include:

```text
optimized_v2/metadata/overall_metrics/random_forest_validation_testing_metrics.xlsx
optimized_v2/metadata/overall_metrics/random_forest_testing_predictions.xlsx
optimized_v2/benchmark/random_forest/Day_Ahead_24H_RANDOM_FOREST.xlsx
optimized_v2/benchmark/random_forest/Day_Ahead_24H_RANDOM_FOREST.csv
optimized_v2/metadata/overall_metrics/xgboost_validation_testing_metrics.xlsx
optimized_v2/metadata/overall_metrics/xgboost_testing_predictions.xlsx
optimized_v2/benchmark/xgboost/Day_Ahead_24H_XGBOOST.xlsx
optimized_v2/benchmark/xgboost/Day_Ahead_24H_XGBOOST.csv
```

### Cell 4: Testing Plots

Cell 4 creates full testing-split actual-vs-forecast comparison plots for RBFNN, Random Forest, and XGBoost.

Expected output location:

```text
optimized_v2/outputs/testing_plots/full_testing_split/
```

## Modeling Rules

The project follows these core rules:

- RBFNN remains the primary thesis model.
- Random Forest and XGBoost remain benchmark models only.
- Train, validation, and test splits must be chronological.
- The split convention is 70% training, 15% validation, and 15% testing.
- Data must not be randomly shuffled for time-series evaluation.
- Planned outage logic must preserve unit-level binary availability.
- True outage-related zero generation should not be removed as an error.
- Metrics must be reported honestly and must not be hard-coded.

## Forecasting Method

The forecast horizon is 24 hours at hourly resolution. Forecasts begin immediately after the latest timestamp in the cleaned historical dataset.

The model should use operationally meaningful features such as:

- Hour, day, month, and weekend features
- Plant generation lags
- Rolling generation statistics
- Unit availability and running-unit counts
- Rainfall, outflow, elevation, spillway, and gate features when available
- Upstream generation and flow features for cascade behavior

The Agus cascade direction is:

```text
Agus 1 -> Agus 2 -> Agus 4 -> Agus 5 -> Agus 6 -> Agus 7
```

## Planned Outage Logic

Planned outage values use binary status:

```text
1 = unit is ON / available
0 = unit is OFF / unavailable
```

The forecast compares the latest historical unit status against the planned status:

| Baseline Status | Planned Status | Forecast Action |
|---|---|---|
| 1 | 1 | No change |
| 0 | 0 | No change |
| 1 | 0 | Reduce generation for that unit |
| 0 | 1 | Restore generation for that unit |

If all planned units for a plant are unavailable, the adjusted generation for that plant should be zero.

## Evaluation Metrics

The project reports:

- MAPE
- MAE
- RMSE
- R2

Metrics are generated for validation and testing splits. MAPE must use safe division to avoid invalid results when actual generation is zero or near zero.

Preferred performance targets:

| Metric | Target |
|---|---|
| MAPE | Below 10%, preferably 7% or lower |
| R2 | At least 0.80, preferably near 0.90 or higher |
| RMSE | Low relative to each plant capacity |
| MAE | Low relative to each plant capacity |

## Git and Output Policy

Commit source code, documentation, environment files, and configuration.

Do not commit generated runtime artifacts such as:

- Raw data files
- Cleaned output data
- Trained model files
- Pickle/joblib scaler files
- Forecast output workbooks
- Benchmark output folders
- Metadata output folders
- Python caches

The `.gitignore` file already excludes these generated files and folders.

## Reproducibility Checklist

Before sharing or tagging a release, confirm that:

- Dependencies install from `requirements.txt`.
- The raw source workbook is placed under `optimized_v2/data/`.
- Cell 1 completes and writes cleaned data.
- Cell 2 writes RBFNN metrics and the 24-hour forecast.
- Cell 3 writes benchmark metrics and forecasts.
- Cell 4 writes testing-split plots.
- RBFNN, Random Forest, and XGBoost metrics are saved separately.
- No generated data, models, or outputs are staged for Git.

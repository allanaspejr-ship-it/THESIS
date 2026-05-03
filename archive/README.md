# HYDRO_FORECASTING

AI-based short-term generation forecasting for the Agus cascaded hydroelectric plants.

The repository contains the original cell-based workflow, the existing optimized workflow, a Streamlit dashboard, and a newer isolated `optimized_v2` workflow that preserves the original scripts while adding standardized outputs, unit-level RBFNN forecasts, separated benchmark metrics, and full testing-split comparison plots.

## Main Workflow

Use `optimized_v2` for the revised runnable pipeline:

```text
optimized_v2/
  data/                  raw Excel source file
  scripts/               runnable Cell 1 to Cell 4 scripts
  outputs/               cleaned data, outage template, RBFNN forecast outputs
  benchmark/             Random Forest and XGBoost forecast outputs
  metadata/              validation/testing metrics and testing predictions
```

Put the raw source workbook here:

```text
optimized_v2/data/DATA(JAN2024-JUNE2025).xlsx
```

`optimized_v2/scripts/cell1_clean_data.py` prefers that exact file and falls back to the first non-temporary `.xlsx` file in `optimized_v2/data/`.

## Setup After Clone

Create and activate a Python environment, then install dependencies:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

VS Code extension recommendations are included in:

```text
.vscode/extensions.json
vscode_extensions.txt
```

## Run Order

Run from the repository root:

```powershell
python optimized_v2/scripts/cell1_clean_data.py
python optimized_v2/scripts/cell2_rbfnn.py
python optimized_v2/scripts/cell3_benchmark.py
python optimized_v2/scripts/cell4_testing_plots.py
```

Use `--train` when retraining models intentionally:

```powershell
python optimized_v2/scripts/cell2_rbfnn.py --train
python optimized_v2/scripts/cell3_benchmark.py --train
```

Without `--train`, Cell 2 and Cell 3 use saved models when available.

## Required Outputs

Cleaned data:

- `optimized_v2/outputs/cleaned_data/cleaned_hourly_data.xlsx`
- `optimized_v2/outputs/cleaned_data/cleaned_hourly_data.parquet`

RBFNN:

- `optimized_v2/metadata/overall_metrics/rbfnn_validation_testing_metrics.xlsx`
- `optimized_v2/outputs/rbfnn_forecast/Day_Ahead_24H_RBFNN_Forecast.xlsx`
- `optimized_v2/outputs/rbfnn_forecast/Day_Ahead_24H_RBFNN_Forecast.csv`
- `optimized_v2/metadata/overall_metrics/rbfnn_testing_predictions.xlsx`

Random Forest:

- `optimized_v2/metadata/overall_metrics/random_forest_validation_testing_metrics.xlsx`
- `optimized_v2/benchmark/random_forest/Day_Ahead_24H_RANDOM_FOREST.xlsx`
- `optimized_v2/benchmark/random_forest/Day_Ahead_24H_RANDOM_FOREST.csv`
- `optimized_v2/metadata/overall_metrics/random_forest_testing_predictions.xlsx`

XGBoost:

- `optimized_v2/metadata/overall_metrics/xgboost_validation_testing_metrics.xlsx`
- `optimized_v2/benchmark/xgboost/Day_Ahead_24H_XGBOOST.xlsx`
- `optimized_v2/benchmark/xgboost/Day_Ahead_24H_XGBOOST.csv`
- `optimized_v2/metadata/overall_metrics/xgboost_testing_predictions.xlsx`

Testing plots:

- `data/outputs/05_testing_plots/full_testing_split/*.png`

## Notes

- The time-series split remains chronological 70/15/15.
- The data is not randomly shuffled.
- Planned outage edits in `optimized_v2/outputs/outages_planning/Planned_Outages_Input.xlsx` affect the 24-hour forecasts.
- Generated data, outputs, models, and caches are intentionally ignored by Git.

# Optimized Agus Hydropower Forecasting Pipeline

## 1. Purpose

This folder contains the optimized version of the Agus cascaded hydropower day-ahead forecasting workflow. It was created separately from the original working pipeline so the baseline scripts remain untouched and runnable.

The optimized pipeline keeps the thesis identity:

> Cascade-aware, outage-aware, RBFNN-based short-term hydropower generation forecasting.

The primary model remains RBFNN. Random Forest and XGBoost are benchmark models only.

## 2. Branch And Preservation

- Working branch: `rbfnn-beat-benchmark-mape`
- Original scripts under `scripts/` were not modified.
- Optimized code lives under `optimized_version/scripts/`.
- Generated optimized models and outputs live locally under `optimized_version/models/` and `optimized_version/outputs/`.
- The repository `.gitignore` ignores model/output folders, so large generated artifacts are not committed by default.

## 3. Folder Structure

```text
optimized_version/
├── scripts/
│   ├── optimized_cell1_clean.py
│   ├── optimized_cell2_rbfnn.py
│   └── optimized_cell3_benchmark.py
├── models/
│   ├── rbfnn_<plant>.keras
│   ├── x_scaler_<plant>.pkl
│   ├── y_scaler_<plant>.pkl
│   ├── meta_<plant>.json
│   └── benchmarks/
├── outputs/
│   ├── cleaned_hourly_data.parquet
│   ├── Planned_Outages_Input.xlsx
│   ├── Day_Ahead_24H_Optimized_RBFNN_Forecast.xlsx
│   ├── Day_Ahead_24H_Optimized_RBFNN_Forecast.csv
│   └── benchmarks/
└── metadata/
    ├── validation_testing_metrics/
    ├── validation_daily_metrics/
    ├── testing_daily_metrics/
    └── benchmark_metrics/
```

## 4. Environment

Use the Conda environment that already has TensorFlow and the scientific stack installed:

```powershell
& 'C:\Users\Allen Mae\anaconda3\envs\ALLANTHESIS\python.exe' <script>
```

The default `python` on PATH is not sufficient because it does not include NumPy/TensorFlow.

## 5. Pipeline From The Start

### Step 1: Prepare optimized inputs

```powershell
& 'C:\Users\Allen Mae\anaconda3\envs\ALLANTHESIS\python.exe' optimized_version\scripts\optimized_cell1_clean.py
```

What it does:

- Reads the original cleaned hourly data and planned outage file from `data/outputs/01_runtime_outputs/`.
- Copies them into `optimized_version/outputs/`.
- Validates required plant target columns and checks for numeric NaNs.
- Does not overwrite original Cell 1 outputs.

### Step 2: Train optimized RBFNN and forecast

```powershell
& 'C:\Users\Allen Mae\anaconda3\envs\ALLANTHESIS\python.exe' optimized_version\scripts\optimized_cell2_rbfnn.py
```

What it does:

- Builds time, lag, rolling, ramp, outage, and cascade features.
- Trains one RBFNN residual/delta model per plant.
- Uses chronological split: 70% train, 15% validation, 15% testing.
- Tunes residual shrinkage on validation data.
- Saves optimized RBFNN models, scalers, metadata, metrics, daily metrics, and 24-hour forecast.
- Preserves outage adjustment logic:
  - `1` means unit ON/available.
  - `0` means unit OFF/unavailable.
  - Forecasts are adjusted by planned-vs-baseline unit availability.

### Step 3: Train benchmark models and forecast

```powershell
& 'C:\Users\Allen Mae\anaconda3\envs\ALLANTHESIS\python.exe' optimized_version\scripts\optimized_cell3_benchmark.py
```

What it does:

- Trains Random Forest and XGBoost benchmark models for each plant.
- Uses the same chronological split and operational MAPE definition.
- Saves benchmark metrics to `optimized_version/metadata/benchmark_metrics/`.
- Saves benchmark forecast outputs to:

```text
optimized_version/outputs/benchmarks/
├── Day_Ahead_24H_RANDOM_FOREST_Forecast.xlsx
├── Day_Ahead_24H_RANDOM_FOREST_Forecast.csv
├── Day_Ahead_24H_XGBOOST_Forecast.xlsx
└── Day_Ahead_24H_XGBOOST_Forecast.csv
```

If an Excel workbook is open and Windows locks the canonical `.xlsx` output, Cell 3 writes a fallback workbook beside it using the `_regenerated.xlsx` suffix. The canonical CSV is still refreshed.

## 6. Optimized RBFNN Method

The optimized RBFNN predicts the next-hour change in generation rather than direct absolute generation. The forecast is anchored to the latest actual generation:

```text
forecast = latest_generation + shrinkage * predicted_delta + validation_bias
```

This improves short-term continuity and reduces recursive drift. Model selection prioritizes lowest validation operational MAPE, then validation RMSE, then validation R2. A plant-specific ramp limit based on historical ramp behavior is applied during 24-hour forecasting.

The latest RBFNN optimization also uses:

- Hyperparameter search over RBF center counts `80`, `120`, and `180`.
- Learning-rate search over `0.001` and `0.0005`.
- Validation-tuned shrinkage over `0.05`, `0.10`, `0.20`, `0.35`, `0.50`, `0.75`, `1.00`, and `1.15`.
- Validation-derived bias correction for the residual forecast.
- Rainfall lag features in the optimized RBFNN feature set.
- Validation-derived bin calibration for Agus 1 and Agus 5.

## 7. Cascade Logic

The cascade order is preserved:

```text
Agus 1 -> Agus 2 -> Agus 4 -> Agus 5 -> Agus 6 -> Agus 7
```

Feature engineering includes upstream generation information, especially for the downstream sequence:

```text
Agus 5 -> Agus 6 -> Agus 7
```

During recursive forecasting, newly predicted plant generation is fed back into the forecast history so downstream features can use updated upstream behavior.

## 8. Operational MAPE

Operational MAPE is used because true zero or near-zero generation makes ordinary percentage error unstable.

For each plant:

```text
MAPE threshold = max(1 MW, 1% of installed plant capacity)
```

Rows where actual generation is below that threshold are excluded from MAPE averaging. MAE, RMSE, and R2 are still computed on all rows.

This prevents real outage/shutdown hours from creating mathematically meaningless MAPE spikes while still preserving outage behavior through separate generation and availability logic.

## 9. Current Optimized RBFNN Results

Saved file:

```text
optimized_version/metadata/validation_testing_metrics/optimized_rbfnn_validation_testing_metrics.xlsx
```

Current summary after the final benchmark-MAPE optimization:

| Plant | Validation MAPE | Validation R2 | Testing MAPE | Testing R2 |
|---|---:|---:|---:|---:|
| agus1 | 3.314 | 0.932 | 3.435 | 0.899 |
| agus2 | 1.080 | 0.822 | 1.372 | 0.912 |
| agus4 | 0.827 | 0.953 | 0.611 | 0.945 |
| agus5 | 4.141 | 0.903 | 4.044 | 0.901 |
| agus6 | 2.413 | 0.835 | 1.404 | 0.990 |
| agus7 | 3.829 | 0.823 | 4.222 | 0.904 |

All plants meet:

- Operational MAPE below 10%.
- Non-negative R2.
- R2 at or above 0.80.

## 10. Benchmark Comparison Results

Saved file:

```text
optimized_version/metadata/validation_testing_metrics/rbfnn_vs_benchmark_mape_comparison.xlsx
```

The final comparison confirms that RBFNN beats the best benchmark MAPE on both validation and testing for all six plants:

| Plant | Best Validation Benchmark | Validation Margin | Best Testing Benchmark | Testing Margin | RBFNN Wins Both |
|---|---|---:|---|---:|---|
| agus1 | XGBoost | 0.121 | Random Forest | 0.052 | True |
| agus2 | XGBoost | 0.640 | Random Forest | 0.789 | True |
| agus4 | XGBoost | 0.805 | XGBoost | 0.461 | True |
| agus5 | XGBoost | 0.023 | Random Forest | 0.157 | True |
| agus6 | XGBoost | 0.279 | Random Forest | 18.475 | True |
| agus7 | XGBoost | 0.245 | Random Forest | 0.072 | True |

## 11. Output Files

Primary optimized RBFNN forecast:

```text
optimized_version/outputs/Day_Ahead_24H_Optimized_RBFNN_Forecast.xlsx
optimized_version/outputs/Day_Ahead_24H_Optimized_RBFNN_Forecast.csv
```

Benchmark forecasts:

```text
optimized_version/outputs/benchmarks/Day_Ahead_24H_RANDOM_FOREST_Forecast.xlsx
optimized_version/outputs/benchmarks/Day_Ahead_24H_RANDOM_FOREST_Forecast.csv
optimized_version/outputs/benchmarks/Day_Ahead_24H_XGBOOST_Forecast.xlsx
optimized_version/outputs/benchmarks/Day_Ahead_24H_XGBOOST_Forecast.csv
```

Metrics:

```text
optimized_version/metadata/validation_testing_metrics/optimized_rbfnn_validation_testing_metrics.xlsx
optimized_version/metadata/validation_testing_metrics/rbfnn_vs_benchmark_mape_comparison.xlsx
optimized_version/metadata/validation_daily_metrics/
optimized_version/metadata/testing_daily_metrics/
optimized_version/metadata/benchmark_metrics/optimized_benchmark_validation_testing_metrics.xlsx
optimized_version/metadata/optimized_vs_original_summary.xlsx
```

## 12. Notes And Limitations

- The optimized Cell 1 currently uses the original cleaned dataset as its trusted input and isolates it under `optimized_version/outputs/`.
- The original summary workbook available at the time of comparison only contained Agus 7, so unavailable original rows are marked as `not_available`.
- TensorFlow prints CPU/GPU and retracing warnings on Windows. These are runtime warnings and did not prevent successful training or output generation.
- TensorFlow/Keras model saving may need permission to write temporary files outside the workspace on Windows.
- Cell 3 suppresses pandas fragmentation warnings and uses single-threaded Random Forest/XGBoost training to avoid Windows joblib handle permission errors in restricted shells.
- If benchmark `.xlsx` forecast files are open in Excel, Cell 3 writes `_regenerated.xlsx` fallback files and still refreshes the canonical `.csv` outputs.

## 13. Recommended Rerun Order

Run the full optimized pipeline in this order:

```powershell
& 'C:\Users\Allen Mae\anaconda3\envs\ALLANTHESIS\python.exe' optimized_version\scripts\optimized_cell1_clean.py
& 'C:\Users\Allen Mae\anaconda3\envs\ALLANTHESIS\python.exe' optimized_version\scripts\optimized_cell2_rbfnn.py
& 'C:\Users\Allen Mae\anaconda3\envs\ALLANTHESIS\python.exe' optimized_version\scripts\optimized_cell3_benchmark.py
```

After rerunning, inspect:

```powershell
& 'C:\Users\Allen Mae\anaconda3\envs\ALLANTHESIS\python.exe' -c "import pandas as pd; print(pd.read_excel('optimized_version/metadata/validation_testing_metrics/rbfnn_vs_benchmark_mape_comparison.xlsx').to_string(index=False))"
```

Success means `rbfnn_wins_both` is `True` for all six plants.

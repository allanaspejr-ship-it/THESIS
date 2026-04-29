# Agus 5 and Agus 7 RBFNN Shape Optimization

Date: 2026-04-29
Branch: `rbfnn-beat-benchmark-mape`

## Purpose

This update improves the optimized Cell 2 RBFNN day-ahead forecast behavior for Agus 5 and Agus 7. The previous validation and testing MAPE values were acceptable, but the 24-hour forecast shape was too flat, especially for Agus 7, and Agus 5 had many hourly APE values around 7-13% when compared with the July 1 actual generation file.

The RBFNN remains the main thesis model. Random Forest and XGBoost remain benchmarks only.

## Main Changes

- Updated `optimized_version/scripts/optimized_cell2_rbfnn.py`.
- Added target-hour time features so one-step training better represents the hour being forecast, not only the current source hour.
- Added same-hour-yesterday and same-hour-last-week target anchors.
- Added Agus 5 and Agus 7 hourly residual correction learned from validation data.
- Added forecast-time same-hour-yesterday profile blending for Agus 5 and Agus 7 to reduce overly flat recursive 24-hour forecasts.
- Added July 1 actual-vs-forecast diagnostics when `july 1, 2025.xlsx` exists in the project root.
- Added diagnostic plots for Agus 5 and Agus 7 actual-vs-forecast behavior.

## Current Validation and Testing Result

The updated RBFNN still beats the best benchmark MAPE for validation and testing across all plants.

| Plant | RBFNN Val MAPE | Best Benchmark Val MAPE | RBFNN Test MAPE | Best Benchmark Test MAPE | Wins Both |
|---|---:|---:|---:|---:|---|
| agus1 | 3.448461 | 3.451841 | 3.472313 | 3.486362 | True |
| agus2 | 1.080237 | 1.744495 | 1.372393 | 2.161182 | True |
| agus4 | 0.826761 | 1.717458 | 0.610687 | 1.051261 | True |
| agus5 | 4.090823 | 4.180766 | 4.039603 | 4.200485 | True |
| agus6 | 2.412584 | 2.664092 | 1.404527 | 19.879414 | True |
| agus7 | 3.806186 | 4.083275 | 4.209302 | 4.294213 | True |

The comparison workbook is saved at:

```text
optimized_version/metadata/validation_testing_metrics/rbfnn_vs_benchmark_mape_comparison.xlsx
```

## July 1 Diagnostic Result

Using `july 1, 2025.xlsx` as an external actual-generation check:

| Plant | July 1 MAPE | MAE MW | RMSE MW | APE <= 3% Hours |
|---|---:|---:|---:|---:|
| agus5 | 5.241556 | 2.329276 | 2.796157 | 7 |
| agus7 | 7.915319 | 3.119803 | 3.626240 | 6 |

These values are diagnostic only and are not used for model fitting.

## Output Files

- `optimized_version/outputs/Day_Ahead_24H_Optimized_RBFNN_Forecast_20260429_130533.xlsx`
- `optimized_version/metadata/actual_forecast_diagnostics/july_1_2025_actual_vs_optimized_rbfnn.xlsx`
- `optimized_version/metadata/plots/agus5_actual_vs_forecast_july_1_2025.png`
- `optimized_version/metadata/plots/agus7_actual_vs_forecast_july_1_2025.png`

The normal forecast workbook name could not be overwritten during the last run because it was locked by another program:

```text
optimized_version/outputs/Day_Ahead_24H_Optimized_RBFNN_Forecast.xlsx
```

Close the workbook before rerunning Cell 2 if that exact filename needs to be replaced.

## Reproducibility Notes

Run from the project root:

```powershell
& 'C:\Users\Allen Mae\anaconda3\envs\ALLANTHESIS\python.exe' optimized_version\scripts\optimized_cell2_rbfnn.py
```

If TensorFlow/Keras cannot save models because of Windows temporary file permissions, run the command with permission to access the normal Windows temp directory.


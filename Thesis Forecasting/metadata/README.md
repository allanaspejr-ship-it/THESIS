# Metadata Folder Guide

Use these folders as the source of truth:

- `day_ahead_backtest/`: true leakage-safe rolling 24-hour day-ahead validation/testing outputs. Use this for methodology and reviewer questions about day-ahead forecasting.
- `overall_metrics/`: strict day-ahead model-level summaries and prediction exports used by scripts.
- `overall_comparison/`: compact strict day-ahead comparison tables for thesis reporting.
- `training_validation_loss/`: RBFNN training history workbooks.
- `rbfnn/`: RBFNN-specific calibration/profile reports only.
- `day_ahead_outage_informed/`: supplementary scenario analysis using forecast-day unit availability as a known-availability proxy. Do not cite this as the primary leakage-free day-ahead result.

Removed duplicate organized copies, one-step validation/testing outputs, and stale `_regenerated.xlsx` files to avoid conflicting metrics.

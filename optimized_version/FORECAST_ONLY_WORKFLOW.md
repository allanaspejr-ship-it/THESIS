# Optimized Forecast-Only Workflow

This optimized pipeline separates daily forecasting from model retraining.
For normal forecast updates, use the saved trained models and do not retrain.

## Directory Layout

The optimized pipeline uses these main folders:

- `data/` - raw historical data source files.
- `outputs/cleaned_data/` - cleaned historical data used by the models.
- `outputs/outages_planning/` - outage plan input for the forecast day.
- `outputs/rbfnn_forecast/` - optimized RBFNN forecast outputs.
- `models/rbfnn/` - saved trained RBFNN models, scalers, and model metadata.
- `models/random_forest/` - saved trained Random Forest benchmark models.
- `models/xgboost/` - saved trained XGBoost benchmark models.
- `metadata/validation_metrics/` - validation metrics.
- `metadata/testing_metrics/` - testing metrics.
- `metadata/overall_metrics/` - overall all-Agus metrics and comparisons.
- `benchmark/random_forest/` - Random Forest benchmark forecast outputs.
- `benchmark/xgboost/` - XGBoost benchmark forecast outputs.
- `scripts/` - runnable pipeline scripts.

## Normal Daily Forecast

Use this flow when the trained models already exist and only the latest data
or outage planning changed.

1. Update the latest data and outage plan.
2. Run Cell 1 only if the cleaned data must be refreshed:

```powershell
python optimized_version/scripts/optimized_cell1_clean.py
```

3. Run Cell 2 to generate the RBFNN forecast from saved models:

```powershell
python optimized_version/scripts/optimized_cell2_rbfnn.py
```

4. Run Cell 3 to generate benchmark forecasts from saved models:

```powershell
python optimized_version/scripts/optimized_cell3_benchmark.py
```

Cell 2 and Cell 3 are forecast-only by default. They load existing trained
models and write new forecasts. They do not retrain unless `--train` is used.

## Outage Planning Test

To test whether outage changes affect the forecast:

1. Edit:

```text
optimized_version/outputs/outages_planning/Planned_Outages_Input.xlsx
```

2. Use outage values:

```text
1 = unit available
0 = unit unavailable / outage
```

3. Save and close Excel.
4. Run:

```powershell
python optimized_version/scripts/optimized_cell2_rbfnn.py
python optimized_version/scripts/optimized_cell3_benchmark.py
```

Do not rerun Cell 1 after manual outage edits unless you want to restore the
baseline copied outage file.

## Intentional Retraining

Retrain only when model code, feature engineering, optimization settings, or
training data strategy changes.

Retrain all RBFNN models:

```powershell
python optimized_version/scripts/optimized_cell2_rbfnn.py --train
```

Retrain one RBFNN plant:

```powershell
python optimized_version/scripts/optimized_cell2_rbfnn.py --train agus1
```

Retrain benchmark models:

```powershell
python optimized_version/scripts/optimized_cell3_benchmark.py --train
```

Without `--train`, missing or broken saved models stop the script with an
error instead of retraining silently. This protects optimized saved models from
being changed by accident.

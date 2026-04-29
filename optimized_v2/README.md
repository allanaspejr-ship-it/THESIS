# Optimized V2 Workflow

`optimized_v2` is an isolated revision of the HYDRO_FORECASTING pipeline. It keeps the original `scripts/` and existing `optimized_version/` workflow intact while providing the revised cleaned-data schema, model outputs, metrics files, and testing-split plots.

## Raw Data Source

Place the raw Excel workbook in:

```text
optimized_v2/data/DATA(JAN2024-JUNE2025).xlsx
```

Cell 1 will use that file first. If it is missing, it will use the first non-temporary `.xlsx` file found in `optimized_v2/data/`.

## Scripts

- `scripts/cell1_clean_data.py` cleans raw data, prepares hourly records, saves cleaned Excel/parquet files, and creates the planned outage workbook.
- `scripts/cell2_rbfnn.py` evaluates or trains RBFNN models, creates metrics, testing predictions, and the unit-level 24-hour RBFNN forecast.
- `scripts/cell3_benchmark.py` evaluates or trains Random Forest and XGBoost benchmark models, creates separated metrics files, testing predictions, and 24-hour benchmark forecasts.
- `scripts/cell4_testing_plots.py` reads the three testing prediction workbooks and creates full testing-split actual-vs-forecast comparison plots.

## Commands

Run from the repository root:

```powershell
python optimized_v2/scripts/cell1_clean_data.py
python optimized_v2/scripts/cell2_rbfnn.py
python optimized_v2/scripts/cell3_benchmark.py
python optimized_v2/scripts/cell4_testing_plots.py
```

Retrain intentionally:

```powershell
python optimized_v2/scripts/cell2_rbfnn.py --train
python optimized_v2/scripts/cell3_benchmark.py --train
```

## Output Conventions

Metrics files use the same columns for RBFNN, Random Forest, and XGBoost:

```text
model
plant
feature_count
val_operational_mape
val_mae
val_rmse
val_r2
test_operational_mape
test_mae
test_rmse
test_r2
```

Testing prediction files use:

```text
Date
Hour
datetime
plant
actual_generation
predicted_generation
model
```

The RBFNN forecast includes all Agus unit-level generation columns, plant totals, and `total_cascade_generation`.

## Git Policy

Generated files under `data/`, `outputs/`, `models/`, and caches are ignored. Commit source code, documentation, environment files, and configuration only.

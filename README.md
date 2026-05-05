# HYDRO_FORECASTING

Hydropower forecasting workflow for the Agus cascade thesis project. The active pipeline is in `Thesis Forecasting/` and produces cleaned hourly datasets, RBFNN forecasts, Random Forest and XGBoost benchmark forecasts, model metrics, and thesis-ready figures.

## Repository Layout

```text
HYDRO_FORECASTING/
├── Thesis Forecasting/
│   ├── scripts/                 # Executable workflow cells
│   ├── thesis_figures/          # Tracked thesis figures
│   ├── README.md                # Workflow-specific notes
│   ├── PROJECT_SETUP_AND_USER_GUIDE.txt
│   ├── THESIS_FORECASTING_DOCUMENTATION.md
│   └── CHAPTER_3_4_5_CONTEXT.md
├── archive/                     # Legacy scripts, GUI, and historical outputs
├── requirements.txt             # Pinned Python dependencies
├── vscode_extensions.txt        # Recommended VS Code extensions
└── main.pdf                     # Thesis manuscript/reference PDF
```

Generated runtime folders such as `data/`, `outputs/`, `models/`, `benchmark/`, and `metadata/` are intentionally ignored by Git.

## Requirements

- 64-bit Python 3.11 on Windows
- Git
- Raw Excel workbook for the forecasting dataset

Install the pinned Python dependencies from the repository root:

```powershell
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

## Input Data

Place the raw workbook here:

```text
Thesis Forecasting/data/DATA(JAN2024-JUNE2025).xlsx
```

If that exact file is missing, Cell 1 uses the first non-temporary `.xlsx` workbook found in `Thesis Forecasting/data/`.

## Workflow

Run commands from the repository root.

### 1. Clean Raw Data

```powershell
python "Thesis Forecasting/scripts/cell1_clean_data.py"
```

Creates cleaned hourly data and outage planning inputs under:

```text
Thesis Forecasting/outputs/cleaned_data/
Thesis Forecasting/outputs/outages_planning/
```

### 2. Run RBFNN Forecast

Use saved models when available:

```powershell
python "Thesis Forecasting/scripts/cell2_rbfnn.py"
```

Retrain all RBFNN plant models intentionally:

```powershell
python "Thesis Forecasting/scripts/cell2_rbfnn.py" --train
```

Retrain selected plants only:

```powershell
python "Thesis Forecasting/scripts/cell2_rbfnn.py" --train agus5 agus7
```

Primary forecast outputs:

```text
Thesis Forecasting/outputs/rbfnn_forecast/Day_Ahead_24H_RBFNN_Forecast.xlsx
Thesis Forecasting/outputs/rbfnn_forecast/Day_Ahead_24H_RBFNN_Forecast.csv
```

### 3. Run Benchmark Forecasts

Use saved Random Forest and XGBoost models when available:

```powershell
python "Thesis Forecasting/scripts/cell3_benchmark.py"
```

Retrain benchmark models:

```powershell
python "Thesis Forecasting/scripts/cell3_benchmark.py" --train
```

Primary benchmark outputs:

```text
Thesis Forecasting/benchmark/random_forest/Day_Ahead_24H_RANDOM_FOREST.xlsx
Thesis Forecasting/benchmark/xgboost/Day_Ahead_24H_XGBOOST.xlsx
```

### 4. Generate Thesis Figures and Organized Metrics

```powershell
python "Thesis Forecasting/scripts/cell4_generate_thesis_and_metrics.py"
```

Creates thesis figures under:

```text
Thesis Forecasting/thesis_figures/
```

and organized metric workbooks under:

```text
Thesis Forecasting/metadata/
```

## Output Conventions

Model metrics use a common schema across RBFNN, Random Forest, and XGBoost:

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

Day-ahead forecasts include date, hour, plant-level totals, unit-level generation columns, and total cascade generation.

## Documentation Map

- `README.md`: repository entry point and quick start
- `Thesis Forecasting/README.md`: active workflow notes and output conventions
- `Thesis Forecasting/THESIS_FORECASTING_DOCUMENTATION.md`: detailed script responsibilities and folder migration notes
- `Thesis Forecasting/PROJECT_SETUP_AND_USER_GUIDE.txt`: user setup guide
- `Thesis Forecasting/CHAPTER_3_4_5_CONTEXT.md`: thesis chapter context
- `archive/`: historical documentation and older workflow versions

## Git and Release Notes

Commit source code, documentation, dependency files, configuration, and thesis figures that are intentionally versioned. Avoid committing raw datasets, generated model binaries, generated runtime outputs, and local environment files.

Recommended release flow:

```powershell
git status --short
git add README.md
git commit -m "Add project documentation"
git tag v1.19-documentation
git push origin HEAD
git push origin v1.19-documentation
```

# Agus Hydropower Forecasting Streamlit GUI

This folder contains a thesis-ready Streamlit dashboard for the Agus hydropower forecasting project. It does not rewrite the model scripts. The GUI reads existing pipeline outputs and calls the optimized scripts through subprocess.

## Install Requirements

From the project root:

```powershell
pip install -r streamlit_gui_app/requirements_gui.txt
```

Use the same Python or Conda environment that can already run TensorFlow and the forecasting scripts. In the project notes, this is the `ALLANTHESIS` environment.

## Run The GUI

From the project root:

```powershell
streamlit run streamlit_gui_app/app.py
```

For subprocess pipeline runs, the app automatically prefers:

```text
C:\Users\Allen Mae\anaconda3\envs\ALLANTHESIS\python.exe
```

If that environment is moved or renamed, edit `PYTHON_EXE` near the top of `streamlit_gui_app/app.py`.

## Expected Paths

The dashboard expects these project paths:

| Purpose | Path |
|---|---|
| Raw Excel input | `data/raw/DATA(JAN2024-JUNE2025).xlsx` |
| Original cleaning script | `scripts/cell1_clean_data.py` |
| Optimized cleaner | `optimized_version/scripts/optimized_cell1_clean.py` |
| Optimized RBFNN forecast script | `optimized_version/scripts/optimized_cell2_rbfnn.py` |
| Benchmark script | `optimized_version/scripts/optimized_cell3_benchmark.py` |
| Cleaned data | `optimized_version/outputs/cleaned_data/cleaned_hourly_data.parquet` |
| Outage planning file | `optimized_version/outputs/outages_planning/Planned_Outages_Input.xlsx` |
| RBFNN forecast output | `optimized_version/outputs/rbfnn_forecast/Day_Ahead_24H_Optimized_RBFNN_Forecast.csv` |
| RBFNN metrics | `optimized_version/metadata/overall_metrics/optimized_rbfnn_validation_testing_metrics.xlsx` |
| Benchmark metrics | `optimized_version/metadata/overall_metrics/optimized_benchmark_validation_testing_metrics.xlsx` |

## What Each Button Does

### Upload / Update Excel Data

Saves the uploaded Excel file to:

```text
data/raw/DATA(JAN2024-JUNE2025).xlsx
```

This replaces the raw input file used by the existing original cleaning script.

### Run Cleaning Pipeline

Runs:

```powershell
python scripts/cell1_clean_data.py
python optimized_version/scripts/optimized_cell1_clean.py
```

The first script regenerates baseline cleaned outputs. The second script copies and validates those outputs into the optimized workflow folder.

### Run Optimized Input Refresh Only

Runs only:

```powershell
python optimized_version/scripts/optimized_cell1_clean.py
```

Use this when the baseline cleaned outputs already exist and only the optimized input folder needs to be refreshed.

### Load Current Outage Plan

Loads:

```text
optimized_version/outputs/outages_planning/Planned_Outages_Input.xlsx
```

The file is shown in an editable Streamlit table.

### Save Edited Outage Plan

Validates the outage table and saves it back to:

```text
optimized_version/outputs/outages_planning/Planned_Outages_Input.xlsx
```

Only `0` and `1` values are allowed in outage status columns.

### Run Day-Ahead Forecast

Runs:

```powershell
python optimized_version/scripts/optimized_cell2_rbfnn.py
```

This uses saved optimized RBFNN models. It does not pass `--train`, so it does not intentionally retrain or replace model files.

## Outage Planning Notes

Outage values use the existing project convention:

```text
1 = unit available/running
0 = unit outage/unavailable
```

After saving an edited outage plan, the next forecast run uses that edited file automatically.

## Dashboard Sections

- **Home**: Project title, latest timestamp, and artifact status cards.
- **Update Excel Data**: Excel upload, cleaning pipeline execution, and cleaned-output preview.
- **Cleaned Data**: Cleaned hourly table, date range, dimensions, generation columns, and downloads.
- **Outage Planning**: Editable 24-hour outage table with validation.
- **Forecast**: RBFNN forecast run button, forecast tables, cascade totals, downloads, and Plotly charts.
- **Metrics**: RBFNN metrics, benchmark metrics, daily metrics, and comparison plots.
- **About System**: Integration paths and workflow notes.

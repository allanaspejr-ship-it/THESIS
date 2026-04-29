# Streamlit Dashboard Documentation

## Purpose

The Streamlit GUI provides a thesis-ready interface for the Agus hydropower day-ahead forecasting workflow. It is designed to help users update source Excel data, refresh cleaned inputs, edit planned outages, run the optimized RBFNN forecast, inspect forecast results, and review validation/testing metrics.

The dashboard is isolated in `streamlit_gui_app/` and does not modify the existing model or cleaning scripts.

## Integrated Pipeline

The GUI uses the optimized workflow:

| Step | Script or Artifact |
|---|---|
| Raw Excel input | `data/raw/DATA(JAN2024-JUNE2025).xlsx` |
| Original cleaning | `scripts/cell1_clean_data.py` |
| Optimized input refresh | `optimized_version/scripts/optimized_cell1_clean.py` |
| RBFNN forecast | `optimized_version/scripts/optimized_cell2_rbfnn.py` |
| Benchmark outputs and metrics | `optimized_version/scripts/optimized_cell3_benchmark.py` |
| Cleaned optimized data | `optimized_version/outputs/cleaned_data/cleaned_hourly_data.parquet` |
| Outage planning input | `optimized_version/outputs/outages_planning/Planned_Outages_Input.xlsx` |
| RBFNN forecast output | `optimized_version/outputs/rbfnn_forecast/Day_Ahead_24H_Optimized_RBFNN_Forecast.csv` |

## Dashboard Pages

### Home

Shows the project title, a short system description, latest cleaned-data timestamp, and status cards for:

- Cleaned data availability
- Planned outage file availability
- RBFNN forecast availability
- Validation/testing metrics availability

### Update Excel Data

Allows users to upload a new Excel file and save it into the raw input path expected by the existing cleaner. The page can run:

```powershell
python scripts/cell1_clean_data.py
python optimized_version/scripts/optimized_cell1_clean.py
```

This refreshes the baseline cleaned outputs first, then updates the optimized input folder used by forecasting.

### Cleaned Data

Displays the optimized cleaned hourly data with:

- Table preview
- Date range
- Row and column count
- Available Agus generation columns
- CSV and Excel download buttons

### Outage Planning

Loads and edits:

```text
optimized_version/outputs/outages_planning/Planned_Outages_Input.xlsx
```

Outage values follow the project convention:

```text
1 = unit available/running
0 = unit outage/unavailable
```

The dashboard validates that all outage status columns contain only `0` or `1` before saving.

### Forecast

Runs:

```powershell
python optimized_version/scripts/optimized_cell2_rbfnn.py
```

The command runs forecast-only mode and does not pass `--train`, so saved optimized RBFNN models are used. The page displays:

- 24-hour forecast table
- Per-Agus total generation table
- Total cascade generation table
- CSV and Excel forecast downloads
- Plotly forecast visualizations

The dashboard automatically detects per-unit forecast columns if future model outputs include them.

### Metrics

Loads RBFNN, benchmark, validation, and testing metrics from optimized metadata folders. It displays summary tables and Plotly charts for:

- MAPE
- MAE
- RMSE
- R2
- RBFNN vs Random Forest and XGBoost comparisons when benchmark files exist

### About System

Summarizes the configured paths, pipeline entrypoints, and forecast behavior.

## Runtime Notes

The app prefers the thesis Conda environment for subprocess execution:

```text
C:\Users\Allen Mae\anaconda3\envs\ALLANTHESIS\python.exe
```

If that environment is unavailable, it falls back to the Python interpreter running Streamlit.

## Run Command

From the project root:

```powershell
& 'C:\Users\Allen Mae\anaconda3\envs\ALLANTHESIS\python.exe' -m streamlit run streamlit_gui_app\app.py --browser.gatherUsageStats false
```

## Safety Boundaries

- The dashboard does not edit existing scripts.
- Forecast execution does not retrain models.
- Model files are not overwritten unless the underlying optimized script is run manually with `--train`.
- Uploaded Excel data only replaces the expected raw input file after the user clicks the upload/update button.
- Outage edits are limited to the optimized planned outage workbook.

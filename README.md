# HYDRO_FORECASTING

Hydropower day-ahead forecasting project for the Agus cascade. The cleaned repository contains the active Streamlit dashboard, data preparation script, RBFNN forecasting workflow, benchmark model workflow, saved model artifacts, and generated runtime outputs.

## Active Project

The working application is inside:

```text
Thesis Forecasting/
```

Current structure:

```text
Thesis Forecasting/streamlit_app.py              Streamlit dashboard
Thesis Forecasting/RUN_DASHBOARD.bat             Windows dashboard launcher
Thesis Forecasting/PROJECT_SETUP_AND_USER_GUIDE.txt
Thesis Forecasting/data/                         Raw source Excel file
Thesis Forecasting/outputs/                      Cleaned data, outage plan, forecasts
Thesis Forecasting/models/                       Saved RBFNN, Random Forest, and XGBoost models
Thesis Forecasting/metadata/                     Metrics, predictions, audits, backtests
Thesis Forecasting/scripts/cell1_clean_data.py   Data cleaning workflow
Thesis Forecasting/scripts/cell2_rbfnn.py        RBFNN training and forecast workflow
Thesis Forecasting/scripts/cell3_benchmark.py    Random Forest and XGBoost benchmark workflow
Thesis Forecasting/scripts/cell4_generate_thesis_and_metrics.py
requirements.txt                                 Python dependencies
```

## Requirements

Use 64-bit Python 3.11 on Windows.

Install dependencies from the repository root:

```powershell
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

## Run The Dashboard

Open this file from Windows Explorer:

```text
Thesis Forecasting/RUN_DASHBOARD.bat
```

Or run manually from the repository root:

```powershell
python -m streamlit run "Thesis Forecasting/streamlit_app.py"
```

## Full Workflow

Run these from the repository root.

Clean raw Excel data and create the planned outage workbook:

```powershell
python "Thesis Forecasting/scripts/cell1_clean_data.py"
```

Generate the RBFNN day-ahead forecast using saved models:

```powershell
python "Thesis Forecasting/scripts/cell2_rbfnn.py" --forecast-only
```

Generate benchmark forecasts using saved Random Forest and XGBoost models:

```powershell
python "Thesis Forecasting/scripts/cell3_benchmark.py" --forecast-only
```

Retrain models only when intentionally updating model artifacts:

```powershell
python "Thesis Forecasting/scripts/cell2_rbfnn.py" --train
python "Thesis Forecasting/scripts/cell3_benchmark.py" --train
```

Optional: regenerate metrics and thesis-style figures/tables:

```powershell
python "Thesis Forecasting/scripts/cell4_generate_thesis_and_metrics.py"
```

## Documentation

Use the setup guide for a complete step-by-step installation and forecasting process:

```text
Thesis Forecasting/PROJECT_SETUP_AND_USER_GUIDE.txt
```

## Important Outputs

```text
Thesis Forecasting/outputs/cleaned_data/cleaned_hourly_data.xlsx
Thesis Forecasting/outputs/outages_planning/Planned_Outages_Input.xlsx
Thesis Forecasting/outputs/rbfnn_forecast/Day_Ahead_24H_RBFNN_Forecast.xlsx
Thesis Forecasting/outputs/random_forest_forecast/Day_Ahead_24H_RANDOM_FOREST.xlsx
Thesis Forecasting/outputs/xgboost_forecast/Day_Ahead_24H_XGBOOST.xlsx
```

The dashboard reads from `data`, `outputs`, `models`, `metadata`, and `scripts` under `Thesis Forecasting/`.

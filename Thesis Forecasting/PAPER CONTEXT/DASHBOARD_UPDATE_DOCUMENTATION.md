# NPC Agus Cascade Streamlit Dashboard Update Documentation

This document summarizes the Streamlit dashboard, launcher, dependency, and VS Code setup updates made for the thesis project:

```text
AI-Based Short-Term Generation Forecasting of Cascaded Hydroelectric Power Plants
```

The dashboard is focused on NPC Agus Cascade hydropower day-ahead forecasting using the main RBFNN model only.

## Updated Files

```text
Thesis Forecasting/streamlit_app.py
Thesis Forecasting/RUN_DASHBOARD.bat
requirements.txt
vscode_extensions.txt
.vscode/extensions.json
```

No forecasting backend scripts were changed:

```text
Thesis Forecasting/scripts/cell1_clean_data.py
Thesis Forecasting/scripts/cell2_rbfnn.py
```

## Streamlit Dashboard

The main dashboard file is:

```text
Thesis Forecasting/streamlit_app.py
```

Run it from inside the `Thesis Forecasting` folder:

```powershell
streamlit run streamlit_app.py
```

The app opens locally at:

```text
http://localhost:8501
```

## Dashboard Pages

The dashboard now uses five operational pages:

```text
Dashboard Overview
Data Management
Planned Outage Planning
RBFNN Forecasting
System Information
```

### Dashboard Overview

Read-only operational summary page.

Shows:

```text
Total Cascade Forecast
Peak Cascade Output
Unavailable Unit-Hours
Latest Cleaned Timestamp
Latest Forecast Timestamp
Total Generation per Agus Plant Hourly
Total Generation per Agus Plant Daily Total
Recent Activity and Status
```

The only action button is:

```text
Refresh Data From Disk
```

### Data Management

Used only for raw Excel update and data cleaning.

Actions:

```text
Upload Excel File Optional
Refresh Data From Disk
Run Data Cleaning
```

The cleaning button runs:

```powershell
python scripts/cell1_clean_data.py
```

Confirmation cards show:

```text
Raw Excel Status
Raw Excel Last Modified
Latest Raw Data Date
Cleaned Data Status
Cleaned Data Last Modified
Latest Cleaned Data Date
Total Cleaned Rows
Total Cleaned Columns
```

### Planned Outage Planning

Used to edit planned unit availability.

Input file:

```text
outputs/outages_planning/Planned_Outages_Input.xlsx
```

Rules:

```text
1 = ON / available / running
0 = OFF / unavailable / planned outage
```

Buttons:

```text
Save Outage Plan
Forecast Day-Ahead
Retrain RBFNN Model
```

Forecast button runs:

```powershell
python scripts/cell2_rbfnn.py
```

Retrain button runs:

```powershell
python scripts/cell2_rbfnn.py --train
```

### RBFNN Forecasting

View-only page for forecast results.

Loads:

```text
outputs/rbfnn_forecast/Day_Ahead_24H_RBFNN_Forecast.xlsx
outputs/rbfnn_forecast/Day_Ahead_24H_RBFNN_Forecast.csv
```

Shows:

```text
Total Cascade Forecast
Peak Cascade Output
Affected Plants
Forecast Generated
Forecast Table
Download Forecast Excel
Download Forecast CSV
Total Generation per Agus Plant Hourly
Total Generation per Agus Plant Daily Total
```

### System Information

Shows:

```text
Forecasting Workflow
File Status Overview
System Details
```

System details include:

```text
Python Version
Streamlit Version
Forecast Model: RBFNN
Forecast Horizon: 24 Hours Day-Ahead
Timezone: Asia/Manila
Last System Check
```

## UI and Layout Updates

The dashboard was redesigned with a dark navy/blue industrial monitoring style.

Main UI changes:

```text
Dark navy dashboard background
Fixed dark sidebar titled NPC Agus Operations
Unified page title block on every page
Blue section headers with white text
Dark KPI and status cards
Readable white/light-blue text
Blue buttons
Red only for outage/off warnings
Plotly charts styled for dark background
No benchmark model pages
No operational metrics page
No long file paths shown in user-facing cards
```

The following layout issues were fixed:

```text
Removed empty blue/light-blue boxes
Moved section titles into proper header boxes
Removed floating text outside boxes
Improved card spacing and row gaps
Improved upload panel visibility
Changed washed-out light cards into dark navy cards
Styled graph labels, ticks, legends, hover labels, and grids for dark mode
```

## Preserved Backend Logic

The Streamlit app only calls existing scripts and reads existing outputs.

It does not change model architecture, training logic, cleaning logic, or forecast logic.

Preserved commands:

```powershell
python scripts/cell1_clean_data.py
python scripts/cell2_rbfnn.py
python scripts/cell2_rbfnn.py --train
```

Preserved input/output paths:

```text
data/DATA(JAN2024-JUNE2025).xlsx
outputs/cleaned_data/cleaned_hourly_data.xlsx
outputs/cleaned_data/cleaned_hourly_data.parquet
outputs/outages_planning/Planned_Outages_Input.xlsx
outputs/rbfnn_forecast/Day_Ahead_24H_RBFNN_Forecast.xlsx
outputs/rbfnn_forecast/Day_Ahead_24H_RBFNN_Forecast.csv
models/rbfnn/
metadata/
scripts/
```

## Model Files Used by Forecasting

The Streamlit dashboard does not directly load the model. It runs:

```powershell
python scripts/cell2_rbfnn.py
```

The script loads the saved RBFNN models from:

```text
models/rbfnn/
```

Expected model files:

```text
rbfnn_agus1.keras
rbfnn_agus2.keras
rbfnn_agus4.keras
rbfnn_agus5.keras
rbfnn_agus6.keras
rbfnn_agus7.keras
```

Expected scaler and metadata files follow the same plant pattern:

```text
x_scaler_agus1.pkl
y_scaler_agus1.pkl
meta_agus1.json
```

and similarly for:

```text
agus2
agus4
agus5
agus6
agus7
```

## Windows Batch Launcher

A Windows launcher was added:

```text
Thesis Forecasting/RUN_DASHBOARD.bat
```

Content:

```bat
@echo off
cd /d "C:\Users\Allen Mae\Desktop\HYDRO_FORECASTING\Thesis Forecasting"

python -m streamlit run streamlit_app.py

pause
```

Use it by double-clicking:

```text
RUN_DASHBOARD.bat
```

The `pause` command keeps the terminal open if there is an error.

## Python Requirements

The root `requirements.txt` was updated for the dashboard and forecasting workflow.

Install with:

```powershell
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

Important included packages:

```text
streamlit
pandas
openpyxl
plotly
pyarrow
watchdog
numpy
scikit-learn
tensorflow
joblib
xgboost
matplotlib
jupyter
ipykernel
```

## VS Code Extensions

The recommended extension list was updated in:

```text
vscode_extensions.txt
.vscode/extensions.json
```

Recommended extensions:

```text
ms-python.python
ms-python.vscode-pylance
ms-toolsai.jupyter
ms-python.debugpy
redhat.vscode-yaml
ms-toolsai.datawrangler
GrapeCity.gc-excelviewer
mechatroner.rainbow-csv
charliermarsh.ruff
```

Install from the VS Code Extensions panel, or use:

```powershell
code --install-extension ms-python.python
code --install-extension ms-python.vscode-pylance
code --install-extension ms-toolsai.jupyter
code --install-extension ms-python.debugpy
code --install-extension redhat.vscode-yaml
code --install-extension ms-toolsai.datawrangler
code --install-extension GrapeCity.gc-excelviewer
code --install-extension mechatroner.rainbow-csv
code --install-extension charliermarsh.ruff
```

## Recommended Workflow

1. Install requirements:

```powershell
python -m pip install -r requirements.txt
```

2. Open the dashboard:

```powershell
cd /d "C:\Users\Allen Mae\Desktop\HYDRO_FORECASTING\Thesis Forecasting"
streamlit run streamlit_app.py
```

3. Update or confirm raw Excel data in the Data Management page.

4. Run data cleaning.

5. Edit planned outages.

6. Save the outage plan.

7. Run Forecast Day-Ahead.

8. View RBFNN Forecasting results and download outputs.

## Notes

The dashboard is intended for thesis presentation and possible NPC deployment-style demonstration. It is operationally focused and excludes benchmark model pages from the deployed interface.

Benchmark-related files and scripts may still exist in the project for thesis documentation, but the Streamlit dashboard uses the RBFNN workflow only.

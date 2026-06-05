# HYDRO_FORECASTING

Hydropower forecasting project for the Agus cascade. The repository contains the thesis-ready forecasting workflow, Streamlit dashboard, model evaluation scripts, benchmark comparisons, generated figures, and paper/defense support materials.

## Project Scope

The main production workflow forecasts day-ahead hydropower generation with an RBFNN model. Benchmark models such as Random Forest and XGBoost are included for thesis comparison, but the dashboard is focused on the final RBFNN forecasting workflow.

The system supports:

- Raw Excel data cleaning and hourly data preparation.
- RBFNN model training, validation, testing, and fast forecast-only execution.
- Day-ahead hydropower generation forecasting.
- Planned outage input management.
- Thesis figures, metrics tables, and benchmark comparison outputs.
- A Streamlit dashboard for daily forecasting use.

## Main Folder

The active thesis-ready application is in:

```text
Thesis Forecasting/
```

Important files and folders:

```text
Thesis Forecasting/streamlit_app.py                         Streamlit dashboard
Thesis Forecasting/RUN_DASHBOARD.bat                        Windows dashboard launcher
Thesis Forecasting/scripts/cell1_clean_data.py              Cleaning workflow
Thesis Forecasting/scripts/cell2_rbfnn.py                   RBFNN training and forecasting
Thesis Forecasting/scripts/cell3_benchmark.py               Benchmark model workflow
Thesis Forecasting/scripts/cell4_generate_thesis_and_metrics.py
Thesis Forecasting/PAPER CONTEXT/README.md                  Thesis workflow details
Thesis Forecasting/PROJECT_SETUP_AND_USER_GUIDE.txt         End-user setup guide
requirements.txt                                            Python dependencies
```

## Setup

Use 64-bit Python 3.11 on Windows.

```powershell
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

For non-programmer Windows users, follow:

```text
Thesis Forecasting/PROJECT_SETUP_AND_USER_GUIDE.txt
```

## Run The Dashboard

From Windows Explorer, open:

```text
Thesis Forecasting/RUN_DASHBOARD.bat
```

Or run from the repository root:

```powershell
streamlit run "Thesis Forecasting/streamlit_app.py"
```

## Run The Workflow Scripts

Run these commands from the repository root:

```powershell
python "Thesis Forecasting/scripts/cell1_clean_data.py"
python "Thesis Forecasting/scripts/cell2_rbfnn.py" --forecast-only
python "Thesis Forecasting/scripts/cell3_benchmark.py"
python "Thesis Forecasting/scripts/cell4_generate_thesis_and_metrics.py"
```

Retrain intentionally:

```powershell
python "Thesis Forecasting/scripts/cell2_rbfnn.py" --train
python "Thesis Forecasting/scripts/cell3_benchmark.py" --train
```

Daily dashboard forecasting should use the saved RBFNN artifacts and forecast-only workflow. Retraining, benchmark comparison, and thesis figure generation should be run only when updating thesis results or model artifacts.

## Documentation Map

- `Thesis Forecasting/PROJECT_SETUP_AND_USER_GUIDE.txt` - step-by-step Windows user guide.
- `Thesis Forecasting/PAPER CONTEXT/README.md` - thesis workflow commands, outputs, and conventions.
- `Thesis Forecasting/PAPER CONTEXT/THESIS_FORECASTING_DOCUMENTATION.md` - thesis-oriented technical documentation.
- `PAPERS/README.md` - paper and defense material index.
- `archive/PROJECT_DOCUMENTATION.md` - legacy/archive documentation.

## Output Areas

The active workflow writes cleaned data, outage inputs, metrics, model artifacts, dashboard outputs, and thesis figures under `Thesis Forecasting/`.

Typical output locations:

```text
Thesis Forecasting/outputs/
Thesis Forecasting/metadata/
Thesis Forecasting/models/
Thesis Forecasting/thesis_figures/
```

## Git Notes

Source code, documentation, requirements, and thesis support materials should be committed. Generated caches, temporary files, raw local data, and environment folders should stay out of version control unless they are intentionally part of the thesis deliverable.

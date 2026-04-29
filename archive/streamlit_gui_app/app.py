"""
Streamlit GUI for the Agus hydropower day-ahead forecasting workflow.

This dashboard is intentionally isolated from the existing pipeline scripts.
It reads existing outputs and calls the optimized scripts through subprocess.
"""

from __future__ import annotations

import io
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# ============================================================
# PATH CONFIGURATION
# ============================================================

APP_DIR = Path(__file__).resolve().parent
PROJECT_DIR = APP_DIR.parent

# Prefer the thesis Conda environment for subprocess pipeline runs when it is
# present, because TensorFlow/model dependencies are installed there.
CONDA_PYTHON = Path(r"C:\Users\Allen Mae\anaconda3\envs\ALLANTHESIS\python.exe")
PYTHON_EXE = str(CONDA_PYTHON) if CONDA_PYTHON.exists() else sys.executable

RAW_EXCEL_PATH = PROJECT_DIR / "data" / "raw" / "DATA(JAN2024-JUNE2025).xlsx"

ORIGINAL_CLEAN_SCRIPT = PROJECT_DIR / "scripts" / "cell1_clean_data.py"
OPTIMIZED_CLEAN_SCRIPT = PROJECT_DIR / "optimized_version" / "scripts" / "optimized_cell1_clean.py"
RBFNN_SCRIPT = PROJECT_DIR / "optimized_version" / "scripts" / "optimized_cell2_rbfnn.py"
BENCHMARK_SCRIPT = PROJECT_DIR / "optimized_version" / "scripts" / "optimized_cell3_benchmark.py"

OPT_DIR = PROJECT_DIR / "optimized_version"
OPT_OUTPUTS_DIR = OPT_DIR / "outputs"
CLEANED_DIR = OPT_OUTPUTS_DIR / "cleaned_data"
OUTAGE_DIR = OPT_OUTPUTS_DIR / "outages_planning"
RBFNN_FORECAST_DIR = OPT_OUTPUTS_DIR / "rbfnn_forecast"
METADATA_DIR = OPT_DIR / "metadata"
OVERALL_METRICS_DIR = METADATA_DIR / "overall_metrics"
VALIDATION_METRICS_DIR = METADATA_DIR / "validation_metrics"
TESTING_METRICS_DIR = METADATA_DIR / "testing_metrics"
BENCHMARK_DIR = OPT_DIR / "benchmark"

CLEANED_PARQUET = CLEANED_DIR / "cleaned_hourly_data.parquet"
CLEANED_XLSX = CLEANED_DIR / "cleaned_hourly_data.xlsx"
OUTAGE_XLSX = OUTAGE_DIR / "Planned_Outages_Input.xlsx"
RBFNN_FORECAST_CSV = RBFNN_FORECAST_DIR / "Day_Ahead_24H_Optimized_RBFNN_Forecast.csv"
RBFNN_FORECAST_XLSX = RBFNN_FORECAST_DIR / "Day_Ahead_24H_Optimized_RBFNN_Forecast.xlsx"
CELL1_METADATA_JSON = OVERALL_METRICS_DIR / "optimized_cell1_metadata.json"
RBFNN_METRICS_XLSX = OVERALL_METRICS_DIR / "optimized_rbfnn_validation_testing_metrics.xlsx"
BENCHMARK_METRICS_XLSX = OVERALL_METRICS_DIR / "optimized_benchmark_validation_testing_metrics.xlsx"
COMPARISON_METRICS_XLSX = OVERALL_METRICS_DIR / "rbfnn_vs_benchmark_mape_comparison.xlsx"
ACTUAL_FORECAST_XLSX = (
    OVERALL_METRICS_DIR
    / "actual_forecast_diagnostics"
    / "july_1_2025_actual_vs_optimized_rbfnn.xlsx"
)

PLANTS = ["agus1", "agus2", "agus4", "agus5", "agus6", "agus7"]
PLANT_LABELS = {plant: plant.upper().replace("AGUS", "AGUS ") for plant in PLANTS}
FORECAST_TOTAL_COLUMNS = ["AGUS1", "AGUS2", "AGUS4", "AGUS5", "AGUS6", "AGUS7"]
METRIC_COLUMNS = ["operational_mape", "mae", "rmse", "r2"]


# ============================================================
# PAGE SETUP AND STYLES
# ============================================================

st.set_page_config(
    page_title="Agus Hydropower Forecasting Dashboard",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main .block-container { padding-top: 1.3rem; padding-bottom: 2rem; }
    .app-title { font-size: 2.1rem; font-weight: 760; line-height: 1.18; margin-bottom: .3rem; }
    .app-subtitle { color: #4b5563; font-size: 1.02rem; margin-bottom: 1.1rem; }
    .status-card {
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 1rem 1.05rem;
        background: #ffffff;
        min-height: 112px;
        box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
    }
    .status-label { font-size: .82rem; color: #6b7280; margin-bottom: .4rem; }
    .status-value { font-size: 1.25rem; font-weight: 720; color: #111827; }
    .status-note { font-size: .78rem; color: #6b7280; margin-top: .25rem; overflow-wrap: anywhere; }
    .section-note {
        color: #4b5563;
        background: #f8fafc;
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: .85rem 1rem;
        margin-bottom: 1rem;
    }
    div[data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: .8rem .9rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HELPERS
# ============================================================


def file_state(path: Path) -> tuple[str, str]:
    if not path.exists():
        return "Missing", str(path.relative_to(PROJECT_DIR) if path.is_relative_to(PROJECT_DIR) else path)
    modified = datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
    return "Available", f"Updated {modified}"


def status_card(label: str, path: Path | None = None, value: str | None = None, note: str | None = None) -> None:
    if path is not None:
        value, note = file_state(path)
    st.markdown(
        f"""
        <div class="status-card">
            <div class="status-label">{label}</div>
            <div class="status-value">{value or "Unknown"}</div>
            <div class="status-note">{note or ""}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_note(text: str) -> None:
    st.markdown(f'<div class="section-note">{text}</div>', unsafe_allow_html=True)


def format_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_DIR))
    except ValueError:
        return str(path)


def script_exists(path: Path) -> bool:
    if path.exists():
        return True
    st.error(f"Required script is missing: {format_path(path)}")
    return False


def run_script(script_path: Path, label: str) -> subprocess.CompletedProcess[str] | None:
    if not script_exists(script_path):
        return None
    with st.spinner(f"Running {label}..."):
        result = subprocess.run(
            [PYTHON_EXE, str(script_path)],
            cwd=str(PROJECT_DIR),
            capture_output=True,
            text=True,
        )
    with st.expander(f"{label} terminal output", expanded=result.returncode != 0):
        if result.stdout:
            st.code(result.stdout, language="text")
        if result.stderr:
            st.code(result.stderr, language="text")
    if result.returncode == 0:
        st.success(f"{label} completed successfully.")
    else:
        st.error(f"{label} failed with exit code {result.returncode}.")
    clear_caches()
    return result


def clear_caches() -> None:
    load_cleaned_data.clear()
    load_excel_file.clear()
    load_forecast_data.clear()
    load_json.clear()
    load_metrics.clear()
    load_daily_metrics.clear()


@st.cache_data(show_spinner=False)
def load_json(path_text: str) -> dict:
    path = Path(path_text)
    if not path.exists():
        return {}
    return json.loads(path.read_text())


@st.cache_data(show_spinner=False)
def load_excel_file(path_text: str, sheet_name=0) -> pd.DataFrame:
    path = Path(path_text)
    if not path.exists():
        return pd.DataFrame()
    return pd.read_excel(path, sheet_name=sheet_name)


@st.cache_data(show_spinner=False)
def load_cleaned_data() -> pd.DataFrame:
    if CLEANED_PARQUET.exists():
        df = pd.read_parquet(CLEANED_PARQUET)
    elif CLEANED_XLSX.exists():
        df = pd.read_excel(CLEANED_XLSX)
    else:
        return pd.DataFrame()
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    return df


@st.cache_data(show_spinner=False)
def load_forecast_data() -> pd.DataFrame:
    if RBFNN_FORECAST_CSV.exists():
        df = pd.read_csv(RBFNN_FORECAST_CSV)
    elif RBFNN_FORECAST_XLSX.exists():
        df = pd.read_excel(RBFNN_FORECAST_XLSX)
    else:
        return pd.DataFrame()
    return normalize_forecast(df)


@st.cache_data(show_spinner=False)
def load_metrics(path_text: str) -> pd.DataFrame:
    path = Path(path_text)
    if not path.exists():
        return pd.DataFrame()
    return pd.read_excel(path)


@st.cache_data(show_spinner=False)
def load_daily_metrics(folder_text: str) -> pd.DataFrame:
    folder = Path(folder_text)
    if not folder.exists():
        return pd.DataFrame()
    frames: list[pd.DataFrame] = []
    for path in sorted(folder.glob("*.xlsx")):
        df = pd.read_excel(path)
        plant = path.stem.split("_")[0]
        df.insert(0, "plant", plant)
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def to_excel_bytes(df: pd.DataFrame, sheet_name: str = "data") -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name=sheet_name[:31], index=False)
    return output.getvalue()


def download_buttons(df: pd.DataFrame, base_name: str) -> None:
    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            "Download CSV",
            df.to_csv(index=False).encode("utf-8"),
            file_name=f"{base_name}.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with col2:
        st.download_button(
            "Download Excel",
            to_excel_bytes(df, base_name),
            file_name=f"{base_name}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )


def normalize_forecast(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "Date" in out.columns:
        out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
    if "Hour" in out.columns:
        out["Hour"] = pd.to_numeric(out["Hour"], errors="coerce").astype("Int64")
    if "Date" in out.columns and "Hour" in out.columns:
        out["DateHour"] = out["Date"] + pd.to_timedelta(out["Hour"].fillna(1).astype(int) - 1, unit="h")
    total_cols = [c for c in FORECAST_TOTAL_COLUMNS if c in out.columns]
    if total_cols:
        out["Total_Cascade_Generation"] = out[total_cols].sum(axis=1)
    return out


def forecast_total_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in FORECAST_TOTAL_COLUMNS if c in df.columns]


def forecast_unit_columns(df: pd.DataFrame) -> list[str]:
    totals = set(FORECAST_TOTAL_COLUMNS + ["Date", "Hour", "DateHour", "Total_Cascade_Generation"])
    unit_markers = ("unit", "UNIT", "Unit")
    return [c for c in df.columns if c not in totals and any(marker in c for marker in unit_markers)]


def metric_rename(df: pd.DataFrame) -> pd.DataFrame:
    rename = {
        "val_operational_mape": "Validation MAPE",
        "val_mae": "Validation MAE",
        "val_rmse": "Validation RMSE",
        "val_r2": "Validation R2",
        "test_operational_mape": "Testing MAPE",
        "test_mae": "Testing MAE",
        "test_rmse": "Testing RMSE",
        "test_r2": "Testing R2",
        "operational_mape": "MAPE",
        "mae": "MAE",
        "rmse": "RMSE",
        "r2": "R2",
    }
    return df.rename(columns=rename)


def display_dataframe(df: pd.DataFrame, height: int = 420) -> None:
    st.dataframe(df, use_container_width=True, height=height)


def available_generation_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("gen_agus") or c.startswith("total_gen_agus")]


def outage_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("out_agus") and "unit" in c]


def validate_outage_plan(df: pd.DataFrame) -> tuple[bool, list[str]]:
    errors: list[str] = []
    if "Date" not in df.columns:
        errors.append("Missing Date column.")
    if "Hour" not in df.columns:
        errors.append("Missing Hour column.")
    if "Hour" in df.columns:
        hours = pd.to_numeric(df["Hour"], errors="coerce")
        if hours.isna().any() or ~hours.between(1, 24).all():
            errors.append("Hour values must be numeric from 1 to 24.")
    for col in outage_columns(df):
        values = pd.to_numeric(df[col], errors="coerce")
        if values.isna().any() or ~values.isin([0, 1]).all():
            errors.append(f"{col} contains values other than 0 or 1.")
    return not errors, errors


def save_outage_plan(df: pd.DataFrame) -> None:
    OUTAGE_DIR.mkdir(parents=True, exist_ok=True)
    out = df.copy()
    if "Date" in out.columns:
        out["Date"] = pd.to_datetime(out["Date"], errors="coerce").dt.date
    for col in outage_columns(out):
        out[col] = pd.to_numeric(out[col], errors="coerce").astype(int)
    out.to_excel(OUTAGE_XLSX, index=False)
    clear_caches()


def latest_cleaned_timestamp() -> str:
    metadata = load_json(str(CELL1_METADATA_JSON))
    if metadata.get("latest_timestamp"):
        return str(metadata["latest_timestamp"])
    df = load_cleaned_data()
    if not df.empty and "datetime" in df.columns:
        return str(df["datetime"].max())
    return "Not available"


def missing_file_message(label: str, path: Path) -> None:
    st.warning(f"{label} is not available yet: `{format_path(path)}`")


# ============================================================
# PAGES
# ============================================================


def page_home() -> None:
    st.markdown(
        '<div class="app-title">AI-Based Short-Term Generation Forecasting of Cascaded Hydroelectric Power Plants</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="app-subtitle">A Streamlit dashboard for day-ahead Agus cascade generation forecasting, outage planning, forecast review, and thesis metrics.</div>',
        unsafe_allow_html=True,
    )

    st.subheader("System Overview")
    section_note(
        "This dashboard uses the optimized saved-model workflow. It refreshes cleaned data through the existing cleaning scripts, edits the planned outage input, and runs the optimized RBFNN forecast without retraining models."
    )

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        status_card("Cleaned Data", CLEANED_PARQUET)
    with col2:
        status_card("Planned Outage File", OUTAGE_XLSX)
    with col3:
        status_card("RBFNN Forecast", RBFNN_FORECAST_CSV)
    with col4:
        status_card("Validation / Testing Metrics", RBFNN_METRICS_XLSX)

    st.divider()
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.metric("Latest Cleaned Timestamp", latest_cleaned_timestamp())
    with col_b:
        cleaned = load_cleaned_data()
        st.metric("Cleaned Rows", f"{len(cleaned):,}" if not cleaned.empty else "0")
    with col_c:
        forecast = load_forecast_data()
        st.metric("Forecast Horizon", f"{len(forecast):,} hours" if not forecast.empty else "Not available")


def page_update_excel() -> None:
    st.header("Upload / Update Excel Data")
    section_note(
        f"Uploaded Excel files are saved to `{format_path(RAW_EXCEL_PATH)}`. Running the cleaning pipeline first regenerates baseline cleaned outputs, then refreshes the optimized input folder used by forecasting."
    )

    uploaded = st.file_uploader("Upload Excel input file", type=["xlsx", "xls"])
    if uploaded is not None:
        try:
            preview = pd.read_excel(uploaded, sheet_name=0, header=None, nrows=12)
            st.caption("Uploaded file preview")
            display_dataframe(preview, height=260)
            uploaded.seek(0)
        except Exception as exc:
            st.error(f"Could not preview uploaded Excel file: {exc}")

        if st.button("Upload / Update Excel Data", type="primary", use_container_width=True):
            RAW_EXCEL_PATH.parent.mkdir(parents=True, exist_ok=True)
            RAW_EXCEL_PATH.write_bytes(uploaded.getvalue())
            clear_caches()
            st.success(f"Saved uploaded file to `{format_path(RAW_EXCEL_PATH)}`.")

    st.subheader("Run Cleaning Pipeline")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Run Cleaning Pipeline", use_container_width=True):
            first = run_script(ORIGINAL_CLEAN_SCRIPT, "Original Cell 1 cleaning pipeline")
            if first and first.returncode == 0:
                run_script(OPTIMIZED_CLEAN_SCRIPT, "Optimized Cell 1 input refresh")
    with col2:
        if st.button("Run Optimized Input Refresh Only", use_container_width=True):
            run_script(OPTIMIZED_CLEAN_SCRIPT, "Optimized Cell 1 input refresh")

    cleaned = load_cleaned_data()
    if not cleaned.empty:
        st.subheader("Latest Cleaned Output Preview")
        display_dataframe(cleaned.head(50), height=360)
    else:
        missing_file_message("Cleaned output", CLEANED_PARQUET)


def page_cleaned_data() -> None:
    st.header("Cleaned Hourly Data")
    df = load_cleaned_data()
    if df.empty:
        missing_file_message("Cleaned data", CLEANED_PARQUET)
        return

    col1, col2, col3 = st.columns(3)
    with col1:
        if "datetime" in df.columns:
            st.metric("Date Range", f"{df['datetime'].min():%Y-%m-%d} to {df['datetime'].max():%Y-%m-%d}")
        else:
            st.metric("Date Range", "Unavailable")
    with col2:
        st.metric("Rows", f"{len(df):,}")
    with col3:
        st.metric("Columns", f"{len(df.columns):,}")

    gen_cols = available_generation_columns(df)
    st.subheader("Available Agus Generation Columns")
    st.write(", ".join(gen_cols) if gen_cols else "No Agus generation columns detected.")

    st.subheader("Table Preview")
    display_dataframe(df.head(500))
    download_buttons(df, "cleaned_hourly_data")


def page_outage_planning() -> None:
    st.header("Outage Planning")
    section_note(
        "Edit planned outage status for the next 24 hours. Use 1 for unit available/running and 0 for outage/unavailable. Saved edits are written to the optimized outage file used by the RBFNN forecasting pipeline."
    )

    if "outage_editor_df" not in st.session_state:
        st.session_state.outage_editor_df = pd.DataFrame()

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Load Current Outage Plan", type="primary", use_container_width=True):
            df = load_excel_file(str(OUTAGE_XLSX))
            if df.empty:
                missing_file_message("Outage plan", OUTAGE_XLSX)
            else:
                st.session_state.outage_editor_df = df
                st.success("Current outage plan loaded.")
    with col2:
        if st.button("Reload From File", use_container_width=True):
            load_excel_file.clear()
            df = load_excel_file(str(OUTAGE_XLSX))
            st.session_state.outage_editor_df = df

    if st.session_state.outage_editor_df.empty:
        df = load_excel_file(str(OUTAGE_XLSX))
        if not df.empty:
            st.session_state.outage_editor_df = df

    if st.session_state.outage_editor_df.empty:
        missing_file_message("Outage plan", OUTAGE_XLSX)
        return

    df = st.session_state.outage_editor_df.copy()
    out_cols = outage_columns(df)
    column_config = {
        "Date": st.column_config.DateColumn("Date"),
        "Hour": st.column_config.NumberColumn("Hour", min_value=1, max_value=24, step=1),
    }
    for col in out_cols:
        column_config[col] = st.column_config.NumberColumn(col, min_value=0, max_value=1, step=1)

    edited = st.data_editor(
        df,
        use_container_width=True,
        height=520,
        num_rows="fixed",
        column_config=column_config,
        key="outage_editor",
    )

    valid, errors = validate_outage_plan(edited)
    if not valid:
        for error in errors:
            st.error(error)

    if st.button("Save Edited Outage Plan", type="primary", disabled=not valid, use_container_width=True):
        save_outage_plan(edited)
        st.session_state.outage_editor_df = edited
        st.success(f"Saved edited outage plan to `{format_path(OUTAGE_XLSX)}`.")


def page_forecast() -> None:
    st.header("Day-Ahead Forecast")
    section_note(
        "Run the optimized RBFNN forecast using saved models, latest cleaned data, and the currently saved outage plan. This does not retrain or overwrite model files."
    )

    if st.button("Run Day-Ahead Forecast", type="primary", use_container_width=True):
        run_script(RBFNN_SCRIPT, "Optimized RBFNN day-ahead forecast")

    df = load_forecast_data()
    if df.empty:
        missing_file_message("RBFNN forecast", RBFNN_FORECAST_CSV)
        return

    total_cols = forecast_total_columns(df)
    unit_cols = forecast_unit_columns(df)
    if not unit_cols:
        st.info("Per-unit forecast columns are unavailable in the current model output. The dashboard will display them automatically if future forecast files include per-unit columns.")

    tabs = st.tabs(["24-Hour Forecast", "Per Agus Totals", "Cascade Total", "Plots", "Downloads"])

    with tabs[0]:
        display_dataframe(df, height=480)

    with tabs[1]:
        cols = [c for c in ["Date", "Hour", "DateHour"] if c in df.columns] + total_cols
        display_dataframe(df[cols], height=480)

    with tabs[2]:
        cols = [c for c in ["Date", "Hour", "DateHour", "Total_Cascade_Generation"] if c in df.columns]
        display_dataframe(df[cols], height=480)

    with tabs[3]:
        plot_forecast_charts(df, total_cols)

    with tabs[4]:
        download_buttons(df, "Day_Ahead_24H_Optimized_RBFNN_Forecast")


def plot_forecast_charts(df: pd.DataFrame, total_cols: list[str]) -> None:
    x_col = "DateHour" if "DateHour" in df.columns else "Hour"
    if "Total_Cascade_Generation" in df.columns:
        fig = px.line(
            df,
            x=x_col,
            y="Total_Cascade_Generation",
            markers=True,
            title="Day-Ahead Forecasted Generation of Agus Cascade",
            labels={x_col: "Date/Hour", "Total_Cascade_Generation": "Generation (MW)"},
        )
        st.plotly_chart(fig, use_container_width=True)

    if total_cols:
        long_df = df.melt(
            id_vars=[x_col],
            value_vars=total_cols,
            var_name="Agus Plant",
            value_name="Generation (MW)",
        )
        fig = px.line(
            long_df,
            x=x_col,
            y="Generation (MW)",
            color="Agus Plant",
            markers=True,
            title="Per-Agus Total Generation Forecast Comparison",
            labels={x_col: "Date/Hour"},
        )
        st.plotly_chart(fig, use_container_width=True)

        fig = px.area(
            long_df,
            x=x_col,
            y="Generation (MW)",
            color="Agus Plant",
            title="Contribution of Each Agus Plant to Cascade Output",
            labels={x_col: "Date/Hour"},
        )
        st.plotly_chart(fig, use_container_width=True)

    actual = load_actual_forecast_diagnostics()
    if not actual.empty:
        st.subheader("Actual vs Forecast")
        st.plotly_chart(actual_vs_forecast_figure(actual), use_container_width=True)


def load_actual_forecast_diagnostics() -> pd.DataFrame:
    if not ACTUAL_FORECAST_XLSX.exists():
        return pd.DataFrame()
    try:
        sheets = pd.read_excel(ACTUAL_FORECAST_XLSX, sheet_name=None)
    except Exception:
        return pd.DataFrame()
    frames = []
    for sheet_name, df in sheets.items():
        if sheet_name.lower() == "summary":
            continue
        if {"Hour", "actual_mw", "forecast_mw"}.issubset(df.columns):
            work = df[["Hour", "actual_mw", "forecast_mw"]].copy()
            work["plant"] = sheet_name.upper()
            frames.append(work)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def actual_vs_forecast_figure(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for plant in sorted(df["plant"].unique()):
        plant_df = df[df["plant"] == plant]
        fig.add_trace(go.Scatter(x=plant_df["Hour"], y=plant_df["actual_mw"], mode="lines+markers", name=f"{plant} Actual"))
        fig.add_trace(go.Scatter(x=plant_df["Hour"], y=plant_df["forecast_mw"], mode="lines+markers", name=f"{plant} Forecast"))
    fig.update_layout(
        title="Actual vs Forecast Generation",
        xaxis_title="Date/Hour",
        yaxis_title="Generation (MW)",
        legend_title="Series",
    )
    return fig


def page_metrics() -> None:
    st.header("Validation and Testing Metrics")

    rbfnn = load_metrics(str(RBFNN_METRICS_XLSX))
    benchmarks = load_metrics(str(BENCHMARK_METRICS_XLSX))
    comparison = load_metrics(str(COMPARISON_METRICS_XLSX))
    val_daily = load_daily_metrics(str(VALIDATION_METRICS_DIR))
    test_daily = load_daily_metrics(str(TESTING_METRICS_DIR))

    tabs = st.tabs(["RBFNN Metrics", "Benchmark Metrics", "Daily Metrics", "Charts"])

    with tabs[0]:
        if rbfnn.empty:
            missing_file_message("RBFNN metrics", RBFNN_METRICS_XLSX)
        else:
            display_dataframe(metric_rename(rbfnn), height=420)

    with tabs[1]:
        if benchmarks.empty:
            missing_file_message("Benchmark metrics", BENCHMARK_METRICS_XLSX)
        else:
            display_dataframe(metric_rename(benchmarks), height=420)
        if not comparison.empty:
            st.subheader("RBFNN vs Benchmark MAPE Comparison")
            display_dataframe(comparison, height=360)

    with tabs[2]:
        st.subheader("Daily Validation Metrics")
        if val_daily.empty:
            missing_file_message("Daily validation metrics", VALIDATION_METRICS_DIR)
        else:
            display_dataframe(metric_rename(val_daily), height=360)
        st.subheader("Daily Testing Metrics")
        if test_daily.empty:
            missing_file_message("Daily testing metrics", TESTING_METRICS_DIR)
        else:
            display_dataframe(metric_rename(test_daily), height=360)

    with tabs[3]:
        plot_metric_charts(rbfnn, benchmarks)


def plot_metric_charts(rbfnn: pd.DataFrame, benchmarks: pd.DataFrame) -> None:
    if not rbfnn.empty:
        metric_map = {
            "Validation MAPE": "val_operational_mape",
            "Validation MAE": "val_mae",
            "Validation RMSE": "val_rmse",
            "Validation R2": "val_r2",
            "Testing MAPE": "test_operational_mape",
            "Testing MAE": "test_mae",
            "Testing RMSE": "test_rmse",
            "Testing R2": "test_r2",
        }
        selected = st.selectbox("RBFNN metric to chart", list(metric_map.keys()))
        fig = px.bar(
            rbfnn,
            x="plant",
            y=metric_map[selected],
            title=f"RBFNN {selected} by Plant",
            labels={"plant": "Agus Plant", metric_map[selected]: selected},
            text_auto=".3f",
        )
        st.plotly_chart(fig, use_container_width=True)

    if not benchmarks.empty:
        options = {
            "Validation MAPE": "val_operational_mape",
            "Validation MAE": "val_mae",
            "Validation RMSE": "val_rmse",
            "Validation R2": "val_r2",
            "Testing MAPE": "test_operational_mape",
            "Testing MAE": "test_mae",
            "Testing RMSE": "test_rmse",
            "Testing R2": "test_r2",
        }
        selected = st.selectbox("Benchmark comparison metric", list(options.keys()))
        fig = px.bar(
            benchmarks,
            x="plant",
            y=options[selected],
            color="model",
            barmode="group",
            title=f"Random Forest and XGBoost {selected} by Plant",
            labels={"plant": "Agus Plant", options[selected]: selected, "model": "Model"},
            text_auto=".3f",
        )
        st.plotly_chart(fig, use_container_width=True)


def page_about() -> None:
    st.header("About System")
    st.write(
        "This dashboard supports a thesis workflow for one-day-ahead, 24-hour generation forecasting of the cascaded Agus hydropower plants: Agus 1, Agus 2, Agus 4, Agus 5, Agus 6, and Agus 7."
    )
    st.subheader("Pipeline Integration")
    st.table(
        pd.DataFrame(
            [
                ["Raw Excel input", format_path(RAW_EXCEL_PATH)],
                ["Original cleaning script", format_path(ORIGINAL_CLEAN_SCRIPT)],
                ["Optimized input refresh", format_path(OPTIMIZED_CLEAN_SCRIPT)],
                ["RBFNN forecast script", format_path(RBFNN_SCRIPT)],
                ["Benchmark script", format_path(BENCHMARK_SCRIPT)],
                ["Cleaned data output", format_path(CLEANED_PARQUET)],
                ["Outage plan", format_path(OUTAGE_XLSX)],
                ["RBFNN forecast output", format_path(RBFNN_FORECAST_CSV)],
            ],
            columns=["Item", "Path"],
        )
    )
    st.subheader("Forecast Notes")
    st.write(
        "The forecast button calls the optimized RBFNN script without `--train`, so saved model files are used. Existing models are not retrained or replaced by the dashboard."
    )
    st.write(
        "Per-unit forecasts will appear automatically if future model outputs include unit-level columns. The current optimized RBFNN output provides total generation per Agus plant."
    )


# ============================================================
# ROUTING
# ============================================================


def main() -> None:
    st.sidebar.title("Agus Forecasting")
    page = st.sidebar.radio(
        "Navigation",
        [
            "Home",
            "Update Excel Data",
            "Cleaned Data",
            "Outage Planning",
            "Forecast",
            "Metrics",
            "About System",
        ],
    )

    st.sidebar.divider()
    st.sidebar.caption("Optimized workflow")
    st.sidebar.write(f"Python: `{PYTHON_EXE}`")
    st.sidebar.write(f"Project: `{PROJECT_DIR.name}`")

    if page == "Home":
        page_home()
    elif page == "Update Excel Data":
        page_update_excel()
    elif page == "Cleaned Data":
        page_cleaned_data()
    elif page == "Outage Planning":
        page_outage_planning()
    elif page == "Forecast":
        page_forecast()
    elif page == "Metrics":
        page_metrics()
    elif page == "About System":
        page_about()


if __name__ == "__main__":
    main()

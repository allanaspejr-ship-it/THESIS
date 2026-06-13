# Requirements note: pip install streamlit pandas openpyxl plotly pyarrow

from __future__ import annotations

import platform
import subprocess
import sys
from datetime import datetime
from io import BytesIO
from pathlib import Path

import openpyxl  # noqa: F401 - required by pandas Excel read/write workflows.
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


APP_DIR = Path(__file__).resolve().parent

DATA_DIR = APP_DIR / "data"
OUTPUTS_DIR = APP_DIR / "outputs"
CLEANED_DIR = OUTPUTS_DIR / "cleaned_data"
OUTAGES_DIR = OUTPUTS_DIR / "outages_planning"
FORECAST_DIR = OUTPUTS_DIR / "rbfnn_forecast"
RANDOM_FOREST_FORECAST_DIR = OUTPUTS_DIR / "random_forest_forecast"
XGBOOST_FORECAST_DIR = OUTPUTS_DIR / "xgboost_forecast"
MODELS_ROOT = APP_DIR / "models"
MODELS_DIR = MODELS_ROOT / "rbfnn"
RANDOM_FOREST_MODELS_DIR = MODELS_ROOT / "random_forest"
XGBOOST_MODELS_DIR = MODELS_ROOT / "xgboost"
METADATA_DIR = APP_DIR / "metadata"
OVERALL_METRICS_DIR = METADATA_DIR / "overall_metrics"
DAY_AHEAD_BACKTEST_DIR = METADATA_DIR / "day_ahead_backtest"
BENCHMARK_DIR = APP_DIR / "benchmark"
THESIS_FIGURES_DIR = APP_DIR / "thesis_figures"
SCRIPTS_DIR = APP_DIR / "scripts"

RAW_EXCEL_PATH = DATA_DIR / "DATA(JAN2024-JUNE2025).xlsx"
CLEANED_EXCEL = CLEANED_DIR / "cleaned_hourly_data.xlsx"
CLEANED_PARQUET = CLEANED_DIR / "cleaned_hourly_data.parquet"
OUTAGE_PLAN_PATH = OUTAGES_DIR / "Planned_Outages_Input.xlsx"
FORECAST_EXCEL = FORECAST_DIR / "Day_Ahead_24H_RBFNN_Forecast.xlsx"
FORECAST_CSV = FORECAST_DIR / "Day_Ahead_24H_RBFNN_Forecast.csv"
RANDOM_FOREST_FORECAST_EXCEL = RANDOM_FOREST_FORECAST_DIR / "Day_Ahead_24H_RANDOM_FOREST.xlsx"
XGBOOST_FORECAST_EXCEL = XGBOOST_FORECAST_DIR / "Day_Ahead_24H_XGBOOST.xlsx"
RBFNN_TESTING_PREDICTIONS = OVERALL_METRICS_DIR / "rbfnn_testing_predictions.xlsx"
RANDOM_FOREST_TESTING_PREDICTIONS = OVERALL_METRICS_DIR / "random_forest_testing_predictions.xlsx"
XGBOOST_TESTING_PREDICTIONS = OVERALL_METRICS_DIR / "xgboost_testing_predictions.xlsx"
RBFNN_DAY_AHEAD_TESTING_PREDICTIONS = DAY_AHEAD_BACKTEST_DIR / "rbfnn_testing_day_ahead_predictions.xlsx"
RANDOM_FOREST_DAY_AHEAD_TESTING_PREDICTIONS = DAY_AHEAD_BACKTEST_DIR / "random_forest_testing_day_ahead_predictions.xlsx"
XGBOOST_DAY_AHEAD_TESTING_PREDICTIONS = DAY_AHEAD_BACKTEST_DIR / "xgboost_testing_day_ahead_predictions.xlsx"
CLEAN_SCRIPT = SCRIPTS_DIR / "cell1_clean_data.py"
RBFNN_SCRIPT = SCRIPTS_DIR / "cell2_rbfnn.py"
BENCHMARK_SCRIPT = SCRIPTS_DIR / "cell3_benchmark.py"
THESIS_METRICS_SCRIPT = SCRIPTS_DIR / "cell4_generate_thesis_and_metrics.py"

PLANTS = ["Agus 1", "Agus 2", "Agus 4", "Agus 5", "Agus 6", "Agus 7"]
PLANT_LABELS = {
    "agus1": "Agus 1",
    "agus2": "Agus 2",
    "agus4": "Agus 4",
    "agus5": "Agus 5",
    "agus6": "Agus 6",
    "agus7": "Agus 7",
}
PLANT_COLORS = {
    "Agus 1": "#2F6BFF",
    "Agus 2": "#19B7D3",
    "Agus 4": "#16B978",
    "Agus 5": "#8657E8",
    "Agus 6": "#FDBF27",
    "Agus 7": "#F04444",
}
MODEL_COLORS = {
    "RBFNN": "#6670FF",
    "Random Forest": "#FF5A3C",
    "XGBoost": "#16D6AA",
}
CHART_PANEL_COLOR = "#12304F"
CHART_BACKGROUND = CHART_PANEL_COLOR
CHART_PLOT_BACKGROUND = CHART_PANEL_COLOR
CHART_GRID_COLOR = "rgba(190, 215, 235, 0.14)"
AXIS_TEXT_COLOR = "#EAF2FA"
TICK_TEXT_COLOR = "#D8E6F3"
CHART_TEXT_COLOR = AXIS_TEXT_COLOR
CHART_MUTED_TEXT = TICK_TEXT_COLOR
FORECAST_TOTAL_COLUMNS = {
    "Agus 1": "total_gen_agus1",
    "Agus 2": "total_gen_agus2",
    "Agus 4": "total_gen_agus4",
    "Agus 5": "total_gen_agus5",
    "Agus 6": "total_gen_agus6",
    "Agus 7": "total_gen_agus7",
}
PLANT_KEYS_BY_LABEL = {label: key for key, label in PLANT_LABELS.items()}
CAPACITY_MW = {"agus1": 80.0, "agus2": 180.0, "agus4": 158.1, "agus5": 55.0, "agus6": 219.0, "agus7": 54.0}
CASCADE_COLUMN = "total_cascade_generation"
MODEL_CONFIG = {
    "RBFNN": {
        "forecast_path": FORECAST_EXCEL,
        "testing_path": RBFNN_DAY_AHEAD_TESTING_PREDICTIONS,
        "script": RBFNN_SCRIPT,
        "script_args": [],
        "models_dir": MODELS_DIR,
        "description": "Radial Basis Function Neural Network day-ahead generation forecast.",
    },
    "Random Forest": {
        "forecast_path": RANDOM_FOREST_FORECAST_EXCEL,
        "testing_path": RANDOM_FOREST_DAY_AHEAD_TESTING_PREDICTIONS,
        "script": BENCHMARK_SCRIPT,
        "script_args": ["--forecast-only", "--model", "random_forest"],
        "models_dir": RANDOM_FOREST_MODELS_DIR,
        "description": "Random Forest benchmark forecast generated from saved benchmark artifacts.",
    },
    "XGBoost": {
        "forecast_path": XGBOOST_FORECAST_EXCEL,
        "testing_path": XGBOOST_DAY_AHEAD_TESTING_PREDICTIONS,
        "script": BENCHMARK_SCRIPT,
        "script_args": ["--forecast-only", "--model", "xgboost"],
        "models_dir": XGBOOST_MODELS_DIR,
        "description": "XGBoost benchmark forecast generated from saved benchmark artifacts.",
    },
}
NAV_ITEMS = [
    "Dashboard Overview",
    "Data Management",
    "Planned Outage Planning",
    "Forecast",
    "System Information",
]


st.set_page_config(
    page_title="NPC Agus Cascade Forecasting",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
    <style>
    :root {
        --bg0: #0a1d31;
        --bg1: #0d2a45;
        --panel: #061b34;
        --panel2: #08213f;
        --panel3: #082f63;
        --line: rgba(59, 130, 246, .45);
        --line2: rgba(96, 165, 250, .26);
        --blue: #0b5ed7;
        --blue2: #1d8cf8;
        --text: #f8fafc;
        --muted: #cbd5e1;
        --soft: #94a3b8;
        --red: #ef4444;
        --green: #22c55e;
        --yellow: #facc15;
    }

    html, body, [class*="css"] {
        font-size: 16px !important;
    }

    html, body, .stApp {
        background: #0a1d31 !important;
        color: var(--text);
        font-family: "Inter", "Segoe UI", system-ui, sans-serif;
    }

    .block-container {
        max-width: none !important;
        width: 100% !important;
        margin-left: 0 !important;
        margin-right: 0 !important;
        padding-top: 4.2rem !important;
        padding-left: 1.5rem !important;
        padding-right: 1.5rem !important;
        padding-bottom: 3rem !important;
    }

    header[data-testid="stHeader"] {
        background: transparent !important;
        height: 2.5rem !important;
    }

    section[data-testid="stSidebar"] {
        background: #081a2c;
        border-right: 1px solid rgba(96, 165, 250, .28);
        min-width: 230px !important;
        width: 230px !important;
        box-shadow: 12px 0 28px rgba(0, 0, 0, .28);
    }

    section[data-testid="stSidebar"] * {
        color: var(--text) !important;
        font-size: 15px !important;
    }

    section[data-testid="stSidebar"] .stRadio label {
        width: 100%;
        min-height: 46px;
        padding: .66rem .75rem;
        margin: .24rem 0 .5rem 0;
        border-radius: 8px;
        border: 1px solid transparent;
        background: transparent;
        transition: all .14s ease;
        font-weight: 750;
        font-size: 15px !important;
    }

    section[data-testid="stSidebar"] .stRadio label:hover {
        background: rgba(15, 102, 216, .18);
        border-color: rgba(96, 165, 250, .30);
    }

    section[data-testid="stSidebar"] .stRadio label:has(input:checked) {
        background: linear-gradient(180deg, #0f66d8, #094fbe);
        border-color: rgba(96, 165, 250, .58);
        box-shadow: 0 10px 22px rgba(11, 94, 215, .28);
    }

    .sidebar-title {
        font-weight: 900;
        font-size: 22px !important;
        margin: 1.05rem 0 .22rem;
    }

    .sidebar-subtitle {
        color: var(--muted);
        font-size: 17px !important;
        font-weight: 700;
        margin-bottom: 1.7rem;
    }

    .sidebar-mini {
        color: var(--muted);
        font-size: 14px !important;
        line-height: 1.8;
        margin-top: 1.8rem;
        padding-top: 1.15rem;
        border-top: 1px solid rgba(96, 165, 250, .18);
    }

    div[data-testid="stHorizontalBlock"] {
        gap: 14px;
        margin-bottom: 14px;
    }

    .dashboard-header {
        width: 100%;
        background: #082f63;
        border: 1px solid #1d8cf8;
        border-radius: 10px;
        padding: 26px 30px;
        margin-bottom: 24px;
        box-sizing: border-box;
        box-shadow: 0 14px 28px rgba(0, 0, 0, .22);
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 20px;
    }

    .page-title {
        color: #ffffff;
        font-size: 30px;
        font-weight: 900;
        line-height: 1.25;
        margin: 0;
    }

    .page-subtitle {
        color: #dbeafe;
        font-size: 16px;
        font-weight: 500;
        margin-top: 10px;
    }

    .dashboard-header h1 {
        font-size: 30px;
        font-weight: 900;
        color: #ffffff;
        margin: 0;
        line-height: 1.25;
    }

    .dashboard-header p {
        font-size: 16px;
        color: #dbeafe;
        margin-top: 10px;
    }

    .header-update {
        min-width: 170px;
        text-align: left;
        color: #dbeafe;
        font-size: 14px;
        line-height: 1.7;
    }

    .header-update strong {
        display: block;
        color: #ffffff;
        font-size: .91rem;
        font-weight: 500;
    }

    .section-title-box {
        width: 100%;
        background: #063b73;
        border: 1px solid #1d8cf8;
        border-radius: 8px;
        padding: 12px 18px;
        margin-top: 22px;
        margin-bottom: 14px;
        box-sizing: border-box;
        min-height: 46px;
        color: #ffffff !important;
        font-size: 20px;
        font-weight: 900;
        line-height: 1.2;
        display: flex;
        align-items: center;
    }

    .section-title-box span {
        color: #ffffff !important;
        font-size: 20px;
        font-weight: 900;
        line-height: 1.2;
    }

    .info-card, .kpi-card, .status-card,
    .metric-card,
    .kpi-card-light, .info-card-light, .status-card-light {
        min-height: 120px;
        background: #08213f;
        border: 1px solid var(--line);
        border-radius: 10px;
        padding: 18px 20px;
        box-sizing: border-box;
        color: #ffffff !important;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        box-shadow: inset 0 1px 0 rgba(255, 255, 255, .04), 0 10px 22px rgba(0, 0, 0, .14);
        margin-bottom: 0;
    }

    .kpi-card.red {
        background: linear-gradient(180deg, #082b57 0%, #061b34 100%);
        border-color: rgba(239, 68, 68, .55);
    }

    .kpi-card.light {
        background: linear-gradient(180deg, #082b57 0%, #061b34 100%);
        border: 1px solid var(--line);
        color: #ffffff !important;
    }

    .kpi-label {
        color: var(--muted) !important;
        font-size: 14px;
        font-weight: 850;
        letter-spacing: .04em;
        text-transform: uppercase;
    }

    .kpi-value {
        color: #ffffff !important;
        font-size: 28px;
        font-weight: 900;
        line-height: 1.1;
        margin-top: .45rem;
    }

    .kpi-card.red .kpi-value,
    .value-red {
        color: var(--red) !important;
    }

    .kpi-card.green .kpi-value,
    .value-green {
        color: var(--green) !important;
    }

    .kpi-note {
        color: #dbeafe !important;
        font-size: 14px;
        line-height: 1.25;
    }

    .instruction, .info-strip, .info-box, .note-card {
        width: 100%;
        background: #08213f;
        border: 1px solid rgba(59, 130, 246, 0.65);
        border-radius: 8px;
        padding: 14px 18px;
        color: #f8fafc !important;
        font-size: 16px;
        font-weight: 600;
        margin-bottom: 18px;
        line-height: 1.45;
        box-sizing: border-box;
    }

    .status-label {
        color: var(--muted) !important;
        font-size: 14px;
        font-weight: 850;
        text-transform: uppercase;
        letter-spacing: .04em;
    }

    .status-value {
        color: var(--green) !important;
        font-size: 24px;
        font-weight: 900;
        margin-top: .35rem;
    }

    .status-note {
        color: var(--muted) !important;
        font-size: 14px;
        margin-top: .7rem;
        padding-top: .62rem;
        border-top: 1px solid rgba(148, 163, 184, .18);
        line-height: 1.45;
    }

    .action-card {
        background: #08213f;
        border: 1px solid rgba(59, 130, 246, 0.65);
        border-radius: 10px;
        padding: 14px 18px;
        min-height: 96px;
        box-sizing: border-box;
        margin-bottom: 10px;
    }

    .action-title {
        font-size: 18px;
        font-weight: 900;
        color: #ffffff;
        margin-bottom: 10px;
    }

    .action-desc {
        font-size: 15px;
        line-height: 1.45;
        color: #dbeafe;
    }

    .action-card-full {
        background: #08213f;
        border: 1px solid rgba(59, 130, 246, 0.65);
        border-radius: 10px;
        padding: 14px 18px;
        min-height: 96px;
        box-sizing: border-box;
        margin-bottom: 10px;
    }

    .chart-card {
        background: #0c2947;
        border: 1px solid rgba(125, 190, 255, 0.85);
        border-radius: 10px;
        padding: 12px;
        box-sizing: border-box;
        min-height: 360px;
    }

    .workflow-card {
        min-height: 150px;
        background: linear-gradient(180deg, #092c5d, #062044);
        border: 1px solid var(--line);
        border-radius: 8px;
        padding: .9rem .75rem;
        text-align: center;
    }

    .workflow-step {
        width: 42px;
        height: 42px;
        border: 1px solid rgba(135, 195, 255, .55);
        border-radius: 8px;
        margin: 0 auto .55rem;
        display: flex;
        align-items: center;
        justify-content: center;
        color: #ffffff;
        font-weight: 900;
        background: rgba(20, 118, 255, .18);
    }

    .workflow-title {
        color: #ffffff;
        font-weight: 800;
        font-size: 15px;
        margin-bottom: .25rem;
    }

    .workflow-note {
        color: var(--muted);
        font-size: 14px;
        line-height: 1.25;
    }

    .data-note {
        color: var(--muted);
        font-size: 14px;
        line-height: 1.45;
    }

    .note-card * {
        color: #f8fafc !important;
    }

    .stMarkdown, p, label, span, div {
        color: inherit;
    }

    h1, h2, h3 {
        color: #ffffff !important;
    }

    div.stButton > button,
    div[data-testid="stDownloadButton"] > button {
        width: 100% !important;
        height: 44px !important;
        font-size: 15px !important;
        font-weight: 800 !important;
        border-radius: 8px !important;
        background: #0b5ed7 !important;
        color: #ffffff !important;
        border: 1px solid #1d8cf8 !important;
        box-shadow: 0 9px 18px rgba(11, 94, 215, .24);
    }

    div.stButton > button:hover,
    div[data-testid="stDownloadButton"] > button:hover {
        background: #1d8cf8;
        border-color: rgba(225, 242, 255, .90);
        color: #ffffff;
    }

    div[data-testid="stDataFrame"], div[data-testid="stDataEditor"] {
        font-size: 14px !important;
        border-radius: 8px;
        overflow: hidden;
        border: 1px solid var(--line2);
        background: #061b34;
    }

    .stDataFrame, .stDataEditor {
        color: #f8fafc;
    }

    div[data-testid="stFileUploader"] {
        background: linear-gradient(180deg, #082b57, #061b34);
        border: 1px solid var(--line);
        border-radius: 8px;
        padding: .85rem;
        margin-bottom: 0;
    }

    div[data-testid="stFileUploader"] label,
    div[data-testid="stFileUploader"] small,
    div[data-testid="stFileUploader"] span,
    div[data-testid="stFileUploader"] p {
        color: #ffffff !important;
    }

    div[data-testid="stFileUploader"] button {
        background: linear-gradient(180deg, #0f66d8, #094fbe) !important;
        color: #ffffff !important;
        border: 1px solid rgba(96, 165, 250, .62) !important;
        border-radius: 8px !important;
        font-weight: 800 !important;
    }

    div[data-testid="stSelectbox"] label,
    div[data-testid="stRadio"] label,
    div[data-testid="stMultiSelect"] label,
    div[data-testid="stCheckbox"] label,
    div[data-testid="stTextInput"] label,
    div[data-testid="stNumberInput"] label,
    div[data-testid="stFileUploader"] label,
    div[data-testid="stSegmentedControl"] label,
    .stSelectbox label,
    .stRadio label,
    .stMultiSelect label,
    .stCheckbox label {
        color: #DDEBFA !important;
        font-weight: 700 !important;
        font-size: 15px !important;
    }

    div[data-testid="stRadio"] label p,
    div[data-testid="stRadio"] label span,
    div[data-testid="stRadio"] label div,
    .stRadio label p,
    .stRadio label span {
        color: #F8FAFC !important;
        font-weight: 750 !important;
        opacity: 1 !important;
    }

    div[data-testid="stRadio"] label:has(input:checked) {
        color: #ffffff !important;
        border-color: rgba(125, 190, 255, .78) !important;
    }

    div[data-testid="stSelectbox"] div,
    div[data-testid="stMultiSelect"] div {
        color: #0a1d31;
    }

    div[data-testid="stCheckbox"] {
        padding-top: 1.72rem;
    }

    div[data-testid="stCheckbox"] label,
    div[data-testid="stCheckbox"] label p,
    div[data-testid="stCheckbox"] span {
        color: #EAF2FA !important;
        font-weight: 700 !important;
        opacity: 1 !important;
    }

    div[data-testid="stCheckbox"] label[data-disabled="false"],
    div[data-testid="stCheckbox"] label:not([aria-disabled="true"]) {
        opacity: 1 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def modified_ns(path: Path) -> int:
    return path.stat().st_mtime_ns if path.exists() else 0


def file_status(path: Path) -> str:
    return "Available" if path.exists() else "Missing"


def file_timestamp(path: Path) -> str:
    if not path.exists():
        return "Not available"
    return datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M")


def cleaned_source_path() -> Path:
    return CLEANED_PARQUET if CLEANED_PARQUET.exists() else CLEANED_EXCEL


def read_file_bytes(path: Path) -> bytes | None:
    return path.read_bytes() if path.exists() else None


def normalize_export_column_name(column: object) -> str:
    return str(column).strip().lower().replace(" ", "_").replace("-", "_")


def prepare_forecast_export(df: pd.DataFrame) -> pd.DataFrame:
    export_df = df.copy()
    technical_columns = {
        col
        for col in export_df.columns
        if normalize_export_column_name(col) in {"forecast_hour", "datetime"}
    }
    return export_df.drop(columns=list(technical_columns), errors="ignore")


def forecast_excel_download_bytes(model_name: str, fallback_df: pd.DataFrame) -> bytes:
    source_path = MODEL_CONFIG[model_name]["forecast_path"]
    if source_path.exists():
        source_df = pd.read_excel(source_path)
        export_df = prepare_forecast_export(source_df)
    else:
        export_df = prepare_forecast_export(fallback_df)
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        export_df.to_excel(writer, index=False, sheet_name="Day-Ahead Forecast")
    buffer.seek(0)
    return buffer.getvalue()


def clean_display_value(value: object, fallback: str = "") -> str:
    if value is None:
        return fallback
    text = str(value).strip()
    blocked_values = {"".join(["un", "defined"]), "none", "nan", "nat"}
    if text.lower() in blocked_values:
        return fallback
    return text


def clear_cached_data(message: str | None = None) -> None:
    st.cache_data.clear()
    if message:
        st.success(message)


@st.cache_data(show_spinner=False)
def load_raw_excel(modified: int) -> pd.DataFrame:
    _ = modified
    if not RAW_EXCEL_PATH.exists():
        raise FileNotFoundError("Raw Excel file is missing.")
    return pd.read_excel(RAW_EXCEL_PATH)


@st.cache_data(show_spinner=False)
def load_cleaned_data(modified: int) -> pd.DataFrame:
    _ = modified
    if CLEANED_PARQUET.exists():
        return pd.read_parquet(CLEANED_PARQUET)
    if CLEANED_EXCEL.exists():
        return pd.read_excel(CLEANED_EXCEL)
    raise FileNotFoundError("Cleaned data file is missing.")


@st.cache_data(show_spinner=False)
def load_outage_plan(modified: int) -> pd.DataFrame:
    _ = modified
    if not OUTAGE_PLAN_PATH.exists():
        raise FileNotFoundError("Planned outage file is missing.")
    return pd.read_excel(OUTAGE_PLAN_PATH)


def simplified_column_name(column: object) -> str:
    return (
        str(column)
        .strip()
        .lower()
        .replace("(mw)", "")
        .replace("_mw", "")
        .replace(" mw", "")
        .replace(" ", "_")
        .replace("-", "_")
    )


def rebuild_datetime_field(df: pd.DataFrame) -> pd.Series:
    if "datetime" in df.columns:
        parsed = pd.to_datetime(df["datetime"], errors="coerce")
        if parsed.notna().any():
            return parsed
    date_col = next((col for col in ["date", "Date"] if col in df.columns), None)
    hour_col = next((col for col in ["hour", "Hour", "time", "TIME"] if col in df.columns), None)
    if date_col is None or hour_col is None:
        return pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns]")
    dates = pd.to_datetime(df[date_col], errors="coerce")
    hours_raw = df[hour_col]

    def parse_hour(value: object) -> float:
        text = str(value).strip()
        if ":" in text:
            parsed = pd.to_timedelta(text + ":00" if text.count(":") == 1 else text, errors="coerce")
            return parsed.total_seconds() / 3600 if pd.notna(parsed) else float("nan")
        numeric = pd.to_numeric(value, errors="coerce")
        if pd.isna(numeric):
            return float("nan")
        hour = int(numeric)
        if 1 <= hour <= 24:
            return float(hour - 1)
        return float(hour)

    offsets = hours_raw.map(parse_hour)
    return dates + pd.to_timedelta(offsets, unit="h")


def normalize_forecast_columns(raw_df: pd.DataFrame, model_name: str) -> pd.DataFrame:
    if raw_df.empty:
        raise ValueError(f"{model_name} forecast workbook is empty.")
    original = raw_df.copy()
    lookup = {simplified_column_name(col): col for col in original.columns}
    normalized = pd.DataFrame(index=original.index)

    date_source = lookup.get("date")
    hour_source = lookup.get("hour") or lookup.get("time")
    if date_source is not None:
        normalized["date"] = pd.to_datetime(original[date_source], errors="coerce").dt.date
    if hour_source is not None:
        normalized["hour"] = original[hour_source]
    normalized["datetime"] = rebuild_datetime_field(original.rename(columns={date_source or "": "Date", hour_source or "": "Hour"}))

    for plant_key in PLANT_LABELS:
        standard = f"total_gen_{plant_key}"
        source = lookup.get(standard)
        if source is None:
            source = lookup.get(f"total_generation_{plant_key}")
        if source is None:
            source = lookup.get(f"total_{plant_key}")
        if source is None:
            raise ValueError(f"{model_name} forecast is missing plant-total column for {PLANT_LABELS[plant_key]}.")
        normalized[standard] = pd.to_numeric(original[source], errors="coerce")

    cascade_source = lookup.get("total_cascade_generation") or lookup.get("cascade_generation") or lookup.get("total_cascade")
    if cascade_source is not None:
        normalized[CASCADE_COLUMN] = pd.to_numeric(original[cascade_source], errors="coerce")
    else:
        normalized[CASCADE_COLUMN] = normalized[list(FORECAST_TOTAL_COLUMNS.values())].sum(axis=1)

    normalized = normalized.sort_values("datetime", na_position="last").reset_index(drop=True)
    normalized["forecast_hour"] = range(1, len(normalized) + 1)
    if len(normalized) != 24:
        raise ValueError(f"{model_name} day-ahead forecast must contain exactly 24 rows; found {len(normalized)}.")
    return normalized


@st.cache_data(show_spinner=False)
def load_model_forecast(model_name: str, modified: int) -> pd.DataFrame:
    _ = modified
    config = MODEL_CONFIG[model_name]
    path = config["forecast_path"]
    if not path.exists():
        raise FileNotFoundError(f"{model_name} forecast file is missing: {path.relative_to(APP_DIR)}")
    try:
        raw_df = pd.read_excel(path)
    except Exception as exc:
        raise ValueError(f"Could not read {model_name} forecast workbook: {exc}") from exc
    return normalize_forecast_columns(raw_df, model_name)


def load_forecast(modified: int) -> pd.DataFrame:
    return load_model_forecast("RBFNN", modified)


def normalize_model_name(value: object, fallback: str) -> str:
    text = str(value if pd.notna(value) else fallback).strip().lower().replace("-", "_").replace(" ", "_")
    if text in {"rbfnn", "rbf"}:
        return "RBFNN"
    if text in {"random_forest", "randomforest", "rf"}:
        return "Random Forest"
    if text in {"xgboost", "xgb"}:
        return "XGBoost"
    return fallback


def normalize_plant_name(value: object) -> str | None:
    text = str(value).strip().lower().replace(" ", "").replace("_", "")
    plant_map = {
        "agus1": "Agus 1",
        "agus2": "Agus 2",
        "agus4": "Agus 4",
        "agus5": "Agus 5",
        "agus6": "Agus 6",
        "agus7": "Agus 7",
    }
    return plant_map.get(text)


def normalize_testing_predictions(raw_df: pd.DataFrame, model_name: str) -> pd.DataFrame:
    if raw_df.empty:
        raise ValueError(f"{model_name} testing prediction workbook is empty.")
    lookup = {simplified_column_name(col): col for col in raw_df.columns}
    required = {
        "plant": lookup.get("plant"),
        "actual_generation": lookup.get("actual_generation") or lookup.get("actual"),
        "predicted_generation": lookup.get("predicted_generation") or lookup.get("prediction") or lookup.get("forecast"),
    }
    missing = [name for name, source in required.items() if source is None]
    if missing:
        raise ValueError(f"{model_name} testing predictions missing required columns: {', '.join(missing)}")

    date_source = lookup.get("date")
    hour_source = lookup.get("hour") or lookup.get("time")
    renamed = raw_df.copy()
    if date_source:
        renamed = renamed.rename(columns={date_source: "Date"})
    if hour_source:
        renamed = renamed.rename(columns={hour_source: "Hour"})
    if lookup.get("datetime"):
        renamed = renamed.rename(columns={lookup["datetime"]: "datetime"})

    out = pd.DataFrame(
        {
            "datetime": rebuild_datetime_field(renamed),
            "plant": raw_df[required["plant"]].map(normalize_plant_name),
            "actual_generation": pd.to_numeric(raw_df[required["actual_generation"]], errors="coerce"),
            "predicted_generation": pd.to_numeric(raw_df[required["predicted_generation"]], errors="coerce"),
        }
    )
    model_source = lookup.get("model")
    out["model"] = raw_df[model_source].map(lambda value: normalize_model_name(value, model_name)) if model_source else model_name
    out = out.dropna(subset=["datetime", "plant", "actual_generation", "predicted_generation"])
    out = out.drop_duplicates(subset=["model", "plant", "datetime"]).sort_values(["model", "plant", "datetime"]).reset_index(drop=True)
    return out


@st.cache_data(show_spinner=False)
def load_testing_predictions(model_name: str, modified: int) -> pd.DataFrame:
    _ = modified
    path = MODEL_CONFIG[model_name]["testing_path"]
    if not path.exists():
        raise FileNotFoundError(f"{model_name} testing predictions are missing: {path.relative_to(APP_DIR)}")
    try:
        raw_df = pd.read_excel(path)
    except Exception as exc:
        raise ValueError(f"Could not read {model_name} testing predictions: {exc}") from exc
    return normalize_testing_predictions(raw_df, model_name)


@st.cache_data(show_spinner=False)
def combine_testing_predictions(modified_tokens: tuple[int, int, int]) -> tuple[pd.DataFrame, list[str]]:
    _ = modified_tokens
    frames: list[pd.DataFrame] = []
    warnings_list: list[str] = []
    for model_name in MODEL_CONFIG:
        try:
            frames.append(load_testing_predictions(model_name, modified_ns(MODEL_CONFIG[model_name]["testing_path"])))
        except Exception as exc:
            warnings_list.append(str(exc))
    if not frames:
        return pd.DataFrame(), warnings_list
    combined = pd.concat(frames, ignore_index=True)
    ranges = combined.groupby("model")["datetime"].agg(["min", "max"])
    if len(ranges.drop_duplicates()) > 1:
        warnings_list.append("Testing date ranges are not identical across all loaded models.")
    return combined, warnings_list


def save_outage_plan(df: pd.DataFrame) -> None:
    OUTAGE_PLAN_PATH.parent.mkdir(parents=True, exist_ok=True)
    saved = df.copy()
    for col in outage_columns(saved):
        saved[col] = pd.to_numeric(saved[col], errors="coerce").fillna(1).clip(0, 1).round().astype(int)
    saved.to_excel(OUTAGE_PLAN_PATH, index=False)
    st.cache_data.clear()


def validate_fast_forecast_ready() -> None:
    if not (CLEANED_PARQUET.exists() or CLEANED_EXCEL.exists()):
        raise FileNotFoundError("Run Data Cleaning first.")
    if not OUTAGE_PLAN_PATH.exists():
        raise FileNotFoundError("Create or save Planned Outage Plan first.")
    for plant in PLANT_LABELS:
        model_exists = (MODELS_DIR / f"rbfnn_{plant}.keras").exists() or (MODELS_DIR / f"rbfnn_{plant}.h5").exists()
        required = [
            MODELS_DIR / f"meta_{plant}.json",
            MODELS_DIR / f"x_scaler_{plant}.pkl",
            MODELS_DIR / f"y_scaler_{plant}.pkl",
        ]
        if not model_exists or not all(path.exists() for path in required):
            raise FileNotFoundError("Retrain RBFNN Model first.")


def validate_model_forecast_ready(model_name: str) -> None:
    if not (CLEANED_PARQUET.exists() or CLEANED_EXCEL.exists()):
        raise FileNotFoundError("Run Data Cleaning first.")
    if not OUTAGE_PLAN_PATH.exists():
        raise FileNotFoundError("Create or save Planned Outage Plan first.")
    if model_name == "RBFNN":
        validate_fast_forecast_ready()
        return
    model_dir = MODEL_CONFIG[model_name]["models_dir"]
    prefix = "random_forest" if model_name == "Random Forest" else "xgboost"
    missing = [f"{prefix}_{plant}.pkl" for plant in PLANT_LABELS if not (model_dir / f"{prefix}_{plant}.pkl").exists()]
    if missing:
        raise FileNotFoundError(f"Missing saved {model_name} model artifacts: {', '.join(missing)}. Run Cell 3 with --train first.")


def run_selected_model_forecast(model_name: str) -> subprocess.CompletedProcess[str]:
    validate_model_forecast_ready(model_name)
    config = MODEL_CONFIG[model_name]
    return run_script(config["script"], config["script_args"])


def run_script(script_path: Path, args: list[str] | None = None) -> subprocess.CompletedProcess[str]:
    if not script_path.exists():
        raise FileNotFoundError(f"Missing script: {script_path.name}")
    command = [sys.executable, str(script_path.relative_to(APP_DIR))]
    if args:
        command.extend(args)
    result = subprocess.run(command, cwd=APP_DIR, capture_output=True, text=True, check=False)
    st.cache_data.clear()
    return result


def show_script_result(result: subprocess.CompletedProcess[str], label: str) -> None:
    if result.returncode == 0:
        st.success(f"{label} completed successfully.")
    else:
        st.error(f"{label} failed with exit code {result.returncode}.")
    with st.expander(f"{label} terminal output", expanded=result.returncode != 0):
        if result.stdout:
            st.code(result.stdout, language="text")
        if result.stderr:
            st.code(result.stderr, language="text")
        if not result.stdout and not result.stderr:
            st.write("No terminal output captured.")


def testing_modified_tokens() -> tuple[int, int, int]:
    return tuple(modified_ns(MODEL_CONFIG[model]["testing_path"]) for model in MODEL_CONFIG)


def filter_testing_data(df: pd.DataFrame, plant_filter: str, model_filter: str) -> pd.DataFrame:
    filtered = df.copy()
    if plant_filter != "All Agus Plants":
        filtered = filtered[filtered["plant"] == plant_filter]
    if model_filter != "All Models":
        filtered = filtered[filtered["model"] == model_filter]
    return filtered


@st.cache_data(show_spinner=False)
def prepare_cdf_data(
    testing_df: pd.DataFrame,
    plant_filter: str,
    model_filter: str,
    error_measure: str,
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    filtered = filter_testing_data(testing_df, plant_filter, model_filter).copy()
    filtered = filtered.dropna(subset=["actual_generation", "predicted_generation"])
    filtered["absolute_error_mw"] = (filtered["actual_generation"] - filtered["predicted_generation"]).abs()

    if error_measure == "Absolute Percentage Error (%)":
        filtered["plant_key"] = filtered["plant"].map(PLANT_KEYS_BY_LABEL)
        filtered["threshold"] = filtered["plant_key"].map(lambda key: max(1.0, 0.01 * CAPACITY_MW[key]) if key in CAPACITY_MW else float("nan"))
        filtered = filtered[(filtered["actual_generation"].abs() >= filtered["threshold"]) & (filtered["actual_generation"].abs() > 0)]
        filtered["error_value"] = filtered["absolute_error_mw"] / filtered["actual_generation"].abs() * 100.0
        value_label = "Absolute Percentage Error (%)"
        within_5_label = "Within 5%"
        within_10_label = "Within 10%"
    else:
        filtered["error_value"] = filtered["absolute_error_mw"]
        value_label = "Absolute Forecast Error (MW)"
        within_5_label = "Within +/-5 MW"
        within_10_label = "Within +/-10 MW"

    cdf_rows: list[pd.DataFrame] = []
    summary_rows: list[dict[str, object]] = []
    for model_name, group in filtered.groupby("model", sort=False):
        values = group["error_value"].dropna().sort_values().reset_index(drop=True)
        n = len(values)
        if n == 0:
            continue
        cdf_rows.append(
            pd.DataFrame(
                {
                    "model": model_name,
                    "plant_filter": plant_filter,
                    "error_value": values,
                    "cdf_probability": [(idx + 1) / n for idx in range(n)],
                    "observations": n,
                }
            )
        )
        summary_rows.append(
            {
                "Model": model_name,
                "Valid Testing Observations": n,
                f"Median {value_label}": values.median(),
                "80th Percentile": values.quantile(0.80),
                "90th Percentile": values.quantile(0.90),
                "95th Percentile": values.quantile(0.95),
                within_5_label: (values <= 5).mean() * 100.0,
                within_10_label: (values <= 10).mean() * 100.0,
            }
        )
    cdf_df = pd.concat(cdf_rows, ignore_index=True) if cdf_rows else pd.DataFrame()
    summary_df = pd.DataFrame(summary_rows)
    return cdf_df, summary_df, value_label


def calculate_error_summary(summary_df: pd.DataFrame) -> pd.DataFrame:
    if summary_df.empty:
        return summary_df
    formatted = summary_df.copy()
    for col in formatted.columns:
        if col == "Valid Testing Observations":
            formatted[col] = formatted[col].map(lambda value: f"{int(value):,}")
        elif col != "Model":
            formatted[col] = pd.to_numeric(formatted[col], errors="coerce").map(lambda value: f"{value:,.2f}" if pd.notna(value) else "N/A")
    return formatted


def calculate_empirical_cdf(values: pd.Series) -> pd.DataFrame:
    sorted_values = pd.to_numeric(values, errors="coerce").dropna().sort_values().reset_index(drop=True)
    n = len(sorted_values)
    return pd.DataFrame({"error_value": sorted_values, "cdf_probability": [(idx + 1) / n for idx in range(n)]}) if n else pd.DataFrame()


def render_header(last_updated: str | None = None) -> None:
    timestamp = last_updated or file_timestamp(FORECAST_EXCEL)
    if timestamp == "Missing":
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    st.markdown(
        f"""
        <div class="dashboard-header">
            <div>
                <h1 class="page-title">NPC Agus Cascade Hydropower Decision-Support Dashboard</h1>
                <p class="page-subtitle">Outage-Aware Forecasting Prototype for Historical Testing-Based Day-Ahead Forecasts</p>
            </div>
            <div class="header-update">
                Last updated
                <strong>{timestamp}</strong>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def metric_card(label: str, value: str, note: str = "", variant: str = "") -> None:
    st.markdown(
        f"""
        <div class="metric-card kpi-card {variant}">
            <div>
                <div class="kpi-label">{label}</div>
                <div class="kpi-value">{value}</div>
            </div>
            <div class="kpi-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def kpi_card(label: str, value: str, note: str, variant: str = "") -> None:
    metric_card(label, value, note, variant)


def status_card(label: str, value_or_path: Path | str, note: str | None = None) -> None:
    if isinstance(value_or_path, Path):
        value = file_status(value_or_path)
        status_note = f"Last modified<br>{file_timestamp(value_or_path)}"
    else:
        value = value_or_path
        status_note = note or ""
    value_class = "status-value" if value == "Available" else "status-value value-red"
    st.markdown(
        f"""
        <div class="status-card">
            <div class="status-label">{label}</div>
            <div class="{value_class}">{value}</div>
            <div class="status-note">{status_note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def instruction_card(text: str) -> None:
    info_box(text)


def section_title(title: str) -> None:
    st.markdown(
        f'''
        <div class="section-title-box">
            <span>{title}</span>
        </div>
        ''',
        unsafe_allow_html=True,
    )


def section_header(title: str) -> None:
    section_title(title)


def page_title_block() -> None:
    render_header()


def note_card(text: str) -> None:
    st.markdown(f'<div class="note-card">{text}</div>', unsafe_allow_html=True)


def info_box(text: str) -> None:
    st.markdown(f'<div class="info-box">{text}</div>', unsafe_allow_html=True)


def info_strip(text: str) -> None:
    info_box(text)


def action_card(title: str, description: str, button_label: str, key: str | None = None) -> bool:
    st.markdown(
        f'''
        <div class="action-card-full">
            <div class="action-title">{title}</div>
            <div class="action-desc">{description}</div>
        </div>
        ''',
        unsafe_allow_html=True,
    )
    return st.button(button_label, use_container_width=True, key=key)


def latest_date_from_df(df: pd.DataFrame | None) -> str:
    if df is None or df.empty:
        return "Not detected"
    candidates: list[pd.Series] = []
    for col in df.columns:
        name = str(col).lower()
        if "date" in name or "time" in name:
            parsed = pd.to_datetime(df[col], errors="coerce")
            if parsed.notna().sum() > 0:
                candidates.append(parsed)
    if not candidates:
        for col in df.columns[:8]:
            parsed = pd.to_datetime(df[col], errors="coerce")
            if parsed.notna().sum() >= max(3, int(len(df) * 0.2)):
                candidates.append(parsed)
    if not candidates:
        return "Not detected"
    latest = pd.concat(candidates, ignore_index=True).dropna().max()
    return latest.strftime("%Y-%m-%d %H:%M") if pd.notna(latest) else "Not detected"


def cleaned_datetime(df: pd.DataFrame | None) -> pd.Series:
    if df is None or "date" not in df.columns:
        return pd.Series(dtype="datetime64[ns]")
    dates = pd.to_datetime(df["date"], errors="coerce")
    if "time" not in df.columns:
        return dates
    hours = pd.to_numeric(df["time"], errors="coerce").fillna(0).astype(int)
    adjusted_hours = hours.where(hours < 24, 0)
    adjusted_dates = dates + pd.to_timedelta((hours == 24).astype(int), unit="D")
    return adjusted_dates + pd.to_timedelta(adjusted_hours, unit="h")


def forecast_datetime(df: pd.DataFrame) -> pd.Series:
    if "datetime" in df.columns:
        parsed = pd.to_datetime(df["datetime"], errors="coerce")
        if parsed.notna().any():
            return parsed
    dates = pd.to_datetime(df["Date"], errors="coerce")
    hours = df["Hour"]

    def parse_one(date_value: pd.Timestamp, hour_value: object) -> pd.Timestamp:
        if pd.isna(date_value):
            return pd.NaT
        hour_text = str(hour_value).strip()
        if ":" in hour_text:
            parsed = pd.to_timedelta(hour_text + ":00" if hour_text.count(":") == 1 else hour_text, errors="coerce")
            return date_value + parsed if pd.notna(parsed) else pd.NaT
        numeric_hour = pd.to_numeric(hour_value, errors="coerce")
        if pd.isna(numeric_hour):
            return pd.NaT
        hour_int = int(numeric_hour)
        hour_int = 0 if hour_int == 24 else hour_int
        return date_value + pd.Timedelta(hours=hour_int)

    return pd.Series([parse_one(date_value, hour_value) for date_value, hour_value in zip(dates, hours)], index=df.index)


def outage_columns(df: pd.DataFrame) -> list[str]:
    return [col for col in df.columns if str(col).startswith("out_agus")]


def outage_summary(df: pd.DataFrame | None) -> dict[str, object]:
    if df is None:
        return {"unavailable": 0, "available": 0, "affected_plants": [], "affected_units": []}
    cols = outage_columns(df)
    if not cols:
        return {"unavailable": 0, "available": 0, "affected_plants": [], "affected_units": []}
    numeric = df[cols].apply(pd.to_numeric, errors="coerce").fillna(1).clip(0, 1).round().astype(int)
    affected_units = [col for col in cols if (numeric[col] == 0).any()]
    affected_plants = [
        label
        for key, label in PLANT_LABELS.items()
        if any(col.startswith(f"out_{key}_") for col in affected_units)
    ]
    return {
        "unavailable": int((numeric == 0).sum().sum()),
        "available": int((numeric == 1).sum().sum()),
        "affected_plants": affected_plants,
        "affected_units": affected_units,
    }


def outage_status_frame(df: pd.DataFrame) -> pd.DataFrame:
    cols = outage_columns(df)
    if not cols:
        return pd.DataFrame()
    status = df[["Date", "Hour", *cols]].copy()
    for col in cols:
        status[col] = pd.to_numeric(status[col], errors="coerce").fillna(1).round().astype(int).map({1: "ON", 0: "OFF"})
    return status


def style_outage_status(df: pd.DataFrame):
    def style_cell(value: object) -> str:
        if value == "ON":
            return "color: #22c55e; font-weight: 800;"
        if value == "OFF":
            return "color: #ef4444; font-weight: 900;"
        return ""

    return df.style.map(style_cell)


def display_forecast_table(forecast: pd.DataFrame) -> pd.DataFrame:
    display = prepare_forecast_export(forecast)
    display.columns = [
        str(col)
        .replace("_", " ")
        .replace("MW", "(MW)")
        .replace("Gen", "Generation")
        .replace("total generation agus", "Total Generation Agus ")
        .replace("total cascade generation", "Total Cascade Generation")
        .replace("forecast hour", "Forecast Hour")
        .title()
        for col in display.columns
    ]
    return display


def forecast_plot_frame(forecast: pd.DataFrame) -> tuple[pd.DataFrame, str, pd.DataFrame]:
    plot_df = forecast.copy()
    plot_df["Forecast Datetime"] = forecast_datetime(plot_df)
    if "forecast_hour" not in plot_df.columns:
        plot_df["forecast_hour"] = range(1, len(plot_df) + 1)
    x_col = "forecast_hour"
    plant_totals = plot_df[["Forecast Datetime", "forecast_hour", *FORECAST_TOTAL_COLUMNS.values()]].rename(
        columns={v: k for k, v in FORECAST_TOTAL_COLUMNS.items()}
    )
    return plot_df, x_col, plant_totals


def forecast_kpis(forecast: pd.DataFrame | None, outage_df: pd.DataFrame | None = None) -> dict[str, str]:
    if forecast is None or forecast.empty or CASCADE_COLUMN not in forecast.columns:
        return {
            "cascade_total": "N/A",
            "peak": "N/A",
            "forecast_generated": file_timestamp(FORECAST_EXCEL),
            "affected": "None",
        }
    cascade = pd.to_numeric(forecast[CASCADE_COLUMN], errors="coerce")
    affected = outage_summary(outage_df)["affected_plants"]
    return {
        "cascade_total": f"{cascade.sum():,.2f} MWh" if cascade.notna().any() else "N/A",
        "peak": f"{cascade.max():,.2f} MW" if cascade.notna().any() else "N/A",
        "forecast_generated": file_timestamp(FORECAST_EXCEL),
        "affected": ", ".join(affected) if affected else "None",
    }


def apply_dashboard_chart_layout(
    fig: go.Figure,
    x_title: str | None = None,
    y_title: str | None = None,
    height: int = 500,
    show_legend: bool = True,
    hovermode: str = "x unified",
    unified_panel: bool = True,
) -> go.Figure:
    panel_color = CHART_PANEL_COLOR if unified_panel else "rgba(0,0,0,0)"
    fig.update_layout(
        title=dict(text=""),
        title_text="",
        legend_title_text="",
        template="plotly_dark",
        paper_bgcolor=panel_color,
        plot_bgcolor=CHART_PLOT_BACKGROUND,
        font=dict(color=CHART_TEXT_COLOR, size=12),
        legend=dict(
            title=dict(text=""),
            font=dict(color=CHART_TEXT_COLOR, size=12),
            bgcolor="rgba(0,0,0,0)",
            orientation="h",
            yanchor="bottom",
            y=1.01,
            xanchor="left",
            x=0.0,
        ),
        showlegend=show_legend,
        hoverlabel=dict(bgcolor="#08213f", font=dict(color=CHART_TEXT_COLOR, size=12)),
        hovermode=hovermode,
        height=height,
        xaxis=dict(
            title=dict(text=x_title or "", font=dict(color=AXIS_TEXT_COLOR, size=13)),
            tickfont=dict(color=TICK_TEXT_COLOR, size=11),
            gridcolor=CHART_GRID_COLOR,
            linecolor="rgba(190, 215, 235, 0.28)",
            zeroline=False,
            automargin=True,
        ),
        yaxis=dict(
            title=dict(text=y_title or "", font=dict(color=AXIS_TEXT_COLOR, size=13)),
            tickfont=dict(color=TICK_TEXT_COLOR, size=11),
            gridcolor=CHART_GRID_COLOR,
            linecolor="rgba(190, 215, 235, 0.28)",
            zeroline=False,
            automargin=True,
        ),
        margin=dict(l=65, r=24, t=45, b=62),
    )
    fig.update_layout(title_text="", legend_title_text="")
    return fig


def style_plotly_chart(fig: go.Figure) -> go.Figure:
    return apply_dashboard_chart_layout(fig)


def render_plotly_line_chart(forecast: pd.DataFrame, height: int = 460, model_name: str = "RBFNN") -> None:
    missing = [col for col in FORECAST_TOTAL_COLUMNS.values() if col not in forecast.columns]
    if missing:
        st.warning("Forecast total generation columns are incomplete.")
        return
    _, x_col, plant_totals = forecast_plot_frame(forecast)
    long_df = plant_totals.melt(
        id_vars=["Forecast Datetime", "forecast_hour"],
        value_vars=PLANTS,
        var_name="Plant",
        value_name="Generation (MW)",
    )
    fig = px.line(
        long_df,
        x=x_col,
        y="Generation (MW)",
        color="Plant",
        markers=True,
        color_discrete_map=PLANT_COLORS,
        title=None,
    )
    fig.update_traces(line=dict(width=2.7), marker=dict(size=6))
    fig = apply_dashboard_chart_layout(
        fig,
        x_title="Forecast Hour",
        y_title="Generation (MW)",
        height=height,
        show_legend=True,
        hovermode="x unified",
    )
    fig.update_xaxes(tickmode="linear", tick0=1, dtick=1)
    fig.update_layout(title_text="", legend_title_text="")
    st.plotly_chart(fig, use_container_width=True)


def render_plotly_bar_chart(forecast: pd.DataFrame, height: int = 460, model_name: str = "RBFNN") -> None:
    missing = [col for col in FORECAST_TOTAL_COLUMNS.values() if col not in forecast.columns]
    if missing:
        st.warning("Forecast total generation columns are incomplete.")
        return
    totals = []
    for plant, col in FORECAST_TOTAL_COLUMNS.items():
        totals.append({"Plant": plant, "Daily Total (MWh)": pd.to_numeric(forecast[col], errors="coerce").sum()})
    daily = pd.DataFrame(totals)
    fig = px.bar(
        daily,
        x="Plant",
        y="Daily Total (MWh)",
        text="Daily Total (MWh)",
        color="Plant",
        color_discrete_map=PLANT_COLORS,
        title=None,
    )
    fig.update_traces(texttemplate="%{text:,.2f}", textposition="outside", marker_line_width=0)
    fig = apply_dashboard_chart_layout(
        fig,
        x_title="Plant",
        y_title="Generation (MWh)",
        height=height,
        show_legend=False,
        hovermode="closest",
    )
    fig.update_layout(title_text="", legend_title_text="")
    st.plotly_chart(fig, use_container_width=True)


def build_cdf_chart(cdf_df: pd.DataFrame, value_label: str, height: int = 520) -> go.Figure:
    fig = go.Figure()
    for model_name, group in cdf_df.groupby("model", sort=False):
        trace_name = clean_display_value(model_name, "Model")
        fig.add_trace(
            go.Scatter(
                x=group["error_value"],
                y=group["cdf_probability"],
                mode="lines",
                name=trace_name,
                line=dict(width=3, shape="hv", color=MODEL_COLORS.get(trace_name)),
                customdata=group[["plant_filter", "observations"]],
                hovertemplate=(
                    "Model: %{fullData.name}<br>"
                    "Plant filter: %{customdata[0]}<br>"
                    f"{value_label}: %{{x:.3f}}<br>"
                    "Cumulative probability: %{y:.3f}<br>"
                    "Observations: %{customdata[1]:,}<extra></extra>"
                ),
            )
        )
    fig = apply_dashboard_chart_layout(
        fig,
        x_title=value_label,
        y_title="Cumulative Probability",
        height=height,
        show_legend=True,
        hovermode="closest",
    )
    fig.update_yaxes(range=[0, 1])
    fig.update_layout(title_text="", legend_title_text="")
    return fig


def forecast_value_column(selection: str) -> tuple[str, str]:
    if selection == "Total Cascade":
        return CASCADE_COLUMN, "Total Cascade Generation (MW)"
    plant_key = PLANT_KEYS_BY_LABEL[selection]
    return f"total_gen_{plant_key}", f"{selection} Generation (MW)"


def rgba_from_hex(hex_color: str, alpha: float) -> str:
    color = hex_color.lstrip("#")
    red, green, blue = (int(color[idx : idx + 2], 16) for idx in (0, 2, 4))
    return f"rgba({red}, {green}, {blue}, {alpha})"


def add_shadowed_line(
    fig: go.Figure,
    hours: pd.Series,
    values: pd.Series,
    name: str,
    line_color: str,
    fill_color: str,
    mode: str = "lines+markers",
    baseline: float | None = None,
    customdata: pd.DataFrame | None = None,
    hovertemplate: str | None = None,
    marker_size: float = 7,
    line_width: float = 3,
) -> None:
    numeric_values = pd.to_numeric(values, errors="coerce")
    trace_name = clean_display_value(name)
    if baseline is None:
        range_size = float(numeric_values.max() - numeric_values.min())
        baseline = float(numeric_values.min() - max(range_size * 0.08, 0.5))
    baseline_values = [baseline] * len(numeric_values)
    fig.add_trace(
        go.Scatter(
            x=hours,
            y=baseline_values,
            mode="lines",
            name="",
            line=dict(color="rgba(0,0,0,0)", width=0),
            hoverinfo="skip",
            hovertemplate=None,
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=hours,
            y=numeric_values,
            mode=mode,
            name=trace_name,
            line=dict(color=line_color, width=line_width),
            marker=dict(size=marker_size, color=line_color),
            fill="tonexty",
            fillcolor=fill_color,
            customdata=customdata,
            hovertemplate=hovertemplate,
            showlegend=bool(trace_name),
        )
    )


def build_all_plants_forecast_chart(forecast: pd.DataFrame, model_name: str, height: int = 550) -> go.Figure:
    plot_df = forecast.copy()
    plot_df["forecast_hour"] = range(1, len(plot_df) + 1)
    plot_df["forecast_date"] = pd.to_datetime(plot_df["datetime"], errors="coerce").dt.strftime("%Y-%m-%d").map(lambda value: clean_display_value(value, "Not available"))
    display_model = clean_display_value(model_name, "Model")
    fig = go.Figure()
    for plant, column in FORECAST_TOTAL_COLUMNS.items():
        display_plant = clean_display_value(plant, "Plant")
        values = pd.to_numeric(plot_df[column], errors="coerce")
        custom = pd.DataFrame({"forecast_date": plot_df["forecast_date"], "forecast_hour": plot_df["forecast_hour"], "plant": display_plant})
        fig.add_trace(
            go.Scatter(
                x=plot_df["forecast_hour"],
                y=values,
                mode="lines+markers",
                name=display_plant,
                line=dict(color=PLANT_COLORS[plant], width=2.6),
                marker=dict(size=5.5, color=PLANT_COLORS[plant]),
                customdata=custom,
                hovertemplate=(
                    f"Model: {display_model}<br>"
                    "Plant: %{customdata[2]}<br>"
                    "Forecast date: %{customdata[0]}<br>"
                    "Forecast hour: %{customdata[1]}<br>"
                    "Generation: %{y:.3f} MW<extra></extra>"
                ),
            )
        )
    fig = apply_dashboard_chart_layout(
        fig,
        x_title="Forecast Hour",
        y_title="Generation (MW)",
        height=height,
        show_legend=True,
        hovermode="x unified",
    )
    fig.update_xaxes(tickmode="linear", tick0=1, dtick=1)
    fig.update_layout(title_text="", legend_title_text="")
    return fig


def build_forecast_chart(forecast: pd.DataFrame, model_name: str, selection: str, height: int = 520) -> go.Figure:
    if selection == "All Plants":
        return build_all_plants_forecast_chart(forecast, model_name, height=550)
    value_col, y_label = forecast_value_column(selection)
    plot_df = forecast.copy()
    plot_df["forecast_hour"] = range(1, len(plot_df) + 1)
    plot_df["generation_mw"] = pd.to_numeric(plot_df[value_col], errors="coerce")
    plot_df["forecast_date"] = pd.to_datetime(plot_df["datetime"], errors="coerce").dt.strftime("%Y-%m-%d").map(lambda value: clean_display_value(value, "Not available"))
    display_model = clean_display_value(model_name, "Model")
    display_selection = clean_display_value(selection, "Forecast")
    line_color = MODEL_COLORS[model_name] if selection == "Total Cascade" else PLANT_COLORS[selection]
    fig = go.Figure()
    add_shadowed_line(
        fig,
        hours=plot_df["forecast_hour"],
        values=plot_df["generation_mw"],
        name=display_model if selection == "Total Cascade" else display_selection,
        line_color=line_color,
        fill_color=rgba_from_hex(line_color, 0.14),
        mode="lines+markers",
        customdata=plot_df[["forecast_date", "forecast_hour"]],
        hovertemplate=(
            f"Model: {display_model}<br>"
            "Forecast date: %{customdata[0]}<br>"
            "Forecast hour: %{customdata[1]}<br>"
            "Generation: %{y:.3f} MW<extra></extra>"
        ),
        marker_size=7,
        line_width=3.1,
    )
    fig = apply_dashboard_chart_layout(
        fig,
        x_title="Forecast Hour",
        y_title=y_label,
        height=height,
        show_legend=True,
        hovermode="x unified",
    )
    fig.update_xaxes(tickmode="linear", tick0=1, dtick=1)
    fig.update_layout(title_text="", legend_title_text="")
    return fig


def forecast_summary(forecast: pd.DataFrame, selection: str) -> dict[str, str]:
    value_col, _ = forecast_value_column(selection)
    values = pd.to_numeric(forecast[value_col], errors="coerce")
    peak_index = int(values.idxmax()) if values.notna().any() else 0
    date_value = pd.to_datetime(forecast["datetime"], errors="coerce").dropna()
    return {
        "date": date_value.iloc[0].strftime("%Y-%m-%d") if not date_value.empty else "N/A",
        "minimum": f"{values.min():,.2f} MW" if values.notna().any() else "N/A",
        "maximum": f"{values.max():,.2f} MW" if values.notna().any() else "N/A",
        "average": f"{values.mean():,.2f} MW" if values.notna().any() else "N/A",
        "peak_hour": f"{int(forecast.loc[peak_index, 'forecast_hour']):02d}" if "forecast_hour" in forecast.columns and values.notna().any() else "N/A",
        "total": f"{values.sum():,.2f} MWh" if selection == "Total Cascade" and values.notna().any() else f"{values.mean():,.2f} MW" if values.notna().any() else "N/A",
    }


def all_plants_summary(forecast: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for plant, column in FORECAST_TOTAL_COLUMNS.items():
        values = pd.to_numeric(forecast[column], errors="coerce")
        peak_idx = int(values.idxmax()) if values.notna().any() else 0
        rows.append(
            {
                "Plant": plant,
                "Minimum Forecast (MW)": values.min(),
                "Maximum Forecast (MW)": values.max(),
                "Average Forecast (MW)": values.mean(),
                "Peak Hour": int(forecast.loc[peak_idx, "forecast_hour"]) if "forecast_hour" in forecast.columns and values.notna().any() else "N/A",
                "24-Hour Energy (MWh)": values.sum(),
            }
        )
    return pd.DataFrame(rows)


def available_forecasts() -> dict[str, pd.DataFrame]:
    forecasts: dict[str, pd.DataFrame] = {}
    for model_name, config in MODEL_CONFIG.items():
        try:
            forecasts[model_name] = load_model_forecast(model_name, modified_ns(config["forecast_path"]))
        except Exception:
            continue
    return forecasts


def build_cascade_comparison_chart(
    forecasts: dict[str, pd.DataFrame],
    selected_models: list[str],
    display_option: str,
    y_axis_option: str,
    show_spread: bool,
    height: int = 610,
) -> go.Figure:
    fig = go.Figure()
    mode = "lines+markers" if display_option == "Line and Markers" else "lines"
    spread_frame = pd.DataFrame({"forecast_hour": range(1, 25)})
    selected_available = [model for model in selected_models if model in forecasts]
    for model_name in selected_available:
        forecast = forecasts[model_name].copy()
        forecast["forecast_hour"] = range(1, len(forecast) + 1)
        forecast["cascade"] = pd.to_numeric(forecast[CASCADE_COLUMN], errors="coerce")
        spread_frame[model_name] = forecast["cascade"].values
    visible_values = spread_frame[selected_available].stack().dropna() if selected_available else pd.Series(dtype=float)
    if visible_values.empty:
        baseline = 0.0
    else:
        visible_min = float(visible_values.min())
        visible_max = float(visible_values.max())
        range_size = visible_max - visible_min
        baseline = visible_min - max(range_size * 0.08, 0.5)
    if show_spread and len(selected_available) >= 2:
        spread_values = spread_frame[selected_available]
        spread_min = spread_values.min(axis=1)
        spread_max = spread_values.max(axis=1)
        fig.add_trace(
            go.Scatter(
                x=list(spread_frame["forecast_hour"]) + list(spread_frame["forecast_hour"])[::-1],
                y=list(spread_max) + list(spread_min)[::-1],
                fill="toself",
                fillcolor="rgba(190, 210, 235, 0.12)",
                line=dict(color="rgba(96, 165, 250, 0)"),
                name="Model Forecast Range",
                hoverinfo="skip",
                hovertemplate=None,
            )
        )
    for model_name in selected_available:
        display_model = clean_display_value(model_name, "Model")
        forecast = forecasts[model_name].copy()
        forecast["forecast_hour"] = range(1, len(forecast) + 1)
        forecast["cascade"] = pd.to_numeric(forecast[CASCADE_COLUMN], errors="coerce")
        color = MODEL_COLORS.get(display_model, "#EAF2FA")
        add_shadowed_line(
            fig,
            hours=forecast["forecast_hour"],
            values=forecast["cascade"],
            name=display_model,
            line_color=color,
            fill_color=rgba_from_hex(color, 0.10),
            mode=mode,
            baseline=baseline,
            hovertemplate=f"Model: {display_model}<br>Forecast hour: %{{x}}<br>Total cascade: %{{y:.3f}} MW<extra></extra>",
            marker_size=7.5,
            line_width=3.2,
        )
    fig = apply_dashboard_chart_layout(
        fig,
        x_title="Forecast Hour",
        y_title="Total Cascade Generation (MW)",
        height=height,
        show_legend=True,
        hovermode="x unified",
    )
    fig.update_xaxes(tickmode="linear", tick0=1, dtick=1)
    fig.update_layout(title_text="", legend_title_text="")
    if y_axis_option == "Start at Zero":
        fig.update_yaxes(rangemode="tozero")
    return fig


def cascade_comparison_summary(forecasts: dict[str, pd.DataFrame], selected_models: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    series_by_model: dict[str, pd.Series] = {}
    for model_name in selected_models:
        if model_name not in forecasts:
            continue
        forecast = forecasts[model_name].copy()
        forecast["forecast_hour"] = range(1, len(forecast) + 1)
        values = pd.to_numeric(forecast[CASCADE_COLUMN], errors="coerce")
        series_by_model[model_name] = values.reset_index(drop=True)
        peak_idx = int(values.idxmax())
        low_idx = int(values.idxmin())
        date_value = pd.to_datetime(forecast["datetime"], errors="coerce").dropna()
        rows.append(
            {
                "Model": model_name,
                "Forecast Date": date_value.iloc[0].strftime("%Y-%m-%d") if not date_value.empty else "N/A",
                "Mean Cascade Forecast (MW)": values.mean(),
                "Minimum Cascade Forecast (MW)": values.min(),
                "Maximum Cascade Forecast (MW)": values.max(),
                "Peak Hour": int(forecast.loc[peak_idx, "forecast_hour"]),
                "Peak Generation (MW)": values.max(),
                "Lowest Hour": int(forecast.loc[low_idx, "forecast_hour"]),
                "Lowest Generation (MW)": values.min(),
            }
        )
    diff_rows: list[dict[str, object]] = []
    pairs = [("RBFNN", "Random Forest"), ("RBFNN", "XGBoost"), ("Random Forest", "XGBoost")]
    for left, right in pairs:
        if left in series_by_model and right in series_by_model:
            diff_rows.append(
                {
                    "Model Pair": f"{left} vs {right}",
                    "Mean Absolute Forecast Difference (MW)": (series_by_model[left] - series_by_model[right]).abs().mean(),
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(diff_rows)


def render_hourly_line_chart(forecast: pd.DataFrame, height: int = 340) -> None:
    render_plotly_line_chart(forecast, height)


def render_daily_bar_chart(forecast: pd.DataFrame, height: int = 340) -> None:
    render_plotly_bar_chart(forecast, height)


def render_testing_cdf_section() -> None:
    section_title("Testing-Period Error CDF Analysis")
    testing_df, warnings_list = combine_testing_predictions(testing_modified_tokens())
    for warning in warnings_list:
        st.warning(warning)
    if testing_df.empty:
        st.info("Day-ahead testing prediction workbooks are not available. Generate model testing predictions before using the CDF analysis.")
        return

    control_cols = st.columns(3, gap="medium")
    with control_cols[0]:
        plant_filter = st.selectbox("Plant", ["All Agus Plants", *PLANTS], key="cdf_plant_filter")
    with control_cols[1]:
        model_filter = st.selectbox("Model", ["All Models", *MODEL_CONFIG.keys()], key="cdf_model_filter")
    with control_cols[2]:
        error_measure = st.selectbox("Error Measure", ["Absolute Error (MW)", "Absolute Percentage Error (%)"], key="cdf_error_measure")

    filtered = filter_testing_data(testing_df, plant_filter, model_filter)
    if filtered.empty:
        st.info("No testing observations match the selected CDF filters.")
        return
    date_min = filtered["datetime"].min().strftime("%Y-%m-%d")
    date_max = filtered["datetime"].max().strftime("%Y-%m-%d")
    info_box(
        f"CDF source: held-out rolling 24-hour day-ahead testing prediction workbooks only. Current filter testing range: {date_min} to {date_max}."
    )

    cdf_df, summary_df, value_label = prepare_cdf_data(testing_df, plant_filter, model_filter, error_measure)
    if cdf_df.empty:
        st.info("No valid testing observations remain after applying the selected error definition and operational threshold.")
        return
    st.plotly_chart(build_cdf_chart(cdf_df, value_label), use_container_width=True)
    note_card(
        "The CDF indicates the proportion of testing observations whose forecast error is less than or equal to a selected error threshold. "
        "A curve that rises more rapidly and remains farther to the upper-left indicates lower forecast error. "
        "This analysis is based only on the held-out testing period."
    )
    st.dataframe(calculate_error_summary(summary_df), use_container_width=True, hide_index=True)


def dashboard_overview() -> None:
    forecast = None
    cleaned = None
    outage = None
    if "overview_forecast_model" not in st.session_state:
        st.session_state["overview_forecast_model"] = "RBFNN"
    overview_model = st.session_state["overview_forecast_model"]
    try:
        forecast = load_model_forecast(overview_model, modified_ns(MODEL_CONFIG[overview_model]["forecast_path"]))
    except Exception as exc:
        forecast = None
        overview_error = str(exc)
    else:
        overview_error = ""
    try:
        cleaned = load_cleaned_data(modified_ns(cleaned_source_path()))
    except Exception:
        pass
    try:
        outage = load_outage_plan(modified_ns(OUTAGE_PLAN_PATH))
    except Exception:
        pass

    kpis = forecast_kpis(forecast, outage)
    cleaned_dt = cleaned_datetime(cleaned)
    outage_info = outage_summary(outage)

    render_header(file_timestamp(MODEL_CONFIG[overview_model]["forecast_path"]))
    cols = st.columns(5, gap="medium")
    with cols[0]:
        metric_card("Total Cascade Forecast", kpis["cascade_total"], "24-hour cascade sum")
    with cols[1]:
        metric_card("Peak Cascade Output", kpis["peak"], "Maximum hourly output")
    with cols[2]:
        metric_card("Unavailable Unit-Hours", f"{outage_info['unavailable']:,}", "Planned OFF entries", "red" if outage_info["unavailable"] else "")
    with cols[3]:
        metric_card("Latest Cleaned Timestamp", cleaned_dt.max().strftime("%Y-%m-%d %H:%M") if cleaned_dt.notna().any() else "N/A", "From cleaned data")
    with cols[4]:
        metric_card("Latest Forecast Timestamp", file_timestamp(MODEL_CONFIG[overview_model]["forecast_path"]), f"{overview_model} forecast file modified")

    section_title("Overview Forecast Model")
    selected_overview_model = st.radio(
        "Overview Forecast Model",
        list(MODEL_CONFIG.keys()),
        horizontal=True,
        index=list(MODEL_CONFIG.keys()).index(overview_model),
        key="overview_forecast_model_radio",
    )
    st.session_state["overview_forecast_model"] = selected_overview_model
    if selected_overview_model != overview_model:
        st.rerun()

    if forecast is not None:
        chart_cols = st.columns(2, gap="large")
        with chart_cols[0]:
            section_title(f"{overview_model} \u2014 Total Generation per Agus Plant Hourly")
            with st.container():
                render_plotly_line_chart(forecast, height=460, model_name=overview_model)
        with chart_cols[1]:
            section_title(f"{overview_model} \u2014 Total Generation per Agus Plant Daily Total")
            with st.container():
                render_plotly_bar_chart(forecast, height=460, model_name=overview_model)
    elif overview_error:
        st.warning(overview_error)

    section_title("Recent Activity and Status")
    with st.container():
        status_cols = st.columns(4, gap="medium")
        status_items = [
            ("Raw Excel Data", RAW_EXCEL_PATH),
            ("Cleaned Data", cleaned_source_path()),
            ("Outage Plan", OUTAGE_PLAN_PATH),
            ("RBFNN Forecast", FORECAST_EXCEL),
        ]
        for col, (label, path) in zip(status_cols, status_items):
            with col:
                status_card(label, path)

    section_title("Model Forecast File Status")
    status_cols = st.columns(3, gap="medium")
    for col, (model_name, config) in zip(status_cols, MODEL_CONFIG.items()):
        with col:
            status_card(f"{model_name} Forecast", config["forecast_path"])

    render_testing_cdf_section()


def data_management_page() -> None:
    render_header(file_timestamp(cleaned_source_path()))
    section_title("Data Management")
    with st.container():
        info_strip("Update raw Excel data, refresh files, and clean hourly dataset.")

    section_title("Data Actions")
    with st.container():
        action_cols = st.columns(3, gap="large")
        with action_cols[0]:
            st.markdown(
                """
                <div class="action-card-full">
                    <div class="action-title">Upload Excel File</div>
                    <div class="action-desc">Upload a new Excel workbook to replace the current raw data source.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            uploaded = st.file_uploader("Upload Excel File", type=["xlsx", "xls"], label_visibility="collapsed")
            if uploaded is not None:
                DATA_DIR.mkdir(parents=True, exist_ok=True)
                RAW_EXCEL_PATH.write_bytes(uploaded.getbuffer())
                clear_cached_data("Uploaded Excel file saved.")
        with action_cols[1]:
            refresh_clicked = action_card(
                "Refresh Data From Disk",
                "Reload the latest raw Excel files from disk without restarting the application.",
                "Refresh Data From Disk",
                key="refresh_data_from_disk",
            )
            if refresh_clicked:
                clear_cached_data("Data refreshed from disk.")
        with action_cols[2]:
            clean_clicked = action_card(
                "Run Data Cleaning",
                "Run the data cleaning process to update the cleaned hourly dataset.",
                "Run Data Cleaning",
                key="run_data_cleaning",
            )
            if clean_clicked:
                with st.spinner("Running data cleaning workflow..."):
                    try:
                        show_script_result(run_script(CLEAN_SCRIPT), "Data cleaning")
                    except Exception as exc:
                        st.error(str(exc))

    raw_df = None
    cleaned_df = None
    try:
        raw_df = load_raw_excel(modified_ns(RAW_EXCEL_PATH))
    except Exception:
        pass
    try:
        cleaned_df = load_cleaned_data(modified_ns(cleaned_source_path()))
    except Exception:
        pass

    cleaned_dt = cleaned_datetime(cleaned_df)
    section_title("Data Confirmation")
    with st.container():
        confirm_cols_1 = st.columns(4, gap="medium")
        with confirm_cols_1[0]:
            metric_card("Raw Excel Status", file_status(RAW_EXCEL_PATH), "Source workbook", "green" if file_status(RAW_EXCEL_PATH) == "Available" else "")
        with confirm_cols_1[1]:
            metric_card("Raw Excel Last Modified", file_timestamp(RAW_EXCEL_PATH), "File timestamp")
        with confirm_cols_1[2]:
            metric_card("Latest Raw Data Date", latest_date_from_df(raw_df), "Detected from Date/Time")
        with confirm_cols_1[3]:
            metric_card("Cleaned Data Status", file_status(cleaned_source_path()), "Cleaned output", "green" if file_status(cleaned_source_path()) == "Available" else "")

        confirm_cols_2 = st.columns(4, gap="medium")
        with confirm_cols_2[0]:
            metric_card("Cleaned Data Last Modified", file_timestamp(cleaned_source_path()), "File timestamp")
        with confirm_cols_2[1]:
            metric_card("Latest Cleaned Data Date", cleaned_dt.max().strftime("%Y-%m-%d %H:%M") if cleaned_dt.notna().any() else "N/A", "Detected from date/time")
        with confirm_cols_2[2]:
            metric_card("Total Cleaned Rows", f"{len(cleaned_df):,}" if cleaned_df is not None else "N/A", "Hourly records")
        with confirm_cols_2[3]:
            metric_card("Total Cleaned Columns", f"{len(cleaned_df.columns):,}" if cleaned_df is not None else "N/A", "Prepared features")

        rainfall_available = cleaned_df is not None and "rainfall" in cleaned_df.columns
        metric_card(
            "Rainfall Input",
            "Available" if rainfall_available else "Missing",
            "Rainfall (daily-derived hourly-equivalent)",
            "green" if rainfall_available else "red",
        )

    section_title("Data Notes")
    with st.container():
        note_card("Uploading replaces the raw Excel workbook used by the cleaning script. Refreshing reloads the latest files from disk without restarting the application. Running data cleaning updates the cleaned hourly dataset used by the forecasting workflow.")


def planned_outage_page() -> None:
    render_header(file_timestamp(OUTAGE_PLAN_PATH))
    section_title("Planned Outage Planning")
    instruction_card("Edit planned unit availability below. Use 1 for ON/available/running unit and 0 for OFF/unavailable/outage unit.")

    if st.session_state.get("outage_saved"):
        st.success("Outage plan saved successfully.")
        st.session_state["outage_saved"] = False

    try:
        outage_df = load_outage_plan(modified_ns(OUTAGE_PLAN_PATH))
    except Exception as exc:
        st.error(str(exc))
        return

    outage_cols = outage_columns(outage_df)
    editable = outage_df.copy()
    for col in outage_cols:
        editable[col] = pd.to_numeric(editable[col], errors="coerce").fillna(1).clip(0, 1).round().astype(int)

    summary = outage_summary(editable)
    kpi_cols = st.columns(4, gap="medium")
    with kpi_cols[0]:
        metric_card("Unavailable Unit-Hours", f"{summary['unavailable']:,}", "Cells marked 0", "red" if summary["unavailable"] else "")
    with kpi_cols[1]:
        metric_card("Available Unit-Hours", f"{summary['available']:,}", "Cells marked 1", "green")
    with kpi_cols[2]:
        affected = ", ".join(summary["affected_plants"]) if summary["affected_plants"] else "None"
        metric_card("Affected Plants", str(len(summary["affected_plants"])), affected, "light")
    with kpi_cols[3]:
        metric_card("Outage Plan Last Modified", file_timestamp(OUTAGE_PLAN_PATH), "Saved outage workbook", "light")

    column_config = {
        col: st.column_config.NumberColumn(col, min_value=0, max_value=1, step=1, format="%d")
        for col in outage_cols
    }
    edited = st.data_editor(
        editable,
        use_container_width=True,
        height=330,
        hide_index=True,
        column_config=column_config,
        disabled=[col for col in editable.columns if col not in outage_cols],
    )

    button_cols = st.columns(4, gap="large")
    with button_cols[0]:
        if st.button("Save Outage Plan", use_container_width=True):
            try:
                save_outage_plan(edited)
                st.session_state["outage_saved"] = True
                st.rerun()
            except Exception as exc:
                st.error(str(exc))
    with button_cols[1]:
        if st.button("Forecast Day-Ahead", use_container_width=True):
            with st.spinner("Generating day-ahead forecasts for RBFNN, Random Forest, and XGBoost..."):
                try:
                    save_outage_plan(edited)
                    validate_fast_forecast_ready()
                    show_script_result(run_script(RBFNN_SCRIPT, ["--forecast-only"]), "RBFNN day-ahead forecast")
                    show_script_result(run_script(BENCHMARK_SCRIPT, ["--forecast-only"]), "Benchmark day-ahead forecasts")
                    st.cache_data.clear()
                    load_forecast(modified_ns(FORECAST_EXCEL))
                    st.success("Day-ahead forecasts completed for RBFNN, Random Forest, and XGBoost using the saved outage plan.")
                except Exception as exc:
                    st.error(str(exc))
    with button_cols[2]:
        if st.button("Retrain RBFNN Model", use_container_width=True):
            with st.spinner("Retraining RBFNN model..."):
                try:
                    show_script_result(run_script(RBFNN_SCRIPT, ["--train"]), "RBFNN retraining")
                except Exception as exc:
                    st.error(str(exc))
    with button_cols[3]:
        if st.button("Retrain Benchmark Models", use_container_width=True):
            with st.spinner("Retraining Random Forest and XGBoost benchmark models..."):
                try:
                    show_script_result(run_script(BENCHMARK_SCRIPT, ["--train"]), "Benchmark model retraining")
                except Exception as exc:
                    st.error(str(exc))

    section_title("ON/OFF Status View")
    with st.container():
        status_view = outage_status_frame(edited)
        if not status_view.empty:
            st.dataframe(style_outage_status(status_view), use_container_width=True, height=320)
        if summary["affected_units"]:
            note_card(f'Affected Units: {", ".join(summary["affected_units"])}')


def forecast_file_status_cards() -> None:
    cols = st.columns(3, gap="medium")
    for col, (model_name, config) in zip(cols, MODEL_CONFIG.items()):
        with col:
            status_card(f"{model_name} Forecast", config["forecast_path"])


def render_selected_forecast_section(model_name: str, forecast: pd.DataFrame) -> None:
    section_title(f"{model_name} Day-Ahead Forecast")
    plant_selection = st.selectbox(
        "Forecast View",
        ["Total Cascade", "All Plants", *PLANTS],
        key=f"{model_name}_forecast_view",
    )
    st.plotly_chart(build_forecast_chart(forecast, model_name, plant_selection), use_container_width=True)
    if plant_selection == "All Plants":
        date_value = pd.to_datetime(forecast["datetime"], errors="coerce").dropna()
        forecast_date = date_value.iloc[0].strftime("%Y-%m-%d") if not date_value.empty else "N/A"
        note_card(f"{model_name} forecast date: {forecast_date}")
        st.dataframe(all_plants_summary(forecast), use_container_width=True, hide_index=True)
    else:
        summary = forecast_summary(forecast, plant_selection)
        cols = st.columns(6, gap="medium")
        with cols[0]:
            metric_card("Forecast Date", summary["date"], "Operational forecast day")
        with cols[1]:
            metric_card("Minimum", summary["minimum"], "Lowest hourly value")
        with cols[2]:
            metric_card("Maximum", summary["maximum"], "Highest hourly value")
        with cols[3]:
            metric_card("Average", summary["average"], "Mean hourly value")
        with cols[4]:
            metric_card("Peak Hour", summary["peak_hour"], "Forecast hour")
        with cols[5]:
            metric_card("Daily Total" if plant_selection == "Total Cascade" else "Average Output", summary["total"], "Selected series")

    st.dataframe(display_forecast_table(forecast), use_container_width=True, height=320)
    excel_bytes = forecast_excel_download_bytes(model_name, forecast)
    if excel_bytes:
        st.download_button(
            f"Download {model_name} Forecast Excel",
            data=excel_bytes,
            file_name=MODEL_CONFIG[model_name]["forecast_path"].name,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )


def render_cascade_comparison_section() -> None:
    section_title("Total Cascade Day-Ahead Forecast Comparison")
    forecast_file_status_cards()
    forecasts = available_forecasts()
    missing_models = [model for model in MODEL_CONFIG if model not in forecasts]
    if missing_models:
        st.info(f"Run the missing model forecast to include it in the comparison: {', '.join(missing_models)}.")
    if not forecasts:
        st.warning("No valid 24-hour forecast workbooks are available for comparison.")
        return

    control_cols = st.columns([2.2, 1.2, 1.2, 1.2], gap="medium")
    with control_cols[0]:
        selected_models = st.multiselect(
            "Visible Models",
            list(MODEL_CONFIG.keys()),
            default=list(forecasts.keys()),
            key="comparison_visible_models",
        )
    with control_cols[1]:
        display_option = st.selectbox("Display", ["Line and Markers", "Lines Only"], key="comparison_display")
    with control_cols[2]:
        y_axis_option = st.selectbox("Y-Axis", ["Automatic Range", "Start at Zero"], key="comparison_y_axis")
    with control_cols[3]:
        show_spread = st.checkbox("Show hourly model spread", key="comparison_spread")

    selected_models = [model for model in selected_models if model in forecasts]
    if not selected_models:
        st.info("Select at least one available model to display the comparison graph.")
        return
    if len(selected_models) == 1:
        st.info("Comparison requires additional model outputs; showing the available selected forecast.")

    st.plotly_chart(
        build_cascade_comparison_chart(forecasts, selected_models, display_option, y_axis_option, show_spread),
        use_container_width=True,
    )
    summary_df, diff_df = cascade_comparison_summary(forecasts, selected_models)
    if not summary_df.empty:
        st.dataframe(summary_df, use_container_width=True, hide_index=True)
    if not diff_df.empty:
        st.dataframe(diff_df, use_container_width=True, hide_index=True)


def forecast_page() -> None:
    if "selected_forecast_model" not in st.session_state:
        st.session_state["selected_forecast_model"] = "RBFNN"

    try:
        initial_forecast = load_model_forecast(st.session_state["selected_forecast_model"], modified_ns(MODEL_CONFIG[st.session_state["selected_forecast_model"]]["forecast_path"]))
        header_time = file_timestamp(MODEL_CONFIG[st.session_state["selected_forecast_model"]]["forecast_path"])
    except Exception:
        initial_forecast = None
        header_time = None

    render_header(header_time)
    section_title("Forecast")
    selected_model = st.radio(
        "Forecast Model",
        list(MODEL_CONFIG.keys()),
        horizontal=True,
        index=list(MODEL_CONFIG.keys()).index(st.session_state["selected_forecast_model"]),
        key="forecast_model_radio",
    )
    st.session_state["selected_forecast_model"] = selected_model
    note_card(MODEL_CONFIG[selected_model]["description"])

    button_cols = st.columns([1.3, 1.3, 3.4], gap="medium")
    with button_cols[0]:
        if st.button(f"Run {selected_model} Forecast", use_container_width=True):
            with st.spinner(f"Running {selected_model} forecast without retraining..."):
                try:
                    result = run_selected_model_forecast(selected_model)
                    show_script_result(result, f"{selected_model} forecast")
                    if result.returncode == 0:
                        st.cache_data.clear()
                        if selected_model in {"Random Forest", "XGBoost"}:
                            st.success("Benchmark forecast-only mode completed using saved Random Forest and XGBoost model artifacts.")
                        else:
                            st.success("RBFNN forecast completed using saved model artifacts.")
                except Exception as exc:
                    st.error(str(exc))
    with button_cols[1]:
        if st.button("Refresh Forecast Files", use_container_width=True):
            clear_cached_data("Forecast files refreshed from disk.")

    try:
        forecast = load_model_forecast(selected_model, modified_ns(MODEL_CONFIG[selected_model]["forecast_path"]))
    except Exception as exc:
        st.error(str(exc))
        forecast = None

    try:
        outage_df = load_outage_plan(modified_ns(OUTAGE_PLAN_PATH))
    except Exception:
        outage_df = None

    if forecast is not None:
        kpis = forecast_kpis(forecast, outage_df)
        kpi_cols = st.columns(4, gap="medium")
        with kpi_cols[0]:
            metric_card("Total Cascade Forecast", kpis["cascade_total"], "24-hour cascade sum")
        with kpi_cols[1]:
            metric_card("Peak Cascade Output", kpis["peak"], "Maximum hourly output")
        with kpi_cols[2]:
            metric_card("Affected Plants", kpis["affected"], "Saved outage plan", "red" if kpis["affected"] != "None" else "")
        with kpi_cols[3]:
            metric_card("Forecast Generated", file_timestamp(MODEL_CONFIG[selected_model]["forecast_path"]), "Forecast file modified")
        render_selected_forecast_section(selected_model, forecast)

    render_cascade_comparison_section()


def system_information_page() -> None:
    render_header()

    section_title("Forecasting Workflow")
    workflow = [
        ("1", "Update Raw Data", "Upload or refresh Excel data"),
        ("2", "Clean Data", "Prepare hourly dataset"),
        ("3", "Edit Outage Plan", "Set unit availability"),
        ("4", "Forecast / Retrain", "Generate or retrain RBFNN"),
        ("5", "View Forecast", "Review results and download"),
    ]
    with st.container():
        cols = st.columns(5, gap="medium")
        for col, (num, title, note) in zip(cols, workflow):
            with col:
                st.markdown(
                    f"""
                    <div class="workflow-card">
                        <div class="workflow-step">{num}</div>
                        <div class="workflow-title">{title}</div>
                        <div class="workflow-note">{note}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    section_title("File Status Overview")
    with st.container():
        status_cols = st.columns(4, gap="medium")
        status_items = [
            ("Raw Excel Data", RAW_EXCEL_PATH),
            ("Cleaned Data", cleaned_source_path()),
            ("Outage Plan", OUTAGE_PLAN_PATH),
            ("RBFNN Forecast", FORECAST_EXCEL),
        ]
        for col, (label, path) in zip(status_cols, status_items):
            with col:
                status_card(label, path)

    section_title("System Details")
    try:
        cleaned_df = load_cleaned_data(modified_ns(cleaned_source_path()))
        rainfall_status = "Available" if "rainfall" in cleaned_df.columns else "Missing"
    except Exception:
        rainfall_status = "Unavailable"
    details = pd.DataFrame(
        [
            {"Detail": "Python Version", "Value": platform.python_version()},
            {"Detail": "Streamlit Version", "Value": st.__version__},
            {"Detail": "Forecast Models", "Value": "RBFNN, Random Forest, XGBoost"},
            {"Detail": "Forecast Horizon", "Value": "24 Hours Day-Ahead"},
            {"Detail": "Hydrologic Input", "Value": f"Rainfall (daily-derived hourly-equivalent): {rainfall_status}"},
            {"Detail": "Timezone", "Value": "Asia/Manila"},
            {"Detail": "Last System Check", "Value": datetime.now().strftime("%Y-%m-%d %H:%M")},
        ]
    )
    with st.container():
        st.dataframe(details, use_container_width=True, hide_index=True, height=250)


def render_sidebar() -> str:
    st.markdown('<div class="sidebar-title">NPC Agus Decision Support</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-subtitle">Hydropower Forecasting Prototype Panel</div>', unsafe_allow_html=True)
    selected_page = st.radio("Navigation", NAV_ITEMS, label_visibility="collapsed")
    st.markdown(
        f"""
        <div class="sidebar-mini">
            Raw Excel Data: {file_status(RAW_EXCEL_PATH)}<br>
            RBFNN Forecast: {file_status(FORECAST_EXCEL)}<br>
            RF Forecast: {file_status(RANDOM_FOREST_FORECAST_EXCEL)}<br>
            XGBoost Forecast: {file_status(XGBOOST_FORECAST_EXCEL)}
        </div>
        """,
        unsafe_allow_html=True,
    )
    return selected_page


with st.sidebar:
    selected_page = render_sidebar()

if selected_page == "Dashboard Overview":
    dashboard_overview()
elif selected_page == "Data Management":
    data_management_page()
elif selected_page == "Planned Outage Planning":
    planned_outage_page()
elif selected_page == "Forecast":
    forecast_page()
elif selected_page == "System Information":
    system_information_page()

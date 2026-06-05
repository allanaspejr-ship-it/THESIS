# Requirements note: pip install streamlit pandas openpyxl plotly pyarrow

from __future__ import annotations

import platform
import subprocess
import sys
from datetime import datetime
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
MODELS_DIR = APP_DIR / "models" / "rbfnn"
METADATA_DIR = APP_DIR / "metadata"
BENCHMARK_DIR = APP_DIR / "benchmark"
THESIS_FIGURES_DIR = APP_DIR / "thesis_figures"
SCRIPTS_DIR = APP_DIR / "scripts"

RAW_EXCEL_PATH = DATA_DIR / "DATA(JAN2024-JUNE2025).xlsx"
CLEANED_EXCEL = CLEANED_DIR / "cleaned_hourly_data.xlsx"
CLEANED_PARQUET = CLEANED_DIR / "cleaned_hourly_data.parquet"
OUTAGE_PLAN_PATH = OUTAGES_DIR / "Planned_Outages_Input.xlsx"
FORECAST_EXCEL = FORECAST_DIR / "Day_Ahead_24H_RBFNN_Forecast.xlsx"
FORECAST_CSV = FORECAST_DIR / "Day_Ahead_24H_RBFNN_Forecast.csv"
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
FORECAST_TOTAL_COLUMNS = {
    "Agus 1": "Total_Gen_Agus1_MW",
    "Agus 2": "Total_Gen_Agus2_MW",
    "Agus 4": "Total_Gen_Agus4_MW",
    "Agus 5": "Total_Gen_Agus5_MW",
    "Agus 6": "Total_Gen_Agus6_MW",
    "Agus 7": "Total_Gen_Agus7_MW",
}
CASCADE_COLUMN = "Total_Cascade_Generation_MW"
NAV_ITEMS = [
    "Dashboard Overview",
    "Data Management",
    "Planned Outage Planning",
    "RBFNN Forecasting",
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


@st.cache_data(show_spinner=False)
def load_forecast(modified: int) -> pd.DataFrame:
    _ = modified
    if not FORECAST_EXCEL.exists():
        raise FileNotFoundError("RBFNN forecast file is missing.")
    return pd.read_excel(FORECAST_EXCEL)


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


PLANT_COLORS = {
    "Agus 1": "#2563eb",
    "Agus 2": "#06b6d4",
    "Agus 4": "#10b981",
    "Agus 5": "#8b5cf6",
    "Agus 6": "#fbbf24",
    "Agus 7": "#ef4444",
}


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
    display = forecast.copy()
    display.columns = [
        str(col)
        .replace("_", " ")
        .replace("MW", "(MW)")
        .replace("Gen", "Generation")
        .replace("Total Cascade Generation (MW)", "Total Cascade (MW)")
        for col in display.columns
    ]
    return display


def forecast_plot_frame(forecast: pd.DataFrame) -> tuple[pd.DataFrame, str, pd.DataFrame]:
    plot_df = forecast.copy()
    plot_df["Forecast Datetime"] = forecast_datetime(plot_df)
    x_col = "Forecast Datetime" if plot_df["Forecast Datetime"].notna().any() else "Hour"
    plant_totals = plot_df[["Forecast Datetime", "Hour", *FORECAST_TOTAL_COLUMNS.values()]].rename(
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


def style_plotly_chart(fig: go.Figure) -> go.Figure:
    fig.update_layout(
        title_text="",
        annotations=[],
        template="plotly_dark",
        paper_bgcolor="#0c2947",
        plot_bgcolor="#0c2947",
        font=dict(color="#f8fafc", size=14),
        title_font=dict(size=20, color="#ffffff"),
        legend=dict(font=dict(color="#f8fafc", size=13), bgcolor="rgba(0,0,0,0)"),
        hoverlabel=dict(bgcolor="#08213f", font=dict(color="#f8fafc")),
        xaxis=dict(
            title_font=dict(color="#f8fafc", size=14),
            tickfont=dict(color="#f8fafc", size=12),
            gridcolor="rgba(148,163,184,0.18)",
            zerolinecolor="rgba(148,163,184,0.28)",
        ),
        yaxis=dict(
            title_font=dict(color="#f8fafc", size=14),
            tickfont=dict(color="#f8fafc", size=12),
            gridcolor="rgba(148,163,184,0.18)",
            zerolinecolor="rgba(148,163,184,0.28)",
        ),
        margin=dict(l=50, r=30, t=20, b=50),
    )
    return fig


def render_plotly_line_chart(forecast: pd.DataFrame, height: int = 340) -> None:
    missing = [col for col in FORECAST_TOTAL_COLUMNS.values() if col not in forecast.columns]
    if missing:
        st.warning("Forecast total generation columns are incomplete.")
        return
    _, x_col, plant_totals = forecast_plot_frame(forecast)
    long_df = plant_totals.melt(
        id_vars=["Forecast Datetime", "Hour"],
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
    fig.update_layout(
        template="plotly_dark",
        height=height,
        title_text="",
        annotations=[],
        margin=dict(l=50, r=30, t=20, b=50),
        xaxis_title="Forecast Hour",
        yaxis_title="Generation (MW)",
        hovermode="x unified",
    )
    fig = style_plotly_chart(fig)
    st.plotly_chart(fig, use_container_width=True)


def render_plotly_bar_chart(forecast: pd.DataFrame, height: int = 340) -> None:
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
    fig.update_layout(
        template="plotly_dark",
        height=height,
        title_text="",
        annotations=[],
        margin=dict(l=50, r=30, t=20, b=50),
        showlegend=False,
        xaxis_title="",
        yaxis_title="Generation (MWh)",
    )
    fig = style_plotly_chart(fig)
    st.plotly_chart(fig, use_container_width=True)


def render_hourly_line_chart(forecast: pd.DataFrame, height: int = 340) -> None:
    render_plotly_line_chart(forecast, height)


def render_daily_bar_chart(forecast: pd.DataFrame, height: int = 340) -> None:
    render_plotly_bar_chart(forecast, height)


def dashboard_overview() -> None:
    forecast = None
    cleaned = None
    outage = None
    try:
        forecast = load_forecast(modified_ns(FORECAST_EXCEL))
    except Exception:
        pass
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

    render_header(kpis["forecast_generated"])
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
        metric_card("Latest Forecast Timestamp", file_timestamp(FORECAST_EXCEL), "Forecast file modified")

    if forecast is not None:
        chart_cols = st.columns(2, gap="large")
        with chart_cols[0]:
            section_title("Total Generation per Agus Plant Hourly")
            with st.container():
                render_plotly_line_chart(forecast)
        with chart_cols[1]:
            section_title("Total Generation per Agus Plant Daily Total")
            with st.container():
                render_plotly_bar_chart(forecast)

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

    button_cols = st.columns(3, gap="large")
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
            with st.spinner("Generating fast forecast-only output..."):
                try:
                    save_outage_plan(edited)
                    validate_fast_forecast_ready()
                    show_script_result(run_script(RBFNN_SCRIPT, ["--forecast-only"]), "Day-ahead forecast")
                    st.cache_data.clear()
                    load_forecast(modified_ns(FORECAST_EXCEL))
                    st.success("Fast forecast-only mode completed using saved RBFNN models and saved outage plan.")
                except Exception as exc:
                    st.error(str(exc))
    with button_cols[2]:
        if st.button("Retrain RBFNN Model", use_container_width=True):
            with st.spinner("Retraining RBFNN model..."):
                try:
                    show_script_result(run_script(RBFNN_SCRIPT, ["--train"]), "RBFNN retraining")
                except Exception as exc:
                    st.error(str(exc))

    section_title("ON/OFF Status View")
    with st.container():
        status_view = outage_status_frame(edited)
        if not status_view.empty:
            st.dataframe(style_outage_status(status_view), use_container_width=True, height=320)
        if summary["affected_units"]:
            note_card(f'Affected Units: {", ".join(summary["affected_units"])}')


def rbfnn_forecasting_page() -> None:
    try:
        forecast = load_forecast(modified_ns(FORECAST_EXCEL))
    except Exception as exc:
        st.error(str(exc))
        return

    try:
        outage_df = load_outage_plan(modified_ns(OUTAGE_PLAN_PATH))
    except Exception:
        outage_df = None

    kpis = forecast_kpis(forecast, outage_df)
    render_header(kpis["forecast_generated"])
    refresh_cols = st.columns([5, 1.25], gap="medium")
    with refresh_cols[1]:
        if st.button("Refresh Forecast", use_container_width=True):
            clear_cached_data("Forecast refreshed from disk.")

    kpi_cols = st.columns(4, gap="medium")
    with kpi_cols[0]:
        metric_card("Total Cascade Forecast", kpis["cascade_total"], "24-hour cascade sum")
    with kpi_cols[1]:
        metric_card("Peak Cascade Output", kpis["peak"], "Maximum hourly output")
    with kpi_cols[2]:
        metric_card("Affected Plants", kpis["affected"], "Saved outage plan", "red" if kpis["affected"] != "None" else "")
    with kpi_cols[3]:
        metric_card("Forecast Generated", kpis["forecast_generated"], "Forecast file modified")

    section_title("Forecast Table")
    with st.container():
        st.dataframe(display_forecast_table(forecast), use_container_width=True, height=320)
    dl_cols = st.columns(2, gap="large")
    with dl_cols[0]:
        excel_bytes = read_file_bytes(FORECAST_EXCEL)
        if excel_bytes:
            st.download_button(
                "Download Forecast Excel",
                data=excel_bytes,
                file_name="Day_Ahead_24H_RBFNN_Forecast.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )
    with dl_cols[1]:
        csv_bytes = read_file_bytes(FORECAST_CSV) or forecast.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download Forecast CSV",
            data=csv_bytes,
            file_name="Day_Ahead_24H_RBFNN_Forecast.csv",
            mime="text/csv",
            use_container_width=True,
        )

    chart_cols = st.columns(2, gap="large")
    with chart_cols[0]:
        section_title("Total Generation per Agus Plant Hourly")
        with st.container():
            render_plotly_line_chart(forecast)
    with chart_cols[1]:
        section_title("Total Generation per Agus Plant Daily Total")
        with st.container():
            render_plotly_bar_chart(forecast)


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
    details = pd.DataFrame(
        [
            {"Detail": "Python Version", "Value": platform.python_version()},
            {"Detail": "Streamlit Version", "Value": st.__version__},
            {"Detail": "Forecast Model", "Value": "RBFNN"},
            {"Detail": "Forecast Horizon", "Value": "24 Hours Day-Ahead"},
            {"Detail": "Timezone", "Value": "Asia/Manila"},
            {"Detail": "Last System Check", "Value": datetime.now().strftime("%Y-%m-%d %H:%M")},
        ]
    )
    with st.container():
        st.dataframe(details, use_container_width=True, hide_index=True, height=250)


def render_sidebar() -> str:
    st.markdown('<div class="sidebar-title">NPC Agus Decision Support</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-subtitle">RBFNN Forecasting Prototype Panel</div>', unsafe_allow_html=True)
    selected_page = st.radio("Navigation", NAV_ITEMS, label_visibility="collapsed")
    st.markdown(
        f"""
        <div class="sidebar-mini">
            Raw Excel Data: {file_status(RAW_EXCEL_PATH)}<br>
            RBFNN Forecast: {file_status(FORECAST_EXCEL)}
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
elif selected_page == "RBFNN Forecasting":
    rbfnn_forecasting_page()
elif selected_page == "System Information":
    system_information_page()

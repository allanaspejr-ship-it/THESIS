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
        --bg0: #020b18;
        --bg1: #06172f;
        --bg2: #082a57;
        --panel: #062044;
        --panel2: #0a3267;
        --line: rgba(91, 166, 255, .45);
        --line2: rgba(135, 195, 255, .26);
        --blue: #1476ff;
        --blue2: #1f8bff;
        --text: #f4f9ff;
        --muted: #b7d5f4;
        --red: #d20f23;
        --green: #44d487;
    }

    html, body, .stApp {
        background:
            radial-gradient(circle at 75% 0%, rgba(31, 139, 255, .16), transparent 28%),
            linear-gradient(135deg, #020b18 0%, #06172f 48%, #082a57 100%) !important;
        color: var(--text);
    }

    .main .block-container {
        max-width: 1500px;
        padding: 1.05rem 1rem 1.8rem 1rem;
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #020b18 0%, #06172f 58%, #041126 100%);
        border-right: 1px solid rgba(135, 195, 255, .22);
        min-width: 245px !important;
        width: 245px !important;
    }

    section[data-testid="stSidebar"] * {
        color: var(--text) !important;
    }

    section[data-testid="stSidebar"] .stRadio label {
        width: 100%;
        min-height: 42px;
        padding: .52rem .65rem;
        margin: .18rem 0 .38rem 0;
        border-radius: 8px;
        border: 1px solid rgba(135, 195, 255, .20);
        background: rgba(10, 50, 103, .50);
        transition: all .14s ease;
    }

    section[data-testid="stSidebar"] .stRadio label:hover {
        background: rgba(20, 118, 255, .38);
        border-color: rgba(135, 195, 255, .56);
    }

    section[data-testid="stSidebar"] .stRadio label:has(input:checked) {
        background: linear-gradient(135deg, #1476ff, #0a5ed8);
        border-color: rgba(210, 232, 255, .70);
        box-shadow: 0 8px 18px rgba(20, 118, 255, .28);
    }

    .sidebar-title {
        font-weight: 850;
        font-size: 1rem;
        margin: .35rem 0 .12rem;
    }

    .sidebar-subtitle {
        color: var(--muted);
        font-size: .76rem;
        margin-bottom: 1rem;
    }

    .sidebar-mini {
        color: var(--muted);
        font-size: .78rem;
        line-height: 1.8;
        margin-top: 1.4rem;
    }

    div[data-testid="stHorizontalBlock"] {
        gap: 16px;
        margin-bottom: 16px;
    }

    .page-title-block {
        width: 100%;
        background: linear-gradient(90deg, #082b57, #0b3f82);
        border: 1px solid #1e88ff;
        border-radius: 12px;
        padding: 22px 24px;
        margin-bottom: 22px;
        box-shadow: 0 0 12px rgba(30,136,255,0.18);
    }

    .page-title {
        color: #ffffff;
        font-size: 30px;
        font-weight: 800;
        line-height: 1.2;
        margin-bottom: 6px;
    }

    .page-subtitle {
        color: #b9d9ff;
        font-size: 14px;
        font-weight: 500;
    }

    .card {
        background: linear-gradient(180deg, rgba(10, 50, 103, .94), rgba(6, 32, 68, .96));
        border: 1px solid var(--line);
        border-radius: 8px;
        padding: .9rem;
        box-shadow: 0 18px 34px rgba(0, 0, 0, .18);
        margin-bottom: 18px;
    }

    .section-header {
        background: linear-gradient(90deg, #083b78, #0b5eb8);
        border: 1px solid #1e88ff;
        border-radius: 8px;
        padding: 10px 14px;
        margin: 20px 0 10px 0;
        color: #ffffff !important;
        font-size: 15px;
        font-weight: 800;
        box-sizing: border-box;
    }

    .section-header span {
        color: #ffffff !important;
    }

    .info-card, .kpi-card, .status-card,
    .kpi-card-light, .info-card-light, .status-card-light {
        height: 122px;
        background: linear-gradient(180deg, #0b2d5c 0%, #08264d 100%);
        border: 1px solid #2f8fff;
        border-radius: 12px;
        padding: 18px;
        min-height: 120px;
        box-sizing: border-box;
        color: #ffffff !important;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        box-shadow: 0 0 10px rgba(30,136,255,0.12);
        margin-bottom: 16px;
    }

    .kpi-card.red {
        background: linear-gradient(180deg, #d20f23, #8f0818);
        border-color: rgba(255, 170, 180, .55);
    }

    .kpi-card.light {
        background: linear-gradient(180deg, #0b2d5c 0%, #08264d 100%);
        border: 1px solid #2f8fff;
        color: #ffffff !important;
        box-shadow: 0 0 10px rgba(30,136,255,0.12);
    }

    .kpi-card-light h4,
    .kpi-card-light h3,
    .kpi-card-light p,
    .info-card-light h4,
    .info-card-light h3,
    .info-card-light p,
    .status-card-light h4,
    .status-card-light h3,
    .status-card-light p {
        color: #ffffff !important;
    }

    .kpi-label {
        color: #b9d9ff !important;
        font-size: .70rem;
        font-weight: 850;
        letter-spacing: .04em;
        text-transform: uppercase;
    }

    .kpi-value {
        color: #ffffff !important;
        font-size: 1.45rem;
        font-weight: 800;
        line-height: 1.1;
        margin-top: .18rem;
    }

    .kpi-note {
        color: #d6e8ff !important;
        font-size: .77rem;
        line-height: 1.2;
    }

    .instruction {
        background: linear-gradient(180deg, #0b3a78, #062b5b);
        border: 1px solid rgba(91, 166, 255, .72);
        border-radius: 8px;
        padding: .8rem .9rem;
        color: #eaf5ff;
        font-size: .86rem;
        margin-bottom: .8rem;
    }

    .status-label {
        color: #b9d9ff !important;
        font-size: .70rem;
        font-weight: 850;
        text-transform: uppercase;
    }

    .status-value {
        color: #ffffff !important;
        font-size: 1.18rem;
        font-weight: 850;
        margin-top: .25rem;
    }

    .status-note {
        color: #d6e8ff !important;
        font-size: .78rem;
        margin-top: .5rem;
        line-height: 1.28;
    }

    .workflow-card {
        min-height: 150px;
        background: linear-gradient(180deg, #092c5d, #062044);
        border: 1px solid var(--line);
        border-radius: 8px;
        padding: .9rem .75rem;
        text-align: center;
    }

    .workflow-icon {
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
        font-size: .83rem;
        margin-bottom: .25rem;
    }

    .workflow-note {
        color: var(--muted);
        font-size: .72rem;
        line-height: 1.25;
    }

    .data-note {
        color: var(--muted);
        font-size: .84rem;
        line-height: 1.45;
    }

    .note-card {
        background: #dbeeff;
        color: #06264d !important;
        border-left: 5px solid #1e88ff;
        border-radius: 8px;
        padding: 14px 16px;
        margin-top: 8px;
        margin-bottom: 16px;
        line-height: 1.45;
    }

    .note-card * {
        color: #06264d !important;
    }

    .stMarkdown, p, label, span, div {
        color: inherit;
    }

    h1, h2, h3 {
        color: #ffffff !important;
    }

    div.stButton > button,
    div[data-testid="stDownloadButton"] > button {
        height: 42px;
        border-radius: 8px;
        border: 1px solid rgba(135, 195, 255, .60);
        background: linear-gradient(180deg, #1476ff, #0b58cc);
        color: #ffffff;
        font-weight: 800;
        box-shadow: 0 8px 18px rgba(20, 118, 255, .22);
    }

    div.stButton > button:hover,
    div[data-testid="stDownloadButton"] > button:hover {
        background: linear-gradient(180deg, #2b91ff, #1476ff);
        border-color: rgba(225, 242, 255, .90);
        color: #ffffff;
    }

    div[data-testid="stDataFrame"], div[data-testid="stDataEditor"] {
        border-radius: 8px;
        overflow: hidden;
        border: 1px solid var(--line2);
    }

    .stDataFrame, .stDataEditor {
        color: #06172f;
    }

    div[data-testid="stFileUploader"] {
        background: linear-gradient(180deg, #092c5d, #062044);
        border: 1px solid #1e88ff;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 18px;
    }

    div[data-testid="stFileUploader"] label,
    div[data-testid="stFileUploader"] small,
    div[data-testid="stFileUploader"] span,
    div[data-testid="stFileUploader"] p {
        color: #ffffff !important;
    }

    div[data-testid="stFileUploader"] button {
        background: linear-gradient(180deg, #1476ff, #0b58cc) !important;
        color: #ffffff !important;
        border: 1px solid #1e88ff !important;
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


def kpi_card(label: str, value: str, note: str, variant: str = "") -> None:
    st.markdown(
        f"""
        <div class="kpi-card {variant}">
            <div>
                <div class="kpi-label">{label}</div>
                <div class="kpi-value">{value}</div>
            </div>
            <div class="kpi-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def status_card(label: str, path: Path) -> None:
    st.markdown(
        f"""
        <div class="status-card">
            <div class="status-label">{label}</div>
            <div class="status-value">{file_status(path)}</div>
            <div class="status-note">Last modified:<br>{file_timestamp(path)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def instruction_card(text: str) -> None:
    st.markdown(f'<div class="instruction">{text}</div>', unsafe_allow_html=True)


def section_header(title: str) -> None:
    st.markdown(
        f"""
        <div class="section-header">
            <span>{title}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def page_title_block() -> None:
    st.markdown(
        """
        <div class="page-title-block">
            <div class="page-title">
                NPC Agus Cascade Hydropower Generation Forecasting Dashboard
            </div>
            <div class="page-subtitle">
                RBFNN-Based Day-Ahead Forecasting and Planned Outage Management
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def note_card(text: str) -> None:
    st.markdown(f'<div class="note-card">{text}</div>', unsafe_allow_html=True)


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
            return "color: #44d487; font-weight: 800;"
        if value == "OFF":
            return "color: #ff5468; font-weight: 900;"
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
        title=None,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#061a33",
        font=dict(color="#ffffff"),
        legend=dict(font=dict(color="#ffffff")),
        hoverlabel=dict(bgcolor="#061a33", font=dict(color="#ffffff")),
        xaxis=dict(
            title_font=dict(color="#ffffff"),
            tickfont=dict(color="#ffffff"),
            gridcolor="rgba(255,255,255,0.15)",
            zerolinecolor="rgba(255,255,255,0.25)",
        ),
        yaxis=dict(
            title_font=dict(color="#ffffff"),
            tickfont=dict(color="#ffffff"),
            gridcolor="rgba(255,255,255,0.15)",
            zerolinecolor="rgba(255,255,255,0.25)",
        ),
        margin=dict(l=40, r=20, t=20, b=40),
    )
    return fig


def render_hourly_line_chart(forecast: pd.DataFrame, height: int = 340) -> None:
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
        color_discrete_sequence=["#00a2ff", "#2f80ed", "#4dabf7", "#74c0fc", "#a5d8ff", "#ff4d5e"],
    )
    fig.update_traces(line=dict(width=2.6), marker=dict(size=5))
    fig.update_layout(
        template="plotly_dark",
        height=height,
        margin=dict(l=36, r=16, t=20, b=34),
        font=dict(color="white", size=11),
        legend=dict(font=dict(size=10, color="white"), bgcolor="rgba(0,0,0,0)"),
        xaxis_title="Forecast Hour",
        yaxis_title="Generation (MW)",
        hovermode="x unified",
    )
    fig = style_plotly_chart(fig)
    st.plotly_chart(fig, width="stretch")


def render_daily_bar_chart(forecast: pd.DataFrame, height: int = 340) -> None:
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
        color_discrete_sequence=["#1476ff", "#2f80ed", "#1c7ed6", "#339af0", "#74c0fc", "#4dabf7"],
    )
    fig.update_traces(texttemplate="%{text:,.2f}", textposition="outside", marker_line_width=0)
    fig.update_layout(
        template="plotly_dark",
        height=height,
        margin=dict(l=40, r=16, t=20, b=34),
        font=dict(color="white", size=11),
        showlegend=False,
        xaxis_title="",
        yaxis_title="Generation (MWh)",
    )
    fig = style_plotly_chart(fig)
    st.plotly_chart(fig, width="stretch")


def dashboard_overview() -> None:
    page_title_block()
    refresh_cols = st.columns([5, 1.25])
    with refresh_cols[1]:
        if st.button("Refresh Data From Disk", width="stretch"):
            clear_cached_data("Data refreshed from disk.")

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

    cols = st.columns(5)
    with cols[0]:
        kpi_card("Total Cascade Forecast", kpis["cascade_total"], "24-hour cascade sum")
    with cols[1]:
        kpi_card("Peak Cascade Output", kpis["peak"], "Maximum hourly output")
    with cols[2]:
        kpi_card("Unavailable Unit-Hours", f"{outage_info['unavailable']:,}", "Planned OFF entries", "red" if outage_info["unavailable"] else "")
    with cols[3]:
        kpi_card("Latest Cleaned Timestamp", cleaned_dt.max().strftime("%Y-%m-%d %H:%M") if cleaned_dt.notna().any() else "N/A", "From cleaned data")
    with cols[4]:
        kpi_card("Latest Forecast Timestamp", file_timestamp(FORECAST_EXCEL), "Forecast file modified")

    if forecast is not None:
        chart_cols = st.columns(2)
        with chart_cols[0]:
            section_header("Total Generation per Agus Plant Hourly")
            render_hourly_line_chart(forecast)
        with chart_cols[1]:
            section_header("Total Generation per Agus Plant Daily Total")
            render_daily_bar_chart(forecast)

    section_header("Recent Activity and Status")
    status_cols = st.columns(4)
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
    page_title_block()
    section_header("Data Management")
    note_card("Update raw Excel data, refresh files, and clean hourly dataset.")

    section_header("Data Actions")
    action_cols = st.columns([1.35, 1, 1])
    with action_cols[0]:
        uploaded = st.file_uploader("Upload Excel File Optional", type=["xlsx", "xls"], label_visibility="visible")
        if uploaded is not None:
            DATA_DIR.mkdir(parents=True, exist_ok=True)
            RAW_EXCEL_PATH.write_bytes(uploaded.getbuffer())
            clear_cached_data("Uploaded Excel file saved.")
    with action_cols[1]:
        st.write("")
        st.write("")
        if st.button("Refresh Data From Disk", width="stretch"):
            clear_cached_data("Data refreshed from disk.")
    with action_cols[2]:
        st.write("")
        st.write("")
        if st.button("Run Data Cleaning", width="stretch"):
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
    section_header("Data Confirmation")
    confirm_cols_1 = st.columns(4)
    with confirm_cols_1[0]:
        kpi_card("Raw Excel Status", file_status(RAW_EXCEL_PATH), "Source workbook")
    with confirm_cols_1[1]:
        kpi_card("Raw Excel Last Modified", file_timestamp(RAW_EXCEL_PATH), "File timestamp")
    with confirm_cols_1[2]:
        kpi_card("Latest Raw Data Date", latest_date_from_df(raw_df), "Detected from Date/Time")
    with confirm_cols_1[3]:
        kpi_card("Cleaned Data Status", file_status(cleaned_source_path()), "Cleaned output")

    confirm_cols_2 = st.columns(4)
    with confirm_cols_2[0]:
        kpi_card("Cleaned Data Last Modified", file_timestamp(cleaned_source_path()), "File timestamp")
    with confirm_cols_2[1]:
        kpi_card("Latest Cleaned Data Date", cleaned_dt.max().strftime("%Y-%m-%d %H:%M") if cleaned_dt.notna().any() else "N/A", "Detected from date/time")
    with confirm_cols_2[2]:
        kpi_card("Total Cleaned Rows", f"{len(cleaned_df):,}" if cleaned_df is not None else "N/A", "Hourly records")
    with confirm_cols_2[3]:
        kpi_card("Total Cleaned Columns", f"{len(cleaned_df.columns):,}" if cleaned_df is not None else "N/A", "Prepared features")

    section_header("Data Notes")
    note_card("Uploading replaces the raw Excel workbook used by the cleaning script. Refreshing reloads the latest files from disk without restarting Streamlit. Running data cleaning updates the cleaned hourly dataset used by the forecasting workflow.")


def planned_outage_page() -> None:
    page_title_block()
    section_header("Planned Outage Planning")
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
    kpi_cols = st.columns(4)
    with kpi_cols[0]:
        kpi_card("Unavailable Unit-Hours", f"{summary['unavailable']:,}", "Cells marked 0", "red" if summary["unavailable"] else "")
    with kpi_cols[1]:
        kpi_card("Available Unit-Hours", f"{summary['available']:,}", "Cells marked 1")
    with kpi_cols[2]:
        affected = ", ".join(summary["affected_plants"]) if summary["affected_plants"] else "None"
        kpi_card("Affected Plants", str(len(summary["affected_plants"])), affected, "light")
    with kpi_cols[3]:
        kpi_card("Outage Plan Last Modified", file_timestamp(OUTAGE_PLAN_PATH), "Saved outage workbook", "light")

    column_config = {
        col: st.column_config.NumberColumn(col, min_value=0, max_value=1, step=1, format="%d")
        for col in outage_cols
    }
    edited = st.data_editor(
        editable,
        width="stretch",
        height=350,
        hide_index=True,
        column_config=column_config,
        disabled=[col for col in editable.columns if col not in outage_cols],
    )

    button_cols = st.columns(3)
    with button_cols[0]:
        if st.button("Save Outage Plan", width="stretch"):
            try:
                save_outage_plan(edited)
                st.session_state["outage_saved"] = True
                st.rerun()
            except Exception as exc:
                st.error(str(exc))
    with button_cols[1]:
        if st.button("Forecast Day-Ahead", width="stretch"):
            with st.spinner("Running day-ahead RBFNN forecast..."):
                try:
                    save_outage_plan(edited)
                    st.cache_data.clear()
                    load_outage_plan(modified_ns(OUTAGE_PLAN_PATH))
                    show_script_result(run_script(RBFNN_SCRIPT), "Day-ahead forecast")
                    st.cache_data.clear()
                    load_forecast(modified_ns(FORECAST_EXCEL))
                    st.success("Forecast updated using the saved hourly outage plan.")
                except Exception as exc:
                    st.error(str(exc))
    with button_cols[2]:
        if st.button("Retrain RBFNN Model", width="stretch"):
            with st.spinner("Retraining RBFNN model..."):
                try:
                    show_script_result(run_script(RBFNN_SCRIPT, ["--train"]), "RBFNN retraining")
                except Exception as exc:
                    st.error(str(exc))

    section_header("ON/OFF Status View")
    status_view = outage_status_frame(edited)
    if not status_view.empty:
        st.dataframe(style_outage_status(status_view), width="stretch", height=255)
    if summary["affected_units"]:
        note_card(f'Affected Units: {", ".join(summary["affected_units"])}')


def rbfnn_forecasting_page() -> None:
    page_title_block()
    refresh_cols = st.columns([5, 1.25])
    with refresh_cols[1]:
        if st.button("Refresh Forecast", width="stretch"):
            clear_cached_data("Forecast refreshed from disk.")

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
    kpi_cols = st.columns(4)
    with kpi_cols[0]:
        kpi_card("Total Cascade Forecast", kpis["cascade_total"], "24-hour cascade sum")
    with kpi_cols[1]:
        kpi_card("Peak Cascade Output", kpis["peak"], "Maximum hourly output")
    with kpi_cols[2]:
        kpi_card("Affected Plants", kpis["affected"], "Saved outage plan")
    with kpi_cols[3]:
        kpi_card("Forecast Generated", kpis["forecast_generated"], "Forecast file modified")

    section_header("Forecast Table")
    st.dataframe(display_forecast_table(forecast), width="stretch", height=285)
    dl_cols = st.columns(2)
    with dl_cols[0]:
        excel_bytes = read_file_bytes(FORECAST_EXCEL)
        if excel_bytes:
            st.download_button(
                "Download Forecast Excel",
                data=excel_bytes,
                file_name="Day_Ahead_24H_RBFNN_Forecast.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                width="stretch",
            )
    with dl_cols[1]:
        csv_bytes = read_file_bytes(FORECAST_CSV) or forecast.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download Forecast CSV",
            data=csv_bytes,
            file_name="Day_Ahead_24H_RBFNN_Forecast.csv",
            mime="text/csv",
            width="stretch",
        )

    chart_cols = st.columns(2)
    with chart_cols[0]:
        section_header("Total Generation per Agus Plant Hourly")
        render_hourly_line_chart(forecast)
    with chart_cols[1]:
        section_header("Total Generation per Agus Plant Daily Total")
        render_daily_bar_chart(forecast)


def system_information_page() -> None:
    page_title_block()

    section_header("Forecasting Workflow")
    workflow = [
        ("1", "Update Raw Data", "Upload or refresh Excel data"),
        ("2", "Clean Data", "Prepare hourly dataset"),
        ("3", "Edit Outage Plan", "Set unit availability"),
        ("4", "Forecast / Retrain", "Generate or retrain RBFNN"),
        ("5", "View Forecast", "Review results and download"),
    ]
    cols = st.columns(5)
    for col, (num, title, note) in zip(cols, workflow):
        with col:
            st.markdown(
                f"""
                <div class="workflow-card">
                    <div class="workflow-icon">{num}</div>
                    <div class="workflow-title">{title}</div>
                    <div class="workflow-note">{note}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    section_header("File Status Overview")
    status_cols = st.columns(4)
    status_items = [
        ("Raw Excel Data", RAW_EXCEL_PATH),
        ("Cleaned Data", cleaned_source_path()),
        ("Outage Plan", OUTAGE_PLAN_PATH),
        ("RBFNN Forecast", FORECAST_EXCEL),
    ]
    for col, (label, path) in zip(status_cols, status_items):
        with col:
            status_card(label, path)

    section_header("System Details")
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
    st.dataframe(details, width="stretch", hide_index=True)


with st.sidebar:
    st.markdown('<div class="sidebar-title">NPC Agus Operations</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-subtitle">RBFNN forecasting control panel</div>', unsafe_allow_html=True)
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

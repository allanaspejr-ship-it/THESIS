"""
Optimized Cell 1

Creates isolated optimized inputs from the already-cleaned baseline outputs.
The original pipeline under data/outputs is read only and remains untouched.
"""

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[2]
OPT_DIR = PROJECT_DIR / "optimized_version"
SRC_RUNTIME = PROJECT_DIR / "data" / "outputs" / "01_runtime_outputs"

DATA_DIR = OPT_DIR / "data"
OUT_DIR = OPT_DIR / "outputs"
CLEANED_DATA_DIR = OUT_DIR / "cleaned_data"
OUTAGES_DIR = OUT_DIR / "outages_planning"
META_DIR = OPT_DIR / "metadata"
OVERALL_METRICS_DIR = META_DIR / "overall_metrics"
DATA_DIR.mkdir(parents=True, exist_ok=True)
CLEANED_DATA_DIR.mkdir(parents=True, exist_ok=True)
OUTAGES_DIR.mkdir(parents=True, exist_ok=True)
OVERALL_METRICS_DIR.mkdir(parents=True, exist_ok=True)

PLANTS = ["agus1", "agus2", "agus4", "agus5", "agus6", "agus7"]


def main():
    clean_src = SRC_RUNTIME / "cleaned_hourly_data.parquet"
    clean_xlsx_src = SRC_RUNTIME / "cleaned_hourly_data.xlsx"
    planned_src = SRC_RUNTIME / "Planned_Outages_Input.xlsx"

    if not clean_src.exists():
        raise FileNotFoundError(f"Missing baseline cleaned parquet: {clean_src}")
    if not planned_src.exists():
        raise FileNotFoundError(f"Missing baseline planned outage file: {planned_src}")

    df = pd.read_parquet(clean_src)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)

    required = ["date", "time", "datetime"] + [f"total_gen_{p}" for p in PLANTS]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Cleaned data is missing required columns: {missing}")

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if df[numeric_cols].isna().any().any():
        raise ValueError("Cleaned data contains NaNs in numeric columns.")

    clean_out = CLEANED_DATA_DIR / "cleaned_hourly_data.parquet"
    clean_xlsx_out = CLEANED_DATA_DIR / "cleaned_hourly_data.xlsx"
    planned_out = OUTAGES_DIR / "Planned_Outages_Input.xlsx"

    df.to_parquet(clean_out, index=False)
    if clean_xlsx_src.exists():
        shutil.copy2(clean_xlsx_src, clean_xlsx_out)
    else:
        df.to_excel(clean_xlsx_out, index=False)
    shutil.copy2(planned_src, planned_out)

    metadata = {
        "source_cleaned_parquet": str(clean_src),
        "optimized_cleaned_parquet": str(clean_out),
        "rows": int(len(df)),
        "latest_timestamp": str(df["datetime"].max()),
        "forecast_start": str(df["datetime"].max() + pd.Timedelta(hours=1)),
        "plants": PLANTS,
        "note": (
            "Optimized Cell 1 isolates inputs for optimized training. "
            "The baseline cleaned outputs are not overwritten."
        ),
    }
    (OVERALL_METRICS_DIR / "optimized_cell1_metadata.json").write_text(json.dumps(metadata, indent=2))

    print("Optimized Cell 1 complete")
    print("Saved:", clean_out)
    print("Saved:", planned_out)
    print("Latest timestamp:", df["datetime"].max())


if __name__ == "__main__":
    main()

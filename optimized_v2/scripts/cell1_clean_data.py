# ============================================================
# CELL 1 — VS CODE VERSION
# SETUP + LOAD + CLEAN + RESAMPLE TO HOURLY + SAVE CLEAN DATA
# + PLANNED OUTAGE TEMPLATE
# ============================================================

import os
import re
import json
import warnings
import numpy as np
import pandas as pd

from pathlib import Path
from sklearn.impute import KNNImputer

warnings.filterwarnings("ignore")
np.random.seed(42)

# ============================================================
# PATH CONFIG — VS CODE LOCAL PATHS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parents[2]
VERSION_DIR = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = VERSION_DIR / "data"

CONFIG = {
    "input_file": RAW_DATA_DIR / "DATA(JAN2024-JUNE2025).xlsx",
    "base_output_dir": VERSION_DIR / "outputs",

    "plants": ["agus1", "agus2", "agus4", "agus5", "agus6", "agus7"],

    "approx_zero_threshold": 0.05,
    "short_zero_run_max": 2,
    "long_zero_run_min": 3,
    "short_missing_gap_max": 2,
    "spike_factor": 3.5,
    "knn_neighbors": 5,

    "dirs": {
        "runtime": "cleaned_data",
        "outages": "outages_planning",
    }
}

DIRS = {}
for k, v in CONFIG["dirs"].items():
    path = CONFIG["base_output_dir"] / v
    path.mkdir(parents=True, exist_ok=True)
    DIRS[k] = path


# ============================================================
# HELPERS
# ============================================================

def sanitize_name(x):
    if pd.isna(x):
        return None
    x = str(x).strip().lower()
    x = re.sub(r"\s+", "_", x)
    x = x.replace("-", "_").replace("/", "_")
    x = x.replace("(", "").replace(")", "")
    x = x.replace(".", "")
    x = x.replace(":", "")
    return x if x else None


def flatten_three_row_header(raw_df):
    h0 = raw_df.iloc[0].ffill()
    h1 = raw_df.iloc[1].ffill()
    h2 = raw_df.iloc[2]

    cols = []
    for a, b, c in zip(h0, h1, h2):
        a = sanitize_name(a)
        b = sanitize_name(b)
        c = sanitize_name(c)

        if a == "date" or b == "date" or c == "date":
            cols.append("date")
            continue
        if a == "time" or b == "time" or c == "time":
            cols.append("time")
            continue

        parts = [p for p in [a, b, c] if p is not None and p != "nan"]
        cols.append("_".join(parts) if parts else "unnamed")

    seen = {}
    final = []
    for c in cols:
        if c not in seen:
            seen[c] = 0
            final.append(c)
        else:
            seen[c] += 1
            final.append(f"{c}_{seen[c]}")
    return final


def standardize_columns(df):
    renamed = {}
    for c in df.columns:
        nc = c

        nc = nc.replace("generated_mw_", "")
        nc = nc.replace("outage_of_plants_", "")
        nc = nc.replace("spillage_", "")
        nc = nc.replace("spillway_gate_opening_in_meters_", "")
        nc = nc.replace("total_gate_open_per_agus_in_meters_", "")
        nc = nc.replace("elevation_", "")

        nc = nc.replace("gen_agus_1_", "gen_agus1_")
        nc = nc.replace("gen_agus_2_", "gen_agus2_")
        nc = nc.replace("gen_agus_4_", "gen_agus4_")
        nc = nc.replace("gen_agus_5_", "gen_agus5_")
        nc = nc.replace("gen_agus_6_", "gen_agus6_")
        nc = nc.replace("gen_agus_7_", "gen_agus7_")

        nc = nc.replace("out_agus_1_", "out_agus1_")
        nc = nc.replace("out_agus_2_", "out_agus2_")
        nc = nc.replace("out_agus_4_", "out_agus4_")
        nc = nc.replace("out_agus_5_", "out_agus5_")
        nc = nc.replace("out_agus_6_", "out_agus6_")
        nc = nc.replace("out_agus_7_", "out_agus7_")

        nc = nc.replace("spill_agus_1_", "spill_agus1_")
        nc = nc.replace("spill_agus_2_", "spill_agus2_")
        nc = nc.replace("spill_agus_4_", "spill_agus4_")
        nc = nc.replace("spill_agus_5_", "spill_agus5_")
        nc = nc.replace("spill_agus_6_", "spill_agus6_")
        nc = nc.replace("spill_agus_7_", "spill_agus7_")

        nc = nc.replace("tot_agus_1", "tot_agus1")
        nc = nc.replace("tot_agus_2", "tot_agus2")
        nc = nc.replace("tot_agus_4", "tot_agus4")
        nc = nc.replace("tot_agus_5", "tot_agus5")
        nc = nc.replace("tot_agus_6", "tot_agus6")
        nc = nc.replace("tot_agus_7", "tot_agus7")

        nc = nc.replace("elev_agus_1", "elev_agus1")
        nc = nc.replace("elev_agus_2", "elev_agus2")
        nc = nc.replace("elev_agus_4", "elev_agus4")
        nc = nc.replace("elev_agus_5", "elev_agus5")
        nc = nc.replace("elev_agus_6", "elev_agus6")
        nc = nc.replace("elev_agus_7", "elev_agus7")

        nc = re.sub(r"^out_(agus[124567])_\1_(unit\d+)$", r"out_\1_\2", nc)
        nc = re.sub(r"^gen_(agus[124567])_\1_(unit\d+)$", r"gen_\1_\2", nc)

        nc = nc.replace("gen_agus1_total_gen_agus1", "total_gen_agus1")
        nc = nc.replace("gen_agus2_total_gen_agus2", "total_gen_agus2")
        nc = nc.replace("gen_agus4_total_gen_agus4", "total_gen_agus4")
        nc = nc.replace("gen_agus5_total_gen_agus5", "total_gen_agus5")
        nc = nc.replace("gen_agus6_total_gen_agus6", "total_gen_agus6")
        nc = nc.replace("gen_agus7_total_gen_agus7", "total_gen_agus7")

        renamed[c] = nc

    return df.rename(columns=renamed)


def build_datetime(df):
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["time"] = pd.to_numeric(df["time"], errors="coerce")
    df["hour0"] = df["time"] - 1
    df["datetime"] = df["date"] + pd.to_timedelta(df["hour0"], unit="h")
    df = df.drop(columns=["hour0"])
    return df.sort_values("datetime").reset_index(drop=True)


def run_lengths(mask):
    arr = np.asarray(mask).astype(int)
    runs = []
    i, n = 0, len(arr)

    while i < n:
        if arr[i] == 1:
            j = i
            while j < n and arr[j] == 1:
                j += 1
            runs.append((i, j - 1, j - i))
            i = j
        else:
            i += 1

    return runs


def interpolate_short_missing(s, max_gap):
    return s.interpolate(method="linear", limit=max_gap, limit_direction="both")


def interpolate_short_zero_runs(s, short_max, long_min):
    s = s.copy()
    zero_mask = s.fillna(np.nan).eq(0)

    for start, end, length in run_lengths(zero_mask):
        if 1 <= length <= short_max:
            s.iloc[start:end + 1] = np.nan
        elif length >= long_min:
            s.iloc[start:end + 1] = 0.0

    return s.interpolate(method="linear", limit_direction="both")


def smooth_spikes(s, factor):
    s = s.copy()

    for i in range(1, len(s) - 1):
        prev_v, curr_v, next_v = s.iloc[i - 1], s.iloc[i], s.iloc[i + 1]

        if pd.notna(prev_v) and pd.notna(curr_v) and pd.notna(next_v):
            local_mean = (prev_v + next_v) / 2.0
            local_diff = abs(prev_v - next_v) + 1e-6

            if abs(curr_v - local_mean) > factor * max(local_diff, 1.0):
                s.iloc[i] = local_mean

    return s


def plant_from_col(c):
    m = re.search(r"(agus[124567])", c)
    return m.group(1) if m else None


def unit_from_col(c):
    m = re.search(r"(unit\d+)", c)
    return m.group(1) if m else None


def final_clean_column_renames():
    renames = {
        "lake_lanao_hourly_outflow_elev_agus7": "lake_lanao_outflow",
    }

    for plant in CONFIG["plants"]:
        renames[f"tot_{plant}"] = f"tot_{plant}_gate"

    for plant in CONFIG["plants"]:
        for unit in range(1, 6):
            renames[f"gen_{plant}_gen_{plant}_unit{unit}"] = f"gen_{plant}_unit{unit}"

    return renames


def prepare_cleaned_output(clean_df):
    # Public cleaned files no longer expose datetime, but model scripts rebuild it from date + time.
    out = clean_df.rename(columns=final_clean_column_renames()).copy()
    out["date"] = pd.to_datetime(out["date"]).dt.strftime("%Y-%m-%d")
    out = out.drop(columns=["datetime"], errors="ignore")
    return out


def display_hour(timestamp):
    hour = pd.Timestamp(timestamp).hour
    return "00:00" if hour == 0 else f"{hour}:00"


def format_excel(path):
    try:
        from openpyxl import load_workbook
    except ImportError:
        return

    wb = load_workbook(path)
    ws = wb.active
    ws.freeze_panes = "A2"
    ws.auto_filter.ref = ws.dimensions

    for column_cells in ws.columns:
        header = str(column_cells[0].value or "")
        max_len = max([len(header)] + [len(str(cell.value)) for cell in column_cells[1:50] if cell.value is not None])
        ws.column_dimensions[column_cells[0].column_letter].width = min(max(max_len + 2, 10), 32)

    wb.save(path)


def resolve_input_file():
    configured = CONFIG["input_file"]
    if configured.exists():
        return configured

    candidates = sorted(
        [
            path for path in RAW_DATA_DIR.glob("*.xlsx")
            if not path.name.startswith("~$")
        ]
    )
    if candidates:
        return candidates[0]

    raise FileNotFoundError(
        f"\nInput Excel file not found:\n{configured}\n\n"
        f"Put the raw source Excel file in:\n{RAW_DATA_DIR}"
    )


# ============================================================
# MAIN
# ============================================================

def main():
    input_file = resolve_input_file()

    print("Loading raw Excel file:")
    print(input_file)

    raw = pd.read_excel(input_file, sheet_name=0, header=None)
    cols = flatten_three_row_header(raw.iloc[:3])

    df = raw.iloc[3:].copy().reset_index(drop=True)
    df.columns = cols
    df = standardize_columns(df)

    for c in df.columns:
        if c not in ["date", "time"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df = build_datetime(df)

    gen_unit_cols = [c for c in df.columns if c.startswith("gen_agus") and "unit" in c]
    out_unit_cols = [c for c in df.columns if c.startswith("out_agus") and "unit" in c]
    spill_cols = [c for c in df.columns if c.startswith("spill_agus") and "gate" in c]
    tot_gate_cols = [c for c in df.columns if c.startswith("tot_agus")]
    elev_cols = [c for c in df.columns if c.startswith("elev_agus")]
    rain_cols = [c for c in df.columns if c == "rainfall"]
    outflow_cols = [c for c in df.columns if "outflow" in c]

    print("Generation unit cols:", len(gen_unit_cols))
    print("Outage unit cols:", len(out_unit_cols))
    print("Spill cols:", len(spill_cols))
    print("Total gate cols:", len(tot_gate_cols))
    print("Elevation cols:", len(elev_cols))
    print("Rain cols:", rain_cols)
    print("Outflow cols:", outflow_cols)

    for c in gen_unit_cols:
        s = df[c].astype(float).copy()
        s = s.mask(s <= CONFIG["approx_zero_threshold"], 0.0)
        s = smooth_spikes(s, CONFIG["spike_factor"])
        s = interpolate_short_missing(s, CONFIG["short_missing_gap_max"])
        s = interpolate_short_zero_runs(
            s,
            CONFIG["short_zero_run_max"],
            CONFIG["long_zero_run_min"]
        )
        s = s.mask(s <= CONFIG["approx_zero_threshold"], 0.0)
        df[c] = s

    for c in spill_cols + tot_gate_cols:
        s = df[c].astype(float).copy()
        s = s.mask(s < 0, 0.0)
        s = smooth_spikes(s, CONFIG["spike_factor"])
        s = interpolate_short_missing(s, CONFIG["short_missing_gap_max"])
        s = interpolate_short_zero_runs(
            s,
            CONFIG["short_zero_run_max"],
            CONFIG["long_zero_run_min"]
        )
        df[c] = s

    for c in elev_cols + rain_cols + outflow_cols:
        s = df[c].astype(float).copy()
        if c in rain_cols:
            s = s.mask(s < 0, 0.0)
        s = smooth_spikes(s, CONFIG["spike_factor"])
        s = interpolate_short_missing(s, CONFIG["short_missing_gap_max"])
        df[c] = s

    num_cols = [c for c in df.columns if c not in ["date", "time", "datetime"]]

    if df[num_cols].isna().sum().sum() > 0:
        imputer = KNNImputer(
            n_neighbors=CONFIG["knn_neighbors"],
            weights="distance"
        )
        df[num_cols] = imputer.fit_transform(df[num_cols])

    for p in CONFIG["plants"]:
        unit_cols = [c for c in gen_unit_cols if plant_from_col(c) == p]
        if unit_cols:
            df[f"total_gen_{p}"] = df[unit_cols].sum(axis=1)

    for gcol in gen_unit_cols:
        p = plant_from_col(gcol)
        u = unit_from_col(gcol)

        if p is None or u is None:
            continue

        match = [
            c for c in out_unit_cols
            if plant_from_col(c) == p and unit_from_col(c) == u
        ]

        if len(match) == 1:
            df[match[0]] = np.where(df[gcol] > 0, 1, 0)

    for c in out_unit_cols:
        df[c] = np.where(
            pd.to_numeric(df[c], errors="coerce").fillna(0) > 0,
            1,
            0
        )

    dfh = df.set_index("datetime").sort_index()
    num_cols_h = dfh.select_dtypes(include=[np.number]).columns.tolist()
    dfh = dfh[num_cols_h].resample("h").mean()

    for c in out_unit_cols:
        if c in dfh.columns:
            dfh[c] = np.where(dfh[c] >= 0.5, 1, 0)

    for p in CONFIG["plants"]:
        unit_cols = [
            c for c in gen_unit_cols
            if plant_from_col(c) == p and c in dfh.columns
        ]

        if unit_cols:
            dfh[f"total_gen_{p}"] = dfh[unit_cols].sum(axis=1)

    dfh["date"] = pd.to_datetime(dfh.index.date)
    dfh["time"] = dfh.index.hour + 1
    dfh["datetime"] = dfh.index
    dfh = dfh.reset_index(drop=True)

    keep_cols = ["date", "time", "datetime"]
    keep_cols += [c for c in dfh.columns if c.startswith("gen_agus") and "unit" in c]
    keep_cols += [c for c in dfh.columns if c.startswith("total_gen_")]
    keep_cols += [c for c in dfh.columns if c.startswith("out_agus") and "unit" in c]
    keep_cols += [c for c in dfh.columns if c.startswith("spill_agus") and "gate" in c]
    keep_cols += [c for c in dfh.columns if c.startswith("tot_agus")]
    keep_cols += [c for c in dfh.columns if c.startswith("elev_agus")]
    keep_cols += [c for c in dfh.columns if c == "rainfall" or "outflow" in c]

    keep_cols = list(dict.fromkeys([c for c in keep_cols if c in dfh.columns]))
    clean_df = dfh[keep_cols].copy()

    cleaned_xlsx = DIRS["runtime"] / "cleaned_hourly_data.xlsx"
    cleaned_parquet = DIRS["runtime"] / "cleaned_hourly_data.parquet"
    meta_json = DIRS["runtime"] / "cell1_metadata.json"

    cleaned_output_df = prepare_cleaned_output(clean_df)
    cleaned_output_df.to_excel(cleaned_xlsx, index=False)
    cleaned_output_df.to_parquet(cleaned_parquet, index=False)
    format_excel(cleaned_xlsx)

    with open(meta_json, "w") as f:
        json.dump(
            {
                "latest_timestamp": str(clean_df["datetime"].max()),
                "rows": int(len(cleaned_output_df)),
                "columns": list(cleaned_output_df.columns),
                "note": "Saved cleaned files omit datetime; downstream scripts rebuild it from date and time.",
            },
            f,
            indent=2
        )

    print("\nDone: Cell 1 cleaning")
    print("Saved Excel:", cleaned_xlsx)
    print("Saved Parquet:", cleaned_parquet)
    print("Saved Metadata:", meta_json)
    print("Latest timestamp:", clean_df["datetime"].max())
    print("Columns kept:", len(clean_df.columns))

    # ========================================================
    # PLANNED OUTAGE TEMPLATE
    # ========================================================

    df_outage = clean_df.copy()
    df_outage["datetime"] = pd.to_datetime(df_outage["datetime"])
    df_outage = df_outage.sort_values("datetime").reset_index(drop=True)

    out_cols = [
        c for c in df_outage.columns
        if c.startswith("out_agus") and "unit" in c
    ]

    def sort_key(col):
        m = re.search(r"out_(agus\d+)_unit(\d+)", col)
        if m:
            plant = m.group(1)
            unit = int(m.group(2))
            return plant, unit
        return col, 999

    out_cols = sorted(out_cols, key=sort_key)

    last_ts = df_outage["datetime"].max()

    future_idx = pd.date_range(
        start=last_ts + pd.Timedelta(hours=1),
        periods=24,
        freq="h"
    )

    planned = pd.DataFrame({
        "Date": future_idx.strftime("%Y-%m-%d"),
        "Hour": [display_hour(ts) for ts in future_idx],
    })

    last_row = df_outage.iloc[-1]

    for c in out_cols:
        planned[c] = int(last_row[c]) if c in df_outage.columns and pd.notna(last_row[c]) else 1

    outage_path = DIRS["outages"] / "Planned_Outages_Input.xlsx"
    planned.to_excel(outage_path, index=False)
    format_excel(outage_path)

    print("\nDone: Planned outage template")
    print("Saved:", outage_path)
    print("\nCleaned data preview:")
    print(clean_df.head())

    print("\nPlanned outage preview:")
    print(planned.head())


if __name__ == "__main__":
    main()

# ============================================================
# CELL 3 — VS CODE VERSION
# BENCHMARK FORECASTS + VALIDATION/TESTING METRICS
# - Random Forest
# - XGBoost
# ============================================================

import re
import math
import numpy as np
import pandas as pd
import joblib

from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

# ============================================================
# VS CODE LOCAL PROJECT PATH
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parents[1]

CONFIG = {
    "base_output_dir": Path(r"C:\Users\Allen Mae\Desktop\HYDRO_FORECASTING\data\outputs"),

    "runtime_dir": "01_runtime_outputs",
    "benchmark_dir": "04_benchmark_outputs",

    "model_root_dir": "02_models",
    "rf_model_dir": "random_forest",
    "xgb_model_dir": "xgboost",

    "plants": ["agus1", "agus2", "agus4", "agus5", "agus6", "agus7"],
    "lags": [1, 2, 3, 6, 12, 24, 48],
    "rolling_window": 24,

    "train_ratio": 0.70,
    "val_ratio": 0.15,

    "rf_n_estimators": 300,
    "rf_random_state": 42,

    "xgb_n_estimators": 300,
    "xgb_learning_rate": 0.05,
    "xgb_max_depth": 6,
    "xgb_subsample": 0.9,
    "xgb_colsample_bytree": 0.9,
    "xgb_random_state": 42,

    "retrain_benchmarks": False,

    "unit_capacity_weights": {
        "agus1": {
            "gen_agus1_unit1": 40 / 80,
            "gen_agus1_unit2": 40 / 80,
        },
        "agus2": {
            "gen_agus2_unit1": 60 / 180,
            "gen_agus2_unit2": 60 / 180,
            "gen_agus2_unit3": 60 / 180,
        },
        "agus4": {
            "gen_agus4_unit1": 52.7 / (52.7 * 3),
            "gen_agus4_unit2": 52.7 / (52.7 * 3),
            "gen_agus4_unit3": 52.7 / (52.7 * 3),
        },
        "agus5": {
            "gen_agus5_unit1": 27.5 / 55,
            "gen_agus5_unit2": 27.5 / 55,
        },
        "agus6": {
            "gen_agus6_unit1": 34.5 / 219,
            "gen_agus6_unit2": 34.5 / 219,
            "gen_agus6_unit3": 50 / 219,
            "gen_agus6_unit4": 50 / 219,
            "gen_agus6_unit5": 50 / 219,
        },
        "agus7": {
            "gen_agus7_unit1": 27 / 54,
            "gen_agus7_unit2": 27 / 54,
        },
    },
}

# ============================================================
# DIRECTORIES
# ============================================================

RUNTIME_DIR = CONFIG["base_output_dir"] / CONFIG["runtime_dir"]
BENCH_DIR = CONFIG["base_output_dir"] / CONFIG["benchmark_dir"]

MODEL_ROOT = CONFIG["base_output_dir"] / CONFIG["model_root_dir"]
RF_DIR = MODEL_ROOT / CONFIG["rf_model_dir"]
XGB_DIR = MODEL_ROOT / CONFIG["xgb_model_dir"]

METRICS_DIR = BENCH_DIR / "validation_metrics"
VAL_DAILY_DIR = BENCH_DIR / "validation_daily_metrics"
TEST_DAILY_DIR = BENCH_DIR / "testing_daily_metrics"

for folder in [
    BENCH_DIR,
    RF_DIR,
    XGB_DIR,
    METRICS_DIR,
    VAL_DAILY_DIR,
    TEST_DAILY_DIR,
]:
    folder.mkdir(parents=True, exist_ok=True)

CLEAN_PATH = RUNTIME_DIR / "cleaned_hourly_data.parquet"
PLANNED_PATH = RUNTIME_DIR / "Planned_Outages_Input.xlsx"


# ============================================================
# HELPERS
# ============================================================

def chronological_split(data, train_ratio=0.70, val_ratio=0.15):
    n = len(data)
    i1 = int(n * train_ratio)
    i2 = int(n * (train_ratio + val_ratio))

    train_df = data.iloc[:i1].copy()
    val_df = data.iloc[i1:i2].copy()
    test_df = data.iloc[i2:].copy()

    return train_df, val_df, test_df


def mape_safe(y_true, y_pred, eps=1e-6):
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)

    denom = np.where(np.abs(y_true) < eps, eps, np.abs(y_true))
    return np.mean(np.abs((y_true - y_pred) / denom)) * 100.0


def compute_metrics(y_true, y_pred):
    return {
        "mape": mape_safe(y_true, y_pred),
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": math.sqrt(mean_squared_error(y_true, y_pred)),
        "r2": r2_score(y_true, y_pred),
    }


def save_daily_metrics(df_part, y_true, y_pred, output_path):
    temp = df_part.copy()
    temp["actual"] = np.array(y_true).flatten()
    temp["predicted"] = np.array(y_pred).flatten()
    temp["date_only"] = pd.to_datetime(temp["datetime"]).dt.date

    rows = []

    for d, g in temp.groupby("date_only"):
        actual = g["actual"].values
        predicted = g["predicted"].values

        rows.append({
            "date": d,
            "mape": mape_safe(actual, predicted),
            "mae": mean_absolute_error(actual, predicted),
            "rmse": math.sqrt(mean_squared_error(actual, predicted)),
            "r2": r2_score(actual, predicted) if len(actual) > 1 else np.nan,
            "samples": len(g),
        })

    daily_df = pd.DataFrame(rows)
    daily_df.to_excel(output_path, index=False)

    return daily_df


def safe_lag(series, lag):
    if len(series) >= lag:
        return float(series.iloc[-lag])
    return float(series.iloc[0])


def make_time_features(date_val, hour_val):
    hour0 = hour_val - 1
    dtt = pd.to_datetime(date_val) + pd.Timedelta(hours=int(hour0))

    return {
        "hour_sin": np.sin(2 * np.pi * hour0 / 24),
        "hour_cos": np.cos(2 * np.pi * hour0 / 24),
        "day_of_week": dtt.dayofweek,
        "month": dtt.month,
        "is_weekend": int(dtt.dayofweek >= 5),
    }


def map_outage_to_gen_col(out_col):
    return out_col.replace("out_", "gen_")


def latest_baseline_status_map(hist_df, plant):
    out_cols = [
        c for c in hist_df.columns
        if re.fullmatch(fr"out_{plant}_unit\d+", str(c))
    ]

    if not out_cols:
        return {}

    latest = hist_df.iloc[-1]
    return {c: float(latest[c]) for c in out_cols}


def edited_status_map(planned_df, i, plant):
    out_cols = [
        c for c in planned_df.columns
        if re.fullmatch(fr"out_{plant}_unit\d+", str(c))
    ]

    return {c: float(planned_df.loc[i, c]) for c in out_cols}


def compute_availability_ratio_from_difference(
    plant,
    baseline_status,
    edited_status,
    unit_capacity_weights
):
    capacity_map = unit_capacity_weights.get(plant, {})

    baseline_share = 0.0
    for out_col, b_status in baseline_status.items():
        gen_col = map_outage_to_gen_col(out_col)
        baseline_share += capacity_map.get(gen_col, 0.0) * b_status

    edited_share = 0.0
    for out_col, e_status in edited_status.items():
        gen_col = map_outage_to_gen_col(out_col)
        edited_share += capacity_map.get(gen_col, 0.0) * e_status

    if baseline_share <= 0 and len(baseline_status) > 0:
        baseline_share = sum(baseline_status.values()) / len(baseline_status)

    if edited_share <= 0 and len(edited_status) > 0 and sum(edited_status.values()) > 0:
        edited_share = sum(edited_status.values()) / len(edited_status)

    if baseline_share <= 0:
        return max(0.0, edited_share)

    return max(0.0, edited_share / baseline_share)


def recursive_forecast_with_outages(model, hist_df, target, x_cols, planned_df, plant):
    hist_local = hist_df.copy()
    preds = []

    baseline_status_fixed = latest_baseline_status_map(hist_df, plant)

    plant_out_cols = [
        c for c in planned_df.columns
        if re.fullmatch(fr"out_{plant}_unit\d+", str(c))
    ]

    print(f"{plant} outage columns used: {plant_out_cols}")

    for i in range(24):
        date_i = planned_df.loc[i, "Date"]
        hour_i = int(planned_df.loc[i, "Hour"])

        row = make_time_features(date_i, hour_i)
        series = hist_local[target].copy()

        for lag in CONFIG["lags"]:
            row[f"{target}_lag{lag}"] = safe_lag(series, lag)

        rw = CONFIG["rolling_window"]
        tail = series.tail(rw)

        row[f"{target}_rollmean{rw}"] = float(tail.mean()) if len(tail) > 0 else 0.0
        row[f"{target}_rollstd{rw}"] = float(tail.std()) if len(tail) > 1 else 0.0
        row[f"{target}_rollmin{rw}"] = float(tail.min()) if len(tail) > 0 else 0.0
        row[f"{target}_rollmax{rw}"] = float(tail.max()) if len(tail) > 0 else 0.0
        row[f"{target}_recent_mean3"] = (
            float(series.tail(3).mean()) if len(series) >= 3 else float(series.mean())
        )

        for c in plant_out_cols:
            row[c] = baseline_status_fixed.get(c, 1.0)

        if plant_out_cols:
            units_running_baseline = sum(
                baseline_status_fixed.get(c, 1.0)
                for c in plant_out_cols
            )

            row[f"{plant}_units_running"] = float(units_running_baseline)
            row[f"{plant}_plant_available"] = 1.0 if units_running_baseline > 0 else 0.0

        x_row = pd.DataFrame([{c: row.get(c, 0.0) for c in x_cols}])
        pred_base = float(model.predict(x_row)[0])

        last_actual = float(series.iloc[-1])
        recent_mean3 = float(series.tail(3).mean()) if len(series) >= 3 else last_actual

        if plant in ["agus2", "agus4"]:
            pred_base = 0.20 * pred_base + 0.80 * last_actual
            pred_base = 0.60 * pred_base + 0.40 * recent_mean3
            max_step_change = max(0.8, abs(last_actual) * 0.008)

        elif plant == "agus6":
            pred_base = 0.20 * pred_base + 0.80 * last_actual
            pred_base = 0.55 * pred_base + 0.45 * recent_mean3
            max_step_change = max(0.8, abs(last_actual) * 0.008)

        elif plant == "agus7":
            pred_base = 0.30 * pred_base + 0.70 * last_actual
            pred_base = 0.65 * pred_base + 0.35 * recent_mean3
            max_step_change = max(1.0, abs(last_actual) * 0.012)

        else:
            pred_base = 0.45 * pred_base + 0.55 * last_actual
            pred_base = 0.70 * pred_base + 0.30 * recent_mean3
            max_step_change = max(1.5, abs(last_actual) * 0.02)

        pred_base = float(
            np.clip(
                pred_base,
                last_actual - max_step_change,
                last_actual + max_step_change
            )
        )

        delta = pred_base - last_actual
        max_delta = abs(last_actual) * 0.01
        pred_base = last_actual + np.clip(delta, -max_delta, max_delta)
        pred_base = max(pred_base, 0.0)

        pred_out = pred_base

        if plant_out_cols:
            edited_status = edited_status_map(planned_df, i, plant)

            ratio = compute_availability_ratio_from_difference(
                plant=plant,
                baseline_status=baseline_status_fixed,
                edited_status=edited_status,
                unit_capacity_weights=CONFIG["unit_capacity_weights"]
            )

            pred_out = pred_base * ratio

            if sum(edited_status.values()) == 0:
                pred_out = 0.0

        pred_out = max(pred_out, 0.0)
        preds.append(pred_out)

        new_row = hist_local.iloc[-1].copy()
        new_row["date"] = pd.to_datetime(date_i)
        new_row["time"] = hour_i
        new_row["datetime"] = pd.to_datetime(date_i) + pd.Timedelta(hours=hour_i - 1)
        new_row[target] = pred_base

        for c in plant_out_cols:
            new_row[c] = baseline_status_fixed.get(c, 1.0)

        hist_local = pd.concat(
            [hist_local, pd.DataFrame([new_row])],
            ignore_index=True
        )

    return preds


# ============================================================
# MAIN PIPELINE
# ============================================================

def main():
    if not CLEAN_PATH.exists():
        raise FileNotFoundError(
            f"\nMissing cleaned data file:\n{CLEAN_PATH}\n\n"
            "Run Cell 1 first:\npython scripts/cell1_clean_data.py"
        )

    if not PLANNED_PATH.exists():
        raise FileNotFoundError(
            f"\nMissing planned outage file:\n{PLANNED_PATH}\n\n"
            "Run Cell 1 first:\npython scripts/cell1_clean_data.py"
        )

    # ------------------------------------------------------------
    # LOAD CLEAN DATA
    # ------------------------------------------------------------
    df = pd.read_parquet(CLEAN_PATH)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)

    print("Loaded cleaned data:", CLEAN_PATH)
    print("Rows:", len(df))
    print("Datetime min:", df["datetime"].min())
    print("Datetime max:", df["datetime"].max())

    # ------------------------------------------------------------
    # LOAD PLANNED OUTAGES
    # ------------------------------------------------------------
    planned = pd.read_excel(PLANNED_PATH)

    planned.columns = [str(c).strip() for c in planned.columns]
    col_map = {c.lower(): c for c in planned.columns}

    if "date" not in col_map:
        raise ValueError(f"Could not find Date column. Found columns: {list(planned.columns)}")

    if "hour" not in col_map:
        raise ValueError(f"Could not find Hour column. Found columns: {list(planned.columns)}")

    planned = planned.rename(columns={
        col_map["date"]: "Date",
        col_map["hour"]: "Hour"
    })

    valid_out_cols = [
        c for c in planned.columns
        if re.fullmatch(r"out_agus[124567]_unit\d+", str(c))
    ]

    planned = planned[["Date", "Hour"] + valid_out_cols].copy()
    planned["Date"] = pd.to_datetime(planned["Date"])
    planned["Hour"] = pd.to_numeric(planned["Hour"])

    for c in valid_out_cols:
        planned[c] = pd.to_numeric(planned[c], errors="coerce").fillna(1)
        planned[c] = np.where(planned[c] > 0, 1, 0)

    print("\nPlanned columns being used:")
    print(list(planned.columns))

    # ------------------------------------------------------------
    # TIME FEATURES
    # ------------------------------------------------------------
    hour0 = df["time"] - 1
    dt = pd.to_datetime(df["datetime"])

    df["hour_sin"] = np.sin(2 * np.pi * hour0 / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour0 / 24)
    df["day_of_week"] = dt.dt.dayofweek
    df["month"] = dt.dt.month
    df["is_weekend"] = (dt.dt.dayofweek >= 5).astype(int)

    # ------------------------------------------------------------
    # FEATURE ENGINEERING
    # ------------------------------------------------------------
    for plant in CONFIG["plants"]:
        target = f"total_gen_{plant}"

        if target in df.columns:
            for lag in CONFIG["lags"]:
                df[f"{target}_lag{lag}"] = df[target].shift(lag)

            rw = CONFIG["rolling_window"]
            df[f"{target}_rollmean{rw}"] = df[target].rolling(rw).mean()
            df[f"{target}_rollstd{rw}"] = df[target].rolling(rw).std()
            df[f"{target}_rollmin{rw}"] = df[target].rolling(rw).min()
            df[f"{target}_rollmax{rw}"] = df[target].rolling(rw).max()
            df[f"{target}_recent_mean3"] = df[target].rolling(3).mean()

    for plant in CONFIG["plants"]:
        unit_out_cols = [
            c for c in df.columns
            if re.fullmatch(fr"out_{plant}_unit\d+", str(c))
        ]

        if unit_out_cols:
            df[f"{plant}_units_running"] = df[unit_out_cols].sum(axis=1)
            df[f"{plant}_plant_available"] = (
                df[f"{plant}_units_running"] > 0
            ).astype(int)

    # ------------------------------------------------------------
    # OUTPUT FRAMES
    # ------------------------------------------------------------
    forecast_rf = pd.DataFrame({
        "Date": planned["Date"].dt.date,
        "Hour": planned["Hour"].astype(int),
    })

    forecast_xgb = pd.DataFrame({
        "Date": planned["Date"].dt.date,
        "Hour": planned["Hour"].astype(int),
    })

    benchmark_metric_rows = []

    # ------------------------------------------------------------
    # TRAIN / LOAD + VALIDATION / TESTING + DAILY METRICS + FORECAST
    # ------------------------------------------------------------
    for plant in CONFIG["plants"]:
        print(f"\n==================== {plant.upper()} ====================")

        target = f"total_gen_{plant}"
        y_col = f"{target}_tplus1"

        if target not in df.columns:
            print(f"[SKIP] Missing target column: {target}")
            continue

        work = df.copy()
        work[y_col] = work[target].shift(-1)

        x_cols = ["hour_sin", "hour_cos", "day_of_week", "month", "is_weekend"]

        x_cols += [c for c in work.columns if c.startswith(target + "_lag")]
        x_cols += [c for c in work.columns if c.startswith(target + "_roll")]
        x_cols += [c for c in work.columns if c.startswith(target + "_recent_mean3")]

        x_cols += [
            c for c in work.columns
            if re.fullmatch(fr"out_{plant}_unit\d+", str(c))
        ]

        x_cols += [
            c for c in work.columns
            if c in [f"{plant}_units_running", f"{plant}_plant_available"]
        ]

        x_cols = list(dict.fromkeys([c for c in x_cols if c in work.columns]))

        work = work.dropna(subset=[y_col] + x_cols).copy()

        if len(work) < 100:
            print(f"[SKIP] Not enough rows for {plant}: {len(work)}")
            continue

        print("Target:", target)
        print("Feature count:", len(x_cols))
        print("Rows after feature prep:", len(work))

        train_df, val_df, test_df = chronological_split(
            work,
            train_ratio=CONFIG["train_ratio"],
            val_ratio=CONFIG["val_ratio"]
        )

        x_train = train_df[x_cols]
        y_train = train_df[y_col]

        x_val = val_df[x_cols]
        y_val = val_df[y_col]

        x_test = test_df[x_cols]
        y_test = test_df[y_col]

        # --------------------------------------------------------
        # RANDOM FOREST
        # --------------------------------------------------------
        rf_path = RF_DIR / f"rf_{plant}.pkl"

        if rf_path.exists() and not CONFIG["retrain_benchmarks"]:
            rf = joblib.load(rf_path)
            print(f"Loaded RF model for {plant}")
        else:
            rf = RandomForestRegressor(
                n_estimators=CONFIG["rf_n_estimators"],
                random_state=CONFIG["rf_random_state"],
                n_jobs=-1
            )

            rf.fit(x_train, y_train)
            joblib.dump(rf, rf_path)

            print(f"Saved RF model for {plant}")

        rf_val_pred = rf.predict(x_val)
        rf_test_pred = rf.predict(x_test)

        rf_val_metrics = compute_metrics(y_val, rf_val_pred)
        rf_test_metrics = compute_metrics(y_test, rf_test_pred)

        benchmark_metric_rows.append({
            "model": "Random Forest",
            "plant": plant,
            "val_mape": rf_val_metrics["mape"],
            "val_mae": rf_val_metrics["mae"],
            "val_rmse": rf_val_metrics["rmse"],
            "val_r2": rf_val_metrics["r2"],
            "test_mape": rf_test_metrics["mape"],
            "test_mae": rf_test_metrics["mae"],
            "test_rmse": rf_test_metrics["rmse"],
            "test_r2": rf_test_metrics["r2"],
        })

        rf_val_daily_path = VAL_DAILY_DIR / f"random_forest_{plant}_validation_daily_metrics.xlsx"
        rf_test_daily_path = TEST_DAILY_DIR / f"random_forest_{plant}_testing_daily_metrics.xlsx"

        save_daily_metrics(val_df, y_val, rf_val_pred, rf_val_daily_path)
        save_daily_metrics(test_df, y_test, rf_test_pred, rf_test_daily_path)

        print("Saved RF validation daily metrics:", rf_val_daily_path)
        print("Saved RF testing daily metrics:", rf_test_daily_path)

        forecast_rf[plant.upper()] = recursive_forecast_with_outages(
            model=rf,
            hist_df=df,
            target=target,
            x_cols=x_cols,
            planned_df=planned,
            plant=plant
        )

        # --------------------------------------------------------
        # XGBOOST
        # --------------------------------------------------------
        xgb_path = XGB_DIR / f"xgb_{plant}.json"

        if xgb_path.exists() and not CONFIG["retrain_benchmarks"]:
            xgb = XGBRegressor()
            xgb.load_model(xgb_path)
            print(f"Loaded XGB model for {plant}")
        else:
            xgb = XGBRegressor(
                n_estimators=CONFIG["xgb_n_estimators"],
                learning_rate=CONFIG["xgb_learning_rate"],
                max_depth=CONFIG["xgb_max_depth"],
                subsample=CONFIG["xgb_subsample"],
                colsample_bytree=CONFIG["xgb_colsample_bytree"],
                random_state=CONFIG["xgb_random_state"],
                objective="reg:squarederror",
                n_jobs=-1
            )

            xgb.fit(x_train, y_train)
            xgb.save_model(xgb_path)

            print(f"Saved XGB model for {plant}")

        xgb_val_pred = xgb.predict(x_val)
        xgb_test_pred = xgb.predict(x_test)

        xgb_val_metrics = compute_metrics(y_val, xgb_val_pred)
        xgb_test_metrics = compute_metrics(y_test, xgb_test_pred)

        benchmark_metric_rows.append({
            "model": "XGBoost",
            "plant": plant,
            "val_mape": xgb_val_metrics["mape"],
            "val_mae": xgb_val_metrics["mae"],
            "val_rmse": xgb_val_metrics["rmse"],
            "val_r2": xgb_val_metrics["r2"],
            "test_mape": xgb_test_metrics["mape"],
            "test_mae": xgb_test_metrics["mae"],
            "test_rmse": xgb_test_metrics["rmse"],
            "test_r2": xgb_test_metrics["r2"],
        })

        xgb_val_daily_path = VAL_DAILY_DIR / f"xgboost_{plant}_validation_daily_metrics.xlsx"
        xgb_test_daily_path = TEST_DAILY_DIR / f"xgboost_{plant}_testing_daily_metrics.xlsx"

        save_daily_metrics(val_df, y_val, xgb_val_pred, xgb_val_daily_path)
        save_daily_metrics(test_df, y_test, xgb_test_pred, xgb_test_daily_path)

        print("Saved XGB validation daily metrics:", xgb_val_daily_path)
        print("Saved XGB testing daily metrics:", xgb_test_daily_path)

        forecast_xgb[plant.upper()] = recursive_forecast_with_outages(
            model=xgb,
            hist_df=df,
            target=target,
            x_cols=x_cols,
            planned_df=planned,
            plant=plant
        )

    # ------------------------------------------------------------
    # SAVE FORECASTS
    # ------------------------------------------------------------
    rf_xlsx = BENCH_DIR / "Day_Ahead_24H_RANDOM_FOREST.xlsx"
    rf_csv = BENCH_DIR / "Day_Ahead_24H_RANDOM_FOREST.csv"

    xgb_xlsx = BENCH_DIR / "Day_Ahead_24H_XGBOOST.xlsx"
    xgb_csv = BENCH_DIR / "Day_Ahead_24H_XGBOOST.csv"

    forecast_rf.to_excel(rf_xlsx, index=False)
    forecast_rf.to_csv(rf_csv, index=False)

    forecast_xgb.to_excel(xgb_xlsx, index=False)
    forecast_xgb.to_csv(xgb_csv, index=False)

    # ------------------------------------------------------------
    # SAVE OVERALL VALIDATION / TESTING METRICS
    # ------------------------------------------------------------
    metrics_path = METRICS_DIR / "benchmark_validation_testing_metrics.xlsx"

    benchmark_metrics_df = pd.DataFrame(benchmark_metric_rows)
    benchmark_metrics_df.to_excel(metrics_path, index=False)

    # ------------------------------------------------------------
    # PRINT RESULTS
    # ------------------------------------------------------------
    print("\nDone: Benchmark models with validation/testing daily metrics")

    print("\nSaved RF forecast:", rf_xlsx)
    print("Saved RF forecast:", rf_csv)

    print("\nSaved XGB forecast:", xgb_xlsx)
    print("Saved XGB forecast:", xgb_csv)

    print("\nSaved overall benchmark validation/testing metrics:", metrics_path)
    print("Saved validation daily metrics folder:", VAL_DAILY_DIR)
    print("Saved testing daily metrics folder:", TEST_DAILY_DIR)

    print("\nBenchmark metrics preview:")
    print(benchmark_metrics_df)

    print("\nRandom Forest forecast preview:")
    print(forecast_rf.head())

    print("\nXGBoost forecast preview:")
    print(forecast_xgb.head())


if __name__ == "__main__":
    main()
"""
Optimized Cell 2: cascade-aware, outage-aware RBFNN.

Key improvements:
- Separate optimized outputs, models, and metadata.
- RBFNN predicts next-hour residual generation change around persistence.
- Validation-tuned shrinkage keeps forecasts continuous and limits drift.
- Operational MAPE excludes near-zero actual generation where percentage error
  is not meaningful; MAE, RMSE, and R2 still use all rows.
- Cascade features preserve upstream influence, including Agus 5 -> 6 -> 7.
"""

import json
import math
import os
import re
import shutil
import sys
import warnings
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
np.random.seed(42)
tf.get_logger().setLevel("ERROR")
tf.random.set_seed(42)

PROJECT_DIR = Path(__file__).resolve().parents[2]
THESIS_DIR = Path(__file__).resolve().parents[1]
OPT_DIR = THESIS_DIR
LEGACY_OPT_DIR = PROJECT_DIR / "archive" / "optimized_version"
OUT_DIR = OPT_DIR / "outputs"
CLEANED_DATA_DIR = OUT_DIR / "cleaned_data"
OUTAGES_DIR = OUT_DIR / "outages_planning"
RBFNN_FORECAST_DIR = OUT_DIR / "rbfnn_forecast"
MODEL_DIR = OPT_DIR / "models" / "rbfnn"
LEGACY_MODEL_DIR = LEGACY_OPT_DIR / "models" / "rbfnn"
META_DIR = OPT_DIR / "metadata"
VALIDATION_METRICS_DIR = META_DIR / "validation_metrics"
TESTING_METRICS_DIR = META_DIR / "testing_metrics"
OVERALL_METRICS_DIR = META_DIR / "overall_metrics"
TRAINING_HISTORY_DIR = META_DIR / "training_validation_loss"
DIAGNOSTICS_DIR = OVERALL_METRICS_DIR / "actual_forecast_diagnostics"
PLOTS_DIR = OVERALL_METRICS_DIR / "plots"

for folder in [
    CLEANED_DATA_DIR,
    OUTAGES_DIR,
    RBFNN_FORECAST_DIR,
    MODEL_DIR,
    VALIDATION_METRICS_DIR,
    TESTING_METRICS_DIR,
    OVERALL_METRICS_DIR,
    TRAINING_HISTORY_DIR,
    DIAGNOSTICS_DIR,
    PLOTS_DIR,
]:
    folder.mkdir(parents=True, exist_ok=True)

CLEAN_PATH = CLEANED_DATA_DIR / "cleaned_hourly_data.parquet"
PLANNED_PATH = OUTAGES_DIR / "Planned_Outages_Input.xlsx"

PLANTS = ["agus1", "agus2", "agus4", "agus5", "agus6", "agus7"]
UPSTREAM_MAP = {"agus1": None, "agus2": "agus1", "agus4": "agus2", "agus5": "agus4", "agus6": "agus5", "agus7": "agus6"}
CAPACITY_MW = {"agus1": 80.0, "agus2": 180.0, "agus4": 158.1, "agus5": 55.0, "agus6": 219.0, "agus7": 54.0}
UNIT_CAPACITY = {
    "agus1": {"unit1": 40.0, "unit2": 40.0},
    "agus2": {"unit1": 60.0, "unit2": 60.0, "unit3": 60.0},
    "agus4": {"unit1": 52.7, "unit2": 52.7, "unit3": 52.7},
    "agus5": {"unit1": 27.5, "unit2": 27.5},
    "agus6": {"unit1": 34.5, "unit2": 34.5, "unit3": 50.0, "unit4": 50.0, "unit5": 50.0},
    "agus7": {"unit1": 27.0, "unit2": 27.0},
}
METRICS_COLUMNS = [
    "model",
    "plant",
    "feature_count",
    "val_operational_mape",
    "val_mae",
    "val_rmse",
    "val_r2",
    "test_operational_mape",
    "test_mae",
    "test_rmse",
    "test_r2",
]
TESTING_PREDICTION_COLUMNS = [
    "Date",
    "Hour",
    "datetime",
    "plant",
    "actual_generation",
    "predicted_generation",
    "model",
]
UNIT_FORECAST_COLUMNS = [
    f"gen_{plant}_{unit}"
    for plant in PLANTS
    for unit in UNIT_CAPACITY[plant]
]
TOTAL_FORECAST_COLUMNS = [f"total_gen_{plant}" for plant in PLANTS]
CASCADE_FORECAST_COLUMN = "total_cascade_generation"

LAGS = [1, 2, 3, 6, 12, 24, 48, 72, 168]
ROLL_WINDOWS = [3, 6, 12, 24, 48, 168]
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
EPOCHS = 80
BATCH_SIZE = 32
N_CENTERS = 120
LEARNING_RATE = 0.001
SEARCH_CENTER_COUNTS = [80, 120, 180]
SEARCH_LEARNING_RATES = [0.001, 0.0005]
SHRINKAGE_GRID = [0.05, 0.10, 0.20, 0.35, 0.50, 0.75, 1.00, 1.15]
BIN_CALIBRATION_PLANTS = {"agus1", "agus5"}
BIN_CALIBRATION_QUANTILES = list(range(3, 21))
BIN_CALIBRATION_QUANTILES_BY_PLANT = {"agus1": list(range(3, 41))}
BIN_CALIBRATION_SCALES = [round(x, 2) for x in np.arange(0.50, 2.55, 0.05)]
BIN_CALIBRATION_BASIS = {"agus1": "base_pred", "agus5": "current"}
SHAPE_OPTIMIZED_PLANTS = {"agus5", "agus7"}
PROFILE_BLEND_GRID = [0.0, 0.10, 0.20, 0.30, 0.40, 0.55, 0.70]
HOURLY_CORRECTION_SCALE_GRID = [0.0, 0.25, 0.50, 0.75, 1.00]
ACTUAL_NEXT_DAY_PATH = PROJECT_DIR / "july 1, 2025.xlsx"
FORECAST_PROFILE_BLEND_FALLBACK = {
    "agus5": {"same_hour_yesterday_weight": 0.70, "source": "forecast_fallback"},
    "agus7": {"same_hour_yesterday_weight": 0.20, "source": "forecast_fallback"},
}


def atomic_save_keras_model(model, path):
    temp_path = path.with_name(f"{path.stem}.tmp{path.suffix}")
    if temp_path.exists():
        temp_path.unlink()
    try:
        model.save(temp_path)
        shutil.move(str(temp_path), str(path))
    except PermissionError:
        if temp_path.exists():
            temp_path.unlink()
        model.save(path.with_suffix(".h5"))


def format_excel(path):
    try:
        from openpyxl import load_workbook
    except ImportError:
        return

    wb = load_workbook(path)
    for ws in wb.worksheets:
        ws.freeze_panes = "A2"
        ws.auto_filter.ref = ws.dimensions
        for column_cells in ws.columns:
            header = str(column_cells[0].value or "")
            max_len = max([len(header)] + [len(str(cell.value)) for cell in column_cells[1:80] if cell.value is not None])
            ws.column_dimensions[column_cells[0].column_letter].width = min(max(max_len + 2, 10), 34)
    wb.save(path)


def save_training_history(history, plant):
    history_df = pd.DataFrame(history.history)
    history_df.insert(0, "epoch", np.arange(1, len(history_df) + 1))
    history_path = TRAINING_HISTORY_DIR / f"{plant}_training_history.xlsx"
    history_df.to_excel(history_path, index=False)
    format_excel(history_path)
    return history_path


def model_file(name):
    current = MODEL_DIR / name
    current_h5 = current.with_suffix(".h5")
    if current_h5.exists():
        return current_h5
    if current.exists():
        return current
    legacy = LEGACY_MODEL_DIR / name
    legacy_h5 = legacy.with_suffix(".h5")
    if legacy_h5.exists():
        return legacy_h5
    if legacy.exists():
        return legacy
    return current


def rebuild_datetime(df):
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"])
    out["time"] = pd.to_numeric(out["time"]).astype(int)
    if "datetime" not in out.columns:
        out["datetime"] = out["date"] + pd.to_timedelta(out["time"] - 1, unit="h")
    else:
        out["datetime"] = pd.to_datetime(out["datetime"])
    return out.sort_values("datetime").reset_index(drop=True)


def add_runtime_compatibility_columns(df):
    # Thesis Forecasting cleaned files use public names; legacy saved models may expect old feature names.
    out = df.copy()
    for plant in PLANTS:
        new_gate = f"tot_{plant}_gate"
        old_gate = f"tot_{plant}"
        if new_gate in out.columns and old_gate not in out.columns:
            out[old_gate] = out[new_gate]
    if "lake_lanao_outflow" in out.columns and "lake_lanao_hourly_outflow_elev_agus7" not in out.columns:
        out["lake_lanao_hourly_outflow_elev_agus7"] = out["lake_lanao_outflow"]
    return out


class RBFLayer(tf.keras.layers.Layer):
    def __init__(self, units, gamma_init=1.0, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.gamma_init = gamma_init

    def build(self, input_shape):
        input_dim = int(input_shape[-1])
        self.centers = self.add_weight(name="centers", shape=(self.units, input_dim), initializer="uniform", trainable=True)
        self.log_gamma = self.add_weight(
            name="log_gamma",
            shape=(self.units,),
            initializer=tf.keras.initializers.Constant(np.log(self.gamma_init)),
            trainable=True,
        )

    def call(self, inputs):
        x = tf.expand_dims(inputs, axis=1)
        c = tf.expand_dims(self.centers, axis=0)
        d2 = tf.reduce_sum(tf.square(x - c), axis=-1)
        return tf.exp(-tf.exp(self.log_gamma) * d2)


def build_model(input_dim, n_centers=N_CENTERS, learning_rate=LEARNING_RATE):
    inputs = tf.keras.Input(shape=(input_dim,))
    x = RBFLayer(n_centers, gamma_init=1.0)(inputs)
    outputs = tf.keras.layers.Dense(1, activation="linear")(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate), loss="mse")
    return model


def init_centers(model, x_train, n_centers=N_CENTERS):
    n_clusters = min(n_centers, len(x_train))
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=5)
    km.fit(x_train)
    centers = km.cluster_centers_
    if n_clusters < n_centers:
        centers = np.vstack([centers, np.repeat(centers[-1:, :], n_centers - n_clusters, axis=0)])
    for layer in model.layers:
        if isinstance(layer, RBFLayer):
            layer.centers.assign(centers)
            return


def operational_threshold(plant):
    return max(1.0, 0.01 * CAPACITY_MW[plant])


def operational_mape(y_true, y_pred, plant):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.abs(y_true) >= operational_threshold(plant)
    if not mask.any():
        return np.nan, 0, len(y_true)
    value = np.mean(np.abs((y_true[mask] - y_pred[mask]) / np.abs(y_true[mask]))) * 100.0
    return float(value), int(mask.sum()), int((~mask).sum())


def metrics_dict(y_true, y_pred, plant):
    mape, included, excluded = operational_mape(y_true, y_pred, plant)
    return {
        "operational_mape": mape,
        "mape_rows_included": included,
        "mape_rows_excluded": excluded,
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": math.sqrt(mean_squared_error(y_true, y_pred)),
        "r2": r2_score(y_true, y_pred),
    }


def chronological_split(data):
    n = len(data)
    i1 = int(n * TRAIN_RATIO)
    i2 = int(n * (TRAIN_RATIO + VAL_RATIO))
    return data.iloc[:i1].copy(), data.iloc[i1:i2].copy(), data.iloc[i2:].copy()


def add_features(df):
    out = df.copy()
    out["datetime"] = pd.to_datetime(out["datetime"])
    hour0 = out["time"].astype(int) - 1
    dt = out["datetime"]
    out["hour_sin"] = np.sin(2 * np.pi * hour0 / 24)
    out["hour_cos"] = np.cos(2 * np.pi * hour0 / 24)
    out["day_sin"] = np.sin(2 * np.pi * dt.dt.dayofweek / 7)
    out["day_cos"] = np.cos(2 * np.pi * dt.dt.dayofweek / 7)
    out["month_sin"] = np.sin(2 * np.pi * dt.dt.month / 12)
    out["month_cos"] = np.cos(2 * np.pi * dt.dt.month / 12)
    out["is_weekend"] = (dt.dt.dayofweek >= 5).astype(int)
    target_dt = dt + pd.Timedelta(hours=1)
    target_hour0 = target_dt.dt.hour
    out["target_hour_sin"] = np.sin(2 * np.pi * target_hour0 / 24)
    out["target_hour_cos"] = np.cos(2 * np.pi * target_hour0 / 24)
    out["target_day_sin"] = np.sin(2 * np.pi * target_dt.dt.dayofweek / 7)
    out["target_day_cos"] = np.cos(2 * np.pi * target_dt.dt.dayofweek / 7)
    out["target_is_weekend"] = (target_dt.dt.dayofweek >= 5).astype(int)

    for plant in PLANTS:
        target = f"total_gen_{plant}"
        out[f"{target}_current"] = out[target]
        out[f"{target}_tplus1"] = out[target].shift(-1)
        out[f"{target}_delta_tplus1"] = out[f"{target}_tplus1"] - out[target]
        for lag in LAGS:
            out[f"{target}_lag{lag}"] = out[target].shift(lag)
        for window in ROLL_WINDOWS:
            roll = out[target].rolling(window, min_periods=max(2, window // 2))
            out[f"{target}_rollmean{window}"] = roll.mean()
            out[f"{target}_rollstd{window}"] = roll.std().fillna(0.0)
            out[f"{target}_rollmin{window}"] = roll.min()
            out[f"{target}_rollmax{window}"] = roll.max()
        out[f"{target}_diff1"] = out[target].diff(1)
        out[f"{target}_diff3"] = out[target].diff(3)
        out[f"{target}_diff24"] = out[target].diff(24)
        out[f"{target}_target_lag24"] = out[target].shift(23)
        out[f"{target}_target_lag168"] = out[target].shift(167)

        out_cols = [c for c in out.columns if re.fullmatch(fr"out_{plant}_unit\d+", c)]
        if out_cols:
            out[f"{plant}_units_running"] = out[out_cols].sum(axis=1)
            out[f"{plant}_plant_available"] = (out[f"{plant}_units_running"] > 0).astype(int)

    for plant, upstream in UPSTREAM_MAP.items():
        if upstream is None:
            continue
        up_target = f"total_gen_{upstream}"
        out[f"{plant}_upstream_current"] = out[up_target]
        for lag in LAGS:
            out[f"{plant}_upstream_gen_lag{lag}"] = out[up_target].shift(lag)

    for col in [c for c in out.columns if c.startswith("tot_agus") or c.startswith("elev_agus") or "outflow" in c or c == "rainfall"]:
        for lag in [1, 3, 6, 12, 24]:
            out[f"{col}_lag{lag}"] = out[col].shift(lag)
    return out


def feature_columns_for(data, plant):
    target = f"total_gen_{plant}"
    cols = [
        "hour_sin",
        "hour_cos",
        "day_sin",
        "day_cos",
        "month_sin",
        "month_cos",
        "is_weekend",
        "target_hour_sin",
        "target_hour_cos",
        "target_day_sin",
        "target_day_cos",
        "target_is_weekend",
        f"{target}_current",
    ]
    prefixes = [f"{target}_lag", f"{target}_roll", f"{target}_diff", f"{target}_target_", f"{plant}_upstream_"]
    cols += [c for c in data.columns if any(c.startswith(prefix) for prefix in prefixes)]
    cols += [c for c in data.columns if re.fullmatch(fr"out_{plant}_unit\d+", c)]
    cols += [f"{plant}_units_running", f"{plant}_plant_available"]
    cols += [c for c in data.columns if c.startswith(f"tot_{plant}") or c.startswith(f"elev_{plant}") or c.startswith(f"spill_{plant}")]
    cols += [c for c in data.columns if "_lag" in c and (c.startswith("tot_agus") or c.startswith("elev_agus") or "outflow" in c or c.startswith("rainfall"))]
    return list(dict.fromkeys([c for c in cols if c in data.columns]))


def save_daily_metrics(part_df, y_true, y_pred, plant, path):
    temp = part_df[["datetime"]].copy()
    temp["actual"] = np.asarray(y_true, dtype=float)
    temp["predicted"] = np.asarray(y_pred, dtype=float)
    temp["date"] = pd.to_datetime(temp["datetime"]).dt.date
    rows = []
    for date, group in temp.groupby("date"):
        daily_metrics = metrics_dict(group["actual"], group["predicted"], plant)
        daily_metrics.pop("r2", None)
        rows.append({"date": date, **daily_metrics, "samples": len(group)})
    pd.DataFrame(rows).to_excel(path, index=False)


def calibrated_level_predictions(current, delta_pred, actual, plant, shrinkage):
    raw_pred = np.clip(current + shrinkage * delta_pred, 0.0, CAPACITY_MW[plant] * 1.05)
    bias = float(np.median(np.asarray(actual, dtype=float) - raw_pred))
    max_bias = operational_threshold(plant)
    bias = float(np.clip(bias, -max_bias, max_bias))
    pred = np.clip(raw_pred + bias, 0.0, CAPACITY_MW[plant] * 1.05)
    return pred, bias


def apply_level_prediction(current, delta_pred, plant, shrinkage, bias):
    pred = np.clip(current + shrinkage * delta_pred + bias, 0.0, CAPACITY_MW[plant] * 1.05)
    return pred


def calibration_bins(values, q):
    _, bins = pd.qcut(pd.Series(values), q, duplicates="drop", retbins=True)
    return np.asarray(bins, dtype=float)


def bin_ids(values, bins):
    return np.digitize(np.asarray(values, dtype=float), bins[1:-1])


def apply_bin_calibration(pred, basis_values, plant, calibration):
    if not calibration:
        return np.asarray(pred, dtype=float)
    bins = np.asarray(calibration["bins"], dtype=float)
    corrections = {int(k): float(v) for k, v in calibration["corrections"].items()}
    scale = float(calibration["scale"])
    ids = bin_ids(basis_values, bins)
    adjustment = np.asarray([corrections.get(int(bin_id), 0.0) for bin_id in ids], dtype=float)
    return np.clip(np.asarray(pred, dtype=float) + scale * adjustment, 0.0, CAPACITY_MW[plant] * 1.05)


def tune_bin_calibration(plant, current, actual, base_pred):
    if plant not in BIN_CALIBRATION_PLANTS:
        return None, base_pred
    best = {"score": candidate_score(metrics_dict(actual, base_pred, plant)), "calibration": None, "pred": base_pred}
    errors = np.asarray(actual, dtype=float) - np.asarray(base_pred, dtype=float)
    basis_name = BIN_CALIBRATION_BASIS.get(plant, "current")
    basis_values = np.asarray(base_pred if basis_name == "base_pred" else current, dtype=float)
    for q in BIN_CALIBRATION_QUANTILES_BY_PLANT.get(plant, BIN_CALIBRATION_QUANTILES):
        bins = calibration_bins(basis_values, q)
        ids = bin_ids(basis_values, bins)
        grouped = pd.DataFrame({"bin": ids, "error": errors}).groupby("bin")["error"].median()
        corrections = {int(k): float(v) for k, v in grouped.items()}
        for scale in BIN_CALIBRATION_SCALES:
            calibration = {"bins": bins.tolist(), "corrections": corrections, "scale": scale, "basis": basis_name}
            pred = apply_bin_calibration(base_pred, basis_values, plant, calibration)
            score = candidate_score(metrics_dict(actual, pred, plant))
            if score < best["score"]:
                best = {"score": score, "calibration": calibration, "pred": pred}
    return best["calibration"], best["pred"]


def candidate_score(metrics):
    mape = metrics["operational_mape"]
    mape_score = float(mape) if pd.notna(mape) else float("inf")
    return (mape_score, metrics["rmse"], -metrics["r2"])


def target_hours_from_rows(df):
    return ((pd.to_numeric(df["time"]).astype(int) % 24) + 1).astype(int).values


def same_hour_target_anchor(df, plant):
    target = f"total_gen_{plant}"
    col = f"{target}_target_lag24"
    if col in df.columns:
        return df[col].values.astype(float)
    return df[target].shift(23).values.astype(float)


def apply_hourly_correction(pred, hours, plant, correction):
    out = np.asarray(pred, dtype=float).copy()
    if not correction:
        return out
    scale = float(correction.get("scale", 0.0))
    by_hour = {int(k): float(v) for k, v in correction.get("hour_corrections", {}).items()}
    offsets = np.asarray([by_hour.get(int(hour), 0.0) for hour in hours], dtype=float)
    return np.clip(out + scale * offsets, 0.0, CAPACITY_MW[plant] * 1.05)


def tune_hourly_correction(plant, pred, actual, hours):
    if plant not in SHAPE_OPTIMIZED_PLANTS:
        return None, np.asarray(pred, dtype=float)
    pred = np.asarray(pred, dtype=float)
    actual = np.asarray(actual, dtype=float)
    hours = np.asarray(hours, dtype=int)
    errors = actual - pred
    grouped = pd.DataFrame({"hour": hours, "error": errors}).groupby("hour")["error"].median()
    max_offset = max(1.0, CAPACITY_MW[plant] * 0.08)
    corrections = {int(k): float(np.clip(v, -max_offset, max_offset)) for k, v in grouped.items()}
    best = {"score": candidate_score(metrics_dict(actual, pred, plant)), "correction": None, "pred": pred}
    for scale in HOURLY_CORRECTION_SCALE_GRID:
        correction = {"hour_corrections": corrections, "scale": scale}
        candidate = apply_hourly_correction(pred, hours, plant, correction)
        score = candidate_score(metrics_dict(actual, candidate, plant))
        if score < best["score"]:
            best = {"score": score, "correction": correction, "pred": candidate}
    return best["correction"], best["pred"]


def apply_profile_blend(pred, anchor, plant, config):
    pred = np.asarray(pred, dtype=float)
    anchor = np.asarray(anchor, dtype=float)
    if not config:
        return pred
    weight = float(config.get("same_hour_yesterday_weight", 0.0))
    valid_anchor = np.isfinite(anchor)
    blended = pred.copy()
    blended[valid_anchor] = (1.0 - weight) * pred[valid_anchor] + weight * anchor[valid_anchor]
    return np.clip(blended, 0.0, CAPACITY_MW[plant] * 1.05)


def tune_profile_blend(plant, pred, actual, anchor):
    if plant not in SHAPE_OPTIMIZED_PLANTS:
        return None, np.asarray(pred, dtype=float)
    pred = np.asarray(pred, dtype=float)
    actual = np.asarray(actual, dtype=float)
    anchor = np.asarray(anchor, dtype=float)
    best = {"score": candidate_score(metrics_dict(actual, pred, plant)), "config": None, "pred": pred}
    for weight in PROFILE_BLEND_GRID:
        config = {"same_hour_yesterday_weight": weight}
        candidate = apply_profile_blend(pred, anchor, plant, config)
        score = candidate_score(metrics_dict(actual, candidate, plant))
        if score < best["score"]:
            best = {"score": score, "config": config, "pred": candidate}
    return best["config"], best["pred"]


def shape_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ape = np.abs(y_true - y_pred) / np.maximum(np.abs(y_true), 1e-6) * 100.0
    true_delta = np.diff(y_true)
    pred_delta = np.diff(y_pred)
    if len(true_delta) == 0:
        trend_match = np.nan
        shape_mae = np.nan
    else:
        trend_match = float(np.mean(np.sign(true_delta) == np.sign(pred_delta)))
        shape_mae = float(np.mean(np.abs(true_delta - pred_delta)))
    return {
        "shape_mape": float(np.mean(ape)),
        "shape_ape_le_3_count": int((ape <= 3.0).sum()),
        "shape_max_ape": float(np.max(ape)),
        "shape_trend_match_rate": trend_match,
        "shape_delta_mae": shape_mae,
    }


def unit_from_outage_col(col):
    match = re.search(r"unit\d+", col)
    return match.group(0) if match else None


def latest_status(hist_df, plant):
    cols = [c for c in hist_df.columns if re.fullmatch(fr"out_{plant}_unit\d+", c)]
    latest = hist_df.iloc[-1]
    return {c: float(latest[c]) for c in cols}


def planned_status(planned, idx, plant):
    cols = [c for c in planned.columns if re.fullmatch(fr"out_{plant}_unit\d+", c)]
    return {c: float(planned.loc[idx, c]) for c in cols}


def availability_ratio(plant, baseline, planned):
    if not planned:
        return 1.0
    base_cap = 0.0
    plan_cap = 0.0
    for col, status in baseline.items():
        base_cap += UNIT_CAPACITY[plant].get(unit_from_outage_col(col), 0.0) * status
    for col, status in planned.items():
        plan_cap += UNIT_CAPACITY[plant].get(unit_from_outage_col(col), 0.0) * status
    if plan_cap <= 0:
        return 0.0
    if base_cap <= 0:
        return plan_cap / CAPACITY_MW[plant]
    return plan_cap / base_cap


def distribute_to_units(plant, plant_forecast, status):
    # Unit-level forecast output honors the edited outage plan: status 0 gets 0 MW.
    available_units = []
    for unit, capacity in UNIT_CAPACITY[plant].items():
        out_col = f"out_{plant}_{unit}"
        if status.get(out_col, 1.0) > 0:
            available_units.append((unit, capacity))

    if not available_units:
        return {f"gen_{plant}_{unit}": 0.0 for unit in UNIT_CAPACITY[plant]}

    available_capacity = sum(capacity for _, capacity in available_units)
    unit_values = {f"gen_{plant}_{unit}": 0.0 for unit in UNIT_CAPACITY[plant]}
    for unit, capacity in available_units:
        unit_values[f"gen_{plant}_{unit}"] = float(plant_forecast) * capacity / available_capacity
    return unit_values


def empty_forecast_frame(planned):
    forecast = pd.DataFrame({"Date": planned["Date"].dt.date, "Hour": planned["Hour"].astype(int)})
    for col in UNIT_FORECAST_COLUMNS + TOTAL_FORECAST_COLUMNS:
        forecast[col] = 0.0
    forecast[CASCADE_FORECAST_COLUMN] = 0.0
    return forecast


def display_hour(hour):
    hour0 = int(hour) - 1
    return "00:00" if hour0 == 0 else f"{hour0}:00"


def parse_planned_hour(value):
    if isinstance(value, str):
        text = value.strip()
        match = re.fullmatch(r"(\d{1,2})(?::00)?", text)
        if match:
            hour0 = int(match.group(1))
            if 0 <= hour0 <= 23:
                return hour0 + 1
    return int(pd.to_numeric(value))


def forecast_display_column(col):
    unit_match = re.fullmatch(r"gen_agus(\d+)_(unit\d+)", col)
    if unit_match:
        plant_num, unit = unit_match.groups()
        return f"Gen_Agus{plant_num}_Unit{unit.replace('unit', '', 1)}_MW"

    total_match = re.fullmatch(r"total_gen_agus(\d+)", col)
    if total_match:
        return f"Total_Gen_Agus{total_match.group(1)}_MW"

    if col == CASCADE_FORECAST_COLUMN:
        return "Total_Cascade_Generation_MW"
    return col


def format_forecast_output(forecast):
    formatted = forecast.copy()
    formatted["Hour"] = formatted["Hour"].apply(display_hour)
    formatted = formatted.rename(columns={col: forecast_display_column(col) for col in formatted.columns})
    return formatted


def ramp_limit(train_series):
    diffs = train_series.diff().abs().dropna()
    return float(max(1.0, diffs.quantile(0.98))) if not diffs.empty else 2.0


def load_planned():
    planned = pd.read_excel(PLANNED_PATH)
    planned.columns = [str(c).strip() for c in planned.columns]
    col_map = {c.lower(): c for c in planned.columns}
    planned = planned.rename(columns={col_map["date"]: "Date", col_map["hour"]: "Hour"})
    planned["Date"] = pd.to_datetime(planned["Date"])
    planned["Hour"] = planned["Hour"].apply(parse_planned_hour).astype(int)
    for col in [c for c in planned.columns if re.fullmatch(r"out_agus[124567]_unit\d+", c)]:
        planned[col] = np.where(pd.to_numeric(planned[col], errors="coerce").fillna(1) > 0, 1, 0)
    return planned


def predict_delta(model, x_scaler, y_scaler, x_frame):
    x_scaled = x_scaler.transform(x_frame.values.astype(np.float32)).astype(np.float32)
    y_scaled = model(x_scaled, training=False).numpy()
    return y_scaler.inverse_transform(y_scaled).flatten()


def feature_row_from_history(hist, plant, x_cols, date_val, hour_val):
    hist_feat = add_features(hist.copy())
    row = hist_feat.iloc[-1].to_dict()
    dt = pd.to_datetime(date_val) + pd.Timedelta(hours=int(hour_val) - 1)
    hour0 = int(hour_val) - 1
    row.update({
        "hour_sin": np.sin(2 * np.pi * hour0 / 24),
        "hour_cos": np.cos(2 * np.pi * hour0 / 24),
        "day_sin": np.sin(2 * np.pi * dt.dayofweek / 7),
        "day_cos": np.cos(2 * np.pi * dt.dayofweek / 7),
        "month_sin": np.sin(2 * np.pi * dt.month / 12),
        "month_cos": np.cos(2 * np.pi * dt.month / 12),
        "is_weekend": int(dt.dayofweek >= 5),
        "target_hour_sin": np.sin(2 * np.pi * hour0 / 24),
        "target_hour_cos": np.cos(2 * np.pi * hour0 / 24),
        "target_day_sin": np.sin(2 * np.pi * dt.dayofweek / 7),
        "target_day_cos": np.cos(2 * np.pi * dt.dayofweek / 7),
        "target_is_weekend": int(dt.dayofweek >= 5),
    })
    return pd.DataFrame([{c: row.get(c, 0.0) if pd.notna(row.get(c, 0.0)) else 0.0 for c in x_cols}])


def apply_forecast_shape_adjustments(base_pred, hist, plant, hour_i, meta):
    if plant not in SHAPE_OPTIMIZED_PLANTS:
        return base_pred
    target = f"total_gen_{plant}"
    adjusted = float(base_pred)
    hourly = meta.get("hourly_residual_correction")
    if hourly:
        adjusted = float(apply_hourly_correction([adjusted], [hour_i], plant, hourly)[0])
    profile = meta.get("profile_blend") or FORECAST_PROFILE_BLEND_FALLBACK.get(plant)
    if profile and len(hist) >= 23:
        same_hour_yesterday = float(hist[target].iloc[-23])
        adjusted = float(apply_profile_blend([adjusted], [same_hour_yesterday], plant, profile)[0])
    return adjusted


def forecast_24h(raw_df, planned):
    forecast = empty_forecast_frame(planned)
    hist = raw_df.copy()
    baseline_status = {plant: latest_status(hist, plant) for plant in PLANTS}

    loaded = {}
    for plant in PLANTS:
        meta = json.loads(model_file(f"meta_{plant}.json").read_text())
        loaded[plant] = {
            "meta": meta,
            "model": tf.keras.models.load_model(model_file(f"rbfnn_{plant}.keras"), custom_objects={"RBFLayer": RBFLayer}, compile=False),
            "x_scaler": joblib.load(model_file(f"x_scaler_{plant}.pkl")),
            "y_scaler": joblib.load(model_file(f"y_scaler_{plant}.pkl")),
        }

    for step in range(24):
        date_i = planned.loc[step, "Date"]
        hour_i = int(planned.loc[step, "Hour"])
        new_row = hist.iloc[-1].copy()
        new_row["date"] = pd.to_datetime(date_i)
        new_row["time"] = hour_i
        new_row["datetime"] = pd.to_datetime(date_i) + pd.Timedelta(hours=hour_i - 1)

        for plant in PLANTS:
            target = f"total_gen_{plant}"
            bundle = loaded[plant]
            meta = bundle["meta"]
            x_row = feature_row_from_history(hist, plant, meta["X_cols"], date_i, hour_i)
            delta = float(predict_delta(bundle["model"], bundle["x_scaler"], bundle["y_scaler"], x_row)[0])
            last_val = float(hist[target].iloc[-1])
            base_pred = last_val + float(meta["best_shrinkage"]) * delta + float(meta.get("bias_correction_mw", 0.0))
            calibration = meta.get("bin_calibration")
            calibration_basis = [base_pred] if calibration and calibration.get("basis") == "base_pred" else [last_val]
            base_pred = float(apply_bin_calibration([base_pred], calibration_basis, plant, calibration)[0])
            base_pred = apply_forecast_shape_adjustments(base_pred, hist, plant, hour_i, meta)
            base_pred = float(np.clip(base_pred, last_val - float(meta["ramp_limit"]), last_val + float(meta["ramp_limit"])))
            base_pred = float(np.clip(base_pred, 0.0, CAPACITY_MW[plant] * 1.05))

            p_status = planned_status(planned, step, plant)
            ratio = availability_ratio(plant, baseline_status[plant], p_status)
            adjusted = float(np.clip(base_pred * ratio, 0.0, CAPACITY_MW[plant] * 1.05)) if ratio > 0 else 0.0
            unit_values = distribute_to_units(plant, adjusted, p_status)
            for unit_col, value in unit_values.items():
                forecast.loc[step, unit_col] = value
            forecast.loc[step, f"total_gen_{plant}"] = sum(unit_values.values())
            new_row[target] = base_pred
            for out_col, value in baseline_status[plant].items():
                new_row[out_col] = value

        hist = pd.concat([hist, pd.DataFrame([new_row])], ignore_index=True)
        forecast.loc[step, CASCADE_FORECAST_COLUMN] = forecast.loc[step, TOTAL_FORECAST_COLUMNS].sum()
    ordered_cols = ["Date", "Hour"] + UNIT_FORECAST_COLUMNS + TOTAL_FORECAST_COLUMNS + [CASCADE_FORECAST_COLUMN]
    forecast = forecast[ordered_cols]
    return forecast


def load_actual_next_day_generation(path):
    if not path.exists():
        return None
    actual = pd.read_excel(path, header=1)
    actual.columns = [str(c).strip() for c in actual.columns]
    if "TIME" not in actual.columns:
        return None
    out = pd.DataFrame({"Hour": actual["TIME"]})
    for plant in PLANTS:
        src = f"{plant.upper()} ACT"
        if src in actual.columns:
            out[plant.upper()] = actual[src]
    actual = out
    actual["Hour"] = pd.to_numeric(actual["Hour"], errors="coerce")
    actual = actual.dropna(subset=["Hour"]).copy()
    actual["Hour"] = actual["Hour"].astype(int)
    actual = actual[(actual["Hour"] >= 1) & (actual["Hour"] <= 24)].head(24)
    for col in [c for c in actual.columns if c != "Hour"]:
        actual[col] = pd.to_numeric(actual[col], errors="coerce")
    return actual


def save_actual_forecast_diagnostics(forecast):
    actual = load_actual_next_day_generation(ACTUAL_NEXT_DAY_PATH)
    if actual is None or actual.empty:
        print(f"Actual next-day file not found or not parseable; skipped diagnostics: {ACTUAL_NEXT_DAY_PATH}")
        return

    diag_dir = DIAGNOSTICS_DIR
    plot_dir = PLOTS_DIR
    merged = actual.merge(forecast, on="Hour", suffixes=("_actual", "_forecast"))
    detail_sheets = {}
    summary_rows = []

    for plant in ["AGUS5", "AGUS7"]:
        actual_col = f"{plant}_actual"
        forecast_col = f"{plant}_forecast"
        if actual_col not in merged.columns or forecast_col not in merged.columns:
            continue
        detail = pd.DataFrame({
            "Hour": merged["Hour"],
            "actual_mw": merged[actual_col].astype(float),
            "forecast_mw": merged[forecast_col].astype(float),
        })
        detail["absolute_error_mw"] = (detail["actual_mw"] - detail["forecast_mw"]).abs()
        detail["ape_percent"] = detail["absolute_error_mw"] / detail["actual_mw"].abs().clip(lower=1e-6) * 100.0
        detail_sheets[plant] = detail
        summary_rows.append({
            "plant": plant.lower(),
            "actual_file": str(ACTUAL_NEXT_DAY_PATH),
            "mape": float(detail["ape_percent"].mean()),
            "mae": float(detail["absolute_error_mw"].mean()),
            "rmse": math.sqrt(float(np.mean(np.square(detail["actual_mw"] - detail["forecast_mw"])))),
            "ape_le_3_count": int((detail["ape_percent"] <= 3.0).sum()),
            "max_ape": float(detail["ape_percent"].max()),
            "trend_match_rate": shape_metrics(detail["actual_mw"], detail["forecast_mw"])["shape_trend_match_rate"],
            "delta_mae": shape_metrics(detail["actual_mw"], detail["forecast_mw"])["shape_delta_mae"],
        })

        plt.figure(figsize=(9, 5))
        plt.plot(detail["Hour"], detail["actual_mw"], marker="o", label="Actual")
        plt.plot(detail["Hour"], detail["forecast_mw"], marker="o", label="Forecast")
        plt.xticks(range(1, 25))
        plt.xlabel("Hour")
        plt.ylabel("Generation (MW)")
        plt.title(f"{plant} Actual vs Optimized RBFNN Forecast")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_dir / f"{plant.lower()}_actual_vs_forecast_july_1_2025.png", dpi=300)
        plt.close()

    if not summary_rows:
        return
    out_path = diag_dir / "july_1_2025_actual_vs_optimized_rbfnn.xlsx"
    with pd.ExcelWriter(out_path) as writer:
        pd.DataFrame(summary_rows).to_excel(writer, sheet_name="summary", index=False)
        for sheet, detail in detail_sheets.items():
            detail.to_excel(writer, sheet_name=sheet.lower(), index=False)
    print("Saved actual-vs-forecast diagnostics:", out_path)


def load_latest_inputs():
    if not CLEAN_PATH.exists() or not PLANNED_PATH.exists():
        raise FileNotFoundError("Run Thesis Forecasting/scripts/cell1_clean_data.py first.")

    raw_df = pd.read_parquet(CLEAN_PATH)
    raw_df = add_runtime_compatibility_columns(rebuild_datetime(raw_df))
    planned = load_planned()
    return raw_df, planned


def evaluate_saved_models(raw_df):
    feat_df = add_features(raw_df)
    summary_rows = []
    testing_prediction_rows = []

    for plant in PLANTS:
        target = f"total_gen_{plant}"
        meta = json.loads(model_file(f"meta_{plant}.json").read_text())
        y_col = meta["y_col"]
        x_cols = meta["X_cols"]
        data = feat_df.dropna(subset=[y_col] + x_cols).copy()
        _, val_df, test_df = chronological_split(data)

        model = tf.keras.models.load_model(model_file(f"rbfnn_{plant}.keras"), custom_objects={"RBFLayer": RBFLayer}, compile=False)
        x_scaler = joblib.load(model_file(f"x_scaler_{plant}.pkl"))
        y_scaler = joblib.load(model_file(f"y_scaler_{plant}.pkl"))

        x_val = x_scaler.transform(val_df[x_cols].values.astype(np.float32))
        x_test = x_scaler.transform(test_df[x_cols].values.astype(np.float32))
        val_delta = y_scaler.inverse_transform(model.predict(x_val, verbose=0)).flatten()
        test_delta = y_scaler.inverse_transform(model.predict(x_test, verbose=0)).flatten()

        val_current = val_df[target].values.astype(float)
        test_current = test_df[target].values.astype(float)
        val_actual = val_df[f"{target}_tplus1"].values.astype(float)
        test_actual = test_df[f"{target}_tplus1"].values.astype(float)

        shrinkage = float(meta.get("best_shrinkage", 1.0))
        bias = float(meta.get("bias_correction_mw", 0.0))
        val_pred = apply_level_prediction(val_current, val_delta, plant, shrinkage, bias)
        test_pred = apply_level_prediction(test_current, test_delta, plant, shrinkage, bias)

        calibration = meta.get("bin_calibration")
        val_basis = val_pred if calibration and calibration.get("basis") == "base_pred" else val_current
        test_basis = test_pred if calibration and calibration.get("basis") == "base_pred" else test_current
        val_pred = apply_bin_calibration(val_pred, val_basis, plant, calibration)
        test_pred = apply_bin_calibration(test_pred, test_basis, plant, calibration)

        val_hours = target_hours_from_rows(val_df)
        test_hours = target_hours_from_rows(test_df)
        val_pred = apply_hourly_correction(val_pred, val_hours, plant, meta.get("hourly_residual_correction"))
        test_pred = apply_hourly_correction(test_pred, test_hours, plant, meta.get("hourly_residual_correction"))
        val_pred = apply_profile_blend(val_pred, same_hour_target_anchor(val_df, plant), plant, meta.get("profile_blend"))
        test_pred = apply_profile_blend(test_pred, same_hour_target_anchor(test_df, plant), plant, meta.get("profile_blend"))

        val_metrics = metrics_dict(val_actual, val_pred, plant)
        test_metrics = metrics_dict(test_actual, test_pred, plant)
        summary_rows.append({
            "model": "RBFNN",
            "plant": plant,
            "feature_count": len(x_cols),
            "val_operational_mape": val_metrics["operational_mape"],
            "val_mae": val_metrics["mae"],
            "val_rmse": val_metrics["rmse"],
            "val_r2": val_metrics["r2"],
            "test_operational_mape": test_metrics["operational_mape"],
            "test_mae": test_metrics["mae"],
            "test_rmse": test_metrics["rmse"],
            "test_r2": test_metrics["r2"],
        })

        test_datetimes = pd.to_datetime(test_df["datetime"]) + pd.Timedelta(hours=1)
        for dt_val, actual, predicted in zip(test_datetimes, test_actual, test_pred):
            testing_prediction_rows.append({
                "Date": dt_val.date(),
                "Hour": int(dt_val.hour) + 1,
                "datetime": dt_val,
                "plant": plant,
                "actual_generation": float(actual),
                "predicted_generation": float(predicted),
                "model": "RBFNN",
            })

        save_daily_metrics(val_df, val_actual, val_pred, plant, VALIDATION_METRICS_DIR / f"{plant}_validation_daily_metrics.xlsx")
        save_daily_metrics(test_df, test_actual, test_pred, plant, TESTING_METRICS_DIR / f"{plant}_testing_daily_metrics.xlsx")

    metrics_path = OVERALL_METRICS_DIR / "rbfnn_validation_testing_metrics.xlsx"
    pd.DataFrame(summary_rows)[METRICS_COLUMNS].to_excel(metrics_path, index=False)
    format_excel(metrics_path)

    predictions_path = OVERALL_METRICS_DIR / "rbfnn_testing_predictions.xlsx"
    pd.DataFrame(testing_prediction_rows)[TESTING_PREDICTION_COLUMNS].to_excel(predictions_path, index=False)
    format_excel(predictions_path)
    print("Saved:", metrics_path)
    print("Saved:", predictions_path)


def run_forecast_only():
    raw_df, planned = load_latest_inputs()
    evaluate_saved_models(raw_df)
    forecast = forecast_24h(raw_df, planned)
    output_forecast = format_forecast_output(forecast)
    xlsx_path = RBFNN_FORECAST_DIR / "Day_Ahead_24H_RBFNN_Forecast.xlsx"
    csv_path = RBFNN_FORECAST_DIR / "Day_Ahead_24H_RBFNN_Forecast.csv"
    output_forecast.to_excel(xlsx_path, index=False)
    output_forecast.to_csv(csv_path, index=False)
    format_excel(xlsx_path)
    print("Optimized RBFNN forecast complete")
    print("Saved:", xlsx_path)
    print("Saved:", csv_path)


def run_training_and_forecast():
    raw_df, planned = load_latest_inputs()
    selected_plants = [arg.lower() for arg in sys.argv[sys.argv.index("--train") + 1:] if not arg.startswith("--")]
    if selected_plants:
        invalid = sorted(set(selected_plants) - set(PLANTS))
        if invalid:
            raise ValueError(f"Unknown plant(s): {invalid}. Valid plants: {PLANTS}")
    feat_df = add_features(raw_df)
    summary_rows = []
    testing_prediction_rows = []

    for plant in PLANTS:
        if selected_plants and plant not in selected_plants:
            continue
        print(f"\n===== OPTIMIZED RBFNN {plant.upper()} =====")
        target = f"total_gen_{plant}"
        y_col = f"{target}_delta_tplus1"
        x_cols = feature_columns_for(feat_df, plant)
        data = feat_df.dropna(subset=[y_col] + x_cols).copy()
        train_df, val_df, test_df = chronological_split(data)

        x_scaler = StandardScaler()
        y_scaler = StandardScaler()
        x_train = x_scaler.fit_transform(train_df[x_cols].values.astype(np.float32))
        x_val = x_scaler.transform(val_df[x_cols].values.astype(np.float32))
        x_test = x_scaler.transform(test_df[x_cols].values.astype(np.float32))
        y_train = y_scaler.fit_transform(train_df[[y_col]].values.astype(np.float32))
        y_val_scaled = y_scaler.transform(val_df[[y_col]].values.astype(np.float32))

        val_current = val_df[target].values.astype(float)
        test_current = test_df[target].values.astype(float)
        val_actual = val_df[f"{target}_tplus1"].values.astype(float)
        test_actual = test_df[f"{target}_tplus1"].values.astype(float)

        best = None
        for n_centers in SEARCH_CENTER_COUNTS:
            for learning_rate in SEARCH_LEARNING_RATES:
                tf.keras.backend.clear_session()
                model = build_model(x_train.shape[1], n_centers=n_centers, learning_rate=learning_rate)
                init_centers(model, x_train, n_centers=n_centers)
                history = model.fit(
                    x_train,
                    y_train,
                    validation_data=(x_val, y_val_scaled),
                    epochs=EPOCHS,
                    batch_size=BATCH_SIZE,
                    verbose=0,
                    callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)],
                )
                val_delta_pred = y_scaler.inverse_transform(model.predict(x_val, verbose=0)).flatten()
                test_delta_pred = y_scaler.inverse_transform(model.predict(x_test, verbose=0)).flatten()

                for shrinkage in SHRINKAGE_GRID:
                    val_pred_candidate, bias = calibrated_level_predictions(val_current, val_delta_pred, val_actual, plant, shrinkage)
                    val_m = metrics_dict(val_actual, val_pred_candidate, plant)
                    score = candidate_score(val_m)
                    if best is None or score < best["score"]:
                        test_pred_candidate = apply_level_prediction(test_current, test_delta_pred, plant, shrinkage, bias)
                        best = {
                            "score": score,
                            "model": model,
                            "history": history,
                            "n_centers": n_centers,
                            "learning_rate": learning_rate,
                            "shrinkage": shrinkage,
                            "bias": bias,
                            "val_pred": val_pred_candidate,
                            "test_pred": test_pred_candidate,
                        }

        model = best["model"]
        history = best["history"]
        shrinkage = best["shrinkage"]
        bias = best["bias"]
        val_pred = best["val_pred"]
        test_pred = best["test_pred"]
        bin_calibration, val_pred = tune_bin_calibration(plant, val_current, val_actual, val_pred)
        test_basis = test_pred if bin_calibration and bin_calibration.get("basis") == "base_pred" else test_current
        test_pred = apply_bin_calibration(test_pred, test_basis, plant, bin_calibration)
        val_hours = target_hours_from_rows(val_df)
        test_hours = target_hours_from_rows(test_df)
        hourly_correction, val_pred = tune_hourly_correction(plant, val_pred, val_actual, val_hours)
        test_pred = apply_hourly_correction(test_pred, test_hours, plant, hourly_correction)
        profile_blend, val_pred = tune_profile_blend(plant, val_pred, val_actual, same_hour_target_anchor(val_df, plant))
        test_pred = apply_profile_blend(test_pred, same_hour_target_anchor(test_df, plant), plant, profile_blend)
        val_metrics = metrics_dict(val_actual, val_pred, plant)
        test_metrics = metrics_dict(test_actual, test_pred, plant)
        val_shape = shape_metrics(val_actual, val_pred)
        test_shape = shape_metrics(test_actual, test_pred)
        limit = ramp_limit(train_df[target])

        row = {
            "model": "RBFNN",
            "plant": plant,
            "feature_count": len(x_cols),
            "val_operational_mape": val_metrics["operational_mape"],
            "val_mae": val_metrics["mae"],
            "val_rmse": val_metrics["rmse"],
            "val_r2": val_metrics["r2"],
            "test_operational_mape": test_metrics["operational_mape"],
            "test_mae": test_metrics["mae"],
            "test_rmse": test_metrics["rmse"],
            "test_r2": test_metrics["r2"],
        }
        summary_rows.append(row)

        test_datetimes = pd.to_datetime(test_df["datetime"]) + pd.Timedelta(hours=1)
        for dt_val, actual, predicted in zip(test_datetimes, test_actual, test_pred):
            testing_prediction_rows.append({
                "Date": dt_val.date(),
                "Hour": int(dt_val.hour) + 1,
                "datetime": dt_val,
                "plant": plant,
                "actual_generation": float(actual),
                "predicted_generation": float(predicted),
                "model": "RBFNN",
            })

        atomic_save_keras_model(model, MODEL_DIR / f"rbfnn_{plant}.keras")
        joblib.dump(x_scaler, MODEL_DIR / f"x_scaler_{plant}.pkl")
        joblib.dump(y_scaler, MODEL_DIR / f"y_scaler_{plant}.pkl")
        history_path = save_training_history(history, plant)
        (MODEL_DIR / f"meta_{plant}.json").write_text(json.dumps({
            "plant": plant,
            "target": target,
            "y_col": y_col,
            "X_cols": x_cols,
            "capacity_mw": CAPACITY_MW[plant],
            "operational_mape_threshold_mw": operational_threshold(plant),
            "best_shrinkage": shrinkage,
            "bias_correction_mw": bias,
            "bin_calibration": bin_calibration,
            "hourly_residual_correction": hourly_correction,
            "profile_blend": profile_blend,
            "ramp_limit": limit,
            "selected_centers": best["n_centers"],
            "selected_learning_rate": best["learning_rate"],
            "selection_metric": "lowest validation operational MAPE, with Agus 5/7 post-calibrated for hourly residual and same-hour-yesterday shape tracking",
            "model_type": "RBFNN residual/delta model anchored to persistence",
        }, indent=2))

        save_daily_metrics(val_df, val_actual, val_pred, plant, VALIDATION_METRICS_DIR / f"{plant}_validation_daily_metrics.xlsx")
        save_daily_metrics(test_df, test_actual, test_pred, plant, TESTING_METRICS_DIR / f"{plant}_testing_daily_metrics.xlsx")
        print("Saved:", history_path)
        print(pd.DataFrame([row]).to_string(index=False))

    summary = pd.DataFrame(summary_rows)[METRICS_COLUMNS]
    summary_path = OVERALL_METRICS_DIR / "rbfnn_validation_testing_metrics.xlsx"
    if selected_plants and summary_path.exists():
        existing_summary = pd.read_excel(summary_path)
        existing_summary = existing_summary[~existing_summary["plant"].isin(selected_plants)]
        summary = pd.concat([existing_summary, summary], ignore_index=True)
        summary["plant_order"] = summary["plant"].map({plant: idx for idx, plant in enumerate(PLANTS)})
        summary = summary.sort_values("plant_order").drop(columns=["plant_order"]).reset_index(drop=True)
    summary.to_excel(summary_path, index=False)
    format_excel(summary_path)
    testing_path = OVERALL_METRICS_DIR / "rbfnn_testing_predictions.xlsx"
    testing_predictions = pd.DataFrame(testing_prediction_rows)[TESTING_PREDICTION_COLUMNS]
    testing_predictions.to_excel(testing_path, index=False)
    format_excel(testing_path)
    if not selected_plants or all(model_file(f"rbfnn_{plant}.keras").exists() for plant in PLANTS):
        forecast = forecast_24h(raw_df, planned)
        output_forecast = format_forecast_output(forecast)
        forecast_xlsx = RBFNN_FORECAST_DIR / "Day_Ahead_24H_RBFNN_Forecast.xlsx"
        forecast_csv = RBFNN_FORECAST_DIR / "Day_Ahead_24H_RBFNN_Forecast.csv"
        output_forecast.to_excel(forecast_xlsx, index=False)
        output_forecast.to_csv(forecast_csv, index=False)
        format_excel(forecast_xlsx)
        print("Saved:", forecast_xlsx)
        print("Saved:", forecast_csv)

    original_path = PROJECT_DIR / "data" / "outputs" / "03_metadata" / "validation_testing_metrics" / "rbfnn_validation_testing_metrics.xlsx"
    comparison = summary.copy()
    if original_path.exists():
        original = pd.read_excel(original_path)
        rename_map = {
            "val_mape": "original_val_mape",
            "val_mae": "original_val_mae",
            "val_rmse": "original_val_rmse",
            "val_r2": "original_val_r2",
            "test_mape": "original_test_mape",
            "test_mae": "original_test_mae",
            "test_rmse": "original_test_rmse",
            "test_r2": "original_test_r2",
        }
        original = original.rename(columns=rename_map)
        keep_cols = ["plant"] + [c for c in rename_map.values() if c in original.columns]
        comparison = comparison.merge(original[keep_cols], on="plant", how="left")
        comparison["original_summary_available"] = comparison["original_test_r2"].notna()
    else:
        comparison["original_summary_available"] = False
    comparison = comparison.fillna("not_available")
    comparison.to_excel(OVERALL_METRICS_DIR / "optimized_vs_original_summary.xlsx", index=False)

    benchmark_path = OVERALL_METRICS_DIR / "optimized_benchmark_validation_testing_metrics.xlsx"
    if benchmark_path.exists():
        benchmarks = pd.read_excel(benchmark_path)
        comparison_rows = []
        for _, rbfnn_row in summary.iterrows():
            plant = rbfnn_row["plant"]
            plant_bench = benchmarks[benchmarks["plant"] == plant].copy()
            best_val = plant_bench.loc[plant_bench["val_operational_mape"].idxmin()]
            best_test = plant_bench.loc[plant_bench["test_operational_mape"].idxmin()]
            val_margin = float(best_val["val_operational_mape"] - rbfnn_row["val_operational_mape"])
            test_margin = float(best_test["test_operational_mape"] - rbfnn_row["test_operational_mape"])
            comparison_rows.append({
                "plant": plant,
                "rbfnn_val_operational_mape": rbfnn_row["val_operational_mape"],
                "best_benchmark_val_model": best_val["model"],
                "best_benchmark_val_operational_mape": best_val["val_operational_mape"],
                "val_mape_margin": val_margin,
                "rbfnn_test_operational_mape": rbfnn_row["test_operational_mape"],
                "best_benchmark_test_model": best_test["model"],
                "best_benchmark_test_operational_mape": best_test["test_operational_mape"],
                "test_mape_margin": test_margin,
                "rbfnn_wins_val_mape": val_margin > 0,
                "rbfnn_wins_test_mape": test_margin > 0,
                "rbfnn_wins_both": val_margin > 0 and test_margin > 0,
            })
        benchmark_comparison = pd.DataFrame(comparison_rows)
        benchmark_comparison_path = OVERALL_METRICS_DIR / "rbfnn_vs_benchmark_mape_comparison.xlsx"
        benchmark_comparison.to_excel(benchmark_comparison_path, index=False)
        print("\nRBFNN vs benchmark MAPE comparison:")
        print(benchmark_comparison.to_string(index=False))
        print("Saved:", benchmark_comparison_path)
    else:
        print("Benchmark metrics not found; run optimized_cell3_benchmark.py before final benchmark comparison.")

    print("\nOptimized RBFNN summary:")
    print(summary.to_string(index=False))
    print("Saved:", summary_path)
    print("Saved:", testing_path)


def main():
    if "--train" in sys.argv:
        run_training_and_forecast()
    else:
        run_forecast_only()


if __name__ == "__main__":
    main()

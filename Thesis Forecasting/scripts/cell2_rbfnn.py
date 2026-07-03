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
import io
import math
import os
import re
import shutil
import sys
import warnings
import zipfile
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import joblib
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
np.random.seed(42)
if hasattr(tf, "get_logger"):
    tf.get_logger().setLevel("ERROR")
elif hasattr(tf, "compat") and hasattr(tf.compat, "v1"):
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
if hasattr(tf, "random") and hasattr(tf.random, "set_seed"):
    tf.random.set_seed(42)

# ============================================================
# PATH AND MODEL CONFIGURATION
# ============================================================

# Defines project paths for cleaned data, saved RBFNN models, metrics, and forecast outputs.
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
RBFNN_META_DIR = META_DIR / "rbfnn"
VALIDATION_METRICS_DIR = META_DIR / "validation_metrics"
TESTING_METRICS_DIR = META_DIR / "testing_metrics"
OVERALL_METRICS_DIR = META_DIR / "overall_metrics"
DAY_AHEAD_BACKTEST_DIR = META_DIR / "day_ahead_backtest"
OUTAGE_INFORMED_DIR = META_DIR / "day_ahead_outage_informed"
TRAINING_HISTORY_DIR = META_DIR / "training_validation_loss"
DIAGNOSTICS_DIR = OVERALL_METRICS_DIR / "actual_forecast_diagnostics"
PLOTS_DIR = OVERALL_METRICS_DIR / "plots"

for folder in [
    CLEANED_DATA_DIR,
    OUTAGES_DIR,
    RBFNN_FORECAST_DIR,
    MODEL_DIR,
    RBFNN_META_DIR,
    OVERALL_METRICS_DIR,
    DAY_AHEAD_BACKTEST_DIR,
    OUTAGE_INFORMED_DIR,
    TRAINING_HISTORY_DIR,
    DIAGNOSTICS_DIR,
    PLOTS_DIR,
]:
    folder.mkdir(parents=True, exist_ok=True)

CLEAN_PATH = CLEANED_DATA_DIR / "cleaned_hourly_data.parquet"
PLANNED_PATH = OUTAGES_DIR / "Planned_Outages_Input.xlsx"

# Stores plant structure, capacities, output schemas, and model search settings.
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
SEARCH_CENTER_COUNTS = [40, 60, 80, 120, 160, 180, 220]
SEARCH_LEARNING_RATES = [0.0012, 0.001, 0.0007, 0.0005, 0.0003, 0.0002]
SEARCH_GAMMA_INITS = [0.5, 1.0, 2.0]
SEARCH_BATCH_SIZES = [16, 32]
SEARCH_PATIENCES = [8, 12, 16]
RBFNN_FINALIST_COUNT = 10
R2_WEIGHT = 20.0
NEGATIVE_R2_PENALTY = 25.0
SHRINKAGE_GRID = [0.05, 0.10, 0.20, 0.35, 0.50, 0.75, 1.00, 1.15]
BIN_CALIBRATION_PLANTS = set(PLANTS)
BIN_CALIBRATION_QUANTILES = list(range(3, 21))
BIN_CALIBRATION_QUANTILES_BY_PLANT = {"agus1": list(range(3, 41))}
BIN_CALIBRATION_SCALES = [round(x, 2) for x in np.arange(0.50, 2.55, 0.05)]
BIN_CALIBRATION_BASIS = {"agus1": "base_pred", "agus5": "current"}
SHAPE_OPTIMIZED_PLANTS = set(PLANTS)
PROFILE_BLEND_GRID = [0.0, 0.10, 0.20, 0.30, 0.40, 0.55, 0.70, 0.85]
HOURLY_CORRECTION_SCALE_GRID = [0.0, 0.25, 0.50, 0.75, 1.00, 1.25]
ACTUAL_NEXT_DAY_PATH = PROJECT_DIR / "july 1, 2025.xlsx"
FEATURE_HISTORY_WINDOW = 240
DAY_AHEAD_EVALUATION_TYPE = "rolling_24h_day_ahead_backtest"
OUTAGE_INPUT_TYPE = "last_known_outage_status_before_forecast_day"
OUTAGE_INPUT_NOTE = (
    "Validation/testing day-ahead backtests persist only unit outage/status columns "
    "from the latest historical row before each forecast day. Same-day actual unit "
    "outage/status values are not used because archived planned outage schedules "
    "were unavailable."
)
HYDROLOGIC_INPUT_TYPE = "lagged_or_persistence_context"
HYDROLOGIC_INPUT_NOTE = (
    "Forecast-day rainfall and Lake Lanao outflow were not taken from actual "
    "same-day records; lagged or most recent known values were used to avoid "
    "future leakage."
)
DAY_AHEAD_THESIS_NOTE = (
    "Each validation and testing day was forecast as a complete 24-hour horizon "
    "using only data available before the forecast day. Lag and rolling features "
    "were reconstructed from historical values and recursive predictions, while "
    "actual forecast-day generation was used only after prediction for metrics."
)
FORECAST_PROFILE_BLEND_FALLBACK = {}


# ============================================================
# OUTPUT AND FILE HELPERS
# ============================================================

# Saves Keras models safely and falls back to H5 if the target file is locked.
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


# Formats Excel outputs for easier review and thesis reporting.
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


def leakage_feature_audit(feature_columns, plant):
    forbidden_patterns = [
        r"_tplus1$",
        r"_delta_tplus1$",
        r"actual",
        r"predicted",
        r"forecast",
        r"testing",
        r"validation",
    ]
    forbidden = [
        col for col in feature_columns
        if any(re.search(pattern, col, flags=re.IGNORECASE) for pattern in forbidden_patterns)
    ]
    return {
        "plant": plant,
        "feature_count": int(len(feature_columns)),
        "leakage_check": "PASS" if not forbidden else "FAIL",
        "forbidden_feature_columns": forbidden,
    }


def save_leakage_audit(per_plant_results):
    audit_path = META_DIR / "leakage_audit.json"
    overall_pass = all(item["leakage_check"] == "PASS" for item in per_plant_results)
    audit = {
        "split_strategy": "chronological 70/15/15",
        "scaler_fitting": "training set only",
        "model_training": "training set only",
        "calibration_selection": "validation set only",
        "testing_use": "final evaluation only",
        "feature_leakage_check": "PASS" if overall_pass else "FAIL",
        "day_ahead_mode": "recursive 24-hour forecasting",
        "future_data_used": False,
        "recursive_forecast_note": "Each forecast hour is appended to the history before building features for the next forecast hour.",
        "unit_outage_inputs": "validation/testing use only last-known unit outage/status before each forecast day; same-day actual unit outage/status is excluded",
        "planned_outage_inputs": "only the operational 24-hour forecast mode may read a planned outage template known before the forecast day",
        "per_plant_feature_audit_results": per_plant_results,
    }
    audit_path.write_text(json.dumps(audit, indent=2))
    print("Saved:", audit_path)


def save_rbfnn_calibration_report(rows):
    report_path = RBFNN_META_DIR / "rbfnn_calibration_report.xlsx"
    columns = [
        "plant",
        "feature_count",
        "selected centers",
        "selected learning rate",
        "selected shrinkage",
        "bias correction",
        "bin calibration used",
        "hourly residual correction used",
        "profile blending used",
        "ramp limit used",
        "capacity/outage adjustment used",
        "derived from",
        "testing used for calibration",
    ]
    pd.DataFrame(rows)[columns].to_excel(report_path, index=False)
    format_excel(report_path)
    print("Saved:", report_path)


def update_model_selection_audit(section, rows):
    path = OVERALL_METRICS_DIR / "model_selection_audit.json"
    audit = {}
    if path.exists():
        audit = json.loads(path.read_text())
    existing_rows = audit.get(section, [])
    if existing_rows and rows:
        merged = {
            (item.get("model"), item.get("plant")): item
            for item in existing_rows
        }
        for item in rows:
            merged[(item.get("model"), item.get("plant"))] = item
        audit[section] = list(merged.values())
    else:
        audit[section] = rows or existing_rows
    audit["controls"] = {
        "split_strategy": "chronological 70/15/15",
        "primary_model": "RBFNN",
        "benchmark_models": ["Random Forest", "XGBoost"],
        "selection_rule": "lowest R2-aware validation score; score penalizes MAPE, RMSE, low R2, and especially negative R2",
        "day_ahead_selection": "RBFNN finalists are selected with R2-aware rolling 24-hour validation backtests",
        "testing_used_for_tuning": False,
        "metric_source": "real model predictions only",
    }
    path.write_text(json.dumps(audit, indent=2, default=str))
    print("Saved:", path)


# Saves epoch-by-epoch training and validation loss history.
def save_training_history(history, plant):
    history_df = pd.DataFrame(history.history)
    history_df.insert(0, "epoch", np.arange(1, len(history_df) + 1))
    history_path = TRAINING_HISTORY_DIR / f"{plant}_training_history.xlsx"
    try:
        history_df.to_excel(history_path, index=False)
        format_excel(history_path)
    except PermissionError:
        history_path = TRAINING_HISTORY_DIR / f"{plant}_training_history_regenerated.xlsx"
        history_df.to_excel(history_path, index=False)
        format_excel(history_path)
    return history_path


# Resolves current model artifacts first, then legacy artifacts if needed.
def model_file(name):
    current = MODEL_DIR / name
    current_h5 = current.with_suffix(".h5")
    if current.exists():
        return current
    if current_h5.exists():
        return current_h5
    legacy = LEGACY_MODEL_DIR / name
    legacy_h5 = legacy.with_suffix(".h5")
    if legacy.exists():
        return legacy
    if legacy_h5.exists():
        return legacy_h5
    return current


# Rebuilds the hourly datetime column from public date and time fields.
def rebuild_datetime(df):
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"])
    out["time"] = pd.to_numeric(out["time"]).astype(int)
    if "datetime" not in out.columns:
        out["datetime"] = out["date"] + pd.to_timedelta(out["time"] - 1, unit="h")
    else:
        out["datetime"] = pd.to_datetime(out["datetime"])
    return out.sort_values("datetime").reset_index(drop=True)


# ============================================================
# INPUT COMPATIBILITY HELPERS
# ============================================================

# Adds legacy-compatible aliases expected by older saved models.
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


# ============================================================
# RBFNN ARCHITECTURE AND TRAINING HELPERS
# ============================================================

# Implements the Gaussian radial basis hidden layer used by the RBFNN.
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


# Builds the RBFNN with an RBF hidden layer and linear output layer.
def build_model(input_dim, n_centers=N_CENTERS, learning_rate=LEARNING_RATE, gamma_init=1.0):
    inputs = tf.keras.Input(shape=(input_dim,))
    x = RBFLayer(n_centers, gamma_init=gamma_init)(inputs)
    outputs = tf.keras.layers.Dense(1, activation="linear")(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate), loss="mse")
    return model


# Initializes RBF centers using K-means clusters from the scaled training data.
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


def _keras_archive_model_config(model_path):
    with zipfile.ZipFile(model_path) as archive:
        return json.loads(archive.read("config.json"))


def _find_serialized_layer(config, class_name):
    stack = [config]
    while stack:
        item = stack.pop()
        if isinstance(item, dict):
            if item.get("class_name") == class_name:
                return item
            stack.extend(item.values())
        elif isinstance(item, list):
            stack.extend(item)
    return None


def load_rbfnn_model(model_path):
    try:
        return tf.keras.models.load_model(model_path, custom_objects={"RBFLayer": RBFLayer}, compile=False)
    except TypeError as exc:
        if "keras.src.models.functional" not in str(exc) and "InputLayer" not in str(exc):
            raise

    config = _keras_archive_model_config(model_path)
    input_layer = _find_serialized_layer(config, "InputLayer")
    rbf_layer = _find_serialized_layer(config, "RBFLayer")
    if input_layer is None or rbf_layer is None:
        raise RuntimeError(f"Cannot reconstruct incompatible Keras archive: {model_path}")

    input_shape = input_layer["config"].get("batch_shape") or input_layer["config"].get("batch_input_shape")
    input_dim = int(input_shape[-1])
    rbf_config = rbf_layer["config"]
    model = build_model(input_dim, n_centers=int(rbf_config["units"]), gamma_init=float(rbf_config.get("gamma_init", 1.0)))

    with zipfile.ZipFile(model_path) as archive:
        weights_bytes = archive.read("model.weights.h5")
    with h5py.File(io.BytesIO(weights_bytes), "r") as weights_file:
        rbf_vars = weights_file["layers"]["rbf_layer"]["vars"]
        dense_vars = weights_file["layers"]["dense"]["vars"]
        for layer in model.layers:
            if isinstance(layer, RBFLayer):
                layer.centers.assign(rbf_vars["0"][()])
                layer.log_gamma.assign(rbf_vars["1"][()])
            elif isinstance(layer, tf.keras.layers.Dense):
                layer.kernel.assign(dense_vars["0"][()])
                layer.bias.assign(dense_vars["1"][()])
    return model


# ============================================================
# METRICS AND DATA SPLITTING
# ============================================================

# Sets the minimum actual-generation level for meaningful MAPE calculation.
def operational_threshold(plant):
    return max(1.0, 0.01 * CAPACITY_MW[plant])


# Computes MAPE only on operational rows to avoid near-zero percentage distortion.
def operational_mape(y_true, y_pred, plant):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.abs(y_true) >= operational_threshold(plant)
    if not mask.any():
        return np.nan, 0, len(y_true)
    value = np.mean(np.abs((y_true[mask] - y_pred[mask]) / np.abs(y_true[mask]))) * 100.0
    return float(value), int(mask.sum()), int((~mask).sum())


# Collects validation/testing metrics used in Chapter 4 tables.
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


# Splits each plant series chronologically into training, validation, and testing sets.
def chronological_split(data):
    n = len(data)
    i1 = int(n * TRAIN_RATIO)
    i2 = int(n * (TRAIN_RATIO + VAL_RATIO))
    return data.iloc[:i1].copy(), data.iloc[i1:i2].copy(), data.iloc[i2:].copy()


# ============================================================
# FEATURE ENGINEERING
# ============================================================

# Builds time, lag, rolling, unit-share, outage, and upstream cascade features.
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

        unit_gen_cols = [c for c in out.columns if re.fullmatch(fr"gen_{plant}_unit\d+", c)]
        for unit_col in unit_gen_cols:
            out[f"{unit_col}_current"] = out[unit_col]
            total_safe = out[target].replace(0, np.nan)
            out[f"{unit_col}_share_current"] = (out[unit_col] / total_safe).replace([np.inf, -np.inf], np.nan).fillna(0.0)
            for lag in LAGS:
                out[f"{unit_col}_lag{lag}"] = out[unit_col].shift(lag)
                out[f"{unit_col}_share_lag{lag}"] = out[f"{unit_col}_share_current"].shift(lag)

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


# Selects the feature columns used by one plant-specific RBFNN model.
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
    cols += [
        c for c in data.columns
        if re.fullmatch(fr"gen_{plant}_unit\d+_(current|lag\d+|share_current|share_lag\d+)", c)
    ]
    cols += [c for c in data.columns if re.fullmatch(fr"out_{plant}_unit\d+", c)]
    cols += [f"{plant}_units_running", f"{plant}_plant_available"]
    cols += [c for c in data.columns if c.startswith(f"tot_{plant}") or c.startswith(f"elev_{plant}") or c.startswith(f"spill_{plant}")]
    cols += [c for c in data.columns if "_lag" in c and (c.startswith("tot_agus") or c.startswith("elev_agus") or "outflow" in c or c.startswith("rainfall"))]
    selected = list(dict.fromkeys([c for c in cols if c in data.columns]))
    assert not any(
        re.fullmatch(r"rainfall|.*outflow.*", c) and "_lag" not in c
        for c in selected
    ), "Contemporaneous rainfall/outflow must not be a feature"
    return selected


# ============================================================
# VALIDATION CALIBRATION AND FORECAST SHAPE ADJUSTMENTS
# ============================================================

# Saves daily error summaries for validation or testing splits.
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


# Converts predicted residual deltas back to level forecasts and tunes bias.
def calibrated_level_predictions(current, delta_pred, actual, plant, shrinkage):
    raw_pred = np.clip(current + shrinkage * delta_pred, 0.0, CAPACITY_MW[plant] * 1.05)
    bias = float(np.median(np.asarray(actual, dtype=float) - raw_pred))
    max_bias = operational_threshold(plant)
    bias = float(np.clip(bias, -max_bias, max_bias))
    pred = np.clip(raw_pred + bias, 0.0, CAPACITY_MW[plant] * 1.05)
    return pred, bias


# Applies the saved residual shrinkage and bias correction during evaluation or forecasting.
def apply_level_prediction(current, delta_pred, plant, shrinkage, bias):
    pred = np.clip(current + shrinkage * delta_pred + bias, 0.0, CAPACITY_MW[plant] * 1.05)
    return pred


# Creates quantile bins for value-based calibration.
def calibration_bins(values, q):
    _, bins = pd.qcut(pd.Series(values), q, duplicates="drop", retbins=True)
    return np.asarray(bins, dtype=float)


# Assigns values to calibration bins.
def bin_ids(values, bins):
    return np.digitize(np.asarray(values, dtype=float), bins[1:-1])


# Applies saved bin-level forecast corrections.
def apply_bin_calibration(pred, basis_values, plant, calibration):
    if not calibration:
        return np.asarray(pred, dtype=float)
    bins = np.asarray(calibration["bins"], dtype=float)
    corrections = {int(k): float(v) for k, v in calibration["corrections"].items()}
    scale = float(calibration["scale"])
    ids = bin_ids(basis_values, bins)
    adjustment = np.asarray([corrections.get(int(bin_id), 0.0) for bin_id in ids], dtype=float)
    return np.clip(np.asarray(pred, dtype=float) + scale * adjustment, 0.0, CAPACITY_MW[plant] * 1.05)


# Tunes value-bin calibration on validation predictions for selected plants.
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


# Ranks candidate configurations with a stronger penalty for poor R2.
def candidate_score(metrics):
    mape = metrics["operational_mape"]
    mape_score = float(mape) if pd.notna(mape) else float("inf")
    r2 = float(metrics["r2"]) if pd.notna(metrics["r2"]) else -999.0
    r2_penalty = max(0.0, -r2) * NEGATIVE_R2_PENALTY
    combined_score = mape_score + metrics["rmse"] + (1.0 - r2) * R2_WEIGHT + r2_penalty
    return (combined_score, -r2, mape_score, metrics["rmse"])


def rolling_selection_score(plant, metrics):
    return candidate_score(metrics)


def rbfnn_search_configs(plant):
    base = [
        {"n_centers": 40, "learning_rate": 0.0012, "gamma_init": 1.0, "batch_size": 32, "patience": 8},
        {"n_centers": 60, "learning_rate": 0.0010, "gamma_init": 1.0, "batch_size": 32, "patience": 8},
        {"n_centers": 60, "learning_rate": 0.0005, "gamma_init": 0.5, "batch_size": 16, "patience": 16},
        {"n_centers": 80, "learning_rate": 0.0010, "gamma_init": 0.5, "batch_size": 32, "patience": 12},
        {"n_centers": 80, "learning_rate": 0.0007, "gamma_init": 2.0, "batch_size": 16, "patience": 12},
        {"n_centers": 120, "learning_rate": 0.0010, "gamma_init": 1.0, "batch_size": 32, "patience": 12},
        {"n_centers": 120, "learning_rate": 0.0007, "gamma_init": 0.5, "batch_size": 16, "patience": 16},
        {"n_centers": 120, "learning_rate": 0.0005, "gamma_init": 2.0, "batch_size": 32, "patience": 16},
        {"n_centers": 160, "learning_rate": 0.0005, "gamma_init": 1.0, "batch_size": 32, "patience": 16},
        {"n_centers": 180, "learning_rate": 0.0005, "gamma_init": 1.0, "batch_size": 32, "patience": 16},
        {"n_centers": 180, "learning_rate": 0.0003, "gamma_init": 0.5, "batch_size": 16, "patience": 16},
        {"n_centers": 220, "learning_rate": 0.0002, "gamma_init": 0.5, "batch_size": 16, "patience": 20},
    ]
    plant_extra = {
        "agus1": [
            {"n_centers": 60, "learning_rate": 0.0007, "gamma_init": 2.0, "batch_size": 16, "patience": 12},
            {"n_centers": 80, "learning_rate": 0.0003, "gamma_init": 1.0, "batch_size": 16, "patience": 20},
            {"n_centers": 120, "learning_rate": 0.0002, "gamma_init": 0.5, "batch_size": 16, "patience": 24},
            {"n_centers": 160, "learning_rate": 0.0003, "gamma_init": 1.0, "batch_size": 16, "patience": 20},
            {"n_centers": 220, "learning_rate": 0.0002, "gamma_init": 1.0, "batch_size": 16, "patience": 24},
        ],
        "agus2": [
            {"n_centers": 40, "learning_rate": 0.0007, "gamma_init": 2.0, "batch_size": 16, "patience": 16},
            {"n_centers": 60, "learning_rate": 0.0003, "gamma_init": 1.0, "batch_size": 16, "patience": 20},
            {"n_centers": 80, "learning_rate": 0.0002, "gamma_init": 0.5, "batch_size": 16, "patience": 24},
            {"n_centers": 120, "learning_rate": 0.0002, "gamma_init": 1.0, "batch_size": 16, "patience": 24},
            {"n_centers": 160, "learning_rate": 0.0003, "gamma_init": 0.5, "batch_size": 16, "patience": 20},
            {"n_centers": 220, "learning_rate": 0.0002, "gamma_init": 2.0, "batch_size": 16, "patience": 24},
        ],
        "agus4": [
            {"n_centers": 80, "learning_rate": 0.0003, "gamma_init": 2.0, "batch_size": 16, "patience": 20},
            {"n_centers": 120, "learning_rate": 0.0002, "gamma_init": 0.5, "batch_size": 16, "patience": 24},
            {"n_centers": 160, "learning_rate": 0.0007, "gamma_init": 1.0, "batch_size": 32, "patience": 16},
            {"n_centers": 180, "learning_rate": 0.0002, "gamma_init": 1.0, "batch_size": 16, "patience": 24},
            {"n_centers": 220, "learning_rate": 0.0003, "gamma_init": 1.0, "batch_size": 16, "patience": 20},
            {"n_centers": 260, "learning_rate": 0.0002, "gamma_init": 0.5, "batch_size": 16, "patience": 24},
        ],
        "agus5": [
            {"n_centers": 80, "learning_rate": 0.0005, "gamma_init": 0.5, "batch_size": 16, "patience": 16},
            {"n_centers": 120, "learning_rate": 0.0003, "gamma_init": 1.0, "batch_size": 16, "patience": 16},
            {"n_centers": 220, "learning_rate": 0.0002, "gamma_init": 1.0, "batch_size": 16, "patience": 20},
        ],
        "agus6": [
            {"n_centers": 180, "learning_rate": 0.0007, "gamma_init": 0.5, "batch_size": 32, "patience": 16},
            {"n_centers": 220, "learning_rate": 0.0003, "gamma_init": 0.5, "batch_size": 16, "patience": 20},
        ],
        "agus7": [
            {"n_centers": 80, "learning_rate": 0.0005, "gamma_init": 1.0, "batch_size": 16, "patience": 16},
            {"n_centers": 120, "learning_rate": 0.0003, "gamma_init": 0.5, "batch_size": 16, "patience": 16},
            {"n_centers": 160, "learning_rate": 0.0002, "gamma_init": 1.0, "batch_size": 16, "patience": 20},
        ],
    }
    configs = base + plant_extra.get(plant, [])
    unique = []
    seen = set()
    for config in configs:
        key = tuple(sorted(config.items()))
        if key not in seen:
            unique.append(config)
            seen.add(key)
    return unique


# Gets the forecast target hour for each validation or testing row.
def target_hours_from_rows(df):
    return ((pd.to_numeric(df["time"]).astype(int) % 24) + 1).astype(int).values


# Uses same-hour-yesterday generation as a shape anchor when available.
def same_hour_target_anchor(df, plant):
    target = f"total_gen_{plant}"
    col = f"{target}_target_lag24"
    if col in df.columns:
        return df[col].values.astype(float)
    return df[target].shift(23).values.astype(float)


# Applies hourly residual corrections learned from validation behavior.
def apply_hourly_correction(pred, hours, plant, correction):
    out = np.asarray(pred, dtype=float).copy()
    if not correction:
        return out
    scale = float(correction.get("scale", 0.0))
    by_hour = {int(k): float(v) for k, v in correction.get("hour_corrections", {}).items()}
    offsets = np.asarray([by_hour.get(int(hour), 0.0) for hour in hours], dtype=float)
    return np.clip(out + scale * offsets, 0.0, CAPACITY_MW[plant] * 1.05)


# Tunes hour-specific residual corrections for shape-optimized plants.
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


# Blends forecasts toward same-hour-yesterday shape anchors.
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


# Tunes same-hour-yesterday profile blending on validation data.
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


# Measures forecast shape behavior beyond standard point-error metrics.
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


# ============================================================
# OUTAGE-AWARE UNIT ALLOCATION
# ============================================================

# Extracts the unit number from an outage/status column.
def unit_from_outage_col(col):
    match = re.search(r"unit\d+", col)
    return match.group(0) if match else None


# Reads the latest unit availability from historical cleaned data.
def latest_status(hist_df, plant):
    cols = [c for c in hist_df.columns if re.fullmatch(fr"out_{plant}_unit\d+", c)]
    latest = hist_df.iloc[-1]
    return {c: float(latest[c]) for c in cols}


# Reads planned unit availability for one forecast hour.
def planned_status(planned, idx, plant):
    cols = [c for c in planned.columns if re.fullmatch(fr"out_{plant}_unit\d+", c)]
    return {c: float(planned.loc[idx, c]) for c in cols}


# Compares planned available capacity against the latest observed baseline.
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


# Lists unit-generation columns for one plant.
def unit_generation_columns(plant):
    return [f"gen_{plant}_{unit}" for unit in UNIT_CAPACITY[plant]]


# Builds the matching outage/status column name for one plant unit.
def unit_status_col(plant, unit):
    return f"out_{plant}_{unit}"


# Sums available unit capacity under the planned outage status.
def available_capacity(plant, status):
    return sum(
        capacity
        for unit, capacity in UNIT_CAPACITY[plant].items()
        if status.get(unit_status_col(plant, unit), 1.0) > 0
    )


def all_units_available_status(plant):
    return {unit_status_col(plant, unit): 1.0 for unit in UNIT_CAPACITY[plant]}


# Computes historical unit generation shares within a plant total.
def _unit_share_frame(plant, hist):
    unit_cols = unit_generation_columns(plant)
    total = hist[f"total_gen_{plant}"].replace(0, np.nan)
    shares = hist[unit_cols].clip(lower=0.0).div(total, axis=0)
    return shares.replace([np.inf, -np.inf], np.nan)


# Learns unit allocation weights from recent, current, and same-hour unit shares.
def learned_unit_weights(plant, hist, status, forecast_hour):
    unit_cols = unit_generation_columns(plant)
    available_cols = [
        f"gen_{plant}_{unit}"
        for unit in UNIT_CAPACITY[plant]
        if status.get(unit_status_col(plant, unit), 1.0) > 0
    ]
    weights = pd.Series(0.0, index=unit_cols, dtype=float)
    if not available_cols:
        return weights

    shares = _unit_share_frame(plant, hist)
    recent = shares.tail(168)
    components = []
    component_weights = []

    current = shares.iloc[-1][available_cols].dropna() if not shares.empty else pd.Series(dtype=float)
    if not current.empty and current.sum() > 0:
        components.append(current / current.sum())
        component_weights.append(0.50)

    median_recent = recent[available_cols].median(skipna=True).dropna()
    if not median_recent.empty and median_recent.sum() > 0:
        components.append(median_recent / median_recent.sum())
        component_weights.append(0.35)

    if "time" in hist.columns:
        hour_series = pd.to_numeric(hist["time"], errors="coerce").astype("Int64")
        same_hour = hist.loc[(hour_series == int(forecast_hour)).fillna(False)]
        same_hour_recent = _unit_share_frame(plant, same_hour.tail(56)) if not same_hour.empty else pd.DataFrame()
        same_hour_median = same_hour_recent[available_cols].median(skipna=True).dropna() if not same_hour_recent.empty else pd.Series(dtype=float)
        if not same_hour_median.empty and same_hour_median.sum() > 0:
            components.append(same_hour_median / same_hour_median.sum())
            component_weights.append(0.15)

    if components:
        combined = sum(w * c.reindex(available_cols).fillna(0.0) for w, c in zip(component_weights, components))
        combined = combined / combined.sum() if combined.sum() > 0 else combined
    else:
        capacity = pd.Series({f"gen_{plant}_{unit}": cap for unit, cap in UNIT_CAPACITY[plant].items()})
        combined = capacity[available_cols] / capacity[available_cols].sum()

    weights.loc[available_cols] = combined.reindex(available_cols).fillna(0.0)
    if weights.sum() <= 0:
        capacity = pd.Series({f"gen_{plant}_{unit}": cap for unit, cap in UNIT_CAPACITY[plant].items()})
        weights.loc[available_cols] = capacity[available_cols] / capacity[available_cols].sum()
    else:
        weights = weights / weights.sum()
    return weights


# Allocates a plant-level forecast to available units without exceeding capacities.
def allocate_with_unit_caps(plant, total_generation, weights, status):
    unit_values = {col: 0.0 for col in unit_generation_columns(plant)}
    available_cols = [
        f"gen_{plant}_{unit}"
        for unit in UNIT_CAPACITY[plant]
        if status.get(unit_status_col(plant, unit), 1.0) > 0
    ]
    if not available_cols:
        return unit_values

    caps = pd.Series({f"gen_{plant}_{unit}": cap for unit, cap in UNIT_CAPACITY[plant].items()}, dtype=float)
    total = float(np.clip(total_generation, 0.0, caps[available_cols].sum()))
    remaining = total
    alloc = pd.Series(0.0, index=available_cols, dtype=float)
    active = list(available_cols)
    active_weights = weights.reindex(active).fillna(0.0)
    if active_weights.sum() <= 0:
        active_weights = caps[active] / caps[active].sum()

    for _ in range(len(active) + 1):
        if remaining <= 1e-9 or not active:
            break
        active_weights = active_weights / active_weights.sum() if active_weights.sum() > 0 else caps[active] / caps[active].sum()
        proposal = active_weights * remaining
        headroom = caps[active] - alloc[active]
        capped = proposal >= headroom
        alloc.loc[active] += np.minimum(proposal, headroom)
        remaining = total - float(alloc.sum())
        active = [col for col in active if not capped.get(col, False) and caps[col] - alloc[col] > 1e-9]
        active_weights = weights.reindex(active).fillna(0.0)

    if remaining > 1e-6:
        headroom = (caps[available_cols] - alloc[available_cols]).clip(lower=0.0)
        if headroom.sum() > 0:
            alloc.loc[available_cols] += remaining * headroom / headroom.sum()

    for col, value in alloc.items():
        unit_values[col] = float(np.clip(value, 0.0, caps[col]))
    return unit_values


# Distributes one plant forecast into unit-level generation values.
def distribute_to_units(plant, plant_forecast, status, hist, forecast_hour, return_base=False):
    base_status = all_units_available_status(plant)
    weights = learned_unit_weights(plant, hist, base_status, forecast_hour)
    base_values = allocate_with_unit_caps(plant, plant_forecast, weights, base_status)
    unit_values = dict(base_values)
    for unit in UNIT_CAPACITY[plant]:
        if status.get(unit_status_col(plant, unit), 1.0) <= 0:
            unit_values[f"gen_{plant}_{unit}"] = 0.0
    if return_base:
        return unit_values, base_values
    return unit_values


# Enforces outages, unit caps, and plant-total consistency in the forecast table.
def validate_and_fix_unit_forecast(forecast, planned):
    for idx in forecast.index:
        for plant in PLANTS:
            unit_cols = unit_generation_columns(plant)
            p_status = planned_status(planned, idx, plant)
            for unit, capacity in UNIT_CAPACITY[plant].items():
                col = f"gen_{plant}_{unit}"
                if p_status.get(unit_status_col(plant, unit), 1.0) <= 0:
                    forecast.loc[idx, col] = 0.0
                forecast.loc[idx, col] = float(np.clip(forecast.loc[idx, col], 0.0, capacity))

            active = [col for col in unit_cols if forecast.loc[idx, col] > 1e-6]
            active_values = pd.to_numeric(forecast.loc[idx, active], errors="coerce").to_numpy(dtype=float) if active else np.array([])
            if len(active_values) > 1 and np.allclose(active_values, active_values[0], atol=1e-6):
                warnings.warn(
                    f"{plant} forecast row {idx + 1} has identical active unit generation; "
                    "check whether historical unit data contains identical unit behavior.",
                    RuntimeWarning,
                )

            total_col = f"total_gen_{plant}"
            unit_sum = float(forecast.loc[idx, unit_cols].sum())
            if not np.isclose(float(forecast.loc[idx, total_col]), unit_sum, atol=1e-6):
                forecast.loc[idx, total_col] = unit_sum
    forecast[CASCADE_FORECAST_COLUMN] = forecast[TOTAL_FORECAST_COLUMNS].sum(axis=1)
    validate_hourly_outage_effect(forecast, planned)
    return forecast


# Checks that hourly outage statuses are reflected in unit and cascade totals.
def validate_hourly_outage_effect(forecast_df, outage_df):
    warnings_found = 0
    for idx in forecast_df.index:
        for plant in PLANTS:
            p_status = planned_status(outage_df, idx, plant)
            unit_cols = unit_generation_columns(plant)
            total_col = f"total_gen_{plant}"

            for unit in UNIT_CAPACITY[plant]:
                status_col = unit_status_col(plant, unit)
                gen_col = f"gen_{plant}_{unit}"
                status = p_status.get(status_col, 1.0)
                generation = float(forecast_df.loc[idx, gen_col])
                if status <= 0 and abs(generation) > 1e-6:
                    warnings.warn(
                        f"Outage validation: {gen_col} row {idx + 1} is {generation:.6f} MW "
                        f"while {status_col}=0.",
                        RuntimeWarning,
                    )
                    warnings_found += 1
                if (
                    idx > 0
                    and planned_status(outage_df, idx - 1, plant).get(status_col, 1.0) <= 0
                    and status > 0
                    and generation <= 1e-6
                    and float(forecast_df.loc[idx, total_col]) > 1e-6
                ):
                    warnings.warn(
                        f"Outage validation: {gen_col} row {idx + 1} is still 0 MW after "
                        f"{status_col} returned to 1; confirm this is allocation-driven.",
                        RuntimeWarning,
                    )
                    warnings_found += 1

            unit_sum = float(forecast_df.loc[idx, unit_cols].sum())
            plant_total = float(forecast_df.loc[idx, total_col])
            if not np.isclose(plant_total, unit_sum, atol=1e-6):
                warnings.warn(
                    f"Outage validation: {total_col} row {idx + 1} is {plant_total:.6f} MW, "
                    f"but unit sum is {unit_sum:.6f} MW.",
                    RuntimeWarning,
                )
                warnings_found += 1

        cascade_total = float(forecast_df.loc[idx, CASCADE_FORECAST_COLUMN])
        plant_sum = float(forecast_df.loc[idx, TOTAL_FORECAST_COLUMNS].sum())
        if not np.isclose(cascade_total, plant_sum, atol=1e-6):
            warnings.warn(
                f"Outage validation: {CASCADE_FORECAST_COLUMN} row {idx + 1} is "
                f"{cascade_total:.6f} MW, but plant sum is {plant_sum:.6f} MW.",
                RuntimeWarning,
            )
            warnings_found += 1

    if warnings_found:
        print(f"Hourly outage validation completed with {warnings_found} warning(s).")
    else:
        print("Hourly outage validation passed.")


# ============================================================
# FORECAST INPUT AND OUTPUT FORMATTING
# ============================================================

# Creates an empty 24-hour forecast table with all plant/unit outputs.
def empty_forecast_frame(planned):
    forecast = pd.DataFrame({"Date": planned["Date"].dt.date, "Hour": planned["Hour"].astype(int)})
    for col in UNIT_FORECAST_COLUMNS + TOTAL_FORECAST_COLUMNS:
        forecast[col] = 0.0
    forecast[CASCADE_FORECAST_COLUMN] = 0.0
    return forecast


# Converts 1-24 forecast hours into display labels.
def display_hour(hour):
    hour0 = int(hour) - 1
    return "00:00" if hour0 == 0 else f"{hour0}:00"


# Converts planned outage hour values into internal 1-24 format.
def parse_planned_hour(value):
    if isinstance(value, str):
        text = value.strip()
        match = re.fullmatch(r"(\d{1,2})(?::00)?", text)
        if match:
            hour0 = int(match.group(1))
            if 0 <= hour0 <= 23:
                return hour0 + 1
    return int(pd.to_numeric(value))


# Converts internal forecast column names into readable output headers.
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


# Formats the final forecast workbook columns and hour labels.
def format_forecast_output(forecast):
    formatted = forecast.copy()
    formatted["Hour"] = formatted["Hour"].apply(display_hour)
    formatted = formatted.rename(columns={col: forecast_display_column(col) for col in formatted.columns})
    return formatted


# Estimates a plant-specific ramp limit from historical generation changes.
def ramp_limit(train_series):
    diffs = train_series.diff().abs().dropna()
    return float(max(1.0, diffs.quantile(0.98))) if not diffs.empty else 2.0


# Loads and standardizes the 24-hour planned outage input workbook.
def load_planned():
    try:
        planned = pd.read_excel(PLANNED_PATH)
    except zipfile.BadZipFile as exc:
        raise RuntimeError(
            f"Cannot read planned outage workbook: {PLANNED_PATH}\n"
            "The file is damaged or was not saved as a valid .xlsx workbook. "
            "Close it in Excel, then rerun Cell 1 to regenerate "
            "outputs/outages_planning/Planned_Outages_Input.xlsx before rerunning this cell."
        ) from exc
    planned.columns = [str(c).strip() for c in planned.columns]
    col_map = {c.lower(): c for c in planned.columns}
    missing = [col for col in ["date", "hour"] if col not in col_map]
    if missing:
        raise ValueError(f"Planned outage workbook is missing required column(s): {', '.join(missing)}")
    planned = planned.rename(columns={col_map["date"]: "Date", col_map["hour"]: "Hour"})
    planned["Date"] = pd.to_datetime(planned["Date"])
    planned["Hour"] = planned["Hour"].apply(parse_planned_hour).astype(int)
    for col in [c for c in planned.columns if re.fullmatch(r"out_agus[124567]_unit\d+", c)]:
        planned[col] = np.where(pd.to_numeric(planned[col], errors="coerce").fillna(1) > 0, 1, 0)
    return planned


# Predicts next-hour generation delta and converts it back from scaled units.
def predict_delta(model, x_scaler, y_scaler, x_frame):
    x_scaled = x_scaler.transform(x_frame.values.astype(np.float32)).astype(np.float32)
    y_scaled = model(x_scaled, training=False).numpy()
    return y_scaler.inverse_transform(y_scaled).flatten()


# ============================================================
# RECURSIVE 24-HOUR RBFNN FORECASTING
# ============================================================

# Builds one forecast feature row from the latest historical data.
def feature_row_from_history(hist, plant, x_cols, date_val, hour_val):
    hist_feat = add_features(hist.copy())
    row = hist_feat.iloc[-1].to_dict()
    return feature_row_from_feature_dict(row, x_cols, date_val, hour_val)


# Builds one forecast feature row from a precomputed latest feature dictionary.
def feature_row_from_feature_dict(row, x_cols, date_val, hour_val):
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


def value_at_lag(frame, column, lag, default=0.0):
    if column not in frame.columns or len(frame) <= lag:
        return default
    value = frame[column].iloc[-(lag + 1)]
    return default if pd.isna(value) else float(value)


def latest_value(frame, column, default=0.0):
    if column not in frame.columns or frame.empty:
        return default
    value = frame[column].iloc[-1]
    return default if pd.isna(value) else float(value)


def fast_backtest_feature_row(hist, plant, date_val, hour_val, x_cols):
    dt = pd.to_datetime(date_val) + pd.Timedelta(hours=int(hour_val) - 1)
    hour0 = int(hour_val) - 1
    target = f"total_gen_{plant}"
    out_cols = [c for c in hist.columns if re.fullmatch(fr"out_{plant}_unit\d+", c)]
    feature_values = {
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
        f"{target}_current": latest_value(hist, target),
        f"{plant}_units_running": sum(latest_value(hist, c, 1.0) for c in out_cols),
    }
    feature_values[f"{plant}_plant_available"] = int(feature_values[f"{plant}_units_running"] > 0)

    for col in x_cols:
        if col in feature_values:
            continue
        if col in hist.columns:
            feature_values[col] = latest_value(hist, col)
        elif m := re.fullmatch(r"(.+)_lag(\d+)", col):
            feature_values[col] = value_at_lag(hist, m.group(1), int(m.group(2)))
        elif m := re.fullmatch(r"(.+)_rollmean(\d+)", col):
            values = pd.to_numeric(hist[m.group(1)].tail(int(m.group(2))), errors="coerce")
            feature_values[col] = float(values.mean()) if values.notna().any() else 0.0
        elif m := re.fullmatch(r"(.+)_rollstd(\d+)", col):
            values = pd.to_numeric(hist[m.group(1)].tail(int(m.group(2))), errors="coerce")
            feature_values[col] = float(values.std()) if values.notna().sum() > 1 else 0.0
        elif m := re.fullmatch(r"(.+)_rollmin(\d+)", col):
            values = pd.to_numeric(hist[m.group(1)].tail(int(m.group(2))), errors="coerce")
            feature_values[col] = float(values.min()) if values.notna().any() else 0.0
        elif m := re.fullmatch(r"(.+)_rollmax(\d+)", col):
            values = pd.to_numeric(hist[m.group(1)].tail(int(m.group(2))), errors="coerce")
            feature_values[col] = float(values.max()) if values.notna().any() else 0.0
        elif m := re.fullmatch(r"(.+)_diff(\d+)", col):
            base, lag = m.group(1), int(m.group(2))
            feature_values[col] = latest_value(hist, base) - value_at_lag(hist, base, lag)
        elif col == f"{target}_target_lag24":
            feature_values[col] = value_at_lag(hist, target, 23)
        elif col == f"{target}_target_lag168":
            feature_values[col] = value_at_lag(hist, target, 167)
        elif m := re.fullmatch(fr"(gen_{plant}_unit\d+)_share_current", col):
            total = latest_value(hist, target)
            feature_values[col] = latest_value(hist, m.group(1)) / total if total else 0.0
        elif m := re.fullmatch(fr"(gen_{plant}_unit\d+)_share_lag(\d+)", col):
            total_lag = value_at_lag(hist, target, int(m.group(2)))
            feature_values[col] = value_at_lag(hist, m.group(1), int(m.group(2))) / total_lag if total_lag else 0.0
        elif col == f"{plant}_upstream_current" and UPSTREAM_MAP.get(plant):
            feature_values[col] = latest_value(hist, f"total_gen_{UPSTREAM_MAP[plant]}")
        elif m := re.fullmatch(fr"{plant}_upstream_gen_lag(\d+)", col):
            upstream = UPSTREAM_MAP.get(plant)
            feature_values[col] = value_at_lag(hist, f"total_gen_{upstream}", int(m.group(1))) if upstream else 0.0
        else:
            feature_values[col] = 0.0

    return pd.DataFrame([{c: feature_values.get(c, 0.0) for c in x_cols}])


# Applies saved shape adjustments during recursive 24-hour forecasting.
def apply_forecast_shape_adjustments(base_pred, hist, plant, hour_i, meta):
    if plant not in SHAPE_OPTIMIZED_PLANTS:
        return base_pred
    target = f"total_gen_{plant}"
    adjusted = float(base_pred)
    hourly = meta.get("hourly_residual_correction")
    if hourly:
        adjusted = float(apply_hourly_correction([adjusted], [hour_i], plant, hourly)[0])
    profile = meta.get("day_ahead_profile_blend") or meta.get("profile_blend") or FORECAST_PROFILE_BLEND_FALLBACK.get(plant)
    if profile and len(hist) > 23:
        same_hour_yesterday = value_at_lag(hist, target, 23)
        adjusted = float(apply_profile_blend([adjusted], [same_hour_yesterday], plant, profile)[0])
    return adjusted


# Generates the recursive 24-hour RBFNN forecast with outage-aware unit allocation.
def forecast_24h(raw_df, planned):
    forecast = empty_forecast_frame(planned)
    hist = raw_df.copy()

    loaded = {}
    for plant in PLANTS:
        meta = json.loads(model_file(f"meta_{plant}.json").read_text())
        loaded[plant] = {
            "meta": meta,
            "model": load_rbfnn_model(model_file(f"rbfnn_{plant}.keras")),
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
        latest_feature_row = add_features(hist.copy()).iloc[-1].to_dict()

        for plant in PLANTS:
            target = f"total_gen_{plant}"
            bundle = loaded[plant]
            meta = bundle["meta"]
            x_row = feature_row_from_feature_dict(latest_feature_row.copy(), meta["X_cols"], date_i, hour_i)
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
            unit_values, recursive_unit_values = distribute_to_units(plant, base_pred, p_status, hist, hour_i, return_base=True)
            for unit_col, value in unit_values.items():
                forecast.loc[step, unit_col] = value
            for unit_col, value in recursive_unit_values.items():
                new_row[unit_col] = value
            forecast.loc[step, f"total_gen_{plant}"] = sum(unit_values.values())
            new_row[target] = sum(recursive_unit_values.values())

        hist = pd.concat([hist, pd.DataFrame([new_row])], ignore_index=True)
        forecast.loc[step, CASCADE_FORECAST_COLUMN] = forecast.loc[step, TOTAL_FORECAST_COLUMNS].sum()
    ordered_cols = ["Date", "Hour"] + UNIT_FORECAST_COLUMNS + TOTAL_FORECAST_COLUMNS + [CASCADE_FORECAST_COLUMN]
    forecast = forecast[ordered_cols]
    return validate_and_fix_unit_forecast(forecast, planned)


# ============================================================
# LEAKAGE-SAFE ROLLING 24-HOUR DAY-AHEAD BACKTEST
# ============================================================

def valid_24h_dates(frame):
    counts = frame.groupby(pd.to_datetime(frame["datetime"]).dt.date).size()
    return {pd.Timestamp(day) for day, count in counts.items() if count == 24}


def day_ahead_backtest_days(raw_df):
    n = len(raw_df)
    train_end = int(n * TRAIN_RATIO)
    val_end = int(n * (TRAIN_RATIO + VAL_RATIO))
    full_dates = valid_24h_dates(raw_df)

    def usable_days(rows):
        days = []
        for day in sorted(valid_24h_dates(rows)):
            if day in full_dates and not raw_df[raw_df["datetime"] < day].empty:
                days.append(day)
        return days

    return {
        "validation": usable_days(raw_df.iloc[train_end:val_end].copy()),
        "testing": usable_days(raw_df.iloc[val_end:].copy()),
    }


def outage_plan_from_history(history, forecast_day):
    """Build a leakage-safe unit outage plan from the last row before forecast_day."""
    planned = pd.DataFrame({
        "Date": [pd.Timestamp(forecast_day).normalize()] * 24,
        "Hour": list(range(1, 25)),
    })
    if history.empty:
        raise ValueError(f"No historical outage context available before {forecast_day}.")
    if pd.to_datetime(history["datetime"]).max() >= pd.Timestamp(forecast_day):
        raise ValueError("Outage history includes forecast-day rows; this would leak same-day unit outage status.")
    latest = history.iloc[-1]
    for col in [c for c in history.columns if re.fullmatch(r"out_agus[124567]_unit\d+", c)]:
        value = pd.to_numeric(pd.Series([latest.get(col, 1)]), errors="coerce").fillna(1).iloc[0]
        planned[col] = 1 if value > 0 else 0
    return planned


def historical_proxy_outage_plan(actual_day):
    raise RuntimeError("Do not use same-day actual outage status for validation/testing backtests.")


def outage_informed_plan_from_actual_day(actual_day, forecast_day):
    """Build the supplementary known-availability scenario from recorded unit status."""
    planned = pd.DataFrame({
        "Date": [pd.Timestamp(forecast_day).normalize()] * 24,
        "Hour": list(range(1, 25)),
    })
    actual = actual_day.copy()
    actual["hour_internal"] = pd.to_numeric(actual["time"]).astype(int)
    actual = actual.set_index("hour_internal")
    for col in [c for c in actual_day.columns if re.fullmatch(r"out_agus[124567]_unit\d+", c)]:
        values = []
        for hour in range(1, 25):
            value = actual.loc[hour, col] if hour in actual.index else 1
            value = pd.to_numeric(pd.Series([value]), errors="coerce").fillna(1).iloc[0]
            values.append(1 if value > 0 else 0)
        planned[col] = values
    return planned


def history_before_forecast_day(raw_df, forecast_day):
    return raw_df[raw_df["datetime"] < pd.Timestamp(forecast_day)].copy().reset_index(drop=True)


def actual_forecast_day(raw_df, forecast_day):
    start = pd.Timestamp(forecast_day)
    end = start + pd.Timedelta(days=1)
    return raw_df[(raw_df["datetime"] >= start) & (raw_df["datetime"] < end)].copy().reset_index(drop=True)


def preferred_rbfnn_model_path(plant):
    keras_path = MODEL_DIR / f"rbfnn_{plant}.keras"
    if keras_path.exists():
        return keras_path
    return model_file(f"rbfnn_{plant}.keras")


def load_rbfnn_backtest_bundle():
    bundle = {}
    for plant in PLANTS:
        bundle[plant] = {
            "meta": json.loads((MODEL_DIR / f"meta_{plant}.json").read_text()),
            "model": load_rbfnn_model(preferred_rbfnn_model_path(plant)),
            "x_scaler": joblib.load(model_file(f"x_scaler_{plant}.pkl")),
            "y_scaler": joblib.load(model_file(f"y_scaler_{plant}.pkl")),
        }
    return bundle


def warn_if_saved_rbfnn_missing_rainfall_features(raw_df):
    if "rainfall" not in raw_df.columns:
        return
    missing = []
    for plant in PLANTS:
        meta_path = model_file(f"meta_{plant}.json")
        if not meta_path.exists():
            continue
        meta = json.loads(meta_path.read_text())
        features = meta.get("X_cols", [])
        if not any(str(col).startswith("rainfall_lag") for col in features):
            missing.append(plant)
    if missing:
        print(
            "WARNING: Rainfall was added to the cleaned feature set. "
            "Retraining is recommended to ensure the RBFNN uses the updated hydrologic inputs. "
            f"Saved RBFNN metadata missing rainfall lag features for: {', '.join(missing)}"
        )


def forecast_24h_from_loaded_bundle(history, planned, bundle):
    forecast = empty_forecast_frame(planned)
    hist = history.copy()

    for step in range(24):
        date_i = planned.loc[step, "Date"]
        hour_i = int(planned.loc[step, "Hour"])
        new_row = hist.iloc[-1].copy()
        new_row["date"] = pd.to_datetime(date_i)
        new_row["time"] = hour_i
        new_row["datetime"] = pd.to_datetime(date_i) + pd.Timedelta(hours=hour_i - 1)
        for plant in PLANTS:
            target = f"total_gen_{plant}"
            model_info = bundle[plant]
            meta = model_info["meta"]
            x_row = fast_backtest_feature_row(hist.tail(FEATURE_HISTORY_WINDOW).copy(), plant, date_i, hour_i, meta["X_cols"])
            delta = float(predict_delta(model_info["model"], model_info["x_scaler"], model_info["y_scaler"], x_row)[0])
            last_val = float(hist[target].iloc[-1])
            base_pred = last_val + float(meta["best_shrinkage"]) * delta + float(meta.get("bias_correction_mw", 0.0))
            calibration = meta.get("bin_calibration")
            calibration_basis = [base_pred] if calibration and calibration.get("basis") == "base_pred" else [last_val]
            base_pred = float(apply_bin_calibration([base_pred], calibration_basis, plant, calibration)[0])
            base_pred = apply_forecast_shape_adjustments(base_pred, hist, plant, hour_i, meta)
            base_pred = float(np.clip(base_pred, last_val - float(meta["ramp_limit"]), last_val + float(meta["ramp_limit"])))
            base_pred = float(np.clip(base_pred, 0.0, CAPACITY_MW[plant] * 1.05))

            p_status = planned_status(planned, step, plant)
            unit_values = distribute_to_units(plant, base_pred, p_status, hist, hour_i)
            for unit_col, value in unit_values.items():
                forecast.loc[step, unit_col] = value
                new_row[unit_col] = value
            forecast.loc[step, f"total_gen_{plant}"] = sum(unit_values.values())
            new_row[target] = forecast.loc[step, f"total_gen_{plant}"]
            for out_col, value in p_status.items():
                new_row[out_col] = value

        hist = pd.concat([hist, pd.DataFrame([new_row])], ignore_index=True)
        forecast.loc[step, CASCADE_FORECAST_COLUMN] = forecast.loc[step, TOTAL_FORECAST_COLUMNS].sum()

    ordered_cols = ["Date", "Hour"] + UNIT_FORECAST_COLUMNS + TOTAL_FORECAST_COLUMNS + [CASCADE_FORECAST_COLUMN]
    return forecast[ordered_cols]


def rolling_plant_day_ahead_predictions(raw_df, forecast_days, plant, model_info):
    rows = []
    target = f"total_gen_{plant}"
    meta = model_info["meta"]
    for day in forecast_days:
        actual_day = actual_forecast_day(raw_df, day)
        if len(actual_day) != 24:
            continue
        hist = history_before_forecast_day(raw_df, day)
        if hist.empty:
            continue
        planned = outage_plan_from_history(hist, day)

        for step in range(24):
            date_i = planned.loc[step, "Date"]
            hour_i = int(planned.loc[step, "Hour"])
            dt_val = pd.to_datetime(date_i) + pd.Timedelta(hours=hour_i - 1)
            new_row = hist.iloc[-1].copy()
            new_row["date"] = pd.to_datetime(date_i)
            new_row["time"] = hour_i
            new_row["datetime"] = dt_val

            x_row = fast_backtest_feature_row(hist.tail(FEATURE_HISTORY_WINDOW).copy(), plant, date_i, hour_i, meta["X_cols"])
            delta = float(predict_delta(model_info["model"], model_info["x_scaler"], model_info["y_scaler"], x_row)[0])
            last_val = float(hist[target].iloc[-1])
            base_pred = last_val + float(meta["best_shrinkage"]) * delta + float(meta.get("bias_correction_mw", 0.0))
            calibration = meta.get("bin_calibration")
            calibration_basis = [base_pred] if calibration and calibration.get("basis") == "base_pred" else [last_val]
            base_pred = float(apply_bin_calibration([base_pred], calibration_basis, plant, calibration)[0])
            base_pred = apply_forecast_shape_adjustments(base_pred, hist, plant, hour_i, meta)
            base_pred = float(np.clip(base_pred, last_val - float(meta["ramp_limit"]), last_val + float(meta["ramp_limit"])))
            base_pred = float(np.clip(base_pred, 0.0, CAPACITY_MW[plant] * 1.05))

            p_status = planned_status(planned, step, plant)
            unit_values = distribute_to_units(plant, base_pred, p_status, hist, hour_i)
            for unit_col, value in unit_values.items():
                new_row[unit_col] = value
            predicted = float(sum(unit_values.values()))
            new_row[target] = predicted
            for out_col, value in p_status.items():
                new_row[out_col] = value

            rows.append({
                "forecast_date": pd.Timestamp(day).date(),
                "forecast_hour": hour_i,
                "datetime": dt_val,
                "plant": plant,
                "actual_generation": float(actual_day.loc[actual_day["datetime"] == dt_val, target].iloc[0]),
                "predicted_generation": predicted,
                "model": "RBFNN",
                "evaluation_type": DAY_AHEAD_EVALUATION_TYPE,
            })
            hist = pd.concat([hist, pd.DataFrame([new_row])], ignore_index=True)
    return pd.DataFrame(rows)


def rolling_candidate_score(raw_df, forecast_days, plant, model_info):
    predictions = rolling_plant_day_ahead_predictions(raw_df, forecast_days, plant, model_info)
    if predictions.empty:
        return (float("inf"), float("inf"), float("inf")), {}
    mape, mae, rmse, r2 = day_ahead_metric_values(
        predictions["actual_generation"],
        predictions["predicted_generation"],
        plant,
    )
    metrics_out = {
        "operational_mape": mape,
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "forecast_days": int(predictions["forecast_date"].nunique()),
        "forecast_hours": int(len(predictions)),
    }
    return rolling_selection_score(plant, metrics_out), metrics_out


def day_ahead_prediction_rows(model_name, forecast_day, actual_day, forecast):
    rows = []
    actual_by_time = actual_day.set_index("datetime")
    for idx in forecast.index:
        dt_val = pd.to_datetime(forecast.loc[idx, "Date"]) + pd.Timedelta(hours=int(forecast.loc[idx, "Hour"]) - 1)
        for plant in PLANTS:
            rows.append({
                "forecast_date": pd.Timestamp(forecast_day).date(),
                "forecast_hour": int(forecast.loc[idx, "Hour"]),
                "datetime": dt_val,
                "plant": plant,
                "actual_generation": float(actual_by_time.loc[dt_val, f"total_gen_{plant}"]),
                "predicted_generation": float(forecast.loc[idx, f"total_gen_{plant}"]),
                "model": model_name,
                "evaluation_type": DAY_AHEAD_EVALUATION_TYPE,
            })
    return rows


def day_ahead_metric_values(y_true, y_pred, plant):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.abs(y_true) >= operational_threshold(plant)
    mape = np.nan if not mask.any() else float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / np.abs(y_true[mask]))) * 100.0)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred) if len(y_true) > 1 else np.nan
    return mape, mae, rmse, r2


def day_ahead_mape_counts(y_true, plant):
    y_true = np.asarray(y_true, dtype=float)
    mask = np.abs(y_true) >= operational_threshold(plant)
    included = int(mask.sum())
    excluded = int(len(mask) - included)
    excluded_fraction = float(excluded / len(mask)) if len(mask) else np.nan
    return included, excluded, excluded_fraction


def day_ahead_metrics_frame(predictions, model_name):
    rows = []
    df = pd.DataFrame(predictions)
    for plant, group in df.groupby("plant", sort=False):
        mape, mae, rmse, r2 = day_ahead_metric_values(group["actual_generation"], group["predicted_generation"], plant)
        included, excluded, excluded_fraction = day_ahead_mape_counts(group["actual_generation"], plant)
        rows.append({
            "model": model_name,
            "plant": plant,
            "number_of_forecast_days": int(group["forecast_date"].nunique()),
            "number_of_forecast_hours": int(len(group)),
            "mape_rows_included": included,
            "mape_rows_excluded": excluded,
            "mape_rows_excluded_fraction": excluded_fraction,
            "low_load_regime": bool(excluded_fraction > 0.40),
            "operational_mape": mape,
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
            "outage_input_type": OUTAGE_INPUT_TYPE,
            "hydrologic_input_type": HYDROLOGIC_INPUT_TYPE,
            "evaluation_type": DAY_AHEAD_EVALUATION_TYPE,
        })
    return pd.DataFrame(rows)


def day_ahead_summary_frame(validation_metrics, testing_metrics, feature_counts):
    validation = validation_metrics.rename(
        columns={
            "operational_mape": "val_operational_mape",
            "mae": "val_mae",
            "rmse": "val_rmse",
            "r2": "val_r2",
        }
    )
    testing = testing_metrics.rename(
        columns={
            "operational_mape": "test_operational_mape",
            "mae": "test_mae",
            "rmse": "test_rmse",
            "r2": "test_r2",
        }
    )
    summary = validation[["model", "plant", "val_operational_mape", "val_mae", "val_rmse", "val_r2"]].merge(
        testing[["plant", "test_operational_mape", "test_mae", "test_rmse", "test_r2"]],
        on="plant",
        how="inner",
    )
    summary["feature_count"] = summary["plant"].map(feature_counts).astype(int)
    return summary[METRICS_COLUMNS]


def testing_prediction_export(predictions):
    out = predictions.copy()
    out["Date"] = pd.to_datetime(out["datetime"]).dt.date
    out["Hour"] = pd.to_numeric(out["forecast_hour"]).astype(int)
    return out[TESTING_PREDICTION_COLUMNS]


def save_day_ahead_backtest_metadata(raw_df, split_days):
    path = DAY_AHEAD_BACKTEST_DIR / "day_ahead_backtest_metadata.json"
    existing = {}
    if path.exists():
        existing = json.loads(path.read_text())
    existing.update({
        "evaluation_type": DAY_AHEAD_EVALUATION_TYPE,
        "models": sorted(set(existing.get("models", []) + ["RBFNN"])),
        "validation_forecast_days": len(split_days["validation"]),
        "testing_forecast_days": len(split_days["testing"]),
        "data_start": str(raw_df["datetime"].min()),
        "data_end": str(raw_df["datetime"].max()),
        "outage_input_type": OUTAGE_INPUT_TYPE,
        "outage_input_note": OUTAGE_INPUT_NOTE,
        "unit_outage_status_rule": "For validation/testing, each 24-hour forecast day receives only the last-known out_agus*_unit* values from before 00:00 of that day.",
        "same_day_actual_unit_outage_used": False,
        "hydrologic_input_type": HYDROLOGIC_INPUT_TYPE,
        "hydrologic_input_note": HYDROLOGIC_INPUT_NOTE,
        "scaling_and_model_fitting_control": "RBFNN scalers and model weights are fitted on the training split only; saved training-fitted scalers are reused for validation/testing backtests.",
        "validation_testing_separation": "Validation may be used for selection/calibration. Testing data are reserved for final evaluation only.",
        "preprocessing_leakage_control": "Cell 1 KNN imputation is offline source-data preparation and is documented as a limitation; rolling backtest model scaling is training-fitted only.",
        "thesis_ready_note": DAY_AHEAD_THESIS_NOTE,
    })
    path.write_text(json.dumps(existing, indent=2))
    print("Saved:", path)


def save_rbfnn_day_ahead_backtest(raw_df):
    print("Running leakage-safe rolling 24-hour day-ahead backtest for RBFNN...")
    print("Using only historical information available before each forecast day.")
    print("Forecast-day actual generation values are used only for scoring, not as inputs.")
    split_days = day_ahead_backtest_days(raw_df)
    bundle = load_rbfnn_backtest_bundle()
    feature_counts = {plant: len(bundle[plant]["meta"]["X_cols"]) for plant in PLANTS}
    split_metrics = {}
    split_predictions = {}

    for split_name, days in split_days.items():
        rows = []
        print(f"RBFNN {split_name} rolling day-ahead days: {len(days)}")
        for day in days:
            actual_day = actual_forecast_day(raw_df, day)
            if len(actual_day) != 24:
                continue
            hist = history_before_forecast_day(raw_df, day)
            planned = outage_plan_from_history(hist, day)
            forecast = forecast_24h_from_loaded_bundle(hist, planned, bundle)
            rows.extend(day_ahead_prediction_rows("RBFNN", day, actual_day, forecast))

        pred_df = pd.DataFrame(rows)
        metrics_df = day_ahead_metrics_frame(rows, "RBFNN") if rows else pd.DataFrame()
        split_predictions[split_name] = pred_df
        split_metrics[split_name] = metrics_df
        pred_path = DAY_AHEAD_BACKTEST_DIR / f"rbfnn_{split_name}_day_ahead_predictions.xlsx"
        metrics_path = DAY_AHEAD_BACKTEST_DIR / f"rbfnn_{split_name}_day_ahead_metrics.xlsx"
        pred_df.to_excel(pred_path, index=False)
        metrics_df.to_excel(metrics_path, index=False)
        format_excel(pred_path)
        format_excel(metrics_path)
        print("Saved:", pred_path)
        print("Saved:", metrics_path)

    if {"validation", "testing"}.issubset(split_metrics):
        metrics_path = OVERALL_METRICS_DIR / "rbfnn_validation_testing_metrics.xlsx"
        summary = day_ahead_summary_frame(split_metrics["validation"], split_metrics["testing"], feature_counts)
        summary.to_excel(metrics_path, index=False)
        format_excel(metrics_path)
        print("Saved:", metrics_path)

    if "testing" in split_predictions and not split_predictions["testing"].empty:
        predictions_path = OVERALL_METRICS_DIR / "rbfnn_testing_predictions.xlsx"
        testing_prediction_export(split_predictions["testing"]).to_excel(predictions_path, index=False)
        format_excel(predictions_path)
        print("Saved:", predictions_path)

    save_day_ahead_backtest_metadata(raw_df, split_days)
    print("Saved RBFNN rolling day-ahead backtest predictions and metrics.")


def save_rbfnn_outage_informed_backtest(raw_df):
    scenario_type = "supplementary_outage_informed_backtest"
    outage_type = "actual_forecast_day_unit_status_proxy_for_known_planned_availability"
    scenario_note = (
        "Uses recorded forecast-day unit outage/status as a proxy for perfectly known "
        "planned unit availability. This is not the primary leakage-free day-ahead "
        "result; it is a sensitivity scenario showing performance when operators "
        "provide forecast-day unit availability before issuing the forecast."
    )
    print("Running supplementary outage-informed 24-hour RBFNN backtest...")
    print("This is a scenario analysis, not the primary leakage-free day-ahead result.")
    split_days = day_ahead_backtest_days(raw_df)
    bundle = load_rbfnn_backtest_bundle()
    feature_counts = {plant: len(bundle[plant]["meta"]["X_cols"]) for plant in PLANTS}
    split_metrics = {}
    split_predictions = {}

    for split_name, days in split_days.items():
        rows = []
        print(f"RBFNN outage-informed {split_name} days: {len(days)}")
        for day in days:
            actual_day = actual_forecast_day(raw_df, day)
            if len(actual_day) != 24:
                continue
            hist = history_before_forecast_day(raw_df, day)
            if hist.empty:
                continue
            planned = outage_informed_plan_from_actual_day(actual_day, day)
            forecast = forecast_24h_from_loaded_bundle(hist, planned, bundle)
            rows.extend(day_ahead_prediction_rows("RBFNN", day, actual_day, forecast))

        pred_df = pd.DataFrame(rows)
        if not pred_df.empty:
            pred_df["evaluation_type"] = scenario_type
            pred_df["outage_input_type"] = outage_type
        metrics_df = day_ahead_metrics_frame(rows, "RBFNN") if rows else pd.DataFrame()
        if not metrics_df.empty:
            metrics_df["evaluation_type"] = scenario_type
            metrics_df["outage_input_type"] = outage_type
        split_predictions[split_name] = pred_df
        split_metrics[split_name] = metrics_df

        pred_path = OUTAGE_INFORMED_DIR / f"rbfnn_{split_name}_outage_informed_predictions.xlsx"
        metrics_path = OUTAGE_INFORMED_DIR / f"rbfnn_{split_name}_outage_informed_metrics.xlsx"
        pred_df.to_excel(pred_path, index=False)
        metrics_df.to_excel(metrics_path, index=False)
        format_excel(pred_path)
        format_excel(metrics_path)
        print("Saved:", pred_path)
        print("Saved:", metrics_path)

    if {"validation", "testing"}.issubset(split_metrics):
        summary = day_ahead_summary_frame(split_metrics["validation"], split_metrics["testing"], feature_counts)
        summary_path = OUTAGE_INFORMED_DIR / "rbfnn_outage_informed_validation_testing_metrics.xlsx"
        summary.to_excel(summary_path, index=False)
        format_excel(summary_path)
        print("Saved:", summary_path)

    metadata = {
        "evaluation_type": scenario_type,
        "scenario_role": "supplementary sensitivity analysis, not primary leakage-free day-ahead backtest",
        "model": "RBFNN",
        "outage_input_type": outage_type,
        "same_day_actual_unit_outage_used": True,
        "testing_used_for_training_or_tuning": False,
        "defense_note": scenario_note,
        "primary_leakage_free_results_remain_in": "metadata/day_ahead_backtest and metadata/overall_metrics",
        "validation_forecast_days": len(split_days["validation"]),
        "testing_forecast_days": len(split_days["testing"]),
    }
    metadata_path = OUTAGE_INFORMED_DIR / "rbfnn_outage_informed_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2))
    print("Saved:", metadata_path)


def save_loaded_rbfnn_selection_audit():
    metrics_path = OVERALL_METRICS_DIR / "rbfnn_validation_testing_metrics.xlsx"
    metrics_df = pd.read_excel(metrics_path) if metrics_path.exists() else pd.DataFrame()
    rows = []
    for plant in PLANTS:
        meta = json.loads(model_file(f"meta_{plant}.json").read_text())
        metrics_row = {}
        if not metrics_df.empty:
            match = metrics_df[metrics_df["plant"] == plant]
            if not match.empty:
                metrics_row = match.iloc[0].to_dict()
        rows.append({
            "plant": plant,
            "model": "RBFNN",
            "feature_count": len(meta.get("X_cols", [])),
            "selected_hyperparameters": {
                "centers": meta.get("selected_centers"),
                "learning_rate": meta.get("selected_learning_rate"),
                "gamma_init": meta.get("selected_gamma_init"),
                "batch_size": meta.get("selected_batch_size"),
                "early_stopping_patience": meta.get("selected_patience"),
                "shrinkage": meta.get("best_shrinkage"),
            },
            "calibration": {
                "bin_calibration": bool(meta.get("bin_calibration")),
                "hourly_residual_correction": bool(meta.get("hourly_residual_correction")),
                "profile_blend": bool(meta.get("profile_blend")),
                "ramp_limit": meta.get("ramp_limit"),
            },
            "validation_score_used_for_selection": {
                "operational_mape": metrics_row.get("val_operational_mape"),
                "rmse": metrics_row.get("val_rmse"),
                "r2": metrics_row.get("val_r2"),
            },
            "testing_score_after_final_evaluation": {
                "operational_mape": metrics_row.get("test_operational_mape"),
                "mae": metrics_row.get("test_mae"),
                "rmse": metrics_row.get("test_rmse"),
                "r2": metrics_row.get("test_r2"),
            },
            "selection_source": "saved optimized RBFNN metadata",
            "testing_used_for_selection_or_calibration": False,
        })
    update_model_selection_audit("rbfnn", rows)


def tune_saved_day_ahead_profile_blends(raw_df):
    print("Tuning RBFNN day-ahead profile blends on validation predictions only...")
    validation_path = DAY_AHEAD_BACKTEST_DIR / "rbfnn_validation_day_ahead_predictions.xlsx"
    if not validation_path.exists():
        save_rbfnn_day_ahead_backtest(raw_df)
    validation = pd.read_excel(validation_path)
    validation["datetime"] = pd.to_datetime(validation["datetime"])
    history = raw_df.copy()
    history["datetime"] = pd.to_datetime(history["datetime"])
    rows = []

    for plant in PLANTS:
        target = f"total_gen_{plant}"
        plant_val = validation[validation["plant"] == plant].copy()
        anchor_source = history[["datetime", target]].copy()
        anchor_source["datetime"] = anchor_source["datetime"] + pd.Timedelta(days=1)
        anchor_source = anchor_source.rename(columns={target: "same_hour_yesterday"})
        plant_val = plant_val.merge(anchor_source, on="datetime", how="left").dropna(subset=["same_hour_yesterday"])
        if plant_val.empty:
            continue

        actual = plant_val["actual_generation"].values.astype(float)
        base_pred = plant_val["predicted_generation"].values.astype(float)
        anchor = plant_val["same_hour_yesterday"].values.astype(float)
        base_metrics = {
            "operational_mape": day_ahead_metric_values(actual, base_pred, plant)[0],
            "mae": day_ahead_metric_values(actual, base_pred, plant)[1],
            "rmse": day_ahead_metric_values(actual, base_pred, plant)[2],
            "r2": day_ahead_metric_values(actual, base_pred, plant)[3],
        }
        best = {"score": rolling_selection_score(plant, base_metrics), "weight": 0.0, "metrics": base_metrics}
        for weight in np.arange(0.05, 0.91, 0.05):
            pred = np.clip((1.0 - weight) * base_pred + weight * anchor, 0.0, CAPACITY_MW[plant] * 1.05)
            mape, mae, rmse, r2 = day_ahead_metric_values(actual, pred, plant)
            metrics_out = {"operational_mape": mape, "mae": mae, "rmse": rmse, "r2": r2}
            score = rolling_selection_score(plant, metrics_out)
            if score < best["score"]:
                best = {"score": score, "weight": float(weight), "metrics": metrics_out}

        meta_path = MODEL_DIR / f"meta_{plant}.json"
        meta = json.loads(meta_path.read_text())
        base_score = rolling_selection_score(plant, base_metrics)
        best_score = rolling_selection_score(plant, best["metrics"])
        robust_improvement = (
            best["weight"] > 0
            and best_score < base_score
            and best["metrics"]["rmse"] <= base_metrics["rmse"]
            and best["metrics"]["r2"] >= base_metrics["r2"]
        )
        if robust_improvement:
            meta["day_ahead_profile_blend"] = {
                "same_hour_yesterday_weight": best["weight"],
                "source": "rolling_validation_only",
                "validation_before": base_metrics,
                "validation_after": best["metrics"],
            }
        else:
            meta["day_ahead_profile_blend"] = None
        meta["testing_used_for_day_ahead_profile_blend"] = False
        meta_path.write_text(json.dumps(meta, indent=2, default=str))
        rows.append({
            "plant": plant,
            "selected_weight": best["weight"],
            "validation_before": base_metrics,
            "validation_after": best["metrics"],
            "testing_used": False,
        })

    path = RBFNN_META_DIR / "rbfnn_day_ahead_profile_blend_report.xlsx"
    pd.DataFrame(rows).to_excel(path, index=False)
    format_excel(path)
    print("Saved:", path)


# Loads optional actual next-day generation for post-forecast diagnostics.
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


# Saves actual-vs-forecast diagnostics when the next-day actual file is available.
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


# ============================================================
# WORKFLOW ENTRY POINTS
# ============================================================

# Loads cleaned data and planned outages required by RBFNN workflows.
def load_latest_inputs():
    cleaned_excel = CLEANED_DATA_DIR / "cleaned_hourly_data.xlsx"
    if not (CLEAN_PATH.exists() or cleaned_excel.exists()) or not PLANNED_PATH.exists():
        raise FileNotFoundError("Run Thesis Forecasting/scripts/cell1_clean_data.py first.")

    if CLEAN_PATH.exists():
        raw_df = pd.read_parquet(CLEAN_PATH)
    else:
        raw_df = pd.read_excel(cleaned_excel)
    raw_df = add_runtime_compatibility_columns(rebuild_datetime(raw_df))
    warn_if_saved_rbfnn_missing_rainfall_features(raw_df)
    planned = load_planned()
    return raw_df, planned


# Recomputes validation/testing metrics from saved RBFNN models.
def evaluate_saved_models(raw_df):
    raise RuntimeError(
        "One-step chronological validation/testing exports are disabled. "
        "Use save_rbfnn_day_ahead_backtest(raw_df) for strict rolling 24-hour day-ahead outputs."
    )
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

        model = load_rbfnn_model(model_file(f"rbfnn_{plant}.keras"))
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


# Checks saved inputs and RBFNN artifacts required for fast forecast-only mode.
def validate_forecast_only_inputs():
    cleaned_excel = CLEANED_DATA_DIR / "cleaned_hourly_data.xlsx"
    if not (CLEAN_PATH.exists() or cleaned_excel.exists()):
        raise FileNotFoundError("Run Data Cleaning first.")
    if not PLANNED_PATH.exists():
        raise FileNotFoundError("Create or save Planned Outage Plan first.")
    for plant in PLANTS:
        required = [
            model_file(f"meta_{plant}.json"),
            model_file(f"rbfnn_{plant}.keras"),
            model_file(f"x_scaler_{plant}.pkl"),
            model_file(f"y_scaler_{plant}.pkl"),
        ]
        if not all(path.exists() for path in required):
            raise FileNotFoundError("Retrain RBFNN Model first.")


# Uses saved RBFNN models to generate only the day-ahead forecast.
def run_forecast_only():
    validate_forecast_only_inputs()
    raw_df, planned = load_latest_inputs()
    forecast = forecast_24h(raw_df, planned)
    output_forecast = format_forecast_output(forecast)
    xlsx_path = RBFNN_FORECAST_DIR / "Day_Ahead_24H_RBFNN_Forecast.xlsx"
    csv_path = RBFNN_FORECAST_DIR / "Day_Ahead_24H_RBFNN_Forecast.csv"
    output_forecast.to_excel(xlsx_path, index=False)
    output_forecast.to_csv(csv_path, index=False)
    format_excel(xlsx_path)
    print("Fast forecast-only mode complete")
    print("Saved:", xlsx_path)
    print("Saved:", csv_path)
    if "--day-ahead-backtest" in sys.argv:
        save_rbfnn_day_ahead_backtest(raw_df)


# Trains/tunes RBFNN models, saves metrics/artifacts, and generates forecasts.
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
    leakage_audit_rows = []
    calibration_report_rows = []
    selection_audit_rows = []
    validation_forecast_days = day_ahead_backtest_days(raw_df)["validation"]

    for plant in PLANTS:
        if selected_plants and plant not in selected_plants:
            continue
        print(f"\n===== OPTIMIZED RBFNN {plant.upper()} =====")
        target = f"total_gen_{plant}"
        y_col = f"{target}_delta_tplus1"
        x_cols = feature_columns_for(feat_df, plant)
        leakage_audit_rows.append(leakage_feature_audit(x_cols, plant))
        data = feat_df.dropna(subset=[y_col] + x_cols).copy()
        train_df, val_df, test_df = chronological_split(data)

        # --- Scaling and chronological split preparation ---
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

        # --- Model configuration search with preliminary validation screening ---
        screened = []
        for config in rbfnn_search_configs(plant):
            tf.keras.backend.clear_session()
            model = build_model(
                x_train.shape[1],
                n_centers=config["n_centers"],
                learning_rate=config["learning_rate"],
                gamma_init=config["gamma_init"],
            )
            init_centers(model, x_train, n_centers=config["n_centers"])
            history = model.fit(
                x_train,
                y_train,
                validation_data=(x_val, y_val_scaled),
                epochs=EPOCHS,
                batch_size=config["batch_size"],
                verbose=0,
                callbacks=[
                    tf.keras.callbacks.EarlyStopping(
                        monitor="val_loss",
                        patience=config["patience"],
                        restore_best_weights=True,
                    )
                ],
            )
            val_delta_pred = y_scaler.inverse_transform(model.predict(x_val, verbose=0)).flatten()
            test_delta_pred = y_scaler.inverse_transform(model.predict(x_test, verbose=0)).flatten()

            for shrinkage in SHRINKAGE_GRID:
                val_pred_candidate, bias = calibrated_level_predictions(val_current, val_delta_pred, val_actual, plant, shrinkage)
                val_m = metrics_dict(val_actual, val_pred_candidate, plant)
                screened.append({
                    "screen_score": candidate_score(val_m),
                    "screen_metrics": val_m,
                    "model": model,
                    "history": history,
                    "n_centers": config["n_centers"],
                    "learning_rate": config["learning_rate"],
                    "gamma_init": config["gamma_init"],
                    "batch_size": config["batch_size"],
                    "patience": config["patience"],
                    "shrinkage": shrinkage,
                    "bias": bias,
                    "val_pred": val_pred_candidate,
                    "test_pred": apply_level_prediction(test_current, test_delta_pred, plant, shrinkage, bias),
                })

        screened = sorted(screened, key=lambda item: item["screen_score"])
        finalists = screened[:RBFNN_FINALIST_COUNT]
        best = None
        finalist_scores = []
        for candidate in finalists:
            meta_candidate = {
                "plant": plant,
                "target": target,
                "y_col": y_col,
                "X_cols": x_cols,
                "capacity_mw": CAPACITY_MW[plant],
                "best_shrinkage": candidate["shrinkage"],
                "bias_correction_mw": candidate["bias"],
                "bin_calibration": None,
                "hourly_residual_correction": None,
                "profile_blend": None,
                "ramp_limit": ramp_limit(train_df[target]),
            }
            model_info = {
                "meta": meta_candidate,
                "model": candidate["model"],
                "x_scaler": x_scaler,
                "y_scaler": y_scaler,
            }
            rolling_score, rolling_metrics = rolling_candidate_score(raw_df, validation_forecast_days, plant, model_info)
            candidate["rolling_score"] = rolling_score
            candidate["rolling_validation_metrics"] = rolling_metrics
            finalist_scores.append({
                "n_centers": candidate["n_centers"],
                "learning_rate": candidate["learning_rate"],
                "gamma_init": candidate["gamma_init"],
                "batch_size": candidate["batch_size"],
                "patience": candidate["patience"],
                "shrinkage": candidate["shrinkage"],
                "preliminary_validation_score": candidate["screen_score"],
                "rolling_validation_score": rolling_score,
                "rolling_validation_metrics": rolling_metrics,
            })
            if best is None or rolling_score < best["rolling_score"]:
                best = candidate

        # --- Validation calibration and testing evaluation ---
        model = best["model"]
        history = best["history"]
        shrinkage = best["shrinkage"]
        bias = best["bias"]
        val_pred = best["val_pred"]
        test_pred = best["test_pred"]
        limit = ramp_limit(train_df[target])
        accepted_roll_score = best["rolling_score"]
        selected_roll_metrics = best["rolling_validation_metrics"]

        bin_candidate, bin_val_pred = tune_bin_calibration(plant, val_current, val_actual, val_pred)
        bin_calibration = None
        if bin_candidate:
            meta_candidate = {
                "plant": plant,
                "target": target,
                "y_col": y_col,
                "X_cols": x_cols,
                "capacity_mw": CAPACITY_MW[plant],
                "best_shrinkage": shrinkage,
                "bias_correction_mw": bias,
                "bin_calibration": bin_candidate,
                "hourly_residual_correction": None,
                "profile_blend": None,
                "ramp_limit": limit,
            }
            roll_score, roll_metrics = rolling_candidate_score(raw_df, validation_forecast_days, plant, {
                "meta": meta_candidate,
                "model": model,
                "x_scaler": x_scaler,
                "y_scaler": y_scaler,
            })
            if roll_score < accepted_roll_score:
                bin_calibration = bin_candidate
                val_pred = bin_val_pred
                test_basis = test_pred if bin_calibration.get("basis") == "base_pred" else test_current
                test_pred = apply_bin_calibration(test_pred, test_basis, plant, bin_calibration)
                accepted_roll_score = roll_score
                selected_roll_metrics = roll_metrics

        val_hours = target_hours_from_rows(val_df)
        test_hours = target_hours_from_rows(test_df)
        hourly_candidate, hourly_val_pred = tune_hourly_correction(plant, val_pred, val_actual, val_hours)
        hourly_correction = None
        if hourly_candidate:
            meta_candidate = {
                "plant": plant,
                "target": target,
                "y_col": y_col,
                "X_cols": x_cols,
                "capacity_mw": CAPACITY_MW[plant],
                "best_shrinkage": shrinkage,
                "bias_correction_mw": bias,
                "bin_calibration": bin_calibration,
                "hourly_residual_correction": hourly_candidate,
                "profile_blend": None,
                "ramp_limit": limit,
            }
            roll_score, roll_metrics = rolling_candidate_score(raw_df, validation_forecast_days, plant, {
                "meta": meta_candidate,
                "model": model,
                "x_scaler": x_scaler,
                "y_scaler": y_scaler,
            })
            if roll_score < accepted_roll_score:
                hourly_correction = hourly_candidate
                val_pred = hourly_val_pred
                test_pred = apply_hourly_correction(test_pred, test_hours, plant, hourly_correction)
                accepted_roll_score = roll_score
                selected_roll_metrics = roll_metrics

        profile_candidate, profile_val_pred = tune_profile_blend(plant, val_pred, val_actual, same_hour_target_anchor(val_df, plant))
        profile_blend = None
        if profile_candidate:
            meta_candidate = {
                "plant": plant,
                "target": target,
                "y_col": y_col,
                "X_cols": x_cols,
                "capacity_mw": CAPACITY_MW[plant],
                "best_shrinkage": shrinkage,
                "bias_correction_mw": bias,
                "bin_calibration": bin_calibration,
                "hourly_residual_correction": hourly_correction,
                "profile_blend": profile_candidate,
                "ramp_limit": limit,
            }
            roll_score, roll_metrics = rolling_candidate_score(raw_df, validation_forecast_days, plant, {
                "meta": meta_candidate,
                "model": model,
                "x_scaler": x_scaler,
                "y_scaler": y_scaler,
            })
            if roll_score < accepted_roll_score:
                profile_blend = profile_candidate
                val_pred = profile_val_pred
                test_pred = apply_profile_blend(test_pred, same_hour_target_anchor(test_df, plant), plant, profile_blend)
                accepted_roll_score = roll_score
                selected_roll_metrics = roll_metrics

        val_metrics = metrics_dict(val_actual, val_pred, plant)
        test_metrics = metrics_dict(test_actual, test_pred, plant)
        val_shape = shape_metrics(val_actual, val_pred)
        test_shape = shape_metrics(test_actual, test_pred)

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

        # --- Artifact export for forecast-only reuse ---
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
            "selected_gamma_init": best["gamma_init"],
            "selected_batch_size": best["batch_size"],
            "selected_patience": best["patience"],
            "selection_metric": "lowest R2-aware rolling day-ahead validation score among preliminarily screened finalists; score penalizes MAPE, RMSE, low R2, and especially negative R2",
            "rolling_validation_metrics": selected_roll_metrics,
            "derived_from": "validation set only",
            "testing_used_for_calibration": "No",
            "model_type": "RBFNN residual/delta model anchored to persistence",
        }, indent=2))
        calibration_report_rows.append({
            "plant": plant,
            "feature_count": len(x_cols),
            "selected centers": best["n_centers"],
            "selected learning rate": best["learning_rate"],
            "selected gamma init": best["gamma_init"],
            "selected batch size": best["batch_size"],
            "selected patience": best["patience"],
            "selected shrinkage": shrinkage,
            "bias correction": bias,
            "bin calibration used": "Yes" if bin_calibration else "No",
            "hourly residual correction used": "Yes" if hourly_correction else "No",
            "profile blending used": "Yes" if profile_blend else "No",
            "ramp limit used": limit,
            "capacity/outage adjustment used": "Yes",
            "derived from": "validation set only",
            "testing used for calibration": "No",
        })

        selection_audit_rows.append({
            "plant": plant,
            "model": "RBFNN",
            "feature_count": len(x_cols),
            "selected_hyperparameters": {
                "centers": best["n_centers"],
                "learning_rate": best["learning_rate"],
                "gamma_init": best["gamma_init"],
                "batch_size": best["batch_size"],
                "early_stopping_patience": best["patience"],
                "shrinkage": shrinkage,
            },
            "calibration": {
                "bin_calibration": bool(bin_calibration),
                "hourly_residual_correction": bool(hourly_correction),
                "profile_blend": bool(profile_blend),
                "ramp_limit": limit,
            },
            "preliminary_validation_score": best["screen_score"],
            "rolling_validation_score": accepted_roll_score,
            "rolling_validation_metrics": selected_roll_metrics,
            "finalists": finalist_scores,
            "validation_metrics_after_calibration": val_metrics,
            "testing_metrics_after_final_evaluation": test_metrics,
            "testing_used_for_selection_or_calibration": False,
        })

        print("Saved:", history_path)
        print(pd.DataFrame([row]).to_string(index=False))

    save_leakage_audit(leakage_audit_rows)
    save_rbfnn_calibration_report(calibration_report_rows)
    update_model_selection_audit("rbfnn", selection_audit_rows)
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
        save_rbfnn_day_ahead_backtest(raw_df)
    else:
        print("Skipped strict day-ahead metric export because not all plant models are available.")


# Selects training mode when --train is passed; otherwise runs forecast-only mode.
def main():
    args = sys.argv[1:]
    allowed_args = {
        "--train",
        "--forecast-only",
        "--day-ahead-backtest",
        "--outage-informed-backtest",
        "--audit-only",
        "--tune-day-ahead-profile",
    }
    unknown = [arg for arg in args if arg.startswith("--") and arg not in allowed_args]
    if unknown:
        raise ValueError(f"Unsupported argument(s): {unknown}. Use --forecast-only or --train.")
    if "--audit-only" in args:
        save_loaded_rbfnn_selection_audit()
        return
    if "--tune-day-ahead-profile" in args:
        raw_df, _ = load_latest_inputs()
        tune_saved_day_ahead_profile_blends(raw_df)
        save_rbfnn_day_ahead_backtest(raw_df)
        save_loaded_rbfnn_selection_audit()
        return
    if "--train" in args and "--forecast-only" in args:
        raise ValueError("Use either --forecast-only or --train, not both.")
    if "--day-ahead-backtest" in args and "--train" not in args:
        raw_df, _ = load_latest_inputs()
        save_rbfnn_day_ahead_backtest(raw_df)
        return
    if "--outage-informed-backtest" in args and "--train" not in args:
        raw_df, _ = load_latest_inputs()
        save_rbfnn_outage_informed_backtest(raw_df)
        return
    if "--train" in args:
        run_training_and_forecast()
    else:
        run_forecast_only()


if __name__ == "__main__":
    main()

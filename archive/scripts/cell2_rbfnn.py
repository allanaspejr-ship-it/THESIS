import os
import re
import json
import math
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

np.random.seed(42)
tf.random.set_seed(42)

# ============================================================
# VS CODE LOCAL PROJECT PATH
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parents[1]

CONFIG = {
    "base_output_dir": Path(r"C:\Users\Allen Mae\Desktop\HYDRO_FORECASTING\data\outputs"),

    "plants": ["agus1", "agus2", "agus4", "agus5", "agus6", "agus7"],

    "delta_plants": ["agus2", "agus4", "agus5", "agus7"],

    "upstream_map": {
        "agus1": None,
        "agus2": "agus1",
        "agus4": "agus2",
        "agus5": "agus4",
        "agus6": "agus5",
        "agus7": "agus6",
    },

    "lags": [1, 2, 3, 6, 12, 24, 48],
    "rolling_window": 24,

    "train_ratio": 0.70,
    "val_ratio": 0.15,

    "epochs": 200,
    "batch_size": 16,
    "patience": 15,
    "learning_rate": 0.0005,

    "n_centers_per_plant": {
        "agus1": 200,
        "agus2": 200,
        "agus4": 200,
        "agus5": 200,
        "agus6": 200,
        "agus7": 200,
    },

    "gamma_init": 1.0,
    "retrain_rbfnn": False,
    "min_rows_after_feature_prep": 100,

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

    "dirs": {
        "runtime": "01_runtime_outputs",
        "models": "02_models",
        "metadata": "03_metadata",
    }
}

DIRS = {
    k: CONFIG["base_output_dir"] / v
    for k, v in CONFIG["dirs"].items()
}

for p in DIRS.values():
    p.mkdir(parents=True, exist_ok=True)

META_DIRS = {
    "loss_plots": DIRS["metadata"] / "training_validation_loss",
    "validation_daily_metrics": DIRS["metadata"] / "validation_daily_metrics",
    "testing_daily_metrics": DIRS["metadata"] / "testing_daily_metrics",
    "summary_metrics": DIRS["metadata"] / "validation_testing_metrics",
}

for p in META_DIRS.values():
    p.mkdir(parents=True, exist_ok=True)

CLEAN_PATH = DIRS["runtime"] / "cleaned_hourly_data.parquet"
PLANNED_PATH = DIRS["runtime"] / "Planned_Outages_Input.xlsx"


# ============================================================
# HELPERS
# ============================================================

def chronological_split(data, train_ratio, val_ratio):
    n = len(data)
    i1 = int(n * train_ratio)
    i2 = int(n * (train_ratio + val_ratio))

    return (
        data.iloc[:i1].copy(),
        data.iloc[i1:i2].copy(),
        data.iloc[i2:].copy()
    )


def mape_safe(y_true, y_pred, eps=1e-6):
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)

    denom = np.where(np.abs(y_true) < eps, eps, np.abs(y_true))
    return np.mean(np.abs((y_true - y_pred) / denom)) * 100.0


def save_daily_metrics(df_part, y_true, y_pred, out_path):
    temp = df_part.copy()
    temp["actual"] = np.array(y_true).flatten()
    temp["pred"] = np.array(y_pred).flatten()
    temp["Date"] = pd.to_datetime(temp["datetime"]).dt.date

    rows = []

    for d, g in temp.groupby("Date"):
        actual = g["actual"].values
        pred = g["pred"].values

        rows.append({
            "date": d,
            "mape": mape_safe(actual, pred),
            "mae": mean_absolute_error(actual, pred),
            "rmse": math.sqrt(mean_squared_error(actual, pred)),
            "r2": r2_score(actual, pred) if len(actual) > 1 else np.nan,
            "samples": len(g),
        })

    daily_df = pd.DataFrame(rows)
    daily_df.to_excel(out_path, index=False)

    return daily_df


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


def safe_lag(series, lag):
    if len(series) >= lag:
        return float(series.iloc[-lag])
    return float(series.iloc[0])


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


# ============================================================
# RBF LAYER
# ============================================================

class RBFLayer(tf.keras.layers.Layer):
    def __init__(self, units, gamma_init=1.0, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.gamma_init = gamma_init

    def build(self, input_shape):
        input_dim = int(input_shape[-1])

        self.centers = self.add_weight(
            name="centers",
            shape=(self.units, input_dim),
            initializer="uniform",
            trainable=True,
        )

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
        gamma = tf.exp(self.log_gamma)
        return tf.exp(-gamma * d2)


def build_rbfnn(input_dim, output_dim, n_centers, gamma_init, learning_rate):
    inp = tf.keras.Input(shape=(input_dim,))
    x = RBFLayer(n_centers, gamma_init=gamma_init)(inp)
    out = tf.keras.layers.Dense(output_dim, activation="linear")(x)

    model = tf.keras.Model(inputs=inp, outputs=out)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse"
    )

    return model


def init_centers_with_kmeans(model, X_train, n_centers):
    km = KMeans(n_clusters=n_centers, random_state=42, n_init=10)
    km.fit(X_train)

    for layer in model.layers:
        if isinstance(layer, RBFLayer):
            layer.centers.assign(km.cluster_centers_)
            break


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
    # LOAD CLEANED DATA
    # ------------------------------------------------------------
    df = pd.read_parquet(CLEAN_PATH)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)

    print("Loaded cleaned data:", CLEAN_PATH)
    print("Rows:", len(df))
    print("Datetime min:", df["datetime"].min())
    print("Datetime max:", df["datetime"].max())

    # ------------------------------------------------------------
    # LOAD / CLEAN PLANNED OUTAGES
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
    # FEATURE ENGINEERING
    # ------------------------------------------------------------
    hour0 = df["time"] - 1
    dt = pd.to_datetime(df["datetime"])

    df["hour_sin"] = np.sin(2 * np.pi * hour0 / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour0 / 24)
    df["day_of_week"] = dt.dt.dayofweek
    df["month"] = dt.dt.month
    df["is_weekend"] = (dt.dt.dayofweek >= 5).astype(int)

    for p in CONFIG["plants"]:
        target = f"total_gen_{p}"

        if target in df.columns:
            for lag in CONFIG["lags"]:
                df[f"{target}_lag{lag}"] = df[target].shift(lag)

            rw = CONFIG["rolling_window"]

            df[f"{target}_rollmean{rw}"] = df[target].rolling(rw).mean()
            df[f"{target}_rollstd{rw}"] = df[target].rolling(rw).std()
            df[f"{target}_rollmin{rw}"] = df[target].rolling(rw).min()
            df[f"{target}_rollmax{rw}"] = df[target].rolling(rw).max()

            df[f"{target}_diff1"] = df[target].diff(1)
            df[f"{target}_diff2"] = df[target].diff(2)
            df[f"{target}_diff3"] = df[target].diff(3)
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

    oper_cols_for_lag = [
        c for c in df.columns
        if c.startswith("tot_agus")
        or c.startswith("elev_agus")
        or c.startswith("spill_agus")
        or c == "rainfall"
        or "outflow" in c
    ]

    for c in oper_cols_for_lag:
        for lag in [1, 2, 3, 6, 12, 24]:
            df[f"{c}_lag{lag}"] = df[c].shift(lag)

    for p, up in CONFIG["upstream_map"].items():
        if up is None:
            continue

        up_gen = f"total_gen_{up}"

        if up_gen in df.columns:
            for lag in CONFIG["lags"]:
                df[f"{p}_upstream_gen_lag{lag}"] = df[up_gen].shift(lag)

        outflow_cols = [c for c in df.columns if "outflow" in c]

        if outflow_cols:
            out_col = outflow_cols[0]

            for lag in CONFIG["lags"]:
                df[f"{p}_upstream_flow_lag{lag}"] = df[out_col].shift(lag)

    # ------------------------------------------------------------
    # TRAIN IF NEEDED
    # ------------------------------------------------------------
    summary_rows = []

    for plant in CONFIG["plants"]:
        target = f"total_gen_{plant}"

        print(f"\n==================== {plant.upper()} ====================")

        if target not in df.columns:
            print(f"[SKIP {plant}] target missing: {target}")
            continue

        n_centers = CONFIG["n_centers_per_plant"][plant]

        model_path = DIRS["models"] / f"rbfnn_{plant}.keras"
        x_scaler_path = DIRS["models"] / f"x_scaler_{plant}.pkl"
        y_scaler_path = DIRS["models"] / f"y_scaler_{plant}.pkl"
        meta_path = DIRS["models"] / f"meta_{plant}.json"

        model_files_exist = all(
            p.exists()
            for p in [model_path, x_scaler_path, y_scaler_path, meta_path]
        )

        if model_files_exist and not CONFIG["retrain_rbfnn"]:
            print(f"[LOAD-ONLY] Existing RBFNN files found for {plant}")
            continue

        data = df.copy()

        y_col = f"{target}_tplus1"
        is_delta_model = plant in CONFIG["delta_plants"]

        if is_delta_model:
            data[y_col] = data[target].shift(-1) - data[target]
        else:
            data[y_col] = data[target].shift(-1)

        X_cols = ["hour_sin", "hour_cos", "day_of_week", "month", "is_weekend"]

        X_cols += [c for c in data.columns if c.startswith(target + "_lag")]
        X_cols += [c for c in data.columns if c.startswith(target + "_roll")]
        X_cols += [c for c in data.columns if c.startswith(target + "_diff")]
        X_cols += [c for c in data.columns if c.startswith(target + "_recent_mean3")]

        X_cols += [
            c for c in data.columns
            if re.fullmatch(fr"out_{plant}_unit\d+", str(c))
        ]

        X_cols += [
            c for c in data.columns
            if c in [f"{plant}_units_running", f"{plant}_plant_available"]
        ]

        X_cols += [
            c for c in data.columns
            if c == "rainfall" or "outflow" in c
        ]

        X_cols += [
            c for c in data.columns
            if c.startswith(f"spill_{plant}")
            or c.startswith(f"tot_{plant}")
            or c.startswith(f"elev_{plant}")
        ]

        X_cols += [
            c for c in data.columns
            if "_lag" in c and (
                c.startswith("tot_agus")
                or c.startswith("elev_agus")
                or c.startswith("spill_agus")
                or c.startswith("rainfall")
                or "outflow" in c
            )
        ]

        X_cols += [
            c for c in data.columns
            if c.startswith(f"{plant}_upstream_")
        ]

        X_cols = list(dict.fromkeys([c for c in X_cols if c in data.columns]))

        print("Target:", target)
        print("Feature count:", len(X_cols))

        data = data.dropna(subset=[y_col] + X_cols).copy()

        print("Rows after feature prep:", len(data))

        if len(data) < CONFIG["min_rows_after_feature_prep"]:
            print(f"[SKIP {plant}] not enough rows after feature prep: {len(data)}")
            continue

        train_df, val_df, test_df = chronological_split(
            data,
            CONFIG["train_ratio"],
            CONFIG["val_ratio"]
        )

        X_train = train_df[X_cols].values.astype(np.float32)
        Y_train = train_df[[y_col]].values.astype(np.float32)

        X_val = val_df[X_cols].values.astype(np.float32)
        Y_val = val_df[[y_col]].values.astype(np.float32)

        X_test = test_df[X_cols].values.astype(np.float32)
        Y_test = test_df[[y_col]].values.astype(np.float32)

        x_scaler = MinMaxScaler()
        y_scaler = MinMaxScaler()

        X_train_s = x_scaler.fit_transform(X_train)
        X_val_s = x_scaler.transform(X_val)
        X_test_s = x_scaler.transform(X_test)

        Y_train_s = y_scaler.fit_transform(Y_train)
        Y_val_s = y_scaler.transform(Y_val)
        Y_test_s = y_scaler.transform(Y_test)

        model = build_rbfnn(
            input_dim=X_train_s.shape[1],
            output_dim=1,
            n_centers=n_centers,
            gamma_init=CONFIG["gamma_init"],
            learning_rate=CONFIG["learning_rate"]
        )

        init_centers_with_kmeans(model, X_train_s, n_centers)

        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=CONFIG["patience"],
                restore_best_weights=True
            )
        ]

        history = model.fit(
            X_train_s,
            Y_train_s,
            validation_data=(X_val_s, Y_val_s),
            epochs=CONFIG["epochs"],
            batch_size=CONFIG["batch_size"],
            verbose=1,
            callbacks=callbacks
        )

        # ------------------------------------------------------------
        # SAVE TRAINING VS VALIDATION LOSS PLOT
        # ------------------------------------------------------------
        plot_path = META_DIRS["loss_plots"] / f"{plant}_training_validation_loss.png"

        plt.figure(figsize=(8, 5))
        plt.plot(history.history["loss"], label="Training Loss")
        plt.plot(history.history["val_loss"], label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss (MSE)")
        plt.title(f"{plant.upper()} Training vs Validation Loss")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(plot_path, dpi=300)
        plt.close()

        print(f"Saved loss plot: {plot_path}")

        # ------------------------------------------------------------
        # VALIDATION AND TEST PREDICTIONS
        # ------------------------------------------------------------
        Y_val_pred_s = model.predict(X_val_s, verbose=0)
        Y_val_pred = y_scaler.inverse_transform(Y_val_pred_s)

        Y_test_pred_s = model.predict(X_test_s, verbose=0)
        Y_test_pred = y_scaler.inverse_transform(Y_test_pred_s)

        # ------------------------------------------------------------
        # DAILY VALIDATION METRICS
        # ------------------------------------------------------------
        val_daily_path = META_DIRS["validation_daily_metrics"] / f"{plant}_validation_daily_metrics.xlsx"

        save_daily_metrics(
            df_part=val_df,
            y_true=Y_val.flatten(),
            y_pred=Y_val_pred.flatten(),
            out_path=val_daily_path
        )

        print(f"Saved daily validation metrics: {val_daily_path}")

        # ------------------------------------------------------------
        # DAILY TESTING METRICS
        # ------------------------------------------------------------
        test_daily_path = META_DIRS["testing_daily_metrics"] / f"{plant}_testing_daily_metrics.xlsx"

        save_daily_metrics(
            df_part=test_df,
            y_true=Y_test.flatten(),
            y_pred=Y_test_pred.flatten(),
            out_path=test_daily_path
        )

        print(f"Saved daily testing metrics: {test_daily_path}")

        # ------------------------------------------------------------
        # OVERALL VALIDATION AND TESTING METRICS
        # ------------------------------------------------------------
        row = {
            "plant": plant,
            "best_n_centers": n_centers,
            "epochs_ran": len(history.history["loss"]),

            "val_mape": mape_safe(Y_val.flatten(), Y_val_pred.flatten()),
            "val_mae": mean_absolute_error(Y_val.flatten(), Y_val_pred.flatten()),
            "val_rmse": math.sqrt(mean_squared_error(Y_val.flatten(), Y_val_pred.flatten())),
            "val_r2": r2_score(Y_val.flatten(), Y_val_pred.flatten()),

            "test_mape": mape_safe(Y_test.flatten(), Y_test_pred.flatten()),
            "test_mae": mean_absolute_error(Y_test.flatten(), Y_test_pred.flatten()),
            "test_rmse": math.sqrt(mean_squared_error(Y_test.flatten(), Y_test_pred.flatten())),
            "test_r2": r2_score(Y_test.flatten(), Y_test_pred.flatten()),
        }

        summary_rows.append(row)

        # ------------------------------------------------------------
        # SAVE MODEL FILES
        # ------------------------------------------------------------
        model.save(model_path)
        joblib.dump(x_scaler, x_scaler_path)
        joblib.dump(y_scaler, y_scaler_path)

        with open(meta_path, "w") as f:
            json.dump({
                "plant": plant,
                "target": target,
                "X_cols": X_cols,
                "y_col": y_col,
                "is_delta_model": is_delta_model,
                "train_min": float(train_df[target].min()),
                "train_max": float(train_df[target].max()),
                "best_n_centers": int(n_centers)
            }, f, indent=2)

        print(f"[SAVED] {plant} | n_centers={n_centers} | delta_model={is_delta_model}")

    # ------------------------------------------------------------
    # SAVE SUMMARY METRICS
    # ------------------------------------------------------------
    if len(summary_rows) > 0:
        summary_df = pd.DataFrame(summary_rows)

        summary_path = META_DIRS["summary_metrics"] / "rbfnn_validation_testing_metrics.xlsx"
        summary_df.to_excel(summary_path, index=False)

        print("\nUpdated RBFNN validation/testing metrics:")
        print("Saved:", summary_path)
        print(summary_df)

    # ------------------------------------------------------------
    # FORECAST USING SAVED / JUST-TRAINED MODELS
    # ------------------------------------------------------------
    forecast = pd.DataFrame({
        "Date": planned["Date"].dt.date,
        "Hour": planned["Hour"].astype(int)
    })

    hist_base = df.copy()

    if "day_of_week" not in hist_base.columns:
        dt = pd.to_datetime(hist_base["datetime"])
        hour0 = hist_base["time"] - 1
        hist_base["hour_sin"] = np.sin(2 * np.pi * hour0 / 24)
        hist_base["hour_cos"] = np.cos(2 * np.pi * hour0 / 24)
        hist_base["day_of_week"] = dt.dt.dayofweek
        hist_base["month"] = dt.dt.month
        hist_base["is_weekend"] = (dt.dt.dayofweek >= 5).astype(int)

    for plant in CONFIG["plants"]:
        model_path = DIRS["models"] / f"rbfnn_{plant}.keras"
        x_scaler_path = DIRS["models"] / f"x_scaler_{plant}.pkl"
        y_scaler_path = DIRS["models"] / f"y_scaler_{plant}.pkl"
        meta_path = DIRS["models"] / f"meta_{plant}.json"

        if not all(p.exists() for p in [model_path, x_scaler_path, y_scaler_path, meta_path]):
            print(f"[FORECAST SKIP {plant}] missing model/scaler/meta files")
            continue

        model = tf.keras.models.load_model(
            model_path,
            custom_objects={"RBFLayer": RBFLayer},
            compile=False
        )

        x_scaler = joblib.load(x_scaler_path)
        y_scaler = joblib.load(y_scaler_path)

        with open(meta_path, "r") as f:
            meta = json.load(f)

        target = meta["target"]
        X_cols = meta["X_cols"]
        is_delta_model = meta.get("is_delta_model", False)
        train_min = meta.get("train_min", None)
        train_max = meta.get("train_max", None)

        hist_local = hist_base.copy()
        preds = []

        baseline_status_fixed = latest_baseline_status_map(hist_base, plant)

        plant_out_cols = [
            c for c in planned.columns
            if re.fullmatch(fr"out_{plant}_unit\d+", str(c))
        ]

        print(f"{plant} outage columns used: {plant_out_cols}")

        for i in range(24):
            date_i = planned.loc[i, "Date"]
            hour_i = int(planned.loc[i, "Hour"])

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

            row[f"{target}_diff1"] = (
                float(series.iloc[-1] - series.iloc[-2])
                if len(series) >= 2 else 0.0
            )

            row[f"{target}_diff2"] = (
                float(series.iloc[-1] - series.iloc[-3])
                if len(series) >= 3 else 0.0
            )

            row[f"{target}_diff3"] = (
                float(series.iloc[-1] - series.iloc[-4])
                if len(series) >= 4 else 0.0
            )

            row[f"{target}_recent_mean3"] = (
                float(series.tail(3).mean())
                if len(series) >= 3 else float(series.mean())
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

            extra_cols = [
                c for c in hist_local.columns
                if c == "rainfall" or "outflow" in c
            ]

            for c in extra_cols:
                row[c] = float(hist_local[c].iloc[-1])

            same_oper_cols = [
                c for c in hist_local.columns
                if c.startswith(f"spill_{plant}")
                or c.startswith(f"tot_{plant}")
                or c.startswith(f"elev_{plant}")
            ]

            for c in same_oper_cols:
                row[c] = float(hist_local[c].iloc[-1])

            oper_cols_for_lag = [
                c for c in hist_local.columns
                if c.startswith("tot_agus")
                or c.startswith("elev_agus")
                or c.startswith("spill_agus")
                or c == "rainfall"
                or "outflow" in c
            ]

            for c in oper_cols_for_lag:
                s = hist_local[c]

                for lag in CONFIG["lags"]:
                    row[f"{c}_lag{lag}"] = safe_lag(s, lag)

            up = CONFIG["upstream_map"][plant]

            if up is not None:
                up_target = f"total_gen_{up}"

                if up_target in hist_local.columns:
                    up_series = hist_local[up_target]

                    for lag in CONFIG["lags"]:
                        row[f"{plant}_upstream_gen_lag{lag}"] = safe_lag(up_series, lag)

                outflow_cols = [
                    c for c in hist_local.columns
                    if "outflow" in c
                ]

                if outflow_cols:
                    flow_series = hist_local[outflow_cols[0]]

                    for lag in CONFIG["lags"]:
                        row[f"{plant}_upstream_flow_lag{lag}"] = safe_lag(flow_series, lag)

            X_row = pd.DataFrame([{c: row.get(c, 0.0) for c in X_cols}])
            x_scaled = x_scaler.transform(X_row.values.astype(np.float32))

            yhat_scaled = model.predict(x_scaled, verbose=0)
            pred_base = float(y_scaler.inverse_transform(yhat_scaled)[0, 0])

            last_actual = float(series.iloc[-1])

            if is_delta_model:
                pred_base = last_actual + pred_base

            recent_mean3 = (
                float(series.tail(3).mean())
                if len(series) >= 3 else last_actual
            )

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

            if train_min is not None and train_max is not None:
                lower = max(0.0, train_min * 0.85)
                upper = train_max * 1.10
                pred_base = float(np.clip(pred_base, lower, upper))

            pred_out = pred_base

            if plant_out_cols:
                edited_status = edited_status_map(planned, i, plant)

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

            new_hist_row = {}

            for c in hist_local.columns:
                if c == "date":
                    new_hist_row[c] = pd.to_datetime(date_i)

                elif c == "time":
                    new_hist_row[c] = hour_i

                elif c == "datetime":
                    new_hist_row[c] = pd.to_datetime(date_i) + pd.Timedelta(hours=hour_i - 1)

                elif c == target:
                    new_hist_row[c] = pred_base

                elif c in plant_out_cols:
                    new_hist_row[c] = baseline_status_fixed.get(c, 1.0)

                elif c in ["hour_sin", "hour_cos", "day_of_week", "month", "is_weekend"]:
                    new_hist_row[c] = row[c]

                else:
                    new_hist_row[c] = hist_local[c].iloc[-1] if c in hist_local.columns else 0.0

            hist_local = pd.concat(
                [hist_local, pd.DataFrame([new_hist_row])],
                ignore_index=True
            )

        forecast[plant.upper()] = preds

    # ------------------------------------------------------------
    # SAVE FORECAST
    # ------------------------------------------------------------
    out_xlsx = DIRS["runtime"] / "Day_Ahead_24H_RBFNN_Forecast.xlsx"
    out_csv = DIRS["runtime"] / "Day_Ahead_24H_RBFNN_Forecast.csv"

    forecast.to_excel(out_xlsx, index=False)
    forecast.to_csv(out_csv, index=False)

    print("\nDone: Unified RBFNN cell")
    print("Saved:", out_xlsx)
    print("Saved:", out_csv)

    print("\nForecast preview:")
    print(forecast)


if __name__ == "__main__":
    main()
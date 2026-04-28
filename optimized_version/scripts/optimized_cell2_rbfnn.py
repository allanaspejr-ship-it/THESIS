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
import re
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
np.random.seed(42)
tf.random.set_seed(42)

PROJECT_DIR = Path(__file__).resolve().parents[2]
OPT_DIR = PROJECT_DIR / "optimized_version"
OUT_DIR = OPT_DIR / "outputs"
MODEL_DIR = OPT_DIR / "models"
META_DIR = OPT_DIR / "metadata"

for folder in [
    OUT_DIR,
    MODEL_DIR,
    META_DIR / "validation_testing_metrics",
    META_DIR / "validation_daily_metrics",
    META_DIR / "testing_daily_metrics",
]:
    folder.mkdir(parents=True, exist_ok=True)

CLEAN_PATH = OUT_DIR / "cleaned_hourly_data.parquet"
PLANNED_PATH = OUT_DIR / "Planned_Outages_Input.xlsx"

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

LAGS = [1, 2, 3, 6, 12, 24, 48, 72, 168]
ROLL_WINDOWS = [3, 6, 12, 24, 48, 168]
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
N_CENTERS = 120
EPOCHS = 80
BATCH_SIZE = 32
LEARNING_RATE = 0.001
SHRINKAGE_GRID = [0.10, 0.20, 0.35, 0.50, 0.75, 1.0]


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


def build_model(input_dim):
    inputs = tf.keras.Input(shape=(input_dim,))
    x = RBFLayer(N_CENTERS, gamma_init=1.0)(inputs)
    outputs = tf.keras.layers.Dense(1, activation="linear")(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(LEARNING_RATE), loss="mse")
    return model


def init_centers(model, x_train):
    n_clusters = min(N_CENTERS, len(x_train))
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=5)
    km.fit(x_train)
    centers = km.cluster_centers_
    if n_clusters < N_CENTERS:
        centers = np.vstack([centers, np.repeat(centers[-1:, :], N_CENTERS - n_clusters, axis=0)])
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
    cols = ["hour_sin", "hour_cos", "day_sin", "day_cos", "month_sin", "month_cos", "is_weekend", f"{target}_current"]
    prefixes = [f"{target}_lag", f"{target}_roll", f"{target}_diff", f"{plant}_upstream_"]
    cols += [c for c in data.columns if any(c.startswith(prefix) for prefix in prefixes)]
    cols += [c for c in data.columns if re.fullmatch(fr"out_{plant}_unit\d+", c)]
    cols += [f"{plant}_units_running", f"{plant}_plant_available"]
    cols += [c for c in data.columns if c.startswith(f"tot_{plant}") or c.startswith(f"elev_{plant}") or c.startswith(f"spill_{plant}")]
    cols += [c for c in data.columns if "_lag" in c and (c.startswith("tot_agus") or c.startswith("elev_agus") or "outflow" in c)]
    return list(dict.fromkeys([c for c in cols if c in data.columns]))


def save_daily_metrics(part_df, y_true, y_pred, plant, path):
    temp = part_df[["datetime"]].copy()
    temp["actual"] = np.asarray(y_true, dtype=float)
    temp["predicted"] = np.asarray(y_pred, dtype=float)
    temp["date"] = pd.to_datetime(temp["datetime"]).dt.date
    rows = []
    for date, group in temp.groupby("date"):
        rows.append({"date": date, **metrics_dict(group["actual"], group["predicted"], plant), "samples": len(group)})
    pd.DataFrame(rows).to_excel(path, index=False)


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


def ramp_limit(train_series):
    diffs = train_series.diff().abs().dropna()
    return float(max(1.0, diffs.quantile(0.98))) if not diffs.empty else 2.0


def load_planned():
    planned = pd.read_excel(PLANNED_PATH)
    planned.columns = [str(c).strip() for c in planned.columns]
    col_map = {c.lower(): c for c in planned.columns}
    planned = planned.rename(columns={col_map["date"]: "Date", col_map["hour"]: "Hour"})
    planned["Date"] = pd.to_datetime(planned["Date"])
    planned["Hour"] = pd.to_numeric(planned["Hour"]).astype(int)
    for col in [c for c in planned.columns if re.fullmatch(r"out_agus[124567]_unit\d+", c)]:
        planned[col] = np.where(pd.to_numeric(planned[col], errors="coerce").fillna(1) > 0, 1, 0)
    return planned


def predict_delta(model, x_scaler, y_scaler, x_frame):
    x_scaled = x_scaler.transform(x_frame.values.astype(np.float32))
    y_scaled = model.predict(x_scaled, verbose=0)
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
    })
    return pd.DataFrame([{c: row.get(c, 0.0) if pd.notna(row.get(c, 0.0)) else 0.0 for c in x_cols}])


def forecast_24h(raw_df, planned):
    forecast = pd.DataFrame({"Date": planned["Date"].dt.date, "Hour": planned["Hour"].astype(int)})
    hist = raw_df.copy()
    baseline_status = {plant: latest_status(hist, plant) for plant in PLANTS}

    loaded = {}
    for plant in PLANTS:
        meta = json.loads((MODEL_DIR / f"meta_{plant}.json").read_text())
        loaded[plant] = {
            "meta": meta,
            "model": tf.keras.models.load_model(MODEL_DIR / f"rbfnn_{plant}.keras", custom_objects={"RBFLayer": RBFLayer}, compile=False),
            "x_scaler": joblib.load(MODEL_DIR / f"x_scaler_{plant}.pkl"),
            "y_scaler": joblib.load(MODEL_DIR / f"y_scaler_{plant}.pkl"),
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
            base_pred = last_val + float(meta["best_shrinkage"]) * delta
            base_pred = float(np.clip(base_pred, last_val - float(meta["ramp_limit"]), last_val + float(meta["ramp_limit"])))
            base_pred = float(np.clip(base_pred, 0.0, CAPACITY_MW[plant] * 1.05))

            p_status = planned_status(planned, step, plant)
            ratio = availability_ratio(plant, baseline_status[plant], p_status)
            adjusted = float(np.clip(base_pred * ratio, 0.0, CAPACITY_MW[plant] * 1.05)) if ratio > 0 else 0.0
            forecast.loc[step, plant.upper()] = adjusted
            new_row[target] = base_pred
            for out_col, value in baseline_status[plant].items():
                new_row[out_col] = value

        hist = pd.concat([hist, pd.DataFrame([new_row])], ignore_index=True)
    return forecast


def main():
    if not CLEAN_PATH.exists() or not PLANNED_PATH.exists():
        raise FileNotFoundError("Run optimized_cell1_clean.py first.")

    raw_df = pd.read_parquet(CLEAN_PATH)
    raw_df["datetime"] = pd.to_datetime(raw_df["datetime"])
    raw_df = raw_df.sort_values("datetime").reset_index(drop=True)
    planned = load_planned()
    feat_df = add_features(raw_df)
    summary_rows = []

    for plant in PLANTS:
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

        model = build_model(x_train.shape[1])
        init_centers(model, x_train)
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
        val_current = val_df[target].values.astype(float)
        test_current = test_df[target].values.astype(float)
        val_actual = val_df[f"{target}_tplus1"].values.astype(float)
        test_actual = test_df[f"{target}_tplus1"].values.astype(float)

        best = None
        for shrinkage in SHRINKAGE_GRID:
            val_pred_candidate = np.clip(val_current + shrinkage * val_delta_pred, 0.0, CAPACITY_MW[plant] * 1.05)
            m = metrics_dict(val_actual, val_pred_candidate, plant)
            score = (m["r2"], -m["operational_mape"], -m["rmse"])
            if best is None or score > best["score"]:
                best = {"shrinkage": shrinkage, "score": score}

        shrinkage = best["shrinkage"]
        val_pred = np.clip(val_current + shrinkage * val_delta_pred, 0.0, CAPACITY_MW[plant] * 1.05)
        test_pred = np.clip(test_current + shrinkage * test_delta_pred, 0.0, CAPACITY_MW[plant] * 1.05)
        val_metrics = metrics_dict(val_actual, val_pred, plant)
        test_metrics = metrics_dict(test_actual, test_pred, plant)
        limit = ramp_limit(train_df[target])

        row = {
            "plant": plant,
            "feature_count": len(x_cols),
            "rows_after_feature_prep": len(data),
            "epochs_ran": len(history.history["loss"]),
            "best_shrinkage": shrinkage,
            "ramp_limit_mw": limit,
            "val_operational_mape": val_metrics["operational_mape"],
            "val_mae": val_metrics["mae"],
            "val_rmse": val_metrics["rmse"],
            "val_r2": val_metrics["r2"],
            "test_operational_mape": test_metrics["operational_mape"],
            "test_mae": test_metrics["mae"],
            "test_rmse": test_metrics["rmse"],
            "test_r2": test_metrics["r2"],
            "meets_thresholds": (
                val_metrics["operational_mape"] < 10
                and test_metrics["operational_mape"] < 10
                and val_metrics["r2"] >= 0.80
                and test_metrics["r2"] >= 0.80
            ),
        }
        summary_rows.append(row)

        model.save(MODEL_DIR / f"rbfnn_{plant}.keras")
        joblib.dump(x_scaler, MODEL_DIR / f"x_scaler_{plant}.pkl")
        joblib.dump(y_scaler, MODEL_DIR / f"y_scaler_{plant}.pkl")
        (MODEL_DIR / f"meta_{plant}.json").write_text(json.dumps({
            "plant": plant,
            "target": target,
            "y_col": y_col,
            "X_cols": x_cols,
            "capacity_mw": CAPACITY_MW[plant],
            "operational_mape_threshold_mw": operational_threshold(plant),
            "best_shrinkage": shrinkage,
            "ramp_limit": limit,
            "model_type": "RBFNN residual/delta model anchored to persistence",
        }, indent=2))

        save_daily_metrics(val_df, val_actual, val_pred, plant, META_DIR / "validation_daily_metrics" / f"{plant}_validation_daily_metrics.xlsx")
        save_daily_metrics(test_df, test_actual, test_pred, plant, META_DIR / "testing_daily_metrics" / f"{plant}_testing_daily_metrics.xlsx")
        print(pd.DataFrame([row]).to_string(index=False))

    summary = pd.DataFrame(summary_rows)
    summary_path = META_DIR / "validation_testing_metrics" / "optimized_rbfnn_validation_testing_metrics.xlsx"
    summary.to_excel(summary_path, index=False)
    forecast = forecast_24h(raw_df, planned)
    forecast.to_excel(OUT_DIR / "Day_Ahead_24H_Optimized_RBFNN_Forecast.xlsx", index=False)
    forecast.to_csv(OUT_DIR / "Day_Ahead_24H_Optimized_RBFNN_Forecast.csv", index=False)

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
    comparison.to_excel(META_DIR / "optimized_vs_original_summary.xlsx", index=False)

    print("\nOptimized RBFNN summary:")
    print(summary.to_string(index=False))
    print("Saved:", summary_path)


if __name__ == "__main__":
    main()

"""
Cell 4: generate thesis figures and organized metadata outputs.

This script only writes thesis_figures and selected organized metadata files.
It does not delete or move benchmark, data, models, outputs, or scripts.
"""

from pathlib import Path
import json
import math
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from openpyxl import load_workbook

warnings.filterwarnings("ignore", category=UserWarning)

# ============================================================
# PATH AND REPORT CONFIGURATION
# ============================================================

# Defines metadata and figure locations used for thesis-ready outputs.
BASE_DIR = Path(__file__).resolve().parents[1]
METADATA_DIR = BASE_DIR / "metadata"
OVERALL_METRICS_DIR = METADATA_DIR / "overall_metrics"
DAY_AHEAD_BACKTEST_DIR = METADATA_DIR / "day_ahead_backtest"
THESIS_FIGURES_DIR = BASE_DIR / "thesis_figures"
MODEL_METRICS_DIR = METADATA_DIR / "model_metrics"

# Stores plant labels, model source files, and figure output folders.
PLANTS = ["agus1", "agus2", "agus4", "agus5", "agus6", "agus7"]
PLANT_LABELS = {
    "agus1": "Agus 1",
    "agus2": "Agus 2",
    "agus4": "Agus 4",
    "agus5": "Agus 5",
    "agus6": "Agus 6",
    "agus7": "Agus 7",
}
MODEL_FILES = {
    "RBFNN": {
        "metrics": OVERALL_METRICS_DIR / "rbfnn_validation_testing_metrics.xlsx",
        "predictions": OVERALL_METRICS_DIR / "rbfnn_testing_predictions.xlsx",
        "folder": METADATA_DIR / "rbfnn",
    },
    "Random Forest": {
        "metrics": OVERALL_METRICS_DIR / "random_forest_validation_testing_metrics.xlsx",
        "predictions": OVERALL_METRICS_DIR / "random_forest_testing_predictions.xlsx",
        "folder": METADATA_DIR / "random_forest",
    },
    "XGBoost": {
        "metrics": OVERALL_METRICS_DIR / "xgboost_validation_testing_metrics.xlsx",
        "predictions": OVERALL_METRICS_DIR / "xgboost_testing_predictions.xlsx",
        "folder": METADATA_DIR / "xgboost",
    },
}
DAY_AHEAD_MODEL_FILES = {
    "RBFNN": {
        "validation": DAY_AHEAD_BACKTEST_DIR / "rbfnn_validation_day_ahead_metrics.xlsx",
        "testing": DAY_AHEAD_BACKTEST_DIR / "rbfnn_testing_day_ahead_metrics.xlsx",
    },
    "Random Forest": {
        "validation": DAY_AHEAD_BACKTEST_DIR / "random_forest_validation_day_ahead_metrics.xlsx",
        "testing": DAY_AHEAD_BACKTEST_DIR / "random_forest_testing_day_ahead_metrics.xlsx",
    },
    "XGBoost": {
        "validation": DAY_AHEAD_BACKTEST_DIR / "xgboost_validation_day_ahead_metrics.xlsx",
        "testing": DAY_AHEAD_BACKTEST_DIR / "xgboost_testing_day_ahead_metrics.xlsx",
    },
}
MODEL_KEYS = {
    "RBFNN": "rbfnn",
    "Random Forest": "random_forest",
    "XGBoost": "xgboost",
}
MODEL_DIRS = {
    "RBFNN": BASE_DIR / "models" / "rbfnn",
    "Random Forest": BASE_DIR / "models" / "random_forest",
    "XGBoost": BASE_DIR / "models" / "xgboost",
}
CAPACITY_MW = {"agus1": 80.0, "agus2": 180.0, "agus4": 158.1, "agus5": 55.0, "agus6": 219.0, "agus7": 54.0}

FIGURE_DIRS = {
    "cleaned_profiles": THESIS_FIGURES_DIR / "cleaned_profiles",
    "testing_actual_vs_forecast": THESIS_FIGURES_DIR / "testing_actual_vs_forecast",
    "testing_actual_vs_forecast_by_model": THESIS_FIGURES_DIR / "testing_actual_vs_forecast_by_model",
    "all_models_testing_actual_vs_forecast": THESIS_FIGURES_DIR / "all_models_testing_actual_vs_forecast",
    "error_analysis": THESIS_FIGURES_DIR / "error_analysis",
    "benchmark_comparison": THESIS_FIGURES_DIR / "benchmark_comparison",
    "day_ahead_forecast": THESIS_FIGURES_DIR / "day_ahead_forecast",
}

SAVED_OUTPUTS = []


# ============================================================
# OUTPUT AND LOADING HELPERS
# ============================================================

# Creates all figure and organized metadata output folders.
def ensure_dirs():
    for path in FIGURE_DIRS.values():
        path.mkdir(parents=True, exist_ok=True)
    (METADATA_DIR / "overall_comparison").mkdir(parents=True, exist_ok=True)
    for model_key in MODEL_KEYS.values():
        (MODEL_METRICS_DIR / model_key).mkdir(parents=True, exist_ok=True)


# Tracks saved files in the console output.
def save_path(path):
    SAVED_OUTPUTS.append(path)
    print(f"Saved: {path}")


# Stops figure or table generation when a required source file is missing.
def require_file(path):
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    return path


# Rebuilds a sortable datetime column from available date and hour fields.
def rebuild_datetime(df):
    out = df.copy()
    if "datetime" in out.columns:
        out["datetime"] = pd.to_datetime(out["datetime"])
        return out.sort_values("datetime").reset_index(drop=True)

    date_col = next((col for col in ["date", "Date"] if col in out.columns), None)
    hour_col = next((col for col in ["time", "Hour", "hour"] if col in out.columns), None)
    if date_col is None or hour_col is None:
        raise ValueError("Cannot rebuild datetime without date and hour/time columns.")

    dates = pd.to_datetime(out[date_col])
    hours_raw = out[hour_col]
    if pd.api.types.is_numeric_dtype(hours_raw):
        hours = pd.to_numeric(hours_raw).astype(int)
        offset = np.where(hours.between(1, 24), hours - 1, hours)
        out["datetime"] = dates + pd.to_timedelta(offset, unit="h")
    else:
        parsed_hours = pd.to_datetime(hours_raw.astype(str), errors="coerce")
        if parsed_hours.notna().any():
            out["datetime"] = dates + pd.to_timedelta(parsed_hours.dt.hour, unit="h")
        else:
            hours = pd.to_numeric(hours_raw.astype(str).str.extract(r"(\d+)")[0]).fillna(0).astype(int)
            offset = np.where((hours >= 1) & (hours <= 24), hours - 1, hours)
            out["datetime"] = dates + pd.to_timedelta(offset, unit="h")
    return out.sort_values("datetime").reset_index(drop=True)


# Formats Excel outputs for readability in thesis appendices and tables.
def format_excel(path):
    wb = load_workbook(path)
    for ws in wb.worksheets:
        ws.freeze_panes = "A2"
        ws.auto_filter.ref = ws.dimensions
        for column_cells in ws.columns:
            header = str(column_cells[0].value or "")
            values = [str(cell.value) for cell in column_cells[1:120] if cell.value is not None]
            max_len = max([len(header)] + [len(value) for value in values])
            ws.column_dimensions[column_cells[0].column_letter].width = min(max(max_len + 2, 10), 36)
    wb.save(path)


# Saves a dataframe to Excel and applies workbook formatting.
def save_excel(df, path, sheet_name="Sheet1"):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.stem}.tmp{path.suffix}")
    if temp_path.exists():
        temp_path.unlink()
    try:
        with pd.ExcelWriter(temp_path, engine="openpyxl") as writer:
            df.to_excel(writer, sheet_name=sheet_name, index=False)
        format_excel(temp_path)
        temp_path.replace(path)
    except PermissionError:
        if temp_path.exists():
            temp_path.unlink()
        raise PermissionError(
            f"Cannot update {path}. Close this workbook in Excel, then rerun cell 4."
        )
    save_path(path)


def copy_excel(src, dst):
    require_file(src)
    df = pd.read_excel(src)
    save_excel(df, dst)


# Saves the current Matplotlib figure using thesis figure settings.
def save_figure(path, tight=True):
    path.parent.mkdir(parents=True, exist_ok=True)
    if tight:
        plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    save_path(path)


# Loads the cleaned hourly dataset used for Chapter 4 profile figures.
def read_cleaned_data():
    parquet_path = BASE_DIR / "outputs" / "cleaned_data" / "cleaned_hourly_data.parquet"
    xlsx_path = BASE_DIR / "outputs" / "cleaned_data" / "cleaned_hourly_data.xlsx"
    if parquet_path.exists():
        df = pd.read_parquet(parquet_path)
    elif xlsx_path.exists():
        df = pd.read_excel(xlsx_path)
    else:
        raise FileNotFoundError(f"Missing cleaned data: {parquet_path} or {xlsx_path}")
    return rebuild_datetime(df)


# Reads model-level validation and testing metrics.
def metric_count_frame(model_name, split):
    pred_path = DAY_AHEAD_BACKTEST_DIR / f"{MODEL_KEYS[model_name]}_{split}_day_ahead_predictions.xlsx"
    if not pred_path.exists():
        return pd.DataFrame()
    predictions = pd.read_excel(pred_path)
    rows = []
    for plant, group in predictions.groupby("plant", sort=False):
        plant = str(plant)
        threshold = max(1.0, 0.01 * CAPACITY_MW[plant])
        actual = pd.to_numeric(group["actual_generation"], errors="coerce").fillna(0.0).abs()
        included = int((actual >= threshold).sum())
        excluded = int((actual < threshold).sum())
        excluded_fraction = float(excluded / len(actual)) if len(actual) else np.nan
        rows.append({
            "plant": plant,
            "mape_rows_included": included,
            "mape_rows_excluded": excluded,
            "mape_rows_excluded_fraction": excluded_fraction,
            "low_load_regime": bool(excluded_fraction > 0.40),
        })
    return pd.DataFrame(rows)


def read_metrics(model_name):
    day_ahead_validation = DAY_AHEAD_MODEL_FILES[model_name]["validation"]
    day_ahead_testing = DAY_AHEAD_MODEL_FILES[model_name]["testing"]
    if day_ahead_validation.exists() and day_ahead_testing.exists():
        validation = pd.read_excel(day_ahead_validation).rename(
            columns={
                "operational_mape": "val_operational_mape",
                "mae": "val_mae",
                "rmse": "val_rmse",
                "r2": "val_r2",
            }
        )
        testing = pd.read_excel(day_ahead_testing).rename(
            columns={
                "operational_mape": "test_operational_mape",
                "mae": "test_mae",
                "rmse": "test_rmse",
                "r2": "test_r2",
            }
        )
        feature_counts = read_feature_counts(model_name)
        validation_cols = ["model", "plant", "val_operational_mape", "val_mae", "val_rmse", "val_r2"]
        testing_cols = ["plant", "test_operational_mape", "test_mae", "test_rmse", "test_r2"]
        for col in ["mape_rows_included", "mape_rows_excluded", "mape_rows_excluded_fraction", "low_load_regime"]:
            if col in validation.columns:
                validation_cols.append(col)
            if col in testing.columns:
                testing_cols.append(col)
        df = validation[validation_cols].merge(
            testing[testing_cols],
            on="plant",
            how="inner",
            suffixes=("_validation", "_testing"),
        )
        for split_name, suffix in [("validation", "_validation"), ("testing", "_testing")]:
            counts = metric_count_frame(model_name, split_name)
            if counts.empty:
                continue
            for col in ["mape_rows_included", "mape_rows_excluded", "mape_rows_excluded_fraction", "low_load_regime"]:
                target_col = f"{col}{suffix}"
                if target_col not in df.columns:
                    df[target_col] = df["plant"].astype(str).map(dict(zip(counts["plant"].astype(str), counts[col])))
        df["feature_count"] = df["plant"].map(feature_counts).fillna(0).astype(int)
        df["evaluation_type"] = "rolling_24h_day_ahead_backtest"
        if "low_load_regime_testing" in df.columns:
            df["low_load_regime"] = df["low_load_regime_testing"].astype(bool)
        elif "low_load_regime" in df.columns:
            df["low_load_regime"] = df["low_load_regime"].astype(bool)
        else:
            df["low_load_regime"] = False
        df["plant"] = pd.Categorical(df["plant"], categories=PLANTS, ordered=True)
        return df.sort_values("plant").reset_index(drop=True)

    warning = (
        f"Missing rolling day-ahead metrics for {model_name}: "
        f"{day_ahead_validation} and/or {day_ahead_testing}. "
        "Strict rolling 24-hour day-ahead metrics are required for thesis outputs."
    )
    warnings.warn(warning)
    print("WARNING:", warning)
    raise FileNotFoundError(warning)


# Reads testing predictions and sorts them by plant and time.
def read_predictions(model_name):
    day_ahead_testing = DAY_AHEAD_BACKTEST_DIR / f"{MODEL_KEYS[model_name]}_testing_day_ahead_predictions.xlsx"
    if day_ahead_testing.exists():
        df = pd.read_excel(day_ahead_testing)
        df = rebuild_datetime(df)
        df["plant"] = pd.Categorical(df["plant"], categories=PLANTS, ordered=True)
        return df.sort_values(["plant", "datetime"]).reset_index(drop=True)

    raise FileNotFoundError(f"Required strict day-ahead prediction file not found: {day_ahead_testing}")


# Reads saved feature counts for day-ahead metric summaries.
def read_feature_counts(model_name):
    if model_name not in MODEL_FILES:
        return {plant: 0 for plant in PLANTS}
    if model_name == "RBFNN":
        counts = {}
        for plant in PLANTS:
            meta_path = MODEL_DIRS[model_name] / f"meta_{plant}.json"
            if meta_path.exists():
                meta = json.loads(meta_path.read_text())
                counts[plant] = len(meta.get("X_cols", []))
        if counts:
            return counts

    metrics_path = MODEL_FILES[model_name]["metrics"]
    if metrics_path.exists():
        metrics = pd.read_excel(metrics_path)
        if {"plant", "feature_count"}.issubset(metrics.columns):
            return dict(zip(metrics["plant"], metrics["feature_count"]))

    audit_path = OVERALL_METRICS_DIR / "benchmark_fairness_audit.xlsx"
    if audit_path.exists():
        audit = pd.read_excel(audit_path)
        if {"model", "plant", "feature count"}.issubset(audit.columns):
            audit = audit[audit["model"] == model_name]
            return dict(zip(audit["plant"], audit["feature count"]))
    return {}


# Converts validation/testing metric columns into one common table schema.
def normalized_metric_frames(metrics_df, model_name):
    base = metrics_df.copy()
    base["model"] = model_name
    base["evaluation_type"] = base.get("evaluation_type", "unknown")
    base["low_load_regime"] = base.get("low_load_regime", False)
    validation_base = base.copy()
    testing_base = base.copy()
    for source, target in [
        ("mape_rows_included_validation", "mape_rows_included"),
        ("mape_rows_excluded_validation", "mape_rows_excluded"),
        ("mape_rows_excluded_fraction_validation", "mape_rows_excluded_fraction"),
    ]:
        validation_base[target] = validation_base[source] if source in validation_base.columns else np.nan
    for source, target in [
        ("mape_rows_included_testing", "mape_rows_included"),
        ("mape_rows_excluded_testing", "mape_rows_excluded"),
        ("mape_rows_excluded_fraction_testing", "mape_rows_excluded_fraction"),
    ]:
        testing_base[target] = testing_base[source] if source in testing_base.columns else np.nan

    output_cols = [
        "model",
        "plant",
        "feature_count",
        "MAPE",
        "MAE",
        "RMSE",
        "R2",
        "evaluation_type",
        "mape_rows_included",
        "mape_rows_excluded",
        "mape_rows_excluded_fraction",
        "low_load_regime",
    ]
    validation = validation_base.rename(
        columns={
            "val_operational_mape": "MAPE",
            "val_mae": "MAE",
            "val_rmse": "RMSE",
            "val_r2": "R2",
        }
    )[output_cols]
    testing = testing_base.rename(
        columns={
            "test_operational_mape": "MAPE",
            "test_mae": "MAE",
            "test_rmse": "RMSE",
            "test_r2": "R2",
        }
    )[output_cols]
    return validation, testing


# ============================================================
# THESIS FIGURE GENERATION
# ============================================================

# Generates cleaned hydropower generation profile figures per plant.
def generate_cleaned_profile_figures():
    df = read_cleaned_data()
    for idx, plant in enumerate(PLANTS, start=1):
        column = f"total_gen_{plant}"
        if column not in df.columns:
            raise ValueError(f"Missing cleaned profile column: {column}")
        plt.figure(figsize=(12, 5))
        plt.plot(df["datetime"], df[column], linewidth=0.8, color="#1f77b4")
        plt.title(f"Figure 4.{idx}: Cleaned Generation Profile - {PLANT_LABELS[plant]}")
        plt.xlabel("Datetime")
        plt.ylabel("Generation (MW)")
        plt.grid(True, alpha=0.25)
        save_figure(FIGURE_DIRS["cleaned_profiles"] / f"figure_4_{idx:02d}_{plant}_cleaned_profile.png")


# Generates actual-versus-RBFNN forecast plots for testing data.
def generate_testing_actual_vs_forecast_figures():
    df = read_predictions("RBFNN")
    for offset, plant in enumerate(PLANTS, start=13):
        plant_df = df[df["plant"] == plant]
        plt.figure(figsize=(12, 5))
        plt.plot(plant_df["datetime"], plant_df["actual_generation"], label="Actual", linewidth=1.0)
        plt.plot(plant_df["datetime"], plant_df["predicted_generation"], label="Forecast", linewidth=1.0)
        plt.title(f"Figure 4.{offset}: Testing Actual vs Forecast - {PLANT_LABELS[plant]} (rolling 24h day-ahead)")
        plt.xlabel("Datetime")
        plt.ylabel("Generation (MW)")
        plt.grid(True, alpha=0.25)
        plt.legend()
        save_figure(FIGURE_DIRS["testing_actual_vs_forecast"] / f"figure_4_{offset:02d}_{plant}_testing_actual_vs_forecast.png")


# Generates one combined RBFNN testing figure with actual and forecast lines by plant.
def generate_rbfnn_testing_actual_vs_forecast_panel():
    df = read_predictions("RBFNN")
    plot_data = df[
        ["datetime", "plant", "actual_generation", "predicted_generation", "model"]
    ].copy()
    plot_data["plant_label"] = plot_data["plant"].astype(str).map(PLANT_LABELS)
    save_excel(
        plot_data,
        OVERALL_METRICS_DIR / "rbfnn_testing_actual_vs_forecast_plot_data.xlsx",
        sheet_name="RBFNN Testing Plot Data",
    )

    fig, axes = plt.subplots(3, 2, figsize=(15, 10), sharex=False)
    axes = axes.flatten()
    for ax, plant in zip(axes, PLANTS):
        plant_df = plot_data[plot_data["plant"].astype(str) == plant]
        ax.plot(
            plant_df["datetime"],
            plant_df["actual_generation"],
            label="Actual",
            linewidth=1.0,
            color="#1f77b4",
        )
        ax.plot(
            plant_df["datetime"],
            plant_df["predicted_generation"],
            label="RBFNN Testing Forecast",
            linewidth=1.0,
            color="#d62728",
        )
        ax.set_title(f"{PLANT_LABELS[plant]}")
        ax.set_ylabel("MW")
        ax.grid(True, alpha=0.25)

    for ax in axes[-2:]:
        ax.set_xlabel("Testing Datetime")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.965))
    fig.suptitle("RBFNN Testing Actual vs Forecast by Plant (rolling 24h day-ahead)", y=0.995)
    fig.autofmt_xdate(rotation=25)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    save_figure(FIGURE_DIRS["testing_actual_vs_forecast"] / "rbfnn_testing_actual_vs_forecast_all_plants.png", tight=False)


def generate_testing_actual_vs_forecast_by_model_figures():
    model_colors = {
        "RBFNN": "#d62728",
        "Random Forest": "#2ca02c",
        "XGBoost": "#9467bd",
    }
    safe_names = {
        "RBFNN": "rbfnn",
        "Random Forest": "random_forest",
        "XGBoost": "xgboost",
    }
    rows = []
    for model_name in MODEL_KEYS:
        model_df = read_predictions(model_name)
        model_dir = FIGURE_DIRS["testing_actual_vs_forecast_by_model"] / safe_names[model_name]
        model_dir.mkdir(parents=True, exist_ok=True)
        for plant in PLANTS:
            plant_df = model_df[model_df["plant"].astype(str) == plant].copy()
            rows.append(
                plant_df[
                    ["datetime", "plant", "actual_generation", "predicted_generation", "model"]
                ]
            )
            plt.figure(figsize=(12, 5))
            plt.plot(
                plant_df["datetime"],
                plant_df["actual_generation"],
                label="Actual",
                linewidth=1.1,
                color="#1f77b4",
            )
            plt.plot(
                plant_df["datetime"],
                plant_df["predicted_generation"],
                label=f"{model_name} Forecast",
                linewidth=1.0,
                color=model_colors[model_name],
            )
            plt.title(f"Testing Actual vs Forecast - {PLANT_LABELS[plant]} ({model_name}, rolling 24h day-ahead)")
            plt.xlabel("Datetime")
            plt.ylabel("Generation (MW)")
            plt.grid(True, alpha=0.25)
            plt.legend()
            save_figure(model_dir / f"{safe_names[model_name]}_{plant}_testing_actual_vs_forecast.png")

    plot_data = pd.concat(rows, ignore_index=True)
    plot_data["plant_label"] = plot_data["plant"].astype(str).map(PLANT_LABELS)
    save_excel(
        plot_data.sort_values(["model", "plant", "datetime"]),
        OVERALL_METRICS_DIR / "testing_actual_vs_forecast_by_model_plot_data.xlsx",
        sheet_name="By Model Plot Data",
    )


def generate_all_models_testing_actual_vs_forecast_figures():
    frames = []
    for model_name in MODEL_KEYS:
        df = read_predictions(model_name)
        df = df[["datetime", "plant", "actual_generation", "predicted_generation", "model"]].copy()
        df["model"] = model_name
        frames.append(df)
    plot_data = pd.concat(frames, ignore_index=True)
    plot_data["plant_label"] = plot_data["plant"].astype(str).map(PLANT_LABELS)
    save_excel(
        plot_data.sort_values(["plant", "model", "datetime"]),
        OVERALL_METRICS_DIR / "all_models_testing_actual_vs_forecast_plot_data.xlsx",
        sheet_name="All Models Testing Plot Data",
    )

    model_colors = {
        "RBFNN": "#d62728",
        "Random Forest": "#2ca02c",
        "XGBoost": "#9467bd",
    }
    for plant in PLANTS:
        plant_df = plot_data[plot_data["plant"].astype(str) == plant].copy()
        actual = plant_df[plant_df["model"] == "RBFNN"][["datetime", "actual_generation"]].drop_duplicates("datetime")
        plt.figure(figsize=(12, 5))
        plt.plot(actual["datetime"], actual["actual_generation"], label="Actual", linewidth=1.1, color="#1f77b4")
        for model_name, color in model_colors.items():
            model_df = plant_df[plant_df["model"] == model_name]
            plt.plot(
                model_df["datetime"],
                model_df["predicted_generation"],
                label=f"{model_name} Forecast",
                linewidth=0.95,
                color=color,
                alpha=0.9,
            )
        plt.title(f"Testing Actual vs Forecast - {PLANT_LABELS[plant]} (All Models, rolling 24h day-ahead)")
        plt.xlabel("Datetime")
        plt.ylabel("Generation (MW)")
        plt.grid(True, alpha=0.25)
        plt.legend(ncol=2)
        save_figure(
            FIGURE_DIRS["all_models_testing_actual_vs_forecast"]
            / f"{plant}_all_models_testing_actual_vs_forecast.png"
        )

    fig, axes = plt.subplots(3, 2, figsize=(16, 10), sharex=False)
    axes = axes.flatten()
    for ax, plant in zip(axes, PLANTS):
        plant_df = plot_data[plot_data["plant"].astype(str) == plant].copy()
        actual = plant_df[plant_df["model"] == "RBFNN"][["datetime", "actual_generation"]].drop_duplicates("datetime")
        ax.plot(actual["datetime"], actual["actual_generation"], label="Actual", linewidth=1.0, color="#1f77b4")
        for model_name, color in model_colors.items():
            model_df = plant_df[plant_df["model"] == model_name]
            ax.plot(model_df["datetime"], model_df["predicted_generation"], label=model_name, linewidth=0.85, color=color)
        ax.set_title(PLANT_LABELS[plant])
        ax.set_ylabel("MW")
        ax.grid(True, alpha=0.25)
    for ax in axes[-2:]:
        ax.set_xlabel("Testing Datetime")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.965))
    fig.suptitle("Testing Actual vs Forecast by Plant (All Models, rolling 24h day-ahead)", y=0.995)
    fig.autofmt_xdate(rotation=25)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    save_figure(
        FIGURE_DIRS["all_models_testing_actual_vs_forecast"] / "all_models_testing_actual_vs_forecast_panel.png",
        tight=False,
    )


# Computes plant-level MAPE from actual and predicted testing values.
def plant_mape(df):
    rows = []
    for plant in PLANTS:
        plant_df = df[df["plant"] == plant].copy()
        actual = plant_df["actual_generation"].astype(float)
        predicted = plant_df["predicted_generation"].astype(float)
        mask = actual.abs() > 0
        mape = np.nan if not mask.any() else np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100.0
        rows.append({"plant": plant, "MAPE": mape})
    return pd.DataFrame(rows)


# Generates scatter, residual, and MAPE plots for RBFNN error analysis.
def generate_error_analysis_figures():
    df = read_predictions("RBFNN")
    actual = df["actual_generation"].astype(float)
    predicted = df["predicted_generation"].astype(float)
    residual = actual - predicted

    plt.figure(figsize=(7, 6))
    plt.scatter(actual, predicted, s=10, alpha=0.35)
    low = min(actual.min(), predicted.min())
    high = max(actual.max(), predicted.max())
    plt.plot([low, high], [low, high], color="black", linestyle="--", linewidth=1)
    plt.title("Figure 4.19: RBFNN Actual vs Predicted Scatter (rolling 24h day-ahead)")
    plt.xlabel("Actual Generation (MW)")
    plt.ylabel("Predicted Generation (MW)")
    plt.grid(True, alpha=0.25)
    save_figure(FIGURE_DIRS["error_analysis"] / "rbfnn_actual_vs_predicted_scatter.png")

    plt.figure(figsize=(8, 5))
    plt.hist(residual, bins=60, color="#4c78a8", edgecolor="white")
    plt.axvline(0, color="black", linestyle="--", linewidth=1)
    plt.title("Figure 4.20: RBFNN Residual Distribution (rolling 24h day-ahead)")
    plt.xlabel("Residual (Actual - Forecast, MW)")
    plt.ylabel("Frequency")
    plt.grid(True, axis="y", alpha=0.25)
    save_figure(FIGURE_DIRS["error_analysis"] / "rbfnn_residual_distribution.png")

    mape_df = plant_mape(df)
    plt.figure(figsize=(8, 5))
    plt.bar([PLANT_LABELS[p] for p in mape_df["plant"]], mape_df["MAPE"], color="#59a14f")
    plt.title("Figure 4.21: RBFNN Testing MAPE per Plant (rolling 24h day-ahead)")
    plt.xlabel("Plant")
    plt.ylabel("MAPE (%)")
    plt.grid(True, axis="y", alpha=0.25)
    save_figure(FIGURE_DIRS["error_analysis"] / "rbfnn_testing_mape_per_plant.png")


# Generates RBFNN, Random Forest, and XGBoost comparison bar charts.
def generate_benchmark_comparison_figures():
    all_metrics = pd.concat([read_metrics(model) for model in DAY_AHEAD_MODEL_FILES], ignore_index=True)
    model_order = [model for model in DAY_AHEAD_MODEL_FILES if model in set(all_metrics["model"])]
    pivot_mape = all_metrics.pivot(index="plant", columns="model", values="test_operational_mape").loc[PLANTS, model_order]
    pivot_rmse = all_metrics.pivot(index="plant", columns="model", values="test_rmse").loc[PLANTS, model_order]
    evaluation_type = ", ".join(sorted(set(all_metrics["evaluation_type"].astype(str))))

    for fig_num, pivot, metric, ylabel, filename in [
        (22, pivot_mape, "Testing MAPE", "MAPE (%)", "testing_mape_model_comparison.png"),
        (23, pivot_rmse, "Testing RMSE", "RMSE (MW)", "testing_rmse_model_comparison.png"),
    ]:
        ax = pivot.rename(index=PLANT_LABELS).plot(kind="bar", figsize=(10, 5), width=0.78)
        ax.set_title(f"Figure 4.{fig_num}: {metric} Model Comparison ({evaluation_type})")
        ax.set_xlabel("Plant")
        ax.set_ylabel(ylabel)
        ax.grid(True, axis="y", alpha=0.25)
        ax.legend(title="Model")
        plt.xticks(rotation=0)
        save_figure(FIGURE_DIRS["benchmark_comparison"] / filename)


# Reads one day-ahead forecast workbook for cascade-level plotting.
def read_day_ahead_forecast(path, model_name):
    df = pd.read_excel(require_file(path))
    df = rebuild_datetime(df)
    if "Total_Cascade_Generation_MW" not in df.columns:
        raise ValueError(f"Missing Total_Cascade_Generation_MW in {path}")
    df["model"] = model_name
    return df[["datetime", "Total_Cascade_Generation_MW", "model"]]


# Generates the total cascade day-ahead comparison figure.
def generate_day_ahead_forecast_figure():
    forecasts = [
        read_day_ahead_forecast(BASE_DIR / "outputs" / "rbfnn_forecast" / "Day_Ahead_24H_RBFNN_Forecast.xlsx", "RBFNN"),
        read_day_ahead_forecast(BASE_DIR / "outputs" / "random_forest_forecast" / "Day_Ahead_24H_RANDOM_FOREST.xlsx", "Random Forest"),
        read_day_ahead_forecast(BASE_DIR / "outputs" / "xgboost_forecast" / "Day_Ahead_24H_XGBOOST.xlsx", "XGBoost"),
    ]
    plt.figure(figsize=(11, 5))
    for df in forecasts:
        plt.plot(df["datetime"], df["Total_Cascade_Generation_MW"], marker="o", linewidth=1.5, markersize=3, label=df["model"].iloc[0])
    plt.title("Figure 4.32: Total Cascade Day-Ahead Forecast with Benchmarks (rolling 24h day-ahead)")
    plt.xlabel("Datetime")
    plt.ylabel("Total Cascade Generation (MW)")
    plt.grid(True, alpha=0.25)
    plt.legend()
    save_figure(FIGURE_DIRS["day_ahead_forecast"] / "total_cascade_day_ahead_forecast_with_benchmarks.png")


# ============================================================
# ORGANIZED METADATA EXPORT
# ============================================================

# Combines plant-level RBFNN daily metrics into one organized workbook.
def write_rbfnn_daily_metrics(split):
    frames = []
    source_dir = METADATA_DIR / f"{split}_metrics"
    expected_paths = [source_dir / f"{plant}_{split}_daily_metrics.xlsx" for plant in PLANTS]
    if source_dir.exists() and all(path.exists() for path in expected_paths):
        for plant in PLANTS:
            path = require_file(source_dir / f"{plant}_{split}_daily_metrics.xlsx")
            df = pd.read_excel(path)
            df.insert(1, "plant", plant)
            frames.append(df[["date", "plant", "operational_mape", "mae", "rmse", "samples"]])
    else:
        pred_path = require_file(DAY_AHEAD_BACKTEST_DIR / f"rbfnn_{split}_day_ahead_predictions.xlsx")
        predictions = pd.read_excel(pred_path)
        predictions["date"] = pd.to_datetime(predictions["datetime"]).dt.date
        for (plant, date), group in predictions.groupby(["plant", "date"], sort=False):
            actual = group["actual_generation"].astype(float)
            pred = group["predicted_generation"].astype(float)
            threshold = max(1.0, 0.01 * CAPACITY_MW[str(plant)])
            mask = actual.abs() >= threshold
            operational_mape = np.nan if not mask.any() else float(((actual[mask] - pred[mask]).abs() / actual[mask].abs()).mean() * 100.0)
            frames.append(pd.DataFrame([{
                "date": date,
                "plant": plant,
                "operational_mape": operational_mape,
                "mae": float((actual - pred).abs().mean()),
                "rmse": float(np.sqrt(np.mean(np.square(actual - pred)))),
                "samples": int(len(group)),
            }]))
    combined = pd.concat(frames, ignore_index=True)
    combined["plant"] = pd.Categorical(combined["plant"], categories=PLANTS, ordered=True)
    combined = combined.sort_values(["plant", "date"]).reset_index(drop=True)
    save_excel(combined, MODEL_FILES["RBFNN"]["folder"] / f"{split}_daily_metrics.xlsx")


# Writes organized per-model metadata tables and testing predictions.
def write_model_metadata():
    for model_name, spec in MODEL_FILES.items():
        metrics = read_metrics(model_name)
        validation, testing = normalized_metric_frames(metrics, model_name)
        summary = metrics.copy()
        summary["model"] = model_name
        predictions = read_predictions(model_name)

        if model_name == "RBFNN":
            write_rbfnn_daily_metrics("validation")
            write_rbfnn_daily_metrics("testing")

        save_excel(validation, spec["folder"] / "validation_average_metrics.xlsx")
        save_excel(testing, spec["folder"] / "testing_average_metrics.xlsx")
        save_excel(summary, spec["folder"] / "validation_testing_summary.xlsx")
        save_excel(predictions, spec["folder"] / "testing_predictions.xlsx")


def write_hydrologic_input_metadata():
    cleaned = read_cleaned_data()
    rows = [
        {
            "variable": "rainfall",
            "label": "Rainfall (daily-derived hourly-equivalent hydrologic context variable)",
            "included_in_cleaned_dataset": "rainfall" in cleaned.columns,
            "numeric": bool("rainfall" in cleaned.columns and pd.api.types.is_numeric_dtype(cleaned["rainfall"])),
            "daily_derived_hourly_equivalent": True,
            "true_hourly_sensor_measurement": False,
            "divided_by_24_again": False,
            "available_for_feature_engineering": "rainfall" in cleaned.columns,
            "feature_use": "lagged hydrologic context only",
        },
        {
            "variable": "lake_lanao_outflow",
            "label": "Lake Lanao outflow (hydrologic context variable)",
            "included_in_cleaned_dataset": "lake_lanao_outflow" in cleaned.columns,
            "numeric": bool("lake_lanao_outflow" in cleaned.columns and pd.api.types.is_numeric_dtype(cleaned["lake_lanao_outflow"])),
            "daily_derived_hourly_equivalent": True,
            "true_hourly_sensor_measurement": False,
            "divided_by_24_again": False,
            "available_for_feature_engineering": "lake_lanao_outflow" in cleaned.columns,
            "feature_use": "lagged hydrologic context only",
        },
    ]
    save_excel(pd.DataFrame(rows), METADATA_DIR / "overall_comparison" / "hydrologic_input_metadata.xlsx")


# Writes overall model comparison and best-model summary workbooks.
def write_overall_comparison():
    validation_frames = []
    testing_frames = []
    for model_name in DAY_AHEAD_MODEL_FILES:
        metrics = read_metrics(model_name)
        validation, testing = normalized_metric_frames(metrics, model_name)
        validation["split"] = "validation"
        testing["split"] = "testing"
        validation_frames.append(validation)
        testing_frames.append(testing)

    combined = pd.concat(validation_frames + testing_frames, ignore_index=True)
    combined["plant"] = pd.Categorical(combined["plant"], categories=PLANTS, ordered=True)
    combined = combined.sort_values(["plant", "split", "model"]).reset_index(drop=True)

    testing_comparison = pd.concat(testing_frames, ignore_index=True).drop(columns=["split"])
    testing_comparison["plant"] = pd.Categorical(testing_comparison["plant"], categories=PLANTS, ordered=True)
    testing_comparison = testing_comparison.sort_values(["plant", "MAPE"]).reset_index(drop=True)

    best = (
        testing_comparison.sort_values(["plant", "MAPE"])
        .groupby("plant", observed=False)
        .head(1)
        .reset_index(drop=True)
        .rename(columns={"model": "best_model", "MAPE": "testing_MAPE", "MAE": "testing_MAE", "RMSE": "testing_RMSE", "R2": "testing_R2"})
    )

    overall_dir = METADATA_DIR / "overall_comparison"
    save_excel(combined, overall_dir / "model_average_validation_testing_comparison.xlsx")
    save_excel(combined, overall_dir / "model_comparison_summary.xlsx")
    save_excel(testing_comparison, overall_dir / "plant_level_testing_comparison.xlsx")
    save_excel(best, overall_dir / "best_model_summary.xlsx")
    update_model_selection_audit_with_comparison(combined)


def update_model_selection_audit_with_comparison(combined):
    audit_path = OVERALL_METRICS_DIR / "model_selection_audit.json"
    audit = {}
    if audit_path.exists():
        audit = json.loads(audit_path.read_text())

    testing = combined[combined["split"] == "testing"].copy()
    comparisons = []
    for plant in PLANTS:
        plant_df = testing[testing["plant"].astype(str) == plant]
        rbfnn = plant_df[plant_df["model"] == "RBFNN"]
        if rbfnn.empty:
            continue
        rbfnn_row = rbfnn.iloc[0]
        for benchmark in ["Random Forest", "XGBoost"]:
            bench = plant_df[plant_df["model"] == benchmark]
            if bench.empty:
                continue
            bench_row = bench.iloc[0]
            comparisons.append({
                "plant": plant,
                "benchmark": benchmark,
                "rbfnn_test_mape": float(rbfnn_row["MAPE"]),
                "benchmark_test_mape": float(bench_row["MAPE"]),
                "rbfnn_test_mae": float(rbfnn_row["MAE"]),
                "benchmark_test_mae": float(bench_row["MAE"]),
                "rbfnn_test_rmse": float(rbfnn_row["RMSE"]),
                "benchmark_test_rmse": float(bench_row["RMSE"]),
                "rbfnn_test_r2": float(rbfnn_row["R2"]),
                "benchmark_test_r2": float(bench_row["R2"]),
                "rbfnn_won_mape": bool(rbfnn_row["MAPE"] < bench_row["MAPE"]),
                "rbfnn_won_mae": bool(rbfnn_row["MAE"] < bench_row["MAE"]),
                "rbfnn_won_rmse": bool(rbfnn_row["RMSE"] < bench_row["RMSE"]),
                "rbfnn_won_r2": bool(rbfnn_row["R2"] > bench_row["R2"]),
            })

    audit["rbfnn_vs_benchmarks"] = comparisons
    audit.setdefault("controls", {})
    audit["controls"].update({
        "benchmark_reporting": "Benchmarks are reported honestly even when they outperform RBFNN on a plant or metric.",
        "comparison_metric_source": "metadata/overall_comparison/model_comparison_summary.xlsx",
    })
    audit_path.write_text(json.dumps(audit, indent=2, default=str))
    print("Saved:", audit_path)


# Writes fair rolling 24-hour day-ahead comparison tables for all models.
def write_day_ahead_backtest_comparison():
    missing = [path for spec in DAY_AHEAD_MODEL_FILES.values() for path in spec.values() if not path.exists()]
    if missing:
        warning = "Day-ahead backtest files incomplete; missing: " + ", ".join(str(path) for path in missing)
        warnings.warn(warning)
        print("WARNING:", warning)
        return

    comparisons = {}
    for split_name in ["validation", "testing"]:
        frames = []
        for model_name, spec in DAY_AHEAD_MODEL_FILES.items():
            df = pd.read_excel(spec[split_name])
            df["model"] = model_name
            frames.append(df)
        comparison = pd.concat(frames, ignore_index=True)
        comparison["plant"] = pd.Categorical(comparison["plant"], categories=PLANTS, ordered=True)
        comparison = comparison.sort_values(["plant", "operational_mape", "model"]).reset_index(drop=True)
        comparisons[split_name] = comparison
        save_excel(comparison, DAY_AHEAD_BACKTEST_DIR / f"all_models_{split_name}_day_ahead_comparison.xlsx")

    summary = pd.concat(
        [df.assign(split=split_name) for split_name, df in comparisons.items()],
        ignore_index=True,
    )
    save_excel(summary, DAY_AHEAD_BACKTEST_DIR / "all_models_day_ahead_backtest_summary.xlsx")


def sync_model_metrics_workbooks():
    for model_name, model_key in MODEL_KEYS.items():
        model_dir = MODEL_METRICS_DIR / model_key
        copy_excel(
            MODEL_FILES[model_name]["metrics"],
            model_dir / f"{model_key}_validation_testing_metrics.xlsx",
        )
        copy_excel(
            DAY_AHEAD_MODEL_FILES[model_name]["validation"],
            model_dir / f"{model_key}_validation_day_ahead_metrics.xlsx",
        )
        copy_excel(
            DAY_AHEAD_MODEL_FILES[model_name]["testing"],
            model_dir / f"{model_key}_testing_day_ahead_metrics.xlsx",
        )


# Runs all thesis figure and organized metadata generation steps.
def main():
    # --- Figure and metadata generation pipeline ---
    ensure_dirs()
    generate_cleaned_profile_figures()
    generate_testing_actual_vs_forecast_figures()
    generate_rbfnn_testing_actual_vs_forecast_panel()
    generate_testing_actual_vs_forecast_by_model_figures()
    generate_all_models_testing_actual_vs_forecast_figures()
    generate_error_analysis_figures()
    generate_benchmark_comparison_figures()
    generate_day_ahead_forecast_figure()
    write_hydrologic_input_metadata()
    write_overall_comparison()
    write_day_ahead_backtest_comparison()
    sync_model_metrics_workbooks()
    print("DONE: Thesis figures and organized metadata files generated successfully.")


if __name__ == "__main__":
    main()

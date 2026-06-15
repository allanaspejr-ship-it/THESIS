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

FIGURE_DIRS = {
    "cleaned_profiles": THESIS_FIGURES_DIR / "cleaned_profiles",
    "testing_actual_vs_forecast": THESIS_FIGURES_DIR / "testing_actual_vs_forecast",
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
    for spec in MODEL_FILES.values():
        spec["folder"].mkdir(parents=True, exist_ok=True)
    (METADATA_DIR / "overall_comparison").mkdir(parents=True, exist_ok=True)


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
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name=sheet_name, index=False)
    format_excel(path)
    save_path(path)


# Saves the current Matplotlib figure using thesis figure settings.
def save_figure(path):
    path.parent.mkdir(parents=True, exist_ok=True)
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
        df = validation[["model", "plant", "val_operational_mape", "val_mae", "val_rmse", "val_r2"]].merge(
            testing[["plant", "test_operational_mape", "test_mae", "test_rmse", "test_r2"]],
            on="plant",
            how="inner",
        )
        df["feature_count"] = df["plant"].map(feature_counts).fillna(0).astype(int)
        df["evaluation_type"] = "rolling_24h_day_ahead_backtest"
        df["plant"] = pd.Categorical(df["plant"], categories=PLANTS, ordered=True)
        return df.sort_values("plant").reset_index(drop=True)

    df = pd.read_excel(require_file(MODEL_FILES[model_name]["metrics"]))
    df["plant"] = pd.Categorical(df["plant"], categories=PLANTS, ordered=True)
    return df.sort_values("plant").reset_index(drop=True)


# Reads testing predictions and sorts them by plant and time.
def read_predictions(model_name):
    day_ahead_testing = DAY_AHEAD_BACKTEST_DIR / f"{MODEL_KEYS[model_name]}_testing_day_ahead_predictions.xlsx"
    if day_ahead_testing.exists():
        df = pd.read_excel(day_ahead_testing)
        df = rebuild_datetime(df)
        df["plant"] = pd.Categorical(df["plant"], categories=PLANTS, ordered=True)
        return df.sort_values(["plant", "datetime"]).reset_index(drop=True)

    df = pd.read_excel(require_file(MODEL_FILES[model_name]["predictions"]))
    df = rebuild_datetime(df)
    df["plant"] = pd.Categorical(df["plant"], categories=PLANTS, ordered=True)
    return df.sort_values(["plant", "datetime"]).reset_index(drop=True)


# Reads saved feature counts for day-ahead metric summaries.
def read_feature_counts(model_name):
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
    validation = base.rename(
        columns={
            "val_operational_mape": "MAPE",
            "val_mae": "MAE",
            "val_rmse": "RMSE",
            "val_r2": "R2",
        }
    )[["model", "plant", "feature_count", "MAPE", "MAE", "RMSE", "R2"]]
    testing = base.rename(
        columns={
            "test_operational_mape": "MAPE",
            "test_mae": "MAE",
            "test_rmse": "RMSE",
            "test_r2": "R2",
        }
    )[["model", "plant", "feature_count", "MAPE", "MAE", "RMSE", "R2"]]
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
        plt.title(f"Figure 4.{offset}: Testing Actual vs Forecast - {PLANT_LABELS[plant]}")
        plt.xlabel("Datetime")
        plt.ylabel("Generation (MW)")
        plt.grid(True, alpha=0.25)
        plt.legend()
        save_figure(FIGURE_DIRS["testing_actual_vs_forecast"] / f"figure_4_{offset:02d}_{plant}_testing_actual_vs_forecast.png")


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
    plt.title("Figure 4.19: RBFNN Actual vs Predicted Scatter")
    plt.xlabel("Actual Generation (MW)")
    plt.ylabel("Predicted Generation (MW)")
    plt.grid(True, alpha=0.25)
    save_figure(FIGURE_DIRS["error_analysis"] / "rbfnn_actual_vs_predicted_scatter.png")

    plt.figure(figsize=(8, 5))
    plt.hist(residual, bins=60, color="#4c78a8", edgecolor="white")
    plt.axvline(0, color="black", linestyle="--", linewidth=1)
    plt.title("Figure 4.20: RBFNN Residual Distribution")
    plt.xlabel("Residual (Actual - Forecast, MW)")
    plt.ylabel("Frequency")
    plt.grid(True, axis="y", alpha=0.25)
    save_figure(FIGURE_DIRS["error_analysis"] / "rbfnn_residual_distribution.png")

    mape_df = plant_mape(df)
    plt.figure(figsize=(8, 5))
    plt.bar([PLANT_LABELS[p] for p in mape_df["plant"]], mape_df["MAPE"], color="#59a14f")
    plt.title("Figure 4.21: RBFNN Testing MAPE per Plant")
    plt.xlabel("Plant")
    plt.ylabel("MAPE (%)")
    plt.grid(True, axis="y", alpha=0.25)
    save_figure(FIGURE_DIRS["error_analysis"] / "rbfnn_testing_mape_per_plant.png")


# Generates RBFNN, Random Forest, and XGBoost comparison bar charts.
def generate_benchmark_comparison_figures():
    all_metrics = pd.concat([read_metrics(model) for model in MODEL_FILES], ignore_index=True)
    model_order = ["RBFNN", "Random Forest", "XGBoost"]
    pivot_mape = all_metrics.pivot(index="plant", columns="model", values="test_operational_mape").loc[PLANTS, model_order]
    pivot_rmse = all_metrics.pivot(index="plant", columns="model", values="test_rmse").loc[PLANTS, model_order]

    for fig_num, pivot, metric, ylabel, filename in [
        (22, pivot_mape, "Testing MAPE", "MAPE (%)", "testing_mape_model_comparison.png"),
        (23, pivot_rmse, "Testing RMSE", "RMSE (MW)", "testing_rmse_model_comparison.png"),
    ]:
        ax = pivot.rename(index=PLANT_LABELS).plot(kind="bar", figsize=(10, 5), width=0.78)
        ax.set_title(f"Figure 4.{fig_num}: {metric} Model Comparison")
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
    plt.title("Figure 4.32: Total Cascade Day-Ahead Forecast with Benchmarks")
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
    for plant in PLANTS:
        path = require_file(source_dir / f"{plant}_{split}_daily_metrics.xlsx")
        df = pd.read_excel(path)
        df.insert(1, "plant", plant)
        frames.append(df[["date", "plant", "operational_mape", "mae", "rmse", "samples"]])
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

        save_excel(validation, spec["folder"] / "validation_average_metrics.xlsx")
        save_excel(testing, spec["folder"] / "testing_average_metrics.xlsx")
        save_excel(summary, spec["folder"] / "validation_testing_summary.xlsx")
        save_excel(predictions, spec["folder"] / "testing_predictions.xlsx")


# Writes overall model comparison and best-model summary workbooks.
def write_overall_comparison():
    validation_frames = []
    testing_frames = []
    for model_name in MODEL_FILES:
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
    if not all(path.exists() for spec in DAY_AHEAD_MODEL_FILES.values() for path in spec.values()):
        print("Day-ahead backtest files incomplete; run Cell 2 and Cell 3 with day-ahead backtest enabled before Cell 4 comparison.")
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


# Runs all thesis figure and organized metadata generation steps.
def main():
    # --- Figure and metadata generation pipeline ---
    ensure_dirs()
    generate_cleaned_profile_figures()
    generate_testing_actual_vs_forecast_figures()
    generate_error_analysis_figures()
    generate_benchmark_comparison_figures()
    generate_day_ahead_forecast_figure()
    write_model_metadata()
    write_overall_comparison()
    write_day_ahead_backtest_comparison()
    print("DONE: Thesis figures and organized metadata files generated successfully.")


if __name__ == "__main__":
    main()

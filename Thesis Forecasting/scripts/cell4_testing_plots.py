"""
Cell 4: full testing-split actual vs forecast comparison plots.

Reads the testing prediction files produced by Cell 2 and Cell 3, combines
RBFNN, Random Forest, and XGBoost predictions, then saves one full-duration
testing plot per Agus plant.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[2]
THESIS_DIR = Path(__file__).resolve().parents[1]
METRICS_DIR = THESIS_DIR / "metadata" / "overall_metrics"
PLOT_DIR = THESIS_DIR / "outputs" / "testing_plots" / "full_testing_split"

PREDICTION_FILES = [
    METRICS_DIR / "rbfnn_testing_predictions.xlsx",
    METRICS_DIR / "random_forest_testing_predictions.xlsx",
    METRICS_DIR / "xgboost_testing_predictions.xlsx",
]
PLANTS = ["agus1", "agus2", "agus4", "agus5", "agus6", "agus7"]
MODEL_LABELS = ["RBFNN", "Random Forest", "XGBoost"]
REQUIRED_COLUMNS = [
    "Date",
    "Hour",
    "datetime",
    "plant",
    "actual_generation",
    "predicted_generation",
    "model",
]


def plant_title(plant):
    return plant.replace("agus", "AGUS ")


def load_predictions():
    missing = [path for path in PREDICTION_FILES if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing testing prediction file(s):\n"
            + "\n".join(str(path) for path in missing)
            + "\nRun Thesis Forecasting/scripts/cell2_rbfnn.py --train and Thesis Forecasting/scripts/cell3_benchmark.py --train first."
        )

    frames = []
    for path in PREDICTION_FILES:
        frame = pd.read_excel(path)
        missing_cols = [col for col in REQUIRED_COLUMNS if col not in frame.columns]
        if missing_cols:
            raise ValueError(f"{path} is missing required columns: {missing_cols}")
        frames.append(frame[REQUIRED_COLUMNS].copy())

    combined = pd.concat(frames, ignore_index=True)
    combined["datetime"] = pd.to_datetime(combined["datetime"])
    combined["plant"] = combined["plant"].astype(str).str.lower()
    combined["model"] = combined["model"].astype(str)
    combined["actual_generation"] = pd.to_numeric(combined["actual_generation"], errors="coerce")
    combined["predicted_generation"] = pd.to_numeric(combined["predicted_generation"], errors="coerce")
    combined = combined.dropna(subset=["datetime", "actual_generation", "predicted_generation"])
    return combined.sort_values(["plant", "datetime", "model"]).reset_index(drop=True)


def plot_plant(combined, plant):
    plant_df = combined[combined["plant"] == plant].copy()
    if plant_df.empty:
        print(f"Skipped {plant}: no testing predictions found.")
        return

    actual = plant_df[["datetime", "actual_generation"]].drop_duplicates("datetime").sort_values("datetime")
    predicted = plant_df.pivot_table(
        index="datetime",
        columns="model",
        values="predicted_generation",
        aggfunc="mean",
    ).sort_index()

    plt.figure(figsize=(16, 7))
    plt.plot(actual["datetime"], actual["actual_generation"], label="Actual Generation", linewidth=1.8, color="black")

    for model in MODEL_LABELS:
        if model in predicted.columns:
            plt.plot(predicted.index, predicted[model], label=f"{model} Forecast", linewidth=1.2)

    plt.xlabel("Testing Datetime")
    plt.ylabel("Generation in MW")
    plt.title(f"{plant_title(plant)} Testing Split Forecast Comparison")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    out_path = PLOT_DIR / f"{plant}_testing_split_forecast_comparison.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print("Saved:", out_path)


def main():
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    combined = load_predictions()
    for plant in PLANTS:
        plot_plant(combined, plant)
    print("Saved testing split plots folder:", PLOT_DIR)


if __name__ == "__main__":
    main()

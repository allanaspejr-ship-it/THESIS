# RBFNN Model Training Context

This file is the working context for training and reproducing the Radial Basis Function Neural Network (RBFNN) model in the HYDRO_FORECASTING project.

## Project Goal

The project forecasts short-term hydroelectric generation for the Agus cascaded hydroelectric plants. The primary model is an optimized RBFNN that generates plant-level and unit-level day-ahead generation forecasts using historical generation, outage status, gate/elevation signals, rainfall, Lake Lanao outflow, and cascade-aware upstream features.

The current production workflow is in:

```text
optimized_v2/
```

Use this folder for model training and evaluation unless there is a specific reason to work with the archived versions.

## Plants Modeled

The RBFNN trains one separate model for each plant:

```text
agus1
agus2
agus4
agus5
agus6
agus7
```

Cascade order:

```text
AGUS 1 -> AGUS 2 -> AGUS 4 -> AGUS 5 -> AGUS 6 -> AGUS 7
```

Plant capacities used by the model:

| Plant | Capacity MW |
|---|---:|
| agus1 | 80.0 |
| agus2 | 180.0 |
| agus4 | 158.1 |
| agus5 | 55.0 |
| agus6 | 219.0 |
| agus7 | 54.0 |

## Main Files

Run scripts from the repository root:

```text
optimized_v2/scripts/cell1_clean_data.py
optimized_v2/scripts/cell2_rbfnn.py
optimized_v2/scripts/cell3_benchmark.py
optimized_v2/scripts/cell4_testing_plots.py
```

RBFNN source code:

```text
optimized_v2/scripts/cell2_rbfnn.py
```

Raw data expected by Cell 1:

```text
optimized_v2/data/DATA(JAN2024-JUNE2025).xlsx
```

If that exact workbook is missing, Cell 1 uses the first non-temporary `.xlsx` file in `optimized_v2/data/`.

## Environment

Install dependencies:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Important packages:

```text
numpy
pandas
openpyxl
pyarrow
joblib
scikit-learn
tensorflow
xgboost
matplotlib
```

## Required Run Order

Clean raw data first:

```powershell
python optimized_v2/scripts/cell1_clean_data.py
```

Train or retrain the RBFNN:

```powershell
python optimized_v2/scripts/cell2_rbfnn.py --train
```

Evaluate saved RBFNN models and generate the 24-hour forecast without retraining:

```powershell
python optimized_v2/scripts/cell2_rbfnn.py
```

Train only selected plants:

```powershell
python optimized_v2/scripts/cell2_rbfnn.py --train agus5 agus7
```

Optional benchmark training:

```powershell
python optimized_v2/scripts/cell3_benchmark.py --train
```

Optional testing comparison plots:

```powershell
python optimized_v2/scripts/cell4_testing_plots.py
```

## Data Cleaning Summary

Cell 1 performs the raw-data preparation:

- Flattens the three-row Excel header.
- Standardizes plant, unit, generation, outage, gate, elevation, rainfall, and outflow column names.
- Builds hourly timestamps from `date` and `time`.
- Converts numeric fields.
- Sets near-zero generation values to zero using a threshold of `0.05`.
- Interpolates short missing gaps up to 2 hours.
- Interpolates short zero runs up to 2 hours.
- Preserves longer zero runs of at least 3 hours as true zero-generation periods.
- Smooths local spikes using a factor of `3.5`.
- Uses KNN imputation with 5 neighbors for remaining numeric missing values.
- Resamples to hourly frequency.
- Recalculates plant total generation from unit generation.
- Converts outage columns to binary operating indicators.

Cleaned outputs:

```text
optimized_v2/outputs/cleaned_data/cleaned_hourly_data.xlsx
optimized_v2/outputs/cleaned_data/cleaned_hourly_data.parquet
optimized_v2/outputs/cleaned_data/cell1_metadata.json
```

The cleaned public files omit `datetime`; downstream scripts rebuild it from `date + time`.

## Planned Outage Input

Cell 1 also creates the editable planned outage workbook:

```text
optimized_v2/outputs/outages_planning/Planned_Outages_Input.xlsx
```

This file contains the next 24 forecast hours and unit-level outage/availability columns. RBFNN forecasting reads this workbook and adjusts unit and plant forecasts according to expected unit availability.

## RBFNN Model Design

The RBFNN is implemented in TensorFlow/Keras with a custom `RBFLayer`.

The model does not directly predict next-hour generation level. It predicts the next-hour residual or delta:

```text
target_delta = total_gen_plant_tplus1 - total_gen_plant_current
```

The level forecast is then reconstructed as:

```text
forecast = current_generation + shrinkage * predicted_delta + bias_correction
```

Forecasts are clipped to:

```text
0 to 105 percent of plant capacity
```

The final model also supports plant-specific calibration:

- Bias correction from validation residuals.
- Validation-tuned shrinkage.
- Bin calibration for selected plants.
- Hourly residual correction for selected plants.
- Same-hour-yesterday profile blending for selected plants.

## RBF Layer

The custom RBF layer has:

- Trainable centers.
- Trainable gamma values stored as `log_gamma`.
- Gaussian radial basis activation:

```text
exp(-gamma * squared_distance(input, center))
```

Centers are initialized with KMeans on scaled training features.

## Training Split

The split is chronological and must not be shuffled:

| Split | Ratio |
|---|---:|
| Training | 70 percent |
| Validation | 15 percent |
| Testing | 15 percent |

The model selection uses validation performance. Testing is reserved for final evaluation.

## Feature Engineering

RBFNN features include:

- Current plant generation.
- Hour, day, month, weekend, and target-hour cyclic features.
- Lagged plant generation at 1, 2, 3, 6, 12, 24, 48, 72, and 168 hours.
- Rolling mean, standard deviation, minimum, and maximum over 3, 6, 12, 24, 48, and 168 hours.
- Generation differences over 1, 3, and 24 hours.
- Same-hour historical references using 24-hour and 168-hour offsets.
- Unit outage status.
- Number of running units.
- Plant availability indicator.
- Plant gate, elevation, and spillway features.
- Rainfall and Lake Lanao outflow lag features.
- Upstream cascade generation features.

The RBFNN feature function is:

```text
feature_columns_for(data, plant)
```

The feature construction function is:

```text
add_features(df)
```

## Training Hyperparameters

Default/search settings in `cell2_rbfnn.py`:

```text
epochs = 80
batch_size = 32
center_search = [80, 120, 180]
learning_rate_search = [0.001, 0.0005]
shrinkage_grid = [0.05, 0.10, 0.20, 0.35, 0.50, 0.75, 1.00, 1.15]
optimizer = Adam
loss = mean squared error
early_stopping = validation loss, patience 10, restore best weights
random_seed = 42
```

Inputs and target deltas are standardized with `StandardScaler`.

## Saved Model Artifacts

RBFNN model artifacts are saved in:

```text
optimized_v2/models/rbfnn/
```

Per plant, expected artifacts include:

```text
rbfnn_<plant>.keras or rbfnn_<plant>.h5
x_scaler_<plant>.pkl
y_scaler_<plant>.pkl
meta_<plant>.json
```

The metadata JSON stores:

- Plant name.
- Target column.
- Delta target column.
- Feature columns.
- Capacity.
- Operational MAPE threshold.
- Selected number of centers.
- Selected learning rate.
- Best shrinkage.
- Bias correction.
- Calibration settings.
- Ramp limit.
- Model type description.

On Windows, `.keras` saving can fail if the internal temporary zip writer is blocked. The script falls back to `.h5` and prefers existing `.h5` files when loading.

## Evaluation Metrics

Metrics are computed for validation and testing:

```text
operational_mape
mae
rmse
r2
```

Operational MAPE excludes low actual generation values where percentage error is misleading. The threshold is:

```text
max(1 MW, 1 percent of plant capacity)
```

MAE, RMSE, and R2 use all rows.

Main RBFNN metrics output:

```text
optimized_v2/metadata/overall_metrics/rbfnn_validation_testing_metrics.xlsx
```

Testing predictions output:

```text
optimized_v2/metadata/overall_metrics/rbfnn_testing_predictions.xlsx
```

Daily validation/testing metric files are saved in:

```text
optimized_v2/metadata/validation_metrics/
optimized_v2/metadata/testing_metrics/
```

Daily metric files intentionally exclude R2 because daily slices are small and can produce misleading values.

## Current Reported RBFNN Performance

From the existing optimized v2 context, average RBFNN performance across all six plants:

| Metric | Validation | Testing |
|---|---:|---:|
| Operational MAPE | 2.615 percent | 2.519 percent |
| MAE | 1.633 MW | 1.355 MW |
| RMSE | 3.789 MW | 3.575 MW |
| R2 | 0.878 | 0.925 |

Plant-level testing operational MAPE:

| Plant | Test MAPE |
|---|---:|
| agus1 | 3.476 percent |
| agus2 | 1.372 percent |
| agus4 | 0.611 percent |
| agus5 | 4.040 percent |
| agus6 | 1.405 percent |
| agus7 | 4.209 percent |

The optimized RBFNN outperformed the Random Forest and XGBoost benchmark models on testing operational MAPE for all six plants in the documented run.

## Forecast Output

The final RBFNN day-ahead forecast is saved to:

```text
optimized_v2/outputs/rbfnn_forecast/Day_Ahead_24H_RBFNN_Forecast.xlsx
optimized_v2/outputs/rbfnn_forecast/Day_Ahead_24H_RBFNN_Forecast.csv
```

The output includes:

- Date.
- Hour.
- Unit-level generation forecast columns.
- Plant total generation columns.
- Total cascade generation.

## Reproducibility Rules

When retraining:

- Use the `optimized_v2` scripts.
- Run from the repository root.
- Do not shuffle the time series.
- Keep the 70/15/15 chronological split.
- Use the cleaned parquet generated by Cell 1.
- Preserve the same plant list and cascade order unless intentionally changing the research scope.
- Compare validation and testing metrics before accepting a retrained model.
- Check `meta_<plant>.json` after training to confirm selected centers, learning rate, shrinkage, and calibration.
- Keep generated models, outputs, and caches out of Git unless explicitly needed.

## Common Troubleshooting

If Cell 2 fails because cleaned data is missing, run:

```powershell
python optimized_v2/scripts/cell1_clean_data.py
```

If saved RBFNN models are missing and forecast-only mode fails, run:

```powershell
python optimized_v2/scripts/cell2_rbfnn.py --train
```

If only one or two plants need retraining:

```powershell
python optimized_v2/scripts/cell2_rbfnn.py --train agus5 agus7
```

If TensorFlow emits oneDNN or low-level logs, the script already sets:

```text
TF_CPP_MIN_LOG_LEVEL=3
TF_ENABLE_ONEDNN_OPTS=0
```

If an Excel workbook is open and locked, close it before rerunning scripts that write `.xlsx` outputs.


# Code Context: Cleaning, Training, Forecasting, RBFNN, and Benchmarks

This file summarizes what the current `Thesis Forecasting` code does from raw data cleaning through model training, day-ahead forecasting, and benchmark comparison. It is written as thesis/code context for explaining the workflow without reading every script line by line.

## Project Scope

The workflow forecasts short-term hydroelectric generation for the Agus cascade:

- Agus 1
- Agus 2
- Agus 4
- Agus 5
- Agus 6
- Agus 7

The active code is in:

```text
Thesis Forecasting/scripts/
```

The main scripts are:

```text
cell1_clean_data.py
cell2_rbfnn.py
cell3_benchmark.py
cell4_generate_thesis_and_metrics.py
```

Run order:

```powershell
python "Thesis Forecasting/scripts/cell1_clean_data.py"
python "Thesis Forecasting/scripts/cell2_rbfnn.py"
python "Thesis Forecasting/scripts/cell3_benchmark.py"
python "Thesis Forecasting/scripts/cell4_generate_thesis_and_metrics.py"
```

Use `--train` when intentionally retraining models:

```powershell
python "Thesis Forecasting/scripts/cell2_rbfnn.py" --train
python "Thesis Forecasting/scripts/cell3_benchmark.py" --train
```

Without `--train`, Cell 2 and Cell 3 load saved models, evaluate them, and generate forecasts.

## Cell 1: Cleaning Code Context

Source script:

```text
Thesis Forecasting/scripts/cell1_clean_data.py
```

Input:

```text
Thesis Forecasting/data/DATA(JAN2024-JUNE2025).xlsx
```

If that exact workbook is missing, the script uses the first non-temporary `.xlsx` file found inside `Thesis Forecasting/data/`.

### What the Cleaning Script Does

The raw Excel workbook has a multi-row header, plant/unit columns, generation columns, outage columns, gate/elevation/rainfall/outflow columns, and date/time fields. The cleaning script converts that into a consistent hourly modeling dataset.

Main cleaning steps:

- Flattens the three-row Excel header into single column names.
- Standardizes column names for generation, outage, spillway, total gate opening, elevation, rainfall, and Lake Lanao outflow.
- Converts `date` and `time` into hourly timestamps.
- Converts numeric fields from Excel text/mixed values into numeric columns.
- Cleans unit generation values using a near-zero threshold of `0.05`.
- Smooths isolated spikes using a local-neighbor spike rule with factor `3.5`.
- Interpolates short missing gaps up to 2 hours.
- Interpolates short zero-runs up to 2 hours.
- Preserves longer zero-runs of at least 3 hours as real zero-generation or outage-like conditions.
- Cleans negative gate, spillway, and rainfall values where applicable.
- Uses KNN imputation with 5 neighbors for remaining missing numeric values.
- Recalculates plant total generation from unit generation columns.
- Converts outage columns into binary unit status indicators.
- Resamples the cleaned data to hourly frequency.
- Saves public cleaned files without a `datetime` column; downstream scripts rebuild `datetime` from `date + time`.

### Cleaning Outputs

Cell 1 writes:

```text
Thesis Forecasting/outputs/cleaned_data/cleaned_hourly_data.xlsx
Thesis Forecasting/outputs/cleaned_data/cleaned_hourly_data.parquet
Thesis Forecasting/outputs/cleaned_data/cell1_metadata.json
Thesis Forecasting/outputs/outages_planning/Planned_Outages_Input.xlsx
```

Current cleaned dataset metadata:

- Latest timestamp: `2025-06-30 23:00:00`
- Rows: `13,128`
- Columns: `55`
- Cleaned files omit `datetime`; model scripts rebuild it.

### Planned Outage Template

Cell 1 also creates:

```text
Thesis Forecasting/outputs/outages_planning/Planned_Outages_Input.xlsx
```

This file contains the next 24 forecast hours after the latest historical timestamp. It includes `Date`, `Hour`, and one outage/status column per unit, such as `out_agus1_unit1`. Forecasting scripts use it to adjust plant and unit forecasts based on planned unit availability.

## Cell 2: RBFNN Training and Forecasting Context

Source script:

```text
Thesis Forecasting/scripts/cell2_rbfnn.py
```

Cell 2 is the main forecasting model. It trains or loads one RBFNN model per plant.

Inputs:

```text
Thesis Forecasting/outputs/cleaned_data/cleaned_hourly_data.parquet
Thesis Forecasting/outputs/outages_planning/Planned_Outages_Input.xlsx
```

Main outputs:

```text
Thesis Forecasting/models/rbfnn/
Thesis Forecasting/metadata/validation_metrics/
Thesis Forecasting/metadata/testing_metrics/
Thesis Forecasting/metadata/overall_metrics/rbfnn_validation_testing_metrics.xlsx
Thesis Forecasting/metadata/overall_metrics/rbfnn_testing_predictions.xlsx
Thesis Forecasting/outputs/rbfnn_forecast/Day_Ahead_24H_RBFNN_Forecast.xlsx
Thesis Forecasting/outputs/rbfnn_forecast/Day_Ahead_24H_RBFNN_Forecast.csv
```

### RBFNN Model Design

The RBFNN is implemented in TensorFlow/Keras.

Architecture:

- Input feature layer
- Custom radial basis function layer
- Trainable RBF centers
- Trainable gamma values
- Linear dense output layer

The model is not a direct level forecaster. It predicts the next-hour generation change:

```text
target = total_gen_plant(t+1) - total_gen_plant(t)
```

The predicted delta is then added to the current generation. This makes the forecast persistence-anchored, which helps avoid unstable jumps in short-term hydro generation forecasts.

### RBFNN Feature Engineering

RBFNN features include:

- Hour, day, month, and weekend time indicators.
- Target-hour cyclical indicators.
- Current plant generation.
- Lags at 1, 2, 3, 6, 12, 24, 48, 72, and 168 hours.
- Rolling mean, standard deviation, minimum, and maximum over 3, 6, 12, 24, 48, and 168 hours.
- Generation differences over 1, 3, and 24 hours.
- Same-hour historical target references.
- Unit-level generation and unit share features.
- Unit outage/status features.
- Number of running units and plant availability indicator.
- Plant-specific gate, elevation, and spillway features.
- Rainfall and Lake Lanao outflow lag features.
- Upstream cascade generation features.

Cascade order used by the RBFNN:

```text
agus1 -> agus2 -> agus4 -> agus5 -> agus6 -> agus7
```

### Training Process

The script uses a chronological split:

- Training: first 70 percent
- Validation: next 15 percent
- Testing: final 15 percent

Training setup:

- Input scaler: `StandardScaler`
- Target delta scaler: `StandardScaler`
- RBF centers initialized with `KMeans`
- Optimizer: Adam
- Loss: mean squared error
- Epochs: 80
- Batch size: 32
- Early stopping on validation loss

Hyperparameter search:

- RBF centers: `80`, `120`, `180`
- Learning rates: `0.001`, `0.0005`
- Shrinkage values: `0.05`, `0.10`, `0.20`, `0.35`, `0.50`, `0.75`, `1.00`, `1.15`

After the model predicts deltas, validation-tuned corrections are applied:

- Shrinkage to control overreaction.
- Bias correction in MW.
- Bin calibration for selected plants.
- Hourly residual correction for selected plants.
- Same-hour-yesterday profile blending for selected plants.
- Forecast ramp limiting during recursive day-ahead forecasting.

### RBFNN Forecasting

The 24-hour forecast is recursive:

1. Start from the latest cleaned historical row.
2. Predict the next hour for each plant.
3. Apply planned outage status.
4. Clip forecasts to plant and unit capacities.
5. Allocate plant-level generation to units using learned historical unit shares and available capacity.
6. Append the forecasted hour back into history.
7. Repeat until 24 hours are produced.

The final RBFNN forecast contains:

- Date
- Hour
- Unit-level generation forecasts
- Plant total generation forecasts
- Total cascade generation

## Cell 3: Benchmark Code Context

Source script:

```text
Thesis Forecasting/scripts/cell3_benchmark.py
```

Benchmark models:

- Random Forest
- XGBoost

Inputs:

```text
Thesis Forecasting/outputs/cleaned_data/cleaned_hourly_data.parquet
Thesis Forecasting/outputs/outages_planning/Planned_Outages_Input.xlsx
```

Main outputs:

```text
Thesis Forecasting/models/random_forest/
Thesis Forecasting/models/xgboost/
Thesis Forecasting/metadata/overall_metrics/random_forest_validation_testing_metrics.xlsx
Thesis Forecasting/metadata/overall_metrics/xgboost_validation_testing_metrics.xlsx
Thesis Forecasting/metadata/overall_metrics/random_forest_testing_predictions.xlsx
Thesis Forecasting/metadata/overall_metrics/xgboost_testing_predictions.xlsx
Thesis Forecasting/benchmark/random_forest/Day_Ahead_24H_RANDOM_FOREST.xlsx
Thesis Forecasting/benchmark/xgboost/Day_Ahead_24H_XGBOOST.xlsx
```

### Benchmark Model Settings

Random Forest:

- `RandomForestRegressor`
- 400 trees
- Minimum samples per leaf: 2
- Random state: 42
- `n_jobs=1`

XGBoost:

- `XGBRegressor`
- 500 estimators
- Learning rate: 0.03
- Maximum depth: 4
- Subsample: 0.9
- Column sample by tree: 0.9
- Objective: squared error
- Random state: 42
- `n_jobs=1`

### Benchmark Training Design

The benchmarks use the same chronological split and the same metric definitions as the RBFNN:

- Training: first 70 percent
- Validation: next 15 percent
- Testing: final 15 percent

However, benchmark targets are direct next-hour generation levels:

```text
target = total_gen_plant(t+1)
```

This is different from the RBFNN, which predicts the next-hour delta.

Benchmark features include:

- Hour sine/cosine
- Day of week
- Month
- Current plant generation
- Plant generation lags
- Rolling mean and standard deviation
- Unit current generation and lag values
- Unit generation share features
- Unit outage/status columns
- Units-running feature

The benchmark feature set is smaller than the RBFNN feature set. It does not include the same full target-hour cyclical features, cascade upstream features, rolling min/max features, generation difference features, rainfall/outflow lag features, shrinkage tuning, bin calibration, hourly correction, or profile blending.

### Benchmark Forecasting

The benchmark 24-hour forecasts are also recursive:

1. Load or train plant-level Random Forest and XGBoost models.
2. Predict next-hour plant total generation.
3. Limit sudden jumps using recent historical ramp behavior.
4. Adjust forecast by planned unit availability.
5. Allocate total generation to available units.
6. Append the forecasted row into history.
7. Repeat for 24 hours.

## Main Difference: RBFNN vs Benchmark Code

The RBFNN and benchmark scripts both solve the same forecasting task, but they differ in model purpose, target design, features, and calibration.

| Area | RBFNN Code | Benchmark Code |
|---|---|---|
| Script | `cell2_rbfnn.py` | `cell3_benchmark.py` |
| Models | Custom TensorFlow/Keras RBFNN | Random Forest and XGBoost |
| Target | Next-hour generation delta | Direct next-hour generation level |
| Forecast anchor | Current generation plus predicted delta | Direct prediction clipped by ramp limit |
| Feature set | Larger, cascade-aware, outage-aware, hydrology-aware | Smaller time-series and outage feature set |
| Upstream cascade features | Yes | No |
| Rainfall/outflow lag features | Yes | No |
| Rolling min/max and generation differences | Yes | Mostly no; uses rolling mean/std |
| Scaling | StandardScaler for X and y delta | No neural-network scaling needed |
| RBF center initialization | KMeans | Not applicable |
| Hyperparameter search | Centers, learning rate, shrinkage | Fixed RF/XGB settings |
| Calibration | Shrinkage, bias, bin correction, hourly correction, profile blending | No equivalent post-calibration |
| Saved models | `.keras` or `.h5`, plus scalers and metadata JSON | Joblib `.pkl` payloads |
| Main role | Final proposed model | Comparison/baseline models |

In short, the RBFNN code is the optimized thesis model. It was built to handle short-term hydro generation behavior through residual forecasting, cascade features, outage awareness, and validation-based correction. The benchmark code is intentionally simpler and is used to compare whether standard machine learning models can match or exceed the RBFNN under the same chronological validation/testing setup.

## Evaluation Metrics

All models are evaluated using:

- Operational MAPE
- MAE
- RMSE
- R2

Operational MAPE excludes very low actual generation values because percentage error becomes misleading near zero. The threshold is:

```text
max(1 MW, 1 percent of plant capacity)
```

MAE, RMSE, and R2 are computed using all rows.

## Current Model Results

Current overall metric files are stored in:

```text
Thesis Forecasting/metadata/overall_metrics/
```

Average performance across all six plants:

| Model | Validation MAPE (%) | Testing MAPE (%) | Validation MAE | Testing MAE | Validation RMSE | Testing RMSE | Validation R2 | Testing R2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| RBFNN | 2.615 | 2.519 | 1.633 | 1.355 | 3.789 | 3.575 | 0.878 | 0.925 |
| Random Forest | 3.234 | 6.185 | 2.231 | 3.225 | 4.033 | 5.014 | 0.867 | 0.905 |
| XGBoost | 3.147 | 8.078 | 2.128 | 3.606 | 3.937 | 5.986 | 0.868 | 0.901 |

Plant-level testing operational MAPE:

| Plant | RBFNN | Random Forest | XGBoost |
|---|---:|---:|---:|
| agus1 | 3.476 | 3.448 | 3.509 |
| agus2 | 1.372 | 2.335 | 3.172 |
| agus4 | 0.611 | 2.239 | 1.123 |
| agus5 | 4.040 | 4.300 | 4.465 |
| agus6 | 1.405 | 20.548 | 31.862 |
| agus7 | 4.209 | 4.239 | 4.335 |

The RBFNN has the best average testing performance. At plant level, Random Forest is slightly lower than RBFNN for `agus1` testing MAPE, but RBFNN is better on the other five plants and is much stronger on the average testing metrics.

## Cell 4: Figures and Organized Metadata

Source script:

```text
Thesis Forecasting/scripts/cell4_generate_thesis_and_metrics.py
```

Cell 4 reads cleaned data, model metric files, model testing prediction files, and day-ahead forecast files. It generates thesis figures and organized metadata.

Outputs:

```text
Thesis Forecasting/thesis_figures/
Thesis Forecasting/metadata/rbfnn/
Thesis Forecasting/metadata/random_forest/
Thesis Forecasting/metadata/xgboost/
Thesis Forecasting/metadata/overall_comparison/
```

It creates figures for:

- Cleaned generation profiles
- RBFNN testing actual vs forecast
- RBFNN residual/scatter/error analysis
- Model comparison charts
- Day-ahead total cascade forecast comparison

## What Was Done Overall

The codebase was organized into a complete thesis forecasting pipeline:

1. Raw data is cleaned and converted into hourly Agus cascade records.
2. A planned outage input workbook is generated for the next 24-hour forecast horizon.
3. RBFNN models are trained or loaded per plant.
4. RBFNN validation, testing, daily metrics, and prediction files are generated.
5. A 24-hour outage-aware RBFNN forecast is generated at unit, plant, and cascade level.
6. Random Forest and XGBoost benchmarks are trained or loaded using the same split.
7. Benchmark metrics, testing predictions, and day-ahead forecasts are generated.
8. Thesis figures and organized metadata are created for Chapter 4 discussion.

The key technical contribution is the RBFNN residual/delta forecasting approach with cascade-aware and outage-aware features, followed by validation-tuned correction and unit-capacity-aware forecast allocation.

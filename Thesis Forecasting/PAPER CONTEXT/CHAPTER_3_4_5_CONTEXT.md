# Chapter 3, 4, and 5 Context for `optimized_v2`

This context file summarizes only the work contained in the `optimized_v2` folder. It is intended as a writing guide for the methodology, results, and conclusion chapters of the paper.

## Scope of `optimized_v2`

The `optimized_v2` folder is an isolated revision of the hydroelectric generation forecasting workflow. It contains the raw input workbook, cleaning scripts, trained models, validation and testing metrics, day-ahead forecast outputs, benchmark outputs, and comparison plots.

The workflow covers the Agus cascade plants:

- AGUS 1
- AGUS 2
- AGUS 4
- AGUS 5
- AGUS 6
- AGUS 7

The source data used in this version is:

`optimized_v2/data/DATA(JAN2024-JUNE2025).xlsx`

The cleaned dataset ends at:

`2025-06-30 23:00:00`

The generated day-ahead forecasts are for:

`2025-07-01`, Hours 1 to 24

## Chapter 3 Context: Methodology

### Research Workflow

The optimized workflow was implemented through four scripts:

1. `scripts/cell1_clean_data.py`
2. `scripts/cell2_rbfnn.py`
3. `scripts/cell3_benchmark.py`
4. `scripts/cell4_testing_plots.py`

The process followed these stages:

1. Raw data preparation and cleaning
2. Hourly dataset generation
3. Planned outage template generation
4. Feature engineering
5. RBFNN model training and evaluation
6. Random Forest and XGBoost benchmark training and evaluation
7. Day-ahead 24-hour forecasting
8. Testing-split visualization

### Data Cleaning and Preprocessing

The raw Excel file was cleaned using `cell1_clean_data.py`. The script flattened the three-row Excel header, standardized plant and unit column names, rebuilt datetime values from date and time fields, converted numeric columns, and created a consistent hourly dataset.

Cleaning steps included:

- Standardization of AGUS plant, generation, outage, gate, elevation, rainfall, and outflow column names
- Conversion of raw date and time into hourly datetime records
- Treatment of near-zero generation values using a 0.05 threshold
- Short missing-gap interpolation with a maximum gap of 2 hours
- Short zero-run interpolation for zero sequences up to 2 hours
- Preservation of longer zero runs of at least 3 hours as actual zero-generation conditions
- Spike smoothing using a local-neighbor rule with a factor of 3.5
- KNN imputation for remaining missing numeric values using 5 neighbors
- Resampling to hourly frequency
- Recalculation of plant-level total generation from unit-level generation
- Conversion of outage status columns into binary operating indicators

The cleaned output contains 13,128 hourly rows and 55 columns. It is stored as:

- `outputs/cleaned_data/cleaned_hourly_data.xlsx`
- `outputs/cleaned_data/cleaned_hourly_data.parquet`
- `outputs/cleaned_data/cell1_metadata.json`

The public cleaned files omit the `datetime` column. Downstream scripts rebuild it using `date + time`.

### Planned Outage Input

The cleaning script also created a 24-hour planned outage workbook for the next forecast horizon after the latest historical timestamp. The file is:

`outputs/outages_planning/Planned_Outages_Input.xlsx`

This file contains the forecast date, hour, and unit-level outage or availability status columns. It is used by the forecasting scripts to adjust forecasts according to expected unit availability.

### Feature Engineering

The optimized RBFNN and benchmark models used time-series and operational features. The RBFNN feature set was more extensive than the benchmark feature set.

RBFNN features included:

- Hour, day, month, and weekend cyclical indicators
- Target-hour cyclical indicators
- Current plant generation
- Lagged generation values at 1, 2, 3, 6, 12, 24, 48, 72, and 168 hours
- Rolling mean, standard deviation, minimum, and maximum values using 3, 6, 12, 24, 48, and 168-hour windows
- Generation differences at 1, 3, and 24 hours
- Same-hour historical target references
- Unit outage status columns
- Number of running units and plant availability indicators
- Plant-specific gate, elevation, and spillway variables
- Rainfall and Lake Lanao outflow lag variables
- Upstream cascade generation features based on plant order

The cascade order used in the optimized RBFNN was:

AGUS 1 -> AGUS 2 -> AGUS 4 -> AGUS 5 -> AGUS 6 -> AGUS 7

### Dataset Split

The dataset was divided chronologically:

- Training: first 70 percent
- Validation: next 15 percent
- Testing: final 15 percent

The validation daily metric files cover January 19, 2025 to April 10, 2025. The testing daily metric files cover April 10, 2025 to June 30, 2025. The testing prediction workbooks cover April 11, 2025 00:00 to June 30, 2025 23:00.

### RBFNN Model

The main model was a Radial Basis Function Neural Network implemented in TensorFlow/Keras. The optimized RBFNN was designed as a residual or delta model. Instead of predicting generation directly, it predicted the next-hour generation change relative to the current generation. The final forecast was anchored to persistence, then adjusted using shrinkage, bias correction, and selected post-calibration steps.

Core RBFNN design:

- RBF layer with trainable centers and gamma values
- KMeans initialization of RBF centers
- Dense linear output layer
- Adam optimizer
- Mean squared error training loss
- Early stopping on validation loss
- Standard scaling for input features and target deltas

Hyperparameter search used:

- RBF centers: 80, 120, and 180
- Learning rates: 0.001 and 0.0005
- Epochs: 80
- Batch size: 32
- Shrinkage values: 0.05 to 1.15

The selected RBFNN settings were:

| Plant | Feature Count | Selected Centers | Learning Rate | Shrinkage | Extra Calibration |
|---|---:|---:|---:|---:|---|
| agus1 | 158 | 80 | 0.001 | 1.15 | Bin calibration |
| agus2 | 169 | 120 | 0.0005 | 1.15 | None |
| agus4 | 169 | 180 | 0.0005 | 1.15 | None |
| agus5 | 168 | 180 | 0.001 | 1.15 | Bin calibration and hourly residual correction |
| agus6 | 171 | 80 | 0.0005 | 1.15 | None |
| agus7 | 168 | 120 | 0.001 | 1.15 | Hourly residual correction |

### Benchmark Models

Two benchmark models were trained for comparison:

- Random Forest Regressor
- XGBoost Regressor

Both benchmark models used the same chronological split and the same operational MAPE definition as the RBFNN. Their feature sets included time indicators, current generation, lagged generation, rolling statistics, outage status, and units-running features.

Random Forest settings:

- 400 trees
- Minimum samples per leaf: 2
- Random state: 42

XGBoost settings:

- 500 estimators
- Learning rate: 0.03
- Maximum depth: 4
- Subsample: 0.9
- Column sample by tree: 0.9
- Objective: squared error
- Random state: 42

### Evaluation Metrics

The models were evaluated using:

- Operational MAPE
- MAE
- RMSE
- R2

Operational MAPE excludes very low actual generation values where percentage error is not meaningful. The operational threshold is the greater of 1 MW or 1 percent of the plant capacity. MAE, RMSE, and R2 are computed using all rows.

## Chapter 4 Context: Results and Discussion

### Main RBFNN Validation and Testing Results

The RBFNN results are stored in:

`metadata/overall_metrics/rbfnn_validation_testing_metrics.xlsx`

| Plant | Validation MAPE (%) | Validation MAE | Validation RMSE | Validation R2 | Testing MAPE (%) | Testing MAE | Testing RMSE | Testing R2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| AGUS 1 | 3.471 | 1.727 | 2.756 | 0.929 | 3.476 | 1.841 | 3.440 | 0.899 |
| AGUS 2 | 1.080 | 0.984 | 4.297 | 0.822 | 1.372 | 1.164 | 4.810 | 0.912 |
| AGUS 4 | 0.827 | 0.717 | 3.362 | 0.953 | 0.611 | 0.697 | 3.077 | 0.945 |
| AGUS 5 | 4.091 | 1.469 | 2.231 | 0.904 | 4.040 | 1.544 | 2.189 | 0.901 |
| AGUS 6 | 2.413 | 3.667 | 7.981 | 0.835 | 1.405 | 1.396 | 5.679 | 0.990 |
| AGUS 7 | 3.806 | 1.234 | 2.109 | 0.824 | 4.209 | 1.491 | 2.254 | 0.905 |

Average RBFNN performance across all plants:

- Validation operational MAPE: 2.615 percent
- Testing operational MAPE: 2.519 percent
- Validation MAE: 1.633 MW
- Testing MAE: 1.355 MW
- Validation RMSE: 3.789 MW
- Testing RMSE: 3.575 MW
- Validation R2: 0.878
- Testing R2: 0.925

### Benchmark Results

Random Forest results are stored in:

`metadata/overall_metrics/random_forest_validation_testing_metrics.xlsx`

XGBoost results are stored in:

`metadata/overall_metrics/xgboost_validation_testing_metrics.xlsx`

Average benchmark performance:

| Model | Validation MAPE (%) | Testing MAPE (%) | Validation MAE | Testing MAE | Validation RMSE | Testing RMSE | Validation R2 | Testing R2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| RBFNN | 2.615 | 2.519 | 1.633 | 1.355 | 3.789 | 3.575 | 0.878 | 0.925 |
| Random Forest | 3.139 | 5.864 | 2.118 | 2.722 | 3.964 | 4.755 | 0.870 | 0.916 |
| XGBoost | 2.974 | 8.876 | 1.927 | 3.764 | 3.796 | 6.498 | 0.880 | 0.898 |

The RBFNN had the lowest average testing operational MAPE, lowest average testing MAE, lowest average testing RMSE, and highest average testing R2 among the three evaluated models.

### Plant-Level Comparison

On the testing set, the RBFNN achieved the lowest operational MAPE for all six Agus plants:

| Plant | RBFNN Test MAPE (%) | Random Forest Test MAPE (%) | XGBoost Test MAPE (%) | Best Model |
|---|---:|---:|---:|---|
| AGUS 1 | 3.476 | 3.486 | 3.573 | RBFNN |
| AGUS 2 | 1.372 | 2.161 | 2.554 | RBFNN |
| AGUS 4 | 0.611 | 1.164 | 1.051 | RBFNN |
| AGUS 5 | 4.040 | 4.200 | 4.228 | RBFNN |
| AGUS 6 | 1.405 | 19.879 | 37.541 | RBFNN |
| AGUS 7 | 4.209 | 4.294 | 4.311 | RBFNN |

This result supports the use of the optimized RBFNN for the final day-ahead forecasting workflow. The largest benchmark weakness appeared in AGUS 6 testing performance, where both benchmark models produced substantially higher percentage errors than the RBFNN.

### Forecast Outputs

The final optimized RBFNN 24-hour forecast is stored in:

- `outputs/rbfnn_forecast/Day_Ahead_24H_RBFNN_Forecast.xlsx`
- `outputs/rbfnn_forecast/Day_Ahead_24H_RBFNN_Forecast.csv`

The RBFNN forecast includes:

- Date
- Hour
- Unit-level generation forecasts
- Plant-level total generation forecasts
- Total cascade generation

Benchmark day-ahead forecasts are stored in:

- `benchmark/random_forest/Day_Ahead_24H_RANDOM_FOREST.xlsx`
- `benchmark/random_forest/Day_Ahead_24H_RANDOM_FOREST.csv`
- `benchmark/xgboost/Day_Ahead_24H_XGBOOST.xlsx`
- `benchmark/xgboost/Day_Ahead_24H_XGBOOST.csv`

The RBFNN forecast for July 1, 2025 produced total cascade generation values around 569 MW to 581 MW during the first 12 forecast hours shown in the workbook preview.

### Testing-Split Plots

The testing prediction files were combined by `cell4_testing_plots.py` to generate actual-versus-forecast plots for the full testing split. The plots compare actual generation against RBFNN, Random Forest, and XGBoost forecasts.

The plot folder is:

`outputs/testing_plots/full_testing_split/`

Generated plots:

- `agus1_testing_split_forecast_comparison.png`
- `agus2_testing_split_forecast_comparison.png`
- `agus4_testing_split_forecast_comparison.png`
- `agus5_testing_split_forecast_comparison.png`
- `agus6_testing_split_forecast_comparison.png`
- `agus7_testing_split_forecast_comparison.png`

These figures can be used in Chapter 4 to visually support the metric results.

## Chapter 5 Context: Summary, Conclusions, and Recommendations

### Summary of Work Done

The `optimized_v2` workflow successfully produced a complete forecasting pipeline for the Agus hydroelectric cascade. The work included data cleaning, hourly resampling, planned outage input preparation, model training, model evaluation, benchmark comparison, 24-hour forecasting, and testing-split visualization.

The main forecasting model was an optimized RBFNN that predicted next-hour generation changes and converted them into generation forecasts using persistence anchoring and validation-tuned corrections. Random Forest and XGBoost models were trained as benchmarks under the same chronological validation and testing framework.

### Main Findings

The optimized RBFNN achieved the strongest overall testing performance. It had the lowest average testing operational MAPE at 2.519 percent, compared with 5.864 percent for Random Forest and 8.876 percent for XGBoost. It also produced the lowest average testing MAE and RMSE and the highest average testing R2.

At plant level, the RBFNN achieved the best testing operational MAPE for AGUS 1, AGUS 2, AGUS 4, AGUS 5, AGUS 6, and AGUS 7. This indicates that the optimized RBFNN generalized better to the final chronological testing period than the benchmark models.

The strongest RBFNN testing performance was observed in AGUS 4, with a testing operational MAPE of 0.611 percent. AGUS 7 and AGUS 5 had higher testing MAPEs, at 4.209 percent and 4.040 percent respectively, suggesting that these plants may still benefit from further calibration or additional explanatory features.

### Conclusion Statement Draft

Based on the optimized validation and testing results, the RBFNN model is the most suitable model among the evaluated approaches for short-term hydroelectric generation forecasting in the Agus cascade system. Its residual forecasting design, cascade-aware features, outage-aware inputs, and validation-tuned correction steps allowed it to outperform Random Forest and XGBoost on the testing period across all six plants.

### Recommendations

Future work may improve the system by:

- Expanding external hydrological inputs such as rainfall forecasts, inflow, reservoir operation schedules, and weather forecasts
- Validating the model with additional months or years of unseen data
- Testing multi-step direct forecasting instead of recursive 24-hour forecasting
- Adding formal uncertainty intervals for day-ahead forecasts
- Improving calibration for AGUS 5 and AGUS 7 due to their higher testing MAPEs
- Automating planned outage updates from operations records instead of relying only on a manually editable workbook
- Comparing the RBFNN with sequence models such as LSTM, GRU, or Temporal Convolutional Networks

## Important Files to Cite or Reference

Source scripts:

- `optimized_v2/scripts/cell1_clean_data.py`
- `optimized_v2/scripts/cell2_rbfnn.py`
- `optimized_v2/scripts/cell3_benchmark.py`
- `optimized_v2/scripts/cell4_testing_plots.py`

Cleaned data:

- `optimized_v2/outputs/cleaned_data/cleaned_hourly_data.xlsx`
- `optimized_v2/outputs/cleaned_data/cleaned_hourly_data.parquet`
- `optimized_v2/outputs/cleaned_data/cell1_metadata.json`

Planned outage file:

- `optimized_v2/outputs/outages_planning/Planned_Outages_Input.xlsx`

Model outputs:

- `optimized_v2/outputs/rbfnn_forecast/Day_Ahead_24H_RBFNN_Forecast.xlsx`
- `optimized_v2/benchmark/random_forest/Day_Ahead_24H_RANDOM_FOREST.xlsx`
- `optimized_v2/benchmark/xgboost/Day_Ahead_24H_XGBOOST.xlsx`

Metrics:

- `optimized_v2/metadata/overall_metrics/rbfnn_validation_testing_metrics.xlsx`
- `optimized_v2/metadata/overall_metrics/random_forest_validation_testing_metrics.xlsx`
- `optimized_v2/metadata/overall_metrics/xgboost_validation_testing_metrics.xlsx`

Testing prediction files:

- `optimized_v2/metadata/overall_metrics/rbfnn_testing_predictions.xlsx`
- `optimized_v2/metadata/overall_metrics/random_forest_testing_predictions.xlsx`
- `optimized_v2/metadata/overall_metrics/xgboost_testing_predictions.xlsx`

Plots:

- `optimized_v2/outputs/testing_plots/full_testing_split/`

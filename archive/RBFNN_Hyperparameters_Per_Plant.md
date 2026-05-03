# RBFNN Hyperparameters per Plant

This table summarizes the optimized RBFNN configuration used for each Agus plant in the `optimized_v2` forecasting pipeline.

| Plant | Feature Count | Selected Centers | Learning Rate | Best Shrinkage | Bias Correction MW | Operational MAPE Threshold MW | Bin Calibration | Hourly Residual Correction | Profile Blend |
|---|---:|---:|---:|---:|---:|---:|---|---|---|
| AGUS 1 | 158 | 80 | 0.001 | 1.15 | -0.0058 | 1.000 | Yes | No | No |
| AGUS 2 | 169 | 120 | 0.0005 | 1.15 | 0.0027 | 1.800 | No | No | No |
| AGUS 4 | 169 | 180 | 0.0005 | 1.15 | 0.0019 | 1.581 | No | No | No |
| AGUS 5 | 168 | 180 | 0.001 | 1.15 | -0.0131 | 1.000 | Yes | Yes | No |
| AGUS 6 | 171 | 80 | 0.0005 | 1.15 | -0.0089 | 2.190 | No | No | No |
| AGUS 7 | 168 | 120 | 0.001 | 1.15 | 0.0075 | 1.000 | No | Yes | No |

## Column Definitions and Purpose

| Column Name | Meaning | Purpose |
|---|---|---|
| Plant | Agus hydropower plant being modeled. | Identifies the specific plant for which the RBFNN model was trained and optimized. |
| Feature Count | Number of engineered input features used by the plant-specific RBFNN model. | Shows the dimensionality of the model input after feature engineering. |
| Selected Centers | Number of radial basis function centers selected for the RBF layer. | Controls the complexity and pattern-learning capacity of the RBFNN. |
| Learning Rate | Step size used by the Adam optimizer during training. | Determines how quickly the model updates its weights during training. |
| Best Shrinkage | Scaling factor applied to the predicted generation change before producing the final forecast. | Stabilizes the residual or delta forecast and helps prevent overcorrection. |
| Bias Correction MW | Small additive correction in megawatts applied after shrinkage. | Reduces systematic underprediction or overprediction observed during validation. |
| Operational MAPE Threshold MW | Minimum actual generation value included in MAPE calculation. | Excludes near-zero actual values where percentage error becomes misleading. |
| Bin Calibration | Indicates whether prediction-bin correction was applied. | Adjusts forecasts based on median residual behavior within prediction or generation ranges. |
| Hourly Residual Correction | Indicates whether hour-of-day residual correction was applied. | Corrects recurring hourly bias patterns in the forecast. |
| Profile Blend | Indicates whether same-hour historical profile blending was applied. | Blends the forecast with a historical profile when useful for preserving daily generation shape. |

## Notes

- The RBFNN models are plant-specific; each plant has its own selected feature set and optimized settings.
- The model predicts generation change or residual behavior, then converts it into final generation forecasts using shrinkage, bias correction, and selected calibration steps.
- Operational MAPE uses the threshold `max(1 MW, 1% of plant capacity)` to avoid misleading percentage errors during near-zero generation periods.

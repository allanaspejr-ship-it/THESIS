# Benchmark Model Testing Metrics Comparison

This report summarizes the testing performance of the benchmark models, Random Forest and XGBoost, for each Agus hydropower plant. It also compares their overall testing performance against the main model, RBFNN.

The metrics are interpreted as follows:

- **Testing MAPE (%)** measures the average percentage forecasting error. Lower values are better.
- **Testing MAE (MW)** measures the average absolute forecasting error in megawatts. Lower values are better.
- **Testing RMSE (MW)** gives more weight to larger errors. Lower values are better.
- **Testing R2** measures how well the predicted values explain the actual generation pattern. Higher values are better and values closer to 1 indicate stronger fit.

## Random Forest Testing Metrics per Agus Plant

| Plant | Feature Count | Testing MAPE (%) | Testing MAE (MW) | Testing RMSE (MW) | Testing R2 |
|---|---:|---:|---:|---:|---:|
| AGUS 1 | 29 | 3.486 | 1.875 | 3.428 | 0.900 |
| AGUS 2 | 30 | 2.161 | 1.934 | 5.005 | 0.905 |
| AGUS 4 | 30 | 1.164 | 1.365 | 3.454 | 0.931 |
| AGUS 5 | 29 | 4.200 | 1.598 | 2.269 | 0.893 |
| AGUS 6 | 32 | 19.879 | 8.033 | 12.198 | 0.956 |
| AGUS 7 | 29 | 4.294 | 1.527 | 2.175 | 0.911 |

Random Forest produced reasonable testing results for most plants, especially AGUS 1, AGUS 2, AGUS 4, AGUS 5, and AGUS 7. However, its AGUS 6 testing error is much higher, with a MAPE of 19.879%, MAE of 8.033 MW, and RMSE of 12.198 MW. This indicates that Random Forest had difficulty forecasting AGUS 6 compared with the other plants, even though its R2 value remained relatively high.

## XGBoost Testing Metrics per Agus Plant

| Plant | Feature Count | Testing MAPE (%) | Testing MAE (MW) | Testing RMSE (MW) | Testing R2 |
|---|---:|---:|---:|---:|---:|
| AGUS 1 | 29 | 3.573 | 1.928 | 3.508 | 0.895 |
| AGUS 2 | 30 | 2.554 | 2.173 | 5.219 | 0.896 |
| AGUS 4 | 30 | 1.051 | 1.266 | 3.273 | 0.938 |
| AGUS 5 | 29 | 4.228 | 1.614 | 2.233 | 0.897 |
| AGUS 6 | 32 | 37.541 | 14.058 | 22.579 | 0.850 |
| AGUS 7 | 29 | 4.311 | 1.542 | 2.174 | 0.911 |

XGBoost also performed acceptably for several plants, particularly AGUS 4, which had the lowest XGBoost MAPE at 1.051%. However, XGBoost showed a very large error for AGUS 6, with a MAPE of 37.541%, MAE of 14.058 MW, and RMSE of 22.579 MW. This makes AGUS 6 the weakest testing case for XGBoost and strongly affects its overall benchmark performance.

## Overall Testing Metrics Comparison

| Model | Role | Average Testing MAPE (%) | Average Testing MAE (MW) | Average Testing RMSE (MW) | Average Testing R2 |
|---|---|---:|---:|---:|---:|
| RBFNN | Main Model | 2.519 | 1.355 | 3.575 | 0.925 |
| Random Forest | Benchmark Model | 5.864 | 2.722 | 4.755 | 0.916 |
| XGBoost | Benchmark Model | 8.876 | 3.764 | 6.498 | 0.898 |

The overall comparison shows that the RBFNN main model achieved the best testing performance among the three models. It had the lowest average MAPE, MAE, and RMSE, meaning its predictions were generally closer to the actual generation values. It also had the highest average R2, showing the strongest overall fit to the testing data.

Random Forest was the stronger benchmark model compared with XGBoost because it had lower average MAPE, MAE, and RMSE and a higher average R2. However, both benchmark models were affected by high AGUS 6 errors, especially XGBoost.

## Summary Interpretation

Based on the testing metrics, the RBFNN model is the most reliable model for the Agus cascade testing set. The benchmark models can still capture the general generation pattern for several plants, but their higher error values show weaker forecasting accuracy. The difference is most visible in AGUS 6, where Random Forest and XGBoost produced much larger errors than RBFNN.

This supports the use of RBFNN as the main forecasting model, while Random Forest and XGBoost serve as useful benchmark models for comparison.

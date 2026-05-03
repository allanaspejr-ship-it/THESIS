# RBFNN Testing Metrics per Plant

This table presents the testing-set performance of the optimized RBFNN model for each Agus hydropower plant. The testing set represents the final chronological 15% of the cleaned dataset and was not used during model training or validation.

| Plant | Feature Count | Testing MAPE (%) | Testing MAE (MW) | Testing RMSE (MW) | Testing R2 |
|---|---:|---:|---:|---:|---:|
| AGUS 1 | 158 | 3.476 | 1.841 | 3.440 | 0.899 |
| AGUS 2 | 169 | 1.372 | 1.164 | 4.810 | 0.912 |
| AGUS 4 | 169 | 0.611 | 0.697 | 3.077 | 0.945 |
| AGUS 5 | 168 | 4.040 | 1.544 | 2.189 | 0.901 |
| AGUS 6 | 171 | 1.405 | 1.396 | 5.679 | 0.990 |
| AGUS 7 | 168 | 4.209 | 1.491 | 2.254 | 0.905 |

## Overall Testing Summary

| Metric | Average Value |
|---|---:|
| Testing MAPE (%) | 2.519 |
| Testing MAE (MW) | 1.355 |
| Testing RMSE (MW) | 3.575 |
| Testing R2 | 0.925 |

## Thesis Explanation

The optimized RBFNN model achieved an average testing MAPE of 2.519%, indicating that the model produced low percentage forecasting error across the Agus cascade during the unseen testing period. The average MAE was 1.355 MW, while the average RMSE was 3.575 MW, showing that the typical prediction error remained small in terms of generation magnitude.

Among the six plants, AGUS 4 achieved the lowest testing MAPE at 0.611%, indicating the strongest percentage-error performance. AGUS 7 recorded the highest testing MAPE at 4.209%, suggesting that this plant had comparatively greater forecasting variability or operational complexity. The highest testing R2 was obtained by AGUS 6 with a value of 0.990, showing that the model explained most of the observed variation for that plant.

Overall, the testing results indicate that the RBFNN generalized well to unseen chronological data. The high R2 values across all plants show strong agreement between actual and predicted generation, while the low MAPE, MAE, and RMSE values support the suitability of the optimized RBFNN for short-term hydropower generation forecasting in the Agus cascade.

Note: MAPE follows the optimized pipeline definition, where near-zero actual generation values are excluded to avoid misleading percentage errors.
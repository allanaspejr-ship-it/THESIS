# RBFNN Overall Testing Metrics for the Agus Cascade

This table summarizes the overall testing performance of the optimized RBFNN model across the Agus cascade. The values are computed as the average of the six plant-level testing metrics.

| Metric | Value |
|---|---:|
| Average Testing MAPE (%) | 2.519 |
| Average Testing MAE (MW) | 1.355 |
| Average Testing RMSE (MW) | 3.575 |
| Average Testing R2 | 0.925 |

## Thesis Summary

The optimized RBFNN model achieved an average testing MAPE of 2.519% across the Agus cascade, indicating low percentage forecasting error on unseen chronological data. The average MAE of 1.355 MW and RMSE of 3.575 MW show that the magnitude of prediction errors remained relatively small. The average R2 value of 0.925 indicates that the model explained a high proportion of the observed variation in hydropower generation across the cascade.

Overall, these results show that the RBFNN generalized well during the testing period and was able to provide accurate short-term generation forecasts for the Agus hydropower cascade.

Note: MAPE follows the optimized pipeline definition, where near-zero actual generation values are excluded to avoid misleading percentage errors.

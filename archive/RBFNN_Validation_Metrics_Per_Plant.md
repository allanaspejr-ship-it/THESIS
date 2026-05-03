# RBFNN Validation Metrics per Plant

| Plant | Feature Count | Validation Operational MAPE (%) | Validation MAE (MW) | Validation RMSE (MW) | Validation R2 |
|---|---:|---:|---:|---:|---:|
| AGUS 1 | 158 | 3.471 | 1.727 | 2.756 | 0.929 |
| AGUS 2 | 169 | 1.080 | 0.984 | 4.297 | 0.822 |
| AGUS 4 | 169 | 0.827 | 0.717 | 3.362 | 0.953 |
| AGUS 5 | 168 | 4.091 | 1.469 | 2.231 | 0.904 |
| AGUS 6 | 171 | 2.413 | 3.667 | 7.981 | 0.835 |
| AGUS 7 | 168 | 3.806 | 1.234 | 2.109 | 0.824 |

These metrics are from the validation split only. Operational MAPE excludes near-zero actual generation values using the optimized pipeline threshold rule.
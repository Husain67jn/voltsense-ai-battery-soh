# Model Comparison: Version 1 vs Version 2

| Version | Target | Rows | Best model | MAE | RMSE | R2 |
| --- | --- | ---: | --- | ---: | ---: | ---: |
| Version 1 | SOH = Capacity / first valid discharge Capacity per battery_id | 2750 | random_forest | 0.971712 | 2.375324 | -17.752190 |
| Version 2 | SOH = Capacity / 2.0 Ah, filtered to 0.5 <= SOH <= 1.2 | 2307 | hist_gradient_boosting | 0.027659 | 0.045350 | 0.840585 |

## R2 Change
Version 2 R2 changed by 18.592774 compared with Version 1.
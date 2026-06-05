# Final Project Summary: Battery SOH Prediction

## Overview

This project builds a leakage-safe machine learning pipeline to predict battery State of Health (SOH) from NASA lithium-ion battery aging data. The pipeline loads metadata, filters discharge cycles, extracts statistical features from per-cycle measurement files, trains multiple regression models, evaluates on held-out batteries, and saves reproducible artifacts.

## Dataset and Target

Only discharge cycles are used as supervised target rows. Charge and impedance rows are excluded from target construction.

Version 1 used:

```text
SOH = Capacity / first valid discharge Capacity per battery_id
```

Diagnostics showed this was unstable because some batteries had abnormal first valid capacity values, producing unrealistic SOH values up to `27.550168`.

Version 2 uses:

```text
SOH = Capacity / 2.0
```

Rows are kept only when:

```text
Capacity > 0
0.5 <= SOH <= 1.2
```

## Validation

The project uses a group-aware split by `battery_id`. Entire batteries are held out for testing, which avoids leakage from cycles of the same battery appearing in both train and test sets.

Version 2 used:

- Rows: `2307`
- Train rows: `2006`
- Test rows: `301`
- Held-out test batteries: `B0029`, `B0038`, `B0042`, `B0044`, `B0047`, `B0049`, `B0050`

## Results

Best Version 2 model:

```text
HistGradientBoostingRegressor
```

Held-out battery test metrics:

| Metric | Value |
| --- | ---: |
| MAE | 0.028529 |
| RMSE | 0.046789 |
| R2 | 0.830306 |

Version 2 improved R2 from `-17.752190` to `0.830306` by replacing the unstable target definition with rated-capacity normalization and realistic SOH filtering.

## Explainability

Permutation feature importance was computed for the Version 2 best model on the same held-out test batteries. Scores represent mean decrease in R2 after permuting each feature.

Top 10 features:

| Rank | Feature | Mean R2 decrease | Std |
| ---: | --- | ---: | ---: |
| 1 | `temperature_mean` | 0.979670 | 0.065129 |
| 2 | `discharge_duration` | 0.459223 | 0.033055 |
| 3 | `voltage_load_mean` | 0.075765 | 0.007540 |
| 4 | `voltage_std` | 0.066423 | 0.014162 |
| 5 | `cycle_index` | 0.057099 | 0.015300 |
| 6 | `voltage_load_std` | 0.052145 | 0.008067 |
| 7 | `current_load_mean` | 0.039138 | 0.007854 |
| 8 | `temperature_max` | 0.030957 | 0.003772 |
| 9 | `voltage_mean` | 0.029363 | 0.002654 |
| 10 | `voltage_max` | 0.019593 | 0.001951 |

## Key Artifacts

- `models/best_model_v2.joblib`
- `reports/metrics_v2.json`
- `reports/predictions_v2.csv`
- `reports/predicted_vs_actual_v2.png`
- `reports/error_by_battery_v2.png`
- `reports/soh_curve_by_battery_v2.png`
- `reports/feature_importance_v2.csv`
- `reports/feature_importance_v2.png`
- `reports/model_comparison_v1_v2.md`

## Limitations

The current model uses simple statistical features and a single held-out battery split. It does not use full sequence modeling, repeated grouped cross-validation, uncertainty estimation, or external dataset validation. It should be treated as a strong portfolio baseline, not a production battery health system.

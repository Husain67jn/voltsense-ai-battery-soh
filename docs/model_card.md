# Model Card: Battery SOH Prediction

## Model Purpose

This model predicts lithium-ion battery State of Health (SOH) from discharge-cycle measurements in the NASA cleaned battery aging dataset. It is intended as a machine learning engineering portfolio project and a reproducible baseline for battery degradation modeling.

It is not intended for direct use in production battery management systems or safety-critical decisions.

## Model Type

The best Version 2 model is:

```text
HistGradientBoostingRegressor
```

The pipeline also compares a mean baseline, Ridge Regression, and RandomForestRegressor.

## Input Features

The model uses numeric features extracted from discharge-cycle CSV files and metadata:

- Ambient temperature
- Cycle index
- Voltage measured: mean, min, max, standard deviation
- Current measured: mean, min, max, standard deviation
- Temperature measured: mean, min, max, standard deviation
- Voltage load: mean, min, max, standard deviation
- Current load: mean, min, max, standard deviation
- Discharge duration from `Time`

The following columns are excluded from model features:

- `battery_id`
- `Capacity`
- `SOH`
- File identifiers and metadata IDs

## Target

Version 2 target:

```text
SOH = Capacity / 2.0
```

Rows are retained only when:

```text
Capacity > 0
0.5 <= SOH <= 1.2
```

This replaced the unstable Version 1 target, which normalized by the first valid discharge capacity per battery and produced unrealistic SOH values for some cells.

## Validation Method

The validation strategy is group-aware by `battery_id`. Entire batteries are held out for testing so the model is evaluated on unseen cells.

Version 2 split:

- Train rows: `2006`
- Test rows: `301`
- Total rows used: `2307`
- Test batteries: `B0029`, `B0038`, `B0042`, `B0044`, `B0047`, `B0049`, `B0050`

This avoids random row leakage across cycles from the same battery.

## Metrics

Final Version 2 held-out test metrics:

| Metric | Value |
| --- | ---: |
| MAE | 0.028529 |
| RMSE | 0.046789 |
| R2 | 0.830306 |

Best model:

```text
HistGradientBoostingRegressor
```

## Limitations

- The model uses simple statistical features, not the full discharge curve as a sequence.
- Results are based on one held-out battery split, not repeated grouped cross-validation.
- The target assumes a nominal capacity of `2.0 Ah`.
- The model is trained on one dataset and may not generalize to other battery chemistries, cell formats, sensors, operating conditions, or aging protocols.
- The model does not provide uncertainty estimates.

## Ethical and Real-World Deployment Caution

Battery health models can influence maintenance, warranty, safety, and replacement decisions. Incorrect SOH predictions may lead to premature replacement, missed degradation, unsafe operation, or financial loss.

Before any real-world deployment, this model would need validation on independent datasets, domain review, uncertainty quantification, monitoring for distribution shift, and integration with safety constraints from battery experts.

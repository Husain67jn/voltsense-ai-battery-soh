# Battery State of Health Prediction

Machine learning pipeline for predicting lithium-ion battery State of Health (SOH) from NASA battery aging data. The project demonstrates practical ML engineering work: dataset inspection, target validation, leakage-safe splitting, feature extraction from cycle files, model comparison, and reproducible saved reports.

## Problem Statement

Battery State of Health estimates how much usable capacity remains compared with a nominal or rated capacity. Accurate SOH prediction helps with maintenance planning, battery lifecycle analysis, and early detection of degradation.

This project predicts SOH from discharge-cycle measurements such as voltage, current, temperature, load values, discharge duration, ambient temperature, and cycle index.

## Dataset

The repository uses the NASA cleaned battery dataset already included locally:

- Metadata: `data/metadata.csv`
- Per-cycle measurement files: `Nasa/cleaned_dataset/data/*.csv`

The metadata contains charge, discharge, and impedance records. Only discharge cycles with valid `Capacity` values are used as supervised target rows. Charge and impedance rows are not used as target rows.

## Target Definition

### Version 1

Version 1 used:

```text
SOH = Capacity / first valid discharge Capacity per battery_id
```

Diagnostics showed this target was unstable. Some batteries had abnormal first valid capacity values, which produced physically unrealistic SOH values as high as `27.550168`. This caused poor model metrics even with a leakage-safe split.

### Version 2

Version 2 uses the rated-capacity target:

```text
SOH = Capacity / 2.0
```

The value `2.0 Ah` is used as the nominal rated capacity for this NASA battery dataset. Version 2 keeps only realistic target rows:

```text
Capacity > 0
0.5 <= SOH <= 1.2
```

This target definition removed the unstable first-capacity normalization issue and produced a much more meaningful learning problem.

## Leakage-Safe Validation

Battery cycles from the same cell are highly correlated. A random row split would leak battery-specific degradation trajectory information from train to test.

This project uses a group-aware train/test split by `battery_id`, holding out entire batteries for testing. The model is evaluated on batteries it did not see during training.

Version 2 split:

- Train batteries: `B0005`, `B0006`, `B0007`, `B0018`, `B0025`, `B0026`, `B0027`, `B0028`, `B0030`, `B0031`, `B0032`, `B0033`, `B0034`, `B0036`, `B0039`, `B0040`, `B0041`, `B0043`, `B0045`, `B0046`, `B0048`, `B0051`, `B0052`, `B0053`, `B0054`, `B0055`, `B0056`
- Held-out test batteries: `B0029`, `B0038`, `B0042`, `B0044`, `B0047`, `B0049`, `B0050`

The model features exclude `battery_id`, `Capacity`, and `SOH`.

## Features

For each discharge cycle CSV, the pipeline extracts simple statistical features:

- Voltage measured: mean, min, max, standard deviation
- Current measured: mean, min, max, standard deviation
- Temperature measured: mean, min, max, standard deviation
- Voltage load: mean, min, max, standard deviation
- Current load: mean, min, max, standard deviation
- Discharge duration from `Time`
- Metadata features: ambient temperature and cycle index

## Models

The training pipeline compares:

- Mean baseline model
- Ridge Regression
- RandomForestRegressor
- HistGradientBoostingRegressor

The best model is selected by test RMSE.

## Results

Final Version 2 results:

| Metric | Value |
| --- | ---: |
| MAE | 0.028529 |
| RMSE | 0.046789 |
| R2 | 0.830306 |
| Best model | HistGradientBoostingRegressor |
| Rows used | 2307 |
| Train rows | 2006 |
| Test rows | 301 |
| Split | Held-out battery IDs |

Version 2 substantially improved over Version 1:

| Version | Target | Best model | R2 |
| --- | --- | --- | ---: |
| Version 1 | `Capacity / first valid discharge capacity per battery_id` | RandomForestRegressor | -17.752190 |
| Version 2 | `Capacity / 2.0`, filtered to `0.5 <= SOH <= 1.2` | HistGradientBoostingRegressor | 0.830306 |

## Explainability

Permutation feature importance was computed for the Version 2 best model on the held-out test batteries. Importance is reported as mean decrease in test-set R2 after randomly permuting each feature.

Top 10 Version 2 features:

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

Generated explainability artifacts:

- `reports/feature_importance_v2.csv`
- `reports/feature_importance_v2.png`

## How to Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Train Version 2:

```bash
python src/train.py --target_strategy rated_capacity
```

Evaluate Version 2 and generate plots:

```bash
python src/evaluate.py --predictions reports/predictions_v2.csv --suffix v2
```

Run the Streamlit demo app:

```bash
streamlit run app.py
```

Compute Version 2 permutation feature importance:

```bash
python src/explain.py
```

Train the original Version 1 target, if needed:

```bash
python src/train.py
```

Predict SOH for one discharge cycle CSV:

```bash
python src/predict.py --csv Nasa/cleaned_dataset/data/00001.csv --ambient-temperature 4 --cycle-index 1 --model models/best_model_v2.joblib
```

## Project Structure

```text
battery-soh-prediction/
|-- app.py
|-- data/
|   `-- metadata.csv
|-- docs/
|   `-- model_card.md
|-- models/
|   |-- best_model.joblib
|   `-- best_model_v2.joblib
|-- Nasa/
|   `-- cleaned_dataset/
|       `-- data/
|-- reports/
|   |-- metrics.json
|   |-- metrics_v2.json
|   |-- predictions_v2.csv
|   |-- feature_importance_v2.csv
|   |-- feature_importance_v2.png
|   |-- model_comparison_v1_v2.md
|   `-- final_summary.md
|-- src/
|   |-- data_loader.py
|   |-- features.py
|   |-- train.py
|   |-- evaluate.py
|   |-- explain.py
|   `-- predict.py
`-- requirements.txt
```

## Limitations

- Version 2 uses statistical summaries of each discharge cycle rather than full time-series sequence modeling.
- `2.0 Ah` is treated as the rated capacity for target construction; real deployments should verify rated capacity for the exact cell type.
- The held-out-battery split is more realistic than random splitting, but it is still based on one split rather than repeated grouped cross-validation.
- The model does not estimate uncertainty.
- This project is for ML engineering demonstration and research-style analysis, not direct safety-critical battery management deployment.

## Future Improvements

- Add richer electrochemical and time-series features such as voltage threshold times, discharge curve slopes, temperature rise, and area-under-curve features.
- Add grouped cross-validation across battery IDs.
- Tune model hyperparameters with leakage-safe validation.
- Compare with XGBoost, LightGBM, and CatBoost.
- Add sequence models such as 1D CNN, GRU, LSTM, or temporal convolutional networks.
- Add model explainability using permutation importance or SHAP.
- Add automated tests for data loading, target filtering, split integrity, and feature generation.

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from features import extract_discharge_features


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "best_model.joblib"


def load_model(model_path=MODEL_PATH):
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}. Run src/train.py first.")
    return joblib.load(model_path)


def predict_from_features(feature_values, model_path=MODEL_PATH):
    bundle = load_model(model_path)
    feature_columns = bundle["feature_columns"]
    model = bundle["model"]

    row = pd.DataFrame([feature_values])
    for column in feature_columns:
        if column not in row.columns:
            row[column] = np.nan
    row = row[feature_columns]
    return float(model.predict(row)[0])


def predict_from_cycle_csv(
    csv_path,
    ambient_temperature=None,
    cycle_index=None,
    model_path=MODEL_PATH,
):
    features = extract_discharge_features(csv_path)
    if ambient_temperature is not None:
        features["ambient_temperature"] = ambient_temperature
    if cycle_index is not None:
        features["cycle_index"] = cycle_index
    return predict_from_features(features, model_path=model_path)


def main():
    parser = argparse.ArgumentParser(description="Predict battery SOH from a discharge cycle CSV.")
    parser.add_argument("--csv", required=True, help="Path to a discharge cycle CSV file.")
    parser.add_argument("--ambient-temperature", type=float, default=None)
    parser.add_argument("--cycle-index", type=float, default=None)
    parser.add_argument(
        "--model",
        default=str(MODEL_PATH),
        help="Path to a saved model bundle.",
    )
    args = parser.parse_args()

    prediction = predict_from_cycle_csv(
        args.csv,
        ambient_temperature=args.ambient_temperature,
        cycle_index=args.cycle_index,
        model_path=args.model,
    )
    print(f"Predicted SOH: {prediction:.6f}")


if __name__ == "__main__":
    main()

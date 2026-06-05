import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from features import build_feature_table, get_feature_columns


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
MODEL_PATH = MODELS_DIR / "best_model.joblib"
METRICS_PATH = REPORTS_DIR / "metrics.json"
PREDICTIONS_PATH = REPORTS_DIR / "predictions.csv"
MODEL_COMPARISON_PATH = REPORTS_DIR / "model_comparison_v1_v2.md"


RANDOM_STATE = 42


def artifact_paths(target_strategy):
    if target_strategy == "rated_capacity":
        return {
            "model": MODELS_DIR / "best_model_v2.joblib",
            "metrics": REPORTS_DIR / "metrics_v2.json",
            "predictions": REPORTS_DIR / "predictions_v2.csv",
        }

    return {
        "model": MODEL_PATH,
        "metrics": METRICS_PATH,
        "predictions": PREDICTIONS_PATH,
    }


def target_description(target_strategy):
    if target_strategy == "rated_capacity":
        return "SOH = Capacity / 2.0 Ah, filtered to 0.5 <= SOH <= 1.2"
    return "SOH = Capacity / first valid discharge Capacity per battery_id"


def regression_metrics(y_true, y_pred):
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)),
    }


def make_models():
    return {
        "baseline_mean": DummyRegressor(strategy="mean"),
        "ridge_regression": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", Ridge(alpha=1.0)),
            ]
        ),
        "random_forest": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=300,
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                        min_samples_leaf=2,
                    ),
                ),
            ]
        ),
        "hist_gradient_boosting": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    HistGradientBoostingRegressor(
                        random_state=RANDOM_STATE,
                        max_iter=300,
                        learning_rate=0.05,
                    ),
                ),
            ]
        ),
    }


def write_v1_v2_comparison(v2_metrics):
    if not METRICS_PATH.exists():
        return

    v1_metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    rows = [
        (
            "Version 1",
            v1_metrics.get("target", ""),
            v1_metrics.get("n_rows"),
            v1_metrics.get("best_model"),
            v1_metrics.get("best_model_metrics", {}),
        ),
        (
            "Version 2",
            v2_metrics.get("target", ""),
            v2_metrics.get("n_rows"),
            v2_metrics.get("best_model"),
            v2_metrics.get("best_model_metrics", {}),
        ),
    ]

    lines = [
        "# Model Comparison: Version 1 vs Version 2",
        "",
        "| Version | Target | Rows | Best model | MAE | RMSE | R2 |",
        "| --- | --- | ---: | --- | ---: | ---: | ---: |",
    ]
    for version, target, n_rows, best_model, metrics in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(version),
                    str(target),
                    str(n_rows),
                    str(best_model),
                    f"{metrics.get('MAE', float('nan')):.6f}",
                    f"{metrics.get('RMSE', float('nan')):.6f}",
                    f"{metrics.get('R2', float('nan')):.6f}",
                ]
            )
            + " |"
        )

    v1_r2 = v1_metrics.get("best_model_metrics", {}).get("R2")
    v2_r2 = v2_metrics.get("best_model_metrics", {}).get("R2")
    lines.extend(["", "## R2 Change"])
    if v1_r2 is not None and v2_r2 is not None:
        lines.append(f"Version 2 R2 changed by {v2_r2 - v1_r2:.6f} compared with Version 1.")
    else:
        lines.append("R2 comparison was unavailable because one metrics file was missing R2.")

    MODEL_COMPARISON_PATH.write_text("\n".join(lines), encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description="Train Battery SOH prediction models.")
    parser.add_argument(
        "--target_strategy",
        choices=["first_valid_capacity", "rated_capacity"],
        default="first_valid_capacity",
        help="Target definition to use for SOH.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    paths = artifact_paths(args.target_strategy)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    feature_table = build_feature_table(target_strategy=args.target_strategy)
    feature_columns = get_feature_columns(feature_table)
    if not feature_columns:
        raise ValueError("No numeric feature columns were created.")

    modeling_table = feature_table.dropna(subset=["SOH", "battery_id"]).copy()
    groups = modeling_table["battery_id"]
    X = modeling_table[feature_columns]
    y = modeling_table["SOH"].astype(float)

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
    train_idx, test_idx = next(splitter.split(X, y, groups=groups))

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    test_metadata = modeling_table.iloc[test_idx].copy()

    results = {}
    trained_models = {}
    for model_name, model in make_models().items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        results[model_name] = regression_metrics(y_test, predictions)
        trained_models[model_name] = model

    best_model_name = min(results, key=lambda name: results[name]["RMSE"])
    best_model = trained_models[best_model_name]
    best_predictions = best_model.predict(X_test)

    model_bundle = {
        "model": best_model,
        "model_name": best_model_name,
        "feature_columns": feature_columns,
    }
    joblib.dump(model_bundle, paths["model"])

    predictions_output = test_metadata[
        ["battery_id", "cycle_index", "Capacity", "SOH", "filename"]
    ].copy()
    predictions_output["actual_soh"] = y_test.to_numpy()
    predictions_output["predicted_soh"] = best_predictions
    predictions_output["error"] = predictions_output["predicted_soh"] - predictions_output["actual_soh"]
    predictions_output.to_csv(paths["predictions"], index=False)

    metrics_output = {
        "target": target_description(args.target_strategy),
        "target_strategy": args.target_strategy,
        "split_strategy": "GroupShuffleSplit by battery_id",
        "random_state": RANDOM_STATE,
        "n_rows": int(len(modeling_table)),
        "n_features": int(len(feature_columns)),
        "train_rows": int(len(train_idx)),
        "test_rows": int(len(test_idx)),
        "train_batteries": sorted(modeling_table.iloc[train_idx]["battery_id"].unique().tolist()),
        "test_batteries": sorted(modeling_table.iloc[test_idx]["battery_id"].unique().tolist()),
        "feature_columns": feature_columns,
        "models": results,
        "best_model": best_model_name,
        "best_model_metrics": results[best_model_name],
    }
    paths["metrics"].write_text(json.dumps(metrics_output, indent=2), encoding="utf-8")
    if args.target_strategy == "rated_capacity":
        write_v1_v2_comparison(metrics_output)

    print("Training complete")
    print(f"Rows: {len(modeling_table)}")
    print(f"Features: {len(feature_columns)}")
    print(f"Train batteries: {', '.join(metrics_output['train_batteries'])}")
    print(f"Test batteries: {', '.join(metrics_output['test_batteries'])}")
    print(f"Best model: {best_model_name}")
    print(json.dumps(results[best_model_name], indent=2))
    print(f"Target strategy: {args.target_strategy}")
    print(f"Saved model: {paths['model']}")
    print(f"Saved metrics: {paths['metrics']}")
    print(f"Saved predictions: {paths['predictions']}")
    if args.target_strategy == "rated_capacity" and MODEL_COMPARISON_PATH.exists():
        print(f"Saved comparison: {MODEL_COMPARISON_PATH}")


if __name__ == "__main__":
    main()

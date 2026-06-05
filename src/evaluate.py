import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = PROJECT_ROOT / "reports"
PREDICTIONS_PATH = REPORTS_DIR / "predictions.csv"
METRICS_PATH = REPORTS_DIR / "metrics.json"


def _plot_path(base_name, suffix):
    if suffix:
        return REPORTS_DIR / f"{base_name}_{suffix}.png"
    return REPORTS_DIR / f"{base_name}.png"


def _metrics_path_for_suffix(suffix):
    if suffix:
        return REPORTS_DIR / f"metrics_{suffix}.json"
    return METRICS_PATH


def _save_predicted_vs_actual(predictions, suffix=""):
    plt.figure(figsize=(7, 6))
    plt.scatter(predictions["actual_soh"], predictions["predicted_soh"], alpha=0.75)
    lower = min(predictions["actual_soh"].min(), predictions["predicted_soh"].min())
    upper = max(predictions["actual_soh"].max(), predictions["predicted_soh"].max())
    plt.plot([lower, upper], [lower, upper], color="black", linestyle="--", linewidth=1)
    plt.xlabel("Actual SOH")
    plt.ylabel("Predicted SOH")
    plt.title("Predicted vs Actual SOH")
    plt.tight_layout()
    output_path = _plot_path("predicted_vs_actual", suffix)
    plt.savefig(output_path, dpi=160)
    plt.close()
    return output_path


def _save_error_by_battery(predictions, suffix=""):
    error_by_battery = (
        predictions.assign(abs_error=predictions["error"].abs())
        .groupby("battery_id", as_index=False)["abs_error"]
        .mean()
        .sort_values("abs_error", ascending=False)
    )

    plt.figure(figsize=(8, 5))
    plt.bar(error_by_battery["battery_id"], error_by_battery["abs_error"])
    plt.xlabel("Battery ID")
    plt.ylabel("Mean Absolute Error")
    plt.title("Prediction Error by Battery")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    output_path = _plot_path("error_by_battery", suffix)
    plt.savefig(output_path, dpi=160)
    plt.close()
    return output_path


def _save_soh_curve_by_battery(predictions, suffix=""):
    if "cycle_index" not in predictions.columns:
        return None

    plt.figure(figsize=(9, 6))
    for battery_id, group in predictions.groupby("battery_id"):
        group = group.sort_values("cycle_index")
        plt.plot(group["cycle_index"], group["actual_soh"], label=f"{battery_id} actual")
        plt.plot(
            group["cycle_index"],
            group["predicted_soh"],
            linestyle="--",
            label=f"{battery_id} predicted",
        )

    plt.xlabel("Cycle Index")
    plt.ylabel("SOH")
    plt.title("SOH Curve by Test Battery")
    plt.legend(fontsize=8)
    plt.tight_layout()
    output_path = _plot_path("soh_curve_by_battery", suffix)
    plt.savefig(output_path, dpi=160)
    plt.close()
    return output_path


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate saved Battery SOH predictions.")
    parser.add_argument(
        "--predictions",
        default=str(PREDICTIONS_PATH),
        help="Path to predictions CSV.",
    )
    parser.add_argument(
        "--suffix",
        default="",
        help="Suffix for output plot names, for example 'v2'.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    predictions_path = Path(args.predictions)
    if not predictions_path.is_absolute():
        predictions_path = PROJECT_ROOT / predictions_path

    if not predictions_path.exists():
        raise FileNotFoundError(
            f"Predictions file not found: {predictions_path}. Run src/train.py first."
        )

    predictions = pd.read_csv(predictions_path)
    mae = mean_absolute_error(predictions["actual_soh"], predictions["predicted_soh"])
    rmse = np.sqrt(mean_squared_error(predictions["actual_soh"], predictions["predicted_soh"]))
    r2 = r2_score(predictions["actual_soh"], predictions["predicted_soh"])

    print("Evaluation metrics")
    print(f"MAE:  {mae:.6f}")
    print(f"RMSE: {rmse:.6f}")
    print(f"R2:   {r2:.6f}")

    metrics_path = _metrics_path_for_suffix(args.suffix)
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        print(f"Best model from training: {metrics.get('best_model')}")
        print(f"Test batteries: {', '.join(metrics.get('test_batteries', []))}")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    plot_paths = [
        _save_predicted_vs_actual(predictions, args.suffix),
        _save_error_by_battery(predictions, args.suffix),
        _save_soh_curve_by_battery(predictions, args.suffix),
    ]

    print("Saved plots")
    for path in plot_paths:
        if path is not None:
            print(path)


if __name__ == "__main__":
    main()

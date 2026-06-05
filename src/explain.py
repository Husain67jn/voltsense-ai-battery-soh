import argparse
import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.inspection import permutation_importance

from features import build_feature_table


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
MODEL_PATH = MODELS_DIR / "best_model_v2.joblib"
METRICS_PATH = REPORTS_DIR / "metrics_v2.json"
IMPORTANCE_CSV_PATH = REPORTS_DIR / "feature_importance_v2.csv"
IMPORTANCE_PLOT_PATH = REPORTS_DIR / "feature_importance_v2.png"
RANDOM_STATE = 42


def load_v2_artifacts(model_path=MODEL_PATH, metrics_path=METRICS_PATH):
    model_path = Path(model_path)
    metrics_path = Path(metrics_path)

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not metrics_path.exists():
        raise FileNotFoundError(f"Metrics file not found: {metrics_path}")

    model_bundle = joblib.load(model_path)
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    return model_bundle, metrics


def get_v2_test_data(metrics, feature_columns):
    feature_table = build_feature_table(target_strategy="rated_capacity")
    test_batteries = set(metrics["test_batteries"])
    test_table = feature_table.loc[feature_table["battery_id"].isin(test_batteries)].copy()

    if test_table.empty:
        raise ValueError("No Version 2 held-out test rows found for permutation importance.")

    X_test = test_table[feature_columns]
    y_test = test_table["SOH"].astype(float)
    return X_test, y_test


def compute_feature_importance(model, X_test, y_test, n_repeats):
    result = permutation_importance(
        model,
        X_test,
        y_test,
        scoring="r2",
        n_repeats=n_repeats,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    importance = pd.DataFrame(
        {
            "feature": X_test.columns,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)
    importance.insert(0, "rank", range(1, len(importance) + 1))
    return importance


def save_importance_plot(importance, output_path=IMPORTANCE_PLOT_PATH, top_n=15):
    plot_data = importance.head(top_n).sort_values("importance_mean")

    plt.figure(figsize=(9, 6))
    plt.barh(plot_data["feature"], plot_data["importance_mean"], xerr=plot_data["importance_std"])
    plt.xlabel("Mean decrease in R2 after permutation")
    plt.ylabel("Feature")
    plt.title("Version 2 Permutation Feature Importance")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()
    return output_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute permutation feature importance for the Version 2 SOH model."
    )
    parser.add_argument("--model", default=str(MODEL_PATH), help="Path to Version 2 model bundle.")
    parser.add_argument("--metrics", default=str(METRICS_PATH), help="Path to Version 2 metrics JSON.")
    parser.add_argument("--n-repeats", type=int, default=20, help="Permutation repeats per feature.")
    return parser.parse_args()


def main():
    args = parse_args()
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    model_bundle, metrics = load_v2_artifacts(args.model, args.metrics)
    feature_columns = model_bundle["feature_columns"]
    model = model_bundle["model"]

    X_test, y_test = get_v2_test_data(metrics, feature_columns)
    importance = compute_feature_importance(model, X_test, y_test, args.n_repeats)

    importance.to_csv(IMPORTANCE_CSV_PATH, index=False)
    save_importance_plot(importance)

    print("Permutation feature importance complete")
    print(f"Scoring: R2 decrease on Version 2 held-out test batteries")
    print(f"Test rows: {len(X_test)}")
    print(f"Saved table: {IMPORTANCE_CSV_PATH}")
    print(f"Saved plot: {IMPORTANCE_PLOT_PATH}")
    print("Top 10 features:")
    print(importance.head(10).to_string(index=False))


if __name__ == "__main__":
    main()

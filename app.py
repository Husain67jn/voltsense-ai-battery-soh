import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent
REPORTS_DIR = PROJECT_ROOT / "reports"
MODELS_DIR = PROJECT_ROOT / "models"
PREDICTIONS_PATH = REPORTS_DIR / "predictions_v2.csv"
METRICS_PATH = REPORTS_DIR / "metrics_v2.json"
FEATURE_IMPORTANCE_PATH = REPORTS_DIR / "feature_importance_v2.csv"
MODEL_PATH = MODELS_DIR / "best_model_v2.joblib"


REQUIRED_PREDICTION_COLUMNS = {
    "battery_id",
    "cycle_index",
    "actual_soh",
    "predicted_soh",
    "error",
}


FALLBACK_FEATURE_RANGES = {
    "ambient_temperature": (4.0, 24.0, 44.0),
    "cycle_index": (1.0, 50.0, 200.0),
    "voltage_mean": (2.8, 3.5, 4.2),
    "voltage_min": (2.0, 2.8, 3.8),
    "voltage_max": (3.4, 4.1, 4.3),
    "voltage_std": (0.0, 0.25, 0.8),
    "current_mean": (-3.0, -1.5, 0.5),
    "current_min": (-5.0, -2.0, 0.0),
    "current_max": (-1.0, 0.0, 2.0),
    "current_std": (0.0, 0.2, 1.0),
    "temperature_mean": (15.0, 32.0, 55.0),
    "temperature_min": (10.0, 25.0, 45.0),
    "temperature_max": (20.0, 38.0, 70.0),
    "temperature_std": (0.0, 2.0, 12.0),
    "voltage_load_mean": (1.5, 2.8, 4.2),
    "voltage_load_min": (0.5, 2.0, 3.8),
    "voltage_load_max": (2.0, 3.5, 4.5),
    "voltage_load_std": (0.0, 0.35, 1.2),
    "current_load_mean": (0.0, 1.5, 3.0),
    "current_load_min": (0.0, 1.0, 3.0),
    "current_load_max": (0.0, 2.0, 4.0),
    "current_load_std": (0.0, 0.2, 1.0),
    "discharge_duration": (100.0, 2500.0, 5000.0),
}


def inject_css():
    st.markdown(
        """
<style>
:root {
  --bg: #070b13;
  --panel: rgba(16, 24, 39, 0.88);
  --panel-soft: rgba(23, 36, 58, 0.72);
  --border: rgba(148, 163, 184, 0.20);
  --text: #e5edf7;
  --muted: #9fb0c6;
  --accent: #38bdf8;
  --accent-2: #22c55e;
}

.stApp {
  background:
    radial-gradient(circle at 12% 0%, rgba(56, 189, 248, 0.10), transparent 30%),
    linear-gradient(135deg, #070b13 0%, #0f172a 48%, #111827 100%);
  color: var(--text);
}

[data-testid="stSidebar"] {
  background: rgba(6, 11, 20, 0.96);
  border-right: 1px solid var(--border);
}

[data-testid="stSidebar"] * {
  color: var(--text);
}

.block-container {
  padding-top: 2rem;
  padding-bottom: 3rem;
  max-width: 1280px;
}

h1, h2, h3 {
  letter-spacing: 0;
}

.hero {
  border: 1px solid var(--border);
  border-radius: 18px;
  padding: 30px 32px;
  background: linear-gradient(135deg, rgba(15, 23, 42, 0.94), rgba(12, 74, 110, 0.28));
  box-shadow: 0 24px 80px rgba(0, 0, 0, 0.30);
  margin-bottom: 22px;
}

.eyebrow {
  color: var(--accent);
  font-size: 0.82rem;
  font-weight: 700;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  margin-bottom: 10px;
}

.hero-title {
  color: #f8fafc;
  font-size: 2.7rem;
  font-weight: 800;
  line-height: 1.05;
  margin: 0;
}

.hero-subtitle {
  color: #cbd5e1;
  font-size: 1.08rem;
  line-height: 1.55;
  max-width: 850px;
  margin-top: 14px;
}

.metric-card, .soft-card, .impact-card {
  border: 1px solid var(--border);
  border-radius: 16px;
  padding: 18px 20px;
  background: var(--panel);
  box-shadow: 0 14px 38px rgba(0, 0, 0, 0.18);
}

.metric-label {
  color: var(--muted);
  font-size: 0.82rem;
  font-weight: 650;
  text-transform: uppercase;
  letter-spacing: 0.04em;
}

.metric-value {
  color: #f8fafc;
  font-size: 1.55rem;
  font-weight: 800;
  margin-top: 6px;
  overflow-wrap: anywhere;
}

.section-note {
  color: #cbd5e1;
  font-size: 1rem;
  line-height: 1.55;
}

.status-pill {
  display: inline-block;
  padding: 8px 12px;
  border-radius: 999px;
  border: 1px solid var(--border);
  color: #f8fafc;
  background: rgba(56, 189, 248, 0.12);
  font-weight: 700;
}

.warning-box {
  border: 1px solid rgba(250, 204, 21, 0.35);
  background: rgba(113, 63, 18, 0.30);
  color: #fef3c7;
  border-radius: 14px;
  padding: 14px 16px;
}

div[data-testid="stDataFrame"] {
  border: 1px solid var(--border);
  border-radius: 14px;
  overflow: hidden;
}

button[kind="primary"], button[kind="secondary"] {
  border-radius: 10px;
}
</style>
        """,
        unsafe_allow_html=True,
    )


def metric_card(label, value):
    st.markdown(
        f"""
<div class="metric-card">
  <div class="metric-label">{label}</div>
  <div class="metric-value">{value}</div>
</div>
        """,
        unsafe_allow_html=True,
    )


def soft_card(title, body):
    st.markdown(
        f"""
<div class="soft-card">
  <div class="metric-label">{title}</div>
  <div class="section-note">{body}</div>
</div>
        """,
        unsafe_allow_html=True,
    )


def render_hero(title, subtitle, eyebrow="VoltSense AI"):
    st.markdown(
        f"""
<div class="hero">
  <div class="eyebrow">{eyebrow}</div>
  <h1 class="hero-title">{title}</h1>
  <div class="hero-subtitle">{subtitle}</div>
</div>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data
def load_predictions():
    if not PREDICTIONS_PATH.exists():
        raise FileNotFoundError(f"Missing predictions file: {PREDICTIONS_PATH}")

    predictions = pd.read_csv(PREDICTIONS_PATH)
    missing_columns = REQUIRED_PREDICTION_COLUMNS - set(predictions.columns)
    if missing_columns:
        raise ValueError(f"predictions_v2.csv is missing columns: {sorted(missing_columns)}")

    numeric_columns = ["cycle_index", "actual_soh", "predicted_soh", "error"]
    for column in numeric_columns:
        predictions[column] = pd.to_numeric(predictions[column], errors="coerce")

    predictions = predictions.dropna(subset=numeric_columns + ["battery_id"]).copy()
    predictions["absolute_error"] = predictions["error"].abs()
    predictions["signed_error"] = predictions["error"]
    return predictions


@st.cache_data
def load_metrics():
    if not METRICS_PATH.exists():
        raise FileNotFoundError(f"Missing metrics file: {METRICS_PATH}")

    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    required_keys = {"best_model_metrics", "best_model", "n_rows", "train_rows", "test_rows"}
    missing_keys = required_keys - set(metrics)
    if missing_keys:
        raise ValueError(f"metrics_v2.json is missing keys: {sorted(missing_keys)}")
    return metrics


@st.cache_data
def load_feature_importance():
    if not FEATURE_IMPORTANCE_PATH.exists():
        return None
    return pd.read_csv(FEATURE_IMPORTANCE_PATH)


@st.cache_resource
def load_model_bundle():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)


def get_importance_columns(feature_importance):
    if feature_importance is None or feature_importance.empty:
        return None, None, None

    lower_map = {column.lower(): column for column in feature_importance.columns}
    feature_column = lower_map.get("feature")
    mean_column = lower_map.get("importance_mean")
    std_column = lower_map.get("importance_std")

    if feature_column is None:
        object_columns = feature_importance.select_dtypes(include="object").columns.tolist()
        feature_column = object_columns[0] if object_columns else feature_importance.columns[0]

    if mean_column is None:
        numeric_columns = [
            column
            for column in feature_importance.columns
            if column != feature_column and pd.api.types.is_numeric_dtype(feature_importance[column])
        ]
        mean_column = numeric_columns[0] if numeric_columns else None

    return feature_column, mean_column, std_column


def status_from_soh(soh):
    if soh >= 0.90:
        return "Excellent", "#22c55e"
    if soh >= 0.80:
        return "Good", "#38bdf8"
    if soh >= 0.70:
        return "Degraded", "#f59e0b"
    return "Critical", "#ef4444"


def format_best_model(metrics):
    if metrics.get("best_model") == "hist_gradient_boosting":
        return "HistGradientBoostingRegressor"
    return str(metrics.get("best_model", "Unknown"))


def make_prediction_plot(battery_predictions, battery_id):
    fig, ax = plt.subplots(figsize=(10, 4.8), facecolor="#0f172a")
    ax.set_facecolor("#0f172a")
    ax.plot(
        battery_predictions["cycle_index"],
        battery_predictions["actual_soh"],
        color="#38bdf8",
        linewidth=2.4,
        marker="o",
        markersize=3.8,
        label="Actual SOH",
    )
    ax.plot(
        battery_predictions["cycle_index"],
        battery_predictions["predicted_soh"],
        color="#22c55e",
        linewidth=2.4,
        linestyle="--",
        marker="s",
        markersize=3.4,
        label="Predicted SOH",
    )
    ax.set_ylim(0.5, 1.2)
    ax.set_xlabel("Cycle index", color="#cbd5e1")
    ax.set_ylabel("State of Health", color="#cbd5e1")
    ax.set_title(f"Actual vs Predicted SOH - {battery_id}", color="#f8fafc", pad=14)
    ax.grid(True, alpha=0.18)
    ax.tick_params(colors="#cbd5e1")
    for spine in ax.spines.values():
        spine.set_color("#334155")
    legend = ax.legend(frameon=True)
    legend.get_frame().set_facecolor("#111827")
    legend.get_frame().set_edgecolor("#334155")
    for text in legend.get_texts():
        text.set_color("#e5e7eb")
    fig.tight_layout()
    return fig


def make_importance_plot(feature_importance, top_n=15):
    feature_column, mean_column, std_column = get_importance_columns(feature_importance)
    if feature_column is None or mean_column is None:
        return None

    plot_data = feature_importance.copy()
    plot_data[mean_column] = pd.to_numeric(plot_data[mean_column], errors="coerce")
    if std_column is not None:
        plot_data[std_column] = pd.to_numeric(plot_data[std_column], errors="coerce")
    plot_data = plot_data.dropna(subset=[mean_column]).sort_values(mean_column, ascending=False)
    plot_data = plot_data.head(top_n).sort_values(mean_column)

    colors = ["#ef4444" if value < 0 else "#38bdf8" for value in plot_data[mean_column]]
    fig_height = max(4.5, 0.38 * len(plot_data))
    fig, ax = plt.subplots(figsize=(10, fig_height), facecolor="#0f172a")
    ax.set_facecolor("#0f172a")
    xerr = plot_data[std_column] if std_column is not None else None
    ax.barh(plot_data[feature_column], plot_data[mean_column], xerr=xerr, color=colors, alpha=0.92)
    ax.axvline(0, color="#94a3b8", linewidth=1)
    ax.set_xlabel("Mean decrease in R2 after permutation", color="#cbd5e1")
    ax.set_ylabel("Feature", color="#cbd5e1")
    ax.set_title("Permutation Feature Importance", color="#f8fafc", pad=14)
    ax.grid(axis="x", alpha=0.18)
    ax.tick_params(colors="#cbd5e1")
    for spine in ax.spines.values():
        spine.set_color("#334155")
    fig.tight_layout()
    return fig


def page_overview(metrics):
    render_hero(
        "VoltSense AI",
        "AI Battery Health Intelligence for Safer and More Sustainable Energy Systems",
        "Battery Health Intelligence",
    )
    st.markdown(
        '<div class="section-note">An AI-powered prototype that estimates battery health from discharge-cycle behavior, built to demonstrate how machine learning can support cleaner, safer, and longer-lasting energy systems.</div>',
        unsafe_allow_html=True,
    )

    st.write("")
    best_metrics = metrics["best_model_metrics"]
    test_battery_count = len(metrics.get("test_batteries", []))
    cols = st.columns(3)
    with cols[0]:
        metric_card("R2 Score", f"{best_metrics['R2']:.6f}")
    with cols[1]:
        metric_card("Average SOH Error", f"{best_metrics['MAE']:.6f}")
    with cols[2]:
        metric_card("Test Batteries", f"{test_battery_count}")

    st.write("")
    model_cols = st.columns(4)
    with model_cols[0]:
        metric_card("MAE", f"{best_metrics['MAE']:.6f}")
    with model_cols[1]:
        metric_card("RMSE", f"{best_metrics['RMSE']:.6f}")
    with model_cols[2]:
        metric_card("Best Model", format_best_model(metrics))
    with model_cols[3]:
        metric_card("Validation", "Held-Out Batteries")

    st.write("")
    data_cols = st.columns(3)
    with data_cols[0]:
        metric_card("Rows Used", f"{metrics['n_rows']:,}")
    with data_cols[1]:
        metric_card("Train Rows", f"{metrics['train_rows']:,}")
    with data_cols[2]:
        metric_card("Held-Out Test Rows", f"{metrics['test_rows']:,}")

    st.write("")
    soh_cols = st.columns([1, 1])
    with soh_cols[0]:
        soft_card(
            "SOH in Simple Terms",
            "State of Health compares current battery capacity with rated capacity. "
            "100% means a fresh battery, 80% is a commonly used degradation threshold, "
            "and lower values mean the battery is aging.",
        )
    with soh_cols[1]:
        soft_card(
            "Prototype Boundary",
            "This dashboard uses saved NASA dataset results. It is a product demo and learning prototype, "
            "not a real-world deployment or safety-certified battery management system.",
        )

    st.write("")
    impact_cols = st.columns(4)
    with impact_cols[0]:
        soft_card("EV Safety", "Earlier degradation insight can support safer inspection and maintenance workflows.")
    with impact_cols[1]:
        soft_card("Clean Storage", "Health prediction can improve reliability for renewable energy storage systems.")
    with impact_cols[2]:
        soft_card("Longer Lifespan", "Better monitoring can help batteries stay useful for longer.")
    with impact_cols[3]:
        soft_card("Less E-Waste", "Avoiding premature replacement can reduce battery waste.")

    st.write("")
    soft_card(
        "WYYFest Vision",
        "VoltSense AI frames this prototype as youth-led AI innovation for battery safety, EVs, "
        "renewable energy, and reducing e-waste. It is intentionally honest about its limits: "
        "trained on NASA data, not validated on real-world BMS fleets, and not deployment ready.",
    )


def page_battery_explorer(predictions):
    render_hero(
        "Battery Explorer",
        "Inspect how the saved Version 2 model tracks actual degradation on held-out test batteries.",
    )

    battery_ids = sorted(predictions["battery_id"].unique())
    selected_battery = st.selectbox("Select held-out test battery", battery_ids)
    battery_predictions = (
        predictions.loc[predictions["battery_id"].eq(selected_battery)]
        .sort_values("cycle_index")
        .copy()
    )

    if battery_predictions.empty:
        st.error("No rows are available for the selected battery.")
        return

    actual = battery_predictions["actual_soh"].to_numpy()
    predicted = battery_predictions["predicted_soh"].to_numpy()
    signed_error = predicted - actual
    absolute_error = np.abs(signed_error)

    metric_cols = st.columns(4)
    with metric_cols[0]:
        metric_card("Battery MAE", f"{absolute_error.mean():.6f}")
    with metric_cols[1]:
        metric_card("Battery RMSE", f"{np.sqrt(np.mean(signed_error ** 2)):.6f}")
    with metric_cols[2]:
        metric_card("Max Error", f"{absolute_error.max():.6f}")
    with metric_cols[3]:
        metric_card("Cycles", f"{len(battery_predictions):,}")

    st.pyplot(make_prediction_plot(battery_predictions, selected_battery), use_container_width=True)
    st.markdown(
        '<div class="section-note">Lower error means the model is tracking degradation more accurately.</div>',
        unsafe_allow_html=True,
    )

    display_table = battery_predictions[
        ["battery_id", "cycle_index", "actual_soh", "predicted_soh", "absolute_error", "signed_error"]
    ].copy()
    for column in ["actual_soh", "predicted_soh", "absolute_error", "signed_error"]:
        display_table[column] = display_table[column].round(6)

    st.subheader("Prediction Detail")
    st.dataframe(display_table, use_container_width=True, hide_index=True)


def page_simulator(metrics):
    render_hero(
        "AI Prediction Simulator",
        "Explore how the saved Version 2 model responds to discharge-cycle feature inputs.",
    )

    model_bundle = load_model_bundle()
    if model_bundle is None:
        st.error(f"Missing model file: {MODEL_PATH}")
        return

    feature_columns = model_bundle.get("feature_columns")
    model = model_bundle.get("model")
    if not feature_columns or model is None:
        st.error("The saved model bundle does not contain the expected model and feature columns.")
        return

    st.markdown(
        """
<div class="warning-box">
The saved Version 2 artifacts do not include the full engineered feature matrix used for training.
This simulator therefore uses conservative prototype slider ranges based on the model's feature names.
It is useful for demonstration, but it is not a calibrated real-world deployment interface.
</div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")
    result_area = st.container()
    st.subheader("Simulation Controls")
    slider_values = {}
    feature_groups = {
        "Temperature Features": [
            feature
            for feature in feature_columns
            if "temperature" in feature or feature == "ambient_temperature"
        ],
        "Voltage Features": [feature for feature in feature_columns if "voltage" in feature],
        "Current Features": [feature for feature in feature_columns if "current" in feature],
        "Cycle/Duration Features": [
            feature
            for feature in feature_columns
            if feature in {"cycle_index", "discharge_duration"}
        ],
    }
    grouped_features = {
        feature for features in feature_groups.values() for feature in features
    }
    other_features = [feature for feature in feature_columns if feature not in grouped_features]
    if other_features:
        feature_groups["Other Model Features"] = other_features

    for section_name, section_features in feature_groups.items():
        if not section_features:
            continue
        with st.expander(section_name, expanded=section_name in {"Temperature Features", "Voltage Features"}):
            columns = st.columns(2)
            for index, feature in enumerate(section_features):
                min_value, default_value, max_value = FALLBACK_FEATURE_RANGES.get(
                    feature, (0.0, 1.0, 10.0)
                )
                step = max((max_value - min_value) / 100.0, 0.001)
                with columns[index % 2]:
                    slider_values[feature] = st.slider(
                        feature,
                        min_value=float(min_value),
                        max_value=float(max_value),
                        value=float(default_value),
                        step=float(step),
                    )

    input_frame = pd.DataFrame([slider_values], columns=feature_columns)
    try:
        predicted_soh = float(model.predict(input_frame)[0])
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
        return

    predicted_soh = float(np.clip(predicted_soh, 0.0, 1.5))
    health_percent = predicted_soh * 100.0
    status, color = status_from_soh(predicted_soh)

    with result_area:
        st.subheader("Live Prediction")
        result_cols = st.columns([1, 1, 1])
        with result_cols[0]:
            metric_card("Predicted SOH", f"{predicted_soh:.3f}")
        with result_cols[1]:
            metric_card("Battery Health", f"{health_percent:.1f}%")
        with result_cols[2]:
            st.markdown(
                f"""
<div class="metric-card">
  <div class="metric-label">Status</div>
  <div class="metric-value" style="color: {color};">{status}</div>
</div>
                """,
                unsafe_allow_html=True,
            )

        progress_value = int(np.clip(predicted_soh, 0.0, 1.0) * 100)
        st.progress(progress_value)
        st.caption(
            "Prototype simulation only. NASA dataset model, not real-world deployment ready "
            "or safety certified."
        )


def page_explainability(feature_importance):
    render_hero(
        "Model Explainability",
        "Understand which discharge-cycle features most influence the Version 2 model on held-out batteries.",
    )

    if feature_importance is None:
        st.error(f"Missing feature importance file: {FEATURE_IMPORTANCE_PATH}")
        return

    feature_column, mean_column, std_column = get_importance_columns(feature_importance)
    if feature_column is None or mean_column is None:
        st.error("Could not identify feature and importance columns in feature_importance_v2.csv.")
        st.dataframe(feature_importance, use_container_width=True)
        return

    importance = feature_importance.copy()
    importance[mean_column] = pd.to_numeric(importance[mean_column], errors="coerce")
    if std_column is not None:
        importance[std_column] = pd.to_numeric(importance[std_column], errors="coerce")
    importance = importance.dropna(subset=[mean_column]).sort_values(mean_column, ascending=False)
    top_features = importance.head(15)

    st.pyplot(make_importance_plot(importance, top_n=15), use_container_width=True)

    if (importance[mean_column] < 0).any():
        st.info(
            "Some features have negative permutation importance. That means the model scored slightly "
            "better when those features were shuffled on this test set, so they may be noisy, redundant, "
            "or not useful for this split."
        )

    st.subheader("Top Feature Table")
    display_columns = [feature_column, mean_column]
    if std_column is not None:
        display_columns.append(std_column)
    st.dataframe(top_features[display_columns], use_container_width=True, hide_index=True)

    st.subheader("Plain-Language Interpretation")
    explanation_cols = st.columns(2)
    with explanation_cols[0]:
        soft_card("temperature_mean", "Thermal behavior signal across the discharge cycle.")
        soft_card("discharge_duration", "How long the battery sustains discharge under test conditions.")
    with explanation_cols[1]:
        soft_card("voltage_load_mean", "Voltage behavior while the battery is under load.")
        soft_card("cycle_index", "Aging progression as the battery moves through repeated cycles.")


def page_impact():
    render_hero(
        "Global Impact & Vision",
        "From battery diagnostics to intelligent energy infrastructure.",
    )

    sections = [
        (
            "Problem",
            "Batteries power EVs, phones, laptops, solar systems, and backup storage, "
            "but degradation is hard to estimate accurately.",
        ),
        (
            "Prototype Solution",
            "This ML prototype predicts SOH from discharge-cycle behavior using saved NASA battery aging results.",
        ),
        (
            "WYYFest Vision",
            "A youth-led AI innovation concept for battery safety, EV reliability, renewable energy storage, "
            "and reducing e-waste through smarter health intelligence.",
        ),
        (
            "Future Product Vision",
            "A Battery Health Intelligence system for battery monitoring, predictive maintenance, "
            "and safer energy systems.",
        ),
        (
            "Potential Customers and Partners",
            "EV manufacturers, battery companies, solar storage providers, e-bike companies, "
            "UPS and backup power companies, and research labs.",
        ),
        (
            "Honest Limitations",
            "Prototype only. Trained on the NASA dataset. Not tested on real Tesla or BYD BMS data. "
            "Needs more datasets, real-world validation, and safety certification before deployment.",
        ),
    ]

    for title, body in sections:
        soft_card(title, body)
        st.write("")


def load_app_data():
    try:
        return load_predictions(), load_metrics(), load_feature_importance()
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
        st.error(str(exc))
        st.stop()


def main():
    st.set_page_config(
        page_title="VoltSense AI",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_css()

    predictions, metrics, feature_importance = load_app_data()

    st.sidebar.markdown("## VoltSense AI")
    st.sidebar.caption("Battery Health Intelligence")
    page = st.sidebar.radio(
        "Navigation",
        [
            "Executive Overview",
            "Battery Explorer",
            "AI Prediction Simulator",
            "Model Explainability",
            "Global Impact & Vision",
        ],
    )
    st.sidebar.markdown("---")
    st.sidebar.caption("Read-only demo. Uses saved Version 2 artifacts only.")

    if page == "Executive Overview":
        page_overview(metrics)
    elif page == "Battery Explorer":
        page_battery_explorer(predictions)
    elif page == "AI Prediction Simulator":
        page_simulator(metrics)
    elif page == "Model Explainability":
        page_explainability(feature_importance)
    else:
        page_impact()


if __name__ == "__main__":
    main()

from pathlib import Path

import pandas as pd

from data_loader import load_discharge_metadata


BASE_METADATA_COLUMNS = [
    "battery_id",
    "test_id",
    "uid",
    "filename",
    "ambient_temperature",
    "cycle_index",
    "Capacity",
    "SOH",
    "file_path",
]


SIGNAL_COLUMNS = {
    "Voltage_measured": "voltage",
    "Current_measured": "current",
    "Temperature_measured": "temperature",
    "Voltage_load": "voltage_load",
    "Current_load": "current_load",
}


def _numeric_stats(frame, column, prefix):
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    if values.empty:
        return {
            f"{prefix}_mean": float("nan"),
            f"{prefix}_min": float("nan"),
            f"{prefix}_max": float("nan"),
            f"{prefix}_std": float("nan"),
        }

    return {
        f"{prefix}_mean": values.mean(),
        f"{prefix}_min": values.min(),
        f"{prefix}_max": values.max(),
        f"{prefix}_std": values.std(ddof=0),
    }


def extract_discharge_features(file_path):
    """Extract simple statistical features from one discharge cycle CSV."""
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Discharge cycle file not found: {file_path}")

    cycle = pd.read_csv(file_path)
    features = {}

    for column, prefix in SIGNAL_COLUMNS.items():
        if column in cycle.columns:
            features.update(_numeric_stats(cycle, column, prefix))

    if "Time" in cycle.columns:
        time_values = pd.to_numeric(cycle["Time"], errors="coerce").dropna()
        features["discharge_duration"] = time_values.max() if not time_values.empty else float("nan")
    else:
        features["discharge_duration"] = float("nan")

    return features


def build_feature_table(metadata_path=None, cycle_data_dir=None, target_strategy="first_valid_capacity"):
    """Build the modeling table by merging metadata and per-cycle features."""
    kwargs = {}
    if metadata_path is not None:
        kwargs["metadata_path"] = metadata_path
    if cycle_data_dir is not None:
        kwargs["cycle_data_dir"] = cycle_data_dir
    kwargs["target_strategy"] = target_strategy

    metadata = load_discharge_metadata(**kwargs)
    rows = []

    for _, row in metadata.iterrows():
        feature_row = {column: row[column] for column in BASE_METADATA_COLUMNS if column in row}
        feature_row.update(extract_discharge_features(row["file_path"]))
        rows.append(feature_row)

    features = pd.DataFrame(rows)
    features["ambient_temperature"] = pd.to_numeric(
        features["ambient_temperature"], errors="coerce"
    )
    numeric_metadata = {"test_id", "uid", "cycle_index", "Capacity", "SOH"}
    non_numeric_columns = {"battery_id", "filename", "file_path"}
    for column in features.columns:
        if column in numeric_metadata or column not in non_numeric_columns:
            features[column] = pd.to_numeric(features[column], errors="coerce")
    return features


def get_feature_columns(feature_table):
    """Return numeric feature columns used for model training."""
    excluded = {
        "battery_id",
        "test_id",
        "uid",
        "filename",
        "Capacity",
        "SOH",
        "file_path",
    }
    return [
        column
        for column in feature_table.columns
        if column not in excluded and pd.api.types.is_numeric_dtype(feature_table[column])
    ]


if __name__ == "__main__":
    table = build_feature_table()
    print(f"Built feature table with shape: {table.shape}")
    print(table.head())

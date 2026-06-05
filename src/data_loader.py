from pathlib import Path
import re

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
METADATA_PATH = PROJECT_ROOT / "data" / "metadata.csv"
CYCLE_DATA_DIR = PROJECT_ROOT / "Nasa" / "cleaned_dataset" / "data"
RATED_CAPACITY_AH = 2.0


def _parse_start_time(value):
    """Parse NASA array-like timestamp strings when possible."""
    if pd.isna(value):
        return pd.NaT

    numbers = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", str(value))
    if len(numbers) < 6:
        return pd.NaT

    try:
        year, month, day, hour, minute = [int(float(x)) for x in numbers[:5]]
        second_float = float(numbers[5])
        second = int(second_float)
        microsecond = int(round((second_float - second) * 1_000_000))
        return pd.Timestamp(
            year=year,
            month=month,
            day=day,
            hour=hour,
            minute=minute,
            second=second,
            microsecond=microsecond,
        )
    except (TypeError, ValueError):
        return pd.NaT


def load_metadata(metadata_path=METADATA_PATH):
    """Load the NASA metadata file."""
    metadata_path = Path(metadata_path)
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    return pd.read_csv(metadata_path)


def load_discharge_metadata(
    metadata_path=METADATA_PATH,
    cycle_data_dir=CYCLE_DATA_DIR,
    target_strategy="first_valid_capacity",
):
    """
    Return one row per valid discharge cycle with SOH target.

    Supported target strategies:
    - first_valid_capacity: SOH = Capacity / first valid discharge Capacity per battery_id
    - rated_capacity: SOH = Capacity / 2.0, filtered to 0.5 <= SOH <= 1.2
    """
    valid_strategies = {"first_valid_capacity", "rated_capacity"}
    if target_strategy not in valid_strategies:
        raise ValueError(
            f"Unknown target_strategy '{target_strategy}'. "
            f"Choose one of: {sorted(valid_strategies)}"
        )

    metadata = load_metadata(metadata_path)
    required_columns = {"type", "battery_id", "test_id", "filename", "Capacity"}
    missing_columns = required_columns - set(metadata.columns)
    if missing_columns:
        raise ValueError(f"metadata.csv is missing columns: {sorted(missing_columns)}")

    discharge = metadata.loc[metadata["type"].eq("discharge")].copy()
    discharge["Capacity"] = pd.to_numeric(discharge["Capacity"], errors="coerce")
    discharge = discharge.dropna(subset=["Capacity", "battery_id", "filename"])
    discharge = discharge.loc[discharge["Capacity"] > 0].copy()

    if discharge.empty:
        raise ValueError("No discharge rows with valid positive Capacity were found.")

    discharge["test_id"] = pd.to_numeric(discharge["test_id"], errors="coerce")
    discharge["uid"] = pd.to_numeric(discharge.get("uid"), errors="coerce")
    discharge["parsed_start_time"] = discharge["start_time"].apply(_parse_start_time)

    sort_columns = ["battery_id", "parsed_start_time", "test_id", "uid"]
    discharge = discharge.sort_values(sort_columns, na_position="last").reset_index(drop=True)
    discharge["cycle_index"] = discharge.groupby("battery_id").cumcount() + 1

    if target_strategy == "first_valid_capacity":
        first_capacity = discharge.groupby("battery_id")["Capacity"].transform("first")
        discharge["SOH"] = discharge["Capacity"] / first_capacity
    else:
        discharge["SOH"] = discharge["Capacity"] / RATED_CAPACITY_AH
        discharge = discharge.loc[discharge["SOH"].between(0.5, 1.2, inclusive="both")].copy()

    if discharge.empty:
        raise ValueError(
            f"No discharge rows remain after applying target_strategy={target_strategy}."
        )

    cycle_data_dir = Path(cycle_data_dir)
    discharge["file_path"] = discharge["filename"].apply(lambda name: cycle_data_dir / str(name))

    return discharge


if __name__ == "__main__":
    df = load_discharge_metadata()
    print(f"Loaded {len(df)} valid discharge cycles from {df['battery_id'].nunique()} batteries.")
    print(df[["battery_id", "cycle_index", "Capacity", "SOH", "file_path"]].head())

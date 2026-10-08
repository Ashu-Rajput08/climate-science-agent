from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import pandas as pd

from .config import PROJECT_ROOT

CITY_DAILY_PATH = PROJECT_ROOT / "india_2000_2024_daily_weather.csv"
DATASET_ID = "india_daily_multicity_2000_2024"


@dataclass
class ClimateData:
    """In-memory view of the active date-only daily city source."""

    hourly: pd.DataFrame
    daily: pd.DataFrame
    metadata: dict
    source: str
    dataset_id: str = DATASET_ID

    def inspect(self) -> dict:
        return {
            "source": self.source,
            "timezone": "not specified",
            # Keep the complete source description available under an explicit key.
            # The sidebar and downstream callers consume this inspect() payload.
            "metadata": self.metadata,
            "location": self.metadata,
            "hourly_rows": 0,
            "daily_rows": int(len(self.daily)),
            "location_count": int(self.metadata.get("location_count", 0)),
            "time_start": self.metadata.get("time_start"),
            "time_end": self.metadata.get("time_end"),
            "columns": list(self.metadata.get("columns", [])),
        }


def load_city_daily_dataset(path: str | Path | None = None) -> tuple[pd.DataFrame, dict]:
    """Read the city CSV without rewriting or normalizing its source fields."""
    source = Path(path or CITY_DAILY_PATH).expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(f"India daily city dataset not found: {source}")
    frame = pd.read_csv(source)
    if "date" not in frame.columns or "city" not in frame.columns:
        raise ValueError("The city daily dataset must contain its observed 'city' and 'date' fields")

    parsed_dates = pd.to_datetime(frame["date"], errors="coerce")
    per_location = {}
    for location, indices in frame.groupby("city", dropna=False, sort=True).groups.items():
        local_dates = parsed_dates.loc[indices].dropna().sort_values()
        per_location[str(location)] = {
            "rows": int(len(indices)),
            "date_start": local_dates.min().date().isoformat() if len(local_dates) else None,
            "date_end": local_dates.max().date().isoformat() if len(local_dates) else None,
            "unique_dates": int(local_dates.nunique()),
            "duplicate_dates": int(local_dates.duplicated().sum()),
            "non_daily_gaps": int(((local_dates.diff().dropna()) != pd.Timedelta(days=1)).sum()),
        }
    metadata = {
        "dataset_id": DATASET_ID,
        "source": str(source),
        "format": "CSV; one row per city and date (as observed)",
        "rows": int(len(frame)),
        "columns": list(frame.columns),
        "shape": [int(len(frame)), int(len(frame.columns))],
        "dtypes": {name: str(dtype) for name, dtype in frame.dtypes.items()},
        "locations": sorted(frame["city"].dropna().astype(str).unique().tolist()),
        "location_field": "city",
        "location_count": int(frame["city"].nunique(dropna=True)),
        "records_by_location": per_location,
        "coordinates": "not provided",
        "time_field": "date",
        "time_semantics": "date-only daily labels; timezone not provided",
        "time_start": parsed_dates.min().date().isoformat() if parsed_dates.notna().any() else None,
        "time_end": parsed_dates.max().date().isoformat() if parsed_dates.notna().any() else None,
        "invalid_date_count": int(parsed_dates.isna().sum()),
        "missing_by_column": {name: int(count) for name, count in frame.isna().sum().items()},
        "duplicate_city_date_rows": int(frame.duplicated(["city", "date"]).sum()),
        "units": "not provided in the file; no unit conversions are registered",
        "provider": "not specified in the file",
        "variables": {
            name: {
                "field": name,
                "dtype": str(frame[name].dtype),
                "missing_count": int(frame[name].isna().sum()),
                "unit": None,
                "description": None,
                "code_legend": "not provided" if name == "weather_code" else None,
            }
            for name in frame.columns if name not in {"city", "date"}
        },
        "notes": [
            "The source CSV is read as-is; this loader does not add, convert, or rename source fields.",
            "Daily city observations are analyzed separately by city when no specific city is named.",
            "Date-only labels are retained without assigning a timezone.",
            "Units and code definitions are not inferred when source documentation is absent.",
        ],
    }
    return frame, metadata


@lru_cache(maxsize=1)
def load_experiment_dataset(dataset_id: str = DATASET_ID) -> ClimateData:
    """Load the single enabled dataset into the analysis container in memory."""
    if dataset_id != DATASET_ID:
        raise ValueError("Only the India-wide daily city dataset is enabled in this app.")
    frame, metadata = load_city_daily_dataset()
    daily = frame.copy()
    # Internal parse alias; retain the original date field and its date-only semantics.
    daily["time_source"] = pd.to_datetime(daily["date"], errors="coerce")
    return ClimateData(
        hourly=pd.DataFrame(), daily=daily, metadata=metadata,
        source=str(CITY_DAILY_PATH.resolve()), dataset_id=dataset_id,
    )


@lru_cache(maxsize=1)
def discover_datasets() -> list[dict]:
    """Return metadata for the only dataset enabled in the application."""
    if not CITY_DAILY_PATH.exists():
        return []
    _, metadata = load_city_daily_dataset()
    metadata["location_dimension"] = True
    return [metadata]


def validate_dataset(data: ClimateData) -> dict:
    if data.dataset_id != DATASET_ID:
        raise ValueError("Only the India-wide daily city dataset is enabled in this app.")
    frame = data.daily
    return {
        "hourly_rows": 0,
        "daily_rows": int(len(frame)),
        "hourly_missing_by_column": {},
        "daily_missing_by_column": {
            key: int(value)
            for key, value in frame.drop(columns=["time_source"], errors="ignore").isna().sum().items()
        },
        "invalid_date_labels": int(frame["time_source"].isna().sum()),
        "physical_range_checks": {},
        "duplicate_city_date_rows": int(data.metadata["duplicate_city_date_rows"]),
        "time_semantics": data.metadata["time_semantics"],
        "units": data.metadata["units"],
        "notes": data.metadata["notes"],
    }

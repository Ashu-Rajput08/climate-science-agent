from __future__ import annotations

import pandas as pd
from scipy import stats

from .data import ClimateData


def daily_frame(data: ClimateData) -> pd.DataFrame:
    """Return source daily records without changing their values or date labels."""
    if data.dataset_id != "india_daily_multicity_2000_2024":
        raise ValueError("Only the India-wide daily city dataset is enabled in this app.")
    return data.daily.copy()


def analysis_frame(data: ClimateData, plan: dict | None = None) -> pd.DataFrame:
    """Daily observations filtered by requested dates and any explicit city selection."""
    frame = daily_frame(data)
    time_range = (plan or {}).get("time_range")
    if time_range:
        start, end = pd.Timestamp(time_range["start"]).date(), pd.Timestamp(time_range["end"]).date()
        local_dates = frame.time_source.dt.date
        frame = frame[(local_dates >= start) & (local_dates <= end)]
    locations = (plan or {}).get("locations") or []
    if locations and "city" in frame:
        frame = frame[frame.city.isin(locations)]
    return frame.copy()


def descriptive(frame: pd.DataFrame, columns: list[str]) -> dict:
    result = {}
    for col in columns:
        if col in frame:
            series = pd.to_numeric(frame[col], errors="coerce").dropna()
            if len(series):
                result[col] = {"n": int(series.size), "mean": float(series.mean()),
                    "median": float(series.median()),
                    "min": float(series.min()), "max": float(series.max()),
                    # Sample standard deviation is undefined for a single value.
                    "std": float(series.std()) if series.size > 1 else None,
                    "p05": float(series.quantile(.05)), "p95": float(series.quantile(.95))}
    return result


def correlation(frame: pd.DataFrame, x: str, y: str) -> dict:
    pair = frame[[x, y]].apply(pd.to_numeric, errors="coerce").dropna()
    if len(pair) < 3:
        raise ValueError("At least three complete paired observations are required")
    if pair[x].nunique() < 2 or pair[y].nunique() < 2:
        raise ValueError("Correlation cannot be calculated because one requested variable is constant in this period")
    pearson, spearman = stats.pearsonr(pair[x], pair[y]), stats.spearmanr(pair[x], pair[y])
    return {"x": x, "y": y, "n": int(len(pair)), "pearson_r": float(pearson.statistic),
        "pearson_p": float(pearson.pvalue), "spearman_rho": float(spearman.statistic),
        "spearman_p": float(spearman.pvalue),
        "interpretation": "Association only; this analysis does not establish causation."}


def seasonal_statistics(frame: pd.DataFrame, value: str) -> dict:
    work = frame[["time_source", value]].dropna().copy()
    months = {12: "Winter", 1: "Winter", 2: "Winter", 3: "Pre-monsoon", 4: "Pre-monsoon",
              5: "Pre-monsoon", 6: "Monsoon", 7: "Monsoon", 8: "Monsoon", 9: "Monsoon",
              10: "Post-monsoon", 11: "Post-monsoon"}
    work["season"] = work.time_source.dt.month.map(months)
    return {str(key): {"n": int(len(group)), "mean": float(group[value].mean()),
        "median": float(group[value].median()),
        "std": float(group[value].std()) if len(group) > 1 else None}
        for key, group in work.groupby("season", sort=False)}


def monthly_statistics(frame: pd.DataFrame, value: str) -> dict:
    work = frame[["time_source", value]].dropna().copy()
    work["month"] = work.time_source.dt.month
    result = {}
    for month, group in work.groupby("month"):
        series = group[value]
        result[pd.Timestamp(2000, int(month), 1).strftime("%B")] = {
            "n": int(len(series)), "mean": float(series.mean()),
            "median": float(series.median()),
            "std": float(series.std()) if len(series) > 1 else None,
        }
    return result


def linear_trend(frame: pd.DataFrame, value: str) -> dict:
    work = frame[["time_source", value]].dropna().copy()
    if len(work) < 3:
        raise ValueError("At least three observations are required for trend analysis")
    # Use elapsed time, not integer datetime storage units (which vary by pandas dtype:
    # seconds, milliseconds, microseconds, or nanoseconds).
    years = (work.time_source - work.time_source.iloc[0]).dt.total_seconds().to_numpy() / (86400 * 365.2425)
    if len(set(years)) < 2:
        raise ValueError("Trend cannot be calculated because all observations have the same date")
    if pd.to_numeric(work[value], errors="coerce").nunique() < 2:
        raise ValueError("Trend cannot be estimated because the requested variable is constant in this period")
    slope, intercept, r, p, error = stats.linregress(years, work[value].astype(float).to_numpy())
    return {"variable": value, "n": int(len(work)), "slope_per_year": float(slope),
        "slope_per_decade": float(slope * 10), "intercept_at_start": float(intercept),
        "r_squared": float(r*r), "p_value": float(p), "slope_standard_error": float(error),
        "method": "ordinary least squares over time",
        "caveat": "The nominal p-value does not account for serial autocorrelation or all climate confounding."}


def anomaly_correlation(frame: pd.DataFrame, x: str, y: str) -> dict:
    work = frame[["time_source", x, y]].dropna().copy()
    work["month_day"] = work.time_source.dt.strftime("%m-%d")
    for col in (x, y):
        work[col + "_anomaly"] = work[col] - work.groupby("month_day")[col].transform("mean")
    result = correlation(work, x + "_anomaly", y + "_anomaly")
    result["method"] = "calendar-day-of-year climatology anomalies"
    result["caveat"] = "Feb 29 and incomplete endpoints may have small climatology samples."
    return result

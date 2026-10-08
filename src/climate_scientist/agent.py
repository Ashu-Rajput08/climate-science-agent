from __future__ import annotations

from datetime import datetime, timezone
from functools import lru_cache
import json
from pathlib import Path
import re
import uuid

import pandas as pd

from .data import (CITY_DAILY_PATH, discover_datasets,
                   load_experiment_dataset, validate_dataset)
from .science import (anomaly_correlation, correlation, descriptive, analysis_frame,
                      linear_trend, monthly_statistics, seasonal_statistics)
from .time_range import parse_time_range
from .visualization import create_analysis_plot

@lru_cache(maxsize=1)
def _available_city_names() -> tuple[str, ...]:
    """Read only the city column once for question routing, not the full CSV each turn."""
    if not CITY_DAILY_PATH.exists():
        return ()
    cities = pd.read_csv(CITY_DAILY_PATH, usecols=["city"])["city"]
    return tuple(sorted(cities.dropna().astype(str).unique()))


def select_source_for_question(question: str) -> tuple[str, list[str], bool]:
    """Route every app question to the India-wide daily city source."""
    if not CITY_DAILY_PATH.exists():
        raise FileNotFoundError(f"India daily city dataset not found: {CITY_DAILY_PATH}")
    q = question.casefold()
    locations = [city for city in _available_city_names()
                 if re.search(rf"\b{re.escape(str(city).casefold())}\b", q)]
    all_cities = bool(re.search(
        r"\b(all cities|across cities|among cities|by city|city-wise|cities|across india|nationwide|region|india)\b", q))
    if not locations:
        all_cities = True
    return "india_daily_multicity_2000_2024", locations, all_cities


def _requests_city_comparison(question: str, locations: list[str] | None = None) -> bool:
    """Recognize comparisons between locations separately from variable correlations."""
    q = question.casefold()
    comparison = re.search(
        r"\b(?:difference|differences|differ|different|compare|comparison|compared|versus|vs)\b", q
    )
    city_dimension = re.search(
        r"\b(?:cities|city-wise|citywise|across india|nationwide|among cities|between cities)\b", q
    )
    return bool(comparison and (city_dimension or len(locations or []) > 1))


def _requested_city_variables(question: str) -> list[str]:
    """Map explicit field mentions to source columns without silently dropping fields."""
    q = question.casefold()
    variables: list[str] = []

    def has(pattern: str) -> bool:
        return re.search(pattern, q) is not None

    apparent = has(r"\b(apparent|feels?[- ]like)\b")
    minimum = has(r"\b(min(?:imum)?|lowest)\b")
    maximum = has(r"\b(max(?:imum)?|highest)\b")
    if apparent:
        if minimum:
            variables.append("apparent_temperature_min")
        if maximum or not minimum:
            variables.append("apparent_temperature_max")
    if has(r"\b(temp(?:erature)?s?)\b") and not apparent:
        if minimum:
            variables.append("temperature_2m_min")
        if maximum or not minimum:
            variables.append("temperature_2m_max")

    if has(r"\b(precip(?:itation)?s?)\b"):
        variables.append("precipitation_sum")
    if has(r"\b(rain(?:fall)?s?)\b"):
        variables.append("rain_sum")
    if has(r"\b(weather|condition)\s+codes?\b|\bweather conditions\b"):
        variables.append("weather_code")

    if has(r"\bwind\s+directions?\b"):
        variables.append("wind_direction_10m_dominant")
    if has(r"\bwind\s+speeds?\b"):
        variables.append("wind_speed_10m_max")
    if has(r"\b(wind\s+)?gusts?\b"):
        variables.append("wind_gusts_10m_max")
    if has(r"\bwind\b") and not any(
        name.startswith("wind_") for name in variables
    ):
        raise ValueError(
            "Please specify wind speed, wind gust, or wind direction; these are separate dataset fields."
        )
    if has(r"\bweather\b") and "weather_code" not in variables and not any(
        name in variables for name in ("temperature_2m_max", "temperature_2m_min",
                                       "apparent_temperature_max", "apparent_temperature_min")
    ):
        variables.append("weather_code")

    # Preserve the source field order and remove duplicates.
    order = ("temperature_2m_max", "temperature_2m_min", "apparent_temperature_max",
             "apparent_temperature_min", "precipitation_sum", "rain_sum", "weather_code",
             "wind_speed_10m_max", "wind_gusts_10m_max", "wind_direction_10m_dominant")
    requested = set(variables)
    return [name for name in order if name in requested]


def interpret_question(question: str, dataset_id: str = "india_daily_multicity_2000_2024",
                       city_comparison: bool = False) -> dict:
    q = question.lower()
    if dataset_id != "india_daily_multicity_2000_2024":
        raise ValueError("Only the India-wide daily city dataset is enabled in this app.")
    city_dataset = True
    if re.search(r"\b(?:uv|ultraviolet)\b", q):
        available = "daily maximum/minimum temperature, apparent temperature, precipitation, rain, weather-code counts, wind speed/gust, and wind direction"
        raise ValueError(f"UV index is not present in the selected source. Available fields include {available}.")
    variables = _requested_city_variables(question)
    unavailable_fields = {
        "humidity": "relative humidity", "pressure": "surface pressure",
        "cloud": "cloud cover", "radiation": "direct solar radiation",
    }
    unavailable = next((label for word, label in unavailable_fields.items()
                        if re.search(rf"\b{word}\b", q)), None)
    if unavailable:
        raise ValueError(
            f"{unavailable.title()} is not included in the India multi-city dataset. "
            "This app uses only the India-wide daily city dataset; it will not substitute another source."
        )
    if city_comparison:
        objective = "city_comparison"
    elif any(term in q for term in ("trend", "long-term", "long term", "over time", "change over time")):
        objective = "trend"
    elif any(term in q for term in ("season", "monsoon", "winter", "summer", "pre-monsoon", "post-monsoon")):
        objective = "seasonal"
    elif "monthly" in q or "by month" in q or "each month" in q:
        objective = "monthly"
    elif "anomal" in q:
        objective = "anomaly_correlation"
    elif any(term in q for term in ("relationship", "correlation", "associated", "relate", "association",
                                    "compare", "versus", " vs ", "impact of", "effect of")):
        objective = "correlation"
    else:
        objective = "descriptive"
    if (city_dataset and objective == "correlation" and len(variables) == 1 and
            re.search(r"\b(compare|comparison|across|among)\b", q) and
            re.search(r"\b(cities|city)\b", q)):
        objective = "descriptive"
    if objective == "city_comparison" and not variables:
        available = "temperature, apparent temperature, precipitation, rain, weather code, wind speed, wind gust, or wind direction"
        raise ValueError(f"Which climate parameter should I compare across cities? Available fields include {available}.")
    if objective in {"correlation", "anomaly_correlation"} and len(variables) < 2:
        raise ValueError("Please name both climate variables to compare (for example, temperature and precipitation).")
    if objective in {"correlation", "anomaly_correlation"} and len(variables) > 2:
        raise ValueError("Please compare two climate variables at a time so each reported association is explicit.")
    if objective in {"trend", "seasonal", "monthly"} and len(variables) > 1:
        raise ValueError("Please ask for a trend, seasonal, or monthly analysis of one variable at a time; this analysis cannot safely summarize multiple fields together.")
    assumptions = []
    source = "deterministic_rules"
    if objective in {"trend", "seasonal", "monthly"} and not variables:
        raise ValueError("Which dataset variable should I analyze? For example: temperature, precipitation, or humidity.")
    if objective == "descriptive" and not variables:
        if any(term in q for term in ("overview", "all variables", "dataset summary")):
            variables = ["temperature_2m_max", "temperature_2m_min", "precipitation_sum"]
        else:
            raise ValueError("I couldn’t identify the variable or analysis. Try asking about temperature, precipitation, humidity, or a trend/seasonal comparison.")
    assumptions += ["Use the supplied location and available observation period."]
    assumptions.append("Interpret source date-only labels on the project calendar (Asia/Kolkata/IST); no timestamp conversion is possible or applied.")
    return {"research_question": question, "domain": "climate science", "variables": variables,
        "objective": objective, "timezone": "Asia/Kolkata (IST calendar dates; source timezone unspecified)", "temporal_resolution": "daily",
        "planner": source, "assumptions": assumptions}


def plan_experiment(question: str, data, locations: list[str] | None = None,
                    all_cities: bool = False) -> dict:
    city_comparison = _requests_city_comparison(question, locations)
    interpretation = interpret_question(question, data.dataset_id, city_comparison)
    time_column = data.daily.time_source
    requested_range = parse_time_range(question, time_column.min().date(), time_column.max().date())
    timezone_label = "Asia/Kolkata (IST calendar dates; source timezone unspecified; no timestamp conversion)"
    return {"experiment_id": "EXP-" + uuid.uuid4().hex[:10].upper(),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "objective": interpretation["objective"], "research_question": question,
        "variables": interpretation["variables"], "dataset": data.source,
        "dataset_id": data.dataset_id, "locations": locations or [],
        "all_cities": all_cities,
        "city_comparison": city_comparison,
        "timezone": timezone_label,
        "time_range": requested_range,
        "preprocessing": ["Use source daily records and date-only labels; do not assign a timezone.",
                          "Retain source field names and values."],
        "methods": [], "refinement_count": 0}


def _run(plan: dict, data) -> dict:
    frame = analysis_frame(data, plan)
    objective, variables = plan["objective"], plan["variables"]
    unavailable = [name for name in variables if name not in frame.columns]
    if unavailable:
        raise ValueError(f"Requested field(s) are not available at the daily analysis resolution: {', '.join(unavailable)}. No substitute field was used.")
    if (data.dataset_id == "india_daily_multicity_2000_2024"
            and objective not in {"descriptive", "city_comparison"}
            and {"weather_code", "wind_direction_10m_dominant"}.intersection(variables)):
        raise ValueError("Weather codes and dominant wind directions are categorical in this source; trend, correlation, and mean-based grouping are unavailable.")
    grouped_city_run = (data.dataset_id == "india_daily_multicity_2000_2024"
                        and "city" in frame
                        and (plan.get("all_cities") or len(plan.get("locations") or []) > 1))
    if grouped_city_run:
        results = {}
        for city, city_frame in frame.groupby("city", sort=True):
            city_plan = dict(plan, locations=[str(city)], all_cities=False)
            results[str(city)] = _run(city_plan, data)
        plan["methods"] = ["Run the selected daily analysis separately for each city; do not pool city records."]
        return {"by_city": results}
    if objective == "trend":
        col = variables[0]
        result = linear_trend(frame, col)
        plan["methods"] = ["ordinary least squares trend"]
    elif objective == "seasonal":
        col = variables[0]
        result = {"variable": col, "seasonal_statistics": seasonal_statistics(frame, col)}
        plan["methods"] = ["source-date meteorological-season grouping"]
    elif objective == "monthly":
        col = variables[0]
        result = {"variable": col, "monthly_statistics": monthly_statistics(frame, col)}
        plan["methods"] = ["monthly climatology grouped by source calendar month"]
    elif objective == "anomaly_correlation":
        result = anomaly_correlation(frame, *variables[:2])
        plan["methods"] = ["calendar-day climatology anomalies", "Pearson and Spearman correlation"]
    elif objective == "correlation":
        result = correlation(frame, *variables[:2])
        plan["methods"] = ["Pearson and Spearman correlation on daily observations"]
    else:
        columns = variables
        result = descriptive(frame, columns)
        if data.dataset_id == "india_daily_multicity_2000_2024":
            for categorical in {"weather_code", "wind_direction_10m_dominant"}.intersection(columns):
                values = frame[categorical].dropna()
                result[categorical] = {"counts": {str(code): int(count)
                    for code, count in values.value_counts().sort_index().items()},
                    "n": int(len(values)),
                    "legend": "not provided" if categorical == "weather_code" else "units/interpretation not provided"}
        plan["methods"] = (["city-wise descriptive statistics; city records summarized separately"]
                            if objective == "city_comparison" else ["descriptive statistics"])
    return result


def _critic(plan: dict, quality: dict) -> dict:
    issues = []
    duplicate_count = quality.get("duplicate_city_date_rows", quality.get("duplicate_utc_timestamps", 0))
    if duplicate_count:
        duplicate_label = "city/date rows" if plan.get("dataset_id") == "india_daily_multicity_2000_2024" else "UTC timestamps"
        issues.append({"severity": "high", "type": "duplicate timestamps",
            "finding": f"{duplicate_count} duplicate {duplicate_label} were found."})
    if any(item["out_of_range_count"] for item in quality["physical_range_checks"].values()):
        issues.append({"severity": "medium", "type": "physical range",
            "finding": "Values outside broad plausibility bounds are retained and should be investigated."})
    if plan["objective"] in {"correlation", "anomaly_correlation"}:
        issues.append({"severity": "medium", "type": "temporal dependence and confounding",
            "finding": "Serial autocorrelation, seasonality, and omitted variables may affect nominal correlation p-values."})
        issues.append({"severity": "low", "type": "causal inference", "finding": "Association does not establish causation."})
    if plan["objective"] == "trend":
        issues.append({"severity": "medium", "type": "trend inference",
            "finding": "The OLS nominal p-value does not adjust for serial autocorrelation, breaks, or confounding."})
    return {"valid": True, "issues": issues, "confidence": 0.78 if issues else 0.9,
        "refinement_recommended": (plan["objective"] == "correlation"
                                   and not plan.get("all_cities")
                                   and len(plan.get("locations") or []) <= 1),
        "decision": "Run a seasonal-anomaly sensitivity analysis for raw correlation." if plan["objective"] == "correlation"
            else "Proceed with stated limitations; the detected inference limitations require scientific interpretation."}


def _report_result_tables(plan: dict, result: dict) -> str:
    """Render analysis outputs as readable tables instead of a raw JSON dump."""
    objective, variables = plan.get("objective"), plan.get("variables", [])
    sections: list[str] = []
    by_city = result.get("by_city")
    if by_city is not None:
        for variable in variables:
            numeric_rows = [(city, values[variable]) for city, values in by_city.items()
                            if variable in values and "mean" in values[variable]]
            if numeric_rows:
                sections.extend([
                    f"### {_display_name(variable)} by city",
                    "| City | Mean | Median | Minimum | Maximum | Standard deviation | Days |\n|---|---:|---:|---:|---:|---:|---:|",
                ])
                for city, metrics in numeric_rows:
                    sections.append(
                        f"| {city} | {_number(metrics['mean'])} | {_number(metrics['median'])} | "
                        f"{_number(metrics['min'])} | {_number(metrics['max'])} | "
                        f"{_number(metrics['std'])} | {metrics['n']:,} |"
                    )
                continue
            categorical_rows = [(city, values[variable].get("counts", {}))
                                for city, values in by_city.items()
                                if variable in values and "counts" in values[variable]]
            if categorical_rows:
                sections.extend([
                    f"### {_display_name(variable)} source values",
                    "| City | Most frequent source values (value: days) |\n|---|---|",
                ])
                for city, counts in categorical_rows:
                    top = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:8]
                    sections.append(f"| {city} | " + ", ".join(f"{code}: {count:,}" for code, count in top) + " |")
        return "\n".join(sections) or "No result table was available for the requested fields."

    if objective in {"seasonal", "monthly"}:
        key = "seasonal_statistics" if objective == "seasonal" else "monthly_statistics"
        groups = result.get(key, {})
        if groups:
            sections.extend(["| Group | Mean | Median | Standard deviation | Days |\n|---|---:|---:|---:|---:|"])
            for name, metrics in groups.items():
                sections.append(f"| {name} | {_number(metrics['mean'])} | {_number(metrics['median'])} | {_number(metrics['std'])} | {metrics['n']:,} |")
    elif objective == "trend":
        sections.extend([
            "| Variable | Change per decade | R² | Nominal p-value | Daily records |\n|---|---:|---:|---:|---:|",
            f"| {_display_name(result.get('variable', variables[0] if variables else 'value'))} | {_number(result.get('slope_per_decade'))} source units | {_number(result.get('r_squared'), 3)} | {_number(result.get('p_value'), 4)} | {result.get('n', 0):,} |",
        ])
    elif objective in {"correlation", "anomaly_correlation"}:
        association = result.get("initial_raw_association", result)
        sections.extend([
            "| Variables | Pearson r | Spearman ρ | Paired days |\n|---|---:|---:|---:|",
            f"| {_display_name(association.get('x', ''))} and {_display_name(association.get('y', ''))} | {_number(association.get('pearson_r'), 3)} | {_number(association.get('spearman_rho'), 3)} | {association.get('n', 0):,} |",
        ])
        adjusted = result.get("seasonality_adjusted_sensitivity")
        if adjusted:
            sections.append("")
            sections.append("Seasonality-adjusted sensitivity check:")
            sections.append("| Pearson r | Spearman ρ | Paired days |\n|---:|---:|---:|")
            sections.append(f"| {_number(adjusted.get('pearson_r'), 3)} | {_number(adjusted.get('spearman_rho'), 3)} | {adjusted.get('n', 0):,} |")
    else:
        numeric_rows = [(variable, metrics) for variable, metrics in result.items()
                        if isinstance(metrics, dict) and "mean" in metrics]
        if numeric_rows:
            sections.extend([
                "| Variable | Mean | Median | Minimum | Maximum | Standard deviation | Days |\n|---|---:|---:|---:|---:|---:|---:|",
            ])
            for variable, metrics in numeric_rows:
                sections.append(f"| {_display_name(variable)} | {_number(metrics['mean'])} | {_number(metrics['median'])} | {_number(metrics['min'])} | {_number(metrics['max'])} | {_number(metrics['std'])} | {metrics['n']:,} |")
        for variable in ("weather_code", "wind_direction_10m_dominant"):
            metrics = result.get(variable, {})
            counts = metrics.get("counts", {}) if isinstance(metrics, dict) else {}
            if counts:
                sections.extend([f"### {_display_name(variable)} source values (no source legend provided)",
                                 "| Source value | Daily records |\n|---|---:|"])
                for value, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:20]:
                    sections.append(f"| {value} | {count:,} |")
    return "\n".join(sections) or "No result table was available for the requested fields."


def _markdown(plan: dict, quality: dict, result: dict, critique: dict) -> str:
    findings = "\n".join(f"- {i['finding']}" for i in critique["issues"])
    if not findings:
        findings = "- No additional limitation was flagged by the automated checks."
    answer = format_chat_answer(plan, quality, result)
    city_names = quality.get("analysis_cities", [])
    city_scope = (
        f"{len(city_names)} cities ({', '.join(city_names)})"
        if plan.get("all_cities") or plan.get("city_comparison")
        else ", ".join(plan.get("locations") or city_names) or "selected dataset records"
    )
    chart_name = Path(quality.get("plot_path", "")).name if quality.get("plot_path") else "not generated"
    return f"""# Climate research report

## Summary

{answer}

## Question and scope

**Question:** {plan['research_question']}

**Dataset:** India-wide daily city records ({plan.get('dataset_id', 'India city dataset')}).  
**Cities:** {city_scope}.  
**Period analyzed:** {quality['analysis_start'][:10]} to {quality['analysis_end'][:10]} (IST calendar labels; source timezone not provided).  
**Daily records analyzed:** {quality.get('analysis_daily_rows', 0):,}.  
**Requested period:** {plan['time_range']['label'] if plan.get('time_range') else 'Full available period'}.

## Detailed results

{_report_result_tables(plan, result)}

## Method and data quality

**Analysis:** {', '.join(plan.get('methods', [])) or plan.get('objective', 'descriptive analysis')}.  
**Duplicate city/date rows:** {quality.get('duplicate_city_date_rows', 0):,}.  
**Invalid date labels:** {quality.get('invalid_date_labels', 0):,}.  
**Source units:** not specified; reported values remain in source units.

## Interpretation and limitations

{findings}

This is observational daily data. Trends and correlations do not establish causes. Source date labels are dates only; the app uses them as calendar dates for the project, without converting timestamps.

## Visualization and provenance

The interactive chart views for this investigation are available in the app’s graph explorer. Static chart file: `{chart_name}`.  
Experiment ID: `{plan['experiment_id']}`. Generated: {plan['created_at_utc']}.
"""


def _strength(value: float) -> str:
    if not pd.notna(value):
        return "an indeterminate"
    magnitude = abs(value)
    label = "weak" if magnitude < 0.3 else "moderate" if magnitude < 0.6 else "strong"
    direction = "positive" if value >= 0 else "negative"
    return f"{label} {direction} association"


def _number(value, precision: int = 2) -> str:
    """Format optional statistics without emitting NaN or raising on n=1 groups."""
    if value is None or not pd.notna(value):
        return "—"
    return f"{value:,.{precision}f}"


def _display_name(variable: str) -> str:
    names = {
        "temperature_2m_max": "Daily maximum temperature",
        "temperature_2m_min": "Daily minimum temperature",
        "apparent_temperature_max": "Daily maximum apparent temperature",
        "apparent_temperature_min": "Daily minimum apparent temperature",
        "precipitation_sum": "Daily precipitation total",
        "rain_sum": "Daily rain total",
        "weather_code": "Weather code",
        "wind_speed_10m_max": "Daily maximum wind speed",
        "wind_gusts_10m_max": "Daily maximum wind gust",
        "wind_direction_10m_dominant": "Dominant wind direction source value",
    }
    return names.get(variable, variable.replace("_", " ").title())


def _append_analysis_scope(lines: list[str], plan: dict, quality: dict) -> None:
    """Add reproducible context so findings are interpretable outside the UI."""
    dataset = plan.get("dataset_id", "climate dataset")
    source_dates = f"{quality.get('analysis_start', '')[:10]} to {quality.get('analysis_end', '')[:10]}"
    observations = quality.get("analysis_daily_rows")
    locations = plan.get("locations") or quality.get("analysis_cities") or []
    if plan.get("all_cities"):
        location_text = "all available cities, analyzed separately"
    elif locations:
        location_text = ", ".join(map(str, locations))
    else:
        location_text = "the selected India daily city records"
    units = "source units not specified"
    lines.extend([
        "", "### Analysis details",
        f"- **Source:** `{dataset}`; {location_text}.",
        f"- **Period analyzed:** {source_dates} ({plan.get('timezone', 'source date labels')}).",
        f"- **Daily records:** {observations:,} across the selected period." if isinstance(observations, int)
        else "- **Daily records:** count unavailable.",
        f"- **Variables:** {', '.join(_display_name(value) for value in plan.get('variables', [])) or 'not identified'} ({units}).",
        f"- **Method:** {', '.join(plan.get('methods', [])) or plan.get('objective', 'descriptive analysis')}.",
    ])
    requested = plan.get("time_range") or {}
    if (requested.get("requested_start") != requested.get("start")
            or requested.get("requested_end") != requested.get("end")):
        lines.append(
            f"- **Coverage note:** The requested interval was {requested.get('requested_start')} to "
            f"{requested.get('requested_end')}, but this dataset only covers {requested.get('start')} to "
            f"{requested.get('end')}; results use the available dates only."
        )


def format_chat_answer(plan: dict, quality: dict, result: dict) -> str:
    """Turn technical outputs into a short answer; the full reproducible report is separate."""
    objective, variables = plan["objective"], plan["variables"]
    lines = ["### Answer"]
    if "by_city" in result:
        lines.append("City records are analyzed separately. For numerical fields, the comparison uses each city’s mean daily value over the selected period.")
        if objective == "city_comparison":
            for variable in variables:
                ranked = [(city, values[variable]) for city, values in result["by_city"].items()
                          if variable in values and "mean" in values[variable]]
                if ranked:
                    ranked.sort(key=lambda item: item[1]["mean"], reverse=True)
                    lines.append(f"**{_display_name(variable)} — city comparison**")
                    lines.append("| Rank | City | Mean daily value | Median | Observed days |\n|---:|---|---:|---:|---:|")
                    for rank, (city, metrics) in enumerate(ranked, start=1):
                        lines.append(f"| {rank} | {city} | {_number(metrics['mean'])} | {_number(metrics['median'])} | {metrics['n']:,} |")
                    high_city, high_metrics = ranked[0]
                    low_city, low_metrics = ranked[-1]
                    difference = high_metrics["mean"] - low_metrics["mean"]
                    lines.append(
                        f"The largest and smallest city means were in **{high_city}** ({_number(high_metrics['mean'])}) "
                        f"and **{low_city}** ({_number(low_metrics['mean'])}), a difference of **{_number(difference)} source units**."
                    )
                elif variable in {"weather_code", "wind_direction_10m_dominant"}:
                    lines.append(f"**{_display_name(variable)} — source-value counts by city** (the file provides no legend):")
                    for city, city_result in result["by_city"].items():
                        counts = city_result.get(variable, {}).get("counts", {})
                        if counts:
                            top = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:5]
                            lines.append(f"- {city}: " + ", ".join(f"{code}: {count:,}" for code, count in top))
            lines.append("Measurement units are not specified in the source file.")
        elif objective == "descriptive":
            numeric_available = any(
                "mean" in metrics
                for city_result in result["by_city"].values()
                for metrics in city_result.values()
            )
            if numeric_available:
                lines.append("| City | Variable | Mean | Median | Std. dev. | n |\n|---|---|---:|---:|---:|---:|")
            for city, city_result in result["by_city"].items():
                for variable, metrics in city_result.items():
                    if "mean" not in metrics:
                        continue
                    lines.append(f"| {city} | {_display_name(variable)} | {_number(metrics['mean'])} | "
                                 f"{_number(metrics['median'])} | {_number(metrics['std'])} | {metrics['n']:,} |")
            for variable in variables:
                ranked = [(city, values[variable]["mean"]) for city, values in result["by_city"].items()
                          if variable in values and "mean" in values[variable]]
                if len(ranked) > 1:
                    high = max(ranked, key=lambda item: item[1])
                    low = min(ranked, key=lambda item: item[1])
                    lines.append(f"For {_display_name(variable)}, {high[0]} had the highest mean ({high[1]:.2f}) "
                                 f"and {low[0]} the lowest ({low[1]:.2f}) among the cities shown.")
            for categorical in ("weather_code", "wind_direction_10m_dominant"):
                if categorical not in variables:
                    continue
                lines.append(f"**{_display_name(categorical)} frequency by city** (codes are not interpreted):")
                for city, city_result in result["by_city"].items():
                    counts = city_result.get(categorical, {}).get("counts", {})
                    if counts:
                        top = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:8]
                        omitted = len(counts) - len(top)
                        summary = ", ".join(f"{code}: {count:,}" for code, count in top)
                        if omitted:
                            summary += f"; {omitted} less frequent values omitted here (see chart/report)"
                        lines.append(f"- {city}: {summary}")
            if plan.get("dataset_id") == "india_daily_multicity_2000_2024":
                lines.append("Units are not stated in the source file.")
        elif objective in {"seasonal", "monthly"}:
            key = "seasonal_statistics" if objective == "seasonal" else "monthly_statistics"
            lines.append("| City | Group | Mean | Median | n |\n|---|---|---:|---:|---:|")
            for city, city_result in result["by_city"].items():
                for group, metrics in city_result.get(key, {}).items():
                    lines.append(f"| {city} | {group} | {_number(metrics['mean'])} | "
                                 f"{_number(metrics['median'])} | {metrics['n']:,} |")
            lines.append("Units are not stated in the source file.")
        elif objective == "trend":
            lines.append("| City | Change per decade (source units) | n |\n|---|---:|---:|")
            for city, city_result in result["by_city"].items():
                lines.append(f"| {city} | {city_result['slope_per_decade']:.3f} | {city_result['n']:,} |")
            lines.append("The source does not specify measurement units.")
        elif objective in {"correlation", "anomaly_correlation"}:
            lines.append("| City | Pearson r | Spearman ρ | Paired days |\n|---|---:|---:|---:|")
            for city, city_result in result["by_city"].items():
                lines.append(f"| {city} | {city_result['pearson_r']:.3f} | "
                             f"{city_result['spearman_rho']:.3f} | {city_result['n']:,} |")
            lines.append("Associations do not establish causation.")
        else:
            for city, city_result in result["by_city"].items():
                lines.append(f"- **{city}:** {json.dumps(city_result, ensure_ascii=False)}")
        lines.append(f"Source date labels: {quality['analysis_start'][:10]} to {quality['analysis_end'][:10]}; interpreted as IST calendar dates (the source has no timezone metadata).")
        _append_analysis_scope(lines, plan, quality)
        return "\n".join(lines)
    if objective in {"seasonal", "monthly"}:
        key = "seasonal_statistics" if objective == "seasonal" else "monthly_statistics"
        stats_by_group = result.get(key, {})
        if stats_by_group:
            best_group, best_stats = max(stats_by_group.items(), key=lambda item: item[1]["mean"])
            value = variables[0]
            display = _display_name(value)
            base_name, _, unit = display.partition("(")
            unit = unit.rstrip(")")
            unit_suffix = f" {unit}" if unit else ""
            unit_heading = f" ({unit})" if unit else ""
            mean_display = f"{best_stats['mean']:.2f}{unit_suffix}"
            lines.append(f"The highest average {base_name.strip().lower()} was in **{best_group}** "
                         f"({mean_display}).")
            low_group, low_stats = min(stats_by_group.items(), key=lambda item: item[1]["mean"])
            if low_group != best_group:
                lines.append(f"The lowest average was in **{low_group}** ({low_stats['mean']:.2f}{unit_suffix}). "
                             f"The difference between these group averages was "
                             f"{best_stats['mean'] - low_stats['mean']:.2f}{unit_suffix}.")
            lines.append(f"| Group | Mean{unit_heading} | Median | Std. dev. | Days |\n|---|---:|---:|---:|---:|")
            for group, metrics in stats_by_group.items():
                lines.append(f"| {group} | {_number(metrics['mean'])} | {_number(metrics['median'])} | "
                             f"{_number(metrics['std'])} | {metrics['n']:,} |")
    elif objective == "trend":
        value = result["slope_per_decade"]
        verb = "increased" if value > 0 else "decreased"
        display = _display_name(variables[0])
        _, has_unit, unit_text = display.partition("(")
        unit = "" if plan.get("dataset_id") == "india_daily_multicity_2000_2024" or not has_unit else f" {unit_text.rstrip(')')}"
        period = "daily source records"
        lines.append(f"{_display_name(variables[0])} {verb} by about **{abs(value):.3f}{unit} per decade** "
                     f"in a linear fit across {result['n']:,} {period}.")
        lines.append(f"The fit explains about {result['r_squared'] * 100:.1f}% of the variation in this linear model "
                     f"(R² = {result['r_squared']:.3f}; nominal p = {result['p_value']:.3g}).")
        lines.append("This is a descriptive trend; serial correlation, breaks, and confounding are not adjusted for, "
                     "so the nominal p-value is not a climate attribution test.")
    elif objective == "correlation":
        initial = result.get("initial_raw_association", result)
        adjusted = result.get("seasonality_adjusted_sensitivity")
        lines.append(f"Daily {_display_name(initial['x'])} and {_display_name(initial['y'])} had a "
                     f"**{_strength(initial['pearson_r'])}** (Pearson r = {initial['pearson_r']:.2f}; "
                     f"{initial['n']:,} paired days).")
        lines.append(f"The rank-based check was Spearman ρ = {initial['spearman_rho']:.2f} "
                     f"(nominal p = {initial['pearson_p']:.3g} for Pearson r).")
        if adjusted:
            lines.append(f"After removing the average seasonal cycle, the association was "
                         f"{_strength(adjusted['pearson_r'])} (Pearson r = {adjusted['pearson_r']:.2f}; "
                         f"Spearman ρ = {adjusted['spearman_rho']:.2f}; n = {adjusted['n']:,}).")
        lines.append("These are associations, not evidence that one variable causes the other.")
    elif objective == "anomaly_correlation":
        lines.append(f"After removing the typical calendar-day pattern, {_display_name(result['x'])} and {_display_name(result['y'])} "
                     f"showed a **{_strength(result['pearson_r'])}** (r = {result['pearson_r']:.2f}; "
                     f"Spearman ρ = {result['spearman_rho']:.2f}; {result['n']:,} paired days). "
                     "This does not establish causation.")
    else:
        numeric_results = {variable: metrics for variable, metrics in result.items() if "mean" in metrics}
        if numeric_results:
            lines.append("Summary statistics for the requested variables:")
            lines.append("| Variable | Mean | Median | Observed min | Observed max | Std. dev. | n |\n|---|---:|---:|---:|---:|---:|---:|")
            for variable, metrics in numeric_results.items():
                lines.append(f"| {_display_name(variable)} | {_number(metrics['mean'])} | {_number(metrics['median'])} | "
                             f"{_number(metrics['min'])} | {_number(metrics['max'])} | "
                             f"{_number(metrics['std'])} | {metrics['n']:,} |")
        for categorical in ("weather_code", "wind_direction_10m_dominant"):
            if categorical in result:
                counts = result[categorical]["counts"]
                top = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:8]
                summary = ", ".join(f"{code}: {count:,}" for code, count in top)
                omitted = len(counts) - len(top)
                suffix = f" {omitted} less frequent values are shown in the chart/report." if omitted else ""
                lines.append(f"{_display_name(categorical)} frequencies (meanings are not inferred): {summary}.{suffix}")

    if plan.get("dataset_id") == "india_daily_multicity_2000_2024":
        lines.append("\n*Dates use source date-only labels interpreted on the project IST calendar; no time conversion was applied. Source timezone and units are unspecified.*")
        if "weather_code" in variables:
            lines.append("Weather-code meanings are not interpreted because this file has no code legend.")
    _append_analysis_scope(lines, plan, quality)
    return "\n".join(lines)


def inspect_and_plan(question: str, data_path=None, dataset_id: str | None = None) -> dict:
    locations, all_cities = [], False
    selected_id, locations, all_cities = select_source_for_question(question)
    if dataset_id and dataset_id != selected_id:
        raise ValueError("This app supports only the India-wide daily city dataset.")
    if data_path is not None:
        raise ValueError("Custom data paths are disabled; this app uses only its registered India-wide daily city dataset.")
    data = load_experiment_dataset(selected_id)
    plan = plan_experiment(question, data, locations, all_cities)
    quality = validate_dataset(data)
    quality["available_datasets"] = discover_datasets()
    dates = data.daily.time_source
    quality["source_start"] = str(dates.min())
    quality["source_end"] = str(dates.max())
    frame = analysis_frame(data, plan)
    if frame.empty:
        raise ValueError("No records are available in the requested date range and city selection.")
    if plan.get("city_comparison") and frame.city.nunique() < 2:
        raise ValueError("I could identify fewer than two cities for this comparison. Name at least two available cities, or ask for a comparison across all cities.")
    quality["analysis_start"] = str(frame.time_source.min())
    quality["analysis_end"] = str(frame.time_source.max())
    quality["analysis_daily_rows"] = int(len(frame))
    quality["analysis_cities"] = sorted(frame.city.unique().tolist()) if "city" in frame else []
    return {"data": data, "plan": plan, "quality": quality}


def execute_experiment(state: dict) -> dict:
    return {"result": _run(state["plan"], state["data"])}


def critique_and_refine(state: dict) -> dict:
    plan, quality, data = state["plan"], state["quality"], state["data"]
    result = state["result"]
    critique = _critic(plan, quality)
    history = [{"stage": "initial", "objective": plan["objective"], "results": result,
                "critic_decision": critique["decision"]}]
    if critique["refinement_recommended"] and len(plan["variables"]) >= 2:
        refined = anomaly_correlation(analysis_frame(data, plan), *plan["variables"][:2])
        plan["refinement_count"] = 1
        plan["methods"].append("refinement: calendar-day climatology anomaly sensitivity analysis")
        result = {"initial_raw_association": result, "seasonality_adjusted_sensitivity": refined}
        history.append({"stage": "refinement", "objective": "anomaly_correlation",
                        "results": refined, "reason": "Critic flagged seasonal confounding risk."})
        critique["decision"] = "Report initial correlation alongside refined seasonal-anomaly sensitivity analysis."
    return {"result": result, "critique": critique, "history": history}


def produce_report(state: dict, output_dir="experiments/runs") -> dict:
    plan, quality = state["plan"], state["quality"]
    result, critique, history = state["result"], state["critique"], state["history"]
    plot_plan = dict(plan)
    if plan["refinement_count"]:
        plot_plan["objective"] = "anomaly_correlation"
    quality["plot_path"] = create_analysis_plot(state["data"], plot_plan,
        Path(output_dir) / f"{plan['experiment_id'].lower()}.png")
    record = {"experiment": plan, "data_quality": quality, "results": result, "critic": critique}
    record["experiment_history"] = history
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = plan["experiment_id"].lower()
    json_path, report_path = out / f"{stem}.json", out / f"{stem}.md"
    json_path.write_text(json.dumps(record, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    report = _markdown(plan, quality, result, critique)
    report_path.write_text(report, encoding="utf-8")
    record.update(answer=format_chat_answer(plan, quality, result), report=report,
                  report_path=str(report_path.resolve()), record_path=str(json_path.resolve()))
    return record


def run_investigation(question: str, data_path=None, output_dir="experiments/runs") -> dict:
    state = inspect_and_plan(question, data_path)
    state.update(execute_experiment(state))
    state.update(critique_and_refine(state))
    return produce_report(state, output_dir)

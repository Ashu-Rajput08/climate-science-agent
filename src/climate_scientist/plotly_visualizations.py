"""Question-specific Plotly figures with filters that respect source date semantics."""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats

from .data import ClimateData
from .science import analysis_frame

MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August",
          "September", "October", "November", "December"]
SEASON_ORDER = ["Winter", "Pre-monsoon", "Monsoon", "Post-monsoon"]
SEASON_MAP = {12: "Winter", 1: "Winter", 2: "Winter", 3: "Pre-monsoon", 4: "Pre-monsoon",
              5: "Pre-monsoon", 6: "Monsoon", 7: "Monsoon", 8: "Monsoon", 9: "Monsoon",
              10: "Post-monsoon", 11: "Post-monsoon"}
PALETTE = px.colors.qualitative.Safe
# The multi-city CSV contains city names but no coordinates. These approximate city
# centers are used only to draw contextual markers, never as scientific observations.
CITY_MAP_CENTERS = {
    "Delhi": (28.6139, 77.2090),
    "Mumbai": (19.0760, 72.8777),
    "Kolkata": (22.5726, 88.3639),
    "Chennai": (13.0827, 80.2707),
    "Bangalore": (12.9716, 77.5946),
    "Hyderabad": (17.3850, 78.4867),
    "Ahmedabad": (23.0225, 72.5714),
    "Pune": (18.5204, 73.8567),
    "Jaipur": (26.9124, 75.7873),
    "Lucknow": (26.8467, 80.9462),
}


def _label(variable: str) -> str:
    labels = {
        "temperature_2m_max": "Daily maximum temperature (source units)",
        "temperature_2m_min": "Daily minimum temperature (source units)",
        "apparent_temperature_max": "Daily maximum apparent temperature (source units)",
        "apparent_temperature_min": "Daily minimum apparent temperature (source units)",
        "precipitation_sum": "Daily precipitation total (source units)",
        "rain_sum": "Daily rain total (source units)",
        "weather_code": "Weather code (source value)",
        "wind_speed_10m_max": "Daily maximum wind speed (source units)",
        "wind_gusts_10m_max": "Daily maximum wind gust (source units)",
        "wind_direction_10m_dominant": "Dominant wind direction (source value)",
    }
    return labels.get(variable, variable.replace("_", " ").title())


def build_coverage_map(sources: list[dict]) -> go.Figure:
    """Offline India-focused coverage map; city markers are approximate centers."""
    fig = go.Figure()
    city_source = next((source for source in sources
                        if source.get("dataset_id") == "india_daily_multicity_2000_2024"), None)
    if city_source:
        cities = [city for city in city_source.get("locations", []) if city in CITY_MAP_CENTERS]
        fig.add_trace(go.Scattergeo(
            lon=[CITY_MAP_CENTERS[city][1] for city in cities],
            lat=[CITY_MAP_CENTERS[city][0] for city in cities],
            text=cities, mode="markers+text", textposition="top center",
            textfont={"size": 9, "color": "#DCE7F7"},
            name="Daily multi-city data",
            hovertemplate="<b>%{text}</b><br>Daily city observations<br>2000–2024<extra></extra>",
            marker={"size": 10, "color": "#67D5C4", "line": {"width": 1, "color": "#10232E"}},
        ))
    fig.update_layout(
        template="plotly_dark", height=350, margin={"l": 8, "r": 8, "t": 8, "b": 8},
        showlegend=True, legend={"orientation": "h", "y": 0.02, "x": 0.02,
                                 "bgcolor": "rgba(16,20,28,.72)"},
        geo={
            "scope": "asia", "projection": {"type": "mercator"},
            "center": {"lat": 22.5, "lon": 79.0},
            "lonaxis": {"range": [66, 92]}, "lataxis": {"range": [5, 38]},
            "showland": True, "landcolor": "#1D2938",
            "showocean": True, "oceancolor": "#111A27",
            "showlakes": True, "lakecolor": "#111A27",
            "showcountries": True, "countrycolor": "#64748B",
            "showcoastlines": True, "coastlinecolor": "#64748B",
            "bgcolor": "rgba(0,0,0,0)",
        },
    )
    return fig


def date_bounds(data: ClimateData, plan: dict | None = None) -> tuple:
    frame = analysis_frame(data, plan)
    return frame.time_source.min().date(), frame.time_source.max().date()


def _filtered_daily(base_frame: pd.DataFrame, date_range=None, selected_months=None,
                    selected_seasons=None) -> pd.DataFrame:
    frame = base_frame.copy()
    if date_range and len(date_range) == 2:
        start, end = date_range
        frame = frame[(frame.time_source.dt.date >= start) & (frame.time_source.dt.date <= end)]
    if selected_months is not None and len(selected_months) < 12:
        month_numbers = [MONTHS.index(month) + 1 for month in selected_months]
        frame = frame[frame.time_source.dt.month.isin(month_numbers)]
    if selected_seasons is not None and len(selected_seasons) < len(SEASON_ORDER):
        frame = frame[frame.time_source.dt.month.map(SEASON_MAP).isin(selected_seasons)]
    return frame.copy()


def _empty_figure(message: str) -> go.Figure:
    figure = go.Figure()
    figure.add_annotation(text=message, x=.5, y=.5, xref="paper", yref="paper", showarrow=False)
    return _style(figure)


def _style(figure: go.Figure, title: str | None = None) -> go.Figure:
    figure.update_layout(template="plotly_dark", title=title or figure.layout.title.text,
        colorway=PALETTE, height=430, margin={"l": 45, "r": 25, "t": 65, "b": 45},
        legend_title_text="")
    figure.update_xaxes(showgrid=True, gridcolor="rgba(150,165,190,.14)")
    figure.update_yaxes(showgrid=True, gridcolor="rgba(150,165,190,.14)")
    return figure


def _association_plots(frame: pd.DataFrame, full_frame: pd.DataFrame,
                       variables: list[str], anomaly: bool) -> list[tuple[str, go.Figure]]:
    if len(variables) < 2 or any(name not in frame for name in variables[:2]):
        return [("Association", _empty_figure("The requested variables are not available for this chart."))]
    x, y = variables[:2]
    raw = frame[["time_source", x, y] + (["city"] if "city" in frame else [])].copy()
    raw[x] = pd.to_numeric(raw[x], errors="coerce")
    raw[y] = pd.to_numeric(raw[y], errors="coerce")
    raw = raw.dropna()
    if len(raw) < 3:
        return [("Association", _empty_figure("Select a wider date range with at least three paired days."))]
    plots: list[tuple[str, go.Figure]] = []
    if not anomaly:
        scatter = px.scatter(raw, x=x, y=y, opacity=.55,
            color="city" if "city" in raw else None,
            color_discrete_sequence=[PALETTE[0]],
            labels={x: _label(x), y: _label(y)},
            title="Daily relationship (one dot per source date)")
        if "city" in raw:
            scatter.update_layout(title="Daily relationship by city")
            for city, group in raw.groupby("city"):
                if group[x].nunique() > 1 and group[y].nunique() > 1:
                    fit = stats.linregress(group[x], group[y])
                    limits = [group[x].min(), group[x].max()]
                    scatter.add_scatter(x=limits, y=[fit.intercept + fit.slope * value for value in limits],
                        mode="lines", name=f"{city} fit")
        elif raw[x].nunique() > 1 and raw[y].nunique() > 1:
            fit = stats.linregress(raw[x], raw[y])
            x_limits = [raw[x].min(), raw[x].max()]
            scatter.add_scatter(x=x_limits, y=[fit.intercept + fit.slope * value for value in x_limits],
                mode="lines", name="Linear fit", line={"color": PALETTE[3], "width": 3})
        plots.append(("Daily relationship", _style(scatter)))

    # Calculate each calendar-day climatology on the full available record before filtering.
    work = raw.copy()
    full = full_frame[["time_source", x, y] + (["city"] if "city" in full_frame else [])].copy()
    full[x] = pd.to_numeric(full[x], errors="coerce")
    full[y] = pd.to_numeric(full[y], errors="coerce")
    full = full.dropna()
    full["month_day"] = full.time_source.dt.strftime("%m-%d")
    group_fields = ["month_day"] + (["city"] if "city" in full else [])
    climatology = full.groupby(group_fields)[[x, y]].mean()
    work["month_day"] = work.time_source.dt.strftime("%m-%d")
    if "city" in work:
        work = work.merge(climatology.reset_index(), on=["month_day", "city"], suffixes=("", "_climatology"))
        for variable in (x, y):
            work[variable + "_anomaly"] = work[variable] - work[variable + "_climatology"]
    else:
        for variable in (x, y):
            work[variable + "_anomaly"] = work[variable] - work.month_day.map(climatology[variable])
    xa, ya = x + "_anomaly", y + "_anomaly"
    anomaly_scatter = px.scatter(work, x=xa, y=ya, opacity=.55,
        color="city" if "city" in work else None,
        color_discrete_sequence=[PALETTE[2]],
        labels={xa: f"Anomaly of {_label(x)}", ya: f"Anomaly of {_label(y)}"},
        title="Relationship after removing the average day-of-year cycle")
    if "city" in work:
        for city, group in work.groupby("city"):
            if group[xa].nunique() > 1 and group[ya].nunique() > 1:
                fit = stats.linregress(group[xa], group[ya])
                limits = [group[xa].min(), group[xa].max()]
                anomaly_scatter.add_scatter(x=limits,
                    y=[fit.intercept + fit.slope * value for value in limits],
                    mode="lines", name=f"{city} fit")
    elif work[xa].nunique() > 1 and work[ya].nunique() > 1:
        fit = stats.linregress(work[xa], work[ya])
        x_limits = [work[xa].min(), work[xa].max()]
        anomaly_scatter.add_scatter(x=x_limits,
            y=[fit.intercept + fit.slope * value for value in x_limits],
            mode="lines", name="Linear fit", line={"color": PALETTE[4], "width": 3})
    plots.append(("Seasonality-adjusted relationship", _style(anomaly_scatter)))
    return plots


def build_charts(data: ClimateData, plan: dict, date_range=None,
                 selected_months=None, selected_seasons=None) -> list[tuple[str, go.Figure]]:
    """Return several complementary, filter-aware charts for a saved experiment."""
    full_frame = analysis_frame(data, plan)
    frame = _filtered_daily(full_frame, date_range, selected_months, selected_seasons)
    if frame.empty:
        return [("No matching data", _empty_figure("No source records match these filters."))]
    objective, variables = plan["objective"], plan["variables"]
    charts: list[tuple[str, go.Figure]] = []

    if objective in {"correlation", "anomaly_correlation"}:
        charts.extend(_association_plots(frame, full_frame, variables, objective == "anomaly_correlation"))
    elif objective == "seasonal":
        variable = variables[0]
        if variable not in frame:
            return [("Seasonal view", _empty_figure("This variable is unavailable in the daily data."))]
        work = frame[["time_source", variable] + (["city"] if "city" in frame else [])].dropna().copy()
        work["Season"] = work.time_source.dt.month.map(SEASON_MAP)
        box = px.box(work, x="Season", y=variable, color="city" if "city" in work else "Season",
                     category_orders={"Season": SEASON_ORDER},
                     labels={variable: _label(variable)},
                     points=False, title=f"{variable.replace('_', ' ')} by season")
        charts.append(("Seasonal distribution", _style(box)))
        group_columns = ["Season"] + (["city"] if "city" in work else [])
        summary = work.assign(Season=work.time_source.dt.month.map(SEASON_MAP)).groupby(group_columns)[variable].agg(["mean", "std"]).reset_index()
        bars = px.bar(summary, x="Season", y="mean", error_y="std", color="city" if "city" in summary else "Season",
                      category_orders={"Season": SEASON_ORDER}, title="Average daily value by season",
                      labels={"mean": _label(variable)})
        charts.append(("Seasonal averages", _style(bars)))
    elif objective == "monthly":
        variable = variables[0]
        if variable not in frame:
            return [("Monthly view", _empty_figure("This variable is unavailable in the daily data."))]
        work = frame[["time_source", variable] + (["city"] if "city" in frame else [])].dropna().copy()
        work["Month"] = work.time_source.dt.month.map(lambda month: MONTHS[month - 1])
        box = px.box(work, x="Month", y=variable, category_orders={"Month": MONTHS},
                     points=False, color="city" if "city" in work else "Month", labels={variable: _label(variable)},
                     title=f"{_label(variable)} by month")
        charts.append(("Monthly distribution", _style(box)))
        group_columns = ["Month"] + (["city"] if "city" in work else [])
        summary = work.groupby(group_columns)[variable].agg(["mean", "std"]).reset_index()
        bars = px.bar(summary, x="Month", y="mean", error_y="std", color="city" if "city" in summary else "Month",
                      category_orders={"Month": MONTHS}, title="Average daily value by month",
                      labels={"mean": _label(variable)})
        charts.append(("Monthly averages", _style(bars)))
    elif objective == "trend":
        variable = variables[0]
        if variable not in frame:
            return [("Trend", _empty_figure("This variable is unavailable in the daily data."))]
        work = frame[["time_source", variable] + (["city"] if "city" in frame else [])].dropna().copy()
        date_label = "Date (source date label)"
        if "city" in work:
            line = px.line(work, x="time_source", y=variable, color="city",
                title=f"Daily {variable.replace('_', ' ')} by city over time",
                labels={"time_source": date_label, variable: _label(variable), "city": "City"})
        else:
            line = px.line(work, x="time_source", y=variable, title=f"Daily {variable.replace('_', ' ')} over time",
                           labels={"time_source": date_label, variable: _label(variable)})
            line.add_scatter(x=work.time_source, y=work[variable].rolling(30, min_periods=15).mean(),
                             mode="lines", name="30-day rolling mean", line={"color": PALETTE[3], "width": 3})
        charts.append(("Trend over time", _style(line)))
        work["Year"] = work.time_source.dt.year
        annual_groups = ["Year"] + (["city"] if "city" in work else [])
        annual = work.groupby(annual_groups)[variable].mean().reset_index()
        charts.append(("Annual means", _style(px.bar(annual, x="Year", y=variable,
            title="Annual means by city",
            color="city" if "city" in annual else "Year", color_continuous_scale="Viridis",
            labels={variable: _label(variable)}))))
    elif objective == "city_comparison":
        categorical = {"weather_code", "wind_direction_10m_dominant"}
        for variable in variables:
            if variable not in frame:
                continue
            if variable in categorical:
                counts = (frame.groupby(["city", variable], dropna=False).size()
                          .rename("Daily records").reset_index())
                top_values = (counts.groupby(variable, as_index=False)["Daily records"].sum()
                              .nlargest(12, "Daily records")[variable])
                counts = counts[counts[variable].isin(top_values)].copy()
                counts[variable] = counts[variable].astype(str)
                figure = px.bar(counts, x="city", y="Daily records", color=variable,
                    barmode="group", title=f"{_label(variable)} counts by city",
                    labels={"city": "City", variable: "Source value"})
                charts.append((f"{_label(variable)} by city", _style(figure)))
                continue
            work = frame[["city", variable]].copy()
            work[variable] = pd.to_numeric(work[variable], errors="coerce")
            work = work.dropna()
            if work.empty:
                continue
            means = work.groupby("city")[variable].mean().sort_values(ascending=False).rename("Mean daily value").reset_index()
            bars = px.bar(means, x="city", y="Mean daily value", color="city",
                title=f"Mean daily {_label(variable)} by city",
                labels={"city": "City", "Mean daily value": _label(variable)})
            charts.append((f"{_label(variable)} · city means", _style(bars)))
            boxes = px.box(work, x="city", y=variable, color="city", points=False,
                title=f"Daily {_label(variable)} distributions by city",
                labels={"city": "City", variable: _label(variable)})
            charts.append((f"{_label(variable)} · distributions", _style(boxes)))
    else:
        variables = [value for value in variables if value in frame]
        if not variables:
            return [("Daily series", _empty_figure("No supported variable was selected."))]
        categorical = {"weather_code", "wind_direction_10m_dominant"}
        for variable in [value for value in variables if value in categorical]:
            group_columns = (["city", variable] if "city" in frame else [variable])
            counts = frame.groupby(group_columns, dropna=False).size().rename("Records").reset_index()
            top_values = (counts.groupby(variable, as_index=False)["Records"].sum()
                          .nlargest(20, "Records")[variable])
            remainder = counts[~counts[variable].isin(top_values)]
            counts = counts[counts[variable].isin(top_values)].copy()
            if not remainder.empty:
                if "city" in remainder:
                    remainder = remainder.groupby("city", as_index=False)["Records"].sum()
                    remainder[variable] = "Other source values"
                else:
                    remainder = pd.DataFrame({variable: ["Other source values"],
                                              "Records": [int(remainder["Records"].sum())]})
                counts = pd.concat([counts, remainder], ignore_index=True)
            counts[variable] = counts[variable].astype(str)
            figure = px.bar(counts, x=variable, y="Records",
                color="city" if "city" in counts else None,
                barmode="group", labels={variable: "Source code/value", "Records": "Daily records"},
                title=f"{_label(variable)} frequencies (source codes are not interpreted)")
            charts.append((f"{_label(variable)} frequencies", _style(figure)))
        variables = [value for value in variables if value not in categorical]
        if len(variables) == 1:
            variable = variables[0]
            charts.append(("Daily time series", _style(px.line(frame, x="time_source", y=variable,
                color="city" if "city" in frame else None,
                title=f"Daily {_label(variable)}", labels={"time_source": "Date (source date label)", variable: _label(variable)}))))
            charts.append(("Value distribution", _style(px.histogram(frame, x=variable, nbins=40,
                title=f"Distribution of {_label(variable)}", color_discrete_sequence=[PALETTE[1]],
                color="city" if "city" in frame else None,
                labels={variable: _label(variable)}))))
        elif len(variables) > 1:
            for index, variable in enumerate(variables):
                charts.append((f"{_label(variable)} · time series", _style(px.line(frame, x="time_source", y=variable,
                    color="city" if "city" in frame else None,
                    title=f"Daily {_label(variable)}", labels={"time_source": "Date (source date label)", variable: _label(variable)},
                    color_discrete_sequence=[PALETTE[index % len(PALETTE)]],))))
                charts.append((f"{_label(variable)} · distribution", _style(px.histogram(frame, x=variable, nbins=40,
                    title=f"Distribution of {_label(variable)}", labels={variable: _label(variable)},
                    color="city" if "city" in frame else None,
                    color_discrete_sequence=[PALETTE[(index + 1) % len(PALETTE)]], opacity=.8))))
    return charts

"""Parse explicit and relative calendar periods from climate questions."""
from __future__ import annotations

from calendar import monthrange
from datetime import date
import re

NUMBER_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
                "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
                "eleven": 11, "twelve": 12, "fifteen": 15, "twenty": 20}
MONTHS = {name.casefold(): index for index, name in enumerate((
    "January", "February", "March", "April", "May", "June", "July", "August",
    "September", "October", "November", "December"), start=1)}
MONTHS.update({name[:3].casefold(): index for index, name in enumerate((
    "January", "February", "March", "April", "May", "June", "July", "August",
    "September", "October", "November", "December"), start=1)})
MONTH_PATTERN = "|".join(sorted(MONTHS, key=len, reverse=True))


def _year(value: str, latest_year: int) -> int:
    number = int(value)
    if len(value) == 4:
        return number
    return (latest_year // 100) * 100 + number if number <= latest_year % 100 else (latest_year // 100 - 1) * 100 + number


def _end_year(value: str, start_year: int, latest_year: int) -> int:
    """Resolve a short range endpoint relative to its start (including future years)."""
    if len(value) == 4:
        return int(value)
    candidate = (start_year // 100) * 100 + int(value)
    if candidate < start_year:
        candidate += 100
    return candidate


def parse_time_range(question: str, available_start: date, available_end: date) -> dict | None:
    """Return a date range only when the question contains a recognizable time scope.

    Calendar years, including labels prefixed with FY, mean January through
    December. Thus FY2020-21 covers January 2020 through December 2021.
    Relative year windows begin on January 1.
    """
    text = question.lower()
    end_year = available_end.year

    iso_dates = re.search(
        r"(\d{4}-\d{2}-\d{2})\s*(?:to|through|until|[-–])\s*(\d{4}-\d{2}-\d{2})",
        text,
    )
    iso_month_range = re.search(
        r"\b(\d{4})-(\d{2})\s*(?:to|through|until|[-–])\s*(\d{4})-(\d{2})\b", text)
    named_month_range = re.search(
        rf"\b({MONTH_PATTERN})\s+(\d{{4}})\s*(?:to|through|until|[-–])\s*"
        rf"({MONTH_PATTERN})\s+(\d{{4}})\b", text, re.I)
    year_to_month = re.search(
        rf"\b(\d{{4}})\s*(?:to|through|until|[-–])\s*({MONTH_PATTERN})\s+(\d{{4}})\b", text, re.I)
    month_to_year = re.search(
        rf"\b({MONTH_PATTERN})\s+(\d{{4}})\s*(?:to|through|until|[-–])\s*(\d{{4}})\b", text, re.I)
    named_date_range = re.search(
        rf"\b({MONTH_PATTERN})\s+(\d{{1,2}})(?:st|nd|rd|th)?[,]?\s+(\d{{4}})\s*"
        rf"(?:to|through|until|[-–])\s*({MONTH_PATTERN})\s+(\d{{1,2}})(?:st|nd|rd|th)?[,]?\s+(\d{{4}})\b",
        text, re.I)
    iso_date = re.search(r"\b(\d{4}-\d{2}-\d{2})\b", text)
    iso_month = re.search(r"\b(\d{4})-(0[1-9]|1[0-2])\b", text)
    named_date = re.search(
        rf"\b({MONTH_PATTERN})\s+(\d{{1,2}})(?:st|nd|rd|th)?[,]?\s+(\d{{4}})\b", text, re.I)
    named_month = re.search(
        rf"\b({MONTH_PATTERN})\s+(\d{{4}})\b|\b(\d{{4}})\s+({MONTH_PATTERN})\b", text, re.I)
    fiscal_range = re.search(r"\bfy\s*(\d{4})\s*[-–/]\s*(\d{2}|\d{4})\b", text, re.I)
    fiscal_year = re.search(r"\bfy\s*\d{4}\b", text, re.I)
    year_range = re.search(
        r"\b(?:(?:from|between)\s+)?(\d{4}|\d{2})\s*(?:to|through|until|and|[-–])\s*(\d{4}|\d{2})\b",
        text,
    )
    single_year = re.search(
        r"\b(?:calendar\s+)?year(?:\s+of)?\s+(\d{4})\b|\b(?:in|during|for)\s+(\d{4})\b",
        text,
    )

    if iso_dates:
        start, end = date.fromisoformat(iso_dates.group(1)), date.fromisoformat(iso_dates.group(2))
        label = f"{start.isoformat()} to {end.isoformat()}"
    elif iso_month_range:
        start_year, start_month, end_year, end_month = map(int, iso_month_range.groups())
        start = date(start_year, start_month, 1)
        end = date(end_year, end_month, monthrange(end_year, end_month)[1])
        label = f"{start.isoformat()} to {end.isoformat()}"
    elif named_month_range:
        start_month_name, start_year, end_month_name, end_year = named_month_range.groups()
        start_month, end_month = MONTHS[start_month_name.casefold()], MONTHS[end_month_name.casefold()]
        start_year, end_year = int(start_year), int(end_year)
        start = date(start_year, start_month, 1)
        end = date(end_year, end_month, monthrange(end_year, end_month)[1])
        label = f"{start.isoformat()} to {end.isoformat()}"
    elif year_to_month:
        start_year, end_month_name, end_year = year_to_month.groups()
        end_year = int(end_year)
        end_month = MONTHS[end_month_name.casefold()]
        start, end = date(int(start_year), 1, 1), date(end_year, end_month, monthrange(end_year, end_month)[1])
        label = f"{start.isoformat()} to {end.isoformat()}"
    elif month_to_year:
        start_month_name, start_year, end_year = month_to_year.groups()
        start_month, start_year, end_year = MONTHS[start_month_name.casefold()], int(start_year), int(end_year)
        start, end = date(start_year, start_month, 1), date(end_year, 12, 31)
        label = f"{start.isoformat()} to {end.isoformat()}"
    elif named_date_range:
        start_month, start_day, start_year, end_month, end_day, end_year = named_date_range.groups()
        start = date(int(start_year), MONTHS[start_month.casefold()], int(start_day))
        end = date(int(end_year), MONTHS[end_month.casefold()], int(end_day))
        label = f"{start.isoformat()} to {end.isoformat()}"
    elif iso_date:
        start = end = date.fromisoformat(iso_date.group(1))
        label = start.isoformat()
    elif named_date:
        month_name, day, year = named_date.groups()
        start = end = date(int(year), MONTHS[month_name.casefold()], int(day))
        label = start.isoformat()
    elif iso_month:
        year, month = map(int, iso_month.groups())
        start = date(year, month, 1)
        end = date(year, month, monthrange(year, month)[1])
        label = f"{start.isoformat()} to {end.isoformat()}"
    elif named_month:
        month_prefix, year_after_month, year_prefix, month_suffix = named_month.groups()
        if month_prefix:
            month_name, year = month_prefix, int(year_after_month)
        else:
            month_name, year = month_suffix, int(year_prefix)
        month = MONTHS[month_name.casefold()]
        start, end = date(year, month, 1), date(year, month, monthrange(year, month)[1])
        label = f"{start.isoformat()} to {end.isoformat()}"
    elif fiscal_range:
        start_year = int(fiscal_range.group(1))
        fiscal_end_year = _end_year(fiscal_range.group(2), start_year, end_year)
        if fiscal_end_year < start_year:
            raise ValueError("The ending year must not be before the starting year (for example, FY2020-21).")
        start, end = date(start_year, 1, 1), date(fiscal_end_year, 12, 31)
        label = f"calendar years {start_year} to {fiscal_end_year} (requested as FY{start_year}-{fiscal_range.group(2)})"
    elif fiscal_year:
        year = int(re.search(r"\d{4}", fiscal_year.group(0)).group(0))
        start, end = date(year, 1, 1), date(year, 12, 31)
        label = f"calendar year {year} (requested as FY{year})"
    elif year_range:
        start_year = _year(year_range.group(1), end_year)
        end_year_explicit = _end_year(year_range.group(2), start_year, end_year)
        start, end = date(start_year, 1, 1), date(end_year_explicit, 12, 31)
        label = f"{start_year} to {end_year_explicit}"
    elif single_year:
        year = int(next(value for value in single_year.groups() if value))
        start, end = date(year, 1, 1), date(year, 12, 31)
        label = f"calendar year {year}"
    else:
        since = re.search(r"\b(?:since|from)\s+(\d{4}|\d{2})\b", text)
        if since:
            start_year = _year(since.group(1), end_year)
            start, end = date(start_year, 1, 1), available_end
            label = f"since {start_year}"
        else:
            relative = re.search(
                r"\b(?:past|last|previous|preceding)\s+(\d+|[a-z]+)\s+(years?|yrs?)\b",
                text,
            )
            decade = re.search(r"\b(?:past|last|previous)\s+decade\b", text)
            if decade:
                count = 10
            elif relative:
                raw = relative.group(1)
                count = int(raw) if raw.isdigit() else NUMBER_WORDS.get(raw)
            else:
                count = None
            if count is None:
                return None
            start_year = end_year - count
            start, end = date(start_year, 1, 1), available_end
            label = f"past {count} years ({start_year} to {available_end.year})"

    requested_start, requested_end = start, end
    # Respect the dataset's actual limits. The right endpoint is inclusive.
    start = max(start, available_start)
    end = min(end, available_end)
    if start > end:
        raise ValueError(f"The requested period is outside this dataset ({available_start} to {available_end}).")
    return {"start": start.isoformat(), "end": end.isoformat(), "label": label,
            "requested_start": requested_start.isoformat(),
            "requested_end": requested_end.isoformat()}

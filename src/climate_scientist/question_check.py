"""Conservative typo suggestions for climate questions; never silently rewrites input."""
from difflib import get_close_matches
import re

TERMS = {
    "temperature", "precipitation", "rainfall", "rain", "humidity", "pressure", "wind",
    "cloud", "season", "seasonal", "monthly", "month", "trend", "relationship",
    "correlation", "associated", "association", "anomaly", "anomalies", "compare",
    "comparison", "average", "mean", "median", "maximum", "minimum", "increase",
    "decrease", "change", "historical", "daily", "hourly", "yearly", "year", "month",
    "data", "dataset", "show", "plot", "graph", "analyze", "analyse", "analysis", "does",
    "vary", "across", "between", "during", "with", "from", "over", "how", "what", "which",
    "the", "and", "for", "is", "are", "in", "on", "of", "to", "by", "my", "this", "region",
    "uv", "index", "radiation", "sunlight", "surface", "speed", "direction", "monsoon",
    "seasons", "trends", "months", "years", "temperatures", "graphs", "charts", "variables",
    "values", "relationships", "rainy", "rains", "precipitations",
    "minimum", "maximum", "highest", "lowest", "solar", "increase", "decrease",
    "pre", "post", "winter", "summer", "spring", "autumn", "february", "january", "march",
    "april", "may", "june", "july", "august", "september", "october", "november", "december",
}


def suggest_corrected_question(question: str) -> str | None:
    """Suggest likely domain-word corrections only; the UI must ask before using them."""
    replacements: dict[str, str] = {}
    for match in re.finditer(r"[A-Za-z]+", question):
        token = match.group(0)
        lower = token.lower()
        if lower in TERMS or len(lower) < 5:
            continue
        options = get_close_matches(lower, sorted(TERMS), n=1, cutoff=0.86)
        if options and options[0] != lower:
            replacements[token] = options[0]
    if not replacements:
        return None

    def replace(match: re.Match) -> str:
        old = match.group(0)
        new = replacements.get(old)
        if not new:
            return old
        return new.capitalize() if old[:1].isupper() else new

    corrected = re.sub(r"[A-Za-z]+", replace, question)
    return corrected if corrected.casefold() != question.casefold() else None

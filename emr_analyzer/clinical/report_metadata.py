"""Deterministic report metadata, kept separate from clinical event dates."""
from datetime import date
import re

_HEADER = re.compile(r"\A<!-- emr-report-date:v1 -->\n.*?\n<!-- /emr-report-date -->\n\n", re.S)


def report_date(value):
    """Accept only a complete ISO date; never infer dates from arbitrary text."""
    if not isinstance(value, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        return None
    try:
        return date.fromisoformat(value).isoformat()
    except ValueError:
        return None


def clinical_body(text):
    """Remove only the application-owned metadata header."""
    return _HEADER.sub("", text, count=1)


def with_report_date(text, value):
    body = clinical_body(text)
    value = report_date(value) or "non disponibile"
    return (
        "<!-- emr-report-date:v1 -->\n"
        f"Data del referto (metadato document_date): {value}\n"
        "Questa data identifica il referto, non necessariamente gli eventi clinici descritti.\n"
        "<!-- /emr-report-date -->\n\n" + body
    )

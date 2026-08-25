"""Conservative temporal normalization for Italian clinical evidence."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
import calendar
import re
from typing import Optional

from ..utils.date_utils import parse_italian_date


@dataclass(frozen=True, slots=True)
class NormalizedClinicalDate:
    start: Optional[str]
    end: Optional[str]
    precision: str
    source: str
    original_text: str = ""
    approximate: bool = False


_MONTHS = {
    "gennaio": 1, "febbraio": 2, "marzo": 3, "aprile": 4,
    "maggio": 5, "giugno": 6, "luglio": 7, "agosto": 8,
    "settembre": 9, "ottobre": 10, "novembre": 11, "dicembre": 12,
}

_NUMBER_WORDS = {
    "un": 1, "uno": 1, "una": 1, "due": 2, "tre": 3, "quattro": 4,
    "cinque": 5, "sei": 6, "sette": 7, "otto": 8, "nove": 9,
    "dieci": 10, "undici": 11, "dodici": 12,
}


def date_sort_key(value: str | None) -> tuple[int, int, int]:
    """Sortable lower bound for partial ISO dates; unknown sorts last."""
    text = str(value or "")
    match = re.fullmatch(r"(\d{4})(?:-(\d{2}))?(?:-(\d{2}))?", text)
    if not match:
        return (9999, 12, 31)
    return (
        int(match.group(1)), int(match.group(2) or 1),
        int(match.group(3) or 1),
    )


def precision_for_iso(value: str | None) -> str:
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", str(value or "")):
        return "day"
    if re.fullmatch(r"\d{4}-\d{2}", str(value or "")):
        return "month"
    if re.fullmatch(r"\d{4}", str(value or "")):
        return "year"
    return "unknown"


def normalize_clinical_date(
    value: object,
    *,
    document_date: str | None = None,
    explicit_precision: str | None = None,
    date_end: object = None,
) -> NormalizedClinicalDate:
    """Normalize exact, partial, interval and approximate clinical dates.

    An absent/unparseable event date remains unknown. ``document_date`` is
    used only to resolve explicit retrospective durations; callers retain it
    separately as the date of documentation.
    """
    original = " ".join(str(value or "").strip().split())
    lowered = original.lower()
    approximate = bool(re.search(
        r"\b(?:circa|approssimativamente|verosimilmente|intorno|da\s+anni)\b",
        lowered,
    ))

    start, precision = _parse_single_date(original)
    end, end_precision = _parse_single_date(str(date_end or "").strip())

    # Clinical notes often omit the year for dates inside a document. Resolve
    # only explicit Italian day+month expressions against the report year;
    # never assign the report date to a fact lacking its own date expression.
    reference_year = re.match(r"((?:19|20)\d{2})", str(document_date or ""))
    compact_interval = re.fullmatch(
        r"(?i)dal\s+(\d{1,2})\s+al\s+(\d{1,2})\s+"
        r"(gennaio|febbraio|marzo|aprile|maggio|giugno|luglio|agosto|"
        r"settembre|ottobre|novembre|dicembre)(?:\s+((?:19|20)\d{2}))?",
        original,
    )
    if compact_interval:
        year = compact_interval.group(4) or (
            reference_year.group(1) if reference_year else None
        )
        if year:
            start, _ = _parse_single_date(
                f"{compact_interval.group(1)} {compact_interval.group(3)} {year}"
            )
            end, _ = _parse_single_date(
                f"{compact_interval.group(2)} {compact_interval.group(3)} {year}"
            )
            precision = end_precision = "interval"
    if start is None and reference_year and re.fullmatch(
        r"(?i)(?:il\s+|dal\s+|del\s+)?\d{1,2}\s+"
        r"(?:gennaio|febbraio|marzo|aprile|maggio|giugno|luglio|agosto|"
        r"settembre|ottobre|novembre|dicembre)",
        original.strip(),
    ):
        start, precision = _parse_single_date(
            f"{original} {reference_year.group(1)}"
        )
    raw_end = str(date_end or "").strip()
    if end is None and reference_year and re.fullmatch(
        r"(?i)(?:il\s+|al\s+|del\s+)?\d{1,2}\s+"
        r"(?:gennaio|febbraio|marzo|aprile|maggio|giugno|luglio|agosto|"
        r"settembre|ottobre|novembre|dicembre)",
        raw_end,
    ):
        end, end_precision = _parse_single_date(
            f"{raw_end} {reference_year.group(1)}"
        )

    if start is None and original:
        interval = _parse_interval(original)
        if interval:
            start, end = interval
            precision = "interval"

    if start is not None:
        if explicit_precision in {
            "day", "month", "year", "interval", "approximate"
        }:
            precision = explicit_precision
        if approximate and precision != "interval":
            precision = "approximate"
        if end is not None and end_precision != "unknown":
            precision = "interval"
        return NormalizedClinicalDate(
            start=start, end=end, precision=precision,
            source="explicit_or_retroactive", original_text=original,
            approximate=approximate,
        )

    relative = re.search(
        r"(?i)\b(?:da(?:\s+circa)?|circa\s+da)\s+"
        r"(\d+|un|uno|una|due|tre|quattro|cinque|sei|sette|otto|nove|"
        r"dieci|undici|dodici)\s+"
        r"(giorn(?:o|i)|settiman(?:a|e)|mes(?:e|i)|ann(?:o|i))\b",
        original,
    )
    if relative:
        raw_quantity = relative.group(1).lower()
        quantity = (
            int(raw_quantity) if raw_quantity.isdigit()
            else _NUMBER_WORDS[raw_quantity]
        )
        retroactive = subtract_duration(
            document_date, quantity, relative.group(2).lower()
        )
        if retroactive:
            return NormalizedClinicalDate(
                start=retroactive, end=None, precision="approximate",
                source="retrospective_duration", original_text=original,
                approximate=True,
            )

    return NormalizedClinicalDate(
        start=None, end=None, precision="unknown", source="unknown",
        original_text=original, approximate=approximate,
    )


def temporal_distance_days(
    first: str | None, second: str | None
) -> int | None:
    """Conservative distance between partial dates using interval bounds."""
    first_bounds = date_bounds(first)
    second_bounds = date_bounds(second)
    if not first_bounds or not second_bounds:
        return None
    a_start, a_end = first_bounds
    b_start, b_end = second_bounds
    if a_start <= b_end and b_start <= a_end:
        return 0
    if a_end < b_start:
        return (b_start - a_end).days
    return (a_start - b_end).days


def date_bounds(value: str | None) -> tuple[date, date] | None:
    text = str(value or "")
    try:
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", text):
            parsed = datetime.strptime(text, "%Y-%m-%d").date()
            return parsed, parsed
        if re.fullmatch(r"\d{4}-\d{2}", text):
            year, month = map(int, text.split("-"))
            last = calendar.monthrange(year, month)[1]
            return date(year, month, 1), date(year, month, last)
        if re.fullmatch(r"\d{4}", text):
            year = int(text)
            return date(year, 1, 1), date(year, 12, 31)
    except ValueError:
        return None
    return None


def subtract_duration(
    reference_date: str | None, quantity: int, unit: str
) -> str | None:
    """Resolve a simple retrospective duration against an exact date."""
    bounds = date_bounds(reference_date)
    if not bounds or bounds[0] != bounds[1] or quantity < 0:
        return None
    reference = bounds[0]
    if unit.startswith("giorn"):
        return (reference - timedelta(days=quantity)).isoformat()
    if unit.startswith("settiman"):
        return (reference - timedelta(days=quantity * 7)).isoformat()
    if unit.startswith("mes"):
        total = reference.year * 12 + reference.month - 1 - quantity
        year, month0 = divmod(total, 12)
        day = min(reference.day, calendar.monthrange(year, month0 + 1)[1])
        return date(year, month0 + 1, day).isoformat()
    if unit.startswith("ann"):
        try:
            return reference.replace(year=reference.year - quantity).isoformat()
        except ValueError:
            return reference.replace(
                year=reference.year - quantity, day=28
            ).isoformat()
    return None


def _parse_single_date(value: str) -> tuple[str | None, str]:
    text = " ".join(str(value or "").strip().split())
    if not text:
        return None, "unknown"
    cleaned = re.sub(
        r"(?i)\b(?:circa|dal|dalla|da|il|nel|nell'|in\s+data|"
        r"approssimativamente|intorno\s+al)\b", " ", text,
    ).strip(" ,;:")
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", cleaned):
        try:
            datetime.strptime(cleaned, "%Y-%m-%d")
            return cleaned, "day"
        except ValueError:
            return None, "unknown"
    match = re.fullmatch(r"(\d{4})-(\d{1,2})", cleaned)
    if match:
        year, month = map(int, match.groups())
        if 1 <= month <= 12:
            return f"{year:04d}-{month:02d}", "month"
    if re.fullmatch(r"(?:19|20)\d{2}", cleaned):
        return cleaned, "year"
    parsed = parse_italian_date(cleaned)
    if parsed:
        return parsed, "day"
    match = re.fullmatch(
        r"(?i)(gennaio|febbraio|marzo|aprile|maggio|giugno|luglio|"
        r"agosto|settembre|ottobre|novembre|dicembre)\s+((?:19|20)\d{2})",
        cleaned,
    )
    if match:
        return f"{int(match.group(2)):04d}-{_MONTHS[match.group(1).lower()]:02d}", "month"
    return None, "unknown"


def _parse_interval(value: str) -> tuple[str, str] | None:
    match = re.search(
        r"(?i)(?:tra|dal)\s+(.+?)\s+(?:e|al)\s+(.+)$", value.strip()
    )
    if not match:
        return None
    start, _ = _parse_single_date(match.group(1))
    end, _ = _parse_single_date(match.group(2))
    return (start, end) if start and end else None

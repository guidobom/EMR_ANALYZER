"""Italian date parsing utilities."""

import re
from datetime import datetime
from typing import Optional

# Italian month names
ITALIAN_MONTHS = {
    "gennaio": 1, "febbraio": 2, "marzo": 3, "aprile": 4,
    "maggio": 5, "giugno": 6, "luglio": 7, "agosto": 8,
    "settembre": 9, "ottobre": 10, "novembre": 11, "dicembre": 12,
}

# Common Italian date formats
DATE_PATTERNS = [
    # 12/04/2026, 12-04-2026, 12.04.2026 (Italian dot format)
    (re.compile(r'(\d{1,2})[/\-.](\d{1,2})[/\-.](\d{2,4})'), "DMY"),
    # 2026-04-12, 2026/04/12, 2026.04.12
    (re.compile(r'(\d{4})[/\-.](\d{1,2})[/\-.](\d{1,2})'), "YMD"),
    # 12 aprile 2026, 12 Aprile 2026
    (re.compile(r'(\d{1,2})\s+(gennaio|febbraio|marzo|aprile|maggio|'
                r'giugno|luglio|agosto|settembre|ottobre|novembre|dicembre)'
                r'\s+(\d{4})', re.IGNORECASE), "DMY_TEXT"),
    # aprile 12, 2026
    (re.compile(r'(gennaio|febbraio|marzo|aprile|maggio|'
                r'giugno|luglio|agosto|settembre|ottobre|novembre|dicembre)'
                r'\s+(\d{1,2}),?\s+(\d{4})', re.IGNORECASE), "MDY_TEXT"),
]


def parse_italian_date(text: str) -> Optional[str]:
    """
    Try to parse an Italian date string.
    Returns ISO format YYYY-MM-DD or None.
    """
    if not text:
        return None

    text = text.strip()

    for pattern, date_type in DATE_PATTERNS:
        match = pattern.search(text)
        if match:
            if date_type == "DMY":
                d, m, y = int(match.group(1)), int(match.group(2)), int(normalize_year(match.group(3)))
            elif date_type == "YMD":
                d, m, y = int(match.group(3)), int(match.group(2)), int(match.group(1))
            elif date_type == "DMY_TEXT":
                d = int(match.group(1))
                m = ITALIAN_MONTHS.get(match.group(2).lower(), 1)
                y = int(match.group(3))
            elif date_type == "MDY_TEXT":
                d = int(match.group(2))
                m = ITALIAN_MONTHS.get(match.group(1).lower(), 1)
                y = int(match.group(3))
            else:
                continue

            try:
                dt = datetime(y, m, d)
                if dt.year < 1900 or dt.year > 2100:
                    continue
                return dt.strftime("%Y-%m-%d")
            except ValueError:
                continue

    return None


def normalize_year(year_str: str) -> str:
    """Normalize 2-digit year to 4-digit."""
    y = int(year_str)
    if y < 100:
        return f"{2000 + y}" if y < 50 else f"{1900 + y}"
    return str(y)


def is_valid_date(date_str: str) -> bool:
    """Check if a string is a valid ISO date."""
    try:
        dt = datetime.fromisoformat(date_str)
        return 1900 <= dt.year <= 2100
    except (ValueError, TypeError):
        return False


def compare_dates(date1: Optional[str], date2: Optional[str]) -> int:
    """
    Compare two ISO date strings.
    Returns -1 if date1 < date2, 0 if equal, 1 if date1 > date2.
    """
    if not date1 and not date2:
        return 0
    if not date1:
        return 1
    if not date2:
        return -1

    try:
        d1 = datetime.fromisoformat(date1)
        d2 = datetime.fromisoformat(date2)
    except (ValueError, TypeError):
        return 0

    if d1 < d2:
        return -1
    elif d1 > d2:
        return 1
    return 0

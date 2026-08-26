"""Deterministic chronological registry serialized from atomic evidence.

The timeline is built ONLY by code (date resolution, deduplication, sorting):
an LLM never reconstructs chronology from raw reports.  The compact textual
form is what an LLM receives for interrogation, with source references that
allow every answer to be verified against the original document.

Ordering rules:
- entries sort by resolved date (day > month > year precision, unknown last)
- identical facts repeated across documents collapse into one entry
  (deduplicate_atomic_evidence); sources remain traceable via the entry ref
- dates with month/year precision or approximate markers stay, marked, and
  are never discarded
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re

from .atomic_evidence import deduplicate_atomic_evidence
from .temporal import date_sort_key

_FACT_TYPE_LABELS = {
    "laboratory_test": "lab",
    "hospitalization": "ricovero",
    "discharge": "dimissione",
    "medication": "terapia",
    "procedure": "procedura",
    "radiology_finding": "imaging",
    "instrumental_finding": "imaging",
    "diagnosis": "diagnosi",
    "symptom": "sintomo",
    "clinical_sign": "segno",
    "vital_sign": "vitale",
    "biomarker": "biomarcatore",
    "histopathology": "istologia",
    "clinical_decision": "decisione",
}

# Intents from the clinical query service mapped to timeline fact types.
INTENT_TYPES = {
    "oncology": {
        "medication", "diagnosis", "biomarker", "radiology_finding",
        "instrumental_finding",
    },
    "therapy": {"medication", "procedure"},
    "toxicity": {"diagnosis"},
    "laboratory": {"laboratory_test", "biomarker"},
    "respiratory": {
        "symptom", "clinical_sign", "vital_sign", "radiology_finding",
        "instrumental_finding", "laboratory_test",
    },
    "active": {"diagnosis", "symptom", "clinical_sign", "medication"},
}


@dataclass(frozen=True)
class TimelineEntry:
    """One fixed-schema row of the chronological registry."""

    date: str                 # ISO: YYYY-MM-DD, YYYY-MM or YYYY
    date_precision: str       # day | month | year | unknown
    fact_type: str
    description: str
    value: float | None = None
    unit: str | None = None
    document_id: str = ""
    abnormal: bool = False
    polarity: str = "present"
    approximate: bool = False
    source_text: str = ""
    source_occurrences: tuple = field(default_factory=tuple)


def build_timeline(evidence, *, deduplicate: bool = True) -> list[TimelineEntry]:
    """Build the deterministic chronological registry for one patient."""
    items = (
        deduplicate_atomic_evidence(list(evidence))
        if deduplicate else list(evidence)
    )
    entries = [_entry_for(item) for item in items]
    entries = [entry for entry in entries if entry is not None]
    entries.sort(key=_entry_sort_key)
    return entries


def filter_entries(
    entries: list[TimelineEntry],
    *,
    types: set[str] | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
) -> list[TimelineEntry]:
    """Deterministic pre-query filtering on the rigid schema."""
    filtered = entries
    if types:
        filtered = [e for e in filtered if e.fact_type in types]
    if date_from:
        filtered = [e for e in filtered if date_sort_key(e.date) >=
                    date_sort_key(date_from)]
    if date_to:
        filtered = [e for e in filtered if date_sort_key(e.date) <=
                    date_sort_key(date_to)]
    return filtered


def format_compact(
    entries: list[TimelineEntry],
    *,
    limit: int | None = None,
    max_chars: int | None = None,
) -> str:
    """Compact chronological text for LLM interrogation.

    One line per entry: ``2023-03-14 [lab] Creatinina 2.1 mg/dL (rif: DOC_01)``
    """
    lines = []
    size = 0
    for entry in entries:
        if limit is not None and len(lines) >= limit:
            break
        line = _format_entry(entry)
        if max_chars is not None and size + len(line) + 1 > max_chars:
            break
        lines.append(line)
        size += len(line) + 1
    return "\n".join(lines)


def _entry_for(item) -> TimelineEntry | None:
    entity = str(item.normalized_entity or item.concept_original or "").strip()
    if not entity:
        return None
    date = item.observed_date or item.document_date or ""
    precision = (
        item.date_precision or _precision_of(date) or "unknown"
    )
    if not date and precision != "unknown":
        precision = "unknown"
    approximate = bool((item.data or {}).get("date_approximate"))
    description = _description(item, entity)
    return TimelineEntry(
        date=date,
        date_precision=precision,
        fact_type=item.fact_type or item.category,
        description=description,
        value=item.numeric_value,
        unit=item.unit,
        document_id=item.document_id,
        abnormal=bool((item.typed_payload or {}).get("flag")),
        polarity=item.data.get("polarity", "present"),
        approximate=approximate,
        source_text=item.source_text,
        source_occurrences=tuple(
            (item.data or {}).get("source_occurrences", ())
        ),
    )


def _description(item, entity: str) -> str:
    parts = []
    if item.data.get("polarity") == "negated" or item.assertion == "absent":
        parts.append("non")
    parts.append(entity)
    value = item.value_text or (
        f"{item.numeric_value:g}" if item.numeric_value is not None else ""
    )
    if value:
        parts.append(str(value))
        if item.unit:
            parts.append(str(item.unit))
    if item.data.get("polarity") == "suspected" or (
        item.certainty == "suspected"
    ):
        parts.append("(sospetto)")
    if (item.typed_payload or {}).get("flag") and not value:
        parts.append("(anomalo)")
    return " ".join(part for part in parts if part).strip()


def _precision_of(date: str) -> str:
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", date):
        return "day"
    if re.fullmatch(r"\d{4}-\d{2}", date):
        return "month"
    if re.fullmatch(r"\d{4}", date):
        return "year"
    return "unknown"


def _entry_sort_key(entry: TimelineEntry):
    return (
        date_sort_key(entry.date or None),
        entry.date_precision,
        entry.fact_type,
        entry.description,
        entry.document_id,
    )


def _format_entry(entry: TimelineEntry) -> str:
    label = _FACT_TYPE_LABELS.get(entry.fact_type, entry.fact_type or "altro")
    date = entry.date or "????-??-??"
    if entry.approximate and entry.date:
        date = f"~{date}"
    precision_mark = {
        "month": " [mese]",
        "year": " [anno]",
        "unknown": " [data referto]" if entry.date else "",
    }.get(entry.date_precision, "")
    description = entry.description
    if entry.abnormal and "anomalo" not in description and entry.value is None:
        description = f"{description} (anomalo)"
    reference = f" (rif: {entry.document_id})" if entry.document_id else ""
    return f"{date}{precision_mark} [{label}] {description}{reference}"

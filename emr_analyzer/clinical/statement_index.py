"""Patient-level index of clinical statements.

A *statement* is one sentence together with the context that gives it meaning:
the sentence itself, the sentence before it and the section heading in force.
Reports of one patient repeat whole statements verbatim (copy-forward), so the
index marks the first occurrence of each statement in time as its ``origin``
and every later one as a ``copy``.  Only origins are sent to the model; a copy
receives the projected annotation with its own offsets.

The index is deterministic and cheap: it reads text, never a model.  Identity
is content only — no document, no date, no model digest — so the same statement
keeps its key when the report that carries it changes.
"""
from dataclasses import dataclass, replace
import re

from .evidence_utils import SentenceSpan, content_hash
from .sentence_groups import clinical_sentences

#: Version of the statement identity rule (sentence + previous + heading).
STATEMENT_KEY_VERSION = "statement-v1"
#: Version of the projection rule; part of every document's input hash.
PROJECTION_VERSION = "statement-projection-v1"
#: Version of the canonical payload stored for a statement.
PAYLOAD_VERSION = "statement-payload-v1"

#: Marker that ends the report-date metadata header written by the importer.
REPORT_DATE_END = "<!-- /emr-report-date -->"
#: A heading is a short line that ends with a colon or is only digits/dates,
#: like the group planner, plus short uppercase section titles.
_DIGITS_OR_DATE = re.compile(r"[\d/ .:-]+")


def normalized(value) -> str:
    return " ".join(str(value or "").split())


def is_heading(text: str) -> bool:
    stripped = (text or "").strip()
    if not stripped:
        return False
    if len(stripped) <= 120 and (stripped.endswith(":") or _DIGITS_OR_DATE.fullmatch(stripped)):
        return True
    return len(stripped) <= 100 and stripped.isupper() and any(c.isalpha() for c in stripped)


def statement_key(sentence: str, previous: str, heading: str) -> str:
    """Identity of a statement: its text and the context that qualifies it."""
    return content_hash(STATEMENT_KEY_VERSION, normalized(sentence), normalized(previous),
                        normalized(heading))


def occurrence_id(document_id: str, start: int, end: int) -> str:
    return content_hash("statement-occ-v1", document_id, start, end)


@dataclass(frozen=True)
class StatementOccurrence:
    occurrence_id: str
    document_id: str
    document_date: str | None
    ordinal: int
    start: int
    end: int
    text: str
    normalized_text: str
    previous_text: str
    heading_text: str
    statement_key: str
    source_version: str
    role: str = "origin"
    carrier_occurrence_id: str = ""
    carrier_document_id: str = ""
    carrier_document_date: str | None = None

    @property
    def sort_order(self) -> tuple:
        return (self.document_date or "9999-99-99", self.document_id, self.start)


def index_document(document_id: str, document_date: str | None, text: str, *,
                   source_version: str = "") -> list[StatementOccurrence]:
    """Occurrences of every clinical sentence in one document, without roles.

    The report-date header is metadata, not a clinical source: its sentences
    are skipped so they can never become the carrier of a statement.
    """
    header_end = text.find(REPORT_DATE_END)
    header_end = header_end + len(REPORT_DATE_END) if header_end >= 0 else 0
    spans: list[SentenceSpan] = clinical_sentences(text)
    occurrences, heading = [], ""
    for span in spans:
        if span.end <= header_end:
            continue
        if is_heading(span.text):
            heading = normalized(span.text)
        previous = normalized(spans[span.sentence_id - 2].text) if span.sentence_id > 1 else ""
        occurrences.append(StatementOccurrence(
            occurrence_id=occurrence_id(document_id, span.start, span.end),
            document_id=document_id, document_date=document_date, ordinal=span.sentence_id,
            start=span.start, end=span.end, text=span.text,
            normalized_text=normalized(span.text), previous_text=previous, heading_text=heading,
            statement_key=statement_key(span.text, previous, heading),
            source_version=source_version or content_hash(text)))
    return occurrences


def assign_roles(occurrences: list[StatementOccurrence]) -> list[StatementOccurrence]:
    """Mark the earliest occurrence of each statement as its carrier.

    Order is the clinical order (document date, then document, then offset), so
    the carrier is the report where the statement first appears.  A document can
    be the carrier of one statement and a copy of another.
    """
    by_key: dict[str, list[StatementOccurrence]] = {}
    for item in occurrences:
        by_key.setdefault(item.statement_key, []).append(item)
    result = []
    for items in by_key.values():
        items.sort(key=lambda item: item.sort_order)
        carrier = items[0]
        result.append(replace(carrier, role="origin", carrier_occurrence_id=carrier.occurrence_id,
                              carrier_document_id=carrier.document_id,
                              carrier_document_date=carrier.document_date))
        for item in items[1:]:
            result.append(replace(item, role="copy", carrier_occurrence_id=carrier.occurrence_id,
                                  carrier_document_id=carrier.document_id,
                                  carrier_document_date=carrier.document_date))
    return result


def document_digest(occurrences: list[StatementOccurrence]) -> str:
    """Digest of how this document's statements are carried.

    Part of the document's ``input_hash``: when an older report takes over a
    statement, the copies change carrier and must be re-planned, without
    depending on whether the annotation exists yet.
    """
    shape = sorted(f"{item.statement_key}|{item.role}|{item.carrier_occurrence_id}"
                   for item in occurrences)
    return content_hash(PROJECTION_VERSION, *shape)

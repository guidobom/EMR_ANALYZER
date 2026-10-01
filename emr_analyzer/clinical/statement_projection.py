"""Deterministic projection of a certified statement onto its copies.

A statement is annotated once, in the report where it first appears.  Every
later report that repeats it shows the same sentence at its own offsets, so
the certified payload is re-anchored there: same clinical content, same
resolved date, this document's provenance.  No model is called and nothing is
resolved against the copy's own date — a copy never invents an onset.

Every step can fail; a failure is a *miss*, never a silent drop.  The pipeline
re-plans the sentence so it is annotated in its own right.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime
import json
import re

from .evidence_utils import locate_quote
from .statement_index import PAYLOAD_VERSION, StatementOccurrence, content_hash
from ..models.clinical_evidence import ClinicalEvidence

WORDS = re.compile(r"\S+")
#: Fields copied verbatim from the carrier's validated row.
ROW_FIELDS = ("assertion", "temporality", "clinical_status", "value_text",
              "numeric_value", "unit", "anatomical_site", "laterality", "severity",
              "significance", "certainty", "mapping_status", "terminology_system",
              "terminology_code", "model_name", "prompt_version")


def sentence_tokens(text: str):
    return list(WORDS.finditer(text))


def anchor_offsets(sentence_text: str, first_word: int, last_word: int):
    """Character offsets of a word interval inside one sentence, or None."""
    tokens = sentence_tokens(sentence_text)
    if not isinstance(first_word, int) or not isinstance(last_word, int):
        return None
    if not 1 <= first_word <= last_word <= len(tokens):
        return None
    return sentence_text.index(tokens[0].group()), tokens[first_word - 1].start(), tokens[last_word - 1].end()


def same_text(left: str, right: str) -> bool:
    return " ".join((left or "").split()) == " ".join((right or "").split())


# --------------------------------------------------------------------------
# Canonical payload: what one validated row means, without its offsets.
# --------------------------------------------------------------------------

def event_payload(row: ClinicalEvidence) -> dict | None:
    """The certified content of one row, addressable in any identical sentence."""
    data = row.data or {}
    selection = data.get("source_selection") or {}
    sentence = data.get("source_sentence") or {}
    date = data.get("date_provenance") or {}
    if not sentence.get("text") or not isinstance(selection.get("first_word"), int):
        return None
    context = []
    if date.get("quote") and date.get("mode") in ("relative", "document"):
        context.append(date["quote"])
    return {
        "label": row.normalized_entity, "fact_type": row.fact_type,
        "category": row.category, "concept_original": row.concept_original,
        "canonical_label": row.canonical_label,
        "fields": {key: getattr(row, key) for key in ROW_FIELDS},
        "subject": data.get("experiencer") or "patient",
        "attributes": data.get("attributes") or {},
        "typed_payload": row.typed_payload or {},
        "lexicon_term_id": data.get("lexicon_term_id"),
        "episode_quote": data.get("episode_quote"),
        "event_kind": data.get("event_kind"),
        "status": row.status,
        "anchor": {"first_word": selection["first_word"], "last_word": selection["last_word"],
                   "quote": row.source_text},
        "date": {"mode": date.get("mode"), "quote": date.get("quote"),
                 "anchor_date": date.get("anchor_date"),
                 "needs_review": date.get("needs_review"),
                 "resolved_date": row.observed_date, "resolved_date_end": row.observed_date_end,
                 "precision": row.date_precision, "source": row.date_source},
        "payload_version": PAYLOAD_VERSION,
    }


def payload_from_rows(rows) -> list[dict]:
    payload = []
    for row in rows:
        if row.extraction_method != "fhir_events_v1":
            continue
        item = event_payload(row)
        if item is not None:
            payload.append(item)
    return payload


# --------------------------------------------------------------------------
# Projection of one occurrence
# --------------------------------------------------------------------------

def project_occurrence(occurrence: StatementOccurrence, payload: list[dict], *,
                       patient_id: str, text: str, document_date: str | None,
                       document_type: str, carrier_document_id: str,
                       carrier_document_date: str | None, geometry=None):
    """Rows for one copy, or the reasons why the copy must be annotated afresh."""
    rows, misses = [], []
    sentence = text[occurrence.start:occurrence.end]
    for item in payload:
        anchor = item.get("anchor") or {}
        offsets = anchor_offsets(sentence, anchor.get("first_word"), anchor.get("last_word"))
        if offsets is None:
            misses.append({"reason": "intervallo di parole fuori dalla copia",
                           "label": item.get("label")})
            continue
        _, first, last = offsets
        start, end = occurrence.start + first, occurrence.start + last
        quote = text[start:end]
        if not same_text(quote, anchor.get("quote")):
            misses.append({"reason": "citazione diversa nella copia", "label": item.get("label")})
            continue
        date = item.get("date") or {}
        if date.get("needs_review"):
            misses.append({"reason": "data non risolta nel referto d'origine", "label": item.get("label")})
            continue
        rows.append(_row(item, occurrence, patient_id=patient_id, text=text,
                         start=start, end=end, quote=quote, document_date=document_date,
                         document_type=document_type, carrier_document_id=carrier_document_id,
                         carrier_document_date=carrier_document_date, geometry=geometry))
    return rows, misses


def _status(item) -> str:
    """A repeated statement is only as strong as its date.

    A discrete event (a procedure, an access, an examination) whose date came
    from the carrier's report rather than from the sentence itself is kept for
    review: two identical sentences can describe two different episodes.
    """
    status = item.get("status") or "proposed"
    if item.get("event_kind") == "discrete" and (item.get("date") or {}).get("mode") in (
            "relative", "document", None):
        return "needs_review"
    return status


def _row(item, occurrence, *, patient_id, text, start, end, quote, document_date, document_type,
         carrier_document_id, carrier_document_date, geometry) -> ClinicalEvidence:
    fields = dict(item.get("fields") or {})
    date = item.get("date") or {}
    page, bbox = None, None
    if geometry is not None:
        page, bbox = geometry.locate_source(quote)
    provenance = {"mode": date.get("mode"), "quote": date.get("quote"),
                  "anchor_date": date.get("anchor_date"),
                  "projected_from_document": carrier_document_id,
                  "projected_from_date": carrier_document_date,
                  "statement_occurrence": occurrence.occurrence_id}
    data = {
        "fhir_pipeline": True, "experiencer": item.get("subject") or "patient",
        "attributes": item.get("attributes") or {},
        "lexicon_term_id": item.get("lexicon_term_id"),
        "episode_quote": item.get("episode_quote"), "event_kind": item.get("event_kind"),
        "date_provenance": provenance, "source_version": content_hash(text),
        "source_spans": [{"start": start, "end": end, "text": quote,
                          "sentence_id": occurrence.ordinal}],
        "source_sentence": {"start": occurrence.start, "end": occurrence.end, "text": occurrence.text},
        "source_selection": {"sentence": occurrence.ordinal,
                             "first_word": item["anchor"]["first_word"],
                             "last_word": item["anchor"]["last_word"]},
        "statement_reuse": {"statement_key": occurrence.statement_key, "role": "copy",
                            "carrier_occurrence_id": occurrence.carrier_occurrence_id,
                            "carrier_document_id": carrier_document_id,
                            "carrier_document_date": carrier_document_date},
        "extraction_pipeline": fields.get("prompt_version"),
        "snomed_mapping_status": "pending",
    }
    row = ClinicalEvidence(
        patient_id=patient_id, document_id=occurrence.document_id, category=item["category"],
        normalized_entity=item["label"], fact_type=item.get("fact_type"),
        concept_original=item.get("concept_original"), canonical_label=item.get("canonical_label"),
        source_text=quote, document_date=document_date, source_page=page, bbox=bbox,
        observed_date=date.get("resolved_date"), observed_date_end=date.get("resolved_date_end"),
        date_precision=date.get("precision") or "unknown", date_source=date.get("source"),
        extraction_method="fhir_events_v1",
        typed_payload=item.get("typed_payload") or {}, data=data,
        status=_status(item), **fields)
    identity = content_hash(patient_id, occurrence.document_id, content_hash(text), start, end,
                            row.normalized_entity, row.fact_type, row.assertion, row.certainty,
                            data["experiencer"], row.observed_date, row.observed_date_end,
                            row.clinical_status, json.dumps(row.data.get("attributes") or {},
                                                            sort_keys=True))
    return replace(row, evidence_id="EVD_" + identity)


# --------------------------------------------------------------------------
# Service: certify a document's statements, project them onto the copies
# --------------------------------------------------------------------------

class StatementProjectionService:
    """Ties the index, the certified payloads and the projector together."""

    def __init__(self, statements, annotations):
        self.statements = statements
        self.annotations = annotations

    def certify(self, patient_id: str, document_id: str, rows, *, model_digest: str,
                prompt_version: str, model_name: str | None = None,
                complete: bool = True) -> int:
        """Store the payload of every statement this document carries.

        With ``complete`` the statements that yielded no event are certified as
        empty, so their copies are not re-read.  After a partial extraction only
        the statements that produced rows are certified: the others stay open
        for a fresh reading.
        """
        occurrences = self.statements.for_document(document_id)
        if not occurrences:
            return 0
        carried: dict[str, list] = {}
        for row in rows:
            if getattr(row, "extraction_method", None) != "fhir_events_v1":
                continue
            span = ((row.data or {}).get("source_spans") or [{}])[0]
            start = span.get("start")
            if start is None:
                continue
            occurrence = next((item for item in occurrences
                               if item.start <= start < item.end), None)
            if occurrence is None or occurrence.role != "origin":
                continue
            # The carrier is one of the occurrences of the statement: stamping
            # it lets the FHIR projection fold all of them into one resource.
            (row.data or {}).setdefault("statement_reuse", {
                "statement_key": occurrence.statement_key, "role": "origin",
                "carrier_occurrence_id": occurrence.occurrence_id,
                "carrier_document_id": document_id,
                "carrier_document_date": occurrence.document_date})
            carried.setdefault(occurrence.statement_key, []).append(row)
        certified = 0
        for occurrence in occurrences:
            if occurrence.role != "origin":
                continue
            statement_rows = carried.get(occurrence.statement_key)
            if statement_rows is None and not complete:
                continue
            payload = payload_from_rows(statement_rows or [])
            self.annotations.save(
                patient_id=patient_id, statement_key=occurrence.statement_key,
                carrier_occurrence_id=occurrence.occurrence_id,
                carrier_document_id=document_id,
                carrier_document_date=occurrence.document_date,
                carrier_sentence=occurrence.normalized_text,
                status="completed" if payload else "empty", payload=payload,
                model_digest=model_digest, prompt_version=prompt_version,
                model_name=model_name)
            certified += 1
        return certified

    def project(self, *, patient_id: str, document_id: str, text: str,
                document_date: str | None, document_type: str, geometry=None,
                model_digest: str, prompt_version: str):
        """Rows for the copies of this document, and what could not be projected."""
        occurrences = self.statements.for_document(document_id)
        rows, misses = [], []
        for occurrence in occurrences:
            if occurrence.role != "copy":
                continue
            annotation = self.annotations.get(patient_id, occurrence.statement_key,
                                              model_digest, prompt_version)
            if annotation is None:
                misses.append(_miss(document_id, occurrence, "nessuna annotazione certificata"))
                continue
            if annotation["carrier_occurrence_id"] != occurrence.carrier_occurrence_id:
                misses.append(_miss(document_id, occurrence, "il referto d'origine è cambiato"))
                continue
            if annotation["status"] != "completed":
                continue
            projected, missed = project_occurrence(
                occurrence, annotation["payload"], patient_id=patient_id, text=text,
                document_date=document_date, document_type=document_type,
                carrier_document_id=annotation["carrier_document_id"],
                carrier_document_date=annotation["carrier_document_date"], geometry=geometry)
            rows.extend(projected)
            for item in missed:
                misses.append(_miss(document_id, occurrence, item["reason"],
                                    label=item.get("label")))
        return rows, misses


def _miss(document_id: str, occurrence: StatementOccurrence, reason: str,
          label: str | None = None) -> dict:
    return {"document_id": document_id, "occurrence_id": occurrence.occurrence_id,
            "ordinal": occurrence.ordinal,
            "statement_key": occurrence.statement_key, "reason": reason,
            "label": label or occurrence.text[:80],
            "carrier_document_id": occurrence.carrier_document_id}


def document_geometry(path):
    """Same geometry source the extractor uses, so copies keep page and box."""
    from pathlib import Path
    if not path:
        return None
    file = Path(path)
    if not file.exists():
        return None
    try:
        from ..pipeline.pdf_extractor import PdfExtractionResult, PdfPlumberExtractor
        if file.suffix.lower() == ".pdf":
            return PdfPlumberExtractor().convert(file)
        return PdfExtractionResult.from_dict(json.loads(file.read_text(encoding="utf-8")))
    except (OSError, ValueError, TypeError):
        return None

"""Local-LLM clinical projection of immutable document content."""

from __future__ import annotations

from datetime import datetime
import json
from typing import Iterable

from ..models.clinical_evidence import ClinicalEvidence
from ..models.clinical_event import ClinicalEvent, EventType
from ..utils.text_utils import fuzzy_find


PROMPT_VERSION = "document_clinical_projection_v1"
SCHEMA_VERSION = "1.0"


EVIDENCE_SCHEMA = {
    "type": "object",
    "properties": {
        "observations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "category": {"type": "string"},
                    "normalized_entity": {"type": "string"},
                    "assertion": {"type": "string"},
                    "temporality": {"type": "string"},
                    "clinical_status": {"type": ["string", "null"]},
                    "observed_date": {"type": ["string", "null"]},
                    "value_text": {"type": ["string", "null"]},
                    "numeric_value": {"type": ["number", "null"]},
                    "unit": {"type": ["string", "null"]},
                    "source_page": {"type": ["integer", "null"]},
                    "source_text": {"type": "string"},
                    "confidence": {"type": "number"},
                },
                "required": [
                    "category", "normalized_entity", "assertion",
                    "temporality", "source_text", "confidence",
                ],
            },
        }
    },
    "required": ["observations"],
}


class ClinicalContentIsolator:
    """Extract a non-destructive structured clinical projection."""

    EVENT_CATEGORIES = {event_type.value for event_type in EventType}
    ALLOWED_CATEGORIES = EVENT_CATEGORIES | {
        "physical_exam", "medical_history", "negative_finding",
        "medication_current", "laboratory_finding", "care_plan",
        "biomarker", "vital_sign", "other",
    }

    def __init__(self, llm_client):
        self.llm = llm_client

    def extract(self, text: str, patient_id: str, document_id: str,
                document_date: str | None = None,
                parsing_result=None) -> list[ClinicalEvidence]:
        if not self.llm or not self.llm.is_available:
            return []
        evidence = []
        attempted_chunks = 0
        completed_chunks = 0
        for chunk in self._chunks(text, parsing_result):
            attempted_chunks += 1
            prompt = self._prompt(chunk, document_date)
            try:
                data = self.llm.generate_structured(
                    prompt,
                    (
                        "Sei un estrattore di evidenze cliniche da referti italiani. "
                        "Non riassumere e non inventare. Ogni osservazione deve "
                        "contenere una citazione letterale presente nell'input."
                    ),
                    EVIDENCE_SCHEMA,
                )
            except Exception:
                continue
            if not isinstance(data, dict) or not isinstance(
                data.get("observations"), list
            ):
                continue
            completed_chunks += 1
            for item in data.get("observations", []):
                converted = self._convert_item(
                    item, chunk, patient_id, document_id, document_date,
                    parsing_result,
                )
                if converted:
                    evidence.append(converted)
        if attempted_chunks and completed_chunks == 0:
            raise RuntimeError(
                "Il modello locale non ha restituito alcuna risposta strutturata valida"
            )
        return self._deduplicate(evidence)

    def _prompt(self, text: str, document_date: str | None) -> str:
        categories = ", ".join(sorted(self.ALLOWED_CATEGORIES))
        return f"""Crea una proiezione clinica strutturata del documento seguente.

Regole:
- conserva tutte le osservazioni clinicamente rilevanti, incluse negazioni;
- distingui dato osservato, dichiarazione del paziente e valutazione clinica;
- includi diagnosi, sintomi, esame obiettivo, anamnesi, terapie correnti,
  inizio/fine/modifica terapia, procedure, ricoveri, radiologia, tossicità,
  eventi avversi, risposta/progressione e piani di follow-up;
- non reinterpretare tabelle di laboratorio già estratte deterministicamente;
- source_text deve essere una citazione letterale breve dell'input;
- source_page deve derivare dal marcatore PAGINA più vicino;
- se la data dell'osservazione non è esplicita usa la data documento soltanto
  quando l'osservazione è chiaramente corrente;
- categorie consentite: {categories}.

Data documento: {document_date or 'non determinata'}

DOCUMENTO:
{text}
"""

    def _convert_item(self, item: dict, chunk: str, patient_id: str,
                      document_id: str, document_date: str | None,
                      parsing_result) -> ClinicalEvidence | None:
        source_text = str(item.get("source_text") or "").strip()
        entity = str(item.get("normalized_entity") or "").strip()
        if not source_text or not entity:
            return None
        if not fuzzy_find(chunk, source_text, threshold=0.78):
            return None
        category = str(item.get("category") or "other").strip().lower()
        if category not in self.ALLOWED_CATEGORIES:
            category = "other"
        page = self._int_or_none(item.get("source_page"))
        bbox = None
        if parsing_result and hasattr(parsing_result, "locate_source"):
            located_page, bbox = parsing_result.locate_source(source_text, page)
            page = located_page or page
        observed_date = self._validated_date(item.get("observed_date"))
        if not observed_date and item.get("temporality") == "current":
            observed_date = document_date
        confidence = max(0.0, min(1.0, float(item.get("confidence", 0.5))))
        return ClinicalEvidence(
            patient_id=patient_id,
            document_id=document_id,
            category=category,
            normalized_entity=entity,
            assertion=str(item.get("assertion") or "present").lower(),
            temporality=str(item.get("temporality") or "current").lower(),
            clinical_status=item.get("clinical_status"),
            observed_date=observed_date,
            value_text=item.get("value_text"),
            numeric_value=item.get("numeric_value"),
            unit=item.get("unit"),
            source_page=page,
            source_text=source_text,
            bbox=bbox,
            confidence=confidence,
            extraction_method="llm_document_projection",
            model_name=self.llm.model,
            prompt_version=PROMPT_VERSION,
            schema_version=SCHEMA_VERSION,
            data={"document_date": document_date},
        )

    @staticmethod
    def _chunks(text: str, parsing_result, max_chars: int = 12000) -> Iterable[str]:
        if parsing_result and getattr(parsing_result, "pages", None):
            buffer = []
            size = 0
            for page in parsing_result.pages:
                page_text = f"\n--- PAGINA {page.page} ---\n{page.text.strip()}"
                if buffer and size + len(page_text) > max_chars:
                    yield "".join(buffer)
                    buffer, size = [], 0
                if len(page_text) > max_chars:
                    raw_page_text = page.text.strip()
                    marker = f"\n--- PAGINA {page.page} ---\n"
                    payload_size = max(1, max_chars - len(marker))
                    for start in range(0, len(raw_page_text), payload_size):
                        yield marker + raw_page_text[start:start + payload_size]
                    continue
                buffer.append(page_text)
                size += len(page_text)
            if buffer:
                yield "".join(buffer)
            return
        for start in range(0, len(text), max_chars):
            yield text[start:start + max_chars]

    @staticmethod
    def _validated_date(value) -> str | None:
        if not value:
            return None
        value = str(value).strip()
        try:
            parsed = datetime.fromisoformat(value)
            if 1850 <= parsed.year <= 2100:
                return value
        except ValueError:
            # Preserve partial or explicitly estimated dates for later review.
            if len(value) >= 4:
                return value
        return None

    @staticmethod
    def _int_or_none(value) -> int | None:
        try:
            return int(value) if value is not None else None
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _deduplicate(evidence: list[ClinicalEvidence]) -> list[ClinicalEvidence]:
        seen = set()
        result = []
        for item in evidence:
            key = (
                item.category, item.normalized_entity.lower(), item.assertion,
                item.observed_date, item.source_text.lower(),
            )
            if key in seen:
                continue
            seen.add(key)
            result.append(item)
        return result

    @classmethod
    def to_events(cls, evidence: list[ClinicalEvidence]) -> list[ClinicalEvent]:
        events = []
        for item in evidence:
            if item.category not in cls.EVENT_CATEGORIES:
                continue
            if item.assertion in {"absent", "negated", "not_present"}:
                continue
            events.append(ClinicalEvent(
                event_id="",
                patient_id=item.patient_id,
                event_date=item.observed_date or "",
                event_type=item.category,
                entity=item.normalized_entity,
                value=item.numeric_value,
                unit=item.unit,
                source_document_id=item.document_id,
                page=item.source_page,
                source_text=item.source_text,
                confidence=item.confidence,
            ))
        return events

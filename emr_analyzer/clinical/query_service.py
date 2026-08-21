"""Structured + full-text local retrieval for clinical questions."""

from __future__ import annotations

import re
from typing import Iterable


_INTENT_CATEGORIES = {
    "oncology": {
        "oncology_treatment_line", "medication", "toxicity", "response",
        "progression", "diagnosis", "imaging_finding", "biomarker",
    },
    "therapy": {"medication", "oncology_treatment_line", "procedure"},
    "toxicity": {"toxicity", "adverse_event", "clinical_syndrome"},
    "laboratory": {"laboratory_finding", "laboratory_trend", "biomarker"},
    "respiratory": {
        "clinical_syndrome", "symptom", "clinical_sign", "vital_sign",
        "imaging_finding", "laboratory_finding",
    },
    "active": {
        "diagnosis", "comorbidity", "symptom", "clinical_sign",
        "medication", "oncology_treatment_line", "toxicity",
        "clinical_syndrome",
    },
}


class ClinicalQueryService:
    def __init__(self, registry_repo):
        self.registry_repo = registry_repo

    def retrieve(
        self, patient_id: str, question: str, *, limit: int = 500
    ) -> list[dict]:
        intents = infer_intents(question)
        broad = is_broad_question(question)
        selected = {}
        if broad:
            for event in self.registry_repo.get_events(patient_id):
                selected[event.event_id] = event
        else:
            for event in self.registry_repo.search_events(
                patient_id, question, limit=limit
            ):
                selected[event.event_id] = event
            categories = set().union(*(
                _INTENT_CATEGORIES[intent] for intent in intents
            )) if intents else set()
            for category in categories:
                status = "active" if "active" in intents else None
                for event in self.registry_repo.get_events(
                    patient_id, category=category, status=status
                ):
                    selected[event.event_id] = event
        ranked = sorted(
            selected.values(),
            key=lambda event: event_rank(event, question, intents),
            reverse=True,
        )[:limit]
        details = []
        for event in ranked:
            detail = self.registry_repo.get_event_detail(event.event_id)
            if detail:
                details.append(detail)
        details.sort(key=lambda item: (
            item["event"].get("first_evidence_date") or "9999",
            item["event"].get("event_id") or "",
        ))
        return details

    @staticmethod
    def format_chunks(
        details: Iterable[dict], *, max_chars: int = 45_000
    ) -> list[str]:
        blocks = []
        for detail in details:
            block = format_event_detail(detail)
            blocks.extend(_split_oversized_event(block, max_chars))
        if not blocks:
            return ["Nessun evento pertinente recuperato dal registro."]
        chunks, current, size = [], [], 0
        for block in blocks:
            if current and size + len(block) + 2 > max_chars:
                chunks.append("\n\n".join(current))
                current, size = [], 0
            current.append(block)
            size += len(block) + 2
        if current:
            chunks.append("\n\n".join(current))
        return chunks


def _split_oversized_event(block: str, max_chars: int) -> list[str]:
    """Split a source-rich event without dropping evidence or its event ID."""
    if len(block) <= max_chars:
        return [block]
    lines = block.splitlines()
    header = lines[0] if lines else "[evento]"
    units = []
    for line in lines[1:]:
        if len(line) <= max_chars - len(header) - 2:
            units.append(line)
            continue
        width = max(1000, max_chars - len(header) - 2)
        units.extend(line[start:start + width] for start in range(0, len(line), width))
    result, current, size = [], [header], len(header)
    for unit in units:
        if len(current) > 1 and size + len(unit) + 1 > max_chars:
            result.append("\n".join(current))
            current, size = [header], len(header)
        current.append(unit)
        size += len(unit) + 1
    if len(current) > 1:
        result.append("\n".join(current))
    return result or [header]


def format_event_detail(detail: dict) -> str:
    event = detail["event"]
    lines = [
        f"[#{event['event_id']}] "
        f"[{event.get('first_evidence_date') or 'data n.d.'}] "
        f"[{event.get('category')}] [{event.get('status')}] "
        f"[{event.get('certainty')}] {event.get('summary_short', '')}",
    ]
    if event.get("summary_detail") and (
        event["summary_detail"] != event.get("summary_short")
    ):
        lines.append(f"Dettaglio: {event['summary_detail']}")
    if detail.get("updates"):
        lines.append("Aggiornamenti:")
        for update in detail["updates"]:
            citations = ",".join(update.get("evidence_ids") or [])
            lines.append(
                f"- {update.get('update_date') or 'data n.d.'}: "
                f"{update.get('summary', '')} [evidenze:{citations}]"
            )
    lines.append("Fonti verificabili:")
    for evidence in detail.get("evidence", []):
        page = evidence.get("source_page") or "n.d."
        relation = evidence.get("relation") or "supports"
        lines.append(
            f"- [{evidence.get('evidence_id')}; "
            f"{evidence.get('document_id')}:p.{page}; {relation}] "
            f"{evidence.get('source_text', '')}"
        )
    return "\n".join(lines)


def infer_intents(question: str) -> set[str]:
    text = str(question or "").casefold()
    result = set()
    if any(token in text for token in (
        "oncol", "linea terapeut", "immunoterap", "chemioterap",
        "progression", "risposta"
    )):
        result.add("oncology")
    if any(token in text for token in (
        "terapi", "farmac", "trattament", "assunt", "prescritt"
    )):
        result.add("therapy")
    if any(token in text for token in (
        "tossicit", "evento avvers", "irae", "immuno-correlat"
    )):
        result.add("toxicity")
    if any(token in text for token in (
        "laborator", "ematochim", "trend", "emocromo", "tsh", "creatinin"
    )):
        result.add("laboratory")
    if any(token in text for token in (
        "respir", "dispnea", "tosse", "saturazione", "spo2", "polmon"
    )):
        result.add("respiratory")
    if any(token in text for token in (
        "attual", "attiv", "in corso", "corrente"
    )):
        result.add("active")
    return result


def is_broad_question(question: str) -> bool:
    text = str(question or "").casefold()
    return any(phrase in text for phrase in (
        "storia completa", "riassunto cronologico completo",
        "intera storia", "tutti gli eventi", "quadro clinico complessivo",
    ))


def event_rank(event, question: str, intents: set[str]) -> tuple:
    text = f"{event.canonical_entity} {event.summary_short}".casefold()
    terms = {
        token for token in re.findall(r"[a-zà-öø-ÿ0-9]+", question.casefold())
        if len(token) >= 3
    }
    term_hits = sum(1 for token in terms if token in text)
    intent_categories = set().union(*(
        _INTENT_CATEGORIES[intent] for intent in intents
    )) if intents else set()
    return (
        1 if event.category in intent_categories else 0,
        term_hits,
        1 if event.status in {"active", "ongoing"} and "active" in intents else 0,
        event.first_evidence_date or "",
    )


def answer_has_valid_citations(
    answer: str,
    valid_event_ids: set[str],
    valid_event_document_pairs: set[tuple[str, str]] | None = None,
) -> bool:
    cited = set(re.findall(r"#(EVT_[A-Fa-f0-9]+)", str(answer or "")))
    if not cited or not cited.issubset(valid_event_ids):
        return False
    if valid_event_document_pairs is None:
        return True
    pairs = set(re.findall(
        r"\[#(EVT_[A-Fa-f0-9]+)\s*;\s*([^\]:;]+)\s*:p\.",
        str(answer or ""),
    ))
    return bool(pairs) and pairs.issubset(valid_event_document_pairs)

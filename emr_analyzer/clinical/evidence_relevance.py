"""Registry disposition of immutable atomic clinical evidence.

Atomic evidence is the auditable source layer and is never deleted merely
because it is not suitable for a chronological registry.  This module assigns
one of three downstream roles:

``primary``
    May anchor or update a clinical episode.
``contextual``
    A normal/negative observation that may support an existing episode but
    must not create a standalone row by itself.
``administrative_or_methodological``
    Scheduling, report metadata or acquisition technique.  It remains in
    ``clinical_evidence`` but is excluded from events, summaries and LLM
    registry contexts.

Classification deliberately uses the extracted *entity*, not the complete
source span.  A long radiology sentence can contain both dose boilerplate and
real findings; filtering the whole span would discard the findings as well.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
import unicodedata
from typing import Iterable

from ..models.clinical_evidence import ClinicalEvidence


PRIMARY = "primary"
CONTEXTUAL = "contextual"
ADMINISTRATIVE_OR_METHODOLOGICAL = "administrative_or_methodological"


@dataclass(frozen=True, slots=True)
class EvidenceDisposition:
    role: str
    reason: str

    @property
    def registry_eligible(self) -> bool:
        return self.role != ADMINISTRATIVE_OR_METHODOLOGICAL

    @property
    def can_anchor_episode(self) -> bool:
        return self.role == PRIMARY


def _surface(value: object) -> str:
    text = unicodedata.normalize("NFKD", str(value or "").casefold())
    text = "".join(char for char in text if not unicodedata.combining(char))
    text = text.replace("_", " ")
    return " ".join(text.split())


_HEADER_ENTITY_PATTERNS = tuple(re.compile(pattern) for pattern in (
    r"^(?:paziente|dati paziente)(?:\s*[:\-].*)?$",
    r"^(?:data|ora) (?:dell'?\s*)?esame(?:\s*[:\-].*)?$",
    r"^data di nascita(?:\s*[:\-].*)?$",
    r"^(?:acc(?:ession)? number|id paziente|numero referto|n esame)(?:\s*[:\-].*)?$",
    r"^(?:richiesto da|provenienza|quesito clinico)(?:\s*[:\-].*)?$",
    r"^classe dose(?:\s*[:\-].*)?$",
    r"^prestazioni eseguite(?:\s+e\s+indicazione\s+di\s+dose)?.*$",
    r"^indicazione di dose secondo .*$",
    r"^esame confrontato con precedente.*$",
    r"^modalita di esecuzione.*$",
))

_APPOINTMENT = re.compile(
    r"\b(?:appuntament\w*|prenotazion\w*|prenotat[oaie]?|"
    r"prossim[oaie]?\s+(?:visita|controllo|accesso))\b"
)
_APPOINTMENT_BLOCK = re.compile(
    r"^(?:(?:prossim|possibil|possim)[oaie]?\s+appuntament\w*|"
    r"appuntament\w*\s+(?:successiv\w*|fissat\w*))\b"
)
_TRACER = re.compile(
    r"\b(?:18f[- ]?fdg|fdg|radiotracciant\w*|tracciant\w*|"
    r"68ga|ga[- ]?68|99mtc|tc[- ]?99m|dotatate|psma)\b"
)
_ADMINISTRATION = re.compile(
    r"\b(?:somministrazion\w*|somministrat[oaie]?|iniezion\w*|"
    r"iniettat[oaie]?|infusion\w*|e\.?v\.?)\b"
)
_METHOD_ONLY = re.compile(
    r"\b(?:eseguit[oa] a digiuno|previa idratazione|tempo di uptake|"
    r"acquisizione (?:delle )?immagini|ricostruzione delle immagini|"
    r"protocollo di acquisizione|indicazione di dose|classe dose|"
    r"modalita di esecuzione)\b"
)
_REGULATORY_ONLY = re.compile(
    r"\b(?:ai sensi|secondo|in conformita)\s+(?:dell['’]?\s*)?art\.?\s*\d+|"
    r"\b(?:d\.?lgs\.?|decreto legislativo)\s*\d+"
)
_ROUTINE_NORMAL_VARIANT = re.compile(
    r"^(?:uter[oa]\s+antiversofless[oa]|uterus\s+antevert\w*|"
    r"vescica\s+(?:scarsamente\s+)?repleta\s+a\s+pareti\s+regolari)\b"
)
_ABNORMAL_IMAGING_CUE = re.compile(
    r"\b(?:lesion\w*|formazion\w*|massa|nodul\w*|cist\w*|metastas\w*|"
    r"secondari\w*|edema|versamento|falda|alterat\w*|patologic\w*)\b"
)

_CONTEXT_STATUS = {
    "normal", "within_range", "negative", "absent", "excluded",
}
_NORMAL_OR_NEGATIVE = re.compile(
    r"\b(?:nei limiti|nella norma|regolare|negativ[oaie]?|"
    r"assenza di|non evidenza di|non si evidenzia|esclus[oaie]?)\b"
)
_CONTEXT_CATEGORIES = {
    "laboratory_finding", "imaging_finding", "histopathology",
    "biomarker", "clinical_sign", "vital_sign", "symptom", "diagnosis",
}


def classify_evidence(item: ClinicalEvidence) -> EvidenceDisposition:
    """Classify one atom for downstream registry use without deleting it."""
    entity = _surface(item.normalized_entity)
    quote = _surface(item.source_text)

    administrative = _administrative_disposition(entity, quote)
    if administrative is not None:
        return administrative

    status = _surface(item.clinical_status)
    assertion = _surface(item.assertion)
    certainty = _surface(item.certainty)
    if (
        assertion in {"absent", "conditional", "hypothetical"}
        or certainty == "excluded"
        or status in _CONTEXT_STATUS
        or (
            item.category in _CONTEXT_CATEGORIES
            and _NORMAL_OR_NEGATIVE.search(quote)
        )
    ):
        return EvidenceDisposition(CONTEXTUAL, "normal_or_negative_context")

    return EvidenceDisposition(PRIMARY, "clinical_episode_candidate")


def _administrative_disposition(
    entity: str, quote: str
) -> EvidenceDisposition | None:
    """Return the non-clinical disposition shared by rows and dataclasses."""

    if (
        _ROUTINE_NORMAL_VARIANT.search(entity or quote)
        and not _ABNORMAL_IMAGING_CUE.search(quote)
    ):
        return EvidenceDisposition(
            ADMINISTRATIVE_OR_METHODOLOGICAL, "routine_normal_finding",
        )

    if any(pattern.search(entity) for pattern in _HEADER_ENTITY_PATTERNS):
        return EvidenceDisposition(
            ADMINISTRATIVE_OR_METHODOLOGICAL, "report_metadata",
        )
    # An extractor may label one bullet as "PET", "prelievi" or "visita"
    # even though its complete grounded span is only the future-appointments
    # block.  The anchored source check removes those false clinical events
    # without penalising a long visit note that later contains appointments.
    if _APPOINTMENT.search(entity) or _APPOINTMENT_BLOCK.search(quote):
        return EvidenceDisposition(
            ADMINISTRATIVE_OR_METHODOLOGICAL, "scheduling",
        )
    if _TRACER.search(entity) and _ADMINISTRATION.search(entity):
        return EvidenceDisposition(
            ADMINISTRATIVE_OR_METHODOLOGICAL,
            "diagnostic_tracer_administration",
        )
    if _METHOD_ONLY.search(entity):
        return EvidenceDisposition(
            ADMINISTRATIVE_OR_METHODOLOGICAL, "diagnostic_method",
        )
    if _REGULATORY_ONLY.search(entity):
        return EvidenceDisposition(
            ADMINISTRATIVE_OR_METHODOLOGICAL, "regulatory_boilerplate",
        )
    return None


def classify_nonclinical_passage(text: str) -> EvidenceDisposition | None:
    """Classify one standalone line before it reaches the extraction LLM."""
    value = _surface(text)
    if not value:
        return None
    return _administrative_disposition(value, value)


def is_administrative_mapping(item: dict) -> bool:
    """Defensive downstream check for database/Quick View row mappings."""
    data = item.get("data") or {}
    if isinstance(data, dict) and data.get("registry_role") == (
        ADMINISTRATIVE_OR_METHODOLOGICAL
    ):
        return True
    return _administrative_disposition(
        _surface(item.get("normalized_entity")),
        _surface(item.get("source_text")),
    ) is not None


def annotate_evidence_disposition(
    item: ClinicalEvidence,
) -> EvidenceDisposition:
    """Persist the reproducible classification in the evidence JSON payload."""
    disposition = classify_evidence(item)
    item.data = dict(item.data or {})
    item.data["registry_role"] = disposition.role
    item.data["registry_role_reason"] = disposition.reason
    if disposition.role == PRIMARY:
        item.clinical_relevance = "accepted_clinical"
    elif disposition.role == CONTEXTUAL:
        item.clinical_relevance = "accepted_low_relevance"
    elif disposition.reason == "scheduling":
        item.clinical_relevance = "excluded_administrative"
    elif disposition.reason in {
        "diagnostic_method", "diagnostic_tracer_administration",
    }:
        item.clinical_relevance = "excluded_methodological"
    elif disposition.reason == "routine_normal_finding":
        item.clinical_relevance = "excluded_non_informative"
    else:
        item.clinical_relevance = "excluded_boilerplate"
    return disposition


def partition_evidence(
    evidence: Iterable[ClinicalEvidence],
) -> tuple[list[ClinicalEvidence], list[ClinicalEvidence], list[ClinicalEvidence]]:
    """Return ``(primary, contextual, administrative/methodological)``."""
    primary: list[ClinicalEvidence] = []
    contextual: list[ClinicalEvidence] = []
    excluded: list[ClinicalEvidence] = []
    for item in evidence:
        role = annotate_evidence_disposition(item).role
        if role == PRIMARY:
            primary.append(item)
        elif role == CONTEXTUAL:
            contextual.append(item)
        else:
            excluded.append(item)
    return primary, contextual, excluded


def is_administrative_evidence(item: ClinicalEvidence) -> bool:
    """Compatibility predicate for callers and older tests."""
    return classify_evidence(item).role == ADMINISTRATIVE_OR_METHODOLOGICAL

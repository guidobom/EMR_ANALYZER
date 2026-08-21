"""Cautious cross-document, cross-modality clinical correlations."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable

from .consolidation import (
    ConsolidatedBundle,
    best_date_precision,
    calibrated_proxy,
    display_entity,
    evidence_claim,
    infer_significance,
    stable_id,
)
from .temporal import date_sort_key, temporal_distance_days
from ..models.clinical_evidence import ClinicalEvidence
from ..models.clinical_registry import (
    ClinicalEpisode,
    ClinicalEvent,
    EventEvidenceLink,
    EventUpdate,
)


_SYSTEM_TERMS = {
    "respiratorio": {
        "dispnea", "tosse", "iposs", "spo2", "saturazione", "polm",
        "torace", "pleur", "bronch", "ossigen", "addensamento", "tac",
        "tc torace", "respir", "opacit", "vetro smerigliato", "pneumon",
    },
    "tiroideo": {
        "tsh", "ft4", "ft3", "tiroide", "ipotiroid", "ipertiroid",
        "tireotoss", "levotirox",
    },
    "epatico": {
        "ast", "alt", "got", "gpt", "bilirubin", "fegato", "epat",
        "transaminas", "gamma gt",
    },
    "renale": {
        "creatinin", "egfr", "filtrato", "rene", "renal", "proteinuri",
        "ematuri",
    },
    "ematologico": {
        "emoglobin", "anemia", "piastrin", "neutrop", "linfop", "leucop",
        "emocromo", "ematolog",
    },
    "gastrointestinale": {
        "diarrea", "vomito", "nausea", "colite", "addome", "intestin",
        "gastro", "enter",
    },
    "neurologico": {
        "cefalea", "vertigin", "pares", "neuropat", "encefal", "cerebr",
        "confusion", "neurolog",
    },
    "cardiovascolare": {
        "troponina", "cardiac", "cuore", "aritm", "tachic", "bradic",
        "pressione", "ecg", "miocard",
    },
    "cutaneo": {
        "rash", "prurito", "eritem", "cute", "cutane", "dermat",
    },
}

_WINDOWS = {
    "respiratorio": 14,
    "tiroideo": 30,
    "epatico": 21,
    "renale": 21,
    "ematologico": 21,
    "gastrointestinale": 14,
    "neurologico": 14,
    "cardiovascolare": 7,
    "cutaneo": 14,
}

_CLINICAL_CATEGORIES = {
    "symptom", "clinical_sign", "vital_sign", "diagnosis", "toxicity",
    "adverse_event",
}
_OBJECTIVE_CATEGORIES = {
    "laboratory_finding", "imaging_finding", "histopathology", "biomarker",
    "vital_sign",
}


@dataclass(slots=True)
class CorrelationCandidate:
    system: str
    evidence: list[ClinicalEvidence]


class ClinicalCorrelationBuilder:
    """Create reviewable composite notes without asserting causality."""

    def build(
        self, patient_id: str, evidence: Iterable[ClinicalEvidence]
    ) -> list[ConsolidatedBundle]:
        assigned: dict[str, list[ClinicalEvidence]] = defaultdict(list)
        for item in evidence:
            system = clinical_system(item)
            if system:
                assigned[system].append(item)
        candidates: list[CorrelationCandidate] = []
        for system, items in assigned.items():
            items.sort(key=lambda item: date_sort_key(
                item.observed_date or item.document_date
            ))
            current: list[ClinicalEvidence] = []
            anchor_date: str | None = None
            for item in items:
                item_date = item.observed_date or item.document_date
                distance = temporal_distance_days(anchor_date, item_date)
                if current and distance is not None and distance > _WINDOWS[system]:
                    if is_multimodal(current):
                        candidates.append(CorrelationCandidate(system, current))
                    current = []
                    anchor_date = None
                current.append(item)
                if anchor_date is None and item_date:
                    anchor_date = item_date
            if current and is_multimodal(current):
                candidates.append(CorrelationCandidate(system, current))
        return [self._to_bundle(patient_id, candidate) for candidate in candidates]

    @staticmethod
    def _to_bundle(
        patient_id: str, candidate: CorrelationCandidate
    ) -> ConsolidatedBundle:
        items = candidate.evidence
        dates = [
            item.observed_date or item.document_date for item in items
            if item.observed_date or item.document_date
        ]
        first = min(dates, key=date_sort_key) if dates else None
        documented = [item.document_date for item in items if item.document_date]
        first_documented = min(documented, key=date_sort_key) if documented else None
        anchor = first or "unknown"
        episode_id = stable_id(
            "EPI", patient_id, "clinical_syndrome", candidate.system, anchor
        )
        event_id = stable_id(
            "EVT", patient_id, "clinical_syndrome", candidate.system, anchor
        )
        claims = []
        for item in items:
            claim = evidence_claim(item)
            if claim not in claims:
                claims.append(claim)
        joined = "; ".join(claims)
        short = f"Quadro {candidate.system}: {joined}"
        detail = (
            f"{short}. I reperti sono temporalmente associati e il quadro è "
            "compatibile con una possibile manifestazione clinica unitaria; "
            "la relazione fisiopatologica o causale non è dimostrata e "
            "richiede revisione clinica."
        )
        episode = ClinicalEpisode(
            episode_id=episode_id, patient_id=patient_id,
            category="clinical_syndrome",
            canonical_entity=f"quadro_{candidate.system}",
            onset_date=first,
            onset_precision=best_date_precision(items, first),
            first_documented_date=first_documented,
            status="active",
            data={"correlation_window_days": _WINDOWS[candidate.system]},
        )
        event = ClinicalEvent(
            event_id=event_id, patient_id=patient_id, episode_id=episode_id,
            category="clinical_syndrome",
            canonical_entity=f"quadro_{candidate.system}",
            summary_short=short[:500], summary_detail=detail,
            significance=infer_significance(items), status="active",
            certainty="inferred", assertion="present",
            first_evidence_date=first,
            first_documented_date=first_documented,
            date_precision=best_date_precision(items, first),
            confidence=min(0.7, calibrated_proxy(items, [])),
            review_status="pending",
            structured_data={
                "correlation_system": candidate.system,
                "correlation_window_days": _WINDOWS[candidate.system],
                "evidence_count": len(items),
                "source_document_count": len({item.document_id for item in items}),
                "causality_asserted": False,
            },
        )
        links = [
            EventEvidenceLink(
                link_id=stable_id("LNK", event_id, item.evidence_id, "correlated"),
                event_id=event_id, evidence_id=item.evidence_id,
                relation="correlated", relation_confidence=0.6,
                rationale=(
                    f"Stesso sistema clinico e finestra di "
                    f"{_WINDOWS[candidate.system]} giorni"
                ),
                included_in_summary=True,
            )
            for item in items
        ]
        updates = []
        for item in items:
            date = item.observed_date or item.document_date
            if not date or date == first:
                continue
            updates.append(EventUpdate(
                update_id=stable_id("UPD", event_id, date, item.evidence_id),
                event_id=event_id, update_date=date,
                date_precision=item.date_precision,
                summary=evidence_claim(item),
                status_after=item.clinical_status,
                evidence_ids=[item.evidence_id],
            ))
        return ConsolidatedBundle(episode, event, links, updates)


def clinical_system(item: ClinicalEvidence) -> str | None:
    haystack = " ".join((
        item.normalized_entity or "", item.anatomical_site or "",
        item.source_text or "",
    )).casefold()
    scores = {
        system: sum(1 for term in terms if term in haystack)
        for system, terms in _SYSTEM_TERMS.items()
    }
    system, score = max(scores.items(), key=lambda pair: pair[1])
    return system if score else None


def is_multimodal(items: Iterable[ClinicalEvidence]) -> bool:
    evidence = list(items)
    categories = {item.category for item in evidence}
    return bool(categories & _CLINICAL_CATEGORIES) and bool(
        categories & _OBJECTIVE_CATEGORIES
    ) and len(categories) >= 2

"""Episode-first reconciliation over atomic evidence and draft events.

The atomic layer remains immutable.  This module gives one primary event the
ownership of a contextual observation, keeps autonomous treatment/procedure/
response episodes separate, and records explicit links between those episodes.
It is intentionally deterministic: an LLM may enrich a bounded candidate
group later, but cannot silently delete provenance or change ownership.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable

from .consolidation import canonicalize_entity, stable_id
from .correlation import clinical_system
from .temporal import temporal_distance_days
from ..models.clinical_evidence import ClinicalEvidence
from ..models.clinical_registry import (
    ClinicalEventRelation,
    EventEvidenceLink,
)


@dataclass(frozen=True, slots=True)
class EpisodeAssemblyStats:
    contextual_linked: int = 0
    contextual_unassigned: int = 0
    event_relations: int = 0


def _event_date(bundle) -> str | None:
    return bundle.event.first_evidence_date or bundle.event.first_documented_date


def _bundle_evidence(bundle, evidence_by_id) -> list[ClinicalEvidence]:
    return [
        evidence_by_id[link.evidence_id]
        for link in bundle.links if link.evidence_id in evidence_by_id
    ]


def _context_score(
    item: ClinicalEvidence,
    bundle,
    evidence_by_id: dict[str, ClinicalEvidence],
) -> int:
    """Conservative compatibility score for a normal/negative observation."""
    item_entity = canonicalize_entity(item.normalized_entity)
    event_entity = canonicalize_entity(bundle.event.canonical_entity)
    exact_entity = bool(item_entity and item_entity == event_entity)
    item_system = clinical_system(item)
    bundle_items = _bundle_evidence(bundle, evidence_by_id)
    bundle_systems = {
        system for evidence in bundle_items
        if (system := clinical_system(evidence))
    }
    same_system = bool(item_system and item_system in bundle_systems)
    same_document = any(
        evidence.document_id == item.document_id for evidence in bundle_items
    )
    gap = temporal_distance_days(
        item.observed_date or item.document_date, _event_date(bundle)
    )

    score = 0
    if exact_entity:
        score += 7
    if item.category == bundle.event.category:
        score += 2
    if same_system:
        score += 3
    if same_document:
        score += 3
    if gap is not None and gap <= 7:
        score += 2
    elif gap is not None and gap <= 30:
        score += 1

    # A shared organ system alone is too broad (e.g. every thoracic finding).
    if not exact_entity and not (same_system and same_document and score >= 7):
        return 0
    return score


def attach_contextual_evidence(
    bundles: Iterable,
    contextual: Iterable[ClinicalEvidence],
    *,
    evidence_by_id: dict[str, ClinicalEvidence],
) -> tuple[int, int]:
    """Attach normal/negative atoms to one episode without creating rows.

    The selected episode is the atom's primary registry owner.  The link is
    excluded from automatic prose but remains available to source-rich LLM
    queries and to the event Quick View.
    """
    bundles = list(bundles)
    linked = 0
    unassigned = 0
    for item in contextual:
        candidates = [
            (score, bundle)
            for bundle in bundles
            if (score := _context_score(item, bundle, evidence_by_id)) > 0
        ]
        if not candidates:
            unassigned += 1
            continue
        _, owner = max(
            candidates,
            key=lambda pair: (
                pair[0],
                owner_date(pair[1]),
                pair[1].event.event_id,
            ),
        )
        if any(link.evidence_id == item.evidence_id for link in owner.links):
            continue
        relation = (
            "excluded"
            if item.assertion == "absent" or item.certainty == "excluded"
            else "correlated"
        )
        owner.links.append(EventEvidenceLink(
            link_id=stable_id(
                "LNK", owner.event.event_id, item.evidence_id, relation
            ),
            event_id=owner.event.event_id,
            evidence_id=item.evidence_id,
            relation=relation,
            relation_confidence=0.85,
            rationale=(
                "Osservazione normale/negativa contestuale dello stesso "
                "episodio; non genera un evento autonomo"
            ),
            included_in_summary=False,
        ))
        structured = owner.event.structured_data
        structured.setdefault("contextual_evidence_ids", [])
        if item.evidence_id not in structured["contextual_evidence_ids"]:
            structured["contextual_evidence_ids"].append(item.evidence_id)
        structured.setdefault("evidence_ids", [])
        if item.evidence_id not in structured["evidence_ids"]:
            structured["evidence_ids"].append(item.evidence_id)
        linked += 1
    return linked, unassigned


def owner_date(bundle) -> str:
    return _event_date(bundle) or "9999"


_DERIVED_AGGREGATES = {
    "clinical_syndrome", "laboratory_trend", "oncology_treatment_line",
}
_THERAPY = {"medication", "oncology_treatment_line"}
_TOXICITY = {"toxicity", "adverse_event"}


def _relation_type(source_category: str, target_category: str) -> str | None:
    if source_category in _DERIVED_AGGREGATES:
        return "aggregates"
    if target_category in _DERIVED_AGGREGATES:
        return "component_of"
    if source_category in _TOXICITY and target_category in _THERAPY:
        return "possibly_related_to"
    if source_category in _THERAPY and target_category in _TOXICITY:
        return "has_possible_toxicity"
    if source_category in {"response", "progression"}:
        return "updates_problem"
    if source_category in {"procedure", "surgery", "imaging_finding"} and (
        target_category in {"diagnosis", "comorbidity"}
    ):
        return "evaluates_or_treats"
    return None


def build_event_relations(
    patient_id: str,
    bundles: Iterable,
) -> list[ClinicalEventRelation]:
    """Link autonomous episodes only when they share immutable evidence.

    Shared evidence is a strong, auditable blocking rule and avoids generating
    a dense graph merely because two events occur on the same date.
    """
    bundles = list(bundles)
    by_evidence: dict[str, list] = defaultdict(list)
    for bundle in bundles:
        for evidence_id in {
            link.evidence_id for link in bundle.links
        }:
            by_evidence[evidence_id].append(bundle)

    relations: dict[tuple[str, str, str], ClinicalEventRelation] = {}
    for evidence_id, owners in by_evidence.items():
        for source in owners:
            for target in owners:
                if source is target:
                    continue
                relation_type = _relation_type(
                    source.event.category, target.event.category
                )
                if not relation_type:
                    continue
                key = (
                    source.event.event_id,
                    target.event.event_id,
                    relation_type,
                )
                relations.setdefault(key, ClinicalEventRelation(
                    relation_id=stable_id("REL", *key),
                    patient_id=patient_id,
                    source_event_id=source.event.event_id,
                    target_event_id=target.event.event_id,
                    relation_type=relation_type,
                    confidence=0.9,
                    rationale=(
                        "Episodi autonomi collegati dalla medesima evidenza "
                        f"atomica {evidence_id}"
                    ),
                ))
    return list(relations.values())

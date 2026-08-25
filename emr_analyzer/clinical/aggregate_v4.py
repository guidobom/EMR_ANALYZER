"""Fast deterministic front-end for aggregate clinical-event synthesis.

The v4 path deliberately avoids evidence-pair LLM adjudication.  It creates
lossless base episodes from exact clinical identities, lets the existing
bounded episode synthesizer reason over small multi-modal groups, and records
coverage for every eligible atomic fact.  Source evidence is never rewritten.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable

from .consolidation import canonicalize_entity
from .evidence_graph import EvidenceGraphCluster, EvidenceGraphResult
from .temporal import date_sort_key, temporal_distance_days
from ..models.clinical_evidence import ClinicalEvidence


AGGREGATE_V4_VERSION = "aggregate-events-v4.0"

_CATEGORY_ALIASES = {
    "laboratory": "laboratory_finding",
    "treatment": "medication",
    "treatment_interruption": "medication",
    "treatment_completed": "medication",
    "imaging": "imaging_finding",
    "histology": "histopathology",
    "consultation": "care_plan",
}

_ACUTE_WINDOWS = {
    "symptom": 14,
    "clinical_sign": 14,
    "vital_sign": 7,
    "laboratory_finding": 14,
    "imaging_finding": 30,
    "instrumental_finding": 30,
    "histopathology": 30,
    "toxicity": 42,
    "adverse_event": 21,
    "procedure": 3,
    "surgery": 3,
    "hospitalization": 3,
    "discharge": 3,
}

_RESOLUTION_STATUSES = {
    "resolved", "completed", "stopped", "suspended", "cancelled",
    "within_range", "normal", "negative",
}


@dataclass(frozen=True, slots=True)
class AggregationCoverage:
    evidence_id: str
    disposition: str
    status: str
    event_ids: tuple[str, ...] = ()
    reason: str = ""


@dataclass(frozen=True, slots=True)
class AggregationCoverageStats:
    primary_total: int = 0
    primary_linked: int = 0
    primary_residual: int = 0
    contextual_total: int = 0
    contextual_linked: int = 0
    contextual_residual: int = 0


class AggregateV4CandidateBuilder:
    """Create deterministic base episodes without pairwise model calls."""

    def build(
        self,
        patient_id: str,
        evidence: Iterable[ClinicalEvidence],
    ) -> EvidenceGraphResult:
        items = [item for item in evidence if item.patient_id == patient_id]
        groups: dict[tuple[str, str, str, str], list[ClinicalEvidence]] = (
            defaultdict(list)
        )
        for item in items:
            category = _CATEGORY_ALIASES.get(item.category, item.category)
            entity = canonicalize_entity(
                item.canonical_label or item.normalized_entity
            )
            # Site and side are part of identity: a right-sided finding must
            # never be joined to a left-sided one by a shared label alone.
            groups[(
                category,
                entity or f"evidenza_{item.evidence_id}",
                canonicalize_entity(item.anatomical_site or ""),
                canonicalize_entity(item.laterality or ""),
            )].append(item)

        clusters: list[EvidenceGraphCluster] = []
        for (category, _entity, _site, _side), related in sorted(
            groups.items(), key=lambda pair: pair[0]
        ):
            related.sort(key=_sort_key)
            current: list[ClinicalEvidence] = []
            resolved = False
            for item in related:
                if current and _starts_new_episode(
                    category, current[-1], item, resolved=resolved
                ):
                    clusters.append(_cluster(current))
                    current = []
                    resolved = False
                current.append(item)
                if _is_resolution(item):
                    resolved = True
            if current:
                clusters.append(_cluster(current))

        clusters.sort(key=lambda cluster: (
            _sort_key(next(
                item for item in items
                if item.evidence_id == cluster.evidence_ids[0]
            )),
            tuple(cluster.evidence_ids),
        ))
        return EvidenceGraphResult(
            relations=[],
            clusters=clusters,
            candidate_count=len(clusters),
            llm_calls=0,
            split_count=0,
            cache_hits=0,
            auto_resolved_count=len(clusters),
        )


def build_coverage_ledger(
    primary: Iterable[ClinicalEvidence],
    contextual: Iterable[ClinicalEvidence],
    bundles: Iterable,
) -> tuple[list[AggregationCoverage], AggregationCoverageStats]:
    """Account for every registry-eligible atomic fact after aggregation."""
    owners: dict[str, set[str]] = defaultdict(set)
    for bundle in bundles:
        for link in bundle.links:
            if link.relation != "duplicate_source":
                owners[link.evidence_id].add(bundle.event.event_id)

    primary = list(primary)
    contextual = list(contextual)
    rows: list[AggregationCoverage] = []
    for item in primary:
        event_ids = tuple(sorted(owners.get(item.evidence_id, ())))
        rows.append(AggregationCoverage(
            evidence_id=item.evidence_id,
            disposition="primary",
            status="linked" if event_ids else "residual",
            event_ids=event_ids,
            reason=(
                "linked_to_aggregate_event"
                if event_ids else "eligible_primary_without_event"
            ),
        ))
    for item in contextual:
        event_ids = tuple(sorted(owners.get(item.evidence_id, ())))
        rows.append(AggregationCoverage(
            evidence_id=item.evidence_id,
            disposition="contextual",
            status="linked" if event_ids else "residual_context",
            event_ids=event_ids,
            reason=(
                "linked_as_context"
                if event_ids else "no_compatible_existing_episode"
            ),
        ))
    primary_linked = sum(
        row.disposition == "primary" and row.status == "linked" for row in rows
    )
    contextual_linked = sum(
        row.disposition == "contextual" and row.status == "linked"
        for row in rows
    )
    return rows, AggregationCoverageStats(
        primary_total=len(primary),
        primary_linked=primary_linked,
        primary_residual=len(primary) - primary_linked,
        contextual_total=len(contextual),
        contextual_linked=contextual_linked,
        contextual_residual=len(contextual) - contextual_linked,
    )


def _cluster(items: list[ClinicalEvidence]) -> EvidenceGraphCluster:
    anchor = min(items, key=lambda item: (_anchor_rank(item.category), _sort_key(item)))
    return EvidenceGraphCluster(
        evidence_ids=[item.evidence_id for item in items],
        roles={
            item.evidence_id: (
                "core" if item.evidence_id == anchor.evidence_id
                else _role_for(item)
            )
            for item in items
        },
    )


def _starts_new_episode(
    category: str,
    previous: ClinicalEvidence,
    item: ClinicalEvidence,
    *,
    resolved: bool,
) -> bool:
    if resolved and not _is_resolution(item) and item.assertion != "absent":
        return True
    window = _ACUTE_WINDOWS.get(category)
    if window is None:
        return False
    distance = temporal_distance_days(_clinical_date(previous), _clinical_date(item))
    return distance is not None and distance > window


def _is_resolution(item: ClinicalEvidence) -> bool:
    return bool(
        item.assertion == "absent"
        or item.certainty == "excluded"
        or str(item.clinical_status or "").casefold() in _RESOLUTION_STATUSES
    )


def _role_for(item: ClinicalEvidence) -> str:
    if item.assertion == "absent" or item.certainty == "excluded":
        return "negative_context"
    if item.category in {"symptom", "clinical_sign", "vital_sign", "adverse_event"}:
        return "manifestation"
    if item.category in {"histopathology", "imaging_finding", "instrumental_finding"}:
        return "diagnostic_support"
    if item.category in {"medication", "procedure", "surgery"}:
        return "treatment"
    if item.category in {"laboratory_finding", "vital_sign"}:
        return "monitoring"
    if item.category in {"response", "progression", "discharge"}:
        return "outcome"
    return "core"


def _anchor_rank(category: str) -> int:
    return {
        "diagnosis": 0,
        "histopathology": 1,
        "symptom": 2,
        "clinical_sign": 3,
        "imaging_finding": 4,
        "instrumental_finding": 5,
        "laboratory_finding": 6,
        "medication": 7,
        "procedure": 8,
    }.get(category, 20)


def _clinical_date(item: ClinicalEvidence) -> str | None:
    return item.observed_date or item.document_date


def _sort_key(item: ClinicalEvidence):
    return date_sort_key(_clinical_date(item)), item.document_id, item.evidence_id

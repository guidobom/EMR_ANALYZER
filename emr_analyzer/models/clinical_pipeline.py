"""Versioned contracts for the end-to-end clinical pipeline v3."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional
import uuid


PIPELINE_CONTRACT_VERSION = "3.0"

ATOMIC_FACT_TYPES = (
    "medication", "laboratory_test", "radiology_finding",
    "instrumental_finding", "diagnosis", "symptom",
    "clinical_decision", "procedure", "clinical_sign", "vital_sign",
    "histopathology", "biomarker", "hospitalization", "discharge",
)

EVIDENCE_DISPOSITIONS = (
    "accepted_clinical", "accepted_low_relevance",
    "excluded_administrative", "excluded_methodological",
    "excluded_boilerplate", "excluded_non_informative",
    "rejected_unverifiable", "pending_review",
)

EVIDENCE_RELATION_TYPES = (
    "same_occurrence", "same_process", "manifestation_of",
    "progression_of", "response_to", "caused_by", "complication_of",
    "treats", "evaluates", "decision_about", "rules_out", "contradicts",
    "documents_resolution", "temporally_associated_with",
)

CLUSTER_EFFECTS = (
    "must_link", "cohesive", "context_only", "cannot_link", "uncertain",
)

EVENT_EVIDENCE_ROLES = (
    "core", "manifestation", "diagnostic_support", "treatment",
    "monitoring", "outcome", "temporal_anchor", "negative_context",
    "contradiction",
)

HYPOTHESIS_STATUSES = ("needs_review", "accepted", "rejected")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


@dataclass(slots=True)
class EvidenceSourceReference:
    evidence_id: str
    document_id: str
    passage: str
    source_ref_id: str = field(default_factory=lambda: _id("SRC"))
    sentence_refs: list[int] = field(default_factory=list)
    source_page: Optional[int] = None
    bbox: Optional[tuple[float, float, float, float]] = None
    source_role: str = "primary"
    created_at: str = field(default_factory=_now)


@dataclass(slots=True)
class ExcludedEvidence:
    patient_id: str
    document_id: str
    disposition: str
    reason_code: str
    source_text: str
    excluded_id: str = field(default_factory=lambda: _id("EXC"))
    fact_type: Optional[str] = None
    concept: Optional[str] = None
    source_page: Optional[int] = None
    bbox: Optional[tuple[float, float, float, float]] = None
    sentence_refs: list[int] = field(default_factory=list)
    extraction_method: str = "rule"
    model_name: Optional[str] = None
    prompt_version: Optional[str] = None
    review_status: str = "auto"
    data: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=_now)

    def __post_init__(self) -> None:
        if self.disposition not in EVIDENCE_DISPOSITIONS:
            raise ValueError(f"Disposizione non valida: {self.disposition}")


@dataclass(slots=True)
class TerminologyMapping:
    normalized_concept: str
    canonical_label: str
    mapping_id: str = field(default_factory=lambda: _id("MAP"))
    fact_type: Optional[str] = None
    terminology_system: Optional[str] = None
    terminology_code: Optional[str] = None
    original_unit: Optional[str] = None
    canonical_unit: Optional[str] = None
    multiplier: Optional[float] = None
    offset: Optional[float] = None
    mapping_status: str = "candidate"
    mapping_confidence: Optional[float] = None
    source: str = "local_dictionary"
    review_status: str = "auto"
    created_at: str = field(default_factory=_now)
    updated_at: str = field(default_factory=_now)

    def __post_init__(self) -> None:
        if self.mapping_confidence is not None:
            self.mapping_confidence = min(
                1.0, max(0.0, float(self.mapping_confidence))
            )


@dataclass(slots=True)
class EvidenceRelation:
    patient_id: str
    source_evidence_id: str
    target_evidence_id: str
    relation_type: str
    weight: float
    cluster_effect: str
    relation_id: str = field(default_factory=lambda: _id("ERL"))
    direction: str = "directed"
    rationale: str = ""
    rule_features: dict[str, Any] = field(default_factory=dict)
    model_votes: list[dict[str, Any]] = field(default_factory=list)
    generation_method: str = "hybrid"
    review_status: str = "auto"
    created_at: str = field(default_factory=_now)

    def __post_init__(self) -> None:
        if self.relation_type not in EVIDENCE_RELATION_TYPES:
            raise ValueError(f"Relazione non valida: {self.relation_type}")
        if self.cluster_effect not in CLUSTER_EFFECTS:
            raise ValueError(f"Effetto cluster non valido: {self.cluster_effect}")
        self.weight = min(1.0, max(0.0, float(self.weight)))


@dataclass(slots=True)
class EventClaim:
    event_id: str
    text: str
    claim_type: str
    claim_id: str = field(default_factory=lambda: _id("CLM"))
    evidence_ids: list[str] = field(default_factory=list)
    lab_observation_ids: list[int] = field(default_factory=list)
    certainty: str = "confirmed"
    review_status: str = "auto"
    position: int = 0
    created_at: str = field(default_factory=_now)
    updated_at: str = field(default_factory=_now)

    def __post_init__(self) -> None:
        if not self.text.strip():
            raise ValueError("Un claim non può essere vuoto")
        if not self.evidence_ids and not self.lab_observation_ids:
            raise ValueError("Ogni claim deve citare almeno una fonte")


@dataclass(slots=True)
class ClinicalHypothesis:
    patient_id: str
    source_event_id: str
    target_event_id: str
    hypothesis_type: str
    rationale: str
    hypothesis_id: str = field(default_factory=lambda: _id("HYP"))
    strength: str = "weak"
    known_mechanism: bool = False
    confounders: list[str] = field(default_factory=list)
    survival_score: Optional[float] = None
    status: str = "needs_review"
    created_at: str = field(default_factory=_now)
    updated_at: str = field(default_factory=_now)

    def __post_init__(self) -> None:
        if self.status not in HYPOTHESIS_STATUSES:
            raise ValueError(f"Stato ipotesi non valido: {self.status}")
        if self.survival_score is not None:
            self.survival_score = min(
                1.0, max(0.0, float(self.survival_score))
            )

"""Versioned downstream models for the evidence-based clinical registry.

The normalized document text and the deterministic laboratory rows are source
layers and are never rewritten by these models.  A registry event is a
projection over one or more immutable :class:`ClinicalEvidence` records; links
retain supporting, conflicting, excluded and merely correlated evidence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional
import uuid


REGISTRY_SCHEMA_VERSION = "2.0"

EVENT_CATEGORIES = (
    "diagnosis",
    "comorbidity",
    "symptom",
    "clinical_sign",
    "clinical_syndrome",
    "laboratory_finding",
    "laboratory_trend",
    "imaging_finding",
    "histopathology",
    "biomarker",
    "medication",
    "oncology_treatment_line",
    "procedure",
    "surgery",
    "hospitalization",
    "discharge",
    "toxicity",
    "adverse_event",
    "allergy",
    "vaccination",
    "family_history",
    "risk_factor",
    "functional_status",
    "vital_sign",
    "recommendation",
    "care_plan",
    "response",
    "progression",
    "follow_up",
    "other",
)

EVENT_STATUSES = (
    "proposed",
    "planned",
    "active",
    "ongoing",
    "completed",
    "resolved",
    "suspended",
    "cancelled",
    "unknown",
)

CERTAINTY_LEVELS = (
    "confirmed",
    "suspected",
    "patient_reported",
    "inferred",
    "excluded",
    "unknown",
)

ASSERTION_TYPES = (
    "present",
    "absent",
    "conditional",
    "hypothetical",
    "unknown",
)

SIGNIFICANCE_LEVELS = (
    "critical",
    "high",
    "clinically_relevant",
    "potentially_relevant",
    "uncertain",
)

DATE_PRECISIONS = (
    "day",
    "month",
    "year",
    "interval",
    "approximate",
    "unknown",
)

EVIDENCE_RELATIONS = (
    "supports",
    "contradicts",
    "excluded",
    "correlated",
    "updates",
    "duplicate_source",
)

EVENT_RELATIONS = (
    "same_event",
    "updates",
    "correlated",
    "contradicts",
    "distinct_episode",
    "aggregates",
    "component_of",
    "possibly_related_to",
    "has_possible_toxicity",
    "updates_problem",
    "evaluates_or_treats",
    "related_episode",
)

REVIEW_STATUSES = (
    "auto",
    "pending",
    "accepted",
    "corrected",
    "rejected",
    "deferred",
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _stable_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


@dataclass(slots=True)
class ClinicalEpisode:
    patient_id: str
    category: str
    canonical_entity: str
    episode_id: str = field(default_factory=lambda: _stable_id("EPI"))
    onset_date: Optional[str] = None
    onset_date_end: Optional[str] = None
    onset_precision: str = "unknown"
    first_documented_date: Optional[str] = None
    resolution_date: Optional[str] = None
    status: str = "active"
    recurrence_index: int = 1
    previous_episode_id: Optional[str] = None
    data: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=_now)
    updated_at: str = field(default_factory=_now)


@dataclass(slots=True)
class ClinicalEvent:
    patient_id: str
    category: str
    canonical_entity: str
    summary_short: str
    event_id: str = field(default_factory=lambda: _stable_id("EVT"))
    episode_id: Optional[str] = None
    summary_detail: str = ""
    anatomical_site: Optional[str] = None
    laterality: Optional[str] = None
    severity: Optional[str] = None
    significance: str = "clinically_relevant"
    status: str = "active"
    certainty: str = "confirmed"
    assertion: str = "present"
    first_evidence_date: Optional[str] = None
    first_documented_date: Optional[str] = None
    date_end: Optional[str] = None
    date_precision: str = "unknown"
    confidence: Optional[float] = None
    review_status: str = "auto"
    structured_data: dict[str, Any] = field(default_factory=dict)
    model_name: Optional[str] = None
    prompt_version: Optional[str] = None
    schema_version: str = REGISTRY_SCHEMA_VERSION
    version: int = 1
    created_at: str = field(default_factory=_now)
    updated_at: str = field(default_factory=_now)


@dataclass(slots=True)
class EventEvidenceLink:
    event_id: str
    evidence_id: str
    relation: str = "supports"
    relation_confidence: Optional[float] = None
    rationale: str = ""
    included_in_summary: bool = True
    link_id: str = field(default_factory=lambda: _stable_id("LNK"))
    created_at: str = field(default_factory=_now)


@dataclass(slots=True)
class EventUpdate:
    event_id: str
    update_date: Optional[str]
    summary: str
    update_id: str = field(default_factory=lambda: _stable_id("UPD"))
    date_precision: str = "unknown"
    status_after: Optional[str] = None
    evidence_ids: list[str] = field(default_factory=list)
    structured_data: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=_now)


@dataclass(slots=True)
class ClinicalEventRelation:
    patient_id: str
    source_event_id: str
    target_event_id: str
    relation_type: str
    relation_id: str = field(default_factory=lambda: _stable_id("REL"))
    confidence: Optional[float] = None
    rationale: str = ""
    review_status: str = "auto"
    created_at: str = field(default_factory=_now)


@dataclass(slots=True)
class ProcessingRun:
    patient_id: str
    stage: str
    run_id: str = field(default_factory=lambda: _stable_id("RUN"))
    status: str = "running"
    model_name: Optional[str] = None
    model_digest: Optional[str] = None
    prompt_version: Optional[str] = None
    schema_version: str = REGISTRY_SCHEMA_VERSION
    parameters: dict[str, Any] = field(default_factory=dict)
    started_at: str = field(default_factory=_now)
    completed_at: Optional[str] = None
    error_message: Optional[str] = None


@dataclass(slots=True)
class ProcessingManifestItem:
    patient_id: str
    document_id: str
    stage: str
    input_hash: str
    pipeline_version: str
    manifest_id: str = field(default_factory=lambda: _stable_id("MAN"))
    run_id: Optional[str] = None
    prompt_version: Optional[str] = None
    model_digest: Optional[str] = None
    status: str = "pending"
    output_hash: Optional[str] = None
    output_count: int = 0
    attempts: int = 0
    error_message: Optional[str] = None
    processed_at: Optional[str] = None
    created_at: str = field(default_factory=_now)
    updated_at: str = field(default_factory=_now)


@dataclass(slots=True)
class ReviewDecision:
    patient_id: str
    target_type: str
    target_id: str
    decision: str
    decision_id: str = field(default_factory=lambda: _stable_id("REV"))
    reviewer_id: str = "local_user"
    reviewer_role: str = "clinician"
    reason: str = ""
    previous_value: dict[str, Any] = field(default_factory=dict)
    corrected_value: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=_now)


@dataclass(slots=True)
class DocumentTextOverlay:
    patient_id: str
    document_id: str
    corrected_text: str
    base_text_hash: str
    overlay_id: str = field(default_factory=lambda: _stable_id("OVR"))
    version: int = 1
    reason: str = "correzione manuale"
    author_id: str = "local_user"
    status: str = "active"
    created_at: str = field(default_factory=_now)


@dataclass(slots=True)
class MedicationCourse:
    patient_id: str
    normalized_name: str
    course_id: str = field(default_factory=lambda: _stable_id("MED"))
    original_names: list[str] = field(default_factory=list)
    indication: Optional[str] = None
    intent: Optional[str] = None
    lifecycle_status: str = "unknown"
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    dose: Optional[str] = None
    route: Optional[str] = None
    frequency: Optional[str] = None
    adherence: Optional[str] = None
    episode_id: Optional[str] = None
    event_ids: list[str] = field(default_factory=list)
    data: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=_now)
    updated_at: str = field(default_factory=_now)


@dataclass(slots=True)
class OncologyLine:
    patient_id: str
    line_label: str
    line_id: str = field(default_factory=lambda: _stable_id("ONC"))
    regimen: list[str] = field(default_factory=list)
    setting: Optional[str] = None
    intent: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    status: str = "unknown"
    cycles: list[dict[str, Any]] = field(default_factory=list)
    modifications: list[dict[str, Any]] = field(default_factory=list)
    toxicities: list[str] = field(default_factory=list)
    responses: list[str] = field(default_factory=list)
    progression_event_id: Optional[str] = None
    event_ids: list[str] = field(default_factory=list)
    created_at: str = field(default_factory=_now)
    updated_at: str = field(default_factory=_now)


@dataclass(slots=True)
class LabTrend:
    patient_id: str
    normalized_name: str
    summary: str
    trend_id: str = field(default_factory=lambda: _stable_id("TRD"))
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    direction: str = "variable"
    severity: Optional[str] = None
    resolved: bool = False
    lab_value_ids: list[int] = field(default_factory=list)
    event_id: Optional[str] = None
    data: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=_now)
    updated_at: str = field(default_factory=_now)

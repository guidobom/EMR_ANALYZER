"""Human-annotated clinical gold-set domain models."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
import uuid


GOLD_REVIEWER_SLOTS = ("reviewer_a", "reviewer_b", "adjudicated")
GOLD_SPLITS = ("pilot", "development", "test")
GOLD_CASE_STATUSES = (
    "draft", "annotating", "adjudication", "locked", "excluded",
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(slots=True)
class GoldSetCase:
    patient_id: str
    included: bool = False
    split: str = "pilot"
    status: str = "draft"
    reviewer_a_id: str = ""
    reviewer_b_id: str = ""
    adjudicator_id: str = ""
    reviewer_a_status: str = "draft"
    reviewer_b_status: str = "draft"
    notes: str = ""
    locked_at: str | None = None
    created_at: str = field(default_factory=_now)
    updated_at: str = field(default_factory=_now)


@dataclass(slots=True)
class GoldAnnotation:
    patient_id: str
    reviewer_slot: str
    reviewer_id: str
    category: str
    canonical_entity: str
    summary_short: str
    annotation_id: str = field(
        default_factory=lambda: f"GOLD_{uuid.uuid4().hex}"
    )
    summary_detail: str = ""
    first_evidence_date: str | None = None
    first_documented_date: str | None = None
    date_end: str | None = None
    date_precision: str = "unknown"
    status: str = "active"
    certainty: str = "confirmed"
    assertion: str = "present"
    anatomical_site: str | None = None
    laterality: str | None = None
    severity: str | None = None
    significance: str = "clinically_relevant"
    episode_key: str | None = None
    recurrence_index: int = 1
    evidence_ids: list[str] = field(default_factory=list)
    source_refs: list[dict[str, Any]] = field(default_factory=list)
    structured_data: dict[str, Any] = field(default_factory=dict)
    source_annotation_ids: list[str] = field(default_factory=list)
    created_at: str = field(default_factory=_now)
    updated_at: str = field(default_factory=_now)

    def to_evaluation_dict(self) -> dict[str, Any]:
        """Return the event shape accepted by the evaluation runner."""
        return {
            "annotation_id": self.annotation_id,
            "category": self.category,
            "canonical_entity": self.canonical_entity,
            "summary_short": self.summary_short,
            "summary_detail": self.summary_detail,
            "first_evidence_date": self.first_evidence_date,
            "first_documented_date": self.first_documented_date,
            "date_end": self.date_end,
            "date_precision": self.date_precision,
            "status": self.status,
            "certainty": self.certainty,
            "assertion": self.assertion,
            "anatomical_site": self.anatomical_site,
            "laterality": self.laterality,
            "severity": self.severity,
            "significance": self.significance,
            "episode_key": self.episode_key,
            "recurrence_index": self.recurrence_index,
            "evidence_ids": list(self.evidence_ids),
            "source_refs": list(self.source_refs),
            "structured_data": dict(self.structured_data),
        }


@dataclass(slots=True)
class GoldAtomicAnnotation:
    """Gold label for one atomic fact, exclusion or duplicate occurrence."""

    patient_id: str
    document_id: str
    reviewer_slot: str
    reviewer_id: str
    source_text: str
    annotation_id: str = field(
        default_factory=lambda: f"GATOM_{uuid.uuid4().hex}"
    )
    annotation_kind: str = "evidence"
    fact_type: str | None = None
    concept_original: str | None = None
    canonical_label: str | None = None
    observation_date: str | None = None
    date_precision: str = "unknown"
    polarity: str | None = None
    disposition: str = "accepted_clinical"
    exclusion_reason: str | None = None
    source_page: int | None = None
    bbox: tuple[float, float, float, float] | None = None
    sentence_refs: list[int] = field(default_factory=list)
    value: dict[str, Any] = field(default_factory=dict)
    duplicate_of_annotation_id: str | None = None
    status: str = "draft"
    created_at: str = field(default_factory=_now)
    updated_at: str = field(default_factory=_now)

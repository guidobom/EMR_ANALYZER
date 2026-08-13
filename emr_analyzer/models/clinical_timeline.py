"""Clinical Timeline data models — strictly temporal clinical registry."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


# Categories for timeline entries
TIMELINE_CATEGORIES = [
    "diagnosis",
    "treatment",
    "procedure",
    "surgery",
    "toxicity",
    "adverse_event",
    "imaging_finding",
    "laboratory",
    "symptom",
    "hospitalization",
    "discharge",
    "follow_up",
    "other",
]

CATEGORY_LABELS = {
    "diagnosis": "Diagnosi",
    "treatment": "Terapie",
    "procedure": "Procedure",
    "surgery": "Interventi Chirurgici",
    "toxicity": "Tossicità",
    "adverse_event": "Eventi Avversi",
    "imaging_finding": "Imaging",
    "laboratory": "Laboratorio",
    "symptom": "Sintomi",
    "hospitalization": "Ricoveri",
    "discharge": "Dimissioni",
    "follow_up": "Follow-up",
    "other": "Altre Osservazioni",
}

CATEGORY_ORDER = [
    "diagnosis", "treatment", "surgery", "procedure",
    "hospitalization", "discharge", "imaging_finding",
    "laboratory", "toxicity", "adverse_event", "symptom",
    "follow_up", "other",
]


@dataclass
class ClinicalTimelineEntry:
    """A single temporally-anchored clinical observation in the registry."""

    entry_id: str                           # CTL_000001, CTL_000002, ...
    patient_id: str
    date_observed: str                      # YYYY-MM-DD or YYYY-MM
    date_resolved: Optional[str] = None     # YYYY-MM-DD if known
    category: str = "other"                 # from TIMELINE_CATEGORIES
    description: str = ""                   # concise clinical description
    source_document_ids: list[str] = field(default_factory=list)
    source_texts: list[str] = field(default_factory=list)
    merged_into_ids: list[str] = field(default_factory=list)
    status: str = "active"                  # active, resolved, superseded
    confidence: float = 0.5                 # 0.0-1.0
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict:
        return {
            "entry_id": self.entry_id,
            "patient_id": self.patient_id,
            "date_observed": self.date_observed,
            "date_resolved": self.date_resolved,
            "category": self.category,
            "description": self.description,
            "source_document_ids": self.source_document_ids,
            "source_texts": self.source_texts,
            "merged_into_ids": self.merged_into_ids,
            "status": self.status,
            "confidence": self.confidence,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ClinicalTimelineEntry":
        return cls(
            entry_id=data.get("entry_id", ""),
            patient_id=data.get("patient_id", ""),
            date_observed=data.get("date_observed", ""),
            date_resolved=data.get("date_resolved"),
            category=data.get("category", "other"),
            description=data.get("description", ""),
            source_document_ids=data.get("source_document_ids", []),
            source_texts=data.get("source_texts", []),
            merged_into_ids=data.get("merged_into_ids", []),
            status=data.get("status", "active"),
            confidence=data.get("confidence", 0.5),
            created_at=data.get("created_at", ""),
            updated_at=data.get("updated_at", ""),
        )

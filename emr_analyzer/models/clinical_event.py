"""Clinical event data models (Event Store)."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional


class EventType(str, Enum):
    DIAGNOSIS = "diagnosis"
    PROGRESSION = "progression"
    RESPONSE = "response"
    TREATMENT_STARTED = "treatment_started"
    TREATMENT_ENDED = "treatment_ended"
    TREATMENT_MODIFIED = "treatment_modified"
    HOSPITALIZATION = "hospitalization"
    DISCHARGE = "discharge"
    PROCEDURE = "procedure"
    SURGERY = "surgery"
    TOXICITY = "toxicity"
    ADVERSE_EVENT = "adverse_event"
    SYMPTOM = "symptom"
    LAB_ALTERATION = "lab_alteration"
    RADIOLOGY_FINDING = "radiology_finding"
    CONSULTATION = "consultation"
    DEATH = "death"
    FOLLOW_UP = "follow_up"
    PRESCRIPTION = "prescription"
    OTHER = "other"


class EventStatus(str, Enum):
    PROPOSED = "proposed"
    CONFIRMED = "confirmed"
    REJECTED = "rejected"
    CORRECTED = "corrected"


@dataclass
class ClinicalEvent:
    """A single clinical observation extracted from a document."""
    event_id: str                            # EVT_004521
    patient_id: str
    event_date: str
    event_type: str                          # EventType value
    entity: str                              # e.g., "nivolumab", "carcinoma polmonare"
    value: Optional[float] = None
    unit: Optional[str] = None
    status: str = EventStatus.PROPOSED.value
    source_document_id: str = ""
    page: Optional[int] = None
    source_text: str = ""
    confidence: float = 0.0
    validated_by_user: bool = False
    validated_at: Optional[str] = None
    user_notes: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict:
        return {
            "event_id": self.event_id,
            "patient_id": self.patient_id,
            "event_date": self.event_date,
            "event_type": self.event_type,
            "entity": self.entity,
            "value": self.value,
            "unit": self.unit,
            "status": self.status,
            "source_document_id": self.source_document_id,
            "page": self.page,
            "source_text": self.source_text,
            "confidence": self.confidence,
            "validated_by_user": self.validated_by_user,
            "validated_at": self.validated_at,
            "user_notes": self.user_notes,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ClinicalEvent":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

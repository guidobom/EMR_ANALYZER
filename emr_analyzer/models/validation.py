"""Validation queue data model."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional


class ValidationItemType(str, Enum):
    EVENT = "event"
    LAB_VALUE = "lab_value"
    DELTA = "delta"
    DOCUMENT_TYPE = "document_type"


class ValidationStatus(str, Enum):
    PENDING = "pending"
    ACCEPTED = "accepted"
    CORRECTED = "corrected"
    REJECTED = "rejected"
    DEFERRED = "deferred"


class Severity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass
class ValidationItem:
    """An item awaiting human review in the validation queue."""
    id: Optional[int] = None
    patient_id: str = ""
    item_type: str = ValidationItemType.EVENT.value
    item_id: str = ""
    issue: str = ""                          # Description of what needs validation
    severity: str = Severity.MEDIUM.value
    status: str = ValidationStatus.PENDING.value
    original_value: Optional[str] = None     # JSON: the original data
    corrected_value: Optional[str] = None    # JSON: the corrected data
    user_notes: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    resolved_at: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "patient_id": self.patient_id,
            "item_type": self.item_type,
            "item_id": self.item_id,
            "issue": self.issue,
            "severity": self.severity,
            "status": self.status,
            "original_value": self.original_value,
            "corrected_value": self.corrected_value,
            "user_notes": self.user_notes,
            "created_at": self.created_at,
            "resolved_at": self.resolved_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ValidationItem":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

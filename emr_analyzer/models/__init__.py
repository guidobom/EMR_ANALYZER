"""Patient and Workspace data models."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

# Document projection types are imported at the end of this module to keep the
# long-standing Patient/Workspace public API intact.


@dataclass
class Patient:
    """Patient record with pseudonymized identity."""
    id: str                              # P001, P002, ...
    pseudonym: str
    initials: Optional[str] = None       # AB, MR, ...
    sex: Optional[str] = None            # M, F, Other
    birth_year: Optional[int] = None
    main_pathology: Optional[str] = None
    center: Optional[str] = None
    notes: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "pseudonym": self.pseudonym,
            "initials": self.initials,
            "sex": self.sex,
            "birth_year": self.birth_year,
            "main_pathology": self.main_pathology,
            "center": self.center,
            "notes": self.notes,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Patient":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class Workspace:
    """Patient workspace metadata."""
    patient_id: str
    workspace_path: str
    document_count: int = 0
    admission_count: int = 0
    event_count: int = 0
    lab_value_count: int = 0
    last_document_date: Optional[str] = None
    last_update: str = field(default_factory=lambda: datetime.now().isoformat())
    processing_status: str = "idle"      # idle, processing, error

    def to_dict(self) -> dict:
        return {
            "patient_id": self.patient_id,
            "workspace_path": self.workspace_path,
            "document_count": self.document_count,
            "admission_count": self.admission_count,
            "event_count": self.event_count,
            "lab_value_count": self.lab_value_count,
            "last_document_date": self.last_document_date,
            "last_update": self.last_update,
            "processing_status": self.processing_status,
        }


from .document_projection import (  # noqa: E402
    DocumentClinicalProjection,
    DocumentObservation,
    DocumentObservationRelationship,
    DocumentProjectionConflict,
    DocumentSourceSpan,
)

__all__ = [
    "Patient", "Workspace", "DocumentClinicalProjection",
    "DocumentObservation", "DocumentObservationRelationship",
    "DocumentProjectionConflict", "DocumentSourceSpan",
]

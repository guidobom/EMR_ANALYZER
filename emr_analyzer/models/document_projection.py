"""Document-level clinical projection built from atomic evidence."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional
import uuid


@dataclass
class DocumentSourceSpan:
    evidence_id: str
    page: Optional[int]
    text: str
    bbox: Optional[tuple[float, float, float, float]] = None

    def to_dict(self) -> dict:
        return {
            "evidence_id": self.evidence_id,
            "page": self.page,
            "text": self.text,
            "bbox": list(self.bbox) if self.bbox else None,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "DocumentSourceSpan":
        return cls(
            evidence_id=data.get("evidence_id", ""),
            page=data.get("page"),
            text=data.get("text", ""),
            bbox=tuple(data["bbox"]) if data.get("bbox") else None,
        )


@dataclass
class DocumentObservation:
    category: str
    normalized_entity: str
    evidence_ids: list[str]
    source_spans: list[DocumentSourceSpan]
    observation_id: str = field(
        default_factory=lambda: f"DOBS_{uuid.uuid4().hex}"
    )
    observed_start_date: Optional[str] = None
    observed_end_date: Optional[str] = None
    assertions: list[str] = field(default_factory=list)
    temporalities: list[str] = field(default_factory=list)
    clinical_status: Optional[str] = None
    value_text: Optional[str] = None
    numeric_value: Optional[float] = None
    unit: Optional[str] = None
    confidence: float = 0.0
    requires_review: bool = False
    data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        result = dict(self.__dict__)
        result["source_spans"] = [span.to_dict() for span in self.source_spans]
        return result

    @classmethod
    def from_dict(cls, data: dict) -> "DocumentObservation":
        fields = dict(data)
        fields["source_spans"] = [
            DocumentSourceSpan.from_dict(span)
            for span in data.get("source_spans", [])
        ]
        return cls(**{
            key: value for key, value in fields.items()
            if key in cls.__dataclass_fields__
        })


@dataclass
class DocumentObservationRelationship:
    from_observation_id: str
    to_observation_id: str
    relationship_type: str
    evidence_ids: list[str] = field(default_factory=list)
    confidence: float = 0.0

    def to_dict(self) -> dict:
        return dict(self.__dict__)


@dataclass
class DocumentProjectionConflict:
    evidence_ids: list[str]
    conflict_type: str
    requires_review: bool = True

    def to_dict(self) -> dict:
        return dict(self.__dict__)


@dataclass
class DocumentClinicalProjection:
    patient_id: str
    document_id: str
    observations: list[DocumentObservation]
    projection_id: str = field(
        default_factory=lambda: f"DPROJ_{uuid.uuid4().hex}"
    )
    document_date: Optional[str] = None
    document_type: Optional[str] = None
    relationships: list[DocumentObservationRelationship] = field(
        default_factory=list
    )
    conflicts: list[DocumentProjectionConflict] = field(default_factory=list)
    consolidation_method: str = "deterministic"
    model_name: Optional[str] = None
    prompt_version: Optional[str] = None
    schema_version: str = "1.0"
    status: str = "proposed"
    warnings: list[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())

    @property
    def evidence_ids(self) -> list[str]:
        return list(dict.fromkeys(
            evidence_id
            for observation in self.observations
            for evidence_id in observation.evidence_ids
        ))

    def to_dict(self) -> dict:
        return {
            "projection_id": self.projection_id,
            "patient_id": self.patient_id,
            "document_id": self.document_id,
            "document_date": self.document_date,
            "document_type": self.document_type,
            "observations": [item.to_dict() for item in self.observations],
            "relationships": [item.to_dict() for item in self.relationships],
            "conflicts": [item.to_dict() for item in self.conflicts],
            "evidence_count": len(self.evidence_ids),
            "observation_count": len(self.observations),
            "consolidation_method": self.consolidation_method,
            "model_name": self.model_name,
            "prompt_version": self.prompt_version,
            "schema_version": self.schema_version,
            "status": self.status,
            "warnings": self.warnings,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "DocumentClinicalProjection":
        return cls(
            projection_id=data.get("projection_id") or f"DPROJ_{uuid.uuid4().hex}",
            patient_id=data.get("patient_id", ""),
            document_id=data.get("document_id", ""),
            document_date=data.get("document_date"),
            document_type=data.get("document_type"),
            observations=[
                DocumentObservation.from_dict(item)
                for item in data.get("observations", [])
            ],
            relationships=[
                DocumentObservationRelationship(**{
                    key: value for key, value in item.items()
                    if key in DocumentObservationRelationship.__dataclass_fields__
                })
                for item in data.get("relationships", [])
            ],
            conflicts=[
                DocumentProjectionConflict(**{
                    key: value for key, value in item.items()
                    if key in DocumentProjectionConflict.__dataclass_fields__
                })
                for item in data.get("conflicts", [])
            ],
            consolidation_method=data.get("consolidation_method", "deterministic"),
            model_name=data.get("model_name"),
            prompt_version=data.get("prompt_version"),
            schema_version=data.get("schema_version", "1.0"),
            status=data.get("status", "proposed"),
            warnings=data.get("warnings", []),
            created_at=data.get("created_at") or datetime.now().isoformat(),
            updated_at=data.get("updated_at") or datetime.now().isoformat(),
        )

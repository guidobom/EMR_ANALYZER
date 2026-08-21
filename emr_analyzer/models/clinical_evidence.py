"""Normalized clinical evidence linked to exact document provenance."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional
import uuid


@dataclass
class ClinicalEvidence:
    patient_id: str
    document_id: str
    category: str
    normalized_entity: str
    source_text: str
    evidence_id: str = field(default_factory=lambda: f"EVD_{uuid.uuid4().hex}")
    assertion: str = "present"
    temporality: str = "current"
    clinical_status: Optional[str] = None
    observed_date: Optional[str] = None
    observed_date_end: Optional[str] = None
    document_date: Optional[str] = None
    date_precision: str = "unknown"
    date_source: Optional[str] = None
    anatomical_site: Optional[str] = None
    laterality: Optional[str] = None
    severity: Optional[str] = None
    significance: str = "clinically_relevant"
    certainty: str = "confirmed"
    value_text: Optional[str] = None
    numeric_value: Optional[float] = None
    unit: Optional[str] = None
    source_page: Optional[int] = None
    bbox: Optional[tuple[float, float, float, float]] = None
    confidence: float = 0.0
    extraction_method: str = "llm"
    model_name: Optional[str] = None
    prompt_version: Optional[str] = None
    schema_version: str = "2.0"
    status: str = "proposed"
    data: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict:
        result = dict(self.__dict__)
        result["bbox"] = list(self.bbox) if self.bbox else None
        return result

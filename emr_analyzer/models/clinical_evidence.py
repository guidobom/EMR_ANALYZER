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
    fact_type: Optional[str] = None
    concept_original: Optional[str] = None
    canonical_label: Optional[str] = None
    terminology_system: Optional[str] = None
    terminology_code: Optional[str] = None
    mapping_status: str = "unmapped"
    mapping_confidence: Optional[float] = None
    clinical_relevance: str = "accepted_clinical"
    typed_payload: dict[str, Any] = field(default_factory=dict)
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
    schema_version: str = "3.0"
    status: str = "proposed"
    data: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict:
        result = dict(self.__dict__)
        result["bbox"] = list(self.bbox) if self.bbox else None
        return result

    def to_atomic_dict(self) -> dict:
        """Return the explicit post-extraction atomic evidence contract."""
        fact_type = self.fact_type or self.data.get("fact_type") or {
            "laboratory_finding": "laboratory_test",
            "imaging_finding": "radiology_finding",
            "care_plan": "clinical_decision",
            "recommendation": "clinical_decision",
        }.get(self.category, self.category)
        polarity = self.data.get("polarity") or (
            "suspected" if self.certainty == "suspected"
            else "negated" if (
                self.assertion == "absent" or self.certainty == "excluded"
            ) else "present"
        )
        source_reference = self.data.get("source_reference") or {
            "document_id": self.document_id,
            "page": self.source_page,
            "bbox": list(self.bbox) if self.bbox else None,
            "passage": self.source_text,
        }
        return {
            "evidence_id": self.evidence_id,
            "patient_id": self.patient_id,
            "document_id": self.document_id,
            "category": self.category,
            "fact_type": fact_type,
            "concept": self.concept_original or self.normalized_entity,
            "canonical_label": self.canonical_label,
            "terminology_system": self.terminology_system,
            "terminology_code": self.terminology_code,
            "mapping_status": self.mapping_status,
            "mapping_confidence": self.mapping_confidence,
            "clinical_relevance": self.clinical_relevance,
            "value_text": self.value_text,
            "numeric_value": self.numeric_value,
            "unit": self.unit,
            "observation_date": self.observed_date,
            "observation_date_end": self.observed_date_end,
            "report_date": self.document_date,
            "date_precision": self.date_precision,
            "date_source": self.date_source,
            "polarity": polarity,
            "assertion": self.assertion,
            "temporality": self.temporality,
            "clinical_status": self.clinical_status,
            "anatomical_site": self.anatomical_site,
            "laterality": self.laterality,
            "severity": self.severity,
            "significance": self.significance,
            "certainty": self.certainty,
            "confidence": self.confidence,
            "review_status": self.status,
            "source_reference": source_reference,
            "typed_payload": dict(self.typed_payload),
            "extraction_method": self.extraction_method,
            "schema_version": self.schema_version,
        }

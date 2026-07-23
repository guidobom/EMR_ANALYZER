"""Contracts for incremental, bitemporal Clinical State reconstruction."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
import uuid


PROMPT_VERSION = "longitudinal_delta_v1"
SCHEMA_VERSION = "1.0"

OPERATIONS = (
    "new", "reaffirm", "evolution", "correction", "resolution",
    "conflict", "no_change",
)

CATEGORIES = (
    "diagnosis", "symptom", "clinical_sign", "specialist_assessment",
    "medication", "oncologic_treatment", "procedure", "surgery",
    "hospitalization", "radiology_finding", "laboratory_finding",
    "pathology_finding", "biomarker", "toxicity", "adverse_event",
    "allergy", "comorbidity", "functional_status", "staging",
    "response_assessment", "follow_up", "other",
)


def _nullable(kind: str) -> dict:
    return {"type": [kind, "null"]}


OPERATION_JSON_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "operation", "target_evidence_id", "category",
        "normalized_entity", "assertion", "clinical_status",
        "valid_start_date", "valid_end_date", "date_precision",
        "original_date_expression", "anchor_date",
        "date_derivation_rule", "value_text", "numeric_value", "unit",
        "grade", "grading_system", "source_page", "source_text",
        "confidence", "objective_basis", "correction_explicit",
        "rationale",
    ],
    "properties": {
        "operation": {"type": "string", "enum": list(OPERATIONS)},
        "target_evidence_id": _nullable("string"),
        "category": {"type": "string", "enum": list(CATEGORIES)},
        "normalized_entity": {"type": "string"},
        "assertion": {
            "type": "string",
            "enum": ["present", "absent", "suspected", "historical"],
        },
        "clinical_status": _nullable("string"),
        "valid_start_date": _nullable("string"),
        "valid_end_date": _nullable("string"),
        "date_precision": {
            "type": "string",
            "enum": ["exact", "partial", "estimated", "unknown"],
        },
        "original_date_expression": _nullable("string"),
        "anchor_date": _nullable("string"),
        "date_derivation_rule": _nullable("string"),
        "value_text": _nullable("string"),
        "numeric_value": _nullable("number"),
        "unit": _nullable("string"),
        "grade": _nullable("integer"),
        "grading_system": _nullable("string"),
        "source_page": _nullable("integer"),
        "source_text": {"type": "string"},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "objective_basis": {"type": "boolean"},
        "correction_explicit": {"type": "boolean"},
        "rationale": {"type": "string"},
    },
}

DOCUMENT_DELTA_JSON_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["operations", "document_summary", "warnings"],
    "properties": {
        "operations": {
            "type": "array",
            "items": OPERATION_JSON_SCHEMA,
        },
        "document_summary": {"type": "string"},
        "warnings": {"type": "array", "items": {"type": "string"}},
    },
}


@dataclass
class LongitudinalEvidence:
    patient_id: str
    document_id: str
    run_id: str
    operation: str
    category: str
    normalized_entity: str
    source_text: str
    asserted_at: str
    evidence_id: str = field(
        default_factory=lambda: f"LEV_{uuid.uuid4().hex.upper()}"
    )
    target_evidence_id: str | None = None
    assertion: str = "present"
    clinical_status: str | None = None
    valid_start_date: str | None = None
    valid_end_date: str | None = None
    date_precision: str = "unknown"
    original_date_expression: str | None = None
    anchor_date: str | None = None
    date_derivation_rule: str | None = None
    value_text: str | None = None
    numeric_value: float | None = None
    unit: str | None = None
    grade: int | None = None
    grading_system: str | None = None
    source_page: int | None = None
    confidence: float = 0.0
    objective_basis: bool = False
    correction_explicit: bool = False
    canonical_status: str = "active"
    model_name: str = ""
    prompt_version: str = PROMPT_VERSION
    schema_version: str = SCHEMA_VERSION
    data: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    @classmethod
    def from_operation(
        cls,
        payload: dict,
        *,
        patient_id: str,
        document_id: str,
        run_id: str,
        asserted_at: str,
        model_name: str,
    ) -> "LongitudinalEvidence":
        return cls(
            patient_id=patient_id,
            document_id=document_id,
            run_id=run_id,
            operation=str(payload.get("operation") or "new"),
            target_evidence_id=payload.get("target_evidence_id"),
            category=str(payload.get("category") or "other"),
            normalized_entity=str(payload.get("normalized_entity") or "").strip(),
            assertion=str(payload.get("assertion") or "present"),
            clinical_status=payload.get("clinical_status"),
            valid_start_date=payload.get("valid_start_date"),
            valid_end_date=payload.get("valid_end_date"),
            asserted_at=asserted_at,
            date_precision=str(payload.get("date_precision") or "unknown"),
            original_date_expression=payload.get("original_date_expression"),
            anchor_date=payload.get("anchor_date"),
            date_derivation_rule=payload.get("date_derivation_rule"),
            value_text=payload.get("value_text"),
            numeric_value=payload.get("numeric_value"),
            unit=payload.get("unit"),
            grade=payload.get("grade"),
            grading_system=payload.get("grading_system"),
            source_page=payload.get("source_page"),
            source_text=str(payload.get("source_text") or "").strip(),
            confidence=float(payload.get("confidence") or 0.0),
            objective_basis=bool(payload.get("objective_basis")),
            correction_explicit=bool(payload.get("correction_explicit")),
            model_name=model_name,
            data={"rationale": payload.get("rationale") or ""},
        )

    def to_dict(self) -> dict:
        return {
            key: getattr(self, key)
            for key in self.__dataclass_fields__
        }


@dataclass
class ModelCapabilityResult:
    model_name: str
    available: bool
    json_schema_supported: bool
    clinical_delta_supported: bool
    message: str
    checked_at: str = field(default_factory=lambda: datetime.now().isoformat())
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def usable(self) -> bool:
        return (
            self.available
            and self.json_schema_supported
            and self.clinical_delta_supported
        )

    def to_dict(self) -> dict:
        return {
            "model_name": self.model_name,
            "available": self.available,
            "json_schema_supported": self.json_schema_supported,
            "clinical_delta_supported": self.clinical_delta_supported,
            "usable": self.usable,
            "message": self.message,
            "checked_at": self.checked_at,
            "details": self.details,
        }

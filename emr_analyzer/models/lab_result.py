"""Lab result data models."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class LabValue:
    """A single lab test result."""
    patient_id: str
    document_id: str
    parameter_name: str               # Original name from the report
    normalized_name: str              # Canonical name (after synonym mapping)
    value: Optional[float] = None     # Numeric result (None when result is textual)
    value_text: Optional[str] = None  # Textual result (e.g. "NEGATIVO", "POSITIVO")
    operator: Optional[str] = None    # <, >, <=, >=
    unit: Optional[str] = None
    reference_low: Optional[float] = None
    reference_high: Optional[float] = None
    reference_text: str = ""          # Original reference range text
    is_abnormal: bool = False
    flag: Optional[str] = None        # H, L, *, !
    sample_date: Optional[str] = None
    biological_material: Optional[str] = None
    lab_name: Optional[str] = None
    page: Optional[int] = None
    source_text: str = ""
    confidence: float = 1.0
    validated_by_user: bool = False

    def to_dict(self) -> dict:
        return {
            "patient_id": self.patient_id,
            "document_id": self.document_id,
            "parameter_name": self.parameter_name,
            "normalized_name": self.normalized_name,
            "value": self.value,
            "value_text": self.value_text,
            "operator": self.operator,
            "unit": self.unit,
            "reference_low": self.reference_low,
            "reference_high": self.reference_high,
            "reference_text": self.reference_text,
            "is_abnormal": self.is_abnormal,
            "flag": self.flag,
            "sample_date": self.sample_date,
            "biological_material": self.biological_material,
            "lab_name": self.lab_name,
            "page": self.page,
            "source_text": self.source_text,
            "confidence": self.confidence,
            "validated_by_user": self.validated_by_user,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "LabValue":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class LabTimepoint:
    """A lab value at a specific point in time, for temporal series."""
    parameter_name: str
    normalized_name: str
    value: Optional[float] = None     # Numeric result (None when result is textual)
    value_text: Optional[str] = None  # Textual result (e.g. "NEGATIVO")
    unit: Optional[str] = None
    date: str = ""
    reference_low: Optional[float] = None
    reference_high: Optional[float] = None
    is_abnormal: bool = False
    document_id: str = ""
    source_text: str = ""
    page: Optional[int] = None

    def to_dict(self) -> dict:
        return {
            "parameter_name": self.parameter_name,
            "normalized_name": self.normalized_name,
            "value": self.value,
            "value_text": self.value_text,
            "unit": self.unit,
            "date": self.date,
            "reference_low": self.reference_low,
            "reference_high": self.reference_high,
            "is_abnormal": self.is_abnormal,
            "document_id": self.document_id,
            "source_text": self.source_text,
            "page": self.page,
        }


@dataclass
class LabTemporalSeries:
    """Complete temporal series for a single lab parameter."""
    normalized_name: str
    parameter_names: list[str]        # All original names mapped to this
    timepoints: list[LabTimepoint] = field(default_factory=list)
    unit: Optional[str] = None
    reference_low: Optional[float] = None
    reference_high: Optional[float] = None

    @property
    def baseline(self) -> Optional[float]:
        """First value in the series."""
        if self.timepoints:
            return self.timepoints[0].value
        return None

    @property
    def min_value(self) -> Optional[float]:
        numeric = [t.value for t in self.timepoints if t.value is not None]
        if numeric:
            return min(numeric)
        return None

    @property
    def max_value(self) -> Optional[float]:
        numeric = [t.value for t in self.timepoints if t.value is not None]
        if numeric:
            return max(numeric)
        return None

    @property
    def absolute_change(self) -> Optional[float]:
        """Absolute change from first to last."""
        if len(self.timepoints) >= 2:
            first, last = self.timepoints[0].value, self.timepoints[-1].value
            if first is not None and last is not None:
                return last - first
        return None

    @property
    def percent_change(self) -> Optional[float]:
        """Percent change from first to last."""
        if len(self.timepoints) >= 2:
            first, last = self.timepoints[0].value, self.timepoints[-1].value
            if first is not None and last is not None and first != 0:
                return ((last - first) / abs(first)) * 100
        return None

    @property
    def abnormal_count(self) -> int:
        return sum(1 for t in self.timepoints if t.is_abnormal)

    def to_dict(self) -> dict:
        return {
            "normalized_name": self.normalized_name,
            "parameter_names": self.parameter_names,
            "timepoints": [t.to_dict() for t in self.timepoints],
            "unit": self.unit,
            "reference_low": self.reference_low,
            "reference_high": self.reference_high,
        }

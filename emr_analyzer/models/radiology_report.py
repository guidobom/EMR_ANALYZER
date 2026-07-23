"""Radiology report data model."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class RadiologyReport:
    """Structured extraction from a radiology report."""
    technique: Optional[str] = None          # Exam technique / modality
    findings: Optional[str] = None           # Radiological findings
    conclusions: Optional[str] = None        # Diagnostic conclusions / impression
    full_text: str = ""
    exam_type: Optional[str] = None          # TC, RM, RX, ECO, etc.
    body_region: Optional[str] = None        # Thorax, Abdomen, etc.
    contrast_used: bool = False
    comparison_available: bool = False       # Whether comparison with prior is mentioned
    document_id: str = ""
    page_start: Optional[int] = None
    page_end: Optional[int] = None

    def to_dict(self) -> dict:
        return {
            "technique": self.technique,
            "findings": self.findings,
            "conclusions": self.conclusions,
            "full_text": self.full_text,
            "exam_type": self.exam_type,
            "body_region": self.body_region,
            "contrast_used": self.contrast_used,
            "comparison_available": self.comparison_available,
            "document_id": self.document_id,
            "page_start": self.page_start,
            "page_end": self.page_end,
        }

"""Shared result types without model or prompt initialization."""
from dataclasses import dataclass, field
from ..pipeline.sensitive_data import DEIDENTIFICATION_VERSION

@dataclass
class ClinicalTextIsolationResult:
    text: str
    model_name: str
    prompt_version: str = "unspecified"
    chunk_count: int = 0
    warnings: list[str] = field(default_factory=list)
    redaction_counts: dict[str, int] = field(default_factory=dict)
    deidentification_version: str = DEIDENTIFICATION_VERSION
    retention_audit: dict = field(default_factory=dict)


class ClinicalTextIsolationError(RuntimeError):
    """The document model did not produce a complete safe text."""

    def __init__(self, message: str, *, systemic: bool = False):
        super().__init__(message)
        self.systemic = systemic

"""Patient identity evidence extracted from report headers.

The raw values live only in memory while an import batch is routed.  The
database stores keyed fingerprints plus provenance, so the workspace code
(``P001``, ``P002``...) remains independent from directly identifying data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class IdentityField:
    """One identity value and the exact PDF evidence supporting it."""

    value: str
    normalized: str
    page: int = 1
    bbox: Optional[tuple[float, float, float, float]] = None
    method: str = "native_text"
    confidence: float = 0.0


@dataclass
class PatientIdentityEvidence:
    """Identity evidence collected from the first page of one document."""

    source_path: str
    name: Optional[IdentityField] = None
    fiscal_code: Optional[IdentityField] = None
    birth_date: Optional[IdentityField] = None
    sex: Optional[IdentityField] = None
    hospital_patient_id: Optional[IdentityField] = None
    extraction_method: str = "native_text"
    warnings: list[str] = field(default_factory=list)

    @property
    def fields(self) -> dict[str, IdentityField]:
        return {
            key: value
            for key, value in {
                "name": self.name,
                "fiscal_code": self.fiscal_code,
                "birth_date": self.birth_date,
                "sex": self.sex,
                "hospital_patient_id": self.hospital_patient_id,
            }.items()
            if value is not None
        }

    @property
    def confidence(self) -> float:
        """Confidence used for automatic routing, not clinical extraction."""

        # Strong per-hospital identifier, treated like the fiscal code: a
        # stable anchor that uniquely names a patient within a health board.
        def strong() -> Optional[IdentityField]:
            if self.fiscal_code:
                return self.fiscal_code
            if self.hospital_patient_id:
                return self.hospital_patient_id
            return None

        anchor = strong()
        if anchor and (self.name or self.birth_date):
            return min(
                anchor.confidence,
                max(
                    self.name.confidence if self.name else 0.0,
                    self.birth_date.confidence if self.birth_date else 0.0,
                ),
            )
        if self.name and self.birth_date:
            return min(self.name.confidence, self.birth_date.confidence)
        if anchor:
            return min(anchor.confidence, 0.94)
        if self.name:
            return min(self.name.confidence, 0.70)
        return 0.0

    @property
    def is_strong(self) -> bool:
        """Whether the evidence is sufficient for an automatic assignment."""

        return bool(
            (self.fiscal_code and (self.name or self.birth_date))
            or (self.name and self.birth_date)
            or self.hospital_patient_id
        ) and self.confidence >= 0.80

    @property
    def group_key(self) -> Optional[str]:
        """In-memory batch key. Raw values are never persisted as this key."""

        if self.fiscal_code:
            return f"cf:{self.fiscal_code.normalized}"
        if self.hospital_patient_id:
            return f"hpid:{self.hospital_patient_id.normalized}"
        if self.name and self.birth_date:
            canonical_name = " ".join(sorted(self.name.normalized.split()))
            return f"name_birth:{canonical_name}|{self.birth_date.normalized}"
        return None

    def masked_label(self) -> str:
        """Short label for the local routing preview without showing the CF."""

        if self.name:
            words = self.name.value.split()
            initials = " ".join(f"{word[0].upper()}." for word in words if word)
            if initials:
                return initials
        return "identità non completa"

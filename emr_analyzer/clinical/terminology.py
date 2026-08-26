"""Deterministic terminology and UCUM-like unit resolution after extraction."""

from __future__ import annotations

from dataclasses import asdict
import re
import unicodedata

from ..extraction.normalizer import LabNormalizer
from ..models.clinical_pipeline import TerminologyMapping


# Conservative seed set: a code is emitted only for canonical analytes whose
# specimen/property are sufficiently constrained by the local lab parser.  A
# project mapping accepted by a reviewer always takes precedence.
_LOINC_LAB = {
    "emoglobina": (
        "Hemoglobin [Mass/volume] in Blood", "718-7",
        {"g/dL", "g/L"},
    ),
    "globuli_bianchi": (
        "Leukocytes [#/volume] in Blood", "6690-2",
        {"/μL", "×10³/μL", "10^9/L"},
    ),
    "ormone_tireostimolante": (
        "Thyrotropin [Units/volume] in Serum or Plasma", "3016-3",
        {"mIU/L", "mU/L", "μIU/mL", "m[IU]/L"},
    ),
    "alanina_aminotransferasi": (
        "Alanine aminotransferase [Enzymatic activity/volume] in Serum or Plasma",
        "1742-6", {"U/L"},
    ),
    "aspartato_aminotransferasi": (
        "Aspartate aminotransferase [Enzymatic activity/volume] in Serum or Plasma",
        "1920-8", {"U/L"},
    ),
    "creatinina": (
        "Creatinine [Mass/volume] in Serum or Plasma", "2160-0",
        {"mg/dL", "mg/L", "μmol/L"},
    ),
    "proteina_c_reattiva": (
        "C reactive protein [Mass/volume] in Serum or Plasma", "1988-5",
        {"mg/L", "mg/dL"},
    ),
    # Urine variants are only emitted when the specimen detector has
    # suffixed the canonical name (e.g. emoglobina_urine).  726-0 is the
    # quantitative urine hemoglobin term; the dipstick ordinal is a
    # different code and must never be used for numeric rows.
    "emoglobina_urine": (
        "Hemoglobin [Mass/volume] in Urine", "726-0", {"mg/dL"},
    ),
    "proteine_urine": (
        "Protein [Mass/volume] in Urine", "2888-6", {"mg/dL"},
    ),
}


def normalize_concept(value: object) -> str:
    text = unicodedata.normalize("NFKD", str(value or "").casefold())
    text = "".join(char for char in text if not unicodedata.combining(char))
    return "_".join(re.findall(r"[a-z0-9]+", text))


class DeterministicTerminologyResolver:
    """Apply reviewed project mappings, then conservative built-in mappings."""

    def __init__(self, repository=None):
        self.repository = repository
        self.lab_normalizer = LabNormalizer()

    def resolve_all(self, evidence):
        return [self.resolve(item) for item in evidence]

    def resolve(self, item):
        # A code already assigned by the constrained SNOMED stage is final.
        # Without this early return the lab block below would rewrite a
        # SNOMED-coded laboratory atom through the parameter normalizer and a
        # LOINC mapping could then overwrite the code.  Preserve the label,
        # promote an unmapped status, and stop.
        if item.terminology_system == "SNOMED CT" and item.terminology_code:
            if item.mapping_status == "unmapped":
                item.mapping_status = "resolved_llm"
            item.canonical_label = item.canonical_label or (
                item.normalized_entity or item.concept_original
            )
            return item

        original_unit = str(item.unit or "")
        if item.fact_type == "laboratory_test" or item.category == (
            "laboratory_finding"
        ):
            item.fact_type = "laboratory_test"
            item.unit = self.lab_normalizer.normalize_unit(original_unit) or None
            item.normalized_entity = self.lab_normalizer.normalize_parameter(
                item.normalized_entity or item.concept_original or "",
                item.unit or "",
            )
        normalized = normalize_concept(
            item.normalized_entity or item.concept_original
        )
        if not normalized:
            return item

        mapping = None
        if self.repository is not None:
            mapping = self.repository.find_mapping(
                normalized, item.fact_type, original_unit or item.unit
            )
        if mapping is None and item.fact_type == "laboratory_test":
            mapping = self._builtin_lab_mapping(normalized, item.unit)
            if mapping is not None and self.repository is not None:
                self.repository.save_mapping(mapping)
        if mapping is None:
            item.canonical_label = item.canonical_label or (
                item.normalized_entity or item.concept_original
            )
            if item.mapping_status == "unmapped" and item.normalized_entity:
                item.mapping_status = "normalized_not_coded"
            return item

        if isinstance(mapping, TerminologyMapping):
            mapping = asdict(mapping)
        self._apply(item, mapping, original_unit)
        return item

    def _builtin_lab_mapping(
        self, normalized: str, unit: str | None
    ) -> TerminologyMapping | None:
        entry = _LOINC_LAB.get(normalized)
        if entry is None:
            return None
        label, code, expected_units = entry
        canonical_unit = str(unit or "")
        # The analyte remains normalized even if a surprising unit prevents a
        # safe code assignment. This avoids silently coding a molar assay as a
        # mass-concentration assay.
        if canonical_unit and canonical_unit not in expected_units:
            return None
        return TerminologyMapping(
            normalized_concept=normalized,
            fact_type="laboratory_test",
            canonical_label=label,
            terminology_system="LOINC",
            terminology_code=code,
            original_unit=canonical_unit,
            canonical_unit=canonical_unit or None,
            mapping_status="resolved_deterministic",
            mapping_confidence=0.9 if canonical_unit else 0.82,
            source="builtin_loinc_verified",
            review_status="auto",
        )

    @staticmethod
    def _apply(item, mapping: dict, original_unit: str) -> None:
        item.canonical_label = mapping.get("canonical_label") or (
            item.canonical_label or item.normalized_entity
        )
        item.terminology_system = mapping.get("terminology_system")
        item.terminology_code = mapping.get("terminology_code")
        item.mapping_status = mapping.get("mapping_status") or "candidate"
        item.mapping_confidence = mapping.get("mapping_confidence")
        canonical_unit = mapping.get("canonical_unit")
        multiplier = mapping.get("multiplier")
        offset = mapping.get("offset")
        item.typed_payload = dict(item.typed_payload or {})
        item.typed_payload.update({
            "mapping_source": mapping.get("source"),
            "original_unit": original_unit or None,
        })
        if canonical_unit:
            item.unit = canonical_unit
        if item.numeric_value is not None and (
            multiplier is not None or offset is not None
        ):
            multiplier = 1.0 if multiplier is None else float(multiplier)
            offset = 0.0 if offset is None else float(offset)
            item.numeric_value = item.numeric_value * multiplier + offset
            for field in ("reference_low", "reference_high"):
                value = item.typed_payload.get(field)
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    item.typed_payload[field] = value * multiplier + offset

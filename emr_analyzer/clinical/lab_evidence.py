"""Narrative laboratory events versus the deterministic laboratory parser."""

from __future__ import annotations

import re
import unicodedata
from typing import Iterable

from ..models.clinical_evidence import ClinicalEvidence
from ..models.lab_result import LabValue

# Evidence methods of earlier builds; their rows are cleared on re-extraction.
LEGACY_LAB_METHODS = ("deterministic_lab", "fhir_laboratory_v1")


def filter_narrative_lab_duplicates(
    evidence: Iterable[ClinicalEvidence],
    lab_values: Iterable[LabValue],
) -> tuple[list[ClinicalEvidence], int]:
    """Remove only exact LLM restatements of parsed laboratory results.

    Narrative patterns (for example ``anemia``) remain distinct from their
    supporting analytes.  An event is suppressed only when analyte, compatible
    date and, when supplied by both sources, value/unit/specimen agree: the
    parsed result stays the single auditable record of that measurement.
    """
    references = [key for lab in lab_values for key in _lab_value_keys(lab)]
    kept: list[ClinicalEvidence] = []
    removed = 0
    for item in evidence:
        candidate = _event_key(item)
        if candidate and any(_same_measurement(candidate, reference) for reference in references):
            removed += 1
            continue
        kept.append(item)
    return kept, removed


def _lab_value_keys(lab: LabValue) -> list[tuple]:
    material = _normalize_parameter(lab.biological_material) if lab.biological_material else ""
    names = {_normalize_parameter(lab.parameter_name),
             _normalize_parameter((lab.normalized_name or "").replace("_", " "))}
    return [(name, str(lab.sample_date or ""), lab.value, _normalize_unit(lab.unit), material)
            for name in names if name]


def _event_key(item: ClinicalEvidence):
    if not (item.fact_type == "laboratory_test" or item.category == "laboratory_finding"):
        return None
    payload = item.typed_payload or {}
    parameter = _normalize_parameter(
        payload.get("parameter_name") or item.canonical_label
        or item.normalized_entity or item.concept_original)
    if not parameter:
        return None
    material = payload.get("specimen") or payload.get("biological_material")
    return (parameter, str(item.observed_date or ""), item.numeric_value,
            _normalize_unit(item.unit), _normalize_parameter(material) if material else "")


def _same_measurement(candidate, reference) -> bool:
    if candidate[0] != reference[0]:
        return False
    if candidate[1] and reference[1] and candidate[1] != reference[1]:
        return False
    if candidate[2] is not None and reference[2] is not None:
        try:
            tolerance = max(1e-9, abs(float(reference[2])) * 1e-6)
            if abs(float(candidate[2]) - float(reference[2])) > tolerance:
                return False
        except (TypeError, ValueError):
            return False
    # Specimen mismatch means a different measurement (urine Hb vs blood Hb).
    if candidate[4] and reference[4] and candidate[4] != reference[4]:
        return False
    return not (candidate[3] and reference[3] and candidate[3] != reference[3])


def _normalize_parameter(value) -> str:
    text = unicodedata.normalize("NFKD", str(value or "").casefold())
    text = "".join(char for char in text if not unicodedata.combining(char))
    text = re.sub(
        r"\b(?:alto|alta|alti|alte|basso|bassa|bassi|basse|"
        r"aumentat\w*|ridott\w*|elevat\w*|diminuit\w*|"
        r"sopra|sotto|fuori|range|limiti?|valori?)\b",
        " ",
        text,
    )
    return re.sub(r"[^a-z0-9]+", "", text)


def _normalize_unit(value) -> str:
    return re.sub(r"\s+", "", str(value or "").casefold())

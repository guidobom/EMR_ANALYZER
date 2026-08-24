"""Deterministic conversion of abnormal laboratory rows into atomic evidence."""

from __future__ import annotations

import hashlib
import json
from datetime import date
from pathlib import Path
from typing import Iterable
import uuid

from ..models.clinical_evidence import ClinicalEvidence
from ..models.lab_result import LabValue
from ..settings import LabEvidencePolicy


LAB_EVIDENCE_SCHEMA_VERSION = "3.0"
LAB_EXTRACTION_METHOD = "deterministic_lab"
_ABNORMAL_FLAGS = {
    "H", "HIGH", "ALTO", "ALTA", "↑",
    "L", "LOW", "BASSO", "BASSA", "↓",
    "*", "**", "***", "!", "ABNORMAL", "PATOLOGICO", "PATOLOGICA",
}


def abnormality_reasons(
    lab: LabValue,
    policy: LabEvidencePolicy | None = None,
    *,
    previous: LabValue | None = None,
) -> list[str]:
    """Return configured, independently auditable inclusion reasons."""
    policy = policy or LabEvidencePolicy()
    reasons: list[str] = []
    flag = str(lab.flag or "").strip().upper()
    if policy.explicit_abnormal_flag and (
        flag in _ABNORMAL_FLAGS
        or (bool(lab.is_abnormal) and not _has_assessable_range(lab)
            and not lab.value_text)
    ):
        reasons.append("explicit_abnormal_flag")
    if policy.outside_reference_range and lab.value is not None and (
        (lab.reference_low is not None and lab.value < lab.reference_low)
        or (lab.reference_high is not None and lab.value > lab.reference_high)
    ):
        reasons.append("outside_reference_range")
    if policy.textual_abnormality and lab.value_text and bool(lab.is_abnormal):
        reasons.append("textual_abnormality")
    if (
        policy.significant_delta_within_range
        and _significant_delta(lab, previous, policy)
    ):
        reasons.append("significant_delta_within_range")
    return reasons


def is_out_of_range(
    lab: LabValue,
    policy: LabEvidencePolicy | None = None,
    *,
    previous: LabValue | None = None,
) -> bool:
    """Return whether configured objective abnormality evidence is present."""
    return bool(abnormality_reasons(lab, policy, previous=previous))


def abnormal_lab_evidence(
    *,
    patient_id: str,
    document_id: str,
    document_date: str | None,
    lab_values: Iterable[LabValue],
    geometry=None,
    policy: LabEvidencePolicy | None = None,
    prior_values: Iterable[LabValue] = (),
) -> list[ClinicalEvidence]:
    """Build one immutable atom for every out-of-range laboratory value.

    Normal and non-assessable values are deliberately omitted.  The structured
    row, not an LLM, supplies the concept, value, range, date and polarity.
    """
    evidence: list[ClinicalEvidence] = []
    policy = policy or LabEvidencePolicy()
    previous_by_parameter: dict[tuple[str, str], LabValue] = {}
    for prior in sorted(
        prior_values,
        key=lambda item: (
            item.sample_date or "", item.document_id, item.page or 0,
        ),
    ):
        if prior.value is None:
            continue
        previous_by_parameter[(
            str(prior.normalized_name or prior.parameter_name).casefold(),
            str(prior.unit or "").casefold(),
        )] = prior
    ordered = sorted(
        lab_values,
        key=lambda item: (
            item.sample_date or "9999-99-99", item.document_id,
            item.page or 0, item.parameter_name,
        ),
    )
    for lab in ordered:
        parameter_key = (
            str(lab.normalized_name or lab.parameter_name).casefold(),
            str(lab.unit or "").casefold(),
        )
        previous = previous_by_parameter.get(parameter_key)
        reasons = abnormality_reasons(lab, policy, previous=previous)
        if lab.value is not None:
            previous_by_parameter[parameter_key] = lab
        if not reasons:
            continue
        source_text = str(lab.source_text or "").strip()
        if not source_text:
            # An atomic claim without an exact source passage cannot be cited.
            continue
        page = lab.page
        bbox = None
        if geometry is not None and hasattr(geometry, "locate_source"):
            located_page, located_bbox = geometry.locate_source(source_text, page)
            page = located_page or page
            bbox = located_bbox
        observed_date = lab.sample_date or None
        direction = _abnormal_direction(lab)
        polarity = "present"
        evidence_id = _stable_lab_evidence_id(
            document_id=document_id,
            lab=lab,
            observed_date=observed_date,
            source_text=source_text,
            page=page,
        )
        source_reference = {
            "document_id": document_id,
            "page": page,
            "bbox": list(bbox) if bbox else None,
            "passage": source_text,
        }
        evidence.append(ClinicalEvidence(
            evidence_id=evidence_id,
            patient_id=patient_id,
            document_id=document_id,
            category="laboratory_finding",
            normalized_entity=(lab.normalized_name or lab.parameter_name).strip(),
            fact_type="laboratory_test",
            concept_original=lab.parameter_name.strip(),
            canonical_label=(
                lab.normalized_name or lab.parameter_name
            ).strip(),
            mapping_status="normalized_name" if lab.normalized_name else "unmapped",
            clinical_relevance="accepted_clinical",
            typed_payload={
                "parameter_name": lab.parameter_name,
                "operator": lab.operator,
                "reference_low": lab.reference_low,
                "reference_high": lab.reference_high,
                "reference_text": lab.reference_text,
                "flag": lab.flag,
                "abnormal_direction": direction,
                "inclusion_reasons": reasons,
                "biological_material": lab.biological_material,
                "lab_name": lab.lab_name,
            },
            assertion="present",
            temporality="current",
            clinical_status=direction,
            observed_date=observed_date,
            document_date=document_date,
            date_precision=_date_precision(observed_date),
            date_source="sample_date" if observed_date else "unknown",
            significance="clinically_relevant",
            certainty="confirmed",
            value_text=(
                lab.value_text
                if lab.value_text is not None
                else _numeric_display(lab)
            ),
            numeric_value=lab.value,
            unit=lab.unit,
            source_page=page,
            source_text=source_text,
            bbox=bbox,
            confidence=max(0.0, min(float(lab.confidence or 1.0), 1.0)),
            extraction_method=LAB_EXTRACTION_METHOD,
            prompt_version=None,
            schema_version=LAB_EVIDENCE_SCHEMA_VERSION,
            status="auto",
            data={
                "fact_type": "laboratory_test",
                "polarity": polarity,
                "report_date": document_date,
                "source_reference": source_reference,
                "parameter_name": lab.parameter_name,
                "operator": lab.operator,
                "reference_low": lab.reference_low,
                "reference_high": lab.reference_high,
                "reference_text": lab.reference_text,
                "flag": lab.flag,
                "abnormal_direction": direction,
                "inclusion_reasons": reasons,
                "biological_material": lab.biological_material,
                "lab_name": lab.lab_name,
                "validated_by_user": lab.validated_by_user,
            },
        ))
    return evidence


def load_document_geometry(path: Path | None):
    """Load optional PDF text geometry without making it a hard dependency."""
    if path is None or not path.exists():
        return None
    try:
        from ..pipeline.pdf_extractor import PdfExtractionResult

        return PdfExtractionResult.from_dict(
            json.loads(path.read_text(encoding="utf-8"))
        )
    except (OSError, ValueError, TypeError):
        return None


def _abnormal_direction(lab: LabValue) -> str:
    flag = str(lab.flag or "").strip().upper()
    if flag in {"H", "HIGH", "ALTO", "ALTA", "↑"}:
        return "high"
    if flag in {"L", "LOW", "BASSO", "BASSA", "↓"}:
        return "low"
    if lab.value is not None:
        if lab.reference_low is not None and lab.value < lab.reference_low:
            return "low"
        if lab.reference_high is not None and lab.value > lab.reference_high:
            return "high"
    return "abnormal"


def _numeric_display(lab: LabValue) -> str:
    if lab.value is None:
        return ""
    return f"{lab.operator or ''}{lab.value:g}"


def _date_precision(value: str | None) -> str:
    length = len(str(value or ""))
    return (
        "day" if length == 10 else "month" if length == 7
        else "year" if length == 4 else "unknown"
    )


def _has_assessable_range(lab: LabValue) -> bool:
    return lab.value is not None and (
        lab.reference_low is not None or lab.reference_high is not None
    )


def _significant_delta(
    current: LabValue,
    previous: LabValue | None,
    policy: LabEvidencePolicy,
) -> bool:
    if (
        previous is None or current.value is None or previous.value is None
        or not current.sample_date or not previous.sample_date
    ):
        return False
    try:
        elapsed = abs((
            date.fromisoformat(current.sample_date[:10])
            - date.fromisoformat(previous.sample_date[:10])
        ).days)
    except (TypeError, ValueError):
        return False
    rule = policy.analyzer_rules.get(
        str(current.normalized_name or current.parameter_name), {}
    )
    if not isinstance(rule, dict):
        rule = {}
    try:
        window = int(rule.get("window_days", policy.delta_window_days))
        threshold = float(rule.get(
            "relative_delta", policy.default_relative_delta
        ))
    except (TypeError, ValueError):
        return False
    if elapsed > max(1, window):
        return False
    denominator = abs(previous.value)
    if denominator == 0:
        return abs(current.value) > 0
    return abs(current.value - previous.value) / denominator >= max(0.0, threshold)


def _stable_lab_evidence_id(
    *,
    document_id: str,
    lab: LabValue,
    observed_date: str | None,
    source_text: str,
    page: int | None,
) -> str:
    payload = json.dumps({
        "document_id": document_id,
        "parameter": str(lab.normalized_name or lab.parameter_name).casefold(),
        "value": lab.value,
        "value_text": lab.value_text,
        "operator": lab.operator,
        "unit": str(lab.unit or "").casefold(),
        "observed_date": observed_date,
        "page": page,
        "source_text": " ".join(source_text.casefold().split()),
    }, ensure_ascii=False, sort_keys=True)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return "EVD_" + uuid.uuid5(uuid.NAMESPACE_URL, digest).hex

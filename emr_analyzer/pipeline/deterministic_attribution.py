"""Deterministic (non-LLM) document attribution.

A document routed to a patient is re-checked without any LLM call: its own
deterministic evidence — read from the PDF header by the extractor, falling
back to a full-text scan for a checksum-valid fiscal code — is digested with
the same HMAC key as the registry and matched against the registered
identities.

The strong identifiers used here are the same the LLM verification trusts:
a checksum-valid fiscal code, or the exact (name, birth date) pair.  Only
HMAC digests are compared; raw identifiers are never persisted (the privacy
invariant of the workspace registry).
"""

from __future__ import annotations

import re

from ..models.patient_identity import (
    IdentityField,
    PatientIdentityEvidence,
)
from ..pipeline.patient_identity import (
    fiscal_code_has_valid_checksum,
    normalize_fiscal_code,
)

# Structure-only pattern for an uppercase Italian fiscal code; the checksum
# is always validated before the code is used as an anchor.
_CF_RE = re.compile(r"\b[A-Z]{6}\d{2}[A-Z]\d{2}[A-Z]\d{3}[A-Z]\b")


def strong_evidence_confirms(
    repo, evidence: PatientIdentityEvidence, patient_id: str
) -> bool:
    """Whether a strong identifier in ``evidence`` confirms ``patient_id``.

    Each strong identifier — the fiscal code alone, or the (name, birth
    date) pair — is matched against the registry on a reduced evidence.  A
    single strong identifier that uniquely resolves to ``patient_id`` without
    a new conflict makes the attribution trustworthy even when a secondary
    field disagrees (an LLM misread or a source-form artifact).
    """
    strong_subsets = []
    if evidence.fiscal_code is not None:
        strong_subsets.append(
            PatientIdentityEvidence(
                source_path=evidence.source_path,
                fiscal_code=evidence.fiscal_code,
            )
        )
    if evidence.name is not None and evidence.birth_date is not None:
        strong_subsets.append(
            PatientIdentityEvidence(
                source_path=evidence.source_path,
                name=evidence.name,
                birth_date=evidence.birth_date,
            )
        )
    for subset in strong_subsets:
        match = repo.find_match(subset)
        if match.patient_id == patient_id and not match.conflict:
            return True
    return False


def evidence_from_fulltext(full_text: str | None) -> PatientIdentityEvidence | None:
    """Fallback evidence from a checksum-valid fiscal code anywhere in text.

    Used when the header extractor yields nothing (scanned documents,
    non-standard headers): a valid CF in the full text is the strongest
    identifier available and uniquely names a patient in the registry.
    """
    if not full_text:
        return None
    for found in _CF_RE.findall(full_text.upper()):
        if fiscal_code_has_valid_checksum(found):
            normalized = normalize_fiscal_code(found)
            return PatientIdentityEvidence(
                source_path="fulltext",
                fiscal_code=IdentityField(
                    found,
                    normalized,
                    method="fulltext_regex",
                    confidence=0.98,
                ),
                extraction_method="fulltext_regex",
            )
    return None


def deterministic_verdict(
    extractor, pdf_path: str | None
) -> tuple[str, PatientIdentityEvidence | None]:
    """Deterministic attribution verdict for one document, no LLM.

    Runs the header extractor on the PDF and falls back to a full-text scan
    for a valid CF.  Returns ``(status, evidence)`` where ``status`` is::

      * ``"confirmed"``  — usable evidence was found (the caller decides
        confirmation vs conflict against the assigned patient via
        :func:`strong_evidence_confirms` and ``repo.find_match``).
      * ``"inconclusive"`` — no usable evidence found.

    Raw identifiers never leave this function: only the caller's
    digest-based checks run on the returned evidence.
    """
    evidence = None
    if extractor and pdf_path:
        try:
            evidence = extractor.extract(pdf_path)
        except Exception:
            evidence = None
    if evidence is None or not evidence.fields:
        if pdf_path:
            try:
                import fitz

                pdf = fitz.open(pdf_path)
                try:
                    raw = " ".join(
                        pdf[i].get_text()
                        for i in range(min(pdf.page_count, 5))
                    )
                finally:
                    pdf.close()
                evidence = evidence_from_fulltext(raw)
            except Exception:
                evidence = None
    if evidence is None or not evidence.fields:
        return "inconclusive", None
    return "confirmed", evidence

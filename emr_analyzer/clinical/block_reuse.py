"""Lossless reuse of exact clinical blocks across longitudinal documents.

Normalized source files are never changed.  The planner removes only long,
byte-equivalent clinical lines from the *LLM view* of later documents, under
the same document type and section heading. Evidence from the first occurrence
is then cloned with the later document's provenance. A missing mapping triggers
one focused verification per unique block and, when needed, a target-specific
segment pass. The full document is re-read only after a verification failure.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace
from datetime import datetime
import hashlib
import re
from typing import Iterable

from .atomic_evidence import locate_quote, stable_evidence_id
from .temporal import normalize_clinical_date
from ..models.clinical_evidence import ClinicalEvidence


_PAGE_MARKER = re.compile(
    r"(?i)^\s*(?:---\s*PAGINA\s+(\d+)\s*---|"
    r"\[PAGINA\s*:?\s*(\d+)\]|<!--\s*page\s*:\s*(\d+)\s*-->)\s*$"
)


@dataclass(frozen=True, slots=True)
class ReuseLink:
    fingerprint: str
    source_document_id: str
    target_document_id: str
    source_text: str
    target_text: str
    target_page: int | None
    source_section: str = ""
    target_section: str = ""


@dataclass(frozen=True, slots=True)
class ReuseVerificationBatch:
    """Unique unresolved blocks that share source-document context."""

    source_document_id: str
    text: str
    links: tuple[ReuseLink, ...]


@dataclass(frozen=True, slots=True)
class PlannedText:
    document_id: str
    extraction_text: str
    reuse_links: tuple[ReuseLink, ...]
    original_chars: int
    extraction_chars: int


def plan_exact_block_reuse(document_rows: Iterable[tuple]) -> dict[str, PlannedText]:
    """Plan conservative cross-document reuse in chronological input order.

    Each input row is ``(document, effective_text, input_hash)``.  Document
    objects need only expose ``id`` and ``document_type``.
    """
    first_occurrence: dict[str, tuple[str, str, str]] = {}
    plans: dict[str, PlannedText] = {}
    for document, text, _input_hash in document_rows:
        section = ""
        section_text = ""
        current_page = None
        kept_lines: list[str] = []
        links: list[ReuseLink] = []
        for raw_line in str(text or "").splitlines():
            marker = _PAGE_MARKER.match(raw_line)
            if marker:
                current_page = int(next(value for value in marker.groups() if value))
                kept_lines.append(raw_line)
                continue
            normalized = _normalize_block(raw_line)
            if _is_section_heading(raw_line):
                section = normalized
                section_text = raw_line.strip()
                kept_lines.append(raw_line)
                continue
            fingerprint = _reusable_fingerprint(
                raw_line,
                document_type=str(getattr(document, "document_type", "") or ""),
                section=section,
            )
            source = first_occurrence.get(fingerprint) if fingerprint else None
            if source and source[0] != document.id:
                links.append(ReuseLink(
                    fingerprint=fingerprint,
                    source_document_id=source[0],
                    target_document_id=document.id,
                    source_text=source[1],
                    target_text=raw_line.strip(),
                    target_page=current_page,
                    source_section=source[2],
                    target_section=section_text,
                ))
                # Preserve line count and separation without exposing the
                # repeated clinical content to the model again.
                kept_lines.append("")
                continue
            kept_lines.append(raw_line)
            if fingerprint:
                first_occurrence.setdefault(
                    fingerprint, (
                        document.id, raw_line.strip(), section_text,
                    )
                )
        extraction_text = "\n".join(kept_lines).strip()
        plans[document.id] = PlannedText(
            document_id=document.id,
            extraction_text=extraction_text,
            reuse_links=tuple(links),
            original_chars=len(str(text or "")),
            extraction_chars=len(extraction_text),
        )
    return plans


def plan_reuse_verification_batches(
    links: Iterable[ReuseLink],
) -> list[ReuseVerificationBatch]:
    """Verify every unresolved fingerprint once, not once per occurrence.

    Blocks are grouped by their first source document so document date, type
    and PDF geometry remain valid.  Section headings are retained as local
    context, while each clinical line remains byte-identical and therefore
    citable by the atomic extractor.
    """
    unique: dict[str, ReuseLink] = {}
    for link in links:
        unique.setdefault(link.fingerprint, link)

    grouped: dict[str, list[ReuseLink]] = {}
    for link in unique.values():
        grouped.setdefault(link.source_document_id, []).append(link)

    batches = []
    for source_document_id, source_links in grouped.items():
        parts: list[str] = []
        previous_section = None
        for link in source_links:
            section = link.source_section.strip()
            if section and section != previous_section:
                parts.append(section)
            parts.append(link.source_text.strip())
            previous_section = section
        batches.append(ReuseVerificationBatch(
            source_document_id=source_document_id,
            text="\n\n".join(part for part in parts if part).strip(),
            links=tuple(source_links),
        ))
    return batches


def build_targeted_reuse_text(links: Iterable[ReuseLink]) -> str:
    """Build the smallest target-specific text that preserves context.

    A page marker and the target section are emitted when available.  Exact
    duplicate occurrences inside the same page/section are included once;
    the clinical source lines themselves are never rewritten.
    """
    parts: list[str] = []
    seen: set[tuple[int | None, str, str]] = set()
    previous_context: tuple[int | None, str] | None = None
    for link in links:
        section = link.target_section.strip()
        key = (link.target_page, section.casefold(), _normalize_block(
            link.target_text
        ))
        if key in seen:
            continue
        seen.add(key)
        context = (link.target_page, section)
        if context != previous_context:
            if link.target_page is not None:
                parts.append(f"--- PAGINA {link.target_page} ---")
            if section:
                parts.append(section)
        parts.append(link.target_text.strip())
        previous_context = context
    return "\n\n".join(part for part in parts if part).strip()


def evidence_matching_reuse_block(
    evidence: Iterable[ClinicalEvidence], link: ReuseLink,
) -> list[ClinicalEvidence]:
    """Return evidence whose exact citation contains, or is inside, a block."""
    block = _normalize_block(link.source_text)
    if not block:
        return []
    matches = []
    for item in evidence:
        quote = _normalize_block(item.source_text)
        if quote and (quote in block or block in quote):
            matches.append(item)
    return matches


def clone_reused_evidence(
    source_evidence: Iterable[ClinicalEvidence],
    link: ReuseLink,
    *,
    patient_id: str,
    target_document_date: str | None,
    target_full_text: str,
    target_geometry=None,
) -> list[ClinicalEvidence]:
    """Clone evidence fully contained in an exact repeated source block."""
    source_normalized = _normalize_block(link.source_text)
    clones = []
    for item in source_evidence:
        quote_normalized = _normalize_block(item.source_text)
        if not quote_normalized or not (
            quote_normalized in source_normalized
            or source_normalized in quote_normalized
        ):
            continue
        verified, target_quote = locate_quote(item.source_text, target_full_text)
        if not verified:
            continue
        target_page, target_bbox = link.target_page, None
        if target_geometry is not None:
            target_page, target_bbox = target_geometry.locate_source(
                target_quote, link.target_page
            )
        observed_date = item.observed_date
        observed_end = item.observed_date_end
        precision = item.date_precision
        date_source = item.date_source
        data = deepcopy(item.data)
        if item.date_source == "retrospective_duration":
            temporal = normalize_clinical_date(
                data.get("date_original_text"),
                document_date=target_document_date,
                explicit_precision=None,
            )
            observed_date = temporal.start
            observed_end = temporal.end
            precision = temporal.precision
            date_source = temporal.source
            data["date_approximate"] = temporal.approximate
        data.update({
            "quote_verified": True,
            "reused_exact_block": True,
            "reused_from_document_id": link.source_document_id,
            "reused_block_fingerprint": link.fingerprint,
        })
        evidence_id = stable_evidence_id(
            document_id=link.target_document_id,
            category=item.category,
            entity=item.normalized_entity,
            quote=target_quote,
            observed_date=observed_date,
            page=target_page,
            assertion=item.assertion,
            certainty=item.certainty,
        )
        clones.append(replace(
            item,
            evidence_id=evidence_id,
            patient_id=patient_id,
            document_id=link.target_document_id,
            source_text=target_quote,
            observed_date=observed_date,
            observed_date_end=observed_end,
            document_date=target_document_date,
            date_precision=precision,
            date_source=date_source,
            source_page=target_page,
            bbox=target_bbox,
            data=data,
            created_at=datetime.now().isoformat(),
        ))
    return clones


def _normalize_block(value: str) -> str:
    return " ".join(str(value or "").casefold().split())


def _is_section_heading(line: str) -> bool:
    value = str(line or "").strip()
    if not value or len(value) > 100:
        return False
    letters = [char for char in value if char.isalpha()]
    return bool(
        value.startswith("#")
        or value.endswith(":")
        or (letters and sum(char.isupper() for char in letters) / len(letters) > 0.85)
    )


def _reusable_fingerprint(
    line: str, *, document_type: str, section: str
) -> str | None:
    normalized = _normalize_block(line)
    # Short labels and isolated values need their surrounding context and are
    # deliberately never reused.
    if len(normalized) < 80 or len(normalized.split()) < 10:
        return None
    payload = f"{document_type.casefold()}\x1f{section}\x1f{normalized}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()

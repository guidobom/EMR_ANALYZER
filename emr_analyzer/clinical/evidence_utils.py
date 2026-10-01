"""Source identity and deduplication utilities, independent of extraction models."""
from __future__ import annotations
import copy
from dataclasses import dataclass
import hashlib
import json
import re
from typing import Iterable
import unicodedata
from .concept_canonicalization import canonical_concept, canonical_severity
from ..models.clinical_evidence import ClinicalEvidence

@dataclass(frozen=True)
class SentenceSpan:
    """A citable, exact slice of one prompt chunk."""
    sentence_id: int
    start: int
    end: int
    text: str

def locate_quote(quote: str, full_text: str) -> tuple[bool, str]:
    """Verify a citation, tolerating whitespace differences only."""
    source = str(full_text or '')
    wanted = ' '.join(str(quote or '').split())
    if not wanted:
        return (False, '')
    if wanted in source:
        return (True, wanted)
    flexible = re.compile('\\s+'.join((re.escape(token) for token in wanted.split())), re.IGNORECASE)
    match = flexible.search(source)
    if match:
        return (True, source[match.start():match.end()])
    return (False, wanted)

def deduplicate_atomic_evidence(evidence: Iterable[ClinicalEvidence]) -> list[ClinicalEvidence]:
    """Return one clinical atom for repeated source evidence.

    ``document_id``, quote wording and page coordinates deliberately do not
    participate in the identity.  Reworded evidence copied within or between
    reports is one clinical fact when concept, state, date, value and the other
    discriminating fields agree.  Every physical occurrence is kept
    in ``data['source_occurrences']`` so provenance/Quick View remain lossless.

    A changed clinical date, value, treatment state, site, side or severity is
    a different atom.  This prevents a true follow-up measurement or therapy
    transition from being swallowed by copy-and-paste suppression.
    """
    from .lexicon_dedup import deduplicate_lexicon_occurrences
    evidence = list(evidence)
    exact = {item.evidence_id:item for item in evidence if item.data.get('fhir_pipeline')}
    evidence = [item for item in evidence if not item.data.get('fhir_pipeline')]
    lexicon = deduplicate_lexicon_occurrences(
        item for item in evidence if item.data.get('lexicon_term_id')
    )
    grouped: dict[tuple, list[ClinicalEvidence]] = {}
    for item in evidence:
        if item.data.get('lexicon_term_id'):
            continue
        grouped.setdefault(_atomic_identity_key(item), []).append(item)
    _absorb_severity_wildcard_groups(grouped)
    _absorb_undated_wildcard_groups(grouped)
    result: list[ClinicalEvidence] = []
    for items in grouped.values():
        canonical = copy.deepcopy(min(items, key=_canonical_source_key))
        richest = max(items, key=_atomic_quality_key)
        _enrich_canonical_atom(canonical, richest)
        occurrences = []
        seen_ids: set[str] = set()
        for occurrence in sorted(items, key=_canonical_source_key):
            if occurrence.evidence_id in seen_ids:
                continue
            seen_ids.add(occurrence.evidence_id)
            occurrences.append({'evidence_id': occurrence.evidence_id, 'document_id': occurrence.document_id, 'document_date': occurrence.document_date, 'observed_date': occurrence.observed_date, 'source_page': occurrence.source_page, 'bbox': list(occurrence.bbox) if occurrence.bbox else None, 'source_text': occurrence.source_text, 'source_spans': occurrence.data.get('source_spans', []), 'source_version': occurrence.data.get('source_version')})
        duplicate_ids = [occurrence['evidence_id'] for occurrence in occurrences if occurrence['evidence_id'] != canonical.evidence_id]
        canonical.data['atomic_duplicate_count'] = len(duplicate_ids)
        if duplicate_ids:
            canonical.data['duplicate_source_evidence_ids'] = duplicate_ids
            canonical.data['source_occurrences'] = occurrences
        else:
            canonical.data.pop('duplicate_source_evidence_ids', None)
            canonical.data.pop('source_occurrences', None)
        result.append(canonical)
    from .historical_reuse import consolidate_historical
    return result + lexicon + consolidate_historical(exact.values())

def _atomic_identity_key(item: ClinicalEvidence) -> tuple:
    """Clinical identity independent of the report containing the quote."""
    therapy = item.data.get('therapy') or {}
    oncology = item.data.get('oncology') or {}
    if item.date_source == 'retrospective_duration':
        date_key = ('relative', _identity_text(item.data.get('date_original_text')))
        date_end_key = None
    else:
        date_key = item.observed_date
        date_end_key = item.observed_date_end
    return (item.patient_id, _identity_text(item.category), canonical_concept(item.category, item.normalized_entity), date_key, date_end_key, _identity_text(item.assertion), _identity_text(item.value_text), _identity_number(item.numeric_value), _identity_text(item.unit), _identity_text(item.anatomical_site), _identity_text(item.laterality), canonical_severity(item.severity), _identity_text(item.clinical_status), _identity_text(therapy.get('lifecycle_status')), _identity_text(therapy.get('dose')), _identity_text(therapy.get('route')), _identity_text(therapy.get('frequency')), _identity_text(oncology.get('line_label') or oncology.get('line')), _identity_json(oncology.get('regimen')), _identity_text(oncology.get('cycle')), _identity_text(oncology.get('modification')), _identity_text(item.data.get('experiencer', 'patient')))

def _absorb_severity_wildcard_groups(grouped: dict[tuple, list[ClinicalEvidence]]) -> None:
    """Merge filler-severity restatements into the informative group.

    The key index 11 is the canonical severity.  A restatement with an
    empty severity (missing or filler like "none") adds no information: it
    is absorbed by the single non-empty severity group sharing the rest of
    the identity, so the registry keeps the informative atom and the
    provenance of every occurrence.  Ambiguous cases (several distinct
    non-empty severities) stay split.
    """
    by_key_without_severity: dict[tuple, list[tuple]] = {}
    for key in grouped:
        by_key_without_severity.setdefault(key[:11] + key[12:], []).append(key)
    for keys in by_key_without_severity.values():
        if len(keys) < 2:
            continue
        informative = [key for key in keys if key[11]]
        empty = [key for key in keys if not key[11]]
        if len(informative) == 1 and len(empty) == 1:
            grouped[informative[0]].extend(grouped[empty[0]])
            del grouped[empty[0]]

def _absorb_undated_wildcard_groups(grouped: dict[tuple, list[ClinicalEvidence]]) -> None:
    """Absorb restatements that lost their observation date.

    A state fact restated across visits (``nivolumab ongoing``, a planned
    biopsy, a diagnosis) is one atom anchored to the documented date; the
    copies whose model-produced ``observation_date`` is empty add no new
    clinical information.  The key index 3 is the observation date.  Only
    groups with a single distinct dated anchor are merged, and
    measurements (``numeric_value``) never participate: an undated
    measurement can never be attributed to another value.
    """
    by_key_without_date: dict[tuple, list[tuple]] = {}
    for key, items in grouped.items():
        if any((item.numeric_value is not None for item in items)):
            continue
        by_key_without_date.setdefault(key[:3] + key[4:], []).append(key)
    for keys in by_key_without_date.values():
        dated = [key for key in keys if key[3]]
        undated = [key for key in keys if not key[3]]
        if len(dated) == 1 and len(undated) == 1:
            grouped[dated[0]].extend(grouped[undated[0]])
            del grouped[undated[0]]

def _identity_text(value: object) -> str:
    normalized = unicodedata.normalize('NFKD', str(value or '').casefold())
    without_marks = ''.join((char for char in normalized if not unicodedata.combining(char)))
    return ' '.join(''.join((char if char.isalnum() else ' ' for char in without_marks)).split())

def _identity_number(value: object) -> str:
    parsed = _safe_float(value)
    return '' if parsed is None else format(parsed, '.12g')

def _identity_json(value: object) -> str:
    if value in (None, '', [], {}):
        return ''
    if isinstance(value, list):
        value = [_identity_text(item) for item in value]
    return json.dumps(value, ensure_ascii=False, sort_keys=True)

def _canonical_source_key(item: ClinicalEvidence) -> tuple:
    verified = bool(item.data.get('quote_verified'))
    return (0 if verified else 1, item.document_date or '9999-99-99', item.document_id, item.source_page or 10 ** 9, item.evidence_id)

def _atomic_quality_key(item: ClinicalEvidence) -> tuple:
    populated = sum((value not in (None, '', [], {}) for value in (item.observed_date, item.observed_date_end, item.clinical_status, item.anatomical_site, item.laterality, item.value_text, item.numeric_value, item.unit, item.data)))
    if canonical_severity(item.severity):
        populated += 1
    certainty_rank = {'confirmed': 5, 'patient_reported': 4, 'suspected': 3, 'inferred': 2, 'excluded': 1, 'unknown': 0}.get(str(item.certainty or '').casefold(), 0)
    significance_rank = {'critical': 5, 'high': 4, 'clinically_relevant': 3, 'potentially_relevant': 2, 'uncertain': 1}.get(str(item.significance or '').casefold(), 0)
    return (1 if item.data.get('quote_verified') else 0, populated, float(item.confidence or 0.0), certainty_rank, significance_rank, item.evidence_id)

def _enrich_canonical_atom(canonical: ClinicalEvidence, richest: ClinicalEvidence) -> None:
    """Keep first-source provenance while adopting stronger metadata."""
    canonical.confidence = max(float(canonical.confidence or 0.0), float(richest.confidence or 0.0))
    if _atomic_quality_key(richest) > _atomic_quality_key(canonical):
        canonical.certainty = richest.certainty
        canonical.significance = richest.significance
        if canonical.status == 'needs_review' and richest.status != 'needs_review':
            canonical.status = richest.status
    if not canonical_severity(canonical.severity) and canonical_severity(richest.severity):
        canonical.severity = richest.severity
    canonical.data = _merge_missing_evidence_data(copy.deepcopy(canonical.data), richest.data)

def _merge_missing_evidence_data(left: dict, right: dict) -> dict:
    for key, value in right.items():
        if value in (None, '', [], {}):
            continue
        current = left.get(key)
        if current in (None, '', [], {}):
            left[key] = copy.deepcopy(value)
        elif isinstance(current, dict) and isinstance(value, dict):
            left[key] = _merge_missing_evidence_data(current, value)
    return left

def content_hash(*values: object) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(str(value or '').encode('utf-8'))
        digest.update(b'\x00')
    return digest.hexdigest()

def _safe_float(value: object) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None

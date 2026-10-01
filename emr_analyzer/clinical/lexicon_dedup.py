"""Lossless, patient-scoped projection of lexicon occurrences into dated events."""
from copy import deepcopy
from datetime import date

WINDOW_DAYS = 15


def _day(item):
    if item.date_precision != 'day' or item.observed_date_end:
        return None
    try:
        return date.fromisoformat(item.observed_date)
    except (ValueError, TypeError):
        return None


def _identity(item):
    data = item.data
    return (item.patient_id, data['lexicon_term_id'],
            item.fact_type if data.get('snomed_concept_id') else None, item.assertion,
            item.certainty, data.get('experiencer'),
            item.temporality == 'hypothetical', data.get('lexicon_dedup_rule', 'auto'),
            data.get('event_kind', 'continuous') if data.get('lexicon_dedup_rule', 'auto') == 'auto' else None)


def _compatible(first, second):
    def agrees(a, b):
        return a in (None, '') or b in (None, '') or a == b
    if first.clinical_status != second.clinical_status:
        return False
    fields = ('anatomical_site', 'laterality', 'value_text',
              'numeric_value', 'unit', 'severity')
    if not all(agrees(getattr(first, name), getattr(second, name)) for name in fields):
        return False
    if first.data.get('lexicon_dedup_rule') == 'therapy':
        # Incomplete regimens cannot establish that two administrations match.
        a, b = first.data.get('attributes', {}), second.data.get('attributes', {})
        if not all(a.get(k) and a.get(k) == b.get(k) for k in ('farmaco', 'dose', 'stato')):
            return False
    a, b = first.data.get('attributes', {}), second.data.get('attributes', {})
    return all(agrees(a[key], b[key]) for key in a.keys() & b.keys())


def deduplicate_lexicon_occurrences(items):
    unique = {}
    for item in items:
        sources = item.data.get('source_occurrences')
        if not sources:
            unique[item.evidence_id] = item
            continue
        # Projections may be consumed again by query/export code. Expand their
        # sources before regrouping so the operation stays lossless/idempotent.
        for source in sources:
            if source.get('occurrence'):
                payload = deepcopy(source['occurrence'])
                if payload.get('bbox') is not None:
                    payload['bbox'] = tuple(payload['bbox'])
                occurrence = type(item)(**payload)
                unique[occurrence.evidence_id] = occurrence
                continue
            occurrence = deepcopy(item)
            for name in ('evidence_id', 'document_id', 'document_date', 'observed_date',
                         'observed_date_end', 'date_precision', 'date_source', 'source_page',
                         'source_text', 'bbox'):
                if name in source:
                    setattr(occurrence, name, source[name])
            for name in ('source_spans', 'source_version', 'date_provenance'):
                occurrence.data[name] = deepcopy(source.get(name))
            for name in ('source_occurrences', 'duplicate_source_evidence_ids', 'atomic_duplicate_count'):
                occurrence.data.pop(name, None)
            unique[occurrence.evidence_id] = occurrence
    groups = {}
    for item in sorted(unique.values(), key=lambda i: (i.observed_date or '9999', i.evidence_id)):
        day = _day(item)
        key = _identity(item)
        # Unknown/month/year/approximate dates cannot establish a 15-day gap.
        # Explicit administrations/measurements remain distinct unless both an
        # exact day and the same explicit episode identifier support identity.
        rule = item.data.get('lexicon_dedup_rule', 'auto')
        discrete = rule in {'measurement', 'action'} or (rule == 'auto' and (
                    item.data.get('event_kind') == 'discrete' or
                    item.fact_type in {'laboratory_test', 'vital_sign', 'procedure'}))
        if day is None or rule == 'source_only':
            key += (item.evidence_id,)
        elif discrete:
            episode = item.data.get('episode_quote')
            key += (item.observed_date, ' '.join(episode.casefold().split()) if episode else item.evidence_id)
        clusters = groups.setdefault(key, [])
        candidates = [cluster for cluster in clusters
            if day is not None and _day(cluster[0]) is not None
            and (day - _day(cluster[0])).days <= WINDOW_DAYS
            and all(_compatible(previous, item) for previous in cluster)]
        # An underspecified occurrence matching two incompatible episodes is
        # ambiguous; never let it bridge both clusters.
        if len(candidates) == 1:
            candidates[0].append(item)
        else:
            clusters.append([item])
    result = []
    for clusters in groups.values():
        for cluster in clusters:
            canonical = deepcopy(cluster[0])  # earliest observation, never earliest report
            sources = []
            for item in cluster:
                sources.append(dict(occurrence=deepcopy(item.to_dict()),
                    evidence_id=item.evidence_id, document_id=item.document_id,
                    document_date=item.document_date, observed_date=item.observed_date,
                    observed_date_end=item.observed_date_end, date_precision=item.date_precision,
                    date_source=item.date_source, source_page=item.source_page,
                    bbox=list(item.bbox) if item.bbox else None, source_text=item.source_text,
                    source_spans=deepcopy(item.data.get('source_spans', [])),
                    source_version=item.data.get('source_version'),
                    date_provenance=deepcopy(item.data.get('date_provenance', {}))))
            canonical.data.update(source_occurrences=sources,
                duplicate_source_evidence_ids=[i.evidence_id for i in cluster[1:]],
                atomic_duplicate_count=len(cluster)-1,
                deduplication_rule='lexicon_earliest_anchor_15_days_v1')
            result.append(canonical)
    return sorted(result, key=lambda i: (i.observed_date or '9999', i.evidence_id))

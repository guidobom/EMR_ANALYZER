"""Identity of the copies of one statement, for deduplication and FHIR.

The extraction-side reuse that used to live here (``historical-block-v2``) has
been replaced by the patient-level statement index: identical statements are
annotated once and projected onto their copies.
"""
from copy import deepcopy
import json

from .evidence_utils import content_hash


def historical_identity(item):
    """Stable event identity only for certified exact reuse, with conflict guards."""
    key = item.data.get('historical_reuse_id')
    if not key:
        return None
    return content_hash(item.patient_id,key,item.normalized_entity,item.fact_type,
        item.observed_date,item.observed_date_end,item.assertion,item.certainty,item.clinical_status,
        item.data.get('experiencer'),json.dumps(item.data.get('attributes',{}),sort_keys=True),
        item.data.get('snomed_concept_id'),json.dumps(item.typed_payload,sort_keys=True))


def consolidate_historical(items):
    """One displayed event, all source occurrences preserved, including relations."""
    groups = {}
    for item in items:
        key = historical_identity(item) or item.evidence_id
        groups.setdefault(key,[]).append(item)
    result = []
    for members in groups.values():
        row = deepcopy(members[0])
        sources = {}
        for member in members:
            existing = member.data.get('source_occurrences')
            if existing:
                for source in existing: sources[source['evidence_id']] = deepcopy(source)
            else:
                sources[member.evidence_id] = dict(occurrence=deepcopy(member.to_dict()),
                    evidence_id=member.evidence_id,document_id=member.document_id,
                    document_date=member.document_date,source_text=member.source_text,
                    source_spans=deepcopy(member.data.get('source_spans',[])))
        if len(sources)>1:
            row.data.update(source_occurrences=list(sources.values()),
                duplicate_source_evidence_ids=[i for i in sources if i != row.evidence_id],
                atomic_duplicate_count=len(sources)-1,deduplication_rule='exact_historical_block_v1')
        result.append(row)
    return result

"""Laboratory observations support documented events; they never diagnose one."""
from dataclasses import asdict, replace
from .temporal import temporal_distance_days


def is_laboratory(item):
    return getattr(item, 'fact_type', None) == 'laboratory_test' or item.category == 'laboratory_finding'


def support_targets(items, relations, *, threshold=0.72, max_days=90):
    by_id = {item.evidence_id: item for item in items}
    result = {}
    for relation in relations:
        r = relation if isinstance(relation, dict) else asdict(relation)
        a, b = by_id.get(r['source_evidence_id']), by_id.get(r['target_evidence_id'])
        if a is None or b is None or is_laboratory(a) == is_laboratory(b):
            continue
        lab, target = (a, b) if is_laboratory(a) else (b, a)
        if (lab.patient_id != target.patient_id or lab.assertion != 'present'
                or target.assertion != 'present' or target.temporality == 'hypothetical'
                or lab.data.get('experiencer', 'patient') != 'patient'
                or target.data.get('experiencer', 'patient') != 'patient'):
            continue
        if (r.get('review_status') == 'rejected'
                or r.get('cluster_effect') not in {'must_link', 'cohesive', 'context_only'}
                or float(r.get('weight', 0)) < threshold
                or r.get('relation_type') not in {'same_occurrence', 'same_process', 'manifestation_of', 'evaluates'}):
            continue
        human = r.get('review_status') == 'accepted'
        votes = r.get('model_votes') or []
        positive = sum(bool(v.get('linked')) for v in votes)
        if not human and (not votes or positive <= len(votes)//2):
            continue  # temporal proximity and provisional category rules are insufficient
        distance = temporal_distance_days(lab.observed_date, target.observed_date)
        if not human and ((distance is not None and distance > max_days)
                          or (distance is None and lab.document_id != target.document_id)):
            continue
        result.setdefault(lab.evidence_id, set()).add(target.evidence_id)
    return {key: sorted(value) for key, value in result.items()}


def as_support_relations(items, relations, *, threshold=0.72, max_days=90, eligible_target_ids=None):
    """Keep labs out of event connectivity so one lab cannot bridge diagnoses."""
    links = support_targets(items, relations, threshold=threshold, max_days=max_days)
    if eligible_target_ids is not None:
        links = {lab: [target for target in targets if target in eligible_target_ids] for lab, targets in links.items()}
        links = {lab: targets for lab, targets in links.items() if targets}
    by_id = {item.evidence_id: item for item in items}
    normalized = []
    for relation in relations:
        a, b = relation.source_evidence_id, relation.target_evidence_id
        supporting = b in links.get(a, []) or a in links.get(b, [])
        if supporting:
            lab_id, target_id = (a, b) if a in links else (b, a)
            normalized.append(replace(relation, cluster_effect='context_only',
                rule_features={**relation.rule_features, 'laboratory_support': {
                    'laboratory_evidence_id': lab_id, 'clinical_evidence_id': target_id}}))
        elif (a in by_id and b in by_id
              and (is_laboratory(by_id[a]) != is_laboratory(by_id[b]) or a in links or b in links)
              and relation.cluster_effect in {'must_link', 'cohesive', 'context_only'}):
            # Unsupported laboratory-clinical edges must not silently attach
            # results or merge otherwise unrelated events through a lab node.
            normalized.append(replace(relation, cluster_effect='uncertain'))
        else:
            normalized.append(relation)
    return normalized, links

"""Conservative exact historical-block reuse, with independently verified sources.

Only complete locally dated blocks are reusable. Similarity is never evidence
of identity. The cache contains source annotations, not copied document offsets.
"""
from copy import deepcopy
from datetime import date, datetime
import json
import re

from .evidence_utils import content_hash, SentenceSpan
from .sentence_groups import SentenceGroup, clinical_sentences, plan_sentence_groups

DAY = r'(?:\d{4}-\d{2}-\d{2}|\d{1,2}[/.]\d{1,2}[/.](?:\d{4}|\d{2}))'
START = re.compile(r'^\s*-?\s*(' + DAY + r')(?!\d)', re.M)
RELATIVE = re.compile(r'\b(?:oggi|ieri|domani|attual\w*|odiern\w*|prossim\w*|recent\w*|'
                      r'ultim\w*|successiv\w*|precedent\w*|programm\w*|previst\w*|'
                      r'fra|entro|fa|u\.s\.)\b', re.I)


def normalized(value):
    return ' '.join(value.split())


def absolute_day(value):
    if not re.fullmatch(DAY, value or ''):
        return None
    for fmt in ('%Y-%m-%d', '%d/%m/%Y', '%d/%m/%y', '%d.%m.%Y', '%d.%m.%y'):
        try:
            return datetime.strptime(value, fmt).date()
        except ValueError:
            pass
    return None


def historical_groups(text, budget):
    """Stable dated-entry boundaries; retain every character and original offsets."""
    cuts = sorted({0, len(text), *(m.start() for m in START.finditer(text))})
    groups = []
    global_spans = clinical_sentences(text)
    for a, b in zip(cuts, cuts[1:]):
        for group in plan_sentence_groups(text[a:b], lambda s: max(1, len(s.encode())//3), budget):
            targets = tuple(SentenceSpan(s.sentence_id, s.start+a, s.end+a, s.text) for s in group.targets)
            context = [SentenceSpan(s.sentence_id, s.start+a, s.end+a, s.text) for s in group.context]
            # Include exact adjacent context, not a generated summary. If it changes,
            # reuse is refused even if the target passage is identical.
            before = [s for s in global_spans if s.end <= targets[0].start]
            after = [s for s in global_spans if s.start >= targets[-1].end]
            context.extend(before[-2:]); context.extend(after[:1])
            # Preserve all section headings preceding the entry in the cache key/prompt.
            for s in before:
                if len(s.text) <= 100 and s.text.isupper() and any(c.isalpha() for c in s.text):
                    context.append(s)
            positions = {(s.start,s.end) for s in targets}
            unique = {(s.start,s.end):s for s in context if (s.start,s.end) not in positions}
            groups.append(SentenceGroup(targets, tuple(unique.values())))
    return groups


class HistoricalReuseMixin:
    def _plan_historical_groups(self, text, budget):
        units = historical_groups(text,budget)
        self._local.history_units = units
        kwargs = getattr(self._local,'history_document',{})
        cards = self._cards(text)
        # Batch NEW entries together. Stable cache units must not multiply the
        # number of LLM calls on a first extraction.
        result, pending, cost = [], [], 0
        def flush():
            if not pending: return
            targets = tuple(s for g in pending for s in g.targets)
            positions = {(s.start,s.end) for s in targets}
            contexts = {(s.start,s.end):s for g in pending for s in g.context if (s.start,s.end) not in positions}
            result.append(SentenceGroup(targets,tuple(contexts.values())))
            pending.clear()
        for unit in units:
            key = self._history_key(unit,cards,kwargs.get('patient_id'),kwargs.get('document_type'),kwargs.get('document_date'))
            prior = self.checkpoints.load(kwargs.get('patient_id'),key) if key else None
            size = sum(max(1,len(s.text.encode())//3)+10 for s in unit.targets)
            if prior and prior['status']=='completed':
                flush();cost=0;result.append(unit)
            else:
                if pending and cost+size > budget: flush();cost=0
                pending.append(unit);cost+=size
        flush()
        return result

    def _history_key(self, group, cards, patient_id, document_type, document_date):
        if not self.checkpoints or not group.targets:
            return None
        first = START.match(group.targets[0].text)
        try:
            report_day = date.fromisoformat(document_date)
        except (ValueError, TypeError):
            return None
        historical_day = absolute_day(first.group(1)) if first else None
        if not historical_day or historical_day >= report_day:
            return None
        if RELATIVE.search(' '.join(s.text for s in group.spans)):
            return None
        shape = [(s.sentence_id, s.sentence_id in group.target_ids, normalized(s.text)) for s in group.numbered]
        return content_hash('historical-block-v2', patient_id, self.model_digest, document_type,
                            json.dumps(shape,ensure_ascii=False), json.dumps(cards,sort_keys=True,ensure_ascii=False))

    def _load_historical_group(self, group, cards, patient_id, document_type, document_date, text, document_id, contract):
        key = self._history_key(group,cards,patient_id,document_type,document_date)
        if not key:
            return None
        cached = self.checkpoints.load(patient_id,key)
        if not cached or cached['status'] != 'completed':
            return None
        result = cached['result']
        rows, errors = self._validate(result,contract,group,cards,text,patient_id,document_id,document_date)
        # Always validate quotes, dates and offsets against THIS document.
        if errors or not rows or any(not r.observed_date or r.observed_date >= document_date
                                    or r.data.get('date_provenance',{}).get('needs_review') for r in rows):
            return None
        self._local.metrics['historical_groups_reused'] = self._local.metrics.get('historical_groups_reused',0)+1
        self._local.metrics['historical_characters_skipped'] = self._local.metrics.get('historical_characters_skipped',0)+sum(len(s.text) for s in group.targets)
        return result

    def _save_historical_group(self, group, cards, patient_id, document_type, document_date,
                               text, document_id, contract, result, retained):
        if not isinstance(result,dict) or not result.get('events'):
            return
        positions = {(s.start,s.end) for s in group.targets}
        for unit in getattr(self._local,'history_units', [group]):
            if not all((s.start,s.end) in positions for s in unit.targets):
                continue
            mapping = {(s.start,s.end):s.sentence_id for s in unit.numbered}
            targets = {(s.start,s.end) for s in unit.targets}
            translated = []
            valid = True
            for raw in result['events']:
                source = group.numbered[raw['s']-1]
                if (source.start,source.end) not in targets: continue
                event = deepcopy(raw)
                event['s'] = mapping[(source.start,source.end)]
                if 'date_s' in event:
                    if type(event['date_s']) is not int or not 1 <= event['date_s'] <= len(group.numbered):
                        valid = False;break
                    source = group.numbered[event['date_s']-1]
                    if (source.start,source.end) not in mapping:
                        valid = False;break
                    event['date_s'] = mapping[(source.start,source.end)]
                translated.append(event)
            if valid:
                self._save_historical_unit(unit,cards,patient_id,document_type,document_date,
                    text,document_id,contract,{'events':translated},retained)

    def _save_historical_unit(self, group, cards, patient_id, document_type, document_date,
                              text, document_id, contract, result, retained):
        key = self._history_key(group,cards,patient_id,document_type,document_date)
        if not key or not isinstance(result,dict) or not result.get('events'):
            return
        canonical, identities = [], {}
        for raw in result['events']:
            event = deepcopy(raw)
            if event.get('temporal') in ('current','hypothetical') or event.get('date_unresolved'):
                return
            if 'time' in event:
                source, quote, _ = self._local.times[event.pop('time')]
                local = next((s for s in group.numbered if s.start == source.start and s.end == source.end), None)
                if local is None or local.sentence_id not in group.target_ids:
                    return
                event.update(date_quote=quote,date_s=local.sentence_id,date_mode='explicit')
            if event.get('date_mode') != 'explicit' or not absolute_day(event.get('date_quote')):
                return
            if event.get('date_s',event['s']) not in group.target_ids:
                return
            rows, errors = self._validate({'events':[event]},contract,group,cards,text,patient_id,document_id,document_date)
            if errors or len(rows)!=1 or not rows[0].observed_date or rows[0].observed_date >= document_date:
                return
            # Ensure an unresolved relation/date cannot be turned into trusted cache.
            if rows[0].data.get('date_provenance',{}).get('needs_review'):
                return
            canonical.append(event)
            identities[rows[0].evidence_id] = content_hash(key,json.dumps(event,sort_keys=True,ensure_ascii=False))
        for row in retained:
            if row.evidence_id in identities:
                row.data['historical_reuse_id'] = identities[row.evidence_id]
        self.checkpoints.save(patient_id,document_id,key,'completed',{'events':canonical},self.last_extraction_metrics())
        self._local.metrics['historical_groups_eligible'] = self._local.metrics.get('historical_groups_eligible',0)+1


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

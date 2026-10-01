"""Compact source annotations: the program expands provenance, never the LLM."""
import json
import re
import time

import jsonschema

from .grounded_sources import GroundedSourceReader, schema, AtomicExtractionCancelled
from .sentence_groups import SentenceGroup, clinical_sentences, plan_sentence_groups
from .evidence_utils import content_hash
from ..models.clinical_pipeline import ATOMIC_FACT_TYPES

MONTH = r'(?:gennaio|febbraio|marzo|aprile|maggio|giugno|luglio|agosto|settembre|ottobre|novembre|dicembre)'
DATES = re.compile(r'\b(?:\d{4}-\d{2}-\d{2}|\d{1,2}[/.]\d{1,2}[/.]\d{2,4}|(?:\d{1,2}\s+)?'+MONTH+r'\s+(?:19|20)\d{2}|(?:19|20)\d{2}|ieri|oggi|(?:\d+|un|due|tre|quattro|cinque|sei|sette) giorni? fa)\b', re.I)


def complete_annotations(content):
    """Recover only complete JSON objects from the expected top-level events array."""
    match = re.match(r'\s*\{\s*"events"\s*:\s*\[', content)
    if not match:
        return {'events': []}
    rows, offset, decoder = [], match.end(), json.JSONDecoder()
    while offset < len(content):
        while offset < len(content) and content[offset] in ' \r\n\t,':
            offset += 1
        try:
            row, end = decoder.raw_decode(content, offset)
        except ValueError:
            break
        if not isinstance(row, dict):
            break
        rows.append(row)
        offset = end
    return {'events': rows}


class CompactSourceReader(GroundedSourceReader):
    partial_repairs = True

    def _initial_groups(self, text):
        self._local.document_spans = tuple(clinical_sentences(text))
        self._local.times = {}
        for span in self._local.document_spans:
            for match in DATES.finditer(span.text):
                ref = 'T'+str(len(self._local.times)+1)
                self._local.times[ref] = (span, match.group(), span.start+match.start())
        # Bound output density as well as input. No tokenization HTTP calls per sentence.
        budget = max(200, min(600, int(getattr(self.llm, 'max_output_tokens', 4096))//8))
        return plan_sentence_groups(text, lambda s: max(1, len(s.encode('utf-8'))//3), budget)

    def _request_key(self, patient_id, document_id, text, document_type, document_date, prompt):
        # Safe patient-local reuse: prompt includes exact context, time anchors and report date.
        # Only raw annotations are reused; document provenance is always rebuilt.
        return content_hash(patient_id, self.model_digest, text, prompt)

    def _schema(self, cards):
        string = {'type': 'string', 'minLength': 1, 'maxLength': 200}
        def enum(values):
            return {'type': 'string', 'enum': list(values)}
        props = {
            's': {'type': 'integer', 'minimum': 1},
            'anchor': {**string, 'maxLength': 600},
            'type': enum(ATOMIC_FACT_TYPES),
            'assertion': enum(('present', 'absent', 'unknown')),
            'certainty': enum(('confirmed', 'suspected', 'possible', 'unknown')),
            'subject': enum(('patient', 'family', 'other', 'unknown')),
            'label': string,
            'time': enum(self._local.times) if self._local.times else enum(('unknown',)),
            'temporal': enum(('current', 'historical', 'hypothetical', 'unknown')),
            'state': string,
            'attributes': {'type': 'object', 'additionalProperties': {'type': 'string', 'maxLength': 300}},
            'date_quote': string,
            'date_s': {'type': 'integer', 'minimum': 1},
            'date_mode': enum(('explicit', 'relative', 'document')),
            'date_unresolved': {'type': 'boolean'},

        }
        return {'type': 'object', 'additionalProperties': False, 'required': ['events'],
            'properties': {'events': {'type': 'array', 'items': {
                'type': 'object', 'additionalProperties': False, 'properties': props,
                'required': ['s', 'anchor', 'type', 'assertion', 'certainty', 'subject']}}}}

    def _prompt(self, group, cards, document_type, document_date):
        # Index only. Each visible sentence is already present once in the source.
        visible = {s.start: s.sentence_id for s in group.numbered}
        temporal = [{'id': ref, 'text': quote,
                     'source': 'S'+str(visible[span.start]) if span.start in visible else 'D'+str(span.sentence_id)}
                    for ref, (span, quote, _) in self._local.times.items()]
        return (group.prompt(document_type, document_date)
                + '\nRIFERIMENTI TEMPORALI DEL DOCUMENTO (non attribuirli per vicinanza):\n'
                + json.dumps(temporal, ensure_ascii=False, separators=(',', ':'))
                + '\nESEMPI FACOLTATIVI, NON FATTI DEL PAZIENTE:\n'
                + json.dumps(cards[0].get('guidance', []), ensure_ascii=False, separators=(',', ':')))

    def _repair_prompt(self, prompt, result, errors):
        indices = {int(m.group(1))-1 for error in errors
                   for m in [re.match(r'Evento (\d+):', error)] if m}
        events = result.get('events', []) if isinstance(result, dict) else []
        invalid = [dict(index=i+1, event=events[i]) for i in sorted(indices) if i < len(events)]
        return (prompt + '\nCorreggi soltanto gli eventi problematici; quelli validi sono già conservati.\n'
                + '; '.join(errors)[:800] + '\nEVENTI DA CORREGGERE:\n'
                + json.dumps(invalid, ensure_ascii=False, separators=(',', ':')))

    def _resolve_context(self, result, group, **kwargs):
        """Fetch literal distant evidence only for proposed remote date links."""
        if not isinstance(result, dict) or not isinstance(result.get('events'), list):
            return result
        visible = {s.start for s in group.numbered}
        pending = []
        for i, raw in enumerate(result['events']):
            if not isinstance(raw, dict):
                continue
            ref = raw.get('time')
            if not isinstance(ref, str) or ref not in self._local.times:
                continue
            span, quote, _ = self._local.times[ref]
            if span.start in visible:
                continue
            s = raw.get('s')
            if type(s) is not int or s not in group.target_ids:
                continue
            pending.append((i, ref, span, quote))
        for offset in range(0, len(pending), 4):
            if kwargs.get('cancel_check') and kwargs['cancel_check']():
                raise AtomicExtractionCancelled('Verifica temporale interrotta')
            batch = pending[offset:offset+4]
            sources = {ref: {'date': quote, 'sentence': span.text} for _,ref,span,quote in batch}
            events = [{'id': str(i), 'event': result['events'][i],
                       'sentence': group.numbered[result['events'][i]['s']-1].text}
                      for i,_,_,_ in batch]
            prompt = json.dumps({'events':events,'sources':sources},ensure_ascii=False,separators=(',',':'))
            contract = {'type':'object','additionalProperties':False,'required':['links'],
                'properties':{'links':{'type':'array','minItems':len(batch),'maxItems':len(batch),
                    'items':{'type':'object','additionalProperties':False,'required':['id','confirmed'],
                        'properties':{'id':{'type':'string','enum':[str(i) for i,_,_,_ in batch]},
                                      'confirmed':{'type':'boolean'}}}}}}
            system = ('Verifica se la frase sorgente della data sostiene il collegamento alla specifica '
                      'occorrenza clinica proposta. Stesso concetto o vicinanza non bastano. '
                      'Non attribuire date di terapia alla tossicità, né esordio a regressione. '
                      'Se non dimostrabile restituisci confirmed=false. Solo JSON.')
            confirmed = {}
            try:
                if not self._fits(prompt+system, contract):
                    raise ValueError('Contesto insufficiente per verifica temporale')
                started = time.monotonic()
                try:
                    answer = self.llm.generate_structured(prompt, system, contract,
                        max_tokens=min(512,int(getattr(self.llm,'max_output_tokens',4096))))
                finally:
                    elapsed = time.monotonic()-started
                    self._local.metrics['llm_calls'] += 1
                    self._local.metrics['elapsed_seconds'] += elapsed
                    meta = getattr(self.llm,'last_generation_metadata',lambda:{})() or {}
                    for name in ('prompt_tokens','completion_tokens','prompt_ms','predicted_ms'):
                        self._local.metrics[name] += meta.get(name,0) or 0
                    if self.checkpoints:
                        self.checkpoints.record_call(kwargs['patient_id'],kwargs['document_id'],
                            content_hash(prompt), 'temporal_context', 'attempted', {**meta,'elapsed_seconds':elapsed})
                jsonschema.validate(answer,contract)
                confirmed = {x['id']:x['confirmed'] for x in answer['links']}
                if len(confirmed)!=len(batch):
                    raise ValueError('Identificativi temporali duplicati')
            except AtomicExtractionCancelled:
                raise
            except Exception:
                confirmed = {}
            for i,ref,_,_ in batch:
                if not confirmed.get(str(i)):
                    for field in ('time', 'date_quote', 'date_s', 'date_mode'):
                        result['events'][i].pop(field,None)
                    result['events'][i]['date_unresolved'] = True
        return result

    def _recover_partial(self, content):
        return complete_annotations(content)

    def _validate(self, result, contract, group, cards, text, pid, did, doc_date, *, source_selection=None):
        if not isinstance(result, dict) or set(result) != {'events'} or not isinstance(result['events'], list):
            return [], ['Risposta non conforme: manca events']
        rows, errors = [], []
        for index, raw in enumerate(result['events']):
            try:
                # Relations are not part of this stage. An unsolicited relation
                # must never force regeneration of a valid event.
                if isinstance(raw, dict):
                    raw = {k:v for k,v in raw.items() if k != 'relations'}
                jsonschema.validate(raw, contract['properties']['events']['items'])
                ref = raw['s']
                if ref not in group.target_ids:
                    raise ValueError('Frase non OBIETTIVO')
                source = group.numbered[ref-1]
                augmented = group
                date_mode, date_quote, date_ref = 'unknown', None, None
                if 'time' in raw:
                    span, date_quote, _ = self._local.times[raw['time']]
                    if not any(s.start == span.start and s.end == span.end for s in group.spans):
                        augmented = SentenceGroup(group.targets, (*group.context, span))
                    date_ref = next(s.sentence_id for s in augmented.numbered if s.start == span.start)
                    date_mode = 'relative' if re.fullmatch(r'ieri|oggi|.+ giorni? fa', date_quote, re.I) else 'explicit'
                elif 'date_quote' in raw:
                    date_mode = raw.get('date_mode', 'explicit')
                    date_quote, date_ref = raw['date_quote'], raw.get('date_s', ref)
                elif 'date_mode' in raw or 'date_s' in raw:
                    raise ValueError('Riferimento temporale privo di citazione')
                item = dict(term_id='new', type=raw['type'], quote=raw['anchor'],
                    sentence=next(s.sentence_id for s in augmented.numbered if s.start == source.start),
                    assertion=raw['assertion'], certainty=raw['certainty'], subject=raw['subject'],
                    temporality=raw.get('temporal', 'unknown'), state=raw.get('state'),
                    date_mode=date_mode, date_quote=date_quote, date_ref=date_ref,
                    event_kind='discrete' if raw['type'] in ('procedure','laboratory_test','hospitalization','discharge') else 'continuous',
                    episode_quote=None, attributes=raw.get('attributes', {}))
                label = raw.get('label', raw['anchor'])
                known = next((c for c in cards[0].get('guidance', [])
                              if c['label'].strip().casefold() == label.strip().casefold()), None)
                card = {**(known or {}), 'term_id': 'new', 'label': label}
                found, problems = super()._validate({'occurrences': [item]}, schema(['new']), augmented,
                    [card], text, pid, did, doc_date, source_selection=source_selection)
                errors.extend('Evento '+str(index+1)+': '+p for p in problems)
                for row in found:
                    row.terminology_system = row.terminology_code = None
                    row.mapping_status = 'unmapped'
                    row.prompt_version = self.prompt_version
                    row.data.update(fhir_pipeline=True, extraction_pipeline=self.prompt_version,
                        lexicon_dedup_rule='source_only', snomed_search_en='',
                        snomed_mapping_status='pending', source_relations=[],
                        annotation_sentence=source.sentence_id)
                    row.data['lexicon_term_id'] = 'mention_'+content_hash(card['label'],row.fact_type)[:24]
                    if raw.get('date_unresolved'):
                        row.status = 'needs_review'
                        row.data['date_provenance']['needs_review'] = 'Collegamento temporale distante non confermato'
                    if known:
                        row.data['local_lexicon_term_id'] = known['term_id']
                    # Stable across recursive splits and repair calls: local sentence
                    # numbers are not part of the identity of a source occurrence.
                    row.evidence_id = 'EVD_'+content_hash(pid, did, content_hash(text),
                        row.data['source_spans'][0]['start'], row.data['source_spans'][0]['end'],
                        row.fact_type, row.assertion, row.certainty, raw['subject'],
                        row.observed_date, row.observed_date_end, row.clinical_status,
                        json.dumps(row.data['attributes'], sort_keys=True))
                    rows.append(row)
            except (ValueError, TypeError, KeyError, IndexError, jsonschema.ValidationError) as exc:
                errors.append(f'Evento {index+1}: {str(exc).splitlines()[0][:240]}')
        return rows, errors

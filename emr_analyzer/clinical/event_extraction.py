"""Compact source annotation followed by targeted terminology linking."""
import json
import time

import jsonschema

from .grounded_sources import (GroundedSourceReader, schema, AtomicExtractionCancelled,
                                 IncompleteAtomicExtraction)
from .evidence_utils import content_hash

VERSION = 'fhir_events_referenced_v7'
from ..prompt_catalog import load_prompt


from .referenced_annotations import ReferencedSourceReader
from .historical_reuse import HistoricalReuseMixin


class EventExtractor(HistoricalReuseMixin, ReferencedSourceReader):
    def _initial_groups(self, text):
        # Initialize source spans and the clinical date index before reuse planning.
        super()._initial_groups(text)
        budget = max(200, min(600, int(getattr(self.llm, 'max_output_tokens', 4096))//8))
        return self._plan_historical_groups(text, budget)

    def __init__(self, llm_client, *, snomed_catalog=None, **kwargs):
        self.snomed = snomed_catalog
        self.teaching_catalog = list(kwargs.pop('catalog', ()))
        super().__init__(llm_client, catalog=self.teaching_catalog or [dict(term_id='new', label='', fields=[], examples=[])],
                         system_prompt=load_prompt('compact_events_system'), **kwargs)
        self.snomed_fingerprint = self.snomed.metadata().get('sha256') if self.snomed else None
        self.mapping_system = load_prompt('snomed_mapping_system')
        self.prompt_version = VERSION
        self.prompt_digest = content_hash(self.system, self.mapping_system)
        self.catalog_digest = content_hash(self.catalog_digest, self.snomed.digest if self.snomed else 'catalog-missing')

    @property
    def model_digest(self):
        return content_hash(VERSION, super().model_digest)

    def _cards(self, text):
        # One bundle prevents splitting the source once per subset of teaching terms.
        # Examples guide recognition; they never constrain which events are extracted.
        from .lexicon_guidance import words
        query = words(text)
        ranked = sorted(self.teaching_catalog, key=lambda c: -len(query & words(c['label']+' '+c.get('definition',''))))
        cards = []
        for card in ranked[:12]:
            cards.append({**card, 'examples': card.get('examples', [])[:2]})
        # Bound optional guidance, reserving context for the source itself.
        while cards and len(json.dumps(cards, ensure_ascii=False)) > 6000:
            cards.pop()
        return [dict(term_id='new', guidance=cards)]

    def extract_document(self, **kwargs):
        ready = kwargs.pop('evidence_ready_callback', None)
        self._local.history_document = {k:kwargs.get(k) for k in ('patient_id','document_type','document_date')}
        self._local.retrieval_cache = {}
        self._local.cancel_check = kwargs.get('cancel_check')
        try:
            rows = super().extract_document(**kwargs)
        except IncompleteAtomicExtraction as exc:
            if ready:
                ready(exc.evidence)
            self._map(exc.evidence, kwargs)
            exc.metrics = self.last_extraction_metrics()
            raise
        if ready:
            ready(rows)
        self._map(rows, kwargs)
        return rows

    def _map(self, rows, kwargs):
        cancelled = kwargs.get('cancel_check')
        progress = kwargs.get('chunk_progress_callback')
        stage_progress = kwargs.get('stage_progress_callback')
        if stage_progress:
            stage_progress(f'Codifica SNOMED: {len(rows)} eventi estratti')
        def check():
            if cancelled and cancelled():
                raise AtomicExtractionCancelled('Codifica SNOMED interrotta')
        if self.snomed is None or not self.snomed.available:
            if stage_progress:
                stage_progress('Catalogo SNOMED assente: eventi conservati con codifica da completare')
            for row in rows:self._unmapped(row, 'Catalogo SNOMED CT non importato.')
            self._local.metrics.update(snomed_mapped=0, snomed_unmapped=len(rows))
            return
        if self.snomed.metadata().get('sha256') != self.snomed_fingerprint:
            raise ValueError('Catalogo SNOMED CT cambiato: riavviare estrazione.')
        for row in rows:
            history_id = row.data.get('historical_reuse_id')
            if history_id and self.checkpoints:
                cached = self.checkpoints.load(kwargs['patient_id'],content_hash('historical-mapping-v1',self.model_digest,history_id))
                if cached and cached['status']=='completed':
                    saved = cached['result']
                    concept = self.snomed.lookup(saved.get('snomed_concept_id'))
                    if concept and concept['active']:
                        row.data.update({k:v for k,v in saved.items() if k.startswith('snomed_')})
                        self._apply_code(row,concept,saved.get('snomed_mapping_reason','Riuso storico verificato'),'historical_reuse')
                        self._local.metrics['historical_mappings_reused'] = self._local.metrics.get('historical_mappings_reused',0)+1
        # English labels are requested only at linking time when multilingual
        # retrieval is unavailable. They never replace the Italian source.
        if not self.snomed.metadata().get('embedding_model'):
            self._translate_queries(rows, kwargs)
        pending = []
        for index, row in enumerate(rows):
            check()
            if row.data.get('snomed_mapping_status') == 'proposed':
                continue
            if row.fact_type == 'laboratory_test':
                self._unmapped(row, 'Risultato di laboratorio: codifica LOINC, non SNOMED CT.')
                continue
            row.data['snomed_release'] = self.snomed.metadata().get('release')
            try:
                candidates = self.snomed.search(row.normalized_entity, row.data['snomed_search_en'])
                row.data['snomed_candidates'] = candidates
                if not candidates:
                    self._unmapped(row, 'Nessun candidato nel catalogo.')
                    continue
                pending.append((row, dict(id=str(index), label=row.normalized_entity, quote=row.source_text,
                    assertion=row.assertion, certainty=row.certainty, type=row.fact_type,
                    attributes=row.data.get('attributes', {}), candidates=candidates)))
            except Exception as exc:
                self._unmapped(row, f'Ricerca non disponibile: {exc}')
        while pending:
            check()
            batch = pending[:8]
            del pending[:8]
            while len(batch)>1 and not self._mapping_fits(batch):
                pending.insert(0,batch.pop())
            if not self._mapping_fits(batch):
                for row,_ in batch:
                    self._unmapped(row, 'Contesto insufficiente per codifica SNOMED.')
                continue
            payload = [data for _,data in batch]
            contract = self._mapping_schema(payload)
            prompt = json.dumps(payload, ensure_ascii=False)
            key = content_hash('snomed_mapping', self.model_digest, kwargs['patient_id'], kwargs['document_id'], prompt)
            cached = self.checkpoints.load(kwargs['patient_id'], key) if self.checkpoints else None
            try:
                if cached and cached['status']=='completed':
                    response = cached['result']
                    self._local.metrics['groups_cached'] += 1
                else:
                    started = time.monotonic()
                    try:
                        response = self.llm.generate_structured(prompt, self.mapping_system, contract,
                            max_tokens=int(getattr(self.llm,'max_output_tokens',4096)))
                    finally:
                        meta = getattr(self.llm, 'last_generation_metadata', lambda: {})() or {}
                        for name in ('prompt_tokens', 'completion_tokens', 'prompt_ms', 'predicted_ms'):
                            self._local.metrics[name] += meta.get(name, 0) or 0
                        if self.checkpoints:
                            self.checkpoints.record_call(kwargs['patient_id'], kwargs['document_id'], key,
                                'snomed_mapping', 'attempted', {**meta,'elapsed_seconds':time.monotonic()-started})
                        self._local.metrics['llm_calls'] += 1
                        self._local.metrics['elapsed_seconds'] += time.monotonic()-started
                check()
                jsonschema.validate(response, contract)
                selections = response['mappings']
                if len(selections)!=len(payload) or {s['id'] for s in selections}!={p['id'] for p in payload}:
                    raise ValueError('Risposta incompleta o identificativi duplicati.')
                # Validate the complete response before mutating any occurrence.
                by_id = {s['id']:s for s in selections}
                for _,data in batch:
                    code = by_id[data['id']]['code']
                    if code is not None and code not in {c['code'] for c in data['candidates']}:
                        raise ValueError('Codice non presente nei candidati della menzione.')
                for row,data in batch:
                    selected = by_id[data['id']]
                    if selected['code'] is None:
                        self._unmapped(row, selected['reason'])
                        continue
                    concept = self.snomed.lookup(selected['code'])
                    if not concept or not concept['active']:
                        self._unmapped(row, 'Concetto non più attivo: rivedere.')
                        continue
                    self._apply_code(row, concept, selected['reason'], 'recovery')
                if self.checkpoints and (not cached or cached['status'] != 'completed'):
                    self.checkpoints.save(kwargs['patient_id'], kwargs['document_id'], key,
                        'completed', response, self.last_extraction_metrics(), None)
            except AtomicExtractionCancelled:
                raise
            except Exception as exc:
                for row,_ in batch:
                    self._unmapped(row, f'Codifica da rivedere: {exc}')
            if progress:
                done = len(rows)-len(pending)
                progress(done, len(rows))
        self._local.metrics['snomed_mapped'] = sum(r.data.get('snomed_mapping_status')=='proposed' for r in rows)
        self._local.metrics['snomed_unmapped'] = len(rows)-self._local.metrics['snomed_mapped']
        self._local.metrics['snomed_joint_mapped'] = sum(r.data.get('snomed_mapping_phase')=='joint' for r in rows)
        self._local.metrics['snomed_recovery_mapped'] = sum(r.data.get('snomed_mapping_phase')=='recovery' for r in rows)
        if self.checkpoints:
            for row in rows:
                history_id = row.data.get('historical_reuse_id')
                if history_id and row.data.get('snomed_mapping_status')=='proposed':
                    self.checkpoints.save(kwargs['patient_id'],kwargs['document_id'],
                        content_hash('historical-mapping-v1',self.model_digest,history_id),'completed',
                        {k:v for k,v in row.data.items() if k.startswith('snomed_')},self.last_extraction_metrics())

    def _translate_queries(self, rows, kwargs):
        pending = [r for r in rows if r.fact_type != 'laboratory_test' and not r.data.get('snomed_search_en')]
        for start in range(0, len(pending), 8):
            if kwargs.get('cancel_check') and kwargs['cancel_check']():
                raise AtomicExtractionCancelled('Traduzione terminologica interrotta')
            batch = pending[start:start+8]
            payload = [{'id': str(i), 'label': r.normalized_entity, 'context': r.source_text}
                       for i,r in enumerate(batch)]
            contract = {'type':'object','additionalProperties':False,'required':['queries'],
                'properties':{'queries':{'type':'array','minItems':len(batch),'maxItems':len(batch),
                    'items':{'type':'object','additionalProperties':False,'required':['id','english'],
                        'properties':{'id':{'type':'string','enum':[p['id'] for p in payload]},
                                      'english':{'type':'string','minLength':1,'maxLength':160}}}}}}
            prompt = json.dumps(payload,ensure_ascii=False)
            system = ('Translate each clinical label into a short English terminology search query. '
                      'Use context to disambiguate; add no diagnosis or undocumented specificity. '
                      'Do not generate codes. Return queries with the original id and english.')
            key = content_hash(self.model_digest, 'english_query', prompt)
            cached = self.checkpoints.load(kwargs['patient_id'],key) if self.checkpoints else None
            try:
                if cached and cached['status']=='completed':
                    result = cached['result']
                else:
                    t = time.monotonic()
                    try:
                        result = self.llm.generate_structured(prompt,system,contract,
                            max_tokens=min(1024,int(getattr(self.llm,'max_output_tokens',4096))))
                    finally:
                        self._local.metrics['llm_calls'] += 1
                        self._local.metrics['elapsed_seconds'] += time.monotonic()-t
                        meta = getattr(self.llm,'last_generation_metadata',lambda:{})() or {}
                        for name in ('prompt_tokens','completion_tokens','prompt_ms','predicted_ms'):
                            self._local.metrics[name] += meta.get(name,0) or 0
                        if self.checkpoints:
                            self.checkpoints.record_call(kwargs['patient_id'],kwargs['document_id'],key,
                                'terminology_query','attempted',{**meta,'elapsed_seconds':time.monotonic()-t})
                jsonschema.validate(result,contract)
                selections={x['id']:x['english'] for x in result['queries']}
                if len(selections)!=len(batch):raise ValueError('Query mancanti o duplicate')
                for i,row in enumerate(batch):row.data['snomed_search_en']=selections[str(i)]
                if self.checkpoints and not cached:
                    self.checkpoints.save(kwargs['patient_id'],kwargs['document_id'],key,'completed',result,{},None)
            except Exception as exc:
                for row in batch:row.data['snomed_query_error']=str(exc)[:300]

    def _apply_code(self, row, concept, reason, phase):
        row.terminology_system = 'SNOMED_CT'
        row.terminology_code = concept['code']
        row.mapping_status = 'mapped'
        row.data.update(snomed_mapping_status='proposed', snomed_mapping_reason=reason,
            snomed_mapping_phase=phase, snomed_release=self.snomed.metadata().get('version_uri') or self.snomed.metadata().get('release'),
            snomed_fsn=concept['fsn'], snomed_term=concept['term'], snomed_concept_id=concept['code'])
        row.data['lexicon_term_id'] = 'snomed:'+concept['code']

    @staticmethod
    def _unmapped(row, reason):
        row.data.update(snomed_mapping_status='needs_review', snomed_mapping_reason=str(reason)[:500])
        row.status = 'needs_review'

    def _mapping_fits(self, batch):
        prompt = json.dumps([data for _,data in batch], ensure_ascii=False)+self.mapping_system
        return self._fits(prompt, self._mapping_schema([data for _,data in batch]))

    @staticmethod
    def _mapping_schema(payload):
        codes = sorted({c['code'] for p in payload for c in p['candidates']})
        return dict(type='object', additionalProperties=False, required=['mappings'], properties={
            'mappings':dict(type='array', items=dict(type='object', additionalProperties=False,
                required=['id','code','reason'], properties={
                    'id':dict(type='string', enum=[p['id'] for p in payload]),
                    'code':{'type':['string','null'], 'enum':codes+[None]},
                    'reason':dict(type='string')}))})

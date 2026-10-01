"""Optional, independently checkpointed relations between existing source events."""
import json
import re
import time

import jsonschema

from .evidence_utils import content_hash
from .grounded_sources import AtomicExtractionCancelled
from .sentence_groups import clinical_sentences

# A recall-limited trigger for enrichment, never a filter on event extraction.
CUES = re.compile(r'\b(?:sospes\w*\s+per|dovut\w*\s+a|a causa di|secondari\w*\s+a|'
                  r'regred\w*|risolt\w*|trattat\w*\s+con|terapia\s+per|prima\s+di|'
                  r'dopo|successivamente|in seguito a)\b', re.I)


def enrich_relations(extractor, rows, kwargs):
    spans = clinical_sentences(kwargs['text'])
    metrics = extractor._local.metrics
    metrics.update(relation_calls=0, relation_links=0, relation_issues=0)
    for row in rows:
        row.data['relation_review'] = {'status': 'not_requested', 'issues': []}
    visited = set()
    for i, span in enumerate(spans):
        if not CUES.search(span.text):
            continue
        if kwargs.get('cancel_check') and kwargs['cancel_check']():
            raise AtomicExtractionCancelled('Relazioni interrotte; annotazioni conservate')
        neighborhood = spans[max(0,i-1):i+2]
        left, right = neighborhood[0].start, neighborhood[-1].end
        selected = [r for r in rows if any(left <= p['start'] < right
                    for p in r.data.get('source_spans', []))]
        if len(selected) < 2:
            continue
        identity = (left, right, tuple(r.evidence_id for r in selected))
        if identity in visited:
            continue
        visited.add(identity)
        by_id = {str(n): row for n,row in enumerate(selected)}
        passage = kwargs['text'][left:right]
        payload = {'text': passage, 'events': [
            {'id': k, 'label': row.normalized_entity, 'quote': row.source_text,
             'assertion': row.assertion, 'certainty': row.certainty,
             'subject': row.data.get('experiencer'), 'state': row.clinical_status}
            for k,row in by_id.items()]}
        contract = {'type':'object','additionalProperties':False,'required':['links'],
            'properties':{'links':{'type':'array','maxItems':20,'items':{
                'type':'object','additionalProperties':False,
                'required':['source','target','kind','support'],
                'properties':{'source':{'type':'string','enum':list(by_id)},
                    'target':{'type':'string','enum':list(by_id)},
                    'kind':{'type':'string','enum':['before','after','reason_for','resolution_of','treats']},
                    'support':{'type':'string','minLength':1,'maxLength':300}}}}}}
        prompt = json.dumps(payload,ensure_ascii=False,separators=(',',':'))
        key = content_hash(extractor.model_digest,'relations',kwargs['document_id'],prompt)
        cached = extractor.checkpoints.load(kwargs['patient_id'],key) if extractor.checkpoints else None
        errors, response = [], None
        progress = kwargs.get('stage_progress_callback')
        if progress:
            progress(f'Relazioni mirate: {len(selected)} eventi; annotazioni già salvate nei checkpoint')
        try:
            if len(selected)>20 or not extractor._fits(prompt+extractor.relation_system,contract):
                raise ValueError('Passaggio troppo denso: relazioni da rivedere separatamente')
            if cached and cached['status']=='completed':
                response = cached['result']
            else:
                started = time.monotonic()
                status = 'failed'
                try:
                    response = extractor.llm.generate_structured(prompt,extractor.relation_system,contract,
                        max_tokens=min(1536,int(getattr(extractor.llm,'max_output_tokens',4096))))
                    status = 'completed'
                finally:
                    elapsed = time.monotonic()-started
                    metrics['llm_calls'] += 1
                    metrics['relation_calls'] += 1
                    metrics['elapsed_seconds'] += elapsed
                    meta = getattr(extractor.llm,'last_generation_metadata',lambda:{})() or {}
                    for name in ('prompt_tokens','completion_tokens','prompt_ms','predicted_ms'):
                        metrics[name] += meta.get(name,0) or 0
                    if extractor.checkpoints:
                        extractor.checkpoints.record_call(kwargs['patient_id'],kwargs['document_id'],key,
                            'relations',status,{**meta,'elapsed_seconds':elapsed})
            if not isinstance(response,dict) or set(response)!={'links'} or not isinstance(response['links'],list):
                raise ValueError('Risposta relazioni non conforme')
            if len(response['links'])>20:
                raise ValueError('Troppe relazioni nella risposta')
            for link in response['links']:
                try:
                    jsonschema.validate(link,contract['properties']['links']['items'])
                    if link['source']==link['target']:
                        raise ValueError('Autorelazione')
                    pattern = r'\s+'.join(re.escape(w) for w in link['support'].split())
                    matches = list(re.finditer(pattern,passage)) if pattern else []
                    if len(matches)!=1:
                        raise ValueError('Supporto non letterale o ambiguo')
                    target = by_id[link['target']]
                    proposed = {'kind':link['kind'],'target_evidence_id':target.evidence_id,
                        'target_start':target.data['source_spans'][0]['start'],
                        'target_text':target.source_text,'support':matches[0].group(),
                        'support_start':left+matches[0].start(),'status':'proposed'}
                    links = by_id[link['source']].data.setdefault('source_relations',[])
                    if proposed not in links:
                        links.append(proposed)
                        metrics['relation_links'] += 1
                except (ValueError,TypeError,KeyError,jsonschema.ValidationError) as exc:
                    errors.append(str(exc).splitlines()[0][:200])
        except AtomicExtractionCancelled:
            raise
        except Exception as exc:
            errors.append(str(exc)[:300])
        metrics['relation_issues'] += len(errors)
        for row in selected:
            review = row.data['relation_review']
            review['issues'].extend(errors)
            review['status'] = 'needs_review' if review['issues'] else 'completed'
        if extractor.checkpoints and not cached:
            extractor.checkpoints.save(kwargs['patient_id'],kwargs['document_id'],key,
                'needs_review' if errors else 'completed',response or {},metrics,'; '.join(errors) or None)

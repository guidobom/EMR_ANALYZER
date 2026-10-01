"""Internal source grounding, deterministic clinical dates and resumable LLM calls."""
import json
from pathlib import Path
from datetime import date, timedelta
import re
import threading
import time

import jsonschema

from .evidence_utils import content_hash
from .sentence_groups import clinical_sentences, SentenceGroup, split_group
from .temporal import normalize_clinical_date, NormalizedClinicalDate
from ..models.clinical_evidence import ClinicalEvidence
from ..models.clinical_pipeline import ATOMIC_FACT_TYPES
from ..prompt_catalog import load_prompt
from ..settings import ClinicalPipelinePolicy
from ..extraction.llm_client import OutputLimitError

VERSION = 'fhir_events_v1'
METHOD = 'fhir_events_v1'


class AtomicExtractionCancelled(RuntimeError):
    pass


class IncompleteAtomicExtraction(RuntimeError):
    def __init__(self, evidence, issues, metrics):
        self.evidence, self.issues, self.metrics = evidence, issues, metrics
        details = '; '.join(str(e) for issue in issues for e in issue.get('errors', []))[:600]
        super().__init__(f'Estrazione dal Lessico incompleta: {details}')


def catalog_snapshot(repo):
    if repo is None:
        return []
    examples = repo.extraction_examples()
    result = []
    for term in repo.terms():
        result.append(dict(term_id=term['id'], label=term['label'],
            **repo.event_definition(term['id']),
            examples=[{k: row[k] for k in ('id', 'quote', 'assertion', 'role')}
                      for row in examples if row['term_id'] == term['id']]))
    return sorted(result, key=lambda c: c['term_id'])


def schema(term_ids):
    def enum(*values):
        return {'type': 'string', 'enum': list(values)}
    fields = dict(
        term_id=enum(*term_ids), type=enum(*ATOMIC_FACT_TYPES),
        quote={'type': 'string', 'minLength': 1}, sentence={'type': 'integer', 'minimum': 1},
        assertion=enum('present', 'absent', 'unknown'),
        certainty=enum('confirmed', 'suspected', 'possible', 'unknown'),
        subject=enum('patient', 'family', 'other', 'unknown'),
        temporality=enum('current', 'historical', 'hypothetical', 'unknown'),
        state={'type': ['string', 'null']},
        date_mode=enum('explicit', 'relative', 'document', 'unknown'),
        date_quote={'type': ['string', 'null']}, date_ref={'type': ['integer', 'null']},
        event_kind=enum('continuous', 'discrete'), episode_quote={'type': ['string', 'null']},
        attributes={'type': 'object', 'additionalProperties': {'type': 'string'}},
    )
    return {'type': 'object', 'additionalProperties': False, 'required': ['occurrences'],
            'properties': {'occurrences': {'type': 'array', 'items': {
                'type': 'object', 'additionalProperties': False,
                'properties': fields, 'required': list(fields)}}}}


class GroundedSourceReader:
    def __init__(self, llm_client, *, catalog=(), policy=None, checkpoint_repo=None, system_prompt=None):
        self.llm = llm_client
        self.catalog = list(catalog)
        self.policy = policy or ClinicalPipelinePolicy()
        self.checkpoints = checkpoint_repo
        self.system = system_prompt or load_prompt('snomed_mentions_system')
        self.prompt_version = VERSION
        self.prompt_digest = content_hash(self.system)
        self.catalog_digest = content_hash(json.dumps(self.catalog, sort_keys=True, ensure_ascii=False))
        self._local = threading.local()

    @property
    def model_name(self):
        return str(getattr(self.llm, 'model', ''))

    @property
    def model_digest(self):
        return content_hash(VERSION, self.prompt_digest, self.catalog_digest,
            json.dumps({key: getattr(self.llm, key, None) for key in (
                'model', 'model_path', 'temperature', 'seed', 'top_p', 'top_k',
                'context_length', 'max_output_tokens', 'thinking_enabled')}, sort_keys=True))

    def last_extraction_metrics(self):
        return dict(getattr(self._local, 'metrics', {}))

    def _cards(self, text):
        from .lexicon_guidance import words
        query = words(text)
        cards = []
        for term in self.catalog:
            examples = sorted(term.get('examples', []),
                key=lambda r: (-len(query & words(r['quote'])), r['id']))
            # All terms are always searched; only the illustrative examples
            # are ranked. Keep a positive and a counterexample when present.
            picked = []
            for role in ('example', 'counterexample'):
                picked.extend([e for e in examples if e['role'] == role][:2 if role == 'example' else 1])
            cards.append({**term, 'examples': picked})
        return cards

    def _prompt(self, group, cards, document_type, document_date):
        return (group.prompt(document_type, document_date)
                + '\nCATALOGO CHIUSO (esempi didattici, non fonti):\n'
                + json.dumps(cards, ensure_ascii=False))

    def _fits(self, prompt, contract):
        # Include schema cost conservatively; structured backends may inject it.
        payload = prompt + '\n' + json.dumps(contract, ensure_ascii=False)
        counter = getattr(self.llm, 'count_prompt_tokens', None)
        size = counter(payload, self.system) if callable(counter) else len((payload+self.system).encode())
        output = int(getattr(self.llm, 'max_output_tokens', 4096))
        return size + output + 256 <= int(getattr(self.llm, 'context_length', 32768))

    def _prepare_cards(self, group, cards):
        return cards

    def _schema(self, cards):
        return schema([c['term_id'] for c in cards])

    def _initial_groups(self, text):
        return [SentenceGroup(tuple(clinical_sentences(text)))] if text.strip() else []

    def _request_key(self, patient_id, document_id, text, document_type, document_date, prompt):
        return content_hash(patient_id, document_id, text, document_type, document_date,
                            self.model_digest, prompt)

    def _resolve_context(self, result, group, **kwargs):
        return result

    def _repair_prompt(self, prompt, result, errors):
        return prompt + '\nCorreggi la risposta: ' + '; '.join(errors)[:800]

    def extract_document(self, *, patient_id, document_id, document_type, document_date,
                         text, geometry_path=None, cancel_check=None, chunk_progress_callback=None,
                         stage_progress_callback=None):
        if not self.catalog:
            raise ValueError('Il Lessico condiviso è vuoto: crea e conferma almeno un termine prima di estrarre le evidenze.')
        self._local.metrics = dict(llm_calls=0, source_chunks=0, groups_cached=0,
            recovery_calls=0, prompt_tokens=0, completion_tokens=0, elapsed_seconds=0,
            prompt_ms=0, predicted_ms=0, unresolved=0)
        def check():
            if cancel_check and cancel_check():
                raise AtomicExtractionCancelled('Estrazione dal Lessico interrotta')
        check()
        self._local.geometry = None
        if geometry_path and Path(geometry_path).exists():
            from ..pipeline.pdf_extractor import PdfExtractionResult, PdfPlumberExtractor
            path = Path(geometry_path)
            try:
                self._local.geometry = (PdfPlumberExtractor().convert(path) if path.suffix.lower() == '.pdf'
                    else PdfExtractionResult.from_dict(json.loads(path.read_text(encoding='utf-8'))))
            except (OSError, ValueError, TypeError):
                pass
        cards = self._cards(text)
        queue = [(group, cards) for group in self._initial_groups(text)]
        work, issues, evidence = [], [], []
        # Prefer the entire document to resolve distant temporal references.
        # If it cannot fit, partition catalog and/or source without dropping
        # terms or sentences. An indivisible request becomes an explicit error.
        while queue:
            check()
            group, subset = queue.pop(0)
            subset = self._prepare_cards(group, subset)
            contract = self._schema(subset)
            prompt = self._prompt(group, subset, document_type, document_date)
            if self._fits(prompt, contract):
                work.append((group, subset, prompt, contract))
            elif len(subset) > 1:
                mid = len(subset)//2
                queue[:0] = [(group, subset[:mid]), (group, subset[mid:])]
            else:
                children = split_group(group)
                if children:
                    queue[:0] = [(g, subset) for g in children]
                elif len(subset)==1 and (subset[0].get('candidates') or subset[0].get('guidance')):
                    reduced = dict(subset[0])
                    field = 'candidates' if reduced.get('candidates') else 'guidance'
                    reduced[field] = reduced[field][:len(reduced[field])//2]
                    reduced['candidate_limit'] = len(reduced.get('candidates', []))
                    queue.insert(0, (group, [reduced]))
                else:
                    issues.append({'errors': ['Contesto insufficiente per una frase e la sua scheda: aumenta il contesto o riduci il limite di risposta.']})
        self._local.metrics['source_chunks'] = len(work)
        for index, (group, subset, prompt, contract) in enumerate(work):
            check()
            key = self._request_key(patient_id, document_id, text, document_type, document_date, prompt)
            prior = self.checkpoints.load(patient_id, key) if self.checkpoints else None
            reuse = getattr(self, '_load_historical_group', None)
            if callable(reuse) and not (prior and prior['status'] == 'completed'):
                reused = reuse(group, subset, patient_id, document_type, document_date, text, document_id, contract)
                if reused is not None:
                    prior = {'status': 'completed', 'result': reused}
                    if stage_progress_callback:
                        stage_progress_callback(f'Riuso storico verificato: gruppo {index+1}/{len(work)}; nessuna richiesta di annotazione LLM')
            result = None
            accepted_results = []
            if prior and prior['status'] == 'completed':
                result = prior['result']
                self._local.metrics['groups_cached'] += 1
            errors, rows, retained = [], [], []
            previous_errors = []
            rescheduled = False
            for attempt in range(2):
                try:
                    if result is None:
                        check()
                        if stage_progress_callback:
                            stage_progress_callback(
                                f"{'Correzione mirata' if attempt else 'Annotazione'}: gruppo {index+1}/{len(work)}; "
                                f"{len(evidence)+len(retained)} eventi validati")
                        started = time.monotonic()
                        call_status = 'failed'
                        try:
                            result = self.llm.generate_structured(prompt, self.system, contract,
                                max_tokens=int(getattr(self.llm, 'max_output_tokens', 4096)))
                            call_status = 'completed'
                        finally:
                            metrics = self._local.metrics
                            metrics['llm_calls'] += 1
                            metrics['recovery_calls'] += int(attempt > 0)
                            metrics['elapsed_seconds'] += time.monotonic()-started
                            meta = getattr(self.llm, 'last_generation_metadata', lambda: {})() or {}
                            for name in ('prompt_tokens', 'completion_tokens', 'prompt_ms', 'predicted_ms'):
                                metrics[name] += meta.get(name, 0) or 0
                            if self.checkpoints:
                                self.checkpoints.record_call(patient_id, document_id, key,
                                    'recover' if attempt else 'extract', call_status,
                                    {**meta, 'elapsed_seconds': time.monotonic()-started})
                    check()
                    if not prior or prior['status'] != 'completed':
                        result = self._resolve_context(result, group, patient_id=patient_id,
                            document_id=document_id, cancel_check=cancel_check)
                    rows, errors = self._validate(result, contract, group, subset, text,
                                                patient_id, document_id, document_date)
                    if (attempt and getattr(self, 'partial_repairs', False)
                            and previous_errors and not rows and not errors):
                        errors = ['Correzione vuota: problemi precedenti da rivedere.']
                except AtomicExtractionCancelled:
                    raise
                except OutputLimitError as exc:
                    if stage_progress_callback:
                        stage_progress_callback(f'Risposta troncata: recupero eventi completi e suddivisione del gruppo {index+1}')
                    recover = getattr(self, '_recover_partial', None)
                    if callable(recover):
                        partial = recover(exc.partial_content)
                        partial = self._resolve_context(partial, group, patient_id=patient_id,
                            document_id=document_id, cancel_check=cancel_check)
                        partial_rows, _ = self._validate(partial, contract, group, subset, text,
                                                       patient_id, document_id, document_date)
                        evidence.extend(partial_rows)
                        if self.checkpoints:
                            self.checkpoints.save(patient_id, document_id, key+':truncated', 'diagnostic',
                                {'partial_content': exc.partial_content, 'recovered': len(partial_rows)},
                                self.last_extraction_metrics(), str(exc))
                    children = split_group(group)
                    planned = []
                    for child in children:
                        child_cards = self._prepare_cards(child, subset)
                        child_contract = self._schema(child_cards)
                        planned.append((child, child_cards,
                            self._prompt(child, child_cards, document_type, document_date), child_contract))
                    if planned and all(self._fits(p, c) for _, _, p, c in planned):
                        work[index+1:index+1] = planned
                        self._local.metrics['source_chunks'] = len(work)
                        rescheduled = True
                        errors = []
                        break
                    errors = [f'Risposta troncata per una frase indivisibile: {exc}']
                    break
                except Exception as exc:
                    errors = [f'{type(exc).__name__}: {str(exc)[:240]}']
                # A successful corrected full response supersedes the first;
                # on failure retain all source-validated partial occurrences.
                retained.extend(rows)
                if getattr(self, 'partial_repairs', False) and result:
                    accepted_results.extend(result.get('events', []))
                if errors and self.checkpoints:
                    self.checkpoints.save(patient_id, document_id, key+f':attempt:{attempt}', 'diagnostic',
                        {'response': result, 'raw': getattr(self.llm, 'last_response_text', lambda: '')()},
                        self.last_extraction_metrics(), '; '.join(errors))
                if not errors:
                    if not getattr(self, 'partial_repairs', False):
                        retained = rows
                    break
                if attempt == 0:
                    previous_errors = list(errors)
                    repair = self._repair_prompt(prompt, result, errors)
                    if not self._fits(repair, contract):
                        break
                    prompt, result = repair, None
            evidence.extend(retained)
            if errors:
                issues.append({'key': key, 'errors': errors})
            if self.checkpoints:
                saved_result = result or {}
                if getattr(self, 'partial_repairs', False) and not errors and not rescheduled:
                    # Persist the union of valid annotations, excluding failed attempts.
                    valid = []
                    for raw in accepted_results:
                        _, problems = self._validate({'events': [raw]}, contract, group, subset, text,
                                                      patient_id, document_id, document_date)
                        if not problems and raw not in valid:
                            valid.append(raw)
                    saved_result = {'events': valid}
                self.checkpoints.save(patient_id, document_id, key,
                    'split' if rescheduled else 'needs_review' if errors else 'completed', saved_result,
                    self.last_extraction_metrics(), '; '.join(errors) or None)
                remember = getattr(self, '_save_historical_group', None)
                if callable(remember) and not errors and not rescheduled and not previous_errors:
                    remember(group, subset, patient_id, document_type, document_date,
                             text, document_id, contract, saved_result, retained)
            if chunk_progress_callback:
                chunk_progress_callback(index+1, len(work))
        evidence = list({e.evidence_id: e for e in evidence}.values())
        self._local.metrics.update(final_items=len(evidence), unresolved=len(issues))
        if issues:
            raise IncompleteAtomicExtraction(evidence, issues, self.last_extraction_metrics())
        return evidence

    def _validate(self, result, contract, group, cards, text, pid, did, doc_date, *, source_selection=None):
        if not isinstance(result, dict) or set(result) != {'occurrences'} or not isinstance(result['occurrences'], list):
            return [], ['Risposta non conforme: manca occurrences']
        rows, errors = [], []
        spans = group.numbered
        catalog = {c['term_id']: c for c in cards}
        for item in result['occurrences']:
            try:
                jsonschema.validate(item, contract['properties']['occurrences']['items'])
                ref = item['sentence']
                if ref not in group.target_ids:
                    raise ValueError('Evento non ancorato a una frase OBIETTIVO')
                span = spans[ref-1]
                quote = item['quote']
                pattern = r'\s+'.join(re.escape(word) for word in quote.split())
                matches = list(re.finditer(pattern, span.text)) if pattern else []
                if source_selection is not None:
                    start, end = source_selection
                    if not span.start <= start < end <= span.end or text[start:end] != quote:
                        raise ValueError('Intervallo sorgente non verificabile')
                elif len(matches) != 1:
                    raise ValueError('Citazione non letterale o ambigua nella frase indicata')
                else:
                    start = span.start + matches[0].start()
                    end = span.start + matches[0].end()
                quote = text[start:end]
                if item['episode_quote'] and item['episode_quote'] not in quote:
                    raise ValueError('Identificatore episodio non citato nel testo origine')
                if item['assertion'] == 'absent' and re.search(r'non\s+(?:si\s+)?(?:pu[oò]|possibile).*esclud|per\s+esclud', quote, re.I):
                    raise ValueError('Un dubbio diagnostico non è una negazione')
                temporal, provenance = self._date(item, spans, doc_date)
                term = catalog[item['term_id']]
                attrs = item['attributes']
                from .lexicon_structure import occurrence_values
                values, units = occurrence_values(term, attrs)
                main_fields = term.get('field_definitions', [])
                main_value = values.get(main_fields[0]['name']) if main_fields else None
                main_unit = units.get(main_fields[0]['name']) if main_fields else None
                data = dict(lexicon_term_id=term['term_id'], lexicon_catalog_digest=self.catalog_digest,
                    lexicon_label=term['label'], experiencer=item['subject'],
                    lexicon_category=term.get('category', ''),
                    lexicon_dedup_rule=term.get('dedup_rule', 'auto'),
                    structured_values=values, value_units=units, value_type=term.get('value_type', 'none'),
                    event_kind=item['event_kind'], episode_quote=item['episode_quote'],
                    attributes=attrs, date_provenance=provenance,
                    source_spans=[dict(start=start, end=end, text=text[start:end], sentence_id=ref)],
                    source_version=content_hash(text), fact_type=item['type'])
                if item['type'] == 'medication':
                    data['therapy'] = {k: attrs[k] for k in ('dose', 'route', 'frequency', 'indication') if k in attrs}
                    data['therapy']['lifecycle_status'] = item['state']
                geometry = getattr(self._local, 'geometry', None)
                page, bbox = geometry.locate_source(text[start:end]) if geometry else (None, None)
                category = {'laboratory_test': 'laboratory_finding', 'radiology_finding': 'imaging_finding',
                            'clinical_decision': 'care_plan'}.get(item['type'], item['type'])
                ident = content_hash(pid, did, text, term['term_id'], start, end,
                    json.dumps(item, sort_keys=True, ensure_ascii=False))
                rows.append(ClinicalEvidence(patient_id=pid, document_id=did,
                    evidence_id='EVD_'+ident, category=category, fact_type=item['type'],
                    normalized_entity=term['label'], canonical_label=term['label'],
                    concept_original=term['label'], terminology_system='LOCAL_LEXICON',
                    terminology_code=term['term_id'], mapping_status='mapped',
                    source_text=text[start:end], assertion=item['assertion'], certainty=item['certainty'],
                    temporality=item['temporality'], clinical_status=item['state'],
                    observed_date=temporal.start, observed_date_end=temporal.end,
                    date_precision=temporal.precision, date_source=temporal.source,
                    document_date=doc_date, anatomical_site=attrs.get('site'),
                    laterality=attrs.get('laterality'), severity=attrs.get('severity'),
                    extraction_method=METHOD, model_name=self.model_name,
                    prompt_version=VERSION, data=data, typed_payload=dict(attrs, structured_values=values, value_units=units),
                    numeric_value=main_value if term.get('value_type') == 'number' else None,
                    value_text=((' '.join((str(main_value), main_unit or '')).strip()) if main_value is not None else None)
                        if term.get('value_type') in {'text', 'choice', 'number'} else
                        ('; '.join(f'{k}: {v} {units.get(k, "")}'.strip() for k,v in values.items()) or None),
                    unit=main_unit if term.get('value_type') == 'number' else None,
                    source_page=page, bbox=bbox,
                    status='needs_review' if (item['subject'] == 'unknown' or item['assertion'] == 'unknown'
                                             or provenance.get('needs_review')) else 'proposed'))
            except (ValueError, TypeError, IndexError, jsonschema.ValidationError) as exc:
                errors.append(str(exc).splitlines()[0][:240])
        return rows, errors

    @staticmethod
    def _date(item, spans, doc_date):
        mode, quote, ref = item['date_mode'], item['date_quote'], item['date_ref']
        if mode == 'unknown':
            if quote is not None or ref is not None:
                raise ValueError('Data sconosciuta con citazione temporale incoerente')
            return normalize_clinical_date(None), {'mode': mode}
        if type(ref) is not int or not 1 <= ref <= len(spans) or not quote or quote not in spans[ref-1].text:
            raise ValueError('Citazione temporale non verificabile')
        span = spans[ref-1]
        provenance = dict(mode=mode, quote=quote, start=span.start+span.text.index(quote),
                          end=span.start+span.text.index(quote)+len(quote))
        if mode == 'document':
            if item['temporality'] != 'current':
                raise ValueError('Data del documento assegnata a un evento non attuale')
            parsed = normalize_clinical_date(doc_date)
            temporal = NormalizedClinicalDate(parsed.start, parsed.end, parsed.precision,
                                              'document_context' if parsed.start else 'unknown', quote)
        else:
            temporal = normalize_clinical_date(quote, document_date=doc_date)
            relative = re.fullmatch(r"(?i)(ieri|oggi|l'altro ieri|(?:\d+|un|uno|due|tre|quattro|cinque|sei|sette) giorni? fa)", quote.strip())
            if mode == 'relative' and relative and doc_date:
                try:
                    anchor = date.fromisoformat(doc_date)
                    value = quote.strip().casefold()
                    words = {'un': 1, 'uno': 1, 'due': 2, 'tre': 3, 'quattro': 4, 'cinque': 5, 'sei': 6, 'sette': 7}
                    if value in {'oggi', 'ieri', "l'altro ieri"}:
                        days = {'oggi': 0, 'ieri': 1, "l'altro ieri": 2}[value]
                    else:
                        number = value.split()[0]
                        days = int(number) if number.isdigit() else words[number]
                    temporal = NormalizedClinicalDate((anchor-timedelta(days=days)).isoformat(),
                        None, 'day', 'relative_to_document', quote)
                    provenance['anchor_date'] = doc_date
                except (ValueError, TypeError):
                    pass
        if not temporal.start:
            provenance['needs_review'] = 'Espressione temporale non risolvibile: data conservata come sconosciuta'
        return temporal, provenance

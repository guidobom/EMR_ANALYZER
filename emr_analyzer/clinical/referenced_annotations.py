"""Source-addressed intermediate events: the LLM selects, Python quotes.

Word intervals are inclusive and 1-based, relative to the displayed sentence.
Bad temporal links quarantine the date, not a valid source-grounded event.
"""
from copy import deepcopy
import json
import re

import jsonschema

from .compact_annotations import CompactSourceReader, DATES
from .evidence_utils import content_hash

WORDS = re.compile(r'\S+')
TODAY = re.compile(r'\b(?:in data odierna|oggi)\b', re.I)
LETTERS = re.compile(r'[^\W\d_]', re.UNICODE)
GENERIC = re.compile(r'^(?:in (?:aumento|riduzione|diminuzione)|aumento|riduzione|'
                     r'stabile|stabilità|nei limiti|normale|conservata|negativo|positivo)$',re.I)
NUMERIC = re.compile(r'^\s*(<=|>=|<|>|≤|≥)?\s*([+-]?\d+(?:[.,]\d+)?)\s*$')
TEMPORAL_FIELDS = ('time','date_quote','date_s','date_mode')


class ReferencedSourceReader(CompactSourceReader):
    def _initial_groups(self, text):
        groups = super()._initial_groups(text)
        header_end = text.find('<!-- /emr-report-date -->')
        if header_end >= 0:
            header_end += len('<!-- /emr-report-date -->')
        # Metadata is not a dated clinical source. Keep stable explicit IDs for
        # clinical dates, including the complete phrase "in data odierna".
        times = {}
        for span in self._local.document_spans:
            if span.start < header_end:
                continue
            matches = {(m.start(),m.end()):m.group() for m in DATES.finditer(span.text)}
            for m in TODAY.finditer(span.text):
                matches[(m.start(),m.end())] = m.group()
            for (start,end),quote in sorted(matches.items()):
                times['T'+str(len(times)+1)] = (span,quote,span.start+start)
        self._local.times = times
        return groups

    def _schema(self, cards):
        contract = super()._schema(cards)
        item = contract['properties']['events']['items']
        del item['properties']['anchor']
        item['properties']['span'] = {'type':'array','minItems':2,'maxItems':2,
                                     'items':{'type':'integer','minimum':1}}
        item['required'] = ['s','span','label','type','assertion','certainty','subject']
        return contract

    def _prompt(self, group, cards, document_type, document_date):
        lines = [f'CONTESTO: tipo_dichiarato={document_type}; data_documento={document_date or "non disponibile"}',
                 'Ogni parola è preceduta da [numero]. span=[prima,ultima], estremi inclusi. '
                 'I numeri sono indirizzi, non fanno parte del testo. Estrai solo OBIETTIVO.']
        if group.copies:
            lines.append('Le frasi COPIA ripetono un fatto già annotato altrove per questo '
                         'paziente: non estrarne nulla; servono solo a capire il contesto.')
        for s in group.numbered:
            if s.sentence_id in group.target_ids:
                role = 'OBIETTIVO'
            elif s.sentence_id in group.copy_ids:
                role = 'COPIA'
            else:
                role = 'CONTESTO'
            lines.append(f'{role} [S{s.sentence_id}] '+ ' '.join(
                f'[{i}]'+m.group() for i,m in enumerate(WORDS.finditer(s.text),1)))
        # Reuse the date index and examples, without sending the source twice.
        original = super()._prompt(group,cards,document_type,document_date)
        suffix = original[original.index('\nRIFERIMENTI TEMPORALI'):]
        return '\n'.join(lines)+suffix

    @staticmethod
    def _document_time_supported(source, quote, selected):
        """Narrow positive rule; report availability is not procedure time."""
        if not TODAY.fullmatch(quote or '') or quote not in source:
            return False
        if re.search(r'\b(?:disponibil\w*|programm\w*|previst\w*|prenot\w*)\b',source,re.I):
            return False
        # Explicit same-day performance, not a statement that an older report
        # became available today. The selected event must follow that predicate.
        match = re.search(r'\b(?:esegue|eseguit[oaie]|effettua|effettuat[oaie])\s+'
                          r'(?:in data odierna|oggi)\b',source,re.I)
        return bool(match and source.find(selected) >= match.start())

    def _temporal(self, raw, group, selected):
        event = deepcopy(raw)
        repairs, warnings = [], []
        if 'time' in event and any(k in event for k in ('date_quote','date_s','date_mode')):
            warnings.append('Modalità temporali incompatibili: data non assegnata.')
        if 'time' in event and event['time'] in self._local.times:
            span,quote,_ = self._local.times[event['time']]
            if TODAY.fullmatch(quote):
                source = group.numbered[event['s']-1]
                if span.start == source.start and self._document_time_supported(source.text,quote,selected):
                    event.pop('time')
                    event.update(date_quote=quote,date_s=event['s'],date_mode='document',temporal='current')
                    repairs.append('Prestazione esplicitamente odierna: data del documento.')
                else:
                    warnings.append('Riferimento odierno non dimostra la data dello specifico evento.')
        if 'date_quote' in event:
            quote = event['date_quote']
            ref = event.get('date_s',event['s'])
            present = type(ref) is int and 1<=ref<=len(group.numbered) and quote in group.numbered[ref-1].text
            if not present:
                hits = [s for s in group.numbered if quote and s.text.count(quote)==1]
                if len(hits)==1:
                    event['date_s'] = hits[0].sentence_id
                    repairs.append('Riferimento alla citazione temporale ricollocato nell’unica frase visibile che la contiene.')
                else:
                    warnings.append('Citazione temporale assente o ambigua nelle frasi visibili.')
            if event.get('date_mode')=='document' or TODAY.fullmatch(quote):
                source = group.numbered[event['s']-1]
                if event.get('date_s',event['s'])==event['s'] and self._document_time_supported(source.text,quote,selected):
                    event['temporal']='current'
                    event['date_mode']='document'
                else:
                    warnings.append('Data del referto non dimostrata come data dell’evento.')
            elif event.get('date_mode')=='relative' and re.fullmatch(r'\d{1,2}[/.]\d{1,2}[/.]\d{2,4}|\d{4}-\d{2}-\d{2}',quote):
                event['date_mode']='explicit'
                repairs.append('Data assoluta normalizzata come explicit.')
        if warnings:
            for key in TEMPORAL_FIELDS:event.pop(key,None)
            event['date_unresolved']=True
        return event, repairs, warnings

    def _validate(self, result, contract, group, cards, text, pid, did, doc_date):
        if not isinstance(result,dict) or set(result)!={'events'} or not isinstance(result['events'],list):
            return [], ['Risposta non conforme: manca events']
        rows, errors = [], []
        legacy = CompactSourceReader._schema(self,cards)
        legacy['properties']['events']['items']['properties']['anchor'].pop('maxLength',None)
        for index, original in enumerate(result['events']):
            try:
                raw = {k:v for k,v in original.items() if k!='relations'} if isinstance(original,dict) else original
                jsonschema.validate(raw,contract['properties']['events']['items'])
                if raw['s'] in group.copy_ids:
                    # The statement is already certified elsewhere: this copy
                    # brings no new fact and needs no repair call.
                    metrics = getattr(self._local, 'metrics', None)
                    if isinstance(metrics, dict):
                        metrics['statement_copy_events_ignored'] = \
                            metrics.get('statement_copy_events_ignored', 0) + 1
                    continue
                if raw['s'] not in group.target_ids:
                    raise ValueError('Frase non OBIETTIVO')
                source = group.numbered[raw['s']-1]
                words = list(WORDS.finditer(source.text))
                first,last = raw['span']
                if not 1<=first<=last<=len(words):
                    raise ValueError(f'Intervallo parole non valido: S{raw["s"]} contiene {len(words)} parole')
                start,end = source.start+words[first-1].start(),source.start+words[last-1].end()
                quote = text[start:end]
                label = raw['label'].strip(' .;:')
                if not label or GENERIC.fullmatch(label):
                    raise ValueError('Attributo isolato: collega valore/andamento a un concetto clinico nominato')
                if not LETTERS.search(label):
                    # A number or a punctuation run is a value, never a concept.
                    raise ValueError('Etichetta senza parole: un valore non è un concetto clinico')
                prepared, repairs, warnings = self._temporal(raw,group,quote)
                prepared.pop('span')
                prepared['anchor']=quote
                found, problems = super()._validate({'events':[prepared]},legacy,group,cards,text,pid,did,doc_date,
                                                    source_selection=(start,end))
                if problems and all(any(t in p for t in ('temporale','Data del documento','Riferimento temporale')) for p in problems):
                    warnings.extend(problems)
                    for key in TEMPORAL_FIELDS:prepared.pop(key,None)
                    prepared['date_unresolved']=True
                    found,problems = super()._validate({'events':[prepared]},legacy,group,cards,text,pid,did,doc_date,
                                                     source_selection=(start,end))
                if problems:
                    raise ValueError('; '.join(problems))
                for row in found:
                    row.source_text=quote
                    row.data['source_spans']=[dict(start=start,end=end,text=quote,sentence_id=raw['s'])]
                    row.data['source_sentence']=dict(start=source.start,end=source.end,text=source.text)
                    row.data['source_selection']=dict(sentence=raw['s'],first_word=first,last_word=last)
                    row.data['intermediate_repairs']=repairs
                    if warnings:
                        row.status='needs_review'
                        row.data['date_provenance']['needs_review']='; '.join(warnings)
                        row.data['date_proposal']={k:raw[k] for k in TEMPORAL_FIELDS if k in raw}
                    geometry=getattr(self._local,'geometry',None)
                    row.source_page,row.bbox=geometry.locate_source(quote) if geometry else (None,None)
                    attrs=row.data['attributes']
                    value=attrs.get('value')
                    match=NUMERIC.fullmatch(value) if isinstance(value,str) else None
                    if match and row.fact_type in ('laboratory_test','vital_sign','instrumental_finding','biomarker'):
                        number=float(match.group(2).replace(',','.'))
                        # A value must occur in the chosen literal span, not just
                        # in another measurement or in a model-generated label.
                        literal=re.findall(r'(?<!\w)[+-]?\d+(?:[.,]\d+)?',quote)
                        unit=attrs.get('unit')
                        proposed_comparator=match.group(1) or attrs.get('comparator')
                        comparator={'≤':'<=','≥':'>='}.get(proposed_comparator,proposed_comparator)
                        normalized_quote=quote.replace('≤','<=').replace('≥','>=')
                        value_pattern=re.escape(match.group(2)).replace(r'\.',r'[.,]').replace(',',r'[.,]')
                        supported_comparator=not comparator or (comparator in ('<','<=','>','>=') and
                            re.search(re.escape(comparator)+r'\s*'+value_pattern,normalized_quote))
                        supported_unit=not unit or re.search(value_pattern+r'\s*'+re.escape(unit)+r'(?![\w/])',quote)
                        if (not any(float(v.replace(',','.'))==number for v in literal)
                                or not supported_unit or not supported_comparator):
                            row.status='needs_review'
                            row.numeric_value=None;row.unit=None;row.value_text=None
                            row.data['value_review']='Valore, comparatore o unità non riscontrati nel frammento selezionato.'
                        else:
                            row.numeric_value=number;row.unit=unit;row.value_text=value
                            if comparator:attrs['comparator']=comparator
                            row.typed_payload=dict(row.typed_payload,**attrs)
                    # Different concepts may deliberately share one negation
                    # span. Include the normalized label in event identity.
                    row.evidence_id='EVD_'+content_hash(pid,did,content_hash(text),start,end,row.normalized_entity,
                        row.fact_type,row.assertion,row.certainty,raw['subject'],row.observed_date,
                        row.observed_date_end,row.clinical_status,json.dumps(attrs,sort_keys=True))
                    rows.append(row)
            except (ValueError,TypeError,KeyError,IndexError,jsonschema.ValidationError) as exc:
                errors.append(f'Evento {index+1}: {str(exc).splitlines()[0][:240]}')
        return rows,errors

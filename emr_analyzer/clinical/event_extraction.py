"""Compact, source-addressed event annotation of one document.

SNOMED CT coding is a separate, concept-level step (``snomed_coding``):
extraction only identifies events, their literal source and their qualifiers.
"""
import json

from .evidence_utils import content_hash
from .grounded_sources import IncompleteAtomicExtraction
from .historical_reuse import HistoricalReuseMixin
from .referenced_annotations import ReferencedSourceReader
from ..prompt_catalog import load_prompt

VERSION = 'fhir_events_referenced_v8'


class EventExtractor(HistoricalReuseMixin, ReferencedSourceReader):
    def _initial_groups(self, text):
        # Initialize source spans and the clinical date index before reuse planning.
        super()._initial_groups(text)
        budget = max(200, min(600, int(getattr(self.llm, 'max_output_tokens', 4096))//8))
        return self._plan_historical_groups(text, budget)

    def __init__(self, llm_client, **kwargs):
        self.teaching_catalog = list(kwargs.pop('catalog', ()))
        super().__init__(llm_client, catalog=self.teaching_catalog or [dict(term_id='new', label='', fields=[], examples=[])],
                         system_prompt=load_prompt('compact_events_system'), **kwargs)
        self.prompt_version = VERSION
        self.prompt_digest = content_hash(self.system)

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
            raise
        if ready:
            ready(rows)
        return rows

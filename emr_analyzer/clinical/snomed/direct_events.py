"""Percorso B — direct events from RAG + LLM (experimental, never default).

Research prototype: ``DirectSnomedEventBuilder`` produces document-level
``SnomedDirectEvent`` records straight from the SNOMED candidate retrieval and
one constrained LLM call per chunk, skipping the atomic layer entirely
(buckets, typed payloads, dedup/graph/fusion).

The wire contract stays closed-set, like Percorso A:

- ``snomed_code`` is an ``enum`` over the chunk's *event-like* candidate set;
- ``observed_date`` is an ``enum`` over the dates actually grounded in the
  source (plus ``""`` for "no date"), never free-form;
- ``source_refs`` cite exact sentence indices, so every event is grounded.

No typed payload and no per-fact-type buckets.  Laboratory events are
deterministic: out-of-range rows are mapped to SNOMED through the index and
never pass through the LLM.  The record keeps exact citations and dates — the
two attributes RAG alone does not produce.

This module is not imported by the registry or any production path; it runs
only through the dedicated benchmark tool or a GUI button explicitly marked
experimental.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Iterable

from ..atomic_evidence import (
    SentenceSpan,
    TextChunk,
    split_sentence_spans,
    split_text_chunks,
)
from ..lab_evidence import is_out_of_range
from ...models.lab_result import LabValue
from ...settings import LabEvidencePolicy
from .domains import is_event_like
from .models import DEFAULT_LANG_ORDER
from .retrieval import SnomedCandidateRetriever

DIRECT_EVENTS_PROMPT_VERSION = "direct_events_v1"
DIRECT_EVENTS_PROMPT_DIGEST = "direct-events-v1-0001"

_EVENT_LIKE_RETRY_LIMIT = 1

_DIRECT_EVENTS_TASK_INTRO = (
    "Sei un ricercatore clinico. Dai testi clinici normalizzati che seguono, "
    "elenca gli EVENTI CLINICI del paziente (diagnosi, reperti, sintomi, "
    "segni, procedure) usando esclusivamente i CODICI SNOMED della lista "
    "CANDIDATI. Ogni evento è un oggetto JSON.\n"
    "REGOLE:\n"
    "1. snomed_code: scegli il codice PIÙ SPECIFICO fra i CANDIDATI; non "
    "inventare mai codici liberi.\n"
    "2. polarity: 'present' per presente, 'negated' per esplicitamente assente.\n"
    "3. certainty: 'definitive' se chiaramente dichiarato, 'probable' o "
    "'possible' per gradi di incertezza testuali.\n"
    "4. observed_date: solo una delle DATE_GROUNDED (usa '\"\"' se il testo "
    "non fornisce una data per l'evento); mai date inventate.\n"
    "5. source_refs: gli indici [N] delle frasi che supportano l'evento "
    "(almeno uno, massimo 8).\n"
    "6. Non aggiungere campi oltre quelli richiesti.\n\n"
)

_DATE_ISO_RE = re.compile(r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b")
_DATE_DMY_RE = re.compile(r"\b(\d{1,2})[/.](\d{1,2})[/.](\d{4})\b")
_MAX_GROUNDED_DATES = 24


def _valid_month_day(month: str, day: str) -> bool:
    try:
        m, d = int(month), int(day)
    except ValueError:
        return False
    return 1 <= m <= 12 and 1 <= d <= 31


def grounded_dates(
    text: object,
    document_date: str | None = None,
    *,
    cap: int = _MAX_GROUNDED_DATES,
) -> tuple[str, ...]:
    """Return the ISO dates present in ``text`` (plus the document date).

    The closed enum for ``observed_date``; the LLM can never emit a date that
    is not grounded in the source.  Sorted chronologically, capped defensively.
    """
    found: set[str] = set()
    value = str(text or "")
    for year, month, day in _DATE_ISO_RE.findall(value):
        if _valid_month_day(month, day):
            found.add(f"{int(year):04d}-{int(month):02d}-{int(day):02d}")
    for day, month, year in _DATE_DMY_RE.findall(value):
        if _valid_month_day(month, day):
            found.add(f"{int(year):04d}-{int(month):02d}-{int(day):02d}")
    if document_date:
        found.add(str(document_date).strip())
    ordered = sorted(found)
    return tuple(ordered[-max(1, int(cap)):])


def build_direct_event_schema(
    *,
    codes: Iterable[str],
    sentence_count: int,
    dates: Iterable[str],
) -> dict[str, Any]:
    """JSON schema for the event-level constrained generation."""
    codes = tuple(codes)
    dates = ("",) + tuple(dates)
    return {
        "type": "object",
        "properties": {
            "events": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "snomed_code": {
                            "type": "string",
                            "enum": list(codes),
                        },
                        "polarity": {
                            "type": "string",
                            "enum": ["present", "negated"],
                        },
                        "certainty": {
                            "type": "string",
                            "enum": ["definitive", "probable", "possible"],
                        },
                        "observed_date": {
                            "type": "string",
                            "enum": list(dates),
                        },
                        "source_refs": {
                            "type": "array",
                            "items": {
                                "type": "integer",
                                "enum": list(range(1, sentence_count + 1)),
                            },
                            "maxItems": 8,
                        },
                    },
                    "required": [
                        "snomed_code", "polarity", "certainty",
                        "observed_date", "source_refs",
                    ],
                },
            },
        },
        "required": ["events"],
    }


@dataclass(frozen=True, slots=True)
class SnomedDirectEvent:
    """One clinical event produced by Percorso B (experimental store)."""

    event_id: str
    patient_id: str
    document_id: str
    snomed_code: str
    label: str
    observed_date: str | None
    polarity: str
    certainty: str
    source_passages: tuple[str, ...]
    source_refs: tuple[int, ...]
    lab_value_ids: tuple[str, ...]
    origin: str  # "llm_text" | "deterministic_lab"
    release_digest: str = ""
    experimental: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "patient_id": self.patient_id,
            "document_id": self.document_id,
            "snomed_code": self.snomed_code,
            "label": self.label,
            "observed_date": self.observed_date,
            "polarity": self.polarity,
            "certainty": self.certainty,
            "source_passages": list(self.source_passages),
            "source_refs": list(self.source_refs),
            "lab_value_ids": list(self.lab_value_ids),
            "origin": self.origin,
            "release_digest": self.release_digest,
            "experimental": self.experimental,
        }


def _lab_row_id(lab: LabValue) -> str:
    return (
        f"lab:{lab.document_id}:{lab.parameter_name}:{lab.sample_date}:"
        f"{lab.value}:{lab.value_text}:{lab.unit}:{lab.page}"
    )


class DirectSnomedEventBuilder:
    """Direct event extraction (Percorso B); explicitly experimental."""

    def __init__(
        self,
        llm_client,
        *,
        retriever: SnomedCandidateRetriever,
        policy: LabEvidencePolicy | None = None,
        system_prompt: str | None = None,
        repair_system_prompt: str | None = None,
        text_budget: int | None = None,
        max_specialized_retries: int = _EVENT_LIKE_RETRY_LIMIT,
    ):
        self.llm = llm_client
        self.retriever = retriever
        self.policy = policy or LabEvidencePolicy()
        self.langs = tuple(getattr(retriever, "langs", None) or DEFAULT_LANG_ORDER)
        self.text_budget = max(1000, int(text_budget or 12000))
        self.max_specialized_retries = max(0, int(max_specialized_retries))
        self._system_prompt = (
            system_prompt
            or "Sei un ricercatore clinico che codifica eventi clinici con "
            "codici SNOMED scelti da un set chiuso."
        )
        self._repair_system_prompt = (
            repair_system_prompt
            or "Sei un ricercatore clinico. Correggi gli oggetti non validi "
            "rispettando lo schema chiuso; restituisci solo i sostituti."
        )
        self._metrics: dict[str, Any] = {}
        self._seq = 0

    @property
    def release_digest(self) -> str:
        return str(getattr(self.retriever, "release_digest", "") or "")

    def last_metrics(self) -> dict[str, Any]:
        return dict(self._metrics)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_events(
        self,
        *,
        patient_id: str,
        document_id: str,
        document_type: str,
        document_date: str | None,
        text: str,
        lab_values: Iterable[LabValue] = (),
    ) -> list[SnomedDirectEvent]:
        """Build direct events from normalized clinical text plus lab rows."""
        self._metrics = {
            "llm_calls": 0,
            "events": 0,
            "invalid_items": 0,
            "repaired_items": 0,
            "validation_retries": 0,
            "empty_candidate_chunks": 0,
            "candidate_chunks": 0,
            "candidate_set_sizes": [],
            "retrieved_codes": set(),
            "prompt_tokens": 0,
            "completion_tokens": 0,
        }
        self._seq = 0
        events: list[SnomedDirectEvent] = []
        events.extend(self._deterministic_lab_events(
            patient_id, document_id, lab_values,
        ))
        for chunk in split_text_chunks(text, self.text_budget):
            events.extend(self._text_chunk_events(
                chunk, patient_id, document_id, document_type,
                document_date,
            ))
        self._metrics["events"] = len(events)
        self._metrics["retrieved_codes"] = sorted(
            self._metrics["retrieved_codes"]
        )
        return events

    # ------------------------------------------------------------------
    # Deterministic laboratory events
    # ------------------------------------------------------------------

    def _deterministic_lab_events(
        self,
        patient_id: str,
        document_id: str,
        lab_values: Iterable[LabValue],
    ) -> list[SnomedDirectEvent]:
        events: list[SnomedDirectEvent] = []
        ordered = sorted(
            lab_values,
            key=lambda lab: (
                lab.sample_date or "9999-99-99", lab.parameter_name,
            ),
        )
        for lab in ordered:
            if not self._is_out_of_range(lab):
                continue
            query = " ".join(filter(None, (
                lab.normalized_name, lab.parameter_name,
            )))
            candidates = self.retriever.candidates_for_text(query)
            if candidates.empty:
                continue
            best = min(
                candidates.candidates,
                key=lambda candidate: (
                    -candidate.score, candidate.concept.concept_id,
                ),
            )
            passage = str(lab.source_text or "").strip() or (
                f"{lab.parameter_name}: {lab.value} {lab.unit}".strip()
            )
            self._seq += 1
            events.append(SnomedDirectEvent(
                event_id=f"DE-{document_id}-{self._seq:04d}",
                patient_id=patient_id,
                document_id=document_id,
                snomed_code=best.concept.concept_id,
                label=best.concept.display_label(self.langs),
                observed_date=lab.sample_date or None,
                polarity="present",
                certainty="definitive",
                source_passages=(passage,) if passage else (),
                source_refs=(),
                lab_value_ids=(_lab_row_id(lab),),
                origin="deterministic_lab",
                release_digest=self.release_digest,
            ))
        return events

    def _is_out_of_range(self, lab: LabValue) -> bool:
        return is_out_of_range(lab, self.policy)

    # ------------------------------------------------------------------
    # LLM text-chunk events
    # ------------------------------------------------------------------

    def _text_chunk_events(
        self,
        chunk: TextChunk,
        patient_id: str,
        document_id: str,
        document_type: str,
        document_date: str | None,
    ) -> list[SnomedDirectEvent]:
        sentence_spans = split_sentence_spans(chunk.text)
        if not sentence_spans:
            return []
        candidates = self.retriever.candidates_for_text(chunk.text)
        self._metrics["candidate_chunks"] += 1
        self._metrics["candidate_set_sizes"].append(len(candidates.codes))
        self._metrics["retrieved_codes"].update(candidates.codes)
        event_like = [
            candidate for candidate in candidates.candidates
            if is_event_like(candidate.concept, langs=self.langs)
        ]
        codes = tuple(candidate.concept.concept_id for candidate in event_like)
        if not codes:
            self._metrics["empty_candidate_chunks"] += 1
            return []
        concept_by_code = {
            candidate.concept.concept_id: candidate.concept
            for candidate in event_like
        }
        dates = grounded_dates(chunk.text, document_date)
        schema = build_direct_event_schema(
            codes=codes,
            sentence_count=len(sentence_spans),
            dates=dates,
        )
        prompt = self._build_prompt(
            chunk.text, sentence_spans, event_like, dates,
            document_type, document_date,
        )

        def generate(
            system_prompt: str,
            repair_note: str = "",
            *,
            max_tokens: int | None = None,
        ) -> dict:
            data = self.llm.generate_structured(
                prompt + repair_note, system_prompt, schema,
                max_tokens=max_tokens,
            )
            self._metrics["llm_calls"] += 1
            self._record_usage()
            return data or {}

        def valid_items(data: dict) -> list[dict[str, Any]]:
            return [
                item for item in data.get("events", [])
                if isinstance(item, dict)
                and not self._validate_item(
                    item, codes, len(sentence_spans), dates
                )
            ]

        initial = generate(self._system_prompt)
        accepted = list(valid_items(initial))
        pending = [
            item for item in initial.get("events", [])
            if isinstance(item, dict)
            and item not in accepted
        ]
        self._metrics["invalid_items"] += len(pending)
        retries = (
            self.max_specialized_retries if pending else 0
        )
        for _attempt in range(retries):
            self._metrics["validation_retries"] += 1
            issue_payload = [
                {
                    "item_index": index,
                    "item": item,
                    "issues": self._validate_item(
                        item, codes, len(sentence_spans), dates
                    ),
                }
                for index, item in enumerate(pending)
            ]
            repair_note = (
                "\n\nCORREZIONE_MIRATA:\n"
                + json_dumps(issue_payload)
                + "\nRestituisci soltanto i sostituti degli oggetti sopra; "
                "non ripetere gli eventi già validi."
            )
            repaired = generate(
                self._repair_system_prompt,
                repair_note=repair_note,
                max_tokens=min(4096, 1024 + len(pending) * 512),
            )
            repaired_valid = valid_items(repaired)
            accepted.extend(repaired_valid)
            self._metrics["repaired_items"] += len(repaired_valid)
            new_pending = [
                item for item in repaired.get("events", [])
                if isinstance(item, dict)
                and item not in repaired_valid
            ]
            self._metrics["invalid_items"] += len(new_pending)
            pending = new_pending
            if not pending:
                break

        events: list[SnomedDirectEvent] = []
        for item in accepted:
            self._seq += 1
            events.append(self._event_from_item(
                item, concept_by_code, sentence_spans,
                patient_id, document_id,
            ))
        return events

    def _build_prompt(
        self,
        text: str,
        sentence_spans: list[SentenceSpan],
        event_like: list,
        dates: tuple[str, ...],
        document_type: str,
        document_date: str | None,
    ) -> str:
        sentences = "\n".join(
            f"[{span.sentence_id}] {span.text}" for span in sentence_spans
        )
        catalog = "\n".join(
            f"{candidate.concept.concept_id} | "
            f"{candidate.concept.display_label(self.langs)}"
            for candidate in event_like
        )
        dates_text = " | ".join(dates) if dates else "(nessuna data rilevata)"
        return (
            _DIRECT_EVENTS_TASK_INTRO
            + f"TIPO_DOCUMENTO: {document_type}\n"
            + f"DATA_DOCUMENTO: {document_date or '(non disponibile)'}\n\n"
            + f"DATE_GROUNDED: {dates_text}\n\n"
            + f"CANDIDATI:\n{catalog}\n\n"
            + f"TESTO:\n{sentences}\n"
        )

    def _validate_item(
        self,
        item: dict[str, Any],
        codes: tuple[str, ...],
        sentence_count: int,
        dates: tuple[str, ...],
    ) -> list[str]:
        issues: list[str] = []
        code = str(item.get("snomed_code") or "").strip()
        if code not in codes:
            issues.append("snomed_code_outside_candidate_set")
        polarity = str(item.get("polarity") or "")
        if polarity not in {"present", "negated"}:
            issues.append("invalid_polarity")
        certainty = str(item.get("certainty") or "")
        if certainty not in {"definitive", "probable", "possible"}:
            issues.append("invalid_certainty")
        observed = str(item.get("observed_date") or "")
        if observed and observed not in dates:
            issues.append("date_not_grounded")
        refs = item.get("source_refs") or []
        if (
            not refs
            or any(
                not isinstance(index, int)
                or index < 1
                or index > sentence_count
                for index in refs
            )
        ):
            issues.append("invalid_source_refs")
        return issues

    def _event_from_item(
        self,
        item: dict[str, Any],
        concept_by_code: dict[str, object],
        sentence_spans: list[SentenceSpan],
        patient_id: str,
        document_id: str,
    ) -> SnomedDirectEvent:
        code = str(item.get("snomed_code") or "").strip()
        concept = concept_by_code.get(code)
        label = (
            concept.display_label(self.langs)
            if concept is not None else code
        )
        refs = tuple(
            int(index) for index in (item.get("source_refs") or [])
            if isinstance(index, int)
        )
        passages = tuple(
            sentence_spans[index - 1].text
            for index in refs
            if 1 <= index <= len(sentence_spans)
        )
        observed = str(item.get("observed_date") or "")
        return SnomedDirectEvent(
            event_id=f"DE-{document_id}-{self._seq:04d}",
            patient_id=patient_id,
            document_id=document_id,
            snomed_code=code,
            label=label,
            observed_date=observed or None,
            polarity=str(item.get("polarity") or "present"),
            certainty=str(item.get("certainty") or "definitive"),
            source_passages=passages,
            source_refs=refs,
            lab_value_ids=(),
            origin="llm_text",
            release_digest=self.release_digest,
        )

    def _record_usage(self) -> None:
        metadata = getattr(self.llm, "last_generation_metadata", None)
        if not callable(metadata):
            return
        usage = dict((metadata() or {}).get("usage") or {})
        self._metrics["prompt_tokens"] += int(
            usage.get("prompt_tokens", 0) or 0
        )
        self._metrics["completion_tokens"] += int(
            usage.get("completion_tokens", 0) or 0
        )


def json_dumps(payload: list[dict[str, Any]]) -> str:
    import json
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


__all__ = [
    "DIRECT_EVENTS_PROMPT_VERSION",
    "DIRECT_EVENTS_PROMPT_DIGEST",
    "SnomedDirectEvent",
    "DirectSnomedEventBuilder",
    "build_direct_event_schema",
    "grounded_dates",
]

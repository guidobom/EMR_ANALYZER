"""Independent per-document extraction of atomic clinical evidence."""

from __future__ import annotations

import copy
from dataclasses import dataclass
import hashlib
import inspect
import json
from pathlib import Path
import re
import threading
from typing import Any, Callable, Iterable
import unicodedata
import uuid

from .evidence_relevance import (
    annotate_evidence_disposition,
    classify_nonclinical_passage,
)
from .temporal import normalize_clinical_date
from ..extraction.llm_client import OutputLimitError
from ..models.clinical_evidence import ClinicalEvidence
from ..models.clinical_registry import (
    ASSERTION_TYPES,
    CERTAINTY_LEVELS,
    EVENT_CATEGORIES,
)
from ..models.clinical_pipeline import ATOMIC_FACT_TYPES as CONTRACT_FACT_TYPES
from ..settings import ClinicalPipelinePolicy
from ..prompt_catalog import load_prompt, prompts_digest


ATOMIC_PIPELINE_VERSION = "registry_pipeline_v11"
# v11 adds narrative laboratory fallback, explicit planned/performed grounding,
# complete structured serialization and safer within-document deduplication.
# Old extraction checkpoints do not prove that this contract was applied.
ATOMIC_RESUME_COMPATIBLE_PIPELINE_VERSIONS: tuple[str, ...] = ()

_ATOMIC_TASK = load_prompt(
    "atomic_evidence_it",
    required_markers=(
        "source_refs", "medication", "radiology_finding", "vital_sign",
        "laboratory_test",
    ),
    minimum_length=200,
)

_ATOMIC_BASE_SYSTEM_PROMPT = load_prompt("atomic_evidence_system")
_ATOMIC_REPAIR_BASE_SYSTEM_PROMPT = load_prompt(
    "atomic_evidence_repair_system"
)
_ATOMIC_COVERAGE_BASE_SYSTEM_PROMPT = load_prompt(
    "atomic_evidence_coverage_system"
)

_ATOMIC_SYSTEM_PROMPT = _ATOMIC_BASE_SYSTEM_PROMPT + "\n\n" + _ATOMIC_TASK

_ATOMIC_VALIDATION_REPAIR_SYSTEM_PROMPT = (
    _ATOMIC_REPAIR_BASE_SYSTEM_PROMPT + "\n\n" + _ATOMIC_TASK
)

_ATOMIC_COVERAGE_RECOVERY_SYSTEM_PROMPT = (
    _ATOMIC_COVERAGE_BASE_SYSTEM_PROMPT + "\n\n" + _ATOMIC_TASK
)

_THERAPY_LIFECYCLE_STATUSES = (
    "proposed", "planned", "prescribed", "started", "taken",
    "administered", "active", "ongoing", "dose_changed", "interrupted",
    "suspended", "stopped", "resumed", "completed", "cancelled", "unknown",
)

_PLANNED_ACTION_RE = re.compile(
    r"(?i)\b(?:da\s+(?:eseguire|effettuare|programmare)|"
    r"si\s+(?:programma|programmerà|richiede|propone|consiglia)|"
    r"(?:programmat|pianificat|previst|richiest|consigliat|indicat|"
    r"propost|prenotat)\w*|in\s+attesa\s+di|eventuale|"
    r"candidato\s+(?:a|ad))\b"
)
_PERFORMED_OR_RESULT_RE = re.compile(
    r"(?i)\b(?:eseguit|effettuat|praticat|sottopost|operat|ricoverat|"
    r"dimess|somministrat|ha\s+(?:mostrato|evidenziato|documentato)|"
    r"(?:mostra|evidenzia|documenta|dimostra|rileva|referta)(?:to|ta)?|"
    r"esito|risultato)\w*\b"
)

# These are projections produced after atomic extraction. Allowing the LLM to
# emit them here would duplicate syndrome, trend and oncology-line notes.
_DERIVED_EVENT_CATEGORIES = {
    "clinical_syndrome", "laboratory_trend", "oncology_treatment_line",
}
ATOMIC_EVENT_CATEGORIES = tuple(
    category for category in EVENT_CATEGORIES
    if category not in _DERIVED_EVENT_CATEGORIES
)
ATOMIC_FACT_TYPE_TO_CATEGORY = {
    "medication": "medication",
    "laboratory_test": "laboratory_finding",
    "radiology_finding": "imaging_finding",
    "instrumental_finding": "instrumental_finding",
    "diagnosis": "diagnosis",
    "symptom": "symptom",
    "clinical_decision": "care_plan",
    "procedure": "procedure",
    "clinical_sign": "clinical_sign",
    "vital_sign": "vital_sign",
    "histopathology": "histopathology",
    "biomarker": "biomarker",
    "hospitalization": "hospitalization",
    "discharge": "discharge",
}
ATOMIC_FACT_TYPES = tuple(CONTRACT_FACT_TYPES)
LLM_ATOMIC_FACT_TYPES = tuple(ATOMIC_FACT_TYPES)
ATOMIC_POLARITIES = ("present", "negated", "suspected")

_TYPED_PAYLOAD_FIELDS = {
    "laboratory_test": {
        "parameter_name", "operator", "reference_low", "reference_high",
        "reference_text", "flag", "abnormal_direction",
        "biological_material", "interpretation",
    },
    "radiology_finding": {
        "modality", "body_region", "comparison", "impression", "measurement",
        "morphology", "signal_characteristics", "enhancement", "distribution",
        "relation_to_adjacent_structures",
    },
    "instrumental_finding": {
        "modality", "body_region", "measurement", "rhythm", "function",
        "interpretation",
    },
    "diagnosis": {"diagnostic_basis", "stage", "grade", "subtype"},
    "symptom": {"onset", "course", "frequency", "context"},
    "clinical_decision": {
        "action", "target", "rationale", "urgency", "timing",
    },
    "procedure": {"procedure_type", "intent", "outcome", "complication"},
    "clinical_sign": {"course", "context", "measurement_method"},
    "vital_sign": {"context", "measurement_method"},
    "histopathology": {
        "specimen", "morphology", "grade", "margins", "invasion",
        "biomarkers", "measurement",
    },
    "biomarker": {"method", "specimen", "interpretation"},
    "hospitalization": {
        "admission_type", "reason", "department", "outcome",
        "disposition",
    },
    "discharge": {"destination", "condition", "instructions"},
}
_WIRE_COMMON_PROPERTIES = {
    "concept": {"type": "string", "minLength": 1},
    "value_text": {"type": "string"},
    "numeric_value": {"type": "number"},
    "unit": {"type": "string"},
    "observation_date": {"type": "string"},
    "observation_date_end": {"type": "string"},
    "date_precision": {
        "type": "string", "enum": [
            "day", "month", "year", "interval", "approximate", "unknown",
        ],
    },
    "polarity": {"type": "string", "enum": list(ATOMIC_POLARITIES)},
    "source_refs": {
        "type": "array", "minItems": 1, "maxItems": 6,
        "items": {"type": "integer", "minimum": 1},
    },
    "clinical_status": {"type": "string"},
    "anatomical_site": {"type": "string"},
    "laterality": {"type": "string"},
    "severity": {"type": "string"},
    "significance": {
        "type": "string", "enum": [
            "critical", "high", "clinically_relevant",
            "potentially_relevant", "uncertain",
        ],
    },
}

_MEDICATION_WIRE_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "properties": {
        "original_name": {"type": "string"},
        "active_ingredient": {"type": "string"},
        "lifecycle_status": {
            "type": "string", "enum": list(_THERAPY_LIFECYCLE_STATUSES),
        },
        "dose": {"type": "string"},
        "route": {"type": "string"},
        "frequency": {"type": "string"},
        "indication": {"type": "string"},
        "intent": {"type": "string"},
        "adherence": {"type": "string"},
    },
}

_ONCOLOGY_WIRE_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "properties": {
        "line_label": {"type": "string"},
        "regimen": {"type": "array", "items": {"type": "string"}},
        "cycle": {"type": "string"},
        "dose": {"type": "string"},
        "modification": {"type": "string"},
        "toxicity": {"type": "string"},
        "response": {"type": "string"},
        "setting": {"type": "string"},
        "intent": {"type": "string"},
        "indication": {"type": "string"},
    },
}

_PAYLOAD_FIELD_DESCRIPTIONS = {
    "modality": (
        "Solo tecnica/modalità (es. TC, RM, PET, ecografia); non inserire "
        "sede, morfologia o descrizione del reperto."
    ),
    "body_region": "Regione anatomica esaminata.",
    "measurement": "Misura completa del reperto con unità.",
    "morphology": "Forma, margini e caratteristiche morfologiche.",
    "signal_characteristics": "Segnale, densità, diffusione o captazione.",
    "enhancement": "Caratteristiche del potenziamento contrastografico.",
    "distribution": "Distribuzione spaziale del reperto.",
    "relation_to_adjacent_structures": (
        "Rapporto, contiguità, invasione o piano di clivaggio con strutture "
        "adiacenti."
    ),
}
_WIRE_MISSING_TEXT = {
    "n.d.", "n.d", "nd", "n/a", "na", "non disponibile", "non documentato",
    "non documentata", "non specificato", "non specificata",
    "non specificato nel testo", "non specificata nel testo", "unknown",
    "sconosciuto", "sconosciuta",
}


def _wire_item_schema(fact_type: str) -> dict[str, Any]:
    """Schema whose optional fields exactly match the semantic validator."""
    properties = copy.deepcopy(_WIRE_COMMON_PROPERTIES)
    payload_fields = _TYPED_PAYLOAD_FIELDS.get(fact_type, set())
    if payload_fields:
        properties["payload"] = {
            "type": "object", "additionalProperties": False,
            "properties": {
                field: (
                    {"type": "array", "items": {"type": "string"}}
                    if field == "biomarkers" else {
                        "type": "string",
                        **(
                            {"description": _PAYLOAD_FIELD_DESCRIPTIONS[field]}
                            if field in _PAYLOAD_FIELD_DESCRIPTIONS else {}
                        ),
                    }
                )
                for field in sorted(payload_fields)
            },
        }
    if fact_type == "medication":
        properties["medication"] = copy.deepcopy(_MEDICATION_WIRE_SCHEMA)
        properties["oncology"] = copy.deepcopy(_ONCOLOGY_WIRE_SCHEMA)
    return {
        "type": "object", "additionalProperties": False,
        "properties": properties,
        "required": ["concept", "polarity", "source_refs"],
    }


def build_atomic_evidence_schema(
    sentence_count: int | None = None,
    *,
    fact_types: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Build a compact category-bucket schema for one source chunk.

    The decoder can only emit citation identifiers that actually occur in the
    prompt.  This removes the most frequent post-validation failure before a
    token is sampled instead of paying for a full corrective generation.
    """
    selected = tuple(
        fact_type for fact_type in (fact_types or LLM_ATOMIC_FACT_TYPES)
        if fact_type in LLM_ATOMIC_FACT_TYPES
    )
    properties: dict[str, Any] = {}
    for fact_type in selected:
        item_schema = _wire_item_schema(fact_type)
        if sentence_count is not None and sentence_count > 0:
            refs = item_schema["properties"]["source_refs"]
            refs["items"] = {
                "type": "integer", "enum": list(range(1, sentence_count + 1)),
            }
        properties[fact_type] = {
            "type": "array", "items": item_schema,
            # A single sentence can contain a treatment list, but an unlimited
            # array invites attribute-level atom explosion.
            "maxItems": max(4, min(32, (sentence_count or 8) * 3)),
        }
    return {
        "type": "object", "additionalProperties": False,
        "properties": properties,
        # Requiring every bucket costs only a handful of empty-array tokens,
        # but prevents small models from closing the object after the first
        # familiar categories and then triggering a much costlier recall pass.
        "required": list(properties),
    }


ATOMIC_EVIDENCE_SCHEMA = build_atomic_evidence_schema()

ATOMIC_PROMPT_VERSION = "atomic_evidence_it_v11"
ATOMIC_PROMPT_DIGEST = prompts_digest(
    _ATOMIC_SYSTEM_PROMPT,
    _ATOMIC_VALIDATION_REPAIR_SYSTEM_PROMPT,
    _ATOMIC_COVERAGE_RECOVERY_SYSTEM_PROMPT,
    schema=ATOMIC_EVIDENCE_SCHEMA,
)

# This is deliberately an output-safety limit, not a context-window limit.
# Clinical documents are often dense enough to produce more JSON than source
# text.  Sending an entire document merely because it fits in context is slow:
# llama.cpp must finish a doomed generation before the caller can bisect and
# repeat it.  Small, deterministic chunks keep each answer below the output
# ceiling while document-level workers still keep every inference slot busy.
_ATOMIC_SOURCE_CHUNK_CHARS = 2800


@dataclass(frozen=True, slots=True)
class TextChunk:
    index: int
    text: str
    page_start: int | None = None
    page_end: int | None = None


@dataclass(frozen=True, slots=True)
class SentenceSpan:
    """A citable, exact slice of one prompt chunk."""

    sentence_id: int
    start: int
    end: int
    text: str


@dataclass(frozen=True, slots=True)
class WireValidationIssue:
    """One object that could not be made canonical without guessing."""

    item_index: int
    fact_type: str | None
    reasons: tuple[str, ...]
    raw: object

    def for_prompt(self) -> dict[str, Any]:
        return {
            "item_index": self.item_index,
            "fact_type": self.fact_type,
            "errors": list(self.reasons),
            "invalid_item": self.raw,
        }


@dataclass(slots=True)
class WireValidationResult:
    items: list[dict[str, Any]]
    issues: list[WireValidationIssue]
    normalized_items: int = 0


class AtomicExtractionCancelled(RuntimeError):
    """Raised at a safe document/chunk boundary after a stop request."""


class AtomicEvidenceExtractor:
    """Extract every observation before any deduplication or synthesis."""

    def __init__(
        self,
        llm_client,
        *,
        policy: ClinicalPipelinePolicy | None = None,
        task_prompt: str | None = None,
        system_prompt: str | None = None,
        repair_system_prompt: str | None = None,
        coverage_system_prompt: str | None = None,
    ):
        self.llm = llm_client
        self.policy = policy or ClinicalPipelinePolicy()
        effective_task = task_prompt or _ATOMIC_TASK
        self._system_prompt = (
            (system_prompt or _ATOMIC_BASE_SYSTEM_PROMPT)
            + "\n\n" + effective_task
        )
        self._repair_system_prompt = (
            (
                repair_system_prompt
                or _ATOMIC_REPAIR_BASE_SYSTEM_PROMPT
            )
            + "\n\n" + effective_task
        )
        self._coverage_system_prompt = (
            (
                coverage_system_prompt
                or _ATOMIC_COVERAGE_BASE_SYSTEM_PROMPT
            )
            + "\n\n" + effective_task
        )
        # One extractor instance is shared by all registry workers.  Keep
        # request counters thread-local so timings/tokens from simultaneous
        # documents can never contaminate one another.
        self._metrics_local = threading.local()
        self._schema_cache: dict[tuple[int, tuple[str, ...]], dict[str, Any]] = {}
        self._schema_cache_lock = threading.Lock()
        try:
            parameters = inspect.signature(
                self.llm.generate_structured
            ).parameters
            self._supports_generation_limit = (
                "max_tokens" in parameters
                or any(
                    parameter.kind == inspect.Parameter.VAR_KEYWORD
                    for parameter in parameters.values()
                )
            )
        except (AttributeError, TypeError, ValueError):
            # A few native/test callables do not expose a signature.  Preserve
            # the most compatible legacy call shape for those clients.
            self._supports_generation_limit = False

    @property
    def model_name(self) -> str:
        return str(getattr(self.llm, "model", "") or "")

    @property
    def model_digest(self) -> str:
        try:
            info = self.llm.backend.model_info(self.model_name) or {}
        except Exception:
            info = {}
        payload = {
            "name": self.model_name,
            "file": info.get("file"),
            "size_bytes": info.get("size_bytes"),
            "architecture": info.get("architecture"),
            "temperature": getattr(self.llm, "temperature", None),
            "top_p": getattr(self.llm, "top_p", None),
            "top_k": getattr(self.llm, "top_k", None),
            "seed": getattr(self.llm, "seed", None),
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True).encode("utf-8")
        ).hexdigest()

    def extract_document(
        self,
        *,
        patient_id: str,
        document_id: str,
        document_type: str,
        document_date: str | None,
        text: str,
        geometry_path: Path | None = None,
        cancel_check: Callable[[], bool] | None = None,
        chunk_progress_callback: Callable[[int, int], None] | None = None,
    ) -> list[ClinicalEvidence]:
        self._metrics_local.value = {
            "llm_calls": 0,
            "output_limit_retries": 0,
            "validation_retries": 0,
            "coverage_retries": 0,
            "uncovered_signal_groups": 0,
            "invalid_items": 0,
            "normalized_items": 0,
            "unresolved_invalid_items": 0,
            "source_chunks": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "prompt_ms": 0.0,
            "predicted_ms": 0.0,
            "prefiltered_nonclinical": 0,
            "initial_items": 0,
            "repaired_items": 0,
            "coverage_items": 0,
            "items_before_deduplication": 0,
            "within_document_duplicates": 0,
            "final_items": 0,
        }
        geometry = self._load_geometry(geometry_path)
        text_for_llm, deterministic_nonclinical = _isolate_nonclinical_lines(
            text,
            patient_id=patient_id,
            document_id=document_id,
            document_date=document_date,
            geometry=geometry,
        )
        evidence: list[ClinicalEvidence] = list(deterministic_nonclinical)
        position = 0
        chunks = split_text_chunks(text_for_llm, self._text_budget())
        self._current_metrics()["source_chunks"] = len(chunks)
        self._current_metrics()["prefiltered_nonclinical"] = len(
            deterministic_nonclinical
        )
        for chunk_number, chunk in enumerate(chunks, start=1):
            if cancel_check is not None and cancel_check():
                raise AtomicExtractionCancelled(
                    "Estrazione interrotta su richiesta dell'utente"
                )
            extracted_chunks = self._extract_chunk_adaptive(
                chunk, document_type=document_type,
                document_date=document_date,
                cancel_check=cancel_check,
            )
            if cancel_check is not None and cancel_check():
                raise AtomicExtractionCancelled(
                    "Estrazione interrotta su richiesta dell'utente"
                )
            for resolved_chunk, payload, sentence_spans, retry_depth in (
                extracted_chunks
            ):
                for item in payload:
                    parsed = self._to_evidence(
                        _expand_atomic_item(item),
                        patient_id=patient_id,
                        document_id=document_id,
                        document_date=document_date,
                        chunk=resolved_chunk,
                        position=position,
                        full_text=text,
                        geometry=geometry,
                        sentence_spans=sentence_spans,
                        retry_depth=retry_depth,
                    )
                    position += 1
                    if parsed is not None:
                        evidence.append(parsed)
            if chunk_progress_callback is not None:
                chunk_progress_callback(chunk_number, len(chunks))
        evidence.extend(_explicit_performed_procedure_evidence(
            patient_id=patient_id,
            document_id=document_id,
            document_date=document_date,
            full_text=text,
            geometry=geometry,
            model_name=self.model_name,
            existing=evidence,
        ))
        evidence.extend(_explicit_negated_imaging_evidence(
            patient_id=patient_id,
            document_id=document_id,
            document_date=document_date,
            full_text=text,
            geometry=geometry,
            model_name=self.model_name,
            existing=evidence,
        ))
        evidence.extend(_explicit_resolution_evidence(
            patient_id=patient_id,
            document_id=document_id,
            document_date=document_date,
            full_text=text,
            geometry=geometry,
            model_name=self.model_name,
            existing=evidence,
        ))
        metrics = self._current_metrics()
        metrics["items_before_deduplication"] = len(evidence)
        evidence = deduplicate_atomic_evidence(evidence)
        metrics["within_document_duplicates"] = (
            metrics["items_before_deduplication"] - len(evidence)
        )
        metrics["final_items"] = len(evidence)
        # Keep every extracted atom immutable and auditable.  Administrative
        # and methodological atoms are labelled here, then excluded only from
        # downstream registry projections.
        for item in evidence:
            annotate_evidence_disposition(item)
        return evidence

    def _extract_chunk_adaptive(
        self,
        chunk: TextChunk,
        *,
        document_type: str,
        document_date: str | None,
        retry_depth: int = 0,
        cancel_check: Callable[[], bool] | None = None,
    ) -> list[tuple[TextChunk, list[dict[str, Any]], list[SentenceSpan], int]]:
        """Retry a dense chunk by deterministic bisection on output limit."""
        if cancel_check is not None and cancel_check():
            raise AtomicExtractionCancelled(
                "Estrazione interrotta su richiesta dell'utente"
            )
        try:
            payload, spans = self._extract_chunk(
                chunk, document_type=document_type,
                document_date=document_date,
            )
            return [(chunk, payload, spans, retry_depth)]
        except OutputLimitError:
            metrics = self._current_metrics()
            metrics["output_limit_retries"] += 1
            children = bisect_text_chunk(chunk)
            if retry_depth >= 5 or len(children) < 2:
                raise
            extracted = []
            for child in children:
                extracted.extend(self._extract_chunk_adaptive(
                    child,
                    document_type=document_type,
                    document_date=document_date,
                    retry_depth=retry_depth + 1,
                    cancel_check=cancel_check,
                ))
            return extracted

    def _extract_chunk(
        self,
        chunk: TextChunk,
        *,
        document_type: str,
        document_date: str | None,
    ) -> tuple[list[dict[str, Any]], list[SentenceSpan]]:
        sentence_spans = split_sentence_spans(chunk.text)
        prompt = build_atomic_prompt(
            chunk,
            document_type=document_type,
            document_date=document_date,
            sentence_spans=sentence_spans,
        )
        generator = self.llm.generate_structured
        base_schema = self._schema_for(len(sentence_spans))

        def generate(
            system_prompt: str,
            repair_note: str = "",
            *,
            schema: dict[str, Any] = base_schema,
            output_budget: int | None = None,
            source_prompt: str = prompt,
        ):
            effective_prompt = source_prompt + repair_note
            try:
                if self._supports_generation_limit:
                    return generator(
                        effective_prompt, system_prompt, schema,
                        max_tokens=(
                            output_budget or self._output_budget(sentence_spans)
                        ),
                    )
                return generator(effective_prompt, system_prompt, schema)
            finally:
                self._record_last_generation()

        data = generate(self._system_prompt)
        validation = _validate_wire_response(data, sentence_spans)
        items = list(validation.items)
        pending = list(validation.issues)
        metrics = self._current_metrics()
        metrics["invalid_items"] += len(pending)
        metrics["normalized_items"] += validation.normalized_items
        metrics["initial_items"] += len(validation.items)
        retries = (
            self.policy.max_specialized_retries
            if self.policy.adaptive_specialized_retry and pending else 0
        )
        for _attempt in range(retries):
            metrics["validation_retries"] += 1
            relevant_types = tuple(dict.fromkeys(
                issue.fact_type for issue in pending
                if issue.fact_type in LLM_ATOMIC_FACT_TYPES
            ))
            repair_schema = self._schema_for(
                len(sentence_spans),
                fact_types=relevant_types or LLM_ATOMIC_FACT_TYPES,
            )
            issue_payload = [issue.for_prompt() for issue in pending[:24]]
            repaired = generate(
                self._repair_system_prompt,
                repair_note=(
                    "\n\nCORREZIONE_MIRATA:\n"
                    + json.dumps(
                        issue_payload, ensure_ascii=False, separators=(",", ":")
                    )
                    + "\nRestituisci soltanto i sostituti degli oggetti sopra; "
                    "non ripetere le evidenze già valide."
                ),
                schema=repair_schema,
                output_budget=min(
                    self._output_budget(sentence_spans),
                    max(768, 384 + len(pending) * 384),
                ),
            )
            repaired_validation = _validate_wire_response(
                repaired, sentence_spans
            )
            items.extend(repaired_validation.items)
            metrics["repaired_items"] += len(repaired_validation.items)
            metrics["invalid_items"] += len(repaired_validation.issues)
            metrics["normalized_items"] += (
                repaired_validation.normalized_items
            )
            pending = list(repaired_validation.issues)
            if not pending:
                break
        metrics["unresolved_invalid_items"] += len(pending)
        if self.policy.adaptive_specialized_retry:
            recovery_spans, recovery_types = _coverage_recovery_plan(
                sentence_spans, items
            )
            if recovery_spans and recovery_types:
                metrics["coverage_retries"] += 1
                metrics["uncovered_signal_groups"] += len(recovery_types)
                compact_spans = [
                    SentenceSpan(
                        sentence_id=index,
                        start=span.start,
                        end=span.end,
                        text=span.text,
                    )
                    for index, span in enumerate(recovery_spans, start=1)
                ]
                original_refs = {
                    compact.sentence_id: original.sentence_id
                    for compact, original in zip(compact_spans, recovery_spans)
                }
                recovery_chunk = TextChunk(
                    index=chunk.index,
                    text="\n".join(span.text for span in compact_spans),
                    page_start=chunk.page_start,
                    page_end=chunk.page_end,
                )
                recovery_prompt = build_atomic_prompt(
                    recovery_chunk,
                    document_type=document_type,
                    document_date=document_date,
                    sentence_spans=compact_spans,
                )
                recovery_schema = self._schema_for(
                    len(compact_spans), fact_types=recovery_types
                )
                recovered = generate(
                    self._coverage_system_prompt,
                    repair_note=(
                        "\n\nCATEGORIE_DA_RECUPERARE: "
                        + ", ".join(recovery_types)
                        + ". Restituisci tutti i fatti espliciti di questi "
                        "tipi presenti nel TESTO."
                    ),
                    schema=recovery_schema,
                    output_budget=min(
                        self._output_budget(compact_spans),
                        max(1024, 512 + len(compact_spans) * 320),
                    ),
                    source_prompt=recovery_prompt,
                )
                recovered_validation = _validate_wire_response(
                    recovered, compact_spans
                )
                metrics["invalid_items"] += len(
                    recovered_validation.issues
                )
                metrics["normalized_items"] += (
                    recovered_validation.normalized_items
                )
                metrics["unresolved_invalid_items"] += len(
                    recovered_validation.issues
                )
                for recovered_item in recovered_validation.items:
                    recovered_item["source_refs"] = [
                        original_refs[ref]
                        for ref in _wire_refs(recovered_item)
                        if ref in original_refs
                    ]
                    if recovered_item["source_refs"]:
                        items.append(recovered_item)
                        metrics["coverage_items"] += 1
        coalesced = _split_multi_state_medication_items(
            _coalesce_adjacent_wire_items(items), sentence_spans
        )
        return (
            _attach_referential_wire_continuations(coalesced, sentence_spans),
            sentence_spans,
        )

    def last_extraction_metrics(self) -> dict[str, int | float]:
        """Metrics for the document most recently handled by this thread."""
        return dict(self._current_metrics())

    def _schema_for(
        self,
        sentence_count: int,
        *,
        fact_types: Iterable[str] | None = None,
    ) -> dict[str, Any]:
        """Reuse immutable response contracts across parallel chunks."""
        selected = tuple(
            fact_type for fact_type in (
                fact_types or LLM_ATOMIC_FACT_TYPES
            )
            if fact_type in LLM_ATOMIC_FACT_TYPES
        )
        key = (max(0, int(sentence_count)), selected)
        with self._schema_cache_lock:
            cached = self._schema_cache.get(key)
            if cached is None:
                cached = build_atomic_evidence_schema(
                    key[0], fact_types=selected
                )
                self._schema_cache[key] = cached
            return cached

    def _current_metrics(self) -> dict[str, int | float]:
        metrics = getattr(self._metrics_local, "value", None)
        if metrics is None:
            metrics = {
                "llm_calls": 0,
                "output_limit_retries": 0,
                "validation_retries": 0,
                "coverage_retries": 0,
                "uncovered_signal_groups": 0,
                "invalid_items": 0,
                "normalized_items": 0,
                "unresolved_invalid_items": 0,
                "source_chunks": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "prompt_ms": 0.0,
                "predicted_ms": 0.0,
                "prefiltered_nonclinical": 0,
                "initial_items": 0,
                "repaired_items": 0,
                "coverage_items": 0,
                "items_before_deduplication": 0,
                "within_document_duplicates": 0,
                "final_items": 0,
            }
            self._metrics_local.value = metrics
        return metrics

    def _record_last_generation(self) -> None:
        metrics = self._current_metrics()
        metrics["llm_calls"] += 1
        getter = getattr(self.llm, "last_generation_metadata", None)
        if not callable(getter):
            return
        try:
            metadata = getter() or {}
        except Exception:
            return
        numeric_keys = (
            "prompt_tokens", "completion_tokens", "total_tokens",
            "prompt_ms", "predicted_ms",
        )
        for key in numeric_keys:
            value = metadata.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                metrics[key] += value

    def _to_evidence(
        self,
        item: dict[str, Any],
        *,
        patient_id: str,
        document_id: str,
        document_date: str | None,
        chunk: TextChunk,
        position: int,
        full_text: str,
        geometry,
        sentence_spans: list[SentenceSpan] | None = None,
        retry_depth: int = 0,
    ) -> ClinicalEvidence | None:
        entity = _clean_text(item.get("normalized_entity"), 300)
        refs = _safe_sentence_refs(item.get("source_refs"), sentence_spans)
        source_quote = _quote_from_sentence_refs(refs, sentence_spans)
        if not source_quote:
            # Backward compatibility for stored/test v2-v3 payloads. Current
            # compact prompts never ask the model to generate a quotation.
            source_quote = _clean_text(item.get("source_text"), 2000)
        if not entity or not source_quote:
            return None
        category = str(item.get("category") or "other").strip().lower()
        if category == "laboratory_trend":
            category = "laboratory_finding"
        elif category == "clinical_syndrome":
            category = "diagnosis"
        elif category == "oncology_treatment_line":
            category = "medication"
        if category not in EVENT_CATEGORIES:
            category = "other"
        assertion = str(item.get("assertion") or "present").strip().lower()
        if assertion not in ASSERTION_TYPES:
            assertion = "unknown"
        certainty = str(item.get("certainty") or "unknown").strip().lower()
        if certainty not in CERTAINTY_LEVELS:
            certainty = "unknown"
        quote_verified, matched_quote = locate_quote(source_quote, full_text)
        quote_folded = matched_quote.casefold()
        fact_type = str(
            item.get("fact_type") or _fact_type_for_category(category)
        )
        if (
            fact_type in {"instrumental_finding", "clinical_sign", "vital_sign"}
            and _LAB_ANALYTE_RE.fullmatch(entity.strip())
            and re.search(
                r"(?i)\b(?:esami|laboratorio|ematochimic\w*|"
                r"range|intervallo\s+di\s+riferimento)\b",
                matched_quote,
            )
        ):
            # A small model can put an analyte in a neighbouring measurement
            # bucket.  Reclassify it before validation; the deterministic lab
            # path remains authoritative and removes an exact duplicate later.
            fact_type = "laboratory_test"
            category = "laboratory_finding"
            item["fact_type"] = fact_type
            item["category"] = category
        if fact_type == "laboratory_test":
            category = "laboratory_finding"
            if not _llm_laboratory_claim_is_relevant(
                entity=entity, quote=matched_quote, item=item,
            ):
                return None
        fact_type, category = _project_planned_action(
            fact_type=fact_type,
            category=category,
            entity=entity,
            quote=matched_quote,
            item=item,
        )
        category = _specific_atomic_category(
            category=category,
            fact_type=fact_type,
            concept=entity,
            quote=matched_quote,
        )
        negative_instrumental_result = (
            re.search(
                r"(?i)\bnegativ\w*\s+per\s+(?P<target>[^,;:.]{2,100})",
                matched_quote,
            ) if fact_type == "instrumental_finding" else None
        )
        if negative_instrumental_result and re.search(
            r"(?i)\b(?:broncoscopia|BAL|colonscopia|gastroscopia|EGDS)\b",
            entity,
        ):
            entity = negative_instrumental_result.group("target").strip()
            assertion = "absent"
            certainty = "excluded"
            item["polarity"] = "negated"
        biomarker_result_match = (
            re.search(
                rf"(?i)\b{re.escape(entity)}\b\s*(?:[:=]\s*)?"
                r"(?P<result>non\s+mutat\w*|negativ\w*|wild[- ]?type)\b",
                matched_quote,
            ) if fact_type == "biomarker" else None
        )
        negative_biomarker_result = bool(biomarker_result_match)
        if negative_biomarker_result:
            directly_negated = False
            assertion = "present"
            certainty = "confirmed"
            item["polarity"] = "present"
            if not _clean_optional(item.get("value_text"), 300):
                item["value_text"] = biomarker_result_match.group("result")
        else:
            entity, directly_negated = _ground_direct_negation(
                entity, matched_quote
            )
        if directly_negated:
            assertion = "absent"
            certainty = "excluded"
            item["polarity"] = "negated"
        if category == "symptom" and re.search(
            r"\b(?:riferisc|riferit|lament|segnal)\w*", quote_folded
        ):
            certainty = "patient_reported"
        if re.search(r"\bnon\s+(?:è\s+)?documentat\w*", quote_folded):
            certainty = "unknown"
            if category == "laboratory_finding" and re.search(
                r"\b(?:causa\s+infettiva|infezion\w*)\b", quote_folded
            ):
                category = "diagnosis"
        elif (
            assertion not in {"absent", "conditional", "hypothetical"}
            and re.search(
                r"\b(?:quadro\s+)?compatibile\s+con\b|"
                r"\b(?:sospett[oa]|possibile|probabile|verosimile)\b",
                quote_folded,
            )
        ):
            certainty = "suspected"

        if category in {"diagnosis", "adverse_event"} and (
            re.search(r"\bimmuno[- ]?mediat\w*\b", quote_folded)
            and re.search(
                r"\b(?:durante|in\s+corso\s+di)\s+(?:la\s+)?terapia\b|"
                r"\bimmunoterap\w*\b|\bcorrelat\w*\s+(?:al|alla)\s+tratt",
                quote_folded,
            )
        ):
            category = "toxicity"

        temporal_value = item.get("observed_date")
        if _temporal_context_mismatch(category, matched_quote):
            temporal_value = None
        grounded_temporal = _temporal_expression_from_quote(
            matched_quote, category=category
        )
        context_temporal, context_ref, context_text = (
            _contextual_date_for_refs(refs, sentence_spans)
        )
        # Prefer a date/duration that can be verified in the quoted passage,
        # then the nearest preceding date heading.  The model-provided value
        # remains a last resort for expressions such as "ieri" that the
        # deterministic parser intentionally does not enumerate.
        if grounded_temporal:
            temporal_value = grounded_temporal
            context_ref = None
            context_text = None
        elif context_temporal:
            temporal_value = context_temporal
        temporal = normalize_clinical_date(
            temporal_value,
            document_date=_duration_reference_date(
                temporal_value,
                matched_quote=matched_quote,
                contextual_date=context_temporal,
                document_date=document_date,
            ),
            explicit_precision=item.get("date_precision"),
            date_end=item.get("observed_date_end"),
        )
        if temporal.start is None and grounded_temporal:
            if grounded_temporal != temporal_value:
                temporal = normalize_clinical_date(
                    grounded_temporal,
                    document_date=document_date,
                    explicit_precision=None,
                    date_end=None,
                )
        page_hint = _safe_int(item.get("source_page")) or chunk.page_start
        page, bbox = page_hint, None
        if geometry is not None:
            page, bbox = geometry.locate_source(matched_quote, page_hint)
        data = _sanitize_nested_payload(
            item.get("additional_data"),
            allowed={
                "reference_range", "grade", "stage",
            },
        )
        if item.get("wire_normalized"):
            # Audit-only marker: the first response was repaired locally for a
            # harmless representation mismatch, without another LLM call.
            data["wire_normalized"] = True
        therapy = _sanitize_nested_payload(
            item.get("therapy"),
            allowed={
                "original_name", "active_ingredient", "lifecycle_status",
                "dose", "route", "frequency", "indication", "intent",
                "adherence",
            },
        )
        oncology = _sanitize_nested_payload(
            item.get("oncology"),
            allowed={
                "line_label", "regimen", "cycle", "dose", "modification",
                "toxicity", "response", "setting", "intent",
                "indication",
            },
            list_fields={"regimen"},
        )
        typed = _sanitize_nested_payload(
            item.get("typed_payload"),
            allowed=_TYPED_PAYLOAD_FIELDS.get(fact_type, set()),
            list_fields={"biomarkers"},
        )
        if fact_type == "histopathology" and typed.get("biomarkers"):
            measurements = [
                value for value in typed["biomarkers"]
                if re.search(r"(?i)\bBreslow\b", value)
            ]
            if measurements and not typed.get("measurement"):
                typed["measurement"] = measurements[0]
                typed["biomarkers"] = [
                    value for value in typed["biomarkers"]
                    if value not in measurements
                ]
                if not typed["biomarkers"]:
                    typed.pop("biomarkers", None)
        if fact_type == "biomarker":
            source_identity = _identity_text(matched_quote)
            for field in ("method", "specimen"):
                value = typed.get(field)
                if value and _identity_text(value) not in source_identity:
                    typed.pop(field, None)
        if category == "hospitalization" and not typed.get("reason"):
            reason = re.search(
                r"(?i)\b(?:ricoverat\w*|ospedalizzat\w*)\b[^.;]*?"
                r"\bper\s+(?P<reason>[^.;]{2,160})",
                matched_quote,
            )
            if reason:
                typed["reason"] = reason.group("reason").strip(" .")
        if category == "care_plan":
            target = re.search(
                r"(?i)\b(?:con|e)\s+(?P<target>nuov[oa]\s+"
                r"(?:TC|TAC|RM|PET|ecografia|visita|valutazione))\b",
                matched_quote,
            )
            timing = re.search(
                r"(?i)\b(?:dopo|tra|entro)\s+(?:circa\s+)?"
                r"(?:\d+|un|uno|una|due|tre|quattro|cinque|sei|sette|"
                r"otto|nove|dieci|undici|dodici)\s+"
                r"(?:giorn\w*|settiman\w*|mes\w*)\b",
                matched_quote,
            )
            if target and not typed.get("target"):
                typed["target"] = target.group("target")
            if timing and not typed.get("timing"):
                typed["timing"] = timing.group(0)
        if category not in {
            "medication", "toxicity", "response", "progression"
        }:
            oncology = {}
        if category not in {"medication", "toxicity", "adverse_event"}:
            therapy = {}
        severity = _clean_optional(item.get("severity"), 100)
        if severity is None and data.get("grade"):
            severity = f"grado {data['grade']}"
        if severity is None and category in {"diagnosis", "toxicity"}:
            grade = re.search(
                r"(?i)\bgrado\s+([0-5]|I{1,3}|IV|V)\b", matched_quote
            )
            if grade:
                severity = f"grado {grade.group(1)}"
        therapy, oncology = _enrich_medication_payload(
            category, entity, matched_quote, therapy, oncology
        )
        meaningful_oncology_structure = any(
            oncology.get(field) for field in (
                "line_label", "cycle", "setting", "intent", "modification",
            )
        )
        if (
            category == "medication" and oncology
            and not meaningful_oncology_structure
            and not re.search(
                r"(?i)\b(?:linea|schema|regime|ciclo|adiuvant|neoadiuvant|"
                r"palliativ|curativ|mantenimento)\w*\b",
                matched_quote,
            )
        ):
            oncology = {}
        if therapy:
            if therapy.get("lifecycle_status") not in (
                None, *_THERAPY_LIFECYCLE_STATUSES
            ):
                therapy.pop("lifecycle_status", None)
        clinical_status = _clean_optional(item.get("clinical_status"), 100)
        if category == "care_plan" and re.search(
            r"(?i)\b(?:programmat\w*|pianificat\w*|previst\w*)\b",
            matched_quote,
        ):
            clinical_status = "planned"
        if category == "imaging_finding" and re.search(
            r"(?i)\b(?:riduzion\w*|regression\w*|miglior\w*)\b",
            matched_quote,
        ):
            assertion = "present"
            item["polarity"] = "present"
            clinical_status = "improved"
            evolution = typed.get("signal_characteristics")
            if (
                evolution
                and re.search(
                    r"(?i)\b(?:riduzion\w*|regression\w*|miglior\w*)\b",
                    evolution,
                )
                and not typed.get("comparison")
            ):
                typed["comparison"] = evolution
                typed.pop("signal_characteristics", None)
        elif category == "imaging_finding" and re.search(
            r"(?i)\b(?:scompars\w*|risolt\w*)\b", matched_quote
        ):
            assertion = "present"
            item["polarity"] = "present"
            clinical_status = "resolved"
        if category == "hospitalization" and temporal.end:
            clinical_status = "completed"
        if category == "discharge":
            if clinical_status and clinical_status not in {
                "completed", "unknown"
            } and not typed.get("condition"):
                typed["condition"] = clinical_status
            clinical_status = "completed"
        assertion, clinical_status, therapy = _medication_transition(
            category, matched_quote, assertion, clinical_status, therapy
        )
        if therapy:
            data["therapy"] = therapy
        if oncology:
            data["oncology"] = oncology
        data.update({
            "fact_type": fact_type,
            "polarity": str(item.get("polarity") or _legacy_polarity(
                assertion, certainty
            )),
            "report_date": document_date,
            "quote_verified": quote_verified,
            "date_original_text": temporal.original_text,
            "date_approximate": temporal.approximate,
            "chunk_index": chunk.index,
            "sentence_refs": refs,
            "adaptive_retry_depth": retry_depth,
            "source_reference": {
                "document_id": document_id,
                "page": page,
                "bbox": list(bbox) if bbox else None,
                "passage": matched_quote,
                "sentence_refs": refs,
            },
        })
        if context_ref is not None and context_text:
            data["date_context_sentence_ref"] = context_ref
            data["date_context_text"] = context_text
        if item.get("confidence") is not None:
            # Compatibility with evidence created by older prompts. The
            # current prompt no longer asks the model for an uncalibrated
            # self-assessment.
            data["llm_confidence_uncalibrated"] = _bounded_float(
                item.get("confidence"), 0.5
            )
        numeric_value = _safe_float(item.get("numeric_value"))
        unit = _clean_optional(item.get("unit"), 80)
        entity, category, numeric_value, unit = _normalize_measurement(
            entity, category, matched_quote, numeric_value, unit
        )
        if (
            category == "toxicity" and numeric_value is not None
            and str(unit or "").casefold() in {"grado", "grade"}
        ):
            severity = severity or f"grado {numeric_value:g}"
            numeric_value, unit = None, None
        entity, severity = _normalize_entity_severity(entity, severity)
        # A recorded number cannot itself be absent.  This deterministic
        # correction also collapses a common duplicate where the model
        # incorrectly transfers a nearby symptom resolution to the vital.
        if numeric_value is not None and assertion == "absent":
            assertion = "present"
        certainty = _ground_certainty(
            category=category,
            assertion=assertion,
            certainty=certainty,
            quote=matched_quote,
            numeric_value=numeric_value,
            quote_verified=quote_verified,
        )
        evidence_id = stable_evidence_id(
            document_id=document_id,
            category=category,
            entity=entity,
            quote=matched_quote,
            observed_date=temporal.start,
            page=page,
            assertion=assertion,
            certainty=certainty,
        )
        return ClinicalEvidence(
            evidence_id=evidence_id,
            patient_id=patient_id,
            document_id=document_id,
            category=category,
            normalized_entity=entity,
            fact_type=str(data.get("fact_type") or category),
            concept_original=entity,
            canonical_label=entity,
            mapping_status="unmapped",
            typed_payload={
                key: value for key, value in (
                    ("medication", therapy),
                    ("oncology", oncology),
                    (fact_type, typed),
                    ("extra", {
                        key: value for key, value in data.items()
                        if key in {"reference_range", "grade", "stage"}
                    }),
                ) if value
            },
            source_text=matched_quote,
            assertion=assertion,
            certainty=certainty,
            temporality=(
                "historical" if temporal.source in {
                    "explicit_or_retroactive", "retrospective_duration",
                }
                and temporal.start and document_date
                and temporal.start < document_date else "current"
            ),
            clinical_status=clinical_status,
            observed_date=temporal.start,
            observed_date_end=temporal.end,
            document_date=document_date,
            date_precision=temporal.precision,
            date_source=temporal.source,
            anatomical_site=_clean_optional(item.get("anatomical_site"), 200),
            laterality=_clean_optional(item.get("laterality"), 50),
            severity=severity,
            significance=_normalize_significance(item.get("significance")),
            value_text=_clean_optional(item.get("value_text"), 300),
            numeric_value=numeric_value,
            unit=unit,
            source_page=page,
            bbox=bbox,
            confidence=(0.75 if quote_verified else 0.35),
            extraction_method="llm_atomic_v2",
            model_name=self.model_name,
            prompt_version=ATOMIC_PROMPT_VERSION,
            schema_version="3.0",
            status="proposed" if quote_verified else "needs_review",
            data=data,
        )

    def _output_budget(self, spans: list[SentenceSpan]) -> int:
        """Use a small per-operation cap; adaptive splitting handles outliers."""
        configured = max(
            256, int(getattr(self.llm, "max_output_tokens", 4096) or 4096)
        )
        source_chars = sum(len(span.text) for span in spans)
        # The category-bucket contract no longer repeats ``fact_type`` and
        # null optionals for every item. A tighter cap prevents pathological
        # verbosity; genuinely dense passages are still bisected safely.
        estimated = max(1280, 384 + int(source_chars * 1.45))
        return min(configured, estimated)

    def _text_budget(self) -> int:
        context = int(getattr(self.llm, "context_length", 32768) or 32768)
        output = int(getattr(self.llm, "max_output_tokens", 4096) or 4096)
        available = max(2000, context - min(output, context // 2) - 3500)
        context_safe_chars = max(1000, int(available * 2.5))
        return min(_ATOMIC_SOURCE_CHUNK_CHARS, context_safe_chars)

    @staticmethod
    def _load_geometry(path: Path | None):
        if path is None or not path.exists():
            return None
        try:
            from ..pipeline.pdf_extractor import PdfExtractionResult
            return PdfExtractionResult.from_dict(
                json.loads(path.read_text(encoding="utf-8"))
            )
        except (OSError, ValueError, TypeError):
            return None


def build_atomic_prompt(
    chunk: TextChunk,
    *,
    document_type: str,
    document_date: str | None,
    sentence_spans: list[SentenceSpan] | None = None,
) -> str:
    """Return a cache-friendly prompt with deterministic source references."""
    pages = (
        f"{chunk.page_start or 'n.d.'}-{chunk.page_end or 'n.d.'}"
    )
    spans = sentence_spans or split_sentence_spans(chunk.text)
    numbered_text = "\n".join(
        f"[S{span.sentence_id}] {span.text}" for span in spans
    )
    inferred_type = infer_document_content_type(chunk.text)
    return (
        f"CONTESTO: tipo_dichiarato={document_type or 'non classificato'}; "
        f"contenuto_probabile={inferred_type}; "
        f"data_documento={document_date or 'non disponibile'}; "
        f"pagine={pages}\n"
        "In refs usa solo i numeri degli ID S seguenti.\n\n"
        f"TESTO:\n{numbered_text}"
    )


def infer_document_content_type(text: str) -> str:
    """Infer a broad content family when imported document labels are wrong."""
    folded = unicodedata.normalize("NFKD", str(text or "")).casefold()
    signals = (
        ("histopathology", (
            r"\b(?:istologic|istopatologic|immunoistochimic|biopsi|"
            r"materiale inviato|margini? di resezione)\w*\b",
        )),
        ("radiology", (
            r"\b(?:tc|tac|rmn?|pet(?:/tc)?|ecografi|radiografi|rx)\b",
            r"\b(?:18f[- ]?fdg|mezzo di contrasto|reperto radiologic)\w*\b",
            r"\b(?:iperintensit|ipointensit|restrizione (?:del segnale )?"
            r"in diffusione|potenziamento contrastografic|"
            r"volume (?:di studio|in esame))\w*\b",
        )),
        ("laboratory", (
            r"\b(?:emocromo|emoglobina|creatinina|transaminasi|tsh|"
            r"piastrine|leucociti|esami ematochimici)\b",
        )),
        ("procedure", (
            r"\b(?:intervento chirurgico|resezione|asportazione|"
            r"endoscopia|broncoscopia|colonscopia)\b",
        )),
        ("clinical_note", (
            r"\b(?:anamnesi|esame obiettivo|visita|terapia|"
            r"diagnosi|piano terapeutico)\b",
        )),
    )
    scores = {
        label: sum(bool(re.search(pattern, folded)) for pattern in patterns)
        for label, patterns in signals
    }
    best = max(scores, key=scores.get)
    return best if scores[best] else "non_classificato"


def _ground_direct_negation(concept: str, quote: str) -> tuple[str, bool]:
    """Normalize a directly stated absence without semantic inference."""
    entity = " ".join(str(concept or "").split()).strip()
    concept_negative = re.match(
        r"(?i)^(?:assenza di evidenza|non evidenza|assenza)\s+"
        r"(?:di\s+)?(.+)$",
        entity,
    )
    quote_negative = re.match(
        r"(?i)^\s*(?:attualmente\s+)?(?:non\s+(?!si\s+esclud)|"
        r"assenza\s+di|senza\s+evidenza\s+di)\b",
        str(quote or ""),
    )
    if not concept_negative and not quote_negative:
        return entity, False
    if concept_negative:
        entity = concept_negative.group(1).strip(" .,:;-")
        entity = re.sub(r"(?i)^ulterior[ie]\s+", "", entity)
    return entity or str(concept or ""), True


def split_sentence_spans(text: str, maximum_chars: int = 1200) -> list[SentenceSpan]:
    """Split a chunk into exact, addressable spans without rewriting text."""
    value = str(text or "")
    if not value.strip():
        return []
    boundaries = [0]
    # A single newline is usually a PDF line wrap, not a semantic boundary.
    for match in re.finditer(r"(?:\n\s*\n+|(?<=[.!?;])\s+)", value):
        boundaries.append(match.end())
    boundaries.append(len(value))

    raw_ranges: list[tuple[int, int]] = []
    for start, end in zip(boundaries, boundaries[1:]):
        while start < end and value[start].isspace():
            start += 1
        while end > start and value[end - 1].isspace():
            end -= 1
        if start >= end:
            continue
        cursor = start
        while end - cursor > maximum_chars:
            target = cursor + maximum_chars
            split_at = max(
                value.rfind(", ", cursor, target),
                value.rfind(" ", cursor, target),
            )
            if split_at <= cursor + maximum_chars // 2:
                split_at = target
            else:
                split_at += 1
            raw_ranges.append((cursor, split_at))
            cursor = split_at
            while cursor < end and value[cursor].isspace():
                cursor += 1
        if cursor < end:
            raw_ranges.append((cursor, end))
    return [
        SentenceSpan(index, start, end, value[start:end])
        for index, (start, end) in enumerate(raw_ranges, start=1)
    ]


def _isolate_nonclinical_lines(
    text: str,
    *,
    patient_id: str,
    document_id: str,
    document_date: str | None,
    geometry,
) -> tuple[str, list[ClinicalEvidence]]:
    """Archive obvious boilerplate deterministically and omit it from prompts."""
    page_pattern = re.compile(
        r"(?i)^\s*(?:---\s*PAGINA\s+(\d+)\s*---|"
        r"\[PAGINA\s*:?\s*(\d+)\]|"
        r"<!--\s*page\s*:\s*(\d+)\s*-->)\s*$"
    )
    current_page = None
    retained = []
    excluded = []
    for line_number, raw in enumerate(str(text or "").splitlines(True), start=1):
        passage = raw.strip()
        marker = page_pattern.match(passage)
        if marker:
            current_page = int(next(value for value in marker.groups() if value))
            retained.append(raw)
            continue
        disposition = classify_nonclinical_passage(passage)
        if disposition is None:
            retained.append(raw)
            continue
        page, bbox = current_page, None
        if geometry is not None and passage:
            page, bbox = geometry.locate_source(passage, current_page)
        evidence_id = "EVD_" + uuid.uuid5(
            uuid.NAMESPACE_URL,
            f"{document_id}:{page}:{line_number}:{passage.casefold()}",
        ).hex
        excluded.append(ClinicalEvidence(
            evidence_id=evidence_id,
            patient_id=patient_id,
            document_id=document_id,
            category="other",
            normalized_entity=passage[:300] or disposition.reason,
            source_text=passage,
            fact_type=None,
            concept_original=None,
            canonical_label=None,
            clinical_relevance=(
                "excluded_administrative"
                if disposition.reason == "scheduling"
                else "excluded_methodological"
                if disposition.reason in {
                    "diagnostic_method", "diagnostic_tracer_administration",
                }
                else "excluded_boilerplate"
            ),
            document_date=document_date,
            date_precision="unknown",
            source_page=page,
            bbox=bbox,
            confidence=1.0,
            extraction_method="deterministic_nonclinical",
            prompt_version=ATOMIC_PROMPT_VERSION,
            schema_version="3.0",
            status="auto",
            data={
                "registry_role": disposition.role,
                "registry_role_reason": disposition.reason,
                "classification_method": "deterministic_prefilter",
                "source_line": line_number,
            },
        ))
        # Preserve line count/page structure while preventing this passage
        # from consuming tokens or being hallucinated back as a clinical fact.
        retained.append("\n" if raw.endswith("\n") else "")
    return "".join(retained), excluded


def bisect_text_chunk(chunk: TextChunk) -> list[TextChunk]:
    """Split near a sentence boundary while retaining coarse page provenance."""
    spans = split_sentence_spans(chunk.text)
    if len(spans) < 2:
        return []
    total_chars = sum(len(span.text) for span in spans)
    target = total_chars / 2
    consumed = 0
    cut_index = 1
    for index, span in enumerate(spans[:-1], start=1):
        consumed += len(span.text)
        cut_index = index
        if consumed >= target:
            break
    left_end = spans[cut_index - 1].end
    right_start = spans[cut_index].start
    left = chunk.text[:left_end].strip()
    right = chunk.text[right_start:].strip()
    if not left or not right:
        return []
    return [
        TextChunk(
            index=chunk.index * 10 + 1, text=left,
            page_start=chunk.page_start, page_end=chunk.page_end,
        ),
        TextChunk(
            index=chunk.index * 10 + 2, text=right,
            page_start=chunk.page_start, page_end=chunk.page_end,
        ),
    ]


def split_text_chunks(text: str, max_chars: int) -> list[TextChunk]:
    """Output-safe paragraph chunks; no prefix-only truncation.

    A short date heading is carried into the following chunk instead of being
    stranded at the end of the preceding one.  This keeps temporal scope intact
    without overlapping (and therefore duplicating) clinical evidence.
    """
    value = str(text or "").strip()
    if not value:
        return []
    max_chars = max(1000, int(max_chars))
    page_pattern = re.compile(
        r"(?im)^\s*(?:"
        r"---\s*PAGINA\s+(\d+)\s*---|"
        r"\[PAGINA\s*:?\s*(\d+)\]|"
        r"<!--\s*page\s*:\s*(\d+)\s*-->"
        r")\s*$"
    )
    pieces: list[tuple[str, int | None]] = []
    current_page: int | None = None
    cursor = 0
    for match in page_pattern.finditer(value):
        before = value[cursor:match.start()].strip()
        if before:
            pieces.append((before, current_page))
        current_page = int(next(group for group in match.groups() if group))
        cursor = match.end()
    tail = value[cursor:].strip()
    if tail:
        pieces.append((tail, current_page))
    if not pieces:
        pieces = [(value, None)]

    paragraphs: list[tuple[str, int | None]] = []
    for piece, page in pieces:
        blocks = [block.strip() for block in re.split(r"\n\s*\n", piece)]
        for block in blocks:
            if not block:
                continue
            if len(block) <= max_chars:
                paragraphs.append((block, page))
                continue
            start = 0
            while start < len(block):
                end = min(len(block), start + max_chars)
                if end < len(block):
                    boundary = max(
                        block.rfind("\n", start, end),
                        block.rfind(". ", start, end),
                        block.rfind("; ", start, end),
                    )
                    if boundary > start + max_chars // 2:
                        end = boundary + 1
                paragraphs.append((block[start:end].strip(), page))
                start = end

    chunks: list[TextChunk] = []
    buffer: list[str] = []
    buffer_pages: list[int | None] = []
    size = 0
    for paragraph, page in paragraphs:
        added = len(paragraph) + (2 if buffer else 0)
        if buffer and size + added > max_chars:
            carry: tuple[str, int | None] | None = None
            if len(buffer) > 1 and _looks_like_temporal_heading(buffer[-1]):
                carry = (buffer.pop(), buffer_pages.pop())
            chunks.append(TextChunk(
                index=len(chunks), text="\n\n".join(buffer),
                page_start=min(
                    value for value in buffer_pages if value is not None
                ) if any(value is not None for value in buffer_pages) else None,
                page_end=max(
                    value for value in buffer_pages if value is not None
                ) if any(value is not None for value in buffer_pages) else None,
            ))
            if carry:
                buffer, buffer_pages = [carry[0]], [carry[1]]
                size = len(carry[0])
            else:
                buffer, buffer_pages, size = [], [], 0
        buffer.append(paragraph)
        size += len(paragraph) + (2 if len(buffer) > 1 else 0)
        buffer_pages.append(page)
    if buffer:
        chunks.append(TextChunk(
            index=len(chunks), text="\n\n".join(buffer),
            page_start=min(
                value for value in buffer_pages if value is not None
            ) if any(value is not None for value in buffer_pages) else None,
            page_end=max(
                value for value in buffer_pages if value is not None
            ) if any(value is not None for value in buffer_pages) else None,
        ))
    return chunks


def _looks_like_temporal_heading(value: str) -> bool:
    """Recognise compact dated section headers, not ordinary dated prose."""
    text = " ".join(str(value or "").split()).strip(" :-–—")
    return bool(
        len(text) <= 120
        and len(text.split()) <= 8
        and _explicit_date_expression(text)
        and re.match(
            r"(?i)^(?:(?:in\s+)?data\s+)?(?:"
            r"\d{1,2}[./-]\d{1,2}[./-](?:\d{2}|\d{4})|"
            r"(?:19|20)\d{2}-\d{1,2}-\d{1,2}|"
            r"\d{1,2}\s+(?:gennaio|febbraio|marzo|aprile|maggio|giugno|"
            r"luglio|agosto|settembre|ottobre|novembre|dicembre)|"
            r"(?:gennaio|febbraio|marzo|aprile|maggio|giugno|luglio|agosto|"
            r"settembre|ottobre|novembre|dicembre)\s+(?:19|20)\d{2})\b",
            text,
        )
    )


def locate_quote(quote: str, full_text: str) -> tuple[bool, str]:
    """Verify a citation, tolerating whitespace differences only."""
    source = str(full_text or "")
    wanted = " ".join(str(quote or "").split())
    if not wanted:
        return False, ""
    if wanted in source:
        return True, wanted
    flexible = re.compile(
        r"\s+".join(re.escape(token) for token in wanted.split()),
        re.IGNORECASE,
    )
    match = flexible.search(source)
    if match:
        return True, source[match.start():match.end()]
    return False, wanted


def _expand_atomic_item(item: dict[str, Any]) -> dict[str, Any]:
    """Map the v8 contract and legacy wire payloads to internal fields."""
    if "fact_type" in item:
        polarity = str(item.get("polarity") or "").strip().casefold()
        assertion, certainty = {
            "present": ("present", "confirmed"),
            "negated": ("absent", "excluded"),
            "suspected": ("present", "suspected"),
        }.get(polarity, ("unknown", "unknown"))
        return {
            "fact_type": item.get("fact_type"),
            "polarity": polarity,
            "category": ATOMIC_FACT_TYPE_TO_CATEGORY.get(
                str(item.get("fact_type") or ""), "other"
            ),
            "normalized_entity": item.get("concept"),
            "source_refs": item.get("source_refs"),
            "assertion": assertion,
            "certainty": certainty,
            "clinical_status": item.get("clinical_status"),
            "observed_date": item.get("observation_date"),
            "observed_date_end": item.get("observation_date_end"),
            "date_precision": item.get("date_precision"),
            "anatomical_site": item.get("anatomical_site"),
            "laterality": item.get("laterality"),
            "severity": item.get("severity"),
            "significance": item.get("significance"),
            "value_text": item.get("value_text"),
            "numeric_value": item.get("numeric_value"),
            "unit": item.get("unit"),
            "therapy": item.get("medication") or {},
            "oncology": item.get("oncology") or {},
            "additional_data": item.get("extra") or {},
            "typed_payload": item.get("payload") or {},
            "wire_normalized": bool(item.get("_wire_normalized")),
        }
    is_v5 = any(key in item for key in ("c", "e", "r"))
    is_v4 = "entity" in item or "refs" in item
    if not is_v5 and not is_v4:
        return dict(item)
    drug_key = "m" if is_v5 else "drug"
    oncology_key = "o" if is_v5 else "oncology"
    extra_key = "z" if is_v5 else "extra"
    drug = item.get(drug_key) if isinstance(item.get(drug_key), dict) else {}
    oncology = (
        item.get(oncology_key)
        if isinstance(item.get(oncology_key), dict) else {}
    )
    extra = (
        item.get(extra_key) if isinstance(item.get(extra_key), dict) else {}
    )
    def field(short: str, long: str):
        return item.get(short) if is_v5 else item.get(long)

    def nested(payload: dict[str, Any], short: str, long: str):
        return payload.get(short) if is_v5 else payload.get(long)

    therapy = {
        "original_name": nested(drug, "n", "name"),
        "active_ingredient": nested(drug, "i", "ingredient"),
        "lifecycle_status": nested(drug, "l", "lifecycle"),
        "dose": nested(drug, "d", "dose"),
        "route": nested(drug, "r", "route"),
        "frequency": nested(drug, "f", "frequency"),
        "indication": nested(drug, "x", "indication"),
        "intent": nested(drug, "t", "intent"),
        "adherence": nested(drug, "a", "adherence"),
    }

    expanded_oncology = {
        "line_label": nested(oncology, "l", "line"),
        "regimen": nested(oncology, "r", "regimen"),
        "cycle": nested(oncology, "c", "cycle"),
        "dose": nested(oncology, "d", "dose"),
        "modification": nested(oncology, "m", "change"),
        "toxicity": nested(oncology, "t", "toxicity"),
        "response": nested(oncology, "p", "response"),
        "setting": nested(oncology, "s", "setting"),
        "intent": nested(oncology, "i", "intent"),
        "indication": nested(oncology, "x", "indication"),
    }
    additional = {
        "reference_range": nested(extra, "r", "range"),
        "grade": nested(extra, "g", "grade"),
        "stage": nested(extra, "s", "stage"),
    }
    return {
        "category": field("c", "category"),
        "normalized_entity": field("e", "entity"),
        "source_refs": field("r", "refs"),
        "assertion": field("a", "assertion"),
        "certainty": field("q", "certainty"),
        "clinical_status": field("s", "status"),
        "observed_date": field("d", "date"),
        "observed_date_end": field("de", "date_end"),
        "date_precision": field("p", "precision"),
        "anatomical_site": field("i", "site"),
        "laterality": field("l", "side"),
        "severity": field("v", "severity"),
        "significance": field("g", "significance"),
        "value_text": field("x", "value"),
        "numeric_value": field("n", "number"),
        "unit": field("u", "unit"),
        "therapy": therapy,
        "oncology": expanded_oncology,
        "additional_data": additional,
    }


def _normalize_atomic_wire_item(value: object) -> dict[str, Any] | None:
    """Normalize legacy tuple/object payloads kept for stored/test clients."""
    if isinstance(value, dict):
        return dict(value)
    if not isinstance(value, list) or len(value) != 7:
        return None
    category, entity, refs, assertion, certainty, significance, details = value
    if not isinstance(details, dict):
        return None
    item = {
        "c": category, "e": entity, "r": refs, "a": assertion,
        "q": certainty, "g": significance,
    }
    item.update(details)
    return item


def _validated_wire_items(
    data: object, sentence_count: int
) -> tuple[list[dict[str, Any]], int]:
    """Compatibility wrapper for older callers and unit tests."""
    spans = [SentenceSpan(index, 0, 0, "") for index in range(
        1, max(0, int(sentence_count)) + 1
    )]
    result = _validate_wire_response(data, spans)
    return result.items, len(result.issues)


def _validate_wire_response(
    data: object,
    sentence_spans: list[SentenceSpan],
) -> WireValidationResult:
    """Create canonical wire items and reject only clinically unsafe gaps.

    Harmless representation differences (``S2`` instead of ``2``, null
    optionals, numeric strings) are normalized locally.  Missing concepts,
    ambiguous citations and unknown information-bearing fields remain hard
    failures and are eligible for one small targeted repair.
    """
    if not isinstance(data, dict):
        return WireValidationResult([], [WireValidationIssue(
            -1, None, ("La risposta non è un oggetto JSON.",), data,
        )])

    valid: list[dict[str, Any]] = []
    issues: list[WireValidationIssue] = []
    normalized_items = 0

    # Backward compatibility for v4-v7 and synthetic test clients.  Current
    # requests receive category buckets and therefore do not pay for a repeated
    # fact_type field in every object.
    legacy = data.get("evidence")
    if isinstance(legacy, list):
        for index, raw in enumerate(legacy):
            item = _normalize_atomic_wire_item(raw)
            if item is None:
                issues.append(WireValidationIssue(
                    index, None, ("Formato legacy non riconosciuto.",), raw,
                ))
                continue
            if "fact_type" not in item:
                valid.append(item)
                continue
            normalized, reasons, changed = _canonical_wire_item(
                item, str(item.get("fact_type") or ""), sentence_spans
            )
            if normalized is None:
                issues.append(WireValidationIssue(
                    index, str(item.get("fact_type") or "") or None,
                    tuple(reasons), raw,
                ))
            else:
                valid.append(normalized)
                normalized_items += int(changed)
        return WireValidationResult(valid, issues, normalized_items)

    item_index = 0
    known_buckets = set(LLM_ATOMIC_FACT_TYPES)
    unknown_buckets = set(data) - known_buckets
    if unknown_buckets:
        issues.append(WireValidationIssue(
            -1, None,
            ("Chiavi di primo livello non ammesse: "
             + ", ".join(sorted(unknown_buckets)),),
            {key: data.get(key) for key in sorted(unknown_buckets)},
        ))
    for fact_type in LLM_ATOMIC_FACT_TYPES:
        bucket = data.get(fact_type)
        if bucket is None:
            continue
        if not isinstance(bucket, list):
            issues.append(WireValidationIssue(
                item_index, fact_type,
                (f"{fact_type} deve essere un array.",), bucket,
            ))
            item_index += 1
            continue
        for raw in bucket:
            normalized, reasons, changed = _canonical_wire_item(
                raw, fact_type, sentence_spans
            )
            if normalized is None:
                issues.append(WireValidationIssue(
                    item_index, fact_type, tuple(reasons), raw,
                ))
            else:
                valid.append(normalized)
                normalized_items += int(changed)
            item_index += 1
    return WireValidationResult(valid, issues, normalized_items)


def _canonical_wire_item(
    raw: object,
    fact_type: str,
    sentence_spans: list[SentenceSpan],
) -> tuple[dict[str, Any] | None, list[str], bool]:
    """Normalize one v8/v7 object without discarding clinical information."""
    if fact_type not in LLM_ATOMIC_FACT_TYPES:
        return None, [f"fact_type non ammesso: {fact_type!r}."], False
    if not isinstance(raw, dict):
        return None, ["L'item deve essere un oggetto JSON."], False

    value = dict(raw)
    value.pop("fact_type", None)
    allowed = set(_wire_item_schema(fact_type)["properties"])
    # ``extra`` existed in v7. Accept it as a canonical compatibility field;
    # current v8 schemas no longer offer it to the model.
    compatibility = {"extra"}
    unknown = set(value) - allowed - compatibility
    if unknown:
        return None, [
            "Campi non ammessi per " + fact_type + ": "
            + ", ".join(sorted(unknown)) + "."
        ], False

    changed = False
    for key in list(value):
        if value[key] in (None, "", [], {}):
            value.pop(key)
            changed = True

    concept = value.get("concept")
    if not isinstance(concept, str) and concept is not None:
        concept = str(concept)
        changed = True
    concept = " ".join(str(concept or "").split()).strip()
    if not concept:
        return None, ["concept mancante o vuoto."], changed
    value["concept"] = concept

    polarity_aliases = {
        "present": "present", "presente": "present", "positive": "present",
        "negated": "negated", "negato": "negated", "absent": "negated",
        "suspected": "suspected", "sospetto": "suspected",
        "possible": "suspected", "probable": "suspected",
    }
    original_polarity = str(value.get("polarity") or "").strip().casefold()
    polarity = polarity_aliases.get(original_polarity)
    if polarity is None:
        return None, ["polarity mancante o non ammessa."], changed
    changed = changed or polarity != original_polarity
    value["polarity"] = polarity

    raw_refs = value.get("source_refs")
    refs = _safe_sentence_refs(raw_refs, sentence_spans)
    supplied_refs = len(raw_refs) if isinstance(raw_refs, list) else 0
    if not refs:
        inferred = _infer_wire_refs(concept, sentence_spans)
        if not inferred:
            return None, [
                "source_refs assenti, fuori intervallo o ambigui."
            ], changed
        refs = inferred
        changed = True
    elif len(refs) != supplied_refs or refs != raw_refs:
        changed = True
    value["source_refs"] = refs

    fact_type, concept, semantic_changed, semantic_error = (
        _normalize_wire_semantics(
            fact_type, concept, refs, sentence_spans
        )
    )
    if semantic_error:
        return None, [semantic_error], changed or semantic_changed
    if semantic_changed:
        value["concept"] = concept
        changed = True

    string_fields = {
        "value_text", "unit", "observation_date", "observation_date_end",
        "clinical_status", "anatomical_site", "laterality", "severity",
    }
    for key in string_fields:
        if key not in value:
            continue
        if isinstance(value[key], (dict, list, bool)):
            return None, [f"{key} deve essere una stringa o null."], changed
        cleaned = " ".join(str(value[key]).split()).strip()
        if cleaned:
            changed = changed or cleaned != value[key]
            value[key] = cleaned
        else:
            value.pop(key, None)
            changed = True

    if "numeric_value" in value:
        numeric = _safe_float(str(value["numeric_value"]).replace(",", "."))
        if numeric is None:
            return None, ["numeric_value non è numerico."], changed
        changed = changed or numeric != value["numeric_value"]
        value["numeric_value"] = numeric

    precision = value.get("date_precision")
    if precision not in (None, "day", "month", "year", "interval",
                          "approximate", "unknown"):
        value.pop("date_precision", None)
        changed = True
    significance = value.get("significance")
    if significance not in (
        None, "critical", "high", "clinically_relevant",
        "potentially_relevant", "uncertain",
    ):
        value.pop("significance", None)
        changed = True

    nested_specs = {
        "payload": (
            _TYPED_PAYLOAD_FIELDS.get(fact_type, set()), {"biomarkers"}
        ),
        "medication": (
            set(_MEDICATION_WIRE_SCHEMA["properties"]), set()
        ),
        "oncology": (set(_ONCOLOGY_WIRE_SCHEMA["properties"]), {"regimen"}),
        "extra": ({"reference_range", "grade", "stage"}, set()),
    }
    for key, (fields, list_fields) in nested_specs.items():
        if key not in value:
            continue
        if key in {"medication", "oncology"} and fact_type != "medication":
            return None, [f"{key} non è ammesso per {fact_type}."], changed
        normalized, nested_changed, nested_error = _canonical_wire_mapping(
            value[key], fields=fields, list_fields=list_fields
        )
        if nested_error:
            return None, [f"{key}: {nested_error}"], changed
        changed = changed or nested_changed
        if normalized:
            value[key] = normalized
        else:
            value.pop(key, None)

    value["fact_type"] = fact_type
    if changed:
        value["_wire_normalized"] = True
    return value, [], changed


_GENERIC_ATOMIC_CONCEPTS = {
    "anamnesi", "conclusioni", "controllo", "esame", "esami",
    "follow up", "follow-up", "procedura", "referto", "terapia",
    "trattamento", "visita",
}
_ANATOMICAL_FRAGMENT_CONCEPTS = {
    "a destra", "a sinistra", "bilaterale", "destra", "destro",
    "dx", "inguinale", "laterale", "sinistra", "sinistro", "sn",
}
_IMAGING_CONTEXT_RE = re.compile(
    r"(?i)(?:^|\W)(?:TC|TAC|RMN?|PET(?:[- /]?FDG|/TC)?|RX|"
    r"ecografi\w*)(?:\W|$)"
)
_PATHOLOGY_CONTEXT_RE = re.compile(
    r"(?i)\b(?:istolog\w*|istopatolog\w*|citolog\w*|biops\w*|"
    r"immunoistochim\w*|pezzo operatorio|campione tissutale|"
    r"margini? di resezione)\b"
)
_VITAL_CONCEPT_RE = re.compile(
    r"(?i)^(?:pa|pressione(?: arteriosa)?|fc|frequenza cardiaca|"
    r"fr|frequenza respiratoria|spo2|saturazione(?: periferica)?|"
    r"temperatura|febbre|peso|altezza|bmi|indice di massa corporea|"
    r"ecog|performance status)$"
)
_LAB_ANALYTE_RE = re.compile(
    r"(?i)^(?:TSH|FT3|FT4|emoglobina|Hb|leucociti|neutrofili|"
    r"linfociti|piastrine|creatinina|azotemia|urea|AST|ALT|GOT|GPT|"
    r"bilirubina|sodio|potassio|calcio|glicemia|PCR|VES|LDH|"
    r"troponina|CK|CPK|amilasi|lipasi)$"
)
_DRUG_TOKEN_RE = re.compile(
    r"(?i)\b([a-zà-öø-ÿ][a-zà-öø-ÿ0-9-]{2,}(?:mab|nib|ciclib|"
    r"taxel|platin|otecan|trexed|cortene|prednisone|prednisolone|"
    r"desametasone|metilprednisolone))\b"
)


def _semantic_source_for_refs(
    refs: list[int], sentence_spans: list[SentenceSpan]
) -> str:
    """Return cited text plus a short preceding classification context."""
    if not refs or not sentence_spans:
        return ""
    first = max(1, refs[0] - 2)
    last = min(len(sentence_spans), refs[-1])
    return " ".join(
        sentence_spans[index - 1].text for index in range(first, last + 1)
    )


def _medication_name_from_text(text: str) -> str | None:
    match = _DRUG_TOKEN_RE.search(str(text or ""))
    if not match:
        return None
    token = match.group(1).strip(" .,:;()[]")
    return token or None


def _normalize_wire_semantics(
    fact_type: str,
    concept: str,
    refs: list[int],
    sentence_spans: list[SentenceSpan],
) -> tuple[str, str, bool, str | None]:
    """Repair only high-certainty type errors; reject unusable fragments.

    This guard is intentionally deterministic. It does not create a clinical
    fact: it verifies that the model-selected bucket is compatible with the
    cited source and prevents labels such as ``terapia`` or ``3X`` from being
    persisted as autonomous evidence.
    """
    source = _semantic_source_for_refs(refs, sentence_spans)
    identity = _identity_text(concept)
    changed = False
    medication_name = _medication_name_from_text(concept)
    source_medication = _medication_name_from_text(source)

    if (
        _VITAL_CONCEPT_RE.fullmatch(concept.strip())
        and fact_type != "vital_sign"
    ):
        fact_type = "vital_sign"
        changed = True

    if (
        identity in {"ricovero", "ospedalizzazione", "degenza"}
        and _HOSPITALIZATION_SIGNAL_RE.search(source)
        and fact_type != "hospitalization"
    ):
        fact_type = "hospitalization"
        changed = True

    if medication_name and fact_type != "medication":
        fact_type = "medication"
        changed = True
    elif identity in _GENERIC_ATOMIC_CONCEPTS and source_medication:
        fact_type = "medication"
        concept = source_medication
        identity = _identity_text(concept)
        changed = True

    if (
        fact_type in {"histopathology", "clinical_sign", "vital_sign"}
        and _IMAGING_CONTEXT_RE.search(source)
        and not _PATHOLOGY_CONTEXT_RE.search(source)
        and not medication_name
        and not source_medication
    ):
        fact_type = "radiology_finding"
        changed = True

    if concept.casefold().strip(" .") in _WIRE_MISSING_TEXT:
        return fact_type, concept, changed, "concept segnaposto non ammesso."
    if identity in _GENERIC_ATOMIC_CONCEPTS:
        return fact_type, concept, changed, (
            "concept generico: usare il fatto clinico o il farmaco specifico."
        )
    if identity in _ANATOMICAL_FRAGMENT_CONCEPTS:
        return fact_type, concept, changed, (
            "una sede/lateralità isolata non è un'evidenza clinica."
        )
    if re.fullmatch(r"[\d.,x× ]+", concept.strip(), re.IGNORECASE):
        return fact_type, concept, changed, (
            "una misura isolata senza reperto non è un concept clinico."
        )
    if fact_type == "vital_sign" and not _VITAL_CONCEPT_RE.fullmatch(
        concept.strip()
    ):
        return fact_type, concept, changed, (
            "vital_sign ammette soltanto un parametro fisiologico nominato."
        )
    return fact_type, concept, changed, None


_IMAGING_FINDING_SIGNAL_RE = re.compile(
    r"(?i)\b(?:evidenz\w*|mostra\w*|presenza|comparsa|increment\w*|"
    r"riduz\w*|captazione|lesion\w*|nodul\w*|metastas\w*|edema|"
    r"ispessimento|enhancement|ipodensit\w*|iperintensit\w*|"
    r"pseudonodularit\w*|linfoaden\w*|repert\w*)\b"
)
_DIAGNOSIS_SIGNAL_RE = re.compile(
    r"(?i)\b(?:affett[oa]\s+da|diagnos\w*|melanoma|metastas\w*|"
    r"progressione|risposta (?:metabolica )?(?:completa|parziale)|"
    r"tossicit\w*|miocardite|polineuropatia|uveite|allerg\w*|"
    r"familiarit\w*|compatibile\s+con|sospett\w*|probabil\w*)\b"
)
_SYMPTOM_SIGNAL_RE = re.compile(
    r"(?i)\b(?:dolor\w*|tosse|dispnea|affanno|astenia|fatigue|nausea|"
    r"vomito|diarrea|stipsi|prurito|rash|eritema|cefalea|vertigin\w*|"
    r"disfagia|disfonia|parestesi\w*|ipoestesia|debolezza|"
    r"palpitazioni|sincope|brividi|calo ponderale)\b"
)
_PROCEDURE_SIGNAL_RE = re.compile(
    r"(?i)\b(?:sottopost[oa]\s+(?:ad?|a)|exeresi|resezione|"
    r"asportazione|intervento chirurgico|radioterapia|RT\s+(?:su|alla)|"
    r"biopsia|broncoscopia|colonscopia)\b"
)
_PERFORMED_PROCEDURE_RE = re.compile(
    r"(?i)\b(?:eseguit|effettuat|praticat|sottopost)\w*\b[^.;]{0,90}?"
    r"(?P<procedure>broncoscopia(?:\s+con\s+BAL)?|colonscopia|"
    r"gastroscopia|EGDS|toracentesi|paracentesi|biopsia\w*|"
    r"exeresi|resezione|asportazione)\b"
)
_VITAL_SIGNAL_RE = re.compile(
    r"(?i)\b(?:SpO2|saturazione|FC|frequenza cardiaca|PA|"
    r"pressione arteriosa|temperatura|peso|BMI|ECOG|performance status)\b"
)
_DECISION_SIGNAL_RE = re.compile(
    r"(?i)\b(?:si (?:è )?concorda(?:to)?|si prescrive|in programma|"
    r"si propone|indicat[oa]|soprassedere|monitoraggio|programmat\w*|"
    r"rivalutazione|follow[- ]?up)\b"
)
_HOSPITALIZATION_SIGNAL_RE = re.compile(
    r"(?i)\b(?:ricover\w*|ospedalizz\w*|degenza|accesso\s+(?:in|al)\s+"
    r"(?:pronto soccorso|PS))\b"
)
_DISCHARGE_SIGNAL_RE = re.compile(
    r"(?i)\b(?:dimess\w*|dimission\w*)\b"
)
_BIOMARKER_SIGNAL_RE = re.compile(
    r"(?i)\b(?:BRAF|NRAS|KRAS|EGFR|ALK|ROS1|HER2|ERBB2|PD[- ]?L1|"
    r"MSI|dMMR|BRCA[12]?|KIT|RET|MET|NTRK|IDH[12]?|MGMT|"
    r"mutat\w*|wild[- ]?type|amplificat\w*|espressione)\b"
)


def _coverage_signal_types(text: str) -> set[str]:
    """High-precision source signals used only to detect omissions."""
    source = str(text or "")
    result: set[str] = set()
    if _medication_name_from_text(source):
        result.add("medication")
    if (
        _IMAGING_CONTEXT_RE.search(source)
        and _IMAGING_FINDING_SIGNAL_RE.search(source)
    ):
        result.add("radiology_finding")
    if _DIAGNOSIS_SIGNAL_RE.search(source):
        result.add("diagnosis")
    if (
        _SYMPTOM_SIGNAL_RE.search(source)
        and not re.search(
            r"(?i)\b(?:risolt|scompars|regredit)[oaie]*\b", source
        )
    ):
        result.add("symptom")
    if _PROCEDURE_SIGNAL_RE.search(source):
        result.add("procedure")
    if _VITAL_SIGNAL_RE.search(source):
        result.add("vital_sign")
    if _DECISION_SIGNAL_RE.search(source):
        result.add("clinical_decision")
    if _HOSPITALIZATION_SIGNAL_RE.search(source):
        result.add("hospitalization")
    if _DISCHARGE_SIGNAL_RE.search(source):
        result.add("discharge")
    if _BIOMARKER_SIGNAL_RE.search(source):
        result.add("biomarker")
    return result


def _coverage_recovery_plan(
    sentence_spans: list[SentenceSpan],
    items: list[dict[str, Any]],
) -> tuple[list[SentenceSpan], tuple[str, ...]]:
    """Select source spans whose explicit clinical signals remain uncovered.

    This is a recall guard, not a rule-based extractor. The rules merely decide
    whether one compact, category-constrained retry is needed; the LLM must
    still return a source-grounded object and the normal validator is applied.
    """
    covered: dict[str, set[int]] = {}
    for item in items:
        # Legacy/test clients may still return the pre-v8 broad ``category``
        # instead of a wire ``fact_type``. Count those objects as covered too,
        # otherwise the recall guard pays for a needless second generation.
        fact_type = str(
            item.get("fact_type")
            or _fact_type_for_category(str(item.get("category") or ""))
        )
        covered.setdefault(fact_type, set()).update(_wire_refs(item))

    covered_at_ref: dict[int, set[str]] = {}
    for fact_type, refs in covered.items():
        for ref in refs:
            covered_at_ref.setdefault(ref, set()).add(fact_type)

    missing_by_type: dict[str, set[int]] = {}
    for span in sentence_spans:
        for fact_type in _coverage_signal_types(span.text):
            if span.sentence_id not in covered.get(fact_type, set()):
                present_types = covered_at_ref.get(span.sentence_id, set())
                if _coverage_type_is_present(
                    fact_type, present_types, span.text
                ):
                    continue
                pathology_result = bool(_PATHOLOGY_CONTEXT_RE.search(span.text))
                # A pathology-result sentence mentioning the diagnosis or the
                # biopsy is already represented by its histopathology atom.
                # Retrying it as both diagnosis and procedure mostly creates
                # duplicate labels, not new source evidence.
                if (
                    pathology_result
                    and "histopathology" in present_types
                    and fact_type in {"diagnosis", "procedure"}
                ):
                    continue
                if (
                    fact_type == "procedure"
                    and _PERFORMED_PROCEDURE_RE.search(span.text)
                ):
                    # A lossless deterministic supplement runs after the LLM
                    # and is cheaper and more stable than a category retry.
                    continue
                missing_by_type.setdefault(fact_type, set()).add(
                    span.sentence_id
                )
    if not missing_by_type:
        return [], ()

    wanted_refs = set().union(*missing_by_type.values())
    selected = [
        span for span in sentence_spans if span.sentence_id in wanted_refs
    ]
    selected_types = tuple(
        fact_type for fact_type in LLM_ATOMIC_FACT_TYPES
        if fact_type in missing_by_type
    )
    return selected, selected_types


def _coverage_type_is_present(
    wanted: str, present: set[str], source_text: str
) -> bool:
    """Treat only clinically equivalent wire buckets as already covered."""
    if wanted in present:
        return True
    if wanted in {"radiology_finding", "instrumental_finding"} and present & {
        "radiology_finding", "instrumental_finding"
    }:
        return True
    if wanted in {"vital_sign", "clinical_sign"} and present & {
        "vital_sign", "clinical_sign"
    }:
        return True
    if _PLANNED_ACTION_RE.search(source_text):
        # A planned procedure may arrive on the wire in either bucket; it is
        # projected deterministically to clinical_decision before persistence.
        if wanted in {"procedure", "clinical_decision"} and present & {
            "procedure", "clinical_decision"
        }:
            return True
    return False


def _canonical_wire_mapping(
    raw: object,
    *,
    fields: set[str],
    list_fields: set[str],
) -> tuple[dict[str, Any], bool, str | None]:
    if raw is None:
        return {}, True, None
    if not isinstance(raw, dict):
        return {}, False, "deve essere un oggetto."
    unknown = set(raw) - fields
    if unknown:
        return {}, False, (
            "campi non ammessi: " + ", ".join(sorted(unknown)) + "."
        )
    normalized: dict[str, Any] = {}
    changed = False
    for key, item in raw.items():
        if item in (None, "", [], {}):
            changed = True
            continue
        if key in list_fields:
            values = item if isinstance(item, list) else [item]
            cleaned = [
                " ".join(str(value).split()).strip() for value in values
                if not isinstance(value, (dict, list))
                and " ".join(str(value).split()).strip()
            ]
            if cleaned:
                normalized[key] = list(dict.fromkeys(cleaned))
            changed = changed or not isinstance(item, list) or cleaned != item
            continue
        if isinstance(item, (dict, list, bool)):
            return {}, changed, f"{key} deve essere una stringa o null."
        cleaned = " ".join(str(item).split()).strip()
        if cleaned and cleaned.casefold().strip(" .") not in _WIRE_MISSING_TEXT:
            normalized[key] = cleaned
        elif cleaned:
            changed = True
        changed = changed or cleaned != item
    return normalized, changed, None


def _infer_wire_refs(
    concept: str,
    sentence_spans: list[SentenceSpan],
) -> list[int]:
    """Infer a citation only when the mapping is deterministic."""
    if len(sentence_spans) == 1:
        return [1]
    needle = _identity_text(concept)
    if not needle:
        return []
    exact = [
        span.sentence_id for span in sentence_spans
        if needle in _identity_text(span.text)
    ]
    return exact if len(exact) == 1 else []


def _validate_v6_atomic_item(value: dict[str, Any]) -> dict[str, Any] | None:
    """Compatibility helper retained for external/test callers."""
    fact_type = str(value.get("fact_type") or "")
    maximum = max(_wire_refs(value), default=1)
    spans = [SentenceSpan(index, 0, 0, "") for index in range(1, maximum + 1)]
    normalized, _reasons, _changed = _canonical_wire_item(
        value, fact_type, spans
    )
    return normalized


def stable_evidence_id(
    *,
    document_id: str,
    category: str,
    entity: str,
    quote: str,
    observed_date: str | None,
    page: int | None,
    assertion: str,
    certainty: str,
) -> str:
    """Content-derived ID stable across chunk size and worker scheduling."""
    stable_payload = json.dumps(
        {
            "document": document_id,
            "category": category,
            "entity": str(entity or "").casefold(),
            "quote": " ".join(str(quote or "").casefold().split()),
            "date": observed_date,
            "page": page,
            "assertion": assertion,
            "certainty": certainty,
        },
        ensure_ascii=False, sort_keys=True,
    )
    return "EVD_" + uuid.uuid5(uuid.NAMESPACE_URL, stable_payload).hex


def _safe_sentence_refs(
    value: object, spans: list[SentenceSpan] | None
) -> list[int]:
    if not spans or not isinstance(value, list):
        return []
    maximum = len(spans)
    refs = []
    for raw in value[:6]:
        try:
            parsed = int(str(raw).strip().lstrip("Ss"))
        except (TypeError, ValueError):
            continue
        if 1 <= parsed <= maximum and parsed not in refs:
            refs.append(parsed)
    refs.sort()
    return refs


def _quote_from_sentence_refs(
    refs: list[int], spans: list[SentenceSpan] | None
) -> str:
    if not refs or not spans:
        return ""
    # Reassemble a whitespace-normalized contiguous slice. ``locate_quote``
    # maps it back to the exact bytes in the immutable full document.
    selected = spans[refs[0] - 1:refs[-1]]
    return " ".join(span.text for span in selected).strip()


def _coalesce_adjacent_wire_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Join compatible fragments of one concept on adjacent source IDs."""
    result: list[dict[str, Any]] = []
    for raw in items:
        item = copy.deepcopy(raw)
        refs = _wire_refs(item)
        if not result or not refs:
            result.append(item)
            continue
        previous = result[-1]
        previous_refs = _wire_refs(previous)
        same_concept = (
            _clean_text(_wire_value(previous, "c", "category"), 100).casefold()
            == _clean_text(_wire_value(item, "c", "category"), 100).casefold()
            and _clean_text(_wire_value(previous, "e", "entity"), 300).casefold()
            == _clean_text(_wire_value(item, "e", "entity"), 300).casefold()
        )
        adjacent = bool(
            previous_refs
            and refs[0] <= previous_refs[-1] + 1
            and len(set(previous_refs + refs)) <= 6
        )
        if "source_refs" in item or "source_refs" in previous:
            refs_key = "source_refs"
        elif "r" in item or "r" in previous:
            refs_key = "r"
        else:
            refs_key = "refs"
        merged = (
            _merge_wire_values(previous, item, ignored={refs_key})
            if same_concept and adjacent else None
        )
        if merged is None:
            result.append(item)
            continue
        merged[refs_key] = sorted(set(previous_refs + refs))
        result[-1] = merged
    return result


def _attach_referential_wire_continuations(
    items: list[dict[str, Any]],
    spans: list[SentenceSpan],
) -> list[dict[str, Any]]:
    """Attach explicit pronoun continuations to the preceding imaging atom.

    A sentence such as ``Essa è in rapporto di contiguità...`` describes an
    attribute of the lesion in the immediately preceding sentence. Treating
    that relationship as a second clinical finding creates artificial events.
    The source IDs remain combined, so the resulting atom cites both claims.
    """
    result: list[dict[str, Any]] = []
    continuation = re.compile(
        r"(?i)^\s*(?:essa|esso|questa|questo|la (?:lesione|formazione)|"
        r"il reperto)\b"
    )
    relation = re.compile(
        r"(?i)\b(?:rapport\w*|contiguit|clivaggio|impront\w*|"
        r"adiacent\w*|invasion\w*|infiltr\w*)\b"
    )
    for raw in items:
        item = copy.deepcopy(raw)
        refs = _wire_refs(item)
        source = (
            spans[refs[0] - 1].text
            if refs and refs[0] <= len(spans) else ""
        )
        if not (
            result and refs and continuation.search(source)
            and relation.search(source)
            and str(item.get("fact_type") or "") == "radiology_finding"
        ):
            result.append(item)
            continue
        previous = result[-1]
        previous_refs = _wire_refs(previous)
        if (
            str(previous.get("fact_type") or "") != item.get("fact_type")
            or not previous_refs or refs[0] != previous_refs[-1] + 1
            or len(set(previous_refs + refs)) > 6
        ):
            result.append(item)
            continue
        payload = dict(previous.get("payload") or {})
        details = [str(item.get("concept") or "").strip()]
        for value in (item.get("payload") or {}).values():
            cleaned = str(value or "").strip()
            if cleaned and cleaned not in details:
                details.append(cleaned)
        existing = str(payload.get("relation_to_adjacent_structures") or "")
        combined = "; ".join(value for value in (existing, *details) if value)
        payload["relation_to_adjacent_structures"] = combined[:1000]
        previous["payload"] = payload
        previous["source_refs"] = sorted(set(previous_refs + refs))
    # The model can correctly avoid creating a second object yet forget to
    # cite the continuation. Extend the immediately preceding atom directly
    # from the numbered source when the anaphoric relationship is explicit.
    claimed_refs = {
        ref for result_item in result for ref in _wire_refs(result_item)
    }
    for item in result:
        if str(item.get("fact_type") or "") != "radiology_finding":
            continue
        refs = _wire_refs(item)
        while refs and len(refs) < 6:
            next_ref = refs[-1] + 1
            if next_ref > len(spans) or next_ref in claimed_refs:
                break
            continuation_text = spans[next_ref - 1].text
            if not (
                continuation.search(continuation_text)
                and relation.search(continuation_text)
            ):
                break
            payload = dict(item.get("payload") or {})
            existing = str(
                payload.get("relation_to_adjacent_structures") or ""
            ).strip()
            literal = " ".join(continuation_text.split()).strip()
            payload["relation_to_adjacent_structures"] = "; ".join(
                value for value in (existing, literal) if value
            )[:1000]
            item["payload"] = payload
            refs.append(next_ref)
            item["source_refs"] = refs
            claimed_refs.add(next_ref)
    return result


def _wire_value(item: dict[str, Any], short: str, long: str) -> object:
    if short in item:
        return item.get(short)
    if long in item:
        return item.get(long)
    return {
        "category": item.get("fact_type"),
        "entity": item.get("concept"),
        "refs": item.get("source_refs"),
    }.get(long)


def _wire_refs(item: dict[str, Any]) -> list[int]:
    value = _wire_value(item, "r", "refs")
    if not isinstance(value, list):
        return []
    result = []
    for raw in value:
        try:
            parsed = int(raw)
        except (TypeError, ValueError):
            continue
        if parsed > 0 and parsed not in result:
            result.append(parsed)
    return sorted(result)


_MEDICATION_STATE_SIGNAL_RE = re.compile(
    r"(?i)\b(?:propost|programmat|prescritt|avviat|iniziat|assunt|"
    r"somministrat|proseguit|modificat|ridott|aumentat|interrott|"
    r"sospes|ripres|riavviat|completat|terminat|annullat)\w*\b"
)


def _split_multi_state_medication_items(
    items: list[dict[str, Any]], spans: list[SentenceSpan]
) -> list[dict[str, Any]]:
    """Split one model item that cites distinct medication transitions.

    Small models occasionally join treatment start and later suspension in a
    single object. Both source sentences are explicit, so separating them is a
    lossless structural correction; lifecycle, dose and dates are then
    grounded independently by the normal deterministic enrichers.
    """
    result: list[dict[str, Any]] = []
    for raw in items:
        item = copy.deepcopy(raw)
        if str(item.get("fact_type") or "") != "medication":
            result.append(item)
            continue
        refs = _wire_refs(item)
        transition_refs = [
            ref for ref in refs
            if ref <= len(spans)
            and _MEDICATION_STATE_SIGNAL_RE.search(spans[ref - 1].text)
        ]
        if len(transition_refs) < 2:
            result.append(item)
            continue
        for ref in transition_refs:
            source = spans[ref - 1].text.casefold()
            split = copy.deepcopy(item)
            split["source_refs"] = [ref]
            for key in (
                "observation_date", "observation_date_end", "date_precision",
                "clinical_status",
            ):
                split.pop(key, None)
            medication = split.get("medication")
            if isinstance(medication, dict):
                medication.pop("lifecycle_status", None)
                for field in ("dose", "route", "frequency"):
                    value = str(medication.get(field) or "").casefold()
                    if value and value not in source:
                        medication.pop(field, None)
                if not medication:
                    split.pop("medication", None)
            if not re.search(
                r"\b(?:linea|schema|regime|ciclo|adiuvant|neoadiuvant|"
                r"palliativ|curativ|mantenimento)\w*\b", source,
            ):
                split.pop("oncology", None)
            result.append(split)
    return result


def _merge_wire_values(
    left: dict[str, Any],
    right: dict[str, Any],
    *,
    ignored: set[str] | None = None,
) -> dict[str, Any] | None:
    """Deep-merge missing wire fields; return None on any real conflict."""
    ignored = ignored or set()
    merged = copy.deepcopy(left)
    for key, right_value in right.items():
        if key in ignored or right_value in (None, "", [], {}):
            continue
        left_value = merged.get(key)
        if left_value in (None, "", [], {}):
            merged[key] = copy.deepcopy(right_value)
            continue
        if isinstance(left_value, dict) and isinstance(right_value, dict):
            child = _merge_wire_values(left_value, right_value)
            if child is None:
                return None
            merged[key] = child
            continue
        if left_value != right_value:
            return None
    return merged


def _contextual_date_for_refs(
    refs: list[int],
    spans: list[SentenceSpan] | None,
    *,
    maximum_distance: int = 6,
) -> tuple[str | None, int | None, str | None]:
    """Return the nearest explicit date heading governing a later sentence."""
    if not refs or not spans:
        return None, None, None
    first = refs[0]
    current_text = spans[first - 1].text
    explicitly_referential = bool(re.search(
        r"(?i)\b(?:nella\s+stessa\s+data|in\s+tale\s+data|"
        r"in\s+quella\s+data|alla\s+dimissione|successivamente|"
        r"il\s+giorno\s+seguente)\b",
        current_text,
    ))
    lower = max(1, first - maximum_distance)
    for sentence_id in range(first - 1, lower - 1, -1):
        span = spans[sentence_id - 1]
        expression = _explicit_date_expression(span.text)
        if expression and (
            explicitly_referential or _looks_like_temporal_heading(span.text)
        ):
            return expression, sentence_id, span.text
    return None, None, None


_RESOLVABLE_SYMPTOMS = (
    "dispnea", "tosse", "febbre", "dolore", "nausea", "vomito",
    "diarrea", "astenia", "prurito", "rash", "cefalea", "vertigini",
    "edema", "disfagia", "disuria", "ematuria", "parestesie",
)

_NEGATED_IMAGING_CONCEPT_RE = re.compile(
    r"(?i)\b(?:embol\w*|metastas\w*|lesion\w*|nodul\w*|"
    r"linfoaden\w*|adenopati\w*|versament\w*|trombos\w*|frattur\w*|"
    r"ischemi\w*|emorragi\w*|recidiv\w*|progression\w*|"
    r"pneumotorace|consolidament\w*|infiltrat\w*|ostruzion\w*)\b"
)


def _explicit_performed_procedure_evidence(
    *,
    patient_id: str,
    document_id: str,
    document_date: str | None,
    full_text: str,
    geometry,
    model_name: str,
    existing: list[ClinicalEvidence],
) -> list[ClinicalEvidence]:
    """Recover an explicitly performed named procedure without another LLM."""
    spans = split_sentence_spans(full_text)
    result: list[ClinicalEvidence] = []
    for span in spans:
        for match in _PERFORMED_PROCEDURE_RE.finditer(span.text):
            concept = " ".join(match.group("procedure").split()).strip()
            if any(
                item.category in {"procedure", "surgery"}
                and _identity_text(item.normalized_entity) == _identity_text(concept)
                and span.sentence_id in item.data.get("sentence_refs", [])
                for item in (*existing, *result)
            ):
                continue
            quote_verified, matched_quote = locate_quote(span.text, full_text)
            if not quote_verified:
                continue
            temporal_value = _temporal_expression_from_quote(matched_quote)
            temporal = normalize_clinical_date(
                temporal_value, document_date=document_date
            )
            page, bbox = None, None
            if geometry is not None:
                page, bbox = geometry.locate_source(matched_quote, None)
            outcome_match = re.search(
                r"(?i)\b(?:negativ\w*\s+per|con\s+esito)\s+([^.;]{2,120})",
                matched_quote,
            )
            procedure_payload = {
                "procedure_type": concept,
                **(
                    {"outcome": outcome_match.group(0).strip(" .")}
                    if outcome_match else {}
                ),
            }
            data = {
                "fact_type": "procedure",
                "polarity": "present",
                "report_date": document_date,
                "quote_verified": True,
                "date_original_text": temporal.original_text,
                "date_approximate": temporal.approximate,
                "sentence_refs": [span.sentence_id],
                "deterministic_explicit_procedure": True,
                "source_reference": {
                    "document_id": document_id,
                    "page": page,
                    "bbox": list(bbox) if bbox else None,
                    "passage": matched_quote,
                    "sentence_refs": [span.sentence_id],
                },
            }
            result.append(ClinicalEvidence(
                evidence_id=stable_evidence_id(
                    document_id=document_id,
                    category="procedure",
                    entity=concept,
                    quote=matched_quote,
                    observed_date=temporal.start,
                    page=page,
                    assertion="present",
                    certainty="confirmed",
                ),
                patient_id=patient_id,
                document_id=document_id,
                category="procedure",
                normalized_entity=concept,
                fact_type="procedure",
                concept_original=concept,
                canonical_label=concept,
                mapping_status="unmapped",
                typed_payload={"procedure": procedure_payload},
                source_text=matched_quote,
                assertion="present",
                certainty="confirmed",
                clinical_status="completed",
                observed_date=temporal.start,
                document_date=document_date,
                date_precision=temporal.precision,
                date_source=temporal.source,
                source_page=page,
                bbox=bbox,
                confidence=1.0,
                extraction_method="deterministic_explicit_procedure",
                model_name=model_name,
                prompt_version=ATOMIC_PROMPT_VERSION,
                schema_version="3.0",
                status="auto",
                data=data,
            ))
    return result


def _explicit_negated_imaging_evidence(
    *,
    patient_id: str,
    document_id: str,
    document_date: str | None,
    full_text: str,
    geometry,
    model_name: str,
    existing: list[ClinicalEvidence],
) -> list[ClinicalEvidence]:
    """Recover explicit, clinically meaningful negative imaging findings."""
    spans = split_sentence_spans(full_text)
    result: list[ClinicalEvidence] = []
    negative = re.compile(
        r"(?i)\b(?:senza(?:\s+evidenza\s+di)?|assenza\s+di|"
        r"non\s+(?:si\s+)?(?:evidenzia|documenta|rileva)(?:no)?)\s+"
        r"(?P<concept>[^,;:.]{2,100})"
    )
    for span in spans:
        if not _IMAGING_CONTEXT_RE.search(span.text):
            continue
        for match in negative.finditer(span.text):
            candidate = match.group("concept").strip(" .")
            concept_match = _NEGATED_IMAGING_CONCEPT_RE.search(candidate)
            if not concept_match:
                continue
            concept = candidate[concept_match.start():].strip(" .")
            # Do not absorb a following independent clause.
            concept = re.split(
                r"(?i)\s+(?:mentre|ma|tuttavia|con)\s+", concept, maxsplit=1
            )[0].strip(" .")
            if not concept:
                continue
            if any(
                item.category == "imaging_finding"
                and item.assertion == "absent"
                and _identity_text(item.normalized_entity) == _identity_text(concept)
                and span.sentence_id in item.data.get("sentence_refs", [])
                for item in (*existing, *result)
            ):
                continue
            quote_verified, matched_quote = locate_quote(span.text, full_text)
            if not quote_verified:
                continue
            temporal_value = _temporal_expression_from_quote(matched_quote)
            temporal = normalize_clinical_date(
                temporal_value, document_date=document_date
            )
            page, bbox = None, None
            if geometry is not None:
                page, bbox = geometry.locate_source(matched_quote, None)
            modality = re.search(
                r"(?i)\b(?:TC|TAC|RMN?|PET(?:[- /]?FDG|/TC)?|RX|"
                r"ecografi\w*)\b",
                matched_quote,
            )
            typed = {
                "radiology_finding": {
                    "modality": modality.group(0) if modality else ""
                }
            }
            if not typed["radiology_finding"]["modality"]:
                typed = {}
            data = {
                "fact_type": "radiology_finding",
                "polarity": "negated",
                "report_date": document_date,
                "quote_verified": True,
                "date_original_text": temporal.original_text,
                "date_approximate": temporal.approximate,
                "sentence_refs": [span.sentence_id],
                "deterministic_explicit_negative_imaging": True,
                "source_reference": {
                    "document_id": document_id,
                    "page": page,
                    "bbox": list(bbox) if bbox else None,
                    "passage": matched_quote,
                    "sentence_refs": [span.sentence_id],
                },
            }
            result.append(ClinicalEvidence(
                evidence_id=stable_evidence_id(
                    document_id=document_id,
                    category="imaging_finding",
                    entity=concept,
                    quote=matched_quote,
                    observed_date=temporal.start,
                    page=page,
                    assertion="absent",
                    certainty="excluded",
                ),
                patient_id=patient_id,
                document_id=document_id,
                category="imaging_finding",
                normalized_entity=concept,
                fact_type="radiology_finding",
                concept_original=concept,
                canonical_label=concept,
                mapping_status="unmapped",
                typed_payload=typed,
                source_text=matched_quote,
                assertion="absent",
                certainty="excluded",
                clinical_status="excluded",
                observed_date=temporal.start,
                document_date=document_date,
                date_precision=temporal.precision,
                date_source=temporal.source,
                source_page=page,
                bbox=bbox,
                confidence=1.0,
                extraction_method="deterministic_explicit_negative_imaging",
                model_name=model_name,
                prompt_version=ATOMIC_PROMPT_VERSION,
                schema_version="3.0",
                status="auto",
                data=data,
            ))
    return result


def _explicit_resolution_evidence(
    *,
    patient_id: str,
    document_id: str,
    document_date: str | None,
    full_text: str,
    geometry,
    model_name: str,
    existing: list[ClinicalEvidence],
) -> list[ClinicalEvidence]:
    """Recover explicitly documented symptom resolutions without inference."""
    spans = split_sentence_spans(full_text)
    result: list[ClinicalEvidence] = []
    pattern = re.compile(
        r"(?i)(?:\b(?:la|il|lo|l['’]|i|gli|le)\s+|"
        r"\balla\s+dimissione\s+)"
        r"(?P<subject>[a-zà-öø-ÿ][a-zà-öø-ÿ'’\- ]{0,70}?)\s+"
        r"(?:è|e'|sono|risult\w*)\s+"
        r"(?P<state>risolt[oaie]?|scompars[oaie]?|regredit[oaie]?|assente)\b"
    )
    for span in spans:
        for match in pattern.finditer(span.text):
            subject = " ".join(match.group("subject").casefold().split())
            if re.search(r"\bnon$", subject):
                continue
            entities = [
                symptom for symptom in _RESOLVABLE_SYMPTOMS
                if re.search(rf"\b{re.escape(symptom)}\b", subject)
            ]
            if not entities:
                continue
            quote_verified, matched_quote = locate_quote(span.text, full_text)
            if not quote_verified:
                continue
            temporal_value = _temporal_expression_from_quote(matched_quote)
            context_value, context_ref, context_text = (
                _contextual_date_for_refs([span.sentence_id], spans)
            )
            if not temporal_value:
                temporal_value = context_value
            temporal = normalize_clinical_date(
                temporal_value, document_date=document_date
            )
            page, bbox = None, None
            if geometry is not None:
                page, bbox = geometry.locate_source(matched_quote, None)
            for entity in entities:
                if any(
                    item.assertion == "absent"
                    and item.normalized_entity.casefold() == entity
                    and " ".join(item.source_text.casefold().split())
                    == " ".join(matched_quote.casefold().split())
                    for item in (*existing, *result)
                ):
                    continue
                data: dict[str, Any] = {
                    "fact_type": "symptom",
                    "polarity": "negated",
                    "report_date": document_date,
                    "quote_verified": True,
                    "date_original_text": temporal.original_text,
                    "date_approximate": temporal.approximate,
                    "sentence_refs": [span.sentence_id],
                    "deterministic_explicit_resolution": True,
                    "source_reference": {
                        "document_id": document_id,
                        "page": page,
                        "bbox": list(bbox) if bbox else None,
                        "passage": matched_quote,
                        "sentence_refs": [span.sentence_id],
                    },
                }
                if context_ref is not None and context_text:
                    data["date_context_sentence_ref"] = context_ref
                    data["date_context_text"] = context_text
                certainty = (
                    "patient_reported" if re.search(
                        r"(?i)\b(?:riferisc|riferit|segnal)\w*", matched_quote
                    ) else "confirmed"
                )
                result.append(ClinicalEvidence(
                    evidence_id=stable_evidence_id(
                        document_id=document_id,
                        category="symptom",
                        entity=entity,
                        quote=matched_quote,
                        observed_date=temporal.start,
                        page=page,
                        assertion="absent",
                        certainty=certainty,
                    ),
                    patient_id=patient_id,
                    document_id=document_id,
                    category="symptom",
                    normalized_entity=entity,
                    fact_type="symptom",
                    concept_original=entity,
                    canonical_label=entity,
                    mapping_status="unmapped",
                    clinical_relevance="accepted_low_relevance",
                    source_text=matched_quote,
                    assertion="absent",
                    certainty=certainty,
                    clinical_status="resolved",
                    temporality=(
                        "historical" if temporal.start and document_date
                        and temporal.start < document_date else "current"
                    ),
                    observed_date=temporal.start,
                    document_date=document_date,
                    date_precision=temporal.precision,
                    date_source=temporal.source,
                    source_page=page,
                    bbox=bbox,
                    confidence=0.95,
                    extraction_method="llm_atomic_v2",
                    model_name=model_name,
                    prompt_version=ATOMIC_PROMPT_VERSION,
                    schema_version="3.0",
                    status="proposed",
                    data=data,
                ))
    return result


def deduplicate_atomic_evidence(
    evidence: Iterable[ClinicalEvidence],
) -> list[ClinicalEvidence]:
    """Return one clinical atom for repeated source evidence.

    ``document_id``, quote wording and page coordinates deliberately do not
    participate in the identity.  Reworded evidence copied within or between
    reports is one clinical fact when concept, state, date, value and the other
    discriminating fields agree.  Every physical occurrence is kept
    in ``data['source_occurrences']`` so provenance/Quick View remain lossless.

    A changed clinical date, value, treatment state, site, side or severity is
    a different atom.  This prevents a true follow-up measurement or therapy
    transition from being swallowed by copy-and-paste suppression.
    """
    grouped: dict[tuple, list[ClinicalEvidence]] = {}
    for item in evidence:
        grouped.setdefault(_atomic_identity_key(item), []).append(item)

    result: list[ClinicalEvidence] = []
    for items in grouped.values():
        # Prefer a grounded first occurrence.  The earliest source is the
        # canonical one so a copied historical statement cannot move its
        # first-documentation date forward in the registry.
        canonical = copy.deepcopy(min(items, key=_canonical_source_key))
        richest = max(items, key=_atomic_quality_key)
        _enrich_canonical_atom(canonical, richest)

        occurrences = []
        seen_ids: set[str] = set()
        for occurrence in sorted(items, key=_canonical_source_key):
            if occurrence.evidence_id in seen_ids:
                continue
            seen_ids.add(occurrence.evidence_id)
            occurrences.append({
                "evidence_id": occurrence.evidence_id,
                "document_id": occurrence.document_id,
                "document_date": occurrence.document_date,
                "observed_date": occurrence.observed_date,
                "source_page": occurrence.source_page,
                "bbox": list(occurrence.bbox) if occurrence.bbox else None,
                "source_text": occurrence.source_text,
            })
        duplicate_ids = [
            occurrence["evidence_id"] for occurrence in occurrences
            if occurrence["evidence_id"] != canonical.evidence_id
        ]
        canonical.data["atomic_duplicate_count"] = len(duplicate_ids)
        if duplicate_ids:
            canonical.data["duplicate_source_evidence_ids"] = duplicate_ids
            canonical.data["source_occurrences"] = occurrences
        else:
            canonical.data.pop("duplicate_source_evidence_ids", None)
            canonical.data.pop("source_occurrences", None)
        result.append(canonical)
    return result


def _atomic_identity_key(item: ClinicalEvidence) -> tuple:
    """Clinical identity independent of the report containing the quote."""
    therapy = item.data.get("therapy") or {}
    oncology = item.data.get("oncology") or {}
    if item.date_source == "retrospective_duration":
        # Recomputing "da due mesi" against the date of every copied report
        # creates artificial dates.  Its literal relative expression, not the
        # recalculated value, identifies a repeated fact.
        date_key = (
            "relative",
            _identity_text(item.data.get("date_original_text")),
        )
        date_end_key = None
    else:
        date_key = item.observed_date
        date_end_key = item.observed_date_end
    return (
        item.patient_id,
        _identity_text(item.category),
        _identity_text(item.normalized_entity),
        date_key,
        date_end_key,
        _identity_text(item.assertion),
        _identity_text(item.value_text),
        _identity_number(item.numeric_value),
        _identity_text(item.unit),
        _identity_text(item.anatomical_site),
        _identity_text(item.laterality),
        _identity_text(item.severity),
        _identity_text(item.clinical_status),
        _identity_text(therapy.get("lifecycle_status")),
        _identity_text(therapy.get("dose")),
        _identity_text(therapy.get("route")),
        _identity_text(therapy.get("frequency")),
        _identity_text(oncology.get("line_label") or oncology.get("line")),
        _identity_json(oncology.get("regimen")),
        _identity_text(oncology.get("cycle")),
        _identity_text(oncology.get("modification")),
    )


def _identity_text(value: object) -> str:
    normalized = unicodedata.normalize("NFKD", str(value or "").casefold())
    without_marks = "".join(
        char for char in normalized if not unicodedata.combining(char)
    )
    return " ".join(
        "".join(char if char.isalnum() else " " for char in without_marks)
        .split()
    )


def _identity_number(value: object) -> str:
    parsed = _safe_float(value)
    return "" if parsed is None else format(parsed, ".12g")


def _identity_json(value: object) -> str:
    if value in (None, "", [], {}):
        return ""
    if isinstance(value, list):
        value = [_identity_text(item) for item in value]
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _canonical_source_key(item: ClinicalEvidence) -> tuple:
    verified = bool(item.data.get("quote_verified"))
    return (
        0 if verified else 1,
        item.document_date or "9999-99-99",
        item.document_id,
        item.source_page or 10**9,
        item.evidence_id,
    )


def _atomic_quality_key(item: ClinicalEvidence) -> tuple:
    populated = sum(value not in (None, "", [], {}) for value in (
        item.observed_date, item.observed_date_end, item.clinical_status,
        item.anatomical_site, item.laterality, item.severity, item.value_text,
        item.numeric_value, item.unit, item.data,
    ))
    certainty_rank = {
        "confirmed": 5, "patient_reported": 4, "suspected": 3,
        "inferred": 2, "excluded": 1, "unknown": 0,
    }.get(str(item.certainty or "").casefold(), 0)
    significance_rank = {
        "critical": 5, "high": 4, "clinically_relevant": 3,
        "potentially_relevant": 2, "uncertain": 1,
    }.get(str(item.significance or "").casefold(), 0)
    return (
        1 if item.data.get("quote_verified") else 0,
        populated, float(item.confidence or 0.0), certainty_rank,
        significance_rank, item.evidence_id,
    )


def _enrich_canonical_atom(
    canonical: ClinicalEvidence,
    richest: ClinicalEvidence,
) -> None:
    """Keep first-source provenance while adopting stronger metadata."""
    canonical.confidence = max(
        float(canonical.confidence or 0.0), float(richest.confidence or 0.0)
    )
    if _atomic_quality_key(richest) > _atomic_quality_key(canonical):
        canonical.certainty = richest.certainty
        canonical.significance = richest.significance
        if canonical.status == "needs_review" and richest.status != "needs_review":
            canonical.status = richest.status
    canonical.data = _merge_missing_evidence_data(
        copy.deepcopy(canonical.data), richest.data
    )


def _merge_missing_evidence_data(left: dict, right: dict) -> dict:
    for key, value in right.items():
        if value in (None, "", [], {}):
            continue
        current = left.get(key)
        if current in (None, "", [], {}):
            left[key] = copy.deepcopy(value)
        elif isinstance(current, dict) and isinstance(value, dict):
            left[key] = _merge_missing_evidence_data(current, value)
    return left


def _deduplicate_atomic(
    evidence: Iterable[ClinicalEvidence],
) -> list[ClinicalEvidence]:
    """Compatibility alias for callers/tests predating patient-wide dedup."""
    return deduplicate_atomic_evidence(evidence)


def content_hash(*values: object) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(str(value or "").encode("utf-8"))
        digest.update(b"\x00")
    return digest.hexdigest()


def _clean_text(value: object, maximum: int) -> str:
    return " ".join(str(value or "").split())[:maximum].strip()


def _clean_optional(value: object, maximum: int) -> str | None:
    cleaned = _clean_text(value, maximum)
    if cleaned.casefold().strip(" .") in _WIRE_MISSING_TEXT:
        return None
    return cleaned or None


def _safe_float(value: object) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _safe_int(value: object) -> int | None:
    try:
        parsed = int(value) if value is not None else None
        return parsed if parsed and parsed > 0 else None
    except (TypeError, ValueError):
        return None


def _legacy_polarity(assertion: str, certainty: str) -> str:
    """Project legacy assertion/certainty fields onto the v6 polarity enum."""
    if str(certainty or "").casefold() == "suspected":
        return "suspected"
    if (
        str(assertion or "").casefold() == "absent"
        or str(certainty or "").casefold() == "excluded"
    ):
        return "negated"
    return "present"


def _fact_type_for_category(category: str) -> str:
    preferred = {
        "laboratory_finding": "laboratory_test",
        "imaging_finding": "radiology_finding",
        "care_plan": "clinical_decision",
        "recommendation": "clinical_decision",
    }
    return preferred.get(str(category or ""), str(category or "other"))


_NARRATIVE_LAB_ABNORMALITY_RE = re.compile(
    r"(?i)\b(?:anemi\w*|leuco(?:citos|pen)\w*|neutro(?:fil|pen)\w*|"
    r"linfo(?:cit|pen)\w*|trombo(?:cit|pen)\w*|pancitopen\w*|"
    r"iper\w+emi\w*|ipo\w+emi\w*|aumentat\w*|incrementat\w*|"
    r"elevat\w*|ridott\w*|diminuit\w*|soppress\w*|alterat\w*|"
    r"fuori\s+(?:range|intervallo)|sopra\s+(?:il\s+)?(?:range|limite)|"
    r"sotto\s+(?:il\s+)?(?:range|limite)|positiv\w*|negativ\w*|"
    r"patologic\w*|marcatamente)\b"
)


def _llm_laboratory_claim_is_relevant(
    *, entity: str, quote: str, item: dict[str, Any]
) -> bool:
    """Admit only source-grounded narrative laboratory abnormalities.

    Structured rows remain the authoritative numeric path.  This fallback is
    deliberately narrower: it recovers qualitative patterns and values whose
    abnormality is explicit in prose, but rejects an isolated normal number.
    """
    payload = item.get("typed_payload")
    if not isinstance(payload, dict):
        payload = item.get("payload")
    payload = payload if isinstance(payload, dict) else {}
    if any(payload.get(key) not in (None, "", [], {}) for key in (
        "reference_low", "reference_high", "reference_text", "flag",
        "abnormal_direction",
    )):
        return True
    value_text = str(item.get("value_text") or "")
    source = f"{entity} {quote} {value_text}"
    if _NARRATIVE_LAB_ABNORMALITY_RE.search(source):
        return True
    # Explicit visual flags are common in normalized laboratory tables.
    return bool(re.search(r"(?:^|\s)(?:\*{1,3}|[HL])(?:\s|$)", quote))


def _project_planned_action(
    *, fact_type: str, category: str, entity: str, quote: str,
    item: dict[str, Any],
) -> tuple[str, str]:
    """Represent a documented plan without fabricating a performed event."""
    if fact_type not in {
        "procedure", "radiology_finding", "instrumental_finding",
        "hospitalization", "discharge",
    }:
        return fact_type, category
    if not _PLANNED_ACTION_RE.search(quote) or _PERFORMED_OR_RESULT_RE.search(
        quote
    ):
        return fact_type, category
    fact_type = "clinical_decision"
    category = "care_plan"
    item["fact_type"] = fact_type
    item["category"] = category
    item["clinical_status"] = "planned"
    payload = item.get("typed_payload")
    if not isinstance(payload, dict):
        payload = item.get("payload")
    payload = dict(payload) if isinstance(payload, dict) else {}
    item["typed_payload"] = {
        "action": "planned",
        "target": entity,
        **({"timing": payload["timing"]} if payload.get("timing") else {}),
    }
    return fact_type, category


def _specific_atomic_category(
    *, category: str, fact_type: str, concept: str, quote: str
) -> str:
    """Project broad wire buckets onto explicit clinical event categories."""
    entity = _identity_text(concept)
    source = _identity_text(quote)
    if fact_type == "vital_sign" and entity in {
        "ecog", "performance status"
    }:
        return "functional_status"
    if fact_type != "diagnosis":
        return category
    if "familiarita" in source or "anamnesi familiare" in source:
        return "family_history"
    if "allerg" in entity or "allerg" in source:
        return "allergy"
    if "progression" in entity:
        return "progression"
    if any(marker in entity for marker in (
        "risposta completa", "risposta parziale", "risposta metabolica",
        "stabilita di malattia",
    )):
        return "response"
    if (
        any(marker in entity for marker in (
            "tossicita", "miocardite", "polineuropatia", "uveite",
        ))
        and any(marker in source for marker in (
            "tossicita", "correlat", "trattamento", "terapia",
        ))
    ):
        return "toxicity"
    if re.search(r"\bimmuno[- ]?correlat\w*\b", source):
        return "toxicity"
    return category


def _bounded_float(value: object, default: float) -> float:
    parsed = _safe_float(value)
    return max(0.0, min(parsed if parsed is not None else default, 1.0))


def _sanitize_nested_payload(
    value: object,
    *,
    allowed: set[str],
    list_fields: set[str] | None = None,
) -> dict[str, Any]:
    """Keep schema-defined explicit metadata even if a backend ignores it."""
    if not isinstance(value, dict):
        return {}
    list_fields = list_fields or set()
    clean: dict[str, Any] = {}
    for key in allowed:
        raw = value.get(key)
        if raw in (None, "", [], {}):
            continue
        if key in list_fields:
            items = raw if isinstance(raw, list) else [raw]
            normalized = [
                _clean_wire_scalar(item, 300) for item in items
                if _clean_wire_scalar(item, 300)
            ]
            if normalized:
                clean[key] = list(dict.fromkeys(normalized))
            continue
        if isinstance(raw, (dict, list)):
            continue
        cleaned = _clean_wire_scalar(raw, 500)
        if cleaned:
            clean[key] = cleaned
    return clean


def _clean_wire_scalar(value: object, maximum: int) -> str:
    cleaned = _clean_text(value, maximum).strip("{}[]")
    if cleaned.casefold().strip(" .") in {
        "unknown", "n.d", "nd", "non disponibile", "none", "null",
    }:
        return ""
    return cleaned


def _temporal_expression_from_quote(
    quote: str, *, category: str | None = None
) -> str | None:
    """Recover an explicit event date/duration omitted by the model."""
    text = str(quote or "")
    duration = re.search(
        r"(?i)\b(?:da(?:\s+circa)?|circa\s+da)\s+"
        r"(?:\d+|un|uno|una|due|tre|quattro|cinque|sei|sette|otto|nove|"
        r"dieci|undici|dodici)\s+"
        r"(?:giorn(?:o|i)|settiman(?:a|e)|mes(?:e|i)|ann(?:o|i))\b",
        text,
    )
    if duration:
        return duration.group(0)
    interval = re.search(
        r"(?i)\bdal\s+\d{1,2}\s+al\s+\d{1,2}\s+"
        r"(?:gennaio|febbraio|marzo|aprile|maggio|giugno|luglio|agosto|"
        r"settembre|ottobre|novembre|dicembre)(?:\s+(?:19|20)\d{2})?\b",
        text,
    )
    if interval:
        return interval.group(0)
    if _temporal_context_mismatch(category, text):
        return None
    return _explicit_date_expression(text)


def _explicit_date_expression(text: str) -> str | None:
    explicit = re.search(
        r"\b(?:\d{1,2}[./-]\d{1,2}[./-](?:\d{2}|\d{4})|"
        r"(?:19|20)\d{2}-\d{1,2}-\d{1,2}|"
        r"\d{1,2}\s+(?:gennaio|febbraio|marzo|aprile|maggio|giugno|"
        r"luglio|agosto|settembre|ottobre|novembre|dicembre)"
        r"(?:\s+(?:19|20)\d{2})?|"
        r"(?:gennaio|febbraio|marzo|aprile|maggio|giugno|luglio|agosto|"
        r"settembre|ottobre|novembre|dicembre)\s+(?:19|20)\d{2})\b",
        str(text or ""),
        re.IGNORECASE,
    )
    return explicit.group(0) if explicit else None


def _duration_reference_date(
    temporal_value: object,
    *,
    matched_quote: str,
    contextual_date: str | None,
    document_date: str | None,
) -> str | None:
    """Anchor a retrospective duration to its local dated note section."""
    if not re.search(
        r"(?i)\b(?:da(?:\s+circa)?|circa\s+da)\s+"
        r"(?:\d+|un|uno|una|due|tre|quattro|cinque|sei|sette|otto|nove|"
        r"dieci|undici|dodici)\s+"
        r"(?:giorn(?:o|i)|settiman(?:a|e)|mes(?:e|i)|ann(?:o|i))\b",
        str(temporal_value or ""),
    ):
        return document_date
    for candidate in (
        _explicit_date_expression(matched_quote), contextual_date
    ):
        if not candidate:
            continue
        normalized = normalize_clinical_date(candidate)
        if normalized.start and normalized.precision == "day":
            return normalized.start
    return document_date


def _ground_certainty(
    *,
    category: str,
    assertion: str,
    certainty: str,
    quote: str,
    numeric_value: float | None,
    quote_verified: bool,
) -> str:
    """Correct certainty values contradicted by the cited source itself."""
    text = str(quote or "").casefold()
    if category not in {
        "medication", "procedure", "hospitalization", "discharge", "care_plan"
    } and re.search(
        r"\b(?:esclus[oa]|assenza\s+di|non\s+(?:si\s+)?evidenz\w*|"
        r"negativ[oa]\s+per)\b",
        text,
    ):
        return "excluded"
    if category == "symptom" and re.search(
        r"\b(?:riferisc|riferit|lament|segnal)\w*", text
    ):
        return "patient_reported"
    if re.search(
        r"\b(?:quadro\s+)?compatibile\s+con\b|"
        r"\b(?:sospett[oa]|possibile|probabile|verosimile)\b",
        text,
    ):
        return "suspected"
    objective = {
        "diagnosis", "clinical_sign", "vital_sign", "laboratory_finding",
        "imaging_finding", "pathology_finding", "biomarker", "medication",
        "procedure", "toxicity", "response", "progression",
    }
    if (
        quote_verified and assertion == "present"
        and (numeric_value is not None or category in objective)
        and certainty in {"unknown", "excluded"}
    ):
        return "confirmed"
    return certainty


def _temporal_context_mismatch(category: str | None, quote: str) -> bool:
    """Reject a date explicitly scoped to another event in a shared quote."""
    if category == "medication":
        return False
    return bool(re.search(
        r"(?i)\bultima\s+somministrazione\b[^.;]*"
        r"\b\d{1,2}[./-]\d{1,2}[./-](?:\d{2}|\d{4})\b",
        str(quote or ""),
    ))


def _medication_transition(
    category: str,
    quote: str,
    assertion: str,
    clinical_status: str | None,
    therapy: dict[str, Any],
) -> tuple[str, str | None, dict[str, Any]]:
    """Correct common confusion between negation and medication lifecycle."""
    if category != "medication":
        return assertion, clinical_status, therapy
    text = str(quote or "").casefold()
    lifecycle = therapy.get("lifecycle_status")
    if (
        re.search(
            r"\b(?:si\s+)?sospend\w*|\bsospes[oaie]?\b|"
            r"\binterrott[oaie]?\b",
            text,
        )
        and not re.search(r"\bnon\s+(?:si\s+)?sospend", text)
        and assertion not in {"conditional", "hypothetical"}
    ):
        lifecycle = "suspended"
    elif re.search(
        r"\b(?:si\s+)?(?:avvia|inizia)\b|\b(?:avviat|iniziat)[oaie]?\b",
        text,
    ):
        lifecycle = "started"
    elif re.search(
        r"\b(?:dose\s+)?(?:ridott|aumentat)[oaie]?\b|"
        r"\b(?:riduzione|incremento)\s+(?:della\s+)?dose\b",
        text,
    ):
        lifecycle = "dose_changed"
    elif re.search(r"\b(?:ripres|riavviat)[oaie]?\b|\briprende\b", text):
        lifecycle = "resumed"
    elif re.search(r"\b(?:completat|terminat)[oaie]?\b", text):
        lifecycle = "completed"
    elif "ultima somministrazione" in text:
        lifecycle = "administered"
    elif re.search(r"\b(?:è|e)\s+in trattamento con\b", text):
        lifecycle = "active"
    if lifecycle:
        assertion = "present"
        clinical_status = lifecycle
        therapy = dict(therapy)
        therapy["lifecycle_status"] = lifecycle
        if therapy.get("intent") in {
            "sospeso", "sospesa", "avviato", "avviata", "iniziato", "iniziata"
        }:
            therapy.pop("intent", None)
    return assertion, clinical_status, therapy


def _enrich_medication_payload(
    category: str,
    entity: str,
    quote: str,
    therapy: dict[str, Any],
    oncology: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Recover explicit drug attributes without external normalization."""
    if category != "medication":
        return therapy, oncology
    therapy = dict(therapy)
    oncology = dict(oncology)
    therapy.setdefault("original_name", entity)
    dose = re.search(
        r"(?i)\b\d+(?:[.,]\d+)?\s*(?:mg|mcg|µg|g|ml|UI|U\.I\.)"
        r"(?:\s*/\s*(?:die|giorno))?\b",
        quote,
    )
    if dose and not therapy.get("dose"):
        therapy["dose"] = " ".join(dose.group(0).split())
    route = re.search(
        r"(?i)\b(?:e\.v\.|ev|i\.v\.|iv|per\s+os|orale|p\.o\.|po|"
        r"s\.c\.|sc|i\.m\.|im)\b",
        quote,
    )
    if route and not therapy.get("route"):
        therapy["route"] = " ".join(route.group(0).split())
    frequency = re.search(
        r"(?i)\b(?:ogni\s+\d+\s+(?:ore|giorni?|settimane?|mesi)|"
        r"\d+\s+volte\s+(?:al|/\s*)\s*(?:die|giorno)|die)\b",
        quote,
    )
    if frequency and not therapy.get("frequency"):
        therapy["frequency"] = " ".join(frequency.group(0).split())
    intent = str(oncology.get("intent") or "").casefold()
    if intent and not re.search(
        r"\b(?:curativ|palliativ|adiuvant|neoadiuvant|radical|mantenimento)",
        intent,
    ):
        oncology.setdefault("indication", oncology.pop("intent"))
    if oncology and not oncology.get("regimen"):
        oncology["regimen"] = [entity]
    return therapy, oncology


def _normalize_measurement(
    entity: str,
    category: str,
    quote: str,
    numeric_value: float | None,
    unit: str | None,
) -> tuple[str, str, float | None, str | None]:
    """Normalize high-value vital measurements without clinical inference."""
    entity_is_spo2 = bool(re.search(
        r"(?i)^\s*(?:spo2|saturazione(?:\s+di\s+ossigeno)?)"
        r"(?:\s*[<>≤≥]?\s*\d+(?:[.,]\d+)?\s*%?)?\s*$", entity
    ))
    percent_spo2 = bool(
        str(unit or "").strip() == "%"
        and re.search(
            r"(?i)\b(?:spo2|saturazione(?:\s+di\s+ossigeno)?)\b", quote
        )
    )
    if entity_is_spo2 or percent_spo2:
        match = re.search(
            r"(?i)\b(?:spo2|saturazione(?:\s+di\s+ossigeno)?)\s*"
            r"(?:(?:era|pari\s+a|di)\s+|[:=]?\s*)"
            r"([<>≤≥]?\s*\d+(?:[.,]\d+)?)\s*%?",
            quote,
        )
        if numeric_value is None and match:
            numeric_value = _safe_float(
                re.sub(r"[^\d,.-]", "", match.group(1)).replace(",", ".")
            )
        return "SpO2", "vital_sign", numeric_value, unit or "%"
    if (
        _identity_text(entity) in {"temperatura", "febbre"}
        or str(unit or "").casefold().strip() in {"°c", "c", "celsius"}
    ):
        match = re.search(
            r"(?i)\btemperatura\s*"
            r"(?:(?:era|pari\s+a|di)\s+|[:=]?\s*)"
            r"([<>≤≥]?\s*\d+(?:[.,]\d+)?)\s*°?C\b",
            quote,
        )
        if numeric_value is None and match:
            numeric_value = _safe_float(
                re.sub(r"[^\d,.-]", "", match.group(1)).replace(",", ".")
            )
        return "temperatura", "vital_sign", numeric_value, unit or "°C"
    return entity, category, numeric_value, unit


def _normalize_entity_severity(
    entity: str, severity: str | None
) -> tuple[str, str | None]:
    """Keep explicit grade in severity rather than inside the base entity."""
    match = re.search(
        r"(?i)\b(?:di\s+)?grado\s+([0-5]|I{1,3}|IV|V)\b", entity
    )
    if not match:
        return entity, severity
    if severity is None:
        severity = f"grado {match.group(1)}"
    normalized = (entity[:match.start()] + entity[match.end():]).strip(" ,;:-")
    return normalized or entity, severity


def _normalize_significance(value: object) -> str:
    normalized = str(value or "clinically_relevant").strip().lower()
    return normalized if normalized in {
        "critical", "high", "clinically_relevant", "potentially_relevant",
        "uncertain",
    } else "uncertain"

"""Measured llama.cpp slot tuning for the atomic-evidence workload.

The benchmark deliberately uses a synthetic clinical passage.  It exercises
the same prompt, JSON schema, validation and provenance path used by the real
registry builder without reading or persisting any patient data.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, replace
import statistics
import time
from typing import Callable, Iterable

from ..clinical.atomic_evidence import AtomicEvidenceExtractor
from ..extraction.llm_client import LlmClient
from ..settings import LLMRoleConfig
from .hardware import get_safe_max_workers


ATOMIC_SLOT_BENCHMARK_TEXT = """VISITA ONCOLOGICA DEL 14/03/2025.
Diagnosi: melanoma cutaneo metastatico.
La paziente riferisce tosse secca e dispnea insorte da tre giorni.
SpO2 88% in aria ambiente.
TC torace del 13/03/2025: nuove opacità bilaterali a vetro smerigliato.
Il trattamento con nivolumab è stato temporaneamente sospeso.
È stato prescritto prednisone 50 mg per os una volta al giorno.
Il quadro è ritenuto compatibile con sospetta polmonite immuno-correlata."""

_EXPECTED_FACT_TYPES = frozenset({
    "diagnosis",
    "symptom",
    "vital_sign",
    "radiology_finding",
    "medication",
})
ATOMIC_BENCHMARK_MIN_QUALITY = 0.80
ATOMIC_BENCHMARK_MIN_EVIDENCE = 4


@dataclass(frozen=True)
class SlotBenchmarkSample:
    """One measured server shape."""

    slots: int
    requests: int
    completed: int
    elapsed_seconds: float
    documents_per_minute: float
    completion_tokens_per_second: float
    quality_score: float
    median_evidence_count: float
    output_limit_retries: int = 0
    validation_retries: int = 0
    unresolved_invalid_items: int = 0
    error: str = ""

    @property
    def reliable(self) -> bool:
        return bool(
            self.completed == self.requests
            and self.requests > 0
            and not self.error
            and self.output_limit_retries == 0
            and self.unresolved_invalid_items == 0
            and self.quality_score >= ATOMIC_BENCHMARK_MIN_QUALITY
            and self.median_evidence_count >= ATOMIC_BENCHMARK_MIN_EVIDENCE
        )


@dataclass(frozen=True)
class SlotBenchmarkResult:
    """Complete benchmark report and selected slot count."""

    recommended_slots: int
    samples: tuple[SlotBenchmarkSample, ...]
    quality_floor: float
    rationale: str


def slot_benchmark_candidates(
    model_name: str,
    context_length: int,
    current_slots: int,
) -> tuple[int, ...]:
    """Return useful, capacity-safe shapes without benchmarking every value."""

    safe_max = max(1, int(get_safe_max_workers(model_name, context_length)))
    current = max(1, min(int(current_slots or 1), safe_max))
    if safe_max < 4:
        return tuple(range(1, safe_max + 1))
    candidates = {
        current,
        4,
        *(value for value in (6, 8) if value <= safe_max),
    }
    return tuple(sorted(value for value in candidates if value <= safe_max))


def choose_slot_benchmark_result(
    samples: Iterable[SlotBenchmarkSample],
    *,
    plateau_tolerance: float = 0.03,
) -> SlotBenchmarkResult:
    """Select the fastest reliable shape without trading away extraction.

    Quality is compared within the same model/prompt benchmark.  Candidates
    more than five percentage points below the best observed coverage are
    discarded.  If multiple candidates are within three percent of peak
    throughput, the smaller one wins to retain memory headroom.
    """

    measured = tuple(sorted(samples, key=lambda item: item.slots))
    reliable = [item for item in measured if item.reliable]
    if not reliable:
        details = "; ".join(
            f"{item.slots} slot: {item.error or 'risultato non affidabile'}"
            for item in measured
        )
        raise RuntimeError(
            "Nessuna configurazione ha completato il benchmark senza errori"
            + (f" ({details})" if details else "")
        )

    best_quality = max(item.quality_score for item in reliable)
    quality_floor = max(0.0, best_quality - 0.05)
    accurate = [
        item for item in reliable if item.quality_score >= quality_floor
    ]
    peak = max(item.documents_per_minute for item in accurate)
    near_peak = [
        item for item in accurate
        if item.documents_per_minute >= peak * (1.0 - plateau_tolerance)
    ]
    selected = min(near_peak, key=lambda item: item.slots)
    return SlotBenchmarkResult(
        recommended_slots=selected.slots,
        samples=measured,
        quality_floor=quality_floor,
        rationale=(
            f"{selected.slots} slot: {selected.documents_per_minute:.2f} "
            "documenti/min; configurazione più piccola entro il 3% del "
            "massimo misurato e senza perdita clinica rilevata"
        ),
    )


def benchmark_atomic_slots(
    config: LLMRoleConfig,
    *,
    candidates: Iterable[int] | None = None,
    progress_callback: Callable[[str], None] | None = None,
    cancel_check: Callable[[], bool] | None = None,
    client_factory=LlmClient,
    extractor_factory=AtomicEvidenceExtractor,
) -> SlotBenchmarkResult:
    """Measure atomic extraction throughput for several llama-server shapes.

    Every shape runs two full waves so all configured slots are exercised.
    Model loading is excluded from throughput.  Candidate runtimes are always
    stopped before moving on, including after an error or cancellation.
    """

    if config.backend != "llama_cpp":
        raise ValueError(
            "Il benchmark degli slot è disponibile per llama.cpp/GGUF"
        )
    selected_candidates = tuple(candidates or slot_benchmark_candidates(
        config.model, config.context_length, config.parallel_workers
    ))
    if not selected_candidates:
        raise ValueError("Nessun numero di slot compatibile da misurare")

    samples: list[SlotBenchmarkSample] = []
    for candidate_index, slots in enumerate(selected_candidates, start=1):
        if cancel_check is not None and cancel_check():
            raise RuntimeError("Benchmark interrotto")
        slots = max(1, int(slots))
        requests = slots * 2
        candidate_config = replace(config, parallel_workers=slots)
        client = client_factory(config=candidate_config)
        if progress_callback is not None:
            progress_callback(
                f"Configurazione {candidate_index}/{len(selected_candidates)}: "
                f"caricamento di {slots} slot…"
            )

        # Reserving the phase before loading the candidate is essential: a
        # resident model from another role would distort both RAM and speed.
        client.retain_only_this_runtime()
        try:
            client.warmup()
            extractor = extractor_factory(client)
            # Prime the real JSON grammar and extraction path before starting
            # the clock. Otherwise the first candidate alone pays one-time
            # schema/kernel initialization and later, larger shapes look
            # artificially faster.
            extractor.extract_document(
                patient_id="BENCHMARK",
                document_id="SYNTHETIC_WARMUP",
                document_type="visita oncologica",
                document_date="2025-03-14",
                text=ATOMIC_SLOT_BENCHMARK_TEXT,
            )
            if progress_callback is not None:
                progress_callback(
                    f"Misura {slots} slot: {requests} estrazioni sintetiche…"
                )
            started = time.perf_counter()
            evidence_counts: list[int] = []
            quality_scores: list[float] = []
            completion_tokens = 0
            output_retries = 0
            validation_retries = 0
            unresolved = 0
            errors: list[str] = []

            def extract_once(request_index: int):
                evidence = extractor.extract_document(
                    patient_id="BENCHMARK",
                    document_id=f"SYNTHETIC_{request_index:02d}",
                    document_type="visita oncologica",
                    document_date="2025-03-14",
                    text=ATOMIC_SLOT_BENCHMARK_TEXT,
                )
                metrics = extractor.last_extraction_metrics()
                fact_types = {
                    str(item.fact_type or "") for item in evidence
                }
                quality = len(
                    fact_types & _EXPECTED_FACT_TYPES
                ) / len(_EXPECTED_FACT_TYPES)
                return len(evidence), quality, metrics

            with ThreadPoolExecutor(max_workers=slots) as pool:
                futures = [
                    pool.submit(extract_once, request_index)
                    for request_index in range(requests)
                ]
                for future in as_completed(futures):
                    try:
                        count, quality, metrics = future.result()
                        evidence_counts.append(count)
                        quality_scores.append(quality)
                        completion_tokens += int(
                            metrics.get("completion_tokens") or 0
                        )
                        output_retries += int(
                            metrics.get("output_limit_retries") or 0
                        )
                        validation_retries += int(
                            metrics.get("validation_retries") or 0
                        )
                        unresolved += int(
                            metrics.get("unresolved_invalid_items") or 0
                        )
                    except Exception as exc:  # report every failed request
                        errors.append(str(exc))
            elapsed = max(0.001, time.perf_counter() - started)
            completed = len(evidence_counts)
            samples.append(SlotBenchmarkSample(
                slots=slots,
                requests=requests,
                completed=completed,
                elapsed_seconds=elapsed,
                documents_per_minute=completed * 60.0 / elapsed,
                completion_tokens_per_second=(
                    completion_tokens / elapsed if completion_tokens else 0.0
                ),
                quality_score=(
                    statistics.fmean(quality_scores)
                    if quality_scores else 0.0
                ),
                median_evidence_count=(
                    statistics.median(evidence_counts)
                    if evidence_counts else 0.0
                ),
                output_limit_retries=output_retries,
                validation_retries=validation_retries,
                unresolved_invalid_items=unresolved,
                error="; ".join(dict.fromkeys(errors))[:1000],
            ))
        except Exception as exc:
            samples.append(SlotBenchmarkSample(
                slots=slots,
                requests=requests,
                completed=0,
                elapsed_seconds=0.0,
                documents_per_minute=0.0,
                completion_tokens_per_second=0.0,
                quality_score=0.0,
                median_evidence_count=0.0,
                error=str(exc),
            ))
        finally:
            # Do not leave multiple benchmark shapes (and therefore duplicate
            # model weights) resident after the measurement.
            try:
                client.backend.stop_config(client)
            except Exception:
                pass

    return choose_slot_benchmark_result(samples)

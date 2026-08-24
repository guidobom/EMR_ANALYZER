from types import SimpleNamespace
from unittest.mock import patch

import pytest

from emr_analyzer.settings import LLMRoleConfig
from emr_analyzer.utils.slot_benchmark import (
    SlotBenchmarkSample,
    benchmark_atomic_slots,
    choose_slot_benchmark_result,
    slot_benchmark_candidates,
)


def _sample(
    slots: int,
    rate: float,
    *,
    quality: float = 1.0,
    output_retries: int = 0,
) -> SlotBenchmarkSample:
    return SlotBenchmarkSample(
        slots=slots,
        requests=8,
        completed=8,
        elapsed_seconds=8 * 60 / rate,
        documents_per_minute=rate,
        completion_tokens_per_second=20.0,
        quality_score=quality,
        median_evidence_count=6.0,
        output_limit_retries=output_retries,
    )


def test_candidates_include_current_and_capacity_safe_high_values():
    with patch(
        "emr_analyzer.utils.slot_benchmark.get_safe_max_workers",
        return_value=8,
    ):
        assert slot_benchmark_candidates("medgemma", 8192, 3) == (
            3, 4, 6, 8,
        )


def test_selector_keeps_smaller_shape_inside_throughput_plateau():
    result = choose_slot_benchmark_result([
        _sample(4, 10.0),
        _sample(6, 10.2),
        _sample(8, 10.25),
    ])

    assert result.recommended_slots == 4
    assert "entro il 3%" in result.rationale


def test_selector_rejects_faster_shape_with_quality_loss_or_truncation():
    result = choose_slot_benchmark_result([
        _sample(4, 8.0),
        _sample(6, 11.0, quality=0.8),
        _sample(8, 12.0, output_retries=1),
    ])

    assert result.recommended_slots == 4


def test_selector_rejects_formally_complete_but_clinically_empty_runs():
    with pytest.raises(RuntimeError, match="senza errori"):
        choose_slot_benchmark_result([
            _sample(4, 20.0, quality=0.0),
            _sample(6, 25.0, quality=0.2),
        ])


def test_atomic_benchmark_uses_two_waves_and_stops_each_runtime():
    clients = []

    class FakeBackend:
        def __init__(self):
            self.stopped = 0

        def stop_config(self, _client):
            self.stopped += 1
            return True

    class FakeClient:
        def __init__(self, *, config):
            self.config = config
            self.backend = FakeBackend()
            clients.append(self)

        def retain_only_this_runtime(self):
            return 1

        def warmup(self):
            return {"slots": self.config.parallel_workers}

    class FakeExtractor:
        def __init__(self, _client):
            pass

        def extract_document(self, **_kwargs):
            return [
                SimpleNamespace(fact_type=fact_type)
                for fact_type in (
                    "diagnosis", "symptom", "vital_sign",
                    "radiology_finding", "medication",
                )
            ]

        def last_extraction_metrics(self):
            return {
                "completion_tokens": 100,
                "output_limit_retries": 0,
                "validation_retries": 0,
                "unresolved_invalid_items": 0,
            }

    result = benchmark_atomic_slots(
        LLMRoleConfig(model="medgemma", context_length=8192),
        candidates=(2, 4),
        client_factory=FakeClient,
        extractor_factory=FakeExtractor,
    )

    assert [sample.requests for sample in result.samples] == [4, 8]
    assert [sample.completed for sample in result.samples] == [4, 8]
    assert all(sample.quality_score == 1.0 for sample in result.samples)
    assert [client.backend.stopped for client in clients] == [1, 1]


def test_atomic_benchmark_is_llama_cpp_specific():
    with pytest.raises(ValueError, match="llama.cpp"):
        benchmark_atomic_slots(
            LLMRoleConfig(model="local", backend="vllm"),
            candidates=(2, 4),
        )

from unittest.mock import patch

from emr_analyzer.utils.hardware import (
    HardwareProfile,
    ModelProfile,
    RecommendedParams,
    calculate_max_workers,
)


def test_worker_capacity_does_not_double_count_resident_model_ram():
    with (
        patch(
            "emr_analyzer.utils.hardware.get_total_ram_gb",
            return_value=48.0,
        ),
        patch(
            "emr_analyzer.utils.hardware.get_model_size_gb",
            return_value=9.3,
        ),
    ):
        assert calculate_max_workers("qwen3-14b", 32768) == 8


def test_qwen_14b_recommendation_uses_capacity_and_long_clinical_output():
    hardware = HardwareProfile(
        total_ram_gb=48.0,
        available_ram_gb=13.0,
        cpu_cores_physical=12,
        cpu_cores_logical=12,
        has_apple_silicon=True,
        gpu_name="Apple M4 Max",
    )
    model = ModelProfile(
        model_name="qwen3-14b",
        size_gb=9.3,
        max_context_length=40960,
        architecture="qwen3",
    )

    recommendation = RecommendedParams.compute(
        hardware, model, role="clinical_state"
    )

    assert recommendation.context_length == 40960
    assert recommendation.max_output_tokens == 10240
    assert recommendation.parallel_workers == 3
    assert "RAM totale" in recommendation.context_rationale

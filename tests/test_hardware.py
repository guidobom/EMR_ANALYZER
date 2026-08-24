from subprocess import CompletedProcess
from unittest.mock import patch

from emr_analyzer.llm_backend.diagnostics import diagnose_llama_acceleration
from emr_analyzer.utils.hardware import (
    HardwareProfile,
    ModelProfile,
    RecommendedParams,
    calculate_max_workers,
)


def _probe(output: str, *, system: str, gpu: str):
    with patch(
        "emr_analyzer.llm_backend.diagnostics.subprocess.run",
        return_value=CompletedProcess(
            ["/opt/llama-server", "--list-devices"], 0, output, ""
        ),
    ):
        return diagnose_llama_acceleration(
            "/opt/llama-server",
            system_name=system,
            machine="arm64",
            gpu_name=gpu,
        )


def test_llama_diagnostic_confirms_cuda_device_exposed_by_binary():
    diagnostic = _probe(
        "ggml_cuda_init: found 1 CUDA devices:\n"
        "Available devices:\n"
        "  CUDA0: NVIDIA GB10 (119808 MiB, 110000 MiB free)",
        system="Linux",
        gpu="NVIDIA GB10",
    )

    assert diagnostic.status == "accelerated"
    assert diagnostic.runtime_backend == "CUDA"
    assert diagnostic.compiled_backends == ("CUDA",)
    assert diagnostic.devices == (
        "CUDA0: NVIDIA GB10 (119808 MiB, 110000 MiB free)",
    )


def test_llama_diagnostic_distinguishes_cuda_runtime_failure():
    diagnostic = _probe(
        "ggml_cuda_init: failed to initialize CUDA: "
        "no CUDA-capable device is detected\nAvailable devices:\n  (none)",
        system="Linux",
        gpu="NVIDIA GB10",
    )

    assert diagnostic.status == "backend_unavailable"
    assert diagnostic.expected_backend == "CUDA"
    assert diagnostic.runtime_backend == "CPU"
    assert "Supporto CUDA compilato" in diagnostic.summary


def test_llama_diagnostic_confirms_metal_device_on_macos():
    diagnostic = _probe(
        "ggml_metal_init: GPU name: Apple M4 Max\n"
        "Available devices:\n  Metal: Apple M4 Max",
        system="Darwin",
        gpu="Apple M4 Max",
    )

    assert diagnostic.status == "accelerated"
    assert diagnostic.expected_backend == "METAL"
    assert diagnostic.runtime_backend == "METAL"
    assert diagnostic.devices == ("Metal: Apple M4 Max",)


def test_llama_diagnostic_reports_build_without_expected_metal():
    diagnostic = _probe(
        "Available devices:\n  (none)",
        system="Darwin",
        gpu="Apple M2 Pro",
    )

    assert diagnostic.status == "backend_missing"
    assert "non risulta compilato con METAL" in diagnostic.summary


def test_llama_diagnostic_reports_missing_binary():
    with patch(
        "emr_analyzer.llm_backend.diagnostics.find_server_binary",
        return_value=None,
    ):
        diagnostic = diagnose_llama_acceleration(
            system_name="Darwin", machine="arm64", gpu_name="Apple M3"
        )

    assert diagnostic.status == "binary_missing"
    assert diagnostic.runtime_backend == "SCONOSCIUTO"


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

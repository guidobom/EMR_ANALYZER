from subprocess import CompletedProcess
from unittest.mock import patch

from emr_analyzer.llm_backend.diagnostics import (
    AccelerationDiagnostic,
    diagnose_llama_acceleration,
    diagnose_local_acceleration,
)
from emr_analyzer.utils.hardware import (
    ATOMIC_EVIDENCE_CONTEXT_TARGET,
    ATOMIC_EVIDENCE_OUTPUT_TARGET,
    ATOMIC_EVIDENCE_WORKER_CAP,
    HardwareProfile,
    ModelProfile,
    RecommendedParams,
    calculate_max_workers,
    max_llm_instances,
    recommend_output_tokens,
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
    assert "backend CUDA è presente" in diagnostic.summary


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


def test_llama_diagnostic_distinguishes_loaded_metal_without_device():
    diagnostic = _probe(
        "loaded MTL backend from /opt/homebrew/libggml-metal.so\n"
        "Available devices:\n  BLAS: Accelerate (0 MiB, 0 MiB free)",
        system="Darwin",
        gpu="Apple M5 Pro",
    )

    assert diagnostic.status == "backend_unavailable"
    assert diagnostic.compiled_backends == ("METAL",)
    assert diagnostic.runtime_backend == "CPU"
    assert "ricadrebbe sulla CPU" in diagnostic.summary


def test_macos_combined_diagnostic_omits_irrelevant_vllm_warning():
    llama = AccelerationDiagnostic(
        status="accelerated",
        expected_backend="METAL",
        runtime_backend="METAL",
        compiled_backends=("METAL",),
        devices=("Metal: Apple M5 Pro",),
        binary_path="/opt/llama-server",
        system="Darwin",
        machine="arm64",
        gpu_name="Apple M5 Pro",
        summary="Accelerazione METAL disponibile: Apple M5 Pro.",
        details="Dettagli Metal",
    )
    with (
        patch(
            "emr_analyzer.llm_backend.diagnostics."
            "diagnose_llama_acceleration",
            return_value=llama,
        ),
        patch(
            "emr_analyzer.llm_backend.diagnostics."
            "diagnose_vllm_acceleration"
        ) as vllm_probe,
    ):
        diagnostic = diagnose_local_acceleration()

    vllm_probe.assert_not_called()
    assert diagnostic.status == "accelerated"
    assert "vLLM:" not in diagnostic.summary
    assert "non viene verificato su macOS" in diagnostic.details


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


def test_atomic_recommendation_uses_workload_preset_not_model_maximum():
    hardware = HardwareProfile(
        total_ram_gb=48.0,
        available_ram_gb=11.0,
        cpu_cores_physical=18,
        cpu_cores_logical=18,
        has_apple_silicon=True,
        gpu_name="Apple M5 Pro",
    )
    model = ModelProfile(
        model_name="medgemma-4b-it-q4km",
        size_gb=2.3,
        max_context_length=131_072,
        architecture="gemma3",
    )

    recommendation = RecommendedParams.compute(
        hardware, model, role="atomic_evidence"
    )

    assert recommendation.context_length == ATOMIC_EVIDENCE_CONTEXT_TARGET
    assert recommendation.max_output_tokens == ATOMIC_EVIDENCE_OUTPUT_TARGET
    assert recommendation.parallel_workers == ATOMIC_EVIDENCE_WORKER_CAP
    assert "2.800 caratteri" in recommendation.context_rationale
    assert "limite adattivo" in recommendation.output_rationale
    assert "limite prudenziale" in recommendation.workers_rationale


def test_atomic_preset_respects_a_smaller_model_context():
    hardware = HardwareProfile(
        total_ram_gb=16.0,
        available_ram_gb=8.0,
        cpu_cores_physical=8,
        cpu_cores_logical=8,
        has_apple_silicon=False,
    )
    model = ModelProfile(
        model_name="small-model",
        size_gb=2.0,
        max_context_length=4_096,
    )

    recommendation = RecommendedParams.compute(
        hardware, model, role="atomic_evidence"
    )

    assert recommendation.context_length == 4_096
    assert recommendation.max_output_tokens == 2_048
    assert "limite del modello" in recommendation.context_rationale


def test_atomic_output_guidance_matches_the_role_preset():
    output, rationale = recommend_output_tokens(
        ATOMIC_EVIDENCE_CONTEXT_TARGET, "atomic_evidence"
    )

    assert output == ATOMIC_EVIDENCE_OUTPUT_TARGET
    assert "Preset atomico" in rationale


def test_max_llm_instances_is_kv_aware_on_dgx_budget():
    # DGX Spark-like pool: ~116 GiB available, 18.6 GiB model, ctx 16384,
    # np 8 → KV = 16384 × 8 × 100 KB ≈ 13.1 GiB; per instance ≈
    # 18.6 + 13.1 + 3 + 2 ≈ 36.7 GiB → floor(116 / 36.7) = 3.
    with patch("emr_analyzer.utils.hardware.psutil.virtual_memory") as vm:
        vm.return_value.available = 116 * 1024 ** 3
        assert max_llm_instances(18.6 * 1024 ** 3, 16384, 8) == 3


def test_max_llm_instances_respects_cap():
    with patch("emr_analyzer.utils.hardware.psutil.virtual_memory") as vm:
        vm.return_value.available = 116 * 1024 ** 3
        assert (
            max_llm_instances(18.6 * 1024 ** 3, 16384, 8, cap=2) == 2
        )


def test_max_llm_instances_never_below_one():
    # A model larger than the whole available pool still yields one instance.
    with patch("emr_analyzer.utils.hardware.psutil.virtual_memory") as vm:
        vm.return_value.available = 2 * 1024 ** 3
        assert max_llm_instances(100 * 1024 ** 3, 0, 1) == 1

"""DGX Spark compatibility tests without CUDA hardware or model loading."""

from subprocess import CompletedProcess
from unittest.mock import patch

import pytest

from emr_analyzer.utils.hardware import (
    HardwareProfile,
    ModelProfile,
    RecommendedParams,
    get_system_ram_reserve_gb,
)
from emr_analyzer.utils.nvidia import probe_nvidia_gpu


def test_gb10_probe_accepts_unsupported_memory_counter():
    def fake_run(argv, **_kwargs):
        if any(str(item).startswith("--query-gpu=") for item in argv):
            return CompletedProcess(
                argv, 0,
                "NVIDIA GB10, 12.1, 580.159.03, [Not Supported]\n", "",
            )
        return CompletedProcess(
            argv, 0,
            "NVIDIA-SMI 580.159.03  Driver Version: 580.159.03  "
            "CUDA Version: 13.0\n",
            "",
        )

    with (
        patch("emr_analyzer.utils.nvidia.shutil.which", return_value="nvidia-smi"),
        patch("emr_analyzer.utils.nvidia.subprocess.run", side_effect=fake_run),
        patch(
            "emr_analyzer.utils.nvidia._product_name",
            return_value="NVIDIA DGX Spark",
        ),
    ):
        gpu = probe_nvidia_gpu(system_name="Linux", machine="aarch64")

    assert gpu.available
    assert gpu.name == "NVIDIA GB10"
    assert gpu.compute_capability == "12.1"
    assert gpu.cuda_architecture == "121"
    assert gpu.memory_total_gb is None
    assert gpu.is_dgx_spark
    assert gpu.unified_memory
    assert gpu.gpu_count == 1


def _dgx() -> HardwareProfile:
    return HardwareProfile(
        total_ram_gb=128.0,
        available_ram_gb=110.0,
        cpu_cores_physical=20,
        cpu_cores_logical=20,
        has_apple_silicon=False,
        gpu_name="NVIDIA GB10",
        system="Linux",
        machine="aarch64",
        has_nvidia_cuda=True,
        is_dgx_spark=True,
        unified_memory=True,
        cuda_compute_capability="12.1",
        cuda_version="13.0",
        nvidia_gpu_count=1,
    )


def test_dgx_sizing_uses_unified_system_ram_and_keeps_headroom():
    hardware = _dgx()
    model = ModelProfile("qwen3-4b", 2.4, 262_144, "qwen3")

    recommendation = RecommendedParams.compute(
        hardware, model, role="atomic_evidence", backend="llama_cpp"
    )

    assert get_system_ram_reserve_gb(hardware) == pytest.approx(12.8)
    assert recommendation.context_length == 8_192
    assert recommendation.parallel_workers == 8
    assert "DGX Spark" in recommendation.workers_rationale
    assert "memoria unificata" in recommendation.workers_rationale


def test_dgx_vllm_reduces_continuous_batch_for_large_checkpoints():
    small = RecommendedParams.compute(
        _dgx(), ModelProfile("small", 8.0, 65_536),
        role="atomic_evidence", backend="vllm",
    )
    large = RecommendedParams.compute(
        _dgx(), ModelProfile("large", 30.0, 65_536),
        role="atomic_evidence", backend="vllm",
    )

    assert small.parallel_workers == 8
    assert large.parallel_workers == 2

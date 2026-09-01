"""Hardware resource detection for dynamic worker pool sizing and auto-config.

Estimates how many concurrent llama-server slots can safely run given
the host's available RAM and the selected model's characteristics.
Also recommends optimal LLM parameters (context length, output tokens,
parallel workers) based on hardware profiling.
"""

from __future__ import annotations

import os
import platform
import subprocess
from dataclasses import dataclass

import psutil

# Practical limits — more than 8 concurrent LLM requests saturate most
# consumer hardware regardless of theoretical headroom.
MIN_WORKERS = 1
MAX_WORKERS = 8

# Atomic extraction never sends a full longitudinal record to the model.  The
# extractor caps source chunks at 2,800 characters and applies a request-local
# output budget before every call.  Allocating the model's maximum context per
# llama-server slot therefore wastes unified/VRAM and can reduce throughput by
# forcing memory pressure.  Keep these role-specific targets next to the
# hardware policy instead of duplicating them in the GUI.
ATOMIC_EVIDENCE_CONTEXT_TARGET = 8_192
ATOMIC_EVIDENCE_OUTPUT_TARGET = 6_144
ATOMIC_EVIDENCE_WORKER_CAP = 8

# Empirical KV-cache + working-memory overhead per token of context.
# llama-server runs the KV cache in q8_0: ~80 KB per token for a ~14B
# model; 100 KB keeps a safety margin (was 50 KB in the f16 Ollama era).
_BYTES_PER_CONTEXT_TOKEN = 100_000  # ~100 KB per token (KV cache dominant)

# RAM reserved for the OS and non-LLM application processes.
_OS_RESERVE_GB = 2.0
_DGX_SPARK_MIN_RESERVE_GB = 12.0

# Extra working memory (weights arena, CUDA graphs, sampler state) beyond the
# weights + KV cache that a single llama-server instance really occupies.
# Used by :func:`max_llm_instances` to bound how many sibling runtimes can
# share the unified memory pool of a multi-patient irAE queue.
_INSTANCE_MARGIN_BYTES = 3 * 1024 ** 3

# ---------------------------------------------------------------------------
# Hardware profile
# ---------------------------------------------------------------------------


@dataclass
class HardwareProfile:
    """Snapshot of the host hardware relevant to LLM configuration."""

    total_ram_gb: float
    available_ram_gb: float
    cpu_cores_physical: int
    cpu_cores_logical: int
    has_apple_silicon: bool
    gpu_name: str = ""
    system: str = ""
    machine: str = ""
    has_nvidia_cuda: bool = False
    is_dgx_spark: bool = False
    unified_memory: bool = False
    cuda_compute_capability: str = ""
    cuda_version: str = ""
    nvidia_driver_version: str = ""
    gpu_total_memory_gb: float | None = None
    nvidia_gpu_count: int = 0

    @classmethod
    def capture(cls) -> "HardwareProfile":
        """Collect a snapshot of the current host hardware."""
        from .nvidia import cached_nvidia_gpu

        vm = psutil.virtual_memory()
        system = platform.system()
        machine = platform.machine()
        nvidia = cached_nvidia_gpu(system, machine)
        return cls(
            total_ram_gb=vm.total / (1024**3),
            available_ram_gb=vm.available / (1024**3),
            cpu_cores_physical=psutil.cpu_count(logical=False) or 1,
            cpu_cores_logical=psutil.cpu_count(logical=True) or 1,
            has_apple_silicon=_detect_apple_silicon(),
            gpu_name=nvidia.name or _detect_gpu_name(),
            system=system,
            machine=machine,
            has_nvidia_cuda=nvidia.available,
            is_dgx_spark=nvidia.is_dgx_spark,
            unified_memory=nvidia.unified_memory,
            cuda_compute_capability=nvidia.compute_capability,
            cuda_version=nvidia.cuda_version,
            nvidia_driver_version=nvidia.driver_version,
            gpu_total_memory_gb=nvidia.memory_total_gb,
            nvidia_gpu_count=nvidia.gpu_count,
        )


@dataclass
class ModelProfile:
    """Key characteristics of a local GGUF model."""

    model_name: str
    size_gb: float | None
    max_context_length: int | None
    architecture: str = ""

    @classmethod
    def capture(
        cls, model_name: str, backend: str = "llama_cpp"
    ) -> "ModelProfile":
        """Read metadata from the selected local backend without a load."""
        if backend == "vllm":
            from ..llm_backend.vllm_backend import resolve_vllm_model

            info = resolve_vllm_model(model_name) or {}
            size_bytes = info.get("size_bytes")
            return cls(
                model_name=model_name,
                size_gb=(
                    float(size_bytes) / (1024 ** 3)
                    if size_bytes is not None else None
                ),
                max_context_length=info.get("max_context_length"),
                architecture=str(info.get("architecture") or ""),
            )

        from ..extraction.llm_client import LlmClient

        size = get_model_size_gb(model_name)
        try:
            caps = LlmClient(model=model_name).model_capabilities()
        except Exception:
            caps = {}
        return cls(
            model_name=model_name,
            size_gb=size,
            max_context_length=caps.get("max_context_length"),
            architecture=caps.get("architecture", ""),
        )


@dataclass
class RecommendedParams:
    """Optimal LLM parameters derived from hardware + model analysis."""

    context_length: int
    max_output_tokens: int
    parallel_workers: int
    # Explanations for the UI
    context_rationale: str = ""
    output_rationale: str = ""
    workers_rationale: str = ""

    @classmethod
    def compute(
        cls,
        hw: HardwareProfile,
        model: ModelProfile,
        role: str = "document",
        backend: str = "llama_cpp",
    ) -> "RecommendedParams":
        """Compute recommended parameters for *model* running on *hw*.

        *role* distinguishes document normalization, atomic extraction,
        event/episode synthesis and longitudinal analysis.
        """
        # ---- context_length ------------------------------------------------
        context, context_rationale = _recommend_context(hw, model, role)

        # ---- max_output_tokens ---------------------------------------------
        max_out, output_rationale = _recommend_output(
            context, model, role
        )

        # ---- parallel_workers ----------------------------------------------
        workers, workers_rationale = _recommend_workers(
            hw, model, context, role, backend
        )

        return cls(
            context_length=context,
            max_output_tokens=max_out,
            parallel_workers=workers,
            context_rationale=context_rationale,
            output_rationale=output_rationale,
            workers_rationale=workers_rationale,
        )


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def get_available_ram_gb() -> float:
    """RAM currently available (free + reclaimable) in GiB."""
    return psutil.virtual_memory().available / (1024 ** 3)


def get_total_ram_gb() -> float:
    """Physical/unified RAM capacity in GiB."""
    return psutil.virtual_memory().total / (1024 ** 3)


def get_model_size_gb(model_name: str) -> float | None:
    """Return the on-disk size (GiB) of a local GGUF model, or *None*."""
    try:
        from ..llm_backend import model_store

        entry = model_store.resolve(model_name)
        if entry is None or not entry.get("file"):
            return None
        return os.path.getsize(entry["file"]) / (1024 ** 3)
    except OSError:
        return None


def estimate_per_request_ram_gb(context_length: int) -> float:
    """Estimate additional RAM (GiB) consumed by one concurrent request."""
    return (context_length * _BYTES_PER_CONTEXT_TOKEN) / (1024 ** 3)


def estimate_server_ram_gb(
    model_name: str, context_length: int, workers: int
) -> float:
    """Conservative machine requirement for one llama-server runtime."""
    return (
        estimate_runtime_ram_gb(model_name, context_length, workers)
        + get_system_ram_reserve_gb()
    )


def estimate_runtime_ram_gb(
    model_name: str, context_length: int, workers: int
) -> float:
    """Estimated weights + KV cache for one physical runtime, in GiB."""
    model_size = get_model_size_gb(model_name) or 4.0
    return (
        model_size
        + estimate_per_request_ram_gb(context_length) * max(1, int(workers))
    )


def get_system_ram_reserve_gb(
    profile: HardwareProfile | None = None,
) -> float:
    """RAM excluded from LLM sizing for the OS and the application."""
    try:
        hw = profile or HardwareProfile.capture()
    except Exception:
        return _OS_RESERVE_GB
    if hw.is_dgx_spark:
        # GB10 shares the same 128 GB pool among OS, CUDA, weights and KV.
        # A percentage prevents the optimizer from consuming the memory that
        # the desktop, application and CUDA graphs still need.
        return max(_DGX_SPARK_MIN_RESERVE_GB, hw.total_ram_gb * 0.10)
    return _OS_RESERVE_GB


def calculate_max_workers(
    model_name: str,
    context_length: int,
) -> int:
    """Safe number of concurrent llama-server slots for *model_name*.

    Returns a value clamped to ``[MIN_WORKERS, MAX_WORKERS]``.
    """
    # Size a cold runtime against machine capacity.  ``available`` RAM is
    # misleading while this same model is resident because subtracting its
    # weights again double-counts memory already in use.
    capacity = get_total_ram_gb()
    model_size = get_model_size_gb(model_name)

    if model_size is None:
        # Unknown model — conservative fallback
        model_size = 4.0

    per_request = estimate_per_request_ram_gb(context_length)
    if per_request <= 0:
        return MAX_WORKERS

    headroom = capacity - model_size - get_system_ram_reserve_gb()
    if headroom <= 0:
        return MIN_WORKERS

    max_theoretical = int(headroom / per_request)
    return max(MIN_WORKERS, min(MAX_WORKERS, max_theoretical))


def get_worker_options(
    model_name: str,
    context_length: int,
) -> list[int]:
    """Return the list of valid worker counts for a UI dropdown.

    Values above the safe maximum are still included (greyed out in the UI)
    so the user sees the full range.
    """
    return list(range(MIN_WORKERS, MAX_WORKERS + 1))


def get_safe_max_workers(
    model_name: str,
    context_length: int,
) -> int:
    """The highest worker count considered safe."""
    return calculate_max_workers(model_name, context_length)


def max_llm_instances(
    model_size_bytes: int,
    context_length: int = 0,
    workers: int = 1,
    *,
    cap: int = 0,
) -> int:
    """How many independent llama-server instances can share the memory pool.

    Multi-patient irAE queues run one server per patient.  Each instance
    reserves the weights plus its KV cache (``context_length * workers``
    tokens at ~100 KB each — the per-slot context is split across the
    ``workers`` parallel slots, so the total KV for one process scales with
    the product), plus a working-memory margin and a share of the OS
    reserve.  Uses the *currently available* system RAM (``psutil``, which
    on the GB10 unified-memory DGX also reflects what the GPU pool has),
    never Linux-specific ``nvidia-smi``.
    """
    available = psutil.virtual_memory().available
    kv = (
        max(0, int(context_length))
        * max(1, int(workers))
        * _BYTES_PER_CONTEXT_TOKEN
    )
    per = (
        int(model_size_bytes)
        + kv
        + _INSTANCE_MARGIN_BYTES
        + 2 * 1024 ** 3
    )
    if per <= 0:
        return 1
    count = max(1, available // per)
    if cap > 0:
        count = min(count, int(cap))
    return count


# ---------------------------------------------------------------------------
# Recommendation helpers
# ---------------------------------------------------------------------------


def recommend_all(
    model_name: str,
    role: str = "document",
    backend: str = "llama_cpp",
) -> RecommendedParams | None:
    """One-shot: profile hardware + model and return recommended parameters.

    Returns ``None`` when the model is not installed or hardware probing fails.
    """
    try:
        hw = HardwareProfile.capture()
        model = ModelProfile.capture(model_name, backend=backend)
    except Exception:
        return None
    if model.max_context_length is None:
        return None
    return RecommendedParams.compute(hw, model, role, backend)


def recommend_output_tokens(context_length: int, role: str) -> tuple[int, str]:
    """Public request-scoped output recommendation for an existing runtime."""
    return _recommend_output(
        int(context_length),
        ModelProfile("runtime", None, int(context_length)),
        role,
    )


# ---------------------------------------------------------------------------
# Internal
# ---------------------------------------------------------------------------


def _detect_apple_silicon() -> bool:
    """Return True when running on an Apple Silicon Mac."""
    return platform.machine() in ("arm64", "aarch64") and platform.system() == "Darwin"


def _detect_gpu_name() -> str:
    """Best-effort GPU name on macOS or Linux."""
    try:
        if platform.system() == "Darwin":
            result = subprocess.run(
                ["system_profiler", "SPDisplaysDataType"],
                capture_output=True, text=True, timeout=5,
            )
            for line in result.stdout.splitlines():
                stripped = line.strip()
                if stripped.startswith("Chipset Model:"):
                    return stripped.split(":", 1)[1].strip()
        else:
            result = subprocess.run(
                ["lspci"], capture_output=True, text=True, timeout=5,
            )
            for line in result.stdout.splitlines():
                if "VGA" in line or "3D" in line or "Display" in line:
                    return line.strip()
    except Exception:
        pass
    return ""


def _recommend_context(
    hw: HardwareProfile, model: ModelProfile, role: str = "document"
) -> tuple[int, str]:
    """Recommend a context length based on hardware and model limits."""
    model_max = model.max_context_length or 131_072  # Conservative fallback
    model_size = model.size_gb or 4.0

    if role == "atomic_evidence":
        chosen = min(model_max, ATOMIC_EVIDENCE_CONTEXT_TARGET)
        if chosen >= ATOMIC_EVIDENCE_CONTEXT_TARGET:
            rationale = (
                "Preset per chunk clinici ≤2.800 caratteri: il contesto "
                f"massimo del modello ({_fmt_tokens(model_max)}) non viene "
                "preallocato inutilmente per ogni slot"
            )
        else:
            rationale = (
                "Preset atomico ridotto al limite del modello: "
                f"{_fmt_tokens(model_max)} token; i passaggi densi saranno "
                "suddivisi automaticamente"
            )
        return max(512, chosen), rationale

    # Configuration describes a cold server. Use total capacity rather than
    # a volatile post-load snapshot that already excludes resident weights.
    available_for_llm = hw.total_ram_gb - get_system_ram_reserve_gb(hw)
    kv_per_token_gb = _BYTES_PER_CONTEXT_TOKEN / (1024**3)
    ram_budget_tokens = max(
        2048,
        int((available_for_llm - model_size) / kv_per_token_gb),
    )

    # Pick the largest round value that fits both constraints.
    candidates = [
        c for c in sorted({
            model_max, 131_072, 65_536, 49_152, 32_768, 16_384, 8_192,
        }, reverse=True)
        if c <= model_max and c <= ram_budget_tokens
    ]
    chosen = candidates[0] if candidates else 8_192

    parts = []
    if chosen == model_max:
        parts.append(f"Limite del modello: {_fmt_tokens(model_max)}")
    else:
        parts.append(
            f"Ridotto da {_fmt_tokens(model_max)} (limite modello) "
            f"a {_fmt_tokens(chosen)} per rispettare la RAM"
        )
    parts.append(
        f"RAM totale: {hw.total_ram_gb:.1f} GiB "
        f"(modello ~{model_size:.1f} GiB)"
    )
    return chosen, "; ".join(parts)


def _recommend_output(
    context: int, model: ModelProfile, role: str
) -> tuple[int, str]:
    """Recommend max_output_tokens.

    The document model filters full text → needs larger output.
    Clinical State also produces long narrative/irAE reports.
    """
    if role == "atomic_evidence":
        # The extractor applies a tighter per-chunk cap derived from source
        # density (currently at most ~4.5K for a full 2,800-character chunk).
        # 6K here is headroom, not a request to generate 6K tokens every time.
        chosen = (
            ATOMIC_EVIDENCE_OUTPUT_TARGET
            if context >= ATOMIC_EVIDENCE_CONTEXT_TARGET
            else max(1_024, context // 2)
        )
        return chosen, (
            "Preset atomico: tetto di sicurezza; ogni chunk usa un limite "
            "adattivo inferiore e viene suddiviso se la risposta è troppo densa"
        )

    # Base ratio: output gets a fraction of the context.
    if role == "document":
        # Document filtering can produce output up to ~50 % of source
        ratio = 0.30
        workload = "Filtraggio testo"
    elif role == "clinical_events":
        ratio = 0.22
        workload = "Fusione eventi/episodi"
    else:
        # Full-registry and irAE reports need the most response headroom.
        ratio = 0.25
        workload = "Analisi clinica longitudinale"

    recommended = max(4096, int(context * ratio))
    # Round to the nearest nice boundary
    nice = [4096, 6144, 8192, 10_240, 12_288, 16_384, 20_480, 24_576, 32_768]
    chosen = min(nice, key=lambda n: abs(n - recommended))
    chosen = max(4096, chosen)

    rationale = (
        f"{workload}: "
        f"{int(ratio * 100)}% del contesto di {_fmt_tokens(context)} "
        f"→ {_fmt_tokens(chosen)} token"
    )
    return chosen, rationale


def _recommend_workers(
    hw: HardwareProfile,
    model: ModelProfile,
    context: int,
    role: str = "document",
    backend: str = "llama_cpp",
) -> tuple[int, str]:
    """Recommend parallel workers based on RAM headroom."""
    capacity = hw.total_ram_gb
    model_size = model.size_gb or 4.0
    per_request = estimate_per_request_ram_gb(context)
    reserve = get_system_ram_reserve_gb(hw)
    headroom = capacity - model_size - reserve

    if headroom <= 0:
        workers = 1
        rationale = (
            f"Capacità RAM insufficiente ({capacity:.1f} GiB totali, "
            f"modello {model_size:.1f} GiB) — 1 slot forzato"
        )
    else:
        theoretical = int(headroom / per_request) if per_request > 0 else MAX_WORKERS
        accelerated = hw.has_apple_silicon or hw.has_nvidia_cuda
        if backend == "vllm" and hw.is_dgx_spark:
            # vLLM continuously batches sequences. On one GB10, small models
            # can use all eight application tasks, whereas large models must
            # keep the batch deliberately small to protect the unified pool.
            performance_cap = (
                8 if model_size < 10 else
                4 if model_size < 24 else
                2 if model_size < 55 else 1
            )
        elif accelerated:
            performance_cap = (
                8 if model_size < 4 else
                6 if model_size < 8 else
                3 if model_size < 24 else 2
            )
        else:
            performance_cap = (
                4 if model_size < 4 else
                3 if model_size < 8 else 2
            )
        if role == "atomic_evidence":
            performance_cap = min(performance_cap, ATOMIC_EVIDENCE_WORKER_CAP)
        workers = max(
            MIN_WORKERS,
            min(MAX_WORKERS, theoretical, performance_cap),
        )
        hw_desc = (
            f"DGX Spark ({hw.gpu_name or 'NVIDIA GB10'}, memoria unificata)"
            if hw.is_dgx_spark
            else f"NVIDIA CUDA ({hw.gpu_name})"
            if hw.has_nvidia_cuda and hw.gpu_name
            else "NVIDIA CUDA"
            if hw.has_nvidia_cuda
            else
            f"Apple Silicon ({hw.gpu_name})"
            if hw.has_apple_silicon and hw.gpu_name
            else "Apple Silicon"
            if hw.has_apple_silicon
            else f"{hw.cpu_cores_physical} core CPU"
        )
        est_total = model_size + (per_request * workers) + reserve
        rationale = (
            f"{hw_desc} — "
            f"RAM: {capacity:.1f} GiB totali, "
            f"~{est_total:.1f} GiB stimati per {workers} slot "
            f"(modello {model_size:.1f} GiB + "
            f"{per_request * workers:.1f} GiB KV-cache)"
        )
        if role == "atomic_evidence":
            rationale += (
                f"; limite prudenziale controllato di "
                f"{ATOMIC_EVIDENCE_WORKER_CAP} slot "
                "per i chunk atomici; il benchmark locale può confermare "
                "4/6/8 sulla macchina"
            )
    return workers, rationale


def _fmt_tokens(value: int) -> str:
    """Format a token count for display (e.g. 32768 → '32.768')."""
    return f"{value:,}".replace(",", ".")

"""Hardware resource detection for dynamic worker pool sizing and auto-config.

Estimates how many concurrent Ollama requests can safely run given
the host's available RAM and the selected model's characteristics.
Also recommends optimal LLM parameters (context length, output tokens,
parallel workers) based on hardware profiling.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from typing import Optional

import psutil

# Practical limits — more than 8 concurrent LLM requests saturate most
# consumer hardware regardless of theoretical headroom.
MIN_WORKERS = 1
MAX_WORKERS = 8

# Empirical KV-cache + working-memory overhead per token of context.
# KV cache ≈ 2 * layers * kv_heads * head_dim * bytes_per_element
# For a typical 8B model (Q4_K_M): ~50 KB per context token.
_BYTES_PER_CONTEXT_TOKEN = 50_000  # ~50 KB per token (KV cache dominant)

# RAM reserved for the OS and non-LLM application processes.
_OS_RESERVE_GB = 2.0

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

    @classmethod
    def capture(cls) -> "HardwareProfile":
        """Collect a snapshot of the current host hardware."""
        vm = psutil.virtual_memory()
        return cls(
            total_ram_gb=vm.total / (1024**3),
            available_ram_gb=vm.available / (1024**3),
            cpu_cores_physical=psutil.cpu_count(logical=False) or 1,
            cpu_cores_logical=psutil.cpu_count(logical=True) or 1,
            has_apple_silicon=_detect_apple_silicon(),
            gpu_name=_detect_gpu_name(),
        )


@dataclass
class ModelProfile:
    """Key characteristics of a local Ollama model."""

    model_name: str
    size_gb: float | None
    max_context_length: int | None
    architecture: str = ""

    @classmethod
    def capture(cls, model_name: str) -> "ModelProfile":
        """Read model metadata from Ollama."""
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
    ) -> "RecommendedParams":
        """Compute recommended parameters for *model* running on *hw*.

        *role* is ``"document"`` (needs larger output for text filtering) or
        ``"clinical_state"`` (structured output is more compact).
        """
        # ---- context_length ------------------------------------------------
        context, context_rationale = _recommend_context(hw, model)

        # ---- max_output_tokens ---------------------------------------------
        max_out, output_rationale = _recommend_output(
            context, model, role
        )

        # ---- parallel_workers ----------------------------------------------
        workers, workers_rationale = _recommend_workers(
            hw, model, context
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


def get_model_size_gb(model_name: str) -> float | None:
    """Return the on-disk size (GiB) of an Ollama model, or *None*.

    Tries ``ollama list`` first (fast, has size column), then falls back
    to parameter-count estimation from ``ollama show``.
    """
    try:
        # Fast path: ollama list has a SIZE column
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            for line in result.stdout.splitlines():
                # Lines look like: qwen3:8b    500a1f067a9f    5.2 GB    ...
                if line.startswith(model_name):
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if i >= 2 and part.upper() in ("GB", "MB", "TB", "KB"):
                            size_str = parts[i - 1] + " " + parts[i]
                            parsed = _parse_size(size_str)
                            if parsed is not None:
                                return parsed
        # Fallback: estimate from parameter count in ollama show
        result2 = subprocess.run(
            ["ollama", "show", model_name],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result2.returncode == 0:
            for line in result2.stdout.splitlines():
                if "parameters" in line.lower():
                    parts = line.strip().split()
                    for i, part in enumerate(parts):
                        if "parameter" in part.lower() and i + 1 < len(parts):
                            param_str = parts[i + 1].rstrip("B").upper()
                            return _estimate_params_size(param_str)
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    return None


def estimate_per_request_ram_gb(context_length: int) -> float:
    """Estimate additional RAM (GiB) consumed by one concurrent request."""
    return (context_length * _BYTES_PER_CONTEXT_TOKEN) / (1024 ** 3)


def calculate_max_workers(
    model_name: str,
    context_length: int,
) -> int:
    """Safe number of concurrent Ollama requests for *model_name*.

    Returns a value clamped to ``[MIN_WORKERS, MAX_WORKERS]``.
    """
    available = get_available_ram_gb()
    model_size = get_model_size_gb(model_name)

    if model_size is None:
        # Unknown model — conservative fallback
        model_size = 4.0

    per_request = estimate_per_request_ram_gb(context_length)
    if per_request <= 0:
        return MAX_WORKERS

    headroom = available - model_size - _OS_RESERVE_GB
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


# ---------------------------------------------------------------------------
# Recommendation helpers
# ---------------------------------------------------------------------------


def recommend_all(
    model_name: str,
    role: str = "document",
) -> RecommendedParams | None:
    """One-shot: profile hardware + model and return recommended parameters.

    Returns ``None`` when the model is not installed or hardware probing fails.
    """
    try:
        hw = HardwareProfile.capture()
        model = ModelProfile.capture(model_name)
    except Exception:
        return None
    if model.max_context_length is None:
        return None
    return RecommendedParams.compute(hw, model, role)


# ---------------------------------------------------------------------------
# Internal
# ---------------------------------------------------------------------------


def _detect_apple_silicon() -> bool:
    """Return True when running on Apple Silicon (M1/M2/M3/M4)."""
    import platform
    return platform.machine() in ("arm64", "aarch64") and platform.system() == "Darwin"


def _detect_gpu_name() -> str:
    """Best-effort GPU name on macOS or Linux."""
    try:
        import platform
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
    hw: HardwareProfile, model: ModelProfile
) -> tuple[int, str]:
    """Recommend a context length based on hardware and model limits."""
    model_max = model.max_context_length or 131_072  # Conservative fallback
    model_size = model.size_gb or 4.0

    # Estimate how much context fits in available RAM
    available_for_llm = hw.available_ram_gb - _OS_RESERVE_GB
    kv_per_token_gb = _BYTES_PER_CONTEXT_TOKEN / (1024**3)
    ram_budget_tokens = max(
        2048,
        int((available_for_llm - model_size) / kv_per_token_gb),
    )

    # Pick the largest round value that fits both constraints.
    candidates = [
        c for c in (131_072, 65_536, 49_152, 32_768, 16_384, 8_192)
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
        f"RAM disponibile: {hw.available_ram_gb:.1f} GB "
        f"(modello ~{model_size:.1f} GB)"
    )
    return chosen, "; ".join(parts)


def _recommend_output(
    context: int, model: ModelProfile, role: str
) -> tuple[int, str]:
    """Recommend max_output_tokens.

    The document model filters full text → needs larger output.
    The clinical-state model produces structured JSON → compact output.
    """
    # Base ratio: output gets a fraction of the context.
    if role == "document":
        # Document filtering can produce output up to ~50 % of source
        ratio = 0.30
    else:
        # Structured JSON is compact
        ratio = 0.20

    recommended = max(4096, int(context * ratio))
    # Round to the nearest nice boundary
    nice = [4096, 6144, 8192, 10_240, 12_288, 16_384, 20_480, 24_576, 32_768]
    chosen = min(nice, key=lambda n: abs(n - recommended))
    chosen = max(4096, chosen)

    rationale = (
        f"{'Filtraggio testo' if role == 'document' else 'Output strutturato'}: "
        f"{int(ratio * 100)}% del contesto di {_fmt_tokens(context)} "
        f"→ {_fmt_tokens(chosen)} token"
    )
    return chosen, rationale


def _recommend_workers(
    hw: HardwareProfile, model: ModelProfile, context: int
) -> tuple[int, str]:
    """Recommend parallel workers based on RAM headroom."""
    available = hw.available_ram_gb
    model_size = model.size_gb or 4.0
    per_request = estimate_per_request_ram_gb(context)
    headroom = available - model_size - _OS_RESERVE_GB

    if headroom <= 0:
        workers = 1
        rationale = (
            f"RAM insufficiente ({available:.1f} GB disp., "
            f"modello {model_size:.1f} GB) — 1 worker forzato"
        )
    else:
        theoretical = int(headroom / per_request) if per_request > 0 else MAX_WORKERS
        workers = max(MIN_WORKERS, min(MAX_WORKERS, theoretical))
        hw_desc = f"Apple Silicon ({hw.gpu_name})" if hw.has_apple_silicon else f"{hw.cpu_cores_physical} core CPU"
        est_total = model_size + (per_request * workers) + _OS_RESERVE_GB
        rationale = (
            f"{hw_desc} — "
            f"RAM: {available:.1f} GB disp., "
            f"~{est_total:.1f} GB stimati per {workers} worker "
            f"(modello {model_size:.1f} GB + "
            f"{per_request * workers:.1f} GB KV-cache)"
        )
    return workers, rationale


def _fmt_tokens(value: int) -> str:
    """Format a token count for display (e.g. 32768 → '32.768')."""
    return f"{value:,}".replace(",", ".")


def _parse_size(raw: str) -> float | None:
    """Parse a human-readable size string like ``5.2 GB`` or ``986 MB``."""
    raw = raw.strip().upper()
    multipliers = {"B": 1, "KB": 1e3, "MB": 1e6, "GB": 1e9, "TB": 1e12}
    for unit, mult in sorted(multipliers.items(), key=lambda x: -len(x[0])):
        if raw.endswith(unit):
            try:
                value = float(raw[: -len(unit)].strip())
                return value * mult / 1e9  # → GiB
            except ValueError:
                return None
    # Try bare number (bytes)
    try:
        return float(raw) / 1e9
    except ValueError:
        return None


def _estimate_params_size(param_str: str) -> float | None:
    """Estimate model size (GiB) from a parameter-count string like ``8.2B``.

    Assumes Q4_K_M quantization (~0.5 bytes per parameter).
    """
    try:
        raw = param_str.strip().upper()
        multiplier = 1.0
        if raw.endswith("B"):
            multiplier = 1.0
            raw = raw[:-1]
        elif raw.endswith("M"):
            multiplier = 0.001
            raw = raw[:-1]
        params = float(raw) * multiplier  # billions
        return params * 0.6  # Q4_K_M ~0.6 GB per billion params
    except ValueError:
        return None

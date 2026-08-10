"""Hardware resource detection for dynamic worker pool sizing.

Estimates how many concurrent Ollama requests can safely run given
the host's available RAM and the selected model's characteristics.
"""

from __future__ import annotations

import json
import subprocess

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
# Public helpers
# ---------------------------------------------------------------------------


def get_total_ram_gb() -> float:
    """Total physical RAM in GiB."""
    return psutil.virtual_memory().total / (1024 ** 3)


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
# Internal
# ---------------------------------------------------------------------------


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

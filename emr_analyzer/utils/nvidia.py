"""Read-only NVIDIA/DGX hardware probing.

DGX Spark is deliberately handled as a special case: its GB10 GPU shares
the machine's coherent memory and ``nvidia-smi`` may report GPU memory as
``N/A``/``Not Supported``.  Device availability must therefore never depend
on the discrete-VRAM fields used by conventional NVIDIA workstations.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import platform
import re
import shutil
import subprocess
from pathlib import Path


@dataclass(frozen=True)
class NvidiaGpuInfo:
    """Best-effort NVIDIA accelerator metadata without allocating memory."""

    available: bool = False
    name: str = ""
    compute_capability: str = ""
    driver_version: str = ""
    cuda_version: str = ""
    memory_total_gb: float | None = None
    product_name: str = ""
    is_dgx_spark: bool = False
    unified_memory: bool = False
    raw_output: str = ""
    gpu_count: int = 0

    @property
    def cuda_architecture(self) -> str:
        """Return the CMake/CUDA numeric architecture (for example ``121``)."""
        digits = re.sub(r"[^0-9]", "", self.compute_capability)
        return digits


def probe_nvidia_gpu(
    *,
    system_name: str | None = None,
    machine: str | None = None,
    timeout: float = 8.0,
) -> NvidiaGpuInfo:
    """Probe the first NVIDIA GPU while tolerating DGX unified-memory gaps."""

    system = str(system_name or platform.system() or "")
    architecture = _normal_machine(machine or platform.machine())
    product = _product_name() if system.casefold() == "linux" else ""
    command = shutil.which("nvidia-smi")
    if system.casefold() != "linux" or not command:
        return NvidiaGpuInfo(product_name=product)

    query_output = ""
    name = compute = driver = ""
    gpu_count = 0
    memory_gb: float | None = None
    try:
        result = subprocess.run(
            [
                command,
                "--query-gpu=name,compute_cap,driver_version,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=max(1.0, float(timeout)),
        )
        query_output = "\n".join(
            part.strip() for part in (result.stdout, result.stderr) if part
        ).strip()
        if result.returncode == 0 and result.stdout.strip():
            gpu_count = len([
                line for line in result.stdout.splitlines() if line.strip()
            ])
            fields = [
                item.strip() for item in result.stdout.splitlines()[0].split(",")
            ]
            if fields:
                name = fields[0]
            if len(fields) > 1 and _is_value(fields[1]):
                compute = _normal_compute_capability(fields[1])
            if len(fields) > 2 and _is_value(fields[2]):
                driver = fields[2]
            if len(fields) > 3:
                memory_gb = _memory_gib(fields[3])
    except (OSError, subprocess.TimeoutExpired):
        pass

    # Some driver versions do not support a combined ``compute_cap`` query.
    # A name-only query is kept as a compatibility fallback and, importantly,
    # does not ask for the unsupported memory counter on DGX Spark.
    if not name:
        try:
            result = subprocess.run(
                [command, "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=max(1.0, float(timeout)),
            )
            if result.returncode == 0 and result.stdout.strip():
                gpu_count = len([
                    line for line in result.stdout.splitlines() if line.strip()
                ])
                name = result.stdout.strip().splitlines()[0].strip()
                query_output = query_output or result.stdout.strip()
        except (OSError, subprocess.TimeoutExpired):
            pass

    full_output = ""
    try:
        result = subprocess.run(
            [command], capture_output=True, text=True,
            timeout=max(1.0, float(timeout)),
        )
        full_output = "\n".join(
            part.strip() for part in (result.stdout, result.stderr) if part
        ).strip()
    except (OSError, subprocess.TimeoutExpired):
        pass
    if not driver:
        match = re.search(r"Driver Version:\s*([^\s|]+)", full_output, re.I)
        if match:
            driver = match.group(1)
    match = re.search(r"CUDA Version:\s*([^\s|]+)", full_output, re.I)
    cuda = match.group(1) if match else ""

    dgx = _is_dgx_spark(name, product, architecture)
    # GB10 is compute capability 12.1.  Older nvidia-smi builds can expose
    # the device but omit ``compute_cap``; keep setup guidance deterministic.
    if dgx and not compute:
        compute = "12.1"
    return NvidiaGpuInfo(
        available=bool(name),
        name=name,
        compute_capability=compute,
        driver_version=driver,
        cuda_version=cuda,
        memory_total_gb=memory_gb,
        product_name=product,
        is_dgx_spark=dgx,
        unified_memory=dgx,
        raw_output=(query_output + "\n" + full_output).strip()[-6000:],
        gpu_count=gpu_count or int(bool(name)),
    )


@lru_cache(maxsize=8)
def cached_nvidia_gpu(
    system_name: str | None = None,
    machine: str | None = None,
) -> NvidiaGpuInfo:
    """Cached host identity for frequent UI sizing refreshes."""
    return probe_nvidia_gpu(system_name=system_name, machine=machine)


def _normal_machine(value: str) -> str:
    clean = str(value or "").strip().casefold()
    return "aarch64" if clean in {"arm64", "aarch64"} else clean


def _normal_compute_capability(value: str) -> str:
    clean = str(value or "").strip().casefold().removeprefix("sm_")
    if re.fullmatch(r"\d{2,3}", clean):
        return f"{clean[:-1]}.{clean[-1]}"
    return clean if re.fullmatch(r"\d+\.\d+", clean) else ""


def _memory_gib(value: str) -> float | None:
    if not _is_value(value):
        return None
    match = re.search(r"([0-9]+(?:\.[0-9]+)?)", value)
    if not match:
        return None
    # ``nounits`` returns MiB for memory.total.
    return float(match.group(1)) / 1024.0


def _is_value(value: str) -> bool:
    clean = str(value or "").strip().casefold()
    return clean not in {"", "n/a", "na", "not supported", "[not supported]"}


def _is_dgx_spark(name: str, product: str, machine: str) -> bool:
    combined = f"{name} {product}".casefold()
    return machine == "aarch64" and (
        "dgx spark" in combined or "gb10" in combined
    )


def _product_name() -> str:
    for candidate in (
        Path("/sys/class/dmi/id/product_name"),
        Path("/sys/devices/virtual/dmi/id/product_name"),
    ):
        try:
            value = candidate.read_text(encoding="utf-8").strip()
        except OSError:
            continue
        if value:
            return value
    return ""

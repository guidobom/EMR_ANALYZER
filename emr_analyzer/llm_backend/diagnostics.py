"""Non-invasive diagnostics for llama.cpp hardware acceleration.

The operating system seeing a GPU does not prove that the ``llama-server``
binary can use it.  This module asks the exact binary managed by the
application to enumerate its devices, without loading a model or touching
already running server processes.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
import platform
import re
import shutil
import subprocess

from .server_manager import find_server_binary
from .vllm_server_manager import find_vllm_binary


@dataclass(frozen=True)
class AccelerationDiagnostic:
    """Result of a CUDA/Metal capability and device probe."""

    status: str
    expected_backend: str
    runtime_backend: str
    compiled_backends: tuple[str, ...]
    devices: tuple[str, ...]
    binary_path: str
    system: str
    machine: str
    gpu_name: str
    summary: str
    details: str
    raw_output: str = ""
    is_dgx_spark: bool = False
    unified_memory: bool = False
    compute_capability: str = ""
    cuda_version: str = ""
    driver_version: str = ""

    @property
    def accelerated(self) -> bool:
        return self.status == "accelerated"


def diagnose_llama_acceleration(
    binary: str | None = None,
    *,
    system_name: str | None = None,
    machine: str | None = None,
    gpu_name: str | None = None,
    timeout: float = 15.0,
) -> AccelerationDiagnostic:
    """Verify acceleration exposed by the installed ``llama-server``.

    ``--list-devices`` initializes the compiled llama.cpp backends and exits;
    it neither loads a GGUF nor connects to or stops existing servers.  The
    optional host arguments make the classification deterministic in tests.
    """

    system = str(system_name or platform.system() or "Sconosciuto")
    architecture = str(machine or platform.machine() or "sconosciuta")
    detected_gpu = _host_gpu_name() if gpu_name is None else str(gpu_name)
    expected = _expected_backend(system, detected_gpu)
    resolved = binary or find_server_binary()
    if not resolved:
        return _result(
            status="binary_missing",
            expected=expected,
            runtime="SCONOSCIUTO",
            compiled=(),
            devices=(),
            binary="",
            system=system,
            machine=architecture,
            gpu_name=detected_gpu,
            summary="llama-server non trovato: accelerazione non verificabile.",
        )

    try:
        completed = subprocess.run(
            # Dynamic Homebrew backends are only named in verbose output.
            # Without ``-v`` a loaded-but-unusable Metal/CUDA plugin is
            # indistinguishable from a binary compiled without that backend.
            [resolved, "-v", "--list-devices"],
            capture_output=True,
            text=True,
            timeout=max(1.0, float(timeout)),
            env={**os.environ, "NO_COLOR": "1"},
        )
    except subprocess.TimeoutExpired:
        return _result(
            status="error",
            expected=expected,
            runtime="SCONOSCIUTO",
            compiled=(),
            devices=(),
            binary=resolved,
            system=system,
            machine=architecture,
            gpu_name=detected_gpu,
            summary="Diagnostica accelerazione scaduta senza risposta.",
        )
    except OSError as exc:
        return _result(
            status="error",
            expected=expected,
            runtime="SCONOSCIUTO",
            compiled=(),
            devices=(),
            binary=resolved,
            system=system,
            machine=architecture,
            gpu_name=detected_gpu,
            summary=f"Impossibile eseguire llama-server: {exc}",
        )

    output = "\n".join(
        part.strip() for part in (completed.stdout, completed.stderr) if part
    ).strip()
    lowered = output.casefold()
    compiled = _compiled_backends(output)
    devices = _parse_devices(output)
    cuda_active = _cuda_available(output, devices)
    metal_active = _metal_available(output, devices)

    if cuda_active or metal_active:
        runtime = "CUDA" if cuda_active else "METAL"
        device_text = "; ".join(devices) if devices else "dispositivo rilevato"
        return _result(
            status="accelerated",
            expected=expected,
            runtime=runtime,
            compiled=compiled,
            devices=devices,
            binary=resolved,
            system=system,
            machine=architecture,
            gpu_name=detected_gpu,
            summary=f"Accelerazione {runtime} disponibile: {device_text}.",
            raw_output=output,
        )

    unsupported = any(fragment in lowered for fragment in (
        "unknown argument", "unknown option", "unrecognized option",
        "invalid argument: --list-devices",
    ))
    if unsupported:
        return _result(
            status="probe_unsupported",
            expected=expected,
            runtime="SCONOSCIUTO",
            compiled=compiled,
            devices=devices,
            binary=resolved,
            system=system,
            machine=architecture,
            gpu_name=detected_gpu,
            summary=(
                "Questa versione di llama-server non supporta la diagnostica "
                "dei dispositivi; aggiornare llama.cpp."
            ),
            raw_output=output,
        )

    if expected in {"CUDA", "METAL"}:
        if expected in compiled:
            summary = (
                f"Il backend {expected} è presente, ma non espone alcun "
                "dispositivo utilizzabile a llama-server; l'inferenza "
                "ricadrebbe sulla CPU."
            )
            status = "backend_unavailable"
        else:
            summary = (
                f"La macchina richiede {expected}, ma il llama-server "
                f"selezionato non risulta compilato con {expected}."
            )
            status = "backend_missing"
        return _result(
            status=status,
            expected=expected,
            runtime="CPU",
            compiled=compiled,
            devices=devices,
            binary=resolved,
            system=system,
            machine=architecture,
            gpu_name=detected_gpu,
            summary=summary,
            raw_output=output,
        )

    if completed.returncode != 0:
        return _result(
            status="error",
            expected=expected,
            runtime="SCONOSCIUTO",
            compiled=compiled,
            devices=devices,
            binary=resolved,
            system=system,
            machine=architecture,
            gpu_name=detected_gpu,
            summary=(
                "llama-server non ha completato la diagnostica "
                f"(codice {completed.returncode})."
            ),
            raw_output=output,
        )

    return _result(
        status="cpu_only",
        expected=expected,
        runtime="CPU",
        compiled=compiled,
        devices=devices,
        binary=resolved,
        system=system,
        machine=architecture,
        gpu_name=detected_gpu,
        summary="Nessun acceleratore CUDA/Metal rilevato; esecuzione su CPU.",
        raw_output=output,
    )


def diagnose_vllm_acceleration(
    binary: str | None = None,
    *,
    system_name: str | None = None,
    machine: str | None = None,
    gpu_name: str | None = None,
    timeout: float = 20.0,
) -> AccelerationDiagnostic:
    """Verify the optional vLLM CLI and an NVIDIA CUDA device.

    This runs only ``vllm --version`` and ``nvidia-smi``; it does not import a
    model, allocate VRAM, access the network, or touch existing servers.
    """
    system = str(system_name or platform.system() or "Sconosciuto")
    architecture = str(machine or platform.machine() or "sconosciuta")
    detected_gpu = _host_gpu_name() if gpu_name is None else str(gpu_name)
    resolved = binary or find_vllm_binary()
    if system.casefold() != "linux":
        return _result(
            status="backend_unavailable",
            expected="CUDA",
            runtime="NON APPLICABILE",
            compiled=(),
            devices=(),
            binary=resolved or "",
            system=system,
            machine=architecture,
            gpu_name=detected_gpu,
            summary="vLLM non è disponibile su macOS; usa llama.cpp/Metal.",
        )
    if not resolved:
        return _result(
            status="binary_missing",
            expected="CUDA",
            runtime="SCONOSCIUTO",
            compiled=(),
            devices=(),
            binary="",
            system=system,
            machine=architecture,
            gpu_name=detected_gpu,
            summary="vLLM non installato nell'ambiente Python corrente.",
        )
    try:
        version = subprocess.run(
            [resolved, "--version"],
            capture_output=True,
            text=True,
            timeout=max(1.0, float(timeout)),
            env={**os.environ, "NO_COLOR": "1"},
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return _result(
            status="error",
            expected="CUDA",
            runtime="SCONOSCIUTO",
            compiled=(),
            devices=(),
            binary=resolved,
            system=system,
            machine=architecture,
            gpu_name=detected_gpu,
            summary=f"Impossibile verificare vLLM: {exc}",
        )
    output = "\n".join(
        part.strip() for part in (version.stdout, version.stderr) if part
    ).strip()
    cuda_gpu = detected_gpu
    cuda_available = gpu_name is not None and bool(cuda_gpu)
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi:
        try:
            probe = subprocess.run(
                [nvidia_smi, "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if probe.returncode == 0 and probe.stdout.strip():
                cuda_gpu = probe.stdout.strip().splitlines()[0]
                cuda_available = True
        except (OSError, subprocess.TimeoutExpired):
            pass
    if version.returncode == 0 and cuda_available:
        return _result(
            status="accelerated",
            expected="CUDA",
            runtime="CUDA / vLLM",
            compiled=("CUDA",),
            devices=((f"CUDA: {cuda_gpu}" if cuda_gpu else "CUDA"),),
            binary=resolved,
            system=system,
            machine=architecture,
            gpu_name=cuda_gpu,
            summary=(
                "vLLM disponibile con CUDA"
                + (f": {cuda_gpu}." if cuda_gpu else ".")
            ),
            raw_output=output,
        )
    return _result(
        status="backend_unavailable" if version.returncode == 0 else "error",
        expected="CUDA",
        runtime="CPU / NON DISPONIBILE",
        compiled=(),
        devices=(),
        binary=resolved,
        system=system,
        machine=architecture,
        gpu_name=cuda_gpu,
        summary=(
            "vLLM è installato, ma CUDA/NVIDIA non risulta utilizzabile."
            if version.returncode == 0 else
            f"Il comando vLLM è terminato con codice {version.returncode}."
        ),
        raw_output=output,
    )


def diagnose_local_acceleration() -> AccelerationDiagnostic:
    """Combine non-invasive llama.cpp and vLLM diagnostics for the GUI."""
    # This action is explicitly a fresh verification from the GUI.
    try:
        from ..utils.nvidia import cached_nvidia_gpu

        cached_nvidia_gpu.cache_clear()
    except Exception:
        pass
    llama = diagnose_llama_acceleration()

    # vLLM is not a macOS backend.  Reporting its expected absence beside a
    # useful llama.cpp result makes a healthy Metal installation look partly
    # broken and makes a Metal failure unnecessarily noisy.
    if llama.system.casefold() == "darwin":
        return AccelerationDiagnostic(
            status=llama.status,
            expected_backend=llama.expected_backend,
            runtime_backend=llama.runtime_backend,
            compiled_backends=llama.compiled_backends,
            devices=llama.devices,
            binary_path=llama.binary_path,
            system=llama.system,
            machine=llama.machine,
            gpu_name=llama.gpu_name,
            summary=f"llama.cpp: {llama.summary}",
            details=(
                llama.details
                + "\n\nNota: vLLM non viene verificato su macOS perché "
                "richiede Linux/CUDA."
            ),
            raw_output=llama.raw_output,
            is_dgx_spark=llama.is_dgx_spark,
            unified_memory=llama.unified_memory,
            compute_capability=llama.compute_capability,
            cuda_version=llama.cuda_version,
            driver_version=llama.driver_version,
        )

    vllm = diagnose_vllm_acceleration()
    accelerated = [item for item in (llama, vllm) if item.accelerated]
    if accelerated:
        status = "accelerated"
        runtime = " + ".join(item.runtime_backend for item in accelerated)
    elif llama.status == "cpu_only":
        status = "cpu_only"
        runtime = "CPU"
    else:
        status = llama.status if llama.status != "binary_missing" else vllm.status
        runtime = "SCONOSCIUTO"
    return AccelerationDiagnostic(
        status=status,
        expected_backend=llama.expected_backend,
        runtime_backend=runtime,
        compiled_backends=tuple(dict.fromkeys(
            (*llama.compiled_backends, *vllm.compiled_backends)
        )),
        devices=tuple(dict.fromkeys((*llama.devices, *vllm.devices))),
        binary_path="; ".join(filter(None, (llama.binary_path, vllm.binary_path))),
        system=llama.system,
        machine=llama.machine,
        gpu_name=vllm.gpu_name or llama.gpu_name,
        summary=f"llama.cpp: {llama.summary}  vLLM: {vllm.summary}",
        details=(
            "=== llama.cpp ===\n" + llama.details
            + "\n\n=== vLLM ===\n" + vllm.details
        ),
        raw_output=(llama.raw_output + "\n" + vllm.raw_output)[-6000:],
        is_dgx_spark=llama.is_dgx_spark or vllm.is_dgx_spark,
        unified_memory=llama.unified_memory or vllm.unified_memory,
        compute_capability=(
            vllm.compute_capability or llama.compute_capability
        ),
        cuda_version=vllm.cuda_version or llama.cuda_version,
        driver_version=vllm.driver_version or llama.driver_version,
    )


def _result(
    *,
    status: str,
    expected: str,
    runtime: str,
    compiled: tuple[str, ...],
    devices: tuple[str, ...],
    binary: str,
    system: str,
    machine: str,
    gpu_name: str,
    summary: str,
    raw_output: str = "",
) -> AccelerationDiagnostic:
    nvidia = None
    if system.casefold() == "linux":
        try:
            from ..utils.nvidia import cached_nvidia_gpu

            nvidia = cached_nvidia_gpu(system, machine)
        except Exception:
            nvidia = None
    is_dgx = bool(
        (nvidia and nvidia.is_dgx_spark)
        or (
            machine.casefold() in {"arm64", "aarch64"}
            and "gb10" in gpu_name.casefold()
        )
    )
    compute = (
        (nvidia.compute_capability if nvidia else "")
        or ("12.1" if is_dgx else "")
    )
    cuda_version = nvidia.cuda_version if nvidia else ""
    driver_version = nvidia.driver_version if nvidia else ""
    compiled_text = ", ".join(compiled) if compiled else "non rilevato"
    device_text = "; ".join(devices) if devices else "nessuno"
    details = "\n".join((
        f"Sistema: {system} {machine}",
        f"GPU rilevata dal sistema: {gpu_name or 'non identificata'}",
        f"Binario: {binary or 'non trovato'}",
        f"Backend atteso: {expected}",
        f"Supporto compilato rilevato: {compiled_text}",
        f"Backend utilizzabile: {runtime}",
        f"Dispositivi esposti dal backend: {device_text}",
    ))
    if is_dgx:
        details += (
            "\nProfilo: NVIDIA DGX Spark / GB10, memoria coerente unificata"
            "\nNota memoria: la VRAM può risultare N/A in nvidia-smi; "
            "il dimensionamento usa la RAM di sistema"
        )
    if compute:
        details += f"\nCompute capability: {compute} (sm_{compute.replace('.', '')})"
    if cuda_version:
        details += f"\nCUDA dichiarata dal driver: {cuda_version}"
    if driver_version:
        details += f"\nDriver NVIDIA: {driver_version}"
    if raw_output:
        details += "\n\nOutput llama-server:\n" + raw_output[-6000:]
    if is_dgx and "dgx spark" not in summary.casefold():
        summary = f"DGX Spark / GB10 — {summary}"
    return AccelerationDiagnostic(
        status=status,
        expected_backend=expected,
        runtime_backend=runtime,
        compiled_backends=compiled,
        devices=devices,
        binary_path=binary,
        system=system,
        machine=machine,
        gpu_name=gpu_name,
        summary=summary,
        details=details,
        raw_output=raw_output[-6000:],
        is_dgx_spark=is_dgx,
        unified_memory=is_dgx,
        compute_capability=compute,
        cuda_version=cuda_version,
        driver_version=driver_version,
    )


def _host_gpu_name() -> str:
    try:
        from ..utils.hardware import HardwareProfile

        return HardwareProfile.capture().gpu_name
    except Exception:
        return ""


def _expected_backend(system: str, gpu_name: str) -> str:
    if system.casefold() == "darwin":
        return "METAL"
    if system.casefold() == "linux" and (
        "nvidia" in gpu_name.casefold() or shutil.which("nvidia-smi")
    ):
        return "CUDA"
    return "CPU"


def _compiled_backends(output: str) -> tuple[str, ...]:
    lowered = output.casefold()
    backends = []
    if (
        "ggml_cuda" in lowered
        or "loaded cuda backend" in lowered
        or re.search(r"\bcuda\d*\s*:", output, re.I)
    ):
        backends.append("CUDA")
    if (
        "ggml_metal" in lowered
        or "loaded metal backend" in lowered
        or "loaded mtl backend" in lowered
        or re.search(r"\b(?:metal|mtl)\d*\s*:", output, re.I)
    ):
        backends.append("METAL")
    return tuple(backends)


def _parse_devices(output: str) -> tuple[str, ...]:
    devices: list[str] = []
    for raw_line in output.splitlines():
        line = raw_line.strip()
        if not line or line.casefold() == "(none)":
            continue
        match = re.match(r"(CUDA\d*|Metal\d*|MTL\d*)\s*:\s*(.+)", line, re.I)
        if match:
            device_name = match.group(1)
            if device_name.casefold().startswith("mtl"):
                device_name = "Metal" + device_name[3:]
            devices.append(f"{device_name}: {match.group(2).strip()}")
            continue
        if "ggml_metal" in line.casefold() and "gpu name:" in line.casefold():
            devices.append("Metal: " + line.split(":", 2)[-1].strip())
            continue
        if (
            re.search(r"\bDevice\s+\d+\s*:", line, re.I)
            and "nvidia" in line.casefold()
        ):
            devices.append(line)
    # Preserve the order printed by llama.cpp while removing duplicate lines.
    return tuple(dict.fromkeys(devices))


def _cuda_available(output: str, devices: tuple[str, ...]) -> bool:
    lowered = output.casefold()
    if any(device.casefold().startswith("cuda") for device in devices):
        return True
    match = re.search(r"found\s+(\d+)\s+cuda devices?", lowered)
    return bool(match and int(match.group(1)) > 0)


def _metal_available(output: str, devices: tuple[str, ...]) -> bool:
    lowered = output.casefold()
    if any(device.casefold().startswith("metal") for device in devices):
        return True
    has_gpu = "ggml_metal" in lowered and "gpu name:" in lowered
    failed = "metal" in lowered and any(word in lowered for word in (
        "failed", "unavailable", "not available", "no metal device",
    ))
    return has_gpu and not failed

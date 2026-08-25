"""Verified, application-managed llama-server runtime storage.

The clinical process never downloads or executes an unverified remote asset.
An operator builds or obtains llama-server separately, then explicitly imports
the local executable.  The importer validates the executable on the current
host, requires the expected GPU backend when appropriate, copies it atomically
under ``~/.emr_analyzer/runtimes`` and records its SHA-256.  Runtime discovery
rechecks the digest before giving the path to :class:`ServerManager`.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import shutil
import stat
import subprocess
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from ..config import LLAMA_RUNTIME_DIR, LLAMA_RUNTIME_INDEX_PATH


_SCHEMA_VERSION = 1
_COPY_CHUNK_SIZE = 4 * 1024 * 1024
_MAX_BINARY_BYTES = 2 * 1024 * 1024 * 1024


class ServerRuntimeError(RuntimeError):
    """A managed llama-server could not be validated or activated."""


@dataclass(frozen=True)
class ManagedServerRuntime:
    """One active, integrity-checked llama-server installation."""

    installation_id: str
    binary_path: Path
    sha256: str
    version: str
    system: str
    machine: str
    backend: str
    devices: tuple[str, ...]
    installed_at: str

    def as_dict(self) -> dict:
        return {
            "installation_id": self.installation_id,
            "binary_path": str(self.binary_path),
            "sha256": self.sha256,
            "version": self.version,
            "system": self.system,
            "machine": self.machine,
            "backend": self.backend,
            "devices": list(self.devices),
            "installed_at": self.installed_at,
        }


def install_managed_server(
    source: str | os.PathLike,
    *,
    expected_sha256: str | None = None,
    runtime_dir: str | os.PathLike = LLAMA_RUNTIME_DIR,
    index_path: str | os.PathLike = LLAMA_RUNTIME_INDEX_PATH,
    system_name: str | None = None,
    machine: str | None = None,
    require_acceleration: bool | None = None,
) -> ManagedServerRuntime:
    """Validate, atomically copy and activate a local llama-server binary.

    A checksum supplied by the user is checked *before* execution.  Locally
    built binaries may omit it; their computed checksum is still recorded and
    rechecked at every future discovery.  macOS and NVIDIA Linux hosts require
    the corresponding Metal/CUDA device by default.
    """

    source_path = Path(source).expanduser()
    if source_path.is_symlink():
        source_path = source_path.resolve(strict=True)
    if not source_path.is_file():
        raise ServerRuntimeError("Seleziona un file llama-server esistente.")
    size = source_path.stat().st_size
    if size <= 0 or size > _MAX_BINARY_BYTES:
        raise ServerRuntimeError("Dimensione del binario llama-server non valida.")

    checksum = _sha256(source_path)
    expected = str(expected_sha256 or "").strip().casefold()
    if expected and not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise ServerRuntimeError("La checksum SHA-256 deve contenere 64 cifre esadecimali.")
    if expected and checksum != expected:
        raise ServerRuntimeError(
            "Checksum SHA-256 non corrispondente: il file non verrà eseguito."
        )

    system = str(system_name or platform.system() or "unknown")
    architecture = _normalized_machine(machine or platform.machine())
    verification = verify_server_binary(
        source_path,
        system_name=system,
        machine=architecture,
        require_acceleration=require_acceleration,
    )

    root = Path(runtime_dir).expanduser()
    manifest_path = Path(index_path).expanduser()
    platform_root = root / f"{system.casefold()}-{architecture}"
    installation_id = f"{system.casefold()}-{architecture}-{checksum[:16]}"
    destination = platform_root / installation_id / _binary_name(system)

    platform_root.mkdir(parents=True, exist_ok=True)
    if not destination.is_file() or _sha256(destination) != checksum:
        temporary_dir = Path(tempfile.mkdtemp(prefix=".install-", dir=platform_root))
        try:
            temporary_binary = temporary_dir / _binary_name(system)
            _copy_file(source_path, temporary_binary)
            temporary_binary.chmod(
                temporary_binary.stat().st_mode
                | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH
            )
            if _sha256(temporary_binary) != checksum:
                raise ServerRuntimeError("La copia del binario non ha superato la verifica SHA-256.")
            # Re-run the executable from its managed location before activation.
            verify_server_binary(
                temporary_binary,
                system_name=system,
                machine=architecture,
                require_acceleration=require_acceleration,
            )
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.replace(temporary_binary, destination)
        finally:
            shutil.rmtree(temporary_dir, ignore_errors=True)

    installed_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    managed_binary = destination.resolve(strict=True)
    runtime = ManagedServerRuntime(
        installation_id=installation_id,
        binary_path=managed_binary,
        sha256=checksum,
        version=verification["version"],
        system=system,
        machine=architecture,
        backend=verification["backend"],
        devices=tuple(verification["devices"]),
        installed_at=installed_at,
    )
    payload = _load_index(manifest_path)
    installations = dict(payload.get("installations") or {})
    installations[installation_id] = {
        "binary": str(destination.relative_to(root)),
        "sha256": checksum,
        "version": runtime.version,
        "system": system,
        "machine": architecture,
        "backend": runtime.backend,
        "devices": list(runtime.devices),
        "installed_at": installed_at,
        "source_name": source_path.name,
    }
    _save_index(
        manifest_path,
        {
            "schema_version": _SCHEMA_VERSION,
            "active_installation": installation_id,
            "installations": installations,
        },
    )
    return runtime


def active_managed_server(
    *,
    runtime_dir: str | os.PathLike = LLAMA_RUNTIME_DIR,
    index_path: str | os.PathLike = LLAMA_RUNTIME_INDEX_PATH,
    system_name: str | None = None,
    machine: str | None = None,
    verify_checksum: bool = True,
) -> ManagedServerRuntime | None:
    """Return the active runtime only when manifest and binary are intact."""

    root = Path(runtime_dir).expanduser()
    payload = _load_index(Path(index_path).expanduser())
    installation_id = str(payload.get("active_installation") or "")
    raw = (payload.get("installations") or {}).get(installation_id)
    if not installation_id or not isinstance(raw, dict):
        return None
    system = str(system_name or platform.system() or "unknown")
    architecture = _normalized_machine(machine or platform.machine())
    if str(raw.get("system") or "").casefold() != system.casefold():
        return None
    if _normalized_machine(raw.get("machine")) != architecture:
        return None
    relative = Path(str(raw.get("binary") or ""))
    if relative.is_absolute() or ".." in relative.parts:
        return None
    binary = root / relative
    try:
        resolved_root = root.resolve(strict=True)
        resolved_binary = binary.resolve(strict=True)
        resolved_binary.relative_to(resolved_root)
    except (OSError, ValueError):
        return None
    if binary.is_symlink() or not resolved_binary.is_file():
        return None
    checksum = str(raw.get("sha256") or "").casefold()
    if not re.fullmatch(r"[0-9a-f]{64}", checksum):
        return None
    if verify_checksum and _sha256(resolved_binary) != checksum:
        return None
    return ManagedServerRuntime(
        installation_id=installation_id,
        binary_path=resolved_binary,
        sha256=checksum,
        version=str(raw.get("version") or "sconosciuta"),
        system=system,
        machine=architecture,
        backend=str(raw.get("backend") or "CPU"),
        devices=tuple(str(item) for item in raw.get("devices") or ()),
        installed_at=str(raw.get("installed_at") or ""),
    )


def deactivate_managed_server(
    *, index_path: str | os.PathLike = LLAMA_RUNTIME_INDEX_PATH,
) -> bool:
    """Deactivate the managed runtime without deleting recoverable files."""

    manifest_path = Path(index_path).expanduser()
    payload = _load_index(manifest_path)
    if not payload.get("active_installation"):
        return False
    payload["active_installation"] = ""
    payload.setdefault("schema_version", _SCHEMA_VERSION)
    payload.setdefault("installations", {})
    _save_index(manifest_path, payload)
    return True


def verify_server_binary(
    binary: str | os.PathLike,
    *,
    system_name: str | None = None,
    machine: str | None = None,
    require_acceleration: bool | None = None,
    timeout: float = 30.0,
) -> dict:
    """Execute harmless version/device probes and validate host compatibility."""

    path = Path(binary)
    system = str(system_name or platform.system() or "unknown")
    architecture = _normalized_machine(machine or platform.machine())
    version_result = _run_probe(path, ["--version"], timeout)
    version_output = _combined_output(version_result)
    if version_result.returncode != 0 or not re.search(
        r"(?:\bversion\b|llama)", version_output, re.I
    ):
        raise ServerRuntimeError(
            "Il file non risponde come un eseguibile llama-server compatibile."
        )
    _check_reported_platform(version_output, system, architecture)

    devices_result = _run_probe(path, ["-v", "--list-devices"], timeout)
    device_output = _combined_output(devices_result)
    if devices_result.returncode != 0 or "available devices" not in device_output.casefold():
        raise ServerRuntimeError(
            "llama-server non supporta la verifica dei dispositivi richiesta dall'app."
        )
    devices = _device_lines(device_output)
    metal = any(item.casefold().startswith("metal") for item in devices)
    cuda = any(item.casefold().startswith("cuda") for item in devices)
    backend = "METAL" if metal else "CUDA" if cuda else "CPU"
    expected = _expected_backend(system)
    must_accelerate = (
        expected in {"METAL", "CUDA"}
        if require_acceleration is None else bool(require_acceleration)
    )
    if must_accelerate and backend != expected:
        raise ServerRuntimeError(
            f"Il server è eseguibile ma non espone un dispositivo {expected}; "
            "non verrà attivato perché ricadrebbe sulla CPU."
        )
    version = _version_line(version_output)
    return {
        "version": version,
        "backend": backend,
        "devices": devices,
        "raw_output": (version_output + "\n" + device_output)[-6000:],
    }


def _run_probe(path: Path, arguments: list[str], timeout: float):
    try:
        return subprocess.run(
            [str(path), *arguments],
            capture_output=True,
            text=True,
            timeout=max(1.0, float(timeout)),
            env={**os.environ, "NO_COLOR": "1"},
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ServerRuntimeError(f"Impossibile verificare llama-server: {exc}") from exc


def _check_reported_platform(output: str, system: str, machine: str) -> None:
    match = re.search(r"\bfor\s+(Darwin|Linux|Windows)\s+([A-Za-z0-9_+-]+)", output, re.I)
    if not match:
        return
    reported_system = match.group(1)
    reported_machine = _normalized_machine(match.group(2))
    if reported_system.casefold() != system.casefold() or reported_machine != machine:
        raise ServerRuntimeError(
            "Il binario è stato compilato per "
            f"{reported_system}/{reported_machine}, non per {system}/{machine}."
        )


def _device_lines(output: str) -> tuple[str, ...]:
    devices: list[str] = []
    for raw in output.splitlines():
        line = raw.strip()
        match = re.match(r"(CUDA\d*|Metal\d*|MTL\d*)\s*:\s*(.+)", line, re.I)
        if not match:
            continue
        name, description = match.groups()
        # BLAS/CPU entries deliberately do not match.  Empty dynamic CPU
        # device lines are likewise ignored.
        if name.casefold().startswith("mtl"):
            name = "Metal" + name[3:]
        devices.append(f"{name}: {description.strip()}")
    return tuple(dict.fromkeys(devices))


def _expected_backend(system: str) -> str:
    if system.casefold() == "darwin":
        return "METAL"
    if system.casefold() == "linux":
        if shutil.which("nvidia-smi") or Path("/dev/nvidiactl").exists():
            return "CUDA"
        try:
            from ..utils.nvidia import cached_nvidia_gpu

            gpu = cached_nvidia_gpu(system, platform.machine())
            if gpu.available or gpu.is_dgx_spark:
                return "CUDA"
        except Exception:
            pass
    return "CPU"


def _version_line(output: str) -> str:
    for line in output.splitlines():
        clean = line.strip()
        if re.search(r"\bversion\b", clean, re.I):
            return clean[:300]
    return output.strip().splitlines()[0][:300]


def _combined_output(result) -> str:
    return "\n".join(
        part.strip() for part in (result.stdout, result.stderr) if part
    ).strip()


def _normalized_machine(value) -> str:
    machine = str(value or "unknown").strip().casefold()
    return {
        "amd64": "x86_64",
        "x64": "x86_64",
        "aarch64": "arm64",
    }.get(machine, machine)


def _binary_name(system: str) -> str:
    return "llama-server.exe" if system.casefold() == "windows" else "llama-server"


def _copy_file(source: Path, destination: Path) -> None:
    with source.open("rb") as reader, destination.open("xb") as writer:
        while True:
            chunk = reader.read(_COPY_CHUNK_SIZE)
            if not chunk:
                break
            writer.write(chunk)
        writer.flush()
        os.fsync(writer.fileno())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(_COPY_CHUNK_SIZE)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _load_index(path: Path) -> dict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return {"schema_version": _SCHEMA_VERSION, "installations": {}}
    if not isinstance(payload, dict) or payload.get("schema_version") != _SCHEMA_VERSION:
        return {"schema_version": _SCHEMA_VERSION, "installations": {}}
    if not isinstance(payload.get("installations"), dict):
        payload["installations"] = {}
    return payload


def _save_index(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".runtime-index-", suffix=".json", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass

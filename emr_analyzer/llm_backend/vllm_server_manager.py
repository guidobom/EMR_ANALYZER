"""Lifecycle for optional app-managed vLLM OpenAI servers.

vLLM is intentionally a Linux/CUDA-only optional backend.  A process is
started only when a role configured with ``backend='vllm'`` is used.  The
child binds to loopback and receives offline Hugging Face environment flags,
so inference never downloads a model as a side effect.
"""

from __future__ import annotations

import atexit
import collections
import os
import shutil
import socket
import subprocess
import sys
import threading
import time
from dataclasses import dataclass

import httpx

from ..config import (
    VLLM_SERVER_BASE_PORT,
    VLLM_SERVER_BINARY,
    VLLM_SERVER_HOST,
    VLLM_SERVER_LOAD_TIMEOUT,
)
from .server_manager import BackendError

_HEALTH_POLL_INTERVAL = 1.0
_STOP_GRACE_SECONDS = 15.0
_PORT_SCAN_COUNT = 20


@dataclass(frozen=True)
class VllmServerKey:
    """All engine-level options that require a distinct vLLM process."""

    model: str
    ctx_size: int
    max_num_seqs: int
    tensor_parallel_size: int = 1
    dtype: str = "auto"
    gpu_memory_utilization: float = 0.9
    quantization: str = ""
    trust_remote_code: bool = False
    enforce_eager: bool = False


@dataclass
class _VllmInstance:
    key: VllmServerKey
    port: int
    proc: subprocess.Popen
    stderr_lines: collections.deque


def find_vllm_binary() -> str | None:
    """Find the CLI installed in the same environment used by the app."""
    configured = str(
        VLLM_SERVER_BINARY
        or os.environ.get("EMR_ANALYZER_VLLM_BINARY")
        or ""
    ).strip()
    if configured and os.path.isfile(configured):
        return configured
    found = shutil.which("vllm")
    if found:
        return found
    managed = os.path.join(
        os.path.expanduser("~"), ".emr_analyzer", "vllm-env", "bin", "vllm"
    )
    if os.path.isfile(managed):
        return managed
    candidate = os.path.join(os.path.expanduser("~"), ".local", "bin", "vllm")
    return candidate if os.path.isfile(candidate) else None


def _port_is_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            probe.bind((VLLM_SERVER_HOST, port))
            return True
        except OSError:
            return False


class VllmServerManager:
    """Thread-safe registry of vLLM processes owned by this application."""

    def __init__(self, binary: str | None = None) -> None:
        self._binary = binary or find_vllm_binary()
        self._instances: dict[VllmServerKey, _VllmInstance] = {}
        self._lock = threading.RLock()
        self._shutdown_hook_registered = False

    @property
    def binary(self) -> str | None:
        return self._binary

    def usable(self) -> bool:
        return self._binary is not None and sys.platform.startswith("linux")

    def status(self, key: VllmServerKey) -> str | None:
        instance = self._instances.get(key)
        if instance is None or instance.proc.poll() is not None:
            return None
        try:
            response = httpx.get(
                f"http://{VLLM_SERVER_HOST}:{instance.port}/health",
                timeout=2.0,
            )
        except httpx.HTTPError:
            return "loading"
        return "running" if response.status_code == 200 else "loading"

    def base_url(self, key: VllmServerKey) -> str | None:
        instance = self._instances.get(key)
        if instance is None or instance.proc.poll() is not None:
            return None
        return f"http://{VLLM_SERVER_HOST}:{instance.port}"

    def ensure(
        self,
        key: VllmServerKey,
        load_timeout: float = VLLM_SERVER_LOAD_TIMEOUT,
        progress_cb=None,
    ) -> str:
        if self._binary is None:
            raise BackendError(
                "vLLM non trovato nell'ambiente Python corrente. "
                "Esegui tools/setup_vllm_backend.py sulla DGX Spark."
            )
        with self._lock:
            instance = self._instances.get(key)
            if instance is None or instance.proc.poll() is not None:
                if instance is not None:
                    self._instances.pop(key, None)
                instance = self._spawn(key)
                self._register_shutdown_hook()

        deadline = time.monotonic() + load_timeout
        while time.monotonic() < deadline:
            status = self.status(key)
            if progress_cb is not None:
                try:
                    progress_cb(status or "starting")
                except Exception:
                    pass
            if status == "running":
                return self.base_url(key)  # type: ignore[return-value]
            if instance.proc.poll() is not None:
                break
            time.sleep(_HEALTH_POLL_INTERVAL)

        detail = "".join(list(instance.stderr_lines)[-80:]).strip()
        if instance.proc.poll() is not None:
            detail = detail or f"exit code {instance.proc.poll()}"
        raise BackendError(
            "vLLM non ha completato il caricamento del modello in tempo"
            + (f": {detail}" if detail else "")
        )

    def stop(self, key: VllmServerKey) -> bool:
        with self._lock:
            instance = self._instances.pop(key, None)
            if instance is None:
                return False
            self._terminate(instance)
            return True

    def stop_all(self) -> int:
        with self._lock:
            instances = list(self._instances.values())
            self._instances.clear()
        for instance in instances:
            self._terminate(instance)
        return len(instances)

    def running_keys(self) -> list[VllmServerKey]:
        with self._lock:
            return [
                key for key, instance in self._instances.items()
                if instance.proc.poll() is None
            ]

    def shutdown(self) -> None:
        self.stop_all()

    def _spawn(self, key: VllmServerKey) -> _VllmInstance:
        port = self._allocate_port()
        argv = [
            self._binary,
            "serve",
            key.model,
            "--host", VLLM_SERVER_HOST,
            "--port", str(port),
            "--served-model-name", key.model,
            "--max-model-len", str(key.ctx_size),
            "--max-num-seqs", str(key.max_num_seqs),
            "--tensor-parallel-size", str(key.tensor_parallel_size),
            "--dtype", key.dtype,
            "--gpu-memory-utilization",
            f"{key.gpu_memory_utilization:.3f}",
        ]
        if key.quantization:
            argv.extend(["--quantization", key.quantization])
        if key.trust_remote_code:
            argv.append("--trust-remote-code")
        if key.enforce_eager:
            argv.append("--enforce-eager")

        env = os.environ.copy()
        env.update({
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_HUB_DISABLE_TELEMETRY": "1",
        })
        # When vLLM lives in the app-managed isolated environment, its
        # helper executables (notably ninja for FlashInfer JIT kernels) are
        # siblings of the vllm entry point rather than members of the GUI's
        # inherited PATH.
        runtime_bin = os.path.dirname(str(self._binary))
        inherited_path = str(env.get("PATH") or "")
        env["PATH"] = runtime_bin + (
            os.pathsep + inherited_path if inherited_path else ""
        )
        try:
            proc = subprocess.Popen(
                argv,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
                start_new_session=True,
                env=env,
            )
        except OSError as exc:
            raise BackendError(f"Impossibile avviare vLLM: {exc}") from exc

        stderr_lines: collections.deque = collections.deque(maxlen=100)
        threading.Thread(
            target=self._drain_stderr,
            args=(proc, stderr_lines),
            daemon=True,
        ).start()
        instance = _VllmInstance(key, port, proc, stderr_lines)
        self._instances[key] = instance
        return instance

    def _allocate_port(self) -> int:
        managed_ports = {
            instance.port for instance in self._instances.values()
            if instance.proc.poll() is None
        }
        for offset in range(_PORT_SCAN_COUNT):
            port = VLLM_SERVER_BASE_PORT + offset
            if port not in managed_ports and _port_is_free(port):
                return port
        raise BackendError(
            f"Nessuna porta libera nell'intervallo {VLLM_SERVER_BASE_PORT}-"
            f"{VLLM_SERVER_BASE_PORT + _PORT_SCAN_COUNT - 1}"
        )

    @staticmethod
    def _drain_stderr(proc: subprocess.Popen, buffer: collections.deque) -> None:
        stderr = getattr(proc, "stderr", None)
        if stderr is None:
            return
        try:
            for line in stderr:
                buffer.append(line)
        except (OSError, ValueError):
            pass

    @staticmethod
    def _terminate(instance: _VllmInstance) -> None:
        proc = instance.proc
        if proc.poll() is not None:
            return
        try:
            proc.terminate()
            proc.wait(timeout=_STOP_GRACE_SECONDS)
        except subprocess.TimeoutExpired:
            try:
                proc.kill()
                proc.wait(timeout=3.0)
            except (OSError, subprocess.TimeoutExpired):
                pass
        except OSError:
            pass

    def _register_shutdown_hook(self) -> None:
        if not self._shutdown_hook_registered:
            atexit.register(self.shutdown)
            self._shutdown_hook_registered = True

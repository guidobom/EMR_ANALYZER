"""Lifecycle management for the app's own llama-server child processes.

One ``llama-server`` process serves one GGUF model with one fixed context
size and a fixed number of parallel slots.  Processes are keyed by
``ServerKey(gguf_path, ctx_size, np)`` so that changing the model, the
context length, or the worker count transparently restarts the engine.

Processes are spawned on demand, terminated on application quit (both
``atexit`` and ``QApplication.aboutToQuit``), and stale processes left over
from a crashed run are reaped when their port is re-allocated.
"""

from __future__ import annotations

import atexit
import collections
import os
import shutil
import socket
import subprocess
import threading
import time
from dataclasses import dataclass

import httpx

from ..config import (
    LLAMA_SERVER_BASE_PORT,
    LLAMA_SERVER_BINARY,
    LLAMA_SERVER_HOST,
    LLAMA_SERVER_LOAD_TIMEOUT,
)

# How long a freshly spawned server may take before /health reports ok.
_HEALTH_POLL_INTERVAL = 0.5
# Grace period between SIGTERM and SIGKILL.
_STOP_GRACE_SECONDS = 5.0
# Ports are allocated upward from the base port.
_PORT_SCAN_COUNT = 20


class BackendError(RuntimeError):
    """The llama-server backend failed to start or answer."""


@dataclass(frozen=True)
class ServerKey:
    """Identity of one server process: model file, context, parallel slots.

    ``instance`` discriminates otherwise identical configurations so the
    multi-patient irAE queue can run one llama-server per patient: two keys
    with the same model/context/slots but a different ``instance`` spawn
    separate processes on separate ports.
    """

    gguf_path: str
    ctx_size: int
    np: int
    speculative_mode: str = "none"
    instance: int = 0


@dataclass
class _ServerInstance:
    key: ServerKey
    port: int
    proc: subprocess.Popen
    stderr_lines: collections.deque


def find_server_binary() -> str | None:
    """Select managed verified runtime first, then external fallbacks."""
    from .server_runtime import active_managed_server

    managed = active_managed_server()
    if managed is not None:
        return str(managed.binary_path)
    return find_external_server_binary()


def find_external_server_binary() -> str | None:
    """Locate an unmanaged compatibility fallback outside the runtime store."""
    configured = str(LLAMA_SERVER_BINARY or "").strip()
    if configured and os.path.isfile(configured):
        return configured
    found = shutil.which("llama-server")
    if found:
        return found
    home = os.path.expanduser("~")
    for prefix in ("/opt/homebrew/opt/llama.cpp/bin",
                   "/usr/local/opt/llama.cpp/bin",
                   "/opt/homebrew/bin",
                   "/usr/local/bin",
                   os.path.join(home, ".local", "bin")):
        candidate = os.path.join(prefix, "llama-server")
        if os.path.isfile(candidate):
            return candidate
    return None


def _port_is_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            probe.bind((LLAMA_SERVER_HOST, port))
            return True
        except OSError:
            return False


class ServerManager:
    """Thread-safe registry of the app's llama-server child processes."""

    def __init__(self, binary: str | None = None) -> None:
        self._binary = binary or find_server_binary()
        self._instances: dict[ServerKey, _ServerInstance] = {}
        self._lock = threading.RLock()
        self._shutdown_hook_registered = False

    # -- public API ---------------------------------------------------------

    @property
    def binary(self) -> str | None:
        return self._binary

    def usable(self) -> bool:
        """Binary present; model availability is checked separately."""
        return self._binary is not None

    def select_binary(self, binary: str | None) -> None:
        """Use *binary* for future servers when no process is still running."""
        with self._lock:
            active = [
                item for item in self._instances.values()
                if item.proc.poll() is None
            ]
            if active:
                raise BackendError(
                    "Scarica i modelli in memoria prima di cambiare "
                    "llama-server."
                )
            if binary is not None and not os.path.isfile(binary):
                raise BackendError(f"llama-server non trovato: {binary}")
            self._binary = binary

    def status(self, key: ServerKey) -> str | None:
        """``"loading"``, ``"running"``, or ``None`` when not started."""
        instance = self._instances.get(key)
        if instance is None or instance.proc.poll() is not None:
            return None
        try:
            response = httpx.get(
                f"http://{LLAMA_SERVER_HOST}:{instance.port}/health",
                timeout=2.0,
            )
        except httpx.HTTPError:
            return None
        if response.status_code == 200:
            return "running"
        if response.status_code == 503:
            return "loading"
        return None

    def base_url(self, key: ServerKey) -> str | None:
        instance = self._instances.get(key)
        if instance is None:
            return None
        return f"http://{LLAMA_SERVER_HOST}:{instance.port}"

    def ensure(
        self,
        key: ServerKey,
        load_timeout: float = LLAMA_SERVER_LOAD_TIMEOUT,
        progress_cb=None,
    ) -> str:
        """Spawn the server for *key* if needed and wait until it answers.

        Returns the base URL.  Raises :class:`BackendError` when the binary
        is missing or the process fails to become healthy in time.
        """
        if self._binary is None:
            raise BackendError(
                "llama-server non trovato: installa llama.cpp "
                "(brew install llama.cpp) o esegui tools/setup_llama_backend.py"
            )
        if not os.path.isfile(key.gguf_path):
            raise BackendError(
                f"File modello GGUF non trovato: {key.gguf_path}"
            )

        with self._lock:
            existing = self._instances.get(key)
            if existing is not None and existing.proc.poll() is None:
                status = self.status(key)
                if status in ("running", "loading"):
                    instance = existing
                else:
                    # Dead process: clean up and respawn.
                    self._terminate(existing)
                    instance = None
            else:
                instance = None

            if instance is None:
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
            if status is None and instance.proc.poll() is not None:
                break  # process died: fail fast with its stderr
            time.sleep(_HEALTH_POLL_INTERVAL)

        detail = "".join(list(instance.stderr_lines)[-15:]).strip()
        if instance.proc.poll() is not None:
            detail = detail or f"exit code {instance.proc.poll()}"
        raise BackendError(
            "llama-server non ha completato il caricamento in tempo"
            + (f": {detail}" if detail else "")
        )

    def stop(self, key: ServerKey) -> bool:
        """Terminate the server for *key* (idempotent).  Returns True if it
        was running."""
        with self._lock:
            instance = self._instances.get(key)
            if instance is None:
                return False
            self._terminate(instance)
            del self._instances[key]
            return True

    def stop_all(self) -> int:
        """Terminate every managed server; returns the number stopped."""
        with self._lock:
            keys = list(self._instances)
            for key in keys:
                instance = self._instances.pop(key)
                self._terminate(instance)
        return len(keys)

    def running_keys(self) -> list[ServerKey]:
        with self._lock:
            return [
                key for key, instance in self._instances.items()
                if instance.proc.poll() is None
            ]

    def shutdown(self) -> None:
        """Stop every child process.  Safe to call multiple times."""
        self.stop_all()

    # -- internals -----------------------------------------------------------

    def _spawn(self, key: ServerKey) -> _ServerInstance:
        """Allocate a port and start llama-server for *key*.

        ``-c`` is the TOTAL context budget, split evenly across the ``np``
        parallel slots (verified against llama-server build 10470): to give
        every request the full configured context the total must be
        ``ctx_size * np``.  KV cache is quantized to q8_0 (≈ half the RAM of
        f16 with negligible quality loss).  ``-rea off`` disables the
        thinking channel for models that support it (qwen3).
        """
        port = self._allocate_port()
        argv = [
            self._binary,
            "-m", key.gguf_path,
            "--host", LLAMA_SERVER_HOST,
            "--port", str(port),
            "-c", str(key.ctx_size * key.np),
            "-np", str(key.np),
            "-ngl", "all",
            "-ctk", "q8_0",
            "-ctv", "q8_0",
            "-rea", "off",
        ]
        if key.speculative_mode == "ngram-cache":
            # Draft tokens are always checked by the target model.  This can
            # accelerate repetitive structured JSON without replacing the
            # clinical model or accepting an unverified token.
            argv.extend(["--spec-type", "ngram-cache"])
        try:
            proc = subprocess.Popen(
                argv,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
                start_new_session=True,
            )
        except OSError as exc:
            raise BackendError(
                f"Impossibile avviare llama-server: {exc}"
            ) from exc

        stderr_lines: collections.deque = collections.deque(maxlen=50)
        threading.Thread(
            target=self._drain_stderr,
            args=(proc, stderr_lines),
            daemon=True,
        ).start()

        instance = _ServerInstance(key, port, proc, stderr_lines)
        self._instances[key] = instance

        # Older llama.cpp builds named the reasoning flag differently
        # (--disable-reasoning).  If the process exits immediately and the
        # error mentions the flag, retry once without it.
        if self._wait_early_exit(proc, 3.0):
            tail = "".join(list(stderr_lines)[-5:])
            if "-rea" in tail or "reasoning" in tail.lower():
                self._terminate(instance)
                # Drop "-rea" together with the value that follows it.
                argv = [arg for i, arg in enumerate(argv)
                        if not (arg == "-rea" or
                                (i > 0 and argv[i - 1] == "-rea"))]
                proc = subprocess.Popen(
                    argv,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                    text=True,
                    start_new_session=True,
                )
                stderr_lines = collections.deque(maxlen=50)
                threading.Thread(
                    target=self._drain_stderr,
                    args=(proc, stderr_lines),
                    daemon=True,
                ).start()
                instance = _ServerInstance(key, port, proc, stderr_lines)
                self._instances[key] = instance
        return instance

    def _allocate_port(self) -> int:
        """First free port from the configured base upward.

        Ports owned by one of our live servers are skipped, never reaped.
        A stale llama-server process of a previous run is terminated so its
        port can be reused; anything else (including this application and
        its client sockets) is left alone.
        """
        for offset in range(_PORT_SCAN_COUNT):
            port = LLAMA_SERVER_BASE_PORT + offset
            if self._port_owned_by_managed(port):
                continue
            if _port_is_free(port):
                return port
            self._reap_port(port)
            if _port_is_free(port):
                return port
        raise BackendError(
            f"Nessuna porta libera nell'intervallo "
            f"{LLAMA_SERVER_BASE_PORT}-"
            f"{LLAMA_SERVER_BASE_PORT + _PORT_SCAN_COUNT - 1}"
        )

    def _port_owned_by_managed(self, port: int) -> bool:
        """True when one of our live servers already listens on *port*."""
        return any(
            instance.port == port and instance.proc.poll() is None
            for instance in self._instances.values()
        )

    def _reap_port(self, port: int) -> None:
        """Kill the process on *port* only when it is an ORPHANED llama-server.

        ``lsof`` also lists client sockets (this app's own keep-alive
        connections), so every candidate PID is verified by name first: the
        application itself and unrelated processes are never signalled.  A
        llama-server whose parent is still alive belongs to a running
        application (possibly another EMR Analyzer process on the same
        machine) and is left alone: only stale servers left behind by a
        crashed run — reparented to launchd/init — are reclaimed.
        """
        try:
            result = subprocess.run(
                ["lsof", "-ti", f"tcp:{port}"],
                capture_output=True, text=True, timeout=5,
            )
        except (OSError, subprocess.TimeoutExpired):
            return
        for raw_pid in result.stdout.split():
            if not raw_pid.isdigit():
                continue
            pid = int(raw_pid)
            if pid == os.getpid():
                continue
            if not self._is_orphaned_llama_server(pid):
                continue
            try:
                os.kill(pid, 15)  # SIGTERM
            except (OSError, ValueError):
                continue

    @staticmethod
    def _is_orphaned_llama_server(pid: int) -> bool:
        """True when *pid* is a llama-server with no living parent.

        Processes are spawned with ``start_new_session=True``: while the
        owning application is alive the server keeps it as parent; when the
        app crashes the server is reparented to PID 1 (launchd/init) and is
        safe to reclaim.
        """
        try:
            result = subprocess.run(
                ["ps", "-p", str(pid), "-o", "ppid=,comm="],
                capture_output=True, text=True, timeout=3,
            )
        except (OSError, subprocess.TimeoutExpired):
            return False
        parts = result.stdout.split()
        if len(parts) < 2 or "llama" not in parts[-1].lower():
            return False
        try:
            return int(parts[0]) == 1
        except ValueError:
            return False

    @staticmethod
    def _wait_early_exit(proc: subprocess.Popen, seconds: float) -> bool:
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                return True
            time.sleep(0.2)
        return False

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
    def _terminate(instance: _ServerInstance) -> None:
        proc = instance.proc
        if proc.poll() is not None:
            return
        try:
            proc.terminate()
        except OSError:
            return
        try:
            proc.wait(timeout=_STOP_GRACE_SECONDS)
        except subprocess.TimeoutExpired:
            try:
                proc.kill()
            except OSError:
                pass
            try:
                proc.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                pass

    def _register_shutdown_hook(self) -> None:
        if self._shutdown_hook_registered:
            return
        atexit.register(self.shutdown)
        self._shutdown_hook_registered = True

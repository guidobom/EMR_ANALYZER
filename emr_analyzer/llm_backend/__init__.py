"""Local LLM backends managed by EMR Analyzer.

The application spawns its own ``llama-server`` (llama.cpp) processes instead
of depending on an externally managed Ollama service.  The server speaks an
OpenAI-compatible API on the loopback interface; model files are plain GGUF
files listed in ``~/.emr_analyzer/models/index.json``.
"""

from __future__ import annotations

import threading

from .backend import LlamaBackend, get_backend as _get_llama_backend
from .server_manager import BackendError, ServerKey, ServerManager
from .vllm_backend import VllmBackend, list_cached_vllm_models
from .vllm_server_manager import VllmServerKey, VllmServerManager

_vllm_backend: VllmBackend | None = None
_vllm_lock = threading.Lock()


def get_backend(name: str = "llama_cpp"):
    """Return the process-wide singleton for the selected backend."""
    normalized = str(name or "llama_cpp").strip().casefold()
    if normalized == "llama_cpp":
        return _get_llama_backend()
    if normalized == "vllm":
        global _vllm_backend
        with _vllm_lock:
            if _vllm_backend is None:
                _vllm_backend = VllmBackend()
            return _vllm_backend
    raise ValueError(f"Backend LLM non supportato: {name}")


def shutdown_all_backends() -> None:
    """Stop every app-owned local model server, regardless of backend."""
    _get_llama_backend().shutdown()
    with _vllm_lock:
        backend = _vllm_backend
    if backend is not None:
        backend.shutdown()

__all__ = [
    "BackendError",
    "LlamaBackend",
    "VllmBackend",
    "ServerKey",
    "ServerManager",
    "VllmServerKey",
    "VllmServerManager",
    "get_backend",
    "list_cached_vllm_models",
    "shutdown_all_backends",
]

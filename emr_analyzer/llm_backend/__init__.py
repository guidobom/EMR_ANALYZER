"""llama.cpp backend: app-managed llama-server child processes.

The application spawns its own ``llama-server`` (llama.cpp) processes instead
of depending on an externally managed Ollama service.  The server speaks an
OpenAI-compatible API on the loopback interface; model files are plain GGUF
files listed in ``~/.emr_analyzer/models/index.json``.
"""

from __future__ import annotations

from .backend import LlamaBackend, get_backend
from .server_manager import BackendError, ServerKey, ServerManager

__all__ = [
    "BackendError",
    "LlamaBackend",
    "ServerKey",
    "ServerManager",
    "get_backend",
]

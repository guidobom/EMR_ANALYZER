"""High-level llama.cpp backend consumed by :class:`LlmClient`.

Maps configured model names to GGUF files and to the server key
``(gguf, context, parallel slots)``, spawns the app's own llama-server
processes, and speaks the OpenAI-compatible API (``/v1/chat/completions``,
``/health``, ``/props``, ``/slots``) over httpx.

Every method takes the caller's configuration object (an
:class:`~emr_analyzer.settings.LLMRoleConfig` or a duck-typed equivalent
with ``model``, ``context_length`` and ``parallel_workers`` attributes);
the key derived from it decides which server process serves the call.
"""

from __future__ import annotations

import threading

import httpx

from ..config import (
    LLAMA_SERVER_HTTP_TIMEOUT,
    LLAMA_SERVER_LOAD_TIMEOUT,
)
from . import model_store
from .server_manager import ServerKey, ServerManager

# KV-cache + working-memory estimate per token of TOTAL context, in bytes.
# q8_0 KV cache for a ~14B model ≈ 80 KB/token; 100 KB keeps a safety
# margin (the old Ollama-era constant was 50 KB with f16 on smaller models).
_BYTES_PER_CONTEXT_TOKEN = 100_000

# RAM reserved for the OS, the GUI and the extraction pipeline.
_OS_RESERVE_BYTES = 2 * 1024 ** 3


class LlamaBackend:
    """Chat completions against the app-managed llama-server(s)."""

    def __init__(
        self,
        manager: ServerManager | None = None,
        load_timeout: float = LLAMA_SERVER_LOAD_TIMEOUT,
    ) -> None:
        self._manager = manager or ServerManager()
        self._load_timeout = load_timeout
        self._key_cache: dict[tuple, ServerKey] = {}
        self._cache_lock = threading.Lock()
        # Generous read timeout: long clinical prompts on the 14B model can
        # legitimately take many minutes, especially with parallel workers.
        self._http = httpx.Client(
            timeout=httpx.Timeout(
                LLAMA_SERVER_HTTP_TIMEOUT, read=LLAMA_SERVER_HTTP_TIMEOUT
            )
        )

    # -- model directory ------------------------------------------------------

    def list_models(self) -> list[str]:
        return model_store.list_models()

    def model_info(self, name: str) -> dict | None:
        return model_store.resolve(name)

    def usable(self) -> bool:
        """Backend usable: binary present and at least one local model."""
        return self._manager.usable() and bool(self.list_models())

    # -- key derivation -------------------------------------------------------

    def key_for(self, config) -> ServerKey:
        """The server key serving *config*'s model/context/worker settings."""
        cache_key = (
            str(getattr(config, "model", "")),
            int(getattr(config, "context_length", 32768) or 32768),
            int(getattr(config, "parallel_workers", 1) or 1),
            bool(getattr(config, "speculative_decoding", False)),
        )
        with self._cache_lock:
            cached = self._key_cache.get(cache_key)
            if cached is not None:
                return cached
        key = self._derive_key(config)
        with self._cache_lock:
            self._key_cache[cache_key] = key
        return key

    def _derive_key(self, config) -> ServerKey:
        entry = model_store.resolve(config.model)
        if entry is None:
            raise KeyError(
                f"Modello non trovato tra i GGUF locali: {config.model}"
            )
        np = int(getattr(config, "parallel_workers", 1) or 1)
        ctx = int(getattr(config, "context_length", 32768) or 32768)
        total_ram = None
        try:
            import psutil
            total_ram = psutil.virtual_memory().total
        except Exception:
            pass
        if total_ram is not None:
            np = min(np, self._memory_safe_np(
                entry.get("size_bytes", 0), ctx, total_ram
            ))
        return ServerKey(
            gguf_path=entry["file"],
            ctx_size=ctx,
            np=max(1, np),
            speculative_mode=(
                "ngram-cache"
                if bool(getattr(config, "speculative_decoding", False))
                else "none"
            ),
        )

    @staticmethod
    def _memory_safe_np(
        model_size_bytes: int, ctx_size: int, total_ram_bytes: int
    ) -> int:
        per_slot = ctx_size * _BYTES_PER_CONTEXT_TOKEN
        headroom = total_ram_bytes - model_size_bytes - _OS_RESERVE_BYTES
        if per_slot <= 0 or headroom <= 0:
            return 1
        return max(1, int(headroom // per_slot))

    # -- server lifecycle ----------------------------------------------------

    def ensure(self, config, progress_cb=None) -> str:
        """Spawn (or reuse) the server for *config*; returns its base URL."""
        return self._manager.ensure(
            self.key_for(config),
            load_timeout=self._load_timeout,
            progress_cb=progress_cb,
        )

    def status(self, config) -> str | None:
        try:
            key = self.key_for(config)
        except KeyError:
            return None
        return self._manager.status(key)

    def runtime_identity(self, config) -> tuple[str, int, int, str]:
        """Stable identity of the physical server used by *config*.

        Generation parameters such as temperature and output length are
        request-scoped.  Only the GGUF file, per-slot context and slot count
        decide whether two logical roles can share one llama-server process.
        """
        key = self.key_for(config)
        return key.gguf_path, key.ctx_size, key.np, key.speculative_mode

    def stop_config(self, config) -> bool:
        """Stop only the exact runtime selected by *config*.

        Unlike :meth:`stop_model`, this does not terminate other servers
        using the same GGUF with a different context or slot count.
        """
        try:
            key = self.key_for(config)
        except KeyError:
            return False
        return self._manager.stop(key)

    def stop_model(self, name: str) -> bool:
        """Stop every server running the model *name*; True if any stopped."""
        entry = model_store.resolve(name)
        if entry is None:
            return False
        stopped = False
        for key in self._manager.running_keys():
            if key.gguf_path == entry["file"]:
                stopped = self._manager.stop(key) or stopped
        return stopped

    def stop_all(self) -> int:
        return self._manager.stop_all()

    def running_model_names(self) -> list[str]:
        """Friendly names of the models with a running server process."""
        running_files = {
            key.gguf_path for key in self._manager.running_keys()
        }
        names = []
        for name, entry in model_store.load_index().items():
            if entry.get("file") in running_files:
                names.append(name)
        return names

    def shutdown(self) -> None:
        self._manager.shutdown()
        try:
            self._http.close()
        except Exception:
            pass

    # -- HTTP API ------------------------------------------------------------

    def chat(
        self,
        config,
        messages: list[dict],
        *,
        temperature: float = 0.1,
        top_p: float = 0.9,
        top_k: int = 40,
        seed: int = 42,
        max_tokens: int = 4096,
        response_format: dict | None = None,
    ) -> dict:
        """One non-streaming chat completion; returns
        ``{"content": str, "finish_reason": str}``."""
        base_url = self.ensure(config)
        payload: dict = {
            "model": str(getattr(config, "model", "")),
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "seed": seed,
            "max_tokens": max_tokens,
            # The app never wants the thinking channel (think=False with
            # Ollama); belt and suspenders together with the -rea off flag.
            "reasoning_effort": "none",
        }
        if response_format is not None:
            payload["response_format"] = response_format
        try:
            response = self._http.post(
                f"{base_url}/v1/chat/completions", json=payload
            )
        except httpx.HTTPError as exc:
            raise RuntimeError(
                f"Errore di comunicazione con llama-server: {exc}"
            ) from exc
        if response.status_code != 200:
            detail = response.text[:300] if response.text else ""
            raise RuntimeError(
                f"llama-server ha risposto {response.status_code}: {detail}"
            )
        data = response.json()
        choices = data.get("choices") or []
        if not choices:
            raise RuntimeError("llama-server ha restituito una risposta vuota")
        choice = choices[0]
        message = choice.get("message") or {}
        return {
            "content": str(message.get("content") or ""),
            "finish_reason": choice.get("finish_reason") or "stop",
            "usage": dict(data.get("usage") or {}),
            "timings": dict(data.get("timings") or {}),
        }

    def slots(self, config) -> list[dict]:
        """Per-slot runtime state of the server for *config*."""
        base_url = self._manager.base_url(self.key_for(config))
        if base_url is None:
            return []
        try:
            # /slots is a local, lightweight status endpoint.  A short
            # timeout keeps periodic GUI refreshes from freezing the dialog
            # when a child process is unhealthy.
            response = self._http.get(f"{base_url}/slots", timeout=1.0)
        except httpx.HTTPError:
            return []
        if response.status_code != 200:
            return []
        return response.json()

    def props(self, config) -> dict:
        """Server properties (total slots, generation defaults) for *config*."""
        base_url = self._manager.base_url(self.key_for(config))
        if base_url is None:
            return {}
        try:
            response = self._http.get(f"{base_url}/props", timeout=1.0)
        except httpx.HTTPError:
            return {}
        if response.status_code != 200:
            return {}
        return response.json()


_backend: LlamaBackend | None = None
_backend_lock = threading.Lock()


def get_backend() -> LlamaBackend:
    """Process-wide singleton backend."""
    global _backend
    with _backend_lock:
        if _backend is None:
            _backend = LlamaBackend()
        return _backend

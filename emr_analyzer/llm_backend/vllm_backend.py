"""Optional vLLM backend for Linux/CUDA systems such as DGX Spark."""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import threading

import httpx

from ..config import VLLM_SERVER_HTTP_TIMEOUT, VLLM_SERVER_LOAD_TIMEOUT
from .vllm_server_manager import VllmServerKey, VllmServerManager


def _hub_cache_root() -> Path:
    explicit = str(os.environ.get("HF_HUB_CACHE") or "").strip()
    if explicit:
        return Path(explicit).expanduser()
    hf_home = str(os.environ.get("HF_HOME") or "").strip()
    if hf_home:
        return Path(hf_home).expanduser() / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


def _cached_repo_dir(model: str) -> Path | None:
    clean = str(model or "").strip()
    if not clean or "/" not in clean:
        return None
    candidate = _hub_cache_root() / ("models--" + clean.replace("/", "--"))
    return candidate if candidate.is_dir() else None


def _latest_snapshot(repo_dir: Path) -> Path | None:
    snapshots = repo_dir / "snapshots"
    if not snapshots.is_dir():
        return None
    candidates = [path for path in snapshots.iterdir() if path.is_dir()]
    if not candidates:
        return None
    candidates.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return candidates[0]


def _has_local_weights(model_path: Path) -> bool:
    """True only when the local snapshot contains actual model weights."""
    if model_path.is_file():
        return model_path.suffix.casefold() in {
            ".safetensors", ".bin", ".pt", ".pth", ".gguf"
        }
    if not model_path.is_dir():
        return False
    weight_suffixes = {".safetensors", ".bin", ".pt", ".pth", ".gguf"}
    try:
        return any(
            path.is_file() and path.suffix.casefold() in weight_suffixes
            for path in model_path.rglob("*")
        )
    except OSError:
        return False


def list_cached_vllm_models() -> list[str]:
    """Return cached repositories that declare a text-generation model."""
    root = _hub_cache_root()
    if not root.is_dir():
        return []
    names = []
    for path in root.glob("models--*--*"):
        if not path.is_dir() or _latest_snapshot(path) is None:
            continue
        name = path.name.removeprefix("models--").replace("--", "/")
        info = resolve_vllm_model(name)
        if (
            info is not None
            and info.get("chat_capable")
            and info.get("weights_present")
        ):
            names.append(name)
    return sorted(set(names), key=str.casefold)


def resolve_vllm_model(name: str) -> dict | None:
    """Resolve only already-local models; never query Hugging Face."""
    clean = str(name or "").strip()
    if not clean:
        return None
    requested_path = Path(clean).expanduser()
    if requested_path.exists():
        source = requested_path.resolve()
    else:
        repo = _cached_repo_dir(clean)
        source = _latest_snapshot(repo) if repo is not None else None
        if source is None:
            return None

    config_path = source / "config.json" if source.is_dir() else None
    config: dict = {}
    if config_path is not None and config_path.is_file():
        try:
            loaded = json.loads(config_path.read_text(encoding="utf-8"))
            config = loaded if isinstance(loaded, dict) else {}
        except (OSError, ValueError, TypeError):
            config = {}
    text_config = config.get("text_config")
    if isinstance(text_config, dict):
        merged = {**config, **text_config}
    else:
        merged = config
    architectures = merged.get("architectures") or []
    architecture = architectures[0] if isinstance(architectures, list) and architectures else ""
    auto_map = merged.get("auto_map") or {}
    architecture_names = (
        [str(item) for item in architectures]
        if isinstance(architectures, list) else []
    )
    chat_capable = any(
        marker in name
        for name in architecture_names
        for marker in (
            "ForCausalLM",
            "ForConditionalGeneration",
            "ForSeq2SeqLM",
        )
    ) or any(
        marker in str(key)
        for key in (auto_map if isinstance(auto_map, dict) else {})
        for marker in ("AutoModelForCausalLM", "AutoModelForSeq2SeqLM")
    )
    maximum = (
        merged.get("max_position_embeddings")
        or merged.get("model_max_length")
        or merged.get("n_positions")
        or merged.get("seq_length")
    )
    try:
        maximum = int(maximum) if maximum is not None else None
    except (TypeError, ValueError):
        maximum = None
    return {
        "name": clean,
        "file": str(source),
        "source": "huggingface_cache" if not requested_path.exists() else "local",
        "architecture": str(merged.get("model_type") or architecture or ""),
        "max_context_length": maximum,
        "size_bytes": None,
        "chat_capable": chat_capable,
        "weights_present": _has_local_weights(source),
    }


class VllmBackend:
    """Chat completions through app-owned, loopback-only vLLM servers."""

    backend_name = "vllm"

    def __init__(
        self,
        manager: VllmServerManager | None = None,
        load_timeout: float = VLLM_SERVER_LOAD_TIMEOUT,
    ) -> None:
        self._manager = manager or VllmServerManager()
        self._load_timeout = load_timeout
        self._key_cache: dict[tuple, VllmServerKey] = {}
        self._cache_lock = threading.Lock()
        self._active_requests: dict[VllmServerKey, int] = {}
        self._active_lock = threading.Lock()
        self._http = httpx.Client(
            timeout=httpx.Timeout(
                VLLM_SERVER_HTTP_TIMEOUT, read=VLLM_SERVER_HTTP_TIMEOUT
            )
        )

    def list_models(self) -> list[str]:
        return list_cached_vllm_models()

    def model_info(self, name: str) -> dict | None:
        info = resolve_vllm_model(name)
        return (
            info
            if (
                info is not None
                and info.get("chat_capable")
                and info.get("weights_present")
            )
            else None
        )

    def usable(self) -> bool:
        return self._manager.usable() and bool(self.list_models())

    def key_for(self, config) -> VllmServerKey:
        raw = (
            str(getattr(config, "model", "")),
            int(getattr(config, "context_length", 32768) or 32768),
            int(getattr(config, "parallel_workers", 1) or 1),
            int(getattr(config, "vllm_tensor_parallel_size", 1) or 1),
            str(getattr(config, "vllm_dtype", "auto") or "auto"),
            round(float(getattr(config, "vllm_gpu_memory_utilization", 0.9) or 0.9), 3),
            str(getattr(config, "vllm_quantization", "") or ""),
            bool(getattr(config, "vllm_trust_remote_code", False)),
            bool(getattr(config, "vllm_enforce_eager", False)),
        )
        with self._cache_lock:
            cached = self._key_cache.get(raw)
            if cached is not None:
                return cached
        if self.model_info(raw[0]) is None:
            raise KeyError(
                f"Modello vLLM non presente localmente: {raw[0]}"
            )
        key = VllmServerKey(
            model=raw[0],
            ctx_size=max(512, raw[1]),
            max_num_seqs=max(1, raw[2]),
            tensor_parallel_size=max(1, raw[3]),
            dtype=raw[4],
            gpu_memory_utilization=min(0.99, max(0.05, raw[5])),
            quantization=raw[6],
            trust_remote_code=raw[7],
            enforce_eager=raw[8],
        )
        with self._cache_lock:
            self._key_cache[raw] = key
        return key

    def ensure(self, config, progress_cb=None) -> str:
        return self._manager.ensure(
            self.key_for(config),
            load_timeout=self._load_timeout,
            progress_cb=progress_cb,
        )

    def status(self, config) -> str | None:
        try:
            return self._manager.status(self.key_for(config))
        except KeyError:
            return None

    def runtime_identity(self, config) -> tuple:
        key = self.key_for(config)
        return (
            f"vllm:{key.model}",
            key.ctx_size,
            key.max_num_seqs,
            "vllm",
            key.tensor_parallel_size,
            key.dtype,
            key.gpu_memory_utilization,
            key.quantization,
            key.trust_remote_code,
            key.enforce_eager,
        )

    def stop_config(self, config) -> bool:
        try:
            return self._manager.stop(self.key_for(config))
        except KeyError:
            return False

    def stop_other_runtimes(self, config) -> int:
        """Stop every vLLM runtime except the selected pipeline runtime."""
        try:
            retained = self.key_for(config)
        except KeyError:
            return 0
        stopped = 0
        for key in self._manager.running_keys():
            if key != retained:
                stopped += int(self._manager.stop(key))
        return stopped

    def stop_model(self, name: str) -> bool:
        clean = str(name or "").strip()
        stopped = False
        for key in self._manager.running_keys():
            if key.model == clean:
                stopped = self._manager.stop(key) or stopped
        return stopped

    def stop_all(self) -> int:
        return self._manager.stop_all()

    def running_model_names(self) -> list[str]:
        return sorted({key.model for key in self._manager.running_keys()})

    def shutdown(self) -> None:
        self._manager.shutdown()
        try:
            self._http.close()
        except Exception:
            pass

    @staticmethod
    def structured_response_format(schema: dict) -> dict:
        return {
            "type": "json_schema",
            "json_schema": {
                "name": "emr_analyzer_response",
                "strict": True,
                "schema": schema,
            },
        }

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
        key = self.key_for(config)
        base_url = self.ensure(config)
        payload: dict = {
            "model": key.model,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "seed": seed,
            "max_tokens": max_tokens,
            # Qwen and compatible templates use this to omit the reasoning
            # channel.  Templates that do not expose the option ignore it.
            "chat_template_kwargs": {"enable_thinking": False},
        }
        if response_format is not None:
            payload["response_format"] = response_format
        with self._active_lock:
            self._active_requests[key] = self._active_requests.get(key, 0) + 1
        try:
            try:
                response = self._http.post(
                    f"{base_url}/v1/chat/completions", json=payload
                )
            except httpx.HTTPError as exc:
                raise RuntimeError(
                    f"Errore di comunicazione con vLLM: {exc}"
                ) from exc
        finally:
            with self._active_lock:
                remaining = self._active_requests.get(key, 1) - 1
                if remaining > 0:
                    self._active_requests[key] = remaining
                else:
                    self._active_requests.pop(key, None)
        if response.status_code != 200:
            detail = response.text[:500] if response.text else ""
            raise RuntimeError(
                f"vLLM ha risposto {response.status_code}: {detail}"
            )
        data = response.json()
        choices = data.get("choices") or []
        if not choices:
            raise RuntimeError("vLLM ha restituito una risposta vuota")
        choice = choices[0]
        message = choice.get("message") or {}
        return {
            "content": str(message.get("content") or ""),
            "finish_reason": choice.get("finish_reason") or "stop",
            "usage": dict(data.get("usage") or {}),
            "timings": {},
        }

    def slots(self, config) -> list[dict]:
        try:
            key = self.key_for(config)
        except KeyError:
            return []
        if self._manager.status(key) != "running":
            return []
        with self._active_lock:
            active = self._active_requests.get(key, 0)
        base_url = self._manager.base_url(key)
        if base_url:
            try:
                response = self._http.get(f"{base_url}/metrics", timeout=1.0)
                if response.status_code == 200:
                    match = re.search(
                        r"^vllm:num_requests_running(?:\{[^}]*\})?\s+([0-9.]+)",
                        response.text,
                        re.MULTILINE,
                    )
                    if match:
                        active = max(active, int(float(match.group(1))))
            except (httpx.HTTPError, ValueError):
                pass
        return [
            {
                "id": index,
                "n_ctx": key.ctx_size,
                "is_processing": index < active,
                "backend": "vllm",
            }
            for index in range(key.max_num_seqs)
        ]

    def props(self, config) -> dict:
        try:
            key = self.key_for(config)
        except KeyError:
            return {}
        return {
            "backend": "vllm",
            "total_slots": key.max_num_seqs,
            "max_model_len": key.ctx_size,
            "tensor_parallel_size": key.tensor_parallel_size,
            "dtype": key.dtype,
            "gpu_memory_utilization": key.gpu_memory_utilization,
        }

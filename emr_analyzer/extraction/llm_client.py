"""Client for configurable local models served by llama.cpp or vLLM.

Generation goes through an app-managed loopback server
(:mod:`emr_analyzer.llm_backend`); no external Ollama service is involved.
The document-normalization path uses ``generate_text``; the structured
methods are retained only for the separate Clinical State layer.
"""

import json
import re
import threading
import time

from ..config import DEFAULT_LLM_MODEL_NAME
from ..llm_backend import get_backend
from ..settings import LLMRoleConfig
from ..prompt_catalog import load_prompt
from .golden_fewshot import (
    DISCHARGE_ALLOWED_CATEGORIES,
    format_examples_section,
)


class OutputLimitError(RuntimeError):
    """The server stopped because the request-specific output cap was hit."""

    def __init__(self, max_tokens: int, partial_content: str = ""):
        self.max_tokens = int(max_tokens)
        self.partial_content = partial_content
        super().__init__(
            "Il modello ha raggiunto il limite di token in output "
            f"(max_output_tokens={self.max_tokens})"
        )


class LlmClient:
    """
    Client for configurable local models served by llama.cpp.

    The document-normalization path uses ``generate_text``; the structured
    methods are retained only for the separate Clinical State layer.
    """

    def __init__(self, base_url: str | None = None,
                 model: str = DEFAULT_LLM_MODEL_NAME,
                 config: LLMRoleConfig | None = None,
                 backend=None,
                 instance_id: int = 0):
        # ``base_url`` is kept for signature compatibility with older
        # callers; the llama.cpp backend owns its server URLs.
        self.base_url = str(base_url or "").rstrip("/")
        self.model = config.model if config is not None else model
        self.backend_type = (
            config.backend if config is not None else "llama_cpp"
        )
        self.temperature = config.temperature if config is not None else 0.1
        self.context_length = (
            config.context_length if config is not None else 32768
        )
        self.max_output_tokens = (
            config.max_output_tokens if config is not None else 4096
        )
        self.top_p = config.top_p if config is not None else 0.9
        self.top_k = config.top_k if config is not None else 40
        self.seed = config.seed if config is not None else 42
        self.thinking_enabled = config.thinking_enabled if config is not None else False
        self.keep_alive_minutes = (
            config.keep_alive_minutes if config is not None else 10
        )
        # Worker slots for this role's server; llama-server is spawned with
        # -np = this value (capped by available RAM in the backend).
        self.parallel_workers = (
            config.parallel_workers if config is not None else 1
        )
        self.speculative_decoding = bool(
            config.speculative_decoding if config is not None else False
        )
        self.vllm_dtype = (
            config.vllm_dtype if config is not None else "auto"
        )
        self.vllm_gpu_memory_utilization = (
            config.vllm_gpu_memory_utilization
            if config is not None else 0.85
        )
        self.vllm_tensor_parallel_size = (
            config.vllm_tensor_parallel_size if config is not None else 1
        )
        self.vllm_quantization = (
            config.vllm_quantization if config is not None else ""
        )
        self.vllm_trust_remote_code = bool(
            config.vllm_trust_remote_code if config is not None else False
        )
        self.vllm_enforce_eager = bool(
            config.vllm_enforce_eager if config is not None else False
        )
        self.backend = (
            backend if backend is not None else get_backend(self.backend_type)
        )
        # Instance discriminator (default 0): sibling clients with a nonzero
        # instance_id resolve to a distinct llama-server process (multi-patient
        # irAE queue).  The backend reads it off the client object.
        self.instance_id = int(instance_id or 0)
        self._available = None  # Lazy check
        # Generation metadata is request-local: one LlmClient is deliberately
        # shared by the parallel registry workers.
        self._generation_local = threading.local()

    @property
    def keep_alive(self) -> str:
        return f"{self.keep_alive_minutes}m"

    def for_instance(self, instance_id: int) -> "LlmClient":
        """A sibling client pinned to a distinct server process (instance k).

        The clone shares the same model, generation parameters and backend
        singleton (so all instances share one port allocator); only
        ``instance_id`` changes, which resolves to a different ``ServerKey``
        and therefore a different llama-server process.
        """
        instance_id = int(instance_id or 0)
        if instance_id == self.instance_id:
            return self
        clone = object.__new__(type(self))
        clone.__dict__.update(self.__dict__)
        clone.instance_id = instance_id
        # Fresh per-thread generation metadata and lazy availability check.
        clone._generation_local = threading.local()
        clone._available = None
        return clone

    def retain_only_this_runtime(self) -> int:
        """Unload app-owned models from inactive pipeline stages."""
        from ..llm_backend import retain_only_runtime

        return retain_only_runtime(self.backend, self)

    @classmethod
    def list_available_models(
        cls,
        base_url: str | None = None,
        backend: str = "llama_cpp",
    ) -> list[str]:
        """Return models already installed for the selected local backend."""
        return get_backend(backend).list_models()

    @property
    def is_available(self) -> bool:
        """Check whether this model exists in the selected local store."""
        if self._available is None:
            try:
                self._available = (
                    self.backend.model_info(self.model) is not None
                )
            except Exception:
                self._available = False
        return self._available

    @property
    def server_available(self) -> bool:
        """Return whether the selected local backend is usable.

        True when the selected engine is installed and at least one local
        model is registered; the server process itself is spawned
        lazily, so it need not be running yet.
        """
        try:
            return self.backend.usable()
        except Exception:
            return False

    def loaded_model_info(self) -> dict | None:
        """Return runtime information if the server for this model is up."""
        try:
            slots = self.backend.slots(self)
        except Exception:
            slots = []
        runtime = self._find_loaded_model(slots, self.model)
        if runtime is None:
            return None
        entry = self.backend.model_info(self.model) or {}
        size = entry.get("size_bytes")
        runtime["size"] = size
        runtime["size_vram"] = size
        runtime["expires_at"] = ""
        runtime["backend"] = self.backend_type
        return runtime

    def runtime_identity(self) -> tuple:
        """Return the stable physical-runtime identity for this client."""
        return self.backend.runtime_identity(self)

    @classmethod
    def unload_runtimes(
        cls, configs: list[LLMRoleConfig] | tuple[LLMRoleConfig, ...]
    ) -> dict:
        """Stop exact configured runtimes without loading them first."""
        unique: dict[tuple[str, tuple], LLMRoleConfig] = {}
        errors: dict[str, str] = {}
        for config in configs:
            if not config.model:
                continue
            backend_name = getattr(config, "backend", "llama_cpp")
            backend = get_backend(backend_name)
            try:
                identity = backend.runtime_identity(config)
            except Exception as exc:
                errors[config.model] = str(exc)
                continue
            unique.setdefault((backend_name, identity), config)

        unloaded = []
        not_loaded = []
        for (backend_name, identity), config in unique.items():
            backend = get_backend(backend_name)
            label = (
                f"{config.model} · {backend_name} · ctx {identity[1]} · "
                f"{identity[2]} slot"
            )
            try:
                if backend.stop_config(config):
                    unloaded.append(label)
                else:
                    not_loaded.append(label)
            except Exception as exc:
                errors[label] = str(exc)
        return {
            "unloaded": unloaded,
            "not_loaded": not_loaded,
            "errors": errors,
        }

    @classmethod
    def unload_models(
        cls,
        model_names: list[str] | tuple[str, ...] | None = None,
        base_url: str | None = None,
    ) -> dict:
        """Stop the servers of selected (or all) local models.

        The llama.cpp backend keeps a model resident until its server
        process is stopped, so unloading means terminating the process.
        The method is idempotent and never loads a model merely to unload it.
        """
        backends = {
            "llama_cpp": get_backend("llama_cpp"),
            "vllm": get_backend("vllm"),
        }
        loaded_by_backend = {
            name: backend.running_model_names()
            for name, backend in backends.items()
        }
        loaded = sorted({
            model
            for models in loaded_by_backend.values()
            for model in models
        })

        if model_names is None:
            targets = loaded
        else:
            requested = {
                cls._normalize_model_name(name)
                for name in model_names if str(name or "").strip()
            }
            targets = [
                name for name in loaded
                if cls._normalize_model_name(name) in requested
            ]

        unloaded = []
        errors = {}
        for model_name in targets:
            stopped = False
            for backend_name, backend in backends.items():
                if model_name not in loaded_by_backend[backend_name]:
                    continue
                try:
                    stopped = backend.stop_model(model_name) or stopped
                except Exception as exc:
                    errors[f"{backend_name}:{model_name}"] = str(exc)
            if stopped:
                unloaded.append(model_name)
            elif not any(key.endswith(f":{model_name}") for key in errors):
                errors[model_name] = "processo già terminato"

        return {
            "unloaded": unloaded,
            "not_loaded": (
                []
                if model_names is None else [
                    name for name in model_names
                    if cls._normalize_model_name(name)
                    not in {
                        cls._normalize_model_name(loaded_name)
                        for loaded_name in loaded
                    }
                ]
            ),
            "errors": errors,
        }

    def model_capabilities(self) -> dict:
        """Read metadata from the selected backend's local model store."""
        entry = self.backend.model_info(self.model) or {}
        return {
            "model": self.model,
            "architecture": str(entry.get("architecture") or ""),
            "max_context_length": entry.get("max_context_length"),
            "capabilities": [],
        }

    @classmethod
    def _find_loaded_model(cls, response, requested_model: str) -> dict | None:
        """Normalize llama-server ``/slots`` lists and legacy ``/api/ps`` payloads."""
        if isinstance(response, list):
            # llama-server /slots: one entry per parallel slot.  The model
            # name is not in the payload; the caller supplies it.
            if not response:
                return None
            rows = [item for item in response if isinstance(item, dict)]
            first = rows[0] if rows else {}
            active_slots = sum(bool(item.get("is_processing")) for item in rows)
            return {
                "model": str(requested_model),
                "size": None,
                "size_vram": None,
                "context_length": cls._as_int(first.get("n_ctx")),
                "processing": active_slots > 0,
                "active_slots": active_slots,
                "slots": len(rows),
                "expires_at": "",
            }
        if hasattr(response, "models"):
            models = response.models
        elif isinstance(response, dict):
            models = response.get("models", [])
        else:
            models = []
        requested = cls._normalize_model_name(requested_model)
        for item in models:
            if isinstance(item, dict):
                getter = item.get
            else:
                getter = lambda key, default=None, obj=item: getattr(
                    obj, key, default
                )
            model_name = getter("model") or getter("name") or ""
            if cls._normalize_model_name(model_name) != requested:
                continue
            expires_at = getter("expires_at")
            return {
                "model": model_name,
                "size": cls._as_int(getter("size")),
                "size_vram": cls._as_int(getter("size_vram")),
                "context_length": cls._as_int(getter("context_length")),
                "expires_at": (
                    expires_at.isoformat()
                    if hasattr(expires_at, "isoformat") else str(expires_at or "")
                ),
            }
        return None

    def warmup(self, keep_alive: str | None = None) -> dict:
        """Load and test the model with a tiny, non-clinical request.

        ``keep_alive`` is accepted for backward compatibility and ignored:
        llama-server keeps the model resident until its process is stopped.
        """
        started = time.perf_counter()
        self.backend.ensure(self)
        result = self.backend.chat(
            self,
            messages=[{
                "role": "user",
                "content": "Test tecnico di disponibilità. Rispondi soltanto OK.",
            }],
            temperature=0.0,
            top_p=self.top_p,
            top_k=self.top_k,
            seed=self.seed,
            # Use the same context configured for real processing: the
            # server was spawned with it, so the first clinical call does
            # not have to restart a different runner.
            max_tokens=8,
        )
        content = result.get("content") or ""
        if not str(content).strip():
            raise RuntimeError("Il modello ha restituito una risposta di test vuota")
        runtime = self.loaded_model_info()
        if runtime is None:
            raise RuntimeError(
                "Il test ha risposto, ma llama-server non segnala il modello in memoria"
            )
        runtime["elapsed_seconds"] = time.perf_counter() - started
        runtime["test_response"] = str(content).strip()[:100]
        return runtime

    @staticmethod
    def _normalize_model_name(value: str) -> str:
        return str(value or "").strip().removesuffix(":latest")

    @staticmethod
    def _as_int(value) -> int | None:
        try:
            return int(value) if value is not None else None
        except (TypeError, ValueError):
            return None

    def _generate(self, prompt: str, system: str = "",
                  stream: bool = False,
                  response_format: dict | str | None = None,
                  seed: int | None = None,
                  temperature: float | None = None,
                  max_tokens: int | None = None) -> str:
        """Internal: chat completion against the app-managed llama-server.

        Thinking follows the pipeline configuration. Only the final answer
        is consumed; the backend keeps reasoning separate from clinical JSON.

        ``seed``/``temperature`` override the configured generation
        parameters for this single call when given (used by corrective
        retries to escape deterministic failures); the client's configured
        values are never modified.
        """
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        request_max_tokens = max(
            1, int(
                self.max_output_tokens if max_tokens is None else max_tokens
            )
        )
        self._generation_local.metadata = {}
        self._generation_local.response_text = ''
        try:
            result = self.backend.chat(
                self,
                messages,
                temperature=(
                    self.temperature if temperature is None else temperature
                ),
                top_p=self.top_p,
                top_k=self.top_k,
                seed=self.seed if seed is None else seed,
                max_tokens=request_max_tokens,
                response_format=response_format,
            )
        except KeyError as exc:
            raise RuntimeError(
                f"Modello non trovato nell'archivio locale del backend "
                f"{self.backend_type}: {exc}"
            ) from exc
        self._generation_local.metadata = {
            "finish_reason": result.get("finish_reason") or "stop",
            "max_tokens": request_max_tokens,
            **dict(result.get("usage") or {}),
            **dict(result.get("timings") or {}),
        }
        self._generation_local.response_text = str(result.get('content') or '')
        if result.get("finish_reason") == "length":
            raise OutputLimitError(request_max_tokens, str(result.get("content") or ""))
        return str(result.get("content") or "")

    def count_tokens(self, text: str) -> int:
        counter = getattr(self.backend, "count_tokens", None)
        if callable(counter):
            return counter(text, self)
        # A byte upper bound is deliberately conservative when a backend
        # does not expose its tokenizer; do not guess Italian tokens/word.
        return len(text.encode("utf-8"))

    def count_prompt_tokens(self, prompt: str, system: str) -> int:
        counter = getattr(self.backend, "count_prompt_tokens", None)
        if callable(counter):
            return counter(prompt, system, self)
        return len((system + prompt).encode("utf-8")) + 256

    def generate_structured(self, prompt: str, system: str,
                            schema: dict,
                            *, max_tokens: int | None = None) -> dict:
        """Generate locally with a backend-enforced JSON schema.

        The request uses llama-server's ``json_object`` response format.
        Some server builds are known to silently ignore the schema, so a
        fence/brace JSON extraction is attempted before failing.
        """

        formatter = getattr(self.backend, "structured_response_format", None)
        response_format = (
            formatter(schema)
            if callable(formatter)
            else {"type": "json_object", "schema": schema}
        )
        response = self._generate(
            prompt, system,
            response_format=response_format,
            max_tokens=max_tokens,
        )
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", response, re.DOTALL)
            if match:
                return json.loads(match.group(0))
            raise

    def last_response_text(self) -> str:
        return str(getattr(self._generation_local, 'response_text', ''))

    def last_generation_metadata(self) -> dict:
        """Return metrics for the last request made by the current thread."""
        return dict(getattr(self._generation_local, "metadata", {}) or {})

    def extract_patient_identity(self, text: str) -> dict:
        """Extract the patient's identity fields from raw document text.

        Runs on the pre-anonymization text: the model must be able to read
        the actual name/CF/birth date so the result can confirm (or refute)
        the workspace attribution made by the deterministic routing.

        Returns a dict with keys ``name``, ``birth_date``, ``fiscal_code``
        and ``confidence``; an empty dict when the call fails or the model
        could not produce valid JSON.
        """
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "birth_date": {"type": "string"},
                "fiscal_code": {"type": "string"},
                "confidence": {"type": "number"},
            },
            "required": ["name", "birth_date", "fiscal_code"],
        }
        system_prompt = load_prompt("patient_identity_system")
        user_prompt = (
            load_prompt("patient_identity_task")
            + "\n\nTESTO:\n"
            + text[:4000]
            + "\n"
        )
        try:
            data = self.generate_structured(
                user_prompt, system_prompt, schema
            )
        except Exception:
            return {}
        if not isinstance(data, dict):
            return {}
        try:
            confidence = float(data.get("confidence", 0.5) or 0.5)
        except (TypeError, ValueError):
            confidence = 0.5
        return {
            "name": (data.get("name") or "").strip(),
            "birth_date": (data.get("birth_date") or "").strip(),
            "fiscal_code": (data.get("fiscal_code") or "").strip().upper(),
            "confidence": confidence,
        }

    def generate_text(self, prompt: str, system: str = "",
                      seed: int | None = None,
                      temperature: float | None = None) -> str:
        """Generate plain text without JSON/schema constraints.

        ``seed``/``temperature`` override the configured parameters for
        this single call when given (keyword-only, backward compatible).
        """
        response = self._generate(
            prompt, system, response_format=None,
            seed=seed, temperature=temperature,
        )
        if not str(response or "").strip():
            raise ValueError(
                "Il modello locale ha restituito una risposta vuota"
            )
        return str(response)

    # Clinical Timeline — strictly temporal extraction & deduplication
    # ------------------------------------------------------------------


    _TIMELINE_DEDUP_SCHEMA = {
        "type": "object",
        "properties": {
            "deduplicated_entries": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "entry_id": {"type": "string"},
                        "date_observed": {"type": "string"},
                        "date_resolved": {"type": ["string", "null"]},
                        "category": {"type": "string"},
                        "description": {"type": "string"},
                        "status": {"type": "string"},
                        "source_document_ids": {
                            "type": "array", "items": {"type": "string"},
                        },
                        "source_texts": {
                            "type": "array", "items": {"type": "string"},
                        },
                        "confidence": {"type": "number"},
                        "enrichment_text": {
                            "type": ["string", "null"],
                            "description": "Dettagli aggiuntivi dalle voci duplicate, o null",
                        },
                    },
                    "required": [
                        "entry_id", "date_observed", "category",
                        "description", "status", "confidence",
                    ],
                },
            },
            "removed_entry_ids": {
                "type": "array",
                "items": {"type": "string"},
            },
            "merge_map": {
                "type": "object",
                "additionalProperties": {"type": "string"},
            },
        },
        "required": ["deduplicated_entries", "removed_entry_ids", "merge_map"],
    }





    _DEDUP_BATCH_SIZE_MIN = 20
    _DEDUP_BATCH_SIZE_MAX = 120
    _DEDUP_OVERLAP = 5
    # Estimated token footprint of one compact dedup entry (~250 chars).
    _DEDUP_TOKENS_PER_ENTRY = 125
    # Instruction block + response reserve (tokens) per dedup call.
    _DEDUP_OVERHEAD_TOKENS = 2500

    def _dedup_batch_size(self) -> int:
        """Entries per LLM dedup call, sized to the configured context.

        The fixed 40-entry batches under-used the context window and turned
        large registries into many serial calls; a compact entry is ~125
        tokens, so a 32K context comfortably compares ~100+ entries at once.
        """
        output_reserve = max(2048, int(self.max_output_tokens * 0.5))
        budget = max(
            8000,
            self.context_length
            - self._DEDUP_OVERHEAD_TOKENS
            - output_reserve,
        )
        size = budget // self._DEDUP_TOKENS_PER_ENTRY
        return max(
            self._DEDUP_BATCH_SIZE_MIN,
            min(self._DEDUP_BATCH_SIZE_MAX, size),
        )

    def deduplicate_timeline(self, entries: list[dict]) -> dict:
        """Semantic dedup with context-sized batches processed in parallel.

        Sends ALL entries to the LLM using a compact format.  Entries are
        ordered by (date, category) so semantic duplicates cluster near
        each other; the registry is then split into overlapping batches
        sized to the context window and each batch is deduplicated with an
        independent LLM call — run in parallel across the server slots.
        The per-batch groups are finally merged with transitivity
        resolution.  The result is a list of groups:

        ``{"groups": [{"kept_id": ..., "merged_into_ids": [...],
                        "canonical_description": ...,
                        "date_observed": ...}]}``

        Entries that are NOT part of any group are left untouched (they are
        not removed) — the caller only needs to drop ``merged_into_ids`` and
        replace each ``kept_id``'s description with the canonical one.
        """
        if not entries:
            return {"groups": []}

        batch_size = self._dedup_batch_size()
        if len(entries) <= batch_size:
            return self._dedup_batch(entries)

        ordered = sorted(
            entries,
            key=lambda e: (e.get("date_observed") or "",
                           e.get("category") or ""),
        )

        overlap = min(self._DEDUP_OVERLAP, batch_size - 1)
        step = batch_size - overlap
        slices = []
        start = 0
        while start < len(ordered):
            end = min(start + batch_size, len(ordered))
            slices.append(ordered[start:end])
            if end == len(ordered):
                break
            start += step

        max_workers = max(1, min(
            int(getattr(self, "parallel_workers", 1) or 1),
            len(slices),
        ))
        if max_workers == 1 or len(slices) == 1:
            batch_results = [self._dedup_batch(s) for s in slices]
        else:
            from concurrent.futures import ThreadPoolExecutor

            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                batch_results = list(pool.map(self._dedup_batch, slices))

        # Global merge with transitivity resolution across parallel
        # batches: a survivor kept by one batch can be merged by another.
        survivor_groups: dict[str, dict] = {}
        for result in batch_results:
            for g in result.get("groups", []):
                kid = g.get("kept_id")
                if not kid:
                    continue
                merged = [
                    m for m in g.get("merged_into_ids", [])
                    if m and m != kid
                ]
                if kid in survivor_groups:
                    cur = survivor_groups[kid]
                    cur["merged_into_ids"] = list(dict.fromkeys(
                        cur.get("merged_into_ids", []) + merged
                    ))
                    if g.get("canonical_description"):
                        cur["canonical_description"] = g[
                            "canonical_description"
                        ]
                else:
                    survivor_groups[kid] = {
                        "kept_id": kid,
                        "merged_into_ids": list(dict.fromkeys(merged)),
                        "canonical_description": (
                            g.get("canonical_description") or ""
                        ),
                        "date_observed": g.get("date_observed") or "",
                        "category": g.get("category") or "",
                        "status": g.get("status") or "",
                    }
                # A prior survivor now merged into this group's kept entry:
                # fold its merged ids into the new survivor.  When two
                # batches keep different ends of the same pair, the group
                # processed later wins and absorbs the earlier one; the
                # self-reference filter below keeps merged lists clean.
                for mid in merged:
                    if mid in survivor_groups and mid != kid:
                        absorbed = survivor_groups.pop(mid)
                        survivor_groups[kid]["merged_into_ids"] = (
                            list(dict.fromkeys(
                                [
                                    m for m in (
                                        survivor_groups[kid][
                                            "merged_into_ids"
                                        ]
                                        + absorbed.get(
                                            "merged_into_ids", []
                                        )
                                    )
                                    if m != kid
                                ]
                            ))
                        )

        return {"groups": list(survivor_groups.values())}

    def _dedup_batch(self, entries: list[dict]) -> dict:
        """Deduplicate a single batch that fits in the context window."""
        import json as _json

        system_prompt = (
            "Sei un medico esperto di oncologia. Il tuo compito e' analizzare "
            "un registro clinico temporale ed eliminare le voci ridondanti "
            "che descrivono lo STESSO evento clinico, fondendo ogni gruppo "
            "di duplicati in un'unica voce canonica. "
            "Rispondi SOLO con JSON valido."
        )

        compact = []
        for e in entries:
            compact.append(_json.dumps({
                "id": e.get("entry_id", ""),
                "d": e.get("date_observed", ""),
                "dr": e.get("date_resolved"),
                "c": e.get("category", ""),
                "t": e.get("description", ""),
                "s": e.get("status", ""),
            }, ensure_ascii=False))
        entries_json = "[\n" + ",\n".join(compact) + "\n]"

        user_prompt = f"""Elimina i duplicati semantici da questo registro ({len(entries)} voci).

REGOLE:
1. Due voci sono DUPLICATE se descrivono lo STESSO evento clinico (stessa data/periodo e stesso contenuto clinico).
2. NON unire: eventi diversi, sospensione/ripresa di trattamento, diagnosi vs progressione, sintomi diversi.
3. Per ogni gruppo di duplicati scegli come kept_id la voce con la descrizione PIU' COMPLETA e la data PIU' PRECISA.
4. SCRIVI una canonical_description: una descrizione sintetica e clinicamente precisa che riassuma il contenuto del gruppo, usando SOLO informazioni presenti nelle voci del gruppo. NON inventare dosaggi, date, indicazioni o dettagli non espliciti.
   Esempio: 'inizia in data odierna terapia con nivolumab' + 'ha iniziato oggi nivolumab con intento adiuvante' -> 'Inizio nivolumab con intento adiuvante'.
5. merged_into_ids: gli entry_id delle voci fuse (tutte quelle del gruppo tranne kept_id).
6. date_observed: la data piu' precoce e piu' precisa tra quelle del gruppo.
7. Le voci che NON sono duplicati vanno OMESSE dai gruppi: restano nel registro senza alcuna modifica.

REGISTRO:
{entries_json}

Restituisci SOLO: ```json {{"groups": [{{"kept_id": "...", "merged_into_ids": [...], "canonical_description": "...", "date_observed": "YYYY-MM-DD", "category": "...", "status": "..."}}]}} ```
"""
        try:
            raw = self.generate_text(user_prompt, system_prompt)
            return self._parse_json_dedup(raw, entries)
        except Exception:
            return {"groups": []}

    @staticmethod
    def _parse_json_dedup(raw: str, fallback_entries: list[dict]) -> dict:
        """Parse the LLM dedup response (compact format: only removed IDs
        and optional enrichments).  Falls back gracefully on any error."""
        import json as _json
        import re as _re

        if not raw or not raw.strip():
            return {
                "removed_entry_ids": [],
                "enrichments": {},
            }

        text = raw.strip()

        # Extract from ```json fence
        m = _re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
        if m:
            text = m.group(1).strip()

        # Find JSON object boundaries
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            text = text[start:end + 1]

        try:
            parsed = _json.loads(text)
            if isinstance(parsed, dict):
                if "groups" in parsed and isinstance(parsed["groups"], list):
                    groups = []
                    for g in parsed["groups"]:
                        if not isinstance(g, dict):
                            continue
                        merged = g.get("merged_into_ids", [])
                        groups.append({
                            "kept_id": g.get("kept_id") or "",
                            "merged_into_ids": (
                                list(merged) if isinstance(merged, list)
                                else []
                            ),
                            "canonical_description": (
                                g.get("canonical_description") or ""
                            ),
                            "date_observed": g.get("date_observed") or "",
                            "category": g.get("category") or "",
                            "status": g.get("status") or "",
                        })
                    return {"groups": groups}

                # Backward compatibility with the old {removed_entry_ids,
                # enrichments} contract, so older stored outputs still parse.
                removed = parsed.get("removed_entry_ids", [])
                enrichments = parsed.get("enrichments", {})
                if "deduplicated_entries" in parsed:
                    return parsed
                return {
                    "removed_entry_ids": (
                        list(removed) if isinstance(removed, list) else []
                    ),
                    "enrichments": (
                        enrichments if isinstance(enrichments, dict) else {}
                    ),
                }
        except (_json.JSONDecodeError, ValueError):
            pass

        return {"groups": []}

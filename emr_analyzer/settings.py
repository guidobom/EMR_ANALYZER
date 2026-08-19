"""Local application settings store (never contains clinical data)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path

from .config import (
    BASE_DIR,
    CLINICAL_STATE_LLM_MODEL_NAME,
    DOCUMENT_LLM_MODEL_NAME,
    LLM_DEFAULT_CONTEXT_LENGTH,
)


SETTINGS_PATH = BASE_DIR / "settings.json"
MODEL_ROLES = {"document", "clinical_state"}


@dataclass(frozen=True)
class LLMRoleConfig:
    """Generation settings for one independently configured LLM role."""

    model: str
    temperature: float = 0.1
    context_length: int = LLM_DEFAULT_CONTEXT_LENGTH
    max_output_tokens: int = 4096
    top_p: float = 0.9
    top_k: int = 40
    seed: int = 42
    # Deprecated with the llama.cpp backend (the model stays resident until
    # its server process is stopped); kept for settings compatibility.
    keep_alive_minutes: int = 10
    parallel_workers: int = 1

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict, default: "LLMRoleConfig") -> "LLMRoleConfig":
        """Load a validated config while tolerating older/partial settings."""
        if not isinstance(payload, dict):
            return default

        def integer(name: str, minimum: int, maximum: int) -> int:
            try:
                value = int(payload.get(name, getattr(default, name)))
            except (TypeError, ValueError):
                return getattr(default, name)
            return min(maximum, max(minimum, value))

        def floating(name: str, minimum: float, maximum: float) -> float:
            try:
                value = float(payload.get(name, getattr(default, name)))
            except (TypeError, ValueError):
                return getattr(default, name)
            return min(maximum, max(minimum, value))

        model = payload.get("model", default.model)
        if not isinstance(model, str):
            model = default.model
        return cls(
            model=model.strip(),
            temperature=floating("temperature", 0.0, 2.0),
            context_length=integer("context_length", 512, 2_000_000),
            max_output_tokens=integer("max_output_tokens", 1, 262_144),
            top_p=floating("top_p", 0.0, 1.0),
            top_k=integer("top_k", 0, 1000),
            seed=integer("seed", -1, 2_147_483_647),
            keep_alive_minutes=integer("keep_alive_minutes", 0, 1440),
            parallel_workers=integer("parallel_workers", 1, 8),
        )


def _auto_workers(model_name: str, context_length: int) -> int:
    """Auto-detect the optimal number of parallel workers for a model."""
    try:
        from .utils.hardware import get_safe_max_workers
        return get_safe_max_workers(model_name, context_length)
    except Exception:
        return 1


def _resolve_legacy_model_name(model: str) -> str:
    """Map a legacy ``family:tag`` model name to a GGUF-index name.

    Settings written by older builds store Ollama-style names like
    ``qwen3:14b``; the llama.cpp backend lists GGUF files under
    ``family-tag`` names.  When the raw name is not in the index, resolve
    it through the model store so existing settings keep working.
    """
    raw = str(model or "").strip()
    if not raw:
        return raw
    try:
        from .llm_backend import model_store
        index = model_store.load_index()
        candidate = raw.removesuffix(":latest").replace(":", "-")
        if candidate in index:
            return candidate
        entry = model_store.resolve(raw)
        if entry is not None:
            for name, data in index.items():
                if data.get("file") == entry.get("file"):
                    return name
        return raw
    except Exception:
        return raw


def default_llm_configs() -> dict[str, LLMRoleConfig]:
    doc_cfg = LLMRoleConfig(model=DOCUMENT_LLM_MODEL_NAME)
    cs_cfg = LLMRoleConfig(model=CLINICAL_STATE_LLM_MODEL_NAME)
    doc_workers = _auto_workers(doc_cfg.model, doc_cfg.context_length)
    cs_workers = _auto_workers(cs_cfg.model, cs_cfg.context_length)
    return {
        "document": LLMRoleConfig(
            model=doc_cfg.model,
            context_length=doc_cfg.context_length,
            max_output_tokens=doc_cfg.max_output_tokens,
            temperature=doc_cfg.temperature,
            parallel_workers=doc_workers,
        ),
        "clinical_state": LLMRoleConfig(
            model=cs_cfg.model,
            context_length=cs_cfg.context_length,
            max_output_tokens=cs_cfg.max_output_tokens,
            temperature=cs_cfg.temperature,
            parallel_workers=cs_workers,
        ),
    }


def _read_payload(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except (OSError, ValueError, TypeError):
        return {}


def load_llm_configs(
    path: str | Path = SETTINGS_PATH,
) -> dict[str, LLMRoleConfig]:
    """Load role-specific models and generation settings.

    The former ``models`` mapping remains a supported migration source so
    existing user choices are not lost when upgrading.
    """
    defaults = default_llm_configs()
    payload = _read_payload(Path(path))
    legacy_models = payload.get("models", {})
    llm_payload = payload.get("llm", {})
    if not isinstance(legacy_models, dict):
        legacy_models = {}
    if not isinstance(llm_payload, dict):
        llm_payload = {}

    result = {}
    for role in MODEL_ROLES:
        default = defaults[role]
        role_payload = llm_payload.get(role, {})
        if not isinstance(role_payload, dict):
            role_payload = {}
        if "model" not in role_payload and isinstance(
            legacy_models.get(role), str
        ):
            role_payload = {
                **role_payload,
                "model": legacy_models[role],
            }
        config = LLMRoleConfig.from_dict(role_payload, default)
        config = replace(
            config, model=_resolve_legacy_model_name(config.model)
        )

        # Auto-detect workers if not explicitly set in the saved payload,
        # using the ACTUAL model and context (not the default values).
        if "parallel_workers" not in role_payload:
            actual_model = config.model or default.model
            actual_ctx = config.context_length
            config = replace(config, parallel_workers=_auto_workers(
                actual_model, actual_ctx
            ))

        result[role] = config
    return result


def save_llm_configs(
    configs: dict[str, LLMRoleConfig],
    path: str | Path = SETTINGS_PATH,
) -> None:
    """Atomically persist both role configurations and legacy model names."""
    if set(configs) != MODEL_ROLES:
        raise ValueError("Sono richieste le configurazioni document e clinical_state")
    if not all(isinstance(configs[role], LLMRoleConfig) for role in MODEL_ROLES):
        raise TypeError("Configurazione LLM non valida")

    settings_path = Path(path)
    payload = _read_payload(settings_path)
    payload["llm"] = {
        role: configs[role].to_dict() for role in sorted(MODEL_ROLES)
    }
    # Keep compatibility with older application builds.
    payload["models"] = {
        role: configs[role].model for role in sorted(MODEL_ROLES)
    }
    settings_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = settings_path.with_suffix(settings_path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(settings_path)


def load_model_assignments(path: str | Path = SETTINGS_PATH) -> dict[str, str]:
    """Backward-compatible view containing only the model assignments."""
    return {
        role: config.model for role, config in load_llm_configs(path).items()
    }


# ---------------------------------------------------------------------------
# Clinical query chat preferences (UI-only; never clinical data)
# ---------------------------------------------------------------------------


def load_chat_preferences(path: str | Path = SETTINGS_PATH) -> dict:
    """Persisted preferences of the clinical query chat panel."""
    payload = _read_payload(Path(path))
    chat = payload.get("chat", {})
    if not isinstance(chat, dict):
        chat = {}
    return {
        "use_conversation_context": bool(
            chat.get("use_conversation_context", False)
        ),
    }


def save_chat_preferences(
    prefs: dict, path: str | Path = SETTINGS_PATH
) -> None:
    """Persist chat preferences, preserving the LLM configuration."""
    settings_path = Path(path)
    payload = _read_payload(settings_path)
    payload["chat"] = {
        "use_conversation_context": bool(
            prefs.get("use_conversation_context", False)
        ),
    }
    settings_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = settings_path.with_suffix(settings_path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(settings_path)


def save_model_assignment(
    role: str,
    model_name: str,
    path: str | Path = SETTINGS_PATH,
) -> None:
    """Backward-compatible update of one model without losing its parameters."""
    if role not in MODEL_ROLES:
        raise ValueError(f"Ruolo modello non valido: {role}")
    configs = load_llm_configs(path)
    configs[role] = replace(configs[role], model=str(model_name or "").strip())
    save_llm_configs(configs, path)

"""Versioned external prompt catalog for every local LLM task."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Iterable

from .config import BASE_DIR


BUILTIN_PROMPT_DIR = Path(
    os.environ.get(
        "EMR_ANALYZER_PROMPT_DIR",
        Path(__file__).resolve().parent / "resources" / "prompts",
    )
).expanduser().resolve()
# Compatibility alias: this directory contains institutional prompts.
PROMPT_DIR = BUILTIN_PROMPT_DIR
CUSTOM_PROMPT_DIR = Path(
    os.environ.get(
        "EMR_ANALYZER_CUSTOM_PROMPT_DIR", BASE_DIR / "prompts" / "custom"
    )
).expanduser().resolve()
PROMPT_SELECTION_PATH = Path(
    os.environ.get(
        "EMR_ANALYZER_PROMPT_SELECTION_PATH",
        BASE_DIR / "prompts" / "active_versions.json",
    )
).expanduser().resolve()
PROMPT_MANIFEST_PATH = BUILTIN_PROMPT_DIR / "manifest.json"

_VERSION_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$")


class PromptConfigurationError(RuntimeError):
    """An external prompt or version selection is structurally unsafe."""


@dataclass(frozen=True, slots=True)
class PromptDefinition:
    key: str
    label: str
    pipeline: str
    institutional_version: str
    required_markers: tuple[str, ...] = ()
    minimum_length: int = 20


@dataclass(frozen=True, slots=True)
class PromptVersion:
    key: str
    version: str
    origin: str
    path: Path
    active: bool = False

    @property
    def display_name(self) -> str:
        origin = "istituzionale" if self.origin == "institutional" else "custom"
        active = " — ATTIVO" if self.active else ""
        return f"{self.version} ({origin}){active}"


def _safe_key(value: str) -> str:
    token = str(value or "").strip()
    if not token or Path(token).name != token or token in {".", ".."}:
        raise PromptConfigurationError(f"Nome prompt non valido: {value!r}")
    return token


def _safe_version(value: str) -> str:
    token = str(value or "").strip()
    if not _VERSION_RE.fullmatch(token):
        raise PromptConfigurationError(
            "Versione non valida: usa 1-80 caratteri fra lettere, numeri, "
            "punto, trattino e underscore."
        )
    return token


def prompt_definitions() -> list[PromptDefinition]:
    try:
        payload = json.loads(PROMPT_MANIFEST_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError) as exc:
        raise PromptConfigurationError(
            f"Manifest dei prompt non leggibile: {PROMPT_MANIFEST_PATH}"
        ) from exc
    rows = payload.get("prompts") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        raise PromptConfigurationError("Manifest dei prompt privo di 'prompts'.")
    result = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        key = _safe_key(row.get("key"))
        result.append(PromptDefinition(
            key=key,
            label=str(row.get("label") or key),
            pipeline=str(row.get("pipeline") or "Altro"),
            institutional_version=_safe_version(
                row.get("institutional_version") or "institutional-v1"
            ),
            required_markers=tuple(
                str(marker) for marker in row.get("required_markers", [])
                if str(marker)
            ),
            minimum_length=max(1, int(row.get("minimum_length", 20) or 20)),
        ))
    if not result:
        raise PromptConfigurationError("Manifest dei prompt vuoto.")
    return result


def prompt_definition(name: str) -> PromptDefinition:
    token = _safe_key(name)
    for definition in prompt_definitions():
        if definition.key == token:
            return definition
    raise PromptConfigurationError(f"Prompt non registrato: {token}")


def _read_selections() -> dict[str, dict[str, str]]:
    try:
        payload = json.loads(PROMPT_SELECTION_PATH.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except (OSError, ValueError, TypeError):
        return {}
    active = payload.get("active") if isinstance(payload, dict) else None
    return dict(active) if isinstance(active, dict) else {}


def active_prompt_selection(name: str) -> tuple[str, str]:
    definition = prompt_definition(name)
    raw = _read_selections().get(definition.key)
    if isinstance(raw, dict):
        origin = str(raw.get("origin") or "")
        version = str(raw.get("version") or "")
        if origin == "custom" and _VERSION_RE.fullmatch(version):
            path = CUSTOM_PROMPT_DIR / definition.key / f"{version}.txt"
            if path.is_file():
                return origin, version
        if (
            origin == "institutional"
            and version == definition.institutional_version
        ):
            return origin, version
    return "institutional", definition.institutional_version


def prompt_path(
    name: str,
    *,
    version: str | None = None,
    origin: str | None = None,
) -> Path:
    definition = prompt_definition(name)
    if version is None or origin is None:
        selected_origin, selected_version = active_prompt_selection(name)
        origin = origin or selected_origin
        version = version or selected_version
    version = _safe_version(version)
    if origin == "institutional":
        if version != definition.institutional_version:
            raise PromptConfigurationError(
                f"Versione istituzionale inesistente per {name}: {version}"
            )
        return PROMPT_DIR / f"{definition.key}.txt"
    if origin == "custom":
        return CUSTOM_PROMPT_DIR / definition.key / f"{version}.txt"
    raise PromptConfigurationError(f"Origine prompt non valida: {origin!r}")


def _validate_text(
    definition: PromptDefinition,
    value: str,
    *,
    required_markers: Iterable[str] = (),
    minimum_length: int | None = None,
) -> str:
    text = str(value or "").strip()
    minimum = max(
        definition.minimum_length,
        int(minimum_length or definition.minimum_length),
    )
    if len(text) < minimum:
        raise PromptConfigurationError(
            f"Prompt '{definition.key}' vuoto o incompleto."
        )
    markers = tuple(dict.fromkeys((
        *definition.required_markers, *tuple(required_markers),
    )))
    missing = [marker for marker in markers if marker not in text]
    if missing:
        raise PromptConfigurationError(
            f"Prompt '{definition.key}' privo dei marcatori obbligatori: "
            + ", ".join(missing)
        )
    return text


def load_prompt(
    name: str,
    *,
    required_markers: Iterable[str] = (),
    minimum_length: int | None = None,
    version: str | None = None,
    origin: str | None = None,
) -> str:
    definition = prompt_definition(name)
    path = prompt_path(name, version=version, origin=origin)
    try:
        value = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise PromptConfigurationError(
            f"Prompt '{name}' non leggibile: {path}"
        ) from exc
    return _validate_text(
        definition,
        value,
        required_markers=required_markers,
        minimum_length=minimum_length,
    )


def prompt_versions(name: str) -> list[PromptVersion]:
    definition = prompt_definition(name)
    active_origin, active_version = active_prompt_selection(name)
    versions = [PromptVersion(
        key=definition.key,
        version=definition.institutional_version,
        origin="institutional",
        path=PROMPT_DIR / f"{definition.key}.txt",
        active=(
            active_origin == "institutional"
            and active_version == definition.institutional_version
        ),
    )]
    directory = CUSTOM_PROMPT_DIR / definition.key
    if directory.is_dir():
        for path in sorted(directory.glob("*.txt"), key=lambda item: item.name):
            version = path.stem
            if not _VERSION_RE.fullmatch(version):
                continue
            versions.append(PromptVersion(
                key=definition.key,
                version=version,
                origin="custom",
                path=path,
                active=active_origin == "custom" and active_version == version,
            ))
    return versions


def save_custom_prompt(
    name: str,
    version: str,
    text: str,
    *,
    overwrite: bool = False,
) -> PromptVersion:
    definition = prompt_definition(name)
    token = _safe_version(version)
    if token == definition.institutional_version:
        raise PromptConfigurationError(
            "Il nome della versione istituzionale è riservato."
        )
    value = _validate_text(definition, text)
    path = CUSTOM_PROMPT_DIR / definition.key / f"{token}.txt"
    if path.exists() and not overwrite:
        raise PromptConfigurationError(
            f"La versione personalizzata '{token}' esiste già."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".txt.tmp")
    temporary.write_text(value + "\n", encoding="utf-8")
    temporary.replace(path)
    return PromptVersion(
        key=definition.key, version=token, origin="custom", path=path
    )


def activate_prompt(name: str, version: str, origin: str) -> None:
    definition = prompt_definition(name)
    token = _safe_version(version)
    # Validate the selected content before making it active.
    load_prompt(name, version=token, origin=origin)
    selections = _read_selections()
    selections[definition.key] = {"origin": origin, "version": token}
    payload = {"schema_version": 1, "active": selections}
    PROMPT_SELECTION_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = PROMPT_SELECTION_PATH.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(PROMPT_SELECTION_PATH)


def prompts_digest(*values: str, schema: object | None = None) -> str:
    """Stable digest used to invalidate only affected cached LLM output."""
    digest = hashlib.sha256()
    for value in values:
        digest.update(str(value).encode("utf-8"))
        digest.update(b"\x1f")
    if schema is not None:
        digest.update(
            json.dumps(schema, sort_keys=True, ensure_ascii=False).encode(
                "utf-8"
            )
        )
    return digest.hexdigest()


def available_prompts() -> list[Path]:
    """Return institutional prompt files in deterministic display order."""
    return [
        PROMPT_DIR / f"{definition.key}.txt"
        for definition in prompt_definitions()
    ]

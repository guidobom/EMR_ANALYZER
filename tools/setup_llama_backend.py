#!/usr/bin/env python
"""One-time setup of the llama.cpp backend for EMR Analyzer.

1. Ensures the llama-server binary is available (brew on macOS; on Linux
   prints the CUDA build instructions for DGX Spark and similar hosts).
2. Copies the GGUF model files that Ollama already stores locally
   (~/.ollama/models/blobs, renamed by digest) into
   ~/.emr_analyzer/models/<family>-<tag>.gguf with readable names, mapping
   them through the Ollama manifests.  Nothing is downloaded and nothing is
   removed: Ollama keeps working.
3. Writes ~/.emr_analyzer/models/index.json with the metadata the app reads
   at runtime (file, size, architecture, max context).
4. Migrates model names in ~/.emr_analyzer/settings.json from the legacy
   Ollama format ("qwen3:14b") to the GGUF-index format ("qwen3-14b").

By default only the models referenced by the saved settings (plus the app
defaults) are copied; pass --all to copy every model Ollama has.  Pass
--dry-run to preview without writing, --skip-brew to leave the binary
alone.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

# Make the repo root importable when the script is run from anywhere.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from emr_analyzer.config import (
    DOCUMENT_LLM_MODEL_NAME,
    CLINICAL_STATE_LLM_MODEL_NAME,
    LLM_MODELS_DIR,
    LLM_MODEL_INDEX_PATH,
)
from emr_analyzer.llm_backend import model_store
from emr_analyzer.settings import (
    SETTINGS_PATH,
    load_llm_configs,
    save_llm_configs,
)

DRY_RUN = "--dry-run" in sys.argv
COPY_ALL = "--all" in sys.argv
SKIP_BREW = "--skip-brew" in sys.argv

OLLAMA_MODELS_DIR = Path.home() / ".ollama" / "models"
OLLAMA_MANIFESTS_DIR = OLLAMA_MODELS_DIR / "manifests"
OLLAMA_BLOBS_DIR = OLLAMA_MODELS_DIR / "blobs"

MODEL_LAYER_TYPE = "application/vnd.ollama.image.model"

BREW_PREFIX_CANDIDATES = (
    "/opt/homebrew/opt/llama.cpp/bin",
    "/usr/local/opt/llama.cpp/bin",
)


def ensure_binary(platform: str | None = None) -> str | None:
    """Locate llama-server, installing it when the platform allows.

    On macOS the binary is installed via Homebrew; on Linux (e.g. NVIDIA
    DGX Spark / DGX OS) it must be a CUDA build from source or from an NGC
    container — the script prints the build instructions instead.
    """
    from emr_analyzer.llm_backend.server_manager import find_server_binary

    found = find_server_binary()
    if found:
        return found
    if SKIP_BREW:
        return None
    system = platform or sys.platform
    if system != "darwin":
        print(
            "\nllama-server non trovato nel PATH.\n"
            "Su Linux serve una build CUDA di llama.cpp (DGX Spark / "
            "Blackwell Ultra = sm_100):\n"
            "\n"
            "  git clone --depth 1 https://github.com/ggml-org/llama.cpp\n"
            "  cd llama.cpp\n"
            "  cmake -B build -DGGML_CUDA=ON -DGGML_NATIVE=ON\n"
            "  cmake --build build --config Release -j\n"
            "  sudo cp build/bin/llama-server /usr/local/bin/\n"
            "\n"
            "oppure usa un container NGC con llama.cpp già compilato.\n"
        )
        return None
    print("llama-server non trovato: installo llama.cpp via Homebrew…")
    if DRY_RUN:
        return "(dry-run: brew install llama.cpp)"
    result = subprocess.run(
        ["brew", "install", "llama.cpp"], text=True
    )
    if result.returncode != 0:
        print("Installazione di llama.cpp fallita; procedi a mano con "
              "`brew install llama.cpp`")
        return None
    return find_server_binary()


def find_ollama_models() -> dict[str, dict]:
    """Map ``family-tag`` names to manifest model layers.

    A layer has ``mediaType``, ``digest`` (sha256:<hex>) and ``size``.  The
    blob on disk is ``blobs/sha256-<hex>`` (dash, not colon).
    """
    if not OLLAMA_MANIFESTS_DIR.is_dir():
        return {}
    models = {}
    for manifest_path in sorted(OLLAMA_MANIFESTS_DIR.rglob("*")):
        if not manifest_path.is_file():
            continue
        relative = manifest_path.relative_to(OLLAMA_MANIFESTS_DIR)
        parts = relative.parts
        if len(parts) < 2:
            continue
        # Real layouts are registry.ollama.ai/library/<family>/<tag>; use
        # the last two components so both prefixed and plain trees work.
        family, tag = parts[-2], parts[-1]
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        for layer in payload.get("layers", []):
            if layer.get("mediaType") == MODEL_LAYER_TYPE:
                models[f"{family}-{tag}"] = {
                    "digest": str(layer.get("digest", "")).removeprefix(
                        "sha256:"
                    ),
                    "size": int(layer.get("size") or 0),
                }
                break
    return models


def blob_path(model: dict) -> Path:
    return OLLAMA_BLOBS_DIR / f"sha256-{model['digest']}"


def validate_blob(model: dict) -> str | None:
    """Return an error message, or None when the blob is a valid GGUF."""
    path = blob_path(model)
    if not path.is_file():
        return "blob mancante"
    with open(path, "rb") as handle:
        if handle.read(4) != b"GGUF":
            return "non è un GGUF"
    if model["size"] and path.stat().st_size != model["size"]:
        return (f"dimensione discorde (manifesto {model['size']}, "
                f"disco {path.stat().st_size})")
    return None


def configured_model_names() -> set[str]:
    """Legacy names from saved settings plus the app defaults."""
    names = {
        DOCUMENT_LLM_MODEL_NAME,
        CLINICAL_STATE_LLM_MODEL_NAME,
    }
    try:
        payload = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        payload = {}
    for role_payload in payload.get("llm", {}).values():
        if isinstance(role_payload, dict) and role_payload.get("model"):
            names.add(str(role_payload["model"]))
    for role_name in payload.get("models", {}).values():
        if isinstance(role_name, str):
            names.add(role_name)
    return names


def pick_models(all_models: dict[str, dict]) -> dict[str, dict]:
    """Restrict the copy set to configured models unless --all is given."""
    if COPY_ALL:
        return all_models
    wanted_legacy = configured_model_names()
    wanted = {
        name.removesuffix(":latest").replace(":", "-")
        for name in wanted_legacy
    }
    chosen = {}
    for name, model in all_models.items():
        if name in wanted:
            chosen[name] = model
    # Keep the `latest` alias for every chosen family so the fuzzy
    # resolution of legacy `family:latest` keeps working.
    for name, model in all_models.items():
        family, _, tag = name.partition("-")
        if tag == "latest" and any(
            other.startswith(f"{family}-") and other != name
            for other in chosen
        ):
            chosen.setdefault(name, model)
    return chosen


def copy_models(models: dict[str, dict]) -> dict[str, dict]:
    """Copy the selected GGUF blobs into the models dir; returns the index."""
    index = dict(model_store.load_index(LLM_MODEL_INDEX_PATH))
    digest_to_name: dict[str, str] = {}
    copied = 0
    for name in sorted(models):
        model = models[name]
        error = validate_blob(model)
        if error is not None:
            print(f"  ✗ {name}: {error}")
            continue
        source = blob_path(model)
        destination = LLM_MODELS_DIR / f"{name}.gguf"
        if destination.exists() and (
            destination.stat().st_size == model["size"] or not model["size"]
        ):
            index.setdefault(name, {
                "file": str(destination),
                "size_bytes": model["size"] or destination.stat().st_size,
                "architecture": "",
                "max_context_length": None,
            })
            continue
        # Same blob already copied under another name: alias it.
        if model["digest"] and model["digest"] in digest_to_name:
            alias_of = digest_to_name[model["digest"]]
            index[name] = index[alias_of]
            print(f"  → {name}: alias di {alias_of}")
            continue
        print(f"  → {name}: copia ({model['size'] / 1e9:.1f} GB)")
        if not DRY_RUN:
            LLM_MODELS_DIR.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
        digest_to_name[model["digest"]] = name
        metadata = model_store.read_gguf_metadata(source)
        index[name] = {
            "file": str(destination),
            "size_bytes": model["size"] or destination.stat().st_size,
            "architecture": (metadata or {}).get("architecture") or "",
            "max_context_length": (
                (metadata or {}).get("max_context_length")
            ),
        }
        copied += 1
    return index, copied


def migrate_settings(index: dict) -> list[tuple[str, str]]:
    """Rewrite legacy Ollama model names in settings.json; returns changes."""
    settings_path = Path(SETTINGS_PATH)
    if not settings_path.exists():
        return []
    try:
        payload = json.loads(settings_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    old_names = {}
    for role_payload in payload.get("llm", {}).values():
        if isinstance(role_payload, dict):
            old_names.setdefault(
                str(role_payload.get("model", "")),
                str(role_payload.get("model", "")),
            )
    for role_name in payload.get("models", {}).values():
        if isinstance(role_name, str):
            old_names.setdefault(role_name, role_name)

    def to_index_name(name: str) -> str:
        candidate = name.removesuffix(":latest").replace(":", "-")
        if candidate in index:
            return candidate
        family = candidate.split("-", 1)[0]
        matches = [n for n in index if n.startswith(f"{family}-")]
        if matches:
            return max(
                matches, key=lambda n: index[n].get("size_bytes", 0)
            )
        return name

    changes = [
        (old, to_index_name(old))
        for old in old_names if to_index_name(old) != old
    ]
    if changes and not DRY_RUN:
        configs = load_llm_configs()
        save_llm_configs(configs)
    return changes


def main() -> int:
    binary = ensure_binary()
    if binary is None:
        if sys.platform == "darwin":
            print("llama-server non disponibile. Installalo con "
                  "`brew install llama.cpp` e riprova.")
        else:
            print("llama-server non disponibile: installa una build CUDA "
                  "(vedi istruzioni sopra) e riprova.")
        return 1
    print(f"llama-server: {binary}")

    models = find_ollama_models()
    if not models:
        print(
            "\nNessun modello Ollama trovato "
            f"({OLLAMA_MODELS_DIR} assente).\n"
            "Scarica un GGUF (ad es. qwen3-14b) e mettilo in "
            f"{LLM_MODELS_DIR}, poi registralo in "
            f"{LLM_MODEL_INDEX_PATH}."
        )
        return 2

    selected = pick_models(models)
    if not selected:
        print("Nessun modello selezionato: controlla i modelli configurati "
              "in ~/.emr_analyzer/settings.json o usa --all.")
        return 2

    print(f"\nModelli da registrare: {', '.join(sorted(selected))}")
    index, copied = copy_models(selected)
    if not DRY_RUN:
        model_store.save_index(index, LLM_MODEL_INDEX_PATH)
        print(f"Indice aggiornato: {LLM_MODEL_INDEX_PATH}")
    else:
        print("(dry-run: indice non scritto)")

    changes = migrate_settings(index)
    for old, new in changes:
        print(f"settings.json: {old} → {new}")
    if not changes:
        print("settings.json: nessuna migrazione necessaria")

    print(f"\nFatto: {len(index)} modelli nell'indice, {copied} copiati.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

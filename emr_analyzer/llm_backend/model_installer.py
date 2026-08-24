"""Install GGUF models without exposing clinical application state.

The GUI launches :mod:`model_download_helper` in a separate process.  This is
intentional: the main EMR Analyzer process remains protected by the strict
loopback-only network policy, while the explicit model-management action can
contact Ollama or an HTTPS endpoint.  Only model identifiers and destinations
are passed to the helper; no workspace or clinical content is involved.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import time
import urllib.request
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Callable
from urllib.parse import unquote, urlparse

from ..config import (
    LLM_MODEL_CATALOG_CACHE_PATH,
    LLM_MODEL_CATALOG_URL,
    LLM_MODEL_INDEX_PATH,
    LLM_MODELS_DIR,
)
from . import model_store
from .model_catalog import cache_catalog_manifest


MODEL_LAYER_TYPE = "application/vnd.ollama.image.model"
_COPY_CHUNK_SIZE = 4 * 1024 * 1024
_MIN_FREE_SPACE = 512 * 1024 * 1024
_MAX_CATALOG_BYTES = 2 * 1024 * 1024
_SHA256_RE = re.compile(r"[0-9a-fA-F]{64}")

ProgressCallback = Callable[[int, int | None, str], None]
CancelCallback = Callable[[], bool]


class ModelInstallError(RuntimeError):
    """A model could not be downloaded, validated or registered."""


class ModelInstallCancelled(ModelInstallError):
    """The caller cancelled an in-progress model installation."""


@dataclass(frozen=True)
class OllamaModel:
    """One locally available model layer discovered from an Ollama manifest."""

    tag: str
    blob_path: Path
    size_bytes: int
    digest: str


def normalize_model_name(value: str) -> str:
    """Return a filesystem-safe friendly model name.

    Ollama-style names (``qwen3:14b``), repository paths and GGUF filenames
    are accepted.  The normalized name is also the key stored in index.json.
    """

    raw = unquote(str(value or "").strip())
    if raw.casefold().endswith(".gguf"):
        raw = raw[:-5]
    raw = re.sub(r"[\s/:\\]+", "-", raw)
    raw = re.sub(r"[^A-Za-z0-9._-]+", "-", raw)
    raw = re.sub(r"-{2,}", "-", raw).strip("-._")
    if not raw:
        raise ModelInstallError("Il nome locale del modello è vuoto.")
    if len(raw) > 160:
        raise ModelInstallError(
            "Il nome locale del modello supera 160 caratteri."
        )
    return raw


def model_name_from_url(url: str) -> str:
    """Derive a friendly local name from the final URL path."""

    parsed = _validate_https_url(url)
    filename = Path(unquote(parsed.path)).name
    if not filename:
        raise ModelInstallError(
            "Impossibile ricavare il nome del modello dall'URL."
        )
    if re.search(r"-\d{5}-of-\d{5}\.gguf$", filename, re.IGNORECASE):
        raise ModelInstallError(
            "Il collegamento indica una parte di un GGUF suddiviso. "
            "Questa versione installa soltanto modelli contenuti in un "
            "singolo file GGUF."
        )
    return normalize_model_name(filename)


def model_name_from_ollama_tag(tag: str) -> str:
    """Convert an Ollama identifier into the local GGUF-index convention."""

    requested = str(tag or "").strip()
    if not requested:
        raise ModelInstallError("Inserisci il nome del modello Ollama.")
    tail = requested.rsplit("/", 1)[-1]
    if ":" not in tail:
        requested += ":latest"
    return normalize_model_name(requested.rsplit("/", 1)[-1])


def _ollama_models_roots() -> list[Path]:
    """Candidate roots for an Ollama model store, most authoritative first.

    Ollama keeps its store under ``<root>/manifests`` and ``<root>/blobs``.
    When the daemon runs as a system service it stores models in the service
    user's home (``/usr/share/ollama/.ollama/models`` on Debian/Ubuntu) rather
    than the caller's ``~/.ollama/models``; ``ollama list`` still reports them
    because it asks the daemon over the socket.  Probing these roots keeps
    discovery working even when the daemon is stopped and makes the local
    archive match what the CLI shows.
    """
    candidates: list[Path] = []
    configured = os.environ.get("OLLAMA_MODELS")
    if configured:
        candidates.append(Path(configured))
    candidates.append(Path("/usr/share/ollama/.ollama/models"))
    candidates.append(Path("/var/lib/ollama"))
    candidates.append(Path.home() / ".ollama" / "models")
    seen: list[Path] = []
    for root in candidates:
        if root not in seen and root.is_dir():
            seen.append(root)
    return seen


def discover_ollama_models(
    ollama_models_dir: str | Path | None = None,
) -> list[OllamaModel]:
    """Read Ollama manifests directly, even when its daemon is stopped.

    With no explicit directory every candidate store found via
    :func:`_ollama_models_roots` is scanned and models are deduplicated by
    tag, so a system-service Ollama (root in the service user's home) and a
    user-local store are both honoured.
    """
    roots = (
        [Path(ollama_models_dir)]
        if ollama_models_dir is not None
        else _ollama_models_roots()
    )

    discovered: dict[str, OllamaModel] = {}
    for root in roots:
        manifests = root / "manifests"
        blobs = root / "blobs"
        if not manifests.is_dir():
            continue
        for manifest_path in sorted(manifests.rglob("*")):
            if not manifest_path.is_file():
                continue
            try:
                relative = manifest_path.relative_to(manifests)
                parts = relative.parts
                payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            # host / namespace... / model / tag
            if len(parts) < 3:
                continue
            namespace = list(parts[1:-2])
            if namespace == ["library"]:
                namespace = []
            model_path = "/".join([*namespace, parts[-2]])
            display_tag = f"{model_path}:{parts[-1]}"
            for layer in payload.get("layers", []):
                if layer.get("mediaType") != MODEL_LAYER_TYPE:
                    continue
                digest = str(layer.get("digest") or "").removeprefix("sha256:")
                blob = blobs / f"sha256-{digest}"
                if not digest or not blob.is_file():
                    continue
                expected_size = int(layer.get("size") or 0)
                actual_size = blob.stat().st_size
                if expected_size and actual_size != expected_size:
                    continue
                if not _has_gguf_magic(blob):
                    continue
                discovered[display_tag] = OllamaModel(
                    tag=display_tag,
                    blob_path=blob,
                    size_bytes=expected_size or actual_size,
                    digest=digest,
                )
                break
    return sorted(discovered.values(), key=lambda item: item.tag.casefold())


def cleanup_abandoned_partials(
    models_dir: str | Path | None = None,
) -> list[Path]:
    """Remove partial files whose dedicated installer process no longer runs."""

    import psutil

    directory = Path(models_dir) if models_dir is not None else LLM_MODELS_DIR
    if not directory.is_dir():
        return []
    removed = []
    pattern = re.compile(r"^\..+\.(\d+)\.[0-9a-f]{32}\.part$")
    for path in directory.glob(".*.part"):
        match = pattern.match(path.name)
        if match is None or psutil.pid_exists(int(match.group(1))):
            continue
        try:
            path.unlink()
            removed.append(path)
        except OSError:
            continue
    return removed


def update_model_catalog(
    *,
    url: str = LLM_MODEL_CATALOG_URL,
    cache_path: str | Path = LLM_MODEL_CATALOG_CACHE_PATH,
    progress: ProgressCallback | None = None,
) -> dict:
    """Download, validate and atomically cache the curated model manifest."""
    parsed = _validate_catalog_url(url)
    request = urllib.request.Request(
        parsed.geturl(),
        headers={"User-Agent": "EMR-Analyzer-catalog-updater/1.0"},
    )
    try:
        opener = urllib.request.build_opener(_CatalogHTTPSRedirectHandler())
        with opener.open(request, timeout=30) as response:
            _validate_catalog_url(response.geturl())
            raw_total = response.headers.get("Content-Length")
            total = int(raw_total) if raw_total and raw_total.isdigit() else None
            if total is not None and total > _MAX_CATALOG_BYTES:
                raise ModelInstallError(
                    "Il manifesto remoto supera il limite di 2 MiB."
                )
            _emit(progress, 0, total, "Download catalogo consigliato…")
            payload = bytearray()
            while True:
                chunk = response.read(min(_COPY_CHUNK_SIZE, 256 * 1024))
                if not chunk:
                    break
                payload.extend(chunk)
                if len(payload) > _MAX_CATALOG_BYTES:
                    raise ModelInstallError(
                        "Il manifesto remoto supera il limite di 2 MiB."
                    )
                _emit(
                    progress, len(payload), total,
                    "Validazione catalogo consigliato…",
                )
        snapshot = cache_catalog_manifest(bytes(payload), cache_path)
        return {
            "catalog_version": snapshot.catalog_version,
            "review_date": snapshot.review_date,
            "model_count": len(snapshot.models),
            "source": parsed.geturl(),
        }
    except ModelInstallError:
        raise
    except ValueError as exc:
        raise ModelInstallError(
            f"Catalogo remoto rifiutato: {exc}"
        ) from exc
    except Exception as exc:
        raise ModelInstallError(
            f"Aggiornamento catalogo non riuscito: {exc}"
        ) from exc


def install_from_ollama(
    tag: str,
    *,
    local_name: str | None = None,
    pull_if_missing: bool = True,
    models_dir: str | Path | None = None,
    index_path: str | Path | None = None,
    ollama_models_dir: str | Path | None = None,
    progress: ProgressCallback | None = None,
    cancelled: CancelCallback | None = None,
) -> tuple[str, dict]:
    """Import a local Ollama GGUF, pulling it first only when absent."""

    requested = str(tag or "").strip()
    if not requested:
        raise ModelInstallError("Inserisci il nome del modello Ollama.")
    lookup = requested if ":" in requested.rsplit("/", 1)[-1] else (
        requested + ":latest"
    )
    available = {item.tag.casefold(): item for item in discover_ollama_models(
        ollama_models_dir
    )}
    candidate = available.get(lookup.casefold())
    if candidate is None and pull_if_missing:
        executable = shutil.which("ollama")
        if not executable:
            raise ModelInstallError(
                "Ollama non è installato. Su DGX/Linux usa un URL HTTPS "
                "diretto verso un file GGUF."
            )
        completed = _pull_with_ollama(executable, requested, progress)
        if completed.returncode != 0:
            details = (completed.stdout or "").strip()
            raise ModelInstallError(
                "Download Ollama non riuscito"
                + (f": {details[-1200:]}" if details else ".")
            )
        available = {
            item.tag.casefold(): item
            for item in discover_ollama_models(ollama_models_dir)
        }
        candidate = available.get(lookup.casefold())
    if candidate is None:
        raise ModelInstallError(
            f"Il modello Ollama {lookup} non è presente nell'archivio locale."
        )

    name = normalize_model_name(
        local_name or model_name_from_ollama_tag(lookup)
    )
    return install_local_gguf(
        candidate.blob_path,
        name,
        models_dir=models_dir,
        index_path=index_path,
        expected_sha256=candidate.digest,
        progress=progress,
        cancelled=cancelled,
        source_label=f"Ollama {candidate.tag}",
    )


def _pull_with_ollama(
    executable: str,
    requested: str,
    progress: ProgressCallback | None,
) -> subprocess.CompletedProcess:
    """Pull a tag, starting a temporary local Ollama daemon when necessary."""

    temporary_server: subprocess.Popen | None = None
    if not _ollama_daemon_ready(executable, timeout=8):
        _emit(progress, 0, None, "Avvio temporaneo del servizio Ollama locale…")
        temporary_server = subprocess.Popen(
            [executable, "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        ready = False
        for _ in range(40):
            if temporary_server.poll() is not None:
                # It may have lost a race against another daemon that has
                # just bound the port; keep probing before declaring failure.
                pass
            time.sleep(0.25)
            if _ollama_daemon_ready(executable, timeout=5):
                ready = True
                break
        if not ready:
            if temporary_server.poll() is None:
                temporary_server.terminate()
            raise ModelInstallError(
                "Impossibile avviare il servizio Ollama locale. In alternativa "
                "usa un URL HTTPS diretto al file GGUF."
            )

    try:
        _emit(progress, 0, None, f"Download Ollama di {requested}…")
        return subprocess.run(
            [executable, "pull", requested],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
    finally:
        if temporary_server is not None and temporary_server.poll() is None:
            temporary_server.terminate()
            try:
                temporary_server.wait(timeout=5)
            except subprocess.TimeoutExpired:
                temporary_server.kill()


def _ollama_daemon_ready(executable: str, *, timeout: int) -> bool:
    try:
        probe = subprocess.run(
            [executable, "list"],
            text=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=timeout,
        )
        return probe.returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def install_from_url(
    url: str,
    *,
    local_name: str | None = None,
    expected_sha256: str | None = None,
    models_dir: str | Path | None = None,
    index_path: str | Path | None = None,
    progress: ProgressCallback | None = None,
    cancelled: CancelCallback | None = None,
) -> tuple[str, dict]:
    """Download, validate and atomically register one GGUF over HTTPS."""

    parsed = _validate_https_url(url)
    name = normalize_model_name(local_name or model_name_from_url(url))
    checksum = _validate_expected_sha256(expected_sha256)
    destination_dir, destination_index = _paths(models_dir, index_path)
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / f"{name}.gguf"
    _ensure_destination_available(destination, destination_index, name)
    temporary = destination_dir / (
        f".{name}.{os.getpid()}.{uuid.uuid4().hex}.part"
    )

    request = urllib.request.Request(
        parsed.geturl(),
        headers={"User-Agent": "EMR-Analyzer-model-installer/1.0"},
    )
    digest = hashlib.sha256()
    written = 0
    try:
        opener = urllib.request.build_opener(_HTTPSRedirectHandler())
        with opener.open(request, timeout=60) as response:
            _validate_https_url(response.geturl())
            raw_total = response.headers.get("Content-Length")
            total = int(raw_total) if raw_total and raw_total.isdigit() else None
            _ensure_disk_capacity(destination_dir, total)
            _emit(progress, 0, total, "Connessione stabilita; download GGUF…")
            with temporary.open("wb") as output:
                while True:
                    _check_cancelled(cancelled)
                    chunk = response.read(_COPY_CHUNK_SIZE)
                    if not chunk:
                        break
                    output.write(chunk)
                    digest.update(chunk)
                    written += len(chunk)
                    if shutil.disk_usage(destination_dir).free < _MIN_FREE_SPACE:
                        raise ModelInstallError(
                            "Download interrotto per conservare almeno 512 MiB "
                            "liberi sul disco."
                        )
                    _emit(progress, written, total, "Download GGUF…")
        if not written:
            raise ModelInstallError("Il server ha restituito un file vuoto.")
        return _finalize_temporary_gguf(
            temporary,
            destination,
            destination_index,
            name,
            digest.hexdigest(),
            checksum,
            source_label=f"HTTPS {parsed.hostname}",
        )
    except ModelInstallError:
        temporary.unlink(missing_ok=True)
        raise
    except Exception as exc:
        temporary.unlink(missing_ok=True)
        raise ModelInstallError(f"Download GGUF non riuscito: {exc}") from exc


def install_local_gguf(
    source: str | Path,
    local_name: str,
    *,
    models_dir: str | Path | None = None,
    index_path: str | Path | None = None,
    expected_sha256: str | None = None,
    progress: ProgressCallback | None = None,
    cancelled: CancelCallback | None = None,
    source_label: str = "file locale",
) -> tuple[str, dict]:
    """Copy and register a local GGUF using an atomic temporary file."""

    source_path = Path(source)
    if not source_path.is_file() or not _has_gguf_magic(source_path):
        raise ModelInstallError("Il file sorgente non è un GGUF valido.")
    name = normalize_model_name(local_name)
    checksum = _validate_expected_sha256(expected_sha256)
    destination_dir, destination_index = _paths(models_dir, index_path)
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / f"{name}.gguf"
    _ensure_destination_available(destination, destination_index, name)
    total = source_path.stat().st_size
    _ensure_disk_capacity(destination_dir, total)
    temporary = destination_dir / (
        f".{name}.{os.getpid()}.{uuid.uuid4().hex}.part"
    )
    digest = hashlib.sha256()
    written = 0
    try:
        with source_path.open("rb") as input_file, temporary.open("wb") as output:
            while True:
                _check_cancelled(cancelled)
                chunk = input_file.read(_COPY_CHUNK_SIZE)
                if not chunk:
                    break
                output.write(chunk)
                digest.update(chunk)
                written += len(chunk)
                _emit(progress, written, total, f"Importazione da {source_label}…")
        return _finalize_temporary_gguf(
            temporary,
            destination,
            destination_index,
            name,
            digest.hexdigest(),
            checksum,
            source_label=source_label,
        )
    except ModelInstallError:
        temporary.unlink(missing_ok=True)
        raise
    except Exception as exc:
        temporary.unlink(missing_ok=True)
        raise ModelInstallError(f"Importazione GGUF non riuscita: {exc}") from exc


def _finalize_temporary_gguf(
    temporary: Path,
    destination: Path,
    index_path: Path,
    name: str,
    actual_sha256: str,
    expected_sha256: str | None,
    *,
    source_label: str,
) -> tuple[str, dict]:
    if expected_sha256 and actual_sha256.casefold() != expected_sha256:
        temporary.unlink(missing_ok=True)
        raise ModelInstallError(
            "Checksum SHA-256 non corrispondente: il file non viene installato."
        )
    if not _has_gguf_magic(temporary):
        temporary.unlink(missing_ok=True)
        raise ModelInstallError(
            "Il file scaricato non è un GGUF valido e non viene installato."
        )
    metadata = model_store.read_gguf_metadata(temporary) or {}
    temporary.replace(destination)
    entry = {
        "file": str(destination),
        "size_bytes": destination.stat().st_size,
        "architecture": metadata.get("architecture") or "",
        "max_context_length": metadata.get("max_context_length"),
        "sha256": actual_sha256,
        "source": source_label,
    }
    index = model_store.load_index(index_path)
    index[name] = entry
    try:
        model_store.save_index(index, index_path)
    except Exception:
        destination.unlink(missing_ok=True)
        raise
    return name, entry


def _paths(
    models_dir: str | Path | None,
    index_path: str | Path | None,
) -> tuple[Path, Path]:
    directory = Path(models_dir) if models_dir is not None else LLM_MODELS_DIR
    index = Path(index_path) if index_path is not None else (
        directory / "index.json"
    )
    return directory, index


def _ensure_destination_available(
    destination: Path, index_path: Path, name: str
) -> None:
    if destination.exists() or name in model_store.load_index(index_path):
        raise ModelInstallError(
            f"Il modello {name} è già registrato. Scegli un nome diverso "
            "oppure rimuovi prima la versione esistente."
        )


def _ensure_disk_capacity(directory: Path, required: int | None) -> None:
    if not required:
        return
    free = shutil.disk_usage(directory).free
    if required + _MIN_FREE_SPACE > free:
        raise ModelInstallError(
            "Spazio su disco insufficiente: servono almeno "
            f"{(required + _MIN_FREE_SPACE) / 1024 ** 3:.1f} GiB, "
            f"disponibili {free / 1024 ** 3:.1f} GiB."
        )


def _validate_expected_sha256(value: str | None) -> str | None:
    checksum = str(value or "").strip().casefold()
    if not checksum:
        return None
    if not _SHA256_RE.fullmatch(checksum):
        raise ModelInstallError(
            "La checksum SHA-256 deve contenere esattamente 64 caratteri "
            "esadecimali."
        )
    return checksum


def _validate_https_url(url: str):
    parsed = urlparse(str(url or "").strip())
    if parsed.scheme.casefold() != "https" or not parsed.hostname:
        raise ModelInstallError("È consentito soltanto un URL HTTPS completo.")
    if parsed.username or parsed.password:
        raise ModelInstallError(
            "Non inserire credenziali direttamente nell'URL del modello."
        )
    return parsed


def _validate_catalog_url(url: str):
    parsed = _validate_https_url(url)
    if (
        parsed.hostname != "raw.githubusercontent.com"
        or parsed.path != (
            "/guidobom/EMR_ANALYZER/main/"
            "emr_analyzer/resources/model_catalog.json"
        )
        or parsed.query
        or parsed.fragment
    ):
        raise ModelInstallError(
            "La sorgente del catalogo non coincide con il repository "
            "ufficiale EMR Analyzer."
        )
    return parsed


class _HTTPSRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        _validate_https_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


class _CatalogHTTPSRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        _validate_catalog_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _has_gguf_magic(path: Path) -> bool:
    try:
        with path.open("rb") as handle:
            return handle.read(4) == b"GGUF"
    except OSError:
        return False


def _emit(
    callback: ProgressCallback | None,
    completed: int,
    total: int | None,
    message: str,
) -> None:
    if callback is not None:
        callback(completed, total, message)


def _check_cancelled(callback: CancelCallback | None) -> None:
    if callback is not None and callback():
        raise ModelInstallCancelled("Installazione annullata dall'utente.")

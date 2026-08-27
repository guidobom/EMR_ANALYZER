from __future__ import annotations

import hashlib
import io
import json
import struct
import subprocess
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError

import pytest

from emr_analyzer.llm_backend import model_store
from emr_analyzer.llm_backend.model_installer import (
    MODEL_LAYER_TYPE,
    ModelInstallError,
    cleanup_abandoned_partials,
    discover_ollama_models,
    install_from_ollama,
    install_from_url,
    install_local_gguf,
    model_name_from_ollama_tag,
    model_name_from_url,
    normalize_model_name,
    update_model_catalog,
    _ollama_models_roots,
    _pull_with_ollama,
)


def _minimal_gguf(payload: bytes = b"weights") -> bytes:
    # Valid header with no tensors/metadata is sufficient for installer
    # boundary tests; production files expose architecture/context metadata.
    return b"GGUF" + struct.pack("<IQQ", 3, 0, 0) + payload


def test_model_names_are_safe_and_derived_from_supported_sources():
    assert normalize_model_name(" Qwen3:14B / Q4_K_M.gguf ") == (
        "Qwen3-14B-Q4_K_M"
    )
    assert model_name_from_ollama_tag("qwen3:30b-a3b") == "qwen3-30b-a3b"
    assert model_name_from_ollama_tag("qwen3") == "qwen3-latest"
    assert model_name_from_url(
        "https://huggingface.co/org/repo/resolve/main/model.Q4_K_M.gguf"
    ) == "model.Q4_K_M"
    with pytest.raises(ModelInstallError, match="GGUF suddiviso"):
        model_name_from_url(
            "https://example.org/model-00001-of-00003.gguf"
        )
    with pytest.raises(ModelInstallError, match="HTTPS"):
        model_name_from_url("http://example.org/model.gguf")


def test_local_gguf_is_copied_hashed_and_registered_atomically(tmp_path):
    source = tmp_path / "source.gguf"
    source.write_bytes(_minimal_gguf())
    destination = tmp_path / "models"
    index_path = destination / "index.json"
    progress = []

    name, entry = install_local_gguf(
        source,
        "clinical:model",
        models_dir=destination,
        index_path=index_path,
        progress=lambda completed, total, message: progress.append(
            (completed, total, message)
        ),
    )

    assert name == "clinical-model"
    installed = destination / "clinical-model.gguf"
    assert installed.read_bytes() == source.read_bytes()
    assert entry["sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert model_store.load_index(index_path)[name]["file"] == str(installed)
    assert progress[-1][0] == source.stat().st_size
    assert not list(destination.glob(".*.part"))


def test_checksum_mismatch_leaves_neither_model_nor_partial(tmp_path):
    source = tmp_path / "source.gguf"
    source.write_bytes(_minimal_gguf())
    destination = tmp_path / "models"

    with pytest.raises(ModelInstallError, match="Checksum"):
        install_local_gguf(
            source,
            "bad-checksum",
            models_dir=destination,
            expected_sha256="0" * 64,
        )

    assert not (destination / "bad-checksum.gguf").exists()
    assert not list(destination.glob(".*.part"))


def test_abandoned_partial_is_removed_but_live_one_is_preserved(tmp_path):
    stale = tmp_path / ".model.111.0123456789abcdef0123456789abcdef.part"
    live = tmp_path / ".model.222.0123456789abcdef0123456789abcdef.part"
    unrelated = tmp_path / ".manual.part"
    for path in (stale, live, unrelated):
        path.write_bytes(b"partial")

    with patch("psutil.pid_exists", side_effect=lambda pid: pid == 222):
        removed = cleanup_abandoned_partials(tmp_path)

    assert removed == [stale]
    assert not stale.exists()
    assert live.exists()
    assert unrelated.exists()


def test_ollama_manifest_is_discovered_and_imported_without_pull(tmp_path):
    ollama = tmp_path / "ollama-models"
    manifest = (
        ollama
        / "manifests/registry.ollama.ai/library/medgemma/4b"
    )
    manifest.parent.mkdir(parents=True)
    blob_bytes = _minimal_gguf(b"medical")
    digest = hashlib.sha256(blob_bytes).hexdigest()
    blob = ollama / "blobs" / f"sha256-{digest}"
    blob.parent.mkdir(parents=True)
    blob.write_bytes(blob_bytes)
    manifest.write_text(
        json.dumps(
            {
                "layers": [
                    {
                        "mediaType": MODEL_LAYER_TYPE,
                        "digest": f"sha256:{digest}",
                        "size": len(blob_bytes),
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    discovered = discover_ollama_models(ollama)
    assert [item.tag for item in discovered] == ["medgemma:4b"]

    destination = tmp_path / "emr-models"
    with patch("shutil.which", side_effect=AssertionError("must not pull")):
        name, entry = install_from_ollama(
            "medgemma:4b",
            ollama_models_dir=ollama,
            models_dir=destination,
            pull_if_missing=True,
        )
    assert name == "medgemma-4b"
    assert Path(entry["file"]).read_bytes() == blob_bytes
    assert entry["source"] == "Ollama medgemma:4b"


def _write_ollama_store(store, tag="27b"):
    """Create a minimal Ollama store with one GGUF model layer."""
    manifest = store / f"manifests/registry.ollama.ai/library/medgemma/{tag}"
    manifest.parent.mkdir(parents=True)
    blob_bytes = _minimal_gguf(b"medgemma-27b")
    digest = hashlib.sha256(blob_bytes).hexdigest()
    blob = store / "blobs" / f"sha256-{digest}"
    blob.parent.mkdir(parents=True)
    blob.write_bytes(blob_bytes)
    manifest.write_text(
        json.dumps(
            {
                "layers": [
                    {
                        "mediaType": MODEL_LAYER_TYPE,
                        "digest": f"sha256:{digest}",
                        "size": len(blob_bytes),
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    return blob, blob_bytes


def test_ollama_models_roots_honour_ollama_models_env(tmp_path, monkeypatch):
    # A system-service Ollama keeps its store in the service user's home, not
    # the caller's ~/.ollama/models — yet `ollama list` reports it.  The
    # resolved roots must include an explicitly configured OLLAMA_MODELS dir.
    store = tmp_path / "system-ollama"
    store.mkdir(parents=True)
    monkeypatch.setenv("OLLAMA_MODELS", str(store))
    roots = _ollama_models_roots()
    assert roots and roots[0] == store


def test_default_roots_discover_system_service_store(tmp_path, monkeypatch):
    # Discovery without an explicit dir scans the resolved roots (a store in
    # the service user's home), so the local archive matches `ollama list`.
    store = tmp_path / "system-ollama"
    blob, _ = _write_ollama_store(store)
    monkeypatch.setenv("OLLAMA_MODELS", str(store))
    with patch(
        "emr_analyzer.llm_backend.model_installer._ollama_models_roots",
        return_value=[store],
    ):
        discovered = discover_ollama_models()
    assert [item.tag for item in discovered] == ["medgemma:27b"]
    assert discovered[0].blob_path == blob


def test_ollama_pull_starts_and_stops_a_temporary_daemon_when_needed():
    class FakeServer:
        def __init__(self):
            self.terminated = False

        def poll(self):
            return None

        def terminate(self):
            self.terminated = True

        def wait(self, timeout):
            assert timeout == 5

        def kill(self):
            raise AssertionError("graceful termination should succeed")

    server = FakeServer()
    failed_probe = subprocess.CompletedProcess([], 1)
    ready_probe = subprocess.CompletedProcess([], 0)
    successful_pull = subprocess.CompletedProcess([], 0, stdout="success")
    messages = []
    with (
        patch(
            "emr_analyzer.llm_backend.model_installer.subprocess.run",
            side_effect=[failed_probe, ready_probe, successful_pull],
        ) as run,
        patch(
            "emr_analyzer.llm_backend.model_installer.subprocess.Popen",
            return_value=server,
        ),
        patch("emr_analyzer.llm_backend.model_installer.time.sleep"),
    ):
        result = _pull_with_ollama(
            "/usr/bin/ollama",
            "new-model:8b",
            lambda _done, _total, message: messages.append(message),
        )

    assert result is successful_pull
    assert server.terminated
    assert run.call_count == 3
    assert messages == [
        "Avvio temporaneo del servizio Ollama locale…",
        "Download Ollama di new-model:8b…",
    ]


class _FakeHTTPSResponse(io.BytesIO):
    def __init__(self, payload: bytes, url: str):
        super().__init__(payload)
        self.headers = {"Content-Length": str(len(payload))}
        self._url = url

    def geturl(self):
        return self._url

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()


class _FakeOpener:
    def __init__(self, payload: bytes, url: str):
        self.payload = payload
        self.url = url

    def open(self, request, timeout):
        assert request.full_url == self.url
        assert timeout == 60
        return _FakeHTTPSResponse(self.payload, self.url)


class _FakeCatalogOpener(_FakeOpener):
    def open(self, request, timeout):
        assert request.full_url == self.url
        assert timeout == 30
        return _FakeHTTPSResponse(self.payload, self.url)


def test_https_download_validates_and_registers_the_file(tmp_path):
    payload = _minimal_gguf(b"downloaded")
    url = "https://example.org/model.gguf"
    destination = tmp_path / "models"
    progress = []
    with patch(
        "urllib.request.build_opener",
        return_value=_FakeOpener(payload, url),
    ):
        name, entry = install_from_url(
            url,
            models_dir=destination,
            expected_sha256=hashlib.sha256(payload).hexdigest(),
            progress=lambda completed, total, message: progress.append(
                (completed, total, message)
            ),
        )

    assert name == "model"
    assert Path(entry["file"]).read_bytes() == payload
    assert progress[-1][:2] == (len(payload), len(payload))


def test_catalog_update_downloads_validates_and_caches_manifest(tmp_path):
    from emr_analyzer.config import LLM_MODEL_CATALOG_URL
    from emr_analyzer.llm_backend.model_catalog import builtin_catalog_manifest

    manifest = builtin_catalog_manifest()
    manifest["catalog_version"] = 3
    manifest["review_date"] = "2026-09-01"
    payload = json.dumps(manifest, ensure_ascii=False).encode("utf-8")
    cache = tmp_path / "catalog.json"
    progress = []
    with patch(
        "urllib.request.build_opener",
        return_value=_FakeCatalogOpener(payload, LLM_MODEL_CATALOG_URL),
    ):
        result = update_model_catalog(
            cache_path=cache,
            progress=lambda completed, total, message: progress.append(
                (completed, total, message)
            ),
        )

    assert result["catalog_version"] == 3
    assert result["updated"] is True
    assert result["model_count"] == len(manifest["models"])
    assert cache.read_bytes() == payload
    assert progress[-1][:2] == (len(payload), len(payload))


def test_catalog_update_uses_integrated_catalog_when_remote_is_private(tmp_path):
    cache = tmp_path / "catalog.json"
    progress = []
    opener = _FakeOpener(b"", "https://unused.test")
    opener.open = lambda request, timeout: (_ for _ in ()).throw(
        HTTPError(request.full_url, 404, "Not Found", {}, None)
    )

    with patch("urllib.request.build_opener", return_value=opener):
        result = update_model_catalog(
            cache_path=cache,
            progress=lambda completed, total, message: progress.append(
                (completed, total, message)
            ),
        )

    assert result["updated"] is False
    assert result["source"] == "integrated"
    assert result["model_count"] > 0
    assert "senza credenziali" in result["notice"]
    assert "catalogo validato" in progress[-1][2]
    assert not cache.exists()


def test_catalog_update_rejects_untrusted_source_and_invalid_manifest(tmp_path):
    from emr_analyzer.config import LLM_MODEL_CATALOG_URL

    with pytest.raises(ModelInstallError, match="repository ufficiale"):
        update_model_catalog(
            url="https://raw.githubusercontent.com/attacker/repo/main/catalog.json",
            cache_path=tmp_path / "catalog.json",
        )

    payload = b'{"schema_version": 1, "models": []}'
    with (
        patch(
            "urllib.request.build_opener",
            return_value=_FakeCatalogOpener(payload, LLM_MODEL_CATALOG_URL),
        ),
        pytest.raises(ModelInstallError, match="rifiutato"),
    ):
        update_model_catalog(cache_path=tmp_path / "catalog.json")
    assert not (tmp_path / "catalog.json").exists()


def test_download_helper_emits_catalog_success(capsys):
    from emr_analyzer.llm_backend import model_download_helper

    result = {
        "catalog_version": 3,
        "review_date": "2026-10-01",
        "model_count": 9,
        "source": "official",
    }
    with patch.object(
        model_download_helper, "update_model_catalog", return_value=result
    ):
        return_code = model_download_helper.main(["--update-catalog"])

    event = json.loads(capsys.readouterr().out)
    assert return_code == 0
    assert event == {"type": "catalog_success", **result}

"""Unit tests for tools/setup_llama_backend.py (tmp dirs only)."""

import importlib.util
import json
import struct
import tempfile
import unittest
from pathlib import Path
from unittest import mock

TOOLS_DIR = Path(__file__).resolve().parents[1] / "tools"
_SPEC = importlib.util.spec_from_file_location(
    "setup_llama_backend", TOOLS_DIR / "setup_llama_backend.py"
)
setup = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(setup)


def _tiny_gguf() -> bytes:
    """Minimal GGUF header (qwen3, context 32768)."""
    def kv(key: bytes, value_type: int, value: bytes) -> bytes:
        return (
            struct.pack("<Q", len(key)) + key
            + struct.pack("<I", value_type) + value
        )

    body = struct.pack("<I", 3)          # version
    body += struct.pack("<Q", 0)         # tensor count
    body += struct.pack("<Q", 2)         # kv count
    body += kv(b"general.architecture", 8,
               struct.pack("<Q", 5) + b"qwen3")
    body += kv(b"qwen3.context_length", 10, struct.pack("<Q", 32768))
    return b"GGUF" + body


def _make_manifest(family: str, tag: str, digest: str, size: int,
                   extra_layers: list | None = None) -> Path:
    base = setup.OLLAMA_MANIFESTS_DIR / family / tag
    base.parent.mkdir(parents=True, exist_ok=True)
    layers = (extra_layers or []) + [{
        "mediaType": "application/vnd.ollama.image.model",
        "digest": f"sha256:{digest}",
        "size": size,
    }]
    base.write_text(json.dumps({"schemaVersion": 2, "layers": layers}))
    return base


class TestFindOllamaModels(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        setup.OLLAMA_MODELS_DIR = Path(self._tmp.name) / "models"
        setup.OLLAMA_MANIFESTS_DIR = setup.OLLAMA_MODELS_DIR / "manifests"
        setup.OLLAMA_BLOBS_DIR = setup.OLLAMA_MODELS_DIR / "blobs"

    def tearDown(self):
        self._tmp.cleanup()

    def test_maps_family_tag_to_model_layer(self):
        _make_manifest("qwen3", "14b", "a8cc", 100, extra_layers=[{
            "mediaType": "application/vnd.ollama.image.template",
            "digest": "sha256:abcd", "size": 10,
        }])
        models = setup.find_ollama_models()
        self.assertEqual(models["qwen3-14b"]["digest"], "a8cc")
        self.assertEqual(models["qwen3-14b"]["size"], 100)

    def test_missing_models_dir_returns_empty(self):
        setup.OLLAMA_MODELS_DIR = Path(self._tmp.name) / "absent"
        setup.OLLAMA_MANIFESTS_DIR = setup.OLLAMA_MODELS_DIR / "manifests"
        self.assertEqual(setup.find_ollama_models(), {})


class TestValidateBlob(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        setup.OLLAMA_BLOBS_DIR = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def _blob(self, digest, payload):
        path = setup.OLLAMA_BLOBS_DIR / f"sha256-{digest}"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        return path

    def test_accepts_valid_gguf(self):
        payload = _tiny_gguf()
        self._blob("a8cc", payload)
        error = setup.validate_blob({
            "digest": "a8cc", "size": len(payload),
        })
        self.assertIsNone(error)

    def test_rejects_json_blob(self):
        self._blob("json", b'{"model":"nope"}')
        error = setup.validate_blob({"digest": "json", "size": 15})
        self.assertEqual(error, "non è un GGUF")

    def test_rejects_size_mismatch(self):
        payload = _tiny_gguf()
        self._blob("a8cc", payload)
        error = setup.validate_blob({
            "digest": "a8cc", "size": len(payload) + 10,
        })
        self.assertIn("dimensione discorde", error)

    def test_rejects_missing_blob(self):
        error = setup.validate_blob({"digest": "ghost", "size": 10})
        self.assertEqual(error, "blob mancante")


class TestEnsureBinary(unittest.TestCase):
    def setUp(self):
        # DRY_RUN is a module-level flag other tests toggle; restore it.
        self._old_dry_run = setup.DRY_RUN
        setup.DRY_RUN = False

    def tearDown(self):
        setup.DRY_RUN = self._old_dry_run

    def test_darwin_missing_binary_prints_managed_runtime_instructions(self):
        with mock.patch(
            "emr_analyzer.llm_backend.server_manager.find_server_binary",
            return_value=None,
        ), mock.patch("builtins.print") as fake_print:
            self.assertIsNone(setup.ensure_binary(platform="darwin"))
        printed = " ".join(
            str(call.args[0]) for call in fake_print.call_args_list
        )
        self.assertIn("--server-binary", printed)
        self.assertIn("Metal", printed)

    def test_linux_missing_binary_prints_cuda_instructions(self):
        with mock.patch(
            "emr_analyzer.llm_backend.server_manager.find_server_binary",
            return_value=None,
        ), mock.patch("builtins.print") as fake_print:
            self.assertIsNone(setup.ensure_binary(platform="linux"))
        printed = " ".join(
            str(call.args[0]) for call in fake_print.call_args_list
        )
        self.assertIn("GGML_CUDA=ON", printed)
        self.assertIn("DGX Spark", printed)

    def test_found_binary_is_returned_verbatim(self):
        with mock.patch(
            "emr_analyzer.llm_backend.server_manager.find_server_binary",
            return_value="/opt/llama/llama-server",
        ):
            self.assertEqual(
                setup.ensure_binary(platform="linux"),
                "/opt/llama/llama-server",
            )


class TestCopyAndMigration(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        root = Path(self._tmp.name)
        setup.LLM_MODELS_DIR = root / "emr_models"
        setup.LLM_MODEL_INDEX_PATH = setup.LLM_MODELS_DIR / "index.json"
        setup.OLLAMA_BLOBS_DIR = root / "blobs"
        setup.SETTINGS_PATH = root / "settings.json"
        self.blob_bytes = _tiny_gguf()

    def tearDown(self):
        self._tmp.cleanup()

    def _write_blob(self, digest, size=None):
        path = setup.OLLAMA_BLOBS_DIR / f"sha256-{digest}"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(self.blob_bytes)
        return path

    def _model(self, name, digest):
        return {name: {"digest": digest, "size": len(self.blob_bytes)}}

    def test_copy_models_builds_index_with_metadata(self):
        self._write_blob("a8cc")
        models = self._model("qwen3-14b", "a8cc")
        index, copied = setup.copy_models(models)
        self.assertEqual(copied, 1)
        entry = index["qwen3-14b"]
        self.assertEqual(entry["architecture"], "qwen3")
        self.assertEqual(entry["max_context_length"], 32768)
        self.assertEqual(entry["size_bytes"], len(self.blob_bytes))
        self.assertTrue(
            Path(entry["file"]).exists(),
            "il GGUF deve essere stato copiato",
        )

    def test_copy_models_aliases_same_digest(self):
        self._write_blob("a8cc")
        models = {
            "qwen3-14b": {"digest": "a8cc", "size": len(self.blob_bytes)},
            "qwen3-latest": {"digest": "a8cc", "size": len(self.blob_bytes)},
        }
        index, _ = setup.copy_models(models)
        self.assertEqual(index["qwen3-latest"]["file"],
                         index["qwen3-14b"]["file"])
        self.assertEqual(len(set(e["file"] for e in index.values())), 1)

    def test_copy_models_skips_invalid_blobs(self):
        self._write_blob("bad")
        models = self._model("qwen3-14b", "bad")
        # Corrupt the blob after writing.
        (setup.OLLAMA_BLOBS_DIR / "sha256-bad").write_bytes(b"not-gguf")
        with mock.patch("builtins.print"):
            index, copied = setup.copy_models(models)
        self.assertEqual(copied, 0)
        self.assertEqual(index, {})

    def test_copy_models_is_idempotent(self):
        self._write_blob("a8cc")
        models = self._model("qwen3-14b", "a8cc")
        setup.copy_models(models)
        _, copied = setup.copy_models(models)
        self.assertEqual(copied, 0)

    def test_pick_models_restricts_to_configured(self):
        settings = {"llm": {
            "document": {"model": "qwen3:14b"},
        }}
        setup.SETTINGS_PATH.write_text(json.dumps(settings))
        models = {
            "qwen3-14b": {"digest": "a8cc", "size": 1},
            "gemma3-4b": {"digest": "bbbb", "size": 2},
        }
        selected = setup.pick_models(models)
        self.assertEqual(set(selected), {"qwen3-14b"})

    def test_migrate_settings_maps_legacy_names(self):
        settings = {"llm": {
            "document": {"model": "qwen3:14b"},
            "clinical_state": {"model": "qwen3:14b"},
        }}
        setup.SETTINGS_PATH.write_text(json.dumps(settings))
        setup.DRY_RUN = True  # verify the mapping without writing
        index = {
            "qwen3-14b": {
                "file": "/models/qwen3-14b.gguf", "size_bytes": 1,
                "architecture": "qwen3", "max_context_length": 32768,
            },
        }
        changes = setup.migrate_settings(index)
        self.assertIn(("qwen3:14b", "qwen3-14b"), changes)

    def test_migrate_settings_leaves_gguf_names_alone(self):
        settings = {"llm": {"document": {"model": "qwen3-14b"}}}
        setup.SETTINGS_PATH.write_text(json.dumps(settings))
        setup.DRY_RUN = True
        self.assertEqual(setup.migrate_settings({"qwen3-14b": {}}), [])


if __name__ == "__main__":
    unittest.main()

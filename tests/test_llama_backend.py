"""Unit tests for the llama.cpp backend package (no real server needed)."""

import json
import os
import struct
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from emr_analyzer.llm_backend import model_store
from emr_analyzer.llm_backend.backend import LlamaBackend
from emr_analyzer.llm_backend.server_manager import (
    BackendError,
    ServerKey,
    ServerManager,
)


# ---------------------------------------------------------------------------
# Synthetic GGUF metadata
# ---------------------------------------------------------------------------


def _gguf_bytes(
    architecture: str = "qwen3",
    context_length: int | None = 32768,
    *,
    extra_kv: int = 0,
    truncated: bool = False,
) -> bytes:
    """Build a minimal, well-formed GGUF header with the two fields we read."""
    kvs: list[tuple[str, int, bytes]] = []
    if architecture:
        payload = architecture.encode("utf-8")
        kvs.append(("general.architecture", 8, struct.pack("<Q", len(payload)) + payload))
    if context_length is not None:
        kvs.append((f"{architecture}.context_length", 10,
                    struct.pack("<Q", context_length)))
    for index in range(extra_kv):
        payload = f"extra{index}".encode("utf-8")
        kvs.append((f"tokenizer.extra.{index}", 8,
                    struct.pack("<Q", len(payload)) + payload))

    body = struct.pack("<I", 3)  # version
    body += struct.pack("<Q", 0)  # tensor count
    body += struct.pack("<Q", len(kvs))
    for key, value_type, value in kvs:
        encoded = key.encode("utf-8")
        body += struct.pack("<Q", len(encoded)) + encoded
        body += struct.pack("<I", value_type) + value
    if truncated:
        return b"GGUF" + body[:-6]
    return b"GGUF" + body


class TestGgufMetadata(unittest.TestCase):
    def test_reads_architecture_and_context(self):
        with tempfile.NamedTemporaryFile(suffix=".gguf") as handle:
            handle.write(_gguf_bytes("qwen3", 32768))
            handle.flush()
            info = model_store.read_gguf_metadata(handle.name)
        self.assertEqual(info, {
            "architecture": "qwen3",
            "max_context_length": 32768,
        })

    def test_skips_other_kv_pairs(self):
        with tempfile.NamedTemporaryFile(suffix=".gguf") as handle:
            handle.write(_gguf_bytes("llama", 8192, extra_kv=5))
            handle.flush()
            info = model_store.read_gguf_metadata(handle.name)
        self.assertEqual(info["max_context_length"], 8192)

    def test_rejects_non_gguf(self):
        with tempfile.NamedTemporaryFile(suffix=".gguf") as handle:
            handle.write(b'{"model": "not a gguf"}\n')
            handle.flush()
            self.assertIsNone(model_store.read_gguf_metadata(handle.name))

    def test_rejects_truncated(self):
        with tempfile.NamedTemporaryFile(suffix=".gguf") as handle:
            handle.write(_gguf_bytes(truncated=True))
            handle.flush()
            self.assertIsNone(model_store.read_gguf_metadata(handle.name))

    def test_missing_file(self):
        self.assertIsNone(
            model_store.read_gguf_metadata("/nonexistent/nope.gguf")
        )

    def test_context_key_prefers_architecture_prefix(self):
        # A wrong-arch context appears first; the arch-prefixed one must win.
        kvs = [
            ("general.architecture", 8, None),
            ("gemma3.context_length", 10, struct.pack("<Q", 4096)),
            ("qwen3.context_length", 10, struct.pack("<Q", 32768)),
        ]
        body = struct.pack("<I", 3) + struct.pack("<Q", 0) + struct.pack("<Q", len(kvs))
        for key, value_type, value in kvs:
            if value_type == 8:
                payload = b"qwen3"
                value = struct.pack("<Q", len(payload)) + payload
            encoded = key.encode("utf-8")
            body += struct.pack("<Q", len(encoded)) + encoded
            body += struct.pack("<I", value_type) + value
        with tempfile.NamedTemporaryFile(suffix=".gguf") as handle:
            handle.write(b"GGUF" + body)
            handle.flush()
            info = model_store.read_gguf_metadata(handle.name)
        self.assertEqual(info["max_context_length"], 32768)


class TestIndex(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.index_path = Path(self._tmp.name) / "index.json"

    def tearDown(self):
        self._tmp.cleanup()

    def test_round_trip(self):
        index = {
            "qwen3-14b": {
                "file": "/models/qwen3-14b.gguf",
                "size_bytes": 9276184896,
                "architecture": "qwen3",
                "max_context_length": 32768,
            },
        }
        model_store.save_index(index, self.index_path)
        loaded = model_store.load_index(self.index_path)
        self.assertEqual(loaded, index)

    def test_load_missing_returns_empty(self):
        self.assertEqual(model_store.load_index(self.index_path), {})

    def test_load_corrupt_returns_empty(self):
        self.index_path.write_text("{nope", encoding="utf-8")
        self.assertEqual(model_store.load_index(self.index_path), {})

    def test_list_models_sorted(self):
        model_store.save_index({
            "qwen3-8b": {"file": "/models/qwen3-8b.gguf", "size_bytes": 1},
            "qwen3-14b": {"file": "/models/qwen3-14b.gguf", "size_bytes": 2},
        }, self.index_path)
        self.assertEqual(
            model_store.list_models(self.index_path),
            ["qwen3-14b", "qwen3-8b"],
        )

    def test_resolve_exact_and_legacy_colon(self):
        model_store.save_index({
            "qwen3-14b": {"file": "/models/qwen3-14b.gguf", "size_bytes": 100},
        }, self.index_path)
        self.assertIsNotNone(model_store.resolve("qwen3-14b", self.index_path))
        self.assertIsNotNone(model_store.resolve("qwen3:14b", self.index_path))
        self.assertIsNotNone(
            model_store.resolve("qwen3:latest", self.index_path)
        )
        self.assertIsNone(model_store.resolve("gemma3:4b", self.index_path))

    def test_resolve_fuzzy_prefers_largest(self):
        model_store.save_index({
            "qwen3-4b": {"file": "/models/qwen3-4b.gguf", "size_bytes": 100},
            "qwen3-14b": {"file": "/models/qwen3-14b.gguf", "size_bytes": 500},
        }, self.index_path)
        entry = model_store.resolve("qwen3:8b", self.index_path)
        self.assertEqual(entry["file"], "/models/qwen3-14b.gguf")


# ---------------------------------------------------------------------------
# Server manager with fakes
# ---------------------------------------------------------------------------


class _FakeProc:
    """Duck-typed Popen stand-in."""

    def __init__(self, argv):
        self.argv = argv
        self._returncode = None
        self.stderr = None

    def poll(self):
        return self._returncode

    def terminate(self):
        self._returncode = -15

    def kill(self):
        self._returncode = -9

    def wait(self, timeout=None):
        self._returncode = 0
        return 0

    def die(self, code=1):
        self._returncode = code


class _FakeResponse:
    def __init__(self, status_code=200, payload=None):
        self.status_code = status_code
        self._payload = payload or {}
        self.text = json.dumps(self._payload)

    def json(self):
        return self._payload


class TestServerManager(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._model_path = Path(self._tmp.name) / "qwen3-14b.gguf"
        self._model_path.write_bytes(b"GGUF-fake")
        self.manager = ServerManager(binary="/fake/llama-server")
        self.key = ServerKey(str(self._model_path), 32768, 4)
        self.patches = []

    def tearDown(self):
        for patcher in self.patches:
            patcher.stop()
        self.manager.stop_all()
        self._tmp.cleanup()

    def _patch_spawn_and_health(self, healthy=True):
        spawn_patch = mock.patch(
            "emr_analyzer.llm_backend.server_manager.subprocess.Popen",
            side_effect=lambda argv, **kwargs: _FakeProc(argv),
        )
        self.fake_popen = spawn_patch.start()
        self.patches.append(spawn_patch)
        status = 200 if healthy else 503
        health_patch = mock.patch(
            "emr_analyzer.llm_backend.server_manager.httpx.get",
            return_value=_FakeResponse(status),
        )
        self.fake_health = health_patch.start()
        self.patches.append(health_patch)
        port_patch = mock.patch(
            "emr_analyzer.llm_backend.server_manager._port_is_free",
            side_effect=lambda port: port in (11435, 11436),
        )
        self.fake_port = port_patch.start()
        self.patches.append(port_patch)
        # Skip the real 3-second early-exit watch: fakes never exit.
        early_patch = mock.patch(
            "emr_analyzer.llm_backend.server_manager.ServerManager._wait_early_exit",
            return_value=False,
        )
        self.fake_early_exit = early_patch.start()
        self.patches.append(early_patch)

    def test_spawn_argv(self):
        self._patch_spawn_and_health()
        url = self.manager.ensure(self.key, load_timeout=5)
        self.assertTrue(url.startswith("http://127.0.0.1:11435"))
        argv = self.fake_popen.call_args[0][0]
        self.assertEqual(argv[0], "/fake/llama-server")
        self.assertIn(str(self._model_path), argv)
        # Total ctx = per-request ctx × slots (server splits it).
        self.assertEqual(argv[argv.index("-c") + 1], str(32768 * 4))
        self.assertEqual(argv[argv.index("-np") + 1], "4")
        self.assertEqual(argv[argv.index("-ngl") + 1], "all")
        self.assertEqual(argv[argv.index("-ctk") + 1], "q8_0")
        self.assertIn("-rea", argv)

    def test_ngram_speculation_is_explicit_and_target_verified(self):
        self._patch_spawn_and_health()
        key = ServerKey(
            str(self._model_path), 32768, 1,
            speculative_mode="ngram-cache",
        )
        self.manager.ensure(key, load_timeout=5)
        argv = self.fake_popen.call_args[0][0]
        self.assertEqual(argv[argv.index("--spec-type") + 1], "ngram-cache")

    def test_ensure_is_idempotent(self):
        self._patch_spawn_and_health()
        first = self.manager.ensure(self.key, load_timeout=5)
        second = self.manager.ensure(self.key, load_timeout=5)
        self.assertEqual(first, second)
        self.assertEqual(self.fake_popen.call_count, 1)

    def test_different_keys_spawn_separate_servers(self):
        self._patch_spawn_and_health()
        self.manager.ensure(self.key, load_timeout=5)
        other = ServerKey(str(self._model_path), 65536, 4)
        self.manager.ensure(other, load_timeout=5)
        self.assertEqual(self.fake_popen.call_count, 2)

    def test_backend_can_reserve_memory_for_one_pipeline_runtime(self):
        retained = self.key
        obsolete_model = ServerKey(
            str(self._model_path) + ".other", 65536, 2
        )
        obsolete_shape = ServerKey(
            str(self._model_path), 131072, 3
        )
        manager = mock.Mock()
        manager.running_keys.return_value = [
            obsolete_model, retained, obsolete_shape,
        ]
        manager.stop.return_value = True
        backend = LlamaBackend(manager=manager)
        backend.key_for = mock.Mock(return_value=retained)

        stopped = backend.stop_other_runtimes(mock.Mock())

        self.assertEqual(stopped, 2)
        self.assertEqual(
            manager.stop.call_args_list,
            [mock.call(obsolete_model), mock.call(obsolete_shape)],
        )
        backend.shutdown()

    def test_second_server_skips_port_owned_by_first(self):
        self._patch_spawn_and_health()
        self.manager.ensure(self.key, load_timeout=5)
        other = ServerKey(str(self._model_path), 65536, 4)
        self.manager.ensure(other, load_timeout=5)
        second_argv = self.fake_popen.call_args_list[1][0][0]
        self.assertEqual(
            second_argv[second_argv.index("--port") + 1], "11436",
            "la seconda istanza non deve riutilizzare la porta 11435",
        )

    def test_reap_port_only_kills_orphaned_llama_servers(self):
        app_pid = os.getpid()
        orphan_pid = 424242      # ppid 1 → stale server of a crashed run
        live_parent_pid = 424243  # ppid != 1 → another live application

        def fake_run(argv, **kwargs):
            result = mock.MagicMock()
            if argv[0] == "lsof":
                # The port has: this app (client socket), an orphaned
                # llama-server, a llama-server with a live parent, and an
                # unrelated process.
                result.stdout = (
                    f"{app_pid}\n{orphan_pid}\n{live_parent_pid}\n99999\n"
                )
            else:  # ps -p <pid> -o ppid=,comm=
                pid = int(argv[2])
                table = {
                    orphan_pid: "1 llama-server",
                    live_parent_pid: "999 llama-server",
                    99999: "4242 python",
                }
                result.stdout = table.get(pid, "4242 python") + "\n"
            return result

        with mock.patch(
            "emr_analyzer.llm_backend.server_manager.subprocess.run",
            side_effect=fake_run,
        ), mock.patch(
            "emr_analyzer.llm_backend.server_manager.os.kill"
        ) as fake_kill:
            self.manager._reap_port(11435)

        fake_kill.assert_called_once_with(orphan_pid, 15)

    def test_early_death_with_bad_flag_retries_without_rea(self):
        self._patch_spawn_and_health()
        self.fake_early_exit.side_effect = [True, False]

        # First process dies on the unknown -rea flag; the retried one lives.
        class _FlagProc(_FakeProc):
            def poll(self):
                return 1

        spawn_calls = []

        def spawner(argv, **kwargs):
            spawn_calls.append(argv)
            if len(spawn_calls) == 1:
                return _FlagProc(argv)
            return _FakeProc(argv)

        self.fake_popen.side_effect = spawner
        # Populate the stderr buffer with the reasoning-flag error line.
        stderr_patch = mock.patch.object(
            ServerManager, "_drain_stderr", autospec=True,
            side_effect=lambda proc, buffer: buffer.append(
                "error: unknown argument -rea\n"
            ),
        )
        stderr_patch.start()
        self.patches.append(stderr_patch)

        self.manager.ensure(self.key, load_timeout=5)
        self.assertEqual(len(spawn_calls), 2)
        # Second attempt must not contain -rea.
        self.assertNotIn("-rea", spawn_calls[1])

    def test_binary_missing_raises(self):
        # ``None`` normally triggers auto-discovery.  Make the missing-binary
        # condition independent from what happens to be installed on the host.
        with mock.patch(
            "emr_analyzer.llm_backend.server_manager.find_server_binary",
            return_value=None,
        ):
            manager = ServerManager(binary=None)
        with self.assertRaises(BackendError):
            manager.ensure(self.key, load_timeout=1)

    def test_missing_gguf_raises(self):
        manager = ServerManager(binary="/fake/llama-server")
        with self.assertRaises(BackendError):
            manager.ensure(
                ServerKey("/nonexistent/model.gguf", 32768, 1),
                load_timeout=1,
            )

    def test_stop_and_stop_all(self):
        self._patch_spawn_and_health()
        self.manager.ensure(self.key, load_timeout=5)
        self.assertTrue(self.manager.stop(self.key))
        self.assertFalse(self.manager.stop(self.key))  # idempotent
        self.manager.ensure(self.key, load_timeout=5)
        self.assertEqual(self.manager.stop_all(), 1)

    def test_status_maps_503_to_loading(self):
        self._patch_spawn_and_health(healthy=False)
        with self.assertRaises(BackendError):
            self.manager.ensure(self.key, load_timeout=0.5)


if __name__ == "__main__":
    unittest.main()

"""vLLM integration tests; no GPU, network, or real server is used."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from emr_analyzer.extraction.llm_client import LlmClient
from emr_analyzer.llm_backend.vllm_backend import (
    VllmBackend,
    list_cached_vllm_models,
    resolve_vllm_model,
)
from emr_analyzer.llm_backend.vllm_server_manager import (
    VllmServerKey,
    VllmServerManager,
)
from emr_analyzer.settings import LLMRoleConfig


class _FakeProc:
    def __init__(self, argv):
        self.argv = argv
        self.stderr = None
        self.returncode = None

    def poll(self):
        return self.returncode

    def terminate(self):
        self.returncode = -15

    def kill(self):
        self.returncode = -9

    def wait(self, timeout=None):
        self.returncode = 0


class _Response:
    def __init__(self, payload=None, status_code=200, text=None):
        self._payload = payload or {}
        self.status_code = status_code
        self.text = text if text is not None else json.dumps(self._payload)

    def json(self):
        return self._payload


class CachedModelTest(unittest.TestCase):
    def test_huggingface_cache_is_discovered_without_network(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            snapshot = (
                root / "models--Qwen--Clinical-Test" / "snapshots" / "abc"
            )
            snapshot.mkdir(parents=True)
            (snapshot / "config.json").write_text(
                json.dumps({
                    "model_type": "qwen3",
                    "architectures": ["Qwen3ForCausalLM"],
                    "max_position_embeddings": 65536,
                }),
                encoding="utf-8",
            )
            (snapshot / "model.safetensors").write_bytes(b"fake-weights")
            with mock.patch.dict("os.environ", {"HF_HUB_CACHE": str(root)}):
                self.assertEqual(
                    list_cached_vllm_models(), ["Qwen/Clinical-Test"]
                )
                info = resolve_vllm_model("Qwen/Clinical-Test")
            self.assertEqual(info["architecture"], "qwen3")
            self.assertEqual(info["max_context_length"], 65536)
            self.assertEqual(info["size_bytes"], len(b"fake-weights"))

    def test_missing_repo_is_not_treated_as_available(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(
            "os.environ", {"HF_HUB_CACHE": tmp}
        ):
            self.assertIsNone(resolve_vllm_model("Org/not-downloaded"))


class VllmManagerTest(unittest.TestCase):
    def test_spawn_uses_official_serve_flags_and_offline_environment(self):
        manager = VllmServerManager(binary="/fake/vllm")
        key = VllmServerKey(
            model="Qwen/Test",
            ctx_size=32768,
            max_num_seqs=4,
            tensor_parallel_size=1,
            dtype="bfloat16",
            gpu_memory_utilization=0.85,
            quantization="awq",
            enforce_eager=True,
        )
        with (
            mock.patch(
                "emr_analyzer.llm_backend.vllm_server_manager.subprocess.Popen",
                side_effect=lambda argv, **kwargs: _FakeProc(argv),
            ) as popen,
            mock.patch(
                "emr_analyzer.llm_backend.vllm_server_manager.httpx.get",
                return_value=_Response(),
            ),
            mock.patch(
                "emr_analyzer.llm_backend.vllm_server_manager._port_is_free",
                return_value=True,
            ),
        ):
            url = manager.ensure(key, load_timeout=2)
        self.assertEqual(url, "http://127.0.0.1:11535")
        argv = popen.call_args.args[0]
        env = popen.call_args.kwargs["env"]
        self.assertEqual(argv[:3], ["/fake/vllm", "serve", "Qwen/Test"])
        self.assertEqual(argv[argv.index("--max-model-len") + 1], "32768")
        self.assertEqual(argv[argv.index("--max-num-seqs") + 1], "4")
        self.assertEqual(argv[argv.index("--quantization") + 1], "awq")
        self.assertIn("--enforce-eager", argv)
        self.assertEqual(env["HF_HUB_OFFLINE"], "1")
        self.assertEqual(env["TRANSFORMERS_OFFLINE"], "1")
        self.assertEqual(env["VLLM_WORKER_MULTIPROC_METHOD"], "spawn")
        self.assertEqual(env["PATH"].split(":", 1)[0], "/fake")
        manager.stop_all()


class VllmBackendTest(unittest.TestCase):
    def test_chat_uses_openai_endpoint_and_json_schema(self):
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp) / "model"
            model_dir.mkdir()
            (model_dir / "config.json").write_text(
                '{"architectures":["TestForCausalLM"],'
                '"max_position_embeddings":32768}', encoding="utf-8"
            )
            (model_dir / "model.safetensors").write_bytes(b"fake-weights")
            manager = mock.MagicMock()
            manager.usable.return_value = True
            manager.ensure.return_value = "http://127.0.0.1:11535"
            backend = VllmBackend(manager=manager)
            backend._http = mock.MagicMock()
            backend._http.post.return_value = _Response({
                "choices": [{
                    "message": {"content": '{"ok":true}'},
                    "finish_reason": "stop",
                }],
                "usage": {"completion_tokens": 4},
            })
            config = LLMRoleConfig(
                model=str(model_dir), backend="vllm",
                context_length=32768, parallel_workers=2,
            )
            response_format = backend.structured_response_format({
                "type": "object",
                "properties": {"ok": {"type": "boolean"}},
                "required": ["ok"],
            })
            result = backend.chat(
                config,
                [{"role": "user", "content": "test"}],
                response_format=response_format,
            )
        request = backend._http.post.call_args.kwargs["json"]
        self.assertTrue(result["content"])
        self.assertEqual(
            request["response_format"]["type"], "json_schema"
        )
        self.assertEqual(
            request["response_format"]["json_schema"]["strict"], True
        )
        self.assertEqual(
            request["chat_template_kwargs"], {"enable_thinking": False}
        )

    def test_llm_client_selects_configured_backend(self):
        fake = mock.MagicMock()
        config = LLMRoleConfig(model="Org/Test", backend="vllm")
        with mock.patch(
            "emr_analyzer.extraction.llm_client.get_backend",
            return_value=fake,
        ) as get_backend:
            client = LlmClient(config=config)
        get_backend.assert_called_once_with("vllm")
        self.assertIs(client.backend, fake)


if __name__ == "__main__":
    unittest.main()

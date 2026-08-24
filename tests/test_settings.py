from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from emr_analyzer.config import (
    CLINICAL_STATE_LLM_MODEL_NAME,
    DOCUMENT_LLM_MODEL_NAME,
)
from emr_analyzer.settings import (
    LLMRoleConfig,
    load_chat_preferences,
    default_llm_configs,
    load_llm_configs,
    load_model_assignments,
    save_chat_preferences,
    save_llm_configs,
    save_model_assignment,
)


class ModelSettingsTest(unittest.TestCase):
    def setUp(self):
        # Isolate from the real ~/.emr_analyzer/models/index.json: legacy
        # names must round-trip unchanged when no GGUF index exists.
        patcher = patch(
            "emr_analyzer.llm_backend.model_store.load_index",
            return_value={},
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_missing_settings_use_function_defaults(self):
        with tempfile.TemporaryDirectory() as tmp:
            assignments = load_model_assignments(Path(tmp) / "missing.json")
            self.assertEqual(assignments["document"], DOCUMENT_LLM_MODEL_NAME)
            self.assertEqual(
                assignments["clinical_state"], CLINICAL_STATE_LLM_MODEL_NAME
            )
            self.assertEqual(
                assignments["atomic_evidence"],
                CLINICAL_STATE_LLM_MODEL_NAME,
            )
            self.assertEqual(
                assignments["clinical_events"],
                CLINICAL_STATE_LLM_MODEL_NAME,
            )
            configs = default_llm_configs()
            self.assertEqual(configs["atomic_evidence"].temperature, 0.0)
            self.assertFalse(
                configs["atomic_evidence"].speculative_decoding
            )

    def test_assignments_persist_independently_and_allow_disabled_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "settings.json"
            save_model_assignment("document", "gemma3:12b", path)
            save_model_assignment("clinical_state", "qwen3:14b", path)
            self.assertEqual(load_model_assignments(path), {
                "document": "gemma3:12b",
                "atomic_evidence": "qwen3-14b",
                "clinical_events": "qwen3-14b",
                "clinical_state": "qwen3:14b",
            })

            save_model_assignment("document", "", path)
            self.assertEqual(load_model_assignments(path)["document"], "")
            self.assertEqual(
                load_model_assignments(path)["clinical_state"], "qwen3:14b"
            )

    def test_full_llm_parameters_roundtrip_independently(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "settings.json"
            configs = {
                "document": LLMRoleConfig(
                    model="gemma3:12b", temperature=0.0,
                    context_length=8192, max_output_tokens=2048,
                    top_p=0.8, top_k=20, seed=7,
                    keep_alive_minutes=4,
                ),
                "clinical_state": LLMRoleConfig(
                    model="qwen3:14b", temperature=0.2,
                    context_length=65536, max_output_tokens=8192,
                    top_p=0.95, top_k=50, seed=42,
                    keep_alive_minutes=30,
                    speculative_decoding=True,
                ),
                "atomic_evidence": LLMRoleConfig(
                    model="atomic-model", temperature=0.0,
                    context_length=32768, max_output_tokens=4096,
                    parallel_workers=4,
                ),
                "clinical_events": LLMRoleConfig(
                    model="event-model", temperature=0.1,
                    context_length=49152, max_output_tokens=6144,
                    parallel_workers=2,
                ),
            }

            save_llm_configs(configs, path)

            self.assertEqual(load_llm_configs(path), configs)
            self.assertEqual(load_model_assignments(path), {
                "document": "gemma3:12b",
                "atomic_evidence": "atomic-model",
                "clinical_events": "event-model",
                "clinical_state": "qwen3:14b",
            })

    def test_legacy_model_mapping_is_migrated_with_parameter_defaults(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "settings.json"
            path.write_text(
                '{"models":{"document":"legacy-doc",'
                '"clinical_state":"legacy-state"}}',
                encoding="utf-8",
            )

            configs = load_llm_configs(path)

            self.assertEqual(configs["document"].model, "legacy-doc")
            self.assertEqual(configs["clinical_state"].model, "legacy-state")
            self.assertEqual(configs["atomic_evidence"].model, "legacy-state")
            self.assertEqual(configs["clinical_events"].model, "legacy-state")
            self.assertGreater(configs["document"].context_length, 0)

    def test_two_role_llm_payload_clones_full_state_config_on_migration(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "settings.json"
            path.write_text(
                '{"llm":{"document":{"model":"doc"},'
                '"clinical_state":{"model":"state","temperature":0.37,'
                '"context_length":49152,"parallel_workers":3}}}',
                encoding="utf-8",
            )

            configs = load_llm_configs(path)

            for role in ("atomic_evidence", "clinical_events"):
                self.assertEqual(configs[role].model, "state")
                self.assertEqual(configs[role].temperature, 0.37)
                self.assertEqual(configs[role].context_length, 49152)
                self.assertEqual(configs[role].parallel_workers, 3)

    def test_vllm_parameters_roundtrip_and_legacy_defaults_to_llama(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "settings.json"
            configs = {
                "document": LLMRoleConfig(model="qwen3-14b"),
                "atomic_evidence": LLMRoleConfig(
                    model="Qwen/Clinical",
                    backend="vllm",
                    vllm_dtype="bfloat16",
                    vllm_gpu_memory_utilization=0.85,
                    vllm_tensor_parallel_size=2,
                    vllm_quantization="awq",
                    vllm_enforce_eager=True,
                ),
                "clinical_events": LLMRoleConfig(model="qwen3-14b"),
                "clinical_state": LLMRoleConfig(model="qwen3-14b"),
            }
            save_llm_configs(configs, path)
            loaded = load_llm_configs(path)
            self.assertEqual(loaded, configs)

            path.write_text(
                '{"llm":{"document":{"model":"legacy"},'
                '"clinical_state":{"model":"legacy"}}}',
                encoding="utf-8",
            )
            self.assertEqual(
                load_llm_configs(path)["document"].backend, "llama_cpp"
            )


class ChatPreferencesTest(unittest.TestCase):
    """Per-patient chat panel preferences (payload["chat"])."""

    def test_default_when_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "settings.json"
            self.assertEqual(
                load_chat_preferences(path),
                {"use_conversation_context": False},
            )

    def test_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "settings.json"
            save_chat_preferences({"use_conversation_context": True}, path)
            self.assertEqual(
                load_chat_preferences(path),
                {"use_conversation_context": True},
            )
            save_chat_preferences({"use_conversation_context": False}, path)
            self.assertEqual(
                load_chat_preferences(path),
                {"use_conversation_context": False},
            )

    def test_preserves_llm_keys_and_vice_versa(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "settings.json"
            save_chat_preferences({"use_conversation_context": True}, path)
            save_model_assignment("document", "qwen3-14b", path)
            self.assertEqual(
                load_chat_preferences(path),
                {"use_conversation_context": True},
            )
            self.assertEqual(
                load_model_assignments(path)["document"], "qwen3-14b"
            )

    def test_non_dict_prefs_coerced(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "settings.json"
            save_chat_preferences({"use_conversation_context": "si"}, path)
            self.assertEqual(
                load_chat_preferences(path),
                {"use_conversation_context": True},
            )

    def test_corrupt_payload_tolerated(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "settings.json"
            path.write_text("{not-json", encoding="utf-8")
            self.assertEqual(
                load_chat_preferences(path),
                {"use_conversation_context": False},
            )


if __name__ == "__main__":
    unittest.main()

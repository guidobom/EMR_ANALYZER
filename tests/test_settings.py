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

    def test_assignments_persist_independently_and_allow_disabled_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "settings.json"
            save_model_assignment("document", "gemma3:12b", path)
            save_model_assignment("clinical_state", "qwen3:14b", path)
            self.assertEqual(load_model_assignments(path), {
                "document": "gemma3:12b",
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
                ),
            }

            save_llm_configs(configs, path)

            self.assertEqual(load_llm_configs(path), configs)
            self.assertEqual(load_model_assignments(path), {
                "document": "gemma3:12b",
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
            self.assertGreater(configs["document"].context_length, 0)


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

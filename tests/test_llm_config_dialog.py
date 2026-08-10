from __future__ import annotations

import os
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtWidgets import QApplication

from emr_analyzer.extraction.llm_client import LlmClient
from emr_analyzer.gui.llm_config_dialog import LLMConfigDialog
from emr_analyzer.settings import LLMRoleConfig


class LLMConfigDialogTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_dialog_exposes_per_model_and_global_gpu_release(self):
        configs = {
            "document": LLMRoleConfig(model="gemma3:12b"),
            "clinical_state": LLMRoleConfig(model="qwen3:14b"),
        }
        with (
            patch.object(
                LlmClient,
                "model_capabilities",
                return_value={"max_context_length": 32768},
            ),
            patch.object(LlmClient, "loaded_model_info", return_value=None),
        ):
            dialog = LLMConfigDialog(
                configs, ["gemma3:12b", "qwen3:14b"]
            )

        self.assertEqual(
            dialog._widgets["document"]["unload"].text(),
            "■ Scarica dalla GPU",
        )
        self.assertEqual(
            dialog._widgets["clinical_state"]["unload"].text(),
            "■ Scarica dalla GPU",
        )
        self.assertEqual(
            dialog._unload_all_button.text(), "■ Libera tutta la GPU"
        )
        self.assertFalse(dialog._widgets["document"]["unload"].isEnabled())
        dialog._runtime_timer.stop()
        dialog.deleteLater()


if __name__ == "__main__":
    unittest.main()

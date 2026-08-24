from __future__ import annotations

import os
import tempfile
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtWidgets import QApplication

from emr_analyzer.extraction.llm_client import LlmClient
from emr_analyzer.gui.llm_config_dialog import LLMConfigDialog
from emr_analyzer.llm_backend.diagnostics import AccelerationDiagnostic
from emr_analyzer.settings import LLMRoleConfig, MODEL_ROLES


class LLMConfigDialogTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_dialog_exposes_per_model_and_global_memory_release(self):
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
            "■ Scarica dalla memoria",
        )
        self.assertEqual(tuple(dialog._widgets), MODEL_ROLES)
        self.assertEqual(
            dialog._widgets["clinical_state"]["unload"].text(),
            "■ Scarica dalla memoria",
        )
        self.assertEqual(
            dialog._unload_all_button.text(), "■ Libera tutti i modelli"
        )
        self.assertFalse(dialog._widgets["document"]["unload"].isEnabled())
        dialog._runtime_timer.stop()
        dialog.deleteLater()

    def test_dialog_displays_acceleration_diagnostic(self):
        configs = {
            "document": LLMRoleConfig(model=""),
            "clinical_state": LLMRoleConfig(model=""),
        }
        dialog = LLMConfigDialog(configs, [])
        diagnostic = AccelerationDiagnostic(
            status="accelerated",
            expected_backend="CUDA",
            runtime_backend="CUDA",
            compiled_backends=("CUDA",),
            devices=("CUDA0: NVIDIA GB10",),
            binary_path="/opt/llama-server",
            system="Linux",
            machine="aarch64",
            gpu_name="NVIDIA GB10",
            summary="Accelerazione CUDA disponibile: CUDA0: NVIDIA GB10.",
            details="Dettagli diagnostici",
        )

        dialog._show_acceleration_diagnostic(diagnostic)

        self.assertIn("CUDA disponibile", dialog._acceleration_status.text())
        self.assertEqual(
            dialog._acceleration_status.toolTip(), "Dettagli diagnostici"
        )
        self.assertEqual(
            dialog._acceleration_probe_button.text(), "Verifica di nuovo"
        )
        dialog._runtime_timer.stop()
        dialog.deleteLater()

    def test_dialog_exposes_vllm_engine_parameters_per_role(self):
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = os.path.join(tmp, "clinical-model")
            os.mkdir(model_dir)
            with open(
                os.path.join(model_dir, "config.json"), "w", encoding="utf-8"
            ) as handle:
                handle.write(
                    '{"architectures":["TestForCausalLM"],'
                    '"max_position_embeddings":65536}'
                )
            with open(
                os.path.join(model_dir, "model.safetensors"), "wb"
            ) as handle:
                handle.write(b"fake-weights")
            configs = {
                "document": LLMRoleConfig(
                    model=model_dir,
                    backend="vllm",
                    vllm_dtype="bfloat16",
                    vllm_gpu_memory_utilization=0.85,
                    vllm_tensor_parallel_size=1,
                ),
                "clinical_state": LLMRoleConfig(model=""),
            }
            with patch.object(LlmClient, "loaded_model_info", return_value=None):
                dialog = LLMConfigDialog(configs, [])

        widgets = dialog._widgets["document"]
        self.assertEqual(widgets["backend"].currentData(), "vllm")
        self.assertFalse(widgets["vllm_group"].isHidden())
        self.assertTrue(widgets["speculative_decoding"].isHidden())
        collected = dialog._collect_config("document")
        self.assertEqual(collected.backend, "vllm")
        self.assertEqual(collected.vllm_dtype, "bfloat16")
        self.assertAlmostEqual(
            collected.vllm_gpu_memory_utilization, 0.85
        )
        dialog._runtime_timer.stop()
        dialog.deleteLater()

    def test_dialog_exposes_installer_and_refreshes_model_selectors(self):
        configs = {
            "document": LLMRoleConfig(model="qwen3-14b"),
            "clinical_state": LLMRoleConfig(model="qwen3-14b"),
        }
        with (
            patch.object(
                LlmClient,
                "model_capabilities",
                return_value={"max_context_length": 32768},
            ),
            patch.object(LlmClient, "loaded_model_info", return_value=None),
        ):
            dialog = LLMConfigDialog(configs, ["qwen3-14b"])
            dialog._register_installed_model("medgemma-4b")

        self.assertEqual(
            dialog._manage_models_button.text(),
            "⬇ Scarica o importa modelli…",
        )
        self.assertIn("2", dialog._model_count_label.text())
        for widgets in dialog._widgets.values():
            self.assertGreaterEqual(
                widgets["model"].findData("medgemma-4b"), 0
            )
            self.assertEqual(
                widgets["model"].currentData(), "qwen3-14b"
            )
        dialog._runtime_timer.stop()
        dialog.deleteLater()

    def test_same_runtime_is_reported_as_one_shared_physical_server(self):
        configs = {
            "document": LLMRoleConfig(
                model="qwen3-14b", temperature=0.0,
                context_length=32768, max_output_tokens=10240,
                parallel_workers=3,
            ),
            "clinical_state": LLMRoleConfig(
                model="qwen3-14b", temperature=0.1,
                context_length=32768, max_output_tokens=6144,
                parallel_workers=3,
            ),
        }
        runtime = {
            "context_length": 32768,
            "slots": 3,
            "active_slots": 0,
            "processing": False,
        }
        with (
            patch.object(
                LlmClient,
                "model_capabilities",
                return_value={"max_context_length": 40960},
            ),
            patch.object(
                LlmClient,
                "runtime_identity",
                new=lambda client: (
                    "/models/qwen3-14b.gguf",
                    client.context_length,
                    client.parallel_workers,
                ),
            ),
            patch.object(
                LlmClient, "loaded_model_info", return_value=runtime
            ),
            patch(
                "emr_analyzer.utils.hardware.get_total_ram_gb",
                return_value=48.0,
            ),
            patch(
                "emr_analyzer.utils.hardware.get_available_ram_gb",
                return_value=13.0,
            ),
            patch(
                "emr_analyzer.utils.hardware.get_model_size_gb",
                return_value=9.3,
            ),
        ):
            dialog = LLMConfigDialog(configs, ["qwen3-14b"])

        self.assertIn(
            "Un solo server fisico condiviso",
            dialog._runtime_summary.text(),
        )
        self.assertIn("caricato · condiviso", dialog._widgets["document"]["status"].text())
        self.assertIn("3 reali", dialog._slots_label.text())
        worker_combo = dialog._widgets["document"]["workers_combo"]
        self.assertNotIn("RAM insufficiente", worker_combo.currentText())
        self.assertIn(
            "Capacità 48.0 GiB",
            dialog._widgets["document"]["workers_info"].text(),
        )
        clinical_warning = dialog._widgets["clinical_state"]["output_warning"]
        self.assertFalse(clinical_warning.isHidden())
        self.assertIn("8.192 token", clinical_warning.text())
        self.assertIn("analisi irAE complete", clinical_warning.text())
        dialog._runtime_timer.stop()
        dialog.deleteLater()

    def test_different_shapes_warn_and_can_be_aligned_for_sharing(self):
        configs = {
            "document": LLMRoleConfig(
                model="qwen3-14b", temperature=0.0,
                context_length=16384, max_output_tokens=4096,
                parallel_workers=1,
            ),
            "clinical_state": LLMRoleConfig(
                model="qwen3-14b", temperature=0.1,
                context_length=32768, max_output_tokens=6144,
                parallel_workers=3,
            ),
        }
        with (
            patch.object(
                LlmClient,
                "model_capabilities",
                return_value={"max_context_length": 40960},
            ),
            patch.object(
                LlmClient,
                "runtime_identity",
                new=lambda client: (
                    "/models/qwen3-14b.gguf",
                    client.context_length,
                    client.parallel_workers,
                ),
            ),
            patch.object(LlmClient, "loaded_model_info", return_value=None),
            patch(
                "emr_analyzer.utils.hardware.get_total_ram_gb",
                return_value=48.0,
            ),
            patch(
                "emr_analyzer.utils.hardware.get_model_size_gb",
                return_value=9.3,
            ),
        ):
            dialog = LLMConfigDialog(configs, ["qwen3-14b"])
            self.assertIn(
                "Lo stesso GGUF richiede più server fisici",
                dialog._runtime_summary.text(),
            )
            self.assertFalse(dialog._align_runtime_button.isHidden())
            dialog._align_shared_runtime()

        document = dialog._collect_config("document")
        clinical = dialog._collect_config("clinical_state")
        self.assertEqual(document.context_length, clinical.context_length)
        self.assertEqual(document.parallel_workers, clinical.parallel_workers)
        self.assertEqual(document.temperature, 0.0)
        self.assertEqual(clinical.temperature, 0.1)
        self.assertIn(
            "Un solo server fisico condiviso",
            dialog._runtime_summary.text(),
        )
        dialog._runtime_timer.stop()
        dialog.deleteLater()

    def test_unsafe_saved_slot_count_is_clamped_to_enabled_capacity(self):
        configs = {
            "document": LLMRoleConfig(
                model="qwen3-14b", context_length=32768,
                max_output_tokens=8192, parallel_workers=8,
            ),
            "clinical_state": LLMRoleConfig(model=""),
        }
        with (
            patch.object(
                LlmClient,
                "model_capabilities",
                return_value={"max_context_length": 40960},
            ),
            patch.object(
                LlmClient,
                "runtime_identity",
                new=lambda client: (
                    "/models/qwen3-14b.gguf",
                    client.context_length,
                    client.parallel_workers,
                ),
            ),
            patch.object(LlmClient, "loaded_model_info", return_value=None),
            patch(
                "emr_analyzer.utils.hardware.get_total_ram_gb",
                return_value=16.0,
            ),
            patch(
                "emr_analyzer.utils.hardware.get_model_size_gb",
                return_value=9.3,
            ),
        ):
            dialog = LLMConfigDialog(configs, ["qwen3-14b"])

        combo = dialog._widgets["document"]["workers_combo"]
        self.assertEqual(combo.currentData(), 1)
        self.assertFalse(combo.model().item(combo.findData(8)).isEnabled())
        self.assertIn(
            "Un solo server fisico configurato",
            dialog._runtime_summary.text(),
        )
        dialog._runtime_timer.stop()
        dialog.deleteLater()

    def test_cancel_unloads_runtime_started_only_for_dialog_test(self):
        configs = {
            "document": LLMRoleConfig(
                model="qwen3-14b", context_length=32768,
                max_output_tokens=8192, parallel_workers=3,
            ),
            "clinical_state": LLMRoleConfig(model=""),
        }
        with (
            patch.object(
                LlmClient,
                "model_capabilities",
                return_value={"max_context_length": 40960},
            ),
            patch.object(
                LlmClient,
                "runtime_identity",
                new=lambda client: (
                    "/models/qwen3-14b.gguf",
                    client.context_length,
                    client.parallel_workers,
                ),
            ),
            patch.object(LlmClient, "loaded_model_info", return_value=None),
        ):
            dialog = LLMConfigDialog(configs, ["qwen3-14b"])
        config = dialog._collect_config("document")
        identity = ("/models/qwen3-14b.gguf", 32768, 3)
        dialog._runtimes_started_in_dialog[identity] = config
        with patch.object(
            LlmClient,
            "unload_runtimes",
            return_value={"errors": {}, "unloaded": ["qwen3-14b"]},
        ) as unload:
            dialog.reject()

        unload.assert_called_once_with([config])
        dialog.deleteLater()


if __name__ == "__main__":
    unittest.main()

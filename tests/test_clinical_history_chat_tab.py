"""Offscreen tests for the per-patient chat trace in the Clinical History tab."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtWidgets import QApplication

from emr_analyzer.gui.clinical_history_tab import ClinicalHistoryTab
from emr_analyzer.models.chat_message import ChatMessage

from unittest import mock
from PyQt5.QtWidgets import QMessageBox


class FakeChatRepo:
    """In-memory chat repository keyed by patient."""

    def __init__(self, initial: dict[str, list[ChatMessage]] | None = None):
        self._store: dict[str, list[ChatMessage]] = {
            pid: list(msgs) for pid, msgs in (initial or {}).items()
        }
        self.added: list[ChatMessage] = []

    def add_message(self, message: ChatMessage) -> None:
        self.added.append(message)
        self._store.setdefault(message.patient_id, []).append(message)

    def get_by_patient(self, patient_id: str) -> list[ChatMessage]:
        return list(self._store.get(patient_id, []))

    def count_by_patient(self, patient_id: str) -> int:
        return len(self._store.get(patient_id, []))

    def clear_for_patient(self, patient_id: str) -> None:
        self._store.pop(patient_id, None)


class FakeTimelineRepo:
    def get_by_patient(self, patient_id: str):
        if not patient_id:
            return []
        return [FakeEntry()]


class FakeEntry:
    entry_id = "E1"
    patient_id = "P"
    date_observed = "2020-05-01"
    date_resolved = ""
    category = "diagnosis"
    description = "Melanoma dorsale"
    status = "active"
    is_golden = 0
    confidence = 0.8
    merged_into_ids: list[str] = []
    source_texts: list[str] = []

    def to_dict(self):
        return {
            "entry_id": self.entry_id,
            "date_observed": self.date_observed,
            "date_resolved": self.date_resolved,
            "category": self.category,
            "description": self.description,
        }


class FakeCSRepo:
    def load(self, patient_id: str):
        return None


class FakeLLM:
    is_available = True
    model = "qwen3-14b"

    def generate_text(self, prompt, system=""):
        return "risposta sintetica"


def make_message(patient_id: str, role: str, content: str,
                 model_used="", context_mode=0) -> ChatMessage:
    return ChatMessage(
        id=f"CHAT_{patient_id}_{role}_{content[:8]}",
        patient_id=patient_id,
        role=role,
        content=content,
        model_used=model_used,
        context_mode=context_mode,
        created_at="2026-08-19T10:00:00.000000",
    )


class ClinicalHistoryChatTabTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.settings_path = Path(self._tmp.name) / "settings.json"
        self.chat_repo = FakeChatRepo()
        self.tab = ClinicalHistoryTab(settings_path=self.settings_path)
        self.tab.set_services({
            "timeline_repo": FakeTimelineRepo(),
            "cs_repo": FakeCSRepo(),
            "chat_repo": self.chat_repo,
        })
        # Silence every modal dialog.
        for patch in (
            mock.patch.object(QMessageBox, "question",
                              return_value=QMessageBox.Yes),
            mock.patch.object(QMessageBox, "information",
                              return_value=None),
            mock.patch.object(QMessageBox, "warning", return_value=None),
            mock.patch.object(QMessageBox, "critical", return_value=None),
        ):
            patch.start()
            self.addCleanup(patch.stop)

    def _wait_query_worker(self):
        worker = self.tab._query_worker
        if worker is not None:
            worker.wait(2000)

    def tearDown(self):
        self._wait_query_worker()
        self.tab.deleteLater()
        self._tmp.cleanup()

    def test_load_patient_shows_only_that_patients_chat(self):
        self.chat_repo._store = {
            "P001": [make_message("P001", "user", "domanda A"),
                     make_message("P001", "assistant", "risposta A")],
            "P002": [make_message("P002", "user", "domanda B")],
        }
        self.tab.load_patient("P001")
        html = self.tab._chat_view.toHtml()
        self.assertIn("domanda A", html)
        self.assertIn("risposta A", html)
        self.assertNotIn("domanda B", html)

    def test_switching_patient_replaces_history(self):
        self.chat_repo._store = {
            "P001": [make_message("P001", "user", "domanda A")],
            "P002": [make_message("P002", "user", "domanda B")],
        }
        self.tab.load_patient("P001")
        self.tab.load_patient("P002")
        html = self.tab._chat_view.toHtml()
        self.assertNotIn("domanda A", html)
        self.assertIn("domanda B", html)

    def test_empty_patient_clears_panel(self):
        self.chat_repo._store = {
            "P001": [make_message("P001", "user", "domanda A")],
        }
        self.tab.load_patient("P001")
        self.tab.load_patient("")
        self.assertEqual(self.tab._chat_messages, [])
        self.assertNotIn("domanda A", self.tab._chat_view.toHtml())

    def test_query_persists_user_and_assistant_with_metadata(self):
        self.tab.load_patient("P001")
        self.tab._services["clinical_state_llm_client"] = FakeLLM()
        self.tab._query_text.setPlainText("Quale terapia?")
        self.tab._run_query()
        self.tab._on_query_result("Terapia con pembrolizumab.")

        roles = [m.role for m in self.chat_repo.added]
        self.assertEqual(roles, ["user", "assistant"])
        assistant = self.chat_repo.added[1]
        self.assertEqual(assistant.model_used, "qwen3-14b")
        self.assertEqual(assistant.context_mode, 0)
        html = self.tab._chat_view.toHtml()
        self.assertIn("Quale terapia?", html)
        self.assertIn("Terapia con pembrolizumab.", html)

    def test_toggle_persists_across_tab_instances(self):
        self.tab._use_context_check.setChecked(True)
        other = ClinicalHistoryTab(settings_path=self.settings_path)
        try:
            self.assertTrue(other._use_context_check.isChecked())
            self.assertTrue(other._use_conversation_context)
        finally:
            other.deleteLater()

    def test_local_search_fallback_records_pair(self):
        self.tab.load_patient("P001")
        # No LLM client in services → local keyword search path.
        self.tab._query_text.setPlainText("melanoma")
        self.tab._run_query()

        roles = [m.role for m in self.chat_repo.added]
        self.assertEqual(roles, ["user", "assistant"])
        assistant = self.chat_repo.added[1]
        self.assertEqual(assistant.model_used, "local_search")
        self.assertIn("Melanoma dorsale", assistant.content)

    def test_answer_after_patient_switch_goes_to_right_patient(self):
        self.tab.load_patient("P001")
        self.tab._services["clinical_state_llm_client"] = FakeLLM()
        self.tab._query_text.setPlainText("domanda per P001")
        self.tab._run_query()

        # User switches patient while the worker is in flight.
        self.chat_repo._store.setdefault(
            "P002", [make_message("P002", "user", "domanda B")]
        )
        self.tab.load_patient("P002")

        self.tab._on_query_result("risposta per P001")

        # Persisted to P001, never rendered on P002's panel.
        self.assertIn("risposta per P001",
                      [m.content for m in self.chat_repo.get_by_patient("P001")
                       if m.role == "assistant"])
        self.assertNotIn("risposta per P001",
                         self.tab._chat_view.toHtml())
        self.assertNotIn("risposta per P001",
                         [m.content for m in self.chat_repo.get_by_patient("P002")])


if __name__ == "__main__":
    unittest.main()


class ChatClearTest(ClinicalHistoryChatTabTest):
    """Deleting the per-patient chat history."""

    def _seed_history(self):
        self.chat_repo._store = {
            "P001": [make_message("P001", "user", "domanda A"),
                     make_message("P001", "assistant", "risposta A")],
            "P002": [make_message("P002", "user", "domanda B")],
        }
        self.tab.load_patient("P001")

    def test_clear_removes_only_current_patient(self):
        self._seed_history()
        self.assertTrue(self.tab._clear_chat_btn.isEnabled())

        self.tab._on_clear_chat()  # QMessageBox.question → Yes (patched)

        self.assertEqual(self.tab._chat_messages, [])
        self.assertNotIn("domanda A", self.tab._chat_view.toHtml())
        self.assertFalse(self.tab._clear_chat_btn.isEnabled())
        # The other patient's history is untouched.
        self.assertEqual(len(self.chat_repo.get_by_patient("P002")), 1)

    def test_clear_works_with_context_toggle_on(self):
        self._seed_history()
        self.tab._use_context_check.setChecked(True)
        self.tab._on_clear_chat()
        self.assertEqual(self.tab._chat_messages, [])
        self.assertEqual(self.chat_repo.count_by_patient("P001"), 0)

    def test_clear_works_with_context_toggle_off(self):
        self._seed_history()
        self.tab._use_context_check.setChecked(False)
        self.tab._on_clear_chat()
        self.assertEqual(self.chat_repo.count_by_patient("P001"), 0)

    def test_clear_without_history_is_noop(self):
        self.tab.load_patient("P001")
        self.assertFalse(self.tab._clear_chat_btn.isEnabled())
        self.tab._on_clear_chat()
        self.assertEqual(self.chat_repo.count_by_patient("P001"), 0)


if __name__ == "__main__":
    unittest.main()


class IraePresetTest(ClinicalHistoryChatTabTest):
    """The dropdown preset launches the full-registry irAE analysis."""

    def test_preset_triggers_analysis_and_resets_combo(self):
        index = self.tab._query_preset.findData("__IRAE_ANALYSIS__")
        self.assertGreaterEqual(index, 0)
        with mock.patch.object(
            self.tab, "_on_irae_analysis"
        ) as analysis:
            self.tab._query_preset.setCurrentIndex(index)
            analysis.assert_called_once()
            # The combo returns to the placeholder entry.
            self.assertEqual(self.tab._query_preset.currentIndex(), 0)
            self.assertEqual(self.tab._query_preset.currentData(), "")

    def test_preset_does_not_fill_query_box(self):
        index = self.tab._query_preset.findData("__IRAE_ANALYSIS__")
        with mock.patch.object(self.tab, "_on_irae_analysis"):
            self.tab._query_preset.setCurrentIndex(index)
        self.assertEqual(self.tab._query_text.toPlainText(), "")


class BusyGuardTest(ClinicalHistoryChatTabTest):
    """The busy guard names the running operation and greys the entry."""

    def _fake_running_worker(self):
        worker = mock.MagicMock()
        worker.isRunning.return_value = True
        self.tab._query_worker = worker
        self.tab._update_busy_ui()
        return worker

    def test_busy_state_disables_preset_and_button(self):
        self.tab.load_patient("P001")
        self._fake_running_worker()
        index = self.tab._query_preset.findData("__IRAE_ANALYSIS__")
        self.assertFalse(
            self.tab._query_preset.model().item(index).isEnabled()
        )
        self.assertFalse(self.tab._irae_btn.isEnabled())

    def test_guard_message_names_the_operation(self):
        self.tab.load_patient("P001")
        self._fake_running_worker()
        self.assertEqual(
            self.tab._running_operation_name(), "l'interrogazione"
        )
        with mock.patch.object(
            QMessageBox, "information", return_value=None
        ) as info:
            self.assertTrue(self.tab._guard_busy())
        self.assertIn("l'interrogazione", info.call_args[0][2])

    def test_release_reenables_entry_points(self):
        self.tab.load_patient("P001")
        worker = self._fake_running_worker()
        worker.isRunning.return_value = False
        self.tab._release_worker("_query_worker")
        index = self.tab._query_preset.findData("__IRAE_ANALYSIS__")
        self.assertTrue(
            self.tab._query_preset.model().item(index).isEnabled()
        )
        self.assertTrue(self.tab._irae_btn.isEnabled())


class IraeNoRegistryTest(ClinicalHistoryChatTabTest):
    """The irAE entry points need a registry: no silent no-ops."""

    def _without_registry(self):
        with mock.patch.object(
            FakeTimelineRepo, "get_by_patient", return_value=[]
        ):
            self.tab.load_patient("P001")

    def test_preset_and_button_disabled_without_registry(self):
        self._without_registry()
        index = self.tab._query_preset.findData("__IRAE_ANALYSIS__")
        self.assertFalse(
            self.tab._query_preset.model().item(index).isEnabled()
        )
        self.assertFalse(self.tab._irae_btn.isEnabled())

    def test_direct_call_without_evidence_shows_message(self):
        # The single-patient button now runs the 3-layer pipeline, whose data
        # prerequisite is the ATOMIC EVIDENCE (not the timeline registry): a
        # direct call without any must explain the blocker, never no-op.
        self._without_registry()
        with mock.patch.object(
            QMessageBox, "information", return_value=None
        ) as info:
            self.tab._on_irae_analysis()
        self.assertIn("Nessuna evidenza atomica", info.call_args[0][1])
        self.assertIsNone(self.tab._irae_worker)

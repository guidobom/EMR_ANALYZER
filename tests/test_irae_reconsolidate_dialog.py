"""GUI tests: the re-consolidation button in ``IraeResultDialog`` and the
failure summary surfaced by ``IraeQueueResultDialog``."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtWidgets import QApplication, QMessageBox

from emr_analyzer.gui.irae_queue_result_dialog import (
    IraeQueueResultDialog,
    collect_irae_failures,
)
from emr_analyzer.gui.irae_result_dialog import IraeResultDialog


class FakeLlm:
    is_available = True


def _report(*, applied: bool = False, with_organs: bool = True) -> dict:
    return {
        "analyzed_at": "2026-09-01T10:00:00",
        "anchor": {
            "first_drug": "nivolumab", "first_raw": "01/09/2022",
            "first_date": "2022-09-01",
            "last_drug": "nivolumab", "last_raw": "01/09/2022",
            "last_date": "2022-09-01",
            "occurrences": 1,
        },
        "organ_results": (
            {
                "Miocardite/Cardiotossicità": {
                    "organ": "Miocardite/Cardiotossicità",
                    "iraes": [{
                        "organ": "Miocardite/Cardiotossicità",
                        "irAE_type": "Miocardite", "ctcae_grade": "G2",
                        "first_onset_date": "2022-11-22",
                        "probability_immune": "PROBABILE",
                        "key_evidence_ids": ["E-TROP"],
                    }],
                },
            }
            if with_organs
            else {}
        ),
        "iraes": [{"irAE_type": "Miocardite da ICI", "ctcae_grade": "G2"}],
        "consolidation": {
            "applied": applied,
            "error": None if applied else "OutputLimitError: x",
        },
        "evidence": [],
        "candidates_total": 1,
    }


class ReconsolidateDialogTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self._ws = mock.patch(
            "emr_analyzer.gui.irae_result_dialog.active_workspace.path",
            Path(self._tmp.name),
        )
        self._ws.start()
        self.addCleanup(self._ws.stop)

    def _dialog(self, report=None, services=None):
        return IraeResultDialog(
            report if report is not None else _report(),
            patient_id="P001",
            services=(
                services
                if services is not None
                else {"clinical_state_llm_client": FakeLlm()}
            ),
        )

    def test_button_enabled_when_failed_and_client(self):
        dialog = self._dialog()
        self.assertTrue(dialog._can_reconsolidate())
        self.assertTrue(dialog._recon_btn.isEnabled())

    def test_button_disabled_without_organ_results(self):
        dialog = self._dialog(report=_report(with_organs=False))
        self.assertFalse(dialog._can_reconsolidate())
        self.assertFalse(dialog._recon_btn.isEnabled())

    def test_button_disabled_without_client(self):
        dialog = self._dialog(services={"clinical_state_llm_client": None})
        self.assertFalse(dialog._can_reconsolidate())

    def test_button_disabled_when_consolidation_applied(self):
        # A patient whose consolidation already succeeded must NOT be touched.
        dialog = self._dialog(report=_report(applied=True))
        self.assertFalse(dialog._can_reconsolidate())
        self.assertFalse(dialog._recon_btn.isEnabled())

    def test_button_disabled_while_busy(self):
        dialog = self._dialog()
        worker = mock.MagicMock()
        worker.isRunning.return_value = True
        dialog._recon_worker = worker
        self.assertFalse(dialog._can_reconsolidate())

    def test_handler_launches_worker_and_updates_view_on_ready(self):
        import emr_analyzer.gui.workers as workers

        llm = FakeLlm()
        dialog = self._dialog(services={"clinical_state_llm_client": llm})
        with mock.patch.object(
            QMessageBox, "question", return_value=QMessageBox.Yes
        ), mock.patch.object(
            workers, "IraeReconsolidateWorker"
        ) as worker_cls:
            dialog._on_reconsolidate()
            worker_cls.assert_called_once()
            args, kwargs = worker_cls.call_args
            self.assertIs(args[0], llm)
            self.assertEqual(kwargs["patient_id"], "P001")
            worker_cls.return_value.start.assert_called_once()

            updated = dict(dialog._structured_report,
                           **{"reconsolidated": True})
            ready_cb = (
                worker_cls.return_value.ready.connect.call_args[0][0]
            )
            ready_cb(updated)
        self.assertIs(dialog._structured_report, updated)
        self.assertIs(dialog._tab._raw_report, updated)

    def test_handler_aborts_when_confirm_no(self):
        import emr_analyzer.gui.workers as workers

        dialog = self._dialog()
        with mock.patch.object(
            QMessageBox, "question", return_value=QMessageBox.No
        ), mock.patch.object(workers, "IraeReconsolidateWorker") as worker_cls:
            dialog._on_reconsolidate()
            worker_cls.assert_not_called()

    def test_ready_with_reconsolidation_error_warns(self):
        dialog = self._dialog()
        with mock.patch.object(
            QMessageBox, "warning", return_value=None
        ) as warning:
            updated = dict(
                dialog._structured_report,
                reconsolidation_error="OutputLimitError: ancora",
            )
            dialog._on_reconsolidate_ready(updated)
            warning.assert_called_once()

    def test_error_shows_message_box(self):
        dialog = self._dialog()
        with mock.patch.object(
            QMessageBox, "critical", return_value=None
        ) as critical:
            dialog._on_reconsolidate_error("boom")
            critical.assert_called_once()
            self.assertIn("boom", critical.call_args[0][2])


# ---------------------------------------------------------------------------
# Failure summary (IraeQueueResultDialog.collect_irae_failures)
# ---------------------------------------------------------------------------


class CollectIraeFailuresTest(unittest.TestCase):
    def test_queue_level_error(self):
        failures = collect_irae_failures([{
            "patient_id": "P001", "error": "Errore di coda", "structured": None,
        }])
        self.assertEqual(len(failures), 1)
        self.assertEqual(failures[0]["problem"], "Errore di analisi")
        self.assertEqual(failures[0]["detail"], "Errore di coda")

    def test_consolidation_failed_with_error(self):
        structured = _report()  # applied=False
        failures = collect_irae_failures([{
            "patient_id": "P001", "error": None, "structured": structured,
        }])
        self.assertEqual(len(failures), 1)
        self.assertEqual(
            failures[0]["problem"], "Consolidamento finale non applicato"
        )
        self.assertIn("OutputLimitError", failures[0]["detail"])

    def test_consolidation_failed_without_error_has_default_detail(self):
        structured = _report()
        structured["consolidation"] = {"applied": False, "error": None}
        failures = collect_irae_failures([{
            "patient_id": "P001", "error": None, "structured": structured,
        }])
        self.assertEqual(
            failures[0]["detail"],
            "Consolidamento non applicato (nessun irAE confermato).",
        )

    def test_organ_error(self):
        structured = _report()
        structured["organ_results"]["Epatite"] = {
            "organ": "Epatite", "iraes": [],
            "error": "OutputLimitError: organo",
        }
        failures = collect_irae_failures([{
            "patient_id": "P001", "error": None, "structured": structured,
        }])
        problems = [f["problem"] for f in failures]
        self.assertIn("Analisi Epatite", problems)

    def test_clean_result_excluded(self):
        structured = _report(applied=True)
        failures = collect_irae_failures([{
            "patient_id": "P001", "error": None, "structured": structured,
        }])
        self.assertEqual(failures, [])

    def test_all_clean_returns_empty(self):
        self.assertEqual(collect_irae_failures([]), [])


class ProblemsTabTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_problems_tab_created_only_when_failures(self):
        results = [{
            "patient_id": "P001", "label": "P001", "markdown": "",
            "error": "Errore di coda", "structured": None,
        }]
        dialog = IraeQueueResultDialog(results, services=None)
        try:
            self.assertEqual(dialog._tabs.tabText(0), "⚠️ Problemi (1)")
        finally:
            dialog.deleteLater()

    def test_no_problems_tab_when_all_clean(self):
        results = [{
            "patient_id": "P001", "label": "P001", "markdown": "# ok",
            "error": None,
            "structured": {"consolidation": {"applied": True},
                           "organ_results": {}, "iraes": []},
        }]
        dialog = IraeQueueResultDialog(results, services=None)
        try:
            texts = [
                dialog._tabs.tabText(i) for i in range(dialog._tabs.count())
            ]
            self.assertNotIn("⚠️ Problemi (1)", texts)
        finally:
            dialog.deleteLater()


if __name__ == "__main__":
    unittest.main()

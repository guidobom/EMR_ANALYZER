"""Tests for the multi-patient irAE analysis queue."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtWidgets import QApplication, QMessageBox

from emr_analyzer.database.engine import DatabaseEngine
from emr_analyzer.database.migrations import init_database
from emr_analyzer.database.patient_repo import PatientRepository
from emr_analyzer.database.timeline_repo import TimelineRepository
from emr_analyzer.gui.irae_queue_dialog import IraeQueueDialog
from emr_analyzer.gui.irae_queue_result_dialog import IraeQueueResultDialog
from emr_analyzer.gui.workers import IraeQueueWorker
from emr_analyzer.models import Patient
from emr_analyzer.models.clinical_timeline import ClinicalTimelineEntry


def make_entry(entry_id: str, patient_id: str) -> ClinicalTimelineEntry:
    return ClinicalTimelineEntry(
        entry_id=entry_id, patient_id=patient_id,
        date_observed="2023-05-01", category="toxicity",
        description="descrizione", source_document_ids=[],
        source_texts=[], status="active", confidence=0.8,
    )


class PatientsWithEntriesTest(unittest.TestCase):
    def test_lists_only_patients_with_registry(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = DatabaseEngine(Path(tmp) / "registry.db")
            init_database(db)
            patient_repo = PatientRepository(db)
            patient_repo.insert(Patient(id="P001", pseudonym="uno", sex="M"))
            patient_repo.insert(Patient(id="P002", pseudonym="due", sex="F"))
            patient_repo.insert(Patient(id="P003", pseudonym="tre", sex="M"))
            timeline_repo = TimelineRepository(db)
            timeline_repo.save_batch([
                make_entry("T1", "P001"),
                make_entry("T2", "P001"),
                make_entry("T3", "P002"),
            ])

            summaries = timeline_repo.patients_with_entries()

            by_id = {s["id"]: s for s in summaries}
            self.assertEqual(set(by_id), {"P001", "P002"})
            self.assertEqual(by_id["P001"]["timeline_count"], 2)
            self.assertEqual(by_id["P002"]["timeline_count"], 1)
            self.assertEqual(by_id["P001"]["pseudonym"], "uno")


class FakeLlm:
    def __init__(self, responses=None, fail_patient=None):
        self.responses = list(responses or [])
        self.fail_patient = fail_patient
        self.calls = []

    def generate_text(self, prompt, system=""):
        self.calls.append(prompt)
        if self.fail_patient and f"paziente {self.fail_patient}" in prompt:
            raise RuntimeError("errore simulato")
        return "| tabella |"


class IraeQueueWorkerTest(unittest.TestCase):
    def _plans(self):
        return [
            ("P001", [f"prompt paziente P001 chunk 1"]),
            ("P002", [f"prompt paziente P002 chunk 1"]),
        ]

    def test_runs_patients_in_order_with_combined_parts(self):
        worker = IraeQueueWorker(FakeLlm(), self._plans())
        started, finished_ok, errors = [], [], []
        worker.patient_started.connect(lambda i, n, p: started.append((i, n, p)))
        worker.patient_finished.connect(
            lambda p, m: finished_ok.append((p, m))
        )
        worker.patient_error.connect(lambda p, e: errors.append((p, e)))
        worker.run()

        self.assertEqual(started, [(1, 2, "P001"), (2, 2, "P002")])
        self.assertEqual([p for p, _ in finished_ok], ["P001", "P002"])
        self.assertEqual(len(errors), 0)
        self.assertIn("### Parte 1/1", finished_ok[0][1])

    def test_per_patient_error_does_not_stop_queue(self):
        worker = IraeQueueWorker(
            FakeLlm(fail_patient="P002"), self._plans()
        )
        finished_ok, errors = [], []
        worker.patient_finished.connect(lambda p, m: finished_ok.append(p))
        worker.patient_error.connect(lambda p, e: errors.append(p))
        worker.run()

        self.assertEqual(finished_ok, ["P001"])
        self.assertEqual(errors, ["P002"])

    def test_multi_chunk_report_is_finally_reconciled(self):
        llm = FakeLlm()
        worker = IraeQueueWorker(llm, [(
            "P001",
            ["paziente P001 [#T1] parte 1", "paziente P001 [#T2] parte 2"],
        )])
        finished = []
        worker.patient_finished.connect(lambda patient, report: finished.append(report))
        worker.run()
        self.assertEqual(len(llm.calls), 4)
        self.assertIn("RICONCILIAZIONE FINALE", llm.calls[-2])

    def test_cancel_stops_between_patients(self):
        class CancellingLlm(FakeLlm):
            def __init__(self, worker):
                super().__init__()
                self.worker = worker
                self.count = 0

            def generate_text(self, prompt, system=""):
                self.count += 1
                if self.count == 1:
                    self.worker.cancel()  # cancella dopo il primo paziente
                return "| tabella |"

        worker = IraeQueueWorker(CancellingLlm(None), self._plans())
        worker.llm_client.worker = worker
        finished_ok = []
        worker.patient_finished.connect(lambda p, m: finished_ok.append(p))
        worker.run()
        self.assertEqual(finished_ok, ["P001"])


class BuildPlansTest(unittest.TestCase):
    def test_build_irae_plans_uses_repos(self):
        class FakeTimelineRepo:
            def get_by_patient(self, pid):
                return [make_entry(f"T-{pid}", pid)]

        class FakeCSRepo:
            def load(self, pid):
                return None

        from emr_analyzer.gui.workspace_tabs import WorkspaceTabs

        plans = WorkspaceTabs._build_irae_plans(
            {"timeline_repo": FakeTimelineRepo(), "cs_repo": FakeCSRepo()},
            ["P001", "P002"],
            "PROTOCOLLO",
        )
        self.assertEqual([pid for pid, _ in plans], ["P001", "P002"])
        self.assertEqual(len(plans[0][1]), 1)
        self.assertIn("[#T-P001]", plans[0][1][0])
        self.assertIn("PROTOCOLLO", plans[0][1][0])


class IraeQueueDialogTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def _summaries(self):
        return [
            {"id": "P001", "pseudonym": "uno", "timeline_count": 12},
            {"id": "P002", "pseudonym": "due", "timeline_count": 3},
        ]

    def test_all_selected_by_default_in_row_order(self):
        dialog = IraeQueueDialog(self._summaries())
        self.assertEqual(
            dialog.selected_patient_ids(), ["P001", "P002"]
        )
        self.assertTrue(dialog._run_btn.isEnabled())
        dialog.deleteLater()

    def test_unchecking_updates_selection(self):
        dialog = IraeQueueDialog(self._summaries())
        item = dialog._table.item(1, 0)
        item.setCheckState(0)  # Unchecked
        self.assertEqual(dialog.selected_patient_ids(), ["P001"])
        dialog.deleteLater()

    def test_toggle_all(self):
        dialog = IraeQueueDialog(self._summaries())
        dialog._toggle_all()
        self.assertEqual(dialog.selected_patient_ids(), [])
        self.assertFalse(dialog._run_btn.isEnabled())
        dialog._toggle_all()
        self.assertEqual(dialog.selected_patient_ids(), ["P001", "P002"])
        dialog.deleteLater()


class IraeQueueResultDialogTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_tabs_and_bulk_save(self):
        with tempfile.TemporaryDirectory() as tmp:
            results = [
                {"patient_id": "P001", "label": "P001",
                 "markdown": "| A | B |\n|---|---|\n| 1 | 2 |", "error": None},
                {"patient_id": "P002", "label": "P002",
                 "markdown": "tabella due", "error": None},
                {"patient_id": "P003", "label": "P003",
                 "markdown": "", "error": "errore simulato"},
            ]
            dialog = IraeQueueResultDialog(results)
            self.assertEqual(dialog._tabs.count(), 3)
            self.assertIn("<table", dialog._tabs.widget(0).toHtml())

            with mock.patch(
                "emr_analyzer.gui.irae_queue_result_dialog."
                "QFileDialog.getExistingDirectory",
                return_value=tmp,
            ), mock.patch.object(
                QMessageBox, "information", return_value=None
            ) as info:
                dialog._save_all()

            self.assertTrue(Path(tmp, "P001_irae.md").exists())
            self.assertTrue(Path(tmp, "P002_irae.md").exists())
            self.assertFalse(Path(tmp, "P003_irae.md").exists())
            self.assertIn("2", info.call_args[0][2])
            dialog.deleteLater()


if __name__ == "__main__":
    unittest.main()

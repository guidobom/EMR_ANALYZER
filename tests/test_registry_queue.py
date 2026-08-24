"""Tests for multi-patient chronological-registry generation."""

from __future__ import annotations

import os
import unittest
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication

from emr_analyzer.gui.registry_queue_dialog import (
    RegistryQueueDialog,
    build_registry_queue_summaries,
)
from emr_analyzer.gui.registry_queue_result_dialog import (
    RegistryQueueResultDialog,
)
from emr_analyzer.gui.workers import ClinicalHistoryWorker, RegistryQueueWorker
from emr_analyzer.models.document import ExtractionStatus


class FakeBuilder:
    def __init__(self, fail_patient: str | None = None):
        self.fail_patient = fail_patient
        self.calls: list[tuple] = []

    def _result(self, patient_id: str) -> dict:
        if patient_id == self.fail_patient:
            raise RuntimeError("errore simulato")
        return {
            "documents_processed": 2,
            "documents_skipped": 1,
            "atomic_evidence_extracted": 5,
            "final_entries": 3,
            "elapsed_seconds": 12.5,
        }

    def build_incremental(self, patient_id, *, progress_callback,
                          generate_narrative, cancel_check=None):
        self.calls.append(("incremental", patient_id, generate_narrative))
        progress_callback(25, "estrazione")
        progress_callback(100, "completato")
        return self._result(patient_id)

    def build_from_documents_parallel(
        self, patient_id, *, num_workers, progress_callback,
        generate_narrative, cancel_check=None,
    ):
        self.calls.append((
            "rebuild", patient_id, num_workers, generate_narrative,
        ))
        progress_callback(100, "completato")
        return self._result(patient_id)

    def extract_atomic_evidence(
        self, patient_id, *, incremental, num_workers, progress_callback,
        cancel_check=None,
    ):
        self.calls.append((
            "atomic", patient_id, incremental, num_workers,
        ))
        progress_callback(100, "evidenze completate")
        return {**self._result(patient_id), "stage": "atomic"}

    def build_structured_events(
        self, patient_id, *, progress_callback, cancel_check=None,
    ):
        self.calls.append(("events", patient_id))
        progress_callback(100, "eventi completati")
        return {**self._result(patient_id), "stage": "events"}

    def prepare_validation(self, patient_id):
        self.calls.append(("validation", patient_id))
        return {"stage": "validation", "validation_pending": 3}


class RegistryQueueWorkerTest(unittest.TestCase):
    def test_runs_incrementally_in_patient_order(self):
        builder = FakeBuilder()
        worker = RegistryQueueWorker(builder, ["P001", "P002"])
        started, progress, completed = [], [], []
        worker.patient_started.connect(
            lambda i, n, p: started.append((i, n, p))
        )
        worker.patient_progress.connect(
            lambda p, percent, msg: progress.append((p, percent, msg))
        )
        worker.patient_finished.connect(
            lambda p, result: completed.append((p, result))
        )

        worker.run()

        self.assertEqual(
            started, [(1, 2, "P001"), (2, 2, "P002")]
        )
        self.assertEqual(
            [call[:2] for call in builder.calls],
            [("incremental", "P001"), ("incremental", "P002")],
        )
        self.assertEqual([item[0] for item in completed], ["P001", "P002"])
        self.assertIn(("P001", 25, "estrazione"), progress)

    def test_patient_failure_does_not_stop_queue(self):
        builder = FakeBuilder(fail_patient="P001")
        worker = RegistryQueueWorker(builder, ["P001", "P002"])
        completed, errors = [], []
        worker.patient_finished.connect(lambda p, result: completed.append(p))
        worker.patient_error.connect(lambda p, error: errors.append((p, error)))

        worker.run()

        self.assertEqual(completed, ["P002"])
        self.assertEqual(errors[0][0], "P001")
        self.assertIn("errore simulato", errors[0][1])

    def test_cancel_is_honoured_between_patients(self):
        builder = FakeBuilder()
        worker = RegistryQueueWorker(builder, ["P001", "P002"])
        worker.patient_finished.connect(
            lambda patient_id, result: worker.cancel()
        )

        worker.run()

        self.assertEqual(
            [call[1] for call in builder.calls], ["P001"]
        )

    def test_full_rebuild_uses_configured_document_workers(self):
        builder = FakeBuilder()
        worker = RegistryQueueWorker(
            builder, ["P001"], num_workers=3, force_rebuild=True,
        )

        worker.run()

        self.assertEqual(builder.calls, [("rebuild", "P001", 3, False)])

    def test_each_explicit_queue_stage_calls_only_its_builder_method(self):
        atomic_builder = FakeBuilder()
        RegistryQueueWorker(
            atomic_builder, ["P001"], num_workers=3, stage="atomic"
        ).run()
        self.assertEqual(
            atomic_builder.calls, [("atomic", "P001", True, 3)]
        )

        event_builder = FakeBuilder()
        RegistryQueueWorker(
            event_builder, ["P001"], stage="events"
        ).run()
        self.assertEqual(event_builder.calls, [("events", "P001")])

        validation_builder = FakeBuilder()
        RegistryQueueWorker(
            validation_builder, ["P001"], stage="validation"
        ).run()
        self.assertEqual(
            validation_builder.calls, [("validation", "P001")]
        )


class ClinicalHistoryWorkerResumeTest(unittest.TestCase):
    def test_empty_final_registry_resumes_completed_atomic_checkpoints(self):
        class Timeline:
            @staticmethod
            def count_by_patient(_patient_id):
                return 0

        class RegistryBuilder:
            @staticmethod
            def has_atomic_checkpoint(_patient_id):
                return True

        class Builder:
            def __init__(self):
                self._timeline_repo = Timeline()
                self._registry_builder = RegistryBuilder()
                self.calls = []

            def build_incremental(self, patient_id, **_kwargs):
                self.calls.append(("incremental", patient_id))
                return {}

            def build_from_documents_parallel(self, patient_id, **_kwargs):
                self.calls.append(("rebuild", patient_id))
                return {}

            def build_from_documents(self, patient_id, **_kwargs):
                self.calls.append(("serial", patient_id))
                return {}

        builder = Builder()
        worker = ClinicalHistoryWorker(builder, "P001", num_workers=3)
        worker.run()
        self.assertEqual(builder.calls, [("incremental", "P001")])


class RegistryQueueSummaryTest(unittest.TestCase):
    def test_reports_readiness_and_existing_registry_counts(self):
        patients = [
            SimpleNamespace(id="P001", pseudonym="uno"),
            SimpleNamespace(id="P002", pseudonym="due"),
            SimpleNamespace(id="P003", pseudonym="senza documenti"),
        ]
        documents = [
            SimpleNamespace(
                patient_id="P001",
                extraction_status=ExtractionStatus.DONE.value,
            ),
            SimpleNamespace(
                patient_id="P001",
                extraction_status=ExtractionStatus.PENDING.value,
            ),
            SimpleNamespace(
                patient_id="P002",
                extraction_status=ExtractionStatus.ERROR.value,
            ),
        ]

        class Repo:
            def __init__(self, rows):
                self.rows = rows

            def list_all(self):
                return self.rows

        class RegistryRepo:
            def count_by_patient(self, patient_id):
                return {"P001": 7, "P002": 0}[patient_id]

        summaries = build_registry_queue_summaries({
            "patient_repo": Repo(patients),
            "document_repo": Repo(documents),
            "registry_repo": RegistryRepo(),
        })

        by_id = {summary["id"]: summary for summary in summaries}
        self.assertEqual(set(by_id), {"P001", "P002"})
        self.assertTrue(by_id["P001"]["eligible"])
        self.assertEqual(by_id["P001"]["normalized_count"], 1)
        self.assertEqual(by_id["P001"]["pending_count"], 1)
        self.assertEqual(by_id["P001"]["registry_count"], 7)
        self.assertFalse(by_id["P002"]["eligible"])


class RegistryQueueDialogTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    @staticmethod
    def _summaries():
        return [
            {
                "id": "P001", "pseudonym": "uno", "document_count": 4,
                "normalized_count": 3, "pending_count": 1,
                "registry_count": 8, "eligible": True,
            },
            {
                "id": "P002", "pseudonym": "due", "document_count": 2,
                "normalized_count": 0, "pending_count": 2,
                "registry_count": 0, "eligible": False,
            },
        ]

    def test_only_eligible_patients_are_selected_by_default(self):
        dialog = RegistryQueueDialog(self._summaries())
        self.assertEqual(dialog.selected_patient_ids(), ["P001"])
        self.assertFalse(dialog.force_rebuild())
        self.assertEqual(dialog.selected_stage(), "atomic")
        self.assertFalse(
            bool(dialog._table.item(1, 0).flags() & Qt.ItemIsEnabled)
        )
        dialog.deleteLater()

    def test_event_and_validation_phases_disable_full_document_rebuild(self):
        dialog = RegistryQueueDialog(self._summaries())
        dialog._mode.setCurrentIndex(1)
        dialog._phase.setCurrentIndex(
            dialog._phase.findData(RegistryQueueDialog.EVENTS)
        )
        self.assertEqual(dialog.selected_stage(), "events")
        self.assertFalse(dialog.force_rebuild())
        self.assertFalse(dialog._mode.isEnabled())

        dialog._phase.setCurrentIndex(
            dialog._phase.findData(RegistryQueueDialog.VALIDATION)
        )
        self.assertEqual(dialog.selected_stage(), "validation")
        self.assertFalse(dialog.force_rebuild())
        dialog.deleteLater()

    def test_can_choose_full_rebuild(self):
        dialog = RegistryQueueDialog(self._summaries())
        dialog._mode.setCurrentIndex(1)
        self.assertTrue(dialog.force_rebuild())
        self.assertIn("Rigenera", dialog._run_btn.text())
        dialog.deleteLater()

    def test_toggle_all_never_selects_ineligible_patient(self):
        dialog = RegistryQueueDialog(self._summaries())
        dialog._toggle_all()
        self.assertEqual(dialog.selected_patient_ids(), [])
        dialog._toggle_all()
        self.assertEqual(dialog.selected_patient_ids(), ["P001"])
        dialog.deleteLater()


class RegistryQueueResultDialogTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_distinguishes_current_completed_and_failed(self):
        dialog = RegistryQueueResultDialog([
            {
                "patient_id": "P001", "error": None,
                "result": {"documents_processed": 0,
                           "documents_skipped": 4},
            },
            {
                "patient_id": "P002", "error": None,
                "result": {"documents_processed": 2,
                           "documents_skipped": 0},
            },
            {
                "patient_id": "P003", "error": "boom", "result": {},
            },
        ])

        self.assertEqual(dialog._table.item(0, 1).text(), "Già aggiornato")
        self.assertEqual(dialog._table.item(1, 1).text(), "Completato")
        self.assertIn("boom", dialog._table.item(2, 1).text())
        dialog.deleteLater()


if __name__ == "__main__":
    unittest.main()

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
from emr_analyzer.database.evidence_repo import EvidenceRepository
from emr_analyzer.database.migrations import init_database
from emr_analyzer.database.patient_repo import PatientRepository
from emr_analyzer.database.timeline_repo import TimelineRepository
from emr_analyzer.gui.irae_queue_dialog import IraeQueueDialog
from emr_analyzer.gui.irae_queue_result_dialog import IraeQueueResultDialog
from emr_analyzer.gui.workers import IraeQueueWorker
from emr_analyzer.models import Patient
from emr_analyzer.models.clinical_evidence import ClinicalEvidence
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


class PatientsWithEvidenceTest(unittest.TestCase):
    def _insert_document(self, db, doc_id: str, patient_id: str) -> None:
        db.execute(
            """INSERT INTO documents
               (id, patient_id, filename, original_path, file_hash,
                document_date, document_type, import_date)
               VALUES (?, ?, 'd.pdf', '/d.pdf', 'hash', '2025-01-10',
                       'referto', '2026-01-01')""",
            (doc_id, patient_id),
        )

    def test_lists_only_patients_with_atomic_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = DatabaseEngine(Path(tmp) / "registry.db")
            init_database(db)
            patient_repo = PatientRepository(db)
            patient_repo.insert(Patient(id="P001", pseudonym="uno", sex="M"))
            patient_repo.insert(Patient(id="P002", pseudonym="due", sex="F"))
            patient_repo.insert(Patient(id="P003", pseudonym="tre", sex="M"))
            for doc_id, pid in (("D1", "P001"), ("D2", "P001"), ("D3", "P002")):
                self._insert_document(db, doc_id, pid)
            db.commit()
            evidence_repo = EvidenceRepository(db)
            evidence_repo.insert_batch([
                ClinicalEvidence(
                    evidence_id="E1", patient_id="P001", document_id="D1",
                    category="diagnosis", normalized_entity="melanoma",
                    source_text="Melanoma", observed_date="2025-01-10",
                ),
                ClinicalEvidence(
                    evidence_id="E2", patient_id="P001", document_id="D2",
                    category="diagnosis", normalized_entity="melanoma",
                    source_text="Melanoma", observed_date="2025-01-12",
                ),
                ClinicalEvidence(
                    evidence_id="E3", patient_id="P002", document_id="D3",
                    category="diagnosis", normalized_entity="melanoma",
                    source_text="Melanoma", observed_date="2025-02-01",
                ),
            ])

            summaries = evidence_repo.patients_with_evidence()

            by_id = {s["id"]: s for s in summaries}
            self.assertEqual(set(by_id), {"P001", "P002"})
            self.assertEqual(by_id["P001"]["evidence_count"], 2)
            self.assertEqual(by_id["P002"]["evidence_count"], 1)
            self.assertEqual(by_id["P001"]["pseudonym"], "uno")


class MergeIraeQueueSummariesTest(unittest.TestCase):
    def test_registry_kept_and_evidence_only_patients_added(self):
        from emr_analyzer.gui.irae_queue_dialog import (
            merge_irae_queue_summaries,
        )

        merged = merge_irae_queue_summaries(
            [
                {"id": "P001", "pseudonym": "uno", "timeline_count": 12},
                {"id": "P002", "pseudonym": "due", "timeline_count": 3},
            ],
            [
                {"id": "P002", "pseudonym": "due", "evidence_count": 40},
                {"id": "P003", "pseudonym": "tre", "evidence_count": 7},
            ],
        )

        by_id = {s["id"]: s for s in merged}
        self.assertEqual(list(by_id), ["P001", "P002", "P003"])
        # Registry patients keep their timeline count untouched.
        self.assertEqual(by_id["P001"]["timeline_count"], 12)
        self.assertEqual(by_id["P002"]["timeline_count"], 3)
        # Evidence-only patients fall back to the evidence count.
        self.assertEqual(by_id["P003"]["timeline_count"], 7)
        self.assertEqual(by_id["P003"]["pseudonym"], "tre")

    def test_empty_lists_return_empty(self):
        from emr_analyzer.gui.irae_queue_dialog import (
            merge_irae_queue_summaries,
        )

        self.assertEqual(merge_irae_queue_summaries([], []), [])


class ChooseIraeMethodTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def _choose(self, click_index: int):
        """Call ``choose_irae_method`` faking a click on button ``index``."""
        def fake_exec(box):
            setattr(box, "_fake_clicked", box.buttons()[click_index])
            return QMessageBox.Accepted

        def fake_clicked(box):
            return box._fake_clicked

        # Le funzioni reali (patch.object new=...) vengono legate al box:
        # un MagicMock come side_effect non riceverebbe l'istanza.
        with mock.patch.object(QMessageBox, "exec_", fake_exec), \
                mock.patch.object(QMessageBox, "clickedButton", fake_clicked):
            from emr_analyzer.gui.workspace_tabs import choose_irae_method
            return choose_irae_method(None)

    def test_confirm_button_returns_structured(self):
        self.assertEqual(self._choose(0), "structured")

    def test_cancel_button_returns_none(self):
        # Il protocollo classico a chunk è temporaneamente disattivato: il
        # dialogo offre solo "Avvia analisi strutturata" / "Annulla".
        self.assertIsNone(self._choose(1))

    def test_dismiss_returns_none(self):
        # Regressione: QMessageBox.question(parent, ..., "a", "b") non esiste
        # su PyQt5 (TypeError) — la scelta usa addButton ed è annullabile.
        def fake_reject(box):
            return QMessageBox.Rejected

        with mock.patch.object(QMessageBox, "exec_", fake_reject), \
                mock.patch.object(QMessageBox, "clickedButton",
                                  lambda box: None):
            from emr_analyzer.gui.workspace_tabs import choose_irae_method
            self.assertIsNone(choose_irae_method(None))


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

    def test_instance_selector_defaults_to_auto(self):
        dialog = IraeQueueDialog(self._summaries(), max_instances=3)
        self.assertEqual(dialog.selected_instances(), 0)  # Auto (memoria)
        self.assertTrue(dialog._instance_combo.isEnabled())
        self.assertEqual(dialog._instance_combo.count(), 4)  # Auto + 1..3
        dialog.deleteLater()

    def test_instance_selector_forced_single_when_no_headroom(self):
        dialog = IraeQueueDialog(self._summaries(), max_instances=1)
        self.assertEqual(dialog.selected_instances(), 0)
        self.assertFalse(dialog._instance_combo.isEnabled())
        self.assertEqual(dialog._instance_combo.count(), 1)
        dialog.deleteLater()

    def test_instance_selector_caps_at_six(self):
        dialog = IraeQueueDialog(self._summaries(), max_instances=12)
        self.assertEqual(dialog._instance_combo.count(), 7)  # Auto + 1..6
        self.assertEqual(dialog.selected_instances(), 0)
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

    def _structured_report(self) -> dict:
        """A consolidated structured report with one definitive irAE."""
        finding = {
            "organ": "Miocardite/Cardiotossicità",
            "irAE_type": "Miocardite da ICI",
            "ctcae_grade": "G2",
            "first_onset_date": "2022-11-22",
            "probability_immune": "PROBABILE",
            "new_onset_vs_exacerbation": "nuova insorgenza",
            "alternative_causes": "nessuna",
            "confidence": 0.9,
            "key_evidence_ids": ["E-TROP"],
        }
        return {
            "anchor": {
                "first_drug": "nivolumab", "first_date": "2022-09-01",
                "first_raw": "2022-09-01", "last_drug": "nivolumab",
                "last_date": "2022-09-01", "last_raw": "2022-09-01",
                "occurrences": 1,
            },
            "candidates_total": 1,
            "iraes": [finding],
            "consolidation": {
                "applied": True, "error": None,
                "input_count": 1, "output_count": 1,
                "iraes": [finding],
                "suspects": [],
            },
            "candidates": [{
                "organ": "Miocardite/Cardiotossicità",
                "evidence_id": "E-TROP",
            }],
            "evidence": [{
                "evidence_id": "E-TROP",
                "document_id": "DOC-1",
                "source_page": 1,
                "bbox": [1, 2, 3, 4],
                "source_text": "troponina 1117 ng/L",
                "normalized_entity": "troponina_i_hs",
                "category": "laboratory_finding",
                "observed_date": "2022-11-22",
                "value_text": "1117",
            }],
        }

    def test_structured_result_uses_inspectable_tab(self):
        from emr_analyzer.config import active_workspace
        from emr_analyzer.gui.irae_patient_tab import IraePatientTab

        with tempfile.TemporaryDirectory() as tmp:
            results = [{
                "patient_id": "P001", "label": "P001",
                "markdown": "", "error": None,
                "structured": self._structured_report(),
            }]
            with mock.patch.object(active_workspace, "path", Path(tmp)):
                dialog = IraeQueueResultDialog(results, services={})

        self.assertIsInstance(dialog._tabs.widget(0), IraePatientTab)
        # the result entry is refreshed to the CORRECTED report, so the bulk
        # save / Excel export read the annotated (and possibly corrected) data.
        self.assertEqual(
            results[0]["structured"]["iraes"][0]["finding_id"], "irAE-1"
        )
        dialog.deleteLater()

    def test_legacy_result_without_structured_keeps_text_tab(self):
        from PyQt5.QtWidgets import QTextBrowser

        results = [{
            "patient_id": "P001", "label": "P001",
            "markdown": "| A | B |\n|---|---|\n| 1 | 2 |", "error": None,
        }]
        # services provided, but no ``structured`` key: the read-only legacy
        # tab is kept (backward compatibility with the classic queue).
        dialog = IraeQueueResultDialog(results, services={})
        self.assertIsInstance(dialog._tabs.widget(0), QTextBrowser)
        self.assertIn("<table", dialog._tabs.widget(0).toHtml())
        dialog.deleteLater()


if __name__ == "__main__":
    unittest.main()

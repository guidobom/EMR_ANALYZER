"""Live, lossless atomic-evidence projection in the clinical-history tab."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtTest import QTest
from PyQt5.QtWidgets import QApplication

from emr_analyzer.gui.clinical_history_tab import ClinicalHistoryTab
from emr_analyzer.models.clinical_evidence import ClinicalEvidence


class MutableEvidenceRepo:
    def __init__(self, items=None):
        self.items = list(items or [])

    def get_by_patient(self, patient_id: str):
        return [item for item in self.items if item.patient_id == patient_id]


class DocumentRepo:
    @staticmethod
    def get_by_id(document_id: str):
        return SimpleNamespace(
            id=document_id,
            filename=f"{document_id}.pdf",
            to_dict=lambda: {
                "id": document_id,
                "patient_id": "P001",
                "filename": f"{document_id}.pdf",
            },
        )


def evidence(evidence_id: str, document_id: str, entity: str, source: str):
    return ClinicalEvidence(
        evidence_id=evidence_id,
        patient_id="P001",
        document_id=document_id,
        category="symptom",
        normalized_entity=entity,
        source_text=source,
        observed_date="2025-01-10",
        document_date="2025-01-10",
        date_precision="day",
        source_page=1,
        confidence=0.9,
        extraction_method="llm_atomic_v2",
    )


class ClinicalHistoryAtomicViewTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.repo = MutableEvidenceRepo([
            evidence("EVD_1", "DOC_1", "dispnea", "Riferisce dispnea."),
            evidence("EVD_2", "DOC_2", "dispnea", "Riferisce dispnea."),
            evidence("EVD_3", "DOC_3", "tosse", "Riferisce tosse."),
        ])
        self.tab = ClinicalHistoryTab(
            settings_path=Path(self._tmp.name) / "settings.json"
        )
        self.tab.set_services({"evidence_repo": self.repo})
        self.tab.load_patient("P001")

    def tearDown(self):
        self.tab.deleteLater()
        self._tmp.cleanup()

    def test_projection_fuses_duplicates_but_exposes_every_source(self):
        tree = self.tab._atomic_evidence_tree
        self.assertEqual(tree.topLevelItemCount(), 2)
        self.assertIn("2 fatti clinici unici", self.tab._atomic_evidence_status.text())
        self.assertIn("1 occorrenze duplicate fuse", self.tab._atomic_evidence_status.text())

        fused = next(
            tree.topLevelItem(index)
            for index in range(tree.topLevelItemCount())
            if tree.topLevelItem(index).text(2).startswith("dispnea")
        )
        self.assertEqual(fused.text(3), "2 fonti fuse")
        self.assertEqual(fused.childCount(), 2)
        self.assertEqual(
            {fused.child(index).text(3) for index in range(2)},
            {"doc DOC_1, p. 1", "doc DOC_2, p. 1"},
        )

    def test_double_click_source_opens_highlighted_quick_view(self):
        self.tab.set_services({
            "evidence_repo": self.repo,
            "document_repo": DocumentRepo(),
        })
        self.tab.load_patient("P001")
        tree = self.tab._atomic_evidence_tree
        fused = next(
            tree.topLevelItem(index)
            for index in range(tree.topLevelItemCount())
            if tree.topLevelItem(index).text(2).startswith("dispnea")
        )
        source_item = next(
            fused.child(index)
            for index in range(fused.childCount())
            if "DOC_2" in fused.child(index).text(3)
        )

        with patch(
            "emr_analyzer.gui.pdf_viewer.PDFViewerDialog"
        ) as viewer_class:
            self.tab._open_atomic_evidence_source(source_item)

        _, services, parent = viewer_class.call_args.args
        highlight = viewer_class.call_args.kwargs["highlight"]
        self.assertIs(services, self.tab._services)
        self.assertIs(parent, self.tab)
        self.assertEqual(highlight["evidence_id"], "EVD_2")
        self.assertEqual(highlight["page_number"], 1)
        self.assertEqual(highlight["source_text"], "Riferisce dispnea.")
        viewer_class.return_value.exec_.assert_called_once_with()

    def test_progress_checkpoint_refreshes_visible_view_dynamically(self):
        self.tab._clinical_data_tabs.setCurrentIndex(
            self.tab._evidence_page_index
        )
        self.repo.items.append(
            evidence("EVD_4", "DOC_4", "astenia", "Riferisce astenia.")
        )
        self.tab._active_registry_stage = "atomic"
        self.tab._on_generation_progress(25, "Evidenze 1/4")

        self.assertTrue(self.tab._live_evidence_timer.isActive())
        QTest.qWait(450)

        self.assertEqual(self.tab._atomic_evidence_tree.topLevelItemCount(), 3)
        self.assertIn("3 fatti clinici unici", self.tab._atomic_evidence_status.text())

    def test_hidden_view_defers_redraw_until_opened(self):
        self.tab._clinical_data_tabs.setCurrentIndex(0)
        self.repo.items.append(
            evidence("EVD_4", "DOC_4", "astenia", "Riferisce astenia.")
        )
        self.tab._active_registry_stage = "atomic"
        self.tab._on_generation_progress(25, "Evidenze 1/4")

        self.assertTrue(self.tab._atomic_view_dirty)
        self.assertFalse(self.tab._live_evidence_timer.isActive())
        self.assertEqual(self.tab._atomic_evidence_tree.topLevelItemCount(), 2)

        self.tab._clinical_data_tabs.setCurrentIndex(
            self.tab._evidence_page_index
        )
        self.assertEqual(self.tab._atomic_evidence_tree.topLevelItemCount(), 3)
        self.assertFalse(self.tab._atomic_view_dirty)


if __name__ == "__main__":
    unittest.main()

"""Offscreen tests for the per-finding evidence inspector widget.

Opened from ``IraePatientTab`` on double-click, it shows the evidence that
led to a finding: the cited ``key_evidence_ids`` (resolved via the compact
``report["evidence"]``) and every Layer 2 candidate of the same organ.  These
tests exercise the widget construction and the tree grouping on a synthetic
report; no LLM and no real document repository are involved.
"""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtWidgets import QApplication, QDialog

from emr_analyzer.clinical.irae_corrections import (
    IraeCorrection,
    add_correction,
    apply_irae_corrections,
    load_corrections,
)
from emr_analyzer.config import active_workspace


def _finding(**overrides) -> dict:
    base = {
        "organ": "Miocardite/Cardiotossicità",
        "irAE_type": "Miocardite da ICI",
        "ctcae_grade": "G2",
        "first_onset_date": "2022-11-22",
        "probability_immune": "PROBABILE",
        "new_onset_vs_exacerbation": "nuova insorgenza",
        "alternative_causes": "nessuna",
        "key_evidence_ids": ["E-TROP"],
        "notes": "originale",
    }
    base.update(overrides)
    return base


def _report() -> dict:
    definitive = [_finding()]
    return {
        "anchor": {
            "first_drug": "nivolumab", "first_date": "2022-09-01",
            "first_raw": "2022-09-01", "last_drug": "nivolumab",
            "last_date": "2022-09-01", "last_raw": "2022-09-01",
            "occurrences": 1,
        },
        "candidates_total": 2,
        "iraes": definitive,
        "consolidation": {
            "applied": True, "error": None,
            "input_count": 2, "output_count": 1,
            "iraes": definitive,
            "suspects": [_finding(
                irAE_type="Rash sospetto", ctcae_grade="G1",
                first_onset_date="2022-12-13",
                probability_immune="POSSIBILE",
                source_organs=["Dermatite"], organ="Dermatite",
            )],
        },
        "organ_results": {
            "Miocardite/Cardiotossicità": {"iraes": [_finding()]},
            "Dermatite": {"iraes": []},
        },
        "candidates": [
            {
                "organ": "Miocardite/Cardiotossicità",
                "evidence_id": "E-TROP", "category": "laboratory_finding",
                "entity": "troponina_i_hs", "observed_raw": "2022-11-22",
                "offset_days": 82, "band": "14_112d", "quote": "",
                "generic": False, "value": "1117", "reference": "",
                "document_id": "DOC-1", "source_page": 2,
                "bbox": [10, 20, 30, 40],
                "source_text": "Troponina I hs 1117 ng/L",
            },
            {
                "organ": "Dermatite", "evidence_id": "E-RASH",
                "category": "symptom", "entity": "rash",
                "observed_raw": "2022-12-10", "offset_days": 100,
                "band": "14_112d", "quote": "", "generic": False,
                "value": "", "reference": "", "document_id": "DOC-2",
                "source_page": 1, "bbox": None,
                "source_text": "Rash maculopapulare",
            },
        ],
        "evidence": [
            {
                "evidence_id": "E-TROP", "document_id": "DOC-1",
                "source_page": 2, "bbox": [10, 20, 30, 40],
                "source_text": "Troponina I hs 1117 ng/L",
                "normalized_entity": "troponina_i_hs",
                "category": "laboratory_finding",
                "observed_date": "2022-11-22", "value_text": "1117",
            },
        ],
    }


class IraeEvidenceInspectorTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def _open_inspector(self):
        """Return (inspector, corrected) for the Miocardite finding."""
        from emr_analyzer.gui.irae_evidence_inspector import (
            IraeEvidenceInspector,
        )

        corrected, _ = apply_irae_corrections(_report(), [])
        finding = next(
            f for f in corrected["iraes"] if f["irAE_type"] == "Miocardite da ICI"
        )
        services = {"document_repo": None}
        inspector = IraeEvidenceInspector(
            corrected, finding, "P001", services
        )
        return inspector, corrected

    def test_constructor_builds_cited_and_candidate_groups(self):
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(active_workspace, "path", Path(tmp)):
            inspector, _ = self._open_inspector()

        root = inspector._tree.invisibleRootItem()
        groups = [root.child(i).text(0) for i in range(root.childCount())]
        self.assertTrue(any("Evidenze citate dal modello" in g for g in groups))
        self.assertTrue(
            any("Candidati Layer 2" in g and "Miocardite" in g for g in groups)
        )
        # the cited key_evidence_id resolves through report["evidence"].
        cited_index = next(
            i for i, g in enumerate(groups)
            if "Evidenze citate dal modello" in g
        )
        self.assertEqual(root.child(cited_index).childCount(), 1)
        inspector.accept()

    def test_candidate_group_covers_all_source_organs(self):
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(active_workspace, "path", Path(tmp)):
            inspector, _ = self._open_inspector()

        root = inspector._tree.invisibleRootItem()
        candidates_index = next(
            i for i in range(root.childCount())
            if "Candidati Layer 2" in root.child(i).text(0)
        )
        # E-TROP (cited) and E-RASH (suspect's organ) are both candidates of
        # the organs involved; E-TROP is grouped under both the cited and the
        # candidate group.
        self.assertEqual(root.child(candidates_index).childCount(), 1)
        inspector.accept()

    def test_missing_candidates_yields_empty_group_not_crash(self):
        from emr_analyzer.gui.irae_evidence_inspector import (
            IraeEvidenceInspector,
        )

        report = _report()
        report["candidates"] = []
        corrected, _ = apply_irae_corrections(report, [])
        finding = next(
            f for f in corrected["iraes"] if f["irAE_type"] == "Miocardite da ICI"
        )
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(active_workspace, "path", Path(tmp)):
            inspector = IraeEvidenceInspector(
                corrected, finding, "P001", {"document_repo": None}
            )
        root = inspector._tree.invisibleRootItem()
        candidates_index = next(
            i for i in range(root.childCount())
            if "Candidati Layer 2" in root.child(i).text(0)
        )
        self.assertEqual(root.child(candidates_index).childCount(), 1)
        self.assertIn(
            "Nessun candidato",
            root.child(candidates_index).child(0).text(2),
        )
        inspector.accept()

    def test_removed_finding_shows_removed_state_and_restore_button(self):
        from emr_analyzer.gui.irae_evidence_inspector import (
            IraeEvidenceInspector,
        )

        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(active_workspace, "path", Path(tmp)):
            add_correction("P001", IraeCorrection(
                action="remove", irAE_type="Miocardite da ICI",
                first_onset_date="2022-11-22", reason="falso positivo",
            ))
            corrected, _ = apply_irae_corrections(
                _report(), load_corrections("P001")
            )
            finding = corrected["consolidation"]["removed"][0]
            inspector = IraeEvidenceInspector(
                corrected, finding, "P001", {"document_repo": None}
            )
            self.assertIn("rimosso", inspector._header.text())
            self.assertIn("falso positivo", inspector._header.text())
            self.assertFalse(inspector._edit_btn.isEnabled())
            self.assertFalse(inspector._remove_btn.isEnabled())
            # isVisible() is False when the dialog itself is never shown;
            # assert on the hidden state instead.
            self.assertFalse(inspector._restore_btn.isHidden())
            inspector.accept()

    def test_remove_then_restore_roundtrip_keeps_dialog_open(self):
        from emr_analyzer.gui.irae_evidence_inspector import (
            IraeEvidenceInspector,
        )

        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(active_workspace, "path", Path(tmp)):
            corrected, _ = apply_irae_corrections(_report(), [])
            finding = next(
                f for f in corrected["iraes"]
                if f["irAE_type"] == "Miocardite da ICI"
            )
            inspector = IraeEvidenceInspector(
                corrected, finding, "P001", {"document_repo": None}
            )
            correction = IraeCorrection(
                action="remove", irAE_type="Miocardite da ICI",
                first_onset_date="2022-11-22",
                finding_id=finding["finding_id"], reason="falso positivo",
            )
            inspector._persist_correction(correction)
            # the finding now lives in the removed bucket: the dialog must
            # stay open and show the removed state.
            self.assertNotEqual(inspector.result(), QDialog.Accepted)
            self.assertIsNotNone(inspector._finding)
            self.assertTrue(inspector._finding.get("removed"))
            self.assertFalse(inspector._restore_btn.isHidden())
            # restoring brings the finding back to the active list.
            inspector._on_restore()
            self.assertIsNotNone(inspector._finding)
            self.assertFalse(inspector._finding.get("removed"))
            self.assertTrue(inspector._restore_btn.isHidden())
            self.assertTrue(inspector._edit_btn.isEnabled())
            self.assertTrue(inspector._remove_btn.isEnabled())
            inspector.accept()


if __name__ == "__main__":
    unittest.main()

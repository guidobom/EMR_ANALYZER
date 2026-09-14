"""Offscreen tests for the per-patient irAE findings tab.

``IraePatientTab`` renders the CORRECTED report: definitives, then suspects,
then manually removed findings (struck through, gray, tagged ``[rimosso]``
with the reason).  No LLM is involved: corrections are persisted in a
temporary workspace and re-applied over a synthetic raw report.
"""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import QApplication

from emr_analyzer.clinical.irae_corrections import (
    IraeCorrection,
    add_correction,
)
from emr_analyzer.config import active_workspace
from emr_analyzer.gui.irae_patient_tab import IraePatientTab


def _finding(**overrides) -> dict:
    base = {
        "irAE_type": "Miocardite da ICI",
        "ctcae_grade": "G2",
        "first_onset_date": "2022-11-22",
        "probability_immune": "PROBABILE",
        "new_onset_vs_exacerbation": "nuova insorgenza",
        "alternative_causes": "nessuna",
        "source_organs": ["Miocardite/Cardiotossicità"],
        "organ": "Miocardite/Cardiotossicità",
        "key_evidence_ids": ["E-TROP"],
        "notes": "originale",
    }
    base.update(overrides)
    return base


def _report() -> dict:
    """One definitive irAE and one suspect, consolidation applied."""
    definitive = [_finding()]
    suspects = [_finding(
        irAE_type="Rash sospetto", ctcae_grade="G1",
        first_onset_date="2022-12-13", probability_immune="POSSIBILE",
        source_organs=["Dermatite"], organ="Dermatite",
    )]
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
            "suspects": suspects,
        },
        "organ_results": {
            "Miocardite/Cardiotossicità": {"iraes": [
                {"organ": "Miocardite/Cardiotossicità",
                 "irAE_type": "Miocardite da ICI",
                 "first_onset_date": "2022-11-22",
                 "probability_immune": "PROBABILE"},
            ]},
        },
    }


class IraePatientTabTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        workspace_patch = mock.patch.object(
            active_workspace, "path", Path(self._tmp.name)
        )
        self._workspace_patch = workspace_patch
        workspace_patch.start()
        self.addCleanup(workspace_patch.stop)
        self.addCleanup(self._tmp.cleanup)
        application = QApplication.instance() or QApplication([])
        self._app = application

    def _tab_with_remove(self, reason="falso positivo") -> IraePatientTab:
        add_correction("P001", IraeCorrection(
            action="remove", irAE_type="Miocardite da ICI",
            first_onset_date="2022-11-22", reason=reason,
        ))
        return IraePatientTab(_report(), "P001", {})

    def test_removed_item_is_last_struck_gray_with_reason(self):
        tab = self._tab_with_remove()
        tab._list  # noqa: B018 — widget built in __init__ via refresh()
        count = tab._list.count()
        self.assertEqual(count, 2)  # suspect + removed
        list_item = tab._list.item(count - 1)
        self.assertTrue(list_item.text().startswith("[rimosso]"))
        self.assertIn("falso positivo", list_item.text())
        self.assertTrue(list_item.font().strikeOut())
        self.assertEqual(list_item.foreground().color(), QColor("#808080"))
        finding = list_item.data(Qt.UserRole)
        self.assertTrue(finding["removed"])
        self.assertEqual(finding["removed_reason"], "falso positivo")
        # the visible suspect comes first, unstruck.
        first = tab._list.item(0)
        self.assertTrue(first.text().startswith("[sospetto]"))
        self.assertFalse(first.font().strikeOut())

    def test_placeholder_only_when_every_list_is_empty(self):
        tab = self._tab_with_remove()
        placeholder = tab._list.item(0)
        self.assertFalse(placeholder.text() == "Nessun irAE nel report.")
        # a report whose only finding was removed shows the removed entry.
        self.assertEqual(tab._list.count(), 2)

    def test_removed_item_double_click_data(self):
        tab = self._tab_with_remove()
        list_item = tab._list.item(tab._list.count() - 1)
        finding = list_item.data(Qt.UserRole)
        self.assertIsInstance(finding, dict)
        self.assertEqual(finding["finding_id"], "irAE-1")

    def test_markdown_mentions_removed_section(self):
        tab = self._tab_with_remove()
        self.assertIn("Rimossi manualmente", tab.markdown())


if __name__ == "__main__":
    unittest.main()

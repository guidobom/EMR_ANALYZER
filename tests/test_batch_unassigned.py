"""Tests for the "Da assegnare" bucket in the multi-patient import dialog."""

from __future__ import annotations

import os
import shutil
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtCore import QEvent, Qt
from PyQt5.QtGui import QKeyEvent
from PyQt5.QtWidgets import QApplication

import emr_analyzer.gui.batch_import_dialog as batch_module
from emr_analyzer.config import active_workspace
from emr_analyzer.database.document_repo import DocumentRepository
from emr_analyzer.database.engine import DatabaseEngine
from emr_analyzer.database.migrations import init_database
from emr_analyzer.database.patient_repo import PatientRepository
from emr_analyzer.gui.batch_import_dialog import BatchImportDialog
from emr_analyzer.models import Patient


class _FakeMessageBox:
    @staticmethod
    def critical(*args, **kwargs):
        pass

    @staticmethod
    def information(*args, **kwargs):
        pass


class _FakeViewer:
    """Records the doc_data it is built with; never opens a real window."""

    def __init__(self, doc_data, services, parent=None):
        self.doc_data = doc_data
        self.parent = parent

    def exec_(self):
        self.opened = True


class BatchUnassignedTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self._tmp = tempfile.mkdtemp(prefix="emr_unassigned_")
        self._root = Path(self._tmp)
        self._orig_workspace_path = active_workspace.path
        active_workspace.set_path(self._root / "workspaces")

        db = DatabaseEngine(self._root / "workspaces" / "registry.db")
        init_database(db)
        self._doc_repo = DocumentRepository(db)
        self._patient_repo = PatientRepository(db)
        self._patient_repo.insert(Patient(id="P001", pseudonym="001"))
        self._services = {
            "document_repo": self._doc_repo,
            "patient_repo": self._patient_repo,
        }
        self._original_qmsgbox = batch_module.QMessageBox
        batch_module.QMessageBox = _FakeMessageBox

    def tearDown(self):
        batch_module.QMessageBox = self._original_qmsgbox
        active_workspace.set_path(self._orig_workspace_path)
        shutil.rmtree(self._tmp, ignore_errors=True)

    def _make_png(self, name="ref.png") -> Path:
        path = self._root / name
        path.write_bytes(b"png-content")
        return path

    def _dialog(self, unassigned, candidates):
        return BatchImportDialog(
            {}, self._services,
            unassigned_files=[str(p) for p in unassigned],
            candidate_pids=candidates,
        )

    def _unassigned_parent(self, dialog):
        for i in range(dialog._tree.topLevelItemCount()):
            item = dialog._tree.topLevelItem(i)
            if item.data(0, Qt.UserRole) == dialog._UNASSIGNED:
                return item
        return None

    def test_bucket_default_target_does_not_import(self):
        png = self._make_png()
        dialog = self._dialog([png], ["P001"])
        parent = self._unassigned_parent(dialog)
        self.assertIsNotNone(parent)
        self.assertEqual(parent.childCount(), 1)
        child = parent.child(0)
        child.setCheckState(0, Qt.Checked)
        # Target defaults to "(non importare)": checked files stay unassigned.
        self.assertIsNone(dialog._unassigned_target_combos[0].currentData())

        dialog._on_import(run_queue=False)

        self.assertEqual(dialog.imported_by_patient, {})

    def test_bucket_imports_checked_file_to_chosen_patient(self):
        png = self._make_png()
        dialog = self._dialog([png], ["P001"])
        parent = self._unassigned_parent(dialog)
        child = parent.child(0)
        child.setCheckState(0, Qt.Checked)
        combo = dialog._unassigned_target_combos[0]
        combo.setCurrentIndex(combo.findData("P001"))

        dialog._on_import(run_queue=False)

        imported = dialog.imported_by_patient.get("P001", [])
        self.assertEqual(len(imported), 1)
        self.assertIsNotNone(self._doc_repo.get_by_id(imported[0]))
        workspace_copy = (
            active_workspace.path / "P001" / "documents" / "original" / "ref.png"
        )
        self.assertTrue(workspace_copy.exists())
        # A patient that received files is queued for extraction.
        self.assertEqual(dialog.queue_patient_ids, ["P001"])

    def test_bucket_without_candidates_offers_only_skip(self):
        png = self._make_png()
        dialog = self._dialog([png], [])
        parent = self._unassigned_parent(dialog)
        combo = dialog._unassigned_target_combos[0]
        self.assertEqual(combo.count(), 1)
        self.assertIsNone(combo.currentData())

    # --- PDF quick-view (double-click + spacebar) --------------------------

    def _dialog_with_patient(self, name="a.pdf"):
        path = self._root / name
        path.write_bytes(b"pdf-content")
        return BatchImportDialog(
            {"P001": [str(path)]}, self._services,
        )

    @staticmethod
    def _file_child(dialog):
        parent = dialog._tree.topLevelItem(0)
        return parent.child(0)

    def test_double_click_opens_viewer_for_file_row(self):
        dialog = self._dialog_with_patient()
        child = self._file_child(dialog)
        seen = []
        class _Capture(_FakeViewer):
            def __init__(self, doc_data, services, parent=None):
                super().__init__(doc_data, services, parent)
                seen.append(doc_data)
        original = batch_module.PDFViewerDialog
        batch_module.PDFViewerDialog = _Capture
        try:
            dialog._on_item_double_clicked(child, 1)
        finally:
            batch_module.PDFViewerDialog = original
        self.assertEqual(len(seen), 1)
        self.assertEqual(seen[0]["original_path"], str(self._root / "a.pdf"))
        self.assertEqual(seen[0]["filename"], "a.pdf")

    def test_double_click_on_checkbox_column_does_not_open(self):
        dialog = self._dialog_with_patient()
        child = self._file_child(dialog)
        original = batch_module.PDFViewerDialog
        seen = []
        class _Capture(_FakeViewer):
            def __init__(self, doc_data, services, parent=None):
                super().__init__(doc_data, services, parent)
                seen.append(doc_data)
        batch_module.PDFViewerDialog = _Capture
        try:
            dialog._on_item_double_clicked(child, 0)
        finally:
            batch_module.PDFViewerDialog = original
        self.assertEqual(seen, [])

    def test_spacebar_quick_view_opens_current_file(self):
        dialog = self._dialog_with_patient()
        child = self._file_child(dialog)
        dialog._tree.setCurrentItem(child)
        original = batch_module.PDFViewerDialog
        seen = []
        class _Capture(_FakeViewer):
            def __init__(self, doc_data, services, parent=None):
                super().__init__(doc_data, services, parent)
                seen.append(doc_data)
        batch_module.PDFViewerDialog = _Capture
        try:
            event = QKeyEvent(QEvent.KeyPress, Qt.Key_Space, Qt.NoModifier)
            consumed = dialog.eventFilter(dialog._tree, event)
        finally:
            batch_module.PDFViewerDialog = original
        self.assertTrue(consumed)
        self.assertEqual(len(seen), 1)

    def test_spacebar_on_patient_row_keeps_checkbox_toggle(self):
        dialog = self._dialog_with_patient()
        dialog._tree.setCurrentItem(dialog._tree.topLevelItem(0))
        original = batch_module.PDFViewerDialog
        seen = []
        class _Capture(_FakeViewer):
            def __init__(self, doc_data, services, parent=None):
                super().__init__(doc_data, services, parent)
                seen.append(doc_data)
        batch_module.PDFViewerDialog = _Capture
        try:
            event = QKeyEvent(QEvent.KeyPress, Qt.Key_Space, Qt.NoModifier)
            consumed = dialog.eventFilter(dialog._tree, event)
        finally:
            batch_module.PDFViewerDialog = original
        self.assertFalse(consumed)
        self.assertEqual(seen, [])


if __name__ == "__main__":
    unittest.main()

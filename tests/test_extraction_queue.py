"""Tests for the sequential multi-patient extraction queue."""

from __future__ import annotations

import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtWidgets import QApplication

import emr_analyzer.gui.workspace_tabs as workspace_tabs
from emr_analyzer.gui.workspace_tabs import WorkspaceTabs


class ExtractionQueueTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_queue_runs_patients_in_order(self):
        tabs = WorkspaceTabs()
        loaded = []
        extracted_labels = []
        tabs.load_patient = lambda pid: loaded.append(pid)
        tabs._documents_tab.extract_clinical_text = (
            lambda **kw: extracted_labels.append(kw.get("patient_label"))
        )

        class _FakeProgressDialog:
            def __init__(self, title="", parent=None):
                self.title = title
                self.done_count = 0

            def show(self):
                pass

            def is_cancelled(self):
                return False

            def reset_for_reuse(self):
                pass

            def setWindowTitle(self, title):
                pass

            def add_log(self, message):
                pass

            def mark_done(self):
                self.done_count += 1

            def exec_(self):
                pass

        original = workspace_tabs.ProgressDialog
        workspace_tabs.ProgressDialog = _FakeProgressDialog
        try:
            tabs.run_extraction_queue(["P005", "P006"])
        finally:
            workspace_tabs.ProgressDialog = original

        self.assertEqual(loaded, ["P005", "P006"])
        self.assertEqual(extracted_labels, [
            "Paziente 1/2: P005",
            "Paziente 2/2: P006",
        ])

    def test_queue_stops_between_patients_on_cancel(self):
        tabs = WorkspaceTabs()
        loaded = []
        tabs.load_patient = lambda pid: loaded.append(pid)
        tabs._documents_tab.extract_clinical_text = lambda **kw: None

        class _FakeProgressDialog:
            def __init__(self, title="", parent=None):
                self.cancel_checks = 0
                self.done_count = 0

            def show(self):
                pass

            def is_cancelled(self):
                self.cancel_checks += 1
                return self.cancel_checks >= 2  # cancel before the 2nd patient

            def reset_for_reuse(self):
                pass

            def setWindowTitle(self, title):
                pass

            def add_log(self, message):
                pass

            def mark_done(self):
                self.done_count += 1

            def exec_(self):
                pass

        original = workspace_tabs.ProgressDialog
        workspace_tabs.ProgressDialog = _FakeProgressDialog
        try:
            tabs.run_extraction_queue(["P005", "P006", "P007"])
        finally:
            workspace_tabs.ProgressDialog = original

        # Only the first patient is processed before the cancel is seen.
        self.assertEqual(loaded, ["P005"])

    def test_empty_queue_is_noop(self):
        tabs = WorkspaceTabs()
        tabs.load_patient = lambda pid: self.fail("load_patient must not be called")
        # No ProgressDialog should be created, so nothing is patched.
        tabs.run_extraction_queue([])


if __name__ == "__main__":
    unittest.main()

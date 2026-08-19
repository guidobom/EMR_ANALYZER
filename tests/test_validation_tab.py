"""Offscreen tests for the attribution flows of the Validazione tab."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtWidgets import QApplication, QMessageBox

from emr_analyzer.clinical.document_reattribution import ReattributionResult
from emr_analyzer.database.audit_repo import AuditRepository
from emr_analyzer.database.engine import DatabaseEngine
from emr_analyzer.database.migrations import init_database
from emr_analyzer.database.patient_repo import PatientRepository
from emr_analyzer.gui.validation_tab import ValidationTab
from emr_analyzer.models import Patient
from emr_analyzer.models.document import DocumentRecord


class FakeDocRepo:
    """Serves a single DocumentRecord by id."""

    def __init__(self, doc):
        self.doc = doc

    def get_by_id(self, doc_id: str):
        return self.doc if doc_id == self.doc.id else None


class FakeLabRepo:
    def __init__(self):
        self.validated: list[int] = []

    def update_validation(self, lab_id: int, valid: bool) -> None:
        self.validated.append(lab_id)


class FakeReattribution:
    """Records calls and resolves the queue row like the real service."""

    def __init__(self, db, ok=True):
        self.db = db
        self.calls: list[tuple] = []
        self.confirms: list[tuple] = []
        self.ok = ok

    def move_document(self, doc_id, target, queue_item_id=None,
                      resolution_status="accepted"):
        self.calls.append((doc_id, target, queue_item_id, resolution_status))
        if self.ok and queue_item_id is not None:
            self.db.execute(
                "UPDATE validation_queue SET status=?, patient_id=?, "
                "resolved_at='2026-08-19T11:00:00' WHERE id=?",
                (resolution_status, target, queue_item_id),
            )
            self.db.commit()
        return ReattributionResult(
            document_id=doc_id,
            source_patient_id="P001",
            target_patient_id=target,
            queue_resolved=self.ok and queue_item_id is not None,
            error=None if self.ok else "errore simulato",
        )

    def confirm_attribution(self, doc_id, queue_item_id=None,
                            resolution_status="accepted"):
        self.confirms.append((doc_id, queue_item_id, resolution_status))
        if self.ok and queue_item_id is not None:
            self.db.execute(
                "UPDATE validation_queue SET status=?, "
                "resolved_at='2026-08-19T11:00:00' WHERE id=?",
                (resolution_status, queue_item_id),
            )
            self.db.commit()
        return ReattributionResult(
            document_id=doc_id,
            source_patient_id="P001",
            target_patient_id="P001",
            queue_resolved=self.ok and queue_item_id is not None,
            error=None if self.ok else "errore simulato",
        )


class ValidationTabTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        root = Path(self._tmp.name)
        self.db = DatabaseEngine(root / "registry.db")
        init_database(self.db)
        patient_repo = PatientRepository(self.db)
        patient_repo.insert(Patient(id="P001", pseudonym="001", sex="M"))
        patient_repo.insert(Patient(id="P002", pseudonym="002", sex="F"))
        self.patient_repo = patient_repo
        self.lab_repo = FakeLabRepo()
        self.reattribution = FakeReattribution(self.db)
        self.audit_repo = AuditRepository(self.db)

        self.doc_file = root / "doc.pdf"
        self.doc_file.write_bytes(b"%PDF-fake")
        self.doc = DocumentRecord(
            id="DOC_000001", patient_id="P001", filename="doc.pdf",
            original_path=str(self.doc_file), file_hash="h",
            document_type="lettera_dimissione",
            import_date="2026-08-19T00:00:00",
        )
        self.doc_repo = FakeDocRepo(self.doc)

        self.tab = ValidationTab()
        self.tab.set_services({
            "db": self.db,
            "patient_repo": patient_repo,
            "lab_repo": self.lab_repo,
            "audit_repo": self.audit_repo,
            "document_repo": self.doc_repo,
            "document_reattribution": self.reattribution,
        })
        self.emitted: list[tuple] = []
        self.tab.document_reattributed.connect(
            lambda s, t: self.emitted.append((s, t))
        )
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

    def tearDown(self):
        self.tab.deleteLater()
        self._tmp.cleanup()

    def _insert_queue_row(self, item_type, item_id, original_value=None,
                          patient_id="P001"):
        self.db.execute(
            """INSERT INTO validation_queue
               (patient_id, item_type, item_id, issue, severity, status,
                original_value, created_at)
               VALUES (?, ?, ?, 'problema', 'high', 'pending', ?, ?)""",
            (
                patient_id, item_type, item_id,
                original_value,
                "2026-08-19T10:00:00",
            ),
        )
        self.db.commit()
        return self.db.execute(
            "SELECT MAX(id) AS mid FROM validation_queue"
        ).fetchone()["mid"]

    def _select_first_row(self):
        self.tab.load_patient("P001")
        self.tab._table.selectRow(0)

    def test_accept_attribution_moves_to_suggested(self):
        queue_id = self._insert_queue_row(
            "attribution", "DOC_000001",
            original_value=json.dumps({"suggested_patient_id": "P002"}),
        )
        self._select_first_row()

        self.tab._resolve("accepted")

        self.assertEqual(self.reattribution.calls, [
            ("DOC_000001", "P002", queue_id, "accepted"),
        ])
        self.assertEqual(self.emitted, [("P001", "P002")])
        self.assertEqual(self.tab._table.rowCount(), 0)  # risolta e spostata

    def test_accept_with_missing_suggested_falls_back_to_chooser(self):
        queue_id = self._insert_queue_row(
            "attribution", "DOC_000001",
            original_value=json.dumps({"suggested_patient_id": "P999"}),
        )
        self._select_first_row()

        with mock.patch(
            "emr_analyzer.gui.validation_tab.QInputDialog.getItem",
            return_value=("P002 — 002", True),
        ) as get_item:
            self.tab._resolve("accepted")

        get_item.assert_called_once()
        self.assertEqual(self.reattribution.calls, [
            ("DOC_000001", "P002", queue_id, "accepted"),
        ])

    def test_correct_attribution_uses_chooser_not_text_dialog(self):
        queue_id = self._insert_queue_row(
            "attribution", "DOC_000001",
            original_value=json.dumps({"suggested_patient_id": "P002"}),
        )
        self._select_first_row()

        with mock.patch(
            "emr_analyzer.gui.validation_tab.QInputDialog.getItem",
            return_value=("P002 — 002", True),
        ) as get_item, mock.patch(
            "emr_analyzer.gui.validation_tab.QInputDialog.getText"
        ) as get_text:
            self.tab._on_correct()

        get_text.assert_not_called()
        get_item.assert_called_once()
        self.assertEqual(self.reattribution.calls, [
            ("DOC_000001", "P002", queue_id, "corrected"),
        ])

    def test_lab_value_accept_unchanged(self):
        self._insert_queue_row("lab_value", "42")
        self._select_first_row()

        self.tab._resolve("accepted")

        self.assertEqual(self.lab_repo.validated, [42])
        self.assertEqual(self.reattribution.calls, [])
        row = self.db.execute(
            "SELECT status FROM validation_queue WHERE item_type='lab_value'"
        ).fetchone()
        self.assertEqual(row["status"], "accepted")

    def test_reject_attribution_does_not_move(self):
        self._insert_queue_row(
            "attribution", "DOC_000001",
            original_value=json.dumps({"suggested_patient_id": "P002"}),
        )
        self._select_first_row()

        self.tab._resolve("rejected")

        self.assertEqual(self.reattribution.calls, [])
        row = self.db.execute(
            "SELECT status FROM validation_queue "
            "WHERE item_type='attribution'"
        ).fetchone()
        self.assertEqual(row["status"], "rejected")

    def test_service_failure_leaves_row_pending(self):
        self.reattribution.ok = False
        queue_id = self._insert_queue_row(
            "attribution", "DOC_000001",
            original_value=json.dumps({"suggested_patient_id": "P002"}),
        )
        self._select_first_row()

        self.tab._resolve("accepted")

        self.assertEqual(len(self.reattribution.calls), 1)
        self.assertEqual(self.emitted, [])
        row = self.db.execute(
            "SELECT status FROM validation_queue WHERE id=?", (queue_id,)
        ).fetchone()
        self.assertEqual(row["status"], "pending")

    def test_chooser_cancel_aborts(self):
        self._insert_queue_row(
            "attribution", "DOC_000001",
            original_value=json.dumps({"suggested_patient_id": "P999"}),
        )
        self._select_first_row()

        with mock.patch(
            "emr_analyzer.gui.validation_tab.QInputDialog.getItem",
            return_value=("", False),
        ):
            self.tab._resolve("accepted")

        self.assertEqual(self.reattribution.calls, [])


if __name__ == "__main__":
    unittest.main()


class ValidationTabViewerTest(ValidationTabTest):
    """PDF inspection of attribution rows (double-click + Quick Look)."""

    def _attribution_row(self):
        self._insert_queue_row(
            "attribution", "DOC_000001",
            original_value=json.dumps({"suggested_patient_id": "P002"}),
        )
        self._select_first_row()

    def test_quick_look_resolves_attribution_document(self):
        self._attribution_row()
        resolved = self.tab._quick_look_path()
        self.assertIsNotNone(resolved)
        path, filename = resolved
        self.assertEqual(filename, "doc.pdf")
        self.assertTrue(os.path.isfile(path))

    def test_double_click_opens_pdf_viewer(self):
        self._attribution_row()
        with mock.patch(
            "emr_analyzer.gui.validation_tab.PDFViewerDialog"
        ) as viewer_cls:
            index = mock.MagicMock(column=lambda: 1)
            self.tab._on_double_click(index)
        viewer_cls.assert_called_once()
        doc_data = viewer_cls.call_args[0][0]
        self.assertEqual(doc_data["id"], "DOC_000001")
        self.assertEqual(doc_data["filename"], "doc.pdf")

    def test_lab_value_row_has_no_document_view(self):
        self._insert_queue_row("lab_value", "42")
        self._select_first_row()
        self.assertIsNone(self.tab._quick_look_path())
        self.assertFalse(self.tab._open_file())

    def test_missing_document_returns_nothing(self):
        self._insert_queue_row(
            "attribution", "DOC_999999",
            original_value=json.dumps({"suggested_patient_id": "P002"}),
        )
        self._select_first_row()
        self.assertIsNone(self.tab._quick_look_path())


if __name__ == "__main__":
    unittest.main()


class ValidationTabConfirmTest(ValidationTabTest):
    """In-place confirmation for ambiguous (conflict) attribution items."""

    def _conflict_row(self):
        return self._insert_queue_row(
            "attribution", "DOC_000001",
            original_value=json.dumps({"suggested_patient_id": "ignoto"}),
        )

    def test_accept_conflict_with_current_patient_confirms_in_place(self):
        queue_id = self._conflict_row()
        self._select_first_row()

        with mock.patch(
            "emr_analyzer.gui.validation_tab.QInputDialog.getItem",
            return_value=(
                "P001 — 001 (paziente corrente — conferma "
                "l'attribuzione attuale)", True,
            ),
        ):
            self.tab._resolve("accepted")

        self.assertEqual(self.reattribution.calls, [])
        self.assertEqual(self.reattribution.confirms, [
            ("DOC_000001", queue_id, "accepted"),
        ])
        self.assertEqual(self.emitted, [("P001", "P001")])
        self.assertEqual(self.tab._table.rowCount(), 0)

    def test_correct_conflict_with_current_patient_confirms_in_place(self):
        queue_id = self._conflict_row()
        self._select_first_row()

        with mock.patch(
            "emr_analyzer.gui.validation_tab.QInputDialog.getItem",
            return_value=("P001 — 001 (paziente corrente — conferma "
                          "l'attribuzione attuale)", True),
        ):
            self.tab._on_correct()

        self.assertEqual(self.reattribution.confirms, [
            ("DOC_000001", queue_id, "corrected"),
        ])

    def test_chooser_includes_current_patient_first(self):
        self._conflict_row()
        self._select_first_row()

        with mock.patch(
            "emr_analyzer.gui.validation_tab.QInputDialog.getItem",
            return_value=("P002 — 002", True),
        ) as get_item:
            self.tab._on_correct()

        labels = get_item.call_args[0][3]
        self.assertIn("paziente corrente", labels[0])

    def test_accept_with_valid_suggested_still_moves(self):
        queue_id = self._insert_queue_row(
            "attribution", "DOC_000001",
            original_value=json.dumps({"suggested_patient_id": "P002"}),
        )
        self._select_first_row()
        self.tab._resolve("accepted")
        self.assertEqual(self.reattribution.calls, [
            ("DOC_000001", "P002", queue_id, "accepted"),
        ])
        self.assertEqual(self.reattribution.confirms, [])


if __name__ == "__main__":
    unittest.main()

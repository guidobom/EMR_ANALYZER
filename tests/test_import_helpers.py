"""Tests for the reusable import helpers (run_file_checks, import_checked_documents)."""

from __future__ import annotations

from pathlib import Path
import shutil
import tempfile
import unittest

from emr_analyzer.config import active_workspace
from emr_analyzer.database.document_repo import DocumentRepository
from emr_analyzer.database.engine import DatabaseEngine
from emr_analyzer.database.migrations import init_database
from emr_analyzer.database.patient_repo import PatientRepository
from emr_analyzer.gui.import_dialog import (
    run_file_checks,
    import_checked_documents,
    guess_document_type,
)
from emr_analyzer.models import Patient
from emr_analyzer.models.document import DocumentRecord, DocumentType, ParsingStatus
from emr_analyzer.utils.file_utils import compute_file_hash


class ImportHelpersTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.mkdtemp(prefix="emr_import_")
        self._root = Path(self._tmp)
        self._staging = self._root / "staging"
        self._staging.mkdir()
        # Point the active workspace at the temp project for this test.
        self._orig_workspace_path = active_workspace.path
        active_workspace.set_path(self._root / "workspaces")

        db = DatabaseEngine(self._root / "workspaces" / "registry.db")
        init_database(db)
        self._doc_repo = DocumentRepository(db)
        self._patient_repo = PatientRepository(db)
        self._patient_repo.insert(Patient(id="P001", pseudonym="001", sex="M"))
        self._services = {
            "document_repo": self._doc_repo,
            "patient_repo": self._patient_repo,
        }

    def tearDown(self):
        active_workspace.set_path(self._orig_workspace_path)
        shutil.rmtree(self._tmp, ignore_errors=True)

    def _write(self, name: str, content: bytes) -> str:
        path = self._staging / name
        path.write_bytes(content)
        return str(path)

    def test_run_file_checks_flags_unsupported_and_duplicate(self):
        supported = self._write("esami.png", b"png-content")
        unsupported = self._write("note.txt", b"not-supported")
        duplicate = self._write("tac.png", b"dup-content")
        dup_hash = compute_file_hash(duplicate)
        self._doc_repo.insert(DocumentRecord(
            id="DOC_000001", patient_id="P001", filename="tac.png",
            original_path=duplicate, file_hash=dup_hash,
        ))

        checks = run_file_checks(
            self._services, [supported, unsupported, duplicate]
        )
        by_path = {c["path"]: c for c in checks}

        # Supported PNG → a real check entry
        self.assertIn("check", by_path[supported])
        self.assertEqual(by_path[supported]["status"], "Pronto")

        # Unsupported → flagged, no check entry
        self.assertEqual(by_path[unsupported]["status"], "Formato non supportato")
        self.assertNotIn("check", by_path[unsupported])

        # Duplicate → flagged via global hash lookup
        self.assertTrue(by_path[duplicate]["is_duplicate"])
        self.assertNotEqual(by_path[duplicate]["status"], "Pronto")

    def test_appledouble_sidecar_files_are_rejected(self):
        # macOS ._* AppleDouble sidecars are metadata, never documents: they
        # carry no readable text, would always fail identity extraction and
        # pollute the "Da assegnare" bucket forever.
        from emr_analyzer.utils.file_utils import is_supported_file

        self.assertFalse(is_supported_file("._10a9b448416DC157.pdf"))
        self.assertFalse(is_supported_file(self._write("._esami.png", b"x")))
        self.assertTrue(is_supported_file(self._write("esami.pdf", b"x")))

    def test_import_checked_documents_copies_and_creates_record(self):
        src = self._write("esami.png", b"png-content")
        checks = run_file_checks(self._services, [src])

        imported = import_checked_documents(self._services, "P001", checks, [0])

        self.assertEqual(len(imported), 1)
        doc = self._doc_repo.get_by_id(imported[0])
        self.assertIsNotNone(doc)
        self.assertEqual(doc.patient_id, "P001")
        self.assertEqual(doc.parsing_status, ParsingStatus.PENDING.value)
        workspace_copy = (
            active_workspace.path / "P001" / "documents" / "original" / "esami.png"
        )
        self.assertTrue(workspace_copy.exists())
        self.assertEqual(doc.original_path, str(workspace_copy))

    def test_import_checked_documents_skips_duplicate(self):
        dup = self._write("tac.png", b"dup-content")
        dup_hash = compute_file_hash(dup)
        self._doc_repo.insert(DocumentRecord(
            id="DOC_000001", patient_id="P001", filename="tac.png",
            original_path=dup, file_hash=dup_hash,
        ))
        checks = run_file_checks(self._services, [dup])

        imported = import_checked_documents(self._services, "P001", checks, [0])

        self.assertEqual(imported, [])

    def test_import_checked_documents_raises_for_unknown_patient(self):
        src = self._write("esami.png", b"png-content")
        checks = run_file_checks(self._services, [src])

        with self.assertRaises(ValueError):
            import_checked_documents(self._services, "P999", checks, [0])

    def test_guess_document_type_from_filename(self):
        self.assertEqual(
            guess_document_type("referto_emocromo.pdf"),
            DocumentType.LABORATORIO.value,
        )
        self.assertEqual(
            guess_document_type("lettera_dimissione.pdf"),
            DocumentType.LETTERA_DIMISSIONE.value,
        )
        self.assertEqual(
            guess_document_type("tac_encefalo.pdf"),
            DocumentType.RADIOLOGIA.value,
        )


if __name__ == "__main__":
    unittest.main()

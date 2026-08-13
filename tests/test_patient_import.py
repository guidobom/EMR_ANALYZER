"""Regression tests for the patient import service.

Covers the v1.0 blocker: ``list_source_patients`` / ``_import_one`` used to
SELECT a ``clinical_profile`` DB column that does not exist — the profile
lives inside ``clinical_state.state_json``.  The crash made any import from
another project fail with ``sqlite3.OperationalError``.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from emr_analyzer.clinical.patient_import import PatientImportService
from emr_analyzer.config import active_workspace
from emr_analyzer.database.clinical_state_repo import ClinicalStateRepository
from emr_analyzer.database.document_repo import DocumentRepository
from emr_analyzer.database.engine import DatabaseEngine
from emr_analyzer.database.migrations import init_database
from emr_analyzer.database.patient_repo import PatientRepository
from emr_analyzer.models import Patient
from emr_analyzer.models.clinical_state import ClinicalState
from emr_analyzer.models.document import DocumentRecord
from emr_analyzer.utils.file_utils import compute_file_hash


class PatientImportTest(unittest.TestCase):
    def setUp(self):
        self._root = Path(tempfile.mkdtemp(prefix="emr_import_"))
        self._orig_workspace_path = active_workspace.path
        active_workspace.set_path(self._root / "target")
        (self._root / "target").mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        active_workspace.set_path(self._orig_workspace_path)

    def _make_source(self, with_profile: bool = True):
        """Build a source project with one patient carrying all data."""
        source = self._root / "source"
        db = DatabaseEngine(source / "emr_registry.db")
        init_database(db)

        patient_repo = PatientRepository(db)
        patient_repo.insert(Patient(
            id="P001", pseudonym="001", sex="M",
            birth_year=1950, initials="MB",
        ))

        original = source / "P001" / "documents" / "original" / "ref.pdf"
        original.parent.mkdir(parents=True, exist_ok=True)
        original.write_bytes(b"source document bytes")

        doc_repo = DocumentRepository(db)
        doc_repo.insert(DocumentRecord(
            id="DOC_000001", patient_id="P001", filename="ref.pdf",
            original_path=str(original),
            file_hash=compute_file_hash(original),
        ))

        db.execute(
            """INSERT INTO clinical_timeline
               (entry_id, patient_id, date_observed, date_resolved, category,
                description, source_document_ids, source_texts,
                merged_into_ids, status, confidence, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            ("CTL_000001", "P001", "2020-05-01", "", "diagnosis",
             "Melanoma", '["DOC_000001"]', '["text"]', "[]", "active", 0.8,
             "now", "now"),
        )

        if with_profile:
            state = ClinicalState(patient_id="P001")
            state.clinical_profile = "Profilo clinico di test"
            ClinicalStateRepository(db).save(state)

        db.close()
        return source

    def test_list_source_patients_reads_profile_from_state_json(self):
        """The schema has no clinical_profile column; it must not crash."""
        source = self._make_source()
        patients = PatientImportService().list_source_patients(source)
        self.assertEqual(len(patients), 1)
        self.assertEqual(patients[0]["id"], "P001")
        self.assertTrue(patients[0]["has_profile"])
        self.assertEqual(patients[0]["document_count"], 1)
        self.assertEqual(patients[0]["timeline_entries"], 1)

    def test_import_patient_copies_profile_and_provenance(self):
        source = self._make_source()
        stats = PatientImportService().import_patients(source, ["P001"])
        self.assertEqual(stats["patients"], 1)
        self.assertEqual(stats["documents"], 1)
        self.assertEqual(stats["timeline"], 1)
        self.assertEqual(stats["profiles"], 1)

        target_db = DatabaseEngine(active_workspace.path / "emr_registry.db")
        row = target_db.execute(
            "SELECT state_json FROM clinical_state"
        ).fetchone()
        self.assertIsNotNone(row)
        data = json.loads(row["state_json"])
        self.assertEqual(data["clinical_profile"], "Profilo clinico di test")
        self.assertEqual(data["patient_id"], "P001")
        target_db.close()

    def test_import_patient_without_profile_succeeds(self):
        source = self._make_source(with_profile=False)
        stats = PatientImportService().import_patients(source, ["P001"])
        self.assertEqual(stats["patients"], 1)
        self.assertEqual(stats["profiles"], 0)


if __name__ == "__main__":
    unittest.main()

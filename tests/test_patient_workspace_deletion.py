from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from emr_analyzer.clinical.patient_deletion import (
    PatientWorkspaceDeletionService,
)
from emr_analyzer.database.document_repo import DocumentRepository
from emr_analyzer.database.engine import DatabaseEngine
from emr_analyzer.database.migrations import init_database
from emr_analyzer.database.patient_repo import PatientRepository
from emr_analyzer.models import Patient
from emr_analyzer.models.document import DocumentRecord
from emr_analyzer.utils.file_utils import compute_file_hash


class FailingPatientDeletionService(PatientWorkspaceDeletionService):
    def _delete_database_records(self, patient_id: str) -> dict[str, int]:
        raise RuntimeError("synthetic database failure")


class PatientWorkspaceDeletionTest(unittest.TestCase):
    def _fixture(self, root: Path):
        workspaces = root / "workspaces"
        cache = root / "cache"
        db = DatabaseEngine(workspaces / "registry.db")
        init_database(db)
        patient_repo = PatientRepository(db)
        patient_repo.insert(Patient(id="P001", pseudonym="001"))
        patient_repo.insert(Patient(id="P002", pseudonym="002"))

        original = workspaces / "P001" / "documents" / "original" / "report.pdf"
        original.parent.mkdir(parents=True)
        original.write_bytes(b"patient-one-document")
        document_hash = compute_file_hash(original)
        DocumentRepository(db).insert(DocumentRecord(
            id="DOC_000001", patient_id="P001", filename="report.pdf",
            original_path=str(original), file_hash=document_hash,
        ))
        extraction = workspaces / "P001" / "extraction"
        extraction.mkdir(parents=True)
        (extraction / "DOC_000001.md").write_text(
            "testo clinico", encoding="utf-8"
        )
        cache_patient = cache / "P001"
        cache_patient.mkdir(parents=True)
        (cache_patient / "cache.bin").write_bytes(b"cache")

        matching_inbox = workspaces / "_inbox" / "BATCH_TEST" / "00001" / "copy.pdf"
        matching_inbox.parent.mkdir(parents=True)
        matching_inbox.write_bytes(b"patient-one-document")
        unrelated_inbox = workspaces / "_inbox" / "BATCH_TEST" / "00002" / "other.pdf"
        unrelated_inbox.parent.mkdir(parents=True)
        unrelated_inbox.write_bytes(b"another-patient")
        trash = workspaces / "_trash" / "old_DOC_000001_residue"
        trash.mkdir(parents=True)
        (trash / "source.pdf").write_bytes(b"patient-one-document")

        now = "2024-01-01T00:00:00"
        statements = [
            ("""INSERT INTO patient_identities
                (patient_id, confidence, status, source_document_id,
                 created_at, updated_at) VALUES (?, 1, 'validated', ?, ?, ?)""",
             ("P001", "DOC_000001", now, now)),
            ("""INSERT INTO document_identity_evidence
                (document_id, patient_id, field_name, value_key,
                 extraction_method, confidence, created_at)
                VALUES (?, ?, 'name', 'key', 'test', 1, ?)""",
             ("DOC_000001", "P001", now)),
            ("""INSERT INTO clinical_evidence
                (evidence_id, patient_id, document_id, category,
                 normalized_entity, source_text, extraction_method,
                 schema_version, created_at)
                VALUES ('EVD1', ?, ?, 'diagnosis', 'x', 'x',
                        'test', '1', ?)""", ("P001", "DOC_000001", now)),
            ("""INSERT INTO lab_values
                (patient_id, document_id, parameter_name, normalized_name, value)
                VALUES (?, ?, 'Hb', 'emoglobina', 10)""",
             ("P001", "DOC_000001")),
            ("""INSERT INTO clinical_state
                (patient_id, state_json, updated_at) VALUES (?, '{}', ?)""",
             ("P001", now)),
            ("""INSERT INTO validation_queue
                (patient_id, item_type, item_id, issue, created_at)
                VALUES (?, 'document', ?, 'review', ?)""",
             ("P001", "DOC_000001", now)),
            ("""INSERT INTO audit_log
                (patient_id, action, target_type, target_id, timestamp)
                VALUES (?, 'test', 'patient', ?, ?)""", ("P001", "P001", now)),
            ("""INSERT INTO audit_log
                (patient_id, action, target_type, target_id, timestamp)
                VALUES (?, 'test', 'patient', ?, ?)""", ("P002", "P002", now)),
            ("""INSERT INTO clinical_chat
                (id, patient_id, role, content, model_used, context_mode,
                 created_at)
                VALUES (?, ?, 'user', 'domanda', '', 0, ?)""",
             ("CHAT_1", "P001", now)),
        ]
        with db:
            for sql, params in statements:
                db.execute(sql, params)
        return {
            "db": db,
            "patient_repo": patient_repo,
            "workspaces": workspaces,
            "cache": cache,
            "original": original,
            "matching_inbox": matching_inbox,
            "unrelated_inbox": unrelated_inbox,
            "trash": trash,
        }

    def test_complete_deletion_removes_all_patient_records_and_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = self._fixture(Path(tmp))
            service = PatientWorkspaceDeletionService(
                fixture["db"], fixture["patient_repo"],
                fixture["workspaces"], fixture["cache"],
            )

            result = service.delete("P001")

            self.assertTrue(result.deleted, result.error)
            self.assertIsNone(fixture["patient_repo"].get_by_id("P001"))
            self.assertIsNotNone(fixture["patient_repo"].get_by_id("P002"))
            for table in service._tables_with_patient_id():
                self.assertEqual(fixture["db"].execute(
                    f'SELECT COUNT(*) FROM "{table}" WHERE patient_id=?',
                    ("P001",),
                ).fetchone()[0], 0, table)
            self.assertEqual(
                fixture["db"].get_table_count(
                    "audit_log", "patient_id='P002'"
                ), 1
            )
            self.assertFalse((fixture["workspaces"] / "P001").exists())
            self.assertFalse((fixture["cache"] / "P001").exists())
            self.assertFalse(fixture["matching_inbox"].exists())
            self.assertTrue(fixture["unrelated_inbox"].exists())
            self.assertFalse(fixture["trash"].exists())
            self.assertFalse(
                (fixture["workspaces"] / "_deletion_staging").exists()
            )

    def test_database_failure_restores_every_moved_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = self._fixture(Path(tmp))
            service = FailingPatientDeletionService(
                fixture["db"], fixture["patient_repo"],
                fixture["workspaces"], fixture["cache"],
            )

            result = service.delete("P001")

            self.assertFalse(result.deleted)
            self.assertIn("synthetic database failure", result.error)
            self.assertIsNotNone(fixture["patient_repo"].get_by_id("P001"))
            self.assertTrue(fixture["original"].exists())
            self.assertTrue((fixture["cache"] / "P001" / "cache.bin").exists())
            self.assertTrue(fixture["matching_inbox"].exists())
            self.assertTrue(fixture["trash"].exists())
            self.assertFalse(
                (fixture["workspaces"] / "_deletion_staging").exists()
            )


if __name__ == "__main__":
    unittest.main()

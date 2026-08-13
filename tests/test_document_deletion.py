from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from emr_analyzer.clinical.document_deletion import DocumentDeletionService
from emr_analyzer.database.audit_repo import AuditRepository
from emr_analyzer.database.document_repo import DocumentRepository
from emr_analyzer.database.engine import DatabaseEngine
from emr_analyzer.database.migrations import init_database
from emr_analyzer.database.patient_repo import PatientRepository
from emr_analyzer.models import Patient
from emr_analyzer.models.document import DocumentRecord


class DocumentDeletionTest(unittest.TestCase):
    def test_document_deletion_removes_files_and_derived_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            db = DatabaseEngine(tmp_path / "registry.db")
            init_database(db)
            patient_repo = PatientRepository(db)
            document_repo = DocumentRepository(db)
            patient_repo.insert(Patient(id="P001", pseudonym="001"))

            original_dir = tmp_path / "P001" / "documents" / "original"
            original_dir.mkdir(parents=True)
            original = original_dir / "report.pdf"
            original.write_bytes(b"synthetic-pdf")
            document = DocumentRecord(
                id="DOC_000001",
                patient_id="P001",
                filename="report.pdf",
                original_path=str(original),
                file_hash="abc123",
                page_count=1,
            )
            document_repo.insert(document)

            extraction_dir = tmp_path / "P001" / "extraction"
            extraction_dir.mkdir(parents=True)
            for name in (
                "DOC_000001.md", "DOC_000001.json", "DOC_000001_raw.md",
                "DOC_000001_pages.jsonl", "DOC_000001_words.jsonl",
                "DOC_000001_tables.json", "DOC_000001_clinical_evidence.json",
            ):
                (extraction_dir / name).write_text("derived", encoding="utf-8")

            db.execute(
        """INSERT INTO lab_values
           (patient_id, document_id, parameter_name, normalized_name, value)
           VALUES ('P001', 'DOC_000001', 'Hb', 'emoglobina', 10.0)"""
            )
            db.execute(
        """INSERT INTO clinical_evidence
           (evidence_id, patient_id, document_id, category, normalized_entity,
            source_text, extraction_method, schema_version, created_at)
           VALUES ('EVD_000001', 'P001', 'DOC_000001', 'symptom', 'dispnea',
                   'dispnea', 'deterministic_lab', '1.0', '2024-01-01')"""
            )
            db.execute(
        """INSERT INTO validation_queue
           (patient_id, item_type, item_id, issue, severity, status, created_at)
           VALUES ('P001', 'evidence', 'EVD_000001', 'review',
                   'medium', 'pending', '2024-01-01')"""
            )
            db.commit()

            service = DocumentDeletionService(
                db,
                document_repo,
                audit_repo=AuditRepository(db),
                workspaces_dir=tmp_path,
            )
            result = service.delete("DOC_000001")

            self.assertTrue(result.deleted)
            self.assertIsNone(result.error)
            self.assertIsNone(document_repo.get_by_id("DOC_000001"))
            self.assertFalse(original.exists())
            self.assertFalse(any(extraction_dir.iterdir()))
            self.assertEqual(db.get_table_count("lab_values"), 0)
            self.assertEqual(db.get_table_count("clinical_evidence"), 0)
            self.assertEqual(db.get_table_count("validation_queue"), 0)
            self.assertEqual(db.get_table_count("audit_log", "action='delete'"), 1)


if __name__ == "__main__":
    unittest.main()

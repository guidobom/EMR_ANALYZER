"""Tests for single-document re-attribution between patient workspaces."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from emr_analyzer.clinical.document_reattribution import (
    DocumentReattributionService,
    ReattributionResult,
)
from emr_analyzer.database.audit_repo import AuditRepository
from emr_analyzer.database.document_repo import DocumentRepository
from emr_analyzer.database.engine import DatabaseEngine
from emr_analyzer.database.migrations import init_database
from emr_analyzer.database.patient_repo import PatientRepository
from emr_analyzer.models import Patient
from emr_analyzer.models.document import DocumentRecord
from emr_analyzer.utils.file_utils import compute_file_hash


class FailingReattributionService(DocumentReattributionService):
    """Raise inside the DB transaction to exercise the restore path."""

    def _repoint_timeline(self, source, target, doc_id, result):
        raise RuntimeError("synthetic database failure")


class DocumentReattributionTest(unittest.TestCase):
    def _fixture(self, root: Path):
        workspaces = root / "workspaces"
        db = DatabaseEngine(workspaces / "registry.db")
        init_database(db)
        patient_repo = PatientRepository(db)
        patient_repo.insert(Patient(id="P001", pseudonym="001", sex="M"))
        patient_repo.insert(Patient(id="P002", pseudonym="002", sex="F"))
        doc_repo = DocumentRepository(db)
        audit_repo = AuditRepository(db)

        original = (
            workspaces / "P001" / "documents" / "original" / "referto.pdf"
        )
        original.parent.mkdir(parents=True, exist_ok=True)
        original.write_bytes(b"lab-report-content")
        file_hash = compute_file_hash(original)
        extraction = workspaces / "P001" / "extraction" / "DOC_000001.md"
        extraction.parent.mkdir(parents=True, exist_ok=True)
        extraction.write_text("testo estratto", encoding="utf-8")
        docling = workspaces / "P001" / "docling" / "DOC_000001.md"
        docling.parent.mkdir(parents=True, exist_ok=True)
        docling.write_text("docling", encoding="utf-8")

        doc = DocumentRecord(
            id="DOC_000001", patient_id="P001", filename="referto.pdf",
            original_path=str(original), file_hash=file_hash,
            document_type="laboratorio", import_date="2026-08-19T00:00:00",
            parsing_status="completed", extraction_status="error",
            error_message="Possibile attribuzione errata",
        )
        doc_repo.insert(doc)

        now = "2026-08-19T10:00:00.000000"
        with db:
            db.execute(
                """INSERT INTO document_identity_evidence
                   (document_id, patient_id, field_name, value_key,
                    extraction_method, created_at)
                   VALUES ('DOC_000001', 'P001', 'cf', 'key', 'import', ?)""",
                (now,),
            )
            db.execute(
                """INSERT INTO clinical_evidence
                   (evidence_id, patient_id, document_id, category,
                    normalized_entity, source_text, extraction_method,
                    schema_version, created_at)
                   VALUES ('EV1', 'P001', 'DOC_000001', 'lab', 'emoglobina',
                           'testo', 'deterministic_lab', '1', ?)""",
                (now,),
            )
            db.execute(
                """INSERT INTO lab_values
                   (patient_id, document_id, parameter_name, normalized_name,
                    value, unit, is_abnormal, sample_date)
                   VALUES ('P001', 'DOC_000001', 'Emoglobina', 'emoglobina',
                           10.2, 'g/dL', 1, '2020-05-01')"""
            )
            db.execute(
                """INSERT INTO validation_queue
                   (patient_id, item_type, item_id, issue, severity, status,
                    original_value, created_at)
                   VALUES ('P001', 'attribution', 'DOC_000001',
                           'Possibile attribuzione errata', 'high', 'pending',
                           ?, ?)""",
                (json.dumps({"suggested_patient_id": "P002"}), now),
            )
            db.execute(
                """INSERT INTO audit_log
                   (patient_id, action, target_type, target_id, timestamp)
                   VALUES ('P001', 'attribution_mismatch', 'document',
                           'DOC_000001', ?)""",
                (now,),
            )
            db.execute(
                """INSERT INTO clinical_timeline
                   (entry_id, patient_id, date_observed, category,
                    description, source_document_ids, source_texts, status,
                    created_at, updated_at)
                   VALUES ('T1', 'P001', '2020-05-01', 'laboratory',
                           'emoglobina 10.2', '["DOC_000001"]', '[]',
                           'active', ?, ?)""",
                (now, now),
            )
            db.execute(
                """INSERT INTO clinical_timeline
                   (entry_id, patient_id, date_observed, category,
                    description, source_document_ids, source_texts, status,
                    created_at, updated_at)
                   VALUES ('T2', 'P001', '2020-05-02', 'other',
                           'mista', '["DOC_000001","DOC_000009"]', '[]',
                           'active', ?, ?)""",
                (now, now),
            )
            db.execute(
                """INSERT INTO clinical_state
                   (patient_id, state_json, updated_at)
                   VALUES ('P001', '{}', ?), ('P002', '{}', ?)""",
                (now, now),
            )

        return {
            "db": db,
            "patient_repo": patient_repo,
            "doc_repo": doc_repo,
            "audit_repo": audit_repo,
            "workspaces": workspaces,
            "original": original,
            "extraction": extraction,
            "docling": docling,
        }

    def _service(self, fixture, cls=DocumentReattributionService):
        return cls(
            fixture["db"], fixture["doc_repo"], fixture["patient_repo"],
            fixture["audit_repo"], fixture["workspaces"],
        )

    def _queue_row(self, fixture):
        return fixture["db"].execute(
            "SELECT * FROM validation_queue WHERE item_id='DOC_000001' "
            "AND item_type='attribution' ORDER BY id LIMIT 1"
        ).fetchone()

    def test_successful_accept_move(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = self._fixture(Path(tmp))
            service = self._service(fixture)
            result = service.move_document(
                "DOC_000001", "P002", queue_item_id=1,
                resolution_status="accepted",
            )
            self.assertTrue(result.ok, result.error)

            doc = fixture["doc_repo"].get_by_id("DOC_000001")
            self.assertEqual(doc.patient_id, "P002")
            self.assertTrue(
                doc.original_path.startswith(
                    str(fixture["workspaces"] / "P002")
                )
            )
            self.assertEqual(doc.extraction_status, "pending")
            self.assertIsNone(doc.error_message)

            # Bound rows repointed.
            for table in ("clinical_evidence", "lab_values",
                          "document_identity_evidence"):
                row = fixture["db"].execute(
                    f"SELECT patient_id FROM {table} "
                    "WHERE document_id='DOC_000001' LIMIT 1"
                ).fetchone()
                self.assertEqual(row["patient_id"], "P002", table)

            # Queue row resolved and repointed.
            qrow = self._queue_row(fixture)
            self.assertEqual(qrow["patient_id"], "P002")
            self.assertEqual(qrow["status"], "accepted")
            self.assertIsNotNone(qrow["resolved_at"])
            self.assertIsNone(qrow["corrected_value"])
            self.assertTrue(result.queue_resolved)

            # Audit: mismatch row repointed + new reattribution row.
            audit_rows = fixture["db"].execute(
                "SELECT patient_id, action FROM audit_log "
                "WHERE target_id='DOC_000001' ORDER BY id"
            ).fetchall()
            actions = {r["action"]: r["patient_id"] for r in audit_rows}
            self.assertEqual(actions.get("attribution_mismatch"), "P002")
            self.assertEqual(actions.get("document_reattributed"), "P002")

            # Timeline: single-doc row repointed, mixed row stays.
            t1 = fixture["db"].execute(
                "SELECT patient_id FROM clinical_timeline "
                "WHERE entry_id='T1'"
            ).fetchone()
            t2 = fixture["db"].execute(
                "SELECT patient_id FROM clinical_timeline "
                "WHERE entry_id='T2'"
            ).fetchone()
            self.assertEqual(t1["patient_id"], "P002")
            self.assertEqual(t2["patient_id"], "P001")
            self.assertTrue(any(
                "più documenti" in w for w in result.warnings
            ))

            # Clinical state gone for both patients.
            for pid in ("P001", "P002"):
                self.assertIsNone(fixture["db"].execute(
                    "SELECT 1 FROM clinical_state WHERE patient_id=?",
                    (pid,),
                ).fetchone())

            # Files physically moved.
            self.assertFalse(fixture["original"].exists())
            self.assertFalse(fixture["extraction"].exists())
            self.assertTrue(Path(doc.original_path).is_file())

    def test_corrected_resolution_sets_corrected_value(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = self._fixture(Path(tmp))
            result = self._service(fixture).move_document(
                "DOC_000001", "P002", queue_item_id=1,
                resolution_status="corrected",
            )
            self.assertTrue(result.ok, result.error)
            qrow = self._queue_row(fixture)
            self.assertEqual(qrow["status"], "corrected")
            self.assertEqual(qrow["corrected_value"], "P002")

    def test_db_failure_restores_files_and_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = self._fixture(Path(tmp))
            result = self._service(
                fixture, cls=FailingReattributionService
            ).move_document("DOC_000001", "P002", queue_item_id=1)
            self.assertFalse(result.ok)
            self.assertIn("synthetic database failure", result.error or "")

            doc = fixture["doc_repo"].get_by_id("DOC_000001")
            self.assertEqual(doc.patient_id, "P001")
            self.assertEqual(doc.extraction_status, "error")
            self.assertTrue(fixture["original"].exists())
            self.assertTrue(fixture["extraction"].exists())
            self.assertEqual(self._queue_row(fixture)["status"], "pending")

    def test_target_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = self._fixture(Path(tmp))
            result = self._service(fixture).move_document(
                "DOC_000001", "P999"
            )
            self.assertFalse(result.ok)
            self.assertIn("non trovato", result.error or "")
            self.assertTrue(fixture["original"].exists())
            doc = fixture["doc_repo"].get_by_id("DOC_000001")
            self.assertEqual(doc.patient_id, "P001")

    def test_target_equals_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = self._fixture(Path(tmp))
            result = self._service(fixture).move_document(
                "DOC_000001", "P001"
            )
            self.assertFalse(result.ok)
            self.assertIn("appartiene già", result.error or "")

    def test_document_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = self._fixture(Path(tmp))
            result = self._service(fixture).move_document(
                "DOC_999999", "P002"
            )
            self.assertFalse(result.ok)
            self.assertIn("non trovato", result.error or "")

    def test_byte_identical_duplicate_in_target(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = self._fixture(Path(tmp))
            workspaces = fixture["workspaces"]
            # The same file already exists under the target patient.
            existing = (
                workspaces / "P002" / "documents" / "original" / "referto.pdf"
            )
            existing.parent.mkdir(parents=True, exist_ok=True)
            existing.write_bytes(b"lab-report-content")
            duplicate = DocumentRecord(
                id="DOC_000009", patient_id="P002", filename="referto.pdf",
                original_path=str(existing),
                file_hash=compute_file_hash(existing),
                document_type="laboratorio", import_date="2026-08-19T00:00:00",
            )
            fixture["doc_repo"].insert(duplicate)

            result = self._service(fixture).move_document(
                "DOC_000001", "P002", queue_item_id=1
            )
            self.assertTrue(result.ok, result.error)
            doc = fixture["doc_repo"].get_by_id("DOC_000001")
            self.assertEqual(doc.patient_id, "P002")
            # The record points at the target's identical file.
            self.assertTrue(Path(doc.original_path).is_file())
            self.assertFalse(fixture["original"].exists())
            self.assertEqual(doc.extraction_status, "pending")

    def test_missing_queue_item_warns_but_moves(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = self._fixture(Path(tmp))
            result = self._service(fixture).move_document(
                "DOC_000001", "P002", queue_item_id=999
            )
            self.assertTrue(result.ok, result.error)
            self.assertFalse(result.queue_resolved)
            self.assertTrue(any(
                "coda" in w for w in result.warnings
            ))


if __name__ == "__main__":
    unittest.main()


class ConfirmAttributionTest(DocumentReattributionTest):
    """In-place confirmation: nothing moves, the document is unlocked."""

    def test_confirm_unlocks_and_resolves(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = self._fixture(Path(tmp))
            service = self._service(fixture)
            result = service.confirm_attribution(
                "DOC_000001", queue_item_id=1, resolution_status="accepted"
            )
            self.assertTrue(result.ok, result.error)

            doc = fixture["doc_repo"].get_by_id("DOC_000001")
            self.assertEqual(doc.patient_id, "P001")  # non spostato
            self.assertEqual(doc.extraction_status, "pending")
            self.assertIsNone(doc.error_message)

            qrow = fixture["db"].execute(
                "SELECT * FROM validation_queue WHERE id=1"
            ).fetchone()
            self.assertEqual(qrow["status"], "accepted")
            self.assertIsNotNone(qrow["resolved_at"])
            self.assertEqual(qrow["patient_id"], "P001")

            # Files untouched.
            self.assertTrue(fixture["original"].exists())

            # Narrative invalidated for the source patient only.
            self.assertIsNone(fixture["db"].execute(
                "SELECT 1 FROM clinical_state WHERE patient_id='P001'"
            ).fetchone())
            self.assertIsNotNone(fixture["db"].execute(
                "SELECT 1 FROM clinical_state WHERE patient_id='P002'"
            ).fetchone())

            audit = fixture["db"].execute(
                "SELECT action FROM audit_log "
                "WHERE action='attribution_confirmed'"
            ).fetchone()
            self.assertIsNotNone(audit)

    def test_confirm_missing_document(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = self._fixture(Path(tmp))
            result = self._service(fixture).confirm_attribution(
                "DOC_999999", queue_item_id=1
            )
            self.assertFalse(result.ok)
            self.assertIn("non trovato", result.error or "")

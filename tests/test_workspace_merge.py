from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from emr_analyzer.clinical.patient_deletion import (
    PatientWorkspaceDeletionService,
)
from emr_analyzer.clinical.workspace_merge import WorkspaceMergeService
from emr_analyzer.database.document_repo import DocumentRepository
from emr_analyzer.database.engine import DatabaseEngine
from emr_analyzer.database.migrations import init_database
from emr_analyzer.database.patient_repo import PatientRepository
from emr_analyzer.models import Patient
from emr_analyzer.models.document import DocumentRecord
from emr_analyzer.utils.file_utils import compute_file_hash


class FailingMergeService(WorkspaceMergeService):
    """Raise inside the DB transaction to exercise the restore path."""

    def _merge_timeline(self, source_pid, target_pid, doc_map):
        raise RuntimeError("synthetic database failure")


class WorkspaceMergeTest(unittest.TestCase):
    def _fixture(self, root: Path):
        """Build a project with P001 (source) and P002 (target).

        P001 documents:
          DOC_000001 report.pdf (unique content) + extraction artifact
          DOC_000002 dup.pdf   (byte-identical to P002's DOC_000004)
        P002 documents:
          DOC_000003 own.pdf   (unique content)
          DOC_000004 dup.pdf   (byte-identical to P001's DOC_000002)
        """
        workspaces = root / "workspaces"
        cache = root / "cache"
        db = DatabaseEngine(workspaces / "registry.db")
        init_database(db)
        patient_repo = PatientRepository(db)
        patient_repo.insert(Patient(id="P001", pseudonym="001", sex="M"))
        patient_repo.insert(Patient(id="P002", pseudonym="002", sex="F"))

        def make_original(pid: str, name: str, content: bytes) -> Path:
            path = workspaces / pid / "documents" / "original" / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
            return path

        p1_original = make_original("P001", "report.pdf", b"patient-one-document")
        p1_hash = compute_file_hash(p1_original)
        p1_dup = make_original("P001", "dup.pdf", b"duplicate-content")
        p1_dup_hash = compute_file_hash(p1_dup)

        p2_own = make_original("P002", "own.pdf", b"target-own-document")
        p2_own_hash = compute_file_hash(p2_own)
        p2_dup = make_original("P002", "dup.pdf", b"duplicate-content")
        p2_dup_hash = compute_file_hash(p2_dup)
        self.assertEqual(p1_dup_hash, p2_dup_hash)

        doc_repo = DocumentRepository(db)
        doc_repo.insert(DocumentRecord(
            id="DOC_000001", patient_id="P001", filename="report.pdf",
            original_path=str(p1_original), file_hash=p1_hash,
        ))
        doc_repo.insert(DocumentRecord(
            id="DOC_000002", patient_id="P001", filename="dup.pdf",
            original_path=str(p1_dup), file_hash=p1_dup_hash,
        ))
        doc_repo.insert(DocumentRecord(
            id="DOC_000003", patient_id="P002", filename="own.pdf",
            original_path=str(p2_own), file_hash=p2_own_hash,
        ))
        doc_repo.insert(DocumentRecord(
            id="DOC_000004", patient_id="P002", filename="dup.pdf",
            original_path=str(p2_dup), file_hash=p2_dup_hash,
        ))

        extraction = workspaces / "P001" / "extraction"
        extraction.mkdir(parents=True)
        (extraction / "DOC_000001.md").write_text(
            "testo clinico", encoding="utf-8"
        )

        now = "2024-01-01T00:00:00"
        statements = [
            # Identities: P001 and P002 share name+birth (duplicate person),
            # but have different fiscal codes.
            ("""INSERT INTO patient_identities
                (patient_id, fiscal_code_key, normalized_name_key,
                 birth_date_key, sex, confidence, status, created_at, updated_at)
                VALUES (?, 'CF1', 'N1', 'B1', 'M', 1.0, 'validated', ?, ?)""",
             ("P001", now, now)),
            ("""INSERT INTO patient_identities
                (patient_id, fiscal_code_key, normalized_name_key,
                 birth_date_key, sex, confidence, status, created_at, updated_at)
                VALUES (?, 'CF2', 'N1', 'B1', 'F', 0.9, 'validated', ?, ?)""",
             ("P002", now, now)),
            # Per-document data for the moved document.
            ("""INSERT INTO docling_elements
                (document_id, element_id, type, text)
                VALUES (?, 'el1', 'text', 'clinical')""", ("DOC_000001",)),
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
            ("""INSERT INTO document_clinical_projections
                (projection_id, patient_id, document_id, projection_json,
                 consolidation_method, schema_version, created_at, updated_at)
                VALUES ('PRJ1', ?, ?, '{}', 'test', '1', ?, ?)""",
             ("P001", "DOC_000001", now, now)),
            ("""INSERT INTO clinical_events
                (event_id, patient_id, event_date, event_type, entity,
                 source_document_id, source_text, created_at)
                VALUES ('EVT1', ?, '2024-01-01', 'diagnosis', 'x', ?, 'x', ?)""",
             ("P001", "DOC_000001", now)),
            ("""INSERT INTO lab_values
                (patient_id, document_id, parameter_name, normalized_name, value)
                VALUES (?, ?, 'Hb', 'emoglobina', 10)""",
             ("P001", "DOC_000001")),
            ("""INSERT INTO clinical_state
                (patient_id, state_json, updated_at) VALUES (?, '{}', ?)""",
             ("P001", now)),
            ("""INSERT INTO clinical_state_deltas
                (patient_id, delta_json, document_id, applied_at)
                VALUES (?, '{}', ?, ?)""", ("P001", "DOC_000001", now)),
            ("""INSERT INTO clinical_state_runs
                (run_id, patient_id, model_name, prompt_version,
                 schema_version, status, source_signature, current_document_id,
                 started_at)
                VALUES ('RUN1', ?, 'model', 'prompt', '1', 'completed',
                        'signature', ?, ?)""", ("P001", "DOC_000001", now)),
            ("""INSERT INTO longitudinal_evidence
                (evidence_id, run_id, patient_id, document_id, operation,
                 category, normalized_entity, asserted_at, source_text,
                 model_name, prompt_version, schema_version, created_at)
                VALUES ('LE1', 'RUN1', ?, ?, 'add', 'diagnosis', 'x', ?, 'x',
                        'model', 'prompt', '1', ?)""",
             ("P001", "DOC_000001", now, now)),
            ("""INSERT INTO clinical_state_checkpoints
                (run_id, document_id, sequence_index, status)
                VALUES ('RUN1', ?, 0, 'completed')""", ("DOC_000001",)),
            ("""INSERT INTO validation_queue
                (patient_id, item_type, item_id, issue, created_at)
                VALUES (?, 'document', ?, 'review', ?)""",
             ("P001", "DOC_000001", now)),
            # Timeline entry referencing the skipped duplicate document, to
            # verify the JSON id rewrite on merge.
            ("""INSERT INTO clinical_timeline
                (entry_id, patient_id, date_observed, category, description,
                 source_document_ids, status, confidence, created_at, updated_at)
                VALUES ('CTL1', ?, '2024-01-01', 'other', 'desc',
                        '["DOC_000002"]', 'active', 0.5, ?, ?)""",
             ("P001", now, now)),
        ]
        with db:
            for sql, params in statements:
                db.execute(sql, params)

        deletion = PatientWorkspaceDeletionService(
            db, patient_repo, workspaces, cache
        )
        service = WorkspaceMergeService(
            db=db, patient_repo=patient_repo, document_repo=doc_repo,
            identity_repo=None, audit_repo=None,
            deletion_service=deletion, workspaces_dir=workspaces,
        )
        return {
            "db": db,
            "patient_repo": patient_repo,
            "doc_repo": doc_repo,
            "workspaces": workspaces,
            "deletion": deletion,
            "service": service,
            "p1_hash": p1_hash,
            "p2_dup_hash": p2_dup_hash,
            "p1_original": p1_original,
            "p1_dup": p1_dup,
        }

    # ------------------------------------------------------------------

    def test_basic_move_remaps_patient_and_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            fx = self._fixture(Path(tmp))
            result = fx["service"].merge("P001", "P002")

            self.assertTrue(result.ok, (result.errors, result.warnings))
            self.assertEqual(result.moved_documents, 1)      # DOC_000001
            self.assertEqual(result.already_present, 1)      # DOC_000002 dup
            self.assertGreaterEqual(result.moved_files, 2)   # original + artifact
            self.assertTrue(result.source_removed)

            # Moved document now belongs to P002 with a new path.
            doc = fx["doc_repo"].get_by_id("DOC_000001")
            self.assertEqual(doc.patient_id, "P002")
            self.assertIn("P002", doc.original_path)
            self.assertTrue(Path(doc.original_path).exists())

            # Files physically moved.
            self.assertFalse(fx["p1_original"].exists())
            self.assertTrue(
                (fx["workspaces"] / "P002" / "documents" / "original"
                 / "report.pdf").exists()
            )
            self.assertFalse(
                (fx["workspaces"] / "P001" / "extraction" / "DOC_000001.md").exists()
            )
            self.assertTrue(
                (fx["workspaces"] / "P002" / "extraction" / "DOC_000001.md").exists()
            )

            # Source patient fully removed from the registry.
            self.assertIsNone(fx["patient_repo"].get_by_id("P001"))
            self.assertIsNotNone(fx["patient_repo"].get_by_id("P002"))
            for table in fx["deletion"]._tables_with_patient_id():
                self.assertEqual(fx["db"].execute(
                    f'SELECT COUNT(*) FROM "{table}" WHERE patient_id=?',
                    ("P001",),
                ).fetchone()[0], 0, table)

            # Target keeps its own documents and gains the moved one.
            docs = fx["db"].execute(
                "SELECT id FROM documents WHERE patient_id='P002'"
            ).fetchall()
            self.assertEqual(sorted(r["id"] for r in docs),
                             ["DOC_000001", "DOC_000003", "DOC_000004"])

    def test_duplicate_document_is_skipped_and_removed_with_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            fx = self._fixture(Path(tmp))
            result = fx["service"].merge("P001", "P002")

            self.assertEqual(result.already_present, 1)
            # Exactly one document with the shared hash survives, in P002.
            rows = fx["db"].execute(
                "SELECT id FROM documents WHERE file_hash=?", (fx["p2_dup_hash"],)
            ).fetchall()
            self.assertEqual([r["id"] for r in rows], ["DOC_000004"])
            # The source copy (file + record) is gone with the workspace.
            self.assertFalse(fx["p1_dup"].exists())
            self.assertIsNone(fx["doc_repo"].get_by_id("DOC_000002"))

    def test_patient_identities_merged_when_both_present(self):
        with tempfile.TemporaryDirectory() as tmp:
            fx = self._fixture(Path(tmp))
            result = fx["service"].merge("P001", "P002")
            self.assertTrue(result.ok, result.errors)

            rows = fx["db"].execute(
                "SELECT patient_id, fiscal_code_key, normalized_name_key, "
                "birth_date_key, sex, confidence FROM patient_identities"
            ).fetchall()
            self.assertEqual(len(rows), 1)
            merged = rows[0]
            self.assertEqual(merged["patient_id"], "P002")
            # Target's own values kept (COALESCE), confidence maxed.
            self.assertEqual(merged["fiscal_code_key"], "CF2")
            self.assertEqual(merged["sex"], "F")
            self.assertEqual(merged["confidence"], 1.0)

    def test_clinical_state_conflict_keeps_target_and_warns(self):
        with tempfile.TemporaryDirectory() as tmp:
            fx = self._fixture(Path(tmp))
            fx["db"].execute(
                "INSERT INTO clinical_state (patient_id, state_json, updated_at) "
                "VALUES ('P002', '{}', '2024-01-01')"
            )
            fx["db"].commit()
            result = fx["service"].merge("P001", "P002")

            self.assertTrue(result.ok, result.errors)
            self.assertTrue(any("Clinical State" in w for w in result.warnings))
            self.assertIsNone(fx["db"].execute(
                "SELECT 1 FROM clinical_state WHERE patient_id='P001'"
            ).fetchone())
            self.assertIsNotNone(fx["db"].execute(
                "SELECT 1 FROM clinical_state WHERE patient_id='P002'"
            ).fetchone())

    def test_timeline_json_rewrites_duplicate_references(self):
        with tempfile.TemporaryDirectory() as tmp:
            fx = self._fixture(Path(tmp))
            result = fx["service"].merge("P001", "P002")
            self.assertTrue(result.ok, result.errors)

            row = fx["db"].execute(
                "SELECT source_document_ids FROM clinical_timeline "
                "WHERE entry_id='CTL1'"
            ).fetchone()
            self.assertIn("DOC_000004", row["source_document_ids"])
            self.assertNotIn("DOC_000002", row["source_document_ids"])

    def test_database_failure_restores_files_and_keeps_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            fx = self._fixture(Path(tmp))
            failing = FailingMergeService(
                db=fx["db"], patient_repo=fx["patient_repo"],
                document_repo=fx["doc_repo"], identity_repo=None,
                audit_repo=None, deletion_service=fx["deletion"],
                workspaces_dir=fx["workspaces"],
            )
            result = failing.merge("P001", "P002")

            self.assertFalse(result.ok)
            self.assertTrue(result.errors)
            self.assertFalse(result.source_removed)
            # Files restored to the source workspace.
            self.assertTrue(fx["p1_original"].exists())
            self.assertFalse(
                (fx["workspaces"] / "P002" / "documents" / "original"
                 / "report.pdf").exists()
            )
            # Source patient intact in the database.
            self.assertIsNotNone(fx["patient_repo"].get_by_id("P001"))
            self.assertEqual(fx["doc_repo"].get_by_id("DOC_000001").patient_id,
                             "P001")

    def test_find_identity_matches(self):
        with tempfile.TemporaryDirectory() as tmp:
            fx = self._fixture(Path(tmp))
            # P001 and P002 share name+birth, differ in fiscal code.
            matches = fx["service"].find_identity_matches("P001")
            self.assertEqual([m["patient_id"] for m in matches], ["P002"])
            self.assertEqual(matches[0]["reason"], "nome e data di nascita")

            # Same fiscal code -> stronger match.
            fx["db"].execute(
                "UPDATE patient_identities SET fiscal_code_key='CF1' "
                "WHERE patient_id='P002'"
            )
            fx["db"].commit()
            matches = fx["service"].find_identity_matches("P001")
            self.assertEqual(matches[0]["reason"], "codice fiscale")

    def test_validation_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            fx = self._fixture(Path(tmp))
            with self.assertRaises(ValueError):
                fx["service"].merge("P001", "P001")
            with self.assertRaises(ValueError):
                fx["service"].merge("P001", "P999")
            with self.assertRaises(ValueError):
                fx["service"].merge("P999", "P002")


if __name__ == "__main__":
    unittest.main()

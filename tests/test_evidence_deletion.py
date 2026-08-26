from __future__ import annotations

from pathlib import Path
import json
import tempfile
import unittest

from emr_analyzer.clinical.evidence_deletion import EvidenceDeletionService
from emr_analyzer.database.audit_repo import AuditRepository
from emr_analyzer.database.clinical_state_repo import ClinicalStateRepository
from emr_analyzer.database.document_repo import DocumentRepository
from emr_analyzer.database.engine import DatabaseEngine
from emr_analyzer.database.migrations import init_database
from emr_analyzer.database.patient_repo import PatientRepository
from emr_analyzer.database.registry_repo import ClinicalRegistryRepository
from emr_analyzer.database.review_repo import ReviewDecisionRepository
from emr_analyzer.models import Patient
from emr_analyzer.models.document import DocumentRecord


class FailingEvidenceDeletionService(EvidenceDeletionService):
    def _delete_database_records(self, patient_id, events, state_json,
                                 result) -> None:
        raise RuntimeError("synthetic database failure")


class EvidenceDeletionTest(unittest.TestCase):
    def _fixture(self, db):
        patient_repo = PatientRepository(db)
        document_repo = DocumentRepository(db)
        patient_repo.insert(Patient(id="P001", pseudonym="001"))
        patient_repo.insert(Patient(id="P002", pseudonym="002"))
        document_repo.insert(DocumentRecord(
            id="DOC_000001", patient_id="P001", filename="a.pdf",
            original_path="/dev/null", file_hash="h1",
        ))
        document_repo.insert(DocumentRecord(
            id="DOC_000002", patient_id="P002", filename="b.pdf",
            original_path="/dev/null", file_hash="h2",
        ))

        # Atomic evidence layer: two atoms for P001, one for P002.
        db.execute(
            """INSERT INTO clinical_evidence
               (evidence_id, patient_id, document_id, category,
                normalized_entity, source_text, extraction_method,
                schema_version, created_at)
               VALUES ('EVD_000001', 'P001', 'DOC_000001', 'symptom',
                       'dispnea', 'dispnea', 'llm_atomic_v2', '1.0',
                       '2024-01-01')"""
        )
        db.execute(
            """INSERT INTO clinical_evidence
               (evidence_id, patient_id, document_id, category,
                normalized_entity, source_text, extraction_method,
                schema_version, created_at)
               VALUES ('EVD_000002', 'P001', 'DOC_000001', 'symptom',
                       'astenia', 'astenia', 'llm_atomic_v2', '1.0',
                       '2024-01-01')"""
        )
        db.execute(
            """INSERT INTO clinical_evidence
               (evidence_id, patient_id, document_id, category,
                normalized_entity, source_text, extraction_method,
                schema_version, created_at)
               VALUES ('EVD_000003', 'P002', 'DOC_000002', 'symptom',
                       'febbre', 'febbre', 'llm_atomic_v2', '1.0',
                       '2024-01-01')"""
        )
        db.execute(
            """INSERT INTO evidence_source_refs
               (source_ref_id, evidence_id, document_id, passage, created_at)
               VALUES ('SRC_1', 'EVD_000001', 'DOC_000001', 'dispnea',
                       '2024-01-01')"""
        )
        db.execute(
            """INSERT INTO evidence_duplicate_groups
               (duplicate_group_id, patient_id, canonical_evidence_id,
                occurrence_key, created_at, updated_at)
               VALUES ('DUP_1', 'P001', 'EVD_000001', 'k', '2024-01-01',
                       '2024-01-01')"""
        )
        db.execute(
            """INSERT INTO evidence_duplicate_members
               (duplicate_group_id, evidence_id, created_at)
               VALUES ('DUP_1', 'EVD_000001', '2024-01-01')"""
        )
        db.execute(
            """INSERT INTO evidence_duplicate_members
               (duplicate_group_id, evidence_id, created_at)
               VALUES ('DUP_1', 'EVD_000002', '2024-01-01')"""
        )
        db.execute(
            """INSERT INTO evidence_relations
               (relation_id, patient_id, source_evidence_id,
                target_evidence_id, relation_type, direction, weight,
                cluster_effect, rationale, generation_method, review_status,
                created_at)
               VALUES ('REL_1', 'P001', 'EVD_000001', 'EVD_000002',
                       'related', 'directed', 1.0, '', '', 'test', 'auto',
                       '2024-01-01')"""
        )
        db.execute(
            """INSERT INTO evidence_relation_adjudication_cache
               (patient_id, namespace, candidate_id, round_index, cache_key,
                source_evidence_id, target_evidence_id, decision_json,
                created_at, updated_at)
               VALUES ('P001', 'ns', 'c1', 0, 'key1', 'EVD_000001',
                       'EVD_000002', '{}', '2024-01-01', '2024-01-01')"""
        )
        db.execute(
            """INSERT INTO clinical_aggregation_coverage
               (run_id, patient_id, evidence_id, disposition, status,
                event_ids_json, reason, created_at)
               VALUES ('R1', 'P001', 'EVD_000001', 'included', 'completed',
                       '[]', '', '2024-01-01')"""
        )
        db.execute(
            """INSERT INTO excluded_evidence
               (excluded_id, patient_id, document_id, disposition,
                reason_code, source_text, extraction_method, created_at)
               VALUES ('EXC_1', 'P001', 'DOC_000001', 'excluded',
                       'administrative', 'boilerplate', 'llm_atomic_v2',
                       '2024-01-01')"""
        )
        db.execute(
            """INSERT INTO processing_manifest
               (manifest_id, patient_id, document_id, stage, input_hash,
                pipeline_version, status, created_at, updated_at)
               VALUES ('MAN_1', 'P001', 'DOC_000001', 'atomic_evidence',
                       'h1', 'v1', 'completed', '2024-01-01', '2024-01-01')"""
        )

        # Registry layer: one event per patient, linked to evidence.
        db.execute(
            """INSERT INTO clinical_events
               (event_id, patient_id, category, canonical_entity,
                summary_short, schema_version, created_at, updated_at)
               VALUES ('EVT_000001', 'P001', 'symptom', 'dispnea', 'dispnea',
                       '1.0', '2024-01-01', '2024-01-01')"""
        )
        db.execute(
            """INSERT INTO clinical_events
               (event_id, patient_id, category, canonical_entity,
                summary_short, schema_version, created_at, updated_at)
               VALUES ('EVT_000002', 'P002', 'symptom', 'febbre', 'febbre',
                       '1.0', '2024-01-01', '2024-01-01')"""
        )
        db.execute(
            """INSERT INTO clinical_event_evidence
               (link_id, event_id, evidence_id, created_at)
               VALUES ('LINK_1', 'EVT_000001', 'EVD_000001', '2024-01-01')"""
        )
        db.execute(
            """INSERT INTO clinical_event_claims
               (claim_id, event_id, claim_type, text, created_at, updated_at)
               VALUES ('CLM_1', 'EVT_000001', 'observation', 'dispnea',
                       '2024-01-01', '2024-01-01')"""
        )
        db.execute(
            """INSERT INTO clinical_event_claim_sources
               (claim_id, source_type, source_id, source_role, created_at)
               VALUES ('CLM_1', 'evidence', 'EVD_000001', 'supports',
                       '2024-01-01')"""
        )
        db.execute(
            """INSERT INTO clinical_event_claim_sources
               (claim_id, source_type, source_id, source_role, created_at)
               VALUES ('CLM_1', 'lab_observation', '999', 'supports',
                       '2024-01-01')"""
        )
        db.execute(
            """INSERT INTO validation_queue
               (patient_id, item_type, item_id, issue, severity, status,
                created_at)
               VALUES ('P001', 'evidence', 'EVD_000001', 'review', 'medium',
                       'pending', '2024-01-01')"""
        )
        db.execute(
            """INSERT INTO validation_queue
               (patient_id, item_type, item_id, issue, severity, status,
                created_at)
               VALUES ('P001', 'atomic_duplicate_v3', 'DUP_1', 'review',
                       'medium', 'pending', '2024-01-01')"""
        )
        db.execute(
            """INSERT INTO validation_queue
               (patient_id, item_type, item_id, issue, severity, status,
                created_at)
               VALUES ('P001', 'evidence_relation_v3', 'REL_1', 'review',
                       'medium', 'pending', '2024-01-01')"""
        )
        db.execute(
            """INSERT INTO validation_queue
               (patient_id, item_type, item_id, issue, severity, status,
                created_at)
               VALUES ('P001', 'clinical_event_v3', 'EVT_000001', 'review',
                       'medium', 'pending', '2024-01-01')"""
        )
        db.execute(
            """INSERT INTO clinical_timeline
               (entry_id, patient_id, date_observed, description, created_at,
                updated_at)
               VALUES ('CTL_000001', 'P001', '2024-01-01', 'dispnea',
                       '2024-01-01', '2024-01-01')"""
        )
        db.execute(
            """INSERT INTO clinical_timeline
               (entry_id, patient_id, date_observed, description, created_at,
                updated_at)
               VALUES ('CTL_000002', 'P002', '2024-01-01', 'febbre',
                       '2024-01-01', '2024-01-01')"""
        )
        db.execute(
            """INSERT INTO clinical_state (patient_id, state_json, updated_at)
               VALUES ('P001',
                       '{"clinical_profile": "profilo P001"}', '2024-01-01')"""
        )
        db.execute(
            """INSERT INTO clinical_state (patient_id, state_json, updated_at)
               VALUES ('P002',
                       '{"clinical_profile": "profilo P002"}', '2024-01-01')"""
        )
        db.execute(
            """INSERT INTO lab_values
               (patient_id, document_id, parameter_name, normalized_name,
                value)
               VALUES ('P001', 'DOC_000001', 'Hb', 'emoglobina', 10.0)"""
        )
        db.execute(
            """INSERT INTO audit_log (patient_id, action, timestamp)
               VALUES ('P001', 'test', '2024-01-01')"""
        )
        db.commit()

    def _service(self, db):
        return EvidenceDeletionService(
            db,
            ReviewDecisionRepository(db),
            ClinicalRegistryRepository(db),
            ClinicalStateRepository(db),
            audit_repo=AuditRepository(db),
        )

    def test_evidence_deletion_resets_derived_stack_for_one_patient(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = DatabaseEngine(Path(tmp) / "registry.db")
            init_database(db)
            self._fixture(db)

            result = self._service(db).delete("P001")

            self.assertTrue(result.deleted)
            self.assertIsNone(result.error)
            self.assertEqual(result.removed_evidence_count, 2)
            self.assertEqual(result.rejected_event_count, 1)
            self.assertTrue(result.profile_cleared)
            self.assertEqual(result.timeline_entries_removed, 1)

            # Evidence layer gone for P001, untouched for P002.
            self.assertEqual(
                db.get_table_count(
                    "clinical_evidence", "patient_id='P001'"), 0)
            self.assertEqual(
                db.get_table_count(
                    "clinical_evidence", "patient_id='P002'"), 1)
            self.assertEqual(db.get_table_count("evidence_source_refs"), 0)
            self.assertEqual(db.get_table_count("evidence_duplicate_groups"), 0)
            self.assertEqual(db.get_table_count("evidence_duplicate_members"), 0)
            self.assertEqual(db.get_table_count("evidence_relations"), 0)
            self.assertEqual(
                db.get_table_count("evidence_relation_adjudication_cache"), 0)
            self.assertEqual(
                db.get_table_count("clinical_aggregation_coverage"), 0)
            self.assertEqual(db.get_table_count("clinical_event_evidence"), 0)
            self.assertEqual(
                db.get_table_count(
                    "excluded_evidence", "patient_id='P001'"), 0)
            self.assertEqual(
                db.get_table_count(
                    "processing_manifest", "patient_id='P001'"), 0)

            # Queue: evidence-scoped rows removed, event row survives.
            self.assertEqual(
                db.get_table_count(
                    "validation_queue",
                    "patient_id='P001' AND item_type IN "
                    "('evidence','atomic_duplicate_v3',"
                    "'evidence_relation_v3')"), 0)
            self.assertEqual(
                db.get_table_count(
                    "validation_queue",
                    "patient_id='P001' AND item_type='clinical_event_v3'"), 1)

            # Claim sources: evidence source removed, lab source survives.
            self.assertEqual(
                db.get_table_count(
                    "clinical_event_claim_sources",
                    "source_type='evidence'"), 0)
            self.assertEqual(
                db.get_table_count(
                    "clinical_event_claim_sources",
                    "source_type='lab_observation'"), 1)

            # Events rejected for P001 only, with a review decision.
            row = db.execute(
                "SELECT review_status FROM clinical_events "
                "WHERE event_id='EVT_000001'").fetchone()
            self.assertEqual(row["review_status"], "rejected")
            row = db.execute(
                "SELECT review_status FROM clinical_events "
                "WHERE event_id='EVT_000002'").fetchone()
            self.assertEqual(row["review_status"], "auto")
            self.assertEqual(
                db.get_table_count(
                    "review_decisions",
                    "target_id='EVT_000001' AND decision='rejected'"), 1)

            # Timeline and narrative reset for P001, intact for P002.
            self.assertEqual(
                db.get_table_count(
                    "clinical_timeline", "patient_id='P001'"), 0)
            self.assertEqual(
                db.get_table_count(
                    "clinical_timeline", "patient_id='P002'"), 1)
            row = db.execute(
                "SELECT state_json FROM clinical_state "
                "WHERE patient_id='P001'").fetchone()
            self.assertEqual(
                json.loads(row["state_json"])["clinical_profile"], "")
            row = db.execute(
                "SELECT state_json FROM clinical_state "
                "WHERE patient_id='P002'").fetchone()
            self.assertEqual(
                json.loads(row["state_json"])["clinical_profile"],
                "profilo P002")

            # Documents and structured labs are never touched.
            self.assertEqual(db.get_table_count("documents"), 2)
            self.assertEqual(db.get_table_count("lab_values"), 1)

            # One audit entry for the reset.
            self.assertEqual(
                db.get_table_count(
                    "audit_log",
                    "action='delete' AND target_type='evidence'"), 1)

    def test_gold_set_rows_are_deleted_with_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = DatabaseEngine(Path(tmp) / "registry.db")
            init_database(db)
            self._fixture(db)
            db.execute(
                """INSERT INTO gold_set_cases (patient_id, created_at,
                                               updated_at)
                   VALUES ('P001', '2024-01-01', '2024-01-01')"""
            )
            db.execute(
                """INSERT INTO gold_set_cases (patient_id, created_at,
                                               updated_at)
                   VALUES ('P002', '2024-01-01', '2024-01-01')"""
            )
            db.execute(
                """INSERT INTO gold_annotations
                   (annotation_id, patient_id, reviewer_slot, reviewer_id,
                    category, canonical_entity, summary_short, created_at,
                    updated_at)
                   VALUES ('ANN_1', 'P001', 'a', 'rev', 'symptom', 'dispnea',
                           'dispnea', '2024-01-01', '2024-01-01')"""
            )
            db.execute(
                """INSERT INTO gold_annotation_sources
                   (source_id, annotation_id, patient_id, evidence_id,
                    document_id, source_text, created_at)
                   VALUES ('SRC_G1', 'ANN_1', 'P001', 'EVD_000001',
                           'DOC_000001', 'dispnea', '2024-01-01')"""
            )
            db.execute(
                """INSERT INTO gold_atomic_annotations
                   (annotation_id, patient_id, document_id, reviewer_slot,
                    reviewer_id, source_text, created_at, updated_at)
                   VALUES ('ATOM_1', 'P001', 'DOC_000001', 'a', 'rev',
                           'dispnea', '2024-01-01', '2024-01-01')"""
            )
            db.execute(
                """INSERT INTO gold_adjudication_decisions
                   (decision_id, patient_id, source_annotation_id, decision,
                    adjudicator_id, created_at, updated_at)
                   VALUES ('DEC_1', 'P001', 'ANN_1', 'accept', 'adj',
                           '2024-01-01', '2024-01-01')"""
            )
            db.commit()

            result = self._service(db).delete("P001")

            self.assertTrue(result.deleted)
            self.assertIsNone(result.error)
            self.assertEqual(result.gold_rows_removed, 5)
            self.assertEqual(
                db.get_table_count(
                    "gold_set_cases", "patient_id='P001'"), 0)
            self.assertEqual(
                db.get_table_count(
                    "gold_set_cases", "patient_id='P002'"), 1)
            self.assertEqual(db.get_table_count("gold_annotations"), 0)
            self.assertEqual(db.get_table_count("gold_annotation_sources"), 0)
            self.assertEqual(db.get_table_count("gold_atomic_annotations"), 0)
            self.assertEqual(
                db.get_table_count("gold_adjudication_decisions"), 0)
            self.assertEqual(
                db.get_table_count(
                    "clinical_evidence", "patient_id='P001'"), 0)

    def test_database_failure_rolls_back_everything(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = DatabaseEngine(Path(tmp) / "registry.db")
            init_database(db)
            self._fixture(db)

            result = FailingEvidenceDeletionService(
                db,
                ReviewDecisionRepository(db),
                ClinicalRegistryRepository(db),
                ClinicalStateRepository(db),
                audit_repo=AuditRepository(db),
            ).delete("P001")

            self.assertFalse(result.deleted)
            self.assertIn("synthetic database failure", result.error)
            self.assertEqual(
                db.get_table_count(
                    "clinical_evidence", "patient_id='P001'"), 2)
            row = db.execute(
                "SELECT review_status FROM clinical_events "
                "WHERE event_id='EVT_000001'").fetchone()
            self.assertEqual(row["review_status"], "auto")
            self.assertEqual(
                db.get_table_count(
                    "clinical_timeline", "patient_id='P001'"), 1)
            self.assertEqual(
                db.get_table_count(
                    "validation_queue",
                    "patient_id='P001' AND item_type='evidence'"), 1)
            self.assertEqual(
                db.get_table_count("audit_log", "action='delete'"), 0)


if __name__ == "__main__":
    unittest.main()

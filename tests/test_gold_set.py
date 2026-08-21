from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from emr_analyzer.clinical.consolidation import ClinicalConsolidator
from emr_analyzer.database.audit_repo import AuditRepository
from emr_analyzer.database.document_repo import DocumentRepository
from emr_analyzer.database.engine import DatabaseEngine
from emr_analyzer.database.evidence_repo import EvidenceRepository
from emr_analyzer.database.gold_set_repo import (
    GoldSetRepository,
    manual_evidence_id,
)
from emr_analyzer.database.migrations import init_database
from emr_analyzer.database.registry_repo import ClinicalRegistryRepository
from emr_analyzer.gui.gold_set_tab import GoldSetTab
from emr_analyzer.models.clinical_evidence import ClinicalEvidence
from emr_analyzer.models.gold_set import GoldAnnotation


class GoldSetWorkflowTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = DatabaseEngine(Path(self.tmp.name) / "gold.sqlite")
        init_database(self.db)
        self.db.execute(
            """INSERT INTO patients (id, pseudonym, created_at, updated_at)
               VALUES ('P001', 'P001', '2026-01-01', '2026-01-01')"""
        )
        self.db.execute(
            """INSERT INTO documents
               (id, patient_id, filename, original_path, file_hash,
                document_date, document_type, import_date)
               VALUES ('D1', 'P001', 'D1.pdf', '/D1.pdf', 'hash-D1',
                       '2025-01-10', 'radiologia', '2026-01-01')"""
        )
        self.db.commit()

        self.registry = ClinicalRegistryRepository(self.db)
        self.audit = AuditRepository(self.db)
        self.repo = GoldSetRepository(
            self.db, registry_repo=self.registry, audit_repo=self.audit
        )
        evidence = ClinicalEvidence(
            evidence_id="EVD_MATCH", patient_id="P001", document_id="D1",
            category="imaging_finding", normalized_entity="nodulo_polmonare",
            source_text="Nodulo polmonare di 8 mm", source_page=3,
            observed_date="2025-01-10", document_date="2025-01-10",
        )
        EvidenceRepository(self.db).insert_batch([evidence])
        bundle = ClinicalConsolidator().consolidate("P001", [evidence])[0]
        self.registry.save_episode(bundle.episode)
        self.registry.save_event(bundle.event, bundle.links, bundle.updates)

    def tearDown(self):
        self.db.close()
        self.tmp.cleanup()

    def _annotation(self, slot: str, reviewer: str) -> GoldAnnotation:
        quote = "Nodulo polmonare di 8 mm"
        evidence_id = manual_evidence_id(
            "D1", 3, quote, category="imaging_finding",
            entity="nodulo_polmonare", date="2025-01-10",
        )
        return GoldAnnotation(
            patient_id="P001", reviewer_slot=slot, reviewer_id=reviewer,
            category="imaging_finding", canonical_entity="nodulo_polmonare",
            summary_short="Nodulo polmonare di 8 mm",
            first_evidence_date="2025-01-10", date_precision="day",
            evidence_ids=[evidence_id], source_refs=[{
                "evidence_id": evidence_id, "document_id": "D1", "page": 3,
                "quote": quote, "relation": "supports",
            }],
        )

    def test_blinded_submission_adjudication_lock_and_export(self):
        case = self.repo.configure_case(
            "P001", included=True, split="test",
            reviewer_a_id="med_a", reviewer_b_id="med_b",
            adjudicator_id="med_c",
        )
        self.assertEqual(case.status, "annotating")
        annotation_a = self.repo.save_annotation(
            self._annotation("reviewer_a", "med_a")
        )
        source_row = self.db.execute(
            """SELECT document_id, source_page, source_text
               FROM gold_annotation_sources WHERE annotation_id=?""",
            (annotation_a.annotation_id,),
        ).fetchone()
        self.assertEqual(dict(source_row), {
            "document_id": "D1", "source_page": 3,
            "source_text": "Nodulo polmonare di 8 mm",
        })
        annotation_b = self.repo.save_annotation(
            self._annotation("reviewer_b", "med_b")
        )
        self.assertEqual(
            [item.annotation_id for item in self.repo.list_annotations(
                "P001", "reviewer_a"
            )],
            [annotation_a.annotation_id],
        )
        self.repo.submit_reviewer("P001", "reviewer_a", "med_a")
        with self.assertRaisesRegex(ValueError, "già consegnata"):
            self.repo.save_annotation(annotation_a)
        case = self.repo.submit_reviewer("P001", "reviewer_b", "med_b")
        self.assertTrue(self.repo.can_adjudicate(case))

        final = self.repo.copy_to_adjudicated(
            annotation_a.annotation_id, "med_c",
            related_annotation_ids=[annotation_b.annotation_id],
        )
        self.assertEqual(
            set(final.source_annotation_ids),
            {annotation_a.annotation_id, annotation_b.annotation_id},
        )
        locked = self.repo.lock_case("P001", "med_c")
        self.assertEqual(locked.status, "locked")
        with self.assertRaisesRegex(ValueError, "bloccato"):
            self.repo.delete_annotation(final.annotation_id)

        record = self.repo.evaluation_record("P001")
        self.assertEqual(record["gold_events"][0]["evidence_ids"], ["EVD_MATCH"])
        metrics = self.repo.evaluate_patient("P001")
        self.assertEqual(metrics["f1"], 1.0)
        self.assertEqual(metrics["citation_completeness"], 1.0)

        destination = Path(self.tmp.name) / "gold.jsonl"
        self.assertEqual(self.repo.export_jsonl(destination), 1)
        exported = json.loads(destination.read_text(encoding="utf-8"))
        self.assertEqual(exported["patient_id"], "P001")
        self.assertEqual(exported["gold_metadata"]["status"], "locked")
        self.assertTrue(self.audit.verify_chain("P001")[0])

    def test_lock_requires_a_decision_for_every_source_annotation(self):
        self.repo.configure_case(
            "P001", included=True, split="pilot",
            reviewer_a_id="a", reviewer_b_id="b", adjudicator_id="c",
        )
        annotation_a = self.repo.save_annotation(
            self._annotation("reviewer_a", "a")
        )
        annotation_b = self.repo.save_annotation(
            self._annotation("reviewer_b", "b")
        )
        self.repo.submit_reviewer("P001", "reviewer_a", "a")
        self.repo.submit_reviewer("P001", "reviewer_b", "b")
        self.repo.copy_to_adjudicated(annotation_a.annotation_id, "c")
        with self.assertRaisesRegex(ValueError, "Adjudication incompleta"):
            self.repo.lock_case("P001", "c")
        self.repo.exclude_from_gold(
            "P001", [annotation_b.annotation_id], "c",
            "Duplicato non confermato nella lettura finale",
        )
        self.assertEqual(self.repo.lock_case("P001", "c").status, "locked")

    def test_gui_keeps_adjudication_hidden_before_both_submissions(self):
        tab = GoldSetTab()
        tab.set_services({
            "gold_set_repo": self.repo,
            "document_repo": DocumentRepository(self.db),
        })
        tab.load_patient("P001")
        self.assertFalse(tab._workflow_tabs.isTabEnabled(1))
        tab.deleteLater()

    def test_repository_rejects_unassigned_and_cross_patient_sources(self):
        self.repo.configure_case(
            "P001", included=True, split="pilot",
            reviewer_a_id="a", reviewer_b_id="b", adjudicator_id="c",
        )
        wrong_reviewer = self._annotation("reviewer_a", "intruso")
        with self.assertRaisesRegex(ValueError, "assegnazione"):
            self.repo.save_annotation(wrong_reviewer)

        self.db.execute(
            """INSERT INTO patients (id, pseudonym, created_at, updated_at)
               VALUES ('P002', 'P002', '2026-01-01', '2026-01-01')"""
        )
        self.db.execute(
            """INSERT INTO documents
               (id, patient_id, filename, original_path, file_hash,
                document_type, import_date)
               VALUES ('D2', 'P002', 'D2.pdf', '/D2.pdf', 'hash-D2',
                       'altro', '2026-01-01')"""
        )
        self.db.commit()
        cross_patient = self._annotation("reviewer_a", "a")
        cross_patient.source_refs[0]["document_id"] = "D2"
        with self.assertRaisesRegex(ValueError, "altro paziente"):
            self.repo.save_annotation(cross_patient)


if __name__ == "__main__":
    unittest.main()

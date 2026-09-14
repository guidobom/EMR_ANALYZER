"""Embedding-based near-miss duplicate candidates for the review queue."""

import json
import tempfile
from pathlib import Path
import unittest

import numpy as np

from emr_analyzer.clinical.embedding_candidates import (
    atom_signature,
    embedding_duplicate_pairs,
)
from emr_analyzer.database.engine import DatabaseEngine
from emr_analyzer.database.evidence_repo import EvidenceRepository
from emr_analyzer.database.migrations import init_database
from emr_analyzer.database.pipeline_repo import ClinicalPipelineRepository
from emr_analyzer.models.clinical_evidence import ClinicalEvidence


def _atom(evidence_id, category, entity, *, document_id="D1",
          date=None, assertion="present", site=None, severity=None,
          value_text=None, numeric_value=None, unit=None):
    return ClinicalEvidence(
        evidence_id=evidence_id, patient_id="P1", document_id=document_id,
        category=category, normalized_entity=entity, source_text=entity,
        observed_date=date, assertion=assertion, certainty="confirmed",
        anatomical_site=site, severity=severity, value_text=value_text,
        numeric_value=numeric_value, unit=unit,
    )


class _FakeIndex:
    model_name = "fake-model"

    def __init__(self, vectors, *, fail_load=False):
        self._vectors = vectors
        self._fail_load = fail_load
        self.encoded = []

    def loaded(self):
        return None if self._fail_load else object()

    def encode(self, signatures):
        self.encoded.extend(signatures)
        return np.asarray(
            [self._vectors[signature] for signature in signatures],
            dtype=np.float32,
        )


class _FakeRepo:
    def __init__(self, cached=None):
        self.cached = cached or {}
        self.put_rows = []

    def get_embeddings(self, evidence_ids, *, model_name):
        return {
            evidence_id: blob
            for evidence_id, blob in self.cached.items()
            if evidence_id in evidence_ids
        }

    def put_embeddings(self, rows):
        self.put_rows.extend(rows)


class EmbeddingCandidateTest(unittest.TestCase):
    def test_atom_signature_uses_structured_identity(self):
        atom = _atom(
            "E1", "imaging_finding", "tc total body mdc",
            site="torace", severity="lieve",
        )
        signature = atom_signature(atom)
        self.assertIn("tc total body", signature)
        self.assertIn("sede torace", signature)
        self.assertIn("gravità lieve", signature)

    def test_near_paraphrases_above_threshold_are_proposed(self):
        left = _atom("E1", "imaging_finding", "nodulo al polmone",
                     date="2025-01-10", site="polmone")
        right = _atom("E2", "imaging_finding", "nodulo polmonare",
                      document_id="D2", date="2025-01-10", site="polmone")
        same = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        index = _FakeIndex({
            atom_signature(left): same,
            atom_signature(right): same,
        })
        pairs = embedding_duplicate_pairs(
            [left, right], threshold=0.86, index=index,
        )
        self.assertEqual(len(pairs), 1)
        left_id, right_id, similarity, source = pairs[0]
        self.assertEqual({left_id, right_id}, {"E1", "E2"})
        self.assertAlmostEqual(similarity, 1.0, places=3)
        self.assertEqual(source, "embedding")

    def test_below_threshold_not_proposed(self):
        left = _atom("E1", "symptom", "dispnea")
        right = _atom("E2", "symptom", "astenia", document_id="D2")
        index = _FakeIndex({
            atom_signature(left): np.array([1.0, 0.0], dtype=np.float32),
            atom_signature(right): np.array([0.0, 1.0], dtype=np.float32),
        })
        pairs = embedding_duplicate_pairs(
            [left, right], threshold=0.86, index=index,
        )
        self.assertEqual(pairs, [])

    def test_true_followups_on_different_dates_are_never_proposed(self):
        left = _atom("E1", "laboratory_finding", "creatinina",
                     date="2025-01-10", numeric_value=1.1, unit="mg/dl")
        right = _atom("E2", "laboratory_finding", "creatinina",
                      document_id="D2", date="2025-02-10",
                      numeric_value=1.1, unit="mg/dl")
        same = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        index = _FakeIndex({
            atom_signature(left): same,
            atom_signature(right): same,
        })
        pairs = embedding_duplicate_pairs(
            [left, right], threshold=0.86, index=index,
        )
        self.assertEqual(pairs, [])

    def test_different_assertions_are_never_merged(self):
        left = _atom("E1", "symptom", "febbre", assertion="present")
        right = _atom("E2", "symptom", "febbre", document_id="D2",
                      assertion="absent")
        same = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        index = _FakeIndex({
            atom_signature(left): same,
            atom_signature(right): same,
        })
        pairs = embedding_duplicate_pairs(
            [left, right], threshold=0.86, index=index,
        )
        self.assertEqual(pairs, [])

    def test_identity_equivalent_pairs_already_merged_are_excluded(self):
        left = _atom("E1", "diagnosis", "melanoma metastatico",
                     date="2025-01-10")
        right = _atom("E2", "diagnosis", "Melanoma Metastatico",
                      document_id="D2", date="2025-01-10")
        same = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        index = _FakeIndex({
            atom_signature(left): same,
            atom_signature(right): same,
        })
        pairs = embedding_duplicate_pairs(
            [left, right], threshold=0.86, index=index,
        )
        self.assertEqual(pairs, [])

    def test_unavailable_model_degrades_gracefully(self):
        left = _atom("E1", "symptom", "dispnea")
        right = _atom("E2", "symptom", "astenia", document_id="D2")
        index = _FakeIndex({}, fail_load=True)
        pairs = embedding_duplicate_pairs(
            [left, right], threshold=0.86, index=index,
        )
        self.assertEqual(pairs, [])

    def test_cached_vectors_avoid_reencoding(self):
        left = _atom("E1", "symptom", "dispnea")
        right = _atom("E2", "symptom", "astenia", document_id="D2")
        close = np.array([0.8, 0.6], dtype=np.float32)
        close = close / np.linalg.norm(close)
        far = np.array([0.2, -0.98], dtype=np.float32)
        far = far / np.linalg.norm(far)
        repo = _FakeRepo(cached={
            "E1": close.astype(np.float32).tobytes(),
            "E2": far.astype(np.float32).tobytes(),
        })

        class _ExplodingIndex(_FakeIndex):
            def encode(self, signatures):
                raise AssertionError("encode non deve essere chiamato")

        pairs = embedding_duplicate_pairs(
            [left, right], threshold=0.86, repo=repo,
            index=_ExplodingIndex({}),
        )
        # dot(close, far) is below threshold: cached vectors drive the result.
        self.assertEqual(pairs, [])

    def test_missing_vectors_are_computed_and_persisted(self):
        left = _atom("E1", "symptom", "dispnea")
        right = _atom("E2", "symptom", "astenia", document_id="D2")
        same = np.array([1.0, 0.0], dtype=np.float32)
        repo = _FakeRepo()
        index = _FakeIndex({
            atom_signature(left): same,
            atom_signature(right): same,
        })
        pairs = embedding_duplicate_pairs(
            [left, right], threshold=0.86, repo=repo, index=index,
        )
        self.assertEqual(len(pairs), 1)
        self.assertEqual(len(repo.put_rows), 2)
        self.assertEqual(
            {row[0] for row in repo.put_rows}, {"E1", "E2"},
        )
        self.assertEqual({row[1] for row in repo.put_rows}, {"fake-model"})


class EmbeddingRepositoryTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = DatabaseEngine(Path(self.tmp.name) / "emr.sqlite")
        init_database(self.db)
        self.db.execute(
            """INSERT INTO patients
               (id, pseudonym, created_at, updated_at)
               VALUES ('P1', 'PAZIENTE_1', '2026-01-01', '2026-01-01')"""
        )
        self.db.execute(
            """INSERT INTO documents
               (id, patient_id, filename, original_path, file_hash,
                document_date, document_type, import_date)
               VALUES ('D1', 'P1', 'd1.pdf', '/d1.pdf', 'h1',
                       '2025-01-10', 'referto', '2026-01-01')"""
        )
        self.db.execute(
            """INSERT INTO documents
               (id, patient_id, filename, original_path, file_hash,
                document_date, document_type, import_date)
               VALUES ('D2', 'P1', 'd2.pdf', '/d2.pdf', 'h2',
                       '2025-01-12', 'referto', '2026-01-01')"""
        )
        self.db.commit()
        EvidenceRepository(self.db).insert_batch([
            _atom("E1", "symptom", "dispnea"),
            _atom("E2", "symptom", "astenia", document_id="D2"),
        ])

    def tearDown(self):
        self.tmp.cleanup()

    def test_embedding_pairs_persist_with_embedding_reason(self):
        pipeline = ClinicalPipelineRepository(self.db)
        pending = pipeline.replace_duplicate_groups(
            "P1", [], [("E1", "E2", 0.93, "embedding")],
        )
        row = self.db.execute(
            """SELECT decision_reason, review_status
               FROM evidence_duplicate_groups WHERE patient_id='P1'"""
        ).fetchone()
        self.assertEqual(
            row["decision_reason"], "embedding_near_copy_similarity=0.930"
        )
        self.assertEqual(row["review_status"], "pending")
        self.assertEqual(len(pending), 1)

    def test_text_pairs_keep_plain_reason(self):
        pipeline = ClinicalPipelineRepository(self.db)
        pipeline.replace_duplicate_groups("P1", [], [("E1", "E2", 0.80)])
        row = self.db.execute(
            """SELECT decision_reason FROM evidence_duplicate_groups
               WHERE patient_id='P1'"""
        ).fetchone()
        self.assertEqual(row["decision_reason"], "near_copy_similarity=0.800")

    def test_embedding_cache_roundtrip(self):
        pipeline = ClinicalPipelineRepository(self.db)
        blob = np.array([0.5, -0.25, 0.75], dtype=np.float32).tobytes()
        pipeline.put_embeddings([("E1", "fake-model", blob)])
        cached = pipeline.get_embeddings(
            ["E1", "E2"], model_name="fake-model"
        )
        self.assertEqual(set(cached), {"E1"})
        self.assertEqual(
            np.frombuffer(cached["E1"], dtype=np.float32).tolist(),
            [0.5, -0.25, 0.75],
        )
        other = pipeline.get_embeddings(
            ["E1", "E2"], model_name="altro-modello"
        )
        self.assertEqual(other, {})


if __name__ == "__main__":
    unittest.main()

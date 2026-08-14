"""Tests for multiple hospital patient IDs per patient.

A person legitimately holds several hospital patient IDs across the units of
a hospital trust (one per referral path).  The registry keeps a set per
patient (``patient_hospital_ids``) rather than the single legacy column, so a
report carrying any of the patient's own IDs must match cleanly, while a
genuinely foreign ID must still raise a conflict.
"""

from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

from emr_analyzer.clinical.workspace_merge import WorkspaceMergeService
from emr_analyzer.database.engine import DatabaseEngine
from emr_analyzer.database.migrations import init_database
from emr_analyzer.database.patient_identity_repo import (
    IdentityKeyService, PatientIdentityRepository,
)
from emr_analyzer.database.patient_repo import PatientRepository
from emr_analyzer.models import Patient
from emr_analyzer.models.patient_identity import IdentityField, PatientIdentityEvidence
from emr_analyzer.pipeline.import_staging import StagedDocument
from emr_analyzer.pipeline.patient_routing import PatientRoutingService


class _StubKeyService:
    """Deterministic key service: no local key file is ever touched."""

    def digest(self, field_name, normalized_value):
        return f"{field_name}:{normalized_value}"


def _evidence(name=None, birth=None, hpid=None):
    fields = {}
    if name:
        fields["name"] = IdentityField(name, name, confidence=0.97)
    if birth:
        fields["birth_date"] = IdentityField(birth, birth, confidence=0.97)
    if hpid:
        fields["hospital_patient_id"] = IdentityField(hpid, hpid, confidence=0.97)
    return PatientIdentityEvidence(source_path="/tmp/doc.pdf", **fields)


def _staged(evidence, index):
    return StagedDocument(
        original_path=f"/tmp/doc{index}.pdf",
        staged_path=f"/tmp/staged{index}.pdf",
        original_name=f"doc{index}.pdf",
        file_hash=f"hash{index}",
        check={},
        evidence=evidence,
    )


class MultiHospitalIdTest(unittest.TestCase):
    def setUp(self):
        self._tmp = Path(tempfile.mkdtemp(prefix="emr_multihpid_"))
        self._db = DatabaseEngine(self._tmp / "registry.db")
        init_database(self._db)
        self._patient_repo = PatientRepository(self._db)
        self._patient_repo.insert(Patient(id="P001", pseudonym="001"))
        self._repo = PatientIdentityRepository(
            self._db, key_service=_StubKeyService()
        )

    def tearDown(self):
        shutil.rmtree(self._tmp, ignore_errors=True)

    def _register_patient_with_hpid(self, hpid):
        self._repo.upsert(
            "P001",
            _evidence(name="GUERRA VILLIAM SILVESTRO",
                      birth="1950-12-31", hpid=hpid),
            status="auto",
        )

    def test_upsert_registers_hpid_in_multi_table(self):
        self._register_patient_with_hpid("8100455504")
        rows = self._db.execute(
            "SELECT hospital_patient_id_key FROM patient_hospital_ids "
            "WHERE patient_id='P001'"
        ).fetchall()
        self.assertEqual([r["hospital_patient_id_key"] for r in rows],
                         ["hospital_patient_id:8100455504"])

    def test_second_hpid_matches_without_conflict(self):
        # The real-world case behind the duplicate workspaces: the same person
        # carries different hospital IDs on different reports.  Registering a
        # second ID must not turn the first into a conflict.
        self._register_patient_with_hpid("8100455504")
        self._repo.add_hospital_patient_id("P001", "FE204467")

        for hpid in ("8100455504", "FE204467"):
            match = self._repo.find_match(
                _evidence(name="GUERRA VILLIAM SILVESTRO",
                          birth="1950-12-31", hpid=hpid)
            )
            self.assertEqual(match.patient_id, "P001")
            self.assertFalse(match.conflict, hpid)

    def test_hpid_alone_matches(self):
        self._register_patient_with_hpid("8100455504")
        self._repo.add_hospital_patient_id("P001", "FE204467")
        match = self._repo.find_match(_evidence(hpid="FE204467"))
        self.assertEqual(match.patient_id, "P001")
        self.assertFalse(match.conflict)
        self.assertIn("ospedaliero", match.reason)

    def test_unregistered_hpid_conflicts(self):
        self._register_patient_with_hpid("8100455504")
        match = self._repo.find_match(
            _evidence(name="GUERRA VILLIAM SILVESTRO",
                      birth="1950-12-31", hpid="9999999999")
        )
        self.assertEqual(match.patient_id, "P001")
        self.assertTrue(match.conflict)
        self.assertIn("hospital_patient_id", match.reason)

    def test_no_hpid_incoming_still_matches_by_name_birth(self):
        self._register_patient_with_hpid("8100455504")
        match = self._repo.find_match(
            _evidence(name="GUERRA VILLIAM SILVESTRO", birth="1950-12-31")
        )
        self.assertEqual(match.patient_id, "P001")
        self.assertFalse(match.conflict)

    def test_legacy_single_column_hpid_still_matches(self):
        # A database that predates the multi-valued table keeps the ID in the
        # column only: matching and conflict must behave identically.
        self._db.execute(
            """INSERT INTO patient_identities
               (patient_id, normalized_name_key, birth_date_key,
                hospital_patient_id_key, confidence, status, created_at, updated_at)
               VALUES ('P001', 'name:GUERRA SILVESTRO VILLIAM',
                       'birth_date:1950-12-31',
                       'hospital_patient_id:8100455504', 1.0, 'auto',
                       '2024-01-01', '2024-01-01')"""
        )
        match = self._repo.find_match(
            _evidence(name="GUERRA VILLIAM SILVESTRO",
                      birth="1950-12-31", hpid="8100455504")
        )
        self.assertEqual(match.patient_id, "P001")
        self.assertFalse(match.conflict)
        foreign = self._repo.find_match(
            _evidence(name="GUERRA VILLIAM SILVESTRO",
                      birth="1950-12-31", hpid="FE204467")
        )
        self.assertTrue(foreign.conflict)
        self.assertIn("hospital_patient_id", foreign.reason)

    def test_no_raw_values_persisted(self):
        # With the real HMAC key service (fresh key in a temp dir), only the
        # digests are persisted — the raw hospital IDs never appear.
        repo = PatientIdentityRepository(
            self._db, key_service=IdentityKeyService(self._tmp / "identity.key")
        )
        repo.upsert(
            "P001", _evidence(name="GUERRA VILLIAM SILVESTRO",
                              birth="1950-12-31", hpid="8100455504")
        )
        repo.add_hospital_patient_id("P001", "FE204467")
        serialized = " ".join(
            str(tuple(row))
            for row in self._db.execute(
                "SELECT * FROM patient_hospital_ids"
            ).fetchall()
        )
        self.assertNotIn("8100455504", serialized)
        self.assertNotIn("FE204467", serialized)
        self.assertNotIn("GUERRA VILLIAM SILVESTRO", serialized)

    def test_routing_resolves_existing_patient_via_second_hpid(self):
        self._register_patient_with_hpid("8100455504")
        self._repo.add_hospital_patient_id("P001", "FE204467")
        router = PatientRoutingService(
            self._repo, None, self._patient_repo, None, audit_repo=None
        )
        groups = router.resolve([
            _staged(_evidence(name="GUERRA VILLIAM SILVESTRO",
                              birth="1950-12-31", hpid="FE204467"), 0),
        ])
        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0].patient_id, "P001")
        self.assertFalse(groups[0].create_new)
        self.assertFalse(groups[0].conflict)

    def test_workspace_merge_folds_hospital_ids(self):
        self._patient_repo.insert(Patient(id="P002", pseudonym="002"))
        now = "2024-01-01T00:00:00"
        for pid in ("P001", "P002"):
            self._db.execute(
                """INSERT INTO patient_identities
                   (patient_id, normalized_name_key, birth_date_key,
                    confidence, status, created_at, updated_at)
                   VALUES (?, 'N', 'B', 1.0, 'auto', ?, ?)""",
                (pid, now, now),
            )
        for hpid in ("8100455504", "FE204467"):
            self._db.execute(
                """INSERT INTO patient_hospital_ids
                   (patient_id, hospital_patient_id_key, created_at, updated_at)
                   VALUES ('P001', ?, ?, ?)""",
                (f"hospital_patient_id:{hpid}", now, now),
            )
        merger = WorkspaceMergeService(
            self._db, None, None, None, None, None
        )
        merger._merge_identity_rows("P001", "P002")
        rows = self._db.execute(
            "SELECT hospital_patient_id_key FROM patient_hospital_ids "
            "WHERE patient_id='P002'"
        ).fetchall()
        self.assertEqual(
            sorted(r["hospital_patient_id_key"] for r in rows),
            ["hospital_patient_id:8100455504", "hospital_patient_id:FE204467"],
        )
        leftover = self._db.execute(
            "SELECT 1 FROM patient_hospital_ids WHERE patient_id='P001'"
        ).fetchone()
        self.assertIsNone(leftover)


if __name__ == "__main__":
    unittest.main()

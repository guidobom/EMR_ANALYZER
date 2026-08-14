"""Tests for name-reconciliation in the routing conflict check.

The same person's name is often extracted differently across layouts
(``VITALI REMO`` / ``RA VITALI REMO`` / ``VITALI REMO COPPARO``).  Such
variants must not send a whole group to "Da assegnare": a group anchored by
a single fiscal code or hospital patient ID is the same person, so name
discordance there is layout noise.  Genuine contradictions (different birth
dates, different fiscal codes) still block.
"""

from __future__ import annotations

import unittest

from emr_analyzer.models.patient_identity import IdentityField, PatientIdentityEvidence
from emr_analyzer.pipeline.import_staging import StagedDocument
from emr_analyzer.pipeline.patient_routing import PatientRoutingService

VALID_CF = "RSSMRA85M01H501Q"


def _evidence(name=None, birth=None, cf=None, hpid=None, sex=None):
    fields = {}
    if name:
        fields["name"] = IdentityField(name, name, confidence=0.97)
    if birth:
        fields["birth_date"] = IdentityField(birth, birth, confidence=0.97)
    if cf:
        fields["fiscal_code"] = IdentityField(cf, cf, confidence=0.97)
    if hpid:
        fields["hospital_patient_id"] = IdentityField(hpid, hpid, confidence=0.97)
    if sex:
        fields["sex"] = IdentityField(sex, sex, confidence=0.97)
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


class CleanNameTest(unittest.TestCase):
    def test_strips_ra_prefix(self):
        self.assertEqual(
            PatientRoutingService._clean_name("RA VITALI REMO"), "VITALI REMO"
        )

    def test_strips_address_and_cf_tokens(self):
        self.assertEqual(
            PatientRoutingService._clean_name("VITALI REMO VIA ACQUEDOTTO"),
            "VITALI REMO ACQUEDOTTO",
        )
        self.assertEqual(
            PatientRoutingService._clean_name("ADIL GUENNANE CF"),
            "ADIL GUENNANE",
        )

    def test_plain_name_unchanged(self):
        self.assertEqual(
            PatientRoutingService._clean_name("REMO VITALI"), "REMO VITALI"
        )


class ConflictingFieldsTest(unittest.TestCase):
    def test_ra_variant_in_anchored_group_is_not_a_conflict(self):
        # "RA VITALI REMO" cleans to "VITALI REMO": same person, no conflict.
        documents = [
            _staged(_evidence(name="VITALI REMO", birth="1942-05-15", cf=VALID_CF), 0),
            _staged(_evidence(name="RA VITALI REMO", birth="1942-05-15", cf=VALID_CF), 1),
        ]
        self.assertEqual(
            PatientRoutingService._conflicting_fields(documents), []
        )

    def test_disjoint_name_without_anchor_is_still_a_conflict(self):
        # A name_birth group (no CF/hpid) with two different people's names
        # must still block: the name is the only anchor there.
        documents = [
            _staged(_evidence(name="MASIERI MARIO", birth="1936-02-05"), 0),
            _staged(_evidence(name="MONTANARI MARCELLO", birth="1936-02-05"), 1),
        ]
        self.assertEqual(
            PatientRoutingService._conflicting_fields(documents), ["name"]
        )

    def test_disjoint_name_in_cf_anchored_group_is_tolerated(self):
        # One document of a CF group grabs another person's name (an anchor
        # error).  The CF anchors the identity, so the group is not blocked.
        documents = [
            _staged(_evidence(name="VITALI REMO", birth="1942-05-15", cf=VALID_CF), 0),
            _staged(_evidence(name="RA PATRIZIA TROMBINI", birth="1942-05-15",
                              cf=VALID_CF), 1),
        ]
        self.assertEqual(
            PatientRoutingService._conflicting_fields(documents), []
        )

    def test_hpid_anchor_tolerates_name_discordance(self):
        documents = [
            _staged(_evidence(name="MONTANARI MARCELLO", birth="1960-01-01",
                              hpid="FE61321"), 0),
            _staged(_evidence(name="TIPO DOCUMENTO", birth="1960-01-01",
                              hpid="FE61321"), 1),
        ]
        self.assertEqual(
            PatientRoutingService._conflicting_fields(documents), []
        )

    def test_birth_date_conflict_still_blocks_anchored_group(self):
        # Same CF but two different birth dates is a real contradiction.
        documents = [
            _staged(_evidence(name="VITALI REMO", birth="1942-05-15", cf=VALID_CF), 0),
            _staged(_evidence(name="VITALI REMO", birth="1943-06-20", cf=VALID_CF), 1),
        ]
        self.assertEqual(
            PatientRoutingService._conflicting_fields(documents), ["birth_date"]
        )

    def test_two_distinct_cfs_still_conflict(self):
        documents = [
            _staged(_evidence(name="VITALI REMO", birth="1942-05-15", cf=VALID_CF), 0),
            _staged(_evidence(name="VITALI REMO", birth="1942-05-15",
                              cf="RSSMRA85M01H501R"), 1),
        ]
        conflicts = PatientRoutingService._conflicting_fields(documents)
        self.assertIn("fiscal_code", conflicts)

    def test_sex_f_and_m_in_group_still_conflicts(self):
        documents = [
            _staged(_evidence(name="VITALI REMO", birth="1942-05-15", cf=VALID_CF,
                              sex="M"), 0),
            _staged(_evidence(name="VITALI REMO", birth="1942-05-15", cf=VALID_CF,
                              sex="F"), 1),
        ]
        self.assertEqual(
            PatientRoutingService._conflicting_fields(documents), ["sex"]
        )


if __name__ == "__main__":
    unittest.main()

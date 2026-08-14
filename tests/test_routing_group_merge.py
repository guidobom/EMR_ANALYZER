"""Regression tests: duplicate workspaces for one patient must merge.

A patient whose reports arrive in two flavours — one batch anchored by the
fiscal code, another by a hospital patient ID — used to end up in two
workspaces.  The group representative is chosen as the most *identifying*
evidence, and ``_prefer_evidence`` ranks a fiscal-code + hospital-ID document
(17) above a fiscal-code + name + birth one (12).  So the representative of a
CF group could be a document with **no name**, the group then contributed no
``name+birth`` fingerprint, and the two halves of the same person never
merged — producing two folders, the second one without initials.

The representative must therefore keep the identifier-rich evidence but fill
in the name (and birth) from a document that names the patient.
"""

from __future__ import annotations

import unittest

from emr_analyzer.models.patient_identity import IdentityField, PatientIdentityEvidence
from emr_analyzer.pipeline.import_staging import StagedDocument
from emr_analyzer.pipeline.patient_routing import (
    PatientRoutingService,
    RoutingGroup,
)

CF_A = "SNTMTH43A41Z306G"   # MARTHE fiscal code
HP_A = "4000173956"         # an hospital ID carried by the MARTHE CF group
HP_B = "8100736863"         # the hospital ID of the MARTHE HP group
BIRTH = "1943-01-01"
NAME_A = "MARTHE SONTIA EPSE TOLLE"
NAME_B = "SONTIA EPSE TOLLE MARTHE"


def _evidence(name=None, birth=None, cf=None, hpid=None):
    fields = {}
    if name:
        fields["name"] = IdentityField(name, name, confidence=0.97)
    if birth:
        fields["birth_date"] = IdentityField(birth, birth, confidence=0.97)
    if cf:
        fields["fiscal_code"] = IdentityField(cf, cf, confidence=0.97)
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


def _group(documents):
    """An initial routing group, exactly as resolve() builds it."""
    return RoutingGroup(
        key=documents[0].evidence.group_key,
        documents=list(documents),
        evidence=PatientRoutingService._group_evidence(documents),
    )


class GroupEvidenceTest(unittest.TestCase):
    def test_identifier_rich_document_without_name_is_filled(self):
        # The strongest evidence (fiscal code + hospital ID + birth, score 17)
        # has no name; a sibling document names the patient.  The
        # representative must keep both identifiers AND the name.
        documents = [
            _staged(_evidence(cf=CF_A, hpid=HP_A, birth=BIRTH), 0),
            _staged(_evidence(name=NAME_A, birth=BIRTH, cf=CF_A), 1),
        ]
        evidence = PatientRoutingService._group_evidence(documents)
        self.assertEqual(evidence.fiscal_code.normalized, CF_A)
        self.assertEqual(evidence.hospital_patient_id.normalized, HP_A)
        self.assertEqual(evidence.birth_date.normalized, BIRTH)
        self.assertEqual(evidence.name.normalized, NAME_A)

    def test_group_without_any_name_keeps_identifier_evidence(self):
        # No document names the patient: nothing to fill, keep the
        # identifiers.  A name cannot be invented.
        documents = [_staged(_evidence(cf=CF_A, hpid=HP_A, birth=BIRTH), 0)]
        evidence = PatientRoutingService._group_evidence(documents)
        self.assertIsNone(evidence.name)
        self.assertEqual(evidence.fiscal_code.normalized, CF_A)

    def test_group_whose_best_evidence_has_name_is_unchanged(self):
        # The richest document already carries name + birth: no merge needed.
        documents = [
            _staged(_evidence(name=NAME_A, birth=BIRTH, cf=CF_A), 0),
            _staged(_evidence(name=NAME_A, birth=BIRTH, cf=CF_A, hpid=HP_A), 1),
        ]
        evidence = PatientRoutingService._group_evidence(documents)
        self.assertEqual(evidence.name.normalized, NAME_A)
        self.assertEqual(evidence.hospital_patient_id.normalized, HP_A)

    def test_named_document_with_birth_wins_over_richer_name_only(self):
        # A richer document that has a name but no birth must not let the
        # group lose the birth date needed for a ``name+birth`` fingerprint.
        documents = [
            _staged(_evidence(cf=CF_A, hpid=HP_A, name=NAME_A), 0),
            _staged(_evidence(name=NAME_A, birth=BIRTH, cf=CF_A), 1),
        ]
        evidence = PatientRoutingService._group_evidence(documents)
        self.assertEqual(evidence.birth_date.normalized, BIRTH)
        self.assertEqual(evidence.name.normalized, NAME_A)
        self.assertEqual(evidence.hospital_patient_id.normalized, HP_A)


class BatchMergeTest(unittest.TestCase):
    def test_cf_and_hpid_groups_merge_via_name_birth(self):
        # Group A anchors on the fiscal code (its richest document has no
        # name), group B anchors on a *different* hospital ID.  The only
        # shared signal is name+birth, so the name must survive extraction.
        group_a = _group([
            _staged(_evidence(cf=CF_A, hpid=HP_A, birth=BIRTH), 0),
            _staged(_evidence(name=NAME_A, birth=BIRTH, cf=CF_A), 1),
        ])
        group_b = _group([
            _staged(_evidence(name=NAME_B, birth=BIRTH, hpid=HP_B), 2),
        ])
        merged = PatientRoutingService._merge_batch_identities([group_a, group_b])
        self.assertEqual(len(merged), 1)
        self.assertEqual(len(merged[0].documents), 3)
        # The merged workspace is named and keeps every fingerprint.
        evidence = merged[0].evidence
        self.assertEqual(evidence.name.normalized, NAME_A)
        self.assertEqual(evidence.fiscal_code.normalized, CF_A)
        self.assertEqual(evidence.hospital_patient_id.normalized, HP_A)
        self.assertEqual(evidence.birth_date.normalized, BIRTH)

    def test_same_person_different_name_order_merges(self):
        # ``PIUNNO REMO`` (hospital ID group) and ``REMO PIUNNO`` (CF group)
        # share the birth date and the canonical sorted name.
        cf_piunno = "PNNRME54D11G916N"
        hpid_piunno = "01000126"
        group_a = _group([
            _staged(_evidence(name="REMO PIUNNO", birth="1954-04-11",
                              cf=cf_piunno), 0),
            _staged(_evidence(cf=cf_piunno, birth="1954-04-11"), 1),
        ])
        group_b = _group([
            _staged(_evidence(name="PIUNNO REMO", birth="1954-04-11",
                              hpid=hpid_piunno), 2),
        ])
        merged = PatientRoutingService._merge_batch_identities([group_a, group_b])
        self.assertEqual(len(merged), 1)
        self.assertEqual(len(merged[0].documents), 3)
        self.assertIsNotNone(merged[0].evidence.name)

    def test_distinct_persons_do_not_merge(self):
        # Same birth date but disjoint names: a genuine split, must remain.
        group_a = _group([
            _staged(_evidence(name="FABIA TRALLI", birth="1956-09-15"), 0),
        ])
        group_b = _group([
            _staged(_evidence(name="GIORDANO SOFFRITTI", birth="1956-09-15"), 1),
        ])
        merged = PatientRoutingService._merge_batch_identities([group_a, group_b])
        self.assertEqual(len(merged), 2)


if __name__ == "__main__":
    unittest.main()

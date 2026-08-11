from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import fitz

from emr_analyzer.database.engine import DatabaseEngine
from emr_analyzer.database.migrations import init_database
from emr_analyzer.database.patient_identity_repo import (
    IdentityKeyService,
    PatientIdentityRepository,
)
from emr_analyzer.database.patient_repo import PatientRepository
from emr_analyzer.models import Patient
from emr_analyzer.models.patient_identity import IdentityField, PatientIdentityEvidence
from emr_analyzer.pipeline.import_staging import StagedDocument
from emr_analyzer.pipeline.patient_identity import PatientIdentityExtractor
from emr_analyzer.pipeline.patient_routing import PatientRoutingService


def _staged(evidence: PatientIdentityEvidence, index: int) -> StagedDocument:
    return StagedDocument(
        original_path=f"/tmp/doc{index}.pdf",
        staged_path=f"/tmp/staged{index}.pdf",
        original_name=f"doc{index}.pdf",
        file_hash=f"hash{index}",
        check={},
        evidence=evidence,
    )


def _evidence(name: str, birth: str, cf: str | None = None) -> PatientIdentityEvidence:
    fields = {
        "name": IdentityField(name, name, confidence=0.97),
        "birth_date": IdentityField(birth, birth, confidence=0.97),
    }
    if cf:
        fields["fiscal_code"] = IdentityField(cf, cf, confidence=0.97)
    return PatientIdentityEvidence(source_path="/tmp/doc.pdf", **fields)


VALID_CF = "RSSMRA85M01H501Q"


def _make_header_pdf(path: Path) -> None:
    document = fitz.open()
    page = document.new_page(width=595, height=842)
    # Values are visually above their labels, as in the real 110-document set.
    page.insert_text((72, 180), "MARIO ROSSI")
    page.insert_text((72, 195), "NOME E COGNOME")
    page.insert_text((260, 180), VALID_CF)
    page.insert_text((260, 195), "CODICE FISCALE")
    page.insert_text((72, 225), "ROMA, 01.08.1985")
    page.insert_text((72, 240), "LUOGO E DATA DI NASCITA")
    page.insert_text((300, 225), "M")
    page.insert_text((300, 240), "SESSO")
    document.save(path)
    document.close()


class PatientRoutingTest(unittest.TestCase):
    def test_coordinate_identity_extraction(self):
        with tempfile.TemporaryDirectory() as tmp:
            pdf_path = Path(tmp) / "header.pdf"
            _make_header_pdf(pdf_path)

            evidence = PatientIdentityExtractor().extract(pdf_path)

            self.assertTrue(evidence.is_strong)
            self.assertIsNotNone(evidence.name)
            self.assertEqual(evidence.name.normalized, "MARIO ROSSI")
            self.assertIsNotNone(evidence.fiscal_code)
            self.assertEqual(evidence.fiscal_code.normalized, VALID_CF)
            self.assertIsNotNone(evidence.birth_date)
            self.assertEqual(evidence.birth_date.normalized, "1985-08-01")
            self.assertIsNotNone(evidence.sex)
            self.assertEqual(evidence.sex.normalized, "M")
            self.assertIsNotNone(evidence.name.bbox)

    def test_identity_repository_matches_keys_and_detects_conflict(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            db = DatabaseEngine(tmp_path / "registry.db")
            init_database(db)
            patient_repo = PatientRepository(db)
            patient_repo.insert(Patient(id="P001", pseudonym="001"))
            identity_repo = PatientIdentityRepository(
                db, IdentityKeyService(tmp_path / "identity.key")
            )
            evidence = PatientIdentityEvidence(
                source_path="synthetic.pdf",
                name=IdentityField("MARIO ROSSI", "MARIO ROSSI", confidence=0.97),
                fiscal_code=IdentityField(VALID_CF, VALID_CF, confidence=0.97),
                birth_date=IdentityField("01.08.1985", "1985-08-01", confidence=0.97),
            )
            identity_repo.upsert("P001", evidence)

            match = identity_repo.find_match(evidence)
            self.assertEqual(match.patient_id, "P001")
            self.assertFalse(match.conflict)

            reversed_name = PatientIdentityEvidence(
                source_path="reversed.pdf",
                name=IdentityField("ROSSI MARIO", "ROSSI MARIO", confidence=0.97),
                birth_date=IdentityField("01.08.1985", "1985-08-01", confidence=0.97),
            )
            reversed_match = identity_repo.find_match(reversed_name)
            self.assertEqual(reversed_match.patient_id, "P001")
            self.assertFalse(reversed_match.conflict)

            conflicting = PatientIdentityEvidence(
                source_path="conflict.pdf",
                name=IdentityField("LUIGI VERDI", "LUIGI VERDI", confidence=0.97),
                fiscal_code=IdentityField(VALID_CF, VALID_CF, confidence=0.97),
                birth_date=IdentityField("01.08.1985", "1985-08-01", confidence=0.97),
            )
            conflict = identity_repo.find_match(conflicting)
            self.assertEqual(conflict.patient_id, "P001")
            self.assertTrue(conflict.conflict)

            # No raw identifying value is persisted in the registry.
            serialized_rows = " ".join(
                str(tuple(row))
                for row in db.execute("SELECT * FROM patient_identities").fetchall()
            )
            self.assertNotIn("MARIO ROSSI", serialized_rows)
            self.assertNotIn(VALID_CF, serialized_rows)

    def test_resolve_merges_cf_and_cfless_groups_for_same_patient(self):
        # One report carries the fiscal code, another the same name (initials
        # inverted between the two) and birth date but no code.  Both describe
        # the same person, so a single workspace must be created.
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            db = DatabaseEngine(tmp_path / "registry.db")
            init_database(db)
            identity_repo = PatientIdentityRepository(
                db, IdentityKeyService(tmp_path / "identity.key")
            )
            router = PatientRoutingService(
                identity_repo, None, None, None, audit_repo=None
            )

            with_cf = _evidence("MARIO ROSSI", "1985-08-01", VALID_CF)
            without_cf = _evidence("ROSSI MARIO", "1985-08-01")
            documents = [_staged(with_cf, 0), _staged(without_cf, 1)]

            self.assertNotEqual(
                with_cf.group_key, without_cf.group_key,
                "precondition: the two documents must land in different groups",
            )

            groups = router.resolve(documents)
            self.assertEqual(len(groups), 1)
            self.assertTrue(groups[0].create_new)
            self.assertFalse(groups[0].conflict)
            self.assertEqual(len(groups[0].documents), 2)
            # The created identity keeps the richest evidence of the batch.
            self.assertIsNotNone(groups[0].evidence.fiscal_code)


if __name__ == "__main__":
    unittest.main()

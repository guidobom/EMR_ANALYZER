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
from emr_analyzer.pipeline.patient_identity import PatientIdentityExtractor


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


if __name__ == "__main__":
    unittest.main()

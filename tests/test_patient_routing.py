from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import fitz

from emr_analyzer.database.engine import DatabaseEngine
from emr_analyzer.database.migrations import init_database
from emr_analyzer.database.patient_identity_repo import (
    IdentityMatch,
    IdentityKeyService,
    PatientIdentityRepository,
)
from emr_analyzer.database.patient_repo import PatientRepository
from emr_analyzer.models import Patient
from emr_analyzer.models.patient_identity import IdentityField, PatientIdentityEvidence
from emr_analyzer.pipeline.patient_identity import PatientIdentityExtractor
from emr_analyzer.pipeline.import_staging import StagedDocument
from emr_analyzer.pipeline.patient_routing import (
    PatientRoutingService,
    RoutingGroup,
    build_routing_plan,
)


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


def _make_false_department_identity_pdf(path: Path) -> None:
    document = fitz.open()
    page = document.new_page(width=595, height=842)
    page.insert_text((72, 180), "DIREZIONE UNITA GESTIONE DOCUMENTALE")
    page.insert_text((72, 195), "NOME E COGNOME")
    page.insert_text((72, 225), "Data referto: 24.07.2026")
    document.save(path)
    document.close()


def _make_competing_birth_dates_pdf(path: Path) -> None:
    document = fitz.open()
    page = document.new_page(width=595, height=842)
    page.insert_text((72, 180), "MARIO ROSSI")
    page.insert_text((72, 195), "NOME E COGNOME")
    page.insert_text((105, 225), "24.07.2026")
    page.insert_text((72, 240), "DATA DI NASCITA")
    page.insert_text((200, 240), "01.08.1985")
    document.save(path)
    document.close()


def _staged(index: int, evidence: PatientIdentityEvidence) -> StagedDocument:
    return StagedDocument(
        original_path=f"/source/{index}.pdf",
        staged_path=f"/staged/{index}.pdf",
        original_name=f"{index}.pdf",
        file_hash=f"hash-{index}",
        check={},
        evidence=evidence,
    )


class _NoIdentityMatchRepository:
    @staticmethod
    def find_match(evidence):
        return IdentityMatch(reason="Nessuna identità registrata compatibile")


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

    def test_department_heading_and_report_date_do_not_form_patient_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            pdf_path = Path(tmp) / "department.pdf"
            _make_false_department_identity_pdf(pdf_path)

            evidence = PatientIdentityExtractor().extract(pdf_path)

            self.assertIsNone(evidence.name)
            self.assertIsNone(evidence.birth_date)
            self.assertFalse(evidence.is_strong)
            self.assertIsNone(evidence.group_key)

    def test_same_line_birth_date_wins_over_unrelated_date_above_label(self):
        with tempfile.TemporaryDirectory() as tmp:
            pdf_path = Path(tmp) / "competing-dates.pdf"
            _make_competing_birth_dates_pdf(pdf_path)

            evidence = PatientIdentityExtractor().extract(pdf_path)

            self.assertIsNotNone(evidence.birth_date)
            self.assertEqual(evidence.birth_date.normalized, "1985-08-01")

    def test_batch_identity_bridge_merges_cf_name_order_and_partial_birth(self):
        birth = IdentityField(
            "01.08.1985", "1985-08-01", confidence=0.97
        )
        sex = IdentityField("F", "F", confidence=0.94)
        full = PatientIdentityEvidence(
            source_path="full.pdf",
            name=IdentityField(
                "MARIA GIULIA NERI", "MARIA GIULIA NERI", confidence=0.97
            ),
            fiscal_code=IdentityField(VALID_CF, VALID_CF, confidence=0.97),
            birth_date=birth,
            sex=sex,
        )
        cf_only_layout = PatientIdentityEvidence(
            source_path="cf.pdf",
            fiscal_code=IdentityField(VALID_CF, VALID_CF, confidence=0.97),
            birth_date=birth,
            sex=sex,
        )
        reversed_name = PatientIdentityEvidence(
            source_path="reversed.pdf",
            name=IdentityField(
                "NERI MARIA GIULIA", "NERI MARIA GIULIA", confidence=0.95
            ),
            birth_date=birth,
            sex=sex,
        )
        partial = PatientIdentityEvidence(
            source_path="partial.pdf",
            birth_date=birth,
            sex=sex,
        )
        unrelated = PatientIdentityEvidence(
            source_path="unrelated.pdf",
            birth_date=IdentityField(
                "02.08.1985", "1985-08-02", confidence=0.97
            ),
        )
        router = PatientRoutingService(
            _NoIdentityMatchRepository(), None, None, None
        )

        groups = router.resolve([
            _staged(1, full),
            _staged(2, cf_only_layout),
            _staged(3, reversed_name),
            _staged(4, partial),
            _staged(5, unrelated),
        ])

        self.assertEqual(len(groups), 2)
        merged = next(group for group in groups if group.create_new)
        blocked = next(group for group in groups if group.needs_review)
        self.assertEqual(len(merged.documents), 4)
        self.assertEqual(merged.partial_matches, 1)
        self.assertEqual(len(blocked.documents), 1)
        self.assertIsNone(blocked.patient_id)

    def test_partial_birth_is_not_attached_in_multi_patient_batch(self):
        first = PatientIdentityEvidence(
            source_path="first.pdf",
            name=IdentityField("MARIO ROSSI", "MARIO ROSSI", confidence=0.97),
            fiscal_code=IdentityField(VALID_CF, VALID_CF, confidence=0.97),
            birth_date=IdentityField(
                "01.08.1985", "1985-08-01", confidence=0.97
            ),
        )
        second = PatientIdentityEvidence(
            source_path="second.pdf",
            name=IdentityField("LUIGI VERDI", "LUIGI VERDI", confidence=0.97),
            fiscal_code=IdentityField(
                "VRDLGI80A01H501X", "VRDLGI80A01H501X", confidence=0.97
            ),
            birth_date=IdentityField(
                "01.01.1980", "1980-01-01", confidence=0.97
            ),
        )
        partial = PatientIdentityEvidence(
            source_path="partial.pdf",
            birth_date=IdentityField(
                "01.08.1985", "1985-08-01", confidence=0.97
            ),
        )
        router = PatientRoutingService(
            _NoIdentityMatchRepository(), None, None, None
        )

        groups = router.resolve([
            _staged(1, first),
            _staged(2, second),
            _staged(3, partial),
        ])

        self.assertEqual(len(groups), 3)
        self.assertEqual(sum(group.create_new for group in groups), 2)
        blocked = [group for group in groups if group.needs_review]
        self.assertEqual(len(blocked), 1)
        self.assertEqual(blocked[0].documents[0].evidence, partial)

    def test_routing_plan_keeps_strong_identities_separate_and_blocks_unknowns(self):
        first = RoutingGroup(
            key="patient:first",
            documents=[object()],
            create_new=True,
        )
        second = RoutingGroup(
            key="patient:second",
            documents=[object(), object()],
            create_new=True,
        )
        unresolved = RoutingGroup(
            key="unresolved:1",
            documents=[object()] * 259,
            needs_review=True,
            reason="Identità insufficiente",
        )

        plan = build_routing_plan([first, second, unresolved])

        self.assertEqual(plan.new_groups, [first, second])
        self.assertEqual(plan.blocked_groups, [unresolved])
        self.assertEqual(plan.importable_count, 3)
        self.assertTrue(first.create_new)
        self.assertTrue(second.create_new)
        self.assertIsNone(unresolved.patient_id)

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

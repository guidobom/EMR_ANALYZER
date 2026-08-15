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


def _make_lines_pdf(path: Path, *lines: tuple[float, float, str]) -> None:
    """PDF whose first page contains the given (x, baseline, text) items."""
    document = fitz.open()
    page = document.new_page(width=595, height=842)
    for x, baseline, text in lines:
        page.insert_text((x, baseline), text)
    document.save(path)
    document.close()


def _make_text_pdf(path: Path, *lines: str) -> None:
    """PDF with arbitrary raw text, used for the auto-assignment second pass."""
    document = fitz.open()
    page = document.new_page(width=595, height=842)
    y = 150
    for line in lines:
        page.insert_text((72, y), line)
        y += 20
    document.save(path)
    document.close()


def _staged_on_disk(path: Path, evidence: PatientIdentityEvidence,
                    index: int) -> StagedDocument:
    return StagedDocument(
        original_path=str(path),
        staged_path=str(path),
        original_name=path.name,
        file_hash=f"hash{index}",
        check={},
        evidence=evidence,
    )


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

    def test_extraction_birth_label_without_di(self):
        # Ferrara Radiologia header uses "Data Nascita:" (no "DI"): it must
        # still be recognised and yield the value on the label's own line.
        with tempfile.TemporaryDirectory() as tmp:
            pdf_path = Path(tmp) / "header.pdf"
            _make_lines_pdf(
                pdf_path,
                (72, 200, "Data Nascita:"),
                (200, 200, "31/12/1950"),
            )
            evidence = PatientIdentityExtractor().extract(pdf_path)
            self.assertIsNotNone(evidence.birth_date)
            self.assertEqual(evidence.birth_date.normalized, "1950-12-31")

    def test_extraction_space_separated_birth_date(self):
        # Anatomia Patologica prints "31 12 1950" (spaces, no "./-") inline.
        with tempfile.TemporaryDirectory() as tmp:
            pdf_path = Path(tmp) / "header.pdf"
            _make_lines_pdf(pdf_path, (72, 200, "Data di nascita: 31 12 1950"))
            evidence = PatientIdentityExtractor().extract(pdf_path)
            self.assertIsNotNone(evidence.birth_date)
            self.assertEqual(evidence.birth_date.normalized, "1950-12-31")

    def test_extraction_numeric_hospital_id_excludes_letterhead(self):
        # "Id Paz:" carries a purely numeric patient ID.  A letterhead phone
        # number ABOVE the label (within the vertical window) must not be
        # mistaken for it.
        with tempfile.TemporaryDirectory() as tmp:
            pdf_path = Path(tmp) / "header.pdf"
            _make_lines_pdf(
                pdf_path,
                (72, 178, "Segreteria Tel. 0532/236656"),
                (72, 200, "Id Paz:"),
                (140, 200, "8100455504"),
            )
            evidence = PatientIdentityExtractor().extract(pdf_path)
            self.assertIsNotNone(evidence.hospital_patient_id)
            self.assertEqual(
                evidence.hospital_patient_id.normalized, "8100455504"
            )

    def test_extraction_wrapped_name_inline_is_merged(self):
        # "Cognome e nome: GUERRA VILLIAM" with "SILVESTRO" on the next line:
        # the inline value must be extended, not returned truncated.
        with tempfile.TemporaryDirectory() as tmp:
            pdf_path = Path(tmp) / "header.pdf"
            _make_lines_pdf(
                pdf_path,
                (72, 200, "Cognome e nome: GUERRA VILLIAM"),
                (72, 214, "SILVESTRO"),
                (72, 230, "Data di nascita: 31/12/1950"),
            )
            evidence = PatientIdentityExtractor().extract(pdf_path)
            self.assertIsNotNone(evidence.name)
            self.assertEqual(evidence.name.normalized, "GUERRA VILLIAM SILVESTRO")
            self.assertTrue(evidence.is_strong)

    def test_extraction_sig_lab_report_name(self):
        # Lab-report header has no dedicated name label; the patient is named
        # by "Sig. COGNOME NOME" inside the demographic block.
        with tempfile.TemporaryDirectory() as tmp:
            pdf_path = Path(tmp) / "header.pdf"
            _make_lines_pdf(
                pdf_path,
                (72, 180, "Sig. GUERRA VILLIAM SILVESTRO"),
                (72, 196, "Data Nascita: 31/12/1950"),
                (72, 212, "Id. Paz.: 6100573725"),
            )
            evidence = PatientIdentityExtractor().extract(pdf_path)
            self.assertIsNotNone(evidence.name)
            self.assertEqual(evidence.name.normalized, "GUERRA VILLIAM SILVESTRO")
            self.assertIsNotNone(evidence.hospital_patient_id)
            self.assertEqual(
                evidence.hospital_patient_id.normalized, "6100573725"
            )
            self.assertTrue(evidence.is_strong)

    def test_extraction_label_row_above_name_not_used_as_name(self):
        # The "Esame Numero:" label sits 25px above "Cognome e nome:"; it must
        # not be validated as the patient name even though its tokens look
        # like words.
        with tempfile.TemporaryDirectory() as tmp:
            pdf_path = Path(tmp) / "header.pdf"
            _make_lines_pdf(
                pdf_path,
                (72, 200, "Esame Numero:"),
                (140, 200, "B2012-004006"),
                (72, 225, "Cognome e nome:"),
                (190, 225, "GUERRA VILLIAM SILVESTRO"),
            )
            evidence = PatientIdentityExtractor().extract(pdf_path)
            self.assertIsNotNone(evidence.name)
            self.assertEqual(evidence.name.normalized, "GUERRA VILLIAM SILVESTRO")
            self.assertNotEqual(evidence.name.normalized, "ESAME NUMERO")

    def test_resolve_merges_truncated_and_full_name_groups_by_birth(self):
        # Even before extraction merges a wrapped name, a batch whose groups
        # split on the truncated name must still resolve to ONE new workspace
        # via the shared birth date (the second-pass surname/birth vote).
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

            full_path = tmp_path / "full.pdf"
            _make_text_pdf(full_path, "GUERRA VILLIAM SILVESTRO", "31/12/1950")
            trunc_path = tmp_path / "trunc.pdf"
            _make_text_pdf(trunc_path, "GUERRA VILLIAM", "31/12/1950")

            documents = [
                _staged_on_disk(
                    full_path, _evidence("GUERRA VILLIAM SILVESTRO",
                                         "1950-12-31"), 0),
                _staged_on_disk(
                    trunc_path, _evidence("GUERRA VILLIAM", "1950-12-31"), 1),
            ]

            groups = router.resolve(documents)
            self.assertEqual(len(groups), 1)
            self.assertEqual(len(groups[0].documents), 2)
            self.assertTrue(groups[0].create_new)
            self.assertFalse(groups[0].conflict)

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

    def test_resolve_minority_fiscal_code_is_noise_not_conflict(self):
        # A group dominated by one checksum-valid fiscal code can carry a
        # stray divergent code — the hospital printed a male-encoded variant
        # of the same person (SRCMND45C13G916J vs SRCMND45C53G916N).  The
        # minority value must not send all 110 documents to "Da assegnare":
        # the group routes as one strong identity and the dominant code wins.
        dominant_cf = "SRCMND45C53G916N"
        minority_cf = "SRCMND45C13G916J"
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

            documents = [
                _staged(_evidence("SERACENI MIRANDA", "1945-03-13", dominant_cf), i)
                for i in range(80)
            ]
            documents.append(
                _staged(_evidence("SERACENI MIRANDA", "1945-03-13", minority_cf), 80)
            )

            groups = router.resolve(documents)
            self.assertEqual(len(groups), 1)
            self.assertTrue(groups[0].create_new)
            self.assertFalse(groups[0].conflict)
            self.assertEqual(len(groups[0].documents), len(documents))
            # The workspace registers the majority code, not the stray one.
            self.assertEqual(groups[0].evidence.fiscal_code.normalized, dominant_cf)

    def test_resolve_balanced_fiscal_code_split_still_conflicts(self):
        # A genuine split between two codes (3 vs 3) means two people share
        # the name; the majority tolerance must not collapse them.
        dominant_cf = "SRCMND45C53G916N"
        other_cf = "SRCMND45C13G916J"
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

            documents = (
                [_staged(_evidence("SERACENI MIRANDA", "1945-03-13", dominant_cf), i)
                 for i in range(3)]
                + [_staged(_evidence("SERACENI MIRANDA", "1945-03-13", other_cf), i)
                   for i in range(3, 6)]
            )

            groups = router.resolve(documents)
            self.assertEqual(len(groups), 1)
            self.assertTrue(groups[0].needs_review)
            self.assertTrue(groups[0].conflict)

    # --- best-effort auto-assignment of unresolved groups -----------------

    def test_resolve_auto_assigns_unresolved_by_surname_and_birth(self):
        # A needs_review document whose header layout defeats the extractor
        # still carries the patient's name and birth date in its text.  The
        # second pass must absorb it into the matching strong group.
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

            pb_path = tmp_path / "pb.pdf"
            _make_text_pdf(
                pb_path, "Paziente: BOCCAFOGLI PAOLO",
                "Data di nascita: 19/12/1958", "CF BCCPLA58T19D548S",
            )
            gb_path = tmp_path / "gb.pdf"
            _make_text_pdf(
                gb_path, "Paziente: BISAN GRAZIELLA",
                "Data di nascita: 04/09/1953", "CF BSNGZL53P44H620F",
            )
            unresolved_path = tmp_path / "unresolved.pdf"
            _make_text_pdf(
                unresolved_path, "BOCCAFOGLI PAOLO", "Paziente:",
                "19/12/1958", "Data Nascita:",
            )

            documents = [
                _staged_on_disk(
                    pb_path, _evidence("PAOLO BOCCAFOGLI", "1958-12-19",
                                       "BCCPLA58T19D548S"), 0),
                _staged_on_disk(
                    gb_path, _evidence("GRAZIELLA BISAN", "1953-09-04",
                                       "BSNGZL53P44H620F"), 1),
                _staged_on_disk(
                    unresolved_path,
                    PatientIdentityEvidence(source_path=str(unresolved_path)),
                    2),
            ]

            groups = router.resolve(documents)

            self.assertEqual(len(groups), 2)
            self.assertTrue(all(group.create_new for group in groups))
            self.assertFalse(any(group.needs_review for group in groups))
            pb = next(g for g in groups
                      if g.evidence.name.normalized == "PAOLO BOCCAFOGLI")
            self.assertEqual(len(pb.documents), 2)

    def test_resolve_auto_assigns_by_birth_date_only(self):
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

            pb_path = tmp_path / "pb.pdf"
            _make_text_pdf(pb_path, "BOCCAFOGLI PAOLO", "19/12/1958")
            birth_only = tmp_path / "birth_only.pdf"
            _make_text_pdf(birth_only, "REFERTO N. 1234", "Data: 19/12/1958")

            documents = [
                _staged_on_disk(
                    pb_path, _evidence("PAOLO BOCCAFOGLI", "1958-12-19",
                                       "BCCPLA58T19D548S"), 0),
                _staged_on_disk(
                    birth_only,
                    PatientIdentityEvidence(source_path=str(birth_only)), 1),
            ]

            groups = router.resolve(documents)

            self.assertEqual(len(groups), 1)
            self.assertEqual(len(groups[0].documents), 2)

    def test_resolve_keeps_ambiguous_group_needs_review(self):
        # The unresolved document mentions both surnames: no candidate wins
        # unambiguously, so it must remain needs_review (and visible).
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

            pb_path = tmp_path / "pb.pdf"
            _make_text_pdf(pb_path, "BOCCAFOGLI PAOLO", "19/12/1958")
            gb_path = tmp_path / "gb.pdf"
            _make_text_pdf(gb_path, "BISAN GRAZIELLA", "04/09/1953")
            both_path = tmp_path / "both.pdf"
            _make_text_pdf(both_path, "BOCCAFOGLI e BISAN")

            documents = [
                _staged_on_disk(
                    pb_path, _evidence("PAOLO BOCCAFOGLI", "1958-12-19",
                                       "BCCPLA58T19D548S"), 0),
                _staged_on_disk(
                    gb_path, _evidence("GRAZIELLA BISAN", "1953-09-04",
                                       "BSNGZL53P44H620F"), 1),
                _staged_on_disk(
                    both_path,
                    PatientIdentityEvidence(source_path=str(both_path)), 2),
            ]

            groups = router.resolve(documents)

            unresolved = [g for g in groups if g.needs_review]
            self.assertEqual(len(unresolved), 1)
            self.assertEqual(len(unresolved[0].documents), 1)
            self.assertEqual(unresolved[0].reason, (
                "Identità insufficiente per l'attribuzione automatica"
            ))

    def test_resolve_without_candidates_leaves_unresolved(self):
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

            path = tmp_path / "only.pdf"
            _make_text_pdf(path, "Nessun nome riconoscibile")
            documents = [
                _staged_on_disk(
                    path,
                    PatientIdentityEvidence(source_path=str(path)), 0),
            ]

            groups = router.resolve(documents)

            self.assertEqual(len(groups), 1)
            self.assertTrue(groups[0].needs_review)
            self.assertIsNone(groups[0].patient_id)
            self.assertFalse(groups[0].create_new)


if __name__ == "__main__":
    unittest.main()

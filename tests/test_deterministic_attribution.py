"""Tests for deterministic (non-LLM) document attribution.

A document whose own evidence — PDF header or a checksum-valid fiscal code
anywhere in the text — strongly confirms the assigned patient skips the LLM
identity call entirely.  Only HMAC digests are compared; the raw values live
only in the freshly-built evidence object.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtWidgets import QApplication

from emr_analyzer.database.engine import DatabaseEngine
from emr_analyzer.database.migrations import init_database
from emr_analyzer.database.patient_identity_repo import (
    PatientIdentityRepository,
)
from emr_analyzer.database.patient_repo import PatientRepository
from emr_analyzer.gui.documents_tab import DocumentsTab
from emr_analyzer.models import Patient
from emr_analyzer.models.document import DocumentRecord
from emr_analyzer.models.patient_identity import (
    IdentityField, PatientIdentityEvidence,
)
from emr_analyzer.pipeline.deterministic_attribution import (
    deterministic_verdict, evidence_from_fulltext, strong_evidence_confirms,
)
from emr_analyzer.pipeline.patient_identity import (
    normalize_fiscal_code, normalize_text, fiscal_code_has_valid_checksum,
)

# Mario Rossi, born 1980-01-01 — checksum-valid Italian fiscal code.
CF_MARIO = "RSSMRA80A01H501U"


class _StubKeyService:
    def digest(self, field_name, normalized_value):
        return f"{field_name}:{normalized_value}"


class _Progress:
    def __init__(self):
        self.messages = []

    def add_log(self, message):
        self.messages.append(message)


class _FakeLlm:
    """An LLM that fails loudly: the deterministic pre-check must avoid it."""

    is_available = True
    model = "fake-model"

    def __init__(self):
        self.calls = 0

    def extract_patient_identity(self, text):
        self.calls += 1
        raise AssertionError("LLM identity call must be skipped when "
                             "the deterministic pre-check confirms")


# The PDF bodies must exceed the 60-char guard in _attribution_raw_text,
# otherwise the LLM path short-circuits before calling extract_patient_identity
# and the tests would pass for the wrong reason.
LONG_TAIL = (
    "Il paziente è seguito dal reparto di oncologia in regime di follow-up "
    "periodico e non presenta complicanze in atto."
)


def _long_report(cf: str | None = None) -> str:
    cf_part = f" Codice fiscale: {cf}." if cf else ""
    return "Referto di controllo per patologia oncologica." + cf_part + " " + LONG_TAIL


def _identity_evidence(name=None, birth=None, cf=None):
    fields = {}
    if name:
        fields["name"] = IdentityField(
            name, normalize_text(name), confidence=0.99
        )
    if birth:
        fields["birth_date"] = IdentityField(
            birth, birth, confidence=0.99
        )
    if cf:
        cf_norm = normalize_fiscal_code(cf)
        fields["fiscal_code"] = IdentityField(
            cf_norm, cf_norm, confidence=0.99
        )
    return PatientIdentityEvidence(source_path="test", **fields)


class StrongEvidenceConfirmsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])
        assert fiscal_code_has_valid_checksum(CF_MARIO)

    def setUp(self):
        self._tmp = Path(tempfile.mkdtemp(prefix="emr_det_attrib_"))
        self._db = DatabaseEngine(self._tmp / "registry.db")
        init_database(self._db)
        self._patient_repo = PatientRepository(self._db)
        self._patient_repo.insert(Patient(id="P001", pseudonym="001"))
        self._identity_repo = PatientIdentityRepository(
            self._db, key_service=_StubKeyService()
        )

    def tearDown(self):
        shutil.rmtree(self._tmp, ignore_errors=True)

    def _register(self, patient_id, name=None, birth=None, cf=None):
        self._identity_repo.upsert(
            patient_id, _identity_evidence(name=name, birth=birth, cf=cf),
            status="auto",
        )

    def test_cf_alone_confirms(self):
        self._register("P001", cf=CF_MARIO)
        evidence = _identity_evidence(cf=CF_MARIO)
        self.assertTrue(strong_evidence_confirms(
            self._identity_repo, evidence, "P001"))

    def test_name_birth_pair_confirms(self):
        self._register("P001", name="Mario Rossi", birth="1980-01-01")
        evidence = _identity_evidence(
            name="Mario Rossi", birth="1980-01-01")
        self.assertTrue(strong_evidence_confirms(
            self._identity_repo, evidence, "P001"))

    def test_wrong_patient_not_confirmed(self):
        self._register("P001", cf=CF_MARIO)
        evidence = _identity_evidence(cf=CF_MARIO)
        self.assertFalse(strong_evidence_confirms(
            self._identity_repo, evidence, "P002"))

    def test_cf_and_name_birth_both_present_pick_confirming(self):
        # LLM-misread pattern: a wrong name/birth pair with the correct CF.
        self._register("P001", name="Mario Rossi", birth="1980-01-01",
                       cf=CF_MARIO)
        evidence = _identity_evidence(
            name="JOLANDA DI SAVOIA", birth="1980-01-01", cf=CF_MARIO)
        self.assertTrue(strong_evidence_confirms(
            self._identity_repo, evidence, "P001"))


class EvidenceFromFulltextTest(unittest.TestCase):
    def test_finds_valid_cf_in_long_text(self):
        text = (
            "REFERTO DI CONTROLLO PER PATOLOGIA ONCOLOGICA.\n"
            f"Il codice fiscale del paziente è {CF_MARIO} e la nascita "
            "è già nota all'anagrafe sanitaria regionale."
        )
        evidence = evidence_from_fulltext(text)
        self.assertIsNotNone(evidence)
        self.assertIsNotNone(evidence.fiscal_code)
        self.assertEqual(evidence.fiscal_code.normalized, CF_MARIO)

    def test_ignores_malformed_cf(self):
        evidence = evidence_from_fulltext(
            "Nessun codice fiscale, solo un numero 8100455504 di telefono.")
        self.assertIsNone(evidence)

    def test_empty_text_returns_none(self):
        self.assertIsNone(evidence_from_fulltext(""))
        self.assertIsNone(evidence_from_fulltext(None))


class DeterministicPrecheckTest(unittest.TestCase):
    """The runtime pre-check skips the LLM when the PDF confirms the patient."""

    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self._tmp = Path(tempfile.mkdtemp(prefix="emr_det_precheck_"))
        self._db = DatabaseEngine(self._tmp / "registry.db")
        init_database(self._db)
        self._patient_repo = PatientRepository(self._db)
        self._patient_repo.insert(Patient(id="P001", pseudonym="001"))
        self._identity_repo = PatientIdentityRepository(
            self._db, key_service=_StubKeyService()
        )

    def tearDown(self):
        shutil.rmtree(self._tmp, ignore_errors=True)

    def _make_pdf(self, body: str) -> Path:
        import fitz

        path = self._tmp / "report.pdf"
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((72, 72), body)
        doc.save(path)
        doc.close()
        return path

    def _doc(self, pdf_path, patient_id="P001", doc_id="DOC_000001"):
        record = DocumentRecord(
            id=doc_id, patient_id=patient_id, filename="report.pdf",
            original_path=str(pdf_path), file_hash="h",
            document_type="visita_specialistica",
        )
        # document_identity_evidence has an FK on documents(id): the doc row
        # must exist for the deterministic evidence to be persisted.
        self._db.execute(
            "INSERT OR REPLACE INTO documents "
            "(id, patient_id, filename, original_path, file_hash, "
            " document_type, import_date) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (record.id, record.patient_id, record.filename,
             record.original_path, record.file_hash,
             record.document_type, "2026-01-01T00:00:00"),
        )
        return record

    def _tab(self, llm=None):
        tab = DocumentsTab()
        services = {
            "identity_repo": self._identity_repo,
            "db": self._db,
        }
        if llm is not None:
            services["document_llm_client"] = llm
        tab.set_services(services)
        return tab

    def test_cf_in_pdf_skips_llm(self):
        # Registered CF appears verbatim in the PDF text: the deterministic
        # pre-check confirms P001 and the LLM identity call is never made.
        self._identity_repo.upsert(
            "P001", _identity_evidence(cf=CF_MARIO), status="auto")
        pdf = self._make_pdf(_long_report(CF_MARIO))
        llm = _FakeLlm()
        tab = self._tab(llm=llm)
        with mock.patch(
            "emr_analyzer.gui.documents_tab.ATTRIBUTION_VERIFICATION_ENABLED",
            True,
        ):
            result = tab._verify_document_attribution(
                self._doc(pdf), "testo", None, _Progress())
        self.assertIsNone(result)
        self.assertEqual(llm.calls, 0)

    def test_evidence_persisted_after_precheck(self):
        # The deterministic evidence lands in document_identity_evidence.
        self._identity_repo.upsert(
            "P001", _identity_evidence(cf=CF_MARIO), status="auto")
        pdf = self._make_pdf(_long_report(CF_MARIO))
        tab = self._tab()
        with mock.patch(
            "emr_analyzer.gui.documents_tab.ATTRIBUTION_VERIFICATION_ENABLED",
            True,
        ):
            tab._verify_document_attribution(
                self._doc(pdf), "testo", None, _Progress())
        rows = self._db.execute(
            "SELECT document_id, patient_id, field_name FROM "
            "document_identity_evidence WHERE document_id=?",
            ("DOC_000001",),
        ).fetchall()
        self.assertTrue(rows, "deterministic evidence must be persisted")
        self.assertEqual(rows[0]["field_name"], "fiscal_code")

    def test_pdf_without_evidence_falls_through_to_llm(self):
        # No CF anywhere in the PDF: the pre-check is inconclusive and the
        # LLM identity call runs (here it would confirm, so no error).
        self._identity_repo.upsert(
            "P001", _identity_evidence(
                name="Mario Rossi", birth="1980-01-01"), status="auto")
        pdf = self._make_pdf(_long_report())
        calls = {"n": 0}

        class _Llm:
            is_available = True
            model = "fake"
            def extract_patient_identity(self, text):
                calls["n"] += 1
                return {"name": "Mario Rossi", "birth_date": "1980-01-01",
                        "confidence": 0.95}

        tab = self._tab(llm=_Llm())
        with mock.patch(
            "emr_analyzer.gui.documents_tab.ATTRIBUTION_VERIFICATION_ENABLED",
            True,
        ):
            result = tab._verify_document_attribution(
                self._doc(pdf), "testo", None, _Progress())
        self.assertIsNone(result)
        self.assertEqual(calls["n"], 1)


if __name__ == "__main__":
    unittest.main()

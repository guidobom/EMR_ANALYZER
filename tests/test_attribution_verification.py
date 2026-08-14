"""Tests for the LLM-based document attribution verification.

A small structured LLM call on the pre-anonymization text confirms that a
document's identity matches the workspace it was routed to.  A mismatch
blocks the isolation (no normalized text under the wrong patient) and flags
the document for review.
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

from emr_analyzer.database.audit_repo import AuditRepository
from emr_analyzer.database.engine import DatabaseEngine
from emr_analyzer.database.migrations import init_database
from emr_analyzer.database.patient_identity_repo import (
    PatientIdentityRepository,
)
from emr_analyzer.database.patient_repo import PatientRepository
from emr_analyzer.gui.documents_tab import (
    AttributionMismatchError, DocumentsTab,
)
from emr_analyzer.models import Patient
from emr_analyzer.models.document import DocumentRecord
from emr_analyzer.models.patient_identity import (
    IdentityField, PatientIdentityEvidence,
)
from emr_analyzer.pipeline.patient_identity import (
    normalize_fiscal_code, normalize_text, fiscal_code_has_valid_checksum,
)

# Mario Rossi, born 1980-01-01 — checksum-valid Italian fiscal code.
CF_MARIO = "RSSMRA80A01H501U"


class _StubKeyService:
    """Deterministic key service: no local key file is ever touched."""

    def digest(self, field_name, normalized_value):
        return f"{field_name}:{normalized_value}"


class _Progress:
    def __init__(self):
        self.messages = []

    def add_log(self, message):
        self.messages.append(message)


class _FakeIsolatorResult:
    warnings = []
    redaction_counts = {}
    text = "testo clinico normalizzato"
    chunk_count = 1
    character_count = len(text)
    prompt_version = "v1"
    deidentification_version = "v1"
    model_name = "fake-model"
    output_format = "normalized_plain_text"


class _FakeIsolator:
    def __init__(self, result=None):
        self.result = result or _FakeIsolatorResult()
        self.calls = []

    def isolate(self, text, **kwargs):
        self.calls.append((text, kwargs))
        return self.result


class _FakeIdentityLlm:
    is_available = True
    model = "fake-model"

    def __init__(self, identity):
        self.identity = identity

    def extract_patient_identity(self, text):
        return self.identity


def _identity_evidence(name=None, birth=None, cf=None, source="test"):
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
    return PatientIdentityEvidence(source_path=source, **fields)


class AttributionVerificationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])
        assert fiscal_code_has_valid_checksum(CF_MARIO), \
            "CF di test non valido"

    def setUp(self):
        self._tmp = Path(tempfile.mkdtemp(prefix="emr_attribution_"))
        self._db = DatabaseEngine(self._tmp / "registry.db")
        init_database(self._db)
        self._patient_repo = PatientRepository(self._db)
        self._patient_repo.insert(Patient(id="P001", pseudonym="001"))
        self._patient_repo.insert(Patient(id="P002", pseudonym="002"))
        self._identity_repo = PatientIdentityRepository(
            self._db, key_service=_StubKeyService()
        )
        self._audit_repo = AuditRepository(self._db)

    def tearDown(self):
        shutil.rmtree(self._tmp, ignore_errors=True)

    def _register(self, patient_id, name=None, birth=None, cf=None):
        self._identity_repo.upsert(
            patient_id,
            _identity_evidence(name=name, birth=birth, cf=cf),
            status="auto",
        )

    @staticmethod
    def _report(payload):
        """Realistic report body: long enough to pass the 60-char guard."""
        return (
            "AZIENDA OSPEDALIERA PROVINCIALE DI RIFERIMENTO - "
            "DIPARTIMENTO ONCOLOGICO - ONCOLOGIA MEDICA\n"
            "Referto di controllo per patologia oncologica in follow-up.\n"
            + payload + "\n"
            "Il presente referto è stato redatto in conformità alla "
            "normativa vigente sulla documentazione sanitaria."
        )

    def _doc(self, patient_id="P002", doc_id="DOC_000001"):
        return DocumentRecord(
            id=doc_id, patient_id=patient_id, filename="report.pdf",
            original_path=str(self._tmp / "report.pdf"), file_hash="h",
        )

    def _tab(self, identity=None, with_verification_services=True):
        tab = DocumentsTab()
        services = {}
        if with_verification_services:
            services.update({
                "identity_repo": self._identity_repo,
                "audit_repo": self._audit_repo,
                "db": self._db,
            })
        if identity is not None:
            services["document_llm_client"] = _FakeIdentityLlm(identity)
        tab.set_services(services)
        return tab

    # --- confirmed / mismatch ------------------------------------------

    def test_confirmed_identity_does_not_raise(self):
        self._register("P001", name="Mario Rossi", birth="1980-01-01")
        tab = self._tab({
            "name": "Mario Rossi", "birth_date": "1980-01-01",
            "confidence": 0.95,
        })
        result = tab._verify_document_attribution(
            self._doc("P001"),
            self._report("Paziente: Mario Rossi nato il 1980-01-01."),
            None, _Progress(),
        )
        self.assertIsNone(result)

    def test_mismatch_name_birth_raises_and_flags(self):
        self._register("P001", name="Mario Rossi", birth="1980-01-01")
        tab = self._tab({
            "name": "Mario Rossi", "birth_date": "1980-01-01",
            "confidence": 0.95,
        })
        progress = _Progress()
        doc = self._doc("P002")
        with self.assertRaises(AttributionMismatchError) as ctx:
            tab._verify_document_attribution(
                doc, self._report("Paziente: Mario Rossi nato il 1980-01-01."),
                None, progress,
            )
        self.assertIn("P001", str(ctx.exception))
        self.assertIn("P002", str(ctx.exception))
        # Review queue + audit trail carry the suggested workspace.
        rows = self._db.execute(
            "SELECT patient_id, item_type, severity, status, original_value "
            "FROM validation_queue WHERE item_id=? AND item_type='attribution'",
            (doc.id,),
        ).fetchall()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["patient_id"], "P002")
        self.assertEqual(rows[0]["severity"], "high")
        self.assertIn(
            '"suggested_patient_id": "P001"', rows[0]["original_value"]
        )
        audit = self._audit_repo.get_by_patient("P002")
        self.assertTrue(any(
            entry["action"] == "attribution_mismatch"
            and entry["target_id"] == doc.id
            for entry in audit
        ))

    def test_mismatch_cf_raises(self):
        self._register("P001", cf=CF_MARIO)
        tab = self._tab({"fiscal_code": CF_MARIO, "confidence": 0.97})
        with self.assertRaises(AttributionMismatchError):
            tab._verify_document_attribution(
                self._doc("P002"),
                self._report(f"Codice fiscale {CF_MARIO} del paziente."),
                None, _Progress(),
            )

    # --- anti-hallucination gates --------------------------------------

    def test_hallucinated_cf_not_in_text_is_inconclusive(self):
        self._register("P001", cf=CF_MARIO)
        tab = self._tab({"fiscal_code": CF_MARIO, "confidence": 0.97})
        result = tab._verify_document_attribution(
            self._doc("P002"),
            self._report("Referto di controllo per patologia oncologica."),
            None, _Progress(),
        )
        self.assertIsNone(result)

    def test_lone_name_without_birth_is_inconclusive(self):
        self._register("P001", name="Mario Rossi", birth="1980-01-01")
        tab = self._tab({"name": "Mario Rossi", "confidence": 0.6})
        result = tab._verify_document_attribution(
            self._doc("P002"),
            self._report("Referto firmato dal dottor Mario Rossi."),
            None, _Progress(),
        )
        self.assertIsNone(result)

    def test_conflict_raises(self):
        self._register("P001", cf=CF_MARIO)
        self._register("P002", cf=CF_MARIO)
        tab = self._tab({"fiscal_code": CF_MARIO, "confidence": 0.97})
        with self.assertRaises(AttributionMismatchError):
            tab._verify_document_attribution(
                self._doc("P001"),
                self._report(f"Codice fiscale {CF_MARIO} del paziente."),
                None, _Progress(),
            )

    # --- malformed fiscal codes read by the LLM --------------------------

    def test_invalid_cf_read_by_llm_does_not_conflict(self):
        # The patient has a registered (valid) CF; the LLM misreads a numeric
        # string (e.g. a phone number) as the fiscal code on one report.  A
        # malformed code must not raise a conflict against the registered one:
        # name+birth still identify the workspace, so the extraction proceeds.
        self._register(
            "P001", name="Mario Rossi", birth="1980-01-01", cf=CF_MARIO
        )
        tab = self._tab({
            "name": "Mario Rossi", "birth_date": "1980-01-01",
            "fiscal_code": "8100455504", "confidence": 0.95,
        })
        result = tab._verify_document_attribution(
            self._doc("P001"),
            self._report(
                "Paziente: Mario Rossi nato il 1980-01-01. "
                "Recapito telefonico 8100455504."
            ),
            None, _Progress(),
        )
        self.assertIsNone(result)

    def test_lone_invalid_cf_is_inconclusive(self):
        # A malformed code alone carries no identity: it is dropped from the
        # evidence and the verification stays inconclusive (nothing blocks).
        tab = self._tab({"fiscal_code": "8100455504", "confidence": 0.9})
        result = tab._verify_document_attribution(
            self._doc("P001"),
            self._report("Referto di controllo. Numero 8100455504."),
            None, _Progress(),
        )
        self.assertIsNone(result)

    # --- service / config gating ---------------------------------------

    def test_disabled_flag_skips_verification(self):
        self._register("P001", name="Mario Rossi", birth="1980-01-01")
        tab = self._tab({
            "name": "Mario Rossi", "birth_date": "1980-01-01",
            "confidence": 0.95,
        })
        with mock.patch(
            "emr_analyzer.gui.documents_tab.ATTRIBUTION_VERIFICATION_ENABLED",
            False,
        ):
            result = tab._verify_document_attribution(
                self._doc("P002"), "Mario Rossi 1980-01-01.",
                None, _Progress(),
            )
        self.assertIsNone(result)

    def test_missing_identity_repo_skips_verification(self):
        tab = self._tab({
            "name": "Mario Rossi", "birth_date": "1980-01-01",
            "confidence": 0.95,
        }, with_verification_services=False)
        result = tab._verify_document_attribution(
            self._doc("P002"), "Mario Rossi 1980-01-01.",
            None, _Progress(),
        )
        self.assertIsNone(result)

    # --- raw-text sourcing ----------------------------------------------

    def test_raw_text_prefers_parsing_plain_text(self):
        tab = self._tab()
        header = (
            "MARIO ROSSI\nNato il 01/01/1980 a Roma\n"
            "Codice fiscale RSSMRA80A01H501U\n"
            "Visita oncologica di controllo ambulatoriale"
        )

        class _Parsing:
            plain_text = header

        doc = self._doc()
        doc.original_path = str(self._tmp / "missing.pdf")
        raw = tab._attribution_raw_text(doc, "fallback text", _Parsing())
        self.assertEqual(raw, header)

    def test_raw_text_falls_back_to_source_text(self):
        tab = self._tab()
        doc = self._doc()
        doc.original_path = str(self._tmp / "missing.pdf")
        raw = tab._attribution_raw_text(doc, "sorgente grezza", None)
        self.assertEqual(raw, "sorgente grezza")

    # --- end-to-end: isolation is skipped on mismatch --------------------

    def test_run_llm_extraction_skips_isolator_on_mismatch(self):
        self._register("P001", name="Mario Rossi", birth="1980-01-01")
        isolator = _FakeIsolator()
        llm = _FakeIdentityLlm({
            "name": "Mario Rossi", "birth_date": "1980-01-01",
            "confidence": 0.95,
        })
        tab = DocumentsTab()
        tab.set_services({
            "clinical_text_isolator": isolator,
            "document_llm_client": llm,
            "identity_repo": self._identity_repo,
            "audit_repo": self._audit_repo,
            "db": self._db,
        })
        tab._save_normalized_clinical_text = lambda doc, result: None
        doc = self._doc("P002")
        with self.assertRaises(AttributionMismatchError):
            tab._run_llm_extraction(
                doc,
                self._report("Paziente: Mario Rossi nato il 1980-01-01."),
                _Progress(),
            )
        # The most important guarantee: no normalization under the wrong
        # patient.
        self.assertEqual(isolator.calls, [])


if __name__ == "__main__":
    unittest.main()

"""Tests for the workspace-wide "non normalizzati / errori" feature."""

from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from emr_analyzer.database.document_repo import DocumentRepository
from emr_analyzer.database.engine import DatabaseEngine
from emr_analyzer.database.migrations import init_database
from emr_analyzer.database.patient_repo import PatientRepository
from emr_analyzer.gui.normalization_dialog import (
    classify_pending_documents,
)
from emr_analyzer.models import Patient
from emr_analyzer.models.document import (
    DocumentRecord, DocumentType, ExtractionStatus, ParsingStatus,
)


def _doc(doc_id: str, patient_id: str, *, doc_type="visita_oncologica",
         metadata_json=None, parsing_status=ParsingStatus.COMPLETED.value,
         extraction_status=ExtractionStatus.PENDING.value,
         error_message=None) -> DocumentRecord:
    return DocumentRecord(
        id=doc_id,
        patient_id=patient_id,
        filename=f"{doc_id}.pdf",
        original_path=f"/tmp/{patient_id}/documents/original/{doc_id}.pdf",
        file_hash=f"hash-{doc_id}",
        document_type=doc_type,
        parsing_status=parsing_status,
        extraction_status=extraction_status,
        error_message=error_message,
        metadata_json=metadata_json,
    )


def _meta(clinical_text=True) -> str:
    meta = {"header": {}}
    if clinical_text:
        meta["clinical_text"] = {"text": "testo normalizzato"}
    return json.dumps(meta)


class ClassifyPendingTest(unittest.TestCase):
    def test_non_normalized_non_lab_doc_is_pending(self):
        docs = [_doc("DOC_000001", "P001", metadata_json=None)]
        result = classify_pending_documents(docs)
        self.assertEqual(result.total, 1)
        self.assertEqual(result.count_needs_norm, 1)
        self.assertEqual(result.count_errors, 0)
        self.assertTrue(result.pending[0].needs_norm)
        self.assertFalse(result.pending[0].has_error)

    def test_metadata_null_is_treated_as_not_normalized(self):
        docs = [_doc("DOC_000002", "P001", metadata_json=None)]
        result = classify_pending_documents(docs)
        self.assertTrue(result.pending[0].needs_norm)

    def test_normalized_doc_is_excluded(self):
        docs = [_doc("DOC_000003", "P001", metadata_json=_meta(True))]
        result = classify_pending_documents(docs)
        self.assertEqual(result.total, 0)
        self.assertEqual(result.count_needs_norm, 0)

    def test_laboratory_doc_never_needs_normalization(self):
        docs = [_doc(
            "DOC_000004", "P001", doc_type=DocumentType.LABORATORIO.value,
            metadata_json=None,
        )]
        result = classify_pending_documents(docs)
        self.assertEqual(result.total, 0)
        self.assertEqual(result.count_needs_norm, 0)

    def test_laboratory_doc_with_error_is_pending(self):
        docs = [_doc(
            "DOC_000005", "P001", doc_type=DocumentType.LABORATORIO.value,
            metadata_json=None, parsing_status=ParsingStatus.ERROR.value,
            error_message="PDF non leggibile",
        )]
        result = classify_pending_documents(docs)
        self.assertEqual(result.total, 1)
        self.assertFalse(result.pending[0].needs_norm)
        self.assertTrue(result.pending[0].has_error)
        self.assertEqual(result.count_errors, 1)

    def test_extraction_error_is_pending(self):
        docs = [_doc(
            "DOC_000006", "P002", metadata_json=_meta(False),
            extraction_status=ExtractionStatus.ERROR.value,
        )]
        result = classify_pending_documents(docs)
        self.assertEqual(result.total, 1)
        self.assertTrue(result.pending[0].needs_norm)
        self.assertTrue(result.pending[0].has_error)
        self.assertEqual(result.count_needs_norm, 1)
        self.assertEqual(result.count_errors, 1)

    def test_done_with_clinical_text_is_excluded(self):
        docs = [_doc(
            "DOC_000007", "P002", metadata_json=_meta(True),
            extraction_status=ExtractionStatus.DONE.value,
        )]
        result = classify_pending_documents(docs)
        self.assertEqual(result.total, 0)

    def test_done_without_clinical_text_is_pending_but_flagged_skipped(self):
        docs = [_doc(
            "DOC_000008", "P002", metadata_json=None,
            extraction_status=ExtractionStatus.DONE.value,
        )]
        result = classify_pending_documents(docs)
        self.assertEqual(result.total, 1)
        self.assertTrue(result.pending[0].needs_norm)
        self.assertEqual(result.count_skipped_done, 1)

    def test_invalid_metadata_json_is_treated_as_not_normalized(self):
        docs = [_doc("DOC_000009", "P002", metadata_json="{not-json")]
        result = classify_pending_documents(docs)
        self.assertEqual(result.total, 1)
        self.assertTrue(result.pending[0].needs_norm)

    def test_counts_across_mixed_batch(self):
        docs = [
            _doc("A", "P001", metadata_json=None),                        # norm
            _doc("B", "P001", metadata_json=_meta(True),
                 extraction_status=ExtractionStatus.DONE.value),          # ok
            _doc("C", "P001", doc_type=DocumentType.LABORATORIO.value,
                 metadata_json=None),                                     # lab ok
            _doc("D", "P002", metadata_json=None,
                 parsing_status=ParsingStatus.ERROR.value),               # norm+err
        ]
        result = classify_pending_documents(docs)
        self.assertEqual(result.total, 2)
        self.assertEqual(result.count_needs_norm, 2)
        self.assertEqual(result.count_errors, 1)


class DocumentRepositoryListAllTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.mkdtemp(prefix="emr_listall_")
        self._db = DatabaseEngine(Path(self._tmp) / "registry.db")
        init_database(self._db)
        self._repo = DocumentRepository(self._db)
        self._patients = PatientRepository(self._db)
        self._patients.insert(Patient(id="P001", pseudonym="001", sex="M"))
        self._patients.insert(Patient(id="P002", pseudonym="002", sex="F"))

    def tearDown(self):
        shutil.rmtree(self._tmp, ignore_errors=True)

    def test_list_all_returns_documents_of_every_patient(self):
        self._repo.insert(_doc("DOC_000001", "P001", metadata_json=_meta(True)))
        self._repo.insert(_doc("DOC_000002", "P002", metadata_json=_meta(False)))
        self._repo.insert(_doc("DOC_000003", "P002", metadata_json=None))

        docs = self._repo.list_all()
        self.assertEqual({d.id for d in docs},
                         {"DOC_000001", "DOC_000002", "DOC_000003"})
        # Ordered by patient_id, then date.
        by_patient = [d.patient_id for d in docs]
        self.assertEqual(by_patient, ["P001", "P002", "P002"])

    def test_list_all_roundtrips_metadata(self):
        self._repo.insert(_doc("DOC_000010", "P001", metadata_json=_meta(True)))
        doc = self._repo.list_all()[0]
        self.assertIsNotNone(doc.metadata_json)
        self.assertIn("clinical_text", json.loads(doc.metadata_json))


if __name__ == "__main__":
    unittest.main()

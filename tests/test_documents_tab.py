from __future__ import annotations

import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtWidgets import QApplication

from emr_analyzer.gui.documents_tab import DocumentsTab
from emr_analyzer.models.document import (
    DocumentType,
    DocumentRecord,
    ExtractionStatus,
    ParsingStatus,
)


class _DocumentRepository:
    def __init__(self, documents):
        self.documents = documents

    def list_by_patient(self, patient_id):
        return [doc for doc in self.documents if doc.patient_id == patient_id]


class _Progress:
    def __init__(self):
        self.messages = []

    def add_log(self, message):
        self.messages.append(message)


class _LabRepository:
    def __init__(self):
        self.deleted = []
        self.inserted = []

    def delete_by_document(self, document_id):
        self.deleted.append(document_id)

    def insert_batch(self, values):
        self.inserted.extend(values)


class _ForbiddenLabParser:
    def parse(self, *args, **kwargs):
        raise AssertionError(
            "Il parser lab non deve essere chiamato per una visita"
        )


class _RecordingLabParser:
    def __init__(self):
        self.calls = 0

    def parse(self, *args, **kwargs):
        self.calls += 1
        return []


class DocumentsTabTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_single_button_runs_full_pipeline_only_for_unprocessed_docs(self):
        pending = DocumentRecord(
            id="DOC_000001", patient_id="P001", filename="pending.pdf",
            original_path="/nonexistent/pending.pdf", file_hash="one",
        )
        failed = DocumentRecord(
            id="DOC_000002", patient_id="P001", filename="failed.pdf",
            original_path="/nonexistent/failed.pdf", file_hash="two",
            parsing_status=ParsingStatus.COMPLETED.value,
            extraction_status=ExtractionStatus.ERROR.value,
        )
        completed = DocumentRecord(
            id="DOC_000003", patient_id="P001", filename="done.pdf",
            original_path="/nonexistent/done.pdf", file_hash="three",
            parsing_status=ParsingStatus.COMPLETED.value,
            extraction_status=ExtractionStatus.DONE.value,
        )
        repository = _DocumentRepository([pending, failed, completed])
        tab = DocumentsTab()
        tab.set_services({"document_repo": repository})
        tab.load_patient("P001")
        observed = []
        tab._process_documents = lambda ids, **kwargs: observed.append(
            (ids, kwargs)
        )

        self.assertEqual(
            tab._extract_clinical_text_btn.text(), "🧠 Estrai testo clinico"
        )
        self.assertTrue(tab._extract_clinical_text_btn.isEnabled())
        tab._extract_clinical_text_btn.click()

        # Two-phase: parse-only for docs still needing parsing, then one
        # parallel LLM pass over every parsed doc (already-parsed first).
        self.assertEqual(observed, [
            (["DOC_000001"], {"parse_only": True}),
            (["DOC_000002", "DOC_000001"], {"llm_only": True}),
        ])

        pending.parsing_status = ParsingStatus.COMPLETED.value
        pending.extraction_status = ExtractionStatus.DONE.value
        failed.extraction_status = ExtractionStatus.DONE.value
        tab._refresh_table()
        self.assertFalse(tab._extract_clinical_text_btn.isEnabled())
        tab.deleteLater()

    def test_non_laboratory_report_never_populates_laboratory_table(self):
        document = DocumentRecord(
            id="DOC_000010", patient_id="P001", filename="visit.pdf",
            original_path="/nonexistent/visit.pdf", file_hash="visit",
            document_type=DocumentType.VISITA_ONCOLOGICA.value,
        )
        lab_repo = _LabRepository()
        progress = _Progress()
        tab = DocumentsTab()
        tab.set_services({
            "lab_parser": _ForbiddenLabParser(),
            "lab_repo": lab_repo,
        })

        result = tab._run_extraction(
            document,
            "Emoglobina 10 g/dL citata nella visita.",
            parsing_result=None,
            tables=[],
            progress=progress,
            skip_llm=True,
        )

        self.assertEqual(result["lab_values"], [])
        self.assertEqual(lab_repo.deleted, ["DOC_000010"])
        self.assertEqual(lab_repo.inserted, [])
        self.assertTrue(any(
            "conservati solo nel testo clinico" in message
            for message in progress.messages
        ))
        tab.deleteLater()

    def test_laboratory_report_still_runs_structured_lab_parser(self):
        document = DocumentRecord(
            id="DOC_000011", patient_id="P001", filename="laboratory.pdf",
            original_path="/nonexistent/laboratory.pdf", file_hash="laboratory",
            document_type=DocumentType.LABORATORIO.value,
        )
        parser = _RecordingLabParser()
        lab_repo = _LabRepository()
        tab = DocumentsTab()
        tab.set_services({"lab_parser": parser, "lab_repo": lab_repo})

        tab._run_extraction(
            document, "Emoglobina 10 g/dL", None, [], _Progress(),
            skip_llm=True,
        )

        self.assertEqual(parser.calls, 1)
        self.assertEqual(lab_repo.deleted, ["DOC_000011"])
        tab.deleteLater()

    def test_batch_stage_progress_never_moves_backwards(self):
        values = []
        total = 20
        for index in range(total):
            values.append(
                DocumentsTab._batch_stage_progress(index, total, 0.0)
            )
            values.append(
                DocumentsTab._batch_stage_progress(index, total, 0.55)
            )

        self.assertEqual(values, sorted(values))
        self.assertGreater(values[-1], values[0])
        self.assertLess(values[-1], 100)


if __name__ == "__main__":
    unittest.main()

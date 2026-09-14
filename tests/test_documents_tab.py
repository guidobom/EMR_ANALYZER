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
    def __init__(self, result):
        self.result = result
        self.calls = []

    def isolate(self, text, **kwargs):
        self.calls.append((text, kwargs))
        return self.result


class _FakeLlmClient:
    is_available = True
    model = "fake-model"


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
            (ids, {k: v for k, v in kwargs.items() if k != "progress"})
        )

        self.assertEqual(
            tab._extract_clinical_text_btn.text(), "🧠 Estrai testo clinico"
        )
        self.assertTrue(tab._extract_clinical_text_btn.isEnabled())
        tab._extract_clinical_text_btn.click()

        # One pass from the original for both new documents and retries.
        self.assertEqual(observed, [
            (["DOC_000002", "DOC_000001"], {}),
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

        tab._run_llm_extraction = lambda *args, **kwargs: []
        result = tab._run_extraction(
            document,
            "Emoglobina 10 g/dL citata nella visita.",
            parsing_result=None,
            tables=[],
            progress=progress,
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
        tab._save_normalized_clinical_text = lambda doc, result: None
        tab.set_services({"lab_parser": parser, "lab_repo": lab_repo})

        tab._run_llm_extraction = lambda *args, **kwargs: []
        tab._run_extraction(
            document, "Emoglobina 10 g/dL", None, [], _Progress(),
        )

        self.assertEqual(parser.calls, 1)
        self.assertEqual(lab_repo.deleted, ["DOC_000011"])
        tab.deleteLater()

    def test_lab_document_processing_writes_no_atomic_evidence(self):
        """Out-of-range labs become evidence only via 'Estrai evidenze'."""
        document = DocumentRecord(
            id="DOC_000013", patient_id="P001", filename="laboratory.pdf",
            original_path="/nonexistent/laboratory.pdf", file_hash="laboratory",
            document_type=DocumentType.LABORATORIO.value,
        )

        class _EvidenceSpy:
            def __init__(self):
                self.calls = []

            def replace_document_method(self, doc_id, method, evidence):
                self.calls.append((doc_id, method, evidence))

        from emr_analyzer.extraction.lab_parser import LabParser

        parser = LabParser()
        lab_repo = _LabRepository()
        evidence_spy = _EvidenceSpy()
        tab = DocumentsTab()
        tab._save_normalized_clinical_text = lambda doc, result: None
        tab.set_services({
            "lab_parser": parser,
            "lab_repo": lab_repo,
            "evidence_repo": evidence_spy,
        })

        tab._run_llm_extraction = lambda *args, **kwargs: []
        tab._run_extraction(
            document,
            "Materiale: Urina\nEmoglobina : 0.20 mg/dL Assente",
            None, [], _Progress(),
        )

        self.assertEqual(evidence_spy.calls, [])
        self.assertEqual(len(lab_repo.inserted), 1)
        self.assertEqual(
            lab_repo.inserted[0].normalized_name, "emoglobina_urine"
        )
        tab.deleteLater()

    def test_lab_parser_receives_raw_text_not_cleaned(self):
        document = DocumentRecord(
            id="DOC_000012", patient_id="P001", filename="laboratory.pdf",
            original_path="/nonexistent/laboratory.pdf", file_hash="laboratory",
            document_type=DocumentType.LABORATORIO.value,
        )

        class _SpecimenStrippingCleaner:
            def clean(self, text):
                # The real cleaner drops repeated "Materiale:" lines.
                return "\n".join(
                    line for line in text.splitlines()
                    if "Materiale" not in line
                )

        class _RecordingTextParser:
            def __init__(self):
                self.calls = 0
                self.received = []

            def parse(self, text, **kwargs):
                self.calls += 1
                self.received.append(text)
                return []

        parser = _RecordingTextParser()
        tab = DocumentsTab()
        tab._save_normalized_clinical_text = lambda doc, result: None
        tab.set_services({
            "lab_parser": parser,
            "lab_repo": _LabRepository(),
            "cleaner": _SpecimenStrippingCleaner(),
        })

        raw_text = "Materiale: Urina\nEmoglobina 0.20 mg/dL"
        tab._run_llm_extraction = lambda *args, **kwargs: []
        tab._run_extraction(
            document, raw_text, None, [], _Progress(),
        )

        self.assertEqual(parser.calls, 1)
        self.assertIn("Materiale: Urina", parser.received[0])
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

    def test_extract_clinical_text_reuses_shared_progress(self):
        """The single extraction pass reuses the shared progress dialog."""
        pending = DocumentRecord(
            id="DOC_000021", patient_id="P001", filename="pending.pdf",
            original_path="/nonexistent/pending.pdf", file_hash="pending",
        )
        repository = _DocumentRepository([pending])
        tab = DocumentsTab()
        tab.set_services({"document_repo": repository})
        tab.load_patient("P001")

        class _SharedProgress:
            def __init__(self):
                self.titles = []

            def setWindowTitle(self, title):
                self.titles.append(title)

        progress = _SharedProgress()
        observed = []
        tab._process_documents = lambda ids, **kwargs: observed.append(
            (ids, kwargs.get("progress"))
        )

        tab.extract_clinical_text(
            progress=progress, patient_label="Paziente 1/2: P001"
        )

        self.assertEqual(
            [ids for ids, _ in observed],
            [["DOC_000021"]],
        )
        self.assertEqual([p for _, p in observed], [progress])
        self.assertEqual(progress.titles, ["Paziente 1/2: P001"])
        tab.deleteLater()

    def test_extract_clinical_text_creates_single_dialog_when_none(self):
        """With progress=None only one dialog is created for extraction."""
        pending = DocumentRecord(
            id="DOC_000022", patient_id="P001", filename="pending.pdf",
            original_path="/nonexistent/pending.pdf", file_hash="pending",
        )
        repository = _DocumentRepository([pending])
        tab = DocumentsTab()
        tab.set_services({"document_repo": repository})
        tab.load_patient("P001")

        import emr_analyzer.gui.progress_dialog as progress_module

        class _FakeProgressDialog:
            instances = []

            def __init__(self, title="", parent=None):
                self.title = title
                _FakeProgressDialog.instances.append(self)

            def show(self):
                pass

            def setWindowTitle(self, title):
                pass

        _FakeProgressDialog.instances = []
        original = progress_module.ProgressDialog
        progress_module.ProgressDialog = _FakeProgressDialog
        observed = []
        tab._process_documents = lambda ids, **kwargs: observed.append(
            kwargs.get("progress")
        )
        try:
            tab.extract_clinical_text()
        finally:
            progress_module.ProgressDialog = original

        self.assertEqual(len(_FakeProgressDialog.instances), 1)
        self.assertEqual(len(observed), 1)
        self.assertIs(observed[0], _FakeProgressDialog.instances[0])
        tab.deleteLater()

    def test_run_llm_extraction_tolerates_none_progress(self):
        """Parallel LLM workers pass progress=None; internal logs swallowed."""
        tab = DocumentsTab()
        tab.set_services({
            "clinical_text_isolator": _FakeIsolator(_FakeIsolatorResult()),
            "document_llm_client": _FakeLlmClient(),
        })
        tab._save_normalized_clinical_text = lambda doc, result: None
        tab._current_patient_id = "P001"
        document = DocumentRecord(
            id="DOC_000020", patient_id="P001", filename="report.pdf",
            original_path="/nonexistent/report.pdf", file_hash="report",
        )

        events = tab._run_llm_extraction(document, "testo clinico", None)

        self.assertEqual(events, [])
        tab.deleteLater()


if __name__ == "__main__":
    unittest.main()

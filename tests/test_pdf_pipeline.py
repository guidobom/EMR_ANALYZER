from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import fitz

from emr_analyzer.database.engine import DatabaseEngine
from emr_analyzer.database.evidence_repo import EvidenceRepository
from emr_analyzer.database.document_projection_repo import (
    DocumentProjectionRepository,
)
from emr_analyzer.database.migrations import SCHEMA_VERSION, init_database
from emr_analyzer.database.patient_repo import PatientRepository
from emr_analyzer.extraction.clinical_isolator import ClinicalContentIsolator
from emr_analyzer.models import Patient
from emr_analyzer.models.clinical_evidence import ClinicalEvidence
from emr_analyzer.models.document import DocumentRecord
from emr_analyzer.database.document_repo import DocumentRepository
from emr_analyzer.database.clinical_state_repo import ClinicalStateRepository
from emr_analyzer.database.event_repo import EventRepository
from emr_analyzer.clinical.clinical_state import ClinicalStateManager
from emr_analyzer.extraction.qwen_client import QwenClient
from emr_analyzer.extraction.document_consolidator import (
    DocumentClinicalConsolidator,
)
from emr_analyzer.extraction.clinical_text_isolator import ClinicalTextIsolator
from emr_analyzer.config import OLLAMA_CONTEXT_LENGTH
from emr_analyzer.settings import LLMRoleConfig
from emr_analyzer.pipeline.pdf_extractor import (
    PdfExtractionResult, PdfPage, PdfPlumberExtractor,
)
from emr_analyzer.pipeline.sensitive_data import SensitiveDataSanitizer


def _make_table_pdf(path: Path) -> None:
    document = fitz.open()
    page = document.new_page(width=595, height=842)
    x_positions = [50, 250, 340, 440, 545]
    y_positions = [100, 125, 150]
    for x in x_positions:
        page.draw_line((x, y_positions[0]), (x, y_positions[-1]))
    for y in y_positions:
        page.draw_line((x_positions[0], y), (x_positions[-1], y))
    headers = ["Parametro", "Risultato", "Unita", "Riferimento"]
    values = ["Emoglobina", "10.2", "g/dL", "12.0 - 16.0"]
    for index, text in enumerate(headers):
        page.insert_text((x_positions[index] + 4, 117), text, fontsize=8)
    for index, text in enumerate(values):
        page.insert_text((x_positions[index] + 4, 142), text, fontsize=8)
    page.insert_text((50, 200), "La paziente riferisce dispnea da sforzo.", fontsize=10)
    document.save(path)
    document.close()


class FakeStructuredLlm:
    model = "local-test-model"
    is_available = True

    def generate_structured(self, prompt, system, schema):
        return {
            "observations": [
                {
                    "category": "symptom",
                    "normalized_entity": "dispnea da sforzo",
                    "assertion": "present",
                    "temporality": "current",
                    "clinical_status": "active",
                    "observed_date": "2024-03-10",
                    "value_text": None,
                    "numeric_value": None,
                    "unit": None,
                    "source_page": 1,
                    "source_text": "La paziente riferisce dispnea da sforzo.",
                    "confidence": 0.95,
                },
                {
                    "category": "diagnosis",
                    "normalized_entity": "dato inventato",
                    "assertion": "present",
                    "temporality": "current",
                    "source_page": 1,
                    "source_text": "Questa frase non esiste nel documento.",
                    "confidence": 0.99,
                },
            ]
        }


class FailingStructuredLlm:
    model = "local-test-model"
    is_available = True

    def generate_structured(self, prompt, system, schema):
        raise RuntimeError("synthetic Ollama failure")


class FakeConsolidationLlm:
    model = "local-consolidation-test"
    is_available = True

    def __init__(self, response):
        self.response = response

    def generate_structured(self, prompt, system, schema):
        return self.response


class FakeTextLlm:
    model = "plain-text-test"
    is_available = True

    def __init__(
        self,
        responses,
        context_length=16_384,
        max_output_tokens=4_096,
    ):
        self.responses = list(responses)
        self.context_length = context_length
        self.max_output_tokens = max_output_tokens
        self.prompts = []

    def generate_text(self, prompt, system=""):
        self.prompts.append((prompt, system))
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class StubHybridExtractor(PdfPlumberExtractor):
    def _extract_with_pymupdf(self, path, page_numbers=None):
        self.requested_pymupdf_pages = page_numbers
        return PdfExtractionResult(
            source_path=str(path), method="pymupdf_native",
            has_native_text=False,
            pages=[PdfPage(
                page=number, width=595, height=842,
                text="", words=[], tables=[],
            ) for number in sorted(page_numbers or set())],
        )

    def _extract_with_local_ocr(self, path, page_numbers=None):
        self.requested_ocr_pages = page_numbers
        return PdfExtractionResult(
            source_path=str(path), method="local_ocr", has_native_text=False,
            pages=[PdfPage(
                page=2, width=595, height=842,
                text="Testo clinico recuperato dalla scansione.",
                words=[], tables=[],
            )],
        )


class StubNativeFallbackExtractor(PdfPlumberExtractor):
    def _extract_with_pymupdf(self, path, page_numbers=None):
        self.requested_pymupdf_pages = page_numbers
        return PdfExtractionResult(
            source_path=str(path), method="pymupdf_native",
            has_native_text=True,
            pages=[PdfPage(
                page=number, width=595, height=842,
                text="Testo elettronico recuperato nativamente con PyMuPDF.",
                words=[{
                    "text": "Testo", "x0": 1, "top": 2,
                    "x1": 10, "bottom": 12, "doctop": 2,
                }], tables=[],
            ) for number in sorted(page_numbers or set())],
        )

    def _extract_with_local_ocr(self, path, page_numbers=None):
        raise AssertionError("OCR non deve essere chiamato se PyMuPDF ha testo")


class PdfPipelineTest(unittest.TestCase):
    def test_qwen_runtime_status_parses_object_and_legacy_responses(self):
        object_response = SimpleNamespace(models=[SimpleNamespace(
            model="gemma3:12b", size=12_000, size_vram=10_000,
            context_length=32768, expires_at=None,
        )])
        info = QwenClient._find_loaded_model(object_response, "gemma3:12b")
        self.assertEqual(info["size_vram"], 10_000)
        self.assertEqual(info["context_length"], 32768)

        legacy_response = {"models": [{
            "name": "qwen3:14b", "size": 14_000, "size_vram": 14_000,
            "context_length": 32768,
        }]}
        self.assertEqual(
            QwenClient._find_loaded_model(
                legacy_response, "qwen3:14b:latest"
            )["model"],
            "qwen3:14b",
        )
        self.assertIsNone(
            QwenClient._find_loaded_model(legacy_response, "gemma3:12b")
        )

    def test_qwen_warmup_uses_real_context_and_confirms_loaded_model(self):
        observed = {}

        class FakeOllamaClient:
            def __init__(self, host):
                observed["host"] = host

            def chat(self, **request):
                observed["request"] = request
                return SimpleNamespace(
                    message=SimpleNamespace(content="OK")
                )

            def ps(self):
                return {"models": [{
                    "model": "gemma3:12b", "size": 12_000,
                    "size_vram": 11_000,
                    "context_length": OLLAMA_CONTEXT_LENGTH,
                }]}

        fake_ollama = SimpleNamespace(Client=FakeOllamaClient)
        with patch.dict("sys.modules", {"ollama": fake_ollama}):
            result = QwenClient(model="gemma3:12b").warmup("10m")

        self.assertEqual(
            observed["request"]["options"]["num_ctx"],
            OLLAMA_CONTEXT_LENGTH,
        )
        self.assertEqual(observed["request"]["keep_alive"], "10m")
        self.assertFalse(observed["request"]["think"])
        self.assertEqual(result["test_response"], "OK")
        self.assertEqual(result["size_vram"], 11_000)

    def test_qwen_unloads_only_requested_resident_model(self):
        observed = []

        class FakeOllamaClient:
            def __init__(self, host):
                self.host = host

            def ps(self):
                return {"models": [
                    {"model": "gemma3:12b"},
                    {"name": "qwen3:14b"},
                ]}

            def generate(self, **request):
                observed.append(request)
                return {}

        fake_ollama = SimpleNamespace(Client=FakeOllamaClient)
        with patch.dict("sys.modules", {"ollama": fake_ollama}):
            result = QwenClient.unload_models(["gemma3:12b"])

        self.assertEqual(result["unloaded"], ["gemma3:12b"])
        self.assertEqual(result["not_loaded"], [])
        self.assertEqual(result["errors"], {})
        self.assertEqual(observed, [{
            "model": "gemma3:12b",
            "prompt": "",
            "keep_alive": 0,
        }])

    def test_qwen_can_unload_all_resident_models(self):
        observed = []

        class FakeOllamaClient:
            def __init__(self, host):
                self.host = host

            def ps(self):
                return SimpleNamespace(models=[
                    SimpleNamespace(model="gemma3:12b"),
                    SimpleNamespace(name="qwen3:14b"),
                ])

            def generate(self, **request):
                observed.append(request["model"])
                return {}

        fake_ollama = SimpleNamespace(Client=FakeOllamaClient)
        with patch.dict("sys.modules", {"ollama": fake_ollama}):
            result = QwenClient.unload_models()

        self.assertEqual(
            result["unloaded"], ["gemma3:12b", "qwen3:14b"]
        )
        self.assertEqual(observed, ["gemma3:12b", "qwen3:14b"])

    def test_qwen_unload_is_idempotent_when_model_is_not_resident(self):
        class FakeOllamaClient:
            def __init__(self, host):
                self.host = host

            def ps(self):
                return {"models": []}

            def generate(self, **request):
                raise AssertionError("Un modello assente non deve essere caricato")

        fake_ollama = SimpleNamespace(Client=FakeOllamaClient)
        with patch.dict("sys.modules", {"ollama": fake_ollama}):
            result = QwenClient.unload_models(["gemma3:12b"])

        self.assertEqual(result["unloaded"], [])
        self.assertEqual(result["not_loaded"], ["gemma3:12b"])
        self.assertEqual(result["errors"], {})

    def test_qwen_uses_role_generation_parameters(self):
        observed = {}

        class FakeOllamaClient:
            def __init__(self, host):
                observed["host"] = host

            def chat(self, **request):
                observed["request"] = request
                return SimpleNamespace(
                    message=SimpleNamespace(content="testo normalizzato"),
                    done_reason="stop",
                )

        config = LLMRoleConfig(
            model="gemma3:12b", temperature=0.0,
            context_length=16_384, max_output_tokens=2_048,
            top_p=0.75, top_k=15, seed=9,
            keep_alive_minutes=22,
        )
        fake_ollama = SimpleNamespace(Client=FakeOllamaClient)
        with patch.dict("sys.modules", {"ollama": fake_ollama}):
            result = QwenClient(config=config).generate_text("sorgente")

        self.assertEqual(result, "testo normalizzato")
        request = observed["request"]
        self.assertEqual(request["model"], "gemma3:12b")
        self.assertEqual(request["options"], {
            "temperature": 0.0,
            "num_predict": 2_048,
            "num_ctx": 16_384,
            "top_p": 0.75,
            "top_k": 15,
            "seed": 9,
        })
        self.assertEqual(request["keep_alive"], "22m")

    def test_qwen_reads_declared_maximum_model_context(self):
        class FakeOllamaClient:
            def __init__(self, host):
                pass

            def show(self, model):
                return SimpleNamespace(
                    modelinfo={
                        "general.architecture": "gemma3",
                        "gemma3.context_length": 131_072,
                    },
                    capabilities=["completion", "vision"],
                )

        fake_ollama = SimpleNamespace(Client=FakeOllamaClient)
        with patch.dict("sys.modules", {"ollama": fake_ollama}):
            capabilities = QwenClient(
                model="gemma3:12b"
            ).model_capabilities()

        self.assertEqual(capabilities["max_context_length"], 131_072)
        self.assertEqual(capabilities["architecture"], "gemma3")
        self.assertIn("vision", capabilities["capabilities"])

    def test_qwen_plain_text_call_does_not_request_json_schema(self):
        client = QwenClient(model="local-test-model")
        observed = {}

        def fake_generate(prompt, system="", stream=False, format_schema=None):
            observed["format_schema"] = format_schema
            return "Testo clinico normalizzato."

        client._ollama_generate = fake_generate

        output = client.generate_text("sorgente")

        self.assertEqual(output, "Testo clinico normalizzato.")
        self.assertIsNone(observed["format_schema"])

    def test_qwen_plain_text_call_rejects_empty_output(self):
        client = QwenClient(model="local-test-model")
        client._ollama_generate = lambda *args, **kwargs: "   "

        with self.assertRaisesRegex(ValueError, "risposta vuota"):
            client.generate_text("sorgente")

    def test_plain_text_isolator_returns_normalized_text_not_json(self):
        llm = FakeTextLlm([
            "Sintomi: cefalea.\nTerapia: prednisone [[VALORE_B]] mg."
        ])
        result = ClinicalTextIsolator(llm).isolate(
            "[PAGINA 1]\nIl paziente riferisce cefalea. "
            "Assume prednisone 5 mg."
        )

        self.assertIn("Sintomi: cefalea", result.text)
        self.assertFalse(result.text.lstrip().startswith("{"))
        self.assertEqual(result.chunk_count, 1)

    def test_sensitive_data_is_removed_before_and_after_document_llm(self):
        source = (
            "[PAGINA 1]\nMario Rossi riferisce cefalea. "
            "E-mail mario.rossi@example.it."
        )
        llm = FakeTextLlm([
            "Mario Rossi riferisce cefalea. "
            "E-mail mario.rossi@example.it."
        ])

        result = ClinicalTextIsolator(llm).isolate(
            source,
            sensitive_identity={"name": "Mario Rossi"},
        )

        prompt = llm.prompts[0][0]
        self.assertNotIn("Mario Rossi", prompt)
        self.assertNotIn("mario.rossi@example.it", prompt)
        self.assertNotIn("Mario Rossi", result.text)
        self.assertNotIn("mario.rossi@example.it", result.text)
        self.assertIn("[PAZIENTE]", result.text)
        self.assertIn("[EMAIL RIMOSSA]", result.text)
        self.assertGreaterEqual(result.redaction_counts["patient_name"], 2)
        self.assertGreaterEqual(result.redaction_counts["email"], 2)

    def test_sensitive_data_filter_preserves_clinical_dates_and_oral_route(self):
        text = """
Paziente Mario Rossi, nato il 01/02/1960.
Codice fiscale RSSMRA60B01H501Z.
Recapiti: mario.rossi@example.it; tel. +39 333 123 4567.
Sig.ra Rossi rivalutata il 13/03/2025.
Terapia per via orale: prednisone 5 mg.
Via Orale, 5 mg/die.
"""
        result = SensitiveDataSanitizer().sanitize(
            text,
            {
                "name": "Mario Rossi",
                "birth_date": "01/02/1960",
                "fiscal_code": "RSSMRA60B01H501Z",
            },
        )

        self.assertNotIn("Mario Rossi", result.text)
        self.assertNotIn("Rossi", result.text)
        self.assertNotIn("01/02/1960", result.text)
        self.assertNotIn("RSSMRA60B01H501Z", result.text)
        self.assertNotIn("mario.rossi@example.it", result.text)
        self.assertNotIn("333 123 4567", result.text)
        self.assertIn("13/03/2025", result.text)
        self.assertIn("via orale: prednisone 5 mg", result.text)
        self.assertIn("Via Orale, 5 mg/die", result.text)
        self.assertGreaterEqual(result.total, 6)

    def test_plain_text_isolator_rejects_json_and_new_numbers(self):
        with self.assertRaisesRegex(RuntimeError, "JSON invece di testo"):
            ClinicalTextIsolator(FakeTextLlm([
                '{"observations": []}',
                '{"observations": []}',
            ])).isolate("[PAGINA 1]\nCefalea.")

        with self.assertRaisesRegex(
            RuntimeError, "prodotti direttamente dal modello"
        ):
            ClinicalTextIsolator(FakeTextLlm([
                "[[PAGINA_A]]\nPrednisone 50 mg.",
                "[[PAGINA_A]]\nPrednisone 50 mg.",
            ])).isolate("[PAGINA 1]\nPrednisone.")

    def test_plain_text_isolator_never_saves_partial_chunks(self):
        long_source = "Cefalea. " * 1600
        isolator = ClinicalTextIsolator(FakeTextLlm(
            ["Cefalea.", RuntimeError("runner stopped")],
            context_length=6_400,
            max_output_tokens=2_048,
        ))

        with self.assertRaisesRegex(RuntimeError, "1/2 chunk falliti"):
            isolator.isolate(long_source)

    def test_plain_text_isolator_processes_fitting_document_as_one_chunk(self):
        source = (
            "<!-- page:1 -->\nPaziente con cefalea.\n"
            "<!-- page:2 -->\nProsegue terapia."
        )
        llm = FakeTextLlm([
            "Paziente con cefalea.\nProsegue terapia."
        ])

        result = ClinicalTextIsolator(llm).isolate(
            source, document_date="2025-03-13"
        )

        self.assertEqual(result.chunk_count, 1)
        self.assertEqual(len(llm.prompts), 1)
        self.assertNotIn("[PAGINA", result.text)
        self.assertNotIn("page:", result.text)

    def test_plain_text_isolator_restores_dates_and_retries_format_change(self):
        source = (
            "[PAGINA 1]\nIn data 07/02 comparsa di febbre. "
            "PCR 24.12 mg/dl."
        )
        llm = FakeTextLlm([
            "[PAGINA 1]\nIn data 07.02 comparsa di febbre.",
            "In data [[VALORE_B]] comparsa di febbre. "
            "PCR [[VALORE_C]] mg/dl.",
        ])

        result = ClinicalTextIsolator(llm).isolate(
            source, document_date="2025-03-13"
        )

        self.assertIn("07/02", result.text)
        self.assertIn("24.12 mg/dl", result.text)
        self.assertNotIn("07.02", result.text)
        self.assertNotIn("[PAGINA", result.text)
        self.assertEqual(len(llm.prompts), 2)
        self.assertTrue(result.warnings)
        self.assertIn("tentativo precedente", llm.prompts[1][0].lower())
        self.assertIn(
            "non produrre alcuna cifra", llm.prompts[1][0].lower()
        )
        self.assertNotIn("2025-03-13", llm.prompts[0][0])

    def test_numeric_validation_preserves_exact_date_separator(self):
        with self.assertRaisesRegex(ValueError, "non presenti esattamente"):
            ClinicalTextIsolator._validate_output(
                "In data 07.02 comparsa di febbre.",
                "In data 07/02 comparsa di febbre.",
                None,
            )

    def test_numeric_placeholders_cannot_be_duplicated_or_reordered(self):
        source = (
            "[PAGINA 1]\nPrednisone 5 mg dal 07/02; "
            "HER2, PD-L1, Ki-67 e stadio pT3N1."
        )
        protected, replacements = (
            ClinicalTextIsolator._protect_numeric_literals(source)
        )
        self.assertIn("[[PAGINA_A]]", protected)
        self.assertIn("[[VALORE_B]]", protected)
        self.assertIn("[[VALORE_C]]", protected)
        self.assertEqual(
            ClinicalTextIsolator._restore_numeric_literals(
                protected, replacements
            ),
            source,
        )
        self.assertEqual(
            ClinicalTextIsolator._restore_numeric_literals(
                "Valore PCR in riduzione.", {}
            ),
            "Valore PCR in riduzione.",
        )
        self.assertFalse(
            any(character.isdigit() for character in protected)
        )

        with self.assertRaisesRegex(ValueError, "duplicati"):
            ClinicalTextIsolator._restore_numeric_literals(
                "[[PAGINA_A]] [[VALORE_B]] [[VALORE_B]]",
                replacements,
            )
        with self.assertRaisesRegex(ValueError, "riordinato"):
            ClinicalTextIsolator._restore_numeric_literals(
                "[[PAGINA_A]] [[VALORE_C]] [[VALORE_B]]",
                replacements,
            )

    def test_numeric_validation_rejects_changed_embedded_biomarker_number(self):
        with self.assertRaisesRegex(ValueError, "non presenti esattamente"):
            ClinicalTextIsolator._validate_output(
                "Neoplasia HER3 positiva.",
                "Neoplasia HER2 positiva.",
                None,
            )

    def test_final_clinical_text_strips_all_page_marker_formats(self):
        output = (
            "[PAGINA 1]\nAnamnesi clinica.\n\n"
            "<!-- page:2 -->\nTerapia in corso.\n"
            "[PAGINE SORGENTE: 1, 2]"
        )

        cleaned = ClinicalTextIsolator._strip_page_markers(output)

        self.assertEqual(
            cleaned, "Anamnesi clinica.\n\nTerapia in corso."
        )

    def test_retry_for_reordered_placeholders_gets_specific_instruction(self):
        source = "[PAGINA 1]\nPrima 5 mg, successivamente 10 mg."
        llm = FakeTextLlm([
            "Prima [[VALORE_C]], successivamente [[VALORE_B]].",
            "Prima [[VALORE_B]] mg, successivamente [[VALORE_C]] mg.",
        ])

        result = ClinicalTextIsolator(llm).isolate(source)

        self.assertIn("5 mg", result.text)
        self.assertIn("10 mg", result.text)
        self.assertIn("stesso ordine", llm.prompts[1][0].lower())
        self.assertIn("segnaposti riordinati", result.warnings[0])

    def test_failed_whole_document_uses_safe_page_section_fallback(self):
        source = (
            "[PAGINA 1]\nEvento in data 07/02.\n"
            "[PAGINA 2]\nControllo in data 12/03."
        )
        llm = FakeTextLlm([
            "Controllo [[VALORE_D]]. Evento [[VALORE_B]].",
            "Controllo [[VALORE_D]]. Evento [[VALORE_B]].",
            "Evento in data [[VALORE_B]].",
            "Controllo in data [[VALORE_B]].",
        ])

        result = ClinicalTextIsolator(llm).isolate(source)

        self.assertEqual(result.chunk_count, 2)
        self.assertIn("Evento in data 07/02.", result.text)
        self.assertIn("Controllo in data 12/03.", result.text)
        self.assertNotIn("[PAGINA", result.text)
        self.assertTrue(any(
            "fallback per 2 sezioni" in warning
            for warning in result.warnings
        ))

    def test_failed_attempt_log_reports_both_validation_reasons(self):
        source = "[PAGINA 1]\nPrima 5 mg, successivamente 10 mg."
        isolator = ClinicalTextIsolator(FakeTextLlm([
            "[[VALORE_C]] poi [[VALORE_B]].",
            "[[VALORE_C]] poi [[VALORE_B]].",
        ]))

        with self.assertRaisesRegex(
            RuntimeError,
            r"tentativo 1:.*riordinato.*tentativo 2:.*riordinato",
        ):
            isolator.isolate(source)

    def test_validation_error_is_not_a_systemic_batch_failure(self):
        source = "[PAGINA 1]\nPrima 5 mg, successivamente 10 mg."
        isolator = ClinicalTextIsolator(FakeTextLlm([
            "[[VALORE_C]] poi [[VALORE_B]].",
            "[[VALORE_C]] poi [[VALORE_B]].",
        ]))

        with self.assertRaises(RuntimeError) as caught:
            isolator.isolate(source)

        self.assertFalse(caught.exception.systemic)

    def test_compute_error_is_a_systemic_batch_failure(self):
        isolator = ClinicalTextIsolator(FakeTextLlm([
            RuntimeError("llama-server Compute error")
        ]))

        with self.assertRaises(RuntimeError) as caught:
            isolator.isolate("[PAGINA 1]\nCefalea.")

        self.assertTrue(caught.exception.systemic)

    def test_pdfplumber_extracts_geometry_and_tables(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "laboratory.pdf"
            _make_table_pdf(path)
            extractor = PdfPlumberExtractor()

            result = extractor.convert(path)

            self.assertEqual(result.method, "pdfplumber")
            self.assertEqual(result.page_count, 1)
            self.assertTrue(result.has_native_text)
            self.assertGreaterEqual(len(result.tables), 1)
            flattened = " ".join(
                cell for table in result.tables for row in table.rows for cell in row
            ).lower()
            self.assertIn("emoglobina", flattened)
            self.assertTrue(result.pages[0].words)
            page, bbox = result.locate_source(
                "La paziente riferisce dispnea da sforzo.", 1
            )
            self.assertEqual(page, 1)
            self.assertIsNotNone(bbox)

            frames = extractor.export_tables(result)
            self.assertTrue(frames)
            self.assertIn("Parametro", list(frames[0].columns))

    def test_pdfplumber_uses_ocr_only_for_sparse_pages(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "mixed.pdf"
            document = fitz.open()
            first = document.new_page()
            first.insert_text(
                (50, 100),
                "Questa pagina contiene sufficiente testo elettronico clinico.",
            )
            document.new_page()
            document.save(path)
            document.close()

            extractor = StubHybridExtractor()
            result = extractor.convert(path)

            self.assertEqual(extractor.requested_pymupdf_pages, {2})
            self.assertEqual(extractor.requested_ocr_pages, {2})
            self.assertEqual(result.method, "pdfplumber+local_ocr")
            self.assertIn("recuperato dalla scansione", result.pages[1].text)

    def test_pymupdf_native_is_used_before_ocr(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "electronic-livecycle-like.pdf"
            document = fitz.open()
            document.new_page()
            document.save(path)
            document.close()

            extractor = StubNativeFallbackExtractor()
            result = extractor.convert(path)

            self.assertEqual(extractor.requested_pymupdf_pages, {1})
            self.assertEqual(result.method, "pymupdf_native")
            self.assertTrue(result.has_native_text)
            self.assertIn("recuperato nativamente", result.pages[0].text)

    def test_structured_isolator_rejects_unsupported_quotes(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "visit.pdf"
            _make_table_pdf(path)
            result = PdfPlumberExtractor().convert(path)
            isolator = ClinicalContentIsolator(FakeStructuredLlm())

            evidence = isolator.extract(
                result.plain_text,
                "P001",
                "DOC_000001",
                document_date="2024-03-10",
                parsing_result=result,
            )

            self.assertEqual(len(evidence), 1)
            self.assertEqual(evidence[0].category, "symptom")
            self.assertEqual(evidence[0].source_page, 1)
            self.assertIsNotNone(evidence[0].bbox)
            events = isolator.to_events(evidence)
            self.assertEqual(len(events), 1)
            self.assertEqual(events[0].event_type, "symptom")

    def test_structured_isolator_does_not_treat_model_failure_as_empty(self):
        isolator = ClinicalContentIsolator(FailingStructuredLlm())

        with self.assertRaisesRegex(RuntimeError, "risposta strutturata"):
            isolator.extract(
                "--- PAGINA 1 ---\nLa paziente riferisce dispnea.",
                "P001", "DOC_000001", document_date="2024-03-10",
            )

    def test_evidence_store_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db = DatabaseEngine(root / "registry.db")
            init_database(db)
            PatientRepository(db).insert(Patient(id="P001", pseudonym="001"))
            DocumentRepository(db).insert(DocumentRecord(
                id="DOC_000001", patient_id="P001", filename="x.pdf",
                original_path=str(root / "x.pdf"), file_hash="hash",
            ))
            repo = EvidenceRepository(db)
            item = ClinicalEvidence(
                patient_id="P001", document_id="DOC_000001",
                category="laboratory_finding", normalized_entity="emoglobina",
                source_text="Emoglobina 10.2 g/dL", numeric_value=10.2,
                unit="g/dL", source_page=1, bbox=(1, 2, 3, 4),
                confidence=0.99, extraction_method="deterministic_lab",
            )

            repo.replace_document("DOC_000001", [item])
            llm_item = ClinicalEvidence(
                patient_id="P001", document_id="DOC_000001",
                category="symptom", normalized_entity="dispnea",
                source_text="Dispnea", extraction_method="llm_document_projection",
            )
            repo.replace_document_method(
                "DOC_000001", "llm_document_projection", [llm_item]
            )
            replacement_lab = ClinicalEvidence(
                patient_id="P001", document_id="DOC_000001",
                category="laboratory_finding", normalized_entity="emoglobina",
                source_text="Emoglobina 9.8 g/dL", numeric_value=9.8,
                unit="g/dL", extraction_method="deterministic_lab",
            )
            repo.replace_document_method(
                "DOC_000001", "deterministic_lab", [replacement_lab]
            )
            loaded = repo.get_by_document("DOC_000001")

            self.assertEqual(len(loaded), 2)
            by_method = {entry.extraction_method: entry for entry in loaded}
            self.assertEqual(
                by_method["deterministic_lab"].numeric_value, 9.8
            )
            self.assertEqual(
                by_method["llm_document_projection"].normalized_entity,
                "dispnea",
            )
            self.assertEqual(db.execute(
                "SELECT MAX(version) FROM schema_version"
            ).fetchone()[0], SCHEMA_VERSION)

            state = ClinicalStateManager(
                ClinicalStateRepository(db), EventRepository(db),
                evidence_repo=repo,
            ).rebuild_from_events("P001")
            self.assertEqual(len(state.observations), 2)
            self.assertEqual(len(state.lab_trends), 1)

    def test_document_projection_groups_cross_page_evidence_without_loss(self):
        rash_first = ClinicalEvidence(
            patient_id="P001", document_id="DOC_000001", category="toxicity",
            normalized_entity="rash cutaneo", observed_date="2024-03-10",
            source_page=1, source_text="Comparsa di rash cutaneo.",
            bbox=(10, 20, 100, 30), confidence=0.95,
        )
        rash_follow_up = ClinicalEvidence(
            patient_id="P001", document_id="DOC_000001", category="toxicity",
            normalized_entity="eruzione cutanea", observed_date="2024-03-10",
            clinical_status="improving", source_page=3,
            source_text="Eruzione cutanea in miglioramento.", confidence=0.9,
        )
        steroid = ClinicalEvidence(
            patient_id="P001", document_id="DOC_000001",
            category="medication_current", normalized_entity="prednisone",
            observed_date="2024-03-10", source_page=4,
            source_text="In terapia con prednisone.", confidence=0.98,
        )
        response = {
            "observation_groups": [
                {
                    "group_key": "rash",
                    "evidence_ids": [
                        rash_first.evidence_id, rash_follow_up.evidence_id,
                    ],
                    "canonical_evidence_id": rash_first.evidence_id,
                    "group_relation": "evolution",
                },
                {
                    "group_key": "steroid",
                    "evidence_ids": [steroid.evidence_id],
                    "canonical_evidence_id": steroid.evidence_id,
                    "group_relation": "same_entity",
                },
            ],
            "relationships": [{
                "from_group_key": "rash",
                "to_group_key": "steroid",
                "relationship_type": "treated_with",
                "evidence_ids": [rash_first.evidence_id, steroid.evidence_id],
                "confidence": 0.88,
            }],
            "conflicts": [],
        }
        projection = DocumentClinicalConsolidator(
            FakeConsolidationLlm(response)
        ).consolidate(
            [rash_first, rash_follow_up, steroid],
            patient_id="P001", document_id="DOC_000001",
            document_date="2024-03-10", document_type="visita_oncologica",
        )

        self.assertEqual(projection.consolidation_method, "llm_validated")
        self.assertEqual(len(projection.observations), 2)
        self.assertEqual(set(projection.evidence_ids), {
            rash_first.evidence_id, rash_follow_up.evidence_id,
            steroid.evidence_id,
        })
        toxicity = next(
            item for item in projection.observations
            if item.category == "toxicity"
        )
        self.assertEqual(
            {span.page for span in toxicity.source_spans}, {1, 3}
        )
        self.assertTrue(toxicity.requires_review)
        self.assertEqual(len(projection.relationships), 1)
        self.assertEqual(
            projection.relationships[0].relationship_type, "treated_with"
        )
        document_context = QwenClient._select_document_context(
            [projection.to_dict()],
            [{"evidence_id": rash_first.evidence_id}],
        )
        self.assertEqual(len(document_context), 1)
        self.assertEqual(len(document_context[0]["observations"]), 2)

    def test_document_projection_repository_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db = DatabaseEngine(root / "registry.db")
            init_database(db)
            PatientRepository(db).insert(Patient(id="P001", pseudonym="001"))
            DocumentRepository(db).insert(DocumentRecord(
                id="DOC_000001", patient_id="P001", filename="x.pdf",
                original_path=str(root / "x.pdf"), file_hash="hash",
            ))
            evidence = ClinicalEvidence(
                patient_id="P001", document_id="DOC_000001",
                category="symptom", normalized_entity="dispnea",
                source_text="Dispnea a riposo", source_page=2,
            )
            projection = DocumentClinicalConsolidator().consolidate(
                [evidence], "P001", "DOC_000001"
            )
            repo = DocumentProjectionRepository(db)
            repo.replace(projection)
            evidence_repo = EvidenceRepository(db)
            evidence_repo.replace_document("DOC_000001", [evidence])

            loaded = repo.get_by_document("DOC_000001")

            self.assertIsNotNone(loaded)
            self.assertEqual(len(loaded.observations), 1)
            self.assertEqual(loaded.observations[0].source_spans[0].page, 2)
            state = ClinicalStateManager(
                ClinicalStateRepository(db), EventRepository(db),
                evidence_repo=evidence_repo, projection_repo=repo,
            ).rebuild_from_events("P001")
            self.assertEqual(len(state.document_projections), 1)

    def test_document_consolidation_rejects_unknown_evidence_ids(self):
        first = ClinicalEvidence(
            patient_id="P001", document_id="DOC_000001",
            category="symptom", normalized_entity="astenia",
            source_text="Riferisce astenia.", source_page=1,
        )
        second = ClinicalEvidence(
            patient_id="P001", document_id="DOC_000001",
            category="medication_current", normalized_entity="prednisone",
            source_text="Assume prednisone.", source_page=2,
        )
        invalid_response = {
            "observation_groups": [{
                "group_key": "invented",
                "evidence_ids": ["EVD_NON_ESISTENTE"],
                "canonical_evidence_id": "EVD_NON_ESISTENTE",
                "group_relation": "same_entity",
            }],
            "relationships": [],
            "conflicts": [],
        }

        projection = DocumentClinicalConsolidator(
            FakeConsolidationLlm(invalid_response)
        ).consolidate([first, second], "P001", "DOC_000001")

        self.assertEqual(
            projection.consolidation_method, "deterministic_fallback"
        )
        self.assertEqual(set(projection.evidence_ids), {
            first.evidence_id, second.evidence_id,
        })
        self.assertTrue(projection.warnings)

    def test_clinical_query_retrieval_keeps_relevant_evidence(self):
        observations = [
            {"category": "diagnosis", "entity": "ipertensione", "date": "2023-01-01"},
            {"category": "toxicity", "entity": "colite", "date": "2024-01-01"},
            {"category": "medication_current", "entity": "prednisone", "date": "2024-01-02"},
        ]

        selected = QwenClient._select_relevant_observations(
            observations,
            "Estrai tutti gli eventi avversi immunorelati e i trattamenti",
        )

        self.assertIn("colite", {item["entity"] for item in selected})
        self.assertIn("prednisone", {item["entity"] for item in selected})


if __name__ == "__main__":
    unittest.main()

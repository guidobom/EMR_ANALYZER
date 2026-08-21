from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import fitz

from emr_analyzer.extraction.llm_client import LlmClient
from emr_analyzer.extraction.clinical_text_isolator import ClinicalTextIsolator
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


class _FakeBackend:
    """Duck-typed llama.cpp backend recording chat/ensure calls."""

    def __init__(self, chat_result="OK"):
        self.chat_calls = []
        self.ensure_calls = []
        self._chat_result = chat_result
        self._slots = [
            {"id": 0, "n_ctx": 32768, "is_processing": False, "state": 0},
        ]
        self._models = {
            "gemma3-12b": {
                "file": "/models/gemma3-12b.gguf",
                "size_bytes": 12_000,
                "architecture": "gemma3",
                "max_context_length": 131_072,
            },
        }

    def list_models(self):
        return sorted(self._models)

    def model_info(self, name):
        normalized = str(name).removesuffix(":latest").replace(":", "-")
        return self._models.get(normalized)

    def usable(self):
        return True

    def ensure(self, config, progress_cb=None):
        self.ensure_calls.append(config)
        return "http://127.0.0.1:11435"

    def chat(self, config, messages, **kwargs):
        # Mirror the real backend: every generation ensures the server first.
        self.ensure(config)
        self.chat_calls.append((config, messages, kwargs))
        return {"content": self._chat_result, "finish_reason": "stop"}

    def slots(self, config):
        return self._slots

    def stop_model(self, name):
        return True

    def running_model_names(self):
        return []


class PdfPipelineTest(unittest.TestCase):
    def test_qwen_runtime_status_parses_object_and_legacy_responses(self):
        object_response = SimpleNamespace(models=[SimpleNamespace(
            model="gemma3:12b", size=12_000, size_vram=10_000,
            context_length=32768, expires_at=None,
        )])
        info = LlmClient._find_loaded_model(object_response, "gemma3:12b")
        self.assertEqual(info["size_vram"], 10_000)
        self.assertEqual(info["context_length"], 32768)

        legacy_response = {"models": [{
            "name": "qwen3:14b", "size": 14_000, "size_vram": 14_000,
            "context_length": 32768,
        }]}
        self.assertEqual(
            LlmClient._find_loaded_model(
                legacy_response, "qwen3:14b:latest"
            )["model"],
            "qwen3:14b",
        )
        self.assertIsNone(
            LlmClient._find_loaded_model(legacy_response, "gemma3:12b")
        )

    def test_qwen_runtime_status_parses_slots_list(self):
        info = LlmClient._find_loaded_model(
            [
                {"id": 0, "n_ctx": 16384, "is_processing": True},
                {"id": 1, "n_ctx": 16384, "is_processing": False},
            ],
            "qwen3:14b",
        )
        self.assertEqual(info["context_length"], 16384)
        self.assertEqual(info["slots"], 2)
        self.assertEqual(info["active_slots"], 1)
        self.assertTrue(info["processing"])
        second_active = LlmClient._find_loaded_model(
            [
                {"id": 0, "n_ctx": 16384, "is_processing": False},
                {"id": 1, "n_ctx": 16384, "is_processing": True},
            ],
            "qwen3:14b",
        )
        self.assertTrue(second_active["processing"])
        self.assertEqual(second_active["active_slots"], 1)
        self.assertIsNone(LlmClient._find_loaded_model([], "qwen3:14b"))

    def test_qwen_warmup_uses_real_context_and_confirms_loaded_model(self):
        backend = _FakeBackend()
        result = LlmClient(model="gemma3:12b", backend=backend).warmup("10m")

        config, messages, request = backend.chat_calls[0]
        self.assertEqual(
            messages[0]["content"],
            "Test tecnico di disponibilità. Rispondi soltanto OK.",
        )
        self.assertEqual(request["max_tokens"], 8)
        # The server is spawned with the configured role context so the
        # first clinical call does not reload a different runner.
        self.assertEqual(config.context_length, backend.ensure_calls[0].context_length)
        self.assertEqual(result["test_response"], "OK")
        self.assertEqual(result["context_length"], 32768)
        self.assertEqual(result["size_vram"], 12_000)

    def test_qwen_uses_role_generation_parameters(self):
        backend = _FakeBackend(chat_result="testo normalizzato")
        config = LLMRoleConfig(
            model="gemma3:12b", temperature=0.0,
            context_length=16_384, max_output_tokens=2_048,
            top_p=0.75, top_k=15, seed=9,
            keep_alive_minutes=22, parallel_workers=4,
        )
        result = LlmClient(config=config, backend=backend).generate_text(
            "sorgente"
        )

        self.assertEqual(result, "testo normalizzato")
        _, messages, request = backend.chat_calls[0]
        self.assertEqual(
            messages, [{"role": "user", "content": "sorgente"}]
        )
        self.assertEqual(request["temperature"], 0.0)
        self.assertEqual(request["top_p"], 0.75)
        self.assertEqual(request["top_k"], 15)
        self.assertEqual(request["seed"], 9)
        self.assertEqual(request["max_tokens"], 2_048)
        self.assertIsNone(request["response_format"])
        # Worker slots flow into the server key/ensure path.
        self.assertEqual(backend.ensure_calls[0].parallel_workers, 4)
        self.assertEqual(backend.ensure_calls[0].context_length, 16_384)

    def test_qwen_reads_declared_maximum_model_context(self):
        capabilities = LlmClient(
            model="gemma3:12b", backend=_FakeBackend()
        ).model_capabilities()

        self.assertEqual(capabilities["max_context_length"], 131_072)
        self.assertEqual(capabilities["architecture"], "gemma3")

    def test_qwen_plain_text_call_does_not_request_json_schema(self):
        client = LlmClient(model="local-test-model")
        observed = {}

        def fake_generate(prompt, system="", stream=False,
                          response_format=None, seed=None,
                          temperature=None):
            observed["response_format"] = response_format
            return "Testo clinico normalizzato."

        client._generate = fake_generate

        output = client.generate_text("sorgente")

        self.assertEqual(output, "Testo clinico normalizzato.")
        self.assertIsNone(observed["response_format"])

    def test_qwen_plain_text_call_rejects_empty_output(self):
        client = LlmClient(model="local-test-model")
        client._generate = lambda *args, **kwargs: "   "

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

        # A duplicated insertion is removed deterministically: the first
        # occurrence wins, the repeated one disappears.
        repaired = ClinicalTextIsolator._restore_numeric_literals(
            "[[PAGINA_A]] [[VALORE_B]] [[VALORE_B]]",
            replacements,
        )
        self.assertIn(replacements["[[PAGINA_A]]"], repaired)
        self.assertEqual(repaired.count(replacements["[[VALORE_B]]"]), 1)
        self.assertNotIn(replacements["[[VALORE_C]]"], repaired)
        # A duplication whose removal breaks the source order is rejected.
        with self.assertRaisesRegex(ValueError, "duplicati"):
            ClinicalTextIsolator._restore_numeric_literals(
                "[[VALORE_B]] [[PAGINA_A]] [[VALORE_B]]",
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

if __name__ == "__main__":
    unittest.main()

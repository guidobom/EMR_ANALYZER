"""Tests for corrective-retry sampling variation in the clinical isolator."""

from __future__ import annotations

import re
import unittest
from unittest import mock

from emr_analyzer.extraction.clinical_text_isolator import ClinicalTextIsolator
from emr_analyzer.extraction.llm_client import LlmClient


class FlakyLlm:
    """Fails validation N times with reordered placeholders, then passes."""

    def __init__(self, fail_times: int):
        self.fail_times = fail_times
        self.calls: list[dict] = []
        self.seed = 42
        self.temperature = 0.1
        self.is_available = True
        self.model = "test"

    def generate_text(self, prompt, system="", seed=None, temperature=None):
        self.calls.append({"seed": seed, "temperature": temperature})
        # Only the real placeholders of the source section, not the
        # literal [[VALORE_X]] examples in the instruction block.
        source_part = prompt.split("TESTO SORGENTE:", 1)[1]
        placeholders = re.findall(r"\[\[VALORE_[A-Z]+\]\]", source_part)
        if len(self.calls) <= self.fail_times:
            swapped = list(placeholders)
            if len(swapped) >= 2:
                swapped[0], swapped[1] = swapped[1], swapped[0]
            return (
                "Cefalea persistente. Terapia: "
                + " ".join(swapped) + "."
            )
        return (
            "Cefalea persistente. Terapia: "
            + " ".join(placeholders) + "."
        )


class RetrySamplingVariationTest(unittest.TestCase):
    def test_first_attempt_uses_configured_params_retries_vary(self):
        llm = FlakyLlm(fail_times=1)
        result = ClinicalTextIsolator(llm).isolate(
            "Il paziente riferisce cefalea.\nAssume prednisone 5 mg dal 07/02."
        )
        # Attempt 1 failed (duplicated placeholder), attempt 2 succeeded.
        self.assertEqual(len(llm.calls), 2)
        # First attempt: no overrides — the configured sampling is used.
        self.assertIsNone(llm.calls[0]["seed"])
        self.assertIsNone(llm.calls[0]["temperature"])
        # Corrective retry: seed advanced, temperature raised internally.
        self.assertEqual(llm.calls[1]["seed"], 43)
        self.assertGreaterEqual(llm.calls[1]["temperature"], 0.4)
        self.assertIn("5", result.text)

    def test_all_retries_fail_still_raises(self):
        llm = FlakyLlm(fail_times=5)
        with self.assertRaises(Exception):
            ClinicalTextIsolator(llm).isolate(
                "Il paziente riferisce cefalea.\nAssume prednisone 5 mg dal 07/02."
            )
        # Both attempts ran; the retry got different sampling.
        self.assertEqual(len(llm.calls), 2)
        self.assertEqual(llm.calls[1]["seed"], 43)

    def test_clients_without_seed_keep_working(self):
        # Duck-typed clients without sampling attributes (test fakes) must
        # not receive unexpected keyword arguments.
        llm = FlakyLlm(fail_times=0)
        llm.seed = None
        llm.temperature = None
        result = ClinicalTextIsolator(llm).isolate(
            "Il paziente riferisce cefalea.\nAssume prednisone 5 mg dal 07/02."
        )
        self.assertEqual(len(llm.calls), 1)
        self.assertIn("5", result.text)


class GenerateTextOverrideTest(unittest.TestCase):
    def test_overrides_forwarded_to_backend(self):
        observed = {}

        class FakeBackend:
            def chat(self, config, messages, **kwargs):
                observed.update(kwargs)
                return {"content": "ok", "finish_reason": "stop"}

            def ensure(self, config, progress_cb=None):
                return "http://fake"

        client = LlmClient(model="local-test", backend=FakeBackend())
        client.generate_text("prompt", seed=99, temperature=0.7)
        self.assertEqual(observed["seed"], 99)
        self.assertEqual(observed["temperature"], 0.7)

        observed.clear()
        client.generate_text("prompt")
        self.assertEqual(observed["seed"], client.seed)
        self.assertEqual(observed["temperature"], client.temperature)


if __name__ == "__main__":
    unittest.main()


class PlaceholderAlphabetTest(unittest.TestCase):
    """Placeholder id design: sequential for small chunks, permuted for large."""

    def test_small_chunk_keeps_sequential_ids(self):
        source = "Valore 1 e 2 e 3."
        protected, replacements = ClinicalTextIsolator._protect_numeric_literals(
            source
        )
        self.assertIn("[[VALORE_A]]", protected)
        self.assertIn("[[VALORE_C]]", protected)

    def test_large_chunk_permutes_alphabet(self):
        numbers = " ".join(str(i) for i in range(20))
        source = f"Esami: {numbers}."
        protected, _ = ClinicalTextIsolator._protect_numeric_literals(source)
        tokens = re.findall(r"\[\[VALORE_[A-Z]+\]\]", protected)
        # Sequential would be A..T; a permutation must differ somewhere.
        sequential = [f"[[VALORE_{chr(65 + i)}]]" for i in range(20)]
        self.assertNotEqual(tokens, sequential)

    def test_permutation_is_deterministic_per_source(self):
        numbers = " ".join(str(i) for i in range(20))
        source = f"Esami: {numbers}."
        first, _ = ClinicalTextIsolator._protect_numeric_literals(source)
        second, _ = ClinicalTextIsolator._protect_numeric_literals(source)
        self.assertEqual(first, second)
        other = f"Esami: {numbers}. diverso"
        third, _ = ClinicalTextIsolator._protect_numeric_literals(other)
        self.assertNotEqual(first, third)

    def test_restore_round_trips_with_permuted_alphabet(self):
        numbers = " ".join(str(i) for i in range(20))
        source = f"Esami: {numbers}."
        protected, replacements = ClinicalTextIsolator._protect_numeric_literals(
            source
        )
        restored = protected
        for token, original in replacements.items():
            restored = restored.replace(token, original)
        self.assertEqual(restored, source)


class DuplicateRepairTest(unittest.TestCase):
    """Deterministic repair of a single token-substitution error."""

    def _protected(self):
        source = "[PAGINA 1]\nPrednisone 5 mg dal 07/02; PCR 10 mg/dl."
        protected, replacements = (
            ClinicalTextIsolator._protect_numeric_literals(source)
        )
        return protected, replacements, source

    def test_duplicated_insertion_is_removed(self):
        # The model inserted token[2] once more (insertion model): the
        # duplicate is dropped and the remaining tokens stay exact.
        _, replacements, _ = self._protected()
        tokens = list(replacements)
        wrong = " ".join(tokens[:3] + [tokens[2]])
        restored = ClinicalTextIsolator._restore_numeric_literals(
            wrong, replacements
        )
        self.assertEqual(restored.count(replacements[tokens[2]]), 1)
        self.assertEqual(restored.count(replacements[tokens[3]]), 0)

    def test_unrepairable_duplication_still_rejected(self):
        # Dropping the duplicate would break the source order: ambiguous.
        _, replacements, _ = self._protected()
        tokens = list(replacements)
        wrong = " ".join([tokens[2], tokens[0], tokens[2]])
        with self.assertRaisesRegex(ValueError, "duplicati"):
            ClinicalTextIsolator._restore_numeric_literals(
                wrong, replacements
            )

    def test_filtered_value_plus_duplicate_is_repaired(self):
        # The model filtered t1 away and duplicated t2: the duplicate is
        # dropped, the filtered value stays absent.
        _, replacements, _ = self._protected()
        tokens = list(replacements)
        wrong = " ".join([tokens[0], tokens[2], tokens[2]])
        restored = ClinicalTextIsolator._restore_numeric_literals(
            wrong, replacements
        )
        self.assertEqual(restored.count(replacements[tokens[1]]), 0)
        self.assertEqual(restored.count(replacements[tokens[2]]), 1)
        self.assertEqual(restored.count(replacements[tokens[3]]), 0)

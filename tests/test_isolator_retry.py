"""Tests for corrective-retry sampling variation in the clinical isolator."""

from __future__ import annotations

import re
import unittest
from unittest import mock

from emr_analyzer.extraction.clinical_text_isolator import ClinicalTextIsolator
from emr_analyzer.extraction.llm_client import LlmClient


class FlakyLlm:
    """Fails validation N times with duplicated placeholders, then passes."""

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
            first = placeholders[0]
            return (
                "Cefalea persistente. Terapia: "
                f"{first} {first}."
            )
        return (
            "Cefalea persistente. Terapia: "
            + " ".join(placeholders) + "."
        )


class RetrySamplingVariationTest(unittest.TestCase):
    def test_first_attempt_uses_configured_params_retries_vary(self):
        llm = FlakyLlm(fail_times=1)
        result = ClinicalTextIsolator(llm).isolate(
            "Il paziente riferisce cefalea.\nAssume prednisone 5 mg."
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
                "Il paziente riferisce cefalea.\nAssume prednisone 5 mg."
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
            "Il paziente riferisce cefalea.\nAssume prednisone 5 mg."
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

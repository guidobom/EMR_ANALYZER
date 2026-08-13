"""Regression tests for LlmClient JSON parsing and golden few-shot prompt
injection."""

from __future__ import annotations

import unittest

from emr_analyzer.extraction.llm_client import LlmClient


class TestParseJsonEntries(unittest.TestCase):
    """``_parse_json_entries`` must always return a list.

    Regression: the ``{"entries": ...}`` dict branch used to return the raw
    value unchecked; ``{"entries": null}`` crashed the caller's iteration.
    """

    def test_plain_list(self):
        raw = '[{"description": "a"}, {"description": "b"}]'
        self.assertEqual(
            LlmClient._parse_json_entries(raw),
            [{"description": "a"}, {"description": "b"}],
        )

    def test_fenced_list(self):
        raw = '```json\n[{"description": "a"}]\n```'
        self.assertEqual(
            LlmClient._parse_json_entries(raw), [{"description": "a"}]
        )

    def test_dict_with_entries_list(self):
        raw = '{"entries": [{"description": "a"}], "meta": 1}'
        self.assertEqual(
            LlmClient._parse_json_entries(raw), [{"description": "a"}]
        )

    def test_dict_with_null_entries_returns_empty_list(self):
        # The pre-fix bug: returned None, crashing the caller's iteration.
        raw = '{"entries": null}'
        self.assertEqual(LlmClient._parse_json_entries(raw), [])

    def test_dict_with_string_entries_returns_empty_list(self):
        raw = '{"entries": "not a list"}'
        self.assertEqual(LlmClient._parse_json_entries(raw), [])

    def test_entries_are_filtered_by_description(self):
        raw = '{"entries": [{"description": "a"}, {"other": 1}]}'
        self.assertEqual(
            LlmClient._parse_json_entries(raw), [{"description": "a"}]
        )

    def test_empty_and_garbage(self):
        self.assertEqual(LlmClient._parse_json_entries(""), [])
        self.assertEqual(LlmClient._parse_json_entries("not json"), [])
        self.assertEqual(LlmClient._parse_json_entries("42"), [])


class TestGoldenSectionInjection(unittest.TestCase):
    """The golden few-shot section must appear (only) when examples are
    passed, between REGISTRO CLINICO ATTUALE and the following section."""

    def _client(self):
        client = LlmClient()  # network-safe: OFFLINE_MODE, no Ollama call
        captured = {}

        def fake_generate(user_prompt, system_prompt):
            captured["prompt"] = user_prompt
            return '[{"description": "x", "category": "symptom", "date_observed": "2024-01-01", "status": "active", "source_text": "t", "confidence": 0.5}]'

        client.generate_text = fake_generate
        return client, captured

    @staticmethod
    def _example(category, description):
        return {
            "date_observed": "2024-01-10",
            "date_resolved": None,
            "category": category,
            "description": description,
            "status": "active",
            "confidence": 0.9,
        }

    def test_timeline_injects_section_when_examples(self):
        client, captured = self._client()
        client.extract_timeline_entries(
            "Testo da analizzare.", "",
            golden_examples=[self._example("diagnosis", "Adenocarcinoma")],
        )
        prompt = captured["prompt"]
        self.assertIn("ESEMPI DI OUTPUT CORRETTO", prompt)
        self.assertIn("Adenocarcinoma", prompt)
        self.assertLess(
            prompt.index("REGISTRO CLINICO ATTUALE"),
            prompt.index("ESEMPI DI OUTPUT CORRETTO"),
        )
        self.assertLess(
            prompt.index("ESEMPI DI OUTPUT CORRETTO"),
            prompt.index("REGOLE GENERALI"),
        )

    def test_timeline_no_section_without_examples(self):
        client, captured = self._client()
        client.extract_timeline_entries("Testo da analizzare.", "")
        self.assertNotIn("ESEMPI DI OUTPUT CORRETTO", captured["prompt"])

    def test_discharge_injects_section_when_examples(self):
        client, captured = self._client()
        client.extract_from_discharge_letter(
            "Lettera di dimissione.", "2024-03-01",
            golden_examples=[self._example("discharge", "Dimissione ok")],
        )
        prompt = captured["prompt"]
        self.assertIn("ESEMPI DI OUTPUT CORRETTO", prompt)
        self.assertIn("Dimissione ok", prompt)
        self.assertLess(
            prompt.index("REGISTRO CLINICO ATTUALE"),
            prompt.index("ESEMPI DI OUTPUT CORRETTO"),
        )
        self.assertLess(
            prompt.index("ESEMPI DI OUTPUT CORRETTO"),
            prompt.index("STRUTTURA TIPICA"),
        )

    def test_discharge_filters_foreign_category(self):
        """A biomarker example is not in the discharge category set and is
        dropped, leaving no section."""
        client, captured = self._client()
        client.extract_from_discharge_letter(
            "Lettera di dimissione.", "2024-03-01",
            golden_examples=[self._example("biomarker", "BRAF mutato")],
        )
        self.assertNotIn("ESEMPI DI OUTPUT CORRETTO", captured["prompt"])


if __name__ == "__main__":
    unittest.main()

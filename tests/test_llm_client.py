"""Regression tests for LlmClient JSON parsing helpers."""

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


if __name__ == "__main__":
    unittest.main()

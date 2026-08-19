"""Tests for the chunked full-registry irAE analysis."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from emr_analyzer.clinical import irae_analysis
from emr_analyzer.gui.workers import format_registry_context


def make_entry(entry_id: str, description: str = "descrizione") -> dict:
    return {
        "entry_id": entry_id,
        "date_observed": "2023-05-01",
        "category": "toxicity",
        "description": description,
    }


class FormatRegistryContextTest(unittest.TestCase):
    def test_includes_citable_ids(self):
        lines = format_registry_context([
            make_entry("T1"), make_entry("T2"),
        ])
        self.assertIn("[#T1] [2023-05-01] [toxicity] descrizione", lines)
        self.assertIn("[#T2]", lines)

    def test_respects_limit(self):
        entries = [make_entry(f"T{i}") for i in range(150)]
        lines = format_registry_context(entries, limit=100).split("\n")
        self.assertEqual(len(lines), 100)
        self.assertIn("[#T149]", lines[-1])
        self.assertNotIn("[#T49]", lines[0])

    def test_missing_fields_fall_back(self):
        lines = format_registry_context([{"description": "solo testo"}])
        self.assertIn("[#?]", lines)
        self.assertIn("solo testo", lines)


class ChunkingTest(unittest.TestCase):
    def test_chunks_split_by_budget_with_overlap(self):
        entries = [
            make_entry(f"T{i}", "x" * 300) for i in range(30)
        ]
        chunks = irae_analysis.chunk_entries(
            entries, chunk_chars=1500, overlap=2
        )
        self.assertGreater(len(chunks), 1)
        # Overlap: the tail of chunk N heads chunk N+1.
        self.assertEqual(chunks[0][-2:], chunks[1][:2])
        # Every entry appears at least once.
        seen = {e["entry_id"] for chunk in chunks for e in chunk}
        self.assertEqual(len(seen), 30)

    def test_small_registry_single_chunk(self):
        chunks = irae_analysis.chunk_entries(
            [make_entry("T1")], chunk_chars=12000
        )
        self.assertEqual(len(chunks), 1)


class AnalysisPlanTest(unittest.TestCase):
    def test_plan_prompts_carry_ids_and_protocol(self):
        entries = [make_entry("T1"), make_entry("T2")]
        prompts = irae_analysis.build_analysis_plan(
            entries, "profilo", "PROTOCOLLO TEST"
        )
        self.assertEqual(len(prompts), 1)
        self.assertIn("[#T1]", prompts[0])
        self.assertIn("[#T2]", prompts[0])
        self.assertIn("PROTOCOLLO TEST", prompts[0])
        self.assertIn("parte 1/1", prompts[0])
        self.assertIn("profilo", prompts[0])

    def test_plan_parts_numbered(self):
        entries = [make_entry(f"T{i}", "x" * 500) for i in range(30)]
        prompts = irae_analysis.build_analysis_plan(
            entries, "", "P",
        )
        self.assertGreater(len(prompts), 1)
        self.assertIn(f"parte {len(prompts)}/{len(prompts)}", prompts[-1])


class PromptFileTest(unittest.TestCase):
    def test_ensure_prompt_creates_from_template(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = irae_analysis.ensure_prompt(Path(tmp) / "irae.txt")
            self.assertTrue(path.exists())
            text = path.read_text(encoding="utf-8")
            self.assertIn("irAE", text)
            self.assertIn("| ID | Organo |", text)

    def test_ensure_prompt_keeps_user_edits(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "irae.txt"
            path.write_text("PROTOCOLLO PERSONALE", encoding="utf-8")
            result = irae_analysis.ensure_prompt(path)
            self.assertEqual(
                result.read_text(encoding="utf-8"), "PROTOCOLLO PERSONALE"
            )

    def test_load_prompt_missing_returns_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(
                irae_analysis.load_prompt(Path(tmp) / "assente.txt"), ""
            )


if __name__ == "__main__":
    unittest.main()

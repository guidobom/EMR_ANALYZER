"""Tests for the golden few-shot injection (Phase 4 follow-up).

Covers:
* ``select_examples`` — exclusion of the current patient, stratification by
  category, recency fill, cap, determinism;
* ``format_examples_section`` — empty section, header, de-identified fields
  (never fenced JSON / source text / ids / patient_id), category filter;
* ``ClinicalHistoryBuilder._load_golden_examples`` — real-DB fetching,
  disabled toggle, missing repo, raising repo;
* ``ClinicalHistoryBuilder._extract_for_document`` — examples are forwarded
  to the LLM as keyword arguments in both the timeline and the discharge
  branches.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from emr_analyzer.clinical.clinical_history_builder import (
    ClinicalHistoryBuilder,
)
from emr_analyzer.database.engine import DatabaseEngine
from emr_analyzer.database.migrations import init_database
from emr_analyzer.database.patient_repo import PatientRepository
from emr_analyzer.database.timeline_repo import TimelineRepository
from emr_analyzer.extraction.golden_fewshot import (
    DISCHARGE_ALLOWED_CATEGORIES,
    format_examples_section,
    select_examples,
)
from emr_analyzer.models import Patient
from emr_analyzer.models.clinical_timeline import ClinicalTimelineEntry


def make_entry(entry_id, patient_id, date, category, description,
               is_golden=1):
    return ClinicalTimelineEntry(
        entry_id=entry_id,
        patient_id=patient_id,
        date_observed=date,
        category=category,
        description=description,
        is_golden=is_golden,
    )


class FakeLlm:
    """Stub of the Clinical State LLM that records golden_examples."""

    is_available = True

    def __init__(self):
        self.timeline_calls = []
        self.discharge_calls = []

    def extract_timeline_entries(self, text, registry_summary,
                                 document_date=None, *, golden_examples=None):
        self.timeline_calls.append(golden_examples)
        return {"entries": []}

    def extract_from_discharge_letter(self, text, document_date=None,
                                      registry_summary="",
                                      *, golden_examples=None):
        self.discharge_calls.append(golden_examples)
        return {"entries": []}


class RaisingRepo:
    def get_golden(self, patient_id=None):
        raise RuntimeError("synthetic DB failure")


class SelectExamplesTest(unittest.TestCase):
    def test_excludes_current_patient(self):
        entries = [
            make_entry("CTL_1", "P001", "2024-01-01", "diagnosis", "A"),
            make_entry("CTL_2", "P002", "2024-01-02", "treatment", "B"),
            make_entry("CTL_3", "P003", "2024-01-03", "surgery", "C"),
        ]
        result = select_examples(entries, current_patient_id="P001")
        self.assertEqual(
            {e["patient_id"] for e in result}, {"P002", "P003"}
        )

    def test_empty_input(self):
        self.assertEqual(select_examples([], "P001"), [])
        self.assertEqual(
            select_examples([make_entry("CTL_1", "P001", "2024-01-01",
                                        "diagnosis", "A")], "P001"),
            [],
        )

    def test_cap_at_max_examples(self):
        entries = [
            make_entry(f"CTL_{i}", "P002", f"2024-01-{i:02d}",
                       "other", f"d{i}")
            for i in range(1, 15)
        ]
        result = select_examples(entries, "P001", max_examples=10)
        self.assertEqual(len(result), 10)

    def test_stratification_then_fill(self):
        entries = [
            make_entry("CTL_1", "P002", "2024-05-01", "A", "A-recent"),
            make_entry("CTL_2", "P002", "2024-04-01", "A", "A-older"),
            make_entry("CTL_3", "P002", "2024-06-01", "B", "B"),
            make_entry("CTL_4", "P002", "2024-03-01", "C", "C"),
        ]
        # Recency order: CTL_3(B), CTL_1(A), CTL_2(A), CTL_4(C).
        # Stratification -> [CTL_3, CTL_1, CTL_4]; fill -> + CTL_2.
        result = select_examples(entries, "P001", max_examples=4)
        self.assertEqual(
            [e["entry_id"] for e in result],
            ["CTL_3", "CTL_1", "CTL_4", "CTL_2"],
        )
        self.assertEqual([e["category"] for e in result[:3]],
                         ["B", "A", "C"])

    def test_stratification_respects_cap(self):
        entries = [
            make_entry("CTL_1", "P002", "2024-05-01", "A", "A"),
            make_entry("CTL_2", "P002", "2024-06-01", "B", "B"),
            make_entry("CTL_3", "P002", "2024-07-01", "C", "C"),
        ]
        result = select_examples(entries, "P001", max_examples=2)
        self.assertEqual(
            [e["entry_id"] for e in result], ["CTL_3", "CTL_2"]
        )

    def test_deterministic(self):
        entries = [
            make_entry("CTL_1", "P002", "2024-05-01", "A", "A"),
            make_entry("CTL_2", "P002", "2024-06-01", "B", "B"),
        ]
        first = select_examples(entries, "P001")
        second = select_examples(list(reversed(entries)), "P001")
        self.assertEqual(first, second)


class FormatSectionTest(unittest.TestCase):
    def test_empty(self):
        self.assertEqual(format_examples_section([]), "")
        self.assertEqual(format_examples_section(None), "")

    def test_header_and_description(self):
        section = format_examples_section([
            make_entry("CTL_1", "P002", "2024-01-10", "diagnosis",
                       "Adenocarcinoma polmonare").to_dict(),
        ])
        self.assertIn("ESEMPI DI OUTPUT CORRETTO", section)
        self.assertIn("Adenocarcinoma polmonare", section)
        self.assertIn("[2024-01-10] [diagnosis]", section)

    def test_only_deidentified_fields(self):
        entry = make_entry("CTL_1", "P002", "2024-01-10", "diagnosis",
                           "Adenocarcinoma").to_dict()
        entry["source_texts"] = ["testo sorgente identificante"]
        entry["source_document_ids"] = ["DOC_1"]
        entry["merged_into_ids"] = ["CTL_0"]
        section = format_examples_section([entry])
        # Never render identifying / noisy fields (E1/E5).
        self.assertNotIn("testo sorgente identificante", section)
        self.assertNotIn("DOC_1", section)
        self.assertNotIn("CTL_1", section)          # entry_id omitted
        self.assertNotIn("P002", section)           # patient_id omitted
        self.assertNotIn("```json", section)        # never fenced (E1)

    def test_category_filter_empties(self):
        entry = make_entry("CTL_1", "P002", "2024-01-10", "biomarker",
                           "BRAF mutato").to_dict()
        # biomarker is not in the discharge category set.
        section = format_examples_section(
            [entry], allowed_categories=DISCHARGE_ALLOWED_CATEGORIES
        )
        self.assertEqual(section, "")

    def test_category_filter_keeps_aligned(self):
        entry = make_entry("CTL_1", "P002", "2024-01-10", "discharge",
                           "Dimissione migliorato").to_dict()
        section = format_examples_section(
            [entry], allowed_categories=DISCHARGE_ALLOWED_CATEGORIES
        )
        self.assertIn("Dimissione migliorato", section)


class LoadGoldenExamplesTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.db = DatabaseEngine(Path(self._tmp.name) / "registry.db")
        init_database(self.db)
        self.patient_repo = PatientRepository(self.db)
        self.patient_repo.insert(Patient(id="P001", pseudonym="001"))
        self.patient_repo.insert(Patient(id="P002", pseudonym="002"))
        self.repo = TimelineRepository(self.db)
        self.repo.save_batch([
            make_entry("CTL_1", "P001", "2024-01-10", "diagnosis", "own"),
            make_entry("CTL_2", "P002", "2024-01-11", "treatment", "other"),
        ])
        self.builder = ClinicalHistoryBuilder(
            timeline_repo=self.repo, document_repo=None, cs_repo=None,
            clinical_state_llm_client=FakeLlm(), audit_repo=None,
        )

    def tearDown(self):
        self.db.close()
        self._tmp.cleanup()

    def test_loads_others_not_current(self):
        result = self.builder._load_golden_examples("P001")
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["patient_id"], "P002")
        self.assertEqual(result[0]["description"], "other")

    def test_disabled_toggle_returns_empty(self):
        with patch(
            "emr_analyzer.clinical.clinical_history_builder."
            "GOLDEN_FEWSHOT_ENABLED", False
        ):
            self.assertEqual(self.builder._load_golden_examples("P001"), [])

    def test_missing_repo_returns_empty(self):
        builder = ClinicalHistoryBuilder(
            timeline_repo=None, document_repo=None, cs_repo=None,
            clinical_state_llm_client=FakeLlm(), audit_repo=None,
        )
        self.assertEqual(builder._load_golden_examples("P001"), [])

    def test_raising_repo_returns_empty(self):
        builder = ClinicalHistoryBuilder(
            timeline_repo=RaisingRepo(), document_repo=None, cs_repo=None,
            clinical_state_llm_client=FakeLlm(), audit_repo=None,
        )
        self.assertEqual(builder._load_golden_examples("P001"), [])


class ExtractForDocumentTest(unittest.TestCase):
    def _builder(self, llm):
        return ClinicalHistoryBuilder(
            timeline_repo=None, document_repo=None, cs_repo=None,
            clinical_state_llm_client=llm, audit_repo=None,
        )

    def test_timeline_branch_forwards_examples(self):
        llm = FakeLlm()
        builder = self._builder(llm)
        examples = [make_entry("CTL_1", "P002", "2024-01-10", "diagnosis",
                               "A").to_dict()]
        builder._extract_for_document(
            {"text": "Referto.", "document_date": "2024-03-01",
             "document_type": "visita_specialistica"},
            "registry", examples,
        )
        self.assertEqual(llm.timeline_calls, [examples])
        self.assertEqual(llm.discharge_calls, [])

    def test_discharge_branch_forwards_examples(self):
        llm = FakeLlm()
        builder = self._builder(llm)
        examples = [make_entry("CTL_1", "P002", "2024-01-10", "discharge",
                               "A").to_dict()]
        builder._extract_for_document(
            {"text": "Lettera di dimissione: il paziente viene dimesso.",
             "document_date": "2024-03-01",
             "document_type": "lettera_dimissione"},
            "registry", examples,
        )
        self.assertEqual(llm.discharge_calls, [examples])
        self.assertEqual(llm.timeline_calls, [])


if __name__ == "__main__":
    unittest.main()

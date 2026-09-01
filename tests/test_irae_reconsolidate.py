"""Tests for ``emr_analyzer.clinical.irae_reconsolidate``.

``reconsolidate_report`` re-runs ONLY the Layer 4 consolidation of a saved
``irae_report.json`` (the per-organ findings + anchor are already persisted),
rebuilds the compact ``evidence`` from ``registry_rows`` and returns a NEW dict
without mutating the input.  The ``IraeReconsolidateWorker`` wraps it and
persists the result when a ``patient_id`` is given.
"""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from emr_analyzer.clinical.irae_reconsolidate import reconsolidate_report
from emr_analyzer.extraction.llm_client import OutputLimitError

from emr_analyzer.gui.workers import IraeReconsolidateWorker

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _consolidated() -> dict:
    return {
        "iraes": [{
            "irAE_type": "Miocardite da ICI",
            "ctcae_grade": "G2",
            "first_onset_date": "2022-11-22",
            "probability_immune": "PROBABILE",
            "key_evidence_ids": ["E-TROP", "E-MED"],
            "source_organs": ["Miocardite/Cardiotossicità", "Epatite"],
        }],
        "suspects": [],
    }


class FakeConsolidationLlm:
    """Minimal ``LlmClient``-like object for the consolidation call.

    ``errors`` is a queue of exceptions raised in order (one per call); when
    the queue is empty the configured ``result`` is returned.
    """

    def __init__(self, result=None, errors=()):
        self.calls: list = []
        self.result = result if result is not None else _consolidated()
        self.errors = list(errors)

    def generate_structured(self, prompt, system="", schema=None, *,
                            max_tokens=None):
        self.calls.append((prompt, max_tokens))
        if self.errors:
            error = self.errors.pop(0)
            if error is not None:
                raise error
        return self.result


def _report() -> dict:
    return {
        "analyzed_at": "2026-09-01T10:00:00",
        "anchor": {
            "first_drug": "nivolumab", "first_date": "2022-09-01",
            "last_drug": "nivolumab", "last_date": "2022-09-01",
            "occurrences": 1,
        },
        "organ_results": {
            "Miocardite/Cardiotossicità": {
                "organ": "Miocardite/Cardiotossicità",
                "iraes": [{
                    "organ": "Miocardite/Cardiotossicità",
                    "irAE_type": "Miocardite", "ctcae_grade": "G2",
                    "first_onset_date": "2022-11-22",
                    "probability_immune": "PROBABILE",
                    "key_evidence_ids": ["E-TROP"],
                }],
            },
            "Epatite": {
                "organ": "Epatite",
                "iraes": [{
                    "organ": "Epatite", "irAE_type": "Epatite",
                    "ctcae_grade": "G1", "first_onset_date": "2022-12-01",
                    "probability_immune": "POSSIBILE",
                    "key_evidence_ids": ["E-MED"],
                }],
            },
        },
        "iraes": [{"irAE_type": "vecchio", "ctcae_grade": "G1",
                   "probability_immune": "PROBABILE"}],
        "consolidation": {"applied": False,
                          "error": "OutputLimitError: precedente"},
        "evidence": [{"evidence_id": "E-OLD"}],
        "candidates_total": 1,
    }


REGISTRY_ROWS = [
    {
        "evidence_id": "E-MED", "category": "medication",
        "normalized_entity": "nivolumab", "observed_date": "2022-09-01",
        "data_json": None, "document_id": "DOC-MED",
        "value_text": "", "numeric_value": None, "unit": "",
        "source_page": 1, "source_text": "nivolumab", "bbox": None,
    },
    {
        "evidence_id": "E-TROP", "category": "laboratory_finding",
        "normalized_entity": "troponina_i_hs", "observed_date": "2022-11-22",
        "data_json": None, "document_id": "DOC-TROP",
        "value_text": "1117", "numeric_value": 1117.0, "unit": "ng/L",
        "source_page": 2, "source_text": "Troponina 1117 ng/L",
        "bbox": [46.2, 439.68, 449.91, 450.34],
    },
    {
        "evidence_id": "E-LAB", "category": "laboratory_finding",
        "normalized_entity": "alt", "observed_date": "2022-12-01",
        "data_json": None, "document_id": "DOC-LAB",
        "value_text": "80", "numeric_value": 80.0, "unit": "U/L",
        "source_page": 3, "source_text": "ALT 80 U/L", "bbox": None,
    },
]


# ---------------------------------------------------------------------------
# reconsolidate_report — pure logic
# ---------------------------------------------------------------------------


class ReconsolidateLogicTest(unittest.TestCase):
    def test_flattens_organ_results_and_applies(self):
        fake = FakeConsolidationLlm()
        result = reconsolidate_report(_report(), fake)
        self.assertEqual(fake.calls[0][1], 16384)  # default budget
        self.assertEqual(result["consolidation"]["input_count"], 2)
        self.assertTrue(result["consolidation"]["applied"])
        self.assertEqual(result["iraes"],
                         _consolidated()["iraes"])
        self.assertTrue(result["reconsolidated"])
        self.assertTrue(result["reconsolidated_at"])
        self.assertIsNone(result["reconsolidation_error"])

    def test_applied_true_rebuilds_evidence_from_cited_ids(self):
        result = reconsolidate_report(
            _report(), FakeConsolidationLlm(),
            registry_rows=list(REGISTRY_ROWS),
        )
        ids = [item["evidence_id"] for item in result["evidence"]]
        self.assertEqual(ids, ["E-TROP", "E-MED"])  # solo i citati, ordine
        trop = result["evidence"][0]
        self.assertEqual(trop["source_page"], 2)
        self.assertEqual(trop["bbox"], [46.2, 439.68, 449.91, 450.34])
        self.assertEqual(trop["normalized_entity"], "troponina_i_hs")

    def test_applied_false_keeps_old_and_sets_error(self):
        report = _report()
        fake = FakeConsolidationLlm(errors=[RuntimeError("boom")])
        result = reconsolidate_report(report, fake)
        self.assertEqual(result["iraes"], report["iraes"])
        self.assertIs(result["consolidation"], report["consolidation"])
        self.assertIs(result["evidence"], report["evidence"])
        self.assertTrue(result["reconsolidated"])
        self.assertIn("boom", result["reconsolidation_error"])
        self.assertEqual(len(fake.calls), 1)  # errore non-limit → nessun retry

    def test_output_limit_retries_until_success(self):
        fake = FakeConsolidationLlm(errors=[OutputLimitError(8192)])
        result = reconsolidate_report(_report(), fake)
        self.assertEqual([m for _, m in fake.calls], [16384, 24576])
        self.assertTrue(result["consolidation"]["applied"])

    def test_all_budgets_exhausted_keeps_old(self):
        report = _report()
        fake = FakeConsolidationLlm(errors=[
            OutputLimitError(8192)] * 3)
        result = reconsolidate_report(report, fake)
        self.assertFalse(result["consolidation"]["applied"])
        self.assertEqual(result["iraes"], report["iraes"])
        self.assertIn("OutputLimitError",
                      result["reconsolidation_error"])

    def test_empty_organ_results_noop(self):
        report = _report()
        report["organ_results"] = {}
        fake = FakeConsolidationLlm()
        result = reconsolidate_report(report, fake)
        self.assertEqual(fake.calls, [])
        self.assertFalse(result["reconsolidated"])
        self.assertIn("Nessun finding", result["reconsolidation_error"])
        self.assertEqual(result["iraes"], report["iraes"])

    def test_no_registry_rows_skips_evidence_rebuild(self):
        report = _report()
        result = reconsolidate_report(report, FakeConsolidationLlm(),
                                      registry_rows=None)
        self.assertIs(result["evidence"], report["evidence"])

    def test_empty_registry_rows_empties_evidence(self):
        result = reconsolidate_report(_report(), FakeConsolidationLlm(),
                                      registry_rows=[])
        self.assertEqual(result["evidence"], [])

    def test_does_not_mutate_input_report(self):
        report = _report()
        reconsolidate_report(report, FakeConsolidationLlm(),
                             registry_rows=list(REGISTRY_ROWS))
        self.assertEqual(report["iraes"],
                         [{"irAE_type": "vecchio", "ctcae_grade": "G1",
                           "probability_immune": "PROBABILE"}])
        self.assertEqual(report["consolidation"]["applied"], False)
        self.assertEqual(report["evidence"], [{"evidence_id": "E-OLD"}])
        self.assertNotIn("reconsolidated", report)

    def test_analyzed_at_preserved_and_reconsolidated_at_added(self):
        result = reconsolidate_report(_report(), FakeConsolidationLlm())
        self.assertEqual(result["analyzed_at"], "2026-09-01T10:00:00")
        self.assertTrue(result["reconsolidated_at"])


# ---------------------------------------------------------------------------
# IraeReconsolidateWorker — persistence + signals
# ---------------------------------------------------------------------------


class ReconsolidateWorkerTest(unittest.TestCase):
    def _run(self, worker):
        ready, errors = [], []
        worker.ready.connect(lambda r: ready.append(r))
        worker.error.connect(lambda e: errors.append(e))
        worker.run()
        return ready, errors

    def test_worker_emits_ready_and_saves_when_patient_id(self):
        from emr_analyzer.clinical.irae_reports import has_report, load_report
        from emr_analyzer.config import active_workspace

        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(active_workspace, "path", Path(tmp)):
                worker = IraeReconsolidateWorker(
                    FakeConsolidationLlm(), _report(), patient_id="P001"
                )
                ready, errors = self._run(worker)
                self.assertEqual(errors, [])
                self.assertEqual(len(ready), 1)
                self.assertTrue(ready[0]["reconsolidated"])
                self.assertTrue(has_report("P001"))
                self.assertEqual(load_report("P001"), ready[0])

    def test_worker_emits_error_on_save_failure(self):
        with mock.patch(
            "emr_analyzer.gui.workers.save_report",
            side_effect=OSError("disco pieno"),
        ):
            worker = IraeReconsolidateWorker(
                FakeConsolidationLlm(), _report(), patient_id="P001"
            )
            ready, errors = self._run(worker)
            self.assertEqual(ready, [])
            self.assertEqual(len(errors), 1)
            self.assertIn("Errore riconsolidamento", errors[0])


if __name__ == "__main__":
    unittest.main()

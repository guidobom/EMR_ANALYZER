"""Tests for the single-patient structured irAE worker."""

from __future__ import annotations

import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from emr_analyzer.clinical import irae_layers
from emr_analyzer.models.clinical_evidence import ClinicalEvidence


def make_evidence(
    patient_id: str,
    evidence_id: str,
    category: str,
    entity: str,
    observed_date: str | None = None,
    *,
    value_text: str | None = None,
    numeric_value: float | None = None,
    unit: str | None = None,
) -> ClinicalEvidence:
    return ClinicalEvidence(
        patient_id=patient_id,
        document_id=f"DOC-{evidence_id}",
        category=category,
        normalized_entity=entity,
        source_text=entity,
        evidence_id=evidence_id,
        observed_date=observed_date,
        value_text=value_text,
        numeric_value=numeric_value,
        unit=unit,
    )


class FakeStructuredLlm:
    """Minimal ``LlmClient``-like object for structured calls."""

    def __init__(self):
        self.calls: list[str] = []

    def generate_structured(self, prompt, system="", schema=None, *,
                            max_tokens=None):
        self.calls.append(prompt)
        if "FASE FINALE" in prompt:
            return {
                "iraes": [{
                    "irAE_type": "irAE consolidato",
                    "ctcae_grade": "G1",
                    "first_onset_date": "2022-11-22",
                    "probability_immune": "PROBABILE",
                    "key_evidence_ids": ["E-TROP"],
                    "source_organs": ["Miocardite/Cardiotossicità"],
                }]
            }
        return {
            "iraes": [{
                "organ": "Miocardite/Cardiotossicità",
                "irAE_type": "irAE di test",
                "ctcae_grade": "G1",
                "first_onset_date": "2022-11-22",
                "probability_immune": "PROBABILE",
                "key_evidence_ids": ["E-TROP"],
            }]
        }


class SinglePatientIraeWorkerTest(unittest.TestCase):
    def _rows(self) -> list[dict]:
        evidence = [
            make_evidence("P001", "E-MED", "medication", "nivolumab",
                          "2022-09-01"),
            make_evidence("P001", "E-TROP", "laboratory_finding",
                          "troponina_i_hs", "2022-11-22",
                          value_text="1117", numeric_value=1117.0,
                          unit="ng/L"),
        ]
        return irae_layers.evidence_rows_from_models(evidence)

    def _run(self, worker):
        reports, errors = [], []
        worker.structured_ready.connect(lambda r: reports.append(r))
        worker.error.connect(lambda e: errors.append(e))
        worker.run()
        return reports, errors

    def test_run_emits_structured_report_with_analyzed_at(self):
        from emr_analyzer.gui.workers import SinglePatientIraeWorker

        worker = SinglePatientIraeWorker(
            FakeStructuredLlm(), self._rows(), max_tokens=512
        )
        reports, errors = self._run(worker)
        self.assertEqual(errors, [])
        self.assertEqual(len(reports), 1)
        report = reports[0]
        self.assertIn("analyzed_at", report)
        self.assertEqual(report["iraes"][0]["irAE_type"], "irAE consolidato")
        # the inspection payload reaches the single-patient dialog too.
        self.assertIn("candidates", report)
        self.assertIn("evidence", report)
        self.assertTrue(report["consolidation"]["applied"])

    def test_empty_rows_errors_without_llm_call(self):
        from emr_analyzer.gui.workers import SinglePatientIraeWorker

        worker = SinglePatientIraeWorker(FakeStructuredLlm(), [])
        reports, errors = self._run(worker)
        self.assertEqual(reports, [])
        self.assertEqual(len(errors), 1)
        self.assertIn("Estrai evidenze", errors[0])

    def test_per_organ_failure_degrades_to_partial_report(self):
        from emr_analyzer.gui.workers import SinglePatientIraeWorker

        class BrokenLlm(FakeStructuredLlm):
            def generate_structured(self, prompt, system="", schema=None, *,
                                    max_tokens=None):
                self.calls.append(prompt)
                raise RuntimeError("boom")

        # ``analyze_irae`` catches per-organ LLM failures and records them in
        # ``organ_results`` (same semantics as the batch queue): the analysis
        # degrades to a partial report instead of failing hard.
        worker = SinglePatientIraeWorker(
            BrokenLlm(), self._rows(), max_tokens=512
        )
        reports, errors = self._run(worker)
        self.assertEqual(errors, [])
        self.assertEqual(len(reports), 1)
        report = reports[0]
        organ_error = (
            report["organ_results"]["Miocardite/Cardiotossicità"]["error"]
        )
        self.assertIn("boom", organ_error)
        self.assertFalse(report["consolidation"]["applied"])
        self.assertEqual(report["iraes"], [])


if __name__ == "__main__":
    unittest.main()

"""Tests for the shared structured 3-layer irAE analysis."""

from __future__ import annotations

import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from emr_analyzer.clinical import irae_layers
from emr_analyzer.clinical.irae_prototype import find_ici_anchor, scan_for_irae
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
    data: dict | None = None,
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
        data=data or {},
    )


def _p001_evidence() -> list[ClinicalEvidence]:
    """A patient with ICI anchor, troponin spike, rash and uveitis."""
    return [
        make_evidence(
            "P001", "E-MED", "medication", "nivolumab", "2022-09-01"
        ),
        make_evidence(
            "P001", "E-TROP", "laboratory_finding", "troponina_i_hs",
            "2022-11-22", value_text="1117", numeric_value=1117.0,
            unit="ng/L",
        ),
        make_evidence(
            "P001", "E-RASH", "symptom", "rash", "2022-12-13"
        ),
        make_evidence(
            "P001", "E-UVE", "diagnosis", "uveite anteriore bilaterale g1",
            "2024-11-18",
        ),
    ]


class FakeStructuredLlm:
    """Minimal ``LlmClient``-like object for structured calls."""

    def __init__(self, fail_patient: str | None = None):
        self.fail_patient = fail_patient
        self.calls: list[str] = []

    def generate_structured(self, prompt, system="", schema=None, *,
                            max_tokens=None):
        self.calls.append(prompt)
        if self.fail_patient and f"paziente {self.fail_patient}" in prompt:
            raise RuntimeError("errore simulato")
        return {
            "iraes": [{
                "organ": "test",
                "irAE_type": "irAE di test",
                "ctcae_grade": "G1",
                "first_onset_date": "2022-11-22",
                "probability_immune": "PROBABILE",
                "new_onset_vs_exacerbation": "nuova insorgenza",
                "alternative_causes": "nessuna",
                "confidence": 0.9,
                "key_evidence_ids": ["E-TROP"],
            }]
        }


class EvidenceBridgeTest(unittest.TestCase):
    def test_models_bridge_data_and_lab_fields(self):
        rows = irae_layers.evidence_rows_from_models([
            make_evidence(
                "P001", "E-TROP", "laboratory_finding", "troponina_i_hs",
                "2022-11-22", value_text="1117", numeric_value=1117.0,
                unit="ng/L", data={"quote_verified": "troponina 1117"},
            ),
        ])
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["evidence_id"], "E-TROP")
        self.assertEqual(row["data_json"]["quote_verified"], "troponina 1117")
        self.assertEqual(row["value_text"], "1117")
        self.assertEqual(row["numeric_value"], 1117.0)
        self.assertEqual(row["unit"], "ng/L")

    def test_missing_data_defaults_to_empty_dict(self):
        rows = irae_layers.evidence_rows_from_models([
            make_evidence("P001", "E-X", "symptom", "rash", "2022-12-13"),
        ])
        self.assertEqual(rows[0]["data_json"], {})


class AnalyzeIraeTest(unittest.TestCase):
    def test_layers_anchor_and_value_flow_into_organ_calls(self):
        rows = irae_layers.evidence_rows_from_models(_p001_evidence())
        anchor = find_ici_anchor(rows)
        self.assertIsNotNone(anchor)
        self.assertEqual(anchor.first_drug, "nivolumab")

        candidates = scan_for_irae(rows, anchor)
        organs = {c.organ for c in candidates}
        self.assertIn("Miocardite/Cardiotossicità", organs)
        self.assertIn("Dermatite", organs)
        self.assertIn("Oculari", organs)
        troponin = next(
            c for c in candidates if c.organ == "Miocardite/Cardiotossicità"
        )
        self.assertEqual(troponin.value, "1117")
        self.assertEqual(troponin.band, "14_112d")

    def test_analyze_irae_runs_one_structured_call_per_organ(self):
        rows = irae_layers.evidence_rows_from_models(_p001_evidence())
        llm = FakeStructuredLlm()
        report = irae_layers.analyze_irae(
            rows, llm, parallel=4, max_tokens=512
        )

        self.assertEqual(report["candidates_total"], 3)
        # Medication and non-toxicity rows are never flagged.
        self.assertIn("immunotherapy_start", report)
        self.assertEqual(report["immunotherapy_start"]["date"], "2022-09-01")
        self.assertEqual(len(report["organ_results"]), 3)
        self.assertEqual(len(report["iraes"]), 3)
        self.assertIn("total", report["timing_seconds"])
        self.assertEqual(len(llm.calls), 3)
        self.assertTrue(
            all("Miocardite" in c or "Dermatite" in c or "Oculari" in c
                for c in llm.calls)
        )

    def test_analyze_irae_empty_patient_has_no_llm_calls(self):
        rows = irae_layers.evidence_rows_from_models([
            make_evidence("P001", "E-NOTE", "symptom", "appuntamento"),
        ])
        llm = FakeStructuredLlm()
        report = irae_layers.analyze_irae(rows, llm)
        self.assertEqual(report["candidates_total"], 0)
        self.assertEqual(report["organ_results"], {})
        self.assertEqual(report["iraes"], [])
        self.assertEqual(len(llm.calls), 0)


class RenderMarkdownTest(unittest.TestCase):
    def test_renders_grade_onset_probability_and_evidence(self):
        report = {
            "anchor": {
                "first_drug": "nivolumab", "first_date": "2022-09-01",
                "first_raw": "2022-09-01", "last_drug": "nivolumab",
                "last_date": "2022-09-01", "last_raw": "2022-09-01",
                "occurrences": 1,
            },
            "candidates_total": 1,
            "organ_results": {
                "Miocardite/Cardiotossicità": {
                    "iraes": [{
                        "organ": "Miocardite/Cardiotossicità",
                        "irAE_type": "Elevazione della troponina",
                        "ctcae_grade": "G2",
                        "first_onset_date": "2022-11-22",
                        "probability_immune": "PROBABILE",
                        "new_onset_vs_exacerbation": "nuova insorgenza",
                        "alternative_causes": "nessuna",
                        "key_evidence_ids": ["E-TROP", "E-2"],
                    }]
                }
            },
        }
        markdown = irae_layers.render_irae_markdown(report)
        self.assertIn("Elevazione della troponina", markdown)
        self.assertIn("G2", markdown)
        self.assertIn("2022-11-22", markdown)
        self.assertIn("[#E-TROP", markdown)
        self.assertIn("nivolumab", markdown)

    def test_renders_per_organ_errors(self):
        markdown = irae_layers.render_irae_markdown({
            "organ_results": {
                "Pancreatite": {"error": "OutputLimitError: boom"},
            },
        })
        self.assertIn("⚠️ Errore", markdown)


class IraeLayer3QueueWorkerTest(unittest.TestCase):
    def _plans(self):
        return [
            ("P001", [("Miocardite/Cardiotossicità", "organo Miocardite paziente P001"),
                      ("Dermatite", "organo Dermatite paziente P001")]),
            ("P002", [("Oculari", "organo Oculari paziente P002")]),
        ]

    def _run_worker(self, worker):
        started, finished_ok, errors, progress, structured = (
            [], [], [], [], []
        )
        worker.patient_started.connect(
            lambda i, n, p: started.append((i, n, p))
        )
        worker.patient_finished.connect(
            lambda p, m: finished_ok.append((p, m))
        )
        worker.patient_structured.connect(
            lambda p, r: structured.append((p, r))
        )
        worker.patient_error.connect(lambda p, e: errors.append((p, e)))
        worker.chunk_progress.connect(
            lambda c, n: progress.append((c, n))
        )
        worker.run()
        return started, finished_ok, errors, progress, structured

    def test_runs_patients_in_order_with_structured_calls(self):
        from emr_analyzer.gui.workers import IraeLayer3QueueWorker

        llm = FakeStructuredLlm()
        worker = IraeLayer3QueueWorker(llm, self._plans())
        started, finished_ok, errors, progress, structured = (
            self._run_worker(worker)
        )

        self.assertEqual(started, [(1, 2, "P001"), (2, 2, "P002")])
        self.assertEqual([p for p, _ in finished_ok], ["P001", "P002"])
        self.assertEqual(errors, [])
        self.assertEqual(len(llm.calls), 3)
        self.assertIn("irAE di test", finished_ok[0][1])
        # chunk_progress covers both organs of P001 then one of P002.
        self.assertEqual(len(progress), 3)
        # patient_structured carries the raw report (for the Excel export).
        self.assertEqual([p for p, _ in structured], ["P001", "P002"])
        self.assertEqual(len(structured[0][1]["iraes"]), 2)
        self.assertEqual(len(structured[1][1]["iraes"]), 1)
        self.assertIn("analyzed_at", structured[0][1])

    def test_meta_anchor_is_rendered_in_markdown(self):
        from emr_analyzer.gui.workers import IraeLayer3QueueWorker

        meta = {
            "candidates_total": 2,
            "anchor": {
                "first_drug": "nivolumab", "first_date": "2022-09-01",
                "first_raw": "2022-09-01", "last_drug": "nivolumab",
                "last_date": "2022-09-01", "last_raw": "2022-09-01",
                "occurrences": 1,
            },
        }
        worker = IraeLayer3QueueWorker(
            FakeStructuredLlm(),
            [("P001", [("Miocardite/Cardiotossicità", "prompt")], meta)],
        )
        finished = []
        structured = []
        worker.patient_finished.connect(lambda p, m: finished.append(m))
        worker.patient_structured.connect(lambda p, r: structured.append(r))
        worker.run()
        self.assertIn("Inizio immunoterapia", finished[0])
        self.assertIn("nivolumab", finished[0])
        self.assertIn("Candidati pre-filtrati", finished[0])
        self.assertEqual(structured[0]["candidates_total"], 2)
        self.assertEqual(structured[0]["anchor"]["first_drug"], "nivolumab")

    def test_empty_plans_raise_explicit_patient_error(self):
        from emr_analyzer.gui.workers import IraeLayer3QueueWorker

        worker = IraeLayer3QueueWorker(FakeStructuredLlm(), [("P003", [])])
        started, finished_ok, errors, _, _ = self._run_worker(worker)
        self.assertEqual(started, [(1, 1, "P003")])
        self.assertEqual(finished_ok, [])
        self.assertEqual(len(errors), 1)
        self.assertIn("Estrai evidenze", errors[0][1])

    def test_all_organs_failed_reports_patient_error_and_continues(self):
        from emr_analyzer.gui.workers import IraeLayer3QueueWorker

        plans = [
            ("P001", [("Oculari", "organo Oculari paziente P001")]),
            ("P002", [("Miocardite/Cardiotossicità",
                       "organo Miocardite paziente P002")]),
        ]
        worker = IraeLayer3QueueWorker(
            FakeStructuredLlm(fail_patient="P002"), plans
        )
        started, finished_ok, errors, _, structured = self._run_worker(worker)

        self.assertEqual([p for p, _ in finished_ok], ["P001"])
        self.assertEqual([p for p, _ in errors], ["P002"])
        self.assertEqual([p for p, _ in structured], ["P001"])

    def test_cancel_stops_between_patients(self):
        from emr_analyzer.gui.workers import IraeLayer3QueueWorker

        class CancellingLlm(FakeStructuredLlm):
            def __init__(self, worker):
                super().__init__()
                self.worker = worker
                self.count = 0

            def generate_structured(self, prompt, system="", schema=None, *,
                                    max_tokens=None):
                self.count += 1
                if self.count == 2:
                    self.worker.cancel()  # durante il secondo organo di P001
                return super().generate_structured(prompt, system, schema,
                                                   max_tokens=max_tokens)

        worker = IraeLayer3QueueWorker(CancellingLlm(None), self._plans())
        worker.llm_client.worker = worker
        started, finished_ok, errors, _, _ = self._run_worker(worker)
        # P001 completes its two organ calls, P002 is skipped.
        self.assertEqual([p for p, _ in finished_ok], ["P001"])
        self.assertEqual(errors, [])
        self.assertEqual([i for i, _, p in started], [1])


class BuildLayer3PlansTest(unittest.TestCase):
    class FakeEvidenceRepo:
        def __init__(self, patients):
            self._patients = patients

        def get_by_patient(self, pid):
            return self._patients.get(pid, [])

    def test_build_irae_layer3_plans_uses_evidence_repo(self):
        from emr_analyzer.gui.workspace_tabs import WorkspaceTabs

        repo = self.FakeEvidenceRepo({
            "P001": _p001_evidence(),
            "P002": [],
        })
        plans = WorkspaceTabs._build_irae_layer3_plans(
            {"evidence_repo": repo}, ["P001", "P002"]
        )
        self.assertEqual([plan[0] for plan in plans], ["P001", "P002"])
        organs, prompts = zip(*plans[0][1])
        self.assertIn("Miocardite/Cardiotossicità", organs)
        self.assertIn("Dermatite", organs)
        self.assertIn("Oculari", organs)
        self.assertTrue(any("1117" in p for p in prompts))
        meta = plans[0][2]
        self.assertEqual(meta["candidates_total"], 3)
        self.assertEqual(meta["anchor"]["first_drug"], "nivolumab")
        self.assertEqual(plans[1][1], [])  # nessuna evidenza atomica
        self.assertEqual(plans[1][2]["anchor"], None)


if __name__ == "__main__":
    unittest.main()

"""Tests for the shared structured 3-layer irAE analysis."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from emr_analyzer.clinical import irae_layers
from emr_analyzer.clinical.irae_prototype import find_ici_anchor, scan_for_irae
from emr_analyzer.config import active_workspace
from emr_analyzer.models.clinical_evidence import ClinicalEvidence


class _IraeWorkspaceIsolated(unittest.TestCase):
    """Patch the active workspace to a temp dir for the whole test.

    The queue/single workers persist ``irae_report.json`` per patient as soon
    as a report is produced, so every test that runs them must isolate the
    workspace or it would write into the real one.
    """

    def setUp(self) -> None:
        self._ws_tmp = tempfile.TemporaryDirectory()
        self._ws_patcher = mock.patch.object(
            active_workspace, "path", Path(self._ws_tmp.name)
        )
        self._ws_patcher.start()
        self.addCleanup(self._ws_patcher.stop)
        self.addCleanup(self._ws_tmp.cleanup)


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
        if "FASE FINALE" in prompt:
            # Layer 4 consolidation: merges all per-organ findings into one.
            return {
                "iraes": [{
                    "irAE_type": "irAE consolidato",
                    "ctcae_grade": "G1",
                    "first_onset_date": "2022-11-22",
                    "probability_immune": "PROBABILE",
                    "new_onset_vs_exacerbation": "nuova insorgenza",
                    "alternative_causes": "nessuna",
                    "confidence": 0.9,
                    "key_evidence_ids": ["E-TROP"],
                    "source_organs": ["test"],
                    "notes": "approfondimento di test",
                }]
            }
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

    def test_rows_carry_document_provenance(self):
        evidence = make_evidence(
            "P001", "E-TROP", "laboratory_finding", "troponina_i_hs",
            "2022-11-22", value_text="1117", numeric_value=1117.0,
            unit="ng/L",
        )
        evidence.source_page = 7
        evidence.bbox = (1.0, 2.0, 3.0, 4.0)
        evidence.source_text = "troponina hs 1117 ng/L"
        rows = irae_layers.evidence_rows_from_models([evidence])
        row = rows[0]
        self.assertEqual(row["document_id"], "DOC-E-TROP")
        self.assertEqual(row["source_page"], 7)
        self.assertEqual(row["bbox"], [1.0, 2.0, 3.0, 4.0])
        self.assertEqual(row["source_text"], "troponina hs 1117 ng/L")

    def test_compact_evidence_drops_data_json(self):
        rows = [{
            "evidence_id": "E1", "document_id": "D1", "source_page": 2,
            "bbox": [1, 2, 3, 4], "source_text": "citazione",
            "normalized_entity": "rash", "category": "symptom",
            "observed_date": "2022-01-01", "value_text": "5",
            "data_json": {"secret": True},
        }]
        compact = irae_layers._compact_evidence(rows)
        self.assertEqual(len(compact), 1)
        self.assertNotIn("data_json", compact[0])
        self.assertEqual(compact[0]["source_page"], 2)
        self.assertEqual(compact[0]["bbox"], [1, 2, 3, 4])


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
        self.assertEqual(report["baseline_excluded"], 0)
        # Medication and non-toxicity rows are never flagged.
        self.assertIn("immunotherapy_start", report)
        self.assertEqual(report["immunotherapy_start"]["date"], "2022-09-01")
        self.assertEqual(len(report["organ_results"]), 3)
        # Layer 4 merges the three per-organ findings into one final irAE.
        self.assertEqual(len(report["iraes"]), 1)
        self.assertEqual(report["iraes"][0]["irAE_type"], "irAE consolidato")
        self.assertTrue(report["consolidation"]["applied"])
        self.assertEqual(report["consolidation"]["input_count"], 3)
        self.assertEqual(report["consolidation"]["output_count"], 1)
        self.assertIn("layer4", report["timing_seconds"])
        self.assertIn("total", report["timing_seconds"])
        # 3 organ calls + 1 consolidation call.
        self.assertEqual(len(llm.calls), 4)
        self.assertTrue(
            all("Miocardite" in c or "Dermatite" in c or "Oculari" in c
                for c in llm.calls[:3])
        )
        self.assertIn("FASE FINALE", llm.calls[3])

    def test_analyze_irae_empty_patient_has_no_llm_calls(self):
        rows = irae_layers.evidence_rows_from_models([
            make_evidence("P001", "E-NOTE", "symptom", "appuntamento"),
        ])
        llm = FakeStructuredLlm()
        report = irae_layers.analyze_irae(rows, llm)
        self.assertEqual(report["candidates_total"], 0)
        self.assertEqual(report["organ_results"], {})
        self.assertEqual(report["iraes"], [])
        self.assertEqual(report["baseline_excluded"], 0)
        self.assertEqual(len(llm.calls), 0)

    def test_analyze_irae_excludes_pre_ici_baseline(self):
        rows = irae_layers.evidence_rows_from_models([
            make_evidence(
                "P001", "M1", "medication", "nivolumab", "2022-01-10"
            ),
            make_evidence(
                "P001", "CREA-PRE", "laboratory_finding", "creatinina",
                "2021-12-20", numeric_value=1.6, unit="mg/dL",
                data={"reference_high": 1.2},
            ),
            make_evidence(
                "P001", "CREA-ON", "laboratory_finding", "creatinina",
                "2022-02-10", numeric_value=1.8, unit="mg/dL",
                data={"reference_high": 1.2},
            ),
        ])
        report = irae_layers.analyze_irae(
            rows, FakeStructuredLlm(), max_tokens=512
        )
        self.assertEqual(report["baseline_excluded"], 1)
        self.assertEqual(report["candidates_total"], 1)
        self.assertEqual(len(report["candidates_by_organ"]), 1)
        self.assertEqual(report["candidates_by_organ"][0]["organ"], "Nefrite")
        self.assertEqual(report["candidates_by_organ"][0]["count"], 1)

    def test_analyze_irae_carries_candidates_and_cited_evidence(self):
        evidence = _p001_evidence()
        troponin = evidence[1]
        troponin.source_page = 7
        troponin.bbox = (1.0, 2.0, 3.0, 4.0)
        troponin.source_text = "troponina hs 1117 ng/L"
        rows = irae_layers.evidence_rows_from_models(evidence)
        report = irae_layers.analyze_irae(
            rows, FakeStructuredLlm(), max_tokens=512
        )

        # Layer 2 candidates carry document provenance for the inspector.
        trop = next(
            c for c in report["candidates"] if c["evidence_id"] == "E-TROP"
        )
        self.assertEqual(trop["document_id"], "DOC-E-TROP")
        self.assertEqual(trop["source_page"], 7)
        self.assertEqual(trop["bbox"], [1.0, 2.0, 3.0, 4.0])
        self.assertEqual(trop["source_text"], "troponina hs 1117 ng/L")

        # The consolidated irAE cites E-TROP → compact evidence resolves it.
        evidence_list = report["evidence"]
        self.assertTrue(
            any(e["evidence_id"] == "E-TROP" for e in evidence_list)
        )
        self.assertTrue(all(
            {"evidence_id", "document_id", "source_page", "bbox",
             "source_text", "normalized_entity", "category",
             "observed_date", "value_text"} <= set(e)
            for e in evidence_list
        ))
        self.assertNotIn("data_json", evidence_list[0])

    def test_evidence_resolves_model_ids_with_hash_prefix(self):
        # The model echoes the prompt's ``[#id]`` notation, while the registry
        # stores ``EVD_x`` without the ``#``: the compact evidence lookup must
        # still resolve them and the markdown must never render "##".
        evidence = _p001_evidence()
        rows = irae_layers.evidence_rows_from_models(evidence)

        class HashPrefixedLlm(FakeStructuredLlm):
            def generate_structured(self, prompt, system="", schema=None, *,
                                    max_tokens=None):
                data = super().generate_structured(
                    prompt, system, schema, max_tokens=max_tokens
                )
                for entry in data.get("iraes", []):
                    entry["key_evidence_ids"] = [
                        f"#{eid}"
                        for eid in (entry.get("key_evidence_ids") or [])
                    ]
                return data

        report = irae_layers.analyze_irae(
            rows, HashPrefixedLlm(), max_tokens=512
        )
        self.assertTrue(
            any(e["evidence_id"] == "E-TROP" for e in report["evidence"])
        )
        md = irae_layers.render_irae_markdown(report)
        self.assertNotIn("##E-", md)
        self.assertIn("[#E-TROP]", md)


class ScanForIraeThresholdTest(unittest.TestCase):
    def _rows(self, anchor_date: str, *lab_rows) -> list[dict]:
        return [{
            "evidence_id": "ICI", "category": "medication",
            "normalized_entity": "nivolumab", "observed_date": anchor_date,
            "data_json": {},
        }, *lab_rows]

    def _lab(self, eid, entity, date, value, unit, *, ref_high=None,
             operator=None, reference_text=None, value_text=None):
        data = {}
        if ref_high is not None:
            data["reference_high"] = ref_high
        if reference_text is not None:
            data["reference_text"] = reference_text
        if operator is not None:
            data["operator"] = operator
        return {
            "evidence_id": eid, "category": "laboratory_finding",
            "normalized_entity": entity, "observed_date": date,
            "numeric_value": value, "unit": unit,
            "value_text": value_text or "", "data_json": data,
        }

    def test_drop_baseline_by_default_and_keep_with_flag(self):
        rows = self._rows(
            "2022-01-10",
            self._lab("ALT-PRE", "alanina_aminotransferasi", "2021-12-01",
                      60.0, "U/L", ref_high=35.0),
        )
        anchor = find_ici_anchor(rows)
        self.assertEqual(scan_for_irae(rows, anchor), [])
        kept = scan_for_irae(rows, anchor, drop_baseline=False)
        self.assertEqual([c.evidence_id for c in kept], ["ALT-PRE"])

    def test_drops_lab_within_reference_by_default(self):
        rows = self._rows(
            "2022-01-10",
            self._lab("ALT-NORM", "alanina_aminotransferasi", "2022-02-01",
                      30.0, "U/L", ref_high=35.0, reference_text="0-35"),
        )
        anchor = find_ici_anchor(rows)
        self.assertEqual(scan_for_irae(rows, anchor), [])
        kept = scan_for_irae(rows, anchor, apply_lab_threshold=False)
        self.assertEqual([c.evidence_id for c in kept], ["ALT-NORM"])

    def test_keeps_elevated_lab_with_reference(self):
        rows = self._rows(
            "2022-01-10",
            self._lab("ALT-ELEV", "alanina_aminotransferasi", "2022-03-01",
                      120.0, "U/L", ref_high=35.0, reference_text="0-35"),
        )
        anchor = find_ici_anchor(rows)
        cand = scan_for_irae(rows, anchor)
        self.assertEqual([c.evidence_id for c in cand], ["ALT-ELEV"])
        self.assertEqual(cand[0].reference, "0-35")

    def test_keeps_lab_without_reference(self):
        rows = self._rows(
            "2022-01-10",
            self._lab("ALT-NOREF", "alanina_aminotransferasi", "2022-04-01",
                      33.0, "U/L"),
        )
        anchor = find_ici_anchor(rows)
        self.assertEqual(
            [c.evidence_id for c in scan_for_irae(rows, anchor)],
            ["ALT-NOREF"],
        )

    def test_keeps_narrative_lab_without_numeric(self):
        rows = self._rows(
            "2022-01-10",
            self._lab("ALT-NAR", "alanina_aminotransferasi", "2022-04-01",
                      None, "", ref_high=35.0, value_text="ALT >40"),
        )
        anchor = find_ici_anchor(rows)
        self.assertEqual(
            [c.evidence_id for c in scan_for_irae(rows, anchor)],
            ["ALT-NAR"],
        )

    def test_keeps_operator_lower_bound(self):
        rows = self._rows(
            "2022-01-10",
            self._lab("ALT-BOUND", "alanina_aminotransferasi", "2022-04-01",
                      40.0, "U/L", ref_high=50.0, operator=">"),
        )
        anchor = find_ici_anchor(rows)
        self.assertEqual(
            [c.evidence_id for c in scan_for_irae(rows, anchor)],
            ["ALT-BOUND"],
        )

    def test_threshold_applies_only_to_lab_organs(self):
        rows = self._rows(
            "2022-01-10",
            self._lab("TSH-NORM", "ormone_tireostimolante", "2022-02-01",
                      2.0, "mU/L", ref_high=4.0),
        )
        anchor = find_ici_anchor(rows)
        self.assertEqual(
            [c.evidence_id for c in scan_for_irae(rows, anchor)],
            ["TSH-NORM"],
        )

    def test_format_candidate_line_appends_reference(self):
        from emr_analyzer.clinical.irae_prototype import ToxicityCandidate

        cand = ToxicityCandidate(
            organ="Epatite", evidence_id="ALT-ELEV",
            category="laboratory_finding",
            entity="alanina_aminotransferasi", observed_raw="2022-03-01",
            offset_days=50, band="14_112d", quote="", value="120 U/L",
            reference="0-35",
        )
        self.assertIn("REFERENZA: 0-35", irae_layers.format_candidate_line(cand))


class ConsolidateIraeTest(unittest.TestCase):
    def _findings(self):
        return [
            {
                "organ": "Miocardite/Cardiotossicità",
                "irAE_type": "Elevazione della troponina",
                "ctcae_grade": "G2", "first_onset_date": "2022-11-22",
                "probability_immune": "PROBABILE",
                "key_evidence_ids": ["E-TROP"],
            },
            {
                "organ": "Dermatite",
                "irAE_type": "Rash maculopapulare",
                "ctcae_grade": "G1", "first_onset_date": "2022-12-13",
                "probability_immune": "POSSIBILE",
                "key_evidence_ids": ["E-RASH"],
            },
        ]

    def test_build_prompt_lists_findings_and_marks_fase_finale(self):
        anchor = {
            "first_drug": "nivolumab", "first_date": "2022-09-01",
            "first_raw": "2022-09-01", "last_drug": "nivolumab",
            "last_date": "2022-09-01", "last_raw": "2022-09-01",
        }
        prompt, truncated = irae_layers.build_consolidation_prompt(
            self._findings(), anchor
        )
        self.assertEqual(truncated, 0)
        self.assertIn("FASE FINALE", prompt)
        self.assertIn("nivolumab dal 2022-09-01", prompt)
        self.assertIn("[1] Organo: Miocardite/Cardiotossicità", prompt)
        # Dermatite è POSSIBILE → va nella sezione sospetti, non nei definitivi.
        self.assertIn("SOSPETTI", prompt)
        self.assertIn("[S1] Organo: Dermatite", prompt)
        self.assertIn("UNISCI i duplicati", prompt)

    def test_consolidate_returns_final_iraes(self):
        llm = FakeStructuredLlm()
        result = irae_layers.consolidate_iraes(
            llm, self._findings(), None, max_tokens=512
        )
        self.assertTrue(result["applied"])
        self.assertEqual(result["input_count"], 2)
        self.assertEqual(result["output_count"], 1)
        self.assertEqual(result["iraes"][0]["irAE_type"], "irAE consolidato")
        self.assertEqual(result["suspects"], [])
        self.assertEqual(len(llm.calls), 1)

    def test_consolidate_fallback_to_input_on_exception(self):
        class BrokenLlm(FakeStructuredLlm):
            def generate_structured(self, prompt, system="", schema=None, *,
                                    max_tokens=None):
                self.calls.append(prompt)
                raise RuntimeError("boom")

        llm = BrokenLlm()
        result = irae_layers.consolidate_iraes(
            llm, self._findings(), None, max_tokens=512
        )
        self.assertFalse(result["applied"])
        self.assertIn("boom", result["error"])
        # Fallback: the analysis is never lost, ma la specificità resta:
        # solo il reperto PROBABILE è definitivo, il POSSIBILE è un sospetto.
        self.assertEqual(result["iraes"], [self._findings()[0]])
        self.assertEqual(result["suspects"], [self._findings()[1]])
        self.assertEqual(result["output_count"], 1)

    def test_consolidate_empty_findings_makes_no_call(self):
        llm = FakeStructuredLlm()
        result = irae_layers.consolidate_iraes(llm, [], None)
        self.assertFalse(result["applied"])
        self.assertEqual(result["iraes"], [])
        self.assertEqual(result["suspects"], [])
        self.assertEqual(len(llm.calls), 0)

    def test_consolidate_partitions_suspects_and_drops_improbable(self):
        class MixedLlm(FakeStructuredLlm):
            def generate_structured(self, prompt, system="", schema=None, *,
                                    max_tokens=None):
                self.calls.append(prompt)
                return {
                    "iraes": [
                        {"irAE_type": "Miocardite da ICI",
                         "probability_immune": "PROBABILE"},
                        {"irAE_type": "Rash sospetto",
                         "probability_immune": "POSSIBILE"},
                        {"irAE_type": "Lesione cutanea del melanoma",
                         "probability_immune": "IMPROBABILE"},
                        {"irAE_type": "Alterazione transitoria",
                         "probability_immune": "INDETERMINATA"},
                    ]
                }

        result = irae_layers.consolidate_iraes(
            MixedLlm(), [{"organ": "x", "irAE_type": "y"}], None
        )
        self.assertTrue(result["applied"])
        self.assertEqual(
            [f["irAE_type"] for f in result["iraes"]], ["Miocardite da ICI"]
        )
        self.assertEqual(
            [f["irAE_type"] for f in result["suspects"]], ["Rash sospetto"]
        )
        self.assertEqual(result["output_count"], 1)
        self.assertEqual(result["input_count"], 1)

    def test_backfills_cited_ids_from_notes(self):
        # il modello omette key_evidence_ids e le scrive come ``#EVD_...``
        # nelle note; il backfill le estrae e normalizza il ``#``.
        class NotesOnlyLlm(FakeStructuredLlm):
            def generate_structured(self, prompt, system="", schema=None, *,
                                    max_tokens=None):
                self.calls.append(prompt)
                return {
                    "iraes": [{
                        "irAE_type": "irAE consolidato",
                        "ctcae_grade": "G1",
                        "first_onset_date": "2022-11-22",
                        "probability_immune": "PROBABILE",
                        "source_organs": ["Miocardite/Cardiotossicità"],
                        "notes": "Le evidenze chiave sono: "
                                 "#EVD_d19bfa0f89d15578, "
                                 "#EVD_f357e94e0c11aa99.",
                    }]
                }

        result = irae_layers.consolidate_iraes(
            NotesOnlyLlm(), self._findings(), None, max_tokens=512
        )
        self.assertTrue(result["applied"])
        self.assertEqual(
            result["iraes"][0]["key_evidence_ids"],
            ["EVD_d19bfa0f89d15578", "EVD_f357e94e0c11aa99"],
        )

    def test_backfills_cited_ids_from_organ_findings_when_notes_lack_them(self):
        class NoIdsLlm(FakeStructuredLlm):
            def generate_structured(self, prompt, system="", schema=None, *,
                                    max_tokens=None):
                self.calls.append(prompt)
                return {
                    "iraes": [{
                        "irAE_type": "irAE consolidato",
                        "ctcae_grade": "G1",
                        "first_onset_date": "2022-11-22",
                        "probability_immune": "PROBABILE",
                        "source_organs": ["Miocardite/Cardiotossicità"],
                        "notes": "approfondimento senza id citati",
                    }]
                }

        result = irae_layers.consolidate_iraes(
            NoIdsLlm(), self._findings(), None, max_tokens=512
        )
        self.assertTrue(result["applied"])
        # fallback: unione dei key_evidence_ids dei findings Layer 3 per organo.
        self.assertEqual(result["iraes"][0]["key_evidence_ids"], ["E-TROP"])

    def test_model_provided_ids_are_kept_not_overwritten(self):
        class KeepsIdsLlm(FakeStructuredLlm):
            def generate_structured(self, prompt, system="", schema=None, *,
                                    max_tokens=None):
                self.calls.append(prompt)
                return {
                    "iraes": [{
                        "irAE_type": "irAE consolidato",
                        "ctcae_grade": "G1",
                        "first_onset_date": "2022-11-22",
                        "probability_immune": "PROBABILE",
                        "key_evidence_ids": ["#E-TROP"],
                        "notes": "Le evidenze chiave sono: #EVD_0000.",
                    }]
                }

        result = irae_layers.consolidate_iraes(
            KeepsIdsLlm(), self._findings(), None, max_tokens=512
        )
        self.assertTrue(result["applied"])
        # le citazioni strutturate del modello vincono (normalizzate dal #).
        self.assertEqual(result["iraes"][0]["key_evidence_ids"], ["E-TROP"])


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

    def test_renders_consolidated_section_with_organs_and_notes(self):
        report = {
            "organ_results": {},
            "consolidation": {
                "applied": True,
                "error": None,
                "input_count": 2,
                "output_count": 1,
                "iraes": [{
                    "irAE_type": "Miocardite da ICI",
                    "ctcae_grade": "G2",
                    "first_onset_date": "2022-11-22",
                    "probability_immune": "PROBABILE",
                    "source_organs": [
                        "Miocardite/Cardiotossicità", "Dermatite",
                    ],
                    "notes": "ripresa con prednisone 1 mg/kg",
                    "key_evidence_ids": ["E-TROP", "E-RASH"],
                }],
            },
        }
        markdown = irae_layers.render_irae_markdown(report)
        self.assertIn("Analisi finale consolidata (Layer 4)", markdown)
        self.assertIn("Miocardite da ICI", markdown)
        self.assertIn("Miocardite/Cardiotossicità, Dermatite", markdown)
        self.assertIn("ripresa con prednisone 1 mg/kg", markdown)
        self.assertIn("Uniti 2 reperti per organo in 1 irAE definitivi", markdown)

    def test_renders_diagnostic_support_and_suspects(self):
        report = {
            "anchor": None,
            "organ_results": {
                "Epatite": {
                    "iraes": [{
                        "organ": "Epatite", "irAE_type": "Epatite G2",
                        "ctcae_grade": "G2",
                        "first_onset_date": "2022-11-22",
                        "probability_immune": "PROBABILE",
                        "diagnostic_support": "trend_evidenze_multiple",
                        "key_evidence_ids": ["E-ALT"],
                    }]
                }
            },
            "consolidation": {
                "applied": True,
                "error": None,
                "input_count": 2,
                "output_count": 1,
                "iraes": [{
                    "irAE_type": "Epatite da ICI", "ctcae_grade": "G2",
                    "first_onset_date": "2022-11-22",
                    "probability_immune": "PROBABILE",
                    "source_organs": ["Epatite"],
                }],
                "suspects": [{
                    "irAE_type": "Pancreatite sospetta", "ctcae_grade": "G1",
                    "first_onset_date": "2022-12-01",
                    "probability_immune": "POSSIBILE",
                    "source_organs": ["Pancreatite"],
                    "alternative_causes": "ipertrigliceridemia",
                }],
            },
        }
        markdown = irae_layers.render_irae_markdown(report)
        self.assertIn("Supporto: trend_evidenze_multiple", markdown)
        self.assertIn("Sospetti da monitorare", markdown)
        self.assertIn("Pancreatite sospetta", markdown)
        self.assertIn("ipertrigliceridemia", markdown)

    def test_renders_ignores_new_inspection_keys(self):
        report = {
            "anchor": None,
            "candidates_total": 1,
            "organ_results": {},
            "candidates": [{"evidence_id": "E1", "bbox": [1, 2, 3, 4]}],
            "evidence": [{"evidence_id": "E1", "source_page": 2}],
            "consolidation": {
                "applied": True, "error": None,
                "input_count": 1, "output_count": 1,
                "iraes": [{
                    "irAE_type": "Miocardite da ICI", "ctcae_grade": "G2",
                    "first_onset_date": "2022-11-22",
                    "probability_immune": "PROBABILE",
                    "source_organs": ["Miocardite/Cardiotossicità"],
                    "key_evidence_ids": ["E1"],
                }],
            },
        }
        markdown = irae_layers.render_irae_markdown(report)
        self.assertIn("Miocardite da ICI", markdown)
        # The inspection payload is inert for the rendered report.
        self.assertNotIn("bbox", markdown)
        self.assertNotIn("source_page", markdown)

    def test_consolidation_not_applied_omits_section(self):
        report = {
            "organ_results": {},
            "consolidation": {
                "applied": False,
                "error": "OutputLimitError: boom",
                "input_count": 2,
                "output_count": 2,
                "iraes": [],
            },
        }
        markdown = irae_layers.render_irae_markdown(report)
        self.assertNotIn("Analisi finale consolidata (Layer 4)", markdown)


class IraeLayer3QueueWorkerTest(_IraeWorkspaceIsolated):
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
        # P001: 2 organs + 1 consolidation; P002: 1 organ + 1 consolidation.
        self.assertEqual(len(llm.calls), 5)
        self.assertIn("irAE consolidato", finished_ok[0][1])
        self.assertIn("irAE di test", finished_ok[0][1])  # sezione per organo
        # chunk_progress: P001 (2 organi + consolidamento) then P002 (1 + 1).
        self.assertEqual(len(progress), 5)
        # patient_structured carries the raw report (for the Excel export).
        self.assertEqual([p for p, _ in structured], ["P001", "P002"])
        self.assertEqual(len(structured[0][1]["iraes"]), 1)
        self.assertEqual(len(structured[1][1]["iraes"]), 1)
        self.assertTrue(structured[0][1]["consolidation"]["applied"])
        self.assertIn("analyzed_at", structured[0][1])

    def test_persists_structured_report_per_patient(self):
        import tempfile
        from pathlib import Path

        from emr_analyzer.clinical.irae_reports import (
            has_report, load_report,
        )
        from emr_analyzer.config import active_workspace
        from emr_analyzer.gui.workers import IraeLayer3QueueWorker

        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(active_workspace, "path", Path(tmp)):
                llm = FakeStructuredLlm()
                worker = IraeLayer3QueueWorker(llm, self._plans())
                _, _, errors, _, structured = self._run_worker(worker)
                self.assertEqual(errors, [])
                self.assertTrue(has_report("P001"))
                self.assertTrue(has_report("P002"))
                # The saved raw report equals the one emitted on the signal.
                self.assertEqual(load_report("P001"), structured[0][1])
                self.assertEqual(load_report("P002"), structured[1][1])

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
        # Tolerant of meta without the inspection payload (legacy callers).
        self.assertEqual(structured[0]["candidates"], [])
        self.assertEqual(structured[0]["evidence"], [])

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


class ParallelFakeLlm(FakeStructuredLlm):
    """Cloneable fake: each sibling instance records its own calls and the
    shared backend tracks which instances the queue stopped.  ``warmup`` must
    never be called (lazy loading — instances spin up on first use)."""

    def __init__(self, fail_patient: str | None = None):
        super().__init__(fail_patient)
        self.instance_id = 0
        self.backend_type = "llama_cpp"
        self.context_length = 16384
        self.model = "qwen3:30b-a3b"
        self.instance_calls: list[str] = []
        self.warmup_called = 0
        self.backend = _ParallelFakeBackend()

    def for_instance(self, instance_id: int):
        instance_id = int(instance_id or 0)
        if instance_id == self.instance_id:
            return self
        clone = object.__new__(type(self))
        clone.__dict__.update(self.__dict__)
        clone.instance_id = instance_id
        clone.instance_calls = []
        return clone

    def generate_structured(self, prompt, system="", schema=None, *,
                            max_tokens=None):
        self.backend.all_calls.append((self.instance_id, prompt))
        self.instance_calls.append(prompt)
        return super().generate_structured(prompt, system, schema,
                                           max_tokens=max_tokens)

    def warmup(self, *_args, **_kwargs):
        self.warmup_called += 1
        raise AssertionError("warmup non deve partire (lazy loading)")


class _ParallelFakeBackend:
    """Shared across clones: records every call and every stopped sibling."""

    def __init__(self):
        self.all_calls: list[tuple[int, str]] = []
        self.stopped: list[int] = []

    def stop_config(self, client):
        self.stopped.append(client.instance_id)

    def model_info(self, name):
        return {"size_bytes": 18.6 * 1024 ** 3}

    def key_for(self, client):
        return mock.Mock(np=8)


    def test_default_max_tokens_matches_single_patient_path(self):
        # The queue must use the same Layer-4-safe output budget as the
        # single-patient worker (8192), not the clinical config's 4096: on
        # large consolidations a 4096 cap raises OutputLimitError.
        from emr_analyzer.gui.workers import IraeLayer3QueueWorker

        worker = IraeLayer3QueueWorker(FakeStructuredLlm(), self._plans())
        self.assertEqual(
            worker.max_tokens, irae_layers.DEFAULT_MAX_TOKENS
        )


class IraeLayer3QueueWorkerParallelTest(_IraeWorkspaceIsolated):
    def _plans(self):
        return [
            ("P001", [("Miocardite/Cardiotossicità",
                       "organo Miocardite paziente P001"),
                      ("Dermatite", "organo Dermatite paziente P001")]),
            ("P002", [("Oculari", "organo Oculari paziente P002")]),
        ]

    def _run_worker(self, worker):
        # Signals emitted from the patient-level thread pool are queued to the
        # main thread: the worker must run in its own QThread while the test
        # pumps the event loop so the slots fire (as in the real GUI).
        from PyQt5.QtCore import QEventLoop, QTimer
        from PyQt5.QtWidgets import QApplication

        QApplication.instance() or QApplication([])
        started, finished_ok, errors, structured = [], [], [], []
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
        loop = QEventLoop()
        worker.finished.connect(loop.quit)
        QTimer.singleShot(60_000, loop.quit)  # watchdog: fail, don't hang
        worker.start()
        loop.exec_()
        if not worker.wait(5_000):
            raise AssertionError("worker non terminato")
        return started, finished_ok, errors, structured

    def test_parallel_uses_distinct_clients_and_cleans_siblings(self):
        from emr_analyzer.gui.workers import IraeLayer3QueueWorker

        llm = ParallelFakeLlm()
        worker = IraeLayer3QueueWorker(llm, self._plans(), instances=2)
        started, finished_ok, errors, structured = self._run_worker(worker)

        self.assertEqual(errors, [])
        # Both patients start (order is racy: they run concurrently).
        self.assertEqual(sorted(i for i, n, _ in started), [1, 2])
        self.assertTrue(all(n == 2 for i, n, _ in started))
        self.assertEqual({p for p, _ in finished_ok}, {"P001", "P002"})
        self.assertEqual({p for p, _ in structured}, {"P001", "P002"})
        # P001 → istanza 0 (2 organi + consolidamento); P002 → istanza 1.
        by_instance: dict[int, list[str]] = {}
        for iid, prompt in llm.backend.all_calls:
            by_instance.setdefault(iid, []).append(prompt)
        self.assertEqual(sorted(by_instance), [0, 1])
        self.assertEqual(len(by_instance[0]), 3)
        self.assertEqual(len(by_instance[1]), 2)
        organ0 = [p for p in by_instance[0] if "organo" in p]
        organ1 = [p for p in by_instance[1] if "organo" in p]
        self.assertEqual(len(organ0), 2)
        self.assertTrue(all("P001" in p for p in organ0))
        self.assertEqual(len(organ1), 1)
        self.assertTrue(all("P002" in p for p in organ1))
        # Lazy: warmup mai chiamato; cleanup del solo sibling, mai l'istanza 0.
        self.assertEqual(llm.warmup_called, 0)
        self.assertEqual(llm.backend.stopped, [1])

    def test_error_on_one_patient_still_cleans_siblings(self):
        from emr_analyzer.gui.workers import IraeLayer3QueueWorker

        llm = ParallelFakeLlm(fail_patient="P002")
        worker = IraeLayer3QueueWorker(llm, self._plans(), instances=2)
        _, finished_ok, errors, _ = self._run_worker(worker)

        self.assertEqual([p for p, _ in finished_ok], ["P001"])
        self.assertEqual([p for p, _ in errors], ["P002"])
        # Il sibling (usato da P002, fallito) viene comunque fermato.
        self.assertEqual(llm.backend.stopped, [1])
        self.assertEqual(llm.warmup_called, 0)

    def test_without_for_instance_falls_back_to_sequential(self):
        from emr_analyzer.gui.workers import IraeLayer3QueueWorker

        llm = FakeStructuredLlm()  # no for_instance → parallel count = 1
        worker = IraeLayer3QueueWorker(llm, self._plans(), instances=2)
        started, finished_ok, errors, _ = self._run_worker(worker)

        self.assertEqual([i for i, _, _ in started], [1, 2])
        self.assertEqual([p for p, _ in finished_ok], ["P001", "P002"])
        self.assertEqual(errors, [])

    def test_instances_override_is_capped_by_plan_count(self):
        from emr_analyzer.gui.workers import IraeLayer3QueueWorker

        llm = ParallelFakeLlm()
        worker = IraeLayer3QueueWorker(llm, self._plans(), instances=9)
        _, finished_ok, errors, _ = self._run_worker(worker)

        self.assertEqual(errors, [])
        self.assertEqual({p for p, _ in finished_ok}, {"P001", "P002"})
        # Solo due client creati (0 e 1): il cap è len(plans) = 2.
        instance_ids = {iid for iid, _ in llm.backend.all_calls}
        self.assertEqual(instance_ids, {0, 1})
        self.assertEqual(llm.backend.stopped, [1])


class ParallelInstanceCountTest(unittest.TestCase):
    def test_auto_uses_kv_aware_memory_budget(self):
        llm = ParallelFakeLlm()
        with mock.patch(
            "emr_analyzer.utils.hardware.psutil.virtual_memory"
        ) as vm:
            vm.return_value.available = 116 * 1024 ** 3
            self.assertEqual(
                irae_layers.parallel_instance_count(llm, override=0), 3
            )

    def test_override_pins_count(self):
        llm = ParallelFakeLlm()
        self.assertEqual(
            irae_layers.parallel_instance_count(llm, override=2), 2
        )

    def test_client_without_for_instance_returns_one(self):
        self.assertEqual(
            irae_layers.parallel_instance_count(
                FakeStructuredLlm(), override=0
            ),
            1,
        )

    def test_unknown_model_size_is_conservative(self):
        llm = ParallelFakeLlm()
        llm.backend.model_info = lambda name: {}
        self.assertEqual(
            irae_layers.parallel_instance_count(llm, override=0), 1
        )


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
        # Inspection payload is carried in the meta for the report.
        self.assertEqual(len(meta["candidates"]), 3)
        self.assertIn("evidence", meta)
        self.assertTrue(any(
            e["evidence_id"] == "E-TROP" for e in meta["evidence"]
        ))
        self.assertEqual(plans[1][1], [])  # nessuna evidenza atomica
        self.assertEqual(plans[1][2]["anchor"], None)
        self.assertEqual(plans[1][2]["candidates"], [])
        self.assertEqual(plans[1][2]["evidence"], [])


if __name__ == "__main__":
    unittest.main()

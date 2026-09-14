"""Tests for the persistent manual corrections of the structured irAE report.

Pure logic only: load/save/add/apply, matching, idempotency and the
anti-duplication guard — no GUI is involved.
"""

from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from emr_analyzer.clinical.irae_corrections import (
    IraeCorrection,
    apply_irae_corrections,
    add_correction,
    annotate_report,
    corrections_path,
    load_corrections,
    remove_correction,
    save_corrections,
)
from emr_analyzer.config import active_workspace


def _finding(**overrides) -> dict:
    base = {
        "irAE_type": "Miocardite da ICI",
        "ctcae_grade": "G2",
        "first_onset_date": "2022-11-22",
        "probability_immune": "PROBABILE",
        "new_onset_vs_exacerbation": "nuova insorgenza",
        "alternative_causes": "nessuna",
        "source_organs": ["Miocardite/Cardiotossicità"],
        "organ": "Miocardite/Cardiotossicità",
        "key_evidence_ids": ["E-TROP"],
        "notes": "originale",
    }
    base.update(overrides)
    return base


def _report() -> dict:
    """A consolidated report with one definitive irAE and one suspect."""
    definitive = [_finding()]
    suspects = [_finding(
        irAE_type="Rash sospetto", ctcae_grade="G1",
        first_onset_date="2022-12-13", probability_immune="POSSIBILE",
        source_organs=["Dermatite"], organ="Dermatite",
    )]
    return {
        "anchor": {
            "first_drug": "nivolumab", "first_date": "2022-09-01",
            "first_raw": "2022-09-01", "last_drug": "nivolumab",
            "last_date": "2022-09-01", "last_raw": "2022-09-01",
            "occurrences": 1,
        },
        "candidates_total": 2,
        "iraes": definitive,
        "consolidation": {
            "applied": True, "error": None,
            "input_count": 2, "output_count": 1,
            "iraes": definitive,
            "suspects": suspects,
        },
        "organ_results": {
            "Miocardite/Cardiotossicità": {
                "iraes": [
                    {"organ": "Miocardite/Cardiotossicità",
                     "irAE_type": "Miocardite da ICI",
                     "first_onset_date": "2022-11-22",
                     "probability_immune": "PROBABILE"},
                ],
            },
        },
    }


class PersistenceTest(unittest.TestCase):
    def test_path_is_inside_patient_workspace(self):
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(active_workspace, "path", Path(tmp)):
                path = corrections_path("P001")
        self.assertEqual(path, Path(tmp) / "P001" / "irae_corrections.json")

    def test_load_missing_returns_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(active_workspace, "path", Path(tmp)):
                self.assertEqual(load_corrections("P001"), [])

    def test_load_corrupt_file_returns_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(active_workspace, "path", Path(tmp)):
                path = corrections_path("P001")
                path.parent.mkdir(parents=True)
                path.write_text("{not json", encoding="utf-8")
                self.assertEqual(load_corrections("P001"), [])

    def test_save_and_load_roundtrip(self):
        corrections = [
            IraeCorrection(action="remove", irAE_type="Miocardite da ICI",
                           first_onset_date="2022-11-22"),
            IraeCorrection(action="add", irAE_type="Colite da ICI",
                           fields={"ctcae_grade": "G2"},
                           reason="vista in lettera"),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(active_workspace, "path", Path(tmp)):
                save_corrections("P001", corrections)
                loaded = load_corrections("P001")
        self.assertEqual(len(loaded), 2)
        self.assertEqual(loaded[0].action, "remove")
        self.assertEqual(loaded[1].fields["ctcae_grade"], "G2")
        self.assertEqual(loaded[1].reason, "vista in lettera")

    def test_add_correction_persists_and_stamps_applied_at(self):
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(active_workspace, "path", Path(tmp)):
                correction = IraeCorrection(
                    action="remove", irAE_type="Rash sospetto"
                )
                self.assertEqual(correction.applied_at, "")
                add_correction("P001", correction)
                self.assertTrue(correction.applied_at)
                loaded = load_corrections("P001")
                self.assertEqual(len(loaded), 1)
                self.assertTrue(loaded[0].applied_at)
                self.assertTrue(corrections_path("P001").exists())

    def test_from_dict_rejects_unknown_action(self):
        self.assertIsNone(IraeCorrection.from_dict({"action": "explode"}))

    def test_remove_correction_deletes_only_matching_remove(self):
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(active_workspace, "path", Path(tmp)):
                add_correction("P001", IraeCorrection(
                    action="remove", irAE_type="Miocardite da ICI",
                    first_onset_date="2022-11-22", finding_id="irAE-1",
                ))
                add_correction("P001", IraeCorrection(
                    action="remove", irAE_type="Rash sospetto",
                    first_onset_date="2022-12-13", finding_id="irAE-2",
                ))
                add_correction("P001", IraeCorrection(
                    action="edit", irAE_type="Epatite", finding_id="irAE-3",
                ))
                remaining = remove_correction("P001", finding_id="irAE-1")
        self.assertEqual(len(remaining), 2)
        self.assertTrue(all(
            not (c.action == "remove" and c.finding_id == "irAE-1")
            for c in remaining
        ))
        self.assertTrue(any(c.action == "remove" for c in remaining))

    def test_remove_correction_matches_type_and_date_without_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(active_workspace, "path", Path(tmp)):
                add_correction("P001", IraeCorrection(
                    action="remove", irAE_type="Rash sospetto",
                    first_onset_date="2022-12-13",
                ))
                remaining = remove_correction(
                    "P001", irAE_type="Rash sospetto",
                    first_onset_date="2022-12-13",
                )
                loaded = load_corrections("P001")
        self.assertEqual(remaining, [])
        self.assertEqual(loaded, [])


class AnnotateReportTest(unittest.TestCase):
    def test_assigns_stable_finding_ids(self):
        corrected = annotate_report(_report())
        self.assertEqual(corrected["iraes"][0]["finding_id"], "irAE-1")
        self.assertEqual(
            corrected["consolidation"]["suspects"][0]["finding_id"], "irAE-2"
        )
        # the definitive list stays the same object under both views.
        self.assertIs(
            corrected["iraes"], corrected["consolidation"]["iraes"]
        )

    def test_does_not_mutate_input(self):
        report = _report()
        before = copy.deepcopy(report)
        annotate_report(report)
        self.assertEqual(report, before)

    def test_defensive_copy_of_preexisting_removed_bucket(self):
        report = _report()
        report["consolidation"]["removed"] = [
            {"irAE_type": "Rash rimosso", "finding_id": "irAE-9"}
        ]
        corrected = annotate_report(report)
        self.assertIsNot(
            corrected["consolidation"]["removed"],
            report["consolidation"]["removed"],
        )
        corrected["consolidation"]["removed"][0]["irAE_type"] = "mutato"
        self.assertEqual(
            report["consolidation"]["removed"][0]["irAE_type"],
            "Rash rimosso",
        )


class ApplyIraeCorrectionsTest(unittest.TestCase):
    def test_remove_from_iraes_and_mirrors_organ_results(self):
        corrected, log = apply_irae_corrections(_report(), [
            IraeCorrection(action="remove", irAE_type="Miocardite da ICI",
                           first_onset_date="2022-11-22"),
        ])
        self.assertEqual(corrected["iraes"], [])
        self.assertTrue(log[0]["applied"])
        self.assertEqual(
            corrected["organ_results"]["Miocardite/Cardiotossicità"]["iraes"],
            [],
        )
        # the suspect is untouched.
        suspects = corrected["consolidation"]["suspects"]
        self.assertEqual([s["irAE_type"] for s in suspects], ["Rash sospetto"])

    def test_remove_moves_finding_to_removed_bucket(self):
        corrected, log = apply_irae_corrections(_report(), [
            IraeCorrection(action="remove", irAE_type="Miocardite da ICI",
                           first_onset_date="2022-11-22",
                           reason="falso positivo"),
        ])
        removed = corrected["consolidation"]["removed"]
        self.assertEqual(len(removed), 1)
        self.assertTrue(removed[0]["removed"])
        self.assertEqual(removed[0]["removed_from"], "iraes")
        self.assertEqual(removed[0]["removed_reason"], "falso positivo")
        # the id assigned before the removal is preserved.
        self.assertEqual(removed[0]["finding_id"], "irAE-1")
        self.assertEqual(log[0]["moved_to"], "removed")
        # the finding no longer lives in the visible lists.
        self.assertEqual(corrected["iraes"], [])
        self.assertIs(
            corrected["iraes"], corrected["consolidation"]["iraes"]
        )

    def test_remove_suspect_lands_in_removed_bucket(self):
        corrected, _ = apply_irae_corrections(_report(), [
            IraeCorrection(action="remove", irAE_type="Rash sospetto",
                           first_onset_date="2022-12-13"),
        ])
        removed = corrected["consolidation"]["removed"]
        self.assertEqual(len(removed), 1)
        self.assertEqual(removed[0]["removed_from"], "suspects")
        self.assertEqual(
            [s["irAE_type"] for s in corrected["consolidation"]["suspects"]],
            [],
        )
        self.assertEqual(len(corrected["iraes"]), 1)

    def test_remove_does_not_renumber_remaining_findings(self):
        corrected, _ = apply_irae_corrections(_report(), [
            IraeCorrection(action="remove", irAE_type="Miocardite da ICI",
                           first_onset_date="2022-11-22"),
        ])
        suspects = corrected["consolidation"]["suspects"]
        self.assertEqual(suspects[0]["finding_id"], "irAE-2")

    def test_remove_idempotent_over_fresh_report(self):
        corrections = [
            IraeCorrection(action="remove", irAE_type="Miocardite da ICI",
                           first_onset_date="2022-11-22",
                           reason="duplicato clinico"),
        ]
        corrected1, _ = apply_irae_corrections(_report(), corrections)
        corrected2, _ = apply_irae_corrections(_report(), corrections)
        self.assertEqual(corrected1, corrected2)
        self.assertEqual(
            corrected1["consolidation"]["removed"][0]["removed_reason"],
            "duplicato clinico",
        )

    def test_second_remove_of_same_finding_is_noop(self):
        corrected, log = apply_irae_corrections(_report(), [
            IraeCorrection(action="remove", irAE_type="Miocardite da ICI",
                           first_onset_date="2022-11-22"),
            IraeCorrection(action="remove", irAE_type="Miocardite da ICI",
                           first_onset_date="2022-11-22"),
        ])
        self.assertTrue(log[0]["applied"])
        self.assertFalse(log[1]["applied"])
        self.assertEqual(len(corrected["consolidation"]["removed"]), 1)

    def test_add_after_remove_gets_non_colliding_id(self):
        corrected, _ = apply_irae_corrections(_report(), [
            IraeCorrection(action="remove", irAE_type="Miocardite da ICI",
                           first_onset_date="2022-11-22"),
            IraeCorrection(action="add", irAE_type="Colite da ICI",
                           fields={"probability_immune": "PROBABILE"}),
        ])
        added = next(
            f for f in corrected["iraes"] if f["irAE_type"] == "Colite da ICI"
        )
        self.assertEqual(added["finding_id"], "irAE-3")
        removed_ids = [
            r["finding_id"] for r in corrected["consolidation"]["removed"]
        ]
        self.assertNotIn(added["finding_id"], removed_ids)

    def test_edit_merges_whitelisted_fields(self):
        corrected, log = apply_irae_corrections(_report(), [
            IraeCorrection(action="edit", irAE_type="Miocardite da ICI",
                           first_onset_date="2022-11-22",
                           fields={"ctcae_grade": "G4"}),
        ])
        finding = corrected["iraes"][0]
        self.assertEqual(finding["ctcae_grade"], "G4")
        self.assertEqual(finding["notes"], "originale")  # untouched kept
        self.assertTrue(log[0]["applied"])

    def test_edit_repartitions_on_probability_change(self):
        corrected, log = apply_irae_corrections(_report(), [
            IraeCorrection(action="edit", irAE_type="Miocardite da ICI",
                           first_onset_date="2022-11-22",
                           fields={"probability_immune": "POSSIBILE",
                                   "ctcae_grade": "G3"}),
        ])
        self.assertEqual(corrected["iraes"], [])
        suspects = corrected["consolidation"]["suspects"]
        self.assertEqual(len(suspects), 2)
        moved = next(s for s in suspects if s["irAE_type"] == "Miocardite da ICI")
        self.assertEqual(moved["probability_immune"], "POSSIBILE")
        self.assertEqual(moved["ctcae_grade"], "G3")
        self.assertEqual(log[0]["moved_to"], "suspects")

    def test_edit_to_excluded_probability_drops_finding(self):
        corrected, log = apply_irae_corrections(_report(), [
            IraeCorrection(action="edit", irAE_type="Miocardite da ICI",
                           first_onset_date="2022-11-22",
                           fields={"probability_immune": "IMPROBABILE"}),
        ])
        self.assertEqual(corrected["iraes"], [])
        self.assertEqual(log[0]["moved_to"], "excluded")
        self.assertEqual(
            corrected["organ_results"]["Miocardite/Cardiotossicità"]["iraes"],
            [],
        )
        # the excluded path never populates the removed bucket.
        self.assertNotIn("removed", corrected["consolidation"])

    def test_add_places_by_probability(self):
        corrected, log = apply_irae_corrections(_report(), [
            IraeCorrection(action="add", irAE_type="Colite da ICI",
                           first_onset_date="2023-01-10",
                           fields={"ctcae_grade": "G2",
                                   "probability_immune": "PROBABILE",
                                   "notes": "aggiunta manuale"}),
        ])
        added = next(
            f for f in corrected["iraes"] if f["irAE_type"] == "Colite da ICI"
        )
        self.assertEqual(added["ctcae_grade"], "G2")
        self.assertEqual(added["notes"], "aggiunta manuale")
        self.assertTrue(added.get("finding_id"))
        self.assertTrue(log[0]["applied"])

    def test_add_possible_goes_to_suspects(self):
        corrected, _ = apply_irae_corrections(_report(), [
            IraeCorrection(action="add", irAE_type="Rash da ICI",
                           fields={"probability_immune": "POSSIBILE"}),
        ])
        suspects = corrected["consolidation"]["suspects"]
        self.assertTrue(any(s["irAE_type"] == "Rash da ICI" for s in suspects))

    def test_add_refuses_duplicate(self):
        corrected, log = apply_irae_corrections(_report(), [
            IraeCorrection(action="add", irAE_type="Miocardite da ICI",
                           first_onset_date="2022-11-22",
                           fields={"probability_immune": "PROBABILE"}),
        ])
        self.assertFalse(log[0]["applied"])
        self.assertEqual(log[0]["reason"], "duplicato")
        self.assertEqual(len(corrected["iraes"]), 1)

    def test_match_by_finding_id(self):
        corrected, _ = apply_irae_corrections(_report(), [])
        finding_id = corrected["iraes"][0]["finding_id"]
        corrected2, log = apply_irae_corrections(_report(), [
            IraeCorrection(action="edit", irAE_type="Miocardite da ICI",
                           finding_id=finding_id,
                           fields={"ctcae_grade": "G4"}),
        ])
        self.assertTrue(log[0]["applied"])
        self.assertEqual(corrected2["iraes"][0]["ctcae_grade"], "G4")

    def test_stale_finding_id_falls_back_to_clinical_identity(self):
        # finding_id is stale (finding set shifted on a re-analysis), but the
        # (type, date) identity still targets the right clinical finding.
        corrected, log = apply_irae_corrections(_report(), [
            IraeCorrection(action="remove", irAE_type="Miocardite da ICI",
                           finding_id="irAE-999"),
        ])
        self.assertTrue(log[0]["applied"])
        self.assertEqual(corrected["iraes"], [])

    def test_unknown_finding_is_noop(self):
        corrected, log = apply_irae_corrections(_report(), [
            IraeCorrection(action="remove", irAE_type="Nessun tale irAE"),
        ])
        self.assertFalse(log[0]["applied"])
        self.assertEqual(len(corrected["iraes"]), 1)

    def test_ambiguous_type_only_match_uses_first_and_logs(self):
        report = _report()
        report["consolidation"]["suspects"] = [
            _finding(irAE_type="Epatite", first_onset_date="2022-11-01",
                     probability_immune="POSSIBILE"),
            _finding(irAE_type="Epatite", first_onset_date="2023-02-01",
                     probability_immune="POSSIBILE"),
        ]
        corrected, log = apply_irae_corrections(report, [
            IraeCorrection(action="remove", irAE_type="Epatite"),
        ])
        remaining = corrected["consolidation"]["suspects"]
        self.assertEqual(
            [s["first_onset_date"] for s in remaining], ["2023-02-01"]
        )
        self.assertTrue(log[0]["ambiguous"])

    def test_whitelist_ignores_unknown_fields(self):
        corrected, _ = apply_irae_corrections(_report(), [
            IraeCorrection(action="edit", irAE_type="Miocardite da ICI",
                           first_onset_date="2022-11-22",
                           fields={"probability_immune": "PROBABILE",
                                   "patient_id": "hack",
                                   "document_id": "evil"}),
        ])
        finding = corrected["iraes"][0]
        self.assertNotIn("patient_id", finding)
        self.assertNotIn("document_id", finding)
        self.assertEqual(finding["probability_immune"], "PROBABILE")

    def test_apply_is_idempotent_and_does_not_mutate_input(self):
        corrections = [
            IraeCorrection(action="remove", irAE_type="Rash sospetto",
                           first_onset_date="2022-12-13"),
            IraeCorrection(action="edit", irAE_type="Miocardite da ICI",
                           first_onset_date="2022-11-22",
                           fields={"ctcae_grade": "G3"}),
            IraeCorrection(action="add", irAE_type="Colite da ICI",
                           fields={"probability_immune": "PROBABILE"}),
        ]
        report = _report()
        before = copy.deepcopy(report)
        corrected1, _ = apply_irae_corrections(report, corrections)
        self.assertEqual(report, before)
        corrected2, _ = apply_irae_corrections(report, corrections)
        self.assertEqual(corrected1, corrected2)
        # replay over a fresh, equivalent report yields the same result.
        fresh = copy.deepcopy(report)
        corrected3, _ = apply_irae_corrections(fresh, corrections)
        self.assertEqual(corrected3, corrected1)

    def test_no_corrections_returns_annotated_copy(self):
        report = _report()
        before = copy.deepcopy(report)
        corrected, log = apply_irae_corrections(report, [])
        self.assertEqual(log, [])
        self.assertEqual(report, before)
        self.assertEqual(corrected["iraes"][0]["finding_id"], "irAE-1")


if __name__ == "__main__":
    unittest.main()

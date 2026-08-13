"""Tests for the golden validation set (Phase 4).

Covers two layers:

* the ``is_golden`` flag roundtrip through ``TimelineRepository``
  (save_batch → set_golden → get_by_patient, and the critical guarantee
  that ``replace_all_for_patient`` PRESERVES the flag of re-saved
  entries, since ``INSERT OR REPLACE`` with an explicit column list
  would otherwise reset it to 0);
* ``build_golden_set`` aggregation: only confirmed timeline entries,
  only RESOLVED validation decisions (never pending/deferred) and only
  user-validated lab values, with the optional ``patient_id`` filter.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from emr_analyzer.database.engine import DatabaseEngine
from emr_analyzer.database.migrations import init_database
from emr_analyzer.database.patient_repo import PatientRepository
from emr_analyzer.database.timeline_repo import TimelineRepository
from emr_analyzer.export.golden_set import build_golden_set, save_golden_set
from emr_analyzer.models import Patient
from emr_analyzer.models.clinical_timeline import ClinicalTimelineEntry


def make_entry(entry_id, patient_id="P001", date="2024-01-10",
               category="diagnosis", description="desc",
               is_golden=0):
    return ClinicalTimelineEntry(
        entry_id=entry_id,
        patient_id=patient_id,
        date_observed=date,
        category=category,
        description=description,
        is_golden=is_golden,
    )


class GoldenSetTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.db = DatabaseEngine(Path(self._tmp.name) / "registry.db")
        init_database(self.db)
        self.patient_repo = PatientRepository(self.db)
        self.patient_repo.insert(Patient(id="P001", pseudonym="001"))
        self.patient_repo.insert(Patient(id="P002", pseudonym="002"))
        self.db.execute(
            """INSERT INTO documents
               (id, patient_id, filename, original_path, file_hash,
                document_type, import_date)
               VALUES ('DOC_1', 'P001', 'lab.pdf', '/tmp/lab.pdf', 'h',
                       'laboratorio', '2024-01-10')"""
        )
        self.repo = TimelineRepository(self.db)

    def tearDown(self):
        self.db.close()
        self._tmp.cleanup()

    def _insert_validation(self, patient_id, status, issue="issue"):
        self.db.execute(
            """INSERT INTO validation_queue
               (patient_id, item_type, item_id, issue, severity, status,
                original_value, corrected_value, created_at, resolved_at)
               VALUES (?, 'lab', 'lab_1', ?, 'medium', ?, 'orig', 'corr',
                       '2024-01-01T00:00:00', '2024-01-02T00:00:00')""",
            (patient_id, issue, status),
        )

    def _insert_lab(self, patient_id, validated):
        self.db.execute(
            """INSERT INTO lab_values
               (patient_id, document_id, parameter_name, normalized_name,
                value, value_text, unit, reference_text, flag, sample_date,
                validated_by_user)
               VALUES (?, 'DOC_1', 'Emoglobina', 'emoglobina', 9.1,
                       NULL, 'g/dl', '13-17', 'L', '2024-01-10', ?)""",
            (patient_id, 1 if validated else 0),
        )

    # ------------------------------------------------------------------
    # is_golden roundtrip through TimelineRepository
    # ------------------------------------------------------------------

    def test_is_golden_defaults_to_zero(self):
        self.repo.save_batch([
            make_entry("CTL_000001"),
            make_entry("CTL_000002"),
        ])
        entries = self.repo.get_by_patient("P001")
        self.assertEqual([e.is_golden for e in entries], [0, 0])

    def test_set_golden_and_get_by_patient(self):
        self.repo.save_batch([make_entry("CTL_000001")])
        self.repo.set_golden("CTL_000001", True)
        entry = self.repo.get_by_patient("P001")[0]
        self.assertEqual(entry.is_golden, 1)

        self.repo.set_golden("CTL_000001", False)
        entry = self.repo.get_by_patient("P001")[0]
        self.assertEqual(entry.is_golden, 0)

    def test_replace_all_preserves_is_golden(self):
        """The critical INSERT OR REPLACE guard: re-saving existing entries
        must NOT reset the flag."""
        self.repo.save_batch([make_entry("CTL_000001")])
        self.repo.set_golden("CTL_000001", True)

        # Simulate a re-save of existing entries (dedup / incremental path).
        self.repo.replace_all_for_patient(
            "P001", [make_entry("CTL_000001", is_golden=1)]
        )
        entries = self.repo.get_by_patient("P001")
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0].is_golden, 1)

    def test_get_golden_filters_by_flag_and_patient(self):
        self.repo.save_batch([
            make_entry("CTL_000001", patient_id="P001", is_golden=1),
            make_entry("CTL_000002", patient_id="P001", is_golden=0),
            make_entry("CTL_000003", patient_id="P002", is_golden=1),
        ])
        all_golden = self.repo.get_golden()
        self.assertEqual(
            {e.entry_id for e in all_golden},
            {"CTL_000001", "CTL_000003"},
        )
        p001_golden = self.repo.get_golden("P001")
        self.assertEqual([e.entry_id for e in p001_golden], ["CTL_000001"])

    def test_update_description_implies_golden(self):
        self.repo.save_batch([make_entry("CTL_000001")])
        self.repo.update_description(
            "CTL_000001", "descrizione canonica corretta"
        )
        entry = self.repo.get_by_patient("P001")[0]
        self.assertEqual(entry.description, "descrizione canonica corretta")
        self.assertEqual(entry.is_golden, 1)

    # ------------------------------------------------------------------
    # build_golden_set aggregation
    # ------------------------------------------------------------------

    def test_build_golden_set_only_golden_and_resolved(self):
        self.repo.save_batch([
            make_entry("CTL_000001", is_golden=1),
            make_entry("CTL_000002", is_golden=0),
        ])
        for status in ("accepted", "corrected", "rejected", "pending",
                       "deferred"):
            self._insert_validation("P001", status)
        self._insert_lab("P001", validated=True)
        self._insert_lab("P001", validated=False)

        data = build_golden_set(self.db, patient_id="P001")

        self.assertEqual(data["schema_version"], 1)
        self.assertIn("exported_at", data)
        self.assertEqual(len(data["patients"]), 1)
        patient = data["patients"][0]
        self.assertEqual(patient["patient_id"], "P001")

        # Only the confirmed timeline entry.
        golden = patient["golden_entries"]
        self.assertEqual(len(golden), 1)
        self.assertEqual(golden[0]["entry_id"], "CTL_000001")
        self.assertEqual(golden[0]["is_golden"], 1)

        # Only resolved decisions, never pending/deferred.
        statuses = {v["status"] for v in patient["validation"]}
        self.assertEqual(statuses, {"accepted", "corrected", "rejected"})
        # Sanity: the resolved rows carry the corrected value.
        corrected = [
            v for v in patient["validation"]
            if v["status"] == "corrected"
        ][0]
        self.assertEqual(corrected["corrected_value"], "corr")

        # Only user-validated labs.
        labs = patient["validated_labs"]
        self.assertEqual(len(labs), 1)
        self.assertEqual(labs[0]["parameter_name"], "Emoglobina")

    def test_build_golden_set_scopes_to_patient(self):
        self.repo.save_batch([
            make_entry("CTL_000001", patient_id="P001", is_golden=1),
            make_entry("CTL_000003", patient_id="P002", is_golden=1),
        ])
        data = build_golden_set(self.db, patient_id="P002")
        self.assertEqual(
            [p["patient_id"] for p in data["patients"]], ["P002"]
        )
        self.assertEqual(
            data["patients"][0]["golden_entries"][0]["entry_id"],
            "CTL_000003",
        )

    def test_build_golden_set_all_patients(self):
        self.repo.save_batch([
            make_entry("CTL_000001", patient_id="P001", is_golden=1),
            make_entry("CTL_000003", patient_id="P002", is_golden=1),
        ])
        data = build_golden_set(self.db)
        self.assertEqual(
            {p["patient_id"] for p in data["patients"]}, {"P001", "P002"}
        )

    def test_save_golden_set_writes_json(self):
        self.repo.save_batch([make_entry("CTL_000001", is_golden=1)])
        path = Path(self._tmp.name) / "golden.json"
        count = save_golden_set(self.db, path, patient_id="P001")
        self.assertEqual(count, 1)
        with open(path, encoding="utf-8") as f:
            payload = json.load(f)
        self.assertEqual(payload["patients"][0]["golden_entries"][0]
                         ["entry_id"], "CTL_000001")


if __name__ == "__main__":
    unittest.main()

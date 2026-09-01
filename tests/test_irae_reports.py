"""Tests for the persistent raw structured irAE report (per-patient JSON).

Pure logic only: path resolution, atomic save, tolerant load, latest-wins
overwrite — no GUI involved.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from emr_analyzer.clinical.irae_reports import (
    REPORT_FILENAME,
    has_report,
    load_report,
    report_path,
    save_report,
)
from emr_analyzer.config import active_workspace


def _report() -> dict:
    return {
        "anchor": {
            "first_drug": "nivolumab", "first_date": "2022-09-01",
            "first_raw": "2022-09-01", "last_drug": "nivolumab",
            "last_date": "2022-09-01", "last_raw": "2022-09-01",
            "occurrences": 1,
        },
        "candidates_total": 2,
        "iraes": [
            {
                "irAE_type": "Miocardite da ICI", "ctcae_grade": "G2",
                "first_onset_date": "2022-11-22",
                "probability_immune": "PROBABILE",
                "key_evidence_ids": ["E-TROP"], "organ": "Miocardite/Cardiotossicità",
            },
        ],
        "consolidation": {
            "applied": True, "input_count": 7, "output_count": 1,
            "iraes": [], "suspects": [],
        },
        "organ_results": {"Miocardite/Cardiotossicità": {"iraes": []}},
        "analyzed_at": "2026-09-01T10:00:00",
    }


class IraeReportsTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self._patcher = mock.patch.object(
            active_workspace, "path", Path(self._tmp.name)
        )
        self._patcher.start()
        self.addCleanup(self._patcher.stop)

    def test_report_path_is_under_patient_dir(self):
        self.assertEqual(
            report_path("P001"),
            Path(self._tmp.name) / "P001" / REPORT_FILENAME,
        )

    def test_save_load_roundtrip(self):
        save_report("P001", _report())
        loaded = load_report("P001")
        self.assertEqual(loaded, _report())
        self.assertTrue(report_path("P001").is_file())

    def test_has_report(self):
        self.assertFalse(has_report("P001"))
        save_report("P001", _report())
        self.assertTrue(has_report("P001"))

    def test_load_missing_returns_none(self):
        self.assertIsNone(load_report("P999"))

    def test_load_corrupt_returns_none(self):
        path = report_path("P001")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{non-json", encoding="utf-8")
        self.assertIsNone(load_report("P001"))

    def test_load_non_dict_returns_none(self):
        save_report("P001", _report())
        path = report_path("P001")
        path.write_text("[1, 2, 3]", encoding="utf-8")
        self.assertIsNone(load_report("P001"))

    def test_save_overwrites_latest(self):
        first = _report()
        second = dict(first, analyzed_at="2026-09-01T11:00:00")
        save_report("P001", first)
        save_report("P001", second)
        self.assertEqual(load_report("P001"), second)

    def test_saved_file_is_utf8_json(self):
        save_report("P001", _report())
        raw = report_path("P001").read_text(encoding="utf-8")
        self.assertEqual(json.loads(raw), _report())


if __name__ == "__main__":
    unittest.main()

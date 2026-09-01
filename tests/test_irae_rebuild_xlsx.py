"""Tests for ``tools/irae_rebuild_from_xlsx.py``.

The tool reconstructs persistent ``irae_report.json`` files from an Excel
export (the only surviving copy of the LLM findings) + the project registry
(which recomputes ``anchor``/``candidates``/``evidence`` deterministically).
These tests exercise the pure pipeline and one end-to-end save/load
round-trip; no GUI involved.
"""

from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from emr_analyzer.clinical.irae_export import write_irae_xlsx
from emr_analyzer.clinical.irae_reports import has_report, load_report
from emr_analyzer.config import active_workspace

from tools.irae_rebuild_from_xlsx import (
    finding_from_row,
    load_registry_rows,
    rebuild_report,
    rows_from_excel,
)

# One exported row, as the queue's Layer 4 would emit it.
BASE_FINDING = {
    "irAE_type": "Miocardite da ICI",
    "ctcae_grade": "G2",
    "first_onset_date": "2022-11-22",
    "probability_immune": "PROBABILE",
    "new_onset_vs_exacerbation": "insorgenza nuova",
    "alternative_causes": "Nessuna alternativa evidente",
    "confidence": 0.85,
    "key_evidence_ids": ["#E-TROP"],
    "source_organs": ["Miocardite/Cardiotossicità"],
    "notes": "Troponina elevata dopo inizio terapia",
}

REGISTRY_ROWS = [
    {
        "evidence_id": "E-MED", "category": "medication",
        "normalized_entity": "nivolumab", "observed_date": "2022-09-01",
        "data_json": None, "document_id": "DOC-MED",
        "value_text": "", "numeric_value": None, "unit": "",
        "source_page": 1, "source_text": "nivolumab",
        "bbox": None,
    },
    {
        "evidence_id": "E-TROP", "category": "laboratory_finding",
        "normalized_entity": "troponina_i_hs", "observed_date": "2022-11-22",
        "data_json": {"quote_verified": "Troponina elevata"},
        "document_id": "DOC-TROP", "value_text": "1117",
        "numeric_value": 1117.0, "unit": "ng/L",
        "source_page": 2, "source_text": "Troponina 1117 ng/L",
        "bbox": [46.2, 439.68, 449.91, 450.34],
    },
]


def _report() -> dict:
    return {
        "anchor": {
            "first_drug": "nivolumab", "first_date": "2022-09-01",
            "first_raw": "2022-09-01", "last_drug": "nivolumab",
            "last_date": "2022-11-20", "last_raw": "2022-11-20",
            "occurrences": 1,
        },
        "iraes": [dict(BASE_FINDING)],
        "candidates_total": 1,
        "analyzed_at": "2026-09-01T10:00:00",
    }


def _make_registry(db_path: Path, patient_id: str, rows: list[dict]) -> None:
    con = sqlite3.connect(db_path)
    con.execute("DROP TABLE IF EXISTS clinical_evidence")
    con.execute(
        "CREATE TABLE clinical_evidence ("
        "  id INTEGER PRIMARY KEY AUTOINCREMENT,"
        "  patient_id TEXT, evidence_id TEXT, category TEXT,"
        "  normalized_entity TEXT, observed_date TEXT, data_json TEXT,"
        "  document_id TEXT, value_text TEXT, numeric_value REAL, unit TEXT,"
        "  source_page INTEGER, source_text TEXT, bbox_json TEXT)"
    )
    for row in rows:
        con.execute(
            "INSERT INTO clinical_evidence ("
            " patient_id, evidence_id, category, normalized_entity,"
            " observed_date, data_json, document_id, value_text,"
            " numeric_value, unit, source_page, source_text, bbox_json)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                patient_id, row["evidence_id"], row["category"],
                row["normalized_entity"], row["observed_date"],
                json.dumps(row["data_json"]) if row.get("data_json") else None,
                row["document_id"], row["value_text"], row["numeric_value"],
                row["unit"], row["source_page"], row["source_text"],
                json.dumps(row["bbox"]) if row.get("bbox") else None,
            ),
        )
    con.commit()
    con.close()


class IraeRebuildXlsxTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.registry = self.root / "emr_registry.db"
        _make_registry(self.registry, "P001", REGISTRY_ROWS)

    def _write_xlsx(self, report: dict, name: str = "out.xlsx") -> Path:
        path = self.root / name
        write_irae_xlsx([{
            "patient_id": "P001",
            "label": "P001",
            "markdown": "",
            "structured": report,
            "error": None,
        }], str(path))
        return path

    # ------------------------------------------------------------------
    # Excel → findings
    # ------------------------------------------------------------------

    def test_rows_from_excel_reads_exported_columns(self):
        rows = rows_from_excel(self._write_xlsx(_report()))
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["patient_id"], "P001")
        self.assertEqual(row["irAE_type"], "Miocardite da ICI")
        self.assertEqual(row["organ"], "Miocardite/Cardiotossicità")
        # the export already strips the "#" prefix from cited ids.
        self.assertEqual(row["key_evidence_ids"], "E-TROP")
        self.assertEqual(row["analyzed_at"], "2026-09-01T10:00:00")
        self.assertEqual(row["confidence"], 0.85)

    def test_rows_from_excel_missing_columns_raises(self):
        from openpyxl import Workbook

        bogus = self.root / "bogus.xlsx"
        workbook = Workbook()
        sheet = workbook.active
        sheet.append(["A", "B", "C"])
        sheet.append(["1", "2", "3"])
        workbook.save(str(bogus))

        with self.assertRaises(ValueError) as ctx:
            rows_from_excel(bogus)
        self.assertIn("mancanti", str(ctx.exception))

    def test_finding_from_row_preserves_fields_and_normalizes_ids(self):
        row = rows_from_excel(self._write_xlsx(_report()))[0]
        finding = finding_from_row(row)
        self.assertEqual(finding["irAE_type"], "Miocardite da ICI")
        self.assertEqual(finding["ctcae_grade"], "G2")
        self.assertEqual(finding["first_onset_date"], "2022-11-22")
        self.assertEqual(finding["probability_immune"], "PROBABILE")
        self.assertEqual(finding["confidence"], 0.85)
        # the "#" prefix is normalized away, as the pipeline stores ids.
        self.assertEqual(finding["key_evidence_ids"], ["E-TROP"])
        # source_organs rebuilt from the comma-joined "organ" column.
        self.assertEqual(
            finding["source_organs"], ["Miocardite/Cardiotossicità"]
        )
        self.assertEqual(finding["organ"], "Miocardite/Cardiotossicità")

    # ------------------------------------------------------------------
    # Registry → deterministic layers
    # ------------------------------------------------------------------

    def test_load_registry_rows_parses_bbox_and_data_json(self):
        rows = load_registry_rows(self.registry, "P001")
        self.assertEqual(len(rows), 2)
        by_id = {row["evidence_id"]: row for row in rows}
        trop = by_id["E-TROP"]
        self.assertEqual(trop["bbox"], [46.2, 439.68, 449.91, 450.34])
        self.assertEqual(trop["data_json"]["quote_verified"], "Troponina elevata")
        self.assertEqual(trop["source_page"], 2)
        self.assertEqual(trop["source_text"], "Troponina 1117 ng/L")

    def test_load_registry_rows_unknown_patient_empty(self):
        self.assertEqual(load_registry_rows(self.registry, "P999"), [])

    # ------------------------------------------------------------------
    # Report reconstruction
    # ------------------------------------------------------------------

    def test_rebuild_report_recomputes_anchor_candidates_evidence(self):
        finding = finding_from_row(rows_from_excel(self._write_xlsx(_report()))[0])
        rows = load_registry_rows(self.registry, "P001")
        report = rebuild_report(rows, [finding], analyzed_at="2026-09-01T10:00:00")

        self.assertEqual(report["anchor"]["first_drug"], "nivolumab")
        self.assertEqual(report["candidates_total"], 1)
        self.assertEqual(report["consolidation"]["applied"], True)
        self.assertEqual(report["consolidation"]["suspects"], [])
        self.assertEqual(report["organ_results"], {})
        self.assertEqual(report["analyzed_at"], "2026-09-01T10:00:00")
        self.assertEqual(report["iraes"], [finding])

        # candidates carry full provenance for the inspector.
        candidate = report["candidates"][0]
        self.assertEqual(candidate["organ"], "Miocardite/Cardiotossicità")
        self.assertEqual(candidate["evidence_id"], "E-TROP")
        self.assertEqual(candidate["source_page"], 2)

        # evidence = only the cited ids, with provenance.
        self.assertEqual(len(report["evidence"]), 1)
        evidence = report["evidence"][0]
        self.assertEqual(evidence["evidence_id"], "E-TROP")
        self.assertEqual(evidence["bbox"], [46.2, 439.68, 449.91, 450.34])

    def test_rebuild_report_with_empty_registry_rows_still_builds(self):
        report = rebuild_report([], [dict(BASE_FINDING, key_evidence_ids=["E-TROP"])])
        self.assertEqual(report["anchor"], None)
        self.assertEqual(report["candidates"], [])
        self.assertEqual(report["evidence"], [])
        self.assertEqual(len(report["iraes"]), 1)

    def test_rebuild_report_evidence_skips_ids_missing_in_registry(self):
        finding = dict(BASE_FINDING, key_evidence_ids=["E-TROP", "E-GONE"])
        report = rebuild_report(
            load_registry_rows(self.registry, "P001"), [finding]
        )
        ids = [item["evidence_id"] for item in report["evidence"]]
        self.assertEqual(ids, ["E-TROP"])

    # ------------------------------------------------------------------
    # End to end: Excel + registry → persisted irae_report.json
    # ------------------------------------------------------------------

    def test_end_to_end_save_load_roundtrip(self):
        xlsx = self._write_xlsx(_report())
        rows = rows_from_excel(xlsx)
        findings = [finding_from_row(row) for row in rows]
        registry_rows = load_registry_rows(self.registry, "P001")
        report = rebuild_report(
            registry_rows, findings,
            analyzed_at=rows[0]["analyzed_at"],
        )

        with mock.patch.object(active_workspace, "path", self.root):
            from tools.irae_rebuild_from_xlsx import save_report

            save_report("P001", report)
            self.assertTrue(has_report("P001"))
            loaded = load_report("P001")
        self.assertEqual(loaded["iraes"], findings)
        self.assertEqual(loaded["candidates"][0]["evidence_id"], "E-TROP")
        self.assertEqual(loaded["consolidation"]["applied"], True)

    # ------------------------------------------------------------------
    # CLI
    # ------------------------------------------------------------------

    def test_main_writes_reports_into_workspace(self):
        from emr_analyzer.config import _ActiveWorkspace

        from tools.irae_rebuild_from_xlsx import main

        xlsx = self._write_xlsx(_report())
        workspace = self.root / "workspace"
        with mock.patch.object(_ActiveWorkspace, "path", workspace):
            exit_code = main([
                "--xlsx", str(xlsx),
                "--project", "RENE",
                "--workspace", str(workspace),
                "--registry", str(self.registry),
            ])
            self.assertEqual(exit_code, 0)
            self.assertTrue(has_report("P001"))
            loaded = load_report("P001")
        self.assertEqual(loaded["iraes"][0]["irAE_type"], "Miocardite da ICI")
        self.assertEqual(loaded["candidates"][0]["evidence_id"], "E-TROP")

    def test_main_missing_registry_returns_2(self):
        from tools.irae_rebuild_from_xlsx import main

        xlsx = self._write_xlsx(_report())
        exit_code = main([
            "--xlsx", str(xlsx),
            "--project", "RENE",
            "--workspace", str(self.root / "nope"),
            "--registry", str(self.root / "missing.db"),
        ])
        self.assertEqual(exit_code, 2)


if __name__ == "__main__":
    unittest.main()

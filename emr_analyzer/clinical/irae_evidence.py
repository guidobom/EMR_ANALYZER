"""Read atomic evidence rows from the project registry (``emr_registry.db``).

``load_registry_rows`` returns rows shaped like
``irae_layers.evidence_rows_from_models`` output (``evidence_id``,
``category``, ``normalized_entity``, ``observed_date``, ``data_json``,
``document_id``, ``value_text``, ``numeric_value``, ``unit``,
``source_page``, ``source_text``, ``bbox``) — the shape
``irae_layers._compact_evidence`` and the deterministic irAE layers need.
Shared by ``tools/irae_rebuild_from_xlsx.py`` and the GUI irAE
reconsolidation feature, so the GUI never imports from ``tools/``.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any


def parse_bbox(raw: Any) -> list | None:
    """``bbox_json`` column → the ``bbox`` list ``_compact_evidence`` wants."""
    if isinstance(raw, (list, tuple)):
        return list(raw)
    if isinstance(raw, str) and raw.strip():
        try:
            parsed = json.loads(raw)
        except ValueError:
            return None
        return list(parsed) if isinstance(parsed, (list, tuple)) else None
    return None


def registry_db_path() -> Path:
    """The project registry database inside the active workspace."""
    from ..config import active_workspace

    return active_workspace.path / "emr_registry.db"


def load_registry_rows(db_path: Path, patient_id: str) -> list[dict[str, Any]]:
    """Atomic evidence rows shaped like ``evidence_rows_from_models`` output.

    The ``clinical_evidence`` table already carries ``source_page``,
    ``source_text`` and ``bbox_json`` at row level (provenance the inspector
    needs to open and highlight the original PDF), so no ``data_json``
    unpacking is required for those.
    """
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    try:
        records = con.execute(
            "SELECT evidence_id, category, normalized_entity, observed_date, "
            "       data_json, document_id, value_text, numeric_value, unit, "
            "       source_page, source_text, bbox_json "
            "FROM clinical_evidence WHERE patient_id = ? "
            "ORDER BY observed_date, id",
            (patient_id,),
        ).fetchall()
    finally:
        con.close()

    rows: list[dict[str, Any]] = []
    for record in records:
        row = dict(record)
        row["bbox"] = parse_bbox(row.get("bbox_json"))
        row.pop("bbox_json", None)
        raw = row.get("data_json")
        if isinstance(raw, str) and raw.strip():
            try:
                row["data_json"] = json.loads(raw)
            except ValueError:
                row["data_json"] = None
        else:
            row["data_json"] = None
        rows.append(row)
    return rows

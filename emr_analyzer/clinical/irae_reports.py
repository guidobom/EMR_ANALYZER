"""Persistent raw structured irAE reports, one file per patient.

The report produced by the structured irAE pipeline (single-patient analysis
and multi-patient queue) is stored as ``<workspace>/<patient_id>/irae_report.json``
so the last analysis of a patient can be reopened at any time — inside the
same session or after an app restart — without re-running the pipeline.

The RAW report is persisted (never the corrected one): manual corrections
live separately in ``irae_corrections.json`` and are re-applied over the raw
report whenever it is shown (see ``apply_irae_corrections``).  A new analysis
atomically overwrites the previous file (the latest analysis wins).

The write pattern mirrors ``irae_corrections.save_corrections`` (tmp + fsync
+ replace) and the read pattern tolerates missing/corrupt files.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from ..config import active_workspace

REPORT_FILENAME = "irae_report.json"


def report_path(patient_id: str) -> Path:
    """Per-patient raw report file inside the active workspace."""
    return active_workspace.path / patient_id / REPORT_FILENAME


def has_report(patient_id: str) -> bool:
    """True when a saved report exists for *patient_id*."""
    return report_path(patient_id).exists()


def load_report(patient_id: str) -> dict[str, Any] | None:
    """The last raw report persisted for *patient_id*; ``None`` when missing
    or corrupt."""
    path = report_path(patient_id)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def save_report(patient_id: str, report: dict[str, Any]) -> None:
    """Atomically persist the raw *report* for *patient_id* (tmp + replace)."""
    path = report_path(patient_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        dir=str(path.parent), prefix=".irae_report_", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise

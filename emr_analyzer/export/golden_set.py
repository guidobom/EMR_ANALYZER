"""Golden validation set export.

Closes the human feedback loop: the corrections and confirmations a clinician
makes in the GUI (confirmed timeline entries, resolved validation decisions,
validated lab values) are exported as a machine-readable golden set used to
evaluate the LLM prompts against ground truth.

Only de-identified clinical content is exported (pseudonymous ``patient_id``,
sanitized clinical text) — never raw patient identity.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

from ..database.timeline_repo import TimelineRepository

# Statuses that represent a human decision. ``pending`` items were never
# reviewed; ``deferred`` items were explicitly postponed and carry no verdict.
_RESOLVED_STATUSES = ("accepted", "corrected", "rejected")


def build_golden_set(db, patient_id: Optional[str] = None) -> dict:
    """Aggregate the golden set for one patient (or the whole registry).

    Combines three sources of human ground truth:

    * timeline entries marked ``is_golden`` (confirmations and canonical
      descriptions edited by the user),
    * resolved ``validation_queue`` rows (accepted / corrected / rejected),
    * ``lab_values`` validated by the user (``validated_by_user=1``).

    Returns a dict shaped as::

        {
          "schema_version": 1,
          "exported_at": "...",
          "patients": [
            {
              "patient_id": "P001",
              "golden_entries": [entry.to_dict(), ...],
              "validation": [...],
              "validated_labs": [...],
            }
          ]
        }
    """
    timeline_repo = TimelineRepository(db)

    if patient_id is not None:
        golden_entries = timeline_repo.get_golden(patient_id)
        entry_rows = [
            {
                "patient_id": patient_id,
                "golden_entries": [e.to_dict() for e in golden_entries],
                "validation": _resolved_validation(db, patient_id),
                "validated_labs": _validated_labs(db, patient_id),
            }
        ]
    else:
        golden_entries = timeline_repo.get_golden()
        # Group golden entries by patient (get_golden() is ordered by patient).
        rows: dict[str, list] = {}
        for e in golden_entries:
            rows.setdefault(e.patient_id, []).append(e)
        entry_rows = []
        for pid, entries in rows.items():
            entry_rows.append({
                "patient_id": pid,
                "golden_entries": [e.to_dict() for e in entries],
                "validation": _resolved_validation(db, pid),
                "validated_labs": _validated_labs(db, pid),
            })

    return {
        "schema_version": 1,
        "exported_at": datetime.now().isoformat(),
        "patients": entry_rows,
    }


def save_golden_set(
    db, path: Path, patient_id: Optional[str] = None
) -> int:
    """Write the golden set to *path* as JSON.

    Returns the number of confirmed (golden) timeline entries exported, so the
    GUI can show a meaningful summary.
    """
    data = build_golden_set(db, patient_id)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return sum(
        len(p["golden_entries"]) for p in data["patients"]
    )


def _resolved_validation(db, patient_id: str) -> list[dict]:
    """Resolved human decisions from the validation queue for a patient."""
    placeholders = ",".join("?" * len(_RESOLVED_STATUSES))
    cursor = db.execute(
        f"""SELECT item_type, item_id, status, issue, severity,
                   original_value, corrected_value, resolved_at
            FROM validation_queue
           WHERE patient_id=? AND status IN ({placeholders})
           ORDER BY resolved_at ASC, id ASC""",
        (patient_id, *_RESOLVED_STATUSES),
    )
    return [dict(r) for r in cursor.fetchall()]


def _validated_labs(db, patient_id: str) -> list[dict]:
    """Lab values explicitly validated by the user for a patient."""
    cursor = db.execute(
        """SELECT id, parameter_name, normalized_name, value, value_text,
                  unit, reference_text, flag, sample_date
             FROM lab_values
            WHERE patient_id=? AND validated_by_user=1
            ORDER BY sample_date ASC, id ASC""",
        (patient_id,),
    )
    return [dict(r) for r in cursor.fetchall()]

"""Clinical State repository."""

import json
from datetime import datetime
from typing import Optional

from .engine import DatabaseEngine
from ..models.clinical_state import ClinicalState, ClinicalStateDelta


class ClinicalStateRepository:
    """Data access for Clinical State persistence."""

    def __init__(self, db: DatabaseEngine):
        self.db = db

    def save(self, state: ClinicalState) -> None:
        """Save or update the clinical state for a patient."""
        state.updated_at = datetime.now().isoformat()
        state.version += 1

        self.db.execute(
            """INSERT OR REPLACE INTO clinical_state (patient_id, state_json,
               updated_at, version)
               VALUES (?, ?, ?, ?)""",
            (state.patient_id, json.dumps(state.to_dict(), ensure_ascii=False),
             state.updated_at, state.version),
        )
        self.db.commit()

    def load(self, patient_id: str) -> Optional[ClinicalState]:
        """Load the clinical state for a patient."""
        cursor = self.db.execute(
            "SELECT * FROM clinical_state WHERE patient_id=?",
            (patient_id,),
        )
        row = cursor.fetchone()
        if row:
            data = json.loads(row["state_json"])
            data["patient_id"] = patient_id
            data["updated_at"] = row["updated_at"]
            data["version"] = row["version"]
            return ClinicalState.from_dict(data)
        return None

    def save_delta(self, patient_id: str, delta: ClinicalStateDelta,
                   document_id: Optional[str] = None,
                   auto_applied: bool = False) -> None:
        """Record a delta in the history table."""
        self.db.execute(
            """INSERT INTO clinical_state_deltas
               (patient_id, delta_json, document_id, applied_at, auto_applied)
               VALUES (?, ?, ?, ?, ?)""",
            (patient_id,
             json.dumps(delta.to_dict(), ensure_ascii=False),
             document_id,
             datetime.now().isoformat(),
             1 if auto_applied else 0),
        )
        self.db.commit()

    def get_delta_history(self, patient_id: str) -> list[dict]:
        cursor = self.db.execute(
            """SELECT * FROM clinical_state_deltas
               WHERE patient_id=? ORDER BY applied_at DESC""",
            (patient_id,),
        )
        return [
            {
                "id": r["id"],
                "patient_id": r["patient_id"],
                "delta": json.loads(r["delta_json"]),
                "document_id": r["document_id"],
                "applied_at": r["applied_at"],
                "auto_applied": bool(r["auto_applied"]),
            }
            for r in cursor.fetchall()
        ]

    def delete(self, patient_id: str) -> None:
        self.db.execute(
            "DELETE FROM clinical_state WHERE patient_id=?", (patient_id,)
        )
        self.db.commit()

    def exists(self, patient_id: str) -> bool:
        cursor = self.db.execute(
            "SELECT COUNT(*) FROM clinical_state WHERE patient_id=?",
            (patient_id,),
        )
        return cursor.fetchone()[0] > 0

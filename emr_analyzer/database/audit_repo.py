"""Audit log repository."""

import json
from datetime import datetime
from typing import Optional

from .engine import DatabaseEngine


class AuditRepository:
    """Data access for the audit log."""

    def __init__(self, db: DatabaseEngine):
        self.db = db

    def log(self, patient_id: str, action: str,
            target_type: Optional[str] = None,
            target_id: Optional[str] = None,
            details: Optional[dict] = None,
            model_used: Optional[str] = None,
            model_version: Optional[str] = None) -> None:
        self.db.execute(
            """INSERT INTO audit_log (patient_id, action, target_type,
               target_id, details_json, model_used, model_version, timestamp)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (patient_id, action, target_type, target_id,
             json.dumps(details, ensure_ascii=False) if details else None,
             model_used, model_version, datetime.now().isoformat()),
        )
        self.db.commit()

    def get_by_patient(self, patient_id: str, limit: int = 100) -> list[dict]:
        cursor = self.db.execute(
            """SELECT * FROM audit_log WHERE patient_id=?
               ORDER BY timestamp DESC LIMIT ?""",
            (patient_id, limit),
        )
        return [
            {
                "id": r["id"],
                "patient_id": r["patient_id"],
                "action": r["action"],
                "target_type": r["target_type"],
                "target_id": r["target_id"],
                "details": json.loads(r["details_json"]) if r["details_json"] else None,
                "model_used": r["model_used"],
                "model_version": r["model_version"],
                "timestamp": r["timestamp"],
            }
            for r in cursor.fetchall()
        ]

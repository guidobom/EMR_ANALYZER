"""Reviewer decisions on extracted events (one row per source occurrence)."""

from __future__ import annotations

from datetime import datetime, timezone
import json

ACTIONS = ("confirm", "correct", "reject", "add")


class EventOverrideRepository:
    def __init__(self, db):
        self.db = db

    def by_patient(self, patient_id: str) -> dict[str, dict]:
        rows = self.db.execute(
            "SELECT * FROM event_overrides WHERE patient_id=? ORDER BY created_at",
            (patient_id,)).fetchall()
        return {row["occurrence_key"]: _row(row) for row in rows}

    def save(self, occurrence_key: str, *, patient_id: str, document_id: str, action: str,
             fields: dict, base_evidence_id: str | None = None, text_hash: str | None = None,
             note: str | None = None, reviewer: str | None = None) -> None:
        if action not in ACTIONS:
            raise ValueError(f"Azione di revisione non valida: {action}")
        now = datetime.now(timezone.utc).isoformat()
        with self.db:
            self.db.execute(
                """INSERT INTO event_overrides (occurrence_key, patient_id, document_id, action,
                       base_evidence_id, fields_json, text_hash, note, reviewer, created_at, updated_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?)
                   ON CONFLICT(occurrence_key) DO UPDATE SET
                       action=excluded.action, base_evidence_id=excluded.base_evidence_id,
                       fields_json=excluded.fields_json, text_hash=excluded.text_hash,
                       note=excluded.note, reviewer=excluded.reviewer, updated_at=excluded.updated_at""",
                (occurrence_key, patient_id, document_id, action, base_evidence_id,
                 json.dumps(fields, ensure_ascii=False, sort_keys=True), text_hash, note,
                 reviewer, now, now))

    def delete(self, occurrence_key: str) -> bool:
        with self.db:
            return self.db.execute("DELETE FROM event_overrides WHERE occurrence_key=?",
                                   (occurrence_key,)).rowcount > 0


def _row(row) -> dict:
    data = dict(row)
    data["fields"] = json.loads(data.pop("fields_json") or "{}")
    return data

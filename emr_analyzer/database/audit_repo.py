"""Audit log repository."""

import json
import hashlib
from datetime import datetime
from typing import Optional

from .engine import DatabaseEngine
from ..security.privacy import sanitize_for_persistence


class AuditRepository:
    """Data access for the audit log."""

    def __init__(self, db: DatabaseEngine):
        self.db = db

    def log(self, patient_id: str, action: str,
            target_type: Optional[str] = None,
            target_id: Optional[str] = None,
            details: Optional[dict] = None,
            model_used: Optional[str] = None,
            model_version: Optional[str] = None,
            *,
            actor_id: str = "system",
            actor_role: str = "system",
            run_id: Optional[str] = None,
            input_hash: Optional[str] = None,
            prompt_hash: Optional[str] = None) -> None:
        safe_details = sanitize_for_persistence(details) if details else None
        timestamp = datetime.now().isoformat()
        details_json = (
            json.dumps(
                safe_details, ensure_ascii=False, sort_keys=True,
                separators=(",", ":"),
            )
            if safe_details else None
        )
        with self.db:
            # INSERT OR IGNORE obtains SQLite's write lock before the chain
            # head is read, serialising concurrent writers across threads.
            self.db.execute(
                """INSERT OR IGNORE INTO audit_chain_heads
                   (patient_id, last_hash, updated_at) VALUES (?, '', ?)""",
                (patient_id, timestamp),
            )
            head = self.db.execute(
                "SELECT last_hash FROM audit_chain_heads WHERE patient_id=?",
                (patient_id,),
            ).fetchone()
            previous_hash = head["last_hash"] if head else ""
            canonical = json.dumps(
                {
                    "patient_id": patient_id,
                    "action": action,
                    "target_type": target_type,
                    "target_id": target_id,
                    "details": safe_details,
                    "model_used": model_used,
                    "model_version": model_version,
                    "timestamp": timestamp,
                    "actor_id": actor_id,
                    "actor_role": actor_role,
                    "run_id": run_id,
                    "input_hash": input_hash,
                    "prompt_hash": prompt_hash,
                    "previous_hash": previous_hash,
                },
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            entry_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
            self.db.execute(
                """INSERT INTO audit_log
                   (patient_id, action, target_type, target_id, details_json,
                    model_used, model_version, timestamp, actor_id, actor_role,
                    run_id, input_hash, prompt_hash, previous_hash, entry_hash)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    patient_id, action, target_type, target_id, details_json,
                    model_used, model_version, timestamp, actor_id, actor_role,
                    run_id, input_hash, prompt_hash, previous_hash, entry_hash,
                ),
            )
            self.db.execute(
                """UPDATE audit_chain_heads SET last_hash=?, updated_at=?
                   WHERE patient_id=?""",
                (entry_hash, timestamp, patient_id),
            )

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
                "actor_id": r["actor_id"],
                "actor_role": r["actor_role"],
                "run_id": r["run_id"],
                "input_hash": r["input_hash"],
                "prompt_hash": r["prompt_hash"],
                "previous_hash": r["previous_hash"],
                "entry_hash": r["entry_hash"],
                "timestamp": r["timestamp"],
            }
            for r in cursor.fetchall()
        ]

    def verify_chain(self, patient_id: str) -> tuple[bool, list[int]]:
        """Verify hashes for post-v12 entries; legacy unhashed rows are skipped."""
        rows = self.db.execute(
            """SELECT * FROM audit_log WHERE patient_id=?
               AND entry_hash IS NOT NULL ORDER BY id""",
            (patient_id,),
        ).fetchall()
        previous = ""
        invalid: list[int] = []
        for row in rows:
            if row["previous_hash"] != previous:
                invalid.append(row["id"])
            details = (
                json.loads(row["details_json"]) if row["details_json"] else None
            )
            canonical = json.dumps(
                {
                    "patient_id": row["patient_id"],
                    "action": row["action"],
                    "target_type": row["target_type"],
                    "target_id": row["target_id"],
                    "details": details,
                    "model_used": row["model_used"],
                    "model_version": row["model_version"],
                    "timestamp": row["timestamp"],
                    "actor_id": row["actor_id"],
                    "actor_role": row["actor_role"],
                    "run_id": row["run_id"],
                    "input_hash": row["input_hash"],
                    "prompt_hash": row["prompt_hash"],
                    "previous_hash": row["previous_hash"],
                },
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            calculated = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
            if calculated != row["entry_hash"]:
                invalid.append(row["id"])
            previous = row["entry_hash"]
        return not invalid, sorted(set(invalid))

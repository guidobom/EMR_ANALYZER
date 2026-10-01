"""Durable results and per-call metrics for independently resumable extraction."""
from datetime import datetime, timezone
import json


class AtomicGroupRepository:
    def __init__(self, db):
        self.db = db

    def load(self, patient_id, request_key):
        row = self.db.execute(
            """SELECT status, result_json FROM atomic_group_results
               WHERE patient_id=? AND request_key=? ORDER BY updated_at DESC LIMIT 1""",
            (patient_id, request_key),
        ).fetchone()
        if row is None:
            return None
        try:
            return {"status": row[0], "result": json.loads(row[1])}
        except (TypeError, ValueError):
            return None

    def save(self, patient_id, document_id, request_key, status, result, metrics, error=None):
        now = datetime.now(timezone.utc).isoformat()
        with self.db:
            self.db.execute(
                """INSERT INTO atomic_group_results
                   (patient_id, document_id, request_key, status, result_json,
                    metrics_json, error_message, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(document_id, request_key) DO UPDATE SET
                   status=excluded.status, result_json=excluded.result_json,
                   metrics_json=excluded.metrics_json,
                   error_message=excluded.error_message, updated_at=excluded.updated_at""",
                (patient_id, document_id, request_key, status,
                 json.dumps(result, ensure_ascii=False), json.dumps(metrics), error, now),
            )

    def record_call(self, patient_id, document_id, request_key, phase, status, metrics):
        self.db.execute(
            """INSERT INTO atomic_group_calls
               (patient_id, document_id, request_key, phase, status, metrics_json, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (patient_id, document_id, request_key, phase, status,
             json.dumps(metrics), datetime.now(timezone.utc).isoformat()),
        )
        self.db.commit()

    def summary(self, patient_id):
        rows = self.db.execute(
            """SELECT status, COUNT(*) AS n FROM atomic_group_results
               WHERE patient_id=? GROUP BY status""", (patient_id,),
        ).fetchall()
        return {row['status']: row['n'] for row in rows}

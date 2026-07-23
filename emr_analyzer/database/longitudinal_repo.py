"""Persistence for longitudinal Clinical State runs and evidence ledger."""

from __future__ import annotations

from datetime import datetime
import json
import uuid

from .engine import DatabaseEngine
from ..models.longitudinal import LongitudinalEvidence


class LongitudinalRepository:
    def __init__(self, db: DatabaseEngine):
        self.db = db

    def find_resumable_run(
        self, patient_id: str, model_name: str, source_signature: str,
        prompt_version: str, schema_version: str,
    ) -> dict | None:
        row = self.db.execute(
            """SELECT * FROM clinical_state_runs
               WHERE patient_id=? AND model_name=? AND source_signature=?
                 AND prompt_version=? AND schema_version=?
                 AND status IN ('running', 'failed', 'cancelled')
               ORDER BY started_at DESC LIMIT 1""",
            (
                patient_id, model_name, source_signature,
                prompt_version, schema_version,
            ),
        ).fetchone()
        return dict(row) if row else None

    def create_run(
        self, patient_id: str, model_name: str, source_signature: str,
        document_count: int, prompt_version: str, schema_version: str,
        capability: dict,
    ) -> str:
        run_id = f"CSR_{uuid.uuid4().hex.upper()}"
        self.db.execute(
            """INSERT INTO clinical_state_runs
               (run_id, patient_id, model_name, prompt_version,
                schema_version, status, source_signature, document_count,
                processed_count, capability_json, started_at)
               VALUES (?, ?, ?, ?, ?, 'running', ?, ?, 0, ?, ?)""",
            (
                run_id, patient_id, model_name, prompt_version,
                schema_version, source_signature, document_count,
                json.dumps(capability, ensure_ascii=False),
                datetime.now().isoformat(),
            ),
        )
        self.db.commit()
        return run_id

    def resume_run(self, run_id: str) -> None:
        self.db.execute(
            """UPDATE clinical_state_runs
               SET status='running', error_message=NULL WHERE run_id=?""",
            (run_id,),
        )
        self.db.commit()

    def prepare_checkpoints(self, run_id: str, documents: list[dict]) -> None:
        self.db.executemany(
            """INSERT OR IGNORE INTO clinical_state_checkpoints
               (run_id, document_id, sequence_index, temporal_group,
                document_date, date_source, status)
               VALUES (?, ?, ?, ?, ?, ?, 'pending')""",
            [
                (
                    run_id, item["document"].id, index,
                    item["temporal_group"], item["document_date"],
                    item["date_source"],
                )
                for index, item in enumerate(documents)
            ],
        )
        self.db.commit()

    def completed_document_ids(self, run_id: str) -> set[str]:
        rows = self.db.execute(
            """SELECT document_id FROM clinical_state_checkpoints
               WHERE run_id=? AND status='completed'""",
            (run_id,),
        ).fetchall()
        return {row["document_id"] for row in rows}

    def start_checkpoint(self, run_id: str, document_id: str) -> None:
        self.db.execute(
            """UPDATE clinical_state_checkpoints SET status='processing',
               error_message=NULL WHERE run_id=? AND document_id=?""",
            (run_id, document_id),
        )
        self.db.execute(
            """UPDATE clinical_state_runs SET current_document_id=?
               WHERE run_id=?""",
            (document_id, run_id),
        )
        self.db.commit()

    def complete_checkpoint(
        self, run_id: str, document_id: str, delta: dict,
    ) -> None:
        now = datetime.now().isoformat()
        with self.db:
            self.db.execute(
                """UPDATE clinical_state_checkpoints
                   SET status='completed', delta_json=?, processed_at=?,
                       error_message=NULL
                   WHERE run_id=? AND document_id=?""",
                (
                    json.dumps(delta, ensure_ascii=False), now,
                    run_id, document_id,
                ),
            )
            self.db.execute(
                """UPDATE clinical_state_runs
                   SET processed_count=(
                       SELECT COUNT(*) FROM clinical_state_checkpoints
                       WHERE run_id=? AND status='completed'
                   ) WHERE run_id=?""",
                (run_id, run_id),
            )

    def fail_checkpoint(
        self, run_id: str, document_id: str, error_message: str,
    ) -> None:
        self.db.execute(
            """UPDATE clinical_state_checkpoints
               SET status='error', error_message=?
               WHERE run_id=? AND document_id=?""",
            (error_message, run_id, document_id),
        )
        self.db.commit()

    def insert_evidence(self, item: LongitudinalEvidence) -> None:
        self.db.execute(
            """INSERT INTO longitudinal_evidence
               (evidence_id, run_id, patient_id, document_id, operation,
                target_evidence_id, category, normalized_entity, assertion,
                clinical_status, valid_start_date, valid_end_date, asserted_at,
                date_precision, original_date_expression, anchor_date,
                date_derivation_rule, value_text, numeric_value, unit, grade,
                grading_system, source_page, source_text, confidence,
                objective_basis, correction_explicit, canonical_status,
                model_name, prompt_version, schema_version, data_json,
                created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                       ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                item.evidence_id, item.run_id, item.patient_id,
                item.document_id, item.operation, item.target_evidence_id,
                item.category, item.normalized_entity, item.assertion,
                item.clinical_status, item.valid_start_date,
                item.valid_end_date, item.asserted_at, item.date_precision,
                item.original_date_expression, item.anchor_date,
                item.date_derivation_rule, item.value_text,
                item.numeric_value, item.unit, item.grade,
                item.grading_system, item.source_page, item.source_text,
                item.confidence, 1 if item.objective_basis else 0,
                1 if item.correction_explicit else 0,
                item.canonical_status, item.model_name, item.prompt_version,
                item.schema_version,
                json.dumps(item.data, ensure_ascii=False), item.created_at,
            ),
        )
        self.db.commit()

    def get_evidence(self, evidence_id: str, run_id: str | None = None) -> dict | None:
        sql = "SELECT * FROM longitudinal_evidence WHERE evidence_id=?"
        params: tuple = (evidence_id,)
        if run_id:
            sql += " AND run_id=?"
            params += (run_id,)
        row = self.db.execute(sql, params).fetchone()
        return self._row(row) if row else None

    def update_canonical_status(
        self, evidence_id: str, status: str,
        valid_end_date: str | None = None,
    ) -> None:
        if valid_end_date is None:
            self.db.execute(
                """UPDATE longitudinal_evidence SET canonical_status=?
                   WHERE evidence_id=?""",
                (status, evidence_id),
            )
        else:
            self.db.execute(
                """UPDATE longitudinal_evidence
                   SET canonical_status=?, valid_end_date=?
                   WHERE evidence_id=?""",
                (status, valid_end_date, evidence_id),
            )
        self.db.commit()

    def list_context(self, run_id: str) -> list[dict]:
        rows = self.db.execute(
            """SELECT * FROM longitudinal_evidence
               WHERE run_id=? AND canonical_status IN
                   ('active', 'historical', 'resolved')
               ORDER BY COALESCE(valid_start_date, asserted_at), id""",
            (run_id,),
        ).fetchall()
        return [self._row(row) for row in rows]

    def list_ledger(self, run_id: str) -> list[dict]:
        rows = self.db.execute(
            """SELECT * FROM longitudinal_evidence
               WHERE run_id=? ORDER BY asserted_at, id""",
            (run_id,),
        ).fetchall()
        return [self._row(row) for row in rows]

    def find_same_clinical_key(
        self, run_id: str, category: str, normalized_entity: str,
        valid_start_date: str | None,
    ) -> list[dict]:
        rows = self.db.execute(
            """SELECT * FROM longitudinal_evidence
               WHERE run_id=? AND category=?
                 AND lower(normalized_entity)=lower(?)
                 AND COALESCE(valid_start_date, '')=COALESCE(?, '')
                 AND canonical_status IN ('active', 'historical', 'resolved')
               ORDER BY id DESC""",
            (run_id, category, normalized_entity, valid_start_date),
        ).fetchall()
        return [self._row(row) for row in rows]

    def add_validation_item(
        self, patient_id: str, evidence_id: str, issue: str,
        severity: str, original_value: dict,
    ) -> None:
        self.db.execute(
            """INSERT INTO validation_queue
               (patient_id, item_type, item_id, issue, severity, status,
                original_value, created_at)
               VALUES (?, 'longitudinal_evidence', ?, ?, ?, 'pending', ?, ?)""",
            (
                patient_id, evidence_id, issue, severity,
                json.dumps(original_value, ensure_ascii=False),
                datetime.now().isoformat(),
            ),
        )
        self.db.commit()

    def complete_run(self, run_id: str) -> None:
        row = self.db.execute(
            "SELECT patient_id FROM clinical_state_runs WHERE run_id=?",
            (run_id,),
        ).fetchone()
        if not row:
            return
        patient_id = row["patient_id"]
        with self.db:
            self.db.execute(
                """UPDATE clinical_state_runs SET status='superseded'
                   WHERE patient_id=? AND status='completed' AND run_id<>?""",
                (patient_id, run_id),
            )
            self.db.execute(
                """UPDATE clinical_state_runs
                   SET status='completed', current_document_id=NULL,
                       completed_at=?, error_message=NULL
                   WHERE run_id=?""",
                (datetime.now().isoformat(), run_id),
            )

    def fail_run(self, run_id: str, error_message: str) -> None:
        self.db.execute(
            """UPDATE clinical_state_runs SET status='failed',
               error_message=? WHERE run_id=?""",
            (error_message, run_id),
        )
        self.db.commit()

    def get_current_run(self, patient_id: str) -> dict | None:
        row = self.db.execute(
            """SELECT * FROM clinical_state_runs
               WHERE patient_id=? AND status='completed'
               ORDER BY completed_at DESC LIMIT 1""",
            (patient_id,),
        ).fetchone()
        return dict(row) if row else None

    @staticmethod
    def _row(row) -> dict:
        value = dict(row)
        value["objective_basis"] = bool(value.get("objective_basis"))
        value["correction_explicit"] = bool(value.get("correction_explicit"))
        try:
            value["data"] = json.loads(value.pop("data_json") or "{}")
        except (TypeError, ValueError):
            value["data"] = {}
        return value

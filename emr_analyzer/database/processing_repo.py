"""Versioned processing runs and document-stage manifest."""

from __future__ import annotations

from datetime import datetime, timezone
import json

from .engine import DatabaseEngine
from ..models.clinical_registry import ProcessingManifestItem, ProcessingRun


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class ProcessingRepository:
    def __init__(self, db: DatabaseEngine):
        self.db = db

    def start_run(self, run: ProcessingRun) -> None:
        # A new run for the same patient/stage is also the recovery boundary
        # for an application that was closed while a worker was active.  The
        # desktop application does not support two concurrent writers for the
        # same patient: leaving old rows as ``running`` would make the UI and
        # audit trail report work that no longer exists.
        interrupted_at = _now()
        stale_rows = self.db.execute(
            """SELECT run_id FROM processing_runs
               WHERE patient_id=? AND stage=? AND status='running'""",
            (run.patient_id, run.stage),
        ).fetchall()
        stale_ids = [row["run_id"] for row in stale_rows]
        for stale_id in stale_ids:
            self.db.execute(
                """UPDATE processing_runs
                   SET status='interrupted', completed_at=?,
                       error_message=COALESCE(
                           error_message,
                           'Esecuzione interrotta prima del completamento'
                       )
                   WHERE run_id=?""",
                (interrupted_at, stale_id),
            )
            self.db.execute(
                """UPDATE processing_manifest
                   SET status='failed',
                       error_message=COALESCE(
                           error_message,
                           'Esecuzione interrotta: ripresa necessaria'
                       ),
                       processed_at=?, updated_at=?
                   WHERE run_id=? AND status='running'""",
                (interrupted_at, interrupted_at, stale_id),
            )
        self.db.execute(
            """INSERT INTO processing_runs
               (run_id, patient_id, stage, status, model_name, model_digest,
                prompt_version, schema_version, parameters_json, started_at,
                completed_at, error_message)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                run.run_id, run.patient_id, run.stage, run.status,
                run.model_name, run.model_digest, run.prompt_version,
                run.schema_version,
                json.dumps(run.parameters, ensure_ascii=False, sort_keys=True),
                run.started_at, run.completed_at, run.error_message,
            ),
        )
        self.db.commit()

    def finish_run(
        self, run_id: str, status: str, error_message: str | None = None
    ) -> None:
        self.db.execute(
            """UPDATE processing_runs
               SET status=?, completed_at=?, error_message=? WHERE run_id=?""",
            (status, _now(), error_message, run_id),
        )
        self.db.commit()

    def is_current(
        self,
        document_id: str,
        stage: str,
        input_hash: str,
        pipeline_version: str,
        prompt_version: str = "",
        model_digest: str = "",
        compatible_pipeline_versions: tuple[str, ...] = (),
    ) -> bool:
        versions = tuple(dict.fromkeys((
            pipeline_version, *compatible_pipeline_versions,
        )))
        placeholders = ", ".join("?" for _ in versions)
        row = self.db.execute(
            f"""SELECT status FROM processing_manifest
               WHERE document_id=? AND stage=? AND input_hash=?
                 AND pipeline_version IN ({placeholders}) AND prompt_version=?
                 AND model_digest=?
               ORDER BY updated_at DESC LIMIT 1""",
            (
                document_id, stage, input_hash, *versions,
                prompt_version or "", model_digest or "",
            ),
        ).fetchone()
        return bool(row and row["status"] == "completed")

    def upsert_manifest(self, item: ProcessingManifestItem) -> str:
        now = _now()
        existing = self.db.execute(
            """SELECT manifest_id, attempts FROM processing_manifest
               WHERE document_id=? AND stage=? AND input_hash=?
                 AND pipeline_version=? AND prompt_version=?
                 AND model_digest=?""",
            (
                item.document_id, item.stage, item.input_hash,
                item.pipeline_version, item.prompt_version or "",
                item.model_digest or "",
            ),
        ).fetchone()
        if existing:
            item.manifest_id = existing["manifest_id"]
            attempts = int(existing["attempts"] or 0) + 1
            self.db.execute(
                """UPDATE processing_manifest
                   SET run_id=?, patient_id=?, status=?, output_hash=?,
                       output_count=?, attempts=?, error_message=?,
                       processed_at=?, updated_at=? WHERE manifest_id=?""",
                (
                    item.run_id, item.patient_id, item.status,
                    item.output_hash, item.output_count, attempts,
                    item.error_message, item.processed_at, now,
                    item.manifest_id,
                ),
            )
        else:
            attempts = 1 if item.status in {"running", "failed"} else 0
            self.db.execute(
                """INSERT INTO processing_manifest
                   (manifest_id, run_id, patient_id, document_id, stage,
                    input_hash, pipeline_version, prompt_version,
                    model_digest, status, output_hash, output_count, attempts,
                    error_message, processed_at, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    item.manifest_id, item.run_id, item.patient_id,
                    item.document_id, item.stage, item.input_hash,
                    item.pipeline_version, item.prompt_version or "",
                    item.model_digest or "", item.status, item.output_hash,
                    item.output_count, attempts, item.error_message,
                    item.processed_at, item.created_at, now,
                ),
            )
        self.db.commit()
        return item.manifest_id

    def mark_result(
        self,
        manifest_id: str,
        *,
        status: str,
        output_hash: str | None = None,
        output_count: int = 0,
        error_message: str | None = None,
    ) -> None:
        self.db.execute(
            """UPDATE processing_manifest
               SET status=?, output_hash=?, output_count=?, error_message=?,
                   processed_at=?, updated_at=? WHERE manifest_id=?""",
            (
                status, output_hash, output_count, error_message,
                _now(), _now(), manifest_id,
            ),
        )
        self.db.commit()

    def processed_document_ids(
        self, patient_id: str, stage: str
    ) -> set[str]:
        rows = self.db.execute(
            """SELECT DISTINCT document_id FROM processing_manifest
               WHERE patient_id=? AND stage=? AND status='completed'""",
            (patient_id, stage),
        ).fetchall()
        return {row["document_id"] for row in rows}

"""Re-attribute a single document to another patient after human review.

Triggered from the Validazione tab on ``attribution`` queue items: after
the clinician confirms (or chooses) the correct patient, the document —
its record, its bound rows and its files on disk — moves to the target
workspace and its extraction is unlocked (``extraction_status='pending'``,
``error_message`` cleared) so the normal pipeline re-processes it there.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from ..config import active_workspace
from .workspace_merge import (
    WorkspaceMergeService,
    move_document_files,
)


@dataclass
class ReattributionResult:
    document_id: str
    source_patient_id: str = ""
    target_patient_id: str = ""
    moved_files: int = 0
    new_original_path: str | None = None
    queue_resolved: bool = False
    warnings: list[str] = field(default_factory=list)
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


class DocumentReattributionService:
    """Move ONE document between patient workspaces, atomically."""

    # Tables carrying both patient_id and document_id; a document move
    # repoints them by document id (the id itself never changes).
    _DOCUMENT_BOUND_TABLES = (
        "clinical_evidence",
        "lab_values",
        "document_identity_evidence",
    )

    def __init__(self, db, document_repo, patient_repo, audit_repo,
                 workspaces_dir: Path | None = None):
        self.db = db
        self.document_repo = document_repo
        self.patient_repo = patient_repo
        self.audit_repo = audit_repo
        self.workspaces_dir = Path(workspaces_dir or active_workspace.path)

    # ------------------------------------------------------------------

    def move_document(
        self,
        doc_id: str,
        target_patient_id: str,
        queue_item_id=None,
        resolution_status: str = "accepted",
    ) -> ReattributionResult:
        """Move *doc_id* to *target_patient_id* and unlock its extraction.

        ``queue_item_id`` (the validation_queue row being resolved) is
        resolved inside the same transaction as the move, so the queue can
        never disagree with the data.  Precondition failures return an
        error result; nothing is touched.
        """
        doc = self.db.execute(
            "SELECT * FROM documents WHERE id=?", (doc_id,)
        ).fetchone()
        if doc is None:
            return ReattributionResult(
                document_id=doc_id, error="Documento non trovato nel database"
            )
        source = doc["patient_id"]
        result = ReattributionResult(
            document_id=doc_id,
            source_patient_id=source,
            target_patient_id=target_patient_id,
        )

        target = self.patient_repo.get_by_id(target_patient_id)
        if target is None:
            result.error = (
                f"Paziente destinazione {target_patient_id} non trovato"
            )
            return result
        if target_patient_id == source:
            result.error = (
                f"Il documento appartiene già al paziente {source}"
            )
            return result

        # ---- Files first (reversible bookkeeping, no transaction) -----
        moved_paths: list[tuple[Path, Path]] = []
        new_original_paths: dict[str, str] = {}
        try:
            result.moved_files = move_document_files(
                self.workspaces_dir, source, target_patient_id, doc,
                moved_paths, new_original_paths,
            )
        except OSError as exc:
            WorkspaceMergeService._restore_files(moved_paths)
            result.error = f"Impossibile spostare i file: {exc}"
            return result
        result.new_original_path = new_original_paths.get(doc_id)

        # ---- Database, atomically -------------------------------------
        try:
            with self.db:
                new_path = result.new_original_path or doc["original_path"]
                self.db.execute(
                    """UPDATE documents
                       SET patient_id=?, original_path=?,
                           extraction_status='pending', error_message=NULL
                       WHERE id=?""",
                    (target_patient_id, new_path, doc_id),
                )
                for table in self._DOCUMENT_BOUND_TABLES:
                    self.db.execute(
                        f'UPDATE "{table}" SET patient_id=? '
                        f"WHERE document_id=?",
                        (target_patient_id, doc_id),
                    )
                # Pending attribution rows for this document follow it (in
                # case a reprocessing duplicated the queue entry).
                self.db.execute(
                    """UPDATE validation_queue SET patient_id=?
                       WHERE item_type='attribution' AND item_id=?
                         AND status='pending'""",
                    (target_patient_id, doc_id),
                )
                if queue_item_id is not None:
                    corrected = (
                        target_patient_id
                        if resolution_status == "corrected" else None
                    )
                    cursor = self.db.execute(
                        """UPDATE validation_queue
                           SET status=?, corrected_value=?, resolved_at=?
                           WHERE id=?""",
                        (
                            resolution_status, corrected,
                            datetime.now().isoformat(), queue_item_id,
                        ),
                    )
                    if cursor.rowcount == 0:
                        result.warnings.append(
                            "Elemento della coda non trovato"
                        )
                    else:
                        result.queue_resolved = True
                # The audit trail follows the document.
                self.db.execute(
                    """UPDATE audit_log SET patient_id=?
                       WHERE target_type='document' AND target_id=?""",
                    (target_patient_id, doc_id),
                )
                self._repoint_timeline(
                    source, target_patient_id, doc_id, result
                )
                # Both patients' narratives are stale: force a rebuild.
                self.db.execute(
                    "DELETE FROM clinical_state WHERE patient_id IN (?, ?)",
                    (source, target_patient_id),
                )
        except Exception as exc:
            WorkspaceMergeService._restore_files(moved_paths)
            result.error = (
                f"Errore durante lo spostamento nel database: {exc}"
            )
            return result

        # ---- Post-commit cleanup + audit (best-effort) -----------------
        # When the original was a byte-identical duplicate already present
        # in the target, the file was not moved: the source copy is now
        # orphaned and can be removed (only after the commit succeeded).
        source_original = Path(doc["original_path"])
        new_path = Path(result.new_original_path or "")
        if (
            source_original.is_file()
            and new_path != source_original
            and new_path.is_file()
        ):
            try:
                source_original.unlink()
            except OSError:
                result.warnings.append(
                    "File originale duplicato non rimosso dalla sorgente"
                )
        try:
            self.audit_repo.log(
                target_patient_id, "document_reattributed", "document",
                doc_id,
                {
                    "source_patient_id": source,
                    "target_patient_id": target_patient_id,
                    "resolution": resolution_status,
                    "moved_files": result.moved_files,
                },
            )
        except Exception:
            pass
        return result

    # ------------------------------------------------------------------

    def _repoint_timeline(self, source: str, target: str, doc_id: str,
                          result: ReattributionResult) -> None:
        """Repoint timeline rows owned exclusively by the moved document.

        Live rows carry a single-element ``source_document_ids``; legacy
        rows referencing multiple documents stay with the source patient
        and produce a warning (a Clinical History rebuild regenerates a
        consistent registry anyway).
        """
        rows = self.db.execute(
            "SELECT entry_id, source_document_ids FROM clinical_timeline "
            "WHERE patient_id=?",
            (source,),
        ).fetchall()
        for row in rows:
            try:
                ids = json.loads(row["source_document_ids"] or "[]")
            except (TypeError, ValueError):
                continue
            if ids == [doc_id]:
                self.db.execute(
                    "UPDATE clinical_timeline SET patient_id=? "
                    "WHERE entry_id=?",
                    (target, row["entry_id"]),
                )
            elif doc_id in ids:
                result.warnings.append(
                    "Voce della timeline con più documenti lasciata al "
                    f"paziente sorgente (entry {row['entry_id']})"
                )

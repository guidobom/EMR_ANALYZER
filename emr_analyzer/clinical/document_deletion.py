"""Safe deletion of a report and all data derived from it."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import shutil
import uuid

from ..config import active_workspace


@dataclass
class DocumentDeletionResult:
    document_id: str
    deleted: bool = False
    warnings: list[str] = field(default_factory=list)
    error: str | None = None


class DocumentDeletionService:
    """Delete DB rows atomically and move files through a reversible trash."""

    def __init__(self, db, document_repo, audit_repo=None,
                 workspaces_dir=None):
        self.db = db
        self.document_repo = document_repo
        self.audit_repo = audit_repo
        self.workspaces_dir = (
            Path(workspaces_dir) if workspaces_dir
            else active_workspace.path
        )

    def delete(self, document_id: str) -> DocumentDeletionResult:
        result = DocumentDeletionResult(document_id=document_id)
        document = self.document_repo.get_by_id(document_id)
        if document is None:
            result.error = "Documento non trovato nel database"
            return result
        gold_reference = self.db.execute(
            """SELECT annotation_id FROM gold_annotation_sources
               WHERE document_id=? LIMIT 1""",
            (document_id,),
        ).fetchone()
        if gold_reference:
            result.error = (
                "Documento citato dal gold set clinico "
                f"({gold_reference['annotation_id']}): eliminazione bloccata"
            )
            return result

        trash_dir = (
            self.workspaces_dir / "_trash" /
            f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{document_id}_{uuid.uuid4().hex[:6]}"
        )
        moved: list[tuple[Path, Path]] = []
        try:
            artifacts = self._document_artifacts(document)
            if artifacts:
                trash_dir.mkdir(parents=True, exist_ok=False)
            for index, source in enumerate(artifacts):
                destination = trash_dir / f"{index:03d}_{source.name}"
                shutil.move(str(source), str(destination))
                moved.append((source, destination))

            # Fetch derived item IDs before the document cascade removes them.
            lab_ids = [
                str(row["id"]) for row in self.db.execute(
                    "SELECT id FROM lab_values WHERE document_id=?", (document_id,)
                ).fetchall()
            ]
            evidence_ids = [
                row["evidence_id"] for row in self.db.execute(
                    "SELECT evidence_id FROM clinical_evidence WHERE document_id=?",
                    (document_id,),
                ).fetchall()
            ]

            with self.db:
                # The queue has no foreign key to heterogeneous item IDs.
                self.db.execute(
                    """DELETE FROM validation_queue
                       WHERE (item_type='document' AND item_id=?)
                          OR item_id=? OR item_id LIKE ?""",
                    (document_id, document_id, f"{document_id}_%"),
                )
                if lab_ids:
                    placeholders = ",".join("?" for _ in lab_ids)
                    self.db.execute(
                        f"""DELETE FROM validation_queue
                            WHERE item_type IN ('lab', 'lab_value')
                              AND item_id IN ({placeholders})""",
                        tuple(lab_ids),
                    )
                if evidence_ids:
                    placeholders = ",".join("?" for _ in evidence_ids)
                    self.db.execute(
                        f"""DELETE FROM validation_queue
                            WHERE item_type='evidence'
                              AND item_id IN ({placeholders})""",
                        tuple(evidence_ids),
                    )
                cursor = self.db.execute(
                    "DELETE FROM documents WHERE id=?", (document_id,)
                )
                if cursor.rowcount != 1:
                    raise RuntimeError("Il record del documento non è stato eliminato")

            result.deleted = True
        except Exception as exc:
            result.error = str(exc)
            self._restore_files(moved, result)
            return result

        if self.audit_repo:
            try:
                self.audit_repo.log(
                    document.patient_id,
                    "delete",
                    "document",
                    document_id,
                    {
                        "filename": document.filename,
                        "file_hash": document.file_hash,
                        "removed_lab_count": len(lab_ids),
                        "removed_evidence_count": len(evidence_ids),
                    },
                )
            except Exception as exc:
                result.warnings.append(f"Audit log non aggiornato: {exc}")

        if trash_dir.exists():
            try:
                shutil.rmtree(trash_dir)
            except OSError as exc:
                result.warnings.append(
                    f"Dati eliminati; residuo temporaneo in {trash_dir}: {exc}"
                )
        return result

    def _document_artifacts(self, document) -> list[Path]:
        artifacts = []
        original = Path(document.original_path)
        seen = set()
        if original.exists():
            artifacts.append(original)
            seen.add(str(original))
        for output_dir_name in ("extraction", "docling"):
            output_dir = self.workspaces_dir / document.patient_id / output_dir_name
            if not output_dir.exists():
                continue
            candidates = [
                output_dir / f"{document.id}.md",
                output_dir / f"{document.id}.json",
                output_dir / f"{document.id}_raw.md",
            ]
            candidates.extend(output_dir.glob(f"{document.id}_*"))
            for candidate in candidates:
                if candidate.exists() and str(candidate) not in seen:
                    artifacts.append(candidate)
                    seen.add(str(candidate))
        return artifacts

    @staticmethod
    def _restore_files(moved: list[tuple[Path, Path]], result: DocumentDeletionResult) -> None:
        for original, temporary in reversed(moved):
            try:
                original.parent.mkdir(parents=True, exist_ok=True)
                if temporary.exists():
                    shutil.move(str(temporary), str(original))
            except OSError as exc:
                result.warnings.append(
                    f"Impossibile ripristinare {original.name}: {exc}"
                )

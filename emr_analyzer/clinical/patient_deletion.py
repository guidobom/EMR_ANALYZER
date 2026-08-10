"""Complete, reversible deletion of one patient workspace."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import re
import shutil
import uuid

from ..config import CACHE_DIR, SUPPORTED_EXTENSIONS, active_workspace
from ..utils.file_utils import compute_file_hash


@dataclass
class PatientWorkspaceDeletionResult:
    patient_id: str
    deleted: bool = False
    removed_database_rows: dict[str, int] = field(default_factory=dict)
    removed_path_count: int = 0
    warnings: list[str] = field(default_factory=list)
    error: str | None = None


class PatientWorkspaceDeletionService:
    """Remove every managed file and database record for one patient.

    Files are first moved to a private staging directory. Database deletion is
    then performed in one transaction. If it fails, every moved path is put
    back; after a successful commit, the staged files are permanently purged.
    """

    def __init__(
        self,
        db,
        patient_repo,
        workspaces_dir=None,
        cache_dir: str | Path = CACHE_DIR,
    ):
        self.db = db
        self.patient_repo = patient_repo
        self.workspaces_dir = (
            Path(workspaces_dir) if workspaces_dir
            else active_workspace.path
        )
        self.cache_dir = Path(cache_dir)

    def delete(self, patient_id: str) -> PatientWorkspaceDeletionResult:
        result = PatientWorkspaceDeletionResult(patient_id=patient_id)
        if not patient_id or self.patient_repo.get_by_id(patient_id) is None:
            result.error = "Paziente non trovato nel registro"
            return result

        documents = self.db.execute(
            "SELECT id, file_hash FROM documents WHERE patient_id=?",
            (patient_id,),
        ).fetchall()
        document_ids = {row["id"] for row in documents}
        document_hashes = {
            row["file_hash"] for row in documents if row["file_hash"]
        }
        workspace = self.workspaces_dir / patient_id
        document_hashes.update(self._workspace_document_hashes(workspace))

        staging_dir = (
            self.workspaces_dir / "_deletion_staging" /
            f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_"
            f"{patient_id}_{uuid.uuid4().hex[:8]}"
        )
        moved: list[tuple[Path, Path]] = []
        emptied_parents: set[Path] = set()
        try:
            managed_paths = self._managed_paths(
                patient_id, document_ids, document_hashes, workspace
            )
            if managed_paths:
                staging_dir.mkdir(parents=True, exist_ok=False)
            for index, source in enumerate(managed_paths):
                destination = staging_dir / f"{index:05d}_{source.name}"
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(source), str(destination))
                moved.append((source, destination))
                emptied_parents.add(source.parent)

            result.removed_database_rows = self._delete_database_records(
                patient_id
            )
        except Exception as exc:
            result.error = str(exc)
            self._restore_paths(moved, result)
            self._remove_empty_staging(staging_dir)
            return result

        # At this point the database transaction is committed. The staging
        # directory contains only data belonging to the deleted patient.
        try:
            if staging_dir.exists():
                shutil.rmtree(staging_dir)
            result.removed_path_count = len(moved)
            self._remove_empty_parents(emptied_parents)
            self._remove_empty_staging(staging_dir)
        except OSError as exc:
            result.error = (
                "Record eliminati, ma non è stato possibile distruggere tutti "
                f"i file temporanei in {staging_dir}: {exc}"
            )
            return result

        result.deleted = True
        return result

    def _delete_database_records(self, patient_id: str) -> dict[str, int]:
        tables = self._tables_with_patient_id()
        counts = {
            table: self.db.execute(
                f'SELECT COUNT(*) FROM "{table}" WHERE patient_id=?',
                (patient_id,),
            ).fetchone()[0]
            for table in tables
        }

        with self.db:
            # The patient delete triggers every declared cascade. The sweep
            # afterwards also covers current or future patient-owned tables
            # that intentionally do not declare a foreign key (e.g. audit_log).
            cursor = self.db.execute(
                "DELETE FROM patients WHERE id=?", (patient_id,)
            )
            if cursor.rowcount != 1:
                raise RuntimeError("Il record del paziente non è stato eliminato")
            for table in tables:
                if table == "patients":
                    continue
                self.db.execute(
                    f'DELETE FROM "{table}" WHERE patient_id=?', (patient_id,)
                )
            residues = {
                table: self.db.execute(
                    f'SELECT COUNT(*) FROM "{table}" WHERE patient_id=?',
                    (patient_id,),
                ).fetchone()[0]
                for table in tables
            }
            residues = {table: count for table, count in residues.items() if count}
            if residues:
                raise RuntimeError(
                    "Residui nel database dopo la cancellazione: "
                    + ", ".join(f"{table}={count}" for table, count in residues.items())
                )
        return {table: count for table, count in counts.items() if count}

    def _tables_with_patient_id(self) -> list[str]:
        tables = []
        rows = self.db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name NOT LIKE 'sqlite_%'"
        ).fetchall()
        for row in rows:
            table = row["name"]
            if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", table):
                continue
            columns = self.db.execute(f'PRAGMA table_info("{table}")').fetchall()
            if any(column["name"] == "patient_id" for column in columns):
                tables.append(table)
        return sorted(tables)

    def _managed_paths(
        self,
        patient_id: str,
        document_ids: set[str],
        document_hashes: set[str],
        workspace: Path,
    ) -> list[Path]:
        paths = []
        if workspace.exists():
            paths.append(workspace)

        cache_workspace = self.cache_dir / patient_id
        if cache_workspace.exists():
            paths.append(cache_workspace)

        trash = self.workspaces_dir / "_trash"
        if trash.exists():
            tokens = {patient_id, *document_ids}
            for child in trash.iterdir():
                if any(token and token in child.name for token in tokens):
                    paths.append(child)

        # Interrupted import batches are not assigned in the database. Delete
        # only staged copies whose content hash matches a document of this
        # workspace; unrelated inbox material is preserved.
        inbox = self.workspaces_dir / "_inbox"
        if inbox.exists() and document_hashes:
            for candidate in inbox.rglob("*"):
                if not candidate.is_file():
                    continue
                try:
                    if compute_file_hash(candidate) in document_hashes:
                        paths.append(candidate)
                except OSError:
                    continue

        # Parents and children must never both be moved.
        unique = []
        for path in sorted(set(paths), key=lambda item: len(item.parts)):
            if any(parent == path or parent in path.parents for parent in unique):
                continue
            unique.append(path)
        return unique

    @staticmethod
    def _workspace_document_hashes(workspace: Path) -> set[str]:
        hashes = set()
        if not workspace.exists():
            return hashes
        for candidate in workspace.rglob("*"):
            if not candidate.is_file() or candidate.suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue
            try:
                hashes.add(compute_file_hash(candidate))
            except OSError:
                continue
        return hashes

    @staticmethod
    def _restore_paths(
        moved: list[tuple[Path, Path]],
        result: PatientWorkspaceDeletionResult,
    ) -> None:
        for original, temporary in reversed(moved):
            try:
                original.parent.mkdir(parents=True, exist_ok=True)
                if temporary.exists():
                    shutil.move(str(temporary), str(original))
            except OSError as exc:
                result.warnings.append(
                    f"Impossibile ripristinare {original}: {exc}"
                )

    def _remove_empty_parents(self, parents: set[Path]) -> None:
        protected = {
            self.workspaces_dir,
            self.workspaces_dir / "_inbox",
            self.workspaces_dir / "_trash",
            self.cache_dir,
        }
        for parent in sorted(parents, key=lambda item: len(item.parts), reverse=True):
            current = parent
            while current not in protected and current.exists():
                try:
                    current.rmdir()
                except OSError:
                    break
                current = current.parent

    @staticmethod
    def _remove_empty_staging(staging_dir: Path) -> None:
        root = staging_dir.parent
        if staging_dir.exists():
            try:
                staging_dir.rmdir()
            except OSError:
                pass
        if root.exists():
            try:
                root.rmdir()
            except OSError:
                pass

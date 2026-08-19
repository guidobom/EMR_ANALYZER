"""Merge one patient workspace into another within the same project.

A "workspace" is a patient: merge moves all documents and every piece of
data derived from them (events, labs, evidence, timeline, ...) from the
SOURCE patient into the TARGET patient, then removes the source workspace
(SPOSTA semantics).  Because the merge happens inside a single project
database, document/event/evidence ids are already globally unique: only
``patient_id`` and ``original_path`` need to change, no id regeneration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import json
import shutil

from ..config import active_workspace
from ..utils.file_utils import compute_file_hash


def move_document_files(workspaces_dir: Path, source_pid: str,
                        target_pid: str, doc, moved: list,
                        new_original_paths: dict) -> int:
    """Move a document's original file and its extraction/docling
    artifacts.  Returns the number of files moved.

    Module-level so it can be reused by single-document operations
    (document re-attribution); *doc* is a dict-like row with
    ``original_path``, ``file_hash`` and ``id`` keys.  *moved* collects
    ``(src, dst)`` pairs for rollback via
    :meth:`WorkspaceMergeService._restore_files`.
    """
    source_root = workspaces_dir / source_pid
    target_root = workspaces_dir / target_pid
    files_moved = 0
    new_original = None

    original = Path(doc["original_path"])
    if original.is_file():
        if WorkspaceMergeService._is_within(original, source_root):
            destination = target_root / original.relative_to(source_root)
        else:
            destination = (
                target_root / "documents" / "original" / original.name
            )
        destination, skipped = WorkspaceMergeService._unique_target_path(
            destination, doc["file_hash"]
        )
        if not skipped:
            WorkspaceMergeService._move(original, destination, moved)
            files_moved += 1
        new_original = str(destination)

    # Extraction / docling artifacts share the (globally unique) doc id,
    # so they can never collide in the target.
    doc_id = doc["id"]
    for subdir in ("extraction", "docling"):
        source_dir = source_root / subdir
        if not source_dir.is_dir():
            continue
        for candidate in sorted(source_dir.glob(f"{doc_id}*")):
            if not candidate.is_file():
                continue
            WorkspaceMergeService._move(
                candidate, target_root / subdir / candidate.name, moved
            )
            files_moved += 1

    if new_original is not None:
        new_original_paths[doc_id] = new_original
    return files_moved


@dataclass
class MergeResult:
    source_patient_id: str
    target_patient_id: str
    moved_documents: int = 0      # documents actually moved into the target
    already_present: int = 0      # byte-identical duplicates skipped
    moved_files: int = 0          # physical files moved (originals + artifacts)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    source_removed: bool = False

    @property
    def ok(self) -> bool:
        return not self.errors and self.source_removed


# Tables whose rows carry a NOT NULL document FK.  The value maps each table
# to the name of its document column.  Only the rows of the documents that
# were actually moved follow the target; the rows of skipped duplicates stay
# with the source and are removed by the final source workspace deletion
# (cascade).
_DOCUMENT_BOUND_TABLES = {
    "clinical_evidence": "document_id",
    "lab_values": "document_id",
    "document_identity_evidence": "document_id",
}

# Tables that can be repointed wholesale; document references are nullable or
# embedded in JSON and either follow the (unchanged) doc ids or are reset to
# NULL via ON DELETE SET NULL when a skipped duplicate is deleted.
_WHOLESALE_TABLES = (
    "validation_queue",
    "audit_log",
)


class WorkspaceMergeService:
    """Move one patient workspace into another and remove the source."""

    def __init__(self, db, patient_repo, document_repo, identity_repo,
                 audit_repo, deletion_service, workspaces_dir=None):
        self.db = db
        self.patient_repo = patient_repo
        self.document_repo = document_repo
        self.identity_repo = identity_repo
        self.audit_repo = audit_repo
        self.deletion = deletion_service
        self.workspaces_dir = (
            Path(workspaces_dir) if workspaces_dir
            else active_workspace.path
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def list_patients(self) -> list[dict]:
        """Return a summary of every patient workspace in the project."""
        rows = self.db.execute(
            "SELECT * FROM patients ORDER BY created_at, id"
        ).fetchall()
        patients = []
        for row in rows:
            pid = row["id"]
            doc_count = self.document_repo.get_count_by_patient(pid)
            has_identity = bool(
                self.identity_repo and self.identity_repo.has_identity(pid)
            )
            patients.append({
                "id": pid,
                "pseudonym": row["pseudonym"],
                "initials": row["initials"],
                "sex": row["sex"],
                "birth_year": row["birth_year"],
                "document_count": doc_count,
                "identity_exists": has_identity,
            })
        return patients

    def find_identity_matches(self, patient_id: str) -> list[dict]:
        """Find patients that look like the same person by stored HMAC keys.

        Compares the already-persisted HMAC fingerprints (same key space
        within a project), mirroring PatientIdentityRepository.find_match:
        strong match on fiscal code, otherwise name + birth date.
        """
        me = self.db.execute(
            "SELECT * FROM patient_identities WHERE patient_id=?",
            (patient_id,),
        ).fetchone()
        if not me:
            return []
        others = self.db.execute(
            "SELECT * FROM patient_identities WHERE patient_id != ?",
            (patient_id,),
        ).fetchall()

        def _cf_match(row) -> bool:
            return bool(
                me["fiscal_code_key"]
                and me["fiscal_code_key"] == row["fiscal_code_key"]
            )

        def _nb_match(row) -> bool:
            return bool(
                me["normalized_name_key"] and me["birth_date_key"]
                and me["normalized_name_key"] == row["normalized_name_key"]
                and me["birth_date_key"] == row["birth_date_key"]
            )

        matches = []
        for other in others:
            cf = _cf_match(other)
            nb = _nb_match(other)
            if not (cf or nb):
                continue
            # conflict: fiscal code and name+birth point to different patients
            conflict = False
            for other2 in others:
                if other2["patient_id"] == other["patient_id"]:
                    continue
                if cf and _nb_match(other2):
                    conflict = True
                if nb and _cf_match(other2):
                    conflict = True
            matches.append({
                "patient_id": other["patient_id"],
                "reason": "codice fiscale" if cf else "nome e data di nascita",
                "confidence": 0.99 if cf else 0.95,
                "conflict": conflict,
            })
        return matches

    def merge(self, source_pid: str, target_pid: str,
              progress_callback=None) -> MergeResult:
        """Move the source workspace into the target and remove the source."""
        result = MergeResult(
            source_patient_id=source_pid,
            target_patient_id=target_pid,
        )
        if source_pid == target_pid:
            raise ValueError(
                "Il workspace sorgente e la destinazione coincidono"
            )
        if self.patient_repo.get_by_id(source_pid) is None:
            raise ValueError(f"Paziente sorgente {source_pid} non trovato")
        if self.patient_repo.get_by_id(target_pid) is None:
            raise ValueError(
                f"Paziente destinazione {target_pid} non trovato"
            )
        self._emit(progress_callback, 5, "Verifica workspace...")

        # ---- 1. Classify documents --------------------------------------
        source_docs = self.db.execute(
            "SELECT * FROM documents WHERE patient_id=? ORDER BY id",
            (source_pid,),
        ).fetchall()
        target_hashes = {
            row["file_hash"]
            for row in self.db.execute(
                "SELECT file_hash FROM documents "
                "WHERE patient_id=? AND file_hash IS NOT NULL",
                (target_pid,),
            ).fetchall()
        }
        doc_map: dict[str, str] = {}
        moved_ids: list[str] = []
        for doc in source_docs:
            if doc["file_hash"] and doc["file_hash"] in target_hashes:
                dup_id = self.db.execute(
                    "SELECT id FROM documents "
                    "WHERE patient_id=? AND file_hash=?",
                    (target_pid, doc["file_hash"]),
                ).fetchone()["id"]
                doc_map[doc["id"]] = dup_id
                result.already_present += 1
            else:
                doc_map[doc["id"]] = doc["id"]
                moved_ids.append(doc["id"])
        self._emit(progress_callback, 15, "Classificazione documenti...")

        # ---- 2. Move files (reversible, non-transactional) ---------------
        moved_paths: list[tuple[Path, Path]] = []
        new_original_paths: dict[str, str] = {}
        try:
            for index, doc_id in enumerate(moved_ids):
                self._emit(
                    progress_callback,
                    15 + int(45 * index / max(len(moved_ids), 1)),
                    f"Spostamento file {index + 1}/{len(moved_ids)}...",
                )
                doc = self.db.execute(
                    "SELECT * FROM documents WHERE id=?", (doc_id,)
                ).fetchone()
                moved_files = self._move_document_files(
                    source_pid, target_pid, doc,
                    moved_paths, new_original_paths,
                )
                result.moved_files += moved_files
            result.moved_documents = len(moved_ids)
        except OSError as exc:
            result.errors.append(f"Impossibile spostare i file: {exc}")
            self._restore_files(moved_paths)
            return result

        # ---- 3. Database transaction -------------------------------------
        try:
            with self.db:
                for doc_id in moved_ids:
                    new_path = new_original_paths.get(doc_id)
                    if new_path:
                        self.db.execute(
                            "UPDATE documents SET patient_id=?, original_path=? "
                            "WHERE id=?",
                            (target_pid, new_path, doc_id),
                        )
                    else:
                        self.db.execute(
                            "UPDATE documents SET patient_id=? WHERE id=?",
                            (target_pid, doc_id),
                        )
                self._merge_identity_rows(source_pid, target_pid)
                self._merge_clinical_state(source_pid, target_pid, result)

                if moved_ids:
                    placeholders = ",".join("?" for _ in moved_ids)
                    for table, doc_col in _DOCUMENT_BOUND_TABLES.items():
                        self.db.execute(
                            f'UPDATE "{table}" SET patient_id=? '
                            f"WHERE patient_id=? AND {doc_col} IN ({placeholders})",
                            (target_pid, source_pid, *moved_ids),
                        )
                for table in _WHOLESALE_TABLES:
                    self.db.execute(
                        f'UPDATE "{table}" SET patient_id=? WHERE patient_id=?',
                        (target_pid, source_pid),
                    )
                self._merge_timeline(source_pid, target_pid, doc_map)
                self._rewrite_validation_queue(source_pid, target_pid, doc_map)
        except Exception as exc:
            result.errors.append(
                f"Errore durante il merge nel database: {exc}"
            )
            self._restore_files(moved_paths)
            return result

        # ---- 4. Audit (after commit) --------------------------------------
        try:
            if self.audit_repo:
                self.audit_repo.log(
                    target_pid, "merge", "patient", source_pid,
                    {
                        "moved_documents": result.moved_documents,
                        "already_present": result.already_present,
                    },
                )
        except Exception as exc:
            result.warnings.append(f"Audit log non aggiornato: {exc}")

        # ---- 5. Remove the source workspace (last) -----------------------
        self._emit(progress_callback, 90, "Eliminazione workspace sorgente...")
        try:
            deletion = self.deletion.delete(source_pid)
            result.source_removed = deletion.deleted
            if not deletion.deleted:
                result.errors.append(
                    deletion.error or "Eliminazione workspace sorgente non riuscita"
                )
                result.warnings.extend(deletion.warnings)
        except Exception as exc:
            result.errors.append(
                f"Eliminazione workspace sorgente fallita: {exc}"
            )
            result.source_removed = False

        self._emit(progress_callback, 100, "Completato")
        return result

    def merge_many(self, pairs: list[tuple[str, str]],
                   progress_callback=None) -> list[MergeResult]:
        """Merge several (source, target) pairs, scaling global progress."""
        results = []
        total = len(pairs)
        for index, (source_pid, target_pid) in enumerate(pairs):
            base = int(100 * index / max(total, 1))
            span = 100 // max(total, 1) if total else 100

            def cb(pct: int, msg: str, _base=base, _span=span):
                if progress_callback:
                    progress_callback(
                        _base + int(pct * _span / 100), msg
                    )

            results.append(
                self.merge(source_pid, target_pid, progress_callback=cb)
            )
        return results

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _move_document_files(self, source_pid: str, target_pid: str,
                             doc, moved, new_original_paths) -> int:
        """Move a document's original file and its extraction/docling
        artifacts.  Returns the number of files moved."""
        return move_document_files(
            self.workspaces_dir, source_pid, target_pid, doc,
            moved, new_original_paths,
        )

    def _merge_identity_rows(self, source_pid: str, target_pid: str) -> None:
        """Fold the source identity into the target's (patient_id is UNIQUE)."""
        source = self.db.execute(
            "SELECT * FROM patient_identities WHERE patient_id=?",
            (source_pid,),
        ).fetchone()
        if not source:
            return
        target = self.db.execute(
            "SELECT * FROM patient_identities WHERE patient_id=?",
            (target_pid,),
        ).fetchone()
        if not target:
            self.db.execute(
                "UPDATE patient_identities SET patient_id=? WHERE patient_id=?",
                (target_pid, source_pid),
            )
            self.db.execute(
                "UPDATE patient_hospital_ids SET patient_id=? WHERE patient_id=?",
                (target_pid, source_pid),
            )
            return
        now = datetime.now().isoformat()
        self.db.execute(
            """UPDATE patient_identities SET
               fiscal_code_key=COALESCE(fiscal_code_key, ?),
               normalized_name_key=COALESCE(normalized_name_key, ?),
               birth_date_key=COALESCE(birth_date_key, ?),
               sex=COALESCE(sex, ?),
               hospital_patient_id_key=COALESCE(hospital_patient_id_key, ?),
               confidence=MAX(confidence, ?),
               updated_at=?
               WHERE patient_id=?""",
            (source["fiscal_code_key"], source["normalized_name_key"],
             source["birth_date_key"], source["sex"],
             source["hospital_patient_id_key"], source["confidence"],
             now, target_pid),
        )
        # Hospital patient IDs are multi-valued per patient: fold the source's
        # set into the target's before removing the source workspace.
        self.db.execute(
            """INSERT OR IGNORE INTO patient_hospital_ids
               (patient_id, hospital_patient_id_key, created_at, updated_at)
               SELECT ?, hospital_patient_id_key, created_at, updated_at
               FROM patient_hospital_ids WHERE patient_id=?""",
            (target_pid, source_pid),
        )
        self.db.execute(
            "DELETE FROM patient_hospital_ids WHERE patient_id=?",
            (source_pid,),
        )
        self.db.execute(
            "DELETE FROM patient_identities WHERE patient_id=?",
            (source_pid,),
        )

    def _merge_clinical_state(self, source_pid: str, target_pid: str,
                              result: MergeResult) -> None:
        """Repoint the source Clinical State or keep the target's (PK
        conflict).  The moved events/evidence are copied, so a rebuild from
        the Clinical History tab can recompute the state afterwards."""
        source = self.db.execute(
            "SELECT 1 FROM clinical_state WHERE patient_id=?", (source_pid,)
        ).fetchone()
        if not source:
            return
        target = self.db.execute(
            "SELECT 1 FROM clinical_state WHERE patient_id=?", (target_pid,)
        ).fetchone()
        if target:
            self.db.execute(
                "DELETE FROM clinical_state WHERE patient_id=?", (source_pid,)
            )
            result.warnings.append(
                "Il Clinical State del paziente sorgente non è stato unito: "
                "ricostruirlo dalla scheda Storia Clinica (i dati spostati "
                "sono disponibili in eventi ed evidenze)."
            )
        else:
            self.db.execute(
                "UPDATE clinical_state SET patient_id=? WHERE patient_id=?",
                (target_pid, source_pid),
            )

    def _merge_timeline(self, source_pid: str, target_pid: str,
                        doc_map: dict[str, str]) -> None:
        """Repoint timeline entries and rewrite the JSON source_document_ids."""
        self.db.execute(
            "UPDATE clinical_timeline SET patient_id=? WHERE patient_id=?",
            (target_pid, source_pid),
        )
        rows = self.db.execute(
            "SELECT entry_id, source_document_ids FROM clinical_timeline "
            "WHERE patient_id=?",
            (target_pid,),
        ).fetchall()
        for row in rows:
            try:
                ids = json.loads(row["source_document_ids"] or "[]")
            except (ValueError, TypeError):
                continue
            new_ids = [doc_map.get(doc_id, doc_id) for doc_id in ids]
            if new_ids != ids:
                self.db.execute(
                    "UPDATE clinical_timeline SET source_document_ids=? "
                    "WHERE entry_id=?",
                    (json.dumps(new_ids, ensure_ascii=False), row["entry_id"]),
                )

    def _rewrite_validation_queue(self, source_pid: str, target_pid: str,
                                  doc_map: dict[str, str]) -> None:
        """Repoint validation rows and rewrite document item_ids."""
        rows = self.db.execute(
            "SELECT id, item_type, item_id FROM validation_queue "
            "WHERE patient_id=?",
            (target_pid,),
        ).fetchall()
        for row in rows:
            if row["item_type"] != "document":
                continue
            new_id = doc_map.get(row["item_id"])
            if new_id and new_id != row["item_id"]:
                self.db.execute(
                    "UPDATE validation_queue SET item_id=? WHERE id=?",
                    (new_id, row["id"]),
                )

    # ------------------------------------------------------------------
    # File helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _move(src: Path, dst: Path, moved: list[tuple[Path, Path]]) -> None:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))
        moved.append((src, dst))

    @staticmethod
    def _unique_target_path(dst: Path, file_hash: str) -> tuple[Path, bool]:
        """Resolve a filename collision.  Returns (path, skipped) where
        ``skipped`` is True when an identical file already exists."""
        if not dst.exists():
            return dst, False
        try:
            if file_hash and compute_file_hash(dst) == file_hash:
                return dst, True
        except OSError:
            pass
        stem, suffix = dst.stem, dst.suffix
        tag = (file_hash[:8] if file_hash else "copia") or "copia"
        new_name = f"{stem}__{tag}{suffix}"
        candidate = dst.with_name(new_name)
        counter = 2
        while candidate.exists():
            candidate = dst.with_name(f"{new_name}__{counter}")
            counter += 1
        return candidate, False

    @staticmethod
    def _restore_files(moved: list[tuple[Path, Path]]) -> None:
        for src, dst in reversed(moved):
            try:
                src.parent.mkdir(parents=True, exist_ok=True)
                if dst.exists():
                    shutil.move(str(dst), str(src))
            except OSError:
                pass

    @staticmethod
    def _is_within(path: Path, root: Path) -> bool:
        try:
            path.relative_to(root)
            return True
        except ValueError:
            return False

    @staticmethod
    def _emit(progress_callback, pct: int, msg: str) -> None:
        if progress_callback:
            progress_callback(pct, msg)

"""Temporary inbox used before a document is assigned to a workspace."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import shutil
import uuid

from ..config import WORKSPACES_DIR
from ..models.patient_identity import PatientIdentityEvidence
from ..utils.file_utils import (
    compute_file_hash,
    get_file_info,
    is_supported_file,
    verify_pdf,
)


@dataclass
class StagedDocument:
    original_path: str
    staged_path: str
    original_name: str
    file_hash: str
    check: dict
    evidence: PatientIdentityEvidence
    duplicate_document_id: str | None = None
    duplicate_patient_id: str | None = None
    error: str | None = None

    @property
    def is_duplicate(self) -> bool:
        return self.duplicate_document_id is not None


@dataclass
class ImportBatch:
    batch_id: str
    directory: Path
    documents: list[StagedDocument] = field(default_factory=list)

    def cleanup(self) -> None:
        if self.directory.exists():
            shutil.rmtree(self.directory, ignore_errors=True)


class ImportStagingService:
    """Copy incoming files into an isolated inbox and inspect their identity."""

    def __init__(self, identity_extractor, document_repo,
                 workspaces_dir: str | Path = WORKSPACES_DIR):
        self._identity_extractor = identity_extractor
        self._document_repo = document_repo
        self._workspaces_dir = Path(workspaces_dir)

    def stage(self, file_paths: list[str], progress_callback=None,
              cancel_check=None) -> ImportBatch:
        batch_id = (
            datetime.now().strftime("BATCH_%Y%m%d_%H%M%S_")
            + uuid.uuid4().hex[:8]
        )
        batch_dir = self._workspaces_dir / "_inbox" / batch_id
        batch_dir.mkdir(parents=True, exist_ok=False)
        batch = ImportBatch(batch_id=batch_id, directory=batch_dir)

        for index, source in enumerate(file_paths, start=1):
            if cancel_check and cancel_check():
                break
            source_path = Path(source)
            if progress_callback:
                progress_callback(index - 1, len(file_paths), source_path.name)
            if not source_path.is_file() or not is_supported_file(source_path):
                continue
            item_dir = batch_dir / f"{index:05d}"
            item_dir.mkdir(parents=True, exist_ok=True)
            staged_path = item_dir / source_path.name
            try:
                shutil.copy2(source_path, staged_path)
                file_hash = compute_file_hash(staged_path)
                duplicate = self._document_repo.get_by_hash_global(file_hash)
                info = get_file_info(staged_path)
                check = (
                    verify_pdf(staged_path)
                    if info["extension"] == ".pdf"
                    else {
                        "readable": True,
                        "page_count": 1,
                        "has_text": False,
                        "is_protected": False,
                        "error": None,
                    }
                )
                evidence = self._identity_extractor.extract(staged_path)
                batch.documents.append(StagedDocument(
                    original_path=str(source_path),
                    staged_path=str(staged_path),
                    original_name=source_path.name,
                    file_hash=file_hash,
                    check=check,
                    evidence=evidence,
                    duplicate_document_id=duplicate.id if duplicate else None,
                    duplicate_patient_id=duplicate.patient_id if duplicate else None,
                    error=check.get("error"),
                ))
            except Exception as exc:
                batch.documents.append(StagedDocument(
                    original_path=str(source_path),
                    staged_path=str(staged_path),
                    original_name=source_path.name,
                    file_hash="",
                    check={"readable": False, "page_count": 0, "has_text": False},
                    evidence=PatientIdentityEvidence(source_path=str(source_path)),
                    error=str(exc),
                ))
        if progress_callback:
            progress_callback(len(file_paths), len(file_paths), "Completato")
        return batch

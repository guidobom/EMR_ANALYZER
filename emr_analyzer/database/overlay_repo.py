"""Versioned correction overlays for immutable normalized documents."""

from __future__ import annotations

from dataclasses import replace
import hashlib

from .engine import DatabaseEngine
from ..models.clinical_registry import DocumentTextOverlay


def text_hash(text: str) -> str:
    return hashlib.sha256(str(text or "").encode("utf-8")).hexdigest()


class DocumentTextOverlayRepository:
    def __init__(self, db: DatabaseEngine):
        self.db = db

    def save(self, overlay: DocumentTextOverlay) -> DocumentTextOverlay:
        row = self.db.execute(
            """SELECT COALESCE(MAX(version), 0) AS version
               FROM document_text_overlays WHERE document_id=?""",
            (overlay.document_id,),
        ).fetchone()
        version = int(row["version"] or 0) + 1
        overlay = replace(overlay, version=version)
        with self.db:
            self.db.execute(
                """UPDATE document_text_overlays SET status='superseded'
                   WHERE document_id=? AND status='active'""",
                (overlay.document_id,),
            )
            self.db.execute(
                """INSERT INTO document_text_overlays
                   (overlay_id, patient_id, document_id, version,
                    base_text_hash, corrected_text, reason, author_id,
                    status, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    overlay.overlay_id, overlay.patient_id,
                    overlay.document_id, overlay.version,
                    overlay.base_text_hash, overlay.corrected_text,
                    overlay.reason, overlay.author_id, overlay.status,
                    overlay.created_at,
                ),
            )
        return overlay

    def get_active(self, document_id: str) -> DocumentTextOverlay | None:
        row = self.db.execute(
            """SELECT * FROM document_text_overlays
               WHERE document_id=? AND status='active'
               ORDER BY version DESC LIMIT 1""",
            (document_id,),
        ).fetchone()
        if row is None:
            return None
        return DocumentTextOverlay(
            overlay_id=row["overlay_id"], patient_id=row["patient_id"],
            document_id=row["document_id"], version=row["version"],
            base_text_hash=row["base_text_hash"],
            corrected_text=row["corrected_text"], reason=row["reason"],
            author_id=row["author_id"], status=row["status"],
            created_at=row["created_at"],
        )

    def effective_text(self, document_id: str, base_text: str) -> str:
        overlay = self.get_active(document_id)
        return overlay.corrected_text if overlay else base_text

    def history(self, document_id: str) -> list[DocumentTextOverlay]:
        rows = self.db.execute(
            """SELECT * FROM document_text_overlays WHERE document_id=?
               ORDER BY version DESC""",
            (document_id,),
        ).fetchall()
        return [
            DocumentTextOverlay(
                overlay_id=row["overlay_id"], patient_id=row["patient_id"],
                document_id=row["document_id"], version=row["version"],
                base_text_hash=row["base_text_hash"],
                corrected_text=row["corrected_text"], reason=row["reason"],
                author_id=row["author_id"], status=row["status"],
                created_at=row["created_at"],
            )
            for row in rows
        ]

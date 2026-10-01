"""Active anonymized text of a report: normalized Markdown plus overlay."""

from __future__ import annotations

import hashlib
from pathlib import Path

from ..config import active_workspace


def normalized_text_path(patient_id: str, document_id: str) -> Path | None:
    candidates = (
        active_workspace.path / patient_id / "extraction" / f"{document_id}.md",
        active_workspace.path / patient_id / "docling" / f"{document_id}.md",
    )
    return next((path for path in candidates if path.exists()), None)


def base_text(patient_id: str, document_id: str) -> str:
    path = normalized_text_path(patient_id, document_id)
    return path.read_text(encoding="utf-8") if path else ""


def effective_text(patient_id: str, document_id: str, overlay_repo=None) -> str:
    """The text extraction reads: the latest overlay, else the Markdown."""
    text = base_text(patient_id, document_id)
    return overlay_repo.effective_text(document_id, text) if (overlay_repo and text) else text


def save_text_overlay(overlay_repo, patient_id: str, document_id: str, new_text: str,
                      *, audit_repo=None) -> None:
    """Save a corrected text as a new overlay version; the Markdown is kept."""
    from ..models.clinical_registry import DocumentTextOverlay

    original = base_text(patient_id, document_id)
    digest = hashlib.sha256(original.encode("utf-8")).hexdigest()
    overlay_repo.save(DocumentTextOverlay(
        patient_id=patient_id, document_id=document_id,
        corrected_text=new_text, base_text_hash=digest))
    if audit_repo is not None:
        audit_repo.log(patient_id, "document_text_overlay_created", "document", document_id,
                       {"base_text_hash": digest}, actor_id="local_user", actor_role="clinician")

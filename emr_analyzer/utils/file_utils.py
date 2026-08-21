"""File utility functions: hashing, validation, type detection."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Optional

from ..config import SUPPORTED_EXTENSIONS


def supported_file_dialog_filter() -> str:
    """Qt file-dialog filter kept in sync with the ingestion whitelist."""
    patterns = " ".join(f"*{extension}" for extension in SUPPORTED_EXTENSIONS)
    return f"Documenti supportati ({patterns});;Tutti i file (*)"


def compute_file_hash(file_path: str | Path, algorithm: str = "sha256") -> str:
    """Compute hash of a file."""
    h = hashlib.new(algorithm)
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def is_supported_file(file_path: str | Path) -> bool:
    """Check if the file extension is supported.

    macOS ``._*`` AppleDouble sidecar files are metadata for the real file
    next to them, never documents themselves: they carry no readable text,
    would always fail identity extraction and would pollute the "Da
    assegnare" bucket forever.  They are filtered out here.
    """
    path = Path(file_path)
    if path.name.startswith("._"):
        return False
    return path.suffix.lower() in SUPPORTED_EXTENSIONS


def is_pdf(file_path: str | Path) -> bool:
    return Path(file_path).suffix.lower() == ".pdf"


def is_image(file_path: str | Path) -> bool:
    return Path(file_path).suffix.lower() in (
        ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp",
    )


def detect_file_type(file_path: str | Path) -> str:
    """Return the deterministic extraction family for a supported file."""
    ext = Path(file_path).suffix.lower()
    if ext == ".pdf":
        return "pdf"
    if ext in (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"):
        return "image"
    if ext in (".txt", ".md", ".hl7"):
        return "text"
    if ext == ".csv":
        return "csv"
    if ext in (".xml", ".cda"):
        return "xml"
    if ext == ".json":
        return "json"
    if ext in (".doc", ".docx"):
        return "word"
    if ext == ".xlsx":
        return "spreadsheet"
    return "unsupported"


def verify_pdf(file_path: str | Path) -> dict:
    """Check if a PDF is readable and gather metadata."""
    result = {
        "readable": False,
        "has_text": False,
        "is_protected": False,
        "page_count": 0,
        "estimated_text_length": 0,
        "error": None,
    }
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(str(file_path))
        result["readable"] = True
        result["page_count"] = doc.page_count

        if doc.is_encrypted:
            result["is_protected"] = True
            doc.close()
            return result

        text = ""
        for page in doc:
            text += page.get_text()
        result["has_text"] = len(text.strip()) > 50
        result["estimated_text_length"] = len(text)
        doc.close()
    except Exception as e:
        result["error"] = str(e)

    return result


def get_file_info(file_path: str | Path) -> dict:
    """Get basic file information."""
    p = Path(file_path)
    return {
        "path": str(p.absolute()),
        "filename": p.name,
        "extension": p.suffix.lower(),
        "size_bytes": p.stat().st_size if p.exists() else 0,
        "size_mb": round(p.stat().st_size / (1024 * 1024), 2) if p.exists() else 0,
    }

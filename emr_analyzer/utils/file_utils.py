"""File utility functions: hashing, validation, type detection."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Optional

from ..config import SUPPORTED_EXTENSIONS


def compute_file_hash(file_path: str | Path, algorithm: str = "sha256") -> str:
    """Compute hash of a file."""
    h = hashlib.new(algorithm)
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def is_supported_file(file_path: str | Path) -> bool:
    """Check if the file extension is supported."""
    return Path(file_path).suffix.lower() in SUPPORTED_EXTENSIONS


def is_pdf(file_path: str | Path) -> bool:
    return Path(file_path).suffix.lower() == ".pdf"


def is_image(file_path: str | Path) -> bool:
    return Path(file_path).suffix.lower() in (".jpg", ".jpeg", ".png")


def detect_file_type(file_path: str | Path) -> str:
    """Returns 'pdf', 'image', or 'unsupported'."""
    ext = Path(file_path).suffix.lower()
    if ext == ".pdf":
        return "pdf"
    elif ext in (".jpg", ".jpeg", ".png"):
        return "image"
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

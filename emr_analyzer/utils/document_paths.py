"""Resolution of stored document paths against the current workspace.

The registry stores absolute ``original_path`` values captured at import
time.  When the project folder moves to another machine or disk (e.g. a
registry built on a Linux workstation and reopened on macOS from an
external drive), those absolute paths stop existing.  The conventional
workspace layout — ``<root>/<patient>/documents/original/<filename>`` —
lets the app rebuild them at open time without rewriting the database.
"""

from __future__ import annotations

import os

from ..config import active_workspace

# Conventional layouts, most common first (the app writes lowercase).
_ORIGINAL_LAYOUTS = (
    ("documents", "original"),
    ("Documents", "Original"),
)


def resolve_document_path(doc, workspace_root=None) -> str:
    """Return an existing path for *doc* (a model or a dict).

    The stored absolute path wins when it exists on disk; otherwise
    candidates are rebuilt from the current workspace root using the
    conventional layout.  Falls back to the stored path unchanged so
    callers surface a meaningful "file not found" instead of a wrong path.
    """
    stored = str(_get(doc, "original_path") or _get(doc, "stored_path") or "")
    if stored and os.path.isfile(stored):
        return stored

    filename = _get(doc, "filename") or os.path.basename(stored) or ""
    patient_id = _get(doc, "patient_id") or ""
    root = workspace_root or getattr(active_workspace, "path", None)
    if not root or not filename:
        return stored

    root_str = str(root)
    candidates = []
    if stored and not os.path.isabs(stored):
        candidates.append(os.path.join(root_str, stored))
    if patient_id:
        for layout in _ORIGINAL_LAYOUTS:
            candidates.append(
                os.path.join(root_str, patient_id, *layout, filename)
            )
        candidates.append(os.path.join(root_str, patient_id, filename))

    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate
    return stored


def _get(doc, key: str):
    if isinstance(doc, dict):
        return doc.get(key)
    return getattr(doc, key, None)

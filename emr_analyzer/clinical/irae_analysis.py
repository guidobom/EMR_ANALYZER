"""Full-registry immune-related adverse event (irAE) analysis.

The identification protocol is a long clinical prompt stored in
``~/.emr_analyzer/irae_prompt.txt`` (editable without code changes; a
template ships with the app).  The chronological registry can exceed the
context window of the Clinical State model, so the analysis runs the
SAME protocol over overlapping chunks of the registry and returns the
combined Markdown tables, with every entry citable by its ``[#id]``.
"""

from __future__ import annotations

from pathlib import Path

from ..config import BASE_DIR
from ..prompt_catalog import load_prompt as load_catalog_prompt

IRAE_PROMPT_PATH = BASE_DIR / "irae_prompt.txt"

# Registry characters per chunk (the protocol prompt itself is large,
# so the registry chunk must stay conservative).
_CHUNK_CHARS = 12000
# Entries repeated at the head of the next chunk so an event split by
# the chunk boundary is never lost.
_OVERLAP_ENTRIES = 10

SYSTEM_PROMPT = load_catalog_prompt("irae_system")


def format_entry(entry: dict) -> str:
    """One registry line with a citable id."""
    return (
        f"[#{entry.get('entry_id', '?')}] "
        f"[{entry.get('date_observed', '?')}] "
        f"[{entry.get('category', '?')}] {entry.get('description', '')}"
    )


def load_prompt(path: Path | None = None) -> str:
    """The protocol text; ``""`` when the file is missing."""
    prompt_path = Path(path or IRAE_PROMPT_PATH)
    try:
        return prompt_path.read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def ensure_prompt(path: Path | None = None) -> Path:
    """Return the protocol path, creating it from the bundled template
    on first use (the user can then edit it freely)."""
    prompt_path = Path(path or IRAE_PROMPT_PATH)
    if not prompt_path.exists():
        template = Path(__file__).parent / "irae_prompt_template.txt"
        if template.exists():
            prompt_path.parent.mkdir(parents=True, exist_ok=True)
            prompt_path.write_text(
                template.read_text(encoding="utf-8"), encoding="utf-8"
            )
    return prompt_path


def chunk_entries(
    entries: list[dict],
    chunk_chars: int = _CHUNK_CHARS,
    overlap: int = _OVERLAP_ENTRIES,
) -> list[list[dict]]:
    """Split the registry into character-budgeted chunks with overlap."""
    if chunk_chars <= 0 or overlap < 0:
        raise ValueError("chunk_chars must be positive and overlap non-negative")
    chunks: list[list[dict]] = []
    current: list[dict] = []
    size = 0
    for entry in entries:
        line_length = len(format_entry(entry)) + 1
        if line_length > chunk_chars:
            raise ValueError("Una voce supera il budget di contesto del registro")
        if current and size + line_length > chunk_chars:
            chunks.append(current)
            current = list(current[-overlap:]) if overlap else []
            size = sum(len(format_entry(e)) + 1 for e in current)
            while current and size + line_length > chunk_chars:
                size -= len(format_entry(current.pop(0))) + 1
        current.append(entry)
        size += line_length
    if current:
        chunks.append(current)
    return chunks


def build_analysis_prompt(
    clinical_profile: str,
    chunk: list[dict],
    protocol_text: str,
    part: int,
    total: int,
) -> str:
    """One analysis call: profile + registry chunk + the full protocol."""
    entries_text = "\n".join(format_entry(entry) for entry in chunk)
    return (
        f"PROFILO CLINICO:\n{clinical_profile}\n\n"
        f"REGISTRO CRONOLOGICO (parte {part}/{total}; ogni voce è "
        f"citabile con il suo [#id]):\n{entries_text}\n\n"
        f"PROTOCOLLO DI ANALISI:\n{protocol_text}"
    )


def build_analysis_plan(
    entries: list[dict],
    clinical_profile: str,
    protocol_text: str,
) -> list[str]:
    """Prompts for the whole registry, one per chunk."""
    chunks = chunk_entries(entries)
    return [
        build_analysis_prompt(
            clinical_profile, chunk, protocol_text, index, len(chunks)
        )
        for index, chunk in enumerate(chunks, start=1)
    ]

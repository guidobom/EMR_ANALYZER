"""Global Patient Set (GPS) loading for the interim pilot.

The SNOMED CT Global Patient Set is distributed under CC BY-ND 4.0 with no
membership or affiliate-license requirement, so it is the legitimate interim
data source while a full RF2 release licence is pending.  The freeset is a
*flat* TSV (``ConceptID | Active | FSN | USPreferredTerm``): no IS-A
hierarchy, no synonyms, and only US-English preferred terms.

``load_gps`` builds the same ``ReleaseSnapshot`` shape as the RF2 loader so the
index, retrieval and constrained extraction run unchanged.  The structural
limits are explicit and surfaced in the snapshot metadata:

- ``ancestors``/``descendants`` are always empty (no relationship data), so
  the hierarchical code metric degenerates to exact matching;
- retrieval matches US-English terms only, so Italian-first lookup degrades to
  English and recall on an Italian corpus underperforms the full release.

The GPS file itself must stay outside the repository (licensed data).
"""

from __future__ import annotations

from pathlib import Path
import zipfile

from .models import (
    FSN_TYPE,
    SYNONYM_TYPE,
    SnomedConcept,
    SnomedDescription,
)
from .rf2_loader import ReleaseSnapshot

# Header aliases, case-folded, mapped to canonical field names.  The official
# freeset header is ``ConceptID | Active | FSN | USPreferredTerm``; the GPS
# extractor and the 2019 guide used equivalent spellings.
_GPS_HEADER_ALIASES = {
    "conceptid": "concept_id",
    "id": "concept_id",
    "active": "active",
    "fsn": "fsn",
    "term_fully_specified_name": "fsn",
    "uspreferredterm": "pt",
    "preferredterm": "pt",
    "term_preferred": "pt",
    "preferred": "pt",
}
# Positional fallback when the file has no header row (official docs show
# headerless sample rows): ConceptID, Active, FSN, USPreferredTerm.
_POSITIONAL_COLUMNS = ("concept_id", "active", "fsn", "pt")


def _read_freeset_text(source: Path) -> str:
    if source.suffix.lower() == ".zip":
        with zipfile.ZipFile(source) as archive:
            name = next(
                (
                    candidate for candidate in archive.namelist()
                    if candidate.lower().endswith((".txt", ".tsv"))
                ),
                None,
            )
            if name is None:
                raise FileNotFoundError(
                    f"Nessun file freeset dentro lo zip: {source}"
                )
            return archive.read(name).decode("utf-8", errors="replace")
    return source.read_text(encoding="utf-8", errors="replace")


def _resolve_freeset(source: str | Path) -> Path:
    path = Path(source)
    if path.is_file():
        return path
    if path.is_dir():
        candidates = [
            candidate for candidate in list(path.rglob("*.txt"))
            + list(path.rglob("*.tsv"))
            if candidate.is_file()
        ]
        if not candidates:
            raise FileNotFoundError(
                f"Nessun file freeset GPS (.txt/.tsv) in: {path}"
            )
        return candidates[0]
    raise FileNotFoundError(f"Percorso GPS non trovato: {path}")


def _split(line: str, delimiter: str) -> list[str]:
    return [part.strip() for part in line.split(delimiter)]


def _column_map(header: list[str]) -> dict[str, int]:
    result: dict[str, int] = {}
    for index, name in enumerate(header):
        canonical = _GPS_HEADER_ALIASES.get(name)
        if canonical is not None:
            result.setdefault(canonical, index)
    return result


def _build_concepts(text: str, source: Path) -> dict[str, SnomedConcept]:
    lines = [line for line in text.splitlines() if line.strip()]
    if not lines:
        raise ValueError(f"Freeset GPS vuoto: {source}")
    # Tab is the official separator; a headerless pipe variant is tolerated.
    delimiter = "\t" if len(_split(lines[0], "\t")) >= 3 else "|"
    header = _split(lines[0], delimiter)
    columns = _column_map(header)
    if "concept_id" in columns:
        data_lines = lines[1:]
    else:
        columns = {name: index for index, name in enumerate(_POSITIONAL_COLUMNS)}
        data_lines = lines
    concepts: dict[str, SnomedConcept] = {}
    for line in data_lines:
        parts = line.split(delimiter)

        def field(name: str) -> str:
            index = columns.get(name)
            if index is None or index >= len(parts):
                return ""
            return parts[index].strip()

        concept_id = field("concept_id")
        if not concept_id.isdigit():
            continue
        # RF2 snapshot semantics: only the active rows are kept.  The GPS file
        # intentionally carries concepts inactivated since 2012 for historical
        # display; they are not retrieval candidates.
        if field("active") not in {"1", "true", "yes"}:
            continue
        fsn = field("fsn")
        preferred = field("pt")
        descriptions: list[SnomedDescription] = []
        if fsn:
            descriptions.append(SnomedDescription(
                description_id=f"gps-{concept_id}-fsn",
                concept_id=concept_id,
                lang="en",
                type_id=FSN_TYPE,
                term=fsn,
            ))
        if preferred:
            descriptions.append(SnomedDescription(
                description_id=f"gps-{concept_id}-pt",
                concept_id=concept_id,
                lang="en",
                type_id=SYNONYM_TYPE,
                term=preferred,
                acceptability="preferred",
            ))
        concepts[concept_id] = SnomedConcept(
            concept_id=concept_id,
            active=True,
            definition_status_id="",
            descriptions=tuple(descriptions),
            parents=frozenset(),
            children=frozenset(),
        )
    if not concepts:
        raise ValueError(f"Freeset GPS senza concetti attivi: {source}")
    return concepts


def load_gps(source: str | Path) -> ReleaseSnapshot:
    """Load a GPS freeset (TSV, directory, or zip) into a ReleaseSnapshot.

    ``languages`` is always ``("en",)``: the GPS carries only US-English terms.
    Edition is marked ``"GPS"`` so callers can tell the interim source apart
    from a licensed RF2 release.
    """
    path = _resolve_freeset(source)
    text = _read_freeset_text(path)
    concepts = _build_concepts(text, path)
    return ReleaseSnapshot(
        release_dir=path.parent,
        edition="GPS",
        release_date="",
        languages=("en",),
        concepts=concepts,
    )


__all__ = ["load_gps"]

"""RF2 release loading with standard snapshot semantics.

The loader only understands the international RF2 file contract; it makes no
assumption about the directory layout of a downloaded release.  It scans the
release directory recursively for the standard ``sct2_*`` files and builds a
``ReleaseSnapshot`` whose row set is the usual "max effectiveTime + active==1
per id" view.  This is the semantics that makes a *Full* file equivalent to a
*Snapshot* file, so either release variant can be consumed without extra
configuration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import re
from typing import Iterable

from .models import (
    ACCEPTABLE_ACCEPTABILITY,
    FSN_TYPE,
    IS_A_TYPE,
    PREFERRED_ACCEPTABILITY,
    SYNONYM_TYPE,
    SnomedConcept,
    SnomedDescription,
)


CONCEPT_FILENAME_RE = re.compile(r"^sct2_Concept_(Full|Snapshot)_INT_.*\.txt$")
DESCRIPTION_FILENAME_RE = re.compile(
    r"^sct2_Description_(Full|Snapshot)-([A-Za-z]+)_INT_.*\.txt$"
)
RELATIONSHIP_FILENAME_RE = re.compile(
    r"^sct2_Relationship_(Full|Snapshot)_INT_.*\.txt$"
)
LANGUAGE_REFSET_FILENAME_RE = re.compile(
    r"^der2_cRefset_Language(?:Snapshot|Full)-([A-Za-z]+)_INT_.*\.txt$"
)

CONCEPT_HEADER = (
    "id", "effectiveTime", "active", "moduleId", "definitionStatusId",
)
DESCRIPTION_HEADER = (
    "id", "effectiveTime", "active", "moduleId", "conceptId", "languageCode",
    "typeId", "term", "caseSignificanceId",
)
RELATIONSHIP_HEADER = (
    "id", "effectiveTime", "active", "moduleId", "sourceId", "destinationId",
    "relationshipGroup", "typeId", "characteristicTypeId", "modifierId",
)
LANGUAGE_REFSET_HEADER = (
    "id", "effectiveTime", "active", "moduleId", "refsetId",
    "referencedComponentId", "acceptabilityId",
)


def _parse_rows(text: str, header: tuple[str, ...]) -> list[dict[str, str]]:
    """Parse pipe-delimited rows, tolerating a missing or reordered header.

    When the first line is a header (every column is a known header name) it is
    dropped and used to map columns to their position; otherwise the rows are
    mapped positionally to ``header`` (headerless input).
    """
    lines = [line for line in text.splitlines() if line.strip()]
    if not lines:
        return []
    first = lines[0].split("|")
    has_header = all(column.strip() in header for column in first)
    column_map = {
        column.strip(): index for index, column in enumerate(first)
    } if has_header else {}
    data_lines = lines[1:] if has_header else lines
    rows = []
    for line in data_lines:
        parts = line.split("|")
        if column_map:
            row = {}
            for column in header:
                index = column_map.get(column)
                row[column] = parts[index] if index is not None else ""
        else:
            row = dict(zip(header, parts))
        rows.append(row)
    return rows


def _latest_active(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    """Keep the newest row per id; ``active==0`` supersedes older rows."""
    latest: dict[str, dict[str, str]] = {}
    for row in rows:
        identifier = row.get("id") or row.get("conceptId")
        if not identifier:
            continue
        current = latest.get(identifier)
        if current is None:
            latest[identifier] = row
            continue
        try:
            newer = int(row.get("effectiveTime") or 0) > int(
                current.get("effectiveTime") or 0
            )
        except (TypeError, ValueError):
            newer = False
        if newer:
            latest[identifier] = row
    return {
        identifier: row for identifier, row in latest.items()
        if str(row.get("active") or "").strip() == "1"
    }


@dataclass(frozen=True)
class ReleaseSnapshot:
    release_dir: Path
    edition: str = "INT"
    release_date: str = ""
    languages: tuple[str, ...] = ("en", "it")
    concepts: dict[str, SnomedConcept] = field(default_factory=dict)

    @property
    def digest(self) -> str:
        """Content digest that changes whenever the release files change."""
        return _release_digest(self.concepts)

    def concept(self, code: str) -> SnomedConcept | None:
        return self.concepts.get(code)

    def iter_concepts(self) -> Iterable[SnomedConcept]:
        return self.concepts.values()

    def resolve_label(
        self, code: str, langs: Iterable[str] | None = None
    ) -> str:
        concept = self.concepts.get(code)
        if concept is None:
            return ""
        return concept.preferred_term(langs or self.languages)


def load_snapshot(
    release_dir: str | Path,
    *,
    languages: Iterable[str] = ("en", "it"),
) -> ReleaseSnapshot:
    """Load one release directory into a ReleaseSnapshot.

    At least one description language must be present in the directory;
    the international English descriptions are the usual baseline and the
    Italian ones are preferred for the Italian clinical corpus.  A missing
    language is tolerated (the loader keeps whatever is found).
    """
    root = Path(release_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Directory release SNOMED non trovata: {root}")
    # Both the sct2_* content files and the der2_* language refsets live in a
    # standard release; glob every RF2 text file and let the filename matchers
    # pick what they need.
    files = list(root.rglob("*.txt"))
    if not files:
        raise FileNotFoundError(
            f"Nessun file RF2 (*.txt) trovato in: {root}"
        )

    concept_file = _first_match(files, CONCEPT_FILENAME_RE)
    relationship_file = _first_match(files, RELATIONSHIP_FILENAME_RE)
    if concept_file is None:
        raise FileNotFoundError(
            f"File sct2_Concept_* mancante nella release: {root}"
        )

    concepts = _load_concepts(concept_file)
    descriptions = _load_descriptions(files)
    relationships = (
        _load_relationships(relationship_file) if relationship_file else {}
    )
    acceptability = _load_language_acceptability(files)
    _merge_descriptions(concepts, descriptions, acceptability)
    _apply_relationships(concepts, relationships)

    langs = tuple(dict.fromkeys(languages))
    found_langs = _present_languages(concepts)
    effective_langs = tuple(
        lang for lang in langs if lang in found_langs
    ) or tuple(found_langs)

    release_date = _release_date(concept_file.name)
    edition = "INT"
    return ReleaseSnapshot(
        release_dir=root,
        edition=edition,
        release_date=release_date,
        languages=effective_langs,
        concepts=concepts,
    )


def _first_match(files: list[Path], pattern: re.Pattern) -> Path | None:
    for path in files:
        if pattern.match(path.name):
            return path
    return None


def _release_date(filename: str) -> str:
    match = re.search(r"(\d{8})", filename)
    return match.group(1) if match else ""


def _load_concepts(path: Path) -> dict[str, SnomedConcept]:
    rows = _latest_active(_parse_rows(path.read_text(encoding="utf-8"), CONCEPT_HEADER))
    return {
        row["id"]: SnomedConcept(
            concept_id=row["id"],
            active=True,
            definition_status_id=row.get("definitionStatusId", ""),
        )
        for row in rows.values()
    }


def _load_descriptions(files: list[Path]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in files:
        match = DESCRIPTION_FILENAME_RE.match(path.name)
        if match is None:
            continue
        lang = match.group(2)
        for row in _parse_rows(
            path.read_text(encoding="utf-8"), DESCRIPTION_HEADER
        ):
            row["_lang"] = lang
            rows.append(row)
    return rows


def _load_relationships(path: Path) -> dict[str, set[str]]:
    rows = _latest_active(_parse_rows(
        path.read_text(encoding="utf-8"), RELATIONSHIP_HEADER
    ))
    parents: dict[str, set[str]] = {}
    for row in rows.values():
        if row.get("typeId", "").strip() != IS_A_TYPE:
            continue
        if str(row.get("active") or "").strip() != "1":
            continue
        source = row.get("sourceId", "").strip()
        destination = row.get("destinationId", "").strip()
        if not source or not destination:
            continue
        parents.setdefault(source, set()).add(destination)
    return parents


def _load_language_acceptability(files: list[Path]) -> dict[str, str]:
    """Map description id -> acceptability id from language refsets."""
    result: dict[str, str] = {}
    for path in files:
        if LANGUAGE_REFSET_FILENAME_RE.match(path.name) is None:
            continue
        for row in _parse_rows(
            path.read_text(encoding="utf-8"), LANGUAGE_REFSET_HEADER
        ):
            if str(row.get("active") or "").strip() != "1":
                continue
            description_id = row.get("referencedComponentId", "").strip()
            acceptability = row.get("acceptabilityId", "").strip()
            if description_id and acceptability:
                result[description_id] = acceptability
    return result


def _merge_descriptions(
    concepts: dict[str, SnomedConcept],
    descriptions: list[dict[str, str]],
    acceptability: dict[str, str],
) -> None:
    by_concept: dict[str, list[SnomedDescription]] = {}
    for row in _latest_active(descriptions).values():
        concept_id = row.get("conceptId", "").strip()
        description_id = row.get("id", "").strip()
        if concept_id not in concepts or not description_id:
            continue
        accept_id = acceptability.get(description_id, "")
        if accept_id == PREFERRED_ACCEPTABILITY:
            acceptance = "preferred"
        elif accept_id == ACCEPTABLE_ACCEPTABILITY:
            acceptance = "acceptable"
        else:
            acceptance = ""
        by_concept.setdefault(concept_id, []).append(SnomedDescription(
            description_id=description_id,
            concept_id=concept_id,
            lang=str(row.get("_lang") or row.get("languageCode") or "en"),
            type_id=row.get("typeId", "").strip(),
            term=str(row.get("term", "")).strip(),
            acceptability=acceptance,
            active=True,
        ))
    for concept_id, rows in by_concept.items():
        concept = concepts[concept_id]
        concepts[concept_id] = SnomedConcept(
            concept_id=concept.concept_id,
            active=concept.active,
            definition_status_id=concept.definition_status_id,
            descriptions=tuple(rows),
            parents=concept.parents,
            children=concept.children,
        )


def _apply_relationships(
    concepts: dict[str, SnomedConcept],
    relationships: dict[str, set[str]],
) -> None:
    """Attach IS-A parent/child edges to every loaded concept."""
    children: dict[str, set[str]] = {}
    for source, parents in relationships.items():
        for parent in parents:
            children.setdefault(parent, set()).add(source)
    for code, concept in concepts.items():
        concepts[code] = SnomedConcept(
            concept_id=concept.concept_id,
            active=concept.active,
            definition_status_id=concept.definition_status_id,
            descriptions=concept.descriptions,
            parents=frozenset(relationships.get(code, ())),
            children=frozenset(children.get(code, ())),
        )


def _present_languages(concepts: dict[str, SnomedConcept]) -> set[str]:
    langs: set[str] = set()
    for concept in concepts.values():
        for description in concept.descriptions:
            langs.add(description.lang)
    return langs


def _release_digest(concepts: dict[str, SnomedConcept]) -> str:
    digest = hashlib.sha256()
    for code in sorted(concepts):
        concept = concepts[code]
        digest.update(code.encode("utf-8"))
        digest.update(b"\x00")
        digest.update(concept.fsn().encode("utf-8"))
        digest.update(b"\x00")
        for description in concept.descriptions:
            digest.update(description.term.encode("utf-8"))
            digest.update(b"\x00")
        for parent in sorted(concept.parents):
            digest.update(parent.encode("utf-8"))
            digest.update(b"\x00")
    return digest.hexdigest()[:16]

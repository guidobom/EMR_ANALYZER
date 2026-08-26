"""Core SNOMED CT value objects used across the SNOMED-enabled pipeline.

The models deliberately stay small and immutable.  They represent a single
release snapshot loaded from RF2 files (see ``rf2_loader.py``); no concept
graph navigation or terminology resolution lives here.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable


# RF2 component type identifiers (international edition, stable).
FSN_TYPE = "900000000000003001"
SYNONYM_TYPE = "900000000000013009"
PREFERRED_ACCEPTABILITY = "900000000000548007"
ACCEPTABLE_ACCEPTABILITY = "900000000000549004"
IS_A_TYPE = "116680003"

DEFAULT_LANG_ORDER = ("it", "en")

_SEMANTIC_TAG_RE = re.compile(r"\s*\(([^()]*)\)\s*$")


@dataclass(frozen=True, slots=True)
class SnomedDescription:
    description_id: str
    concept_id: str
    lang: str
    type_id: str
    term: str
    acceptability: str = ""  # "", "preferred" | "acceptable"
    active: bool = True


@dataclass(frozen=True, slots=True)
class SnomedConcept:
    concept_id: str
    active: bool
    definition_status_id: str
    descriptions: tuple[SnomedDescription, ...] = ()
    parents: frozenset[str] = frozenset()
    children: frozenset[str] = frozenset()

    def _descriptions_for(
        self, langs: Iterable[str]
    ) -> list[SnomedDescription]:
        wanted = tuple(langs)
        return [
            description for description in self.descriptions
            if description.active
            and description.lang in wanted
        ]

    @staticmethod
    def _strip_semantic_tag(term: str) -> str:
        match = _SEMANTIC_TAG_RE.search(str(term or "").strip())
        if match:
            return str(term).strip()[:match.start()].strip()
        return str(term or "").strip()

    def fsn(self, langs: Iterable[str] = DEFAULT_LANG_ORDER) -> str:
        """Return the fully specified name, honouring the language order."""
        for description in self._descriptions_for(langs):
            if description.type_id == FSN_TYPE:
                return description.term
        return ""

    def semantic_tag(self, langs: Iterable[str] = DEFAULT_LANG_ORDER) -> str:
        """Return the SNOMED semantic tag, e.g. ``(disorder)`` or ``(finding)``."""
        fsn = self.fsn(langs)
        match = _SEMANTIC_TAG_RE.search(fsn)
        return match.group(1) if match else ""

    def preferred_term(self, langs: Iterable[str] = DEFAULT_LANG_ORDER) -> str:
        """Return the preferred term, preferring the first available language.

        Selection heuristic: for the first language in order, return a
        ``preferred`` synonym if one exists, else the shortest synonym, else
        the fully specified name with its semantic tag stripped.  Real releases
        encode this preference in language refsets; this heuristic covers the
        synthetic fixture and is a safe fallback for any release.
        """
        wanted = tuple(langs)
        for lang in wanted:
            descriptions = self._descriptions_for((lang,))
            synonyms = [
                d for d in descriptions if d.type_id == SYNONYM_TYPE
            ]
            preferred = [
                d for d in synonyms if d.acceptability == "preferred"
            ]
            if preferred:
                return min(preferred, key=lambda d: len(d.term)).term
            if synonyms:
                return min(synonyms, key=lambda d: len(d.term)).term
            fsns = [d for d in descriptions if d.type_id == FSN_TYPE]
            if fsns:
                return self._strip_semantic_tag(fsns[0].term)
        # No description in the requested languages: fall back to any language.
        for description in self.descriptions:
            if description.active and description.type_id == SYNONYM_TYPE:
                return description.term
        for description in self.descriptions:
            if description.active and description.type_id == FSN_TYPE:
                return self._strip_semantic_tag(description.term)
        return ""

    def all_terms(self, langs: Iterable[str] = DEFAULT_LANG_ORDER) -> tuple[str, ...]:
        """Every active term (FSN + synonyms) in the requested languages."""
        return tuple(
            description.term for description in self._descriptions_for(langs)
        )

    def display_label(self, langs: Iterable[str] = DEFAULT_LANG_ORDER) -> str:
        """Shortest preferred/synonym term; used in the candidate catalog."""
        term = self.preferred_term(langs)
        return term or self.concept_id


@dataclass(frozen=True, slots=True)
class SnomedCandidate:
    concept: SnomedConcept
    score: float
    matched_term: str
    match_kind: str  # exact | phrase | token | prefix | fuzzy


@dataclass(frozen=True, slots=True)
class SnomedCandidateSet:
    """A closed, chunk-scoped candidate set for constrained generation.

    ``codes`` is the deterministic order (ascending) exposed as the JSON-schema
    ``enum``.  ``domain_fact_types`` maps every code to the atomic fact types
    its concept belongs to (see ``domains.py``); it feeds the permissive domain
    consistency guard, never a hard filter.
    """

    codes: tuple[str, ...]
    candidates: tuple[SnomedCandidate, ...]
    domain_fact_types: dict[str, tuple[str, ...]]
    digest: str
    empty: bool = False

    def concept(self, code: str) -> SnomedConcept | None:
        for candidate in self.candidates:
            if candidate.concept.concept_id == code:
                return candidate.concept
        return None

    def to_catalog(self, langs: Iterable[str] = DEFAULT_LANG_ORDER) -> str:
        """One code|label [domains] row per candidate, deterministic order."""
        rows = []
        for candidate in self.candidates:
            concept = candidate.concept
            label = concept.display_label(langs)
            domains = self.domain_fact_types.get(
                concept.concept_id, ()
            )
            domain_text = ", ".join(domains) if domains else "generico"
            rows.append(
                f"{concept.concept_id} | {label} [domini: {domain_text}]"
            )
        return "\n".join(rows)

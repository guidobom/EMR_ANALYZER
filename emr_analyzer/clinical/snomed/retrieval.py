"""Candidate-set retrieval for the constrained SNOMED extraction path.

The retriever is the deterministic front half of the pipeline: given a source
chunk it returns a closed ``SnomedCandidateSet`` (the ``enum`` the LLM is
constrained to) plus the per-code domain mapping used by the consistency guard.
It never generates codes itself; it only *recalls* them from the index.

Abbreviation fallback: when a token in the source text matches the curated
abbreviation dictionary but the index has no literal description token for it
(e.g. ``dm`` in a release that does not carry that synonym), the retriever
expands the abbreviation into its full bilingual phrases, searches those, and
merges the recalled concepts into the candidate set.  The release's own
description surface stays authoritative whenever it already resolves the
token, so existing retrieval behaviour is unchanged.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable

from .domains import fact_types_for_concept
from .index import SnomedIndex, normalize_tokens
from .models import (
    DEFAULT_LANG_ORDER,
    SnomedCandidate,
    SnomedCandidateSet,
    SnomedConcept,
)


_ABBREVIATIONS_RESOURCE = (
    Path(__file__).resolve().parents[2] / "resources" / "snomed_abbreviations.json"
)


def load_abbreviations() -> dict[str, tuple[str, ...]]:
    """Load the curated abbreviation fallback from the packaged resource.

    Keys are lower-cased normalized tokens; values are bilingual search
    phrases resolved through the index.  A missing/unreadable resource is
    tolerated (empty map), keeping the standard path untouched.
    """
    if not _ABBREVIATIONS_RESOURCE.exists():
        return {}
    try:
        data = json.loads(_ABBREVIATIONS_RESOURCE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return {
        str(key).casefold().strip(): tuple(
            str(phrase).strip()
            for phrase in value
            if str(phrase).strip()
        )
        for key, value in data.items()
        if key != "_comment" and value
    }


def candidate_digest(codes: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for code in sorted(set(codes)):
        digest.update(code.encode("utf-8"))
        digest.update(b"\x00")
    return digest.hexdigest()[:16]


class SnomedCandidateRetriever:
    """Return a deterministic candidate set for one source chunk."""

    def __init__(
        self,
        index: SnomedIndex,
        *,
        langs: Iterable[str] = DEFAULT_LANG_ORDER,
        cap: int = 32,
        min_score: float = 0.2,
        release_digest: str = "",
        abbreviations: dict[str, Iterable[str]] | None = None,
    ):
        self.index = index
        self.langs = tuple(langs)
        self.cap = max(8, min(64, int(cap)))
        self.min_score = float(min_score)
        # Stamped onto every set so a release swap invalidates extraction
        # checkpoints even when the candidate codes happen not to change.
        self.release_digest = str(release_digest)
        # ``None`` loads the packaged curated dictionary; ``{}`` disables the
        # abbreviation fallback entirely.
        self.abbreviations = (
            load_abbreviations() if abbreviations is None
            else {
                str(key).casefold().strip(): tuple(value)
                for key, value in abbreviations.items()
                if value
            }
        )

    def candidates_for_text(
        self,
        text: object,
        *,
        cap: int | None = None,
        min_score: float | None = None,
    ) -> SnomedCandidateSet:
        cap = cap or self.cap
        min_score = min_score if min_score is not None else self.min_score
        rows = self.index.search(
            text,
            langs=self.langs,
            cap=cap,
            min_score=min_score,
        )
        rows = self._with_abbreviation_fallback(text, rows, cap=cap,
                                                min_score=min_score)
        candidates: list[SnomedCandidate] = []
        domain_fact_types: dict[str, tuple[str, ...]] = {}
        for concept, score, matched_term, kind in rows:
            candidates.append(SnomedCandidate(
                concept=concept,
                score=score,
                matched_term=matched_term,
                match_kind=kind,
            ))
            domain_fact_types[concept.concept_id] = fact_types_for_concept(
                concept, langs=self.langs
            )
        codes = tuple(candidate.concept.concept_id for candidate in candidates)
        return SnomedCandidateSet(
            codes=codes,
            candidates=tuple(candidates),
            domain_fact_types=domain_fact_types,
            digest=candidate_digest(codes),
            empty=not codes,
            release_digest=self.release_digest,
        )

    def _with_abbreviation_fallback(
        self,
        text: object,
        rows: list[tuple[SnomedConcept, float, str, str]],
        *,
        cap: int,
        min_score: float,
    ) -> list[tuple[SnomedConcept, float, str, str]]:
        """Merge concepts recalled by expanding abbreviations not present as
        literal tokens in the index.

        Only tokens that (a) are in the curated dictionary and (b) have no
        literal posting are expanded: the release's own description surface
        stays authoritative for abbreviations it already carries (e.g. ``spo2``
        in many releases).  Expansions are searched with the same cap/min_score
        and merged, then the combined rows are re-capped by ``(-score, code)``
        and re-ordered by code, matching ``SnomedIndex.search`` ordering.
        """
        if not self.abbreviations:
            return rows
        query_tokens = normalize_tokens(text)
        if not query_tokens:
            return rows
        seen_abbr: set[str] = set()
        hit_codes = {concept.concept_id for concept, *_ in rows}
        expanded: list[tuple[SnomedConcept, float, str, str]] = list(rows)
        for token in query_tokens:
            if token in seen_abbr:
                continue
            seen_abbr.add(token)
            phrases = self.abbreviations.get(token)
            if not phrases:
                continue
            if self.index.has_token(token):
                # The index already resolves this abbreviation literally.
                continue
            for phrase in phrases:
                for concept, score, matched, kind in self.index.search(
                    phrase,
                    langs=self.langs,
                    cap=cap,
                    min_score=min_score,
                ):
                    if kind == "fuzzy":
                        # Expansions are term lookups: partial-token overlap
                        # rows (shared word, e.g. "ipertensione arteriosa"
                        # against "pressione arteriosa") are noise, not the
                        # intended concept.
                        continue
                    if concept.concept_id in hit_codes:
                        continue
                    hit_codes.add(concept.concept_id)
                    expanded.append((concept, score, token, "abbreviation"))
        ranked = sorted(expanded, key=lambda row: (-row[1], row[0].concept_id))
        ranked = ranked[:max(0, int(cap))]
        return sorted(ranked, key=lambda row: row[0].concept_id)

    def candidates_for_concepts(
        self,
        concepts: Iterable[SnomedConcept],
    ) -> SnomedCandidateSet:
        """Build a set from explicit concepts (lab/LOINC mapping, tests)."""
        candidates: list[SnomedCandidate] = []
        domain_fact_types: dict[str, tuple[str, ...]] = {}
        for concept in concepts:
            candidates.append(SnomedCandidate(
                concept=concept, score=1.0, matched_term="", match_kind="exact",
            ))
            domain_fact_types[concept.concept_id] = fact_types_for_concept(
                concept, langs=self.langs
            )
        ordered = sorted(candidates, key=lambda c: c.concept.concept_id)
        codes = tuple(c.concept.concept_id for c in ordered)
        return SnomedCandidateSet(
            codes=codes,
            candidates=tuple(ordered),
            domain_fact_types=domain_fact_types,
            digest=candidate_digest(codes),
            empty=not codes,
            release_digest=self.release_digest,
        )

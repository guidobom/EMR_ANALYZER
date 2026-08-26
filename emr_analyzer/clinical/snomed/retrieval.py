"""Candidate-set retrieval for the constrained SNOMED extraction path.

The retriever is the deterministic front half of the pipeline: given a source
chunk it returns a closed ``SnomedCandidateSet`` (the ``enum`` the LLM is
constrained to) plus the per-code domain mapping used by the consistency guard.
It never generates codes itself; it only *recalls* them from the index.
"""

from __future__ import annotations

import hashlib
from typing import Iterable

from .domains import fact_types_for_concept
from .index import SnomedIndex
from .models import (
    DEFAULT_LANG_ORDER,
    SnomedCandidate,
    SnomedCandidateSet,
    SnomedConcept,
)


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
    ):
        self.index = index
        self.langs = tuple(langs)
        self.cap = max(8, min(64, int(cap)))
        self.min_score = float(min_score)
        # Stamped onto every set so a release swap invalidates extraction
        # checkpoints even when the candidate codes happen not to change.
        self.release_digest = str(release_digest)

    def candidates_for_text(
        self,
        text: object,
        *,
        cap: int | None = None,
        min_score: float | None = None,
    ) -> SnomedCandidateSet:
        rows = self.index.search(
            text,
            langs=self.langs,
            cap=cap or self.cap,
            min_score=min_score if min_score is not None else self.min_score,
        )
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

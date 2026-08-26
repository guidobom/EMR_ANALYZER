"""Inverted term index over a SNOMED release with IS-A navigation.

This is the deterministic retrieval engine behind the SNOMED RAG stage.  For
v1 it is deliberately lexical (normalized NFKD tokens, phrase/token/prefix/
fuzzy scoring) because the clinical sublanguage is closed and the candidate
set only needs to *recall*, while the LLM does the disambiguation.  Embedding
models can be added later if ``retrieval_recall`` proves insufficient in the
benchmark.
"""

from __future__ import annotations

from dataclasses import dataclass
import unicodedata
from typing import Iterable

from .models import DEFAULT_LANG_ORDER, SnomedConcept, SnomedDescription


def normalize_tokens(text: object) -> list[str]:
    """Lower-case, NFKD-normalized alphanumeric tokens (``HbA1c`` -> ``hba1c``)."""
    folded = unicodedata.normalize("NFKD", str(text or "")).casefold()
    folded = "".join(
        char for char in folded if not unicodedata.combining(char)
    )
    return [token for token in __import__("re").findall(r"[a-z0-9]+", folded)]


def _edit_distance(left: str, right: str) -> int:
    if abs(len(left) - len(right)) > 1:
        return 99
    if left == right:
        return 0
    previous = list(range(len(right) + 1))
    for index, char in enumerate(left, start=1):
        current = [index]
        for other_index, other_char in enumerate(right, start=1):
            cost = 0 if char == other_char else 1
            current.append(min(
                current[other_index - 1] + 1,
                previous[other_index] + 1,
                previous[other_index - 1] + cost,
            ))
        previous = current
    return previous[-1]


@dataclass(frozen=True, slots=True)
class _IndexedTerm:
    concept_id: str
    lang: str
    tokens: tuple[str, ...]
    full_tokens: frozenset[str]


class SnomedIndex:
    """Deterministic lexical index plus IS-A transitive navigation."""

    def __init__(
        self,
        concepts: Iterable[SnomedConcept],
        *,
        lang_order: Iterable[str] = DEFAULT_LANG_ORDER,
    ):
        self._lang_order = tuple(lang_order)
        self._concepts: dict[str, SnomedConcept] = {}
        self._terms: list[_IndexedTerm] = []
        self._token_postings: dict[str, set[str]] = {}
        for concept in concepts:
            self._concepts[concept.concept_id] = concept
            for description in concept.descriptions:
                if not description.active:
                    continue
                tokens = tuple(normalize_tokens(description.term))
                if not tokens:
                    continue
                indexed = _IndexedTerm(
                    concept_id=concept.concept_id,
                    lang=description.lang,
                    tokens=tokens,
                    full_tokens=frozenset(tokens),
                )
                self._terms.append(indexed)
                for token in indexed.full_tokens:
                    self._token_postings.setdefault(token, set()).add(
                        concept.concept_id
                    )

    # ------------------------------------------------------------------
    # Navigation
    # ------------------------------------------------------------------

    def concept(self, code: str) -> SnomedConcept | None:
        return self._concepts.get(code)

    def parents(self, code: str) -> frozenset[str]:
        concept = self._concepts.get(code)
        return concept.parents if concept else frozenset()

    def children(self, code: str) -> frozenset[str]:
        concept = self._concepts.get(code)
        return concept.children if concept else frozenset()

    def ancestors(self, code: str) -> set[str]:
        """All transitive IS-A ancestors (excluding the concept itself)."""
        seen: set[str] = set()
        frontier = list(self.parents(code))
        while frontier:
            current = frontier.pop()
            if current in seen:
                continue
            seen.add(current)
            frontier.extend(self.parents(current))
        return seen

    def descendants(self, code: str) -> set[str]:
        seen: set[str] = set()
        frontier = list(self.children(code))
        while frontier:
            current = frontier.pop()
            if current in seen:
                continue
            seen.add(current)
            frontier.extend(self.children(current))
        return seen

    def is_descendant_of(self, code: str, ancestor: str) -> bool:
        return ancestor in self.ancestors(code)

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def search(
        self,
        text: object,
        *,
        langs: Iterable[str] | None = None,
        cap: int = 32,
        min_score: float = 0.2,
    ) -> list[tuple[SnomedConcept, float, str, str]]:
        """Return ``(concept, score, matched_term, match_kind)`` rows.

        Only concepts whose match score reaches ``min_score`` are returned;
        results are capped to the top-``cap`` by score and then ordered
        deterministically by concept id (ascending).
        """
        wanted = set(langs or self._lang_order)
        query_tokens = normalize_tokens(text)
        if not query_tokens:
            return []
        query_full = frozenset(query_tokens)
        query_set = set(query_full)

        scores: dict[str, float] = {}
        match_terms: dict[str, tuple[str, str]] = {}

        # Narrow the candidate pool through the token postings, then score the
        # few survivors against their full term set.
        hit_concepts: set[str] = set()
        for token in query_full:
            hit_concepts.update(self._token_postings.get(token, ()))

        for concept_id in hit_concepts:
            best_score = 0.0
            best_match: tuple[str, str] = ("", "")
            for term in self._terms:
                if term.concept_id != concept_id or term.lang not in wanted:
                    continue
                term_tokens = set(term.tokens)
                if not query_set & term_tokens:
                    # A prefix match can still recall a term whose tokens are a
                    # superstring of the query (e.g. ``diabete`` -> ``diabete
                    # mellito``) even though no exact token overlaps.
                    prefix_hit = any(
                        token.startswith(q) for token in term_tokens
                        for q in query_set if len(q) >= 3
                    )
                    if not prefix_hit:
                        continue
                score, kind = _score_term(query_full, query_set, term_tokens)
                if score > best_score:
                    best_score = score
                    best_match = (term.full_tokens and term.tokens[0], kind)
            if best_score < min_score:
                continue
            scores[concept_id] = best_score
            match_terms[concept_id] = best_match

        ranked = sorted(
            scores.items(), key=lambda pair: (-pair[1], pair[0])
        )[:max(0, int(cap))]
        ordered = sorted(ranked, key=lambda pair: pair[0])
        result = []
        for concept_id, score in ordered:
            concept = self._concepts[concept_id]
            matched_term, kind = match_terms.get(concept_id, ("", ""))
            result.append((concept, score, matched_term, kind))
        return result


def _score_term(
    query_full: frozenset[str],
    query_set: set[str],
    term_tokens: set[str],
) -> tuple[float, str]:
    """Score one indexed term against the query token set.

    Tiers (best wins): exact full match > phrase/token coverage > prefix >
    fuzzy.  Coverage is ``|hit| / |term|`` so a query that mentions every word
    of a multi-word term scores 1.0 while ``diabete`` against
    ``diabete mellito`` scores 0.5.
    """
    if query_full == term_tokens:
        return 1.0, "exact"
    exact_hits = query_set & term_tokens
    coverage = len(exact_hits) / max(1, len(term_tokens))
    prefix_hits = {
        token for token in term_tokens - exact_hits
        if any(q.startswith(token) for q in query_set if len(q) >= 3)
    }
    prefix_boost = 0.4 * len(prefix_hits) / max(1, len(term_tokens))
    if coverage + prefix_boost >= min(1.0, 0.55 + prefix_boost):
        if exact_hits:
            return min(1.0, coverage + prefix_boost), "token"
        return min(1.0, coverage + prefix_boost), "prefix"
    fuzzy_hits = 0
    for token in term_tokens - exact_hits:
        if any(_edit_distance(token, query) <= 1 for query in query_set):
            fuzzy_hits += 1
    fuzzy_boost = 0.5 * fuzzy_hits / max(1, len(term_tokens))
    score = min(1.0, coverage + prefix_boost + fuzzy_boost)
    if score > 0.0:
        return score, "fuzzy"
    return 0.0, ""

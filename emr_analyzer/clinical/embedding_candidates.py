"""Embedding-based near-miss duplicate candidates for atomic evidence.

Deterministic deduplication merges only provably identical facts; free
paraphrases ("nodulo al polmone" vs "nodulo polmonare") escape it by design.
This module closes that gap with a local sentence-transformer: every atom's
*structured signature* is embedded, and same-category pairs above the
configured cosine threshold are proposed to the existing duplicate review
queue.  Merges are always human-confirmed — nothing is merged automatically.

The pipeline runs fully local; if the embedding package or model is
unavailable, candidate generation degrades gracefully (no pairs, the
SequenceMatcher path keeps working).
"""

from __future__ import annotations

import logging
import os

import numpy as np

from .atomic_evidence import _atomic_identity_key
from .concept_canonicalization import canonical_concept, canonical_severity

logger = logging.getLogger(__name__)

DEFAULT_EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
_PAIR_CAP_PER_PATIENT = 200
_PAIR_CAP_PER_ATOM = 3


def atom_signature(item) -> str:
    """Structured identity text: the embedding represents clinical identity,
    not verbatim wording."""
    parts = [
        str(item.category or ""),
        canonical_concept(item.category, item.normalized_entity),
    ]
    if item.anatomical_site:
        parts.append(f"sede {item.anatomical_site}")
    if item.laterality:
        parts.append(item.laterality)
    severity = canonical_severity(item.severity)
    if severity:
        parts.append(f"gravità {severity}")
    if item.value_text:
        parts.append(item.value_text)
    elif item.numeric_value is not None:
        parts.append(str(item.numeric_value))
    if item.unit:
        parts.append(item.unit)
    return " | ".join(parts)


class EmbeddingIndex:
    """Lazy local sentence-transformer index with graceful degradation."""

    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or os.environ.get(
            "EMR_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL
        )
        self._model = None
        self._failed = False

    def loaded(self):
        """Return the SentenceTransformer instance, or None if unavailable."""
        if self._model is not None or self._failed:
            return self._model
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
        except Exception as exc:
            self._failed = True
            logger.warning(
                "Dedup embedding non disponibile (nessun candidato "
                "semantico): %s", exc,
            )
        return self._model

    def encode(self, signatures: list[str]):
        model = self.loaded()
        if model is None or not signatures:
            return None
        vectors = model.encode(
            signatures, normalize_embeddings=True, show_progress_bar=False
        )
        return np.asarray(vectors, dtype=np.float32)


def _norm_entity(value: object) -> str:
    return " ".join(str(value or "").casefold().split())


def embedding_duplicate_pairs(
    evidence,
    *,
    threshold: float,
    repo=None,
    index: EmbeddingIndex | None = None,
) -> list[tuple[str, str, float, str]]:
    """Same-patient, same-category near-miss candidates via cosine.

    Returns 4-tuples ``(left_id, right_id, similarity, "embedding")`` ready
    for ``replace_duplicate_groups``.  Pairs whose deterministic identity key
    already matches are excluded (already merged), as are identical concepts
    on different dates (true follow-ups, not restatements).
    """
    items = [
        item for item in evidence
        if getattr(item, "evidence_id", None)
        and getattr(item, "normalized_entity", None)
    ]
    if len(items) < 2:
        return []
    index = index or EmbeddingIndex()
    model_name = index.model_name

    cached: dict[str, bytes] = {}
    if repo is not None:
        try:
            cached = repo.get_embeddings(
                [item.evidence_id for item in items], model_name=model_name
            )
        except Exception as exc:
            logger.debug("Cache embedding non leggibile: %s", exc)
    vectors: dict[str, np.ndarray] = {
        evidence_id: np.frombuffer(blob, dtype=np.float32)
        for evidence_id, blob in cached.items()
    }
    missing = [
        item for item in items if item.evidence_id not in vectors
    ]
    if missing:
        if index.loaded() is None:
            return []
        computed = index.encode([atom_signature(item) for item in missing])
        if computed is None:
            return []
        for item, vector in zip(missing, computed):
            vectors[item.evidence_id] = vector
        if repo is not None:
            try:
                repo.put_embeddings([
                    (
                        item.evidence_id, model_name,
                        vector.astype(np.float32).tobytes(),
                    )
                    for item, vector in zip(missing, computed)
                ])
            except Exception as exc:
                logger.debug("Cache embedding non scrivibile: %s", exc)

    by_category: dict[str, list] = {}
    identity_keys: dict[str, tuple] = {}
    for item in items:
        identity_keys[item.evidence_id] = _atomic_identity_key(item)
        by_category.setdefault(
            str(item.category or ""), []
        ).append(item)

    results: list[tuple[str, str, float, str]] = []
    for members in by_category.values():
        if len(members) < 2:
            continue
        matrix = np.stack([vectors[item.evidence_id] for item in members])
        scores = matrix @ matrix.T
        count = len(members)
        for i in range(count):
            left = members[i]
            row_candidates: list[tuple[float, str, str]] = []
            for j in range(i + 1, count):
                similarity = float(scores[i, j])
                if similarity < threshold:
                    continue
                right = members[j]
                if (
                    identity_keys[left.evidence_id]
                    == identity_keys[right.evidence_id]
                ):
                    continue
                if _norm_entity(left.normalized_entity) == _norm_entity(
                    right.normalized_entity
                ):
                    continue
                if (
                    left.observed_date and right.observed_date
                    and left.observed_date != right.observed_date
                ):
                    continue
                if (left.assertion or "") != (right.assertion or ""):
                    continue
                row_candidates.append((
                    similarity, left.evidence_id, right.evidence_id,
                ))
            row_candidates.sort(reverse=True)
            for similarity, left_id, right_id in (
                row_candidates[:_PAIR_CAP_PER_ATOM]
            ):
                results.append((
                    left_id, right_id, round(similarity, 4), "embedding",
                ))
    return results[:_PAIR_CAP_PER_PATIENT]

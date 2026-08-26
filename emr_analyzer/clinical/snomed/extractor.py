"""SNOMED-constrained atomic evidence extraction (Percorso A).

``SnomedAtomicEvidenceExtractor`` subclasses the base atomic extractor and
overrides the single hook that switches the constrained path on: every chunk
first asks the deterministic ``SnomedCandidateRetriever`` for a closed candidate
set, which is then exposed as the JSON-schema ``enum`` for ``snomed_code``.  The
LLM can never emit a code outside that set, and the ``concept`` free-text field
is removed from the wire contract.  Attribute extraction (polarity, dates,
exact citations, typed payloads) remains the base pipeline's job.

The base extractor never imports this module, so there is no import cycle; the
registry builder imports it lazily when ``atomic_extraction_variant == "snomed"``.
"""

from __future__ import annotations

from typing import Iterable

from ..atomic_evidence import AtomicEvidenceExtractor, build_atomic_evidence_schema
from ...prompt_catalog import load_prompt, prompts_digest
from ...settings import ClinicalPipelinePolicy
from .retrieval import SnomedCandidateRetriever


_SNOMED_ATOMIC_TASK = load_prompt(
    "atomic_evidence_it_snomed",
    required_markers=("snomed_code", "source_refs", "CANDIDATI"),
    minimum_length=200,
)

_SNOMED_ATOMIC_BASE_SYSTEM_PROMPT = load_prompt(
    "atomic_evidence_system_snomed"
)
_SNOMED_ATOMIC_REPAIR_BASE_SYSTEM_PROMPT = load_prompt(
    "atomic_evidence_repair_system_snomed"
)
_SNOMED_ATOMIC_COVERAGE_BASE_SYSTEM_PROMPT = load_prompt(
    "atomic_evidence_coverage_system_snomed"
)

_SNOMED_ATOMIC_SYSTEM_PROMPT = (
    _SNOMED_ATOMIC_BASE_SYSTEM_PROMPT + "\n\n" + _SNOMED_ATOMIC_TASK
)
_SNOMED_ATOMIC_VALIDATION_REPAIR_SYSTEM_PROMPT = (
    _SNOMED_ATOMIC_REPAIR_BASE_SYSTEM_PROMPT + "\n\n" + _SNOMED_ATOMIC_TASK
)
_SNOMED_ATOMIC_COVERAGE_RECOVERY_SYSTEM_PROMPT = (
    _SNOMED_ATOMIC_COVERAGE_BASE_SYSTEM_PROMPT + "\n\n" + _SNOMED_ATOMIC_TASK
)

# Prompt-contract digest for the constrained variant.  The per-chunk schema
# varies with the candidate set (handled by the schema cache key), so the
# digest anchors on the prompt files plus the SNOMED schema shape; any change
# here invalidates stored extraction checkpoints for this variant.
SNOMED_ATOMIC_PROMPT_DIGEST = prompts_digest(
    _SNOMED_ATOMIC_SYSTEM_PROMPT,
    _SNOMED_ATOMIC_VALIDATION_REPAIR_SYSTEM_PROMPT,
    _SNOMED_ATOMIC_COVERAGE_RECOVERY_SYSTEM_PROMPT,
    schema=build_atomic_evidence_schema(snomed_codes=()),
)


class SnomedAtomicEvidenceExtractor(AtomicEvidenceExtractor):
    """Atomic extraction constrained to a closed per-chunk SNOMED candidate set."""

    _atomic_extraction_method = "llm_atomic_snomed_v1"

    def __init__(
        self,
        llm_client,
        *,
        retriever: SnomedCandidateRetriever,
        policy: ClinicalPipelinePolicy | None = None,
        task_prompt: str | None = None,
        system_prompt: str | None = None,
        repair_system_prompt: str | None = None,
        coverage_system_prompt: str | None = None,
    ):
        # The retriever carries the release digest and stamps it on every
        # candidate set it builds, so a release swap invalidates checkpoints
        # even when the retrieved codes happen to be unchanged.
        self.retriever = retriever
        super().__init__(
            llm_client,
            policy=policy,
            task_prompt=task_prompt or _SNOMED_ATOMIC_TASK,
            system_prompt=(
                system_prompt or _SNOMED_ATOMIC_BASE_SYSTEM_PROMPT
            ),
            repair_system_prompt=(
                repair_system_prompt
                or _SNOMED_ATOMIC_REPAIR_BASE_SYSTEM_PROMPT
            ),
            coverage_system_prompt=(
                coverage_system_prompt
                or _SNOMED_ATOMIC_COVERAGE_BASE_SYSTEM_PROMPT
            ),
        )

    @property
    def release_digest(self) -> str:
        return str(getattr(self.retriever, "release_digest", "") or "")

    def _snomed_candidates_for_chunk(self, chunk):
        """Closed candidate set for the chunk; empty sets degrade upstream."""
        return self.retriever.candidates_for_text(chunk.text)


__all__ = [
    "SNOMED_ATOMIC_PROMPT_DIGEST",
    "SnomedAtomicEvidenceExtractor",
]

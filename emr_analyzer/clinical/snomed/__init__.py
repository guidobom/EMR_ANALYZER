"""SNOMED CT foundation: RF2 loading, indexing, retrieval and domains.

Public surface for the SNOMED RAG stage used by both pipeline paths:

- ``ReleaseSnapshot`` / ``load_snapshot``: standard RF2 release loading.
- ``SnomedIndex``: deterministic lexical index with IS-A navigation.
- ``SnomedCandidateRetriever``: closed candidate sets for constrained LLM
  extraction.
- ``fact_types_for_concept``: SNOMED semantics -> atomic fact types.

The extractor (``extractor.py``) and the experimental direct-events builder
(``direct_events.py``) are imported lazily by the registry to avoid circular
imports.
"""

from __future__ import annotations

# Versioning of the SNOMED pipeline stages.  These are separate from the
# base atomic constants because they must invalidate extraction checkpoints
# whenever the constrained prompt, the candidate retrieval contract or the
# release changes.
SNOMED_ATOMIC_PIPELINE_VERSION = "registry_pipeline_snomed_v1"
SNOMED_ATOMIC_PROMPT_VERSION = "atomic_evidence_it_snomed_v1"

from .domains import (  # noqa: E402
    FSN_FACT_TYPE_HINTS,
    fact_types_for_concept,
    semantic_tag,
)
from .index import SnomedIndex, normalize_tokens  # noqa: E402
from .models import (  # noqa: E402
    DEFAULT_LANG_ORDER,
    SnomedCandidate,
    SnomedCandidateSet,
    SnomedConcept,
    SnomedDescription,
)
from .release_manager import manager as snomed_manager  # noqa: E402
from .release_manager import _ReleaseManager  # noqa: E402
from .retrieval import (  # noqa: E402
    SnomedCandidateRetriever,
    candidate_digest,
)
from .rf2_loader import (  # noqa: E402
    ReleaseSnapshot,
    load_snapshot,
)

__all__ = [
    "DEFAULT_LANG_ORDER",
    "FSN_FACT_TYPE_HINTS",
    "ReleaseSnapshot",
    "SNOMED_ATOMIC_PIPELINE_VERSION",
    "SNOMED_ATOMIC_PROMPT_VERSION",
    "SnomedCandidate",
    "SnomedCandidateRetriever",
    "SnomedCandidateSet",
    "SnomedConcept",
    "SnomedDescription",
    "SnomedIndex",
    "_ReleaseManager",
    "candidate_digest",
    "fact_types_for_concept",
    "load_snapshot",
    "normalize_tokens",
    "semantic_tag",
    "snomed_manager",
]

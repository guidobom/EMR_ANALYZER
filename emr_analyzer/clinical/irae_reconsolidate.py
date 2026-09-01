"""Re-run only the Layer 4 consolidation of a saved irAE report.

A finished multi-patient analysis persists one ``irae_report.json`` per
patient (``irae_reports``).  When the final consolidation call (Layer 4)
exceeded the default output budget (``DEFAULT_MAX_TOKENS`` = 8192), the saved
report has ``consolidation.applied == False`` and its ``iraes`` only hold the
per-organ definitive partition — cross-organ duplicates were never merged.

The per-organ findings and the anchor are already in the saved report, so the
consolidation can be re-run WITHOUT repeating the analysis (Layers 1-3).  This
module does exactly that: flatten ``organ_results`` into the finding list the
pipeline would have built (``analyze_irae``), call ``consolidate_iraes`` with a
higher ``max_tokens`` budget, and — when ``registry_rows`` are supplied —
rebuild the compact ``evidence`` provenance for the newly cited ids.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from .irae_layers import _compact_evidence, consolidate_iraes

# The default 8192 output tokens is too small for large per-organ finding sets
# (e.g. P001 had 80 findings): the model hits the cap mid-generation and the
# JSON is truncated.  16384 fits the observed worst cases; retries go higher.
DEFAULT_RECONSOLIDATE_MAX_TOKENS = 16384
_RETRY_BUDGETS = (16384, 24576, 32768)


def _normalize_evidence_id(eid: Any) -> str:
    """The pipeline stores evidence ids without the model's ``#`` prefix."""
    return str(eid).strip().lstrip("#")


def _rebuild_evidence(
    consolidation: dict, registry_rows: list[dict]
) -> list[dict]:
    """Compact provenance for only the ids cited by the new consolidation."""
    cited: list[str] = []
    for item in (
        consolidation.get("iraes", []) + consolidation.get("suspects", [])
    ):
        for eid in item.get("key_evidence_ids") or []:
            cited.append(_normalize_evidence_id(eid))
    by_id = {str(row.get("evidence_id")): row for row in registry_rows}
    return _compact_evidence([
        by_id[eid]
        for eid in dict.fromkeys(cited)
        if eid in by_id
    ])


def reconsolidate_report(
    report: dict,
    client,
    *,
    max_tokens: int = DEFAULT_RECONSOLIDATE_MAX_TOKENS,
    registry_rows: list[dict] | None = None,
) -> dict:
    """Re-run the Layer 4 consolidation of a saved report.

    Returns a NEW dict (the input ``report`` is never mutated).  ``iraes`` and
    ``consolidation`` are rebuilt from the per-organ findings; ``evidence`` is
    rebuilt from ``registry_rows`` (when given) for the ids cited by the new
    consolidation.  On failure the previous ``iraes``/``consolidation``/
    ``evidence`` are preserved and ``reconsolidation_error`` explains why.
    ``analyzed_at`` is left untouched; ``reconsolidated_at`` records the re-run.
    """
    out = dict(report)
    findings = [
        item
        for result in (report.get("organ_results") or {}).values()
        for item in result.get("iraes", [])
    ]
    if not findings:
        out["reconsolidated"] = False
        out["reconsolidated_at"] = None
        out["reconsolidation_error"] = (
            "Nessun finding per-organo da consolidare (organ_results vuoto)."
        )
        return out

    anchor = report.get("anchor")
    budgets = (max_tokens,) + tuple(
        b for b in _RETRY_BUDGETS if b > max_tokens
    )
    last = None
    for budget in budgets:
        consolidation = consolidate_iraes(
            client, findings, anchor, max_tokens=budget
        )
        if consolidation.get("applied"):
            out["iraes"] = consolidation["iraes"]
            out["consolidation"] = consolidation
            out["reconsolidated"] = True
            out["reconsolidated_at"] = datetime.now().isoformat(
                timespec="seconds"
            )
            out["reconsolidation_error"] = None
            if registry_rows is not None:
                out["evidence"] = _rebuild_evidence(
                    consolidation, registry_rows
                )
            return out
        last = consolidation
        error = consolidation.get("error") or ""
        if not error.startswith("OutputLimitError"):
            break

    out["reconsolidated"] = True
    out["reconsolidated_at"] = datetime.now().isoformat(timespec="seconds")
    out["reconsolidation_error"] = (
        last.get("error") if last is not None else "errore sconosciuto"
    )
    return out

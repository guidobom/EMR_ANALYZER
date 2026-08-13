"""Golden few-shot examples for the timeline-extraction prompts.

The golden set (user-confirmed timeline entries) is injected into the
extraction prompt of new documents so the model replicates the style and
canonical description of past human confirmations. Only de-identified
clinical fields are rendered — never source text, ids or patient identity.
"""

from __future__ import annotations

from typing import Optional

# Fields rendered in the prompt section. Anything else (source_texts, ids,
# timestamps, patient_id) is dropped so the examples stay de-identified.
_PROMPT_FIELDS = ("date_observed", "date_resolved", "category",
                  "description", "status", "confidence")

# Categories accepted by the discharge-letter prompt (its CATEGORIE list
# differs from the generic timeline prompt and does not include e.g.
# biomarker/histopathology). Examples outside this set are filtered at
# render time for discharge letters only.
DISCHARGE_ALLOWED_CATEGORIES = frozenset({
    "diagnosis", "treatment", "treatment_completed", "procedure",
    "surgery", "toxicity", "adverse_event", "consultation",
    "imaging_finding", "laboratory", "hospitalization", "discharge",
    "follow_up", "other",
})


def _project_fields(ex: dict) -> dict:
    """Keep only the de-identified prompt fields of a golden example."""
    return {k: ex.get(k) for k in _PROMPT_FIELDS if k in ex}


def select_examples(entries, current_patient_id: str,
                    max_examples: int = 10) -> list[dict]:
    """Select golden few-shot examples, excluding the current patient.

    Most recent first, stratified by category (one example per category),
    then the remaining budget is filled with the most recent ones. Capped
    at *max_examples*. Returns entry dicts via ``to_dict()``.
    """
    others = [e for e in entries if e.patient_id != current_patient_id]
    if not others:
        return []
    others = sorted(
        others,
        key=lambda e: (e.date_observed or "", e.entry_id or ""),
        reverse=True,
    )
    selected = []
    selected_ids = set()
    seen_categories = set()
    # Stratification: one example per category, most recent first.
    for e in others:
        if len(selected) >= max_examples:
            break
        if e.category in seen_categories:
            continue
        selected.append(e)
        selected_ids.add(e.entry_id)
        seen_categories.add(e.category)
    # Fill the remaining budget with the most recent examples.
    for e in others:
        if len(selected) >= max_examples:
            break
        if e.entry_id not in selected_ids:
            selected.append(e)
            selected_ids.add(e.entry_id)
    return [e.to_dict() for e in selected]


def format_examples_section(examples: Optional[list[dict]] = None,
                            allowed_categories=None) -> str:
    """Build the golden few-shot prompt section, or ``""`` when empty.

    Examples are rendered as plain lines (``[date] [category] description``),
    never as fenced JSON: if the model echoed a fenced example, the answer
    parser (which takes the first code fence) would pick it up and empty the
    extraction. Only :data:`_PROMPT_FIELDS` are rendered. The header makes
    explicit that these are NOT part of the text to analyze.
    """
    if not examples:
        return ""
    if allowed_categories is not None:
        examples = [
            ex for ex in examples
            if ex.get("category") in allowed_categories
        ]
        if not examples:
            return ""
    header = (
        "ESEMPI DI OUTPUT CORRETTO (voci golden confermate dall'utente su "
        "ALTRI pazienti):\n"
        "- Sono SOLO riferimenti di stile e struttura. NON fanno parte del "
        "testo da analizzare.\n"
        "- NON estrarle come osservazioni, NON copiarle nel risultato, "
        "NON citarle.\n"
        "- Usa SOLO le categorie elencate nella sezione CATEGORIE qui sopra."
    )
    lines = [header]
    for raw in examples:
        ex = _project_fields(raw)
        span = ex.get("date_observed") or "?"
        if ex.get("date_resolved"):
            span += f" → {ex['date_resolved']}"
        lines.append(
            f"[{span}] [{ex.get('category', '?')}] "
            f"{str(ex.get('description', '')).strip()}"
        )
    return "\n".join(lines)

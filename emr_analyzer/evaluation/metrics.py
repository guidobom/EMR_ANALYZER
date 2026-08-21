"""Clinically oriented metrics for event extraction and consolidation."""

from __future__ import annotations

from collections import defaultdict
from difflib import SequenceMatcher
from typing import Any

from ..clinical.temporal import date_bounds, temporal_distance_days


def _norm(value: object) -> str:
    return " ".join(str(value or "").casefold().replace("_", " ").split())


def _safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _event_similarity(gold: dict, predicted: dict) -> float:
    category = 1.0 if _norm(gold.get("category")) == _norm(
        predicted.get("category")
    ) else 0.0
    entity = SequenceMatcher(
        None,
        _norm(gold.get("canonical_entity") or gold.get("entity")),
        _norm(predicted.get("canonical_entity") or predicted.get("entity")),
    ).ratio()
    date_distance = temporal_distance_days(
        gold.get("first_evidence_date") or gold.get("date"),
        predicted.get("first_evidence_date") or predicted.get("date"),
    )
    temporal = 0.0 if date_distance is None else max(
        0.0, 1.0 - min(date_distance, 365) / 365
    )
    return category * 0.35 + entity * 0.45 + temporal * 0.20


def match_events(
    gold_events: list[dict], predicted_events: list[dict], threshold: float = 0.62
) -> tuple[list[tuple[int, int, float]], list[int], list[int]]:
    """Greedy one-to-one matching over category, entity and onset date."""
    candidates = sorted(
        (
            (_event_similarity(gold, predicted), gold_index, predicted_index)
            for gold_index, gold in enumerate(gold_events)
            for predicted_index, predicted in enumerate(predicted_events)
        ),
        reverse=True,
    )
    used_gold: set[int] = set()
    used_predicted: set[int] = set()
    matches = []
    for score, gold_index, predicted_index in candidates:
        if score < threshold:
            break
        if gold_index in used_gold or predicted_index in used_predicted:
            continue
        used_gold.add(gold_index)
        used_predicted.add(predicted_index)
        matches.append((gold_index, predicted_index, score))
    return (
        matches,
        [index for index in range(len(gold_events)) if index not in used_gold],
        [
            index for index in range(len(predicted_events))
            if index not in used_predicted
        ],
    )


def evaluate_patient_events(
    gold_events: list[dict], predicted_events: list[dict]
) -> dict[str, Any]:
    matches, false_negative_indices, false_positive_indices = match_events(
        gold_events, predicted_events
    )
    true_positive = len(matches)
    precision = _safe_div(true_positive, len(predicted_events))
    recall = _safe_div(true_positive, len(gold_events))
    f1 = _safe_div(2 * precision * recall, precision + recall)
    temporal_exact = 0
    temporal_compatible = 0
    certainty_correct = 0
    status_correct = 0
    citation_complete = 0
    matched_details = []
    for gold_index, predicted_index, score in matches:
        gold = gold_events[gold_index]
        predicted = predicted_events[predicted_index]
        gold_date = gold.get("first_evidence_date") or gold.get("date")
        predicted_date = predicted.get("first_evidence_date") or predicted.get("date")
        if gold_date and gold_date == predicted_date:
            temporal_exact += 1
        gold_bounds = date_bounds(gold_date)
        predicted_bounds = date_bounds(predicted_date)
        if gold_bounds and predicted_bounds and (
            gold_bounds[0] <= predicted_bounds[1]
            and predicted_bounds[0] <= gold_bounds[1]
        ):
            temporal_compatible += 1
        certainty_correct += int(
            _norm(gold.get("certainty")) == _norm(predicted.get("certainty"))
        )
        status_correct += int(
            _norm(gold.get("status")) == _norm(predicted.get("status"))
        )
        required_sources = set(gold.get("evidence_ids") or [])
        predicted_sources = set(predicted.get("evidence_ids") or [])
        citation_complete += int(
            not required_sources or required_sources.issubset(predicted_sources)
        )
        matched_details.append({
            "gold_index": gold_index,
            "predicted_index": predicted_index,
            "similarity": round(score, 4),
        })
    denominator = max(1, true_positive)
    return {
        "gold_count": len(gold_events),
        "predicted_count": len(predicted_events),
        "true_positive": true_positive,
        "false_positive": len(false_positive_indices),
        "false_negative": len(false_negative_indices),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "temporal_exact_accuracy": round(temporal_exact / denominator, 4),
        "temporal_compatible_accuracy": round(
            temporal_compatible / denominator, 4
        ),
        "certainty_accuracy": round(certainty_correct / denominator, 4),
        "status_accuracy": round(status_correct / denominator, 4),
        "citation_completeness": round(citation_complete / denominator, 4),
        "matches": matched_details,
        "false_negative_indices": false_negative_indices,
        "false_positive_indices": false_positive_indices,
    }


def aggregate_patient_metrics(results: list[dict]) -> dict[str, Any]:
    """Micro and macro aggregates; patients remain the split unit."""
    if not results:
        return {"patient_count": 0}
    totals = defaultdict(float)
    for result in results:
        for key in ("gold_count", "predicted_count", "true_positive",
                    "false_positive", "false_negative"):
            totals[key] += result.get(key, 0)
    precision = _safe_div(totals["true_positive"], totals["predicted_count"])
    recall = _safe_div(totals["true_positive"], totals["gold_count"])
    macro_keys = (
        "precision", "recall", "f1", "temporal_exact_accuracy",
        "temporal_compatible_accuracy", "certainty_accuracy",
        "status_accuracy", "citation_completeness",
    )
    return {
        "patient_count": len(results),
        "micro_precision": round(precision, 4),
        "micro_recall": round(recall, 4),
        "micro_f1": round(_safe_div(2 * precision * recall, precision + recall), 4),
        **{
            f"macro_{key}": round(
                sum(result.get(key, 0.0) for result in results) / len(results), 4
            )
            for key in macro_keys
        },
    }

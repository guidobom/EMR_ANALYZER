#!/usr/bin/env python3
"""Compare Percorso B direct events vs Percorso A coded atoms vs reference.

Event-level metrics over the same corpus:

- recall / precision of Percorso A (coded atoms) and Percorso B (direct
  events) against the reference events, using the same semantic match
  (``_is_match``) as the atomic benchmark.
- code agreement: share of gold ``snomed_code`` values present (exact or
  IS-A ancestor/descendant) in the A emitted codes and in the B emitted codes.
- date agreement: for semantically matched pairs, share with equal dates.
- grounding: share of B events carrying at least one exact source passage.

Percorso B is experimental; this comparison is a development diagnostic.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

TOOLS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[1]
for directory in (TOOLS_DIR, PROJECT_ROOT):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from score_atomic_benchmark import EXCLUDED, _is_match, _pair_score
from score_snomed_benchmark import _gold_snomed_events, _hierarchical_match

from emr_analyzer.clinical.snomed import SnomedIndex, load_snapshot


def _a_candidates(payload: dict) -> list[dict]:
    """Coded Percorso A atoms as matchable candidates."""
    return [
        item for item in payload.get("evidence", [])
        if item.get("clinical_relevance") not in EXCLUDED
    ]


def _a_emitted_codes(payload: dict) -> set[str]:
    return {
        str(item.get("terminology_code")).strip()
        for item in payload.get("evidence", [])
        if item.get("terminology_system") == "SNOMED CT"
        and str(item.get("terminology_code") or "").strip()
    }


def _b_candidates(payload: dict) -> list[dict]:
    """Direct events as matchable candidates."""
    candidates = []
    for event in payload.get("direct_events", []):
        passages = event.get("source_passages") or []
        candidates.append({
            "concept": event.get("label"),
            "fact_type": None,  # Percorso B has no per-fact-type buckets
            "polarity": event.get("polarity"),
            "observation_date": event.get("observed_date"),
            "value_text": None,
            "numeric_value": None,
            "unit": None,
            "source_reference": {"passage": passages[0] if passages else ""},
        })
    return candidates


def _b_emitted_codes(payload: dict) -> set[str]:
    return {
        str(event.get("snomed_code") or "").strip()
        for event in payload.get("direct_events", [])
        if str(event.get("snomed_code") or "").strip()
    }


def _match_counts(
    references: list[dict], candidates: list[dict]
) -> tuple[int, set[int], list[tuple[dict, dict]]]:
    matched_items: set[int] = set()
    matched_refs = 0
    pairs: list[tuple[dict, dict]] = []
    for reference in references:
        ranked = sorted(
            (
                (_pair_score(reference, candidate)[0], index, candidate)
                for index, candidate in enumerate(candidates)
            ),
            reverse=True,
        )
        if ranked and _is_match(reference, ranked[0][2])[0]:
            matched_refs += 1
            matched_items.add(ranked[0][1])
            pairs.append((reference, ranked[0][2]))
    return matched_refs, matched_items, pairs


def _code_agreement(gold_codes, emitted: set[str], index) -> dict:
    exact = sum(1 for code in gold_codes if code in emitted)
    hierarchical = sum(1 for code in gold_codes
                       if _hierarchical_match(code, emitted, index))
    total = len(gold_codes)
    if not total:
        return {"gold": 0, "exact": None, "hierarchical": None}
    return {
        "gold": total,
        "exact": exact / total,
        "hierarchical": hierarchical / total,
    }


def _date_agreement(pairs: list[tuple[dict, dict]]) -> float | None:
    agree = 0
    count = 0
    for reference, candidate in pairs:
        ref_date = reference.get("observation_date")
        cand_date = candidate.get("observation_date")
        if not ref_date and not cand_date:
            continue
        count += 1
        if ref_date == cand_date:
            agree += 1
    return agree / count if count else None


def _run_case_metrics(
    case_id: str,
    references: list[dict],
    a_payload: dict,
    b_payload: dict,
    index: SnomedIndex,
) -> dict:
    a_candidates = _a_candidates(a_payload)
    b_candidates = _b_candidates(b_payload)
    a_matched_refs, a_matched_items, _ = _match_counts(references, a_candidates)
    b_matched_refs, b_matched_items, b_pairs = _match_counts(
        references, b_candidates
    )
    gold_snomed = _gold_snomed_events(references)
    gold_codes = {str(e["snomed_code"]).strip() for e in gold_snomed}
    b_grounded = sum(
        1 for event in b_payload.get("direct_events", [])
        if event.get("source_passages")
    )
    b_total = len(b_payload.get("direct_events", []))
    return {
        "case_id": case_id,
        "reference_events": len(references),
        "A": {
            "items": len(a_candidates),
            "recall": a_matched_refs / len(references) if references else None,
            "precision": (
                len(a_matched_items) / len(a_candidates)
                if a_candidates else None
            ),
            "code": _code_agreement(
                gold_codes, _a_emitted_codes(a_payload), index
            ),
        },
        "B": {
            "events": b_total,
            "recall": b_matched_refs / len(references) if references else None,
            "precision": (
                len(b_matched_items) / len(b_candidates)
                if b_candidates else None
            ),
            "code": _code_agreement(
                gold_codes, _b_emitted_codes(b_payload), index
            ),
            "date_agreement": _date_agreement(b_pairs),
            "grounded": b_grounded / b_total if b_total else None,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--release-dir", type=Path, required=True)
    parser.add_argument("--case-ids", nargs="+", required=True)
    parser.add_argument("--a-dir", default="snomed")
    parser.add_argument("--b-dir", default="events")
    args = parser.parse_args()

    release = load_snapshot(args.release_dir)
    index = SnomedIndex(release.concepts.values())
    source = __import__("json").loads(
        (args.root / "sol" / "reference_all.json").read_text(encoding="utf-8")
    )
    by_case = {case["case_id"]: case.get("events", []) for case in source["cases"]}

    cases = []
    for case_id in args.case_ids:
        if case_id not in by_case:
            raise ValueError(f"Case non presente nel gold: {case_id}")
        a_payload = __import__("json").loads(
            (args.root / args.a_dir / f"{case_id}.json").read_text(encoding="utf-8")
        )
        b_payload = __import__("json").loads(
            (args.root / args.b_dir / f"{case_id}.json").read_text(encoding="utf-8")
        )
        cases.append(_run_case_metrics(
            case_id, by_case[case_id], a_payload, b_payload, index,
        ))

    def nested(block: str, metric: str) -> float | None:
        values = [
            case[block][metric] for case in cases
            if case[block][metric] is not None
        ]
        return sum(values) / len(values) if values else None

    def nested_code(block: str, metric: str) -> float | None:
        values = [
            case[block]["code"][metric] for case in cases
            if case[block]["code"][metric] is not None
        ]
        return sum(values) / len(values) if values else None

    totals = {
        "reference_events": sum(case["reference_events"] for case in cases),
        "A_recall": nested("A", "recall"),
        "A_precision": nested("A", "precision"),
        "A_code_exact": nested_code("A", "exact"),
        "A_code_hierarchical": nested_code("A", "hierarchical"),
        "B_recall": nested("B", "recall"),
        "B_precision": nested("B", "precision"),
        "B_code_exact": nested_code("B", "exact"),
        "B_code_hierarchical": nested_code("B", "hierarchical"),
        "B_date_agreement": nested("B", "date_agreement"),
        "B_grounded": nested("B", "grounded"),
    }

    output = {
        "method": "direct_events_comparison",
        "release_dir": str(args.release_dir),
        "release_digest": release.digest,
        "warning": "Development diagnostics; Percorso B is experimental.",
        "totals": totals,
        "cases": cases,
    }
    print(__import__("json").dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

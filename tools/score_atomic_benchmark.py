#!/usr/bin/env python3
"""Score two atomic-evidence runs against the same augmented reference.

This is a reproducible semantic proxy for A/B development.  It deliberately
does not replace clinician adjudication: the reference is augmented only with
baseline atoms previously marked ``valid_additional`` by manual review.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from difflib import SequenceMatcher
import json
from pathlib import Path
import re
import unicodedata


EXCLUDED = {
    "excluded_administrative", "excluded_boilerplate",
    "excluded_methodological",
}
STOP = set(
    "il lo la i gli le un uno una di del dello della dei degli delle da dal "
    "dallo dalla in nel nello nella nei con su per tra fra e ed o a al allo "
    "alla ai agli alle che cui si è era sono come anche non più meno circa "
    "lieve moderata grado paziente".split()
)
TYPE_GROUP = {
    "radiology_finding": "finding", "instrumental_finding": "finding",
    "clinical_sign": "finding", "vital_sign": "finding",
    "diagnosis": "condition", "symptom": "condition",
    "histopathology": "condition", "biomarker": "lab",
    "laboratory_test": "lab", "medication": "medication",
    "procedure": "action", "clinical_decision": "action",
    "hospitalization": "encounter", "discharge": "encounter",
}


def _norm(value) -> str:
    text = unicodedata.normalize("NFKD", str(value or ""))
    text = "".join(char for char in text if not unicodedata.combining(char))
    return re.sub(r"[^a-z0-9]+", " ", text.casefold()).strip()


def _tokens(value) -> list[str]:
    result = []
    for word in _norm(value).split():
        if word in STOP or len(word) < 2:
            continue
        for suffix in (
            "mente", "zioni", "zione", "amento", "amenti", "iche", "ici",
            "ale", "ali", "oso", "osa", "osi", "ato", "ata", "ati", "ate",
        ):
            if len(word) > len(suffix) + 3 and word.endswith(suffix):
                word = word[:-len(suffix)]
                break
        result.append(word)
    return result


def _similarity(left, right) -> float:
    a, b = _norm(left), _norm(right)
    if not a or not b:
        return 0.0
    left_tokens, right_tokens = set(_tokens(a)), set(_tokens(b))
    intersection = len(left_tokens & right_tokens)
    jaccard = intersection / max(1, len(left_tokens | right_tokens))
    coverage = intersection / max(1, min(len(left_tokens), len(right_tokens)))
    sequence = SequenceMatcher(None, a, b).ratio()
    return max(
        jaccard, 0.72 * coverage + 0.28 * sequence,
        sequence if min(len(a), len(b)) > 5 else 0.0,
    )


def _source_coverage(reference, passage) -> float:
    a, b = _norm(reference), _norm(passage)
    if not a or not b:
        return 0.0
    if a in b or b in a:
        return 1.0
    left, right = set(_tokens(a)), set(_tokens(b))
    return len(left & right) / max(1, len(left))


def _values(item: dict) -> str:
    result = " ".join(
        str(item.get(key) or "")
        for key in ("value_text", "numeric_value", "unit")
    )
    return result + " " + json.dumps(
        item.get("typed_payload") or {}, ensure_ascii=False, sort_keys=True
    )


def _pair_score(reference: dict, candidate: dict) -> tuple[float, ...]:
    concept = _similarity(reference.get("concept"), candidate.get("concept"))
    source = _source_coverage(
        reference.get("source_quote"),
        (candidate.get("source_reference") or {}).get("passage"),
    )
    value = _similarity(
        " ".join(
            str(reference.get(key) or "")
            for key in ("value_text", "numeric_value", "unit")
        ),
        _values(candidate),
    )
    reference_type = reference.get("fact_type")
    candidate_type = candidate.get("fact_type")
    score = 0.57 * concept + 0.33 * source + 0.10 * value
    if reference_type == candidate_type:
        score += 0.08
    elif TYPE_GROUP.get(reference_type) == TYPE_GROUP.get(candidate_type):
        score -= 0.02
    else:
        score -= 0.18
    if (
        reference.get("polarity") != candidate.get("polarity")
        and reference.get("polarity") in {"present", "negated"}
        and candidate.get("polarity") in {"present", "negated"}
    ):
        score -= 0.12
    return score, concept, source, value


def _is_match(reference: dict, candidate: dict) -> tuple[bool, float]:
    score, concept, source, _ = _pair_score(reference, candidate)
    return score >= 0.48 or (concept >= 0.52 and source >= 0.35), score


def _load_augmented_reference(root: Path) -> dict[str, list[dict]]:
    source = json.loads((root / "sol" / "reference_all.json").read_text())
    by_case = {case["case_id"]: list(case["events"]) for case in source["cases"]}
    baseline_cache: dict[str, list[dict]] = {}
    for path in sorted((root / "judgments").glob("judgment_batch_*.json")):
        judgment = json.loads(path.read_text())
        for case in judgment["cases"]:
            case_id = case["case_id"]
            baseline = baseline_cache.setdefault(
                case_id,
                json.loads((root / "qwen" / f"{case_id}.json").read_text())[
                    "evidence"
                ],
            )
            for verdict in case["qwen_verdicts"]:
                if verdict["verdict"] != "valid_additional":
                    continue
                atom = baseline[verdict["qwen_index"]]
                reference = {
                    "concept": atom.get("concept"),
                    "fact_type": atom.get("fact_type"),
                    "polarity": atom.get("polarity"),
                    "observation_date": atom.get("observation_date"),
                    "value_text": atom.get("value_text"),
                    "numeric_value": atom.get("numeric_value"),
                    "unit": atom.get("unit"),
                    "source_quote": (
                        atom.get("source_reference") or {}
                    ).get("passage"),
                }
                if not any(
                    _is_match(reference, current)[0]
                    for current in by_case[case_id]
                ):
                    by_case[case_id].append(reference)
    return by_case


def _duplicate_pairs(items: list[dict]) -> int:
    duplicates = 0
    representatives: list[dict] = []
    for item in items:
        duplicate = False
        for prior in representatives:
            same_group = (
                item.get("fact_type") == prior.get("fact_type")
                or TYPE_GROUP.get(item.get("fact_type"))
                == TYPE_GROUP.get(prior.get("fact_type"))
            )
            dates_compatible = (
                not item.get("observation_date")
                or not prior.get("observation_date")
                or item.get("observation_date") == prior.get("observation_date")
            )
            concept = _similarity(item.get("concept"), prior.get("concept"))
            source = _similarity(
                (item.get("source_reference") or {}).get("passage"),
                (prior.get("source_reference") or {}).get("passage"),
            )
            if same_group and dates_compatible and (
                concept >= 0.90 or 0.72 * concept + 0.28 * source >= 0.82
            ):
                duplicate = True
                break
        if duplicate:
            duplicates += 1
        else:
            representatives.append(item)
    return duplicates


def score_run(
    root: Path, result_dir: str, references: dict[str, list[dict]],
    case_ids: list[str],
) -> dict:
    aggregate = Counter()
    by_type = defaultdict(Counter)
    cases = []
    for case_id in case_ids:
        payload = json.loads(
            (root / result_dir / f"{case_id}.json").read_text()
        )
        items = [
            item for item in payload["evidence"]
            if item.get("clinical_relevance") not in EXCLUDED
        ]
        refs = references[case_id]
        matched_items: set[int] = set()
        matched_refs = 0
        missing_reference = []
        for reference in refs:
            ranked = sorted(
                (
                    (_pair_score(reference, item)[0], index, item)
                    for index, item in enumerate(items)
                ),
                reverse=True,
            )
            if ranked and _is_match(reference, ranked[0][2])[0]:
                matched_refs += 1
                matched_items.add(ranked[0][1])
                by_type[reference.get("fact_type")]["matched"] += 1
            else:
                missing_reference.append({
                    "fact_type": reference.get("fact_type"),
                    "concept": reference.get("concept"),
                })
            by_type[reference.get("fact_type")]["reference"] += 1
        duplicate_count = _duplicate_pairs(items)
        aggregate.update({
            "reference": len(refs), "matched_reference": matched_refs,
            "items": len(items), "matched_items": len(matched_items),
            "semantic_duplicates": duplicate_count,
            "llm_calls": int(payload.get("metrics", {}).get("llm_calls", 0)),
            "prompt_tokens": int(
                payload.get("metrics", {}).get("prompt_tokens", 0)
            ),
            "completion_tokens": int(
                payload.get("metrics", {}).get("completion_tokens", 0)
            ),
        })
        cases.append({
            "case_id": case_id, "reference": len(refs),
            "matched_reference": matched_refs, "items": len(items),
            "matched_items": len(matched_items),
            "semantic_duplicates": duplicate_count,
            "llm_calls": payload.get("metrics", {}).get("llm_calls", 0),
            "missing_reference": missing_reference,
        })
    recall = aggregate["matched_reference"] / max(1, aggregate["reference"])
    # Conservative lower bound: new valid additions cannot be adjudicated by
    # an automatic reference matcher, so unmatched items are not called false.
    unique_items = aggregate["items"] - aggregate["semantic_duplicates"]
    precision_floor = aggregate["matched_items"] / max(1, unique_items)
    return {
        "result_dir": result_dir,
        "cases": cases,
        "aggregate": dict(aggregate),
        "reference_recall_proxy": recall,
        "matched_item_precision_floor": precision_floor,
        "by_reference_fact_type": {
            fact_type: {
                **dict(counts),
                "recall": counts["matched"] / max(1, counts["reference"]),
            }
            for fact_type, counts in sorted(by_type.items())
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--case-ids", nargs="+", required=True)
    args = parser.parse_args()
    references = _load_augmented_reference(args.root)
    output = {
        "method": "semantic_proxy_against_manually_augmented_reference",
        "warning": "Development A/B; not a substitute for clinician review.",
        "runs": [
            score_run(args.root, run, references, args.case_ids)
            for run in args.runs
        ],
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

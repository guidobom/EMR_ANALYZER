#!/usr/bin/env python3
"""Score the SNOMED-constrained run against the user's SNOMED gold annotation.

The gold SNOMED codes are the user's manual annotation (external, in
``sol/reference_all.json``: each event may carry ``snomed_code``).  Metrics:

- ``retrieval_recall`` — share of gold codes present in the run's per-case
  union of retrieved candidate codes.  This is the *ceiling* for constrained
  generation: a code the retriever never recalls can never be emitted.
- ``code_exact`` — share of gold codes emitted verbatim.
- ``code_hierarchical`` — share of gold codes emitted verbatim or as an
  IS-A ancestor/descendant of the emitted code.
- ``code_mismatch`` — share of gold codes with no hierarchical match.
- ``snomed_coverage`` — share of gold SNOMED events the run captured as a
  coded atom (semantic match, independent of code identity).
- ``semantic_proxy`` — the existing atom-level A/B proxies (same reference)
  for a comparable baseline-vs-SNOMED picture.

The release index is used only for IS-A navigation (hierarchical code
matching); no release data is written anywhere.
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

from score_atomic_benchmark import _load_augmented_reference, _is_match, score_run

from emr_analyzer.clinical.snomed import SnomedIndex, load_gps, load_snapshot


def _gold_snomed_events(case_events: list[dict]) -> list[dict]:
    return [
        event for event in case_events
        if str(event.get("snomed_code") or "").strip()
    ]


def _hierarchical_match(
    gold_code: str, emitted_codes: set[str], index: SnomedIndex
) -> bool:
    if gold_code in emitted_codes:
        return True
    neighbours = index.ancestors(gold_code) | index.descendants(gold_code)
    return bool(emitted_codes & neighbours)


def _case_metrics(
    case_id: str,
    gold_snomed: list[dict],
    run_payload: dict,
    index: SnomedIndex,
) -> dict:
    gold_codes = {
        str(event["snomed_code"]).strip() for event in gold_snomed
    }
    emitted_codes = {
        str(item.get("terminology_code")).strip()
        for item in run_payload.get("evidence", [])
        if item.get("terminology_system") == "SNOMED CT"
        and str(item.get("terminology_code") or "").strip()
    }
    retrieved_codes = set(run_payload.get("snomed", {}).get(
        "retrieved_codes", ()
    ))
    run_items = [
        item for item in run_payload.get("evidence", [])
        if item.get("clinical_relevance") not in {
            "excluded_administrative", "excluded_boilerplate",
            "excluded_methodological",
        }
    ]
    coded_items = [
        item for item in run_items
        if item.get("terminology_system") == "SNOMED CT"
        and str(item.get("terminology_code") or "").strip()
    ]
    if not gold_codes:
        return {
            "case_id": case_id,
            "gold_snomed": 0,
            "note": "nessun evento gold con snomed_code",
            "emitted_codes": sorted(emitted_codes),
            "retrieved_codes": sorted(retrieved_codes),
            "retrieval_recall": None,
            "code_exact": None,
            "code_hierarchical": None,
            "code_mismatch": None,
            "snomed_coverage": None,
        }

    code_exact_hits = sum(
        1 for code in gold_codes if code in emitted_codes
    )
    hierarchical_hits = sum(
        1 for code in gold_codes if _hierarchical_match(code, emitted_codes, index)
    )
    retrieval_hits = sum(
        1 for code in gold_codes if code in retrieved_codes
    )
    coverage_hits = 0
    for reference in gold_snomed:
        ref = {
            "concept": reference.get("concept"),
            "fact_type": reference.get("fact_type"),
            "polarity": reference.get("polarity"),
            "observation_date": reference.get("observation_date"),
            "value_text": reference.get("value_text"),
            "numeric_value": reference.get("numeric_value"),
            "unit": reference.get("unit"),
            "source_quote": reference.get("source_quote"),
        }
        if any(_is_match(ref, item)[0] for item in coded_items):
            coverage_hits += 1
    total = len(gold_codes)
    return {
        "case_id": case_id,
        "gold_snomed": total,
        "retrieval_recall": retrieval_hits / total,
        "code_exact": code_exact_hits / total,
        "code_hierarchical": hierarchical_hits / total,
        "code_mismatch": (total - hierarchical_hits) / total,
        "snomed_coverage": coverage_hits / total,
        "retrieval_hits": retrieval_hits,
        "emitted_codes": sorted(emitted_codes),
        "retrieved_codes": sorted(retrieved_codes),
        "missing_codes": sorted(gold_codes - emitted_codes),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--release-dir", type=Path, required=True)
    parser.add_argument("--case-ids", nargs="+", required=True)
    parser.add_argument("--run-dir", default="snomed")
    parser.add_argument("--runs", nargs="+", default=("qwen", "snomed"))
    parser.add_argument(
        "--gps", action="store_true",
        help="Source is the GPS freeset (interim CC BY-ND), not an RF2 release",
    )
    args = parser.parse_args()

    # ``_hierarchical_match`` needs the IS-A graph, which GPS lacks (the metric
    # degrades to exact matching); load via the same kind as the run.
    release = (
        load_gps(args.release_dir)
        if args.gps else load_snapshot(args.release_dir)
    )
    index = SnomedIndex(release.concepts.values())

    reference_path = args.root / "sol" / "reference_all.json"
    if not reference_path.exists():
        raise FileNotFoundError(
            f"Gold SNOMED mancante: {reference_path} "
            "(annotazione utente esterna richiesta)"
        )
    source = __import__("json").loads(reference_path.read_text(encoding="utf-8"))
    by_case = {case["case_id"]: case.get("events", []) for case in source["cases"]}

    references = _load_augmented_reference(args.root)
    cases = []
    for case_id in args.case_ids:
        if case_id not in by_case:
            raise ValueError(f"Case non presente nel gold: {case_id}")
        payload = __import__("json").loads(
            (args.root / args.run_dir / f"{case_id}.json").read_text(
                encoding="utf-8"
            )
        )
        gold_snomed = _gold_snomed_events(by_case[case_id])
        cases.append(_case_metrics(
            case_id, gold_snomed, payload, index
        ))

    totals = {
        "gold_snomed": sum(
            case["gold_snomed"] for case in cases
        ),
    }
    for metric in (
        "retrieval_recall", "code_exact", "code_hierarchical",
        "code_mismatch", "snomed_coverage",
    ):
        values = [case[metric] for case in cases if case[metric] is not None]
        totals[metric] = (
            sum(values) / len(values) if values else None
        )

    output = {
        "method": "snomed_gold_against_manual_annotation",
        "release_dir": str(args.release_dir),
        "release_digest": release.digest,
        "warning": "Development diagnostics; not a substitute for clinician review.",
        "totals": totals,
        "cases": cases,
        "semantic_proxy": [
            score_run(args.root, run, references, args.case_ids)
            for run in args.runs
        ],
    }
    print(__import__("json").dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

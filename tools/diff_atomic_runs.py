#!/usr/bin/env python3
"""Diff atomico tra due run del benchmark (stessi casi, stessi testi).

Confronta gli atomi estratti da due run e produce l'elenco di ciò che un
run trova in più o in meno rispetto all'altro, ordinato per tipo di fatto.
Serve alla revisione clinica delle differenze (non sostituisce il
riferimento annotato).

Usage::

    python tools/diff_atomic_runs.py --root /tmp/atomic_bench_20260905 \
        --runs baseline_qwen3_4b qwen3_14b [--case-ids CASE_001 ...]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _norm(value) -> str:
    text = unicodedata.normalize("NFKD", str(value or ""))
    text = "".join(char for char in text if not unicodedata.combining(char))
    return re.sub(r"[^a-z0-9]+", " ", text.casefold()).strip()


def _atom_key(item: dict) -> tuple:
    concept = _norm(item.get("concept") or item.get("normalized_entity"))
    value = _norm(item.get("value_text"))
    numeric = item.get("numeric_value")
    date = item.get("observation_date") or ""
    return (
        _norm(item.get("fact_type") or item.get("category")),
        concept,
        date,
        value,
        numeric,
    )


def _load(run_dir: Path, case_id: str) -> list[dict]:
    path = run_dir / f"{case_id}.json"
    if not path.is_file():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "ok":
        return []
    return payload.get("evidence") or []


def _load_payload(run_dir: Path, case_id: str) -> dict:
    path = run_dir / f"{case_id}.json"
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


_METRIC_FIELDS = (
    "llm_calls", "prompt_tokens", "completion_tokens", "total_tokens",
    "validation_retries", "coverage_retries", "invalid_items",
    "sentence_calls",
)


def _metrics_report(root: Path, run_a: Path, run_b: Path, cases: list[str]):
    """Compare per-case cost/latency totals between the two runs."""
    def aggregate(run_dir: Path):
        totals: Counter = Counter()
        elapsed = 0.0
        ok_cases = 0
        evidence_total = 0
        for case_id in cases:
            payload = _load_payload(run_dir, case_id)
            if payload.get("status") != "ok":
                continue
            ok_cases += 1
            elapsed += float(payload.get("elapsed_seconds") or 0.0)
            evidence_total += int(payload.get("llm_evidence_count") or 0)
            metrics = payload.get("metrics") or {}
            for field in _METRIC_FIELDS:
                value = metrics.get(field)
                if isinstance(value, (int, float)) and not isinstance(
                    value, bool
                ):
                    totals[field] += value
        return ok_cases, elapsed, evidence_total, totals

    ok_a, elapsed_a, evidence_a, totals_a = aggregate(run_a)
    ok_b, elapsed_b, evidence_b, totals_b = aggregate(run_b)
    print(f"\nMetriche {run_a.name} (A) vs {run_b.name} (B) "
          f"su {min(ok_a, ok_b)} casi completati\n")
    print(f"  casi ok:            A={ok_a}  B={ok_b}")
    print(f"  atomi LLM totali:   A={evidence_a}  B={evidence_b}")
    print(f"  elapsed_seconds:    A={elapsed_a:,.0f}  B={elapsed_b:,.0f}")
    for field in _METRIC_FIELDS:
        print(f"  {field:<20} A={totals_a.get(field, 0):,}  "
              f"B={totals_b.get(field, 0):,}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--runs", nargs=2, required=True)
    parser.add_argument("--case-ids", nargs="*")
    parser.add_argument("--limit", type=int, default=40,
                        help="Massimo di righe di diff mostrate per direzione")
    parser.add_argument("--metrics", action="store_true",
                        help="Confronta anche latenza e costi per caso "
                             "(elapsed, chiamate, token) tra i due run")
    args = parser.parse_args()

    manifest = json.loads(
        (args.root / "manifest.json").read_text(encoding="utf-8")
    )
    cases = [
        case["case_id"] for case in manifest
        if not args.case_ids or case["case_id"] in args.case_ids
    ]
    run_a_dir = args.root / args.runs[0]
    run_b_dir = args.root / args.runs[1]

    only_a: Counter = Counter()
    only_b: Counter = Counter()
    only_a_examples: dict[str, list[tuple]] = defaultdict(list)
    only_b_examples: dict[str, list[tuple]] = defaultdict(list)
    for case_id in cases:
        items_a = _load(run_a_dir, case_id)
        items_b = _load(run_b_dir, case_id)
        keys_a = {_atom_key(item) for item in items_a}
        keys_b = {_atom_key(item) for item in items_b}
        for item in items_a:
            if _atom_key(item) not in keys_b:
                label = item.get("fact_type") or item.get("category") or "?"
                only_a[label] += 1
                if len(only_a_examples[label]) < 5:
                    only_a_examples[label].append((
                        case_id,
                        item.get("concept") or item.get("normalized_entity"),
                        item.get("observation_date") or "-",
                        item.get("value_text") or item.get("numeric_value")
                        or "-",
                    ))
        for item in items_b:
            if _atom_key(item) not in keys_a:
                label = item.get("fact_type") or item.get("category") or "?"
                only_b[label] += 1
                if len(only_b_examples[label]) < 5:
                    only_b_examples[label].append((
                        case_id,
                        item.get("concept") or item.get("normalized_entity"),
                        item.get("observation_date") or "-",
                        item.get("value_text") or item.get("numeric_value")
                        or "-",
                    ))

    print(f"Diff atomico: {args.runs[0]} (A) vs {args.runs[1]} (B) "
          f"su {len(cases)} casi\n")
    print(f"Solo in A ({args.runs[0]}): {sum(only_a.values())}")
    for label, count in only_a.most_common():
        print(f"  {label}: {count}")
        for case_id, concept, date, value in only_a_examples[label]:
            print(f"    {case_id} | {concept} | {date} | {value}")
    print(f"\nSolo in B ({args.runs[1]}): {sum(only_b.values())}")
    for label, count in only_b.most_common():
        print(f"  {label}: {count}")
        for case_id, concept, date, value in only_b_examples[label]:
            print(f"    {case_id} | {concept} | {date} | {value}")
    if args.metrics:
        _metrics_report(args.root, run_a_dir, run_b_dir, cases)


if __name__ == "__main__":
    main()

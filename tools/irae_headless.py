#!/usr/bin/env python3
"""Headless immune-related adverse event (irAE) analysis for one patient.

Three layers, all fed by the atomic evidence registry (``clinical_evidence``)
of a project:

1. Temporal anchoring — the first dated checkpoint-inhibitor dose.
2. Organ toxicity lexicon — deterministic scan of the atomic evidence rows,
   each flagged evidence bucketed by latency from the anchor.
3. LLM hypothesis — one structured call per organ (constrained JSON decoding,
   NCI CTCAE 6.0 as the single reference) over the pre-filtered candidates
   (citable by ``[#id]``), returning for each irAE: type, CTCAE grade, date of
   first onset, and the probability of an immune-related origin.  The
   immunotherapy start date comes from Layer 1.

The deterministic layers never touch a model; the LLM layer works on the
atomic evidence subset, never on the raw documents.  Layers 1-3 are
implemented in ``emr_analyzer.clinical.irae_layers`` (shared with the GUI
batch queue); this tool is a thin CLI around them.

Usage::

    python tools/irae_headless.py --patient P068 --project MELANOMA
    python tools/irae_headless.py --patient P068 --no-llm          # deterministic only
    python tools/irae_headless.py --patient P068 --max-candidates 60
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from emr_analyzer.clinical import irae_layers
from emr_analyzer.clinical.irae_prototype import (
    find_ici_anchor,
    scan_for_irae,
    summarize_candidates,
)
from emr_analyzer.extraction.llm_client import LlmClient
from emr_analyzer.settings import load_llm_configs

DEFAULT_PROJECTS_ROOT = Path("/home/utente/Desktop")


def load_evidence(project_dir: Path, patient_id: str) -> list[dict[str, Any]]:
    """Atomic evidence rows for one patient from the project registry."""
    db_path = project_dir / "emr_registry.db"
    if not db_path.exists():
        raise SystemExit(f"Registro non trovato: {db_path}")
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    try:
        rows = con.execute(
            "SELECT evidence_id, category, normalized_entity, observed_date, "
            "       data_json, document_id, value_text, numeric_value, unit "
            "FROM clinical_evidence WHERE patient_id = ? "
            "ORDER BY observed_date, id",
            (patient_id,),
        ).fetchall()
    finally:
        con.close()
    return [dict(row) for row in rows]


def _load_data_json(rows: list[dict[str, Any]]) -> None:
    for row in rows:
        raw = row.get("data_json")
        if isinstance(raw, str) and raw.strip():
            try:
                row["data_json"] = json.loads(raw)
            except json.JSONDecodeError:
                row["data_json"] = None
        else:
            row["data_json"] = None


def _anchor_dict(anchor) -> dict | None:
    if anchor is None:
        return None
    return {
        "first_drug": anchor.first_drug,
        "first_date": anchor.first_date.isoformat(),
        "first_raw": anchor.first_raw,
        "last_drug": anchor.last_drug,
        "last_date": anchor.last_date.isoformat(),
        "last_raw": anchor.last_raw,
        "occurrences": anchor.occurrences,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--patient", default="P068", help="Patient pseudonym.")
    parser.add_argument("--project", default="MELANOMA", help="Project folder.")
    parser.add_argument(
        "--projects-root",
        type=Path,
        default=DEFAULT_PROJECTS_ROOT,
        help="Directory containing the project folders (default: Desktop).",
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Run only the deterministic layers (no model load).",
    )
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=irae_layers.DEFAULT_MAX_CANDIDATES,
        help="Prompt bound per organ for Layer 3.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=irae_layers.DEFAULT_MAX_TOKENS,
        help="Output token bound per organ call.",
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=irae_layers.DEFAULT_PARALLEL,
        help="Organ calls in parallel (server slots).",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Write the deterministic summary + LLM report to a JSON file.",
    )
    args = parser.parse_args(argv)

    started = time.perf_counter()

    project_dir = args.projects_root / args.project
    rows = load_evidence(project_dir, args.patient)
    _load_data_json(rows)
    print(f"[evidence] {args.project}/{args.patient}: {len(rows)} evidenze atomiche")

    # ---- Layer 1 + Layer 2 (deterministic) --------------------------------
    anchor = find_ici_anchor(rows)
    if anchor is not None:
        print(
            f"[layer1] ancora: {anchor.first_drug} prima dose "
            f"{anchor.first_raw} ({anchor.first_date.isoformat()}), "
            f"ultima {anchor.last_drug} {anchor.last_raw} "
            f"({anchor.last_date.isoformat()}), {anchor.occurrences} occorrenze"
        )
    else:
        print("[layer1] ancora: NON DETERMINATA (nessun farmaco datato)")

    candidates = scan_for_irae(rows, anchor)
    print(f"[layer2] candidati: {len(candidates)}")
    for summary in summarize_candidates(candidates):
        bands = ", ".join(f"{k}:{v}" for k, v in summary["bands"].items())
        print(f"    {summary['organ']:25s} n={summary['count']:4d}  {bands}")

    if args.no_llm:
        print("[layer3] saltato (--no-llm)")
        if args.output_json:
            _write_json(args.output_json, {
                "patient": args.patient,
                "project": args.project,
                "anchor": _anchor_dict(anchor),
                "immunotherapy_start": (
                    {
                        "date": anchor.first_date.isoformat(),
                        "raw": anchor.first_raw,
                        "drug": anchor.first_drug,
                    }
                    if anchor is not None
                    else None
                ),
                "candidates_total": len(candidates),
                "candidates_by_organ": summarize_candidates(candidates),
                "timing_seconds": {},
            })
        return 0

    # ---- Layer 3 (structured per-organ LLM calls, shared module) ----------
    configs = load_llm_configs()
    state_config = configs["clinical_state"]
    print(
        f"[layer3] modello: {state_config.model} · {state_config.backend} · "
        f"ctx {state_config.context_length} · workers {state_config.parallel_workers}"
    )
    client = LlmClient(config=state_config)
    if not client.is_available:
        print(
            f"[layer3] ERRORE: modello {state_config.model} non presente "
            "nell'archivio locale"
        )
        return 2
    if not client.server_available:
        print("[layer3] ERRORE: backend llama.cpp non disponibile")
        return 2

    t_warm_start = time.perf_counter()
    print("[layer3] caricamento modello e warmup…", flush=True)
    runtime = client.warmup()
    warm_seconds = time.perf_counter() - t_warm_start
    print(
        f"[layer3] warmup: {runtime.get('context_length')} ctx, "
        f"{runtime.get('test_response', '')!r} in {warm_seconds:.1f}s"
    )

    report = irae_layers.analyze_irae(
        rows,
        client,
        max_candidates=args.max_candidates,
        max_tokens=args.max_tokens,
        parallel=args.parallel,
    )
    report["patient"] = args.patient
    report["project"] = args.project
    report["timing_seconds"]["warmup"] = round(warm_seconds, 1)

    total_seconds = report["timing_seconds"]["total"]
    print("\n" + "=" * 78)
    print(f"IPOTESI IRae — {args.project}/{args.patient} "
          f"({total_seconds:.1f}s totali)")
    print("=" * 78)
    if anchor is not None:
        print(
            f"INIZIO IMMUNOTERAPIA: {anchor.first_drug} dal "
            f"{anchor.first_raw} ({anchor.first_date.isoformat()})"
        )
    all_iraes: list[dict] = []
    for organ in [s["organ"] for s in report["candidates_by_organ"]]:
        result = report["organ_results"].get(organ, {})
        print(f"\n## {organ}")
        if result.get("error"):
            print(f"    [ERRORE] {result['error']}")
            continue
        for item in result.get("iraes", []):
            all_iraes.append(item)
            grade = item.get("ctcae_grade", "?")
            prob = item.get("probability_immune", "?")
            onset = item.get("first_onset_date", "?")
            ev = ",".join(item.get("key_evidence_ids", [])[:5])
            print(
                f"  • {item.get('irAE_type', '?')} · {grade} · "
                f"insorgenza {onset} · {prob}"
            )
            if item.get("new_onset_vs_exacerbation"):
                print(f"      {item.get('new_onset_vs_exacerbation')}")
            if item.get("alternative_causes"):
                print(f"      cause alt.: {item.get('alternative_causes')}")
            if ev:
                print(f"      evidenze: [{ev}]")
            conf = item.get("confidence")
            if conf is not None:
                print(f"      confidenza: {conf}")
    print("\n" + "=" * 78)

    if args.output_json:
        _write_json(args.output_json, report)
    return 0


def _write_json(path: Path, report: dict) -> None:
    path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[report] JSON salvato in {path}")


if __name__ == "__main__":
    sys.exit(main())

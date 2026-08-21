"""Synthetic, non-clinical benchmark for consolidation throughput."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from emr_analyzer.clinical.consolidation import ClinicalConsolidator
from emr_analyzer.models.clinical_evidence import ClinicalEvidence


def main() -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--documents", type=int, default=200)
    parser.add_argument("--evidence-per-document", type=int, default=20)
    parser.add_argument(
        "--evidence", type=int,
        help="Numero totale di evidenze (sovrascrive --evidence-per-document)",
    )
    args = parser.parse_args()
    if args.documents <= 0 or args.evidence_per_document <= 0:
        parser.error("documents ed evidence-per-document devono essere positivi")
    if args.evidence is not None and args.evidence <= 0:
        parser.error("evidence deve essere positivo")
    total_evidence = (
        args.evidence
        if args.evidence is not None
        else args.documents * args.evidence_per_document
    )
    evidence = []
    per_document_counters = [0] * args.documents
    for global_index in range(total_evidence):
        document_index = global_index % args.documents
        evidence_index = per_document_counters[document_index]
        per_document_counters[document_index] += 1
        entity_index = global_index % 10
        evidence.append(ClinicalEvidence(
            evidence_id=f"E_{document_index}_{evidence_index}",
            patient_id="P_BENCH", document_id=f"D_{document_index}",
            category="symptom", normalized_entity=f"sintomo_{entity_index}",
            source_text=f"Sintomo sintetico {entity_index}",
            observed_date=f"2025-01-{document_index % 28 + 1:02d}",
            document_date=f"2025-01-{document_index % 28 + 1:02d}",
        ))
    started = time.perf_counter()
    bundles = ClinicalConsolidator().consolidate("P_BENCH", evidence)
    elapsed = time.perf_counter() - started
    print(json.dumps({
        "documents": args.documents,
        "evidence": len(evidence),
        "events": len(bundles),
        "elapsed_seconds": round(elapsed, 4),
        "evidence_per_second": round(len(evidence) / max(elapsed, 1e-9), 1),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

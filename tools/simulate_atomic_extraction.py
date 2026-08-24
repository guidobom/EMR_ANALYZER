"""Run a small PHI-free end-to-end check of atomic clinical extraction."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from emr_analyzer.clinical.atomic_evidence import AtomicEvidenceExtractor
from emr_analyzer.extraction.llm_client import LlmClient
from emr_analyzer.settings import LLMRoleConfig


SYNTHETIC_NOTE = """10/03/2025: il paziente riferisce tosse secca e dispnea da sforzo
insorte da circa 5 giorni. SpO2 88% in aria ambiente.
12/03/2025 TC torace: nuove opacità bilaterali a vetro smerigliato.
Valutazione pneumologica del 13/03/2025: quadro compatibile con polmonite
immuno-mediata di grado 2 durante terapia con nivolumab. Nivolumab sospeso;
iniziato prednisone 50 mg/die per os.
Al controllo del 25/03/2025 la dispnea è risolta, SpO2 96% in aria ambiente;
prednisone ridotto a 25 mg/die.
"""


def main() -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--model", default="qwen3-14b")
    parser.add_argument("--context", type=int, default=8192)
    parser.add_argument("--max-output", type=int, default=2048)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--speculative", action="store_true")
    parser.add_argument("--input", type=Path)
    parser.add_argument("--document-type", default="visita_specialistica")
    parser.add_argument("--date", default="2025-03-25")
    parser.add_argument("--show-source", action="store_true")
    args = parser.parse_args()

    source_text = (
        args.input.read_text(encoding="utf-8")
        if args.input is not None else SYNTHETIC_NOTE
    )
    if args.show_source:
        print("--- TESTO SORGENTE ---", file=sys.stderr, flush=True)
        print(source_text, file=sys.stderr, flush=True)
        print("--- ESTRAZIONE ---", file=sys.stderr, flush=True)

    config = LLMRoleConfig(
        model=args.model,
        temperature=0.0,
        context_length=args.context,
        max_output_tokens=args.max_output,
        top_p=0.9,
        top_k=40,
        seed=42,
        parallel_workers=max(1, args.workers),
        speculative_decoding=args.speculative,
    )
    client = LlmClient(config=config)
    extractor = AtomicEvidenceExtractor(client)
    started = time.perf_counter()
    try:
        evidence = extractor.extract_document(
            patient_id="P_SYNTHETIC",
            document_id="D_SYNTHETIC",
            document_type=args.document_type,
            document_date=args.date,
            text=source_text,
            chunk_progress_callback=lambda done, total: print(
                f"Segmento {done}/{total} completato",
                file=sys.stderr, flush=True,
            ),
        )
        payload = {
            "model": args.model,
            "speculative": args.speculative,
            "elapsed_seconds": round(time.perf_counter() - started, 3),
            "metrics": extractor.last_extraction_metrics(),
            "evidence_count": len(evidence),
            "evidence": [asdict(item) for item in evidence],
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0
    finally:
        client.backend.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())

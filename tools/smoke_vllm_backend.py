#!/usr/bin/env python
"""Non-clinical end-to-end smoke test for the app-managed vLLM backend."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from emr_analyzer.extraction.llm_client import LlmClient
from emr_analyzer.llm_backend import shutdown_all_backends
from emr_analyzer.settings import LLMRoleConfig


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-4B")
    parser.add_argument("--context", type=int, default=4096)
    parser.add_argument("--gpu-memory", type=float, default=0.35)
    parser.add_argument("--enforce-eager", action="store_true")
    args = parser.parse_args()
    config = LLMRoleConfig(
        model=args.model,
        backend="vllm",
        temperature=0.0,
        context_length=args.context,
        max_output_tokens=128,
        parallel_workers=1,
        vllm_dtype="bfloat16",
        vllm_gpu_memory_utilization=args.gpu_memory,
        vllm_tensor_parallel_size=1,
        vllm_enforce_eager=args.enforce_eager,
    )
    client = LlmClient(config=config)
    if not client.is_available:
        print(f"Modello locale assente o incompleto: {args.model}")
        return 2
    try:
        print("Avvio e warm-up vLLM…", flush=True)
        runtime = client.warmup()
        print(
            f"Warm-up OK in {runtime['elapsed_seconds']:.1f}s; "
            f"risposta={runtime['test_response']!r}",
            flush=True,
        )
        schema = {
            "type": "object",
            "properties": {
                "status": {"type": "string", "enum": ["ok"]},
                "value": {"type": "integer"},
            },
            "required": ["status", "value"],
            "additionalProperties": False,
        }
        result = client.generate_structured(
            "Test tecnico senza dati clinici: restituisci status ok e value 7.",
            "Rispondi esclusivamente secondo lo schema JSON richiesto.",
            schema,
            max_tokens=64,
        )
        if result.get("status") != "ok" or result.get("value") != 7:
            raise RuntimeError(f"JSON semanticamente inatteso: {result!r}")
        print(f"JSON Schema OK: {result}", flush=True)
        return 0
    finally:
        shutdown_all_backends()
        print("Processo vLLM di test arrestato e memoria rilasciata.", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())

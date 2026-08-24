#!/usr/bin/env python
"""Diagnose or explicitly install the optional vLLM backend.

The default invocation is read-only.  ``--install`` runs the official
``python -m pip install vllm`` command in the current Python environment.
Model downloads are intentionally outside this script: EMR Analyzer only
serves an explicit local directory or a repository already present in the
Hugging Face cache.
"""

from __future__ import annotations

import argparse
import platform
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from emr_analyzer.llm_backend.diagnostics import diagnose_vllm_acceleration
from emr_analyzer.llm_backend.vllm_backend import (
    list_cached_vllm_models,
    resolve_vllm_model,
)

VLLM_ENV_DIR = Path.home() / ".emr_analyzer" / "vllm-env"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Configura il backend vLLM opzionale per EMR Analyzer."
    )
    parser.add_argument(
        "--install",
        action="store_true",
        help="Installa/aggiorna vLLM nell'ambiente Python corrente.",
    )
    parser.add_argument(
        "--model",
        default="",
        help="Verifica un ID Hugging Face già in cache o una directory locale.",
    )
    parser.add_argument(
        "--download-model",
        action="store_true",
        help=(
            "Scarica esplicitamente --model nella cache Hugging Face; "
            "non viene mai eseguito durante l'inferenza."
        ),
    )
    return parser


def _install() -> bool:
    print(f"Interprete di riferimento: {sys.executable}")
    adjacent_uv = Path(sys.executable).with_name("uv")
    uv = shutil.which("uv") or (
        str(adjacent_uv) if adjacent_uv.is_file() else None
    )
    if not uv:
        print(
            "uv non trovato. La documentazione vLLM corrente lo raccomanda "
            "per selezionare correttamente PyTorch/CUDA, soprattutto su "
            "ARM64/Blackwell. Installa uv e riesegui --install; non viene "
            "tentata una compilazione sorgente implicita."
        )
        return False
    runtime_python = VLLM_ENV_DIR / "bin" / "python"
    if runtime_python.is_file():
        print(f"Riutilizzo l'ambiente isolato: {VLLM_ENV_DIR}")
    else:
        print(f"Creo l'ambiente isolato: {VLLM_ENV_DIR}")
        create = subprocess.run(
            [
                uv, "venv", str(VLLM_ENV_DIR), "--python", sys.executable,
                "--seed",
            ],
            text=True,
        )
        if create.returncode != 0:
            return False
    print("Installazione di vLLM con selezione automatica CUDA…")
    result = subprocess.run(
        [
            uv, "pip", "install", "--python", str(runtime_python),
            "--upgrade", "vllm", "--torch-backend=auto",
        ],
        text=True,
    )
    return result.returncode == 0


def _download_model(model: str) -> bool:
    managed_hf = VLLM_ENV_DIR / "bin" / "hf"
    hf = str(managed_hf) if managed_hf.is_file() else shutil.which("hf")
    if not hf:
        print("Comando hf non trovato: installa prima vLLM con --install.")
        return False
    print(f"Download esplicito del modello {model}…")
    result = subprocess.run([hf, "download", model], text=True)
    return result.returncode == 0


def main() -> int:
    args = _parser().parse_args()
    system = platform.system()
    machine = platform.machine()
    print(f"Sistema: {system} {machine}")
    if system != "Linux":
        print(
            "vLLM è destinato a Linux/CUDA. Su macOS usa il backend "
            "llama.cpp con Metal."
        )
        return 2
    if not shutil.which("nvidia-smi"):
        print("AVVISO: nvidia-smi non trovato; CUDA non è verificabile.")

    if args.install and not _install():
        print(
            "Installazione fallita. Usa l'immagine/container NVIDIA adatta "
            "alla versione di DGX OS/CUDA oppure consulta la documentazione "
            "vLLM corrente."
        )
        return 1

    if args.download_model:
        if not args.model:
            print("--download-model richiede anche --model ORGANIZZAZIONE/NOME.")
            return 2
        if not _download_model(args.model):
            return 1

    diagnostic = diagnose_vllm_acceleration()
    print(diagnostic.summary)
    print(diagnostic.details)

    models = list_cached_vllm_models()
    print(f"\nModelli Hugging Face già locali: {len(models)}")
    for model in models:
        print(f"  - {model}")
    if args.model:
        info = resolve_vllm_model(args.model)
        if info is None or not info.get("weights_present"):
            print(
                f"\nModello assente o incompleto: {args.model}\n"
                "Scaricalo esplicitamente prima dell'uso con "
                "--download-model, poi riapri Configura LLM."
            )
            return 3
        if not info.get("chat_capable"):
            print(
                f"\nIl modello {args.model} non dichiara un'architettura "
                "generativa compatibile con Chat Completions."
            )
            return 4
        print(f"\nModello verificato: {args.model}")
        print(f"Percorso locale: {info['file']}")
        print(f"Contesto dichiarato: {info.get('max_context_length') or 'n/d'}")

    if not diagnostic.accelerated:
        print(
            "\nBackend non ancora pronto. Per installare esplicitamente "
            "nell'ambiente corrente:"
        )
        print(f"  {sys.executable} tools/setup_vllm_backend.py --install")
        return 1
    print(
        "\nBackend pronto. In Configura LLM seleziona «vLLM · CUDA», "
        "scegli un modello locale e usa «Carica e testa»."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

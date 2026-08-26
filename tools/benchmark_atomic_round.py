#!/usr/bin/env python3
"""Reproducible, read-only benchmark of the atomic-evidence pipeline.

Clinical source data are never written inside the repository.  Every artifact
is stored below the explicitly supplied output directory (normally /tmp), and
the three project databases are opened immutable/read-only.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random
import sqlite3
import sys
import threading
import time

# Allow direct execution as ``python tools/benchmark_atomic_round.py``.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from emr_analyzer.clinical.atomic_evidence import AtomicEvidenceExtractor
from emr_analyzer.clinical.lab_evidence import abnormal_lab_evidence
from emr_analyzer.extraction.llm_client import LlmClient
from emr_analyzer.llm_backend import shutdown_all_backends
from emr_analyzer.models.lab_result import LabValue
from emr_analyzer.pipeline.sensitive_data import SensitiveDataSanitizer
from emr_analyzer.settings import load_llm_configs, load_pipeline_policy


SEED = "emr-atomic-round-2026-08-26-v1"
QUOTAS = {
    "visita_oncologica": ("short", "medium", "long", "very_long"),
    "visita_specialistica": ("short", "medium", "long"),
    "laboratorio": ("short", "medium"),
    "radiologia": ("short", "medium", "long"),
    "anatomia_patologica": ("short", "medium", "long"),
    "lettera_dimissione": ("medium",),
    "pronto_soccorso": ("long",),
    "verbale_operatorio": ("medium",),
    "cartella_clinica": ("long",),
    "non_classificato": ("short",),
}
PROJECTS = ("MELANOMA", "RENE", "LUNG")


def _stable_key(*parts: object) -> str:
    value = "\x1f".join(str(part) for part in parts)
    return hashlib.sha256(f"{SEED}\x1e{value}".encode()).hexdigest()


def _length_bin(length: int) -> str:
    if length < 1_500:
        return "short"
    if length < 5_000:
        return "medium"
    if length < 12_000:
        return "long"
    return "very_long"


def _connect_readonly(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(
        f"file:{db_path}?mode=ro&immutable=1", uri=True,
    )
    conn.row_factory = sqlite3.Row
    return conn


def _find_normalized(project: Path, patient_id: str, document_id: str) -> Path | None:
    expected = project / patient_id / "extraction" / f"{document_id}.md"
    if expected.is_file():
        return expected
    matches = list(project.glob(f"*/extraction/{document_id}.md"))
    return matches[0] if len(matches) == 1 else None


def _candidates(project: Path) -> list[dict]:
    conn = _connect_readonly(project / "emr_registry.db")
    try:
        rows = conn.execute(
            """
            SELECT id, patient_id, document_type, document_date
            FROM documents
            WHERE parsing_status = 'completed'
              AND extraction_status = 'done'
            """
        ).fetchall()
    finally:
        conn.close()
    sanitizer = SensitiveDataSanitizer()
    found: list[dict] = []
    seen_text: set[str] = set()
    for row in rows:
        path = _find_normalized(project, row["patient_id"], row["id"])
        if path is None:
            continue
        try:
            raw = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError):
            continue
        clean = sanitizer.sanitize(raw)
        text = clean.text.strip()
        if len(text) < 200:
            continue
        digest = hashlib.sha256(text.encode()).hexdigest()
        if digest in seen_text:
            continue
        seen_text.add(digest)
        found.append({
            "project": project.name,
            "project_path": str(project),
            "patient_id": row["patient_id"],
            "document_id": row["id"],
            "document_type": row["document_type"],
            "document_date": row["document_date"],
            "source_path": str(path),
            "text": text,
            "text_sha256": digest,
            "chars": len(text),
            "length_bin": _length_bin(len(text)),
            "redactions": clean.counts,
        })
    return found


def _select_project(candidates: list[dict]) -> list[dict]:
    selected: list[dict] = []
    selected_ids: set[str] = set()
    used_patients: set[str] = set()
    for document_type, targets in QUOTAS.items():
        for target_bin in targets:
            pool = [
                row for row in candidates
                if row["document_type"] == document_type
                and row["document_id"] not in selected_ids
            ]
            exact = [row for row in pool if row["length_bin"] == target_bin]
            unused = [row for row in exact if row["patient_id"] not in used_patients]
            fallback_unused = [
                row for row in pool if row["patient_id"] not in used_patients
            ]
            choices = unused or exact or fallback_unused or pool
            if not choices:
                raise RuntimeError(
                    f"Nessun candidato per {document_type}/{target_bin}"
                )
            chosen = min(
                choices,
                key=lambda row: _stable_key(
                    row["project"], document_type, target_bin,
                    row["patient_id"], row["document_id"],
                ),
            )
            selected.append(chosen)
            selected_ids.add(chosen["document_id"])
            used_patients.add(chosen["patient_id"])
    if len(selected) != 20:
        raise AssertionError(f"Campione inatteso: {len(selected)}")
    return selected


def build_sample(projects_root: Path, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    text_dir = output / "blind_texts"
    text_dir.mkdir(exist_ok=True)
    selected: list[dict] = []
    for project_name in PROJECTS:
        project = projects_root / project_name
        candidates = _candidates(project)
        project_sample = _select_project(candidates)
        selected.extend(project_sample)
        print(
            f"SAMPLE {project_name}: {len(project_sample)}/20 "
            f"da {len(candidates)} candidati", flush=True,
        )

    # Global opaque case identifiers keep the clinical reference reviewer
    # blind to project and patient identity.
    shuffled = sorted(
        selected,
        key=lambda row: _stable_key(
            "global", row["project"], row["document_id"]
        ),
    )
    manifest: list[dict] = []
    for index, row in enumerate(shuffled, start=1):
        case_id = f"CASE_{index:03d}"
        destination = text_dir / f"{case_id}.txt"
        destination.write_text(row.pop("text"), encoding="utf-8")
        manifest.append({
            "case_id": case_id,
            "sanitized_path": str(destination),
            **row,
        })

    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    # Three project-blind, evenly mixed batches for independent Sol review.
    for batch_index in range(3):
        batch = []
        for row in manifest[batch_index::3]:
            batch.append({
                "case_id": row["case_id"],
                "document_type": row["document_type"],
                "report_date": row["document_date"],
                "text_path": row["sanitized_path"],
            })
        (output / f"sol_batch_{batch_index + 1}.json").write_text(
            json.dumps(batch, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    summary = {
        project: {
            "cases": sum(row["project"] == project for row in manifest),
            "patients": len({
                row["patient_id"] for row in manifest
                if row["project"] == project
            }),
            "types": {
                doc_type: sum(
                    row["project"] == project
                    and row["document_type"] == doc_type
                    for row in manifest
                )
                for doc_type in QUOTAS
            },
        }
        for project in PROJECTS
    }
    (output / "sample_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"MANIFEST: {output / 'manifest.json'}", flush=True)


def _lab_values(case: dict) -> list[LabValue]:
    db_path = Path(case["project_path"]) / "emr_registry.db"
    conn = _connect_readonly(db_path)
    try:
        rows = conn.execute(
            "SELECT * FROM lab_values WHERE document_id = ?",
            (case["document_id"],),
        ).fetchall()
    finally:
        conn.close()
    sanitizer = SensitiveDataSanitizer()
    values = []
    allowed = set(LabValue.__dataclass_fields__)
    for row in rows:
        payload = {key: row[key] for key in row.keys() if key in allowed}
        payload["is_abnormal"] = bool(payload.get("is_abnormal"))
        payload["validated_by_user"] = bool(payload.get("validated_by_user"))
        payload["source_text"] = sanitizer.sanitize(
            payload.get("source_text") or ""
        ).text
        values.append(LabValue.from_dict(payload))
    return values


def run_qwen(
    output: Path,
    *,
    workers: int | None = None,
    case_ids: set[str] | None = None,
    result_dir: str = "qwen",
) -> None:
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    if case_ids:
        manifest = [case for case in manifest if case["case_id"] in case_ids]
        missing = case_ids - {case["case_id"] for case in manifest}
        if missing:
            raise ValueError(f"Case non trovati: {', '.join(sorted(missing))}")
    target = output / result_dir
    target.mkdir(exist_ok=True)
    configs = load_llm_configs()
    config = configs["atomic_evidence"]
    policy = load_pipeline_policy()
    effective_workers = max(1, min(workers or config.parallel_workers, 8))
    client = LlmClient(config=config)
    client.retain_only_this_runtime()
    print(
        "QWEN CONFIG "
        f"model={config.model} backend={config.backend} "
        f"ctx={config.context_length} output={config.max_output_tokens} "
        f"workers={effective_workers}",
        flush=True,
    )
    warm = client.warmup()
    print(
        f"QWEN WARMUP {warm.get('elapsed_seconds', 0):.1f}s "
        f"slots={warm.get('slots', '?')}", flush=True,
    )
    extractor = AtomicEvidenceExtractor(client, policy=policy)
    write_lock = threading.Lock()
    started = time.perf_counter()
    completed = 0

    def one(case: dict) -> tuple[str, bool, float, int, int]:
        case_id = case["case_id"]
        result_path = target / f"{case_id}.json"
        if result_path.exists():
            return case_id, True, 0.0, 0, 0
        text = Path(case["sanitized_path"]).read_text(encoding="utf-8")
        case_started = time.perf_counter()
        try:
            llm_items = extractor.extract_document(
                patient_id=case["patient_id"],
                document_id=case["document_id"],
                document_type=case["document_type"],
                document_date=case["document_date"],
                text=text,
            )
            lab_items = abnormal_lab_evidence(
                patient_id=case["patient_id"],
                document_id=case["document_id"],
                document_date=case["document_date"],
                lab_values=_lab_values(case),
                policy=policy.lab,
            )
            payload = {
                "case_id": case_id,
                "status": "ok",
                "elapsed_seconds": time.perf_counter() - case_started,
                "metrics": extractor.last_extraction_metrics(),
                "llm_evidence_count": len(llm_items),
                "deterministic_lab_count": len(lab_items),
                "evidence": [
                    item.to_atomic_dict() for item in (*llm_items, *lab_items)
                ],
                # Keep the complete dataclass contract too: the compact wire
                # view intentionally omits some structured fields used by
                # clinical-quality scoring (site, severity, status, etc.).
                "evidence_full": [
                    item.to_dict() for item in (*llm_items, *lab_items)
                ],
            }
            temporary = result_path.with_suffix(".json.tmp")
            temporary.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            temporary.replace(result_path)
            metrics = payload["metrics"]
            return (
                case_id, False, payload["elapsed_seconds"],
                len(payload["evidence"]), int(metrics.get("llm_calls", 0)),
            )
        except Exception as exc:
            payload = {
                "case_id": case_id,
                "status": "error",
                "elapsed_seconds": time.perf_counter() - case_started,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            result_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            return case_id, False, payload["elapsed_seconds"], -1, -1

    try:
        with ThreadPoolExecutor(max_workers=effective_workers) as pool:
            futures = {pool.submit(one, case): case for case in manifest}
            for future in as_completed(futures):
                case_id, cached, elapsed, evidence_count, calls = future.result()
                with write_lock:
                    completed += 1
                    total_elapsed = time.perf_counter() - started
                    rate = completed / total_elapsed if total_elapsed else 0.0
                    eta = (
                        (len(manifest) - completed) / rate if rate else 0.0
                    )
                    marker = "cached" if cached else (
                        "error" if evidence_count < 0 else "ok"
                    )
                    print(
                        f"QWEN {completed:02d}/{len(manifest)} {case_id} "
                        f"{marker} {elapsed:.1f}s atoms={evidence_count} "
                        f"calls={calls} ETA={eta / 60:.1f}m",
                        flush=True,
                    )
    finally:
        shutdown_all_backends()
    print(
        f"QWEN COMPLETE elapsed={(time.perf_counter() - started) / 60:.1f}m",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    sample = subparsers.add_parser("sample")
    sample.add_argument("--projects-root", type=Path, required=True)
    sample.add_argument("--output", type=Path, required=True)
    qwen = subparsers.add_parser("qwen")
    qwen.add_argument("--output", type=Path, required=True)
    qwen.add_argument("--workers", type=int)
    qwen.add_argument("--case-ids", nargs="*")
    qwen.add_argument("--result-dir", default="qwen")
    args = parser.parse_args()
    if args.command == "sample":
        build_sample(args.projects_root, args.output)
    else:
        run_qwen(
            args.output,
            workers=args.workers,
            case_ids=set(args.case_ids or ()),
            result_dir=args.result_dir,
        )


if __name__ == "__main__":
    main()

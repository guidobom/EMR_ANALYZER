"""Source-grounded event extraction for one patient, persisted per document."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import threading
import time
from typing import Callable, Optional

from .evidence_utils import content_hash, deduplicate_atomic_evidence
from .grounded_sources import (
    catalog_snapshot, IncompleteAtomicExtraction, AtomicExtractionCancelled,
    VERSION, METHOD,
)
from .event_extraction import EventExtractor
from .lab_evidence import LAB_EXTRACTION_METHOD, filter_narrative_lab_duplicates
from ..config import active_workspace
from ..models.clinical_registry import ProcessingManifestItem, ProcessingRun
from ..settings import load_pipeline_policy

STAGE = "atomic_evidence"
FHIR_FILENAME = "clinical_events.fhir.json"


class ExtractionCancelled(RuntimeError):
    pass


class ExtractionPipeline:
    """Extract clinical events from every active text of a patient.

    Each document is committed as soon as it completes, so an interrupted
    run resumes from the processing manifest and the group checkpoints.
    """

    def __init__(
        self,
        *,
        evidence_repo,
        processing_repo,
        document_repo,
        lab_repo,
        overlay_repo,
        llm_client=None,
        audit_repo=None,
        pipeline_policy=None,
        shared_lexicon_repo=None,
        db=None,
    ):
        self.evidence_repo = evidence_repo
        self.processing_repo = processing_repo
        self.document_repo = document_repo
        self.lab_repo = lab_repo
        self.overlay_repo = overlay_repo
        self.llm = llm_client
        self.audit = audit_repo
        self.pipeline_policy = pipeline_policy or load_pipeline_policy()
        self.shared_lexicon_repo = shared_lexicon_repo
        self.db = db or evidence_repo.db
        self.extractor = self.make_extractor()

    def make_extractor(self):
        if self.llm is None:
            return None
        from ..database.atomic_group_repo import AtomicGroupRepository
        return EventExtractor(
            self.llm,
            snomed_catalog=getattr(self.shared_lexicon_repo, "snomed_catalog", None),
            policy=self.pipeline_policy,
            catalog=catalog_snapshot(self.shared_lexicon_repo),
            checkpoint_repo=AtomicGroupRepository(self.db),
        )

    def set_llm(self, llm_client) -> None:
        self.llm = llm_client
        self.extractor = self.make_extractor()

    def reload_policy(self) -> None:
        self.pipeline_policy = load_pipeline_policy()
        self.extractor = self.make_extractor()

    @property
    def prompt_version(self):
        return getattr(self.extractor, "prompt_version", VERSION)

    @property
    def prompt_digest(self):
        from ..prompt_catalog import load_prompt
        return getattr(self.extractor, "prompt_digest",
                       content_hash(load_prompt("compact_events_system")))

    def has_checkpoint(self, patient_id: str) -> bool:
        """Whether completed work exists that a new run would reuse."""
        row = self.db.execute(
            """SELECT 1 FROM processing_manifest
               WHERE patient_id=? AND stage=? AND status='completed' LIMIT 1""",
            (patient_id, STAGE),
        ).fetchone()
        if row is not None:
            return True
        return self.db.execute(
            """SELECT 1 FROM atomic_group_results WHERE patient_id=?
               AND status IN ('completed', 'completed_empty') LIMIT 1""",
            (patient_id,),
        ).fetchone() is not None

    def extract_patient(
        self,
        patient_id: str,
        *,
        incremental: bool = True,
        num_workers: int = 1,
        progress_callback: Optional[Callable[[int, str], None]] = None,
        cancel_check: Optional[Callable[[], bool]] = None,
    ) -> dict:
        started = time.monotonic()
        documents = sorted(
            self.document_repo.list_by_patient(patient_id),
            key=lambda doc: (doc.document_date or "9999", doc.id),
        )
        # The shared Lexicon may have changed since the previous run.
        self.extractor = self.make_extractor()
        model_digest = self.extractor.model_digest if self.extractor else ""
        run = ProcessingRun(
            patient_id=patient_id, stage="atomic_evidence_v3",
            model_name=self.extractor.model_name if self.extractor else None,
            model_digest=model_digest, prompt_version=self.prompt_version,
            parameters={
                "incremental": incremental, "requested_workers": num_workers,
                "document_count": len(documents),
                "atomic_prompt_digest": self.prompt_digest,
                "atomic_model": getattr(self.llm, "model", None),
            },
        )
        self.processing_repo.start_run(run)
        failures: list[dict] = []
        doc_stats: list[dict] = []
        skipped = processed = extracted_count = 0
        progress_lock = threading.Lock()
        progress_value = 0

        def report(value, message):
            nonlocal progress_value
            if progress_callback:
                with progress_lock:
                    progress_value = max(progress_value, value)
                    progress_callback(progress_value, message)

        def check_cancelled() -> None:
            if cancel_check is not None and cancel_check():
                raise ExtractionCancelled("Elaborazione interrotta su richiesta dell'utente")

        try:
            check_cancelled()
            self._sync_laboratory_events(patient_id, documents)
            tasks = []
            for doc in documents:
                path = self.normalized_text_path(patient_id, doc.id)
                if path is None:
                    failures.append({"document_id": doc.id, "error": "Testo anonimizzato non disponibile."})
                    continue
                base_text = path.read_text(encoding="utf-8")
                text = self.overlay_repo.effective_text(doc.id, base_text) if self.overlay_repo else base_text
                input_hash = content_hash(text, doc.document_date, doc.document_type,
                                          self.prompt_version, self.prompt_digest)
                if incremental and self.processing_repo.is_current(
                        doc.id, STAGE, input_hash, self.prompt_version,
                        self.prompt_version, model_digest):
                    skipped += 1
                    continue
                tasks.append((doc, text, input_hash))

            available = bool(self.extractor is not None and self.llm
                             and getattr(self.llm, "is_available", False))
            if tasks and not available:
                failures.extend({"document_id": doc.id, "error":
                    "Modello locale non disponibile: il documento sarà riprovato alla prossima esecuzione."}
                    for doc, _, _ in tasks)
                tasks = []
            if tasks:
                self._retain_only_extraction_runtime()

            workers = max(1, min(int(num_workers or 1), len(tasks) or 1))
            report(0, f"Eventi: {len(tasks)} documenti da elaborare, {skipped} invariati")
            manifests = {}
            for doc, _, input_hash in tasks:
                item = ProcessingManifestItem(
                    patient_id=patient_id, document_id=doc.id, stage=STAGE,
                    input_hash=input_hash, pipeline_version=self.prompt_version,
                    run_id=run.run_id, prompt_version=self.prompt_version,
                    model_digest=model_digest, status="running",
                )
                self.processing_repo.upsert_manifest(item)
                manifests[doc.id] = item.manifest_id

            def extract_one(doc, text):
                task_started = time.monotonic()
                check_cancelled()
                kwargs = dict(
                    patient_id=patient_id, document_id=doc.id,
                    document_type=doc.document_type, document_date=doc.document_date,
                    text=text, geometry_path=_geometry_source(doc, active_workspace.path),
                    stage_progress_callback=lambda message: report(
                        int(90 * processed / max(len(tasks), 1)), f"{doc.id}: {message}"),
                    chunk_progress_callback=lambda done, total: report(
                        int(90 * (processed + done / max(total, 1)) / max(len(tasks), 1)),
                        f"{doc.id}: gruppi {done}/{total} verificati"),
                )
                try:
                    evidence = self.extractor.extract_document(
                        evidence_ready_callback=lambda rows: self.replace_document_evidence(doc.id, rows),
                        cancel_check=cancel_check, **kwargs)
                except AtomicExtractionCancelled as exc:
                    raise ExtractionCancelled(str(exc)) from exc
                except IncompleteAtomicExtraction as exc:
                    self.replace_document_evidence(doc.id, exc.evidence)
                    raise
                check_cancelled()
                return doc, evidence, round(time.monotonic() - task_started, 3), \
                    self.extractor.last_extraction_metrics()

            def persist(result) -> None:
                nonlocal processed, extracted_count
                doc, evidence, duration, metrics = result
                self.replace_document_evidence(doc.id, evidence)
                self.processing_repo.mark_result(
                    manifests[doc.id], status="completed",
                    output_hash=_evidence_hash(evidence), output_count=len(evidence))
                processed += 1
                extracted_count += len(evidence)
                doc_stats.append({"document_id": doc.id, "evidence_count": len(evidence),
                                  "elapsed_seconds": duration, "status": "completed", **metrics})

            def fail(doc, exc) -> None:
                failures.append({"document_id": doc.id, "error": str(exc)})
                incomplete = isinstance(exc, IncompleteAtomicExtraction)
                doc_stats.append({"document_id": doc.id, "status": "failed", "error": str(exc),
                                  "evidence_count": len(exc.evidence) if incomplete else 0,
                                  **(exc.metrics if incomplete else {})})
                self.processing_repo.mark_result(manifests[doc.id], status="failed",
                                                 error_message=str(exc))

            # Longest first reduces the final straggler with identical workers.
            ordered = sorted(tasks, key=lambda task: len(task[1]), reverse=True)
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = {pool.submit(extract_one, doc, text): doc for doc, text, _ in ordered}
                for completed, future in enumerate(as_completed(futures), start=1):
                    doc = futures[future]
                    try:
                        persist(future.result())
                    except ExtractionCancelled:
                        for pending in futures:
                            pending.cancel()
                        raise
                    except Exception as exc:
                        fail(doc, exc)
                    report(int(completed / max(len(ordered), 1) * 90),
                           f"Documenti {completed}/{len(ordered)}")

            check_cancelled()
            fhir = self.write_fhir(patient_id, documents, {
                "documents_total": len(documents), "processed": processed,
                "unchanged": skipped, "failures": failures,
                "complete": not failures, "pipeline": VERSION,
            }, cancel_check=check_cancelled, progress=report)
            stored = self.evidence_repo.get_by_patient(patient_id)
            elapsed = round(time.monotonic() - started, 2)
            self.processing_repo.finish_run(
                run.run_id, "completed_with_warnings" if failures else "completed")
            summary = _summarize(doc_stats)
            if self.audit:
                self.audit.log(patient_id, "clinical_events_extracted", "clinical_evidence",
                               patient_id, {"documents_total": len(documents),
                                            "documents_processed": processed,
                                            "documents_skipped": skipped,
                                            "documents_failed": len(failures),
                                            "evidence_stored_occurrences": len(stored),
                                            "elapsed_seconds": elapsed, **summary},
                               model_used=getattr(self.llm, "model", None),
                               model_version=self.prompt_version, run_id=run.run_id)
            report(100, "Estrazione completata")
            return {
                "fhir_path": fhir["path"], "fhir_events": fhir["events"],
                "fhir_uncoded": fhir["uncoded"],
                "total_entries": len(deduplicate_atomic_evidence(stored)),
                "stored_evidence_occurrences": len(stored),
                "documents_processed": processed, "documents_skipped": skipped,
                "documents_failed": len(failures),
                "failed_doc_ids": [item["document_id"] for item in failures],
                "failures": failures,
                "document_stats": sorted(doc_stats, key=lambda item: item["document_id"]),
                "atomic_evidence_extracted": extracted_count,
                "incremental": incremental, "atomic_model": getattr(self.llm, "model", None),
                "run_id": run.run_id, "elapsed_seconds": elapsed, **summary,
            }
        except ExtractionCancelled as exc:
            self.processing_repo.interrupt_run(run.run_id, str(exc))
            raise
        except Exception as exc:
            self.processing_repo.finish_run(run.run_id, "failed", str(exc))
            raise

    def write_fhir(self, patient_id, documents, coverage, *, cancel_check=None, progress=None) -> dict:
        from .fhir_registry import FhirRegistry
        loinc = getattr(self.shared_lexicon_repo, "loinc_catalog", None)
        lab_values = self.lab_repo.get_by_patient(patient_id)
        proposals = {}
        if loinc and self.llm and getattr(self.llm, "is_available", False):
            if progress:
                progress(92, "Codifica LOINC dei risultati di laboratorio…")
            proposals = loinc.propose(lab_values, self.llm, lambda: (cancel_check() if cancel_check else None) or False)
        exporter = FhirRegistry(patient_id, loinc, proposals)
        for document in documents:
            exporter.document(document.id)
        for item in self.evidence_repo.get_by_patient(patient_id):
            if item.extraction_method == METHOD:
                exporter.clinical(item)
        for ordinal, lab in enumerate(lab_values):
            exporter.laboratory(lab, ordinal)
        path = exporter.write(active_workspace.path / patient_id / FHIR_FILENAME, coverage)
        return {"path": path, "events": exporter.event_count, "uncoded": exporter.unmapped}

    def patient_status(self, patient_id: str) -> dict:
        """Documents with text, current extractions and failures."""
        documents = self.document_repo.list_by_patient(patient_id)
        rows = self.db.execute(
            """SELECT document_id, status FROM processing_manifest
               WHERE patient_id=? AND stage=? ORDER BY updated_at""",
            (patient_id, STAGE),
        ).fetchall()
        latest = {row["document_id"]: row["status"] for row in rows}
        with_text = [doc for doc in documents if self.normalized_text_path(patient_id, doc.id)]
        return {
            "documents": len(documents), "with_text": len(with_text),
            "completed": sum(latest.get(doc.id) == "completed" for doc in with_text),
            "failed": sum(latest.get(doc.id) == "failed" for doc in with_text),
            "fhir_path": str(active_workspace.path / patient_id / FHIR_FILENAME),
        }

    @staticmethod
    def normalized_text_path(patient_id: str, document_id: str) -> Path | None:
        candidates = (
            active_workspace.path / patient_id / "extraction" / f"{document_id}.md",
            active_workspace.path / patient_id / "docling" / f"{document_id}.md",
        )
        return next((path for path in candidates if path.exists()), None)

    def _retain_only_extraction_runtime(self) -> int:
        reserve = getattr(self.llm, "retain_only_this_runtime", None)
        try:
            return int(reserve() or 0) if callable(reserve) else 0
        except Exception:
            return 0

    def _sync_laboratory_events(self, patient_id: str, documents) -> int:
        from .fhir_registry import laboratory_evidence
        values = self.lab_repo.get_by_patient(patient_id)
        catalog = getattr(self.shared_lexicon_repo, "loinc_catalog", None)
        by_document = {}
        for ordinal, value in enumerate(values):
            by_document.setdefault(value.document_id, []).append(
                laboratory_evidence(value, ordinal, catalog))
        for document in documents:
            self.evidence_repo.replace_document_method(document.id, LAB_EXTRACTION_METHOD, [])
            self.evidence_repo.replace_document_method(
                document.id, "fhir_laboratory_v1", by_document.get(document.id, []))
        return sum(bool(value.is_abnormal) for value in values)

    def replace_document_evidence(self, document_id: str, evidence) -> None:
        """Replace this document's extracted events; legacy atoms are cleared."""
        stored = self.evidence_repo.get_by_document(document_id)
        labs = [item for item in stored if item.extraction_method == "fhir_laboratory_v1"]
        evidence, _ = filter_narrative_lab_duplicates(list(evidence), labs)
        grouped: dict[str, list] = {
            METHOD: [], "shared_lexicon_atomic": [], "icd11_extraction": [],
            "llm_atomic_v2": [], "deterministic_nonclinical": [],
        }
        for item in evidence:
            grouped.setdefault(item.extraction_method, []).append(item)
        for method, items in grouped.items():
            self.evidence_repo.replace_document_method(document_id, method, items)


def _summarize(doc_stats) -> dict:
    total = lambda key, cast=int: sum(cast(item.get(key, 0) or 0) for item in doc_stats)
    keys = ("llm_calls", "source_chunks", "prompt_tokens", "completion_tokens",
            "recovery_calls", "snomed_mapped", "snomed_unmapped")
    summary = {key: total(key) for key in keys}
    summary.update(prompt_ms=round(total("prompt_ms", float), 3),
                   predicted_ms=round(total("predicted_ms", float), 3))
    return summary


def _evidence_hash(evidence) -> str:
    payload = [{"evidence_id": item.evidence_id, "category": item.category,
                "entity": item.normalized_entity, "date": item.observed_date,
                "source": item.source_text}
               for item in sorted(evidence, key=lambda item: item.evidence_id)]
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True)
                          .encode("utf-8")).hexdigest()


def _geometry_source(doc, workspace_root):
    """Legacy geometry remains readable; new projects reparse the original."""
    from ..utils.document_paths import resolve_document_path
    legacy = Path(workspace_root) / doc.patient_id / "extraction" / f"{doc.id}.json"
    return legacy if legacy.is_file() else Path(resolve_document_path(doc, workspace_root))

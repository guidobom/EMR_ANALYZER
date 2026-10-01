"""Source-grounded event extraction for one or many patients.

All documents of the selected patients share one worker pool, so the local
server slots stay busy across patient boundaries.  Each document is committed
as soon as it completes and each patient is finalized (FHIR file, run status)
as soon as its last document is done; an interrupted run resumes from the
processing manifest and the group checkpoints.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
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
from .snomed_coding import CodingCancelled, ConceptCoder
from .lab_evidence import LEGACY_LAB_METHODS, filter_narrative_lab_duplicates
from .statement_index import PROJECTION_VERSION
from ..config import active_workspace
from ..models.clinical_registry import ProcessingManifestItem, ProcessingRun
from ..models.document import DocumentType
from ..settings import load_pipeline_policy

STAGE = "atomic_evidence"
FHIR_FILENAME = "clinical_events.fhir.json"


class ExtractionCancelled(RuntimeError):
    pass


@dataclass
class _PatientPlan:
    patient_id: str
    documents: list
    run: ProcessingRun
    tasks: list = field(default_factory=list)       # (doc, text, manifest_id)
    #: sentence id -> 'origin' | 'copy' for every narrative document
    statement_roles: dict = field(default_factory=dict)
    skipped: int = 0
    laboratory_documents: int = 0
    processed: int = 0
    extracted: int = 0
    failures: list = field(default_factory=list)
    doc_stats: list = field(default_factory=list)
    started: float = field(default_factory=time.monotonic)
    remaining: int = 0
    result: dict | None = None
    error: Exception | None = None


class ExtractionPipeline:
    """Extract, persist and export the clinical events of active texts."""

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
        from ..database.statement_repo import (StatementAnnotationRepository,
                                              StatementIndexRepository)
        self.statements = StatementIndexRepository(self.db)
        self.statement_annotations = StatementAnnotationRepository(self.db)
        self.extractor = self.make_extractor()
        self.coder = self.make_coder()

    def make_extractor(self):
        if self.llm is None:
            return None
        from ..database.atomic_group_repo import AtomicGroupRepository
        return EventExtractor(
            self.llm,
            policy=self.pipeline_policy,
            catalog=catalog_snapshot(self.shared_lexicon_repo),
            checkpoint_repo=AtomicGroupRepository(self.db),
        )

    def make_coder(self) -> ConceptCoder:
        return ConceptCoder(
            self.llm,
            getattr(self.shared_lexicon_repo, "snomed_catalog", None),
            getattr(self.shared_lexicon_repo, "concept_mappings", None),
        )

    def set_llm(self, llm_client) -> None:
        self.llm = llm_client
        self.extractor = self.make_extractor()
        self.coder = self.make_coder()

    def reload_policy(self) -> None:
        self.pipeline_policy = load_pipeline_policy()
        self.extractor = self.make_extractor()
        self.coder = self.make_coder()

    def review_service(self):
        """Reviewed events of a patient: occurrences, concept codes, decisions."""
        from .event_review import EventReviewService
        from ..database.event_override_repo import EventOverrideRepository
        return EventReviewService(
            self.evidence_repo, EventOverrideRepository(self.db), self.coder,
            overlay_repo=self.overlay_repo, audit_repo=self.audit,
            snomed=getattr(self.shared_lexicon_repo, "snomed_catalog", None))

    def events(self, patient_id: str, *, include_rejected: bool = False) -> list:
        return self.review_service().events(patient_id, include_rejected=include_rejected)

    def code_patient(self, patient_id: str, cancel_check=None, progress=None) -> dict:
        """Code the patient's concepts that have no SNOMED mapping yet."""
        try:
            return self.coder.code_events(self.events(patient_id), cancel_check=cancel_check,
                                          progress=progress)
        except CodingCancelled as exc:
            raise ExtractionCancelled(str(exc)) from exc

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

    # ------------------------------------------------------------- running
    def extract_patient(self, patient_id: str, **kwargs) -> dict:
        """Extract one patient; failures of the patient itself are raised."""
        plans = self.extract_patients([patient_id], **kwargs)
        plan = plans[patient_id]
        if plan.error is not None:
            raise plan.error
        return plan.result

    def extract_patients(
        self,
        patient_ids: list[str],
        *,
        incremental: bool = True,
        num_workers: int = 1,
        progress_callback: Optional[Callable[[int, str], None]] = None,
        cancel_check: Optional[Callable[[], bool]] = None,
        patient_finished: Optional[Callable[[str, dict], None]] = None,
        patient_error: Optional[Callable[[str, str], None]] = None,
    ) -> dict[str, _PatientPlan]:
        started = time.monotonic()
        lock = threading.Lock()
        progress = {"value": 0}

        def report(value, message):
            if progress_callback:
                with lock:
                    progress["value"] = max(progress["value"], int(value))
                    progress_callback(progress["value"], message)

        def check_cancelled() -> None:
            if cancel_check is not None and cancel_check():
                raise ExtractionCancelled("Elaborazione interrotta su richiesta dell'utente")

        # The shared Lexicon may have changed since the previous run.
        self.extractor = self.make_extractor()
        self.coder = self.make_coder()
        model_digest = self.extractor.model_digest if self.extractor else ""
        plans: dict[str, _PatientPlan] = {}
        try:
            for patient_id in patient_ids:
                check_cancelled()
                plans[patient_id] = self._plan(patient_id, incremental, model_digest, num_workers)
            total = sum(len(plan.tasks) for plan in plans.values())
            skipped = sum(plan.skipped for plan in plans.values())
            available = bool(self.extractor is not None and self.llm
                             and getattr(self.llm, "is_available", False))
            if total and not available:
                for plan in plans.values():
                    for doc, _, manifest_id in plan.tasks:
                        self._fail(plan, doc, manifest_id, RuntimeError(
                            "Modello locale non disponibile: il documento sarà riprovato "
                            "alla prossima esecuzione."))
                    plan.tasks = []
                total = 0
            if total:
                self._retain_only_extraction_runtime()
            report(0, f"{total} documenti da elaborare, {skipped} invariati")

            def finalize(plan):
                try:
                    plan.result = self._finalize(plan, check_cancelled)
                    if patient_finished:
                        patient_finished(plan.patient_id, plan.result)
                except ExtractionCancelled:
                    raise
                except Exception as exc:
                    plan.error = exc
                    self.processing_repo.finish_run(plan.run.run_id, "failed", str(exc))
                    if patient_error:
                        patient_error(plan.patient_id, str(exc))

            for plan in plans.values():
                plan.remaining = len(plan.tasks)
                if not plan.tasks:
                    finalize(plan)
            # Patients in order, longest documents first within each patient:
            # early patients finish early while the pool never idles.
            ordered = [(plan, task) for plan in plans.values()
                       for task in sorted(plan.tasks, key=lambda task: len(task[1]), reverse=True)]
            workers = max(1, min(int(num_workers or 1), total or 1))
            completed = 0
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = {
                    pool.submit(self._extract, plan, doc, text, cancel_check, report, total,
                                lambda: completed): (plan, doc, manifest_id)
                    for plan, (doc, text, manifest_id) in ordered
                }
                try:
                    for future in as_completed(futures):
                        plan, doc, manifest_id = futures[future]
                        try:
                            self._persist(plan, manifest_id, future.result())
                        except ExtractionCancelled:
                            raise
                        except Exception as exc:
                            self._fail(plan, doc, manifest_id, exc)
                        completed += 1
                        plan.remaining -= 1
                        if plan.remaining == 0:
                            finalize(plan)
                        report(int(completed / max(total, 1) * 100),
                               _eta_message(completed, total, time.monotonic() - started, plan.patient_id))
                except ExtractionCancelled:
                    for pending in futures:
                        pending.cancel()
                    raise
        except ExtractionCancelled as exc:
            for plan in plans.values():
                if plan.result is None and plan.error is None:
                    self.processing_repo.interrupt_run(plan.run.run_id, str(exc))
            raise
        report(100, "Elaborazione completata")
        return plans

    def _plan(self, patient_id, incremental, model_digest, num_workers) -> _PatientPlan:
        documents = sorted(
            self.document_repo.list_by_patient(patient_id),
            key=lambda doc: (doc.document_date or "9999", doc.id),
        )
        run = ProcessingRun(
            patient_id=patient_id, stage="atomic_evidence_v3",
            model_name=self.extractor.model_name if self.extractor else None,
            model_digest=model_digest, prompt_version=self.prompt_version,
            parameters={"incremental": incremental, "requested_workers": num_workers,
                        "document_count": len(documents),
                        "atomic_prompt_digest": self.prompt_digest,
                        "atomic_model": getattr(self.llm, "model", None)},
        )
        self.processing_repo.start_run(run)
        plan = _PatientPlan(patient_id=patient_id, documents=documents, run=run)
        # Projections written by earlier builds are no longer read.
        self.evidence_repo.delete_methods(patient_id, LEGACY_LAB_METHODS)
        lab_counts = dict(self.db.execute(
            "SELECT document_id, COUNT(*) FROM lab_values WHERE patient_id=? GROUP BY document_id",
            (patient_id,)).fetchall())
        # Read every text once: the statement index needs all the narrative
        # documents of the patient before any document's input hash is fixed.
        texts, laboratory_documents = {}, {}
        for doc in documents:
            path = self.normalized_text_path(patient_id, doc.id)
            if path is None:
                plan.failures.append({"document_id": doc.id, "error": "Testo anonimizzato non disponibile."})
                continue
            base_text = path.read_text(encoding="utf-8")
            texts[doc.id] = (self.overlay_repo.effective_text(doc.id, base_text)
                             if self.overlay_repo else base_text)
            laboratory_documents[doc.id] = (
                doc.document_type == DocumentType.LABORATORIO.value
                and lab_counts.get(doc.id, 0) > 0)
        indexed = self.statements.rebuild(patient_id, [
            (doc.id, doc.document_date, texts[doc.id]) for doc in documents
            if doc.id in texts and not laboratory_documents[doc.id]], run_id=run.run_id)
        plan.statement_roles = {
            document_id: {occurrence.ordinal: occurrence.role for occurrence in occurrences}
            for document_id, occurrences in indexed.items()}
        for doc in documents:
            if doc.id not in texts:
                continue
            text = texts[doc.id]
            laboratory = laboratory_documents[doc.id]
            guidance = self.extractor.guidance_digest(text) if self.extractor else ""
            # The digest makes a document depend on which report carries its
            # statements: when an older report takes one over, its copies are
            # planned again even though their own text did not change.
            index_digest = "" if laboratory else self.statements.digest(doc.id)
            input_hash = content_hash(text, doc.document_date, doc.document_type,
                                      self.prompt_version, self.prompt_digest, guidance,
                                      PROJECTION_VERSION, index_digest,
                                      "deterministic-laboratory" if laboratory else "")
            if incremental and self.processing_repo.is_current(
                    doc.id, STAGE, input_hash, self.prompt_version,
                    self.prompt_version, model_digest):
                plan.skipped += 1
                continue
            item = ProcessingManifestItem(
                patient_id=patient_id, document_id=doc.id, stage=STAGE,
                input_hash=input_hash, pipeline_version=self.prompt_version,
                run_id=run.run_id, prompt_version=self.prompt_version,
                model_digest=model_digest, status="running",
            )
            self.processing_repo.upsert_manifest(item)
            if laboratory:
                # Tabular results are read by the deterministic parser; the
                # report is complete without a language-model reading.
                self.replace_document_evidence(doc.id, [])
                self.processing_repo.mark_result(item.manifest_id, status="completed",
                                                 output_hash=_evidence_hash([]), output_count=0)
                plan.laboratory_documents += 1
                continue
            plan.tasks.append((doc, text, item.manifest_id))
        return plan

    def _extract(self, plan, doc, text, cancel_check, report, total, completed):
        if cancel_check is not None and cancel_check():
            raise ExtractionCancelled("Elaborazione interrotta su richiesta dell'utente")
        task_started = time.monotonic()
        position = lambda: int(completed() / max(total, 1) * 100)
        try:
            evidence = self.extractor.extract_document(
                patient_id=plan.patient_id, document_id=doc.id,
                document_type=doc.document_type, document_date=doc.document_date,
                text=text, geometry_path=_geometry_source(doc, active_workspace.path),
                statement_roles=plan.statement_roles.get(doc.id),
                evidence_ready_callback=lambda rows: self.replace_document_evidence(doc.id, rows),
                cancel_check=cancel_check,
                stage_progress_callback=lambda message: report(position(), f"{plan.patient_id} · {doc.id}: {message}"),
                chunk_progress_callback=lambda done, all_: report(
                    position(), f"{plan.patient_id} · {doc.id}: gruppi {done}/{all_} verificati"),
            )
        except AtomicExtractionCancelled as exc:
            raise ExtractionCancelled(str(exc)) from exc
        except IncompleteAtomicExtraction as exc:
            self.replace_document_evidence(doc.id, exc.evidence)
            raise
        return doc, evidence, round(time.monotonic() - task_started, 3), \
            self.extractor.last_extraction_metrics()

    def _persist(self, plan, manifest_id, result) -> None:
        doc, evidence, duration, metrics = result
        self.replace_document_evidence(doc.id, evidence)
        self.processing_repo.mark_result(manifest_id, status="completed",
                                         output_hash=_evidence_hash(evidence), output_count=len(evidence))
        plan.processed += 1
        plan.extracted += len(evidence)
        plan.doc_stats.append({"document_id": doc.id, "evidence_count": len(evidence),
                               "elapsed_seconds": duration, "status": "completed", **metrics})

    def _fail(self, plan, doc, manifest_id, exc) -> None:
        plan.failures.append({"document_id": doc.id, "error": str(exc)})
        incomplete = isinstance(exc, IncompleteAtomicExtraction)
        plan.doc_stats.append({"document_id": doc.id, "status": "failed", "error": str(exc),
                               "evidence_count": len(exc.evidence) if incomplete else 0,
                               **(exc.metrics if incomplete else {})})
        self.processing_repo.mark_result(manifest_id, status="failed", error_message=str(exc))

    def _finalize(self, plan, check_cancelled) -> dict:
        check_cancelled()
        cancel = lambda: (check_cancelled() or False)
        coding = self.code_patient(plan.patient_id, cancel_check=cancel)
        fhir = self.write_fhir(plan.patient_id, plan.documents, {
            "documents_total": len(plan.documents), "processed": plan.processed,
            "unchanged": plan.skipped, "laboratory_documents": plan.laboratory_documents,
            "failures": plan.failures, "complete": not plan.failures, "pipeline": VERSION,
        }, cancel_check=check_cancelled)
        stored = self.evidence_repo.get_by_patient(plan.patient_id)
        elapsed = round(time.monotonic() - plan.started, 2)
        self.processing_repo.finish_run(
            plan.run.run_id, "completed_with_warnings" if plan.failures else "completed")
        summary = _summarize(plan.doc_stats)
        counts = self.statements.counts(plan.patient_id)
        summary.update(statement_occurrences=counts["occurrences"],
                       statement_origins=counts["origins"], statement_copies=counts["copies"],
                       statement_reused_characters=counts["copy_chars"])
        if self.audit:
            self.audit.log(plan.patient_id, "clinical_events_extracted", "clinical_evidence",
                           plan.patient_id, {"documents_total": len(plan.documents),
                                             "documents_processed": plan.processed,
                                             "documents_skipped": plan.skipped,
                                             "laboratory_documents": plan.laboratory_documents,
                                             "documents_failed": len(plan.failures),
                                             "evidence_stored_occurrences": len(stored),
                                             "elapsed_seconds": elapsed, **summary},
                           model_used=getattr(self.llm, "model", None),
                           model_version=self.prompt_version, run_id=plan.run.run_id)
        return {
            "fhir_path": fhir["path"], "fhir_events": fhir["events"], "fhir_uncoded": fhir["uncoded"],
            "total_entries": len(deduplicate_atomic_evidence(stored)),
            "stored_evidence_occurrences": len(stored),
            "documents_processed": plan.processed, "documents_skipped": plan.skipped,
            "laboratory_documents": plan.laboratory_documents,
            "documents_failed": len(plan.failures),
            "failed_doc_ids": [item["document_id"] for item in plan.failures],
            "failures": plan.failures,
            "document_stats": sorted(plan.doc_stats, key=lambda item: item["document_id"]),
            "atomic_evidence_extracted": plan.extracted,
            "atomic_model": getattr(self.llm, "model", None),
            "run_id": plan.run.run_id, "elapsed_seconds": elapsed, "coding": coding, **summary,
        }

    def build_fhir(self, patient_id, documents=None, *, cancel_check=None, use_llm=True):
        """FHIR registry of the patient's reviewed events and parsed results.

        ``use_llm=False`` never calls the model (cached LOINC proposals only).
        """
        from .event_review import to_evidence
        from .fhir_registry import FhirRegistry, lab_occurrence_keys
        if documents is None:
            documents = self.document_repo.list_by_patient(patient_id)
        loinc = getattr(self.shared_lexicon_repo, "loinc_catalog", None)
        lab_values = self.lab_repo.get_by_patient(patient_id)
        proposals = {}
        if loinc:
            llm = self.llm if (use_llm and self.llm and getattr(self.llm, "is_available", False)) else None
            proposals = loinc.propose(lab_values, llm,
                                      lambda: (cancel_check() if cancel_check else None) or False)
        exporter = FhirRegistry(patient_id, loinc, proposals)
        for document in documents:
            exporter.document(document.id)
        for event in self.events(patient_id):
            exporter.clinical(to_evidence(event), event.coding, resource_key=event.key)
        for lab, key in zip(lab_values, lab_occurrence_keys(lab_values)):
            exporter.laboratory(lab, key)
        return exporter

    def write_fhir(self, patient_id, documents, coverage, *, cancel_check=None, use_llm=True) -> dict:
        exporter = self.build_fhir(patient_id, documents, cancel_check=cancel_check, use_llm=use_llm)
        path = exporter.write(active_workspace.path / patient_id / FHIR_FILENAME, coverage)
        return {"path": path, "events": exporter.event_count, "uncoded": exporter.unmapped}

    def coverage(self, patient_id: str) -> dict:
        status = self.patient_status(patient_id)
        return {"documents_total": status["documents"], "documents_with_text": status["with_text"],
                "processed": status["completed"], "failed": status["failed"],
                "complete": status["failed"] == 0 and status["completed"] == status["with_text"],
                "pipeline": VERSION}

    def export_patient(self, patient_id: str, *, use_llm: bool = True) -> dict:
        """Rewrite the patient's FHIR file from the current reviewed state."""
        return self.write_fhir(patient_id, None, self.coverage(patient_id), use_llm=use_llm)

    def export_project(self, patient_ids=None, *, target=None, progress=None,
                       cancel_check=None) -> dict:
        """One NDJSON file (one FHIR resource per line) for the whole project.

        Each patient's Bundle is validated and its own file refreshed too.
        """
        from datetime import datetime
        from .fhir_registry import validate_bundle, write_atomic
        if patient_ids is None:
            patient_ids = [row[0] for row in self.db.execute("SELECT id FROM patients ORDER BY id")]
        lines, seen = [], set()
        totals = {"patients": 0, "events": 0, "uncoded": 0}
        for index, patient_id in enumerate(patient_ids, start=1):
            if cancel_check and cancel_check():
                raise ExtractionCancelled("Esportazione interrotta")
            if progress:
                progress(int((index - 1) * 100 / max(len(patient_ids), 1)), f"Esportazione {patient_id}")
            exporter = self.build_fhir(patient_id)
            bundle = exporter.bundle(self.coverage(patient_id))
            validate_bundle(bundle)
            write_atomic(active_workspace.path / patient_id / FHIR_FILENAME,
                         json.dumps(bundle, ensure_ascii=False, indent=2, allow_nan=False))
            for entry in bundle["entry"]:
                resource = entry["resource"]
                identity = (resource["resourceType"], resource["id"])
                if identity not in seen:
                    seen.add(identity)
                    lines.append(json.dumps(resource, ensure_ascii=False, allow_nan=False))
            totals["patients"] += 1
            totals["events"] += exporter.event_count
            totals["uncoded"] += exporter.unmapped
        if target is None:
            stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            target = active_workspace.path / "exports" / "fhir" / f"progetto-{stamp}.ndjson"
        path = write_atomic(target, "\n".join(lines) + ("\n" if lines else ""))
        if progress:
            progress(100, "Esportazione completata")
        return {"path": path, "resources": len(lines), **totals}

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
        from .document_text import normalized_text_path
        return normalized_text_path(patient_id, document_id)

    def _retain_only_extraction_runtime(self) -> int:
        reserve = getattr(self.llm, "retain_only_this_runtime", None)
        try:
            return int(reserve() or 0) if callable(reserve) else 0
        except Exception:
            return 0

    def replace_document_evidence(self, document_id: str, evidence) -> None:
        """Replace this document's extracted events; legacy atoms are cleared."""
        evidence, _ = filter_narrative_lab_duplicates(
            list(evidence), self.lab_repo.get_by_document(document_id))
        grouped: dict[str, list] = {
            METHOD: [], "shared_lexicon_atomic": [], "icd11_extraction": [],
            "llm_atomic_v2": [], "deterministic_nonclinical": [],
        }
        for item in evidence:
            grouped.setdefault(item.extraction_method, []).append(item)
        for method, items in grouped.items():
            self.evidence_repo.replace_document_method(document_id, method, items)


def _eta_message(completed: int, total: int, elapsed: float, patient_id: str) -> str:
    message = f"Documenti {completed}/{total} · ultimo paziente {patient_id}"
    if completed and total > completed and elapsed > 0:
        remaining = elapsed / completed * (total - completed)
        hours, minutes = divmod(int(remaining // 60), 60)
        message += (f" · {completed / elapsed * 60:.1f} documenti/min · fine stimata tra "
                    + (f"{hours} h {minutes} min" if hours else f"{minutes} min"))
    return message


def _summarize(doc_stats) -> dict:
    total = lambda key, cast=int: sum(cast(item.get(key, 0) or 0) for item in doc_stats)
    keys = ("llm_calls", "source_chunks", "prompt_tokens", "completion_tokens",
            "recovery_calls", "snomed_mapped", "snomed_unmapped",
            "statement_groups_skipped", "statement_copy_events_ignored")
    summary = {key: total(key) for key in keys}
    summary.update(prompt_ms=round(total("prompt_ms", float), 3),
                   predicted_ms=round(total("predicted_ms", float), 3),
                   llm_seconds=round(total("elapsed_seconds", float), 3))
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

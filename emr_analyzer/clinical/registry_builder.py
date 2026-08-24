"""Orchestrator for the versioned evidence → event → episode registry."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from difflib import SequenceMatcher
import hashlib
import json
from pathlib import Path
import time
from typing import Callable, Optional

from .atomic_evidence import (
    ATOMIC_PIPELINE_VERSION,
    ATOMIC_PROMPT_DIGEST,
    ATOMIC_PROMPT_VERSION,
    ATOMIC_RESUME_COMPATIBLE_PIPELINE_VERSIONS,
    AtomicExtractionCancelled,
    AtomicEvidenceExtractor,
    content_hash,
    deduplicate_atomic_evidence,
    locate_quote,
)
from .block_reuse import (
    build_targeted_reuse_text,
    clone_reused_evidence,
    evidence_matching_reuse_block,
    plan_exact_block_reuse,
    plan_reuse_verification_batches,
)
from .consolidation import ClinicalConsolidator, stable_id
from .evidence_graph import EvidenceGraphBuilder, EvidenceGraphCancelled
from .episode_assembler import build_event_relations
from .episode_synthesis import EpisodeSynthesisStats
from .event_dedup import deduplicate_bundles
from .evidence_relevance import (
    annotate_evidence_disposition,
    is_administrative_mapping,
    partition_evidence,
)
from .lab_evidence import (
    LAB_EXTRACTION_METHOD,
    abnormal_lab_evidence,
    load_document_geometry,
)
from .projections import LabTrendBuilder, TherapyProjectionBuilder
from ..config import active_workspace
from ..database.clinical_state_repo import ClinicalStateRepository
from ..models.clinical_registry import (
    EventEvidenceLink,
    ProcessingManifestItem,
    ProcessingRun,
)
from ..models.clinical_timeline import ClinicalTimelineEntry
from ..models.clinical_pipeline import (
    EventClaim,
    EvidenceSourceReference,
    ExcludedEvidence,
)
from ..settings import load_pipeline_policy


class RegistryBuildCancelled(RuntimeError):
    """Safe cooperative cancellation of a registry build."""


class ClinicalRegistryBuilder:
    """Incremental, order-independent construction of the clinical registry."""

    def __init__(
        self,
        *,
        registry_repo,
        evidence_repo,
        processing_repo,
        timeline_repo,
        document_repo,
        lab_repo,
        overlay_repo,
        llm_client=None,
        atomic_llm_client=None,
        event_llm_client=None,
        audit_repo=None,
        pipeline_repo=None,
        pipeline_policy=None,
        db=None,
    ):
        self.registry_repo = registry_repo
        self.evidence_repo = evidence_repo
        self.processing_repo = processing_repo
        self.timeline_repo = timeline_repo
        self.document_repo = document_repo
        self.lab_repo = lab_repo
        self.overlay_repo = overlay_repo
        # ``llm_client`` is the compatibility path for callers created before
        # the registry had independent extraction and event models.
        self.atomic_llm = (
            atomic_llm_client if atomic_llm_client is not None else llm_client
        )
        self.event_llm = (
            event_llm_client if event_llm_client is not None else llm_client
        )
        # Public compatibility alias: historically ``llm`` drove both stages;
        # it now denotes the event/episode model only.
        self.llm = self.event_llm
        self.audit = audit_repo
        self.pipeline_repo = pipeline_repo
        self.pipeline_policy = pipeline_policy or load_pipeline_policy()
        self.db = db or timeline_repo.db
        self.atomic_extractor = (
            AtomicEvidenceExtractor(self.atomic_llm, policy=self.pipeline_policy)
            if self.atomic_llm else None
        )

    def reload_policy(self) -> None:
        """Reload non-clinical settings for subsequent registry builds."""
        self.pipeline_policy = load_pipeline_policy()
        if self.atomic_extractor is not None:
            self.atomic_extractor.policy = self.pipeline_policy

    def has_atomic_checkpoint(self, patient_id: str) -> bool:
        """Return whether an interrupted/incremental build can be resumed.

        A registry can still contain zero final events while hundreds of
        document-level extractions are already durable.  The GUI must not use
        the absence of final timeline rows as a reason to discard that work.
        Exact input/prompt/model compatibility is rechecked later by build().
        """
        row = self.db.execute(
            """SELECT 1 FROM processing_manifest
               WHERE patient_id=? AND stage='atomic_evidence'
                 AND status='completed'
               LIMIT 1""",
            (patient_id,),
        ).fetchone()
        return row is not None

    def build(
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
        run = ProcessingRun(
            patient_id=patient_id,
            stage="clinical_registry_v3",
            model_name=(
                self.atomic_extractor.model_name if self.atomic_extractor else None
            ),
            model_digest=(
                self.atomic_extractor.model_digest if self.atomic_extractor else None
            ),
            prompt_version=ATOMIC_PROMPT_VERSION,
            parameters={
                "incremental": incremental,
                "requested_workers": num_workers,
                "document_count": len(documents),
                "atomic_prompt_digest": ATOMIC_PROMPT_DIGEST,
                "atomic_model": getattr(self.atomic_llm, "model", None),
                "event_model": getattr(self.event_llm, "model", None),
            },
        )
        self.processing_repo.start_run(run)
        failures: list[dict] = []
        skipped = 0
        processed = 0
        extracted_count = 0
        abnormal_lab_evidence_count = 0

        def check_cancelled() -> None:
            if cancel_check is not None and cancel_check():
                raise RegistryBuildCancelled(
                    "Elaborazione interrotta su richiesta dell'utente"
                )

        def extract_atomic_document(**kwargs):
            check_cancelled()
            try:
                evidence = self.atomic_extractor.extract_document(
                    **kwargs, cancel_check=cancel_check,
                )
            except AtomicExtractionCancelled as exc:
                raise RegistryBuildCancelled(str(exc)) from exc
            check_cancelled()
            return evidence

        try:
            check_cancelled()
            abnormal_lab_evidence_count = self._sync_abnormal_lab_evidence(
                patient_id, documents
            )
            check_cancelled()
            tasks = []
            reuse_rows = []
            for doc in documents:
                if doc.document_type == "laboratorio":
                    # Deterministic laboratory evidence is already populated
                    # during document extraction and needs no narrative LLM.
                    continue
                path = self._normalized_text_path(patient_id, doc.id)
                if path is None:
                    continue
                base_text = path.read_text(encoding="utf-8")
                effective_text = self.overlay_repo.effective_text(
                    doc.id, base_text
                ) if self.overlay_repo else base_text
                input_hash = content_hash(
                    effective_text, doc.document_date, doc.document_type,
                    ATOMIC_PROMPT_VERSION, ATOMIC_PROMPT_DIGEST,
                )
                # Current documents remain eligible as canonical sources for
                # exact-block reuse by new/interrupted targets.  They are not
                # sent to the LLM again unless focused verification discovers
                # a previously unmapped clinical block.
                reuse_rows.append((doc, effective_text, input_hash))
                current = incremental and self.processing_repo.is_current(
                    doc.id, "atomic_evidence", input_hash,
                    ATOMIC_PIPELINE_VERSION, ATOMIC_PROMPT_VERSION,
                    run.model_digest or "",
                    compatible_pipeline_versions=(
                        ATOMIC_RESUME_COMPATIBLE_PIPELINE_VERSIONS
                    ),
                )
                if current:
                    skipped += 1
                    continue
                tasks.append((doc, effective_text, input_hash))

            atomic_llm_available = bool(
                self.atomic_extractor is not None and self.atomic_llm
                and getattr(self.atomic_llm, "is_available", False)
            )
            event_llm_available = bool(
                self.event_llm
                and getattr(self.event_llm, "is_available", False)
            )
            model_unavailable = bool(tasks) and not atomic_llm_available
            if model_unavailable:
                message = (
                    "Modello locale non disponibile: consolidate le evidenze "
                    "già presenti e di laboratorio; i documenti narrativi "
                    "verranno riprovati alla prossima esecuzione."
                )
                failures.extend(
                    {"document_id": doc.id, "error": message}
                    for doc, _, _ in tasks
                )
                tasks = []

            reuse_plans = plan_exact_block_reuse(reuse_rows)
            task_ids = {doc.id for doc, _, _ in tasks}
            reused_blocks = sum(
                len(plan.reuse_links)
                for doc_id, plan in reuse_plans.items()
                if doc_id in task_ids
            )
            reused_chars = sum(
                plan.original_chars - plan.extraction_chars
                for doc_id, plan in reuse_plans.items()
                if doc_id in task_ids
            )

            configured_workers = max(1, int(num_workers or 1))
            event_workers = max(1, int(getattr(
                self.event_llm, "parallel_workers", configured_workers
            ) or configured_workers))
            actual_workers = max(1, min(configured_workers, len(tasks) or 1))
            if progress_callback:
                progress_callback(
                    0, f"Evidenze atomiche: {len(tasks)} documenti da elaborare, "
                    f"{skipped} invariati"
                )

            manifests = {}
            for doc, _, input_hash in tasks:
                item = ProcessingManifestItem(
                    patient_id=patient_id, document_id=doc.id,
                    stage="atomic_evidence", input_hash=input_hash,
                    pipeline_version=ATOMIC_PIPELINE_VERSION,
                    run_id=run.run_id, prompt_version=ATOMIC_PROMPT_VERSION,
                    model_digest=run.model_digest or "", status="running",
                )
                self.processing_repo.upsert_manifest(item)
                manifests[doc.id] = item.manifest_id

            def extract_one(doc, text, input_hash):
                task_started = time.monotonic()
                geometry_path = (
                    active_workspace.path / patient_id / "extraction" /
                    f"{doc.id}.json"
                )
                evidence = extract_atomic_document(
                    patient_id=patient_id, document_id=doc.id,
                    document_type=doc.document_type,
                    document_date=doc.document_date, text=text,
                    geometry_path=geometry_path,
                )
                metrics = self.atomic_extractor.last_extraction_metrics()
                return (
                    doc, evidence, input_hash,
                    round(time.monotonic() - task_started, 3), metrics,
                )

            partial_results: dict[str, tuple] = {}

            def persist_success(result, *, final: bool = False) -> None:
                """Commit one completed document before scheduling moves on.

                Evidence identifiers are content-derived, so database row
                order has no semantic effect.  Persisting in completion order
                makes a long run resumable after a crash and exposes truthful
                progress in the processing manifest.
                """
                nonlocal processed, extracted_count
                doc, evidence, _, duration, metrics = result
                self._replace_atomic_document_evidence(doc.id, evidence)
                plan = reuse_plans.get(doc.id)
                if plan and plan.reuse_links and not final:
                    # The novel portion is durable already, but the manifest
                    # remains running until every reused source has been
                    # materialized with this document's provenance.
                    partial_results[doc.id] = result
                    return
                output_hash = _evidence_hash(evidence)
                self.processing_repo.mark_result(
                    manifests[doc.id], status="completed",
                    output_hash=output_hash, output_count=len(evidence),
                )
                processed += 1
                extracted_count += len(evidence)
                doc_stats.append({
                    "document_id": doc.id,
                    "evidence_count": len(evidence),
                    "elapsed_seconds": duration,
                    "status": "completed",
                    **metrics,
                })

            doc_stats: list[dict] = []
            initial_reuse_errors: dict[str, str] = {}

            extraction_tasks = []
            for doc, original_text, input_hash in tasks:
                plan = reuse_plans.get(doc.id)
                extraction_text = (
                    plan.extraction_text if plan is not None else original_text
                )
                if extraction_text.strip():
                    extraction_tasks.append(
                        (doc, extraction_text, input_hash)
                    )
                else:
                    partial_results[doc.id] = (
                        doc, [], input_hash, 0.0,
                        {
                            "llm_calls": 0,
                            "output_limit_retries": 0,
                            "source_chunks": 0,
                            "prompt_tokens": 0,
                            "completion_tokens": 0,
                            "total_tokens": 0,
                            "prompt_ms": 0.0,
                            "predicted_ms": 0.0,
                        },
                    )

            # Longest-processing-time first is the standard way to reduce
            # the final straggler with identical workers.  It changes only
            # scheduling: persistence and reused-block materialization remain
            # deterministic and chronological.
            extraction_tasks.sort(key=lambda task: len(task[1]), reverse=True)

            if actual_workers == 1:
                for index, task in enumerate(extraction_tasks, start=1):
                    doc = task[0]
                    try:
                        persist_success(extract_one(*task))
                    except RegistryBuildCancelled:
                        raise
                    except Exception as exc:
                        plan = reuse_plans.get(doc.id)
                        if plan and plan.reuse_links:
                            initial_reuse_errors[doc.id] = str(exc)
                        else:
                            failures.append({
                                "document_id": doc.id, "error": str(exc)
                            })
                            doc_stats.append({
                                "document_id": doc.id,
                                "evidence_count": 0,
                                "status": "failed",
                                "error": str(exc),
                            })
                            self.processing_repo.mark_result(
                                manifests[doc.id], status="failed",
                                error_message=str(exc),
                            )
                    if progress_callback:
                        progress_callback(
                            int(index / max(len(extraction_tasks), 1) * 50),
                            f"Evidenze {index}/{len(extraction_tasks)}",
                        )
            else:
                with ThreadPoolExecutor(max_workers=actual_workers) as pool:
                    future_map = {
                        pool.submit(extract_one, *task): task[0]
                        for task in extraction_tasks
                    }
                    completed = 0
                    for future in as_completed(future_map):
                        completed += 1
                        doc = future_map[future]
                        try:
                            persist_success(future.result())
                        except RegistryBuildCancelled:
                            for pending_future in future_map:
                                pending_future.cancel()
                            raise
                        except Exception as exc:
                            plan = reuse_plans.get(doc.id)
                            if plan and plan.reuse_links:
                                initial_reuse_errors[doc.id] = str(exc)
                            else:
                                failures.append({
                                    "document_id": doc.id, "error": str(exc)
                                })
                                self.processing_repo.mark_result(
                                    manifests[doc.id], status="failed",
                                    error_message=str(exc),
                                )
                                doc_stats.append({
                                    "document_id": doc.id,
                                    "evidence_count": 0,
                                    "status": "failed",
                                    "error": str(exc),
                                })
                        if progress_callback:
                            progress_callback(
                                int(completed / max(len(extraction_tasks), 1) * 50),
                                f"Evidenze {completed}/{len(extraction_tasks)}",
                            )

            # Materialize exact repeated blocks.  Missing mappings are not a
            # reason to re-read every target document: verify each unique
            # source fingerprint once, then use a small target-specific pass
            # only when its citation cannot be transferred.  Full-document
            # extraction remains the final safety net for actual failures.
            reused_evidence_count = 0
            fallback_count = 0
            targeted_verification_count = 0
            unique_blocks_verified = 0
            reuse_verification_batches = 0
            documents_by_id = {doc.id: doc for doc in documents}
            source_text_by_doc = {
                doc.id: text for doc, text, _ in reuse_rows
            }
            task_by_id = {
                doc.id: (doc, original_text, input_hash)
                for doc, original_text, input_hash in tasks
            }
            combined_by_doc: dict[str, list] = {}
            unresolved_by_doc: dict[str, list] = {}
            forced_full_doc_ids: set[str] = set()

            for doc, original_text, input_hash in tasks:
                plan = reuse_plans.get(doc.id)
                if not plan or not plan.reuse_links:
                    continue
                partial = partial_results.get(doc.id)
                if partial is None:
                    forced_full_doc_ids.add(doc.id)
                    continue
                combined = list(partial[1])
                unresolved_links = []
                target_geometry = self.atomic_extractor._load_geometry(
                    active_workspace.path / patient_id / "extraction" /
                    f"{doc.id}.json"
                )
                for link in plan.reuse_links:
                    source_items = [
                        item for item in self.evidence_repo.get_by_document(
                            link.source_document_id
                        )
                        if item.extraction_method == "llm_atomic_v2"
                    ]
                    clones = clone_reused_evidence(
                        source_items,
                        link,
                        patient_id=patient_id,
                        target_document_date=doc.document_date,
                        target_full_text=original_text,
                        target_geometry=target_geometry,
                    )
                    if not clones:
                        unresolved_links.append(link)
                        continue
                    before = len(deduplicate_atomic_evidence(combined))
                    combined = deduplicate_atomic_evidence((*combined, *clones))
                    after = len(combined)
                    reused_evidence_count += max(0, after - before)
                combined_by_doc[doc.id] = combined
                if unresolved_links:
                    unresolved_by_doc[doc.id] = unresolved_links

            unresolved_links = [
                link for links in unresolved_by_doc.values() for link in links
            ]
            verification_batches = plan_reuse_verification_batches(
                unresolved_links
            )
            reuse_verification_batches = len(verification_batches)
            verified_by_fingerprint: dict[str, list] = {}
            source_additions: dict[str, list] = {}
            verification_errors: dict[str, str] = {}

            def accumulate_metrics(total: dict, current: dict) -> None:
                for key, value in current.items():
                    if isinstance(value, (int, float)):
                        total[key] = total.get(key, 0) + value
                    else:
                        total[key] = value

            def verify_source_batch(batch):
                source_doc = documents_by_id[batch.source_document_id]
                source_original = source_text_by_doc[source_doc.id]
                started_batch = time.monotonic()
                geometry_path = (
                    active_workspace.path / patient_id / "extraction" /
                    f"{source_doc.id}.json"
                )
                metrics: dict = {}

                def extract_focused(text: str):
                    extracted = extract_atomic_document(
                        patient_id=patient_id,
                        document_id=source_doc.id,
                        document_type=source_doc.document_type,
                        document_date=source_doc.document_date,
                        text=text,
                        geometry_path=geometry_path,
                    )
                    accumulate_metrics(
                        metrics,
                        self.atomic_extractor.last_extraction_metrics(),
                    )
                    grounded = [
                        item for item in extracted
                        if locate_quote(item.source_text, source_original)[0]
                    ]
                    for item in grounded:
                        item.data["reuse_block_verification"] = True
                    return extracted, grounded

                grouped_error = None
                try:
                    grouped, grouped_grounded = extract_focused(batch.text)
                except RegistryBuildCancelled:
                    raise
                except Exception as exc:
                    grouped, grouped_grounded = [], []
                    grouped_error = str(exc)

                resolved: dict[str, list] = {}
                errors: dict[str, str] = {}
                additions = []
                for link in batch.links:
                    raw_matches = evidence_matching_reuse_block(grouped, link)
                    matches = evidence_matching_reuse_block(
                        grouped_grounded, link
                    )
                    needs_individual = bool(
                        grouped_error or (raw_matches and not matches)
                    )
                    if needs_individual:
                        individual_text = "\n\n".join(
                            value for value in (
                                link.source_section.strip(),
                                link.source_text.strip(),
                            ) if value
                        )
                        try:
                            individual, individual_grounded = extract_focused(
                                individual_text
                            )
                            raw_matches = evidence_matching_reuse_block(
                                individual, link
                            )
                            matches = evidence_matching_reuse_block(
                                individual_grounded, link
                            )
                        except RegistryBuildCancelled:
                            raise
                        except Exception as exc:
                            errors[link.fingerprint] = str(exc)
                            continue
                    if raw_matches and not matches:
                        errors[link.fingerprint] = (
                            "Citazione del blocco non localizzabile nel "
                            "documento sorgente"
                        )
                        continue
                    resolved[link.fingerprint] = matches
                    additions.extend(matches)
                return (
                    batch, resolved, errors, additions,
                    round(time.monotonic() - started_batch, 3), metrics,
                )

            def accept_source_verification(result) -> None:
                nonlocal unique_blocks_verified
                batch, resolved, errors, additions, duration, metrics = result
                verified_by_fingerprint.update(resolved)
                verification_errors.update(errors)
                unique_blocks_verified += len(resolved)
                stored_additions = source_additions.setdefault(
                    batch.source_document_id, []
                )
                stored_additions.extend(additions)
                doc_stats.append({
                    "document_id": batch.source_document_id,
                    "evidence_count": len(
                        deduplicate_atomic_evidence(additions)
                    ),
                    "elapsed_seconds": duration,
                    "status": "reuse_source_verification",
                    **metrics,
                })

            if verification_batches:
                ordered_batches = sorted(
                    verification_batches,
                    key=lambda batch: len(batch.text), reverse=True,
                )
                with ThreadPoolExecutor(max_workers=actual_workers) as pool:
                    future_map = {
                        pool.submit(verify_source_batch, batch): batch
                        for batch in ordered_batches
                    }
                    completed = 0
                    for future in as_completed(future_map):
                        completed += 1
                        batch = future_map[future]
                        try:
                            accept_source_verification(future.result())
                        except RegistryBuildCancelled:
                            for pending_future in future_map:
                                pending_future.cancel()
                            raise
                        except Exception as exc:
                            for link in batch.links:
                                verification_errors[link.fingerprint] = str(exc)
                        if progress_callback:
                            progress_callback(
                                50 + int(
                                    5 * completed /
                                    max(len(ordered_batches), 1)
                                ),
                                "Verifica blocchi unici "
                                f"{completed}/{len(ordered_batches)}",
                            )

            # New evidence discovered by the focused verifier belongs to the
            # first source document too; otherwise an earlier first-evidence
            # date would be lost.  Merge it before cloning target occurrences.
            for source_doc_id, additions in source_additions.items():
                if not additions:
                    continue
                current = [
                    item for item in self.evidence_repo.get_by_document(
                        source_doc_id
                    ) if item.extraction_method == "llm_atomic_v2"
                ]
                merged_items = deduplicate_atomic_evidence(
                    (*current, *additions)
                )
                self._replace_atomic_document_evidence(
                    source_doc_id, merged_items, clear_missing=False
                )
                partial = partial_results.get(source_doc_id)
                if partial is not None:
                    partial_results[source_doc_id] = (
                        partial[0], merged_items, partial[2],
                        partial[3], partial[4],
                    )
                if source_doc_id in combined_by_doc:
                    combined_by_doc[source_doc_id] = (
                        deduplicate_atomic_evidence((
                            *combined_by_doc[source_doc_id], *additions,
                        ))
                    )
                elif (
                    source_doc_id in manifests
                    and source_doc_id not in forced_full_doc_ids
                ):
                    self.processing_repo.mark_result(
                        manifests[source_doc_id], status="completed",
                        output_hash=_evidence_hash(merged_items),
                        output_count=len(merged_items),
                    )

            target_checks: dict[str, list] = {}
            for doc_id, links in unresolved_by_doc.items():
                doc, original_text, _ = task_by_id[doc_id]
                target_geometry = self.atomic_extractor._load_geometry(
                    active_workspace.path / patient_id / "extraction" /
                    f"{doc.id}.json"
                )
                combined = combined_by_doc[doc_id]
                for link in links:
                    templates = verified_by_fingerprint.get(link.fingerprint)
                    if link.fingerprint in verification_errors:
                        target_checks.setdefault(doc_id, []).append(link)
                        continue
                    if not templates:
                        # A successful focused pass with no evidence is a
                        # verified-empty repeated block.
                        continue
                    clones = clone_reused_evidence(
                        templates, link, patient_id=patient_id,
                        target_document_date=doc.document_date,
                        target_full_text=original_text,
                        target_geometry=target_geometry,
                    )
                    if not clones:
                        target_checks.setdefault(doc_id, []).append(link)
                        continue
                    before = len(deduplicate_atomic_evidence(combined))
                    combined = deduplicate_atomic_evidence((*combined, *clones))
                    combined_by_doc[doc_id] = combined
                    after = len(combined)
                    reused_evidence_count += max(0, after - before)

            def merge_metrics(*payloads):
                merged = {}
                for payload in payloads:
                    for key, value in payload.items():
                        if isinstance(value, (int, float)):
                            merged[key] = merged.get(key, 0) + value
                        else:
                            merged[key] = value
                return merged

            def verify_target_document(doc_id: str):
                doc, original_text, input_hash = task_by_id[doc_id]
                if doc_id in forced_full_doc_ids:
                    return "full", extract_one(doc, original_text, input_hash)
                links = target_checks[doc_id]
                segment = build_targeted_reuse_text(links)
                started_segment = time.monotonic()
                geometry_path = (
                    active_workspace.path / patient_id / "extraction" /
                    f"{doc.id}.json"
                )
                try:
                    evidence = extract_atomic_document(
                        patient_id=patient_id, document_id=doc.id,
                        document_type=doc.document_type,
                        document_date=doc.document_date, text=segment,
                        geometry_path=geometry_path,
                    )
                    metrics = self.atomic_extractor.last_extraction_metrics()
                    grounded = [
                        item for item in evidence
                        if locate_quote(item.source_text, original_text)[0]
                    ]
                    ambiguous = any(
                        evidence_matching_reuse_block(evidence, link)
                        and not evidence_matching_reuse_block(grounded, link)
                        for link in links
                    )
                    if ambiguous:
                        raise RuntimeError(
                            "Citazione del segmento non localizzabile nel documento"
                        )
                    for item in grounded:
                        item.data["reuse_block_target_verification"] = True
                    partial = partial_results[doc.id]
                    final_items = deduplicate_atomic_evidence((
                        *combined_by_doc[doc.id], *grounded,
                    ))
                    return "targeted", (
                        doc, final_items, input_hash,
                        partial[3] + round(
                            time.monotonic() - started_segment, 3
                        ),
                        merge_metrics(partial[4], metrics),
                    )
                except RegistryBuildCancelled:
                    raise
                except Exception:
                    return "full", extract_one(doc, original_text, input_hash)

            pending_target_ids = set(target_checks) | forced_full_doc_ids
            if pending_target_ids:
                ordered_target_ids = sorted(
                    pending_target_ids,
                    key=lambda doc_id: len(task_by_id[doc_id][1]),
                    reverse=True,
                )
                with ThreadPoolExecutor(max_workers=actual_workers) as pool:
                    future_map = {
                        pool.submit(verify_target_document, doc_id): doc_id
                        for doc_id in ordered_target_ids
                    }
                    completed = 0
                    for future in as_completed(future_map):
                        completed += 1
                        doc_id = future_map[future]
                        try:
                            mode, result = future.result()
                            if mode == "full":
                                fallback_count += 1
                            else:
                                targeted_verification_count += 1
                            persist_success(result, final=True)
                        except RegistryBuildCancelled:
                            for pending_future in future_map:
                                pending_future.cancel()
                            raise
                        except Exception as exc:
                            error = str(exc)
                            if doc_id in initial_reuse_errors:
                                error = (
                                    "Estrazione ridotta: "
                                    f"{initial_reuse_errors[doc_id]}; "
                                    f"fallback completo: {error}"
                                )
                            failures.append({
                                "document_id": doc_id, "error": error,
                            })
                            self.processing_repo.mark_result(
                                manifests[doc_id], status="failed",
                                error_message=error,
                            )
                        if progress_callback:
                            progress_callback(
                                55 + int(
                                    5 * completed /
                                    max(len(ordered_target_ids), 1)
                                ),
                                "Verifica mirata documenti "
                                f"{completed}/{len(ordered_target_ids)}",
                            )

            # Documents fully resolved by source templates need no second LLM
            # call.  Persist them only after source additions have been merged.
            for doc_id, combined in combined_by_doc.items():
                if doc_id in pending_target_ids:
                    continue
                partial = partial_results[doc_id]
                final_evidence = deduplicate_atomic_evidence(combined)
                persist_success((
                    partial[0], final_evidence, partial[2],
                    partial[3], partial[4],
                ), final=True)

            check_cancelled()
            if progress_callback:
                progress_callback(60, "Ricostruzione eventi ed episodi...")

            stored_evidence = self.evidence_repo.get_by_patient(patient_id)
            document_dates = {doc.id: doc.document_date for doc in documents}
            for item in stored_evidence:
                if not item.document_date:
                    item.document_date = document_dates.get(item.document_id)
                if not item.date_precision and item.observed_date:
                    item.date_precision = (
                        "day" if len(item.observed_date) == 10
                        else "month" if len(item.observed_date) == 7
                        else "year" if len(item.observed_date) == 4
                        else "unknown"
                    )

            # Materialize the non-clinical projection separately.  It remains
            # inspectable (with exact source and exclusion reason) but cannot
            # reach event construction, summaries or LLM analysis contexts.
            if self.pipeline_repo is not None:
                excluded_by_document: dict[str, list[ExcludedEvidence]] = {}
                for item in stored_evidence:
                    disposition = annotate_evidence_disposition(item)
                    if disposition.registry_eligible:
                        continue
                    excluded_by_document.setdefault(
                        item.document_id, []
                    ).append(_excluded_record(item, disposition.reason))
                for document in documents:
                    self.pipeline_repo.replace_excluded_document(
                        document.id,
                        excluded_by_document.get(document.id, []),
                    )

            # One clinical fact can be printed verbatim in many subsequent
            # reports.  Collapse those document occurrences before *any*
            # event, trend, episode or LLM projection sees them.  The raw rows
            # remain immutable and are reattached later only as source copies.
            source_evidence = deduplicate_atomic_evidence(stored_evidence)
            if self.pipeline_repo is not None:
                source_evidence = _apply_reviewed_duplicate_groups(
                    source_evidence,
                    self.pipeline_repo.list_accepted_duplicate_groups(patient_id),
                )
            atomic_duplicates_suppressed = max(
                0, len(stored_evidence) - len(source_evidence)
            )
            if self.pipeline_repo is not None:
                for item in source_evidence:
                    self.pipeline_repo.replace_evidence_sources(
                        item.evidence_id, _source_references(item)
                    )
                uncertain_duplicates = _uncertain_duplicate_pairs(
                    stored_evidence
                )
                pending_duplicate_groups = (
                    self.pipeline_repo.replace_duplicate_groups(
                        patient_id, source_evidence, uncertain_duplicates
                    )
                )
            else:
                uncertain_duplicates = []
                pending_duplicate_groups = []

            # The source layer is immutable: classification affects only the
            # registry projection.  Contextual normal/negative observations
            # can be linked to an existing episode but cannot anchor a new row.
            primary_evidence, contextual_evidence, excluded_evidence = (
                partition_evidence(source_evidence)
            )
            evidence_by_id = {
                item.evidence_id: item for item in stored_evidence
            }
            evidence_by_id.update({
                item.evidence_id: item for item in source_evidence
            })

            existing_events = self.registry_repo.get_events(
                patient_id, include_rejected=True
            )
            self._release_atomic_runtime_before_events(
                enabled=event_llm_available
            )
            consolidator = ClinicalConsolidator(
                fusion_llm=(
                    self.event_llm if event_llm_available else None
                ),
                existing_events=existing_events,
            )
            def relation_progress(
                completed: int,
                total: int,
                cache_hits: int,
                auto_resolved: int,
            ) -> None:
                if progress_callback:
                    progress_callback(
                        60 + int(8 * completed / max(total, 1)),
                        "Relazioni cliniche "
                        f"{completed}/{total} "
                        f"(cache {cache_hits}, regole {auto_resolved})",
                    )

            try:
                graph_result = EvidenceGraphBuilder(
                    self.event_llm if event_llm_available else None,
                    policy=self.pipeline_policy,
                ).build(
                    patient_id, primary_evidence + contextual_evidence,
                    reviewed_relations=(
                        self.pipeline_repo.list_evidence_relations(patient_id)
                        if self.pipeline_repo is not None else ()
                    ),
                    num_workers=event_workers,
                    cancel_check=cancel_check,
                    progress_callback=relation_progress,
                    cache_repository=self.pipeline_repo,
                )
            except EvidenceGraphCancelled as exc:
                raise RegistryBuildCancelled(str(exc)) from exc
            check_cancelled()
            if self.pipeline_repo is not None:
                self.pipeline_repo.replace_evidence_relations(
                    patient_id, graph_result.relations
                )

            def fusion_progress(completed: int, total: int) -> None:
                if progress_callback:
                    progress_callback(
                        68 + int(4 * completed / max(total, 1)),
                        f"Fusione eventi {completed}/{total}",
                    )

            bundles = consolidator.consolidate_graph_clusters(
                patient_id,
                graph_result.clusters,
                evidence_by_id=evidence_by_id,
                num_workers=event_workers,
                progress_callback=fusion_progress,
            )
            check_cancelled()

            if progress_callback:
                progress_callback(
                    72,
                    "Grafo clinico: relazioni pesate e split esplicito...",
                )
            correlation_bundles = []

            if progress_callback:
                progress_callback(78, "Trend di laboratorio e terapie...")
            trends, trend_bundles = LabTrendBuilder(self.db).build(
                patient_id, primary_evidence + contextual_evidence
            )
            medications, oncology_lines, oncology_bundles = (
                TherapyProjectionBuilder().build(patient_id, primary_evidence)
            )
            all_bundles = _unique_bundles(
                bundles + trend_bundles + oncology_bundles + correlation_bundles
            )

            # Clean summaries (headers/measurement tables) then merge bundles
            # that describe the same event across documents/dates.  Cleaning
            # before dedup keeps distinct measurement parameters distinguishable.
            linked_contextual_ids = {
                link.evidence_id for bundle in all_bundles
                for link in bundle.links
                if link.role in {"negative_context", "monitoring"}
            }
            contextual_linked = len(linked_contextual_ids)
            contextual_unassigned = max(
                0, len(contextual_evidence) - contextual_linked
            )
            all_bundles, bundles_merged = deduplicate_bundles(
                all_bundles,
                evidence_by_id=evidence_by_id,
                persisted_review_status={
                    e.event_id: e.review_status for e in existing_events
                },
                evidence_relations=graph_result.relations,
            )

            if progress_callback:
                progress_callback(
                    82, "Assemblaggio LLM dei problemi/episodi clinici..."
                )
            # The v3 graph already defines event membership. A second semantic
            # absorption pass would be able to merge evidence without a graph
            # edge, violating the end-to-end contract.
            semantic_relations = []
            assembly_stats = EpisodeSynthesisStats()
            duplicate_source_links = _attach_duplicate_source_links(
                all_bundles, evidence_by_id
            )
            relation_map = {
                (
                    relation.source_event_id,
                    relation.target_event_id,
                    relation.relation_type,
                ): relation
                for relation in (
                    build_event_relations(patient_id, all_bundles)
                    + semantic_relations
                )
            }
            event_relations = list(relation_map.values())

            if progress_callback:
                progress_callback(88, "Salvataggio registro versionato...")
            self.registry_repo.replace_generated_registry(
                patient_id,
                [bundle.episode for bundle in all_bundles],
                [
                    (bundle.event, bundle.links, bundle.updates)
                    for bundle in all_bundles
                ],
            )
            self.registry_repo.replace_generated_relations(
                patient_id, event_relations
            )
            if self.pipeline_repo is not None:
                for bundle in all_bundles:
                    self.pipeline_repo.replace_event_claims(
                        bundle.event.event_id,
                        _claims_for_bundle(bundle, evidence_by_id),
                    )
            self.registry_repo.save_lab_trends(patient_id, trends)
            self.registry_repo.save_medication_courses(
                patient_id, medications
            )
            self.registry_repo.save_oncology_lines(
                patient_id, oncology_lines
            )
            timeline_changed = self._sync_legacy_timeline(
                patient_id, all_bundles, source_evidence
            )
            profile_invalidated = self._invalidate_clinical_profile(
                patient_id
            ) if timeline_changed else False
            self._sync_review_queue(patient_id, all_bundles)
            self._sync_duplicate_review_queue(
                patient_id, pending_duplicate_groups
            )
            self._sync_relation_review_queue(
                patient_id, graph_result.relations
            )

            elapsed = round(time.monotonic() - started, 2)
            self.processing_repo.finish_run(
                run.run_id, "completed_with_warnings" if failures else "completed"
            )
            if self.audit:
                self.audit.log(
                    patient_id, "clinical_registry_v3_built",
                    "clinical_registry", patient_id,
                    {
                        "documents_total": len(documents),
                        "documents_processed": processed,
                        "documents_skipped": skipped,
                        "documents_failed": len(failures),
                        "atomic_model": getattr(
                            self.atomic_llm, "model", None
                        ),
                        "event_model": getattr(
                            self.event_llm, "model", None
                        ),
                        "atomic_evidence_extracted": extracted_count,
                        "abnormal_lab_evidence": abnormal_lab_evidence_count,
                        "exact_blocks_reused": reused_blocks,
                        "exact_reused_characters": reused_chars,
                        "reused_evidence": reused_evidence_count,
                        "unique_reuse_blocks_verified": unique_blocks_verified,
                        "reuse_verification_batches": reuse_verification_batches,
                        "targeted_reuse_verifications": (
                            targeted_verification_count
                        ),
                        "full_document_fallbacks": fallback_count,
                        "llm_calls": sum(
                            int(item.get("llm_calls", 0)) for item in doc_stats
                        ),
                        "source_chunks": sum(
                            int(item.get("source_chunks", 0))
                            for item in doc_stats
                        ),
                        "output_limit_retries": sum(
                            int(item.get("output_limit_retries", 0))
                            for item in doc_stats
                        ),
                        "prompt_tokens": sum(
                            int(item.get("prompt_tokens", 0)) for item in doc_stats
                        ),
                        "completion_tokens": sum(
                            int(item.get("completion_tokens", 0))
                            for item in doc_stats
                        ),
                        "evidence_stored_occurrences": len(stored_evidence),
                        "evidence_total": len(source_evidence),
                        "atomic_duplicates_suppressed": (
                            atomic_duplicates_suppressed
                        ),
                        "duplicate_source_links": duplicate_source_links,
                        "evidence_primary": len(primary_evidence),
                        "evidence_contextual": len(contextual_evidence),
                        "evidence_administrative_or_methodological": len(
                            excluded_evidence
                        ),
                        "contextual_evidence_linked": contextual_linked,
                        "contextual_evidence_unassigned": contextual_unassigned,
                        "events_total": len(all_bundles),
                        "bundles_merged": bundles_merged,
                        "episode_candidate_groups": (
                            assembly_stats.candidate_groups
                        ),
                        "episode_assembly_llm_calls": assembly_stats.llm_calls,
                        "episode_assembly_cached_groups": (
                            assembly_stats.cached_groups
                        ),
                        "episode_observations_absorbed": (
                            assembly_stats.episodes_absorbed
                        ),
                        "episode_autonomous_links": (
                            assembly_stats.autonomous_links
                        ),
                        "event_relations": len(event_relations),
                        "evidence_relation_candidates": graph_result.candidate_count,
                        "evidence_relations": len(graph_result.relations),
                        "evidence_relation_llm_calls": graph_result.llm_calls,
                        "evidence_relation_cache_hits": graph_result.cache_hits,
                        "evidence_relation_auto_resolved": (
                            graph_result.auto_resolved_count
                        ),
                        "graph_explicit_splits": graph_result.split_count,
                        "clinical_profile_invalidated": profile_invalidated,
                        "episodes_total": len({
                            bundle.episode.episode_id for bundle in all_bundles
                        }),
                        "lab_trends": len(trends),
                        "medication_courses": len(medications),
                        "oncology_lines": len(oncology_lines),
                        "elapsed_seconds": elapsed,
                    },
                    model_used=run.model_name,
                    model_version=ATOMIC_PROMPT_VERSION,
                    run_id=run.run_id,
                )
            if progress_callback:
                progress_callback(100, "Registro clinico v3 completato")
            return {
                "total_entries": len(source_evidence),
                "stored_evidence_occurrences": len(stored_evidence),
                "atomic_duplicates_suppressed": (
                    atomic_duplicates_suppressed
                ),
                "duplicate_source_links": duplicate_source_links,
                "registry_primary_evidence": len(primary_evidence),
                "registry_contextual_evidence": len(contextual_evidence),
                "registry_excluded_evidence": len(excluded_evidence),
                "contextual_evidence_linked": contextual_linked,
                "contextual_evidence_unassigned": contextual_unassigned,
                "deduplicated": max(
                    0, len(primary_evidence) - len(all_bundles)
                ),
                "bundles_merged": bundles_merged,
                "episode_candidate_groups": assembly_stats.candidate_groups,
                "episode_assembly_llm_calls": assembly_stats.llm_calls,
                "episode_assembly_cached_groups": assembly_stats.cached_groups,
                "episode_observations_absorbed": (
                    assembly_stats.episodes_absorbed
                ),
                "episode_autonomous_links": assembly_stats.autonomous_links,
                "event_relations": len(event_relations),
                "evidence_relation_candidates": graph_result.candidate_count,
                "evidence_relations": len(graph_result.relations),
                "evidence_relation_llm_calls": graph_result.llm_calls,
                "evidence_relation_cache_hits": graph_result.cache_hits,
                "evidence_relation_auto_resolved": (
                    graph_result.auto_resolved_count
                ),
                "graph_explicit_splits": graph_result.split_count,
                "clinical_profile_invalidated": profile_invalidated,
                "final_entries": len(all_bundles),
                "documents_processed": processed,
                "documents_skipped": skipped,
                "documents_failed": len(failures),
                "failed_doc_ids": [item["document_id"] for item in failures],
                "failures": failures,
                "document_stats": sorted(
                    doc_stats, key=lambda item: item["document_id"]
                ),
                "atomic_evidence_extracted": extracted_count,
                "abnormal_lab_evidence": abnormal_lab_evidence_count,
                "exact_blocks_reused": reused_blocks,
                "exact_reused_characters": reused_chars,
                "reused_evidence": reused_evidence_count,
                "unique_reuse_blocks_verified": unique_blocks_verified,
                "reuse_verification_batches": reuse_verification_batches,
                "targeted_reuse_verifications": targeted_verification_count,
                "full_document_fallbacks": fallback_count,
                "llm_calls": sum(
                    int(item.get("llm_calls", 0)) for item in doc_stats
                ),
                "source_chunks": sum(
                    int(item.get("source_chunks", 0)) for item in doc_stats
                ),
                "output_limit_retries": sum(
                    int(item.get("output_limit_retries", 0))
                    for item in doc_stats
                ),
                "validation_retries": sum(
                    int(item.get("validation_retries", 0))
                    for item in doc_stats
                ),
                "wire_items_normalized": sum(
                    int(item.get("normalized_items", 0))
                    for item in doc_stats
                ),
                "unresolved_invalid_items": sum(
                    int(item.get("unresolved_invalid_items", 0))
                    for item in doc_stats
                ),
                "prompt_tokens": sum(
                    int(item.get("prompt_tokens", 0)) for item in doc_stats
                ),
                "completion_tokens": sum(
                    int(item.get("completion_tokens", 0))
                    for item in doc_stats
                ),
                "lab_trends": len(trends),
                "medication_courses": len(medications),
                "oncology_lines": len(oncology_lines),
                "incremental": incremental,
                "registry_version": 3,
                "atomic_model": getattr(self.atomic_llm, "model", None),
                "event_model": getattr(self.event_llm, "model", None),
                "run_id": run.run_id,
                "elapsed_seconds": elapsed,
            }
        except RegistryBuildCancelled as exc:
            self.processing_repo.interrupt_run(run.run_id, str(exc))
            raise
        except Exception as exc:
            self.processing_repo.finish_run(run.run_id, "failed", str(exc))
            raise

    def rebuild_from_evidence(
        self,
        patient_id: str,
        progress_callback: Optional[Callable[[int, str], None]] = None,
    ) -> dict:
        """Re-consolidate stored evidence without re-running extraction."""
        return self.build(
            patient_id, incremental=True, num_workers=1,
            progress_callback=progress_callback,
        )

    def sync_timeline_projection(self, patient_id: str) -> bool:
        """Refresh the legacy GUI projection after a manual event operation."""
        return self._sync_legacy_timeline(patient_id, [], [])

    def _release_atomic_runtime_before_events(self, *, enabled: bool) -> None:
        """Release distinct extraction weights before event synthesis.

        llama.cpp starts clients lazily.  When the two stages use different
        runtime identities, keeping the extraction server resident would make
        the event model an avoidable second copy in RAM/GPU memory.  Failures
        are intentionally non-fatal: the backend can still try to load the
        event model and report an actionable error from the generation call.
        """
        if not enabled or not self.atomic_llm or not self.event_llm:
            return
        try:
            if (
                self.atomic_llm.runtime_identity()
                == self.event_llm.runtime_identity()
            ):
                return
            self.atomic_llm.backend.stop_config(self.atomic_llm)
        except Exception:
            return

    @staticmethod
    def _normalized_text_path(
        patient_id: str, document_id: str
    ) -> Path | None:
        candidates = (
            active_workspace.path / patient_id / "extraction" /
            f"{document_id}.md",
            active_workspace.path / patient_id / "docling" /
            f"{document_id}.md",
        )
        return next((path for path in candidates if path.exists()), None)

    def _sync_abnormal_lab_evidence(self, patient_id: str, documents) -> int:
        """Rebuild deterministic lab atoms from structured out-of-range rows.

        This makes the post-extraction registry independent from the GUI path
        that originally parsed the report and also removes normal laboratory
        atoms left by older pipeline versions.
        """
        lab_values = self.lab_repo.get_by_patient(patient_id)
        by_document: dict[str, list] = {}
        for value in lab_values:
            by_document.setdefault(value.document_id, []).append(value)
        total = 0
        prior_values = []
        for document in documents:
            values = by_document.get(document.id, [])
            if document.document_type != "laboratorio" and not values:
                continue
            geometry = None
            if values:
                geometry = load_document_geometry(
                    active_workspace.path / patient_id / "extraction" /
                    f"{document.id}.json"
                )
            evidence = abnormal_lab_evidence(
                patient_id=patient_id,
                document_id=document.id,
                document_date=document.document_date,
                lab_values=values,
                geometry=geometry,
                policy=getattr(
                    getattr(self, "pipeline_policy", None), "lab", None
                ),
                prior_values=prior_values,
            )
            self.evidence_repo.replace_document_method(
                document.id, LAB_EXTRACTION_METHOD, evidence
            )
            total += len(evidence)
            prior_values.extend(values)
        return total

    def _replace_atomic_document_evidence(
        self, document_id: str, evidence, *, clear_missing: bool = True
    ) -> None:
        """Refresh LLM and deterministic-prefilter atoms independently."""
        grouped: dict[str, list] = {
            "llm_atomic_v2": [],
            "deterministic_nonclinical": [],
        }
        for item in evidence:
            grouped.setdefault(item.extraction_method, []).append(item)
        for method, items in grouped.items():
            if not clear_missing and not items:
                continue
            self.evidence_repo.replace_document_method(
                document_id, method, items
            )

    def _sync_legacy_timeline(
        self,
        patient_id: str,
        bundles,
        all_evidence,
    ) -> bool:
        # Read the persisted projection back: reviewed/corrected/rejected rows
        # may intentionally differ from the newly generated bundle.
        persisted_events = self.registry_repo.get_events(patient_id)
        entries = []
        for event in persisted_events:
            detail = self.registry_repo.get_event_detail(event.event_id) or {}
            evidence_rows = detail.get("evidence", [])
            if evidence_rows and all(
                is_administrative_mapping(item) for item in evidence_rows
            ):
                continue
            entries.append(ClinicalTimelineEntry(
                entry_id=event.event_id,
                patient_id=patient_id,
                date_observed=event.first_evidence_date or "",
                date_resolved=event.date_end,
                category=event.category,
                description=event.summary_short,
                source_document_ids=list(dict.fromkeys(
                    str(item.get("document_id") or "")
                    for item in evidence_rows if item.get("document_id")
                )),
                source_texts=list(dict.fromkeys(
                    str(item.get("source_text") or "")
                    for item in evidence_rows if item.get("source_text")
                )),
                status=event.status,
                confidence=event.confidence or 0.0,
                is_golden=1 if event.review_status in {
                    "accepted", "corrected"
                } else 0,
            ))
        previous = self.timeline_repo.get_by_patient(patient_id)

        def signature(rows) -> list[tuple]:
            return [(
                row.entry_id, row.date_observed, row.date_resolved,
                row.category, row.description, row.status,
            ) for row in rows]

        changed = signature(previous) != signature(entries)
        self.timeline_repo.replace_all_for_patient(patient_id, entries)
        return changed

    def _invalidate_clinical_profile(self, patient_id: str) -> bool:
        """Clear a narrative derived from a registry projection that changed."""
        repository = ClinicalStateRepository(self.db)
        state = repository.load(patient_id)
        if state is None or not state.clinical_profile:
            return False
        state.clinical_profile = ""
        repository.save(state)
        return True

    def _sync_review_queue(self, patient_id: str, bundles) -> None:
        pending = [
            bundle for bundle in bundles
            if bundle.event.review_status == "pending"
        ]
        with self.db:
            self.db.execute(
                """DELETE FROM validation_queue
                   WHERE patient_id=?
                     AND item_type IN ('clinical_event_v2','clinical_event_v3')
                     AND status='pending'""",
                (patient_id,),
            )
            for bundle in pending:
                reasons = []
                data = bundle.event.structured_data
                if data.get("conflicts"):
                    reasons.append("fonti discordanti")
                if bundle.event.certainty == "inferred":
                    reasons.append("correlazione/inferenza clinica")
                if not data.get("fusion_validated", True):
                    reasons.append("sintesi non verificata completamente")
                if data.get("unit_compatible") is False:
                    reasons.append("unità laboratoristiche incompatibili")
                if data.get("merged_into_ids"):
                    reasons.append("duplicati unificati — verifica")
                issue = ", ".join(reasons) or "evidenza da revisionare"
                self.db.execute(
                    """INSERT INTO validation_queue
                       (patient_id, item_type, item_id, issue, severity,
                        status, original_value, created_at)
                       VALUES (?, 'clinical_event_v3', ?, ?, ?, 'pending', ?, ?)""",
                    (
                        patient_id, bundle.event.event_id, issue,
                        "high" if bundle.event.category in {
                            "diagnosis", "toxicity", "progression",
                            "oncology_treatment_line",
                        } else "medium",
                        json.dumps({
                            "summary_short": bundle.event.summary_short,
                            "category": bundle.event.category,
                            "certainty": bundle.event.certainty,
                            "evidence_count": len(bundle.links),
                        }, ensure_ascii=False),
                        datetime.now(timezone.utc).isoformat(),
                    ),
                )

    def _sync_duplicate_review_queue(
        self, patient_id: str, group_ids: list[str]
    ) -> None:
        with self.db:
            self.db.execute(
                """DELETE FROM validation_queue
                   WHERE patient_id=? AND item_type='atomic_duplicate_v3'
                     AND status='pending'""",
                (patient_id,),
            )
            for group_id in group_ids:
                group = self.db.execute(
                    """SELECT decision_reason FROM evidence_duplicate_groups
                       WHERE duplicate_group_id=? AND patient_id=?""",
                    (group_id, patient_id),
                ).fetchone()
                members = self.db.execute(
                    """SELECT evidence_id FROM evidence_duplicate_members
                       WHERE duplicate_group_id=? ORDER BY evidence_id""",
                    (group_id,),
                ).fetchall()
                if group is None or len(members) < 2:
                    continue
                similarity_text = str(group["decision_reason"] or "")
                try:
                    similarity = float(similarity_text.rsplit("=", 1)[1])
                except (IndexError, ValueError):
                    similarity = None
                self.db.execute(
                    """INSERT INTO validation_queue
                       (patient_id, item_type, item_id, issue, severity,
                        status, original_value, created_at)
                       VALUES (?, 'atomic_duplicate_v3', ?, ?, 'high',
                               'pending', ?, ?)""",
                    (
                        patient_id, group_id,
                        "Possibile copia cross-documento da confermare",
                        json.dumps({
                            "left_evidence_id": members[0]["evidence_id"],
                            "right_evidence_id": members[1]["evidence_id"],
                            "similarity": similarity,
                        }, ensure_ascii=False),
                        datetime.now(timezone.utc).isoformat(),
                    ),
                )

    def _sync_relation_review_queue(self, patient_id: str, relations) -> None:
        """Queue only consequential uncertain edges, not every candidate."""
        pending = [
            relation for relation in relations
            if relation.review_status == "pending"
            and (
                relation.cluster_effect == "uncertain"
                or abs(
                    relation.weight - self.pipeline_policy.cohesive_threshold
                ) <= 0.12
            )
        ]
        pending.sort(key=lambda relation: (
            abs(relation.weight - self.pipeline_policy.cohesive_threshold),
            relation.relation_id,
        ))
        with self.db:
            self.db.execute(
                """DELETE FROM validation_queue
                   WHERE patient_id=? AND item_type='evidence_relation_v3'
                     AND status='pending'""",
                (patient_id,),
            )
            for relation in pending[:500]:
                self.db.execute(
                    """INSERT INTO validation_queue
                       (patient_id,item_type,item_id,issue,severity,status,
                        original_value,created_at)
                       VALUES (?, 'evidence_relation_v3', ?, ?, 'high',
                               'pending', ?, ?)""",
                    (
                        patient_id, relation.relation_id,
                        "Collegamento fra evidenze incerto o vicino alla soglia",
                        json.dumps({
                            "source_evidence_id": relation.source_evidence_id,
                            "target_evidence_id": relation.target_evidence_id,
                            "relation_type": relation.relation_type,
                            "cluster_effect": relation.cluster_effect,
                            "weight": relation.weight,
                            "rationale": relation.rationale,
                        }, ensure_ascii=False),
                        datetime.now(timezone.utc).isoformat(),
                    ),
                )


def _attach_duplicate_source_links(bundles, evidence_by_id: dict) -> int:
    """Expose copied report occurrences without treating them as new facts."""
    added = 0
    for bundle in bundles:
        linked_ids = {link.evidence_id for link in bundle.links}
        aliases: list[str] = []
        for link in list(bundle.links):
            canonical = evidence_by_id.get(link.evidence_id)
            if canonical is None:
                continue
            aliases.extend(
                str(evidence_id) for evidence_id in
                canonical.data.get("duplicate_source_evidence_ids", [])
                if evidence_id
            )
        for evidence_id in dict.fromkeys(aliases):
            if evidence_id in linked_ids or evidence_id not in evidence_by_id:
                continue
            bundle.links.append(EventEvidenceLink(
                link_id=stable_id(
                    "LNK", bundle.event.event_id, evidence_id,
                    "duplicate_source",
                ),
                event_id=bundle.event.event_id,
                evidence_id=evidence_id,
                relation="duplicate_source",
                relation_confidence=1.0,
                rationale=(
                    "Occorrenza documentale ripetuta della stessa evidenza; "
                    "esclusa da sintesi e analisi"
                ),
                included_in_summary=False,
            ))
            linked_ids.add(evidence_id)
            added += 1
    return added


def _excluded_record(item, reason: str) -> ExcludedEvidence:
    """Project one immutable atom into the separate exclusion archive."""
    excluded_id = stable_id(
        "EXC", item.document_id, item.evidence_id, item.clinical_relevance
    )
    return ExcludedEvidence(
        excluded_id=excluded_id,
        patient_id=item.patient_id,
        document_id=item.document_id,
        disposition=item.clinical_relevance,
        reason_code=reason,
        source_text=item.source_text,
        fact_type=item.fact_type,
        concept=item.concept_original or item.normalized_entity,
        source_page=item.source_page,
        bbox=item.bbox,
        sentence_refs=list((item.data or {}).get("sentence_refs") or []),
        extraction_method=item.extraction_method,
        model_name=item.model_name,
        prompt_version=item.prompt_version,
        review_status=(
            "pending" if item.status == "needs_review" else "auto"
        ),
        data={
            "evidence_id": item.evidence_id,
            "registry_role": (item.data or {}).get("registry_role"),
            "classification_method": (item.data or {}).get(
                "classification_method", "post_extraction_rules"
            ),
        },
        created_at=item.created_at,
    )


def _apply_reviewed_duplicate_groups(evidence, groups: list[dict]):
    """Apply accepted near-copy decisions before graph/event construction."""
    items = list(evidence)
    for group in groups:
        member_ids = {
            str(member.get("evidence_id") or "")
            for member in group.get("members", [])
            if member.get("evidence_id")
        }
        owners = [
            item for item in items
            if item.evidence_id in member_ids
            or member_ids.intersection(
                (item.data or {}).get("duplicate_source_evidence_ids") or []
            )
        ]
        if len(owners) < 2:
            continue
        preferred_id = str(group.get("canonical_evidence_id") or "")
        canonical = next(
            (item for item in owners if item.evidence_id == preferred_id),
            min(owners, key=lambda item: (
                item.document_date or "9999", item.document_id,
                item.source_page or 10**9, item.evidence_id,
            )),
        )
        occurrences = []
        duplicate_ids = set()
        for owner in owners:
            owner_occurrences = (owner.data or {}).get(
                "source_occurrences"
            ) or [{
                "evidence_id": owner.evidence_id,
                "document_id": owner.document_id,
                "document_date": owner.document_date,
                "observed_date": owner.observed_date,
                "source_page": owner.source_page,
                "bbox": list(owner.bbox) if owner.bbox else None,
                "source_text": owner.source_text,
            }]
            known_occurrence_ids = {
                row.get("evidence_id") for row in occurrences
            }
            for occurrence in owner_occurrences:
                occurrence_id = str(occurrence.get("evidence_id") or "")
                if occurrence_id and occurrence_id != canonical.evidence_id:
                    duplicate_ids.add(occurrence_id)
                if occurrence_id and occurrence_id not in known_occurrence_ids:
                    occurrences.append(dict(occurrence))
                    known_occurrence_ids.add(occurrence_id)
        canonical.data = dict(canonical.data or {})
        canonical.data.update({
            "duplicate_source_evidence_ids": sorted(duplicate_ids),
            "source_occurrences": sorted(
                occurrences,
                key=lambda row: (
                    row.get("document_date") or "9999",
                    row.get("document_id") or "",
                    row.get("source_page") or 10**9,
                ),
            ),
            "reviewed_duplicate_group_id": group.get("duplicate_group_id"),
            "atomic_duplicate_count": len(duplicate_ids),
        })
        owner_ids = {owner.evidence_id for owner in owners}
        items = [
            item for item in items
            if item.evidence_id == canonical.evidence_id
            or item.evidence_id not in owner_ids
        ]
    return items


def _source_references(item) -> list[EvidenceSourceReference]:
    """Expand canonical and copied occurrences into normalized citations."""
    occurrences = list((item.data or {}).get("source_occurrences") or [])
    if not occurrences:
        occurrences = [{
            "evidence_id": item.evidence_id,
            "document_id": item.document_id,
            "source_page": item.source_page,
            "bbox": list(item.bbox) if item.bbox else None,
            "source_text": item.source_text,
        }]
    result_by_occurrence = {}
    for occurrence in occurrences:
        document_id = str(occurrence.get("document_id") or "")
        passage = str(occurrence.get("source_text") or "")
        if not document_id or not passage:
            continue
        source_role = (
            "primary"
            if occurrence.get("evidence_id") == item.evidence_id
            else "duplicate_source"
        )
        source_ref_id = stable_id(
            "SRC", item.evidence_id, document_id,
            str(occurrence.get("source_page") or ""), passage,
        )
        bbox = occurrence.get("bbox")
        source = EvidenceSourceReference(
            source_ref_id=source_ref_id,
            evidence_id=item.evidence_id,
            document_id=document_id,
            passage=passage,
            sentence_refs=(
                list((item.data or {}).get("sentence_refs") or [])
                if source_role == "primary" else []
            ),
            source_page=occurrence.get("source_page"),
            bbox=tuple(bbox) if bbox else None,
            source_role=source_role,
            created_at=item.created_at,
        )
        key = (document_id, source.source_page, passage)
        existing = result_by_occurrence.get(key)
        if existing is None or (
            source.source_role == "primary"
            and existing.source_role != "primary"
        ):
            result_by_occurrence[key] = source
    return list(result_by_occurrence.values())


def _claims_for_bundle(bundle, evidence_by_id: dict) -> list[EventClaim]:
    """Normalize fusion claims so every statement has source-level citations."""
    known_ids = {
        link.evidence_id for link in bundle.links
        if link.evidence_id in evidence_by_id
        and link.relation != "duplicate_source"
    }
    raw_claims = bundle.event.structured_data.get("claims") or []
    claims: list[EventClaim] = []
    for position, raw in enumerate(raw_claims):
        if not isinstance(raw, dict):
            continue
        text_value = " ".join(str(raw.get("text") or "").split())
        source_ids = list(dict.fromkeys(
            source_id for source_id in (raw.get("evidence_ids") or [])
            if source_id in known_ids
        ))
        if not text_value or not source_ids:
            continue
        claims.append(EventClaim(
            claim_id=stable_id(
                "CLM", bundle.event.event_id, str(position), text_value,
                *source_ids,
            ),
            event_id=bundle.event.event_id,
            text=text_value,
            claim_type="clinical_observation",
            evidence_ids=source_ids,
            certainty=str(raw.get("certainty") or bundle.event.certainty),
            review_status=bundle.event.review_status,
            position=position,
        ))
    if claims:
        return claims
    # Derived projections and legacy deterministic bundles may not carry the
    # normalized claim array yet. They still receive one fully cited claim.
    source_ids = sorted(known_ids)
    text_value = " ".join(str(bundle.event.summary_short or "").split())
    if not source_ids or not text_value:
        return []
    return [EventClaim(
        claim_id=stable_id(
            "CLM", bundle.event.event_id, "0", text_value, *source_ids
        ),
        event_id=bundle.event.event_id,
        text=text_value,
        claim_type="clinical_observation",
        evidence_ids=source_ids,
        certainty=bundle.event.certainty,
        review_status=bundle.event.review_status,
        position=0,
    )]


def _uncertain_duplicate_pairs(evidence) -> list[tuple[str, str, float]]:
    """Find bounded near-copy candidates without suppressing true follow-up."""
    groups: dict[tuple, list] = {}
    result = []
    for item in sorted(evidence, key=lambda value: (
        value.document_date or "9999", value.document_id, value.evidence_id
    )):
        key = (
            str(item.category or "").casefold(),
            " ".join(str(item.normalized_entity or "").casefold().split()),
            item.observed_date,
            item.value_text,
            item.numeric_value,
            str(item.unit or "").casefold(),
            item.assertion,
            item.clinical_status,
        )
        previous = groups.setdefault(key, [])
        current_text = " ".join(str(item.source_text or "").casefold().split())
        for other in previous[-8:]:
            if other.document_id == item.document_id:
                continue
            other_text = " ".join(
                str(other.source_text or "").casefold().split()
            )
            if not current_text or current_text == other_text:
                continue
            similarity = SequenceMatcher(
                None, current_text, other_text, autojunk=False
            ).ratio()
            if 0.78 <= similarity < 0.985:
                result.append((
                    other.evidence_id, item.evidence_id, round(similarity, 4)
                ))
        previous.append(item)
    return result
    return ExcludedEvidence(
        excluded_id=excluded_id,
        patient_id=item.patient_id,
        document_id=item.document_id,
        disposition=item.clinical_relevance,
        reason_code=reason,
        fact_type=item.fact_type,
        concept=item.concept_original or item.normalized_entity,
        source_page=item.source_page,
        bbox=item.bbox,
        sentence_refs=list((item.data or {}).get("sentence_refs") or []),
        source_text=item.source_text,
        extraction_method=item.extraction_method,
        model_name=item.model_name,
        prompt_version=item.prompt_version,
        review_status=(
            "pending" if item.status == "needs_review" else "auto"
        ),
        data={"source_evidence_id": item.evidence_id},
        created_at=item.created_at,
    )


def _evidence_hash(evidence) -> str:
    payload = [
        {
            "evidence_id": item.evidence_id,
            "category": item.category,
            "entity": item.normalized_entity,
            "date": item.observed_date,
            "source": item.source_text,
        }
        for item in sorted(evidence, key=lambda item: item.evidence_id)
    ]
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


def _unique_bundles(bundles):
    """Keep the most information-rich projection for a stable event id."""
    result = {}
    for bundle in bundles:
        current = result.get(bundle.event.event_id)
        if current is None or len(bundle.links) > len(current.links):
            result[bundle.event.event_id] = bundle
    return list(result.values())

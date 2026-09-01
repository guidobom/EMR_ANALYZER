"""QThread workers for background processing.

Each worker runs in a separate thread and communicates
with the main GUI thread via Qt signals.
"""

import threading
from datetime import datetime

from PyQt5.QtCore import QThread, pyqtSignal

from ..prompt_catalog import load_prompt

from ..clinical.registry_builder import RegistryBuildCancelled
from ..clinical import irae_layers, irae_reconsolidate
from ..clinical.irae_reports import save_report


class ClinicalHistoryWorker(QThread):
    """Background worker for building the clinical history timeline."""
    progress = pyqtSignal(int, str)         # percentage, message
    result_ready = pyqtSignal(dict)         # result summary
    cancelled = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(self, builder, patient_id: str,
                 generate_narrative: bool = False,
                 num_workers: int = 1, stage: str = "full", parent=None):
        super().__init__(parent)
        self.builder = builder
        self.patient_id = patient_id
        self.generate_narrative = generate_narrative
        self.num_workers = num_workers
        self.stage = str(stage or "full")
        self._cancel_event = threading.Event()

    def cancel(self) -> None:
        """Request a safe stop after the active LLM call returns."""
        self._cancel_event.set()

    def run(self):
        try:
            progress = lambda pct, msg: self.progress.emit(pct, msg)
            if self.stage == "atomic":
                result = self.builder.extract_atomic_evidence(
                    self.patient_id,
                    incremental=True,
                    num_workers=self.num_workers,
                    progress_callback=progress,
                    cancel_check=self._cancel_event.is_set,
                )
                self.result_ready.emit(result)
                return
            if self.stage == "events":
                result = self.builder.build_structured_events(
                    self.patient_id,
                    progress_callback=progress,
                    cancel_check=self._cancel_event.is_set,
                )
                self.result_ready.emit(result)
                return
            if self.stage == "validation":
                self.result_ready.emit(
                    self.builder.prepare_validation(self.patient_id)
                )
                return
            # A crash can leave no final timeline rows even though many
            # document-level checkpoints are already durable.  Prefer the
            # incremental builder in that case; it validates hashes, prompt
            # and model before deciding which documents can really be skipped.
            existing_count = self.builder._timeline_repo.count_by_patient(
                self.patient_id
            )
            registry_builder = getattr(self.builder, "_registry_builder", None)
            has_checkpoint = bool(
                registry_builder is not None
                and registry_builder.has_atomic_checkpoint(self.patient_id)
            )
            if existing_count > 0 or has_checkpoint:
                result = self.builder.build_incremental(
                    self.patient_id,
                    progress_callback=progress,
                    generate_narrative=self.generate_narrative,
                    cancel_check=self._cancel_event.is_set,
                )
            elif self.num_workers > 1:
                result = self.builder.build_from_documents_parallel(
                    self.patient_id,
                    num_workers=self.num_workers,
                    progress_callback=progress,
                    generate_narrative=self.generate_narrative,
                    cancel_check=self._cancel_event.is_set,
                )
            else:
                result = self.builder.build_from_documents(
                    self.patient_id,
                    progress_callback=progress,
                    generate_narrative=self.generate_narrative,
                    cancel_check=self._cancel_event.is_set,
                )
            self.result_ready.emit(result)
        except RegistryBuildCancelled:
            self.cancelled.emit()
        except Exception as e:
            self.error.emit(str(e))


class RegistryQueueWorker(QThread):
    """Build chronological registries for several patients sequentially.

    One patient already consumes all configured llama-server slots through
    the registry builder's document pool.  Running patients concurrently
    would only make them compete for the same slots, so this worker advances
    to the next patient only after the current registry is durably saved.
    Cancellation is intentionally honoured between patients.
    """

    patient_started = pyqtSignal(int, int, str)
    patient_progress = pyqtSignal(str, int, str)
    patient_finished = pyqtSignal(str, dict)
    patient_error = pyqtSignal(str, str)

    def __init__(
        self,
        builder,
        patient_ids: list[str],
        *,
        num_workers: int = 1,
        force_rebuild: bool = False,
        stage: str = "full",
        parent=None,
    ):
        super().__init__(parent)
        self.builder = builder
        self.patient_ids = list(patient_ids)
        self.num_workers = max(1, int(num_workers or 1))
        self.force_rebuild = bool(force_rebuild)
        self.stage = str(stage or "full")
        self._cancel_event = threading.Event()

    def cancel(self) -> None:
        """Request a safe stop after the patient currently being saved."""
        self._cancel_event.set()

    def run(self) -> None:
        total = len(self.patient_ids)
        for index, patient_id in enumerate(self.patient_ids, start=1):
            if self._cancel_event.is_set():
                break
            self.patient_started.emit(index, total, patient_id)

            def progress(percent: int, message: str, pid=patient_id) -> None:
                self.patient_progress.emit(pid, int(percent), str(message))

            try:
                if self.stage == "atomic":
                    result = self.builder.extract_atomic_evidence(
                        patient_id,
                        incremental=not self.force_rebuild,
                        num_workers=self.num_workers,
                        progress_callback=progress,
                        cancel_check=self._cancel_event.is_set,
                    )
                elif self.stage == "events":
                    result = self.builder.build_structured_events(
                        patient_id,
                        progress_callback=progress,
                        cancel_check=self._cancel_event.is_set,
                    )
                elif self.stage == "validation":
                    result = self.builder.prepare_validation(patient_id)
                    progress(100, "Coda di validazione preparata")
                elif self.force_rebuild:
                    result = self.builder.build_from_documents_parallel(
                        patient_id,
                        num_workers=self.num_workers,
                        progress_callback=progress,
                        generate_narrative=False,
                        cancel_check=self._cancel_event.is_set,
                    )
                else:
                    # The evidence-first builder verifies input hashes,
                    # prompt/model versions and manifests; a patient whose
                    # registry is current therefore completes without LLM
                    # calls and is reported as already up to date.
                    result = self.builder.build_incremental(
                        patient_id,
                        progress_callback=progress,
                        generate_narrative=False,
                        cancel_check=self._cancel_event.is_set,
                    )
                self.patient_finished.emit(patient_id, dict(result or {}))
            except RegistryBuildCancelled:
                break
            except Exception as exc:
                self.patient_error.emit(patient_id, str(exc))


class DedupWorker(QThread):
    """Deduplicate existing timeline entries without re-extracting."""
    result_ready = pyqtSignal(int)          # number of removed entries
    error = pyqtSignal(str)

    def __init__(self, builder, patient_id: str, parent=None):
        super().__init__(parent)
        self.builder = builder
        self.patient_id = patient_id

    def run(self):
        try:
            removed = self.builder.deduplicate_existing(self.patient_id)
            self.result_ready.emit(removed)
        except Exception as e:
            self.error.emit(str(e))


class NarrativeWorker(QThread):
    """Generate the narrative clinical profile from the existing timeline."""
    result_ready = pyqtSignal(str)          # narrative text
    error = pyqtSignal(str)

    def __init__(self, builder, patient_id: str, parent=None):
        super().__init__(parent)
        self.builder = builder
        self.patient_id = patient_id

    def run(self):
        try:
            narrative = self.builder.generate_narrative(self.patient_id)
            self.result_ready.emit(narrative)
        except Exception as e:
            self.error.emit(str(e))


# Conversational context cap: last N messages, truncated per message, so
# the embedded history never crowds out the clinical registry in the
# context window (default LLM context is 32K tokens).
_MAX_CONVERSATION_MESSAGES = 20   # 10 Q&A exchanges
_MAX_CONVERSATION_CHARS_PER_MESSAGE = 300


def format_registry_context(entries: list[dict], limit: int = 100) -> str:
    """Compact registry lines with citable entry ids.

    Each line carries ``[#id]`` so prompts can ask the model to cite the
    exact registry entries behind every claim (traceability).
    """
    return "\n".join(
        f"[#{e.get('entry_id', '?')}] [{e.get('date_observed', '?')}] "
        f"[{e.get('category', '?')}] {e.get('description', '')}"
        for e in entries[-limit:]
    )


def build_query_prompt(
    clinical_profile: str,
    entries_text: str,
    question: str,
    conversation: list[dict] | None = None,
    use_conversation_context: bool = False,
) -> str:
    """Build the user prompt of a clinical query.

    When *use_conversation_context* is True and prior messages exist, a
    ``CONVERSAZIONE PRECEDENTE`` section is embedded before the question so
    follow-up questions can reference earlier answers (capped: last 20
    messages, 300 chars each).  Otherwise the prompt is identical to the
    classic single-shot form.
    """
    conversation_text = ""
    if use_conversation_context and conversation:
        lines = []
        for message in conversation[-_MAX_CONVERSATION_MESSAGES:]:
            speaker = (
                "UTENTE" if message.get("role") == "user" else "ASSISTENTE"
            )
            content = str(message.get("content") or "")
            if len(content) > _MAX_CONVERSATION_CHARS_PER_MESSAGE:
                content = content[:_MAX_CONVERSATION_CHARS_PER_MESSAGE] + "…"
            lines.append(f"[{speaker}] {content}")
        if lines:
            conversation_text = (
                "CONVERSAZIONE PRECEDENTE:\n" + "\n".join(lines) + "\n\n"
            )

    return (
        f"PROFILO CLINICO:\n{clinical_profile}\n\n"
        f"REGISTRO CRONOLOGICO:\n{entries_text}\n\n"
        f"{conversation_text}"
        f"DOMANDA: {question}\n\n"
        f"Rispondi in modo chiaro e conciso, citando date e fonti "
        f"quando disponibili."
    )


class ClinicalHistoryQueryWorker(QThread):
    """Run a clinical query against the timeline history."""
    result_ready = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, llm_client, entries: list[dict],
                 clinical_profile: str, question: str,
                 conversation: list[dict] | None = None,
                 use_conversation_context: bool = False,
                 registry_repo=None, patient_id: str = "",
                 evidence_repo=None, parent=None):
        super().__init__(parent)
        self.llm_client = llm_client
        self.entries = entries
        self.clinical_profile = clinical_profile
        self.question = question
        self.conversation = conversation
        self.use_conversation_context = use_conversation_context
        self.registry_repo = registry_repo
        self.patient_id = patient_id
        self.evidence_repo = evidence_repo

    def run(self):
        try:
            system_prompt = load_prompt("clinical_query_system")

            if self.registry_repo is not None and self.patient_id:
                answer = self._run_registry_query(system_prompt)
            else:
                # Compatibility path for legacy workspaces/tests.
                entries_text = format_registry_context(self.entries, limit=100)
                user_prompt = build_query_prompt(
                    self.clinical_profile,
                    entries_text,
                    self.question,
                    conversation=self.conversation,
                    use_conversation_context=self.use_conversation_context,
                )
                answer = self.llm_client.generate_text(
                    user_prompt, system_prompt
                )
            self.result_ready.emit(answer)
        except Exception as e:
            self.error.emit(f"Errore query: {str(e)}")

    def _run_registry_query(self, system_prompt: str) -> str:
        from ..clinical.query_service import (
            ClinicalQueryService, answer_has_valid_citations,
        )

        service = ClinicalQueryService(self.registry_repo)
        details = service.retrieve(self.patient_id, self.question)
        if not details and self.evidence_repo is not None:
            # Registry not built yet: interrogate the DETERMINISTIC timeline
            # built from atomic evidence.  The LLM never reconstructs the
            # chronology itself.
            return self._run_timeline_query(system_prompt)
        valid_ids = {
            detail["event"]["event_id"] for detail in details
        }
        valid_pairs = {
            (detail["event"]["event_id"], str(evidence.get("document_id")))
            for detail in details for evidence in detail.get("evidence", [])
            if evidence.get("document_id")
        }
        context_length = int(
            getattr(self.llm_client, "context_length", 32768) or 32768
        )
        output_tokens = int(
            getattr(self.llm_client, "max_output_tokens", 4096) or 4096
        )
        context_chars = max(
            12_000,
            int(max(4000, context_length - output_tokens - 2500) * 2.3),
        )
        chunks = service.format_chunks(details, max_chars=context_chars)
        partials = []
        for index, chunk in enumerate(chunks, start=1):
            prompt = build_query_prompt(
                self.clinical_profile if index == 1 else "",
                chunk,
                self.question,
                conversation=self.conversation if index == 1 else None,
                use_conversation_context=(
                    self.use_conversation_context and index == 1
                ),
            ) + (
                "\nUsa citazioni nel formato [#EVT_...; DOC_...:p.N]. "
                "Ogni affermazione clinica deve essere sostenuta da almeno "
                "un evento e da una fonte originale elencata. Non usare "
                "identificativi non presenti nel contesto."
            )
            partials.append(self.llm_client.generate_text(
                prompt,
                system_prompt + (
                    " Il contesto deriva dal registro evidence-based completo, "
                    "non dalle sole voci più recenti."
                ),
            ))
        if len(partials) == 1:
            answer = partials[0]
        else:
            answer = _reconcile_query_partials(
                self.llm_client, partials, self.question, system_prompt,
                max_chars=context_chars,
            )
        if valid_ids and not answer_has_valid_citations(
            answer, valid_ids, valid_pairs
        ):
            retry = (
                "La risposta seguente non contiene citazioni di registro "
                "verificabili. Riformulala senza aggiungere contenuto e cita "
                "ogni affermazione con uno degli ID validi nel formato "
                "[#EVT_...; DOC_...:p.N].\n\n"
                f"RISPOSTA DA CORREGGERE:\n{answer}\n\n"
                "CONTESTO VERIFICABILE:\n" + "\n\n".join(chunks)
            )
            corrected = self.llm_client.generate_text(retry, system_prompt)
            if answer_has_valid_citations(corrected, valid_ids, valid_pairs):
                answer = corrected
            else:
                answer = (
                    "La risposta generativa non ha superato il controllo delle "
                    "citazioni. Eventi pertinenti recuperati:\n\n"
                    + "\n".join(_deterministic_cited_event(detail)
                                for detail in details)
                )
        return answer

    def _run_timeline_query(self, system_prompt: str) -> str:
        """Interrogate the deterministic timeline when the registry is empty.

        The chronology is built and filtered by code; the LLM only reads
        the already-ordered compact text.
        """
        from ..clinical.query_service import (
            build_query_prompt, infer_intents, is_broad_question,
        )
        from ..clinical.timeline_serializer import (
            INTENT_TYPES, build_timeline, filter_entries, format_compact,
        )

        evidence = self.evidence_repo.get_by_patient(self.patient_id)
        entries = build_timeline(evidence)
        intents = infer_intents(self.question)
        if not is_broad_question(self.question) and intents:
            types = set().union(*(
                INTENT_TYPES.get(intent, set()) for intent in intents
            ))
            entries = filter_entries(entries, types=types)
        context_length = int(
            getattr(self.llm_client, "context_length", 32768) or 32768
        )
        output_tokens = int(
            getattr(self.llm_client, "max_output_tokens", 4096) or 4096
        )
        context_chars = max(
            12_000,
            int(max(4000, context_length - output_tokens - 2500) * 2.3),
        )
        timeline_text = format_compact(entries, max_chars=context_chars)
        user_prompt = build_query_prompt(
            "",
            timeline_text or "Nessuna evidenza atomica disponibile.",
            self.question,
            conversation=self.conversation,
            use_conversation_context=self.use_conversation_context,
        )
        return self.llm_client.generate_text(user_prompt, system_prompt)


def _deterministic_cited_event(detail: dict) -> str:
    event = detail["event"]
    evidence = (detail.get("evidence") or [{}])[0]
    document_id = evidence.get("document_id") or "documento_n.d."
    page = evidence.get("source_page") or "n.d."
    return (
        f"- [#{event['event_id']}; {document_id}:p.{page}] "
        f"{event.get('first_evidence_date') or 'data n.d.'}: "
        f"{event.get('summary_short') or ''}"
    )


def _reconcile_query_partials(
    llm_client, partials: list[str], question: str, system_prompt: str,
    *, max_chars: int,
) -> str:
    """Hierarchically reconcile arbitrarily many complete registry chunks."""
    current = list(partials)
    while len(current) > 1:
        groups, group, size = [], [], 0
        for part in current:
            added = len(part) + 20
            if group and size + added > max_chars:
                groups.append(group)
                group, size = [], 0
            group.append(part)
            size += added
        if group:
            groups.append(group)
        # Ensure progress even when every partial nearly fills the budget.
        if len(groups) == len(current):
            groups = [current[index:index + 2] for index in range(0, len(current), 2)]
        reduced = []
        for group in groups:
            if len(group) == 1:
                reduced.append(group[0])
                continue
            prompt = (
                f"DOMANDA: {question}\n\n"
                "SINTESI PARZIALI DA RICONCILIARE:\n"
                + "\n\n".join(
                    f"PARTE {index}:\n{part}"
                    for index, part in enumerate(group, start=1)
                )
                + "\n\nProduci una risposta unica, elimina ripetizioni, "
                  "conserva discordanze e tutte le citazioni verificabili. "
                  "Non introdurre affermazioni nuove."
            )
            reduced.append(llm_client.generate_text(prompt, system_prompt))
        current = reduced
    return current[0]


class IraeQueueWorker(QThread):
    """Run the irAE protocol over several patient registries, in order.

    ``patient_plans`` is a list of ``(patient_id, prompts)`` where the
    prompts are pre-built on the main thread (SQLite is not used
    cross-thread).  Cancellation is honored BETWEEN patients; the LLM
    call in flight cannot be interrupted.
    """
    patient_started = pyqtSignal(int, int, str)  # index, total, patient_id
    chunk_progress = pyqtSignal(int, int)        # chunk_index, chunk_total
    patient_finished = pyqtSignal(str, str)      # patient_id, markdown
    patient_error = pyqtSignal(str, str)         # patient_id, error

    def __init__(self, llm_client, patient_plans, parent=None):
        super().__init__(parent)
        self.llm_client = llm_client
        self.patient_plans = patient_plans
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def run(self):
        from emr_analyzer.clinical.irae_analysis import SYSTEM_PROMPT

        total = len(self.patient_plans)
        for index, (patient_id, prompts) in enumerate(
            self.patient_plans, start=1
        ):
            if self._cancelled:
                break
            self.patient_started.emit(index, total, patient_id)
            try:
                parts = []
                for chunk_index, prompt in enumerate(prompts, start=1):
                    self.chunk_progress.emit(chunk_index, len(prompts))
                    answer = self.llm_client.generate_text(
                        prompt, SYSTEM_PROMPT
                    )
                    parts.append(
                        f"### Parte {chunk_index}/{len(prompts)}\n\n"
                        f"{answer}"
                    )
                self.patient_finished.emit(
                    patient_id,
                    reconcile_irae_parts(
                        self.llm_client, parts, prompts, SYSTEM_PROMPT
                    ),
                )
            except Exception as exc:
                self.patient_error.emit(
                    patient_id, f"Errore analisi irAE: {str(exc)}"
                )


class IraeLayer3QueueWorker(QThread):
    """Run the structured 3-layer irAE analysis over several patients.

    ``patient_plans`` is a list of ``(patient_id, [(organ, prompt), ...])``
    with the per-organ Layer 3 prompts pre-built on the main thread (SQLite
    is not touched here).  Each organ call is a structured LLM call with the
    constrained JSON schema from ``irae_layers``; organ calls of one patient
    run in a bounded thread pool (llama-server slots).  Patients run
    sequentially when the client cannot be cloned, or in parallel — one
    llama-server instance per patient — when ``instances > 0`` or the memory
    budget allows more than one (see ``irae_layers.parallel_instance_count``).
    Results are rendered to Markdown and emitted with the same signal
    contract as :class:`IraeQueueWorker`, so the queue wiring and the result
    dialog stay unchanged.  Cancellation is honoured between patients.
    """
    patient_started = pyqtSignal(int, int, str)  # index, total, patient_id
    chunk_progress = pyqtSignal(int, int)        # organ_index, organ_total
    patient_finished = pyqtSignal(str, str)      # patient_id, markdown
    patient_error = pyqtSignal(str, str)         # patient_id, error
    patient_structured = pyqtSignal(str, object)  # patient_id, report dict

    def __init__(
        self,
        llm_client,
        patient_plans,
        *,
        max_tokens: int = irae_layers.DEFAULT_MAX_TOKENS,
        parallel: int = 4,
        instances: int = 0,
        parent=None,
    ):
        super().__init__(parent)
        self.llm_client = llm_client
        self.patient_plans = patient_plans
        self.max_tokens = max_tokens
        self.parallel = parallel
        self.instances = int(instances or 0)
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def run(self):
        from ..clinical import irae_layers

        n = min(
            irae_layers.parallel_instance_count(
                self.llm_client, override=self.instances
            ),
            len(self.patient_plans),
        )
        if n <= 1:
            self._run_sequential()
        else:
            self._run_parallel(n)

    def _run_sequential(self):
        """Original single-instance loop: patients one after another."""
        for index, plan in enumerate(self.patient_plans, start=1):
            if self._cancelled:
                break
            self._process_plan(plan, self.llm_client, index)

    def _run_parallel(self, n):
        """One cloned client per patient, each backed by its own server."""
        from concurrent.futures import ThreadPoolExecutor

        from ..llm_backend.backend import get_backend

        clients = [self.llm_client] + [
            self.llm_client.for_instance(k) for k in range(1, n)
        ]
        backend = getattr(self.llm_client, "backend", None) or get_backend()
        try:
            with ThreadPoolExecutor(max_workers=len(clients)) as pool:
                futures = []
                for index, plan in enumerate(
                    self.patient_plans, start=1
                ):
                    if self._cancelled:
                        break
                    futures.append(
                        pool.submit(
                            self._process_plan,
                            plan,
                            clients[(index - 1) % len(clients)],
                            index,
                        )
                    )
                for future in futures:
                    try:
                        future.result()
                    except Exception:
                        pass  # _process_plan already emitted patient_error
        finally:
            # Stop every sibling instance, including any never used because a
            # patient errored before its first call.  Instance 0 stays warm so
            # the next single-patient analysis reuses it.
            for sibling in clients[1:]:
                try:
                    backend.stop_config(sibling)
                except Exception:
                    pass

    def _process_plan(self, plan, client, index):
        """Process one patient (all organs + consolidation) on *client*."""
        from concurrent.futures import ThreadPoolExecutor, as_completed

        from ..clinical import irae_layers

        total = len(self.patient_plans)
        patient_id = plan[0]
        organ_prompts = plan[1]
        meta = plan[2] if len(plan) > 2 else {}
        if self._cancelled:
            return
        self.patient_started.emit(index, total, patient_id)
        try:
            if not organ_prompts:
                raise RuntimeError(
                    "nessun candidato irAE (esegui prima lo stadio "
                    "'Estrai evidenze' per questo paziente)"
                )
            organ_results: dict[str, dict] = {}
            organs = [organ for organ, _ in organ_prompts]
            max_workers = max(1, min(self.parallel, len(organ_prompts)))
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                futures = {
                    pool.submit(
                        irae_layers.run_organ_call,
                        client, organ, prompt, self.max_tokens,
                    ): organ
                    for organ, prompt in organ_prompts
                }
                for position, future in enumerate(
                    as_completed(futures), start=1
                ):
                    self.chunk_progress.emit(position, len(organ_prompts))
                    result = future.result()
                    organ_results[result["organ"]] = result
            failed = [o for o in organs if organ_results.get(o, {}).get("error")]
            if len(failed) == len(organs):
                raise RuntimeError(
                    organ_results[organs[0]].get("error")
                    or "tutte le chiamate per organo sono fallite"
                )
            findings = [
                item
                for result in organ_results.values()
                for item in result.get("iraes", [])
            ]
            consolidation = irae_layers.consolidate_iraes(
                client, findings, meta.get("anchor"),
                max_tokens=self.max_tokens,
            )
            self.chunk_progress.emit(
                len(organ_prompts) + 1, len(organ_prompts) + 1
            )
            report: dict = {
                **meta,
                "organ_results": {
                    organ: organ_results[organ] for organ in organs
                },
                "iraes": consolidation["iraes"],
                "consolidation": consolidation,
                "candidates": meta.get("candidates", []),
                "evidence": meta.get("evidence", []),
                "analyzed_at": datetime.now().isoformat(timespec="seconds"),
            }
            save_report(patient_id, report)
            self.patient_structured.emit(patient_id, report)
            self.patient_finished.emit(
                patient_id, irae_layers.render_irae_markdown(report)
            )
        except Exception as exc:
            self.patient_error.emit(
                patient_id, f"Errore analisi irAE strutturata: {str(exc)}"
            )


class SinglePatientIraeWorker(QThread):
    """Structured 3-layer irAE analysis for ONE patient.

    Feeds the deterministic evidence rows to ``irae_layers.analyze_irae``
    (the same pipeline as the batch queue) and emits the structured report so
    the single-patient result dialog can inspect and correct it exactly like
    the queue results.
    """

    structured_ready = pyqtSignal(object)  # report dict
    error = pyqtSignal(str)

    def __init__(
        self,
        llm_client,
        rows,
        *,
        patient_id: str = "",
        max_candidates: int = irae_layers.DEFAULT_MAX_CANDIDATES,
        max_tokens: int = irae_layers.DEFAULT_MAX_TOKENS,
        parallel: int = irae_layers.DEFAULT_PARALLEL,
        parent=None,
    ):
        super().__init__(parent)
        self.llm_client = llm_client
        self.rows = rows
        self.patient_id = str(patient_id or "")
        self.max_candidates = max_candidates
        self.max_tokens = max_tokens
        self.parallel = parallel

    def run(self):
        if not self.rows:
            self.error.emit(
                "Nessuna evidenza atomica disponibile (esegui prima lo stadio "
                "'Estrai evidenze')."
            )
            return
        try:
            report = irae_layers.analyze_irae(
                self.rows,
                self.llm_client,
                max_candidates=self.max_candidates,
                max_tokens=self.max_tokens,
                parallel=self.parallel,
            )
            report["analyzed_at"] = datetime.now().isoformat(timespec="seconds")
            if self.patient_id:
                save_report(self.patient_id, report)
            self.structured_ready.emit(report)
        except Exception as exc:
            self.error.emit(f"Errore analisi irAE strutturata: {str(exc)}")


class IraeReconsolidateWorker(QThread):
    """Re-run ONLY the Layer 4 consolidation of a saved irAE report.

    The consolidation call of the original analysis may have exceeded its
    output budget (``DEFAULT_MAX_TOKENS``), leaving ``consolidation.applied``
    False.  This worker re-runs just that call with a higher budget, rebuilds
    the evidence provenance and persists the updated report.  It never touches
    SQLite: ``registry_rows`` are loaded on the main thread and passed in.
    """

    ready = pyqtSignal(object)  # updated report dict
    error = pyqtSignal(str)

    def __init__(
        self,
        llm_client,
        report,
        *,
        patient_id: str = "",
        max_tokens: int = irae_reconsolidate.DEFAULT_RECONSOLIDATE_MAX_TOKENS,
        registry_rows=None,
        parent=None,
    ):
        super().__init__(parent)
        self.llm_client = llm_client
        self.report = report
        self.patient_id = str(patient_id or "")
        self.max_tokens = int(
            max_tokens or irae_reconsolidate.DEFAULT_RECONSOLIDATE_MAX_TOKENS
        )
        self.registry_rows = registry_rows

    def run(self):
        try:
            updated = irae_reconsolidate.reconsolidate_report(
                self.report,
                self.llm_client,
                max_tokens=self.max_tokens,
                registry_rows=self.registry_rows,
            )
            if self.patient_id:
                save_report(self.patient_id, updated)
            self.ready.emit(updated)
        except Exception as exc:
            self.error.emit(f"Errore riconsolidamento irAE: {str(exc)}")


class IraeAnalysisWorker(QThread):
    """Run the irAE protocol over the whole registry, chunk by chunk."""
    progress = pyqtSignal(int, int)  # chunk_index, chunk_total
    result_ready = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, llm_client, prompts: list[str], parent=None):
        super().__init__(parent)
        self.llm_client = llm_client
        self.prompts = prompts

    def run(self):
        from emr_analyzer.clinical.irae_analysis import SYSTEM_PROMPT

        try:
            parts = []
            total = len(self.prompts)
            for index, prompt in enumerate(self.prompts, start=1):
                self.progress.emit(index, total)
                answer = self.llm_client.generate_text(
                    prompt, SYSTEM_PROMPT
                )
                parts.append(
                    f"### Parte {index}/{total}\n\n{answer}"
                )
            self.result_ready.emit(reconcile_irae_parts(
                self.llm_client, parts, self.prompts, SYSTEM_PROMPT
            ))
        except Exception as exc:
            self.error.emit(f"Errore analisi irAE: {str(exc)}")


def reconcile_irae_parts(
    llm_client, parts: list[str], source_prompts: list[str], system_prompt: str
) -> str:
    """Create one deduplicated irAE report while retaining valid source IDs."""
    if len(parts) <= 1:
        return "\n\n".join(parts)
    import re

    allowed_ids = set(re.findall(
        r"\[#([^\];\s]+)", "\n".join(source_prompts)
    ))
    reconciliation_prompt = (
        "RICONCILIAZIONE FINALE DI ANALISI irAE\n\n"
        "Le sezioni seguenti sono analisi parziali di blocchi dello stesso "
        "registro. Produci un unico rapporto clinico finale secondo il "
        "protocollo: elimina duplicati e sovrapposizioni, unifica lo stesso "
        "episodio, conserva cronologia e discordanze, non aggiungere eventi "
        "nuovi. Mantieni le tabelle richieste e cita solo ID [#...] già "
        "presenti. Distingui sospetto, confermato ed escluso; non affermare "
        "causalità non documentate.\n\n"
        + "\n\n".join(parts)
    )
    answer = llm_client.generate_text(reconciliation_prompt, system_prompt)
    cited = set(re.findall(r"\[#([^\];\s]+)", str(answer)))
    if cited - allowed_ids or (allowed_ids and not cited):
        retry = (
            "Riformula il rapporto riconciliato senza cambiare i contenuti. "
            "Usa esclusivamente questi ID di registro: "
            + ", ".join(f"[#{item}]" for item in sorted(allowed_ids))
            + "\n\nRAPPORTO DA CORREGGERE:\n" + str(answer)
        )
        corrected = llm_client.generate_text(retry, system_prompt)
        corrected_ids = set(re.findall(r"\[#([^\];\s]+)", str(corrected)))
        if not (corrected_ids - allowed_ids) and (
            corrected_ids or not allowed_ids
        ):
            return corrected
        # The partial reports are safer than an uncited reconciliation.
        return "\n\n".join(parts)
    return answer

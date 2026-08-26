"""Central QTabWidget workspace for EMR Analyzer."""

from __future__ import annotations

from PyQt5.QtWidgets import (
    QTabWidget, QWidget, QVBoxLayout, QLabel, QMessageBox,
)
from PyQt5.QtCore import Qt, pyqtSignal

from datetime import datetime

from .documents_tab import DocumentsTab
from .laboratory_tab import LaboratoryTab
from .clinical_history_tab import ClinicalHistoryTab
from .validation_tab import ValidationTab
from .gold_set_tab import GoldSetTab
from .import_dialog import ImportDialog
from .batch_import_dialog import BatchImportDialog
from .progress_dialog import ProgressDialog
from .qt_utils import process_gui_events
from ..models import Patient


class WorkspaceTabs(QTabWidget):
    """Central tab container for the patient workspace."""

    context_requested = pyqtSignal(str, dict)
    patient_created = pyqtSignal(str)  # patient_id

    def __init__(self, parent=None):
        super().__init__(parent)
        self._services = {}
        self._current_patient_id = None
        self._current_document_id = None
        self._irae_queue_worker = None
        self._registry_queue_workers: dict = {}

        # Create tabs
        self._documents_tab = DocumentsTab()
        self._laboratory_tab = LaboratoryTab()
        self._clinical_history_tab = ClinicalHistoryTab()
        self._validation_tab = ValidationTab()
        self._gold_set_tab = GoldSetTab()

        # Add tabs
        self.addTab(self._documents_tab, "📄 Documenti")
        self.addTab(self._laboratory_tab, "🔬 Laboratorio")
        self.addTab(self._clinical_history_tab, "📋 Storia Clinica")
        self.addTab(self._validation_tab, "✓ Validazione")
        self.addTab(self._gold_set_tab, "🧪 Gold Set")

        # Connect signals
        self._documents_tab.document_selected.connect(self._on_document_selected)
        self._documents_tab.import_requested.connect(self._on_import_requested)
        self._documents_tab.processing_complete.connect(self._on_processing_complete)
        self._validation_tab.document_reattributed.connect(
            self._on_document_reattributed
        )
        self._clinical_history_tab.validation_requested.connect(
            self._show_validation_tab
        )

        # Forward tab selections to context panel
        self._laboratory_tab.lab_selected.connect(self._on_lab_selected)

    def set_services(self, services: dict):
        self._services = services
        self._documents_tab.set_services(services)
        self._laboratory_tab.set_services(services)
        self._clinical_history_tab.set_services(services)
        self._validation_tab.set_services(services)
        self._gold_set_tab.set_services(services)

    def load_patient(self, patient_id: str):
        """Load all tabs with data for the given patient."""
        self._current_patient_id = patient_id
        self._documents_tab.load_patient(patient_id)
        self._laboratory_tab.load_patient(patient_id)
        self._clinical_history_tab.load_patient(patient_id)
        self._validation_tab.load_patient(patient_id)
        self._gold_set_tab.load_patient(patient_id)

    def _show_validation_tab(self) -> None:
        """Open and refresh phase 3 after its queue has been prepared."""

        self._validation_tab.load_patient(self._current_patient_id)
        self.setCurrentWidget(self._validation_tab)

    def request_shutdown(self) -> None:
        """Request cancellation without waiting for long LLM timeouts."""

        try:
            self._documents_tab.request_shutdown()
        except Exception:
            pass
        try:
            self._clinical_history_tab.request_shutdown()
        except Exception:
            pass
        for worker in (
            self._irae_queue_worker, *self._registry_queue_workers.values(),
        ):
            if worker is not None and worker.isRunning():
                cancel = getattr(worker, "cancel", None)
                if callable(cancel):
                    cancel()
                worker.requestInterruption()

    def shutdown(self, wait_ms: int = 1500) -> int:
        """Cooperatively stop workers; never wait beyond a small total budget."""

        import time

        self.request_shutdown()
        deadline = time.monotonic() + max(0, int(wait_ms)) / 1000
        try:
            self._clinical_history_tab.shutdown(
                max(0, int((deadline - time.monotonic()) * 1000))
            )
        except Exception:
            pass
        workers = [
            worker for worker in (
                self._irae_queue_worker, *self._registry_queue_workers.values(),
            )
            if worker is not None and worker.isRunning()
        ]
        for worker in workers:
            remaining = max(0, int((deadline - time.monotonic()) * 1000))
            if remaining:
                worker.wait(remaining)
        return sum(
            1 for worker in (
                self._irae_queue_worker, *self._registry_queue_workers.values(),
            )
            if worker is not None and worker.isRunning()
        ) + sum(
            1 for attr in self._clinical_history_tab._WORKER_ATTRS
            if getattr(self._clinical_history_tab, attr, None) is not None
            and getattr(self._clinical_history_tab, attr).isRunning()
        )

    def llm_operation_running(self) -> bool:
        """True while any workspace operation is using an LLM runtime."""
        if self._documents_tab.llm_operation_running():
            return True
        if self._clinical_history_tab._worker_running():
            return True
        return any(
            worker is not None and worker.isRunning()
            for worker in (
                self._irae_queue_worker, *self._registry_queue_workers.values(),
            )
        )

    def _on_document_reattributed(self, source_pid: str,
                                  target_pid: str) -> None:
        """Refresh the visible workspace after a document re-attribution.

        The reviewer is looking at the source patient: reload its tabs
        (the document just left this workspace).  The target refreshes
        lazily the next time it is selected.
        """
        pid = self._current_patient_id
        if not pid:
            return
        self._documents_tab.load_patient(pid)
        self._laboratory_tab.load_patient(pid)
        self._clinical_history_tab.load_patient(pid)
        self._validation_tab.load_patient(pid)
        self._gold_set_tab.load_patient(pid)

    def show_import_dialog(self, files: list[str]):
        """Import documents — routes to existing/new patient workspaces."""
        if not files:
            return

        # Guard against recursive calls during import
        if getattr(self, "_import_in_progress", False):
            return
        self._import_in_progress = True

        try:
            self._do_import(files)
        finally:
            self._import_in_progress = False

    def _do_import(self, files: list[str]):
        patient_repo = self._services.get("patient_repo")
        if not patient_repo:
            QMessageBox.warning(self, "Errore", "Servizio pazienti non disponibile.")
            return

        # ---- 1. Stage & route files ---------------------------------------
        staging = self._services.get("import_staging")
        router = self._services.get("patient_router")

        batch = None
        try:
            if staging and router:
                batch = staging.stage(files)
                groups = router.resolve(batch.documents) if batch else []
            else:
                groups = []
        except Exception as e:
            QMessageBox.warning(
                self, "Errore staging",
                f"Impossibile analizzare i file:\n{e}"
            )
            return
        finally:
            # The staged copies in _inbox carry the full identity (name, CF,
            # address) and are only needed for hashing, verification and
            # identity extraction. Routing and the import dialogs use the
            # original paths, so the inbox batch is removed as soon as
            # staging/routing completes — on success and on failure alike.
            if batch:
                batch.cleanup()

        # Build a map: patient_id → list of file paths
        patient_files: dict[str, list[str]] = {}
        routed_files: set[str] = set()
        unassigned_files: list[str] = []

        if groups:
            for g in groups:
                group_file_paths = [
                    d.original_path for d in g.documents
                    if not d.is_duplicate and not d.error
                ]
                if not group_file_paths:
                    continue

                if g.patient_id:
                    # Existing patient — enrich its identity with any new
                    # strong signal the batch carried (e.g. a hospital
                    # patient ID the registry did not know yet).  The upsert
                    # only fills missing fields (COALESCE) and keeps the
                    # highest confidence, so existing values are untouched.
                    pid = g.patient_id
                    identity_repo = self._services.get("identity_repo")
                    if identity_repo and g.evidence and g.evidence.is_strong:
                        identity_repo.upsert(pid, g.evidence, status="enriched")
                elif g.create_new:
                    # Auto-create workspace with identity info
                    pid = patient_repo.get_next_id()
                    evidence = g.evidence
                    initials = None
                    sex = None
                    birth_year = None
                    if evidence:
                        if evidence.name and evidence.name.normalized:
                            parts = evidence.name.normalized.split()
                            initials = "".join(
                                p[0].upper() for p in parts if p
                            )[:4]
                        if evidence.sex and evidence.sex.normalized:
                            sex = evidence.sex.normalized.upper()
                            if sex not in ("M", "F"):
                                sex = None
                        if evidence.birth_date and evidence.birth_date.normalized:
                            try:
                                birth_year = int(
                                    evidence.birth_date.normalized[:4]
                                )
                            except (ValueError, IndexError):
                                pass
                    patient = Patient(
                        id=pid,
                        pseudonym=pid,
                        initials=initials,
                        sex=sex,
                        birth_year=birth_year,
                        created_at=datetime.now().isoformat(),
                        updated_at=datetime.now().isoformat(),
                    )
                    patient_repo.insert(patient)
                    self.patient_created.emit(pid)
                    # Register the identity
                    identity_repo = self._services.get("identity_repo")
                    if identity_repo and evidence:
                        identity_repo.upsert(pid, evidence)
                else:
                    # Identity insufficient: never drop silently.  Files stay
                    # visible so the user can assign them in the dialog.
                    unassigned_files.extend(group_file_paths)
                    continue

                routed_files.update(group_file_paths)
                if pid not in patient_files:
                    patient_files[pid] = []
                patient_files[pid].extend(group_file_paths)

        # Any files not routed → assign to current patient
        unrouted = [
            f for f in files
            if f not in routed_files and f not in unassigned_files
        ]
        if unrouted:
            pid = self._current_patient_id
            if not pid:
                pid = patient_repo.get_next_id()
                patient = Patient(
                    id=pid, pseudonym=pid,
                    created_at=datetime.now().isoformat(),
                    updated_at=datetime.now().isoformat(),
                )
                patient_repo.insert(patient)
                self.patient_created.emit(pid)
            if pid not in patient_files:
                patient_files[pid] = []
            patient_files[pid].extend(unrouted)

        # ---- 2. Import for each patient -----------------------------------
        imported_any = False
        last_patient_id = self._current_patient_id

        if len(patient_files) > 1 or unassigned_files:
            # Multi-patient (or files still needing a patient): one window
            # lists every workspace with a checkbox and expandable per-file
            # detail; a "Da assegnare" bucket shows the needs_review files;
            # then a single sequential queue runs the extraction for the
            # checked patients without further prompts between them.
            candidates = list(patient_files)
            if (self._current_patient_id
                    and self._current_patient_id not in candidates):
                candidates.append(self._current_patient_id)
            batch_dialog = BatchImportDialog(
                patient_files, self._services, parent=self,
                unassigned_files=unassigned_files,
                candidate_pids=candidates,
            )
            if batch_dialog.exec_() == BatchImportDialog.Accepted:
                if batch_dialog.should_run_queue:
                    self.run_extraction_queue(batch_dialog.queue_patient_ids)
                elif batch_dialog.imported_by_patient:
                    imported_any = True
                    last_patient_id = list(batch_dialog.imported_by_patient)[-1]
        else:
            for pid, pfiles in patient_files.items():
                if pid != self._current_patient_id:
                    self.load_patient(pid)
                try:
                    import_dialog = ImportDialog(
                        file_paths=pfiles,
                        services=self._services,
                        patient_id=pid,
                        parent=self,
                    )
                    if import_dialog.exec_() == ImportDialog.Accepted:
                        imported_any = True
                        last_patient_id = pid
                        if import_dialog.should_auto_process():
                            self._documents_tab.extract_clinical_text(
                                import_dialog.get_imported_doc_ids()
                            )
                except Exception as e:
                    QMessageBox.warning(
                        self, "Errore importazione",
                        f"Errore durante l'importazione per {pid}:\n{e}"
                    )

        if imported_any and last_patient_id:
            self.load_patient(last_patient_id)

    def run_extraction_queue(self, patient_ids: list[str]):
        """Run the clinical-text extraction sequentially over several patients.

        A single shared ProgressDialog is reused across patients: its title
        shows the current patient (i/N) and the queue stops between patients
        when the user cancels.  No confirmation is required between patients.
        """
        if not patient_ids:
            return

        total = len(patient_ids)
        progress = ProgressDialog(
            f"Coda di estrazione — paziente 1/{total}", parent=self,
        )
        progress.show()
        process_gui_events()

        for idx, pid in enumerate(patient_ids, start=1):
            if progress.is_cancelled():
                break
            progress.reset_for_reuse()
            progress.setWindowTitle(
                f"Coda di estrazione — paziente {idx}/{total}"
            )
            progress.add_log(f"\n===== Paziente {idx}/{total}: {pid} =====")
            self.load_patient(pid)
            try:
                self._documents_tab.extract_clinical_text(
                    progress=progress,
                    patient_label=f"Paziente {idx}/{total}: {pid}",
                )
            except Exception as exc:
                progress.add_log(f"❌ Errore per {pid}: {exc}")

        progress.mark_done()
        progress.exec_()

    # ------------------------------------------------------------------
    # Multi-patient chronological-registry queue
    # ------------------------------------------------------------------

    def run_registry_queue(
        self, patient_ids: list[str], *, force_rebuild: bool = False,
        stage: str = "atomic",
    ) -> None:
        """Run one explicit registry phase for several patients in order.

        One queue per phase may run concurrently (atomic evidence and
        clinical events use independent local models); a second queue of
        the same phase, and any per-document LLM operation, still blocks.
        """
        if not patient_ids:
            return
        running = self._registry_queue_workers.get(stage)
        if running is not None and running.isRunning():
            QMessageBox.information(
                self, "Coda già in esecuzione",
                "Una coda della stessa fase è già in esecuzione. "
                "Attendi il suo completamento prima di avviarne un'altra.",
            )
            return
        if (
            self._documents_tab.llm_operation_running()
            or self._clinical_history_tab._worker_running()
        ):
            QMessageBox.information(
                self, "Operazione LLM in corso",
                "Attendi il completamento dell'operazione corrente prima "
                "di avviare la coda dei registri.",
            )
            return

        builder = self._services.get("clinical_history_builder")
        client_key = {
            "atomic": "atomic_evidence_llm_client",
            "events": "clinical_events_llm_client",
        }.get(stage)
        required_llm = self._services.get(client_key) if client_key else None
        if builder is None or (
            client_key and (
                required_llm is None or not required_llm.is_available
            )
        ):
            QMessageBox.warning(
                self, "LLM non disponibile",
                "Il modello richiesto dalla fase selezionata, oppure il "
                "generatore dei registri, non è disponibile.",
            )
            return

        configs = self._services.get("llm_configs") or {}
        state_config = configs.get(
            "clinical_events" if stage == "events" else "atomic_evidence"
        )
        num_workers = max(
            1, int(getattr(state_config, "parallel_workers", 1) or 1)
        )

        other_stage = "events" if stage == "atomic" else "atomic"
        other_queue = self._registry_queue_workers.get(other_stage)
        concurrent = (
            other_queue is not None and other_queue.isRunning()
        )

        from .workers import RegistryQueueWorker

        worker = RegistryQueueWorker(
            builder, patient_ids, num_workers=num_workers,
            force_rebuild=force_rebuild,
            stage=stage,
        )
        self._registry_queue_workers[stage] = worker
        results: list[dict] = []
        state = {"index": 0, "total": len(patient_ids), "patient": ""}
        progress = ProgressDialog(
            f"Coda {stage} — paziente 1/{len(patient_ids)}", parent=self,
        )
        progress.show()
        if concurrent:
            progress.add_log(
                "⚠️ In parallelo alla coda dell'altra fase: i due modelli "
                "locali saranno residenti insieme in RAM e le prestazioni "
                "potranno ridursi."
            )
        process_gui_events()

        def on_started(index: int, total: int, patient_id: str) -> None:
            state.update(index=index, total=total, patient=patient_id)
            progress.setWindowTitle(
                f"Coda {stage} — paziente {index}/{total}"
            )
            progress.add_log(
                f"\n===== Paziente {index}/{total}: {patient_id} ====="
            )
            progress.set_progress(
                int((index - 1) * 100 / total),
                f"Avvio registro di {patient_id}...",
            )

        def on_progress(patient_id: str, percent: int, message: str) -> None:
            index = state["index"]
            total = max(state["total"], 1)
            overall = int(((index - 1) + percent / 100) * 100 / total)
            progress.set_progress(
                overall,
                f"{patient_id} ({percent}%): {message}",
            )

        def on_finished(patient_id: str, result: dict) -> None:
            results.append({
                "patient_id": patient_id, "result": result, "error": None,
            })
            processed = int(result.get("documents_processed", 0) or 0)
            skipped = int(result.get("documents_skipped", 0) or 0)
            result_stage = result.get("stage") or stage
            if result_stage == "events":
                label = (
                    f"eventi completati ({result.get('final_entries', 0)} voci)"
                )
            elif result_stage == "validation":
                label = (
                    "validazione preparata "
                    f"({result.get('validation_pending', 0)} elementi)"
                )
            else:
                label = (
                    "evidenze già aggiornate"
                    if processed == 0 and skipped else
                    f"evidenze completate ({processed} documenti elaborati)"
                )
            progress.add_log(f"✓ {patient_id}: {label}")
            progress.set_progress(
                int(state["index"] * 100 / max(state["total"], 1)),
                f"{patient_id}: {label}",
            )

        def on_error(patient_id: str, error: str) -> None:
            results.append({
                "patient_id": patient_id, "result": {}, "error": error,
            })
            progress.add_log(f"❌ {patient_id}: {error}")

        worker.patient_started.connect(on_started)
        worker.patient_progress.connect(on_progress)
        worker.patient_finished.connect(on_finished)
        worker.patient_error.connect(on_error)
        progress.cancelled.connect(worker.cancel)

        def on_queue_finished() -> None:
            cancelled = progress.is_cancelled()
            worker.deleteLater()
            self._registry_queue_workers.pop(stage, None)
            progress.mark_done()
            progress.accept()
            if self._current_patient_id:
                self._clinical_history_tab.load_patient(
                    self._current_patient_id
                )
            from .registry_queue_result_dialog import (
                RegistryQueueResultDialog,
            )
            RegistryQueueResultDialog(
                results, cancelled=cancelled, parent=self,
            ).exec_()

        worker.finished.connect(on_queue_finished)
        worker.start()

    # ------------------------------------------------------------------
    # Multi-patient irAE analysis queue
    # ------------------------------------------------------------------

    @staticmethod
    def _build_irae_plans(
        services: dict, patient_ids: list[str], protocol: str
    ) -> list[tuple[str, list[str]]]:
        """Pre-build the analysis prompts of every patient (main thread:
        SQLite must not be touched from the worker)."""
        from ..clinical.irae_analysis import build_analysis_plan

        timeline_repo = services.get("timeline_repo")
        cs_repo = services.get("cs_repo")
        plans = []
        for pid in patient_ids:
            entries = timeline_repo.get_by_patient(pid) if timeline_repo else []
            profile = ""
            if cs_repo:
                state = cs_repo.load(pid)
                profile = state.clinical_profile if state else ""
            prompts = build_analysis_plan(
                [e.to_dict() for e in entries], profile, protocol
            )
            plans.append((pid, prompts))
        return plans

    def run_irae_queue(self, patient_ids: list[str]):
        """Run the irAE protocol over several registries, in order.

        Plans are built on the main thread; the LLM calls run in a
        background worker so the UI stays responsive.  A summary dialog
        with one tab per patient opens at the end.
        """
        if not patient_ids:
            return

        llm = self._services.get("clinical_state_llm_client")
        if not llm or not llm.is_available:
            QMessageBox.warning(
                self, "LLM non disponibile",
                "Il modello Clinical State non è disponibile.",
            )
            return

        from ..clinical import irae_analysis

        try:
            prompt_path = irae_analysis.ensure_prompt()
            protocol = irae_analysis.load_prompt(prompt_path)
        except OSError as exc:
            QMessageBox.warning(
                self, "Protocollo non disponibile", str(exc)
            )
            return
        if not protocol:
            QMessageBox.warning(
                self, "Protocollo non disponibile",
                f"Il file del protocollo irAE è vuoto: {prompt_path}",
            )
            return

        plans = self._build_irae_plans(self._services, patient_ids, protocol)

        from .workers import IraeQueueWorker

        self._irae_queue_worker = IraeQueueWorker(llm, plans)
        worker = self._irae_queue_worker
        results: list[dict] = []

        total = len(plans)
        progress = ProgressDialog(
            f"Coda analisi irAE — paziente 1/{total}", parent=self,
        )
        progress.show()
        process_gui_events()

        worker.patient_started.connect(
            lambda idx, n, pid, progress=progress: (
                progress.setWindowTitle(
                    f"Coda analisi irAE — paziente {idx}/{n}"
                ),
                progress.add_log(f"\n===== Paziente {idx}/{n}: {pid} ====="),
            )
        )
        worker.chunk_progress.connect(
            lambda chunk, n, progress=progress: progress.set_progress(
                int(chunk * 100 / n),
                f"Analisi parte {chunk}/{n}...",
            )
        )
        worker.patient_finished.connect(
            lambda pid, markdown, results=results: (
                results.append({
                    "patient_id": pid, "label": pid, "markdown": markdown,
                    "error": None,
                }),
                progress.add_log(f"✓ {pid}: analisi completata"),
            )
        )
        worker.patient_error.connect(
            lambda pid, error, results=results: (
                results.append({
                    "patient_id": pid, "label": pid, "markdown": "",
                    "error": error,
                }),
                progress.add_log(f"❌ {pid}: {error}"),
            )
        )
        progress.cancelled.connect(worker.cancel)

        def _on_queue_finished():
            worker.deleteLater()
            self._irae_queue_worker = None
            progress.mark_done()
            progress.accept()
            if results:
                from .irae_queue_result_dialog import IraeQueueResultDialog
                dialog = IraeQueueResultDialog(results, parent=self)
                dialog.exec_()
            else:
                QMessageBox.information(
                    self, "Nessun risultato",
                    "Nessuna analisi completata.",
                )

        worker.finished.connect(_on_queue_finished)
        worker.start()

    def run_extraction_for_docs(self, grouped: dict[str, list[str]]):
        """Run clinical-text extraction for explicit doc_ids, grouped by patient.

        Mirrors :meth:`run_extraction_queue` but passes the selected documents
        to ``extract_clinical_text``, so only the requested files are parsed
        and LLM-normalized (other pending documents of the patient are left
        untouched).
        """
        patients = [pid for pid, ids in grouped.items() if ids]
        if not patients:
            return

        total = len(patients)
        progress = ProgressDialog(
            f"Estrazione documenti selezionati — paziente 1/{total}",
            parent=self,
        )
        progress.show()
        process_gui_events()

        for idx, pid in enumerate(patients, start=1):
            if progress.is_cancelled():
                break
            progress.reset_for_reuse()
            progress.setWindowTitle(
                f"Estrazione documenti selezionati — paziente {idx}/{total}"
            )
            progress.add_log(f"\n===== Paziente {idx}/{total}: {pid} =====")
            self.load_patient(pid)
            try:
                self._documents_tab.extract_clinical_text(
                    doc_ids=grouped[pid],
                    progress=progress,
                    patient_label=f"Paziente {idx}/{total}: {pid}",
                )
            except Exception as exc:
                progress.add_log(f"❌ Errore per {pid}: {exc}")

        progress.mark_done()
        progress.exec_()

    def _on_document_selected(self, doc_id: str, doc_data: dict):
        self._current_document_id = doc_id
        doc_data["_patient_id"] = self._current_patient_id
        self.context_requested.emit("document", doc_data)

    def _on_import_requested(self, files: list[str]):
        self.show_import_dialog(files)

    def _on_lab_selected(self, lab_data: dict):
        self.context_requested.emit("lab", lab_data)

    def _on_processing_complete(self, patient_id: str):
        """Refresh all tabs after document processing completes."""
        self._laboratory_tab.load_patient(patient_id)
        self._clinical_history_tab.load_patient(patient_id)
        self._validation_tab.load_patient(patient_id)

    def reprocess_document(self, doc_id: str):
        """Re-run the clinical-text pipeline for one document."""
        self._documents_tab.reprocess_document(doc_id)

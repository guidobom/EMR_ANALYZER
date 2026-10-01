"""Main window for EMR Analyzer."""

import os
from pathlib import Path

from PyQt5.QtWidgets import (
    QMainWindow, QToolBar, QStatusBar, QAction,
    QSplitter, QMessageBox, QFileDialog, QWidget,
    QLabel, QApplication, QSizePolicy, QDialog,
)
from PyQt5.QtCore import Qt, QSize

from .patient_panel import PatientPanel
from .workspace_tabs import WorkspaceTabs
from .context_panel import ContextPanel
from .llm_config_dialog import LLMConfigDialog
from .prompt_manager_dialog import PromptManagerDialog
from .styles import MAIN_STYLESHEET
from ..config import APP_NAME, APP_VERSION, active_workspace
from ..extraction.llm_client import LlmClient
from ..settings import MODEL_ROLES, load_llm_configs, save_llm_configs
from ..utils.file_utils import supported_file_dialog_filter


class MainWindow(QMainWindow):
    """Top-level application window."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._current_patient_id = None
        self._services = {}
        self._ollama_available = False
        self._closing = False
        self.setWindowTitle(f"{APP_NAME} v{APP_VERSION}")
        self.resize(1400, 900)
        self.setMinimumSize(1024, 700)

        # Apply stylesheet
        self.setStyleSheet(MAIN_STYLESHEET)

        # Initialize components
        self._setup_menu_bar()
        self._setup_toolbar()
        self._setup_status_bar()
        self._setup_central_widget()

    def set_services(self, services: dict):
        """Inject backend services after construction."""
        self._services = services
        self._pipeline_llm_defaults = dict(services.get("llm_configs") or load_llm_configs())
        services["prepare_pipeline_llm"] = self._prepare_pipeline_llm
        self.patient_panel.set_services(services)
        self.workspace_tabs.set_services(services)
        self.context_panel.set_services(services)
        # Connect context panel
        self.workspace_tabs.context_requested.connect(self._on_context_requested)
        self.workspace_tabs.patient_created.connect(
            lambda pid: self.patient_panel.refresh()
        )
        self._update_llm_summary()

    # ------------------------------------------------------------------
    # Menu Bar
    # ------------------------------------------------------------------
    def _setup_menu_bar(self):
        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu("&File")

        new_patient_action = QAction("&Nuovo Paziente", self)
        new_patient_action.setShortcut("Ctrl+N")
        new_patient_action.triggered.connect(self._on_new_patient)
        file_menu.addAction(new_patient_action)

        open_workspace_action = QAction("&Apri Workspace", self)
        open_workspace_action.setShortcut("Ctrl+O")
        open_workspace_action.triggered.connect(self._on_open_workspace)
        file_menu.addAction(open_workspace_action)

        file_menu.addSeparator()

        import_action = QAction("&Importa Documenti", self)
        import_action.setShortcut("Ctrl+I")
        import_action.triggered.connect(self._on_import_documents)
        file_menu.addAction(import_action)

        merge_action = QAction("Unisci &workspace pazienti...", self)
        merge_action.triggered.connect(self._on_merge_workspaces)
        file_menu.addAction(merge_action)

        file_menu.addSeparator()

        fhir_export_action = QAction("Esporta &FHIR del progetto (NDJSON)...", self)
        fhir_export_action.setToolTip(
            "Rigenera il file FHIR di ogni paziente e scrive un unico file NDJSON "
            "del progetto (una risorsa FHIR per riga)"
        )
        fhir_export_action.triggered.connect(self._on_export_project_fhir)
        file_menu.addAction(fhir_export_action)

        file_menu.addSeparator()

        quit_action = QAction("&Esci", self)
        quit_action.setShortcut("Ctrl+Q")
        quit_action.triggered.connect(self.close)
        file_menu.addAction(quit_action)

        # Tools menu
        tools_menu = menubar.addMenu("&Strumenti")

        reprocess_action = QAction("&Rielabora Documento", self)
        reprocess_action.triggered.connect(self._on_reprocess)
        tools_menu.addAction(reprocess_action)

        registry_queue_action = QAction(
            "Elabora &pazienti (eventi e codici SNOMED)...", self
        )
        registry_queue_action.triggered.connect(self._on_show_registry_queue)
        tools_menu.addAction(registry_queue_action)

        prompts_action = QAction("Gestisci &prompt LLM...", self)
        prompts_action.setToolTip(
            "Apre, modifica, versiona e assegna i prompt alle pipeline"
        )
        prompts_action.triggered.connect(self._open_prompt_manager)
        tools_menu.addAction(prompts_action)

        tools_menu.addSeparator()

        validate_action = QAction("&Attribuzioni dei documenti", self)
        validate_action.setShortcut("Ctrl+V")
        validate_action.triggered.connect(
            lambda: self.workspace_tabs.setCurrentWidget(
                self.workspace_tabs._validation_tab
            )
        )
        tools_menu.addAction(validate_action)

        # Help menu
        help_menu = menubar.addMenu("&Help")

        about_action = QAction("&Informazioni", self)
        about_action.triggered.connect(self._on_about)
        help_menu.addAction(about_action)

    # ------------------------------------------------------------------
    # Toolbar
    # ------------------------------------------------------------------
    def _setup_toolbar(self):
        toolbar = QToolBar("Toolbar principale")
        toolbar.setMovable(False)
        toolbar.setIconSize(QSize(24, 24))
        self.addToolBar(Qt.TopToolBarArea, toolbar)

        new_patient_btn = QAction("👤 Nuovo", self)
        new_patient_btn.triggered.connect(self._on_new_patient)
        toolbar.addAction(new_patient_btn)

        toolbar.addSeparator()

        import_btn = QAction("📥 Importa", self)
        import_btn.triggered.connect(self._on_import_documents)
        toolbar.addAction(import_btn)

        # Workspace-wide view of documents still needing normalization or
        # carrying an error, with the option to launch the extraction on them.
        pending_btn = QAction("🧹 Non normalizzati", self)
        pending_btn.setToolTip(
            "Mostra tutti i documenti ancora da normalizzare o con errori e "
            "avvia l'estrazione del testo clinico sui selezionati"
        )
        pending_btn.triggered.connect(self._on_show_pending)
        toolbar.addAction(pending_btn)

        registry_queue_btn = QAction("▶ Elabora pazienti", self)
        registry_queue_btn.setToolTip(
            "Estrae gli eventi clinici dei pazienti selezionati, li codifica "
            "in SNOMED CT e scrive i file FHIR"
        )
        registry_queue_btn.triggered.connect(self._on_show_registry_queue)
        toolbar.addAction(registry_queue_btn)

        concepts_btn = QAction("🧬 Concetti SNOMED", self)
        concepts_btn.setToolTip(
            "Rivedi le codifiche SNOMED CT per concetto in tutto il progetto"
        )
        concepts_btn.triggered.connect(self._on_show_concepts)
        toolbar.addAction(concepts_btn)

        # Spacer
        spacer = QWidget()
        spacer.setMinimumWidth(20)
        toolbar.addWidget(spacer)

        # Deterministic PDF parsing status.
        self._model_label = QLabel("PDF: ⚡ pipeline pronta")
        self._model_label.setStyleSheet("color: #27ae60; font-weight: bold;")
        self._model_label.setToolTip(
            "Estrazione automatica: pdfplumber → PyMuPDF → OCR locale"
        )
        toolbar.addWidget(self._model_label)

        self._prompt_manager_action = QAction("✎ Prompt", self)
        self._prompt_manager_action.setToolTip(
            "Apri il Prompt Manager e scegli la versione per ogni compito"
        )
        self._prompt_manager_action.triggered.connect(
            self._open_prompt_manager
        )
        toolbar.addAction(self._prompt_manager_action)

        self._ollama_label = QLabel(" 🔌")
        self._ollama_label.setToolTip("Stato del motore locale (llama.cpp)")
        toolbar.addWidget(self._ollama_label)

        # Spacer that pushes the exit button to the right
        right_spacer = QWidget()
        right_spacer.setSizePolicy(
            QSizePolicy.Expanding, QSizePolicy.Preferred
        )
        toolbar.addWidget(right_spacer)

        exit_btn = QAction("🚪 Esci", self)
        exit_btn.setToolTip("Chiudi l'applicazione")
        exit_btn.triggered.connect(self.close)
        toolbar.addAction(exit_btn)

    # ------------------------------------------------------------------
    # Status Bar
    # ------------------------------------------------------------------
    def _setup_status_bar(self):
        self.statusbar = QStatusBar()
        self.setStatusBar(self.statusbar)
        self._status_patient = QLabel("Nessun paziente selezionato")
        self._status_docs = QLabel("")
        self._status_progress = QLabel("")
        self.statusbar.addWidget(self._status_patient)
        self.statusbar.addPermanentWidget(self._status_docs)
        self.statusbar.addPermanentWidget(self._status_progress)

    # ------------------------------------------------------------------
    # Central Widget (3-panel layout)
    # ------------------------------------------------------------------
    def _setup_central_widget(self):
        # Left panel: patient list
        self.patient_panel = PatientPanel()
        self.patient_panel.patient_selected.connect(self._on_patient_selected)

        # Center: tab workspace
        self.workspace_tabs = WorkspaceTabs()

        # Right panel: context/details
        self.context_panel = ContextPanel()

        # Horizontal splitter: left | center | right
        h_splitter = QSplitter(Qt.Horizontal)
        h_splitter.addWidget(self.patient_panel)
        h_splitter.addWidget(self.workspace_tabs)
        h_splitter.addWidget(self.context_panel)
        h_splitter.setStretchFactor(0, 1)    # left: narrow
        h_splitter.setStretchFactor(1, 4)    # center: wide
        h_splitter.setStretchFactor(2, 1)    # right: narrow

        self.setCentralWidget(h_splitter)

    # ------------------------------------------------------------------
    # Event Handlers
    # ------------------------------------------------------------------

    def _open_prompt_manager(self):
        dialog = PromptManagerDialog(
            self,
            services=self._services,
            patient_id=self._current_patient_id,
        )
        dialog.exec_()
        self.statusbar.showMessage(
            "Prompt aggiornati; riavvia l'app se hai attivato o modificato "
            "una versione",
            7000,
        )


    def _on_new_patient(self):
        from .patient_panel import NewPatientDialog
        dialog = NewPatientDialog(self._services.get("patient_repo"), self)
        if dialog.exec_():
            self.patient_panel.refresh()

    def _on_open_workspace(self):
        directory = QFileDialog.getExistingDirectory(
            self, "Apri Workspace Paziente", str(active_workspace.path)
        )
        if directory:
            patient_id = Path(directory).name
            patient_repo = self._services.get("patient_repo")
            if patient_repo and patient_repo.get_by_id(patient_id) is None:
                reply = QMessageBox.question(
                    self,
                    "Workspace non registrato",
                    f"La cartella {patient_id} esiste, ma il paziente non è "
                    "presente nel registro corrente.\n\n"
                    "Registrare nuovamente questo workspace?",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.No,
                )
                if reply != QMessageBox.Yes:
                    return
                from datetime import datetime
                from ..models import Patient
                now = datetime.now().isoformat()
                pseudonym = (
                    patient_id[1:]
                    if patient_id.startswith("P") and patient_id[1:]
                    else patient_id
                )
                try:
                    patient_repo.insert(Patient(
                        id=patient_id,
                        pseudonym=pseudonym,
                        created_at=now,
                        updated_at=now,
                    ))
                    if hasattr(self, "patient_panel"):
                        self.patient_panel.refresh()
                except Exception as exc:
                    QMessageBox.critical(
                        self,
                        "Recupero non riuscito",
                        f"Impossibile registrare il workspace {patient_id}:\n{exc}",
                    )
                    return
            self._on_patient_selected(patient_id)

    def _on_import_documents(self):
        if not self._current_patient_id:
            QMessageBox.warning(self, "Attenzione",
                                "Seleziona prima un paziente.")
            return
        files, _ = QFileDialog.getOpenFileNames(
            self, "Seleziona documenti clinici",
            "", supported_file_dialog_filter()
        )
        if files:
            self.workspace_tabs.show_import_dialog(files)


    def _on_merge_workspaces(self):
        """Merge one or more patient workspaces into others (same project)."""
        from .merge_workspace_dialog import MergeWorkspaceDialog
        dialog = MergeWorkspaceDialog(self._services, self)
        if dialog.exec_() != MergeWorkspaceDialog.Accepted:
            return
        self.patient_panel.refresh()
        if self._current_patient_id in dialog.removed_sources:
            # The selected workspace no longer exists.
            self._on_patient_selected("")
        else:
            # Reload so moved documents appear in the target workspace.
            self.workspace_tabs.load_patient(self._current_patient_id)


    def _on_show_pending(self):
        """List every document needing normalization or with an error.

        The dialog returns the checked documents grouped by patient; the
        extraction queue is launched only after the dialog closes, so the
        modal selection window never sits behind the progress dialog.
        """
        from .normalization_dialog import (
            NormalizationDialog, classify_pending_documents,
        )
        doc_repo = self._services.get("document_repo")
        if not doc_repo:
            QMessageBox.warning(
                self, "Errore", "Servizio documenti non disponibile."
            )
            return
        classification = classify_pending_documents(doc_repo.list_all())
        if classification.total == 0:
            QMessageBox.information(
                self, "Nessun documento pendente",
                "Nessun documento da normalizzare o con errori nel workspace.",
            )
            return
        dialog = NormalizationDialog(
            classification, parent=self, services=self._services
        )
        if dialog.exec_() == QDialog.Accepted:
            grouped = dialog.selected_groups()
            if grouped:
                self.workspace_tabs.run_extraction_for_docs(grouped)


    def _on_export_project_fhir(self):
        """Export the reviewed events of every patient as FHIR NDJSON."""
        pipeline = self._services.get("extraction_pipeline")
        if pipeline is None:
            return
        if self.workspace_tabs.llm_operation_running():
            QMessageBox.information(self, "Elaborazione in corso",
                                    "Attendi il termine dell'elaborazione prima di esportare.")
            return
        from .progress_dialog import ProgressDialog
        from .workers import ProjectExportWorker

        progress = ProgressDialog("Esportazione FHIR del progetto", parent=self)
        worker = ProjectExportWorker(pipeline, self)
        worker.progress.connect(lambda percent, message: progress.set_progress(percent, message))
        progress.cancelled.connect(worker.cancel)

        def finish(message, *, error=False):
            progress.mark_done()
            progress.accept()
            (QMessageBox.warning if error else QMessageBox.information)(
                self, "Esportazione FHIR", message)

        worker.result_ready.connect(lambda result: finish(
            f"{result['patients']} pazienti, {result['events']} risorse cliniche "
            f"({result['uncoded']} senza codice), {result['resources']} righe.\n\n{result['path']}"))
        worker.cancelled.connect(lambda: finish("Esportazione interrotta."))
        worker.error.connect(lambda message: finish(f"Esportazione non riuscita: {message}", error=True))
        worker.finished.connect(worker.deleteLater)
        self._project_export_worker = worker
        progress.show()
        worker.start()


    def _on_show_concepts(self):
        """Project-wide review of concept → SNOMED CT codes."""
        from .concept_review_dialog import ConceptReviewDialog
        ConceptReviewDialog(self._services, self).exec_()
        if self._current_patient_id:
            self.workspace_tabs.load_patient(self._current_patient_id)

    def _on_show_registry_queue(self):
        """Select patients and extract, code and export their events."""
        if self.workspace_tabs.llm_operation_running():
            QMessageBox.information(
                self, "LLM occupato",
                "Attendi il completamento o annulla l'elaborazione LLM "
                "attualmente in corso.",
            )
            return

        from .registry_queue_dialog import (
            RegistryQueueDialog,
            build_registry_queue_summaries,
        )

        summaries = build_registry_queue_summaries(self._services)
        if not summaries:
            QMessageBox.information(
                self, "Nessun documento",
                "Nessun paziente del workspace contiene documenti clinici.",
            )
            return
        if not any(summary.get("eligible") for summary in summaries):
            QMessageBox.information(
                self, "Nessun documento normalizzato",
                "Prima di elaborare gli eventi occorre normalizzare almeno un "
                "documento clinico.",
            )
            return

        dialog = RegistryQueueDialog(summaries, parent=self)
        if dialog.exec_() != QDialog.Accepted:
            return
        selected = dialog.selected_patient_ids()
        if selected:
            self.workspace_tabs.run_registry_queue(
                selected, force_rebuild=dialog.force_rebuild(),
            )

    def _on_about(self):
        QMessageBox.about(
            self, f"Informazioni su {APP_NAME}",
            f"<h3>{APP_NAME} v{APP_VERSION}</h3>"
            f"<p>Ambiente di lavoro clinico-documentale per l'analisi "
            f"longitudinale della documentazione sanitaria.</p>"
            f"<p>Basato su pdfplumber e modelli LLM locali configurabili.</p>"
        )

    def _on_reprocess(self):
        current_doc = getattr(self.workspace_tabs, '_current_document_id', None)
        if current_doc:
            self.workspace_tabs.reprocess_document(current_doc)

    def _on_patient_selected(self, patient_id: str):
        """Called when a patient is selected in the left panel."""
        self._current_patient_id = patient_id
        self._status_patient.setText(
            f"Paziente: {patient_id}" if patient_id else "Nessun paziente selezionato"
        )
        self.workspace_tabs.load_patient(patient_id)
        self.context_panel.clear()

    def _on_context_requested(self, item_type: str, data: dict):
        """Update the context panel when an item is selected."""
        if item_type == "document":
            patient_id = data.pop("_patient_id", self._current_patient_id)
            self.context_panel.show_document_context(data, patient_id or "")
        elif item_type == "event":
            self.context_panel.show_event_context(data)
        elif item_type == "lab":
            self.context_panel.show_lab_context(data)

    def update_model_status(self, ollama_ok: bool = False):
        """Update the shared local-LLM connection indicator."""
        self._model_label.setText("PDF: ⚡ pipeline pronta")
        self._model_label.setStyleSheet("color: #27ae60; font-weight: bold;")
        self._ollama_available = ollama_ok
        if ollama_ok:
            self._ollama_label.setText(" 🔌✓")
            self._ollama_label.setStyleSheet("color: #27ae60; font-weight: bold;")
        else:
            self._ollama_label.setText(" 🔌✗")
            self._ollama_label.setStyleSheet("color: #e74c3c;")
        self._update_llm_summary()

    def _prepare_pipeline_llm(self, key, parent=None) -> bool:
        from .pipeline_llm import pipeline_definition
        from ..settings import load_pipeline_llm_config, save_pipeline_llm_config

        if self.workspace_tabs.llm_operation_running():
            QMessageBox.information(parent or self, "Elaborazione in corso",
                                    "Attendi il termine dell’elaborazione prima di avviare una nuova analisi.")
            return False
        role, title = pipeline_definition(key)
        defaults = self._pipeline_llm_defaults
        config = load_pipeline_llm_config(key, defaults[role])
        try:
            available = LlmClient.list_available_models()
            dialog = LLMConfigDialog({role: config}, available, parent or self,
                                     role=role, pipeline_title=title,
                                     enable_slot_benchmark=(key == "atomic"))
            if dialog.exec_() != QDialog.Accepted:
                return False
            selected = dialog.configurations()[role]
            if key == "atomic" and selected.temperature > 0.2:
                answer = QMessageBox.question(
                    parent or self, "Temperatura elevata",
                    f"La temperatura del modello è {selected.temperature:g}. Per un'estrazione "
                    "riproducibile e conforme allo schema si consiglia 0 (al massimo 0,2).\n\n"
                    "Continuare comunque?",
                    QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
                )
                if answer != QMessageBox.Yes:
                    return False
            if not selected.model or not LlmClient(config=selected).is_available:
                QMessageBox.warning(parent or self, "Modello non disponibile",
                                    "Scarica o importa il modello selezionato prima di avviare l’analisi.")
                return False
            save_pipeline_llm_config(key, selected)
            configs = dict(self._services.get("llm_configs") or defaults)
            configs[role] = selected
            self._apply_llm_configs(configs, persist=False, active_role=role)
            return self._services.get(f"{role}_llm_client") is not None
        except (OSError, ValueError, TypeError, RuntimeError) as exc:
            QMessageBox.warning(parent or self, "Analisi non avviata", str(exc))
            return False

    def _apply_llm_configs(self, configs, *, persist=True, active_role=None) -> None:
        """Persist settings and replace all live clients atomically."""
        old_configs = self._services.get("llm_configs") or {}
        if persist:
            save_llm_configs(configs)

        # A server is keyed by GGUF path, context and slot count. Stop only
        # old shapes no longer referenced by any role. Request-scoped
        # changes such as temperature and output length keep the shared
        # process alive.
        new_runtime_ids = set()
        for config in configs.values():
            if not config.model:
                continue
            try:
                new_runtime_ids.add(
                    LlmClient(config=config).runtime_identity()
                )
            except Exception:
                pass
        obsolete_configs = []
        obsolete_seen = set()
        for config in old_configs.values():
            if not config.model:
                continue
            try:
                identity = LlmClient(config=config).runtime_identity()
            except Exception:
                continue
            if (
                identity not in new_runtime_ids
                and identity not in obsolete_seen
            ):
                obsolete_seen.add(identity)
                obsolete_configs.append(config)
        cleanup_result = (
            LlmClient.unload_runtimes(obsolete_configs)
            if obsolete_configs else {"errors": {}}
        )

        clients = {}
        unavailable = []
        for role in MODEL_ROLES:
            if active_role is not None and role != active_role:
                clients[role] = self._services.get(f"{role}_llm_client")
                continue
            config = configs[role]
            client = LlmClient(config=config) if config.model else None
            if client is not None and not client.is_available:
                unavailable.append(config.model)
                client = None
            clients[role] = client

        configs = {role: configs[role] for role in MODEL_ROLES}
        document_client = clients["document"]
        atomic_client = clients["atomic_evidence"]
        self._services.update({
            "llm_configs": configs,
            "document_llm_client": document_client,
            "atomic_evidence_llm_client": atomic_client,
        })
        if active_role in (None, "document"):
            self._propagate_document_llm(document_client)
        if active_role in (None, "atomic_evidence"):
            self._propagate_atomic_llm(atomic_client)
        self._ollama_available = any(
            client is not None and client.server_available
            for client in clients.values()
        )
        self._services["ollama_available"] = self._ollama_available
        self.update_model_status(self._ollama_available)
        self.statusbar.showMessage("Configurazione LLM salvata e applicata", 6000)

        cleanup_errors = cleanup_result.get("errors") or {}
        if cleanup_errors:
            QMessageBox.warning(
                self,
                "Runtime precedente non scaricato",
                "La configurazione è stata applicata, ma non è stato possibile "
                "fermare uno o più server precedenti:\n- "
                + "\n- ".join(
                    f"{name}: {error}"
                    for name, error in cleanup_errors.items()
                ),
            )

        if unavailable:
            QMessageBox.warning(
                self,
                "Modelli non disponibili",
                "Configurazione salvata, ma questi modelli non risultano "
                "nell'archivio locale del backend selezionato:\n- "
                + "\n- ".join(unavailable),
            )

    def _update_llm_summary(self) -> None:
        configs = self._services.get("llm_configs")
        if not configs:
            self._ollama_label.setToolTip(
                "Motore LLM locale pronto" if self._ollama_available
                else "Motore locale non disponibile — esegui "
                     "lo strumento di setup llama.cpp/vLLM"
            )
            return
        configs = {role: configs[role] for role in MODEL_ROLES if role in configs}
        connection = (
            "Motore LLM locale pronto" if self._ollama_available
            else "Motore locale non disponibile — esegui "
                 "lo strumento di setup llama.cpp/vLLM"
        )
        try:
            runtime_ids = {
                role: (
                    LlmClient(config=config).runtime_identity()
                    if config.model else None
                )
                for role, config in configs.items()
            }
        except Exception:
            runtime_ids = {}
        physical_runtimes = {
            identity for identity in runtime_ids.values()
            if identity is not None
        }
        configured_count = sum(bool(config.model) for config in configs.values())
        if len(physical_runtimes) == 1 and configured_count > 1:
            runtime_note = "Un solo server fisico condiviso"
        elif physical_runtimes:
            runtime_note = f"{len(physical_runtimes)} server fisici configurati"
        else:
            runtime_note = "Nessun server configurato"
        labels = {
            "document": "Documenti",
            "atomic_evidence": "Estrazione e codifica",
        }
        role_lines = [
            f"{labels[role]}: {configs[role].model or 'off'} "
            f"[{configs[role].backend}] — "
            f"ctx {configs[role].context_length}, "
            f"{configs[role].parallel_workers} slot, "
            f"T {configs[role].temperature:g}"
            for role in MODEL_ROLES if role in configs
        ]
        details = "\n".join([connection, runtime_note, *role_lines])
        self._ollama_label.setToolTip(details)

    def _propagate_document_llm(self, client):
        """Update the plain-text document normalizer only."""
        isolator = self._services.get("clinical_text_isolator")
        if isolator is not None:
            isolator.llm = client

    def _propagate_atomic_llm(self, client):
        """Update the event extraction and coding stage."""
        pipeline = self._services.get("extraction_pipeline")
        if pipeline is not None:
            pipeline.set_llm(client)


    def closeEvent(self, event):
        """Always permit an explicit exit, even while background work runs."""

        if getattr(self, "_closing", False):
            event.accept()
            return

        from .application_shutdown import (
            mark_shutdown_requested,
            request_qthread_shutdown,
            running_qthreads,
            schedule_emergency_exit,
        )

        threads = running_qthreads()
        workspace_busy = self.workspace_tabs.llm_operation_running()
        work_in_progress = workspace_busy or bool(threads)
        if work_in_progress:
            reply = QMessageBox.question(
                self,
                "Interrompere le elaborazioni e uscire?",
                "È in corso almeno un'elaborazione. Puoi comunque chiudere "
                "EMR Analyzer.\n\n"
                "I risultati già salvati resteranno disponibili; l'unità di "
                "lavoro attiva potrebbe rimanere incompleta e verrà "
                "riconosciuta come interrotta al prossimo avvio.\n\n"
                "Interrompere ora tutte le attività e chiudere?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if reply != QMessageBox.Yes:
                event.ignore()
                return

        self._closing = True
        mark_shutdown_requested()
        # Accept and hide before cleanup: even if an external library ignores
        # cancellation, the application visibly closes immediately.
        event.accept()
        hide = getattr(self, "hide", None)
        if callable(hide):
            hide()

        if work_in_progress:
            # Absolute guarantee requested by the user.  Normal cooperative
            # shutdown normally wins; this fires only if a worker/library is
            # still stuck after the grace period.
            schedule_emergency_exit(delay_seconds=5.0)

        request_shutdown = getattr(
            self.workspace_tabs, "request_shutdown", None
        )
        if callable(request_shutdown):
            request_shutdown()
        request_qthread_shutdown(threads)

        if work_in_progress:
            # Breaking local HTTP connections makes QThreads blocked in a
            # long LLM request return promptly instead of waiting 30 minutes.
            try:
                from ..llm_backend import shutdown_all_backends
                shutdown_all_backends()
            except Exception:
                pass
        try:
            self.workspace_tabs.shutdown(wait_ms=1000)
        except Exception:
            pass
        if "db" in self._services and self._services["db"]:
            try:
                self._services["db"].close()
            except Exception:
                pass

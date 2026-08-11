"""Main window for EMR Analyzer."""

import os
from pathlib import Path

from PyQt5.QtWidgets import (
    QMainWindow, QToolBar, QStatusBar, QAction,
    QSplitter, QMessageBox, QFileDialog, QWidget,
    QLabel, QApplication, QSizePolicy,
)
from PyQt5.QtCore import Qt, QSize

from .patient_panel import PatientPanel
from .workspace_tabs import WorkspaceTabs
from .context_panel import ContextPanel
from .llm_config_dialog import LLMConfigDialog
from .styles import MAIN_STYLESHEET
from ..config import APP_NAME, APP_VERSION, active_workspace
from ..extraction.llm_client import LlmClient
from ..settings import load_llm_configs, save_llm_configs


class MainWindow(QMainWindow):
    """Top-level application window."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._current_patient_id = None
        self._services = {}
        self._ollama_available = False
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
        self.patient_panel.set_services(services)
        self.workspace_tabs.set_services(services)
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

        import_patient_action = QAction("Importa &paziente da altro progetto...", self)
        import_patient_action.triggered.connect(self._on_import_patient)
        file_menu.addAction(import_patient_action)

        merge_action = QAction("Unisci &workspace pazienti...", self)
        merge_action.triggered.connect(self._on_merge_workspaces)
        file_menu.addAction(merge_action)

        file_menu.addSeparator()

        export_action = QAction("&Esporta...", self)
        export_action.setShortcut("Ctrl+E")
        export_action.triggered.connect(self._on_export)
        file_menu.addAction(export_action)

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

        tools_menu.addSeparator()

        validate_action = QAction("&Validazione", self)
        validate_action.setShortcut("Ctrl+V")
        validate_action.triggered.connect(lambda: self.workspace_tabs.setCurrentIndex(6))
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

        export_btn = QAction("📤 Esporta", self)
        export_btn.triggered.connect(self._on_export)
        toolbar.addAction(export_btn)

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

        # All model assignments and generation parameters live in one dialog.
        self._configure_llm_action = QAction("⚙ Configura LLM", self)
        self._configure_llm_action.setToolTip(
            "Configura i modelli e i parametri per documenti e Clinical State"
        )
        self._configure_llm_action.triggered.connect(self._open_llm_config)
        toolbar.addAction(self._configure_llm_action)

        self._ollama_label = QLabel(" 🔌")
        self._ollama_label.setToolTip("Stato connessione Ollama")
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
            self, "Seleziona Documenti PDF",
            "", "Documenti (*.pdf *.jpg *.jpeg *.png);;Tutti i file (*)"
        )
        if files:
            self.workspace_tabs.show_import_dialog(files)

    def _on_import_patient(self):
        """Import patients from another project."""
        from .import_patient_dialog import ImportPatientDialog
        dialog = ImportPatientDialog(self)
        if dialog.exec_() == ImportPatientDialog.Accepted:
            self.patient_panel.refresh()

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

    def _on_export(self):
        from .export_dialog import ExportDialog
        if not self._current_patient_id:
            return
        dialog = ExportDialog(self._services, self._current_patient_id, self)
        dialog.exec_()

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
        """Update the shared Ollama connection indicator."""
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

    def _open_llm_config(self) -> None:
        """Open the single configuration surface for both local LLMs."""
        configs = self._services.get("llm_configs") or load_llm_configs()
        try:
            available_models = LlmClient.list_available_models()
            self._ollama_available = True
        except Exception:
            available_models = []
            self._ollama_available = False

        dialog = LLMConfigDialog(configs, available_models, self)
        if dialog.exec_() != LLMConfigDialog.Accepted:
            self.update_model_status(self._ollama_available)
            return
        try:
            self._apply_llm_configs(dialog.configurations())
        except (OSError, ValueError, TypeError) as exc:
            QMessageBox.warning(
                self,
                "Configurazione non salvata",
                f"Impossibile salvare la configurazione LLM.\n\n{exc}",
            )

    def _apply_llm_configs(self, configs) -> None:
        """Persist settings and replace both live clients atomically."""
        save_llm_configs(configs)
        clients = {}
        unavailable = []
        for role in ("document", "clinical_state"):
            config = configs[role]
            client = LlmClient(config=config) if config.model else None
            if client is not None and not client.is_available:
                unavailable.append(config.model)
                client = None
            clients[role] = client

        document_client = clients["document"]
        state_client = clients["clinical_state"]
        self._services.update({
            "llm_configs": configs,
            "document_llm_client": document_client,
            "clinical_state_llm_client": state_client,
        })
        self._propagate_document_llm(document_client)
        self._propagate_state_llm(state_client)
        self._ollama_available = LlmClient().server_available
        self._services["ollama_available"] = self._ollama_available
        self.update_model_status(self._ollama_available)
        self.statusbar.showMessage("Configurazione LLM salvata e applicata", 6000)

        if unavailable:
            QMessageBox.warning(
                self,
                "Modelli non disponibili",
                "Configurazione salvata, ma questi modelli non risultano "
                "disponibili in Ollama:\n- " + "\n- ".join(unavailable),
            )

    def _update_llm_summary(self) -> None:
        configs = self._services.get("llm_configs")
        if not configs:
            self._ollama_label.setToolTip(
                "Ollama connesso" if self._ollama_available
                else "Ollama non disponibile"
            )
            return
        document = configs["document"]
        state = configs["clinical_state"]
        connection = (
            "Ollama connesso" if self._ollama_available
            else "Ollama non disponibile"
        )
        details = (
            f"{connection}\n"
            f"Documenti: {document.model or 'off'} — "
            f"ctx {document.context_length}, T {document.temperature:g}\n"
            f"Clinical State: {state.model or 'off'} — "
            f"ctx {state.context_length}, T {state.temperature:g}"
        )
        self._ollama_label.setToolTip(details)
        self._configure_llm_action.setToolTip(details)

    def _propagate_document_llm(self, client):
        """Update the plain-text document normalizer only."""
        isolator = self._services.get("clinical_text_isolator")
        if isolator is not None:
            isolator.llm = client

    def _propagate_state_llm(self, client):
        """Update services that operate on the longitudinal state."""
        history_builder = self._services.get("clinical_history_builder")
        if history_builder is not None:
            history_builder._llm = client

    def closeEvent(self, event):
        """Close the database connection before exiting."""
        if "db" in self._services and self._services["db"]:
            self._services["db"].close()
        event.accept()

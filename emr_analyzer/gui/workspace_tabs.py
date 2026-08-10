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
from .import_dialog import ImportDialog
from ..models import Patient
from .import_dialog import ImportDialog


class WorkspaceTabs(QTabWidget):
    """Central tab container for the patient workspace."""

    context_requested = pyqtSignal(str, dict)
    patient_created = pyqtSignal(str)  # patient_id

    def __init__(self, parent=None):
        super().__init__(parent)
        self._services = {}
        self._current_patient_id = None
        self._current_document_id = None

        # Create tabs
        self._documents_tab = DocumentsTab()
        self._laboratory_tab = LaboratoryTab()
        self._clinical_history_tab = ClinicalHistoryTab()
        self._validation_tab = ValidationTab()

        # Add tabs
        self.addTab(self._documents_tab, "📄 Documenti")
        self.addTab(self._laboratory_tab, "🔬 Laboratorio")
        self.addTab(self._clinical_history_tab, "📋 Storia Clinica")
        self.addTab(self._validation_tab, "✓ Validazione")

        # Connect signals
        self._documents_tab.document_selected.connect(self._on_document_selected)
        self._documents_tab.import_requested.connect(self._on_import_requested)
        self._documents_tab.processing_complete.connect(self._on_processing_complete)

        # Forward tab selections to context panel
        self._laboratory_tab.lab_selected.connect(self._on_lab_selected)

    def set_services(self, services: dict):
        self._services = services
        self._documents_tab.set_services(services)
        self._laboratory_tab.set_services(services)
        self._clinical_history_tab.set_services(services)
        self._validation_tab.set_services(services)

    def load_patient(self, patient_id: str):
        """Load all tabs with data for the given patient."""
        self._current_patient_id = patient_id
        self._documents_tab.load_patient(patient_id)
        self._laboratory_tab.load_patient(patient_id)
        self._clinical_history_tab.load_patient(patient_id)
        self._validation_tab.load_patient(patient_id)

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

        # Build a map: patient_id → list of file paths
        patient_files: dict[str | None, list[str]] = {}
        routed_files: set[str] = set()

        if groups:
            for g in groups:
                group_file_paths = [
                    d.original_path for d in g.documents
                    if not d.is_duplicate and not d.error
                ]
                if not group_file_paths:
                    continue
                routed_files.update(group_file_paths)

                if g.patient_id:
                    # Existing patient
                    pid = g.patient_id
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
                    # Needs review — use current patient if available
                    pid = self._current_patient_id
                    if not pid:
                        continue

                if pid not in patient_files:
                    patient_files[pid] = []
                patient_files[pid].extend(group_file_paths)

        # Any files not routed → assign to current patient
        unrouted = [f for f in files if f not in routed_files]
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

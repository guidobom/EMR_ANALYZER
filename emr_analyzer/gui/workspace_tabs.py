"""Central QTabWidget workspace for EMR Analyzer."""

from __future__ import annotations

from PyQt6.QtWidgets import QTabWidget, QWidget, QVBoxLayout, QLabel
from PyQt6.QtCore import Qt, pyqtSignal

from .documents_tab import DocumentsTab
from .timeline_tab import TimelineTab
from .laboratory_tab import LaboratoryTab
from .clinical_state_tab import ClinicalStateTab
from .validation_tab import ValidationTab
from .dashboard_tab import DashboardTab
from .import_dialog import ImportDialog
from ..pipeline.patient_routing import build_routing_plan


class WorkspaceTabs(QTabWidget):
    """Central tab container for the patient workspace."""

    context_requested = pyqtSignal(str, dict)  # item_type ("event"|"lab"|"document"), data

    def __init__(self, parent=None):
        super().__init__(parent)
        self._services = {}
        self._current_patient_id = None
        self._current_document_id = None

        # Create all tabs
        self._dashboard_tab = DashboardTab()
        self._documents_tab = DocumentsTab()
        self._timeline_tab = TimelineTab()
        self._laboratory_tab = LaboratoryTab()
        self._clinical_state_tab = ClinicalStateTab()
        self._validation_tab = ValidationTab()

        # Add tabs
        self.addTab(self._dashboard_tab, "📊 Dashboard")
        self.addTab(self._documents_tab, "📄 Documenti")
        self.addTab(self._timeline_tab, "⏱ Timeline")
        self.addTab(self._laboratory_tab, "🔬 Laboratorio")
        self.addTab(self._clinical_state_tab, "🏥 Clinical State")
        self.addTab(self._validation_tab, "✓ Validazione")

        # Connect signals
        self._documents_tab.document_selected.connect(self._on_document_selected)
        self._documents_tab.import_requested.connect(self._on_import_requested)
        self._documents_tab.processing_complete.connect(self._on_processing_complete)

        # Forward tab selections to context panel
        self._timeline_tab.event_selected.connect(self._on_event_selected)
        self._laboratory_tab.lab_selected.connect(self._on_lab_selected)

    def set_services(self, services: dict):
        self._services = services
        self._documents_tab.set_services(services)
        self._timeline_tab.set_services(services)
        self._laboratory_tab.set_services(services)
        self._clinical_state_tab.set_services(services)
        self._validation_tab.set_services(services)
        self._dashboard_tab.set_services(services)

    def load_patient(self, patient_id: str):
        """Load all tabs with data for the given patient."""
        self._current_patient_id = patient_id
        self._documents_tab.load_patient(patient_id)
        self._timeline_tab.load_patient(patient_id)
        self._laboratory_tab.load_patient(patient_id)
        self._clinical_state_tab.load_patient(patient_id)
        self._validation_tab.load_patient(patient_id)
        self._dashboard_tab.load_patient(patient_id)

    def show_import_dialog(self, files: list[str]):
        """Stage, identify and route documents before clinical processing."""
        if not self._services.get("patient_repo"):
            return
        from collections import defaultdict
        from datetime import datetime
        from pathlib import Path
        import shutil

        from PyQt6.QtWidgets import QApplication, QMessageBox, QProgressDialog
        from ..config import WORKSPACES_DIR
        from ..models import Patient

        staging = self._services.get("import_staging")
        router = self._services.get("patient_router")
        patient_repo = self._services.get("patient_repo")
        identity_repo = self._services.get("identity_repo")
        audit_repo = self._services.get("audit_repo")
        if not all((staging, router, patient_repo, identity_repo)):
            QMessageBox.warning(
                self, "Importazione non disponibile",
                "I servizi di identificazione del paziente non sono inizializzati."
            )
            return

        # A directory can survive an interrupted import or an old registry.
        # Do not use its name as a foreign key unless it is actually present in
        # the current database. Strong identity routing can then recreate and
        # safely reuse that workspace ID.
        if (
            self._current_patient_id
            and patient_repo.get_by_id(self._current_patient_id) is None
        ):
            self._current_patient_id = None

        progress = QProgressDialog(
            "Preparazione inbox...", "Annulla", 0, len(files), self
        )
        progress.setWindowTitle("Identificazione documenti")
        progress.setMinimumDuration(0)

        def update_progress(value, maximum, filename):
            progress.setMaximum(maximum)
            progress.setValue(value)
            progress.setLabelText(f"Analisi intestazione: {filename}")
            QApplication.processEvents()

        cancelled = {"value": False}

        def cancel_requested():
            if progress.wasCanceled():
                cancelled["value"] = True
            return cancelled["value"]

        batch = staging.stage(
            files,
            progress_callback=update_progress,
            cancel_check=cancel_requested,
        )
        progress.close()
        if cancelled["value"]:
            batch.cleanup()
            return

        created_patient_ids = []
        try:
            router.bootstrap_existing_identities()
            groups = router.resolve(batch.documents)
            duplicates = [document for document in batch.documents if document.is_duplicate]
            errors = [document for document in batch.documents if document.error]

            plan = build_routing_plan(groups)
            matched_groups = plan.matched_groups
            new_groups = plan.new_groups
            blocked_groups = plan.blocked_groups
            # First import into a manually-created empty workspace: bind its
            # identity instead of creating a redundant second patient.
            if (
                self._current_patient_id
                and patient_repo.get_by_id(self._current_patient_id) is not None
                and len(new_groups) == 1
                and not identity_repo.has_identity(self._current_patient_id)
                and self._services["document_repo"].get_count_by_patient(
                    self._current_patient_id
                ) == 0
            ):
                initial_group = new_groups.pop()
                initial_group.create_new = False
                initial_group.patient_id = self._current_patient_id
                initial_group.reason = "prima identità del workspace selezionato"
                matched_groups.append(initial_group)

            importable_count = sum(
                len(group.documents)
                for group in matched_groups + new_groups
            )

            lines = ["Attribuzione proposta:", ""]

            for group in matched_groups:
                partial_note = (
                    f"; {group.partial_matches} con identificatori parziali, "
                    "da verificare"
                    if group.partial_matches else ""
                )
                lines.append(
                    f"• {group.patient_id}: {len(group.documents)} documenti "
                    f"({group.reason}{partial_note})"
                )

            for group in new_groups:
                label = (
                    group.evidence.masked_label()
                    if group.evidence else "identità"
                )
                partial_note = (
                    f" ({group.partial_matches} con identificatori parziali, "
                    "da verificare)"
                    if group.partial_matches else ""
                )
                lines.append(
                    f"• Nuovo workspace ({label}): "
                    f"{len(group.documents)} documenti{partial_note}"
                )

            if duplicates:
                lines.append(f"• Duplicati globali esclusi: {len(duplicates)}")
            if blocked_groups:
                conflict_count = sum(
                    len(group.documents)
                    for group in blocked_groups if group.conflict
                )
                unresolved_count = (
                    sum(len(group.documents) for group in blocked_groups)
                    - conflict_count
                )
                if unresolved_count:
                    lines.append(
                        f"• Identità insufficiente, non importati: "
                        f"{unresolved_count} documenti"
                    )
                if conflict_count:
                    lines.append(
                        f"• Conflitto d'identità, non importati: "
                        f"{conflict_count} documenti"
                    )
            if errors:
                lines.append(
                    f"• File non leggibili esclusi: {len(errors)}"
                )
            lines.extend([
                "",
                f"Confermare l'importazione? "
                f"({importable_count} documenti importabili)",
            ])

            if importable_count == 0:
                QMessageBox.warning(
                    self, "Nessun documento attribuibile", "\n".join(lines)
                )
                return
            reply = QMessageBox.question(
                self, "Assegnazione pazienti", "\n".join(lines),
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                return

            routing = defaultdict(list)
            for group in matched_groups:
                routing[group.patient_id].extend(group.documents)
            for group in new_groups:
                new_id = patient_repo.get_next_id()
                evidence = group.evidence
                birth_year = None
                if evidence and evidence.birth_date:
                    birth_year = int(evidence.birth_date.normalized[:4])
                patient = Patient(
                    id=new_id,
                    pseudonym=new_id[1:],
                    sex=evidence.sex.normalized if evidence and evidence.sex else None,
                    birth_year=birth_year,
                    created_at=datetime.now().isoformat(),
                    updated_at=datetime.now().isoformat(),
                )
                patient_repo.insert(patient)
                created_patient_ids.append(new_id)
                identity_repo.upsert(new_id, evidence, status="auto_created")
                if audit_repo:
                    audit_repo.log(
                        new_id, "auto_create", "patient", new_id,
                        {
                            "document_count": len(group.documents),
                            "identity_confidence": evidence.confidence,
                        },
                    )
                routing[new_id].extend(group.documents)

            imported_patients = []
            for patient_id, staged_documents in routing.items():
                paths = [document.staged_path for document in staged_documents]
                metadata = {
                    document.staged_path: {
                        "original_name": document.original_name,
                        "hash": document.file_hash,
                        "check": document.check,
                        "identity_evidence": document.evidence,
                    }
                    for document in staged_documents
                }
                self._current_patient_id = patient_id
                dialog = ImportDialog(
                    paths, self._services, patient_id, self,
                    workspace_tabs=self, file_metadata=metadata,
                )
                if dialog.exec():
                    imported_patients.append(patient_id)
                    self._documents_tab.load_patient(patient_id)
                    self._dashboard_tab.load_patient(patient_id)
                    if dialog.should_auto_process():
                        doc_ids = dialog.get_imported_doc_ids()
                        if doc_ids:
                            self._documents_tab.extract_clinical_text(doc_ids)

            # Remove automatically-created empty patients if their import was
            # cancelled in the second confirmation dialog.
            for patient_id in created_patient_ids:
                if patient_id not in imported_patients:
                    patient_repo.delete(patient_id)
                    workspace_path = WORKSPACES_DIR / patient_id
                    if workspace_path.exists():
                        shutil.rmtree(workspace_path, ignore_errors=True)

            main_window = self.window()
            if hasattr(main_window, "patient_panel"):
                main_window.patient_panel.refresh()
                if imported_patients:
                    main_window.patient_panel.patient_selected.emit(imported_patients[-1])
        finally:
            batch.cleanup()


    def reprocess_document(self, doc_id: str):
        """Trigger reprocessing of a document."""
        self._documents_tab.reprocess_document(doc_id)

    def _on_document_selected(self, doc_id: str, doc_data: dict):
        self._current_document_id = doc_id
        # Include patient_id
        doc_data["_patient_id"] = self._current_patient_id
        self.context_requested.emit("document", doc_data)

    def _on_import_requested(self, files: list[str]):
        self.show_import_dialog(files)

    def _on_event_selected(self, event_data: dict):
        self.context_requested.emit("event", event_data)

    def _on_lab_selected(self, lab_data: dict):
        self.context_requested.emit("lab", lab_data)

    def _on_processing_complete(self, patient_id: str):
        """Refresh all tabs after document processing completes."""
        self._timeline_tab.load_patient(patient_id)
        self._laboratory_tab.load_patient(patient_id)
        self._clinical_state_tab.load_patient(patient_id)
        self._validation_tab.load_patient(patient_id)
        self._dashboard_tab.load_patient(patient_id)

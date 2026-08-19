"""Central QTabWidget workspace for EMR Analyzer."""

from __future__ import annotations

from PyQt5.QtWidgets import (
    QTabWidget, QWidget, QVBoxLayout, QLabel, QMessageBox, QApplication,
)
from PyQt5.QtCore import Qt, pyqtSignal

from datetime import datetime

from .documents_tab import DocumentsTab
from .laboratory_tab import LaboratoryTab
from .clinical_history_tab import ClinicalHistoryTab
from .validation_tab import ValidationTab
from .import_dialog import ImportDialog
from .batch_import_dialog import BatchImportDialog
from .progress_dialog import ProgressDialog
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
        self._validation_tab.document_reattributed.connect(
            self._on_document_reattributed
        )

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

    def shutdown(self) -> None:
        """Stop background workers of the tabs (application quit path)."""
        try:
            self._clinical_history_tab.shutdown()
        except Exception:
            pass

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
        QApplication.processEvents()

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
        QApplication.processEvents()

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

"""Application orchestrator for EMR Analyzer."""

import sys

from PyQt5.QtCore import Qt

from .config import (
    APP_NAME, APP_VERSION, active_workspace, CACHE_DIR, LOG_DIR,
    OFFLINE_MODE,
    DOCUMENT_LLM_MODEL_NAME,
)
from .security.offline import OfflinePolicy
from .database.engine import DatabaseEngine
from .database.migrations import init_database
from .database.patient_repo import PatientRepository
from .database.patient_identity_repo import PatientIdentityRepository
from .database.evidence_repo import EvidenceRepository
from .database.document_repo import DocumentRepository
from .database.lab_repo import LabRepository
from .database.clinical_state_repo import ClinicalStateRepository
from .database.audit_repo import AuditRepository
from .pipeline.converter import DoclingConverter
from .pipeline.pdf_extractor import PdfPlumberExtractor
from .pipeline.classifier import DocumentClassifier
from .pipeline.cleaner import TextCleaner
from .pipeline.header_metadata import HeaderMetadataExtractor
from .pipeline.patient_identity import PatientIdentityExtractor
from .pipeline.import_staging import ImportStagingService
from .pipeline.patient_routing import PatientRoutingService
from .extraction.lab_parser import LabParser
from .extraction.normalizer import LabNormalizer
from .extraction.llm_client import LlmClient
from .extraction.clinical_text_isolator import ClinicalTextIsolator
from .clinical.clinical_history_builder import ClinicalHistoryBuilder
from .clinical.document_deletion import DocumentDeletionService
from .clinical.patient_deletion import PatientWorkspaceDeletionService
from .clinical.document_reattribution import DocumentReattributionService
from .gui.main_window import MainWindow
from .database.timeline_repo import TimelineRepository
from .database.chat_repo import ChatRepository
from .settings import load_llm_configs


class EMRAnalyzerApp:
    """Application orchestrator — wires together all services."""

    def __init__(self):
        self._services = {}
        if OFFLINE_MODE:
            OfflinePolicy(strict_network=True).activate()
        self._llm_configs = load_llm_configs()

        # QApplication must exist before any dialog
        from PyQt5.QtWidgets import QApplication as QA
        self._qapp = QA(sys.argv)
        self._qapp.setApplicationName(APP_NAME)
        self._qapp.setApplicationVersion(APP_VERSION)
        self._qapp.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
        self._qapp.aboutToQuit.connect(self._shutdown_backend)

        # Project selection — sets active_workspace.path
        if not self._select_project():
            sys.exit(0)

        self._init_dirs()
        self._init_services()
        self._init_gui()

    def _select_project(self) -> bool:
        """Show the project dialog and update active_workspace.

        Returns False if the user cancelled (application should exit).
        """
        from .gui.project_dialog import ProjectDialog
        dialog = ProjectDialog()
        if dialog.exec_() != ProjectDialog.Accepted:
            return False
        selected = dialog.selected_path
        if selected is None:
            return False
        active_workspace.set_path(selected)
        print(f"[EMR Analyzer] Progetto: {selected}")
        return True

    def _init_dirs(self):
        """Create required directory structure."""
        for d in [active_workspace.path, CACHE_DIR, LOG_DIR]:
            d.mkdir(parents=True, exist_ok=True)

    def _init_services(self):
        """Initialize all backend services."""
        print(f"[EMR Analyzer] Initializing services...")

        # ---- Database ----
        # Each patient gets their own .db in their workspace,
        # but we also have a global registry for patient list.
        # For now, use a global DB in the base workspace dir.
        global_db_path = active_workspace.path / "emr_registry.db"
        db = DatabaseEngine(global_db_path)
        init_database(db)
        self._services["db"] = db
        print(f"  ✓ Database: {global_db_path}")

        # ---- Repositories ----
        patient_repo = PatientRepository(db)
        identity_repo = PatientIdentityRepository(db)
        doc_repo = DocumentRepository(db)
        lab_repo = LabRepository(db)
        cs_repo = ClinicalStateRepository(db)
        audit_repo = AuditRepository(db)
        evidence_repo = EvidenceRepository(db)
        timeline_repo = TimelineRepository(db)
        chat_repo = ChatRepository(db)
        self._services.update({
            "patient_repo": patient_repo,
            "identity_repo": identity_repo,
            "document_repo": doc_repo,
            "lab_repo": lab_repo,
            "cs_repo": cs_repo,
            "audit_repo": audit_repo,
            "evidence_repo": evidence_repo,
            "timeline_repo": timeline_repo,
            "chat_repo": chat_repo,
        })
        print(f"  ✓ Repositories initialized")

        # ---- Deterministic PDF parsing ----
        parser_ok = False
        try:
            converter = PdfPlumberExtractor()
            if converter.is_available:
                self._services["converter"] = converter
                parser_ok = True
                print(
                    "  ✓ PDF parser: pdfplumber → PyMuPDF → OCR locale ready"
                )
            else:
                print(f"  ⚠ pdfplumber: not available ({converter._init_error})")
        except Exception as e:
            print(f"  ⚠ pdfplumber: {e}")
            self._services["converter"] = None

        # Temporary safety fallback while the two parsers are benchmarked on
        # heterogeneous institutional templates. It is never the primary path.
        try:
            fallback = DoclingConverter()
            self._services["parser_fallback"] = (
                fallback if fallback.is_available else None
            )
        except Exception:
            self._services["parser_fallback"] = None

        # ---- Function-specific local llama.cpp models ----
        ollama_ok = False
        document_config = self._llm_configs["document"]
        state_config = self._llm_configs["clinical_state"]
        document_model_name = document_config.model
        state_model_name = state_config.model
        try:
            probe_model = (
                document_model_name or state_model_name
                or DOCUMENT_LLM_MODEL_NAME
            )
            ollama_ok = LlmClient(model=probe_model).server_available
            document_llm = (
                LlmClient(config=document_config)
                if document_model_name else None
            )
            clinical_state_llm = (
                LlmClient(config=state_config) if state_model_name else None
            )
            active_document_llm = (
                document_llm
                if document_llm is not None and document_llm.is_available
                else None
            )
            active_clinical_state_llm = (
                clinical_state_llm
                if clinical_state_llm is not None and clinical_state_llm.is_available
                else None
            )
            self._services.update({
                "document_llm_client": active_document_llm,
                "clinical_state_llm_client": active_clinical_state_llm,
                "ollama_available": ollama_ok,
                "llm_configs": self._llm_configs,
            })
            if ollama_ok:
                print(
                    "  ✓ LLM locale (llama.cpp): "
                    f"document={active_document_llm.model if active_document_llm else 'off'}, "
                    "clinical-state="
                    f"{active_clinical_state_llm.model if active_clinical_state_llm else 'off'}"
                )
                # Eager background start: the model load (seconds on this
                # hardware) hides behind the normal startup flow.
                self._eager_start_llm_servers([
                    active_document_llm, active_clinical_state_llm,
                ])
            else:
                print(
                    "  ⚠ Motore locale non disponibile — esegui "
                    "tools/setup_llama_backend.py"
                )
        except Exception as e:
            print(f"  ⚠ Motore locale (llama.cpp): {e}")
            self._services.update({
                "document_llm_client": None,
                "clinical_state_llm_client": None,
                "ollama_available": False,
                "llm_configs": self._llm_configs,
            })

        # ---- Pipeline components ----
        self._services["classifier"] = DocumentClassifier()
        self._services["cleaner"] = TextCleaner()
        self._services["header_metadata_extractor"] = HeaderMetadataExtractor()
        identity_extractor = PatientIdentityExtractor()
        self._services["identity_extractor"] = identity_extractor
        self._services["import_staging"] = ImportStagingService(
            identity_extractor, doc_repo
        )
        self._services["patient_router"] = PatientRoutingService(
            identity_repo,
            identity_extractor,
            patient_repo,
            doc_repo,
            audit_repo,
        )

        # ---- Extraction components ----
        normalizer = LabNormalizer()
        lab_parser = LabParser(normalizer)
        self._services.update({
            "normalizer": normalizer,
            "lab_parser": lab_parser,
            "clinical_text_isolator": ClinicalTextIsolator(
                self._services.get("document_llm_client")
            ),
        })

        # ---- Clinical components ----
        clinical_history_builder = ClinicalHistoryBuilder(
            timeline_repo=timeline_repo,
            document_repo=doc_repo,
            cs_repo=cs_repo,
            clinical_state_llm_client=self._services.get("clinical_state_llm_client"),
            audit_repo=audit_repo,
        )
        self._services.update({
            "clinical_history_builder": clinical_history_builder,
        })
        self._services["document_deletion"] = DocumentDeletionService(
            db, doc_repo, audit_repo=audit_repo
        )
        self._services["patient_workspace_deletion"] = (
            PatientWorkspaceDeletionService(db, patient_repo)
        )
        self._services["document_reattribution"] = (
            DocumentReattributionService(
                db, doc_repo, patient_repo, audit_repo
            )
        )

        print(f"  ✓ Services initialized")
        return parser_ok, ollama_ok

    @staticmethod
    def _eager_start_llm_servers(clients: list) -> None:
        """Spawn the llama-server processes in the background.

        The first clinical request would otherwise pay the model load time;
        a daemon thread hides it behind the startup flow.  Failures are
        non-blocking: generation calls retry lazily on demand.
        """
        import threading

        from .llm_backend import get_backend

        def _start() -> None:
            backend = get_backend()
            for client in clients:
                if client is None:
                    continue
                try:
                    backend.ensure(client)
                    print(f"  ✓ server pronto per {client.model}")
                except Exception as exc:
                    print(f"  ⚠ avvio server {client.model} rimandato: {exc}")

        threading.Thread(target=_start, daemon=True).start()

    @staticmethod
    def _shutdown_backend() -> None:
        """Terminate the app-owned llama-server processes on quit."""
        try:
            from .llm_backend import get_backend
            get_backend().shutdown()
        except Exception:
            pass

    def _init_gui(self):
        """Initialize and show the main window."""
        ollama_ok = self._services.get("ollama_available", False)

        self._window = MainWindow()
        self._window.set_services(self._services)
        self._window.update_model_status(ollama_ok)
        self._window.show()

    def run(self):
        """Run the application event loop."""
        return self._qapp.exec_()

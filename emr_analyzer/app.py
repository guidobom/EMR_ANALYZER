"""Application orchestrator for EMR Analyzer."""

import sys

from PyQt5.QtCore import Qt

from .config import (
    APP_NAME, APP_VERSION, active_workspace, CACHE_DIR, LOG_DIR,
    OFFLINE_MODE,
)
from .security.offline import OfflinePolicy
from .database.engine import DatabaseEngine
from .database.migrations import init_database, reset_stale_processing
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
from .clinical.registry_builder import ClinicalRegistryBuilder
from .clinical.document_deletion import DocumentDeletionService
from .clinical.patient_deletion import PatientWorkspaceDeletionService
from .clinical.document_reattribution import DocumentReattributionService
from .gui.main_window import MainWindow
from .database.timeline_repo import TimelineRepository
from .database.chat_repo import ChatRepository
from .database.registry_repo import ClinicalRegistryRepository
from .database.processing_repo import ProcessingRepository
from .database.overlay_repo import DocumentTextOverlayRepository
from .database.review_repo import ReviewDecisionRepository
from .database.gold_set_repo import GoldSetRepository
from .database.pipeline_repo import ClinicalPipelineRepository
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
        # Self-healing: a document left in 'processing' status by an
        # interrupted run (crash, forced quit) would be skipped by the
        # extraction queue forever.  At startup no queue is running, so
        # every 'processing' flag is stale by definition.
        reset_stale_processing(db)
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
        registry_repo = ClinicalRegistryRepository(db)
        processing_repo = ProcessingRepository(db)
        overlay_repo = DocumentTextOverlayRepository(db)
        review_repo = ReviewDecisionRepository(db)
        pipeline_repo = ClinicalPipelineRepository(db)
        gold_set_repo = GoldSetRepository(
            db, registry_repo=registry_repo, audit_repo=audit_repo
        )
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
            "registry_repo": registry_repo,
            "processing_repo": processing_repo,
            "overlay_repo": overlay_repo,
            "review_repo": review_repo,
            "pipeline_repo": pipeline_repo,
            "gold_set_repo": gold_set_repo,
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

        # ---- Function-specific local llama.cpp / vLLM models ----
        ollama_ok = False
        try:
            clients = {}
            for role, config in self._llm_configs.items():
                client = LlmClient(config=config) if config.model else None
                clients[role] = (
                    client
                    if client is not None and client.is_available else None
                )
            ollama_ok = any(
                client is not None and client.server_available
                for client in clients.values()
            )
            self._services.update({
                "document_llm_client": clients["document"],
                "atomic_evidence_llm_client": clients["atomic_evidence"],
                "clinical_events_llm_client": clients["clinical_events"],
                "clinical_state_llm_client": clients["clinical_state"],
                "ollama_available": ollama_ok,
                "llm_configs": self._llm_configs,
            })
            if ollama_ok:
                print(
                    "  ✓ LLM locale: "
                    + ", ".join(
                        f"{role}="
                        + (
                            f"{client.model} [{client.backend_type}]"
                            if client else "off"
                        )
                        for role, client in clients.items()
                    )
                )
                # Only the document model is warmed at startup.  Registry and
                # analysis models stay lazy so selecting different GGUFs does
                # not make all of them resident at the same time.
                self._eager_start_llm_servers([clients["document"]])
            else:
                print(
                    "  ⚠ Motore locale non disponibile — esegui "
                    "tools/setup_llama_backend.py oppure "
                    "tools/setup_vllm_backend.py"
                )
        except Exception as e:
            print(f"  ⚠ Motore LLM locale: {e}")
            self._services.update({
                "document_llm_client": None,
                "atomic_evidence_llm_client": None,
                "clinical_events_llm_client": None,
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
        registry_builder = ClinicalRegistryBuilder(
            registry_repo=registry_repo,
            evidence_repo=evidence_repo,
            processing_repo=processing_repo,
            timeline_repo=timeline_repo,
            document_repo=doc_repo,
            lab_repo=lab_repo,
            overlay_repo=overlay_repo,
            atomic_llm_client=self._services.get(
                "atomic_evidence_llm_client"
            ),
            event_llm_client=self._services.get(
                "clinical_events_llm_client"
            ),
            audit_repo=audit_repo,
            pipeline_repo=pipeline_repo,
            db=db,
        )
        clinical_history_builder = ClinicalHistoryBuilder(
            timeline_repo=timeline_repo,
            document_repo=doc_repo,
            cs_repo=cs_repo,
            clinical_state_llm_client=self._services.get("clinical_state_llm_client"),
            audit_repo=audit_repo,
            registry_builder=registry_builder,
        )
        self._services.update({
            "clinical_history_builder": clinical_history_builder,
            "registry_builder": registry_builder,
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
        """Spawn the selected local LLM servers in the background.

        The first clinical request would otherwise pay the model load time;
        a daemon thread hides it behind the startup flow.  Failures are
        non-blocking: generation calls retry lazily on demand.
        """
        import threading

        def _start() -> None:
            for client in clients:
                if client is None:
                    continue
                try:
                    client.backend.ensure(client)
                    print(
                        f"  ✓ server {client.backend_type} pronto per "
                        f"{client.model}"
                    )
                except Exception as exc:
                    print(f"  ⚠ avvio server {client.model} rimandato: {exc}")

        threading.Thread(target=_start, daemon=True).start()

    @staticmethod
    def _shutdown_backend() -> None:
        """Terminate every app-owned local LLM process on quit."""
        try:
            from .llm_backend import shutdown_all_backends
            shutdown_all_backends()
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

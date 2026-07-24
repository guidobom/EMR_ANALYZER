"""QThread workers for background processing.

Each worker runs in a separate thread and communicates
with the main GUI thread via Qt signals.
"""

import traceback
from pathlib import Path
from datetime import datetime
from typing import Optional

from PyQt6.QtCore import QThread, pyqtSignal

from ..pipeline.converter import DoclingConverter
from ..pipeline.classifier import DocumentClassifier
from ..pipeline.segmenter import ClinicalSegmenter
from ..pipeline.cleaner import TextCleaner
from ..extraction.lab_parser import LabParser
from ..extraction.qwen_client import QwenClient
from ..clinical.event_extractor import EventExtractor
from ..clinical.clinical_state import ClinicalStateManager
from ..clinical.event_store import EventStore
from ..utils.file_utils import verify_pdf, get_file_info, compute_file_hash
from ..models.document import DocumentRecord, DocumentType
from ..models.lab_result import LabValue
from ..models.clinical_event import ClinicalEvent
from ..models.clinical_state import ClinicalStateDelta


class DoclingWorker(QThread):
    """Docling standard conversion in background."""
    progress = pyqtSignal(int, str)           # percentage, message
    page_progress = pyqtSignal(int, int)      # current_page, total_pages
    finished = pyqtSignal(object)             # ConversionResult
    error = pyqtSignal(str)

    def __init__(self, converter: DoclingConverter, file_path: str,
                 parent=None):
        super().__init__(parent)
        self.converter = converter
        self.file_path = file_path

    def run(self):
        try:
            self.progress.emit(5, "Caricamento documento...")
            result = self.converter.convert(self.file_path)
            self.progress.emit(100, "Completato")
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(f"Errore conversione: {str(e)}")


class ImportCheckWorker(QThread):
    """Validate files during import (hash, PDF check, etc.)."""
    file_checked = pyqtSignal(str, dict)       # filename, check_results
    all_checked = pyqtSignal(list)
    error = pyqtSignal(str)

    def __init__(self, file_paths: list[str], parent=None):
        super().__init__(parent)
        self.file_paths = file_paths

    def run(self):
        try:
            results = []
            for fp in self.file_paths:
                info = get_file_info(fp)
                check = verify_pdf(fp) if info["extension"] == ".pdf" else {}
                result = {
                    "path": fp,
                    "filename": info["filename"],
                    "size_mb": info["size_mb"],
                    "hash": compute_file_hash(fp),
                    "page_count": check.get("page_count", 1),
                    "has_text": check.get("has_text", False),
                    "is_protected": check.get("is_protected", False),
                    "readable": check.get("readable", True),
                    "error": check.get("error"),
                }
                results.append(result)
                self.file_checked.emit(info["filename"], result)
            self.all_checked.emit(results)
        except Exception as e:
            self.error.emit(f"Errore importazione: {str(e)}")


class ExtractionWorker(QThread):
    """
    Full extraction pipeline: classifies, segments, extracts lab values
    and clinical events from a docling-processed document.
    """
    progress = pyqtSignal(int, str)
    lab_values_ready = pyqtSignal(list)        # list[LabValue]
    events_ready = pyqtSignal(list)            # list[ClinicalEvent]
    delta_ready = pyqtSignal(object)           # ClinicalStateDelta
    finished = pyqtSignal(dict)                # Full extraction result
    error = pyqtSignal(str)

    def __init__(self, patient_id: str, doc_id: str,
                 markdown_text: str, docling_result,
                 classifier: DocumentClassifier,
                 segmenter: ClinicalSegmenter,
                 cleaner: TextCleaner,
                 lab_parser: LabParser,
                 qwen_client: QwenClient,
                 event_store: EventStore,
                 cs_manager: ClinicalStateManager,
                 parent=None):
        super().__init__(parent)
        self.patient_id = patient_id
        self.doc_id = doc_id
        self.markdown_text = markdown_text
        self.docling_result = docling_result
        self.classifier = classifier
        self.segmenter = segmenter
        self.cleaner = cleaner
        self.lab_parser = lab_parser
        self.qwen_client = qwen_client
        self.event_store = event_store
        self.cs_manager = cs_manager

    def run(self):
        try:
            # Step 1: Clean text
            self.progress.emit(10, "Pulizia testo...")
            cleaned = self.cleaner.clean(self.markdown_text)

            # Step 2: Classify document
            self.progress.emit(20, "Classificazione documento...")
            doc_type = self.classifier.classify(cleaned)

            # Step 3: Segment into clinical sections
            self.progress.emit(35, "Segmentazione clinica...")
            sections = self.segmenter.segment(cleaned)

            # Step 4: Extract lab values (deterministic)
            self.progress.emit(50, "Estrazione valori laboratorio...")
            tables = []
            if hasattr(self.docling_result, 'document'):
                doc = self.docling_result.document
                for table in doc.tables:
                    try:
                        tables.append(table.export_to_dataframe())
                    except Exception:
                        pass
            lab_values = self.lab_parser.parse(cleaned, tables)
            self.lab_values_ready.emit(lab_values)

            # Step 5: Extract clinical events via Qwen3
            self.progress.emit(65, "Estrazione eventi clinici (Qwen3-14B)...")
            events = []
            try:
                events = self.qwen_client.extract_clinical_events(
                    cleaned, self.patient_id, self.doc_id
                )
            except Exception as e:
                self.progress.emit(70, f"Qwen non disponibile: {e}")
            self.events_ready.emit(events)

            # Step 6: Propose Clinical State delta
            self.progress.emit(85, "Aggiornamento Clinical State...")
            delta = ClinicalStateDelta()
            if events:
                try:
                    delta = self.cs_manager.propose_delta(
                        self.patient_id, events
                    )
                except Exception as e:
                    self.progress.emit(90, f"Delta non calcolabile: {e}")
            self.delta_ready.emit(delta)

            # Step 7: Assemble result
            self.progress.emit(100, "Completato")
            self.finished.emit({
                "patient_id": self.patient_id,
                "document_id": self.doc_id,
                "document_type": doc_type,
                "sections": sections,
                "lab_values": lab_values,
                "events": events,
                "delta": delta,
                "cleaned_text": cleaned,
            })
        except Exception as e:
            traceback.print_exc()
            self.error.emit(f"Errore estrazione: {str(e)}")


class ClinicalStateBuildWorker(QThread):
    """Build a ClinicalState from all normalized clinical texts."""
    progress = pyqtSignal(int, str)           # percent, message
    finished = pyqtSignal(object)             # ClinicalState
    error = pyqtSignal(str)

    def __init__(self, cs_manager, patient_id: str, parent=None):
        super().__init__(parent)
        self.cs_manager = cs_manager
        self.patient_id = patient_id

    def run(self):
        try:
            state = self.cs_manager.build_from_normalized_texts(
                self.patient_id,
                progress_callback=lambda pct, msg: self.progress.emit(pct, msg),
            )
            self.finished.emit(state)
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.error.emit(str(e))


class BatchClinicalStateWorker(QThread):
    """Build ClinicalState for multiple patients concurrently.

    Manages up to *max_concurrent* build workers at a time, cycling through
    the patient list until all have been processed.
    """

    progress = pyqtSignal(int, str)           # percent, message
    patient_started = pyqtSignal(str)         # patient_id
    patient_completed = pyqtSignal(str, int, int)  # patient_id, diag_count, treat_count
    patient_failed = pyqtSignal(str, str)     # patient_id, error
    all_completed = pyqtSignal(int, int)      # success_count, fail_count

    def __init__(self, cs_manager, patient_ids: list[str],
                 max_concurrent: int = 2, parent=None):
        super().__init__(parent)
        self.cs_manager = cs_manager
        self.patient_ids = list(patient_ids)
        self.max_concurrent = max_concurrent

    def run(self):
        import threading
        total = len(self.patient_ids)
        completed = 0
        success = 0
        failed = 0
        lock = threading.Lock()
        errors = []

        def process_one(pid: str):
            nonlocal completed, success, failed
            self.patient_started.emit(pid)
            try:
                state = self.cs_manager.build_from_normalized_texts(pid)
                with lock:
                    completed += 1
                    success += 1
                    pct = int(completed * 100 / total)
                    self.progress.emit(
                        pct,
                        f"[{completed}/{total}] {pid} — "
                        f"{len(state.active_diagnoses)} diagnosi, "
                        f"{len(state.active_treatments)} terapie",
                    )
                    self.patient_completed.emit(
                        pid,
                        len(state.active_diagnoses),
                        len(state.active_treatments),
                    )
            except Exception as e:
                with lock:
                    completed += 1
                    failed += 1
                    errors.append(f"{pid}: {e}")
                    pct = int(completed * 100 / total)
                    self.progress.emit(pct, f"[{completed}/{total}] {pid} — ❌ {e}")
                    self.patient_failed.emit(pid, str(e))

        threads = []
        for pid in self.patient_ids:
            t = threading.Thread(target=process_one, args=(pid,), daemon=True)
            threads.append(t)

        # Start up to max_concurrent threads, wait for one batch to drain
        # before starting the next.
        idx = 0
        running = []
        while idx < total or running:
            while len(running) < self.max_concurrent and idx < total:
                t = threads[idx]
                t.start()
                running.append(t)
                idx += 1
            for t in running[:]:
                t.join(timeout=0.3)
                if not t.is_alive():
                    running.remove(t)

        self.all_completed.emit(success, failed)


class ClinicalQueryWorker(QThread):
    """Run a clinical query against the Clinical State."""
    finished = pyqtSignal(str)                # Answer text
    error = pyqtSignal(str)

    def __init__(self, qwen_client: QwenClient, clinical_state: dict,
                 events: list[dict], question: str, parent=None):
        super().__init__(parent)
        self.qwen_client = qwen_client
        self.clinical_state = clinical_state
        self.events = events
        self.question = question

    def run(self):
        try:
            answer = self.qwen_client.query_clinical_state(
                self.clinical_state, self.events, self.question
            )
            self.finished.emit(answer)
        except Exception as e:
            self.error.emit(f"Errore query: {str(e)}")


class BatchQueryWorker(QThread):
    """Run the same question against the ClinicalState of all patients
    that have one, collecting answers concurrently."""

    progress = pyqtSignal(int, str)           # percent, message
    patient_result = pyqtSignal(str, str)     # patient_id, answer_markdown
    patient_skipped = pyqtSignal(str, str)    # patient_id, reason
    patient_error = pyqtSignal(str, str)      # patient_id, error
    all_completed = pyqtSignal(int, int, int, int)  # total, success, skipped, failed

    def __init__(self, qwen_client: QwenClient,
                 cs_repo, event_repo,
                 patient_ids: list[str], question: str,
                 max_concurrent: int = 2, parent=None):
        super().__init__(parent)
        self.qwen_client = qwen_client
        self.cs_repo = cs_repo
        self.event_repo = event_repo
        self.patient_ids = list(patient_ids)
        self.question = question
        self.max_concurrent = max_concurrent

    def run(self):
        import threading
        total = len(self.patient_ids)
        completed = 0
        success = 0
        skipped = 0
        failed = 0
        lock = threading.Lock()

        def query_one(pid: str):
            nonlocal completed, success, skipped, failed
            try:
                state = self.cs_repo.load(pid)
                if state is None:
                    with lock:
                        completed += 1
                        skipped += 1
                        self.progress.emit(
                            int(completed * 100 / total),
                            f"[{completed}/{total}] {pid} — ⏭️ nessun CS",
                        )
                        self.patient_skipped.emit(pid, "Nessun Clinical State")
                    return

                events = []
                if self.event_repo:
                    events = [e.to_dict()
                              for e in self.event_repo.get_by_patient(pid)]

                answer = self.qwen_client.query_clinical_state(
                    state.to_dict(), events, self.question
                )
                with lock:
                    completed += 1
                    success += 1
                    self.progress.emit(
                        int(completed * 100 / total),
                        f"[{completed}/{total}] {pid} — ✅ risposta ricevuta",
                    )
                    self.patient_result.emit(pid, answer)
            except Exception as e:
                with lock:
                    completed += 1
                    failed += 1
                    self.progress.emit(
                        int(completed * 100 / total),
                        f"[{completed}/{total}] {pid} — ❌ {e}",
                    )
                    self.patient_error.emit(pid, str(e))

        # --- concurrent execution (same pattern as BatchClinicalStateWorker) ---
        threads = [threading.Thread(target=query_one, args=(pid,), daemon=True)
                   for pid in self.patient_ids]
        idx = 0
        running = []
        while idx < total or running:
            while len(running) < self.max_concurrent and idx < total:
                threads[idx].start()
                running.append(threads[idx])
                idx += 1
            for t in running[:]:
                t.join(timeout=0.3)
                if not t.is_alive():
                    running.remove(t)

        self.all_completed.emit(total, success, skipped, failed)

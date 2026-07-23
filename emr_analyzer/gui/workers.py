"""QThread workers for background processing.

Each worker runs in a separate thread and communicates
with the main GUI thread via Qt signals.
"""

import traceback
from pathlib import Path
from datetime import datetime
from typing import Optional

from PyQt5.QtCore import QThread, pyqtSignal

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

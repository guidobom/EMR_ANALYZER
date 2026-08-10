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
from ..extraction.llm_client import LlmClient
from ..utils.file_utils import verify_pdf, get_file_info, compute_file_hash


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


class ClinicalQueryWorker(QThread):
    """Run a clinical query against the Clinical State."""
    finished = pyqtSignal(str)                # Answer text
    error = pyqtSignal(str)

    def __init__(self, llm_client: LlmClient, clinical_state: dict,
                 events: list[dict], question: str, parent=None):
        super().__init__(parent)
        self.llm_client = llm_client
        self.clinical_state = clinical_state
        self.events = events
        self.question = question

    def run(self):
        try:
            answer = self.llm_client.query_clinical_state(
                self.clinical_state, self.events, self.question
            )
            self.finished.emit(answer)
        except Exception as e:
            self.error.emit(f"Errore query: {str(e)}")


class ClinicalHistoryWorker(QThread):
    """Background worker for building the clinical history timeline."""
    progress = pyqtSignal(int, str)         # percentage, message
    finished = pyqtSignal(dict)             # result summary
    error = pyqtSignal(str)

    def __init__(self, builder, patient_id: str,
                 generate_narrative: bool = False,
                 num_workers: int = 1, parent=None):
        super().__init__(parent)
        self.builder = builder
        self.patient_id = patient_id
        self.generate_narrative = generate_narrative
        self.num_workers = num_workers

    def run(self):
        try:
            # Use incremental mode when entries already exist
            existing_count = self.builder._timeline_repo.count_by_patient(
                self.patient_id
            )
            if existing_count > 0:
                result = self.builder.build_incremental(
                    self.patient_id,
                    progress_callback=lambda pct, msg: self.progress.emit(pct, msg),
                    generate_narrative=self.generate_narrative,
                )
            elif self.num_workers > 1:
                result = self.builder.build_from_documents_parallel(
                    self.patient_id,
                    num_workers=self.num_workers,
                    progress_callback=lambda pct, msg: self.progress.emit(pct, msg),
                    generate_narrative=self.generate_narrative,
                )
            else:
                result = self.builder.build_from_documents(
                    self.patient_id,
                    progress_callback=lambda pct, msg: self.progress.emit(pct, msg),
                    generate_narrative=self.generate_narrative,
                )
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))


class DedupWorker(QThread):
    """Deduplicate existing timeline entries without re-extracting."""
    finished = pyqtSignal(int)              # number of removed entries
    error = pyqtSignal(str)

    def __init__(self, builder, patient_id: str, parent=None):
        super().__init__(parent)
        self.builder = builder
        self.patient_id = patient_id

    def run(self):
        try:
            removed = self.builder.deduplicate_existing(self.patient_id)
            self.finished.emit(removed)
        except Exception as e:
            self.error.emit(str(e))


class NarrativeWorker(QThread):
    """Generate the narrative clinical profile from the existing timeline."""
    finished = pyqtSignal(str)             # narrative text
    error = pyqtSignal(str)

    def __init__(self, builder, patient_id: str, parent=None):
        super().__init__(parent)
        self.builder = builder
        self.patient_id = patient_id

    def run(self):
        try:
            narrative = self.builder.generate_narrative(self.patient_id)
            self.finished.emit(narrative)
        except Exception as e:
            self.error.emit(str(e))


class ClinicalHistoryQueryWorker(QThread):
    """Run a clinical query against the timeline history."""
    finished = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, llm_client, entries: list[dict],
                 clinical_profile: str, question: str, parent=None):
        super().__init__(parent)
        self.llm_client = llm_client
        self.entries = entries
        self.clinical_profile = clinical_profile
        self.question = question

    def run(self):
        try:
            system_prompt = (
                "Sei un assistente clinico esperto. Rispondi alla domanda "
                "basandoti ESCLUSIVAMENTE sui dati clinici forniti. "
                "Se un dato non e' disponibile, dichiaralo esplicitamente. "
                "Cita le date quando disponibili. Non inventare informazioni."
            )

            # Merge profile + timeline as compact context
            entries_text = "\n".join(
                f"[{e.get('date_observed', '?')}] [{e.get('category', '?')}] "
                f"{e.get('description', '')}"
                for e in self.entries[-100:]
            )

            user_prompt = (
                f"PROFILO CLINICO:\n{self.clinical_profile}\n\n"
                f"REGISTRO CRONOLOGICO:\n{entries_text}\n\n"
                f"DOMANDA: {self.question}\n\n"
                f"Rispondi in modo chiaro e conciso, citando date e fonti "
                f"quando disponibili."
            )

            answer = self.llm_client.generate_text(user_prompt, system_prompt)
            self.finished.emit(answer)
        except Exception as e:
            self.error.emit(f"Errore query: {str(e)}")

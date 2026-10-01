"""QThread workers for background extraction.

Each worker runs in a separate thread and communicates
with the main GUI thread via Qt signals.
"""

import threading

from PyQt5.QtCore import QThread, pyqtSignal

from ..clinical.extraction_pipeline import ExtractionCancelled


class PatientExtractionWorker(QThread):
    """Extract, code and export the clinical events of one patient."""
    progress = pyqtSignal(int, str)         # percentage, message
    result_ready = pyqtSignal(dict)         # result summary
    cancelled = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(self, pipeline, patient_id: str, *, num_workers: int = 1,
                 force_rebuild: bool = False, parent=None):
        super().__init__(parent)
        self.pipeline = pipeline
        self.patient_id = patient_id
        self.num_workers = max(1, int(num_workers or 1))
        self.force_rebuild = bool(force_rebuild)
        self._cancel_event = threading.Event()

    def cancel(self) -> None:
        """Request a safe stop after the active LLM call returns."""
        self._cancel_event.set()

    def run(self):
        try:
            result = self.pipeline.extract_patient(
                self.patient_id,
                incremental=not self.force_rebuild,
                num_workers=self.num_workers,
                progress_callback=lambda pct, msg: self.progress.emit(int(pct), str(msg)),
                cancel_check=self._cancel_event.is_set,
            )
            self.result_ready.emit(dict(result or {}))
        except ExtractionCancelled:
            self.cancelled.emit()
        except Exception as exc:
            self.error.emit(str(exc))


class RegistryQueueWorker(QThread):
    """Extract several patients through one shared pool of model slots.

    Each patient is finalized (and reported) as soon as its last document is
    done; cancellation is honoured after the active model calls return.
    """

    progress = pyqtSignal(int, str)
    patient_finished = pyqtSignal(str, dict)
    patient_error = pyqtSignal(str, str)
    cancelled = pyqtSignal()

    def __init__(self, pipeline, patient_ids: list[str], *, num_workers: int = 1,
                 force_rebuild: bool = False, parent=None):
        super().__init__(parent)
        self.pipeline = pipeline
        self.patient_ids = list(patient_ids)
        self.num_workers = max(1, int(num_workers or 1))
        self.force_rebuild = bool(force_rebuild)
        self._cancel_event = threading.Event()

    def cancel(self) -> None:
        """Request a safe stop after the active model calls return."""
        self._cancel_event.set()

    def run(self) -> None:
        try:
            self.pipeline.extract_patients(
                self.patient_ids, incremental=not self.force_rebuild,
                num_workers=self.num_workers,
                progress_callback=lambda pct, msg: self.progress.emit(int(pct), str(msg)),
                cancel_check=self._cancel_event.is_set,
                patient_finished=lambda pid, result: self.patient_finished.emit(pid, dict(result or {})),
                patient_error=lambda pid, error: self.patient_error.emit(pid, str(error)),
            )
        except ExtractionCancelled:
            self.cancelled.emit()
        except Exception as exc:
            self.patient_error.emit("", str(exc))


class ProjectExportWorker(QThread):
    """Write every patient's FHIR file and the project NDJSON export."""

    progress = pyqtSignal(int, str)
    result_ready = pyqtSignal(dict)
    cancelled = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(self, pipeline, parent=None):
        super().__init__(parent)
        self.pipeline = pipeline
        self._cancel_event = threading.Event()

    def cancel(self) -> None:
        self._cancel_event.set()

    def run(self):
        try:
            result = self.pipeline.export_project(
                progress=lambda pct, msg: self.progress.emit(int(pct), str(msg)),
                cancel_check=self._cancel_event.is_set,
            )
            self.result_ready.emit(result)
        except ExtractionCancelled:
            self.cancelled.emit()
        except Exception as exc:
            self.error.emit(str(exc))

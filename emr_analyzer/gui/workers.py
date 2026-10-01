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
    """Extract several patients in order; cancellation is honoured between calls."""

    patient_started = pyqtSignal(int, int, str)
    patient_progress = pyqtSignal(str, int, str)
    patient_finished = pyqtSignal(str, dict)
    patient_error = pyqtSignal(str, str)

    def __init__(self, pipeline, patient_ids: list[str], *, num_workers: int = 1,
                 force_rebuild: bool = False, parent=None):
        super().__init__(parent)
        self.pipeline = pipeline
        self.patient_ids = list(patient_ids)
        self.num_workers = max(1, int(num_workers or 1))
        self.force_rebuild = bool(force_rebuild)
        self._cancel_event = threading.Event()

    def cancel(self) -> None:
        """Request a safe stop after the patient currently being saved."""
        self._cancel_event.set()

    def run(self) -> None:
        total = len(self.patient_ids)
        for index, patient_id in enumerate(self.patient_ids, start=1):
            if self._cancel_event.is_set():
                break
            self.patient_started.emit(index, total, patient_id)

            def progress(percent: int, message: str, pid=patient_id) -> None:
                self.patient_progress.emit(pid, int(percent), str(message))

            try:
                result = self.pipeline.extract_patient(
                    patient_id, incremental=not self.force_rebuild,
                    num_workers=self.num_workers, progress_callback=progress,
                    cancel_check=self._cancel_event.is_set,
                )
                self.patient_finished.emit(patient_id, dict(result or {}))
            except ExtractionCancelled:
                break
            except Exception as exc:
                self.patient_error.emit(patient_id, str(exc))

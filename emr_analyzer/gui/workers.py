"""QThread workers for background processing.

Each worker runs in a separate thread and communicates
with the main GUI thread via Qt signals.
"""

from PyQt5.QtCore import QThread, pyqtSignal


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

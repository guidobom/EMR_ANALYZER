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


# Conversational context cap: last N messages, truncated per message, so
# the embedded history never crowds out the clinical registry in the
# context window (default LLM context is 32K tokens).
_MAX_CONVERSATION_MESSAGES = 20   # 10 Q&A exchanges
_MAX_CONVERSATION_CHARS_PER_MESSAGE = 300


def format_registry_context(entries: list[dict], limit: int = 100) -> str:
    """Compact registry lines with citable entry ids.

    Each line carries ``[#id]`` so prompts can ask the model to cite the
    exact registry entries behind every claim (traceability).
    """
    return "\n".join(
        f"[#{e.get('entry_id', '?')}] [{e.get('date_observed', '?')}] "
        f"[{e.get('category', '?')}] {e.get('description', '')}"
        for e in entries[-limit:]
    )


def build_query_prompt(
    clinical_profile: str,
    entries_text: str,
    question: str,
    conversation: list[dict] | None = None,
    use_conversation_context: bool = False,
) -> str:
    """Build the user prompt of a clinical query.

    When *use_conversation_context* is True and prior messages exist, a
    ``CONVERSAZIONE PRECEDENTE`` section is embedded before the question so
    follow-up questions can reference earlier answers (capped: last 20
    messages, 300 chars each).  Otherwise the prompt is identical to the
    classic single-shot form.
    """
    conversation_text = ""
    if use_conversation_context and conversation:
        lines = []
        for message in conversation[-_MAX_CONVERSATION_MESSAGES:]:
            speaker = (
                "UTENTE" if message.get("role") == "user" else "ASSISTENTE"
            )
            content = str(message.get("content") or "")
            if len(content) > _MAX_CONVERSATION_CHARS_PER_MESSAGE:
                content = content[:_MAX_CONVERSATION_CHARS_PER_MESSAGE] + "…"
            lines.append(f"[{speaker}] {content}")
        if lines:
            conversation_text = (
                "CONVERSAZIONE PRECEDENTE:\n" + "\n".join(lines) + "\n\n"
            )

    return (
        f"PROFILO CLINICO:\n{clinical_profile}\n\n"
        f"REGISTRO CRONOLOGICO:\n{entries_text}\n\n"
        f"{conversation_text}"
        f"DOMANDA: {question}\n\n"
        f"Rispondi in modo chiaro e conciso, citando date e fonti "
        f"quando disponibili."
    )


class ClinicalHistoryQueryWorker(QThread):
    """Run a clinical query against the timeline history."""
    finished = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, llm_client, entries: list[dict],
                 clinical_profile: str, question: str,
                 conversation: list[dict] | None = None,
                 use_conversation_context: bool = False, parent=None):
        super().__init__(parent)
        self.llm_client = llm_client
        self.entries = entries
        self.clinical_profile = clinical_profile
        self.question = question
        self.conversation = conversation
        self.use_conversation_context = use_conversation_context

    def run(self):
        try:
            system_prompt = (
                "Sei un assistente clinico esperto. Rispondi alla domanda "
                "basandoti ESCLUSIVAMENTE sui dati clinici forniti. "
                "Se un dato non e' disponibile, dichiaralo esplicitamente. "
                "Cita le date quando disponibili. Non inventare informazioni."
            )

            # Merge profile + timeline as compact context
            entries_text = format_registry_context(self.entries, limit=100)

            user_prompt = build_query_prompt(
                self.clinical_profile,
                entries_text,
                self.question,
                conversation=self.conversation,
                use_conversation_context=self.use_conversation_context,
            )

            answer = self.llm_client.generate_text(user_prompt, system_prompt)
            self.finished.emit(answer)
        except Exception as e:
            self.error.emit(f"Errore query: {str(e)}")


class IraeQueueWorker(QThread):
    """Run the irAE protocol over several patient registries, in order.

    ``patient_plans`` is a list of ``(patient_id, prompts)`` where the
    prompts are pre-built on the main thread (SQLite is not used
    cross-thread).  Cancellation is honored BETWEEN patients; the LLM
    call in flight cannot be interrupted.
    """
    patient_started = pyqtSignal(int, int, str)  # index, total, patient_id
    chunk_progress = pyqtSignal(int, int)        # chunk_index, chunk_total
    patient_finished = pyqtSignal(str, str)      # patient_id, markdown
    patient_error = pyqtSignal(str, str)         # patient_id, error
    finished = pyqtSignal()

    def __init__(self, llm_client, patient_plans, parent=None):
        super().__init__(parent)
        self.llm_client = llm_client
        self.patient_plans = patient_plans
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def run(self):
        from emr_analyzer.clinical.irae_analysis import SYSTEM_PROMPT

        total = len(self.patient_plans)
        for index, (patient_id, prompts) in enumerate(
            self.patient_plans, start=1
        ):
            if self._cancelled:
                break
            self.patient_started.emit(index, total, patient_id)
            try:
                parts = []
                for chunk_index, prompt in enumerate(prompts, start=1):
                    self.chunk_progress.emit(chunk_index, len(prompts))
                    answer = self.llm_client.generate_text(
                        prompt, SYSTEM_PROMPT
                    )
                    parts.append(
                        f"### Parte {chunk_index}/{len(prompts)}\n\n"
                        f"{answer}"
                    )
                self.patient_finished.emit(
                    patient_id, "\n\n".join(parts)
                )
            except Exception as exc:
                self.patient_error.emit(
                    patient_id, f"Errore analisi irAE: {str(exc)}"
                )
        self.finished.emit()


class IraeAnalysisWorker(QThread):
    """Run the irAE protocol over the whole registry, chunk by chunk."""
    progress = pyqtSignal(int, int)  # chunk_index, chunk_total
    finished = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, llm_client, prompts: list[str], parent=None):
        super().__init__(parent)
        self.llm_client = llm_client
        self.prompts = prompts

    def run(self):
        from emr_analyzer.clinical.irae_analysis import SYSTEM_PROMPT

        try:
            parts = []
            total = len(self.prompts)
            for index, prompt in enumerate(self.prompts, start=1):
                self.progress.emit(index, total)
                answer = self.llm_client.generate_text(
                    prompt, SYSTEM_PROMPT
                )
                parts.append(
                    f"### Parte {index}/{total}\n\n{answer}"
                )
            self.finished.emit("\n\n".join(parts))
        except Exception as exc:
            self.error.emit(f"Errore analisi irAE: {str(exc)}")

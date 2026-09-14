"""Editor and per-task version selector for external LLM prompts."""

from __future__ import annotations

import json
import time

from PyQt5.QtCore import QThread, Qt, pyqtSignal
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QComboBox,
    QDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QInputDialog,
    QPlainTextEdit,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from ..config import active_workspace
from ..prompt_catalog import (
    PromptConfigurationError,
    PromptVersion,
    activate_prompt,
    load_prompt,
    prompt_definitions,
    prompt_versions,
    save_custom_prompt,
)


_PROMPT_ROLE = {
    "patient_identity_system": "document",
    "patient_identity_task": "document",
    "clinical_text_system": "document",
    "clinical_text_instructions": "document",
    "atomic_evidence_system": "atomic_evidence",
    "atomic_evidence_it": "atomic_evidence",
    "atomic_evidence_repair_system": "atomic_evidence",
    "atomic_evidence_coverage_system": "atomic_evidence",
    "clinical_fusion_system": "clinical_events",
    "clinical_fusion_task": "clinical_events",
    "evidence_relations_system": "clinical_events",
    "evidence_relations_task": "clinical_events",
    "episode_assembly_system": "clinical_events",
    "episode_assembly_task": "clinical_events",
    "hypothesis_system": "clinical_events",
    "hypothesis_task": "clinical_events",
    "clinical_query_system": "clinical_state",
    "narrative_profile_system": "clinical_state",
    "narrative_profile_task": "clinical_state",
    "irae_system": "clinical_state",
}

_ROLE_SERVICE = {
    "document": "document_llm_client",
    "atomic_evidence": "atomic_evidence_llm_client",
    "clinical_events": "clinical_events_llm_client",
    "clinical_state": "clinical_state_llm_client",
}

_PROMPT_PAIRS = {
    "patient_identity_system": "patient_identity_task",
    "patient_identity_task": "patient_identity_system",
    "clinical_text_system": "clinical_text_instructions",
    "clinical_text_instructions": "clinical_text_system",
    "clinical_fusion_system": "clinical_fusion_task",
    "clinical_fusion_task": "clinical_fusion_system",
    "evidence_relations_system": "evidence_relations_task",
    "evidence_relations_task": "evidence_relations_system",
    "episode_assembly_system": "episode_assembly_task",
    "episode_assembly_task": "episode_assembly_system",
    "hypothesis_system": "hypothesis_task",
    "hypothesis_task": "hypothesis_system",
    "narrative_profile_system": "narrative_profile_task",
    "narrative_profile_task": "narrative_profile_system",
}


class PromptPreviewWorker(QThread):
    """Execute a read-only pipeline preview away from the GUI thread."""

    completed = pyqtSignal(dict)
    failed = pyqtSignal(str)

    def __init__(self, llm_client, request: dict, parent=None):
        super().__init__(parent)
        self.llm = llm_client
        self.request = request

    def run(self) -> None:
        started = time.perf_counter()
        try:
            mode = self.request["mode"]
            if mode == "clinical_text":
                from ..extraction.clinical_text_isolator import (
                    ClinicalTextIsolator,
                )

                isolator = ClinicalTextIsolator(
                    self.llm,
                    system_prompt=self.request["system_prompt"],
                    instructions_prompt=self.request["task_prompt"],
                )
                result = isolator.isolate(
                    self.request["text"],
                    document_date=self.request.get("document_date"),
                )
                output = result.text
                extra = {
                    "chunk_count": result.chunk_count,
                    "warnings": result.warnings,
                    "redaction_counts": result.redaction_counts,
                }
            elif mode == "atomic_evidence":
                from ..clinical.atomic_evidence import AtomicEvidenceExtractor

                extractor = AtomicEvidenceExtractor(
                    self.llm,
                    task_prompt=self.request.get("task_prompt"),
                    system_prompt=self.request.get("system_prompt"),
                    repair_system_prompt=self.request.get(
                        "repair_system_prompt"
                    ),
                    coverage_system_prompt=self.request.get(
                        "coverage_system_prompt"
                    ),
                )
                evidence = extractor.extract_document(
                    patient_id=self.request["patient_id"],
                    document_id=self.request["document_id"],
                    document_type=self.request.get("document_type") or "",
                    document_date=self.request.get("document_date"),
                    text=self.request["text"],
                )
                output = json.dumps(
                    [item.to_atomic_dict() for item in evidence],
                    ensure_ascii=False,
                    indent=2,
                )
                extra = {
                    "evidence_count": len(evidence),
                    "pipeline_metrics": extractor.last_extraction_metrics(),
                }
            elif mode == "structured":
                result = self.llm.generate_structured(
                    self.request["user_prompt"],
                    self.request["system_prompt"],
                    self.request["schema"],
                )
                output = json.dumps(result, ensure_ascii=False, indent=2)
                extra = {}
            else:
                output = self.llm.generate_text(
                    self.request["user_prompt"],
                    self.request["system_prompt"],
                )
                extra = {}
            if self.isInterruptionRequested():
                return
            metadata_getter = getattr(
                self.llm, "last_generation_metadata", None
            )
            generation = (
                metadata_getter() if callable(metadata_getter) else {}
            )
            self.completed.emit({
                "output": str(output or ""),
                "elapsed_seconds": round(time.perf_counter() - started, 2),
                "model": str(getattr(self.llm, "model", "") or ""),
                "generation": generation or {},
                **extra,
            })
        except Exception as exc:
            if not self.isInterruptionRequested():
                self.failed.emit(
                    f"{type(exc).__name__}: {' '.join(str(exc).split())}"
                )


class PromptManagerDialog(QDialog):
    """Inspect, edit, version and activate prompts without changing code."""

    def __init__(
        self,
        parent=None,
        *,
        services: dict | None = None,
        patient_id: str | None = None,
    ):
        super().__init__(parent)
        self._services = services or {}
        self._initial_patient_id = patient_id
        self._preview_worker: PromptPreviewWorker | None = None
        self.setWindowTitle("Prompt Manager")
        self.resize(1080, 760)
        self.setMinimumSize(760, 560)
        self._current_version: PromptVersion | None = None
        self._loading = False
        self._last_task_index = -1
        self._last_version_index = -1
        self._build_ui()
        self._populate_tasks()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        introduction = QLabel(
            "Seleziona il compito e la versione del prompt. Le versioni "
            "istituzionali sono protette; per modificarle salvale come nuova "
            "versione personalizzata. La versione marcata ATTIVA è quella "
            "utilizzata dalla relativa pipeline."
        )
        introduction.setWordWrap(True)
        layout.addWidget(introduction)

        selectors = QFormLayout()
        self.task_combo = QComboBox()
        self.task_combo.setObjectName("promptTaskCombo")
        self.task_combo.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        self.task_combo.currentIndexChanged.connect(self._on_task_changed)
        selectors.addRow("Compito / prompt:", self.task_combo)

        version_row = QWidget()
        version_layout = QHBoxLayout(version_row)
        version_layout.setContentsMargins(0, 0, 0, 0)
        self.version_combo = QComboBox()
        self.version_combo.setObjectName("promptVersionCombo")
        self.version_combo.currentIndexChanged.connect(
            self._on_version_changed
        )
        version_layout.addWidget(self.version_combo, 1)
        self.activate_button = QPushButton("Usa per questo compito")
        self.activate_button.setObjectName("activatePromptVersionButton")
        self.activate_button.clicked.connect(self._activate_selected)
        version_layout.addWidget(self.activate_button)
        selectors.addRow("Versione:", version_row)
        layout.addLayout(selectors)

        self.status_label = QLabel()
        self.status_label.setObjectName("promptVersionStatus")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        self.path_label = QLabel()
        self.path_label.setObjectName("promptPathLabel")
        self.path_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.path_label.setWordWrap(True)
        layout.addWidget(self.path_label)

        self.editor = QPlainTextEdit()
        self.editor.setObjectName("promptEditor")
        self.editor.setLineWrapMode(QPlainTextEdit.WidgetWidth)
        editor_font = QFont("Menlo")
        editor_font.setStyleHint(QFont.Monospace)
        self.editor.setFont(editor_font)
        self.editor.setPlaceholderText("Testo del prompt selezionato")
        self.editor.document().modificationChanged.connect(
            self._update_button_state
        )
        content_splitter = QSplitter(Qt.Vertical)
        content_splitter.addWidget(self.editor)
        content_splitter.addWidget(self._build_preview_panel())
        content_splitter.setStretchFactor(0, 3)
        content_splitter.setStretchFactor(1, 2)
        content_splitter.setSizes([430, 250])
        layout.addWidget(content_splitter, 1)

        button_row = QHBoxLayout()
        self.reload_button = QPushButton("Ripristina testo salvato")
        self.reload_button.clicked.connect(self._load_selected_version)
        button_row.addWidget(self.reload_button)
        button_row.addStretch(1)
        self.save_button = QPushButton("Salva modifiche")
        self.save_button.setObjectName("savePromptButton")
        self.save_button.clicked.connect(self._save_current_custom)
        button_row.addWidget(self.save_button)
        self.save_as_button = QPushButton("Salva come nuova versione…")
        self.save_as_button.setObjectName("savePromptAsButton")
        self.save_as_button.clicked.connect(self._save_as_new_version)
        button_row.addWidget(self.save_as_button)
        close_button = QPushButton("Chiudi")
        close_button.clicked.connect(self.reject)
        button_row.addWidget(close_button)
        layout.addLayout(button_row)

    def _build_preview_panel(self) -> QWidget:
        panel = QGroupBox("Prova non distruttiva su un documento")
        layout = QVBoxLayout(panel)
        controls = QHBoxLayout()
        self.preview_patient_combo = QComboBox()
        self.preview_patient_combo.setObjectName("previewPatientCombo")
        self.preview_patient_combo.currentIndexChanged.connect(
            self._populate_preview_documents
        )
        controls.addWidget(QLabel("Paziente:"))
        controls.addWidget(self.preview_patient_combo, 1)
        self.preview_document_combo = QComboBox()
        self.preview_document_combo.setObjectName("previewDocumentCombo")
        self.preview_document_combo.currentIndexChanged.connect(
            self._update_preview_availability
        )
        controls.addWidget(QLabel("Documento:"))
        controls.addWidget(self.preview_document_combo, 2)
        self.run_preview_button = QPushButton("▶ Esegui prova")
        self.run_preview_button.setObjectName("runPromptPreviewButton")
        self.run_preview_button.clicked.connect(self._run_preview)
        controls.addWidget(self.run_preview_button)
        layout.addLayout(controls)

        self.preview_info_label = QLabel()
        self.preview_info_label.setObjectName("promptPreviewInfo")
        self.preview_info_label.setWordWrap(True)
        layout.addWidget(self.preview_info_label)
        self.preview_result = QPlainTextEdit()
        self.preview_result.setObjectName("promptPreviewResult")
        self.preview_result.setReadOnly(True)
        self.preview_result.setPlaceholderText(
            "Qui comparirà il risultato della prova; il database e il "
            "registro cronologico non verranno modificati."
        )
        layout.addWidget(self.preview_result, 1)
        self._populate_preview_patients()
        return panel

    def _populate_tasks(self) -> None:
        self._loading = True
        try:
            self.task_combo.clear()
            definitions = sorted(
                prompt_definitions(), key=lambda item: (item.pipeline, item.label)
            )
            for definition in definitions:
                if definition.key.startswith("clinical_text_"):
                    continue  # Clinical Markdown is now filtered without prompts.
                self.task_combo.addItem(
                    f"{definition.pipeline} — {definition.label}",
                    definition.key,
                )
        except PromptConfigurationError as exc:
            QMessageBox.critical(self, "Catalogo prompt non valido", str(exc))
        finally:
            self._loading = False
        if self.task_combo.count():
            self._on_task_changed(self.task_combo.currentIndex())

    def _populate_preview_patients(self) -> None:
        patient_repo = self._services.get("patient_repo")
        self.preview_patient_combo.blockSignals(True)
        self.preview_patient_combo.clear()
        selected_index = -1
        if patient_repo is not None:
            try:
                patients = patient_repo.list_all()
            except Exception:
                patients = []
            for patient in patients:
                label = (
                    f"{patient.pseudonym or patient.id} — {patient.id}"
                )
                self.preview_patient_combo.addItem(label, patient.id)
                if patient.id == self._initial_patient_id:
                    selected_index = self.preview_patient_combo.count() - 1
        if selected_index >= 0:
            self.preview_patient_combo.setCurrentIndex(selected_index)
        self.preview_patient_combo.blockSignals(False)
        self._populate_preview_documents()

    def _populate_preview_documents(self, *_args) -> None:
        patient_id = str(self.preview_patient_combo.currentData() or "")
        doc_repo = self._services.get("document_repo")
        self.preview_document_combo.blockSignals(True)
        self.preview_document_combo.clear()
        if patient_id and doc_repo is not None:
            try:
                documents = doc_repo.list_by_patient(patient_id)
            except Exception:
                documents = []
            for document in documents:
                date = document.document_date or "data non disponibile"
                self.preview_document_combo.addItem(
                    f"{document.id} · {date} · {document.filename}",
                    document,
                )
        self.preview_document_combo.blockSignals(False)
        self._update_preview_availability()

    def _update_preview_availability(self, *_args) -> None:
        key = self._selected_key()
        role = _PROMPT_ROLE.get(key, "clinical_state")
        service_name = _ROLE_SERVICE[role]
        client = self._services.get(service_name)
        document = self.preview_document_combo.currentData()
        running = bool(
            self._preview_worker is not None
            and self._preview_worker.isRunning()
        )
        available = bool(
            client is not None
            and getattr(client, "is_available", False)
            and document is not None
            and not running
        )
        self.run_preview_button.setEnabled(available)
        if running:
            self.run_preview_button.setText("Prova in corso…")
            return
        self.run_preview_button.setText("▶ Esegui prova")
        if document is None:
            self.preview_info_label.setText(
                "Seleziona un paziente con almeno un documento importato."
            )
            return
        if client is None or not getattr(client, "is_available", False):
            self.preview_info_label.setText(
                f"Nessun modello disponibile per il ruolo «{role}». "
                "Configurarlo in Configura LLM."
            )
            return
        layer = self._preview_layer_name(key)
        self.preview_info_label.setText(
            f"Modello: {getattr(client, 'model', '—')} · Input: {layer}. "
            "La prova non salva evidenze, eventi o modifiche nel database."
        )

    @staticmethod
    def _preview_layer_name(key: str) -> str:
        if key.startswith("patient_identity"):
            return "testo sorgente pre-normalizzazione"
        if key.startswith("clinical_text"):
            return "testo sorgente pulito"
        if key.startswith((
            "clinical_fusion", "evidence_relations", "episode_assembly",
            "hypothesis",
        )):
            return "evidenze atomiche già estratte dal documento"
        return "testo clinico normalizzato"

    def _source_for_preview(self, key: str, document) -> tuple[str, str]:
        extraction_dir = (
            active_workspace.path / document.patient_id / "extraction"
        )
        event_prompt = key.startswith((
            "clinical_fusion", "evidence_relations", "episode_assembly",
            "hypothesis",
        ))
        if event_prompt:
            evidence_repo = self._services.get("evidence_repo")
            evidence = (
                evidence_repo.get_by_document(document.id)
                if evidence_repo is not None else []
            )
            if evidence:
                return (
                    json.dumps(
                        [item.to_atomic_dict() for item in evidence],
                        ensure_ascii=False,
                        indent=2,
                    ),
                    f"{len(evidence)} evidenze atomiche",
                )

        if key.startswith(("patient_identity", "clinical_text")):
            from ..utils.document_paths import resolve_document_path
            from ..pipeline.pdf_extractor import PdfPlumberExtractor
            parser = PdfPlumberExtractor()
            result = parser.convert(resolve_document_path(document))
            return parser.export_text(result), "testo estratto dall’originale in memoria"
        else:
            candidates = (
                extraction_dir / f"{document.id}.md",
                active_workspace.path / document.patient_id / "docling"
                / f"{document.id}.md",
            )
            layer = "testo clinico normalizzato"
        path = next((item for item in candidates if item.is_file()), None)
        if path is None:
            raise PromptConfigurationError(
                "Il documento selezionato non dispone ancora del livello "
                "testuale richiesto."
            )
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            raise PromptConfigurationError(
                "Il livello testuale del documento selezionato è vuoto."
            )
        if not key.startswith(("patient_identity", "clinical_text")):
            overlay_repo = self._services.get("overlay_repo")
            if overlay_repo is not None:
                text = overlay_repo.effective_text(document.id, text)
        return text, f"{layer}: {path.name}"

    @staticmethod
    def _structured_schema(key: str) -> dict | None:
        if key.startswith("patient_identity"):
            return {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "name": {"type": "string"},
                    "birth_date": {"type": "string"},
                    "fiscal_code": {"type": "string"},
                    "confidence": {
                        "type": "number", "minimum": 0, "maximum": 1,
                    },
                },
                "required": [
                    "name", "birth_date", "fiscal_code", "confidence",
                ],
            }
        if key.startswith("clinical_fusion"):
            from ..clinical.consolidation import FUSION_SCHEMA
            return FUSION_SCHEMA
        if key.startswith("evidence_relations"):
            from ..clinical.evidence_graph import _RELATION_SCHEMA
            return _RELATION_SCHEMA
        if key.startswith("episode_assembly"):
            from ..clinical.episode_synthesis import _SCHEMA
            return _SCHEMA
        if key.startswith("hypothesis"):
            from ..clinical.hypothesis_discovery import _SCHEMA
            return _SCHEMA
        return None

    def _paired_prompts(self, key: str, edited_text: str) -> tuple[str, str]:
        pair = _PROMPT_PAIRS.get(key)
        task_keys = {
            "patient_identity_task", "clinical_text_instructions",
            "clinical_fusion_task", "evidence_relations_task",
            "episode_assembly_task", "hypothesis_task",
            "narrative_profile_task",
        }
        if key in task_keys:
            return load_prompt(pair), edited_text
        task = load_prompt(pair) if pair else (
            "Analizza esclusivamente i dati del documento fornito e "
            "restituisci il risultato richiesto dal prompt."
        )
        if key == "irae_system":
            try:
                from ..clinical.irae_analysis import load_prompt as load_irae
                task = load_irae()
            except Exception:
                pass
        return edited_text, task

    def _build_preview_request(self, key: str, document) -> dict:
        edited_text = self.editor.toPlainText().strip()
        if not edited_text:
            raise PromptConfigurationError("Il prompt nell'editor è vuoto.")
        source, source_description = self._source_for_preview(key, document)
        base = {
            "patient_id": document.patient_id,
            "document_id": document.id,
            "document_type": document.document_type,
            "document_date": document.document_date,
            "text": source,
            "source_description": source_description,
        }
        if key.startswith("clinical_text"):
            system, task = self._paired_prompts(key, edited_text)
            return {
                **base, "mode": "clinical_text",
                "system_prompt": system, "task_prompt": task,
            }
        if key.startswith("atomic_evidence"):
            prompts = {
                "task_prompt": load_prompt("atomic_evidence_it"),
                "system_prompt": load_prompt("atomic_evidence_system"),
                "repair_system_prompt": load_prompt(
                    "atomic_evidence_repair_system"
                ),
                "coverage_system_prompt": load_prompt(
                    "atomic_evidence_coverage_system"
                ),
            }
            field_by_key = {
                "atomic_evidence_it": "task_prompt",
                "atomic_evidence_system": "system_prompt",
                "atomic_evidence_repair_system": "repair_system_prompt",
                "atomic_evidence_coverage_system": (
                    "coverage_system_prompt"
                ),
            }
            prompts[field_by_key[key]] = edited_text
            return {**base, "mode": "atomic_evidence", **prompts}

        system, task = self._paired_prompts(key, edited_text)
        user_prompt = (
            task
            + "\n\nDATI DEL DOCUMENTO DI PROVA (non modificare la fonte):\n"
            + source
        )
        schema = self._structured_schema(key)
        return {
            **base,
            "mode": "structured" if schema else "text",
            "system_prompt": system,
            "task_prompt": task,
            "user_prompt": user_prompt,
            "schema": schema,
        }

    def _run_preview(self) -> None:
        if self._preview_worker is not None and self._preview_worker.isRunning():
            return
        key = self._selected_key()
        document = self.preview_document_combo.currentData()
        role = _PROMPT_ROLE.get(key, "clinical_state")
        client = self._services.get(_ROLE_SERVICE[role])
        if document is None or client is None:
            self._update_preview_availability()
            return
        try:
            request = self._build_preview_request(key, document)
        except (OSError, PromptConfigurationError) as exc:
            QMessageBox.warning(self, "Prova non avviata", str(exc))
            return
        self.preview_result.clear()
        self.preview_info_label.setText(
            f"Prova in corso con {getattr(client, 'model', 'modello locale')} "
            f"su {document.id} · {request['source_description']}…"
        )
        worker = PromptPreviewWorker(client, request, self)
        self._preview_worker = worker
        worker.completed.connect(self._on_preview_completed)
        worker.failed.connect(self._on_preview_failed)
        worker.finished.connect(self._on_preview_finished)
        worker.start()
        self._update_preview_availability()

    def _on_preview_completed(self, result: dict) -> None:
        metrics = {
            "modello": result.get("model"),
            "durata_secondi": result.get("elapsed_seconds"),
            "generazione": result.get("generation") or {},
        }
        for key in ("chunk_count", "evidence_count", "pipeline_metrics"):
            if key in result:
                metrics[key] = result[key]
        self.preview_result.setPlainText(
            "METRICHE\n"
            + json.dumps(metrics, ensure_ascii=False, indent=2)
            + "\n\nRISULTATO\n"
            + result.get("output", "")
        )
        self.preview_info_label.setText(
            "Prova completata senza modificare il database."
        )

    def _on_preview_failed(self, error: str) -> None:
        self.preview_result.setPlainText("ERRORE\n" + error)
        self.preview_info_label.setText(
            "La prova è terminata con errore; nessun dato è stato salvato."
        )

    def _on_preview_finished(self) -> None:
        worker = self._preview_worker
        self._preview_worker = None
        if worker is not None:
            worker.deleteLater()
        self._update_preview_availability()

    def _selected_key(self) -> str:
        return str(self.task_combo.currentData() or "")

    def _on_task_changed(self, _index: int) -> None:
        if self._loading:
            return
        if not self._confirm_discard_editor_changes():
            self._loading = True
            self.task_combo.setCurrentIndex(self._last_task_index)
            self._loading = False
            return
        self._populate_versions()

    def _populate_versions(
        self,
        *,
        select_origin: str | None = None,
        select_version: str | None = None,
    ) -> None:
        key = self._selected_key()
        self._loading = True
        try:
            self.version_combo.clear()
            selected_index = -1
            for version in prompt_versions(key):
                self.version_combo.addItem(version.display_name, version)
                index = self.version_combo.count() - 1
                if select_origin is not None:
                    if (
                        version.origin == select_origin
                        and version.version == select_version
                    ):
                        selected_index = index
                elif version.active:
                    selected_index = index
            if selected_index >= 0:
                self.version_combo.setCurrentIndex(selected_index)
        except PromptConfigurationError as exc:
            QMessageBox.critical(self, "Versioni non disponibili", str(exc))
        finally:
            self._loading = False
        self._on_version_changed(self.version_combo.currentIndex())

    def _on_version_changed(self, _index: int) -> None:
        if self._loading:
            return
        if not self._confirm_discard_editor_changes():
            self._loading = True
            self.version_combo.setCurrentIndex(self._last_version_index)
            self._loading = False
            return
        self._load_selected_version()

    def _confirm_discard_editor_changes(self) -> bool:
        if not self.editor.document().isModified():
            return True
        choice = QMessageBox.question(
            self,
            "Modifiche non salvate",
            "Cambiare selezione e scartare le modifiche non salvate?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if choice != QMessageBox.Yes:
            return False
        self.editor.document().setModified(False)
        return True

    def _load_selected_version(self) -> None:
        version = self.version_combo.currentData()
        if not isinstance(version, PromptVersion):
            self._current_version = None
            self.editor.clear()
            self._update_button_state()
            return
        try:
            text = load_prompt(
                version.key,
                version=version.version,
                origin=version.origin,
            )
        except PromptConfigurationError as exc:
            QMessageBox.critical(self, "Prompt non leggibile", str(exc))
            return
        self._current_version = version
        self._last_task_index = self.task_combo.currentIndex()
        self._last_version_index = self.version_combo.currentIndex()
        self.editor.setPlainText(text)
        self.editor.document().setModified(False)
        self.path_label.setText(f"File: {version.path}")
        if version.origin == "institutional":
            detail = (
                "Versione istituzionale protetta: le modifiche nell'editor "
                "possono essere salvate soltanto come nuova versione."
            )
        else:
            detail = "Versione personalizzata modificabile."
        if version.active:
            detail = "ATTIVA per questo compito. " + detail
        self.status_label.setText(detail)
        self._update_button_state()
        self._update_preview_availability()

    def _update_button_state(self, *_args) -> None:
        version = self._current_version
        has_version = isinstance(version, PromptVersion)
        modified = self.editor.document().isModified()
        self.save_button.setEnabled(
            bool(has_version and version.origin == "custom" and modified)
        )
        self.save_as_button.setEnabled(has_version)
        self.reload_button.setEnabled(has_version and modified)
        self.activate_button.setEnabled(
            bool(has_version and not version.active and not modified)
        )

    def _save_current_custom(self) -> None:
        version = self._current_version
        if version is None or version.origin != "custom":
            return
        try:
            save_custom_prompt(
                version.key,
                version.version,
                self.editor.toPlainText(),
                overwrite=True,
            )
        except PromptConfigurationError as exc:
            QMessageBox.critical(self, "Prompt non salvato", str(exc))
            return
        self.editor.document().setModified(False)
        self._populate_versions(
            select_origin="custom", select_version=version.version
        )
        self.status_label.setText(
            "Versione personalizzata salvata. Se era già attiva, il nuovo "
            "testo verrà usato dalle nuove elaborazioni dopo il riavvio "
            "dell'applicazione."
        )

    def _next_custom_version(self) -> str:
        occupied = {
            version.version for version in prompt_versions(self._selected_key())
            if version.origin == "custom"
        }
        index = 1
        while f"custom-v{index}" in occupied:
            index += 1
        return f"custom-v{index}"

    def _save_as_new_version(self) -> None:
        current = self._current_version
        if current is None:
            return
        version_name, accepted = QInputDialog.getText(
            self,
            "Nuova versione del prompt",
            "Nome versione (lettere, numeri, punto, trattino o underscore):",
            text=self._next_custom_version(),
        )
        if not accepted:
            return
        try:
            saved = save_custom_prompt(
                current.key, version_name, self.editor.toPlainText()
            )
        except PromptConfigurationError as exc:
            QMessageBox.critical(self, "Prompt non salvato", str(exc))
            return
        self.editor.document().setModified(False)
        self._populate_versions(
            select_origin=saved.origin, select_version=saved.version
        )
        self.status_label.setText(
            "Nuova versione salvata. Premi «Usa per questo compito» per "
            "renderla attiva."
        )

    def _activate_selected(self) -> None:
        version = self._current_version
        if version is None:
            return
        if self.editor.document().isModified():
            QMessageBox.information(
                self,
                "Modifiche non salvate",
                "Salva prima le modifiche, poi attiva la versione.",
            )
            return
        try:
            activate_prompt(version.key, version.version, version.origin)
        except PromptConfigurationError as exc:
            QMessageBox.critical(self, "Versione non attivata", str(exc))
            return
        self._populate_versions(
            select_origin=version.origin, select_version=version.version
        )
        self.status_label.setText(
            "Versione ATTIVA per questo compito. Verrà usata dalle nuove "
            "elaborazioni dopo il riavvio dell'applicazione."
        )

    def reject(self) -> None:
        if (
            self._preview_worker is not None
            and self._preview_worker.isRunning()
        ):
            QMessageBox.information(
                self,
                "Prova in corso",
                "Attendi il completamento della prova prima di chiudere il "
                "Prompt Manager. L'uscita generale dall'applicazione resta "
                "comunque disponibile.",
            )
            return
        if self.editor.document().isModified():
            choice = QMessageBox.question(
                self,
                "Modifiche non salvate",
                "Chiudere e scartare le modifiche non salvate?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if choice != QMessageBox.Yes:
                return
        super().reject()

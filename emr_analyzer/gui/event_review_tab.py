"""Patient events: extracted clinical events, SNOMED codes and their source text."""

from __future__ import annotations

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor, QTextCursor
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QSplitter, QLabel, QPushButton,
    QProgressBar, QTableWidget, QTableWidgetItem, QAbstractItemView,
    QHeaderView, QLineEdit, QComboBox, QPlainTextEdit, QTextEdit,
    QMessageBox, QDialog,
)

from .pipeline_llm import prepare_pipeline
from .qt_utils import qt_offset
from ..clinical.evidence_utils import content_hash
from ..clinical.grounded_sources import METHOD
from ..clinical.snomed_coding import concept_key

TYPE_LABELS = {
    "diagnosis": "Diagnosi", "symptom": "Sintomo", "clinical_sign": "Segno",
    "medication": "Farmaco", "procedure": "Procedura", "laboratory_test": "Laboratorio",
    "vital_sign": "Parametro vitale", "radiology_finding": "Imaging",
    "instrumental_finding": "Strumentale", "histopathology": "Istologia",
    "biomarker": "Biomarcatore", "clinical_decision": "Decisione",
    "hospitalization": "Ricovero", "discharge": "Dimissione",
}
ASSERTION_LABELS = {"present": "presente", "absent": "negato", "unknown": "incerto"}
STATUS_LABELS = {"proposed": "proposto", "needs_review": "da rivedere"}
CODING_LABELS = {"proposed": "codice proposto", "confirmed": "codice confermato",
                 "needs_review": "codice da rivedere"}
FILTERS = ("Tutti", "Da rivedere", "Senza codice SNOMED", "Con codice SNOMED")


class SourceTextView(QPlainTextEdit):
    """Read-only active text with one exact source interval highlighted."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setReadOnly(True)
        self.setLineWrapMode(QPlainTextEdit.WidgetWidth)
        self._text = ""

    def show_text(self, text: str, start: int | None = None, end: int | None = None) -> None:
        if text != self._text:
            self._text = text
            self.setPlainText(text)
        selections = []
        if start is not None and end is not None and 0 <= start < end <= len(text):
            highlight = QTextEdit.ExtraSelection()
            highlight.format.setBackground(QColor("#fff176"))
            cursor = self.textCursor()
            cursor.setPosition(qt_offset(text, start))
            cursor.setPosition(qt_offset(text, end), QTextCursor.KeepAnchor)
            highlight.cursor = cursor
            selections.append(highlight)
            caret = self.textCursor()
            caret.setPosition(qt_offset(text, start))
            self.setTextCursor(caret)
            self.centerCursor()
        self.setExtraSelections(selections)


class EventReviewTab(QWidget):
    """One row per extracted event; selecting it shows the source fragment."""

    _WORKER_ATTRS = ("_worker",)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._services: dict = {}
        self._patient_id: str | None = None
        self._events: list = []
        self._mappings: dict = {}
        self._texts: dict[str, str] = {}
        self._worker = None
        self._setup_ui()

    # ------------------------------------------------------------------ UI
    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        actions = QHBoxLayout()
        self._run_btn = QPushButton("▶ Elabora paziente")
        self._run_btn.setToolTip(
            "Estrae gli eventi dai referti anonimizzati non ancora elaborati, "
            "li codifica in SNOMED CT e scrive il file FHIR del paziente.")
        self._run_btn.clicked.connect(self._on_run)
        actions.addWidget(self._run_btn)
        self._cancel_btn = QPushButton("⏹ Interrompi")
        self._cancel_btn.setToolTip("Si ferma al termine della chiamata LLM in corso; "
                                    "i documenti completati restano salvati.")
        self._cancel_btn.clicked.connect(self._on_cancel)
        self._cancel_btn.setVisible(False)
        actions.addWidget(self._cancel_btn)
        self._export_btn = QPushButton("Esporta FHIR")
        self._export_btn.setToolTip("Riscrive il file FHIR del paziente con lo stato attuale della revisione.")
        self._export_btn.clicked.connect(self._export_fhir)
        actions.addWidget(self._export_btn)
        self._fhir_btn = QPushButton("Apri FHIR")
        self._fhir_btn.clicked.connect(self._show_fhir)
        actions.addWidget(self._fhir_btn)
        self._progress = QProgressBar()
        self._progress.setVisible(False)
        actions.addWidget(self._progress, stretch=1)
        actions.addStretch()
        layout.addLayout(actions)

        self._status = QLabel("")
        self._status.setWordWrap(True)
        layout.addWidget(self._status)

        filters = QHBoxLayout()
        self._filter = QComboBox()
        self._filter.addItems(FILTERS)
        self._filter.currentIndexChanged.connect(self._render)
        filters.addWidget(self._filter)
        self._search = QLineEdit()
        self._search.setPlaceholderText("Cerca evento, codice o citazione…")
        self._search.textChanged.connect(self._render)
        filters.addWidget(self._search, stretch=1)
        layout.addLayout(filters)

        splitter = QSplitter(Qt.Horizontal)
        self._table = QTableWidget(0, 7)
        self._table.setHorizontalHeaderLabels(
            ["Data", "Evento", "Tipo", "Asserzione", "SNOMED CT", "Stato", "Documento"])
        self._table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._table.setSelectionMode(QAbstractItemView.SingleSelection)
        self._table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._table.setAlternatingRowColors(True)
        self._table.verticalHeader().setVisible(False)
        self._table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self._table.itemSelectionChanged.connect(self._on_selection)
        splitter.addWidget(self._table)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        self._detail = QLabel("Seleziona un evento per vedere il frammento sorgente.")
        self._detail.setWordWrap(True)
        self._detail.setTextInteractionFlags(Qt.TextSelectableByMouse)
        right_layout.addWidget(self._detail)
        self._source = SourceTextView()
        right_layout.addWidget(self._source, stretch=1)
        self._original_btn = QPushButton("Apri referto originale")
        self._original_btn.clicked.connect(self._open_original)
        self._original_btn.setEnabled(False)
        right_layout.addWidget(self._original_btn)
        splitter.addWidget(right)
        splitter.setSizes([620, 480])
        layout.addWidget(splitter, stretch=1)

    # --------------------------------------------------------------- data
    def set_services(self, services: dict) -> None:
        self._services = services

    def load_patient(self, patient_id: str | None) -> None:
        self._patient_id = patient_id
        self._texts.clear()
        self._refresh()

    def _refresh(self) -> None:
        repo = self._services.get("evidence_repo")
        pipeline = self._services.get("extraction_pipeline")
        self._events = []
        if self._patient_id and repo is not None:
            self._events = [item for item in repo.get_by_patient(self._patient_id)
                            if item.extraction_method == METHOD]
        self._mappings = pipeline.coder.resolve(self._events) if pipeline is not None else {}
        coded = sum(bool(self._coding(item).get("code")) for item in self._events)
        review = sum(item.status == "needs_review" for item in self._events)
        status = pipeline.patient_status(self._patient_id) if (pipeline and self._patient_id) else {}
        self._status.setText(
            f"{len(self._events)} eventi · {coded} con codice SNOMED · {review} da rivedere"
            + (f" · documenti elaborati {status['completed']}/{status['with_text']}"
               f" (falliti {status['failed']})" if status else ""))
        self._render()

    def _coding(self, item) -> dict:
        return self._mappings.get(concept_key(item.normalized_entity, item.fact_type)) or {}

    def _visible_events(self):
        mode = self._filter.currentText()
        needle = self._search.text().strip().casefold()
        for item in self._events:
            coding = self._coding(item)
            code = coding.get("code")
            if mode == "Da rivedere" and item.status != "needs_review":
                continue
            if mode == "Senza codice SNOMED" and code:
                continue
            if mode == "Con codice SNOMED" and not code:
                continue
            if needle and needle not in " ".join(str(value or "") for value in (
                    item.normalized_entity, code, coding.get("term"), coding.get("fsn"),
                    item.source_text, item.document_id)).casefold():
                continue
            yield item

    def _render(self) -> None:
        events = sorted(self._visible_events(), key=lambda item: (
            item.observed_date or item.document_date or "", item.document_id,
            (item.data.get("source_spans") or [{}])[0].get("start", 0)), reverse=True)
        self._table.setSortingEnabled(False)
        self._table.setRowCount(len(events))
        for row, item in enumerate(events):
            coding = self._coding(item)
            code = coding.get("code")
            snomed = f"{code} — {coding.get('term') or coding.get('fsn') or ''}" if code else "—"
            values = (item.observed_date or "data n.d.", item.normalized_entity,
                      TYPE_LABELS.get(item.fact_type, item.fact_type),
                      ASSERTION_LABELS.get(item.assertion, item.assertion), snomed,
                      STATUS_LABELS.get(item.status, item.status),
                      f"{item.document_id} ({item.document_date or 'n.d.'})")
            for column, value in enumerate(values):
                cell = QTableWidgetItem(str(value or ""))
                if column == 0:
                    cell.setData(Qt.UserRole, item)
                if item.assertion == "absent":
                    cell.setForeground(Qt.gray)
                elif item.status == "needs_review":
                    cell.setForeground(Qt.darkYellow)
                self._table.setItem(row, column, cell)
        self._table.resizeColumnsToContents()
        self._table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)

    def _selected(self):
        row = self._table.currentRow()
        cell = self._table.item(row, 0) if row >= 0 else None
        return cell.data(Qt.UserRole) if cell is not None else None

    def _document_text(self, document_id: str) -> str:
        if document_id not in self._texts:
            pipeline = self._services.get("extraction_pipeline")
            path = pipeline.normalized_text_path(self._patient_id, document_id) if pipeline else None
            text = path.read_text(encoding="utf-8") if path else ""
            overlay = self._services.get("overlay_repo")
            self._texts[document_id] = overlay.effective_text(document_id, text) if (overlay and text) else text
        return self._texts[document_id]

    def _on_selection(self) -> None:
        item = self._selected()
        self._original_btn.setEnabled(item is not None)
        if item is None:
            return
        text = self._document_text(item.document_id)
        coding = self._coding(item)
        span = (item.data.get("source_spans") or [{}])[0]
        stale = bool(text) and item.data.get("source_version") not in (None, content_hash(text))
        provenance = item.data.get("date_provenance") or {}
        lines = [
            f"<b>{item.normalized_entity}</b> · {TYPE_LABELS.get(item.fact_type, item.fact_type)}"
            f" · {ASSERTION_LABELS.get(item.assertion, item.assertion)}, certezza {item.certainty}",
            f"SNOMED CT: {coding.get('code') or 'non codificato'} {coding.get('fsn') or ''}"
            f" ({CODING_LABELS.get(coding.get('status'), 'concetto non ancora codificato')})",
            f"Data evento: {item.observed_date or 'non determinata'}"
            + (f" — da «{provenance.get('quote')}»" if provenance.get("quote") else "")
            + (f" — {provenance['needs_review']}" if provenance.get("needs_review") else ""),
        ]
        if coding.get("note") and not coding.get("code"):
            lines.append(f"Codifica: {coding['note']}")
        if stale:
            lines.append("<span style='color:#c0392b'>Il testo del referto è cambiato dopo "
                         "l'estrazione: rielabora il paziente.</span>")
        self._detail.setText("<br>".join(lines))
        if stale:
            self._source.show_text(text)
        else:
            self._source.show_text(text, span.get("start"), span.get("end"))

    def _open_original(self) -> None:
        item = self._selected()
        documents = self._services.get("document_repo")
        document = documents.get_by_id(item.document_id) if (item and documents) else None
        if document is None:
            QMessageBox.information(self, "Referto", "Documento non disponibile nel progetto.")
            return
        from .pdf_viewer import PDFViewerDialog
        PDFViewerDialog(document.to_dict(), self._services, self, highlight={
            "evidence_id": item.evidence_id, "page_number": item.source_page,
            "bbox": item.bbox, "source_text": item.source_text,
            "normalized_entity": item.normalized_entity}).exec_()

    def _export_fhir(self) -> None:
        pipeline = self._services.get("extraction_pipeline")
        if pipeline is None or not self._patient_id or self._worker_running():
            return
        try:
            result = pipeline.export_patient(self._patient_id)
        except Exception as exc:
            QMessageBox.warning(self, "Esportazione FHIR", f"Esportazione non riuscita: {exc}")
            return
        self._status.setText(f"FHIR aggiornato: {result['events']} risorse cliniche, "
                             f"{result['uncoded']} senza codice · {result['path']}")

    def _show_fhir(self) -> None:
        pipeline = self._services.get("extraction_pipeline")
        path = pipeline.patient_status(self._patient_id).get("fhir_path") if (pipeline and self._patient_id) else None
        from pathlib import Path
        if not path or not Path(path).is_file():
            QMessageBox.information(self, "FHIR", "Elabora prima il paziente per generare il file FHIR.")
            return
        dialog = QDialog(self)
        dialog.setWindowTitle(str(path))
        dialog.resize(900, 700)
        layout = QVBoxLayout(dialog)
        view = QPlainTextEdit()
        view.setReadOnly(True)
        view.setPlainText(Path(path).read_text(encoding="utf-8"))
        layout.addWidget(view)
        dialog.exec_()

    # ---------------------------------------------------------- extraction
    def _worker_running(self) -> bool:
        return self._worker is not None and self._worker.isRunning()

    def _on_run(self) -> None:
        pipeline = self._services.get("extraction_pipeline")
        if not self._patient_id or pipeline is None:
            return
        if self._worker_running():
            return
        if not prepare_pipeline(self._services, "atomic", self):
            return
        llm = self._services.get("atomic_evidence_llm_client")
        if llm is None or not getattr(llm, "is_available", False):
            QMessageBox.warning(self, "Modello non disponibile",
                                "Configura e avvia il modello di estrazione prima di elaborare.")
            return
        from .workers import PatientExtractionWorker
        self._worker = PatientExtractionWorker(
            pipeline, self._patient_id,
            num_workers=max(1, int(getattr(llm, "parallel_workers", 1) or 1)))
        self._worker.progress.connect(self._on_progress)
        self._worker.result_ready.connect(self._on_finished)
        self._worker.cancelled.connect(lambda: self._on_stopped("Elaborazione interrotta."))
        self._worker.error.connect(lambda message: self._on_stopped(f"Errore: {message}", error=True))
        self._worker.finished.connect(self._release_worker)
        self._run_btn.setEnabled(False)
        self._cancel_btn.setVisible(True)
        self._progress.setValue(0)
        self._progress.setVisible(True)
        self._worker.start()

    def _on_progress(self, percent: int, message: str) -> None:
        self._progress.setValue(int(percent))
        self._status.setText(message)

    def _on_finished(self, result: dict) -> None:
        self._on_stopped(
            f"Completato: {result.get('documents_processed', 0)} documenti elaborati, "
            f"{result.get('documents_skipped', 0)} invariati, "
            f"{result.get('documents_failed', 0)} non completati · "
            f"{result.get('fhir_events', 0)} risorse FHIR, {result.get('fhir_uncoded', 0)} senza codice.")

    def _on_stopped(self, message: str, *, error: bool = False) -> None:
        self._run_btn.setEnabled(True)
        self._cancel_btn.setVisible(False)
        self._progress.setVisible(False)
        self._cancel_btn.setEnabled(True)
        self._texts.clear()
        self._refresh()
        self._status.setText(message + "  " + self._status.text())
        if error:
            QMessageBox.warning(self, "Elaborazione non completata", message)

    def _release_worker(self) -> None:
        worker, self._worker = self._worker, None
        if worker is not None:
            worker.deleteLater()

    def _on_cancel(self) -> None:
        if self._worker_running():
            self._worker.cancel()
            self._cancel_btn.setEnabled(False)

    def request_shutdown(self) -> None:
        if self._worker_running():
            self._worker.cancel()
            self._worker.requestInterruption()

    def shutdown(self, wait_ms: int = 1500) -> int:
        self.request_shutdown()
        if self._worker_running():
            self._worker.wait(max(0, int(wait_ms)))
        return int(self._worker_running())

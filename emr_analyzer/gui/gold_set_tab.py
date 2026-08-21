"""Blinded double-annotation and adjudication workspace for clinical gold sets."""

from __future__ import annotations

import json
from pathlib import Path

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from ..config import active_workspace
from ..database.gold_set_repo import manual_evidence_id
from ..evaluation.metrics import match_events
from ..models.clinical_registry import (
    ASSERTION_TYPES,
    CERTAINTY_LEVELS,
    DATE_PRECISIONS,
    EVENT_CATEGORIES,
    EVENT_STATUSES,
    SIGNIFICANCE_LEVELS,
)
from ..models.gold_set import GoldAnnotation


_ROLE_LABELS = {
    "reviewer_a": "Revisore A (cieco)",
    "reviewer_b": "Revisore B (cieco)",
    "adjudicated": "Adjudicatore — riferimento finale",
}


class GoldSetTab(QWidget):
    """Patient-level gold-set editor with prediction blinding."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._services: dict = {}
        self._current_patient_id: str | None = None
        self._case = None
        self._annotations: list[GoldAnnotation] = []
        self._current_annotation: GoldAnnotation | None = None
        self._source_refs: list[dict] = []
        self._documents = []
        self._comparison_rows: list[dict] = []
        self._setup_ui()

    # ------------------------------------------------------------------ UI

    def _setup_ui(self) -> None:
        root = QVBoxLayout(self)
        root.addWidget(self._build_case_box())

        self._workflow_tabs = QTabWidget()
        self._workflow_tabs.addTab(
            self._build_annotation_page(), "1. Annotazione indipendente"
        )
        self._workflow_tabs.addTab(
            self._build_adjudication_page(), "2. Confronto e adjudication"
        )
        root.addWidget(self._workflow_tabs, stretch=1)

    def _build_case_box(self) -> QWidget:
        box = QGroupBox("Protocollo del caso")
        layout = QVBoxLayout(box)

        first = QHBoxLayout()
        self._included = QCheckBox("Includi nel gold set")
        first.addWidget(self._included)
        first.addWidget(QLabel("Split:"))
        self._split = QComboBox()
        self._split.addItem("Pilot", "pilot")
        self._split.addItem("Development", "development")
        self._split.addItem("Test bloccato", "test")
        first.addWidget(self._split)
        self._case_status = QLabel("Nessun paziente")
        first.addWidget(self._case_status)
        first.addStretch()
        self._save_case_btn = QPushButton("Salva assegnazioni")
        self._save_case_btn.clicked.connect(self._save_case)
        first.addWidget(self._save_case_btn)
        layout.addLayout(first)

        second = QHBoxLayout()
        self._reviewer_a = QLineEdit()
        self._reviewer_a.setPlaceholderText("ID revisore A")
        self._reviewer_b = QLineEdit()
        self._reviewer_b.setPlaceholderText("ID revisore B")
        self._adjudicator = QLineEdit()
        self._adjudicator.setPlaceholderText("ID adjudicatore")
        for label, field in (
            ("A:", self._reviewer_a), ("B:", self._reviewer_b),
            ("Adjudicatore:", self._adjudicator),
        ):
            second.addWidget(QLabel(label))
            second.addWidget(field)
        second.addWidget(QLabel("Sessione:"))
        self._role = QComboBox()
        for value in ("reviewer_a", "reviewer_b", "adjudicated"):
            self._role.addItem(_ROLE_LABELS[value], value)
        self._role.currentIndexChanged.connect(self._on_role_changed)
        second.addWidget(self._role)
        layout.addLayout(second)

        self._blind_notice = QLabel(
            "I revisori A e B vedono soltanto i documenti e le proprie "
            "annotazioni. Le predizioni restano nascoste fino alle due consegne."
        )
        self._blind_notice.setWordWrap(True)
        self._blind_notice.setStyleSheet("color:#7f6000;")
        layout.addWidget(self._blind_notice)
        return box

    def _build_annotation_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        splitter = QSplitter(Qt.Horizontal)

        source_panel = QWidget()
        source_layout = QVBoxLayout(source_panel)
        source_layout.addWidget(QLabel("Documenti disponibili (sorgenti immutabili)"))
        self._documents_table = QTableWidget(0, 3)
        self._documents_table.setHorizontalHeaderLabels(["Documento", "Data", "Tipo"])
        self._documents_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.Stretch
        )
        self._documents_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._documents_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._documents_table.itemSelectionChanged.connect(
            self._load_selected_document
        )
        source_layout.addWidget(self._documents_table, stretch=1)

        source_row = QHBoxLayout()
        source_row.addWidget(QLabel("Pagina:"))
        self._source_page = QSpinBox()
        self._source_page.setRange(0, 9999)
        self._source_page.setSpecialValueText("n.d.")
        source_row.addWidget(self._source_page)
        self._add_source_btn = QPushButton("Aggiungi selezione come evidenza")
        self._add_source_btn.clicked.connect(self._add_selected_source)
        source_row.addWidget(self._add_source_btn)
        source_layout.addLayout(source_row)
        self._source_text = QPlainTextEdit()
        self._source_text.setReadOnly(True)
        self._source_text.setPlaceholderText(
            "Seleziona un documento per leggere il testo normalizzato."
        )
        source_layout.addWidget(self._source_text, stretch=3)
        splitter.addWidget(source_panel)

        annotation_panel = QWidget()
        annotation_layout = QVBoxLayout(annotation_panel)
        header = QHBoxLayout()
        header.addWidget(QLabel("Eventi annotati nella sessione corrente"))
        header.addStretch()
        self._new_btn = QPushButton("Nuovo")
        self._new_btn.clicked.connect(self._new_annotation)
        self._delete_btn = QPushButton("Elimina")
        self._delete_btn.clicked.connect(self._delete_annotation)
        self._submit_btn = QPushButton("Consegna annotazione cieca")
        self._submit_btn.setObjectName("successButton")
        self._submit_btn.clicked.connect(self._submit_reviewer)
        header.addWidget(self._new_btn)
        header.addWidget(self._delete_btn)
        header.addWidget(self._submit_btn)
        annotation_layout.addLayout(header)

        self._events_table = QTableWidget(0, 5)
        self._events_table.setHorizontalHeaderLabels(
            ["Prima evidenza", "Categoria", "Entità", "Sintesi", "Fonti"]
        )
        self._events_table.horizontalHeader().setSectionResizeMode(
            3, QHeaderView.Stretch
        )
        self._events_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._events_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._events_table.itemSelectionChanged.connect(
            self._load_selected_annotation
        )
        annotation_layout.addWidget(self._events_table, stretch=1)

        editor_scroll = QScrollArea()
        editor_scroll.setWidgetResizable(True)
        editor = QWidget()
        form = QFormLayout(editor)
        self._category = QComboBox()
        self._category.addItems(EVENT_CATEGORIES)
        self._entity = QLineEdit()
        self._summary = QLineEdit()
        self._detail = QTextEdit()
        self._detail.setMaximumHeight(85)
        form.addRow("Categoria*", self._category)
        form.addRow("Entità normalizzata*", self._entity)
        form.addRow("Nota breve*", self._summary)
        form.addRow("Dettaglio", self._detail)

        date_row = QWidget()
        date_layout = QHBoxLayout(date_row)
        date_layout.setContentsMargins(0, 0, 0, 0)
        self._first_date = QLineEdit()
        self._first_date.setPlaceholderText("AAAA-MM-GG / AAAA-MM / AAAA")
        self._first_documented = QLineEdit()
        self._first_documented.setPlaceholderText("Prima documentazione")
        self._date_end = QLineEdit()
        self._date_end.setPlaceholderText("Fine/risoluzione")
        date_layout.addWidget(self._first_date)
        date_layout.addWidget(self._first_documented)
        date_layout.addWidget(self._date_end)
        form.addRow("Date", date_row)

        state_row = QWidget()
        state_layout = QHBoxLayout(state_row)
        state_layout.setContentsMargins(0, 0, 0, 0)
        self._precision = QComboBox()
        self._precision.addItems(DATE_PRECISIONS)
        self._status = QComboBox()
        self._status.addItems(EVENT_STATUSES)
        self._certainty = QComboBox()
        self._certainty.addItems(CERTAINTY_LEVELS)
        self._assertion = QComboBox()
        self._assertion.addItems(ASSERTION_TYPES)
        for widget in (
            self._precision, self._status, self._certainty, self._assertion
        ):
            state_layout.addWidget(widget)
        form.addRow("Precisione / stato / certezza / asserzione", state_row)

        self._significance = QComboBox()
        self._significance.addItems(SIGNIFICANCE_LEVELS)
        form.addRow("Significatività clinica", self._significance)

        anatomy_row = QWidget()
        anatomy_layout = QHBoxLayout(anatomy_row)
        anatomy_layout.setContentsMargins(0, 0, 0, 0)
        self._site = QLineEdit()
        self._site.setPlaceholderText("Sede")
        self._laterality = QLineEdit()
        self._laterality.setPlaceholderText("Lateralità")
        self._severity = QLineEdit()
        self._severity.setPlaceholderText("Gravità")
        anatomy_layout.addWidget(self._site)
        anatomy_layout.addWidget(self._laterality)
        anatomy_layout.addWidget(self._severity)
        form.addRow("Anatomia e gravità", anatomy_row)

        episode_row = QWidget()
        episode_layout = QHBoxLayout(episode_row)
        episode_layout.setContentsMargins(0, 0, 0, 0)
        self._episode_key = QLineEdit()
        self._episode_key.setPlaceholderText("Chiave episodio, es. DISPNEA_1")
        self._recurrence = QSpinBox()
        self._recurrence.setRange(1, 999)
        episode_layout.addWidget(self._episode_key)
        episode_layout.addWidget(QLabel("Recidiva n."))
        episode_layout.addWidget(self._recurrence)
        form.addRow("Episodio", episode_row)

        self._sources_table = QTableWidget(0, 4)
        self._sources_table.setHorizontalHeaderLabels(
            ["Documento", "Pagina", "ID evidenza", "Passaggio"]
        )
        self._sources_table.horizontalHeader().setSectionResizeMode(
            3, QHeaderView.Stretch
        )
        self._sources_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._sources_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        sources_widget = QWidget()
        sources_layout = QVBoxLayout(sources_widget)
        sources_layout.setContentsMargins(0, 0, 0, 0)
        sources_layout.addWidget(self._sources_table)
        self._remove_source_btn = QPushButton("Rimuovi fonte selezionata")
        self._remove_source_btn.clicked.connect(self._remove_source)
        sources_layout.addWidget(self._remove_source_btn)
        form.addRow("Evidenze citate*", sources_widget)

        self._structured = QPlainTextEdit("{}")
        self._structured.setMaximumHeight(70)
        form.addRow("Dati strutturati JSON", self._structured)
        self._save_annotation_btn = QPushButton("Salva evento annotato")
        self._save_annotation_btn.setObjectName("successButton")
        self._save_annotation_btn.clicked.connect(self._save_annotation)
        form.addRow("", self._save_annotation_btn)
        editor_scroll.setWidget(editor)
        annotation_layout.addWidget(editor_scroll, stretch=3)
        splitter.addWidget(annotation_panel)
        splitter.setSizes([500, 850])
        layout.addWidget(splitter)
        return page

    def _build_adjudication_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        self._adjudication_notice = QLabel(
            "Il confronto sarà disponibile dopo la consegna indipendente "
            "di entrambi i revisori."
        )
        self._adjudication_notice.setWordWrap(True)
        layout.addWidget(self._adjudication_notice)

        self._comparison_table = QTableWidget(0, 6)
        self._comparison_table.setHorizontalHeaderLabels([
            "Accordo", "Revisore A", "Revisore B", "Registro automatico",
            "Decisione", "Evento finale",
        ])
        for column in (1, 2, 3, 5):
            self._comparison_table.horizontalHeader().setSectionResizeMode(
                column, QHeaderView.Stretch
            )
        self._comparison_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._comparison_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        layout.addWidget(self._comparison_table, stretch=3)

        actions = QHBoxLayout()
        self._adopt_a_btn = QPushButton("Adotta A e sintetizza")
        self._adopt_a_btn.clicked.connect(lambda: self._adopt_selected("a"))
        self._adopt_b_btn = QPushButton("Adotta B e sintetizza")
        self._adopt_b_btn.clicked.connect(lambda: self._adopt_selected("b"))
        self._exclude_btn = QPushButton("Escludi dal gold")
        self._exclude_btn.clicked.connect(self._exclude_selected)
        self._reopen_a_btn = QPushButton("Riapri A")
        self._reopen_a_btn.clicked.connect(lambda: self._reopen("reviewer_a"))
        self._reopen_b_btn = QPushButton("Riapri B")
        self._reopen_b_btn.clicked.connect(lambda: self._reopen("reviewer_b"))
        self._lock_btn = QPushButton("🔒 Blocca riferimento finale")
        self._lock_btn.setObjectName("successButton")
        self._lock_btn.clicked.connect(self._lock_case)
        for widget in (
            self._adopt_a_btn, self._adopt_b_btn, self._exclude_btn,
            self._reopen_a_btn, self._reopen_b_btn, self._lock_btn,
        ):
            actions.addWidget(widget)
        layout.addLayout(actions)

        bottom = QHBoxLayout()
        self._metrics_btn = QPushButton("Calcola metriche")
        self._metrics_btn.clicked.connect(self._calculate_metrics)
        self._export_current_btn = QPushButton("Esporta questo caso JSONL")
        self._export_current_btn.clicked.connect(self._export_current)
        self._export_project_btn = QPushButton("Esporta progetto bloccato JSONL")
        self._export_project_btn.clicked.connect(self._export_project)
        bottom.addWidget(self._metrics_btn)
        bottom.addWidget(self._export_current_btn)
        bottom.addWidget(self._export_project_btn)
        bottom.addStretch()
        layout.addLayout(bottom)
        self._metrics_text = QPlainTextEdit()
        self._metrics_text.setReadOnly(True)
        self._metrics_text.setMaximumHeight(150)
        layout.addWidget(self._metrics_text)
        return page

    # --------------------------------------------------------------- loading

    def set_services(self, services: dict) -> None:
        self._services = services
        if self._current_patient_id:
            self.load_patient(self._current_patient_id)

    def load_patient(self, patient_id: str) -> None:
        self._current_patient_id = patient_id or None
        self._current_annotation = None
        self._source_refs = []
        repository = self._services.get("gold_set_repo")
        if not patient_id or repository is None:
            self._case = None
            self._events_table.setRowCount(0)
            self._documents_table.setRowCount(0)
            self._update_controls()
            return
        self._case = repository.ensure_case(patient_id)
        self._populate_case()
        self._load_documents()
        self._refresh_annotations()
        self._refresh_adjudication()

    def _populate_case(self) -> None:
        case = self._case
        if not case:
            return
        self._included.setChecked(case.included)
        index = self._split.findData(case.split)
        self._split.setCurrentIndex(max(0, index))
        self._reviewer_a.setText(case.reviewer_a_id)
        self._reviewer_b.setText(case.reviewer_b_id)
        self._adjudicator.setText(case.adjudicator_id)
        self._case_status.setText(
            f"Stato: {case.status} · A: {case.reviewer_a_status} · "
            f"B: {case.reviewer_b_status}"
        )
        self._update_controls()

    def _load_documents(self) -> None:
        repository = self._services.get("document_repo")
        self._documents = (
            repository.list_by_patient(self._current_patient_id)
            if repository and self._current_patient_id else []
        )
        self._documents_table.setRowCount(len(self._documents))
        for row, document in enumerate(self._documents):
            values = (
                document.id, document.document_date or "",
                document.document_type or "",
            )
            for column, value in enumerate(values):
                item = QTableWidgetItem(str(value))
                if column == 0:
                    item.setData(Qt.UserRole, document.id)
                self._documents_table.setItem(row, column, item)

    def _load_selected_document(self) -> None:
        row = self._documents_table.currentRow()
        if row < 0 or not self._current_patient_id:
            return
        item = self._documents_table.item(row, 0)
        document_id = str(item.data(Qt.UserRole) or "") if item else ""
        candidates = (
            active_workspace.path / self._current_patient_id / "extraction"
            / f"{document_id}.md",
            active_workspace.path / self._current_patient_id / "docling"
            / f"{document_id}.md",
        )
        path = next((candidate for candidate in candidates if candidate.exists()), None)
        if path is None:
            self._source_text.setPlainText("Testo normalizzato non disponibile.")
            return
        try:
            base_text = path.read_text(encoding="utf-8")
            overlay = self._services.get("overlay_repo")
            text = overlay.effective_text(document_id, base_text) if overlay else base_text
            self._source_text.setPlainText(text)
        except OSError as exc:
            self._source_text.setPlainText(f"Errore lettura: {exc}")

    # ------------------------------------------------------------- annotation

    def _on_role_changed(self, *_args) -> None:
        slot = self._role.currentData()
        if slot == "adjudicated" and self._case and not self._repository().can_adjudicate(
            self._case
        ):
            self._role.blockSignals(True)
            self._role.setCurrentIndex(0)
            self._role.blockSignals(False)
            slot = "reviewer_a"
        self._current_annotation = None
        self._clear_form()
        self._refresh_annotations()
        self._update_controls()

    def _refresh_annotations(self) -> None:
        if not self._current_patient_id or not self._services.get("gold_set_repo"):
            return
        slot = str(self._role.currentData())
        self._annotations = self._repository().list_annotations(
            self._current_patient_id, slot
        )
        self._events_table.setRowCount(len(self._annotations))
        for row, annotation in enumerate(self._annotations):
            values = (
                annotation.first_evidence_date or "n.d.", annotation.category,
                annotation.canonical_entity, annotation.summary_short,
                str(len(annotation.source_refs)),
            )
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                if column == 0:
                    item.setData(Qt.UserRole, annotation.annotation_id)
                self._events_table.setItem(row, column, item)
        self._update_controls()

    def _load_selected_annotation(self) -> None:
        row = self._events_table.currentRow()
        if row < 0:
            return
        item = self._events_table.item(row, 0)
        annotation_id = item.data(Qt.UserRole) if item else None
        annotation = next(
            (value for value in self._annotations
             if value.annotation_id == annotation_id), None
        )
        if annotation is None:
            return
        self._current_annotation = annotation
        self._category.setCurrentText(annotation.category)
        self._entity.setText(annotation.canonical_entity)
        self._summary.setText(annotation.summary_short)
        self._detail.setPlainText(annotation.summary_detail)
        self._first_date.setText(annotation.first_evidence_date or "")
        self._first_documented.setText(annotation.first_documented_date or "")
        self._date_end.setText(annotation.date_end or "")
        self._precision.setCurrentText(annotation.date_precision)
        self._status.setCurrentText(annotation.status)
        self._certainty.setCurrentText(annotation.certainty)
        self._assertion.setCurrentText(annotation.assertion)
        self._significance.setCurrentText(annotation.significance)
        self._site.setText(annotation.anatomical_site or "")
        self._laterality.setText(annotation.laterality or "")
        self._severity.setText(annotation.severity or "")
        self._episode_key.setText(annotation.episode_key or "")
        self._recurrence.setValue(annotation.recurrence_index)
        self._structured.setPlainText(
            json.dumps(annotation.structured_data, ensure_ascii=False, indent=2)
        )
        self._source_refs = [dict(value) for value in annotation.source_refs]
        self._render_sources()

    def _new_annotation(self) -> None:
        if self._role.currentData() == "adjudicated":
            QMessageBox.information(
                self, "Usa il confronto",
                "Crea l'evento finale adottando una riga A/B, quindi modificalo.",
            )
            return
        self._events_table.clearSelection()
        self._current_annotation = None
        self._clear_form()

    def _clear_form(self) -> None:
        self._category.setCurrentText("diagnosis")
        for field in (
            self._entity, self._summary, self._first_date,
            self._first_documented, self._date_end, self._site,
            self._laterality, self._severity, self._episode_key,
        ):
            field.clear()
        self._detail.clear()
        self._precision.setCurrentText("unknown")
        self._status.setCurrentText("active")
        self._certainty.setCurrentText("confirmed")
        self._assertion.setCurrentText("present")
        self._significance.setCurrentText("clinically_relevant")
        self._recurrence.setValue(1)
        self._structured.setPlainText("{}")
        self._source_refs = []
        self._render_sources()

    def _add_selected_source(self) -> None:
        row = self._documents_table.currentRow()
        if row < 0:
            QMessageBox.information(self, "Documento richiesto", "Seleziona un documento.")
            return
        document_item = self._documents_table.item(row, 0)
        document_id = str(document_item.data(Qt.UserRole) or "")
        quote = self._source_text.textCursor().selectedText().replace("\u2029", "\n")
        quote = " ".join(quote.split())
        if not quote:
            quote, accepted = QInputDialog.getMultiLineText(
                self, "Passaggio sorgente",
                "Seleziona il testo nel documento oppure inserisci il passaggio:"
            )
            if not accepted:
                return
            quote = " ".join(quote.split())
        if not quote:
            QMessageBox.warning(self, "Fonte non valida", "Il passaggio non può essere vuoto.")
            return
        page = self._source_page.value() or None
        evidence_id = manual_evidence_id(
            document_id, page, quote,
            category=self._category.currentText(),
            entity=self._entity.text(), date=self._first_date.text(),
        )
        reference = {
            "evidence_id": evidence_id,
            "document_id": document_id,
            "page": page,
            "quote": quote,
            "relation": "supports",
        }
        if evidence_id not in {value.get("evidence_id") for value in self._source_refs}:
            self._source_refs.append(reference)
        self._render_sources()

    def _render_sources(self) -> None:
        self._sources_table.setRowCount(len(self._source_refs))
        for row, reference in enumerate(self._source_refs):
            values = (
                reference.get("document_id") or "",
                str(reference.get("page") or "n.d."),
                reference.get("evidence_id") or "",
                reference.get("quote") or "",
            )
            for column, value in enumerate(values):
                self._sources_table.setItem(row, column, QTableWidgetItem(value))

    def _remove_source(self) -> None:
        row = self._sources_table.currentRow()
        if 0 <= row < len(self._source_refs):
            self._source_refs.pop(row)
            self._render_sources()

    def _save_annotation(self) -> None:
        if not self._current_patient_id:
            return
        try:
            structured = json.loads(self._structured.toPlainText() or "{}")
            if not isinstance(structured, dict):
                raise ValueError("I dati strutturati devono essere un oggetto JSON")
            slot = str(self._role.currentData())
            existing = self._current_annotation
            annotation = GoldAnnotation(
                annotation_id=(existing.annotation_id if existing else None)
                or GoldAnnotation(
                    patient_id=self._current_patient_id,
                    reviewer_slot=slot, reviewer_id="", category="other",
                    canonical_entity="x", summary_short="x",
                ).annotation_id,
                patient_id=self._current_patient_id,
                reviewer_slot=slot,
                reviewer_id=self._reviewer_id(slot),
                category=self._category.currentText(),
                canonical_entity=self._entity.text().strip(),
                summary_short=self._summary.text().strip(),
                summary_detail=self._detail.toPlainText().strip(),
                first_evidence_date=self._optional(self._first_date.text()),
                first_documented_date=self._optional(
                    self._first_documented.text()
                ),
                date_end=self._optional(self._date_end.text()),
                date_precision=self._precision.currentText(),
                status=self._status.currentText(),
                certainty=self._certainty.currentText(),
                assertion=self._assertion.currentText(),
                significance=self._significance.currentText(),
                anatomical_site=self._optional(self._site.text()),
                laterality=self._optional(self._laterality.text()),
                severity=self._optional(self._severity.text()),
                episode_key=self._optional(self._episode_key.text()),
                recurrence_index=self._recurrence.value(),
                evidence_ids=[
                    str(value.get("evidence_id")) for value in self._source_refs
                    if value.get("evidence_id")
                ],
                source_refs=[dict(value) for value in self._source_refs],
                structured_data=structured,
                source_annotation_ids=(
                    list(existing.source_annotation_ids) if existing else []
                ),
                created_at=existing.created_at if existing else "",
            )
            self._repository().save_annotation(annotation)
        except Exception as exc:
            QMessageBox.warning(self, "Annotazione non salvata", str(exc))
            return
        self._current_annotation = annotation
        self._refresh_annotations()
        self._refresh_adjudication()

    def _delete_annotation(self) -> None:
        if not self._current_annotation:
            return
        if QMessageBox.question(
            self, "Elimina annotazione", "Eliminare l'evento selezionato?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
        ) != QMessageBox.Yes:
            return
        try:
            self._repository().delete_annotation(
                self._current_annotation.annotation_id,
                actor_id=self._reviewer_id(str(self._role.currentData())),
            )
        except Exception as exc:
            QMessageBox.warning(self, "Eliminazione non riuscita", str(exc))
            return
        self._current_annotation = None
        self._clear_form()
        self._refresh_annotations()
        self._refresh_adjudication()

    def _submit_reviewer(self) -> None:
        slot = str(self._role.currentData())
        if slot == "adjudicated":
            return
        if QMessageBox.question(
            self, "Consegna annotazione",
            "Dopo la consegna questa annotazione resterà bloccata finché "
            "l'adjudicatore non la riapre. Confermi?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
        ) != QMessageBox.Yes:
            return
        try:
            self._case = self._repository().submit_reviewer(
                self._current_patient_id, slot, self._reviewer_id(slot)
            )
        except Exception as exc:
            QMessageBox.warning(self, "Consegna non riuscita", str(exc))
            return
        self._populate_case()
        self._refresh_adjudication()

    # ------------------------------------------------------------- case flow

    def _save_case(self) -> None:
        if not self._current_patient_id:
            return
        try:
            self._case = self._repository().configure_case(
                self._current_patient_id,
                included=self._included.isChecked(),
                split=str(self._split.currentData()),
                reviewer_a_id=self._reviewer_a.text(),
                reviewer_b_id=self._reviewer_b.text(),
                adjudicator_id=self._adjudicator.text(),
            )
        except Exception as exc:
            QMessageBox.warning(self, "Configurazione non salvata", str(exc))
            return
        self._populate_case()
        self._refresh_adjudication()

    def _refresh_adjudication(self) -> None:
        if not self._case or not self._current_patient_id:
            self._workflow_tabs.setTabEnabled(1, False)
            return
        can_adjudicate = self._repository().can_adjudicate(self._case)
        self._workflow_tabs.setTabEnabled(1, can_adjudicate)
        adjudicated_index = self._role.findData("adjudicated")
        self._role.model().item(adjudicated_index).setEnabled(can_adjudicate)
        if not can_adjudicate:
            self._comparison_table.setRowCount(0)
            self._adjudication_notice.setText(
                "Il confronto è nascosto: attendere la consegna indipendente "
                "di entrambi i revisori."
            )
            self._update_controls()
            return

        reviewer_a = self._repository().list_annotations(
            self._current_patient_id, "reviewer_a"
        )
        reviewer_b = self._repository().list_annotations(
            self._current_patient_id, "reviewer_b"
        )
        a_dicts = [item.to_evaluation_dict() for item in reviewer_a]
        b_dicts = [item.to_evaluation_dict() for item in reviewer_b]
        matches, unmatched_a, unmatched_b = match_events(a_dicts, b_dicts)
        rows = [
            {"a": reviewer_a[a], "b": reviewer_b[b], "score": score}
            for a, b, score in matches
        ]
        rows.extend(
            {"a": reviewer_a[index], "b": None, "score": 0.0}
            for index in unmatched_a
        )
        rows.extend(
            {"a": None, "b": reviewer_b[index], "score": 0.0}
            for index in unmatched_b
        )
        predicted = self._repository().evaluation_record(
            self._current_patient_id
        )["predicted_events"]
        representatives = [
            (row["a"] or row["b"]).to_evaluation_dict() for row in rows
        ]
        prediction_map = {}
        if representatives and predicted:
            pred_matches, _, _ = match_events(representatives, predicted)
            prediction_map = {
                representative_index: predicted[predicted_index]
                for representative_index, predicted_index, _ in pred_matches
            }
        decisions = {
            row["source_annotation_id"]: row
            for row in self._repository().list_adjudication_decisions(
                self._current_patient_id
            )
        }
        finals = {
            item.annotation_id: item
            for item in self._repository().list_annotations(
                self._current_patient_id, "adjudicated"
            )
        }
        for index, row in enumerate(rows):
            row["predicted"] = prediction_map.get(index)
            source_ids = [
                item.annotation_id for item in (row["a"], row["b"]) if item
            ]
            source_decisions = [decisions.get(value) for value in source_ids]
            if source_decisions and all(source_decisions):
                if all(value["decision"] == "excluded" for value in source_decisions):
                    row["decision"] = "escluso"
                    row["final"] = None
                else:
                    final_ids = {
                        value["final_annotation_id"] for value in source_decisions
                        if value["final_annotation_id"]
                    }
                    row["decision"] = "incluso" if len(final_ids) == 1 else "parziale"
                    row["final"] = finals.get(next(iter(final_ids))) if len(final_ids) == 1 else None
            else:
                row["decision"] = "da decidere"
                row["final"] = None
        self._comparison_rows = rows
        self._render_comparison()
        agreement = self._repository().reviewer_agreement(self._current_patient_id)
        self._adjudication_notice.setText(
            f"Confronto sbloccato · accordo eventi F1={agreement['f1']:.3f} · "
            f"A={agreement['gold_count']} eventi, B={agreement['predicted_count']} eventi."
        )
        self._update_controls()

    def _render_comparison(self) -> None:
        self._comparison_table.setRowCount(len(self._comparison_rows))
        for row_index, row in enumerate(self._comparison_rows):
            a = row.get("a")
            b = row.get("b")
            predicted = row.get("predicted") or {}
            final = row.get("final")
            values = (
                f"{row.get('score', 0.0):.3f}",
                self._annotation_label(a), self._annotation_label(b),
                self._event_label(predicted), row.get("decision", ""),
                self._annotation_label(final),
            )
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                if column == 0:
                    item.setData(Qt.UserRole, row_index)
                self._comparison_table.setItem(row_index, column, item)

    def _adopt_selected(self, side: str) -> None:
        row = self._selected_comparison()
        if row is None:
            return
        chosen = row.get(side)
        if chosen is None:
            QMessageBox.information(self, "Annotazione assente", f"Il revisore {side.upper()} non ha questo evento.")
            return
        related = [
            item.annotation_id for key, item in (("a", row.get("a")), ("b", row.get("b")))
            if item is not None and key != side
        ]
        adjudicator = self._adjudicator.text().strip()
        if not adjudicator:
            QMessageBox.warning(self, "Adjudicatore richiesto", "Inserisci l'ID dell'adjudicatore.")
            return
        try:
            final = self._repository().copy_to_adjudicated(
                chosen.annotation_id, adjudicator,
                related_annotation_ids=related,
                rationale=f"Sintesi adottata dal revisore {side.upper()}",
            )
        except Exception as exc:
            QMessageBox.warning(self, "Adjudication non salvata", str(exc))
            return
        self._role.setCurrentIndex(self._role.findData("adjudicated"))
        self._refresh_annotations()
        self._refresh_adjudication()
        for index, annotation in enumerate(self._annotations):
            if annotation.annotation_id == final.annotation_id:
                self._events_table.selectRow(index)
                break

    def _exclude_selected(self) -> None:
        row = self._selected_comparison()
        if row is None:
            return
        source_ids = [
            item.annotation_id for item in (row.get("a"), row.get("b")) if item
        ]
        rationale, accepted = QInputDialog.getMultiLineText(
            self, "Motivazione esclusione",
            "Perché queste annotazioni non appartengono al riferimento finale?"
        )
        if not accepted:
            return
        try:
            self._repository().exclude_from_gold(
                self._current_patient_id, source_ids,
                self._adjudicator.text(), rationale,
            )
        except Exception as exc:
            QMessageBox.warning(self, "Esclusione non salvata", str(exc))
            return
        self._refresh_adjudication()

    def _reopen(self, slot: str) -> None:
        try:
            self._case = self._repository().reopen_reviewer(
                self._current_patient_id, slot, self._adjudicator.text()
            )
        except Exception as exc:
            QMessageBox.warning(self, "Riapertura non riuscita", str(exc))
            return
        self._role.setCurrentIndex(self._role.findData(slot))
        self._populate_case()
        self._refresh_annotations()
        self._refresh_adjudication()

    def _lock_case(self) -> None:
        if QMessageBox.question(
            self, "Blocca riferimento",
            "Dopo il blocco non saranno più consentite modifiche. Confermi?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
        ) != QMessageBox.Yes:
            return
        try:
            self._case = self._repository().lock_case(
                self._current_patient_id, self._adjudicator.text()
            )
        except Exception as exc:
            QMessageBox.warning(self, "Blocco non riuscito", str(exc))
            return
        self._populate_case()
        self._refresh_adjudication()

    # ------------------------------------------------------------- evaluation

    def _calculate_metrics(self) -> None:
        try:
            agreement = self._repository().reviewer_agreement(
                self._current_patient_id
            )
            model = self._repository().evaluate_patient(self._current_patient_id)
        except Exception as exc:
            QMessageBox.warning(self, "Metriche non disponibili", str(exc))
            return
        self._metrics_text.setPlainText(
            "ACCORDO A↔B\n"
            f"F1 {agreement['f1']:.4f} · precisione {agreement['precision']:.4f} · "
            f"richiamo {agreement['recall']:.4f}\n\n"
            "REGISTRO AUTOMATICO ↔ GOLD ADJUDICATO\n"
            f"F1 {model['f1']:.4f} · precisione {model['precision']:.4f} · "
            f"richiamo {model['recall']:.4f}\n"
            f"data esatta {model['temporal_exact_accuracy']:.4f} · "
            f"data compatibile {model['temporal_compatible_accuracy']:.4f} · "
            f"citazioni complete {model['citation_completeness']:.4f}"
        )

    def _export_current(self) -> None:
        self._export([self._current_patient_id])

    def _export_project(self) -> None:
        self._export(None)

    def _export(self, patient_ids) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Esporta gold set clinico", "gold_set_clinico.jsonl",
            "JSON Lines (*.jsonl)"
        )
        if not path:
            return
        if not path.lower().endswith(".jsonl"):
            path += ".jsonl"
        try:
            count = self._repository().export_jsonl(
                Path(path), patient_ids=patient_ids, locked_only=True
            )
        except Exception as exc:
            QMessageBox.critical(self, "Esportazione non riuscita", str(exc))
            return
        QMessageBox.information(
            self, "Esportazione completata",
            f"Esportati {count} casi bloccati in:\n{path}",
        )

    # --------------------------------------------------------------- helpers

    def _update_controls(self) -> None:
        case = self._case
        has_case = case is not None
        locked = bool(case and case.status == "locked")
        included = bool(case and case.included)
        slot = str(self._role.currentData())
        submitted = bool(
            case and (
                (slot == "reviewer_a" and case.reviewer_a_status == "submitted")
                or (slot == "reviewer_b" and case.reviewer_b_status == "submitted")
            )
        )
        adjudication = bool(
            case and self._repository().can_adjudicate(case)
        ) if self._services.get("gold_set_repo") else False
        editable = has_case and included and not locked and not submitted
        if slot == "adjudicated":
            editable = editable and adjudication
        self._save_case_btn.setEnabled(has_case and not locked)
        for widget in (
            self._included, self._split, self._reviewer_a,
            self._reviewer_b, self._adjudicator,
        ):
            widget.setEnabled(has_case and not locked)
        self._new_btn.setEnabled(editable and slot != "adjudicated")
        self._save_annotation_btn.setEnabled(editable)
        self._delete_btn.setEnabled(editable and self._current_annotation is not None)
        self._add_source_btn.setEnabled(editable)
        self._remove_source_btn.setEnabled(editable)
        self._submit_btn.setVisible(slot in {"reviewer_a", "reviewer_b"})
        self._submit_btn.setEnabled(editable and bool(self._annotations))
        for widget in (
            self._adopt_a_btn, self._adopt_b_btn, self._exclude_btn,
            self._reopen_a_btn, self._reopen_b_btn,
        ):
            widget.setEnabled(adjudication and not locked)
        self._lock_btn.setEnabled(adjudication and not locked)
        self._export_current_btn.setEnabled(bool(case and locked))
        self._metrics_btn.setEnabled(adjudication)

    def _repository(self):
        repository = self._services.get("gold_set_repo")
        if repository is None:
            raise RuntimeError("Repository gold set non disponibile")
        return repository

    def _reviewer_id(self, slot: str) -> str:
        if slot == "reviewer_a":
            return self._reviewer_a.text().strip()
        if slot == "reviewer_b":
            return self._reviewer_b.text().strip()
        return self._adjudicator.text().strip()

    def _selected_comparison(self) -> dict | None:
        row = self._comparison_table.currentRow()
        if row < 0:
            QMessageBox.information(self, "Selezione richiesta", "Seleziona una riga del confronto.")
            return None
        item = self._comparison_table.item(row, 0)
        index = item.data(Qt.UserRole) if item else None
        if not isinstance(index, int) or not 0 <= index < len(self._comparison_rows):
            return None
        return self._comparison_rows[index]

    @staticmethod
    def _optional(value: str) -> str | None:
        value = str(value or "").strip()
        return value or None

    @staticmethod
    def _annotation_label(annotation: GoldAnnotation | None) -> str:
        if annotation is None:
            return "—"
        return (
            f"{annotation.first_evidence_date or 'data n.d.'} · "
            f"{annotation.category} · {annotation.summary_short}"
        )

    @staticmethod
    def _event_label(event: dict) -> str:
        if not event:
            return "—"
        return (
            f"{event.get('first_evidence_date') or 'data n.d.'} · "
            f"{event.get('category') or 'other'} · "
            f"{event.get('summary_short') or event.get('canonical_entity') or ''}"
        )

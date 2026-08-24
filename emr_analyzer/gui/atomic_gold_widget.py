"""Source-first, prediction-blinded editor for atomic-evidence gold labels."""

from __future__ import annotations

import json
from datetime import datetime, timezone
import uuid

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QFormLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..config import active_workspace
from ..models.clinical_pipeline import ATOMIC_FACT_TYPES, EVIDENCE_DISPOSITIONS
from ..models.clinical_registry import DATE_PRECISIONS
from ..models.gold_set import GoldAtomicAnnotation


class AtomicGoldWidget(QWidget):
    """Annotate atoms directly from normalized documents without predictions."""

    def __init__(self, role_getter, parent=None):
        super().__init__(parent)
        self._role_getter = role_getter
        self._services = {}
        self._patient_id = None
        self._documents = []
        self._annotations = []
        self._current = None
        self._document_id = None
        self._setup_ui()

    def _setup_ui(self):
        root = QVBoxLayout(self)
        notice = QLabel(
            "Annotazione cieca delle evidenze atomiche: seleziona il passaggio "
            "dal testo sorgente. Le predizioni automatiche non sono mostrate."
        )
        notice.setWordWrap(True)
        root.addWidget(notice)
        splitter = QSplitter(Qt.Horizontal)

        source = QWidget()
        source_layout = QVBoxLayout(source)
        self._documents_table = QTableWidget(0, 3)
        self._documents_table.setHorizontalHeaderLabels(
            ["Documento", "Data", "Tipo"]
        )
        self._documents_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.Stretch
        )
        self._documents_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._documents_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._documents_table.itemSelectionChanged.connect(self._load_document)
        source_layout.addWidget(self._documents_table, 1)
        page_row = QHBoxLayout()
        page_row.addWidget(QLabel("Pagina:"))
        self._page = QSpinBox()
        self._page.setRange(0, 9999)
        self._page.setSpecialValueText("n.d.")
        page_row.addWidget(self._page)
        self._use_selection = QPushButton("Usa selezione come passaggio")
        self._use_selection.clicked.connect(self._copy_selection)
        page_row.addWidget(self._use_selection)
        source_layout.addLayout(page_row)
        self._source_document = QPlainTextEdit()
        self._source_document.setReadOnly(True)
        source_layout.addWidget(self._source_document, 3)
        splitter.addWidget(source)

        editor = QWidget()
        editor_layout = QVBoxLayout(editor)
        toolbar = QHBoxLayout()
        self._new = QPushButton("Nuova")
        self._new.clicked.connect(self._clear)
        self._delete = QPushButton("Elimina")
        self._delete.clicked.connect(self._delete_annotation)
        toolbar.addWidget(QLabel("Evidenze annotate"))
        toolbar.addStretch()
        toolbar.addWidget(self._new)
        toolbar.addWidget(self._delete)
        editor_layout.addLayout(toolbar)
        self._table = QTableWidget(0, 5)
        self._table.setHorizontalHeaderLabels(
            ["Documento", "Tipo", "Concetto", "Disposizione", "Passaggio"]
        )
        self._table.horizontalHeader().setSectionResizeMode(4, QHeaderView.Stretch)
        self._table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._table.itemSelectionChanged.connect(self._load_annotation)
        editor_layout.addWidget(self._table, 1)

        form = QFormLayout()
        self._kind = QComboBox()
        for label, value in (
            ("Evidenza clinica", "evidence"),
            ("Esclusione", "exclusion"),
            ("Duplicato", "duplicate"),
            ("Invalida/non verificabile", "invalid"),
        ):
            self._kind.addItem(label, value)
        self._fact_type = QComboBox()
        self._fact_type.addItem("—", None)
        self._fact_type.addItems(ATOMIC_FACT_TYPES)
        self._concept = QLineEdit()
        self._canonical = QLineEdit()
        self._observation_date = QLineEdit()
        self._observation_date.setPlaceholderText("AAAA-MM-GG / AAAA-MM / AAAA")
        self._precision = QComboBox()
        self._precision.addItems(DATE_PRECISIONS)
        self._polarity = QComboBox()
        self._polarity.addItem("—", None)
        for value in ("present", "negated", "suspected"):
            self._polarity.addItem(value, value)
        self._disposition = QComboBox()
        self._disposition.addItems(EVIDENCE_DISPOSITIONS)
        self._reason = QLineEdit()
        self._duplicate_of = QLineEdit()
        self._duplicate_of.setPlaceholderText("ID annotazione canonica")
        self._passage = QPlainTextEdit()
        self._passage.setMaximumHeight(90)
        self._value = QPlainTextEdit("{}")
        self._value.setMaximumHeight(65)
        form.addRow("Tipo annotazione", self._kind)
        form.addRow("Tipo di fatto", self._fact_type)
        form.addRow("Concetto originale", self._concept)
        form.addRow("Etichetta canonica", self._canonical)
        form.addRow("Data osservazione", self._observation_date)
        form.addRow("Precisione", self._precision)
        form.addRow("Polarità", self._polarity)
        form.addRow("Disposizione", self._disposition)
        form.addRow("Motivo esclusione", self._reason)
        form.addRow("Duplicato di", self._duplicate_of)
        form.addRow("Passaggio sorgente*", self._passage)
        form.addRow("Valore/payload JSON", self._value)
        editor_layout.addLayout(form)
        self._save = QPushButton("Salva evidenza atomica")
        self._save.setObjectName("successButton")
        self._save.clicked.connect(self._save_annotation)
        editor_layout.addWidget(self._save)
        splitter.addWidget(editor)
        splitter.setSizes([560, 780])
        root.addWidget(splitter, 1)

    def set_services(self, services):
        self._services = services
        self.refresh()

    def load_patient(self, patient_id):
        self._patient_id = patient_id or None
        self._current = None
        self._document_id = None
        self._load_documents()
        self.refresh()

    def role_changed(self):
        self._current = None
        self.refresh()

    def _load_documents(self):
        repository = self._services.get("document_repo")
        self._documents = (
            repository.list_by_patient(self._patient_id)
            if repository and self._patient_id else []
        )
        self._documents_table.setRowCount(len(self._documents))
        for row, document in enumerate(self._documents):
            for column, value in enumerate((
                document.id, document.document_date or "",
                document.document_type or "",
            )):
                item = QTableWidgetItem(str(value))
                if column == 0:
                    item.setData(Qt.UserRole, document.id)
                self._documents_table.setItem(row, column, item)

    def _load_document(self):
        row = self._documents_table.currentRow()
        if row < 0 or not self._patient_id:
            return
        item = self._documents_table.item(row, 0)
        self._document_id = str(item.data(Qt.UserRole) or "") if item else None
        candidates = (
            active_workspace.path / self._patient_id / "extraction"
            / f"{self._document_id}.md",
            active_workspace.path / self._patient_id / "docling"
            / f"{self._document_id}.md",
        )
        path = next((candidate for candidate in candidates if candidate.exists()), None)
        if path is None:
            self._source_document.setPlainText("Testo normalizzato non disponibile.")
            return
        try:
            text = path.read_text(encoding="utf-8")
            overlay = self._services.get("overlay_repo")
            if overlay:
                text = overlay.effective_text(self._document_id, text)
            self._source_document.setPlainText(text)
        except OSError as exc:
            self._source_document.setPlainText(f"Errore lettura: {exc}")

    def _copy_selection(self):
        selected = self._source_document.textCursor().selectedText()
        selected = " ".join(selected.replace("\u2029", "\n").split())
        if not selected:
            QMessageBox.information(
                self, "Nessuna selezione", "Seleziona prima il passaggio nel testo."
            )
            return
        self._passage.setPlainText(selected)

    def refresh(self):
        repository = self._services.get("gold_set_repo")
        slot = self._role_getter()
        if not repository or not self._patient_id or not slot:
            self._annotations = []
        else:
            self._annotations = repository.list_atomic_annotations(
                self._patient_id, slot
            )
        self._table.setRowCount(len(self._annotations))
        for row, annotation in enumerate(self._annotations):
            values = (
                annotation.document_id, annotation.fact_type or annotation.annotation_kind,
                annotation.concept_original or "", annotation.disposition,
                annotation.source_text,
            )
            for column, value in enumerate(values):
                item = QTableWidgetItem(str(value))
                if column == 0:
                    item.setData(Qt.UserRole, annotation.annotation_id)
                self._table.setItem(row, column, item)

    def _load_annotation(self):
        row = self._table.currentRow()
        if row < 0 or row >= len(self._annotations):
            return
        annotation = self._annotations[row]
        self._current = annotation
        self._document_id = annotation.document_id
        self._set_data(self._kind, annotation.annotation_kind)
        self._set_data(self._fact_type, annotation.fact_type)
        self._concept.setText(annotation.concept_original or "")
        self._canonical.setText(annotation.canonical_label or "")
        self._observation_date.setText(annotation.observation_date or "")
        self._precision.setCurrentText(annotation.date_precision)
        self._set_data(self._polarity, annotation.polarity)
        self._disposition.setCurrentText(annotation.disposition)
        self._reason.setText(annotation.exclusion_reason or "")
        self._duplicate_of.setText(annotation.duplicate_of_annotation_id or "")
        self._page.setValue(annotation.source_page or 0)
        self._passage.setPlainText(annotation.source_text)
        self._value.setPlainText(json.dumps(
            annotation.value, ensure_ascii=False, indent=2, sort_keys=True
        ))

    @staticmethod
    def _set_data(combo, value):
        index = combo.findData(value)
        combo.setCurrentIndex(max(0, index))

    def _reviewer_id(self, slot):
        repository = self._services.get("gold_set_repo")
        case = repository.get_case(self._patient_id) if repository else None
        if not case:
            return ""
        field = "adjudicator_id" if slot == "adjudicated" else f"{slot}_id"
        return str(getattr(case, field, "") or "")

    def _save_annotation(self):
        repository = self._services.get("gold_set_repo")
        slot = self._role_getter()
        if not repository or not self._patient_id or not self._document_id:
            QMessageBox.warning(
                self, "Dati mancanti", "Seleziona paziente e documento sorgente."
            )
            return
        try:
            value = json.loads(self._value.toPlainText() or "{}")
            if not isinstance(value, dict):
                raise ValueError("Il payload deve essere un oggetto JSON")
            annotation = GoldAtomicAnnotation(
                annotation_id=(
                    self._current.annotation_id if self._current else
                    f"GATOM_{uuid.uuid4().hex}"
                ),
                patient_id=self._patient_id,
                document_id=self._document_id,
                reviewer_slot=slot,
                reviewer_id=self._reviewer_id(slot),
                annotation_kind=str(self._kind.currentData()),
                fact_type=self._fact_type.currentData(),
                concept_original=self._concept.text().strip() or None,
                canonical_label=self._canonical.text().strip() or None,
                observation_date=self._observation_date.text().strip() or None,
                date_precision=self._precision.currentText(),
                polarity=self._polarity.currentData(),
                disposition=self._disposition.currentText(),
                exclusion_reason=self._reason.text().strip() or None,
                source_page=self._page.value() or None,
                source_text=self._passage.toPlainText().strip(),
                value=value,
                duplicate_of_annotation_id=(
                    self._duplicate_of.text().strip() or None
                ),
                created_at=(
                    self._current.created_at if self._current
                    else datetime.now(timezone.utc).isoformat()
                ),
            )
            repository.save_atomic_annotation(annotation)
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            QMessageBox.warning(self, "Annotazione non salvata", str(exc))
            return
        self._current = annotation
        self.refresh()

    def _delete_annotation(self):
        if self._current is None:
            return
        try:
            self._services["gold_set_repo"].delete_atomic_annotation(
                self._current.annotation_id,
                actor_id=self._reviewer_id(self._role_getter()) or "local_user",
            )
        except ValueError as exc:
            QMessageBox.warning(self, "Annotazione non eliminata", str(exc))
            return
        self._clear()
        self.refresh()

    def _clear(self):
        self._current = None
        self._concept.clear()
        self._canonical.clear()
        self._observation_date.clear()
        self._reason.clear()
        self._duplicate_of.clear()
        self._passage.clear()
        self._value.setPlainText("{}")

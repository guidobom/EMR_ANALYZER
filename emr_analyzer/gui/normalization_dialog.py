"""Dialog listing documents that still need clinical-text normalization.

Workspace-wide counterpart of the per-patient "Estrai testo clinico" button:
it shows every document that (a) is not yet normalized (non-laboratory type
and no ``clinical_text`` in its metadata) or (b) carries a parsing or
extraction error, and lets the user launch the extraction pipeline on the
selected files (grouped by patient).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QAbstractItemView, QDialog, QHeaderView, QHBoxLayout, QLabel,
    QPushButton, QTableWidget, QTableWidgetItem, QVBoxLayout,
)

from ..models.document import (
    DocumentRecord, DocumentType, ExtractionStatus, ParsingStatus,
)


@dataclass
class PendingDoc:
    """A document that needs normalization and/or carries an error."""
    doc: DocumentRecord
    needs_norm: bool
    has_error: bool


@dataclass
class PendingClassification:
    pending: list[PendingDoc] = field(default_factory=list)
    count_needs_norm: int = 0
    count_errors: int = 0
    count_skipped_done: int = 0

    @property
    def total(self) -> int:
        return len(self.pending)


def _has_clinical_text(doc: DocumentRecord) -> bool:
    """True when the document's metadata carries a normalized clinical_text."""
    if not doc.metadata_json:
        return False
    try:
        meta = json.loads(doc.metadata_json)
    except (TypeError, ValueError):
        return False
    return meta.get("clinical_text") is not None


def classify_pending_documents(docs) -> PendingClassification:
    """Classify documents as pending normalization and/or errored.

    Pure function (no Qt) so it can be unit-tested headlessly.  A document is
    pending when it is non-laboratory and still lacks a normalized
    ``clinical_text``, or when its parsing or extraction ended in ``error``.
    Laboratory documents are intentionally never LLM-normalized, so they are
    listed only when they carry an error.
    """
    result = PendingClassification()
    for doc in docs:
        is_lab = doc.document_type == DocumentType.LABORATORIO.value
        needs_norm = (not is_lab) and not _has_clinical_text(doc)
        has_error = (
            doc.parsing_status == ParsingStatus.ERROR.value
            or doc.extraction_status == ExtractionStatus.ERROR.value
        )
        if needs_norm:
            result.count_needs_norm += 1
            if doc.extraction_status == ExtractionStatus.DONE.value:
                # Rare: marked DONE without a clinical_text — the shared
                # extract_clinical_text entry point skips DONE documents, so
                # this one can only be reprocessed individually.
                result.count_skipped_done += 1
        if has_error:
            result.count_errors += 1
        if needs_norm or has_error:
            result.pending.append(PendingDoc(doc, needs_norm, has_error))
    return result


def _pending_status_text(p: PendingDoc) -> str:
    labels = []
    if p.needs_norm:
        labels.append("da normalizzare")
    if p.has_error:
        labels.append("errore")
    return " + ".join(labels)


class NormalizationDialog(QDialog):
    """Table of pending documents with per-row checkboxes and an extract action."""

    def __init__(self, classification: PendingClassification, parent=None):
        super().__init__(parent)
        self._classification = classification
        self._grouped: dict[str, list[str]] = {}
        self.setWindowTitle("Documenti da normalizzare / con errori")
        self.resize(1000, 540)
        self._setup_ui()
        self._populate()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def _setup_ui(self):
        layout = QVBoxLayout(self)

        self._summary_label = QLabel()
        layout.addWidget(self._summary_label)

        self._table = QTableWidget()
        self._table.setColumnCount(7)
        self._table.setHorizontalHeaderLabels(
            ["☑", "Paziente", "File", "Tipo", "Data", "Stato", "Errore"]
        )
        self._table.horizontalHeader().setStretchLastSection(True)
        self._table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        self._table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self._table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._table.setAlternatingRowColors(True)
        self._table.itemChanged.connect(self._on_item_changed)
        layout.addWidget(self._table, stretch=1)

        self._notes_label = QLabel()
        self._notes_label.setWordWrap(True)
        self._notes_label.setStyleSheet("color: #b9770e;")
        layout.addWidget(self._notes_label)

        buttons = QHBoxLayout()
        self._select_all_btn = QPushButton("☑ Seleziona tutti")
        self._select_all_btn.clicked.connect(self._toggle_select_all)
        buttons.addWidget(self._select_all_btn)

        buttons.addStretch()

        self._extract_btn = QPushButton("Estrai testo clinico")
        self._extract_btn.setObjectName("successButton")
        self._extract_btn.clicked.connect(self.accept)
        buttons.addWidget(self._extract_btn)

        close_btn = QPushButton("Chiudi")
        close_btn.clicked.connect(self.reject)
        buttons.addWidget(close_btn)

        layout.addLayout(buttons)

    def _populate(self):
        classification = self._classification
        table = self._table
        table.blockSignals(True)
        table.setRowCount(classification.total)
        for row, p in enumerate(classification.pending):
            doc = p.doc

            check = QTableWidgetItem()
            check.setFlags(
                Qt.ItemIsEnabled | Qt.ItemIsSelectable | Qt.ItemIsUserCheckable
            )
            check.setCheckState(Qt.Checked)
            check.setData(Qt.UserRole, p)
            table.setItem(row, 0, check)

            table.setItem(row, 1, QTableWidgetItem(doc.patient_id or ""))
            table.setItem(row, 2, QTableWidgetItem(doc.filename))
            table.setItem(row, 3, QTableWidgetItem(doc.document_type or ""))
            table.setItem(row, 4, QTableWidgetItem(doc.document_date or ""))
            table.setItem(row, 5, QTableWidgetItem(_pending_status_text(p)))
            table.setItem(row, 6, QTableWidgetItem(doc.error_message or ""))

            if p.has_error:
                table.item(row, 5).setForeground(Qt.red)
        table.blockSignals(False)

        # Default selection: every pending document is checked.
        for p in classification.pending:
            self._grouped.setdefault(p.doc.patient_id, [])
            if p.doc.id not in self._grouped[p.doc.patient_id]:
                self._grouped[p.doc.patient_id].append(p.doc.id)

        self._table.setColumnWidth(0, 40)
        self._summary_label.setText(
            f"{classification.count_needs_norm} da normalizzare, "
            f"{classification.count_errors} con errore "
            f"(totale {classification.total} file)"
        )
        if classification.count_skipped_done:
            self._notes_label.setText(
                f"⚠ {classification.count_skipped_done} documento/i hanno già stato "
                "'completato' ma senza testo clinico: l'estrazione automatica li "
                "salterà — rilanciarli singolarmente dal tab del paziente."
            )
        self._update_extract_button()

    # ------------------------------------------------------------------
    # Selection handling
    # ------------------------------------------------------------------
    def selected_groups(self) -> dict[str, list[str]]:
        """Map patient_id → [doc_ids] for the currently checked rows."""
        return dict(self._grouped)

    def _update_extract_button(self):
        n = sum(len(ids) for ids in self._grouped.values())
        self._extract_btn.setText(f"Estrai testo clinico ({n})")
        self._extract_btn.setEnabled(n > 0)

    def _on_item_changed(self, item):
        if item.column() != 0:
            return
        p = item.data(Qt.UserRole)
        if p is None:
            return
        doc = p.doc
        ids = self._grouped.setdefault(doc.patient_id, [])
        if item.checkState() == Qt.Checked:
            if doc.id not in ids:
                ids.append(doc.id)
        else:
            if doc.id in ids:
                ids.remove(doc.id)
            if not ids:
                self._grouped.pop(doc.patient_id, None)
        self._update_extract_button()

    def _toggle_select_all(self):
        table = self._table
        rows = table.rowCount()
        if rows == 0:
            return
        all_checked = all(
            table.item(r, 0).checkState() == Qt.Checked for r in range(rows)
        )
        target = Qt.Unchecked if all_checked else Qt.Checked
        table.blockSignals(True)
        self._grouped = {}
        for r in range(rows):
            check = table.item(r, 0)
            check.setCheckState(target)
            if target == Qt.Checked:
                p = check.data(Qt.UserRole)
                if p is not None:
                    self._grouped.setdefault(p.doc.patient_id, [])
                    if p.doc.id not in self._grouped[p.doc.patient_id]:
                        self._grouped[p.doc.patient_id].append(p.doc.id)
        table.blockSignals(False)
        self._update_extract_button()

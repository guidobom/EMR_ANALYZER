"""Review queue and source Quick View for excluded clinical evidence."""

from __future__ import annotations

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .pdf_viewer import DocumentEvidencePreview


class ExcludedEvidenceDialog(QDialog):
    """Inspect boilerplate/method/admin atoms beside their original report."""

    def __init__(self, patient_id: str, services: dict, parent=None):
        super().__init__(parent)
        self._patient_id = patient_id
        self._services = services
        self._rows = []
        self.setWindowTitle("Evidenze escluse — Quick View e revisione")
        self.resize(1450, 850)
        self._setup_ui()
        self._refresh()

    def _setup_ui(self):
        root = QVBoxLayout(self)
        notice = QLabel(
            "Queste evidenze restano archiviate con il motivo di esclusione, "
            "ma non entrano in eventi, sintesi o analisi LLM."
        )
        notice.setWordWrap(True)
        root.addWidget(notice)
        splitter = QSplitter(Qt.Horizontal)
        left = QWidget()
        left_layout = QVBoxLayout(left)
        self._table = QTableWidget(0, 6)
        self._table.setHorizontalHeaderLabels([
            "Stato", "Disposizione", "Motivo", "Documento", "Pagina", "Passaggio",
        ])
        self._table.horizontalHeader().setSectionResizeMode(5, QHeaderView.Stretch)
        self._table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._table.itemSelectionChanged.connect(self._show_selected)
        left_layout.addWidget(self._table)
        buttons = QHBoxLayout()
        accept = QPushButton("Conferma esclusione")
        accept.clicked.connect(lambda: self._review("accepted"))
        reject = QPushButton("Contesta esclusione")
        reject.setToolTip(
            "La fonte resta prudentemente fuori dal registro finché non viene corretta"
        )
        reject.clicked.connect(lambda: self._review("rejected"))
        pending = QPushButton("Rimanda")
        pending.clicked.connect(lambda: self._review("pending"))
        for button in (accept, reject, pending):
            buttons.addWidget(button)
        left_layout.addLayout(buttons)
        splitter.addWidget(left)
        self._preview = DocumentEvidencePreview(self._services, self)
        splitter.addWidget(self._preview)
        splitter.setSizes([650, 800])
        root.addWidget(splitter)

    def _refresh(self):
        repository = self._services.get("pipeline_repo")
        self._rows = repository.list_excluded(self._patient_id) if repository else []
        self._table.setRowCount(len(self._rows))
        for row_index, row in enumerate(self._rows):
            values = (
                row.get("review_status"), row.get("disposition"),
                row.get("reason_code"), row.get("document_id"),
                row.get("source_page") or "", row.get("source_text"),
            )
            for column, value in enumerate(values):
                item = QTableWidgetItem(str(value or ""))
                if column == 0:
                    item.setData(Qt.UserRole, row.get("excluded_id"))
                self._table.setItem(row_index, column, item)
        if self._rows:
            self._table.selectRow(0)

    def _show_selected(self):
        row_index = self._table.currentRow()
        if row_index < 0 or row_index >= len(self._rows):
            return
        row = self._rows[row_index]
        document_repo = self._services.get("document_repo")
        document = document_repo.get_by_id(row["document_id"]) if document_repo else None
        if document is None:
            return
        highlight_id = row["excluded_id"]
        self._preview.set_document(
            document.to_dict(),
            highlights=[{
                "evidence_id": highlight_id,
                "page_number": row.get("source_page"),
                "bbox": row.get("bbox"),
                "source_text": row.get("source_text"),
                "normalized_entity": row.get("concept") or row.get("reason_code"),
            }],
            selected_evidence_id=highlight_id,
        )

    def _review(self, decision: str):
        row_index = self._table.currentRow()
        if row_index < 0 or row_index >= len(self._rows):
            return
        reason, accepted = QInputDialog.getText(
            self, "Motivo della revisione", "Nota (facoltativa):"
        )
        if not accepted:
            return
        try:
            self._services["pipeline_repo"].review_exclusion(
                self._rows[row_index]["excluded_id"], decision, reason=reason
            )
        except ValueError as exc:
            QMessageBox.warning(self, "Revisione non salvata", str(exc))
            return
        self._refresh()

    def closeEvent(self, event):
        self._preview.close_document()
        event.accept()

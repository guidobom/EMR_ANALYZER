"""Manual split editor for one evidence-backed clinical event."""

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QDialog, QDialogButtonBox, QFormLayout, QLabel, QLineEdit,
    QTableWidget, QTableWidgetItem, QVBoxLayout,
)


class EventSplitDialog(QDialog):
    def __init__(self, detail: dict, parent=None):
        super().__init__(parent)
        self._detail = detail
        event = detail.get("event") or {}
        self.setWindowTitle("Dividi evento clinico")
        self.resize(950, 620)
        root = QVBoxLayout(self)
        notice = QLabel(
            "Seleziona le evidenze da spostare nel nuovo evento. Almeno una "
            "evidenza deve restare nell'evento originale."
        )
        notice.setWordWrap(True)
        root.addWidget(notice)
        evidences = detail.get("evidence") or []
        self._table = QTableWidget(len(evidences), 5)
        self._table.setHorizontalHeaderLabels(
            ["Sposta", "Data", "Ruolo", "Concetto", "Passaggio"]
        )
        self._table.horizontalHeader().setStretchLastSection(True)
        for row, evidence in enumerate(evidences):
            check = QTableWidgetItem()
            check.setFlags(Qt.ItemIsEnabled | Qt.ItemIsUserCheckable)
            check.setCheckState(Qt.Unchecked)
            check.setData(Qt.UserRole, evidence.get("evidence_id"))
            self._table.setItem(row, 0, check)
            for column, value in enumerate((
                evidence.get("observed_date") or evidence.get("source_document_date") or "",
                evidence.get("role") or "core",
                evidence.get("normalized_entity") or "",
                evidence.get("source_text") or "",
            ), start=1):
                self._table.setItem(row, column, QTableWidgetItem(str(value)))
        root.addWidget(self._table)
        form = QFormLayout()
        self._original_summary = QLineEdit(event.get("summary_short") or "")
        self._new_summary = QLineEdit()
        self._new_summary.setPlaceholderText("Descrizione del nuovo evento")
        form.addRow("Sintesi evento originale", self._original_summary)
        form.addRow("Sintesi nuovo evento", self._new_summary)
        root.addLayout(form)
        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

    @property
    def selected_evidence_ids(self):
        return [
            str(item.data(Qt.UserRole))
            for row in range(self._table.rowCount())
            if (item := self._table.item(row, 0)) is not None
            and item.checkState() == Qt.Checked
        ]

    @property
    def original_summary(self):
        return self._original_summary.text().strip()

    @property
    def new_summary(self):
        return self._new_summary.text().strip()

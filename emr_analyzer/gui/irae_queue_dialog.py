"""Selection dialog for the multi-patient irAE analysis queue."""

from __future__ import annotations

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QTableWidget, QTableWidgetItem,
    QHeaderView, QPushButton, QLabel, QAbstractItemView,
)


def merge_irae_queue_summaries(
    timeline_summaries: list[dict],
    evidence_summaries: list[dict],
) -> list[dict]:
    """Union of the registry and evidence patient lists for the irAE queue.

    Patients with a chronological registry keep their ``timeline_count``
    untouched; patients found only in the atomic evidence (the input of the
    structured NCTCAE 3-layer method) get ``timeline_count`` set to their
    ``evidence_count`` so the dialog's count column stays populated.
    """
    merged: dict[str, dict] = {}
    for summary in timeline_summaries:
        merged[summary["id"]] = dict(summary)
    for summary in evidence_summaries:
        if summary["id"] not in merged:
            merged[summary["id"]] = {
                "id": summary["id"],
                "pseudonym": summary.get("pseudonym") or "",
                "timeline_count": summary.get("evidence_count", 0),
            }
    return list(merged.values())


class IraeQueueDialog(QDialog):
    """Checkbox list of the patients that HAVE a chronological registry."""

    def __init__(self, summaries: list[dict], parent=None):
        super().__init__(parent)
        self.setWindowTitle("Analisi irAE multi-paziente")
        self.resize(720, 480)
        self._summaries = summaries
        self._selected: set[str] = set(s["id"] for s in summaries)

        layout = QVBoxLayout(self)
        intro = QLabel(
            "Seleziona i pazienti da analizzare con il protocollo irAE. "
            "La coda elabora un paziente alla volta; al termine si apre "
            "un riepilogo con una scheda per paziente."
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        self._table = QTableWidget()
        self._table.setColumnCount(4)
        self._table.setHorizontalHeaderLabels(
            ["☑", "Paziente", "Pseudonimo", "Voci registro"]
        )
        self._table.horizontalHeader().setSectionResizeMode(
            2, QHeaderView.Stretch
        )
        self._table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._table.setAlternatingRowColors(True)
        self._table.itemChanged.connect(self._on_item_changed)
        layout.addWidget(self._table, stretch=1)

        self._table.blockSignals(True)
        self._table.setRowCount(len(summaries))
        for row, summary in enumerate(summaries):
            check = QTableWidgetItem()
            check.setFlags(
                Qt.ItemIsEnabled | Qt.ItemIsSelectable
                | Qt.ItemIsUserCheckable
            )
            check.setCheckState(Qt.Checked)
            check.setData(Qt.UserRole, summary["id"])
            self._table.setItem(row, 0, check)
            self._table.setItem(row, 1, QTableWidgetItem(summary["id"]))
            self._table.setItem(
                row, 2, QTableWidgetItem(summary.get("pseudonym") or "")
            )
            self._table.setItem(
                row, 3, QTableWidgetItem(str(summary.get("timeline_count", 0)))
            )
        self._table.blockSignals(False)

        buttons = QHBoxLayout()
        self._toggle_btn = QPushButton("☐ Deseleziona tutti")
        self._toggle_btn.clicked.connect(self._toggle_all)
        buttons.addWidget(self._toggle_btn)
        buttons.addStretch()
        self._run_btn = QPushButton()
        self._run_btn.setObjectName("successButton")
        self._run_btn.clicked.connect(self.accept)
        buttons.addWidget(self._run_btn)
        close_btn = QPushButton("Chiudi")
        close_btn.clicked.connect(self.reject)
        buttons.addWidget(close_btn)
        layout.addLayout(buttons)

        self._update_run_button()

    def _on_item_changed(self, item) -> None:
        if item.column() != 0:
            return
        patient_id = item.data(Qt.UserRole)
        if item.checkState() == Qt.Checked:
            self._selected.add(patient_id)
        else:
            self._selected.discard(patient_id)
        self._update_run_button()

    def _update_run_button(self) -> None:
        self._run_btn.setText(f"⚡ Avvia analisi ({len(self._selected)})")
        self._run_btn.setEnabled(bool(self._selected))

    def _toggle_all(self) -> None:
        check_all = not self._selected
        self._table.blockSignals(True)
        for row in range(self._table.rowCount()):
            item = self._table.item(row, 0)
            item.setCheckState(Qt.Checked if check_all else Qt.Unchecked)
        self._table.blockSignals(False)
        self._selected = (
            set(s["id"] for s in self._summaries) if check_all else set()
        )
        self._toggle_btn.setText(
            "☐ Deseleziona tutti" if check_all else "☑ Seleziona tutti"
        )
        self._update_run_button()

    def selected_patient_ids(self) -> list[str]:
        """Selected ids in the order the rows are shown."""
        return [
            s["id"] for s in self._summaries if s["id"] in self._selected
        ]

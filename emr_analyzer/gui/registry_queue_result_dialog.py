"""Final summary for multi-patient chronological-registry generation."""

from __future__ import annotations

from PyQt5.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QHeaderView,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)


class RegistryQueueResultDialog(QDialog):
    def __init__(self, results: list[dict], *, cancelled: bool = False,
                 parent=None):
        super().__init__(parent)
        self.setWindowTitle("Riepilogo coda registri")
        self.resize(960, 480)

        layout = QVBoxLayout(self)
        completed = sum(not result.get("error") for result in results)
        errors = sum(bool(result.get("error")) for result in results)
        suffix = " — coda interrotta" if cancelled else ""
        label = QLabel(
            f"{completed} pazienti completati, {errors} errori{suffix}."
        )
        layout.addWidget(label)

        self._table = QTableWidget(len(results), 7)
        self._table.setHorizontalHeaderLabels([
            "Paziente", "Esito", "Doc. elaborati", "Doc. saltati",
            "Evidenze", "Voci finali", "Tempo",
        ])
        self._table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._table.setAlternatingRowColors(True)
        self._table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.Stretch
        )
        for row, item in enumerate(results):
            result = item.get("result") or {}
            error = item.get("error")
            processed = int(result.get("documents_processed", 0) or 0)
            skipped = int(result.get("documents_skipped", 0) or 0)
            if error:
                outcome = f"Errore: {error}"
            elif processed == 0 and skipped:
                outcome = "Già aggiornato"
            else:
                outcome = "Completato"
            elapsed = result.get("elapsed_seconds")
            values = (
                item.get("patient_id", ""), outcome, processed, skipped,
                result.get("atomic_evidence_extracted",
                           result.get("total_entries", 0)),
                result.get("final_entries", 0),
                f"{float(elapsed):.0f} s" if elapsed is not None else "—",
            )
            for column, value in enumerate(values):
                self._table.setItem(row, column, QTableWidgetItem(str(value)))
        layout.addWidget(self._table, stretch=1)

        buttons = QHBoxLayout()
        buttons.addStretch()
        close = QPushButton("Chiudi")
        close.clicked.connect(self.accept)
        buttons.addWidget(close)
        layout.addLayout(buttons)

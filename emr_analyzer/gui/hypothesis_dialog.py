"""Separate discovery and human review of clinical hypotheses."""

from __future__ import annotations

from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from ..clinical.hypothesis_discovery import HypothesisDiscovery


class _DiscoveryWorker(QThread):
    completed = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, discovery, patient_id: str, parent=None):
        super().__init__(parent)
        self.discovery = discovery
        self.patient_id = patient_id

    def run(self) -> None:
        try:
            self.completed.emit(self.discovery.run(self.patient_id))
        except Exception as exc:
            self.failed.emit(str(exc))


class HypothesisDialog(QDialog):
    """Keep exploratory output visibly separate from validated registry data."""

    def __init__(self, patient_id: str, services: dict, parent=None):
        super().__init__(parent)
        self.patient_id = patient_id
        self.services = services
        self._worker = None
        self.setWindowTitle(f"Ipotesi esplorative — {patient_id}")
        self.resize(1100, 620)

        root = QVBoxLayout(self)
        notice = QLabel(
            "Le ipotesi sono generate con un comando separato e restano "
            "escluse da eventi, sintesi, analisi e RAG. L'accettazione umana "
            "le rende revisionate, ma non le trasforma in fatti documentati."
        )
        notice.setWordWrap(True)
        root.addWidget(notice)

        self._table = QTableWidget(0, 7)
        self._table.setHorizontalHeaderLabels([
            "Stato", "Forza", "Score", "Tipo", "Evento sorgente",
            "Evento destinazione", "Razionale",
        ])
        self._table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._table.setSelectionMode(QAbstractItemView.SingleSelection)
        self._table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._table.horizontalHeader().setStretchLastSection(True)
        root.addWidget(self._table)

        actions = QHBoxLayout()
        self._generate = QPushButton("Scopri con LLM")
        self._generate.clicked.connect(self._run_discovery)
        self._accept = QPushButton("Accetta ipotesi")
        self._accept.clicked.connect(lambda: self._review("accepted"))
        self._reject = QPushButton("Rifiuta ipotesi")
        self._reject.clicked.connect(lambda: self._review("rejected"))
        actions.addWidget(self._generate)
        actions.addWidget(self._accept)
        actions.addWidget(self._reject)
        actions.addStretch(1)
        root.addLayout(actions)

        self._status = QLabel("")
        root.addWidget(self._status)
        close_buttons = QDialogButtonBox(QDialogButtonBox.Close)
        close_buttons.rejected.connect(self.reject)
        root.addWidget(close_buttons)
        self._refresh()

    def _refresh(self) -> None:
        repository = self.services.get("pipeline_repo")
        registry = self.services.get("registry_repo")
        rows = repository.list_hypotheses(self.patient_id) if repository else []
        event_names = {
            event.event_id: event.summary_short
            for event in (registry.get_events(
                self.patient_id, include_rejected=True
            ) if registry else [])
        }
        self._table.setRowCount(len(rows))
        for row_index, row in enumerate(rows):
            values = (
                row["status"], row["strength"],
                "" if row["survival_score"] is None
                else f"{float(row['survival_score']):.2f}",
                row["hypothesis_type"],
                event_names.get(row["source_event_id"], row["source_event_id"]),
                event_names.get(row["target_event_id"], row["target_event_id"]),
                row["rationale"],
            )
            for column, value in enumerate(values):
                item = QTableWidgetItem(str(value or ""))
                if column == 0:
                    item.setData(Qt.UserRole, row["hypothesis_id"])
                self._table.setItem(row_index, column, item)
        self._table.resizeColumnsToContents()
        self._status.setText(f"{len(rows)} ipotesi archiviate separatamente")

    def _run_discovery(self) -> None:
        if self._worker and self._worker.isRunning():
            return
        llm = self.services.get("clinical_events_llm_client")
        discovery = HypothesisDiscovery(
            self.services.get("registry_repo"),
            self.services.get("pipeline_repo"),
            llm,
        )
        self._set_running(True)
        self._status.setText("Valutazione dei candidati in corso...")
        self._worker = _DiscoveryWorker(discovery, self.patient_id, self)
        self._worker.completed.connect(self._completed)
        self._worker.failed.connect(self._failed)
        self._worker.finished.connect(lambda: self._set_running(False))
        self._worker.start()

    def _completed(self, result: dict) -> None:
        self._refresh()
        self._status.setText(
            f"Completato: {result['candidate_count']} candidati, "
            f"{result['hypothesis_count']} ipotesi, "
            f"{result['llm_calls']} chiamate LLM"
        )

    def _failed(self, message: str) -> None:
        self._status.setText("Scoperta non completata")
        QMessageBox.warning(self, "Ipotesi non generate", message)

    def _review(self, decision: str) -> None:
        row = self._table.currentRow()
        item = self._table.item(row, 0) if row >= 0 else None
        hypothesis_id = item.data(Qt.UserRole) if item else None
        if not hypothesis_id:
            QMessageBox.information(
                self, "Nessuna ipotesi", "Seleziona prima una riga."
            )
            return
        try:
            self.services["pipeline_repo"].review_hypothesis(
                str(hypothesis_id), decision
            )
        except (KeyError, ValueError) as exc:
            QMessageBox.warning(self, "Revisione non salvata", str(exc))
            return
        self._refresh()

    def _set_running(self, running: bool) -> None:
        self._generate.setEnabled(not running)
        self._accept.setEnabled(not running)
        self._reject.setEnabled(not running)

    def reject(self) -> None:
        if self._worker and self._worker.isRunning():
            QMessageBox.information(
                self, "Elaborazione in corso",
                "Attendi il completamento della scoperta delle ipotesi."
            )
            return
        super().reject()

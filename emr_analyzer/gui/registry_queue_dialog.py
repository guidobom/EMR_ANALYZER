"""Selection dialog for sequential multi-patient registry generation."""

from __future__ import annotations

from collections import defaultdict

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDialog,
    QHeaderView,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from ..models.document import ExtractionStatus


def build_registry_queue_summaries(services: dict) -> list[dict]:
    """Return one queue row per patient having at least one document."""
    patient_repo = services.get("patient_repo")
    document_repo = services.get("document_repo")
    if patient_repo is None or document_repo is None:
        return []

    documents_by_patient: dict[str, list] = defaultdict(list)
    for document in document_repo.list_all():
        documents_by_patient[document.patient_id].append(document)

    registry_repo = services.get("registry_repo")
    timeline_repo = services.get("timeline_repo")
    summaries = []
    for patient in patient_repo.list_all():
        documents = documents_by_patient.get(patient.id, [])
        if not documents:
            continue
        ready = sum(
            document.extraction_status == ExtractionStatus.DONE.value
            for document in documents
        )
        if registry_repo is not None:
            registry_count = registry_repo.count_by_patient(patient.id)
        elif timeline_repo is not None:
            registry_count = timeline_repo.count_by_patient(patient.id)
        else:
            registry_count = 0
        summaries.append({
            "id": patient.id,
            "pseudonym": patient.pseudonym or "",
            "document_count": len(documents),
            "normalized_count": ready,
            "pending_count": len(documents) - ready,
            "registry_count": registry_count,
            "eligible": ready > 0,
        })
    return summaries


class RegistryQueueDialog(QDialog):
    """Choose patients and incremental versus full registry generation."""

    INCREMENTAL = "incremental"
    REBUILD = "rebuild"

    def __init__(self, summaries: list[dict], parent=None):
        super().__init__(parent)
        self.setWindowTitle("Generazione registri multi-paziente")
        self.resize(860, 540)
        self._summaries = list(summaries)
        self._selected = {
            summary["id"] for summary in summaries
            if summary.get("eligible", False)
        }

        layout = QVBoxLayout(self)
        intro = QLabel(
            "Seleziona i pazienti da inserire nella coda. I pazienti vengono "
            "elaborati uno alla volta; per ciascuno, i documenti utilizzano "
            "tutti i worker LLM configurati. L'annullamento diventa effettivo "
            "dopo il paziente in corso."
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("Modalità:"))
        self._mode = QComboBox()
        self._mode.addItem(
            "Aggiorna/riprendi; salta i documenti già correnti (consigliato)",
            self.INCREMENTAL,
        )
        self._mode.addItem(
            "Rigenera integralmente tutti i registri selezionati",
            self.REBUILD,
        )
        self._mode.currentIndexChanged.connect(self._update_run_button)
        mode_row.addWidget(self._mode, stretch=1)
        layout.addLayout(mode_row)

        self._warning = QLabel()
        self._warning.setWordWrap(True)
        self._warning.setStyleSheet("color: #a65e00;")
        layout.addWidget(self._warning)

        self._table = QTableWidget()
        self._table.setColumnCount(7)
        self._table.setHorizontalHeaderLabels([
            "☑", "Paziente", "Pseudonimo", "Documenti",
            "Normalizzati", "Da normalizzare", "Voci registro",
        ])
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
            eligible = bool(summary.get("eligible"))
            check = QTableWidgetItem()
            flags = Qt.ItemIsSelectable | Qt.ItemIsUserCheckable
            if eligible:
                flags |= Qt.ItemIsEnabled
            check.setFlags(flags)
            check.setCheckState(Qt.Checked if eligible else Qt.Unchecked)
            check.setData(Qt.UserRole, summary["id"])
            if not eligible:
                check.setToolTip(
                    "Nessun documento clinico normalizzato disponibile"
                )
            self._table.setItem(row, 0, check)
            values = (
                summary["id"], summary.get("pseudonym") or "",
                summary.get("document_count", 0),
                summary.get("normalized_count", 0),
                summary.get("pending_count", 0),
                summary.get("registry_count", 0),
            )
            for column, value in enumerate(values, start=1):
                self._table.setItem(row, column, QTableWidgetItem(str(value)))
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

    def _toggle_all(self) -> None:
        eligible_ids = {
            summary["id"] for summary in self._summaries
            if summary.get("eligible", False)
        }
        check_all = not self._selected
        self._table.blockSignals(True)
        for row in range(self._table.rowCount()):
            item = self._table.item(row, 0)
            patient_id = item.data(Qt.UserRole)
            item.setCheckState(
                Qt.Checked
                if check_all and patient_id in eligible_ids
                else Qt.Unchecked
            )
        self._table.blockSignals(False)
        self._selected = eligible_ids if check_all else set()
        self._toggle_btn.setText(
            "☐ Deseleziona tutti" if check_all else "☑ Seleziona tutti"
        )
        self._update_run_button()

    def _update_run_button(self) -> None:
        rebuild = self.force_rebuild()
        verb = "Rigenera" if rebuild else "Avvia coda"
        self._run_btn.setText(f"📋 {verb} ({len(self._selected)})")
        self._run_btn.setEnabled(bool(self._selected))
        self._warning.setText(
            "La rigenerazione integrale ignora i manifest correnti e può "
            "richiedere molte ore."
            if rebuild else
            "I registri già aggiornati vengono verificati senza ripetere "
            "le chiamate LLM."
        )

    def selected_patient_ids(self) -> list[str]:
        """Selected eligible IDs in their displayed order."""
        return [
            summary["id"] for summary in self._summaries
            if summary["id"] in self._selected
        ]

    def force_rebuild(self) -> bool:
        return self._mode.currentData() == self.REBUILD

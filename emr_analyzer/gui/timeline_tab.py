"""Timeline tab — chronological view of clinical events."""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTableWidget, QTableWidgetItem,
    QHeaderView, QLabel, QComboBox, QPushButton, QAbstractItemView,
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor


class TimelineTab(QWidget):
    """Chronological display of clinical events."""

    event_selected = pyqtSignal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._services = {}
        self._current_patient_id = None
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        # Filters
        filter_layout = QHBoxLayout()
        filter_layout.addWidget(QLabel("Filtro tipo:"))

        self._type_filter = QComboBox()
        self._type_filter.addItem("Tutti gli eventi", "")
        self._type_filter.currentIndexChanged.connect(self._refresh_table)
        filter_layout.addWidget(self._type_filter)

        filter_layout.addWidget(QLabel("Periodo:"))
        self._period_filter = QComboBox()
        self._period_filter.addItem("Tutto", "")
        self._period_filter.addItem("Ultimo anno", "1y")
        self._period_filter.addItem("Ultimi 6 mesi", "6m")
        self._period_filter.addItem("Ultimi 3 mesi", "3m")
        self._period_filter.addItem("Ultimo mese", "1m")
        self._period_filter.currentIndexChanged.connect(self._refresh_table)
        filter_layout.addWidget(self._period_filter)

        filter_layout.addStretch()
        layout.addLayout(filter_layout)

        # Event table
        self._table = QTableWidget()
        self._table.setColumnCount(8)
        self._table.setHorizontalHeaderLabels([
            "Data", "Tipo", "Entità", "Valore", "Stato",
            "Confidenza", "Fonte", "Pagina"
        ])
        self._table.horizontalHeader().setStretchLastSection(True)
        self._table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        self._table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._table.setAlternatingRowColors(True)
        self._table.itemSelectionChanged.connect(self._on_selection_changed)
        self._table.doubleClicked.connect(self._on_double_click)
        layout.addWidget(self._table, stretch=1)

        # Summary
        self._summary_label = QLabel("")
        layout.addWidget(self._summary_label)

    def set_services(self, services: dict):
        self._services = services

    def load_patient(self, patient_id: str):
        self._current_patient_id = patient_id
        self._populate_type_filter()
        self._refresh_table()

    def _populate_type_filter(self):
        """Populate the type filter with distinct event types."""
        self._type_filter.blockSignals(True)
        self._type_filter.clear()
        self._type_filter.addItem("Tutti gli eventi", "")

        event_repo = self._services.get("event_repo")
        if event_repo and self._current_patient_id:
            events = event_repo.get_by_patient(self._current_patient_id)
            types = sorted(set(e.event_type for e in events))
            for t in types:
                self._type_filter.addItem(t, t)

        self._type_filter.blockSignals(False)

    def _refresh_table(self):
        if not self._current_patient_id:
            return

        event_repo = self._services.get("event_repo")
        if not event_repo:
            return

        event_type = self._type_filter.currentData()
        if event_type:
            events = event_repo.get_by_type(self._current_patient_id, event_type)
        else:
            events = event_repo.get_by_patient(self._current_patient_id)

        # Filter by period
        period = self._period_filter.currentData()
        if period and events:
            from datetime import datetime, timedelta
            now = datetime.now()
            if period == "1y":
                cutoff = now - timedelta(days=365)
            elif period == "6m":
                cutoff = now - timedelta(days=180)
            elif period == "3m":
                cutoff = now - timedelta(days=90)
            elif period == "1m":
                cutoff = now - timedelta(days=30)
            else:
                cutoff = None

            if cutoff:
                cutoff_str = cutoff.strftime("%Y-%m-%d")
                events = [e for e in events if e.event_date >= cutoff_str]

        self._table.setRowCount(len(events))
        for i, event in enumerate(events):
            self._table.setItem(i, 0, QTableWidgetItem(event.event_date))
            self._table.setItem(i, 1, QTableWidgetItem(event.event_type))
            self._table.setItem(i, 2, QTableWidgetItem(event.entity))
            val_str = f"{event.value} {event.unit or ''}".strip()
            self._table.setItem(i, 3, QTableWidgetItem(val_str))
            self._table.setItem(i, 4, QTableWidgetItem(event.status))
            self._table.setItem(i, 5, QTableWidgetItem(f"{event.confidence:.2f}"))
            self._table.setItem(i, 6, QTableWidgetItem(event.source_document_id))
            self._table.setItem(i, 7, QTableWidgetItem(str(event.page or "")))

            # Store event data
            self._table.item(i, 0).setData(Qt.ItemDataRole.UserRole, event.to_dict())

            # Color by status
            if event.status == "rejected":
                for col in range(8):
                    item = self._table.item(i, col)
                    if item:
                        item.setForeground(QColor(Qt.GlobalColor.gray))
            elif event.confidence < 0.6:
                for col in range(8):
                    item = self._table.item(i, col)
                    if item:
                        item.setForeground(QColor(Qt.GlobalColor.darkYellow))

        self._summary_label.setText(
            f"{len(events)} eventi visualizzati | "
            f"{sum(1 for e in events if e.status == 'confirmed')} confermati | "
            f"{sum(1 for e in events if e.status == 'proposed')} da validare"
        )

    def _on_selection_changed(self):
        pass

    def _on_double_click(self, index):
        """Emit event to context panel."""
        row = index.row()
        item = self._table.item(row, 0)
        if item:
            event_data = item.data(Qt.ItemDataRole.UserRole)
            self.event_selected.emit(event_data)

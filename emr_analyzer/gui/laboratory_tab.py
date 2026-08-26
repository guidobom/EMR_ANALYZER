"""Laboratory tab — lab values table with temporal charts."""

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTableWidget, QTableWidgetItem,
    QHeaderView, QLabel, QComboBox, QPushButton, QAbstractItemView,
    QSplitter,
)
from PyQt5.QtCore import Qt, pyqtSignal

# Try to import pyqtgraph for charts
try:
    import pyqtgraph as pg
    HAS_PYQTGRAPH = True
except ImportError:
    HAS_PYQTGRAPH = False


class LaboratoryTab(QWidget):
    """Tab for laboratory values with temporal charting."""

    lab_selected = pyqtSignal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._services = {}
        self._current_patient_id = None
        self._current_parameter = None
        self._all_lab_values = []
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        # Parameter selector
        selector_layout = QHBoxLayout()
        selector_layout.addWidget(QLabel("Analita:"))

        self._parameter_combo = QComboBox()
        self._parameter_combo.setMinimumWidth(250)
        self._parameter_combo.setEditable(True)
        self._parameter_combo.currentTextChanged.connect(self._on_parameter_changed)
        selector_layout.addWidget(self._parameter_combo)

        self._abnormal_only_btn = QPushButton("Solo anomali")
        self._abnormal_only_btn.setCheckable(True)
        self._abnormal_only_btn.toggled.connect(self._refresh_table)
        selector_layout.addWidget(self._abnormal_only_btn)

        self._show_chart_btn = QPushButton("📈 Grafico")
        self._show_chart_btn.setCheckable(True)
        self._show_chart_btn.setChecked(True)
        self._show_chart_btn.toggled.connect(self._toggle_chart)
        selector_layout.addWidget(self._show_chart_btn)

        selector_layout.addStretch()

        self._stats_label = QLabel("")
        selector_layout.addWidget(self._stats_label)

        layout.addLayout(selector_layout)

        # Splitter: chart on top, table below
        self._splitter = QSplitter(Qt.Vertical)

        # Chart widget
        self._chart_widget = None
        if HAS_PYQTGRAPH:
            self._chart_widget = pg.PlotWidget()
            self._chart_widget.setLabel("left", "Valore")
            self._chart_widget.setLabel("bottom", "Data")
            self._chart_widget.showGrid(x=True, y=True, alpha=0.3)
            self._splitter.addWidget(self._chart_widget)

        # Lab values table
        self._table = QTableWidget()
        self._table.setColumnCount(10)
        self._table.setHorizontalHeaderLabels([
            "Data", "Parametro", "Valore", "Unità", "Materiale", "Range",
            "Esito", "Confidenza", "Fonte", "Pagina"
        ])
        self._table.horizontalHeader().setStretchLastSection(True)
        self._table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self._table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._table.setAlternatingRowColors(True)
        self._table.setObjectName("labTable")
        self._table.doubleClicked.connect(self._on_double_click)
        self._splitter.addWidget(self._table)

        layout.addWidget(self._splitter, stretch=1)

    def set_services(self, services: dict):
        self._services = services

    def load_patient(self, patient_id: str):
        self._current_patient_id = patient_id
        self._load_data()
        self._populate_parameter_combo()
        self._refresh_table()
        self._update_chart()

    def _load_data(self):
        lab_repo = self._services.get("lab_repo")
        if lab_repo and self._current_patient_id:
            self._all_lab_values = lab_repo.get_by_patient(self._current_patient_id)
        else:
            self._all_lab_values = []

    def _populate_parameter_combo(self):
        self._parameter_combo.blockSignals(True)
        self._parameter_combo.clear()
        self._parameter_combo.addItem("— Tutti i parametri —", "")

        params = sorted(set(
            lv.normalized_name for lv in self._all_lab_values
        ))
        for p in params:
            display = f"{p} ({sum(1 for lv in self._all_lab_values if lv.normalized_name == p)} valori)"
            self._parameter_combo.addItem(display, p)

        self._parameter_combo.blockSignals(False)

    def _on_parameter_changed(self, text: str):
        self._current_parameter = self._parameter_combo.currentData()
        self._refresh_table()
        self._update_chart()

    def _refresh_table(self):
        """Filter and display lab values."""
        values = self._all_lab_values

        # Filter by parameter
        if self._current_parameter:
            values = [v for v in values
                      if v.normalized_name == self._current_parameter]

        # Filter abnormal only
        if self._abnormal_only_btn.isChecked():
            values = [v for v in values if v.is_abnormal]

        # Sort by date
        values.sort(key=lambda v: v.sample_date or "0000-00-00")

        self._table.setRowCount(len(values))
        for i, lv in enumerate(values):
            self._table.setItem(i, 0, QTableWidgetItem(lv.sample_date or ""))
            self._table.setItem(i, 1, QTableWidgetItem(lv.parameter_name))
            val_str = f"{lv.operator or ''}{lv.value if lv.value is not None else ''}".strip()
            if lv.value_text:
                val_str = lv.value_text
            self._table.setItem(i, 2, QTableWidgetItem(val_str))
            self._table.setItem(i, 3, QTableWidgetItem(lv.unit or ""))
            material = (lv.biological_material or "").capitalize()
            self._table.setItem(i, 4, QTableWidgetItem(material))
            self._table.setItem(i, 5, QTableWidgetItem(lv.reference_text))
            if lv.flag == "H":
                outcome = "⚠ Alto"
            elif lv.flag == "L":
                outcome = "⚠ Basso"
            elif lv.is_abnormal:
                outcome = "⚠ Fuori range"
            else:
                outcome = "✓ Normale"
            self._table.setItem(i, 6, QTableWidgetItem(outcome))
            self._table.setItem(i, 7, QTableWidgetItem(f"{lv.confidence:.2f}"))
            self._table.setItem(i, 8, QTableWidgetItem(lv.document_id))
            self._table.setItem(i, 9, QTableWidgetItem(str(lv.page or "")))

            # Color abnormal rows
            if lv.is_abnormal:
                if lv.flag == "H":
                    bg = Qt.red
                elif lv.flag == "L":
                    bg = Qt.blue
                else:
                    bg = Qt.darkYellow
                for col in range(10):
                    item = self._table.item(i, col)
                    if item:
                        item.setBackground(bg)
                        item.setForeground(Qt.white)

            # Store data
            self._table.item(i, 0).setData(Qt.UserRole, lv.to_dict())

        # Update stats
        abnormal_count = sum(1 for v in values if v.is_abnormal)
        self._stats_label.setText(
            f"{len(values)} valori | {abnormal_count} anomali"
        )

    def _update_chart(self):
        """Update the temporal chart for the selected parameter."""
        if not HAS_PYQTGRAPH or not self._chart_widget:
            return
        if not self._current_parameter:
            self._chart_widget.clear()
            return

        self._chart_widget.clear()

        # Filter values for chart
        values = [v for v in self._all_lab_values
                  if v.normalized_name == self._current_parameter
                  and v.sample_date]
        values.sort(key=lambda v: v.sample_date)

        if not values:
            return

        # Build x (timestamps) and y (values)
        import time
        from datetime import datetime

        x_vals = []
        y_vals = []
        for lv in values:
            if lv.value is None:
                continue  # Skip textual results (cannot chart)
            try:
                dt = datetime.fromisoformat(lv.sample_date)
                x_vals.append(dt.timestamp())
                y_vals.append(lv.value)
            except (ValueError, TypeError):
                x_vals.append(len(x_vals))
                y_vals.append(lv.value)

        # Plot values
        scatter = pg.ScatterPlotItem(
            x=x_vals, y=y_vals, size=10, brush=pg.mkBrush(52, 152, 219, 200)
        )
        self._chart_widget.addItem(scatter)

        if len(x_vals) >= 2:
            line = pg.PlotDataItem(x_vals, y_vals, pen=pg.mkPen(52, 152, 219, 150))
            self._chart_widget.addItem(line)

        # Reference range lines
        ref_low = values[0].reference_low
        ref_high = values[0].reference_high
        if ref_low is not None and ref_high is not None and x_vals:
            min_x = min(x_vals)
            max_x = max(x_vals)
            if min_x < max_x:
                low_line = pg.PlotDataItem(
                    [min_x, max_x], [ref_low, ref_low],
                    pen=pg.mkPen(231, 76, 60, 100, style=Qt.DashLine)
                )
                high_line = pg.PlotDataItem(
                    [min_x, max_x], [ref_high, ref_high],
                    pen=pg.mkPen(231, 76, 60, 100, style=Qt.DashLine)
                )
                self._chart_widget.addItem(low_line)
                self._chart_widget.addItem(high_line)

        # Format x-axis as dates
        if x_vals and isinstance(x_vals[0], float) and x_vals[0] > 1000000000:
            from pyqtgraph import DateAxisItem
            date_axis = DateAxisItem(orientation='bottom')
            self._chart_widget.setAxisItems({'bottom': date_axis})

    def _on_double_click(self, index):
        """Emit lab value to context panel."""
        row = index.row()
        item = self._table.item(row, 0)
        if item:
            lab_data = item.data(Qt.UserRole)
            if lab_data:
                self.lab_selected.emit(lab_data)

    def _toggle_chart(self, show: bool):
        """Show/hide the chart panel."""
        if self._chart_widget:
            self._chart_widget.setVisible(show)

"""Laboratory tab — lab values table with temporal charts."""

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTableWidget, QTableWidgetItem,
    QHeaderView, QLabel, QComboBox, QPushButton, QAbstractItemView,
    QSplitter, QListView,
)
from PyQt5.QtCore import Qt, pyqtSignal
from collections import Counter
from datetime import datetime, timezone
import math

from ..config import LAB_SYNONYMS
from ..extraction.normalizer import LabNormalizer
from ..utils.date_utils import parse_italian_date

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
        # Use an explicit list popup on macOS too; selection always refers to
        # a model item, never to uncommitted editable text.
        self._parameter_combo.setView(QListView())
        self._parameter_combo.setMaxVisibleItems(20)
        self._parameter_combo.currentIndexChanged.connect(self._on_parameter_changed)
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

        self._series_combo = QComboBox()
        self._series_combo.setToolTip("Unità di misura e materiale della serie temporale")
        self._series_combo.currentIndexChanged.connect(self._update_chart)
        selector_layout.addWidget(self._series_combo)

        self._stats_label = QLabel("")
        selector_layout.addWidget(self._stats_label)

        layout.addLayout(selector_layout)

        self._chart_status = QLabel()
        self._chart_status.setWordWrap(True)
        layout.addWidget(self._chart_status)

        # Splitter: chart on top, table below
        self._splitter = QSplitter(Qt.Vertical)

        # Chart widget
        self._chart_widget = None
        if HAS_PYQTGRAPH:
            self._chart_widget = pg.PlotWidget(
                axisItems={"bottom": pg.DateAxisItem(orientation="bottom", utcOffset=0)}
            )
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
        if patient_id != self._current_patient_id:
            self._current_parameter = None
        self._current_patient_id = patient_id
        self._load_data()
        self._populate_parameter_combo()
        self._populate_series_combo()
        self._refresh_table()
        self._update_chart()

    def _load_data(self):
        lab_repo = self._services.get("lab_repo")
        if lab_repo and self._current_patient_id:
            self._all_lab_values = lab_repo.get_by_patient(self._current_patient_id)
        else:
            self._all_lab_values = []

    @staticmethod
    def _parameter_key(value):
        return (value.normalized_name or "").strip() or LabNormalizer().normalize_parameter(
            value.parameter_name, value.unit, value.biological_material
        ) or "analita_non_specificato"

    @staticmethod
    def _series_key(value):
        return ((value.unit or "").strip(),
                (value.biological_material or "").strip().casefold())

    @staticmethod
    def _parameter_label(key):
        """Keep familiar abbreviations visible alongside the stored name."""
        name = key.replace("_", " ")
        abbreviations = sorted({alias.upper() for alias, canonical in LAB_SYNONYMS.items()
                                if canonical == key and alias != key
                                and alias.isalnum() and len(alias) <= 5})
        return f"{' / '.join(abbreviations)} — {name}" if abbreviations else name

    def _populate_parameter_combo(self):
        counts = Counter(self._parameter_key(v) for v in self._all_lab_values)
        self._parameter_combo.blockSignals(True)
        try:
            self._parameter_combo.clear()
            self._parameter_combo.addItem("— Tutti i parametri —", "")
            for key in sorted(counts, key=lambda k: self._parameter_label(k).casefold()):
                self._parameter_combo.addItem(
                    f"{self._parameter_label(key)} ({counts[key]} valori)", key
                )
            index = self._parameter_combo.findData(self._current_parameter)
            self._parameter_combo.setCurrentIndex(max(0, index))
            self._current_parameter = self._parameter_combo.currentData()
        finally:
            self._parameter_combo.blockSignals(False)

    def _populate_series_combo(self):
        previous = self._series_combo.currentData()
        keys = sorted({self._series_key(v) for v in self._all_lab_values
                       if self._parameter_key(v) == self._current_parameter})
        self._series_combo.blockSignals(True)
        try:
            self._series_combo.clear()
            for unit, material in keys:
                self._series_combo.addItem(
                    f"{unit or 'Unità non indicata'} · {material or 'Materiale non indicato'}",
                    (unit, material),
                )
            index = next((i for i, key in enumerate(keys) if key == previous), -1)
            self._series_combo.setCurrentIndex(max(0, index) if keys else -1)
            self._series_combo.setVisible(bool(keys))
        finally:
            self._series_combo.blockSignals(False)

    def _on_parameter_changed(self, index):
        self._current_parameter = self._parameter_combo.currentData()
        self._populate_series_combo()
        self._refresh_table()
        self._update_chart()

    def _refresh_table(self):
        """Filter and display lab values."""
        values = list(self._all_lab_values)

        # Filter by parameter
        if self._current_parameter:
            values = [v for v in values
                      if self._parameter_key(v) == self._current_parameter]

        # Filter abnormal only
        if self._abnormal_only_btn.isChecked():
            values = [v for v in values if v.is_abnormal]

        # Sort by date
        values.sort(key=lambda v: self._timestamp(v.sample_date) or float("-inf"))

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

    @staticmethod
    def _timestamp(value):
        text = str(value or "").strip()
        if not text:
            return None
        try:
            dt = datetime.fromisoformat(text)
        except ValueError:
            parsed = parse_italian_date(text)
            if not parsed:
                return None
            dt = datetime.fromisoformat(parsed)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.timestamp()

    def _update_chart(self, *_):
        """Connect only numeric, dated observations in a comparable series."""
        if self._chart_widget is None:
            self._chart_status.setText("Grafici non disponibili: installare pyqtgraph.")
            return
        self._chart_widget.clear()
        self._chart_widget.setTitle("")
        self._chart_widget.setLabel("left", "Valore")
        if not self._all_lab_values:
            self._chart_status.setText("Nessun valore di laboratorio disponibile per questo paziente.")
            return
        if not self._current_parameter:
            self._chart_status.setText("Seleziona un analita nel menu per visualizzare l’andamento temporale.")
            return
        values = [v for v in self._all_lab_values
                  if self._parameter_key(v) == self._current_parameter
                  and self._series_key(v) == self._series_combo.currentData()]
        points = []
        for value in values:
            timestamp = self._timestamp(value.sample_date)
            if timestamp is None or value.value is None or value.operator:
                continue
            try:
                numeric = float(value.value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(numeric):
                points.append((timestamp, numeric, value))
        points.sort(key=lambda p: p[0])
        omitted = len(values) - len(points)
        self._chart_status.setText(
            f"{len(points)} misure nel grafico; {omitted} escluse "
            "(data assente/non valida, risultato non numerico o con soglia < / >). "
            "I trattini rossi indicano i limiti di riferimento del singolo referto. "
            "Il filtro «Solo anomali» riguarda la tabella."
        )
        if not points:
            return
        unit, _ = self._series_combo.currentData()
        self._chart_widget.setTitle(self._parameter_combo.currentText())
        self._chart_widget.setLabel("left", "Valore", units=unit or None)
        self._chart_widget.plot(
            [p[0] for p in points], [p[1] for p in points],
            pen=pg.mkPen(52, 152, 219, width=2), symbol="o", symbolSize=9,
            symbolBrush=pg.mkBrush(52, 152, 219),
        )
        # Reference limits may differ across reports: do not apply the first
        # report's interval to every later observation.
        for attribute in ("reference_low", "reference_high"):
            refs = [(x, getattr(v, attribute)) for x, _, v in points
                    if getattr(v, attribute) is not None
                    and math.isfinite(getattr(v, attribute))]
            if refs:
                self._chart_widget.plot(
                    [p[0] for p in refs], [p[1] for p in refs],
                    pen=None, symbol="_", symbolSize=14,
                    symbolPen=pg.mkPen(231, 76, 60),
                )
        self._chart_widget.enableAutoRange()
        if len({p[0] for p in points}) == 1:
            self._chart_widget.setXRange(points[0][0] - 43200, points[0][0] + 43200)

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

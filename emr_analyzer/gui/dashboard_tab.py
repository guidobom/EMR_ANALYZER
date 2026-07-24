"""Dashboard tab — summary overview of the patient workspace."""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QGroupBox,
    QGridLayout, QFrame,
)
from PyQt6.QtCore import Qt


class StatCard(QFrame):
    """A single KPI card for the dashboard."""

    def __init__(self, title: str, value: str, color: str = "#2c3e50",
                 parent=None):
        super().__init__(parent)
        self.setFrameStyle(QFrame.Shape.Box | QFrame.Shadow.Raised)
        self.setStyleSheet(
            f"QFrame {{ border: 1px solid #dcdde1; border-radius: 8px; "
            f"background-color: #ffffff; padding: 12px; }}"
        )

        layout = QVBoxLayout(self)
        title_label = QLabel(title)
        title_label.setStyleSheet("color: #7f8c8d; font-size: 12px;")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title_label)

        value_label = QLabel(str(value))
        value_label.setStyleSheet(
            f"color: {color}; font-size: 28px; font-weight: bold;"
        )
        value_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(value_label)


class DashboardTab(QWidget):
    """Dashboard showing key patient metrics."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._services = {}
        self._current_patient_id = None
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        # Patient info header
        self._info_label = QLabel("Seleziona un paziente per visualizzare il dashboard")
        self._info_label.setObjectName("heading")
        layout.addWidget(self._info_label)

        # Stats grid
        self._stats_grid = QGridLayout()
        self._stats_grid.setSpacing(12)
        layout.addLayout(self._stats_grid)

        # Placeholder stat cards
        self._cards = {}
        card_specs = [
            ("doc_card", "Documenti", "0", "#2c3e50"),
            ("event_card", "Eventi clinici", "0", "#2980b9"),
            ("lab_card", "Valori laboratorio", "0", "#27ae60"),
            ("diagnosis_card", "Diagnosi attive", "0", "#e74c3c"),
            ("treatment_card", "Terapie attive", "0", "#8e44ad"),
            ("admission_card", "Ricoveri", "0", "#f39c12"),
            ("valid_pending_card", "Da validare", "0", "#e67e22"),
            ("last_update_card", "Ultimo aggiorn.", "—", "#7f8c8d"),
        ]
        for i, (key, title, value, color) in enumerate(card_specs):
            card = StatCard(title, value, color)
            self._cards[key] = card
            self._stats_grid.addWidget(card, i // 4, i % 4)

        layout.addStretch()

    def set_services(self, services: dict):
        self._services = services

    def load_patient(self, patient_id: str):
        self._current_patient_id = patient_id
        self._refresh()

    def _refresh(self):
        if not self._current_patient_id:
            return

        # Get patient info
        patient_repo = self._services.get("patient_repo")
        if patient_repo:
            patient = patient_repo.get_by_id(self._current_patient_id)
            if patient:
                self._info_label.setText(
                    f"Dashboard — {patient.pseudonym} ({self._current_patient_id})"
                )

        # Document count
        doc_repo = self._services.get("document_repo")
        if doc_repo:
            doc_count = doc_repo.get_count_by_patient(self._current_patient_id)
            self._cards["doc_card"].findChildren(QLabel)[1].setText(str(doc_count))

        # Event count
        event_repo = self._services.get("event_repo")
        if event_repo:
            ev_count = event_repo.count_by_patient(self._current_patient_id)
            self._cards["event_card"].findChildren(QLabel)[1].setText(str(ev_count))

        # Lab count
        lab_repo = self._services.get("lab_repo")
        if lab_repo:
            lab_count = lab_repo.get_abnormal_count(self._current_patient_id)
            # We don't have a total count method; use abnormal as indicator
            self._cards["lab_card"].findChildren(QLabel)[1].setText(str(lab_count))

        # Clinical State
        cs_repo = self._services.get("cs_repo")
        if cs_repo:
            cs = cs_repo.load(self._current_patient_id)
            if cs:
                self._cards["diagnosis_card"].findChildren(QLabel)[1].setText(
                    str(len(cs.active_diagnoses)))
                self._cards["treatment_card"].findChildren(QLabel)[1].setText(
                    str(len(cs.active_treatments)))
                self._cards["last_update_card"].findChildren(QLabel)[1].setText(
                    cs.updated_at[:10] if cs.updated_at else "—")

        # Validation queue
        # (simplified — would need validation_repo)
        self._cards["valid_pending_card"].findChildren(QLabel)[1].setText("—")

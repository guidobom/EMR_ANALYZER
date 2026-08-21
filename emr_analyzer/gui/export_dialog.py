"""Export dialog — select format and content for data export."""

from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QRadioButton, QCheckBox,
    QPushButton, QLabel, QFileDialog, QButtonGroup, QGroupBox,
    QMessageBox,
)
from PyQt5.QtCore import Qt


class ExportDialog(QDialog):
    """Dialog for exporting extraction results."""

    def __init__(self, services: dict, patient_id: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Esporta Dati")
        self.setMinimumWidth(450)
        self._services = services
        self._patient_id = patient_id
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        # Format selection
        format_group = QGroupBox("Formato")
        format_layout = QVBoxLayout()

        self._format_group = QButtonGroup(self)
        self._csv_radio = QRadioButton("CSV — Tabella piatta, compatibile con Excel")
        self._json_radio = QRadioButton("JSON — Formato nidificato completo")
        self._xlsx_radio = QRadioButton("Excel (XLSX) — Multi-foglio")
        self._csv_radio.setChecked(True)

        self._format_group.addButton(self._csv_radio, 1)
        self._format_group.addButton(self._json_radio, 2)
        self._format_group.addButton(self._xlsx_radio, 3)

        format_layout.addWidget(self._csv_radio)
        format_layout.addWidget(self._json_radio)
        format_layout.addWidget(self._xlsx_radio)
        format_group.setLayout(format_layout)
        layout.addWidget(format_group)

        # Content selection
        content_group = QGroupBox("Contenuto")
        content_layout = QVBoxLayout()

        self._lab_check = QCheckBox("Valori di laboratorio")
        self._lab_check.setChecked(True)
        content_layout.addWidget(self._lab_check)

        self._events_check = QCheckBox("Registro cronologico")
        self._events_check.setChecked(True)
        content_layout.addWidget(self._events_check)

        self._state_check = QCheckBox("Profilo clinico narrativo")
        self._state_check.setChecked(True)
        content_layout.addWidget(self._state_check)

        self._sources_check = QCheckBox("Includi fonti (testo originale)")
        self._sources_check.setChecked(False)
        content_layout.addWidget(self._sources_check)

        content_group.setLayout(content_layout)
        layout.addWidget(content_group)

        # File path
        path_layout = QHBoxLayout()
        path_layout.addWidget(QLabel("Destinazione:"))
        self._path_label = QLabel("(seleziona file)")
        self._path_label.setStyleSheet("color: #7f8c8d;")
        self._path_label.setWordWrap(True)
        path_layout.addWidget(self._path_label, stretch=1)
        browse_btn = QPushButton("Sfoglia...")
        browse_btn.clicked.connect(self._browse)
        path_layout.addWidget(browse_btn)
        layout.addLayout(path_layout)

        # Buttons
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        export_btn = QPushButton("Esporta")
        export_btn.setDefault(True)
        export_btn.clicked.connect(self._export)
        cancel_btn = QPushButton("Annulla")
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(export_btn)
        btn_layout.addWidget(cancel_btn)
        layout.addLayout(btn_layout)

    def _browse(self):
        ext = "csv"
        if self._json_radio.isChecked():
            ext = "json"
        elif self._xlsx_radio.isChecked():
            ext = "xlsx"

        file_path, _ = QFileDialog.getSaveFileName(
            self, "Salva Esportazione",
            f"emr_export_{self._patient_id}.{ext}",
            f"CSV (*.csv);;JSON (*.json);;Excel (*.xlsx);;Tutti i file (*)"
        )
        if file_path:
            self._path_label.setText(file_path)

    def _export(self):
        file_path = self._path_label.text()
        if file_path == "(seleziona file)":
            QMessageBox.warning(self, "Attenzione",
                                "Seleziona un file di destinazione.")
            return

        try:
            from ..export.registry_export import ClinicalRegistryExporter
            ClinicalRegistryExporter(self._services).export(
                self._patient_id,
                file_path,
                include_sources=self._sources_check.isChecked(),
                include_events=self._events_check.isChecked(),
                include_labs=self._lab_check.isChecked(),
                include_profile=self._state_check.isChecked(),
            )

            QMessageBox.information(self, "Esportazione completata",
                                    f"File salvato con successo:\n{file_path}")
            self.accept()
        except Exception as e:
            QMessageBox.critical(self, "Errore esportazione", str(e))

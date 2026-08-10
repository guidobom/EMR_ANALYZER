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
            if self._csv_radio.isChecked():
                self._export_csv(file_path)
            elif self._json_radio.isChecked():
                self._export_json(file_path)
            elif self._xlsx_radio.isChecked():
                self._export_xlsx(file_path)

            QMessageBox.information(self, "Esportazione completata",
                                    f"File salvato con successo:\n{file_path}")
            self.accept()
        except Exception as e:
            QMessageBox.critical(self, "Errore esportazione", str(e))

    def _export_csv(self, file_path: str):
        import csv
        with open(file_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            # Lab values
            if self._lab_check.isChecked():
                lab_repo = self._services.get("lab_repo")
                if lab_repo:
                    values = lab_repo.get_by_patient(self._patient_id)
                    writer.writerow([
                        "Data", "Parametro", "Valore", "Unità", "Range",
                        "Anomalo", "Flag", "Confidenza", "Documento", "Pagina"
                    ])
                    for lv in values:
                        display_value = (
                            lv.value_text if lv.value_text
                            else f"{lv.operator or ''}{lv.value}".strip()
                            if lv.value is not None else ""
                        )
                        writer.writerow([
                            lv.sample_date, lv.parameter_name, display_value,
                            lv.unit, lv.reference_text,
                            "Sì" if lv.is_abnormal else "No",
                            lv.flag, f"{lv.confidence:.2f}",
                            lv.document_id, lv.page,
                        ])
                        if not self._sources_check.isChecked():
                            pass  # source_text column skipped

            # Registro Cronologico
            if self._events_check.isChecked():
                writer.writerow([])
                timeline_repo = self._services.get("timeline_repo")
                if timeline_repo:
                    entries = timeline_repo.get_by_patient(self._patient_id)
                    writer.writerow([
                        "Data", "Categoria", "Descrizione", "Stato",
                        "Confidenza", "Documenti",
                    ])
                    for e in entries:
                        row = [
                            e.date_observed, e.category, e.description,
                            e.status, f"{e.confidence:.2f}",
                            ", ".join(e.source_document_ids),
                        ]
                        if self._sources_check.isChecked():
                            row.append("\n".join(e.source_texts[:3]))
                        writer.writerow(row)

    def _export_json(self, file_path: str):
        import json
        result = {"patient_id": self._patient_id}

        if self._lab_check.isChecked():
            lab_repo = self._services.get("lab_repo")
            if lab_repo:
                result["lab_values"] = [lv.to_dict()
                                        for lv in lab_repo.get_by_patient(self._patient_id)]

        if self._events_check.isChecked():
            timeline_repo = self._services.get("timeline_repo")
            if timeline_repo:
                result["clinical_timeline"] = [e.to_dict()
                    for e in timeline_repo.get_by_patient(self._patient_id)]

        if self._state_check.isChecked():
            cs_repo = self._services.get("cs_repo")
            if cs_repo:
                cs = cs_repo.load(self._patient_id)
                if cs and cs.clinical_profile:
                    result["clinical_profile"] = cs.clinical_profile

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

    def _export_xlsx(self, file_path: str):
        from openpyxl import Workbook
        wb = Workbook()

        # Lab values sheet
        if self._lab_check.isChecked():
            ws = wb.active
            ws.title = "Laboratorio"
            ws.append([
                "Data", "Parametro", "Valore", "Unità", "Range",
                "Anomalo", "Flag", "Confidenza", "Documento", "Pagina", "Fonte"
            ])
            lab_repo = self._services.get("lab_repo")
            if lab_repo:
                for lv in lab_repo.get_by_patient(self._patient_id):
                    display_value = (
                        lv.value_text if lv.value_text
                        else f"{lv.operator or ''}{lv.value}".strip()
                        if lv.value is not None else ""
                    )
                    row = [
                        lv.sample_date, lv.parameter_name, display_value,
                        lv.unit, lv.reference_text,
                        "Sì" if lv.is_abnormal else "No",
                        lv.flag, lv.confidence,
                        lv.document_id, lv.page,
                        lv.source_text if self._sources_check.isChecked() else "",
                    ]
                    ws.append(row)

        # Registro Cronologico sheet
        if self._events_check.isChecked():
            ws = wb.create_sheet("Registro Cronologico")
            ws.append([
                "Data", "Categoria", "Descrizione", "Stato",
                "Confidenza", "Documenti", "Fonti"
            ])
            timeline_repo = self._services.get("timeline_repo")
            if timeline_repo:
                for e in timeline_repo.get_by_patient(self._patient_id):
                    row = [
                        e.date_observed, e.category, e.description,
                        e.status, e.confidence,
                        ", ".join(e.source_document_ids),
                        "\n---\n".join(e.source_texts[:3])
                        if self._sources_check.isChecked() else "",
                    ]
                    ws.append(row)

        # Profilo Clinico sheet
        if self._state_check.isChecked():
            ws = wb.create_sheet("Profilo Clinico")
            cs_repo = self._services.get("cs_repo")
            if cs_repo:
                cs = cs_repo.load(self._patient_id)
                if cs and cs.clinical_profile:
                    ws.append(["Profilo Clinico Narrativo"])
                    ws.append([cs.clinical_profile])

        wb.save(file_path)

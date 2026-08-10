"""Dialog for importing patients from another project."""

from pathlib import Path

from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QTableWidget, QTableWidgetItem, QHeaderView, QFileDialog,
    QMessageBox, QProgressBar, QCheckBox,
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal


class _ImportWorker(QThread):
    """Import selected patients in a background thread."""
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, source_path, patient_ids):
        super().__init__()
        self.source_path = source_path
        self.patient_ids = patient_ids

    def run(self):
        from ..clinical.patient_import import PatientImportService
        service = PatientImportService()
        try:
            result = service.import_patients(
                self.source_path, self.patient_ids,
                progress_callback=lambda pct, msg: self.progress.emit(pct, msg),
            )
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))


class ImportPatientDialog(QDialog):
    """Select patients from a source project to import."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Importa pazienti da progetto")
        self.setMinimumSize(700, 450)
        self._source_path = None
        self._patients = []
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        # Source selection
        src_layout = QHBoxLayout()
        self._src_label = QLabel("Progetto sorgente: (nessuno)")
        src_layout.addWidget(self._src_label, stretch=1)
        browse_btn = QPushButton("📁 Scegli progetto...")
        browse_btn.clicked.connect(self._on_browse)
        src_layout.addWidget(browse_btn)
        layout.addLayout(src_layout)

        # Patient table
        self._table = QTableWidget()
        self._table.setColumnCount(6)
        self._table.setHorizontalHeaderLabels([
            "", "Paziente", "Iniziali", "Documenti", "Voci registro", "Profilo"
        ])
        self._table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.Stretch
        )
        self._table.setEditTriggers(QTableWidget.NoEditTriggers)
        layout.addWidget(self._table, stretch=1)

        # Progress
        self._progress = QProgressBar()
        self._progress.setVisible(False)
        layout.addWidget(self._progress)

        # Buttons
        btn_layout = QHBoxLayout()
        self._select_all_btn = QPushButton("Seleziona tutti")
        self._select_all_btn.clicked.connect(lambda: self._toggle_all(True))
        self._select_all_btn.setEnabled(False)
        btn_layout.addWidget(self._select_all_btn)

        self._deselect_all_btn = QPushButton("Deseleziona tutti")
        self._deselect_all_btn.clicked.connect(lambda: self._toggle_all(False))
        self._deselect_all_btn.setEnabled(False)
        btn_layout.addWidget(self._deselect_all_btn)

        btn_layout.addStretch()

        self._import_btn = QPushButton("📥 Importa selezionati")
        self._import_btn.clicked.connect(self._on_import)
        self._import_btn.setEnabled(False)
        btn_layout.addWidget(self._import_btn)

        close_btn = QPushButton("Chiudi")
        close_btn.clicked.connect(self.reject)
        btn_layout.addWidget(close_btn)

        layout.addLayout(btn_layout)

    def _on_browse(self):
        folder = QFileDialog.getExistingDirectory(
            self, "Seleziona cartella progetto EMR Analyzer",
            str(Path.home()),
        )
        if not folder:
            return

        from ..clinical.patient_import import PatientImportService
        service = PatientImportService()
        try:
            self._patients = service.list_source_patients(folder)
        except Exception as e:
            QMessageBox.warning(self, "Errore", str(e))
            return

        self._source_path = folder
        self._src_label.setText(f"Progetto sorgente: {folder}")
        self._populate_table()

    def _populate_table(self):
        self._table.setRowCount(len(self._patients))
        for i, p in enumerate(self._patients):
            chk = QCheckBox()
            chk.setChecked(True)
            self._table.setCellWidget(i, 0, chk)

            self._table.setItem(i, 1, QTableWidgetItem(
                f"{p['id']} — {p['pseudonym']}"
            ))
            self._table.setItem(i, 2, QTableWidgetItem(
                f"{p.get('initials', '')} {p.get('sex', '')} {p.get('birth_year', '')}"
            ))
            self._table.setItem(i, 3, QTableWidgetItem(str(p['document_count'])))
            self._table.setItem(i, 4, QTableWidgetItem(str(p['timeline_entries'])))
            self._table.setItem(
                i, 5, QTableWidgetItem("✓" if p['has_profile'] else "—")
            )

        self._select_all_btn.setEnabled(True)
        self._deselect_all_btn.setEnabled(True)
        self._import_btn.setEnabled(True)

    def _toggle_all(self, checked: bool):
        for i in range(self._table.rowCount()):
            widget = self._table.cellWidget(i, 0)
            if isinstance(widget, QCheckBox):
                widget.setChecked(checked)

    def _get_selected_ids(self) -> list[str]:
        ids = []
        for i in range(self._table.rowCount()):
            widget = self._table.cellWidget(i, 0)
            if isinstance(widget, QCheckBox) and widget.isChecked():
                ids.append(self._patients[i]["id"])
        return ids

    def _on_import(self):
        selected = self._get_selected_ids()
        if not selected:
            QMessageBox.information(self, "Nessuno", "Seleziona almeno un paziente.")
            return

        reply = QMessageBox.question(
            self, "Conferma importazione",
            f"Importare {len(selected)} paziente/i dal progetto sorgente?\n\n"
            f"Verranno copiati: PDF, testi estratti, registro cronologico "
            f"e profilo narrativo.",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return

        self._import_btn.setEnabled(False)
        self._progress.setVisible(True)
        self._progress.setValue(0)

        self._worker = _ImportWorker(self._source_path, selected)
        self._worker.progress.connect(
            lambda pct, msg: (self._progress.setValue(pct),)
        )
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _on_finished(self, stats: dict):
        self._progress.setVisible(False)
        QMessageBox.information(
            self, "Importazione completata",
            f"Importati:\n"
            f"• {stats['patients']} pazienti\n"
            f"• {stats['documents']} documenti\n"
            f"• {stats['timeline']} voci di registro\n"
            f"• {stats['profiles']} profili narrativi"
        )
        self.accept()

    def _on_error(self, error: str):
        self._progress.setVisible(False)
        self._import_btn.setEnabled(True)
        QMessageBox.critical(self, "Errore importazione", error)

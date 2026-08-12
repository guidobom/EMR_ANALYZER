"""Batch import dialog — one window to confirm imports for several patients.

Shows one tree row per patient workspace (with a checkbox) and, expanded
underneath, the file list for that patient (per-file checkbox + type
override).  Two actions: import the checked files, optionally followed by the
sequential extraction queue over the patients that received documents.
"""

import os

from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QTreeWidget, QTreeWidgetItem, QComboBox, QMessageBox, QHeaderView,
)
from PyQt5.QtCore import Qt

from ..models.document import DocumentType
from .import_dialog import run_file_checks, import_checked_documents


class BatchImportDialog(QDialog):
    """Confirm the import of several patients' documents in one window."""

    def __init__(self, patient_files: dict[str, list[str]], services: dict,
                 parent=None):
        super().__init__(parent)
        self.setWindowTitle("Importazione Documenti — più pazienti")
        self.setMinimumSize(920, 580)
        self._services = services
        self._patient_files = patient_files
        self._patient_repo = services.get("patient_repo")
        self._file_checks: dict[str, list[dict]] = {}
        self._type_combos: dict[tuple[str, int], QComboBox] = {}
        self._syncing = False

        # Public API consumed by workspace_tabs after accept()
        self.imported_by_patient: dict[str, list[str]] = {}
        self.queue_patient_ids: list[str] = []
        self.should_run_queue = False

        self._setup_ui()
        self._load_patients()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        header = QLabel(
            f"Importazione documenti di {len(self._patient_files)} pazienti — "
            "spunta i workspace da importare"
        )
        header.setObjectName("heading")
        layout.addWidget(header)

        self._tree = QTreeWidget()
        self._tree.setColumnCount(6)
        self._tree.setHeaderLabels([
            "Importa", "Nome / File", "Pagine", "Testo", "Tipo", "Stato",
        ])
        self._tree.header().setStretchLastSection(True)
        self._tree.header().setSectionResizeMode(1, QHeaderView.Stretch)
        self._tree.setAlternatingRowColors(True)
        self._tree.itemChanged.connect(self._on_item_changed)
        layout.addWidget(self._tree, stretch=1)

        buttons = QHBoxLayout()
        buttons.addStretch()
        cancel_btn = QPushButton("Annulla")
        cancel_btn.clicked.connect(self.reject)
        import_btn = QPushButton("Importa")
        import_btn.clicked.connect(lambda: self._on_import(run_queue=False))
        import_extract_btn = QPushButton("⚡ Importa ed estrai")
        import_extract_btn.setToolTip(
            "Importa i documenti spuntati e avvia subito la coda di "
            "estrazione per i workspace che hanno ricevuto file"
        )
        import_extract_btn.setDefault(True)
        import_extract_btn.clicked.connect(
            lambda: self._on_import(run_queue=True)
        )
        buttons.addWidget(cancel_btn)
        buttons.addWidget(import_btn)
        buttons.addWidget(import_extract_btn)
        layout.addLayout(buttons)

    # --- population -----------------------------------------------------

    def _load_patients(self):
        for pid, paths in self._patient_files.items():
            checks = run_file_checks(self._services, paths)
            self._file_checks[pid] = checks
            patient = (
                self._patient_repo.get_by_id(pid) if self._patient_repo else None
            )

            parent = QTreeWidgetItem(self._tree)
            parent.setText(0, str(pid))
            parent.setText(1, self._patient_label(pid, patient))
            parent.setText(2, str(len(checks)))
            parent.setText(
                3, str(sum(1 for c in checks if c.get("is_duplicate")))
            )
            parent.setCheckState(0, Qt.Checked)
            parent.setExpanded(True)
            parent.setData(0, Qt.UserRole, pid)

            for i, check_data in enumerate(checks):
                child = QTreeWidgetItem(parent)
                if "check" not in check_data:
                    child.setText(0, "")
                    child.setText(1, os.path.basename(check_data["path"]))
                    child.setText(5, "❌ Formato non supportato")
                    child.setDisabled(True)
                    continue
                child.setText(1, check_data.get("original_name")
                              or os.path.basename(check_data["path"]))
                child.setText(2, str(check_data["check"].get("page_count", 1)))
                child.setText(
                    3, "Sì" if check_data["check"].get("has_text") else "No"
                )
                child.setText(
                    5, "⚠️ Duplicato" if check_data["is_duplicate"] else "Pronto"
                )
                child.setCheckState(
                    0, Qt.Unchecked if check_data["is_duplicate"] else Qt.Checked
                )
                child.setDisabled(bool(check_data["is_duplicate"]))
                child.setData(0, Qt.UserRole, (pid, i))

                combo = QComboBox()
                for dt in DocumentType:
                    combo.addItem(dt.value, dt.value)
                combo.setCurrentText(check_data["guessed_type"])
                self._tree.setItemWidget(child, 4, combo)
                self._type_combos[(pid, i)] = combo

    @staticmethod
    def _patient_label(pid: str, patient) -> str:
        bits = [str(pid)]
        if patient:
            if getattr(patient, "initials", None):
                bits.append(patient.initials)
            if getattr(patient, "sex", None):
                bits.append(patient.sex)
            if getattr(patient, "birth_year", None):
                bits.append(str(patient.birth_year))
        return " · ".join(bits)

    # --- interactions ---------------------------------------------------

    def _on_item_changed(self, item: QTreeWidgetItem, column: int):
        """Propagate a patient-level checkbox to its file rows."""
        if column != 0 or self._syncing or item.parent() is not None:
            return
        state = item.checkState(0)
        self._syncing = True
        try:
            for i in range(item.childCount()):
                child = item.child(i)
                if not child.isDisabled():
                    child.setCheckState(0, state)
        finally:
            self._syncing = False

    def _make_status_callback(self, parent: QTreeWidgetItem):
        def cb(index: int, message: str):
            child = parent.child(index)
            if child:
                child.setText(5, message)
        return cb

    def _on_import(self, run_queue: bool):
        total_imported = 0
        for i in range(self._tree.topLevelItemCount()):
            parent = self._tree.topLevelItem(i)
            if parent.checkState(0) != Qt.Checked:
                continue
            pid = parent.data(0, Qt.UserRole)
            checks = self._file_checks[pid]

            selected = []
            type_overrides = {}
            for j in range(parent.childCount()):
                child = parent.child(j)
                if child.checkState(0) != Qt.Checked:
                    continue
                selected.append(j)
                combo = self._type_combos.get((pid, j))
                if combo:
                    type_overrides[j] = combo.currentText()

            try:
                imported = import_checked_documents(
                    self._services, pid, checks, selected,
                    status_callback=self._make_status_callback(parent),
                    type_overrides=type_overrides,
                )
            except Exception as exc:
                QMessageBox.critical(
                    self, "Importazione non riuscita",
                    f"Errore per il paziente {pid}:\n{exc}",
                )
                continue
            self.imported_by_patient[pid] = imported
            total_imported += len(imported)

        if total_imported == 0:
            QMessageBox.information(
                self, "Nessun import",
                "Nessun file selezionato per l'importazione.",
            )
            return

        self.queue_patient_ids = [
            pid for pid in self._patient_files
            if self.imported_by_patient.get(pid)
        ]
        self.should_run_queue = run_queue
        self.accept()

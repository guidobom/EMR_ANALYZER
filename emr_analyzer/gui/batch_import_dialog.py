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
from .pdf_viewer import PDFViewerDialog
from .quick_look import QuickLook


class BatchImportDialog(QDialog):
    """Confirm the import of several patients' documents in one window."""

    _UNASSIGNED = "__unassigned__"

    def __init__(self, patient_files: dict[str, list[str]], services: dict,
                 parent=None, unassigned_files: list[str] | None = None,
                 candidate_pids: list[str] | None = None):
        super().__init__(parent)
        self.setWindowTitle("Importazione Documenti — più pazienti")
        self.setMinimumSize(1020, 580)
        self._services = services
        self._patient_files = patient_files
        self._patient_repo = services.get("patient_repo")
        self._file_checks: dict[str, list[dict]] = {}
        self._type_combos: dict[tuple[str, int], QComboBox] = {}
        self._unassigned_files = list(unassigned_files or [])
        self._candidate_pids = list(candidate_pids or [])
        self._unassigned_checks: list[dict] = []
        self._unassigned_target_combos: dict[int, QComboBox] = {}
        self._unassigned_type_combos: dict[int, QComboBox] = {}
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
        self._tree.setColumnCount(7)
        self._tree.setHeaderLabels([
            "Importa", "Nome / File", "Pagine", "Testo", "Tipo", "Stato",
            "Assegna a",
        ])
        self._tree.header().setStretchLastSection(True)
        self._tree.header().setSectionResizeMode(1, QHeaderView.Stretch)
        self._tree.setAlternatingRowColors(True)
        self._tree.itemChanged.connect(self._on_item_changed)
        # Double-click opens the full viewer; spacebar opens an in-app Quick
        # Look preview of the current file (Space/Esc again closes it). The
        # overlay owns the spacebar key via its own event filter on the tree.
        self._tree.itemDoubleClicked.connect(self._on_item_double_clicked)
        self._quick_look = QuickLook(self._tree, self._quick_look_path)
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
                child.setData(0, Qt.UserRole + 1, check_data["path"])
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

        if self._unassigned_files:
            self._load_unassigned_bucket()

    def _load_unassigned_bucket(self):
        """Rows for documents no automatic rule could assign to a patient.

        Each row offers a target-patient combo (default: non importare) so
        the user can assign files that routing could not, without a single
        review dialog per file.
        """
        checks = run_file_checks(self._services, self._unassigned_files)
        self._unassigned_checks = checks
        parent = QTreeWidgetItem(self._tree)
        parent.setText(1, f"⚠️ Documenti senza paziente ({len(checks)})")
        parent.setText(2, str(len(checks)))
        parent.setText(5, "Da assegnare")
        parent.setExpanded(True)
        parent.setData(0, Qt.UserRole, self._UNASSIGNED)
        # Header row only: no checkbox, the user acts on individual files.
        parent.setFlags(parent.flags() & ~Qt.ItemIsUserCheckable)

        for i, check_data in enumerate(checks):
            child = QTreeWidgetItem(parent)
            child.setData(0, Qt.UserRole + 1, check_data["path"])
            if "check" not in check_data:
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
            child.setCheckState(0, Qt.Unchecked)
            child.setDisabled(bool(check_data["is_duplicate"]))
            child.setData(0, Qt.UserRole, (self._UNASSIGNED, i))

            type_combo = QComboBox()
            for dt in DocumentType:
                type_combo.addItem(dt.value, dt.value)
            type_combo.setCurrentText(check_data["guessed_type"])
            self._tree.setItemWidget(child, 4, type_combo)
            self._unassigned_type_combos[i] = type_combo

            target_combo = QComboBox()
            for pid in self._candidate_pids:
                target_combo.addItem(str(pid), str(pid))
            target_combo.addItem("(non importare)", None)
            target_combo.setCurrentIndex(target_combo.count() - 1)
            self._tree.setItemWidget(child, 6, target_combo)
            self._unassigned_target_combos[i] = target_combo

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

    def _on_item_double_clicked(self, item: QTreeWidgetItem, column: int):
        """Open the PDF viewer for a file row (double-click)."""
        if column == 0:
            return  # column 0 is the checkbox; let it toggle
        self._open_file(item)

    def _open_file(self, item: QTreeWidgetItem | None) -> bool:
        """Open the file of a row in the PDF viewer. False when not a file."""
        if item is None or item.parent() is None:
            return False
        path = item.data(0, Qt.UserRole + 1)
        if not path or not os.path.exists(str(path)):
            return False
        self._quick_look.dismiss()
        viewer = PDFViewerDialog(
            {"filename": os.path.basename(str(path)), "original_path": str(path)},
            self._services,
            self,
        )
        viewer.exec_()
        return True

    def _quick_look_path(self):
        """Resolve the current tree row to a PDF path for the Quick Look."""
        item = self._tree.currentItem()
        if item is None or item.parent() is None:
            return None  # a patient/unassigned header row, not a file
        path = item.data(0, Qt.UserRole + 1)
        if not path or not os.path.exists(str(path)):
            return None
        return str(path), os.path.basename(str(path))

    def _make_status_callback(self, parent: QTreeWidgetItem):
        def cb(index: int, message: str):
            child = parent.child(index)
            if child:
                child.setText(5, message)
        return cb

    def _import_unassigned(self, parent: QTreeWidgetItem) -> int:
        """Import checked "Da assegnare" rows into their chosen patients."""
        by_target: dict[str, list[int]] = {}
        type_overrides: dict[int, str] = {}
        for j in range(parent.childCount()):
            child = parent.child(j)
            if child.checkState(0) != Qt.Checked:
                continue
            target_combo = self._unassigned_target_combos.get(j)
            target = target_combo.currentData() if target_combo else None
            if not target:
                continue
            by_target.setdefault(target, []).append(j)
            type_combo = self._unassigned_type_combos.get(j)
            if type_combo:
                type_overrides[j] = type_combo.currentText()

        imported_count = 0
        for target, indexes in by_target.items():
            try:
                imported = import_checked_documents(
                    self._services, target, self._unassigned_checks,
                    indexes,
                    status_callback=self._make_status_callback(parent),
                    type_overrides=type_overrides,
                )
            except Exception as exc:
                QMessageBox.critical(
                    self, "Importazione non riuscita",
                    f"Errore per il paziente {target}:\n{exc}",
                )
                continue
            self.imported_by_patient.setdefault(target, []).extend(imported)
            imported_count += len(imported)
        return imported_count

    def _on_import(self, run_queue: bool):
        total_imported = 0
        for i in range(self._tree.topLevelItemCount()):
            parent = self._tree.topLevelItem(i)
            pid = parent.data(0, Qt.UserRole)
            if pid == self._UNASSIGNED:
                total_imported += self._import_unassigned(parent)
                continue
            if parent.checkState(0) != Qt.Checked:
                continue
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
            pid for pid in self.imported_by_patient
            if self.imported_by_patient[pid]
        ]
        self.should_run_queue = run_queue
        self.accept()

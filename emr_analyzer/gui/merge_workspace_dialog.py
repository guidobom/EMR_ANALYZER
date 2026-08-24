"""Dialog to merge one or more patient workspaces into other workspaces.

Workspaces are moved (not copied): after a successful merge the source
workspace is deleted.  Potential duplicate patients (same identity HMAC
keys) are flagged so the user can decide case by case where each source
should go.
"""

from __future__ import annotations

from datetime import datetime

from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QTableWidget, QTableWidgetItem, QHeaderView, QMessageBox,
    QProgressBar, QCheckBox, QComboBox,
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal


class _MergeWorker(QThread):
    """Run the merge in a background thread."""

    progress = pyqtSignal(int, str)
    result_ready = pyqtSignal(list)
    error = pyqtSignal(str)

    def __init__(self, services: dict, pairs: list[tuple[str, str]]):
        super().__init__()
        self._services = services
        self._pairs = pairs

    def run(self):
        from ..clinical.workspace_merge import WorkspaceMergeService
        service = WorkspaceMergeService(
            db=self._services.get("db"),
            patient_repo=self._services.get("patient_repo"),
            document_repo=self._services.get("document_repo"),
            identity_repo=self._services.get("identity_repo"),
            audit_repo=self._services.get("audit_repo"),
            deletion_service=self._services.get("patient_workspace_deletion"),
        )
        try:
            results = service.merge_many(
                self._pairs,
                progress_callback=lambda pct, msg: self.progress.emit(pct, msg),
            )
            self.result_ready.emit(results)
        except Exception as exc:
            self.error.emit(str(exc))


class MergeWorkspaceDialog(QDialog):
    """Select source workspaces and their destinations to merge (move)."""

    def __init__(self, services: dict, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Unisci workspace pazienti")
        self.setMinimumSize(860, 500)
        self._services = services
        self._patients = []
        self._matches = {}      # patient_id -> list of identity match dicts
        self.removed_sources = []
        self._setup_ui()
        self._load_patients()

    def _make_service(self):
        from ..clinical.workspace_merge import WorkspaceMergeService
        return WorkspaceMergeService(
            db=self._services.get("db"),
            patient_repo=self._services.get("patient_repo"),
            document_repo=self._services.get("document_repo"),
            identity_repo=self._services.get("identity_repo"),
            audit_repo=self._services.get("audit_repo"),
            deletion_service=self._services.get("patient_workspace_deletion"),
        )

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        title = QLabel(
            "<h3>Unisci workspace pazienti</h3>"
            "<p>Sposta (non copia) documenti ed elaborazioni dal workspace "
            "sorgente al workspace di destinazione. Il workspace sorgente "
            "verrà eliminato definitivamente.</p>"
        )
        title.setWordWrap(True)
        layout.addWidget(title)

        self._table = QTableWidget()
        self._table.setColumnCount(6)
        self._table.setHorizontalHeaderLabels([
            "Unisci", "Paziente", "Dettagli", "Documenti",
            "Duplicato identità", "Workspace destinazione",
        ])
        header = self._table.horizontalHeader()
        header.setSectionResizeMode(1, QHeaderView.Stretch)
        header.setSectionResizeMode(4, QHeaderView.Stretch)
        self._table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._table.verticalHeader().setVisible(False)
        layout.addWidget(self._table, stretch=1)

        self._progress = QProgressBar()
        self._progress.setVisible(False)
        layout.addWidget(self._progress)

        btn_layout = QHBoxLayout()
        select_all_btn = QPushButton("Unisci tutti")
        select_all_btn.clicked.connect(lambda: self._toggle_all(True))
        btn_layout.addWidget(select_all_btn)

        deselect_all_btn = QPushButton("Deseleziona tutti")
        deselect_all_btn.clicked.connect(lambda: self._toggle_all(False))
        btn_layout.addWidget(deselect_all_btn)

        btn_layout.addStretch()

        self._merge_btn = QPushButton("🔀 Unisci selezionati")
        self._merge_btn.clicked.connect(self._on_merge)
        self._merge_btn.setEnabled(False)
        btn_layout.addWidget(self._merge_btn)

        close_btn = QPushButton("Chiudi")
        close_btn.clicked.connect(self.reject)
        btn_layout.addWidget(close_btn)

        layout.addLayout(btn_layout)

    def _load_patients(self):
        service = self._make_service()
        self._patients = service.list_patients()
        for patient in self._patients:
            self._matches[patient["id"]] = service.find_identity_matches(
                patient["id"]
            )
        self._populate_table()

    def _populate_table(self):
        self._table.setRowCount(len(self._patients))
        for row_index, patient in enumerate(self._patients):
            pid = patient["id"]

            chk = QCheckBox()
            chk.stateChanged.connect(
                lambda state, row=row_index: self._on_check_toggled(row, state)
            )
            self._table.setCellWidget(row_index, 0, chk)

            self._table.setItem(row_index, 1, QTableWidgetItem(
                f"{pid} — {patient.get('pseudonym', '')}"
            ))

            details = patient.get("initials") or ""
            if patient.get("sex"):
                details += f" • {patient['sex']}"
            if patient.get("birth_year"):
                age = datetime.now().year - patient["birth_year"]
                details += f" • {age} aa"
            self._table.setItem(row_index, 2, QTableWidgetItem(details))

            self._table.setItem(row_index, 3, QTableWidgetItem(
                str(patient["document_count"])
            ))

            matches = self._matches.get(pid, [])
            if matches:
                labels = [
                    f"→ {m['patient_id']} ({m['reason']})"
                    + (" [conflitto]" if m["conflict"] else "")
                    for m in matches
                ]
                badge = "⚠ possibile duplicato " + "; ".join(labels)
            else:
                badge = "—"
            badge_item = QTableWidgetItem(badge)
            badge_item.setToolTip(badge)
            self._table.setItem(row_index, 4, badge_item)

            combo = QComboBox()
            combo.addItem("(seleziona destinazione)", "")
            for other in self._patients:
                if other["id"] != pid:
                    combo.addItem(
                        f"{other['id']} — {other.get('pseudonym', '')}",
                        other["id"],
                    )
            clean = [m for m in matches if not m["conflict"]]
            if len(clean) == 1:
                index = combo.findData(clean[0]["patient_id"])
                if index >= 0:
                    combo.setCurrentIndex(index)
            combo.setEnabled(False)
            combo.currentIndexChanged.connect(
                lambda _idx, row=row_index: self._update_merge_button()
            )
            self._table.setCellWidget(row_index, 5, combo)

        self._update_merge_button()

    # ------------------------------------------------------------------
    # Interaction
    # ------------------------------------------------------------------

    def _on_check_toggled(self, row: int, state: int):
        combo = self._table.cellWidget(row, 5)
        if isinstance(combo, QComboBox):
            combo.setEnabled(state == Qt.Checked)
        self._update_merge_button()

    def _toggle_all(self, checked: bool):
        for row in range(self._table.rowCount()):
            widget = self._table.cellWidget(row, 0)
            if isinstance(widget, QCheckBox):
                widget.setChecked(checked)
        self._update_merge_button()

    def _get_pairs(self) -> list[tuple[str, str]]:
        pairs = []
        for row in range(self._table.rowCount()):
            chk = self._table.cellWidget(row, 0)
            if not isinstance(chk, QCheckBox) or not chk.isChecked():
                continue
            combo = self._table.cellWidget(row, 5)
            if not isinstance(combo, QComboBox):
                continue
            target = combo.currentData()
            if target:
                pairs.append((self._patients[row]["id"], target))
        return pairs

    def _update_merge_button(self):
        self._merge_btn.setEnabled(bool(self._get_pairs()))

    def _on_merge(self):
        pairs = self._get_pairs()
        if not pairs:
            return

        sources = {s for s, _ in pairs}
        targets = {t for _, t in pairs}
        if sources & targets:
            QMessageBox.warning(
                self, "Configurazione non valida",
                "Un paziente non può essere contemporaneamente destinazione "
                "di un merge e sorgente di un altro.",
            )
            return

        dup_lines = []
        for source, _target in pairs:
            for m in self._matches.get(source, []):
                dup_lines.append(f"{source} → {m['patient_id']} ({m['reason']})")
        dup_warning = (
            "\n\nPossibili pazienti duplicati rilevati:\n- "
            + "\n- ".join(sorted(set(dup_lines)))
            if dup_lines else ""
        )

        reply = QMessageBox.question(
            self,
            "Conferma merge",
            f"Unire {len(pairs)} workspace nel workspace di destinazione?\n\n"
            "Verranno SPOSTATI (non copiati): documenti, file estratti, "
            "eventi, esami, timeline e Clinical State.\n\n"
            "I workspace SORGENTE verranno ELIMINATI definitivamente "
            "dopo il merge.\nQuesta azione è IRREVERSIBILE."
            + dup_warning,
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return

        self._merge_btn.setEnabled(False)
        self._progress.setVisible(True)
        self._progress.setValue(0)

        self._worker = _MergeWorker(self._services, pairs)
        self._worker.progress.connect(
            lambda pct, msg: (
                self._progress.setValue(pct),
                self._progress.setToolTip(msg),
            )
        )
        self._worker.result_ready.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _on_finished(self, results: list):
        self._progress.setVisible(False)
        moved = sum(r.moved_documents for r in results)
        present = sum(r.already_present for r in results)
        files = sum(r.moved_files for r in results)
        removed = [r.source_patient_id for r in results if r.source_removed]

        lines = [
            f"• {moved} documenti spostati",
            f"• {present} documenti già presenti (saltati)",
            f"• {files} file spostati",
        ]
        if removed:
            lines.append(f"• workspace rimossi: {', '.join(removed)}")
        errors = [e for r in results for e in r.errors]
        warnings = [w for r in results for w in r.warnings]
        if errors:
            lines.append("\nERRORI:\n- " + "\n- ".join(errors))
        if warnings:
            lines.append("\nNote:\n- " + "\n- ".join(warnings))

        QMessageBox.information(self, "Merge completato", "\n".join(lines))
        self.removed_sources = removed
        self.accept()

    def _on_error(self, error: str):
        self._progress.setVisible(False)
        self._merge_btn.setEnabled(True)
        QMessageBox.critical(self, "Errore merge", error)

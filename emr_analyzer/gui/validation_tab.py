"""Validation tab — review queue for human validation of extractions."""

from datetime import datetime

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTableWidget, QTableWidgetItem,
    QHeaderView, QPushButton, QLabel, QAbstractItemView, QMessageBox,
    QTextEdit, QSplitter,
)
from PyQt5.QtCore import Qt, pyqtSignal

from ..models.validation import ValidationStatus, Severity


class ValidationTab(QWidget):
    """Tab for reviewing and validating extracted data."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._services = {}
        self._current_patient_id = None
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        # Header
        header = QHBoxLayout()
        header.addWidget(QLabel("Coda di Validazione"))
        self._count_label = QLabel("")
        header.addWidget(self._count_label)
        header.addStretch()

        self._accept_all_btn = QPushButton("✓ Accetta tutti a basso rischio")
        self._accept_all_btn.clicked.connect(self._on_accept_all_low_risk)
        header.addWidget(self._accept_all_btn)
        layout.addLayout(header)

        # Splitter: table on top, detail on bottom
        splitter = QSplitter(Qt.Vertical)

        # Validation queue table
        self._table = QTableWidget()
        self._table.setColumnCount(6)
        self._table.setHorizontalHeaderLabels([
            "Tipo", "ID Elemento", "Problema", "Gravità",
            "Stato", "Data"
        ])
        self._table.horizontalHeader().setStretchLastSection(True)
        self._table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        self._table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._table.setAlternatingRowColors(True)
        self._table.itemSelectionChanged.connect(self._on_selection_changed)
        splitter.addWidget(self._table)

        # Detail panel
        detail_widget = QWidget()
        detail_layout = QVBoxLayout(detail_widget)
        detail_layout.setContentsMargins(0, 0, 0, 0)

        self._detail_text = QTextEdit()
        self._detail_text.setReadOnly(True)
        self._detail_text.setMaximumHeight(150)
        detail_layout.addWidget(self._detail_text)

        # Action buttons
        action_layout = QHBoxLayout()
        self._accept_btn = QPushButton("✓ Accetta")
        self._accept_btn.setObjectName("successButton")
        self._accept_btn.clicked.connect(lambda: self._resolve("accepted"))
        action_layout.addWidget(self._accept_btn)

        self._correct_btn = QPushButton("✏️ Correggi")
        self._correct_btn.clicked.connect(self._on_correct)
        action_layout.addWidget(self._correct_btn)

        self._reject_btn = QPushButton("✗ Rifiuta")
        self._reject_btn.setObjectName("dangerButton")
        self._reject_btn.clicked.connect(lambda: self._resolve("rejected"))
        action_layout.addWidget(self._reject_btn)

        self._defer_btn = QPushButton("⏭ Rimanda")
        self._defer_btn.clicked.connect(lambda: self._resolve("deferred"))
        action_layout.addWidget(self._defer_btn)

        detail_layout.addLayout(action_layout)
        splitter.addWidget(detail_widget)

        layout.addWidget(splitter, stretch=1)

    def set_services(self, services: dict):
        self._services = services

    def load_patient(self, patient_id: str):
        self._current_patient_id = patient_id
        self._refresh()

    def _refresh(self):
        """Load validation queue items for the current patient."""
        if not self._current_patient_id:
            return

        db = self._services.get("db")
        if not db:
            return

        # Query validation_queue table directly
        cursor = db.execute(
            """SELECT * FROM validation_queue
               WHERE patient_id=? AND status='pending'
               ORDER BY
                 CASE severity
                   WHEN 'high' THEN 1
                   WHEN 'medium' THEN 2
                   WHEN 'low' THEN 3
                 END,
                 created_at ASC""",
            (self._current_patient_id,),
        )
        items = cursor.fetchall()

        self._table.setRowCount(len(items))
        for i, row in enumerate(items):
            self._table.setItem(i, 0, QTableWidgetItem(row["item_type"]))
            self._table.setItem(i, 1, QTableWidgetItem(row["item_id"]))
            self._table.setItem(i, 2, QTableWidgetItem(row["issue"]))
            severity = row["severity"] or "medium"
            self._table.setItem(i, 3, QTableWidgetItem(severity))
            self._table.setItem(i, 4, QTableWidgetItem(row["status"]))
            self._table.setItem(i, 5, QTableWidgetItem(row["created_at"][:10] if row["created_at"] else ""))

            # Color by severity
            sev_colors = {"high": Qt.red, "medium": Qt.darkYellow, "low": Qt.darkGreen}
            color = sev_colors.get(severity, Qt.black)
            for col in range(6):
                item = self._table.item(i, col)
                if item:
                    item.setForeground(color)

            # Store row data
            self._table.item(i, 0).setData(Qt.UserRole, dict(row))

        self._count_label.setText(
            f"({len(items)} elementi da validare)"
        )

    def _on_selection_changed(self):
        row = self._table.currentRow()
        has_selection = row >= 0
        self._accept_btn.setEnabled(has_selection)
        self._correct_btn.setEnabled(has_selection)
        self._reject_btn.setEnabled(has_selection)
        self._defer_btn.setEnabled(has_selection)

        if has_selection:
            item = self._table.item(row, 0)
            if item:
                row_data = item.data(Qt.UserRole)
                detail = (
                    f"Tipo: {row_data.get('item_type')}\n"
                    f"ID: {row_data.get('item_id')}\n"
                    f"Problema: {row_data.get('issue')}\n"
                    f"Gravità: {row_data.get('severity')}\n"
                )
                if row_data.get("original_value"):
                    detail += f"\nValore originale:\n{row_data['original_value']}"
                self._detail_text.setPlainText(detail)

    def _resolve(self, new_status: str):
        """Resolve selected validation item."""
        row = self._table.currentRow()
        if row < 0:
            return

        item = self._table.item(row, 0)
        if not item:
            return

        row_data = item.data(Qt.UserRole)
        item_id = row_data.get("id")
        db = self._services.get("db")

        if db and item_id is not None:
            db.execute(
                """UPDATE validation_queue
                   SET status=?, resolved_at=?
                   WHERE id=?""",
                (new_status, datetime.now().isoformat(), item_id),
            )
            db.commit()

            # If accepted, also update the underlying item
            if new_status == "accepted":
                self._accept_underlying_item(row_data)

            # Audit log
            audit_repo = self._services.get("audit_repo")
            if audit_repo:
                audit_repo.log(
                    self._current_patient_id,
                    "validate",
                    row_data.get("item_type"),
                    row_data.get("item_id"),
                    {"action": new_status, "issue": row_data.get("issue")},
                )

        self._refresh()

    def _accept_underlying_item(self, row_data: dict):
        """Mark the underlying lab_value as validated."""
        item_type = row_data.get("item_type")
        item_ref_id = row_data.get("item_id")

        if item_type == "lab_value":
            lab_repo = self._services.get("lab_repo")
            if lab_repo and item_ref_id.isdigit():
                lab_repo.update_validation(int(item_ref_id), True)

    def _on_correct(self):
        """Open a dialog to correct the value."""
        row = self._table.currentRow()
        if row < 0:
            return

        item = self._table.item(row, 0)
        if not item:
            return

        row_data = item.data(Qt.UserRole)
        from PyQt5.QtWidgets import QInputDialog

        new_value, ok = QInputDialog.getText(
            self, "Correggi Valore",
            f"Correggi il valore per: {row_data.get('item_id')}",
            text=row_data.get("original_value", "")
        )
        if ok:
            db = self._services.get("db")
            if db:
                db.execute(
                    """UPDATE validation_queue
                       SET status='corrected', corrected_value=?,
                       resolved_at=?
                       WHERE id=?""",
                    (new_value, datetime.now().isoformat(), row_data.get("id")),
                )
                db.commit()
            self._refresh()

    def _on_accept_all_low_risk(self):
        """Accept all low-severity items automatically."""
        db = self._services.get("db")
        if not db:
            return

        reply = QMessageBox.question(
            self, "Conferma",
            "Accettare tutti gli elementi a basso rischio?\n"
            "Questa azione è reversibile solo manualmente.",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            db.execute(
                """UPDATE validation_queue
                   SET status='accepted', resolved_at=?
                   WHERE patient_id=? AND status='pending' AND severity='low'""",
                (datetime.now().isoformat(), self._current_patient_id),
            )
            db.commit()
            self._refresh()

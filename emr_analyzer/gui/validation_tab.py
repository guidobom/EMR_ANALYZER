"""Validation tab — review queue for human validation of extractions."""

import json
from datetime import datetime

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTableWidget, QTableWidgetItem,
    QHeaderView, QPushButton, QLabel, QAbstractItemView, QMessageBox,
    QTextEdit, QSplitter, QInputDialog,
)
from PyQt5.QtCore import Qt, pyqtSignal

from ..models.validation import ValidationStatus, Severity


class ValidationTab(QWidget):
    """Tab for reviewing and validating extracted data."""

    # (source_patient_id, target_patient_id) after a document re-attribution.
    document_reattributed = pyqtSignal(str, str)

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
                if row_data.get("item_type") == "attribution":
                    suggested = self._suggested_patient(row_data)
                    if suggested:
                        detail += f"\nPaziente suggerito: {suggested}"
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

        # Attribution rows follow a different flow: the document moves to
        # the target patient and the queue row is resolved atomically by
        # the re-attribution service (never a bare status update).
        if row_data.get("item_type") == "attribution" and new_status == "accepted":
            self._accept_attribution(row_data)
            return

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

    # ------------------------------------------------------------------
    # Attribution re-assignment (human-confirmed document move)
    # ------------------------------------------------------------------

    @staticmethod
    def _suggested_patient(row_data: dict) -> str | None:
        """Suggested patient from the attribution queue item JSON."""
        try:
            payload = json.loads(str(row_data.get("original_value") or ""))
        except (TypeError, ValueError):
            return None
        if not isinstance(payload, dict):
            return None
        suggested = payload.get("suggested_patient_id")
        return str(suggested) if suggested else None

    def _accept_attribution(self, row_data: dict) -> None:
        """Accept an attribution item: move the document to the suggested
        patient (or a chosen one) and unlock its extraction."""
        doc_id = str(row_data.get("item_id") or "")
        suggested = self._suggested_patient(row_data)
        patient_repo = self._services.get("patient_repo")

        target = None
        if suggested:
            patient = patient_repo.get_by_id(suggested) if patient_repo else None
            if patient is not None and suggested != self._current_patient_id:
                target = suggested

        if target is None:
            # Suggested patient invalid or the document already belongs to
            # the current workspace: fall back to an explicit choice.
            if suggested is None:
                QMessageBox.information(
                    self, "Paziente suggerito non disponibile",
                    "Il paziente suggerito non è disponibile. "
                    "Scegli una destinazione.",
                )
            target = self._choose_target_patient(row_data)
            if target is None:
                return
        else:
            patient = patient_repo.get_by_id(target)
            label = (
                f"{target} ({patient.pseudonym})" if patient else target
            )
            reply = QMessageBox.question(
                self, "Conferma spostamento",
                f"Spostare il documento {doc_id} al paziente {label}?\n\n"
                "Il documento verrà spostato fisicamente e l'estrazione "
                "verrà sbloccata.\nQuesta azione è irreversibile.",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if reply != QMessageBox.Yes:
                return

        self._perform_reattribution(row_data, target, "accepted")

    def _on_correct(self):
        """Correct the value; for attribution items, choose the patient."""
        row = self._table.currentRow()
        if row < 0:
            return

        item = self._table.item(row, 0)
        if not item:
            return

        row_data = item.data(Qt.UserRole)
        if row_data.get("item_type") == "attribution":
            target = self._choose_target_patient(row_data)
            if target is None:
                return
            self._perform_reattribution(row_data, target, "corrected")
            return

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

    def _choose_target_patient(self, row_data: dict) -> str | None:
        """Let the user pick the destination patient for a document."""
        patient_repo = self._services.get("patient_repo")
        patients = [
            p for p in (patient_repo.list_all() if patient_repo else [])
            if p.id != self._current_patient_id
        ]
        if not patients:
            QMessageBox.warning(
                self, "Nessuna destinazione",
                "Non esistono altri pazienti a cui assegnare il documento.",
            )
            return None

        labels = []
        ids = []
        suggested = self._suggested_patient(row_data)
        suggested_index = 0
        for index, patient in enumerate(patients):
            extras = " • ".join(
                part for part in (
                    patient.initials, patient.sex,
                    str(patient.birth_year) if patient.birth_year else "",
                ) if part
            )
            labels.append(
                f"{patient.id} — {patient.pseudonym}"
                + (f" • {extras}" if extras else "")
            )
            ids.append(patient.id)
            if patient.id == suggested:
                suggested_index = index

        chosen, ok = QInputDialog.getItem(
            self, "Correggi attribuzione",
            f"Scegli il paziente a cui assegnare il documento "
            f"{row_data.get('item_id')}:",
            labels, suggested_index, editable=False,
        )
        if not ok:
            return None
        # The label is "P001 — pseudonym • extras": the id is the prefix.
        chosen_id = str(chosen).split(" — ")[0].strip()
        return chosen_id if chosen_id in ids else None

    def _perform_reattribution(
        self, row_data: dict, target: str, resolution_status: str
    ) -> None:
        """Run the move and surface the outcome."""
        service = self._services.get("document_reattribution")
        if not service:
            QMessageBox.critical(
                self, "Errore",
                "Servizio di riassegnazione non disponibile.",
            )
            return

        doc_id = str(row_data.get("item_id") or "")
        result = service.move_document(
            doc_id, target,
            queue_item_id=row_data.get("id"),
            resolution_status=resolution_status,
        )
        if result.ok:
            QMessageBox.information(
                self, "Documento spostato",
                f"Documento {doc_id} spostato al paziente {target}.\n"
                "L'estrazione è stata sbloccata: rielaboralo dalla scheda "
                "Documenti del paziente.",
            )
            self.document_reattributed.emit(
                result.source_patient_id, result.target_patient_id
            )
        else:
            QMessageBox.critical(
                self, "Spostamento non riuscito", result.error or "Errore"
            )
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

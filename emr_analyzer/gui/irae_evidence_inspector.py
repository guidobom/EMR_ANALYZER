"""Evidence inspector for one selected irAE finding.

Opened from ``IraePatientTab`` on double-click: shows the evidence that led
to the finding — the ones the model cited (``key_evidence_ids``) plus every
Layer 2 candidate of the same organ — in a tree with a textual citation panel
and, when the original document is available, an embedded PDF preview with
the passage highlighted (same pattern as ``EventQuickViewDialog``).

Manual corrections (edit / remove / add) are persisted per patient and
re-applied over the raw report, so a re-analysis reproduces the exact same
corrected result.
"""

from __future__ import annotations

import html

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QComboBox, QDialog, QDialogButtonBox, QFormLayout, QHBoxLayout, QLabel,
    QLineEdit, QMessageBox, QPlainTextEdit, QPushButton, QSplitter,
    QTextBrowser, QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget,
)

from .pdf_viewer import DocumentEvidencePreview

GRADE_CHOICES = ["G1", "G2", "G3", "G4", "G5", "non_determinabile"]
PROB_CHOICES = [
    "CERTA_CONFERMATA", "PROBABILE", "POSSIBILE",
    "IMPROBABILE", "INDETERMINATA",
]


class _FindingForm(QDialog):
    """Modal form editing the whitelisted fields of an irAE finding."""

    def __init__(self, parent, title: str, existing: dict | None = None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setMinimumWidth(460)
        existing = existing or {}
        self._fields = {}

        form = QFormLayout(self)
        self._type_edit = QLineEdit(str(existing.get("irAE_type") or ""))
        form.addRow("Tipo irAE:", self._type_edit)
        self._date_edit = QLineEdit(str(existing.get("first_onset_date") or ""))
        self._date_edit.setPlaceholderText("YYYY-MM-DD")
        form.addRow("Data prima insorgenza:", self._date_edit)
        self._grade_combo = QComboBox()
        self._grade_combo.addItems(GRADE_CHOICES)
        self._set_combo(self._grade_combo, existing.get("ctcae_grade"))
        form.addRow("Grado CTCAE:", self._grade_combo)
        self._prob_combo = QComboBox()
        self._prob_combo.addItems(PROB_CHOICES)
        self._set_combo(self._prob_combo, existing.get("probability_immune"))
        form.addRow("Probabilità immuno-correlata:", self._prob_combo)
        self._onset_edit = QLineEdit(
            str(existing.get("new_onset_vs_exacerbation") or "")
        )
        self._onset_edit.setPlaceholderText("insorgenza nuova | riacutizzazione…")
        form.addRow("Insorgenza:", self._onset_edit)
        self._causes_edit = QLineEdit(
            str(existing.get("alternative_causes") or "")
        )
        form.addRow("Cause alternative:", self._causes_edit)
        self._confidence_edit = QLineEdit(
            str(existing.get("confidence") or "")
        )
        self._confidence_edit.setPlaceholderText("0.0 – 1.0")
        form.addRow("Confidenza:", self._confidence_edit)
        self._notes_edit = QPlainTextEdit(str(existing.get("notes") or ""))
        self._notes_edit.setFixedHeight(80)
        form.addRow("Note:", self._notes_edit)

        buttons = QDialogButtonBox(
            QDialogButtonBox.Save | QDialogButtonBox.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)

    @staticmethod
    def _set_combo(combo: QComboBox, value) -> None:
        text = str(value or "")
        index = combo.findText(text)
        combo.setCurrentIndex(index if index >= 0 else 0)

    def values(self) -> dict:
        """The whitelisted fields as a plain dict (empty values omitted)."""
        fields: dict = {}
        for key, widget in (
            ("irAE_type", self._type_edit),
            ("first_onset_date", self._date_edit),
            ("new_onset_vs_exacerbation", self._onset_edit),
            ("alternative_causes", self._causes_edit),
        ):
            text = widget.text().strip()
            if text:
                fields[key] = text
        grade = self._grade_combo.currentText()
        if grade:
            fields["ctcae_grade"] = grade
        prob = self._prob_combo.currentText()
        if prob:
            fields["probability_immune"] = prob
        confidence = self._confidence_edit.text().strip()
        if confidence:
            try:
                fields["confidence"] = float(confidence)
            except ValueError:
                fields["confidence"] = confidence
        notes = self._notes_edit.toPlainText().strip()
        if notes:
            fields["notes"] = notes
        return fields


class IraeEvidenceInspector(QDialog):
    """Inspect and manually correct ONE irAE finding and its evidence."""

    report_updated = pyqtSignal(object)  # corrected report dict

    def __init__(
        self,
        report: dict,
        finding: dict,
        patient_id: str,
        services: dict,
        parent=None,
    ):
        super().__init__(parent)
        self._raw_report = report
        self._patient_id = patient_id
        self._services = services
        self._finding_id = str(finding.get("finding_id") or "")
        self._corrected, _ = self._reapply()
        self._finding = self._locate_finding(self._corrected, self._finding_id)
        label = str(self._finding.get("irAE_type") or "irAE")
        self.setWindowTitle(f"Evidenze irAE — {label} · {patient_id}")
        self.resize(1500, 900)
        self._setup_ui()
        self._populate()

    # ------------------------------------------------------------------
    # Corrections plumbing
    # ------------------------------------------------------------------

    def _reapply(self):
        from ..clinical.irae_corrections import apply_irae_corrections
        from ..clinical.irae_corrections import load_corrections

        return apply_irae_corrections(
            self._raw_report, load_corrections(self._patient_id)
        )

    @staticmethod
    def _locate_finding(corrected: dict, finding_id: str) -> dict | None:
        if not finding_id:
            return None
        consolidation = corrected.get("consolidation") or {}
        for item in list(corrected.get("iraes") or []) + list(
            consolidation.get("suspects") or []
        ):
            if item.get("finding_id") == finding_id:
                return item
        return None

    def _persist_correction(self, correction) -> None:
        from ..clinical.irae_corrections import add_correction, load_corrections

        add_correction(self._patient_id, correction)
        self._corrected, _ = apply_irae_corrections(
            self._raw_report, load_corrections(self._patient_id)
        )
        self._finding = self._locate_finding(self._corrected, self._finding_id)
        self.report_updated.emit(self._corrected)
        if self._finding is None:
            self.accept()  # the finding was removed
            return
        self._populate()

    def _ask_confirm(self, title: str, text: str) -> bool:
        answer = QMessageBox.question(
            self, title, text,
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
        )
        return answer == QMessageBox.Yes

    # ------------------------------------------------------------------
    # Correction buttons
    # ------------------------------------------------------------------

    def _on_edit(self) -> None:
        if self._finding is None:
            return
        dialog = _FindingForm(
            self, "Modifica irAE", existing=self._finding
        )
        if dialog.exec_() != QDialog.Accepted:
            return
        fields = dialog.values()
        if not fields.get("irAE_type"):
            QMessageBox.warning(self, "Tipo mancante", "Il tipo irAE è obbligatorio.")
            return
        from ..clinical.irae_corrections import IraeCorrection

        correction = IraeCorrection(
            action="edit",
            irAE_type=str(fields.pop("irAE_type")),
            first_onset_date=str(fields.pop("first_onset_date", "") or ""),
            finding_id=self._finding_id,
            fields=fields,
        )
        self._persist_correction(correction)

    def _on_remove(self) -> None:
        if self._finding is None:
            return
        if not self._ask_confirm(
            "Rimuovi irAE",
            f"Rimuovere «{self._finding.get('irAE_type')}» dal report?",
        ):
            return
        from ..clinical.irae_corrections import IraeCorrection

        correction = IraeCorrection(
            action="remove",
            irAE_type=str(self._finding.get("irAE_type") or ""),
            first_onset_date=str(self._finding.get("first_onset_date") or ""),
            finding_id=self._finding_id,
        )
        self._persist_correction(correction)

    def _on_add(self) -> None:
        dialog = _FindingForm(self, "Aggiungi irAE manuale")
        if dialog.exec_() != QDialog.Accepted:
            return
        fields = dialog.values()
        if not fields.get("irAE_type"):
            QMessageBox.warning(self, "Tipo mancante", "Il tipo irAE è obbligatorio.")
            return
        from ..clinical.irae_corrections import IraeCorrection

        correction = IraeCorrection(
            action="add",
            irAE_type=str(fields.pop("irAE_type")),
            first_onset_date=str(fields.pop("first_onset_date", "") or ""),
            fields=fields,
        )
        self._persist_correction(correction)

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        splitter = QSplitter(Qt.Horizontal)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        self._header = QLabel()
        left_layout.addWidget(self._header)
        self._tree = QTreeWidget()
        self._tree.setHeaderLabels(["Data", "Categoria", "Entità"])
        self._tree.setColumnWidth(0, 105)
        self._tree.setColumnWidth(1, 120)
        self._tree.itemSelectionChanged.connect(self._show_selected_evidence)
        left_layout.addWidget(self._tree, stretch=1)
        self._quote = QTextBrowser()
        self._quote.setMaximumHeight(150)
        self._quote.setPlaceholderText("Seleziona un'evidenza per leggerne la citazione.")
        left_layout.addWidget(self._quote)
        splitter.addWidget(left)

        self._preview = DocumentEvidencePreview(self._services, self)
        splitter.addWidget(self._preview)
        splitter.setSizes([560, 940])
        layout.addWidget(splitter, stretch=1)

        buttons = QHBoxLayout()
        edit_btn = QPushButton("✏️ Modifica irAE")
        edit_btn.clicked.connect(self._on_edit)
        buttons.addWidget(edit_btn)
        remove_btn = QPushButton("🗑️ Rimuovi irAE")
        remove_btn.clicked.connect(self._on_remove)
        buttons.addWidget(remove_btn)
        add_btn = QPushButton("➕ Aggiungi irAE manuale")
        add_btn.clicked.connect(self._on_add)
        buttons.addWidget(add_btn)
        buttons.addStretch()
        close_btn = QPushButton("Chiudi")
        close_btn.clicked.connect(self.accept)
        buttons.addWidget(close_btn)
        layout.addLayout(buttons)

    def _populate(self) -> None:
        self._tree.clear()
        self._quote.clear()
        finding = self._finding or {}
        grade = finding.get("ctcae_grade") or "?"
        prob = finding.get("probability_immune") or "?"
        onset = finding.get("first_onset_date") or "?"
        self._header.setText(
            f"<b>{html.escape(str(finding.get('irAE_type') or 'irAE'))}</b>"
            f" · {grade} · insorgenza {html.escape(str(onset))} · {prob}"
            f" · <code>{html.escape(self._finding_id)}</code>"
        )
        if finding.get("notes"):
            self._header.setToolTip(str(finding["notes"]))

        self._group_items: list[dict] = []
        self._add_cited_group()
        self._add_candidates_group()

    def _evidence_index(self) -> dict[str, dict]:
        index: dict[str, dict] = {}
        for item in self._corrected.get("evidence") or []:
            if isinstance(item, dict):
                index[str(item.get("evidence_id") or "")] = item
        return index

    def _add_cited_group(self) -> None:
        finding = self._finding or {}
        cited = finding.get("key_evidence_ids") or []
        root = QTreeWidgetItem(self._tree, ["Evidenze citate dal modello", "", ""])
        root.setExpanded(True)
        index = self._evidence_index()
        if not cited:
            QTreeWidgetItem(root, ["", "", "Nessuna evidenza citata dal modello."])
            return
        for evidence_id in cited:
            # the model echoes the prompt's ``[#id]`` notation; the registry
            # stores ids without the leading ``#``, so normalize before lookup.
            evidence = index.get(str(evidence_id).strip().lstrip("#"))
            if evidence is None:
                QTreeWidgetItem(
                    root, ["", "", f"Evidenza {evidence_id}: non trovata nel registro."]
                )
                continue
            item = self._evidence_item(root, evidence)
            self._group_items.append(evidence)
            _ = item

    def _add_candidates_group(self) -> None:
        finding = self._finding or {}
        organs = [
            str(organ) for organ in (finding.get("source_organs") or [])
        ]
        if finding.get("organ"):
            organs.append(str(finding["organ"]))
        organs = list(dict.fromkeys(organs))
        label = (
            "Candidati Layer 2 — " + ", ".join(organs)
            if organs else "Candidati Layer 2"
        )
        root = QTreeWidgetItem(self._tree, [label, "", ""])
        root.setExpanded(True)
        if not organs:
            QTreeWidgetItem(root, ["", "", "Nessun candidato di organo."])
            return
        organ_set = set(organs)
        matching = [
            candidate for candidate in self._corrected.get("candidates") or []
            if isinstance(candidate, dict)
            and str(candidate.get("organ") or "") in organ_set
        ]
        if not matching:
            QTreeWidgetItem(root, ["", "", "Nessun candidato per questo organo."])
            return
        for candidate in matching:
            evidence = self._candidate_to_evidence(candidate)
            self._evidence_item(root, evidence)
            self._group_items.append(evidence)

    @staticmethod
    def _candidate_to_evidence(candidate: dict) -> dict:
        """Compact evidence-shaped view of a Layer 2 candidate."""
        return {
            "evidence_id": candidate.get("evidence_id") or "",
            "document_id": candidate.get("document_id") or "",
            "source_page": candidate.get("source_page"),
            "bbox": candidate.get("bbox"),
            "source_text": candidate.get("source_text") or candidate.get("quote") or "",
            "normalized_entity": candidate.get("entity") or "",
            "category": candidate.get("category") or "",
            "observed_date": candidate.get("observed_raw") or "",
            "value_text": candidate.get("value") or "",
        }

    @staticmethod
    def _evidence_item(parent: QTreeWidgetItem, evidence: dict) -> QTreeWidgetItem:
        entity = evidence.get("normalized_entity") or ""
        value = evidence.get("value_text") or ""
        text = f"{entity}" + (f" — {value}" if value else "")
        item = QTreeWidgetItem(parent, [
            str(evidence.get("observed_date") or "n.d."),
            str(evidence.get("category") or ""),
            text,
        ])
        item.setToolTip(2, str(evidence.get("source_text") or ""))
        item.setData(0, Qt.UserRole, evidence)
        return item

    def _show_selected_evidence(self) -> None:
        item = self._tree.currentItem()
        evidence = item.data(0, Qt.UserRole) if item else None
        if not isinstance(evidence, dict) or not evidence.get("evidence_id"):
            return
        document_id = str(evidence.get("document_id") or "")
        document_repo = self._services.get("document_repo")
        document = document_repo.get_by_id(document_id) if document_repo else None
        self._render_quote(evidence)
        if not document:
            return
        same_document = [
            item_evidence for item_evidence in self._group_items
            if str(item_evidence.get("document_id") or "") == document_id
        ]
        highlights = [{
            "evidence_id": source.get("evidence_id"),
            "page_number": source.get("source_page"),
            "bbox": source.get("bbox"),
            "source_text": source.get("source_text"),
            "normalized_entity": source.get("normalized_entity"),
        } for source in same_document]
        self._preview.set_document(
            document.to_dict(),
            highlights=highlights,
            selected_evidence_id=evidence.get("evidence_id"),
        )

    def _render_quote(self, evidence: dict) -> None:
        source_text = str(evidence.get("source_text") or "").strip()
        value = str(evidence.get("value_text") or "").strip()
        lines = [
            f"<b>{html.escape(str(evidence.get('normalized_entity') or ''))}</b>",
            f"<small>{html.escape(str(evidence.get('observed_date') or 'n.d.'))}"
            f" · {html.escape(str(evidence.get('category') or ''))}</small>",
        ]
        if value:
            lines.append(f"<p><b>Valore:</b> {html.escape(value)}</p>")
        if source_text:
            lines.append(
                f"<blockquote>{html.escape(source_text)}</blockquote>"
            )
        else:
            lines.append("<p><i>Nessuna citazione testuale disponibile.</i></p>")
        self._quote.setHtml("".join(lines))

    def closeEvent(self, event) -> None:
        self._preview.close_document()
        event.accept()

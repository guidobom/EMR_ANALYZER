"""Right panel showing context/details of the selected item."""

import json

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QTextEdit, QGroupBox,
    QFormLayout, QScrollArea, QPushButton, QMessageBox,
)
from PyQt6.QtCore import Qt


class ContextPanel(QWidget):
    """Right-side contextual information panel."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumWidth(220)
        self.setMaximumWidth(350)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        title = QLabel("DETTAGLIO")
        title.setObjectName("subheading")
        layout.addWidget(title)

        # Scrollable area for details
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_widget = QWidget()
        self._scroll_layout = QVBoxLayout(scroll_widget)
        scroll.setWidget(scroll_widget)
        layout.addWidget(scroll, stretch=1)

        # Metadata group
        self._meta_group = QGroupBox("Metadati")
        self._meta_form = QFormLayout()
        self._meta_group.setLayout(self._meta_form)
        self._meta_labels = {}
        self._scroll_layout.addWidget(self._meta_group)

        # Source group
        self._source_group = QGroupBox("Fonte")
        source_layout = QVBoxLayout()
        self._source_label = QLabel("")
        self._source_label.setWordWrap(True)
        source_layout.addWidget(self._source_label)
        self._source_group.setLayout(source_layout)
        self._scroll_layout.addWidget(self._source_group)

        # Compatibility view for workspaces created by the retired structured
        # document pipeline. New runs do not create this JSON.
        self._projection_group = QGroupBox("Proiezione clinica del documento")
        projection_layout = QVBoxLayout()
        self._projection_edit = QTextEdit()
        self._projection_edit.setReadOnly(True)
        self._projection_edit.setMaximumHeight(280)
        self._projection_edit.setPlaceholderText(
            "Proiezione non ancora costruita"
        )
        projection_layout.addWidget(self._projection_edit)
        self._projection_group.setLayout(projection_layout)
        self._projection_group.setVisible(False)  # Legacy structured output only.
        self._scroll_layout.addWidget(self._projection_group)

        # Active text: cleaned parser output before the LLM, normalized
        # clinical text after successful isolation.
        self._text_group = QGroupBox("Testo clinico attivo")
        text_layout = QVBoxLayout()
        self._text_edit = QTextEdit()
        self._text_edit.setMinimumHeight(150)
        self._text_edit.setMaximumHeight(400)
        self._text_edit.setReadOnly(False)  # EDITABLE for manual correction
        self._text_edit.setPlaceholderText("Seleziona un documento per visualizzare il testo...")
        text_layout.addWidget(self._text_edit)

        # Save bar
        save_layout = QHBoxLayout()
        self._save_btn = QPushButton("💾 Salva correzioni")
        self._save_btn.setObjectName("successButton")
        self._save_btn.clicked.connect(self._on_save_text)
        self._save_btn.setEnabled(False)
        self._cancel_btn = QPushButton("↩ Annulla modifiche")
        self._cancel_btn.clicked.connect(self._on_revert_text)
        self._cancel_btn.setEnabled(False)
        save_layout.addWidget(self._save_btn)
        save_layout.addWidget(self._cancel_btn)
        text_layout.addLayout(save_layout)
        self._text_group.setLayout(text_layout)
        self._scroll_layout.addWidget(self._text_group)

        # State for tracking current document
        self._current_doc_id = None
        self._current_patient_id = None
        self._original_text = ""

        # Notes group
        self._notes_group = QGroupBox("Note utente")
        notes_layout = QVBoxLayout()
        self._notes_edit = QTextEdit()
        self._notes_edit.setMaximumHeight(120)
        self._notes_edit.setPlaceholderText("Annotazioni personali su questo documento...")
        notes_layout.addWidget(self._notes_edit)
        self._notes_group.setLayout(notes_layout)
        self._scroll_layout.addWidget(self._notes_group)

        # Stretch at bottom
        self._scroll_layout.addStretch()

    def show_event_context(self, event: dict):
        """Show details for a clinical event."""
        self.clear()
        # Metadata
        fields = [
            ("ID Evento", event.get("event_id", "")),
            ("Tipo", event.get("event_type", "")),
            ("Data", event.get("event_date", "")),
            ("Entità", event.get("entity", "")),
            ("Confidenza", f"{event.get('confidence', 0):.2f}"),
            ("Stato", event.get("status", "")),
            ("Pagina", str(event.get("page", ""))),
            ("Documento", event.get("source_document_id", "")),
        ]
        for label, value in fields:
            self._add_meta_field(label, value)

        # Source
        self._source_label.setText(f"Documento: {event.get('source_document_id', '')}\n"
                                   f"Pagina: {event.get('page', '-')}")

        # Original text
        self._text_edit.setPlainText(event.get("source_text", ""))

    def show_lab_context(self, lab: dict):
        """Show details for a lab value."""
        self.clear()
        fields = [
            ("Parametro", lab.get("parameter_name", "")),
            ("Valore", f"{lab.get('value', '')} {lab.get('unit', '')}"),
            ("Range rif.", lab.get("reference_text", "")),
            ("Anomalo", "Sì" if lab.get("is_abnormal") else "No"),
            ("Confidenza", f"{lab.get('confidence', 0):.2f}"),
            ("Data", lab.get("sample_date", "")),
            ("Materiale", lab.get("biological_material", "")),
            ("Laboratorio", lab.get("lab_name", "")),
        ]
        for label, value in fields:
            self._add_meta_field(label, value)

    def show_document_context(self, doc: dict, patient_id: str = ""):
        """Show details for a document, including editable extracted text."""
        self.clear()
        doc_id = doc.get("id", "")
        self._current_doc_id = doc_id
        self._current_patient_id = patient_id

        fields = [
            ("ID", doc_id),
            ("File", doc.get("filename", "")),
            ("Tipo", doc.get("document_type", "")),
            ("Data doc", doc.get("document_date", "")),
            ("Pagine", str(doc.get("page_count", ""))),
            ("Stato parsing", doc.get("parsing_status", "")),
            ("Stato estrazione", doc.get("extraction_status", "")),
            ("Testo clinico", (
                "disponibile" if doc.get("extraction_status") == "done"
                else "non disponibile"
            )),
            ("Lab values", str(doc.get("lab_value_count", ""))),
        ]
        for label, value in fields:
            self._add_meta_field(label, value)

        # Source
        self._source_label.setText(
            f"File: {doc.get('filename', '')}\n"
            f"Tipo: {doc.get('document_type', '')}"
        )

        # Load extracted text
        self._original_text = ""
        if patient_id and doc_id:
            from pathlib import Path
            from ..config import WORKSPACES_DIR
            md_path = WORKSPACES_DIR / patient_id / "extraction" / f"{doc_id}.md"
            if not md_path.exists():
                md_path = WORKSPACES_DIR / patient_id / "docling" / f"{doc_id}.md"
            if md_path.exists():
                self._original_text = md_path.read_text(encoding="utf-8")
                self._text_edit.setPlainText(self._original_text)
                self._save_btn.setEnabled(True)
                self._cancel_btn.setEnabled(True)
            else:
                self._text_edit.setPlainText("(testo estratto non ancora disponibile)")
                self._save_btn.setEnabled(False)
                self._cancel_btn.setEnabled(False)

            projection_path = (
                WORKSPACES_DIR / patient_id / "extraction" /
                f"{doc_id}_clinical_evidence.json"
            )
            if projection_path.exists():
                try:
                    payload = json.loads(
                        projection_path.read_text(encoding="utf-8")
                    )
                    projection = payload.get("projection") or {}
                    self._projection_group.setVisible(True)
                    observations = projection.get("observations", [])
                    lines = [
                        f"{len(observations)} osservazioni consolidate",
                        f"{len(projection.get('relationships', []))} relazioni",
                        "",
                    ]
                    observation_labels = {}
                    for observation in observations:
                        date = observation.get("observed_start_date") or "data n.d."
                        pages = sorted({
                            span.get("page") for span in observation.get(
                                "source_spans", []
                            ) if span.get("page") is not None
                        })
                        page_text = ",".join(map(str, pages)) or "n.d."
                        lines.append(
                            f"• {date} | {observation.get('category', 'other')} | "
                            f"{observation.get('normalized_entity', '')} "
                            f"[p. {page_text}]"
                        )
                        observation_labels[observation.get("observation_id")] = (
                            observation.get("normalized_entity", "")
                        )
                        for span in observation.get("source_spans", []):
                            quote = " ".join(
                                str(span.get("text") or "").split()
                            )
                            if len(quote) > 220:
                                quote = quote[:217] + "..."
                            lines.append(
                                f"    p. {span.get('page') or 'n.d.'}: {quote}"
                            )
                    relationships = projection.get("relationships", [])
                    if relationships:
                        lines.extend(["", "Relazioni:"])
                        for relationship in relationships:
                            source = observation_labels.get(
                                relationship.get("from_observation_id"), "?"
                            )
                            target = observation_labels.get(
                                relationship.get("to_observation_id"), "?"
                            )
                            lines.append(
                                f"  {source} → "
                                f"{relationship.get('relationship_type', '')} → "
                                f"{target}"
                            )
                    conflicts = projection.get("conflicts", [])
                    if conflicts:
                        lines.extend([
                            "",
                            f"⚠ {len(conflicts)} conflitti da revisionare",
                        ])
                    self._projection_edit.setPlainText("\n".join(lines))
                except (OSError, ValueError, TypeError):
                    self._projection_edit.setPlainText(
                        "Proiezione presente ma non leggibile"
                    )

    def _on_save_text(self):
        """Save edited text back to the markdown file."""
        if not self._current_doc_id or not self._current_patient_id:
            return

        reply = QMessageBox.question(
            self, "Conferma salvataggio",
            "Salvare le modifiche al testo clinico attivo?\n"
            "Il testo grezzo e la sorgente PDF resteranno conservati "
            "separatamente.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, QMessageBox.StandardButton.Yes
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        from pathlib import Path
        from ..config import WORKSPACES_DIR
        md_path = WORKSPACES_DIR / self._current_patient_id / "extraction" / f"{self._current_doc_id}.md"
        if not md_path.exists():
            md_path = WORKSPACES_DIR / self._current_patient_id / "docling" / f"{self._current_doc_id}.md"
        new_text = self._text_edit.toPlainText()
        md_path.write_text(new_text, encoding="utf-8")
        self._original_text = new_text

        self._save_btn.setText("✓ Salvato")
        self._save_btn.setStyleSheet("background-color: #2ecc71; color: white;")

    def _on_revert_text(self):
        """Revert to the original text."""
        self._text_edit.setPlainText(self._original_text)
        self._save_btn.setText("💾 Salva correzioni")
        self._save_btn.setStyleSheet("")

    def clear(self):
        """Clear all context details."""
        while self._meta_form.rowCount() > 0:
            self._meta_form.removeRow(0)
        self._source_label.clear()
        self._projection_edit.clear()
        self._projection_group.setVisible(False)
        self._text_edit.clear()
        self._notes_edit.clear()
        self._save_btn.setEnabled(False)
        self._cancel_btn.setEnabled(False)
        self._save_btn.setText("💾 Salva correzioni")
        self._save_btn.setStyleSheet("")
        self._original_text = ""

    def _add_meta_field(self, label: str, value: str):
        """Add a read-only field to the metadata panel."""
        value_label = QLabel(str(value))
        value_label.setWordWrap(True)
        value_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self._meta_form.addRow(f"{label}:", value_label)

"""Clinical State tab — tree view + query bar."""

import json

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTextEdit, QPushButton,
    QTreeWidget, QTreeWidgetItem, QComboBox, QLabel, QSplitter,
    QMessageBox,
)
from PyQt5.QtCore import Qt

from ..models.clinical_state import ClinicalState


# Predefined query templates
PREDEFINED_QUERIES = [
    ("— Prompt predefiniti —", ""),
    ("Qual è la terapia attuale?", "Qual è la terapia attuale del paziente?"),
    ("Quali diagnosi attive ha il paziente?", "Elenca tutte le diagnosi attive del paziente."),
    ("Qual è lo staging più recente?", "Qual è lo staging tumorale più recente documentato?"),
    ("Mostra i biomarcatori rilevanti", "Elenca i biomarcatori clinicamente rilevanti con i loro valori."),
    ("Quali tossicità sono state registrate?", "Elenca tutte le tossicità e gli eventi avversi registrati."),
    ("Riassumi lo stato clinico attuale", "Fornisci un riassunto completo dello stato clinico attuale del paziente."),
    ("Quali procedure/interventi sono stati eseguiti?", "Elenca tutte le procedure e gli interventi chirurgici eseguiti."),
]


class ClinicalStateTab(QWidget):
    """Tree view of the clinical state with query capability."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._services = {}
        self._current_patient_id = None
        self._clinical_state = None
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        # Query bar
        query_layout = QHBoxLayout()
        query_layout.addWidget(QLabel("Query:"))

        self._query_preset = QComboBox()
        for label, query in PREDEFINED_QUERIES:
            self._query_preset.addItem(label, query)
        self._query_preset.currentIndexChanged.connect(self._on_preset_changed)
        query_layout.addWidget(self._query_preset, stretch=1)

        self._query_text = QTextEdit()
        self._query_text.setMaximumHeight(60)
        self._query_text.setPlaceholderText(
            "Inserisci una domanda sul Clinical State del paziente..."
        )
        query_layout.addWidget(self._query_text, stretch=2)

        self._query_btn = QPushButton("🔍 Interroga")
        self._query_btn.clicked.connect(self._run_query)
        query_layout.addWidget(self._query_btn)

        layout.addLayout(query_layout)

        # Splitter: tree on left, answer on right
        splitter = QSplitter(Qt.Horizontal)

        # Clinical State tree
        self._tree = QTreeWidget()
        self._tree.setHeaderLabels(["Clinical State"])
        self._tree.setAlternatingRowColors(True)
        splitter.addWidget(self._tree)

        # Query answer
        answer_widget = QWidget()
        answer_layout = QVBoxLayout(answer_widget)
        answer_layout.setContentsMargins(0, 0, 0, 0)

        answer_header = QHBoxLayout()
        answer_header.addWidget(QLabel("Risposta"))
        self._clear_answer_btn = QPushButton("Pulisci")
        self._clear_answer_btn.clicked.connect(lambda: self._answer_text.clear())
        answer_header.addWidget(self._clear_answer_btn)
        answer_layout.addLayout(answer_header)

        self._answer_text = QTextEdit()
        self._answer_text.setReadOnly(True)
        answer_layout.addWidget(self._answer_text)

        splitter.addWidget(answer_widget)
        layout.addWidget(splitter, stretch=1)

        # Bottom: refresh + rebuild
        bottom_layout = QHBoxLayout()
        self._refresh_btn = QPushButton("🔄 Aggiorna")
        self._refresh_btn.clicked.connect(lambda: self.load_patient(self._current_patient_id))
        bottom_layout.addWidget(self._refresh_btn)

        self._rebuild_btn = QPushButton("🔨 Ricostruisci da zero")
        self._rebuild_btn.setToolTip(
            "Elimina il Clinical State e lo ricostruisce da eventi ed evidenze.\n"
            "Utile dopo aver cambiato modello LLM o rielaborato molti documenti."
        )
        self._rebuild_btn.clicked.connect(self._on_rebuild)
        bottom_layout.addWidget(self._rebuild_btn)

        self._version_label = QLabel("")
        bottom_layout.addWidget(self._version_label)
        bottom_layout.addStretch()
        layout.addLayout(bottom_layout)

    def set_services(self, services: dict):
        self._services = services

    def load_patient(self, patient_id: str):
        self._current_patient_id = patient_id
        self._refresh()

    def _on_rebuild(self):
        """Wipe and rebuild Clinical State from all events."""
        if not self._current_patient_id:
            return

        if self._has_normalized_document_sources():
            QMessageBox.information(
                self,
                "Ricostruzione non ancora disponibile",
                "I documenti usano la nuova pipeline a testo clinico "
                "normalizzato. Il vecchio ricostruttore da eventi/evidenze "
                "è disattivato perché produrrebbe un Clinical State parziale.\n\n"
                "La ricostruzione longitudinale dai testi normalizzati sarà "
                "implementata nella fase successiva.",
            )
            return

        from PyQt5.QtWidgets import QMessageBox
        reply = QMessageBox.question(
            self, "Ricostruzione Clinical State",
            "Eliminare il Clinical State attuale e ricostruirlo\n"
            "da TUTTI gli eventi e da tutte le evidenze normalizzate?\n\n"
            "Usa questa funzione dopo aver cambiato modello LLM\n"
            "o rielaborato molti documenti.",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )
        if reply != QMessageBox.Yes:
            return

        cs_manager = self._services.get("cs_manager")
        if not cs_manager:
            return

        # Delete existing Clinical State
        cs_repo = self._services.get("cs_repo")
        if cs_repo:
            cs_repo.delete(self._current_patient_id)

        # Rebuild from scratch
        state = cs_manager.rebuild_from_events(self._current_patient_id)

        # Refresh display
        self._refresh()
        QMessageBox.information(
            self, "Ricostruzione completata",
            f"Clinical State ricostruito:\n"
            f"• {len(state.active_diagnoses)} diagnosi attive\n"
            f"• {len(state.active_treatments)} terapie attive\n"
            f"• {len(state.completed_treatments)} terapie concluse\n"
            f"• {len(state.toxicities)} tossicità\n"
            f"• {len(state.procedures)} procedure\n"
            f"• {len(state.observations)} osservazioni cliniche complessive"
        )

    def _refresh(self):
        if not self._current_patient_id:
            return

        cs_repo = self._services.get("cs_repo")
        if not cs_repo:
            return

        self._clinical_state = cs_repo.load(self._current_patient_id)
        self._build_tree()
        self._version_label.setText(
            f"Versione: {self._clinical_state.version if self._clinical_state else 'N/D'}"
        )
        uses_normalized_text = self._has_normalized_document_sources()
        self._rebuild_btn.setEnabled(not uses_normalized_text)
        if uses_normalized_text:
            self._rebuild_btn.setToolTip(
                "Disattivato: il vecchio ricostruttore non legge i testi "
                "clinici normalizzati e produrrebbe uno stato incompleto."
            )
        else:
            self._rebuild_btn.setToolTip(
                "Elimina il Clinical State e lo ricostruisce da eventi ed "
                "evidenze legacy."
            )

    def _has_normalized_document_sources(self) -> bool:
        """Return true when at least one report uses the new text pipeline."""
        doc_repo = self._services.get("document_repo")
        if not doc_repo or not self._current_patient_id:
            return False
        for document in doc_repo.list_by_patient(self._current_patient_id):
            try:
                metadata = json.loads(document.metadata_json or "{}")
            except (TypeError, ValueError):
                continue
            if metadata.get("clinical_text", {}).get("output_format") == (
                "normalized_plain_text"
            ):
                return True
        return False

    def _build_tree(self):
        """Build the tree view from the Clinical State."""
        self._tree.clear()
        if not self._clinical_state:
            self._tree.addTopLevelItem(
                QTreeWidgetItem(["Nessun Clinical State disponibile"])
            )
            return

        cs = self._clinical_state

        # Helper to add a list of items under a parent
        def add_items(parent_key: str, label: str, items: list,
                      fields: list[str]):
            if not items:
                return
            parent = QTreeWidgetItem([f"{label} ({len(items)})"])
            self._tree.addTopLevelItem(parent)
            for item in items:
                if isinstance(item, dict):
                    text = ", ".join(f"{k}: {item.get(k, '')}" for k in fields)
                elif hasattr(item, 'to_dict'):
                    d = item.to_dict()
                    text = ", ".join(f"{k}: {d.get(k, '')}" for k in fields)
                else:
                    text = str(item)
                child = QTreeWidgetItem([text])
                parent.addChild(child)
            parent.setExpanded(True)

        # Profile
        if cs.clinical_profile:
            root = QTreeWidgetItem(["📋 Profilo Clinico"])
            self._tree.addTopLevelItem(root)
            root.addChild(QTreeWidgetItem([cs.clinical_profile]))

        # Staging
        if cs.staging:
            root = QTreeWidgetItem([f"📊 Staging: {cs.staging}"])
            self._tree.addTopLevelItem(root)

        # Diagnoses
        self._add_diagnosis_tree("🩺 Diagnosi Attive", cs.active_diagnoses, True)
        self._add_diagnosis_tree("📝 Diagnosi Pregresse", cs.past_diagnoses, False)

        # Biomarkers
        add_items("biomarkers", "🧬 Biomarcatori", cs.biomarkers,
                  ["name", "value", "date"])

        # Treatments
        self._add_treatment_tree("💊 Terapie Attive", cs.active_treatments, True)
        self._add_treatment_tree("✅ Terapie Concluse", cs.completed_treatments, False)

        # Toxicities
        self._add_toxicity_tree("⚠️ Tossicità", cs.toxicities)

        # Allergies
        if cs.allergies:
            root = QTreeWidgetItem([f"🚫 Allergie ({len(cs.allergies)})"])
            self._tree.addTopLevelItem(root)
            for a in cs.allergies:
                root.addChild(QTreeWidgetItem([a]))
            root.setExpanded(True)

        # Comorbidities
        if cs.comorbidities:
            root = QTreeWidgetItem([f"🏥 Comorbidità ({len(cs.comorbidities)})"])
            self._tree.addTopLevelItem(root)
            for c in cs.comorbidities:
                root.addChild(QTreeWidgetItem([c]))
            root.setExpanded(True)

        # Procedures
        add_items("procedures", "🔧 Procedure", cs.procedures,
                  ["name", "date", "outcome"])

        # Follow-up
        if cs.follow_up:
            root = QTreeWidgetItem([f"📅 Follow-up ({len(cs.follow_up)})"])
            self._tree.addTopLevelItem(root)
            for f in cs.follow_up:
                text = f.get("description", f.get("date", str(f)))
                root.addChild(QTreeWidgetItem([text]))
            root.setExpanded(True)

        # Complete evidence projection, grouped but initially collapsed to
        # keep large longitudinal dossiers responsive.
        if cs.observations:
            root = QTreeWidgetItem([
                f"📚 Osservazioni complete ({len(cs.observations)})"
            ])
            self._tree.addTopLevelItem(root)
            groups = {}
            for observation in cs.observations:
                category = observation.get("category") or "other"
                groups.setdefault(category, []).append(observation)
            for category, observations in sorted(groups.items()):
                group = QTreeWidgetItem([
                    f"{category} ({len(observations)})"
                ])
                root.addChild(group)
                for observation in observations:
                    parts = [
                        value for value in (
                            observation.get("date"),
                            observation.get("entity"),
                            observation.get("value_text"),
                            observation.get("clinical_status"),
                        ) if value not in (None, "")
                    ]
                    child = QTreeWidgetItem([" | ".join(map(str, parts))])
                    child.setToolTip(
                        0,
                        "\n".join(filter(None, [
                            f"Evidence: {observation.get('evidence_id', '')}",
                            f"Documento: {observation.get('source_document_id', '')}",
                            f"Pagina: {observation.get('source_page', '')}",
                            observation.get("source_text") or "",
                        ])),
                    )
                    child.setData(Qt.UserRole, observation)
                    group.addChild(child)
            root.setExpanded(False)

    def _add_diagnosis_tree(self, label: str, diagnoses: list, expanded: bool):
        if not diagnoses:
            return
        root = QTreeWidgetItem([f"{label} ({len(diagnoses)})"])
        self._tree.addTopLevelItem(root)
        for d in diagnoses:
            text = d.name
            if d.date:
                text += f" | {d.date}"
            if d.icd_code:
                text += f" | ICD: {d.icd_code}"
            if d.status:
                text += f" | [{d.status}]"
            child = QTreeWidgetItem([text])
            if d.source_event_id:
                child.setToolTip(0, f"Event: {d.source_event_id}")
            root.addChild(child)
        root.setExpanded(expanded)

    def _add_treatment_tree(self, label: str, treatments: list, expanded: bool):
        if not treatments:
            return
        root = QTreeWidgetItem([f"{label} ({len(treatments)})"])
        self._tree.addTopLevelItem(root)
        for t in treatments:
            text = t.name
            if t.start_date:
                text += f" | dal {t.start_date}"
            if t.line:
                text += f" | {t.line}ª linea"
            if t.setting:
                text += f" | {t.setting}"
            child = QTreeWidgetItem([text])
            root.addChild(child)
        root.setExpanded(expanded)

    def _add_toxicity_tree(self, label: str, toxicities: list):
        if not toxicities:
            return
        root = QTreeWidgetItem([f"{label} ({len(toxicities)})"])
        self._tree.addTopLevelItem(root)
        for t in toxicities:
            text = t.name
            if t.grade is not None:
                text += f" | Grado {t.grade}"
            if t.date:
                text += f" | {t.date}"
            if t.status:
                text += f" | [{t.status}]"
            child = QTreeWidgetItem([text])
            root.addChild(child)
        root.setExpanded(True)

    def _on_preset_changed(self, index: int):
        query = self._query_preset.currentData()
        if query:
            self._query_text.setPlainText(query)

    def _run_query(self):
        """Execute a clinical query."""
        question = self._query_text.toPlainText().strip()
        if not question:
            return

        if not self._clinical_state:
            self._answer_text.setPlainText(
                "Nessun Clinical State disponibile. Elabora prima dei documenti."
            )
            return

        llm_client = self._services.get("clinical_state_llm_client")
        event_repo = self._services.get("event_repo")

        if not llm_client or not llm_client.is_available:
            # Fallback: answer from structured data
            answer = self._local_query(question)
            self._answer_text.setMarkdown(answer)
            return

        # Run the query with the independently selected Clinical State model.
        events = []
        if event_repo:
            events = [e.to_dict() for e in event_repo.get_by_patient(
                self._current_patient_id)]

        from .workers import ClinicalQueryWorker
        self._worker = ClinicalQueryWorker(
            llm_client, self._clinical_state.to_dict(), events, question
        )
        self._worker.finished.connect(self._on_query_result)
        self._worker.error.connect(self._on_query_error)
        self._query_btn.setEnabled(False)
        self._query_btn.setText("⏳ Interrogazione in corso...")
        self._worker.start()

    def _on_query_result(self, answer: str):
        self._answer_text.setMarkdown(answer)
        self._query_btn.setEnabled(True)
        self._query_btn.setText("🔍 Interroga")

    def _on_query_error(self, error: str):
        self._answer_text.setPlainText(f"❌ Errore: {error}")
        self._query_btn.setEnabled(True)
        self._query_btn.setText("🔍 Interroga")

    def _local_query(self, question: str) -> str:
        """Local query when LLM is not available — uses structured data."""
        cs = self._clinical_state
        q_lower = question.lower()

        if "terapia" in q_lower or "trattamento" in q_lower:
            if cs.active_treatments:
                treatments = "\n".join(
                    f"- **{t.name}** (dal {t.start_date or '?'}) [{t.status}]"
                    for t in cs.active_treatments
                )
                return f"### Terapie Attive\n{treatments}"
            return "Nessuna terapia attiva documentata."

        if "diagnosi" in q_lower:
            if cs.active_diagnoses:
                diagnoses = "\n".join(
                    f"- **{d.name}** ({d.date or 'data sconosciuta'})"
                    for d in cs.active_diagnoses
                )
                return f"### Diagnosi Attive\n{diagnoses}"
            return "Nessuna diagnosi attiva documentata."

        if "staging" in q_lower:
            return f"### Staging\n{cs.staging or 'Non documentato'}"

        if "biomarcat" in q_lower:
            if cs.biomarkers:
                markers = "\n".join(
                    f"- **{b.name}**: {b.value or 'N/D'}"
                    for b in cs.biomarkers
                )
                return f"### Biomarcatori\n{markers}"
            return "Nessun biomarcatore documentato."

        if "tossicit" in q_lower or "eventi avversi" in q_lower:
            if cs.toxicities:
                tox = "\n".join(
                    f"- **{t.name}** (Grado {t.grade or '?'}) [{t.status}]"
                    for t in cs.toxicities
                )
                return f"### Tossicità\n{tox}"
            return "Nessuna tossicità documentata."

        # Default: full summary
        parts = []
        if cs.active_diagnoses:
            parts.append(f"**Diagnosi attive:** {', '.join(d.name for d in cs.active_diagnoses)}")
        if cs.active_treatments:
            parts.append(f"**Terapie attive:** {', '.join(t.name for t in cs.active_treatments)}")
        if cs.staging:
            parts.append(f"**Staging:** {cs.staging}")
        parts.append(f"**Ultimo aggiornamento:** {cs.updated_at[:10] if cs.updated_at else 'N/D'}")
        return "### Riassunto Clinico\n\n" + "\n\n".join(parts)

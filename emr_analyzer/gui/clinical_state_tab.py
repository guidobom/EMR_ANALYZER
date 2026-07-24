"""Clinical State tab — tree view + query bar."""

import json

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTextEdit, QPushButton,
    QTreeWidget, QTreeWidgetItem, QComboBox, QLabel, QSplitter,
    QMessageBox, QDialog, QTableWidget, QTableWidgetItem,
    QHeaderView,
)
from PyQt6.QtCore import Qt

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

        self._query_all_btn = QPushButton("🔍 Interroga TUTTI")
        self._query_all_btn.setToolTip(
            "Esegue la stessa domanda sul Clinical State di TUTTI i pazienti."
        )
        self._query_all_btn.clicked.connect(self._on_query_all)
        query_layout.addWidget(self._query_all_btn)

        layout.addLayout(query_layout)

        # Splitter: tree on left, answer on right
        splitter = QSplitter(Qt.Orientation.Horizontal)

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

        self._batch_btn = QPushButton("🔨 Ricostruisci TUTTI")
        self._batch_btn.setToolTip(
            "Ricostruisce il Clinical State per TUTTI i pazienti "
            "che hanno testi clinici normalizzati.\n"
            "Esegue fino a 2 pazienti in parallelo."
        )
        self._batch_btn.clicked.connect(self._on_rebuild_all)
        bottom_layout.addWidget(self._batch_btn)

        self._export_btn = QPushButton("📥 Esporta Excel")
        self._export_btn.setToolTip(
            "Esporta il Clinical State corrente in un file Excel "
            "con un foglio per ogni entità (diagnosi, terapie, ...)"
        )
        self._export_btn.clicked.connect(self._on_export_excel)
        bottom_layout.addWidget(self._export_btn)

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
        """Wipe and rebuild Clinical State."""
        if not self._current_patient_id:
            return

        cs_manager = self._services.get("cs_manager")
        if not cs_manager:
            return

        llm_client = self._services.get("clinical_state_llm_client")
        if not llm_client or not llm_client.is_available:
            QMessageBox.warning(
                self, "LLM non disponibile",
                "Per la ricostruzione longitudinale è necessario un modello "
                "LLM configurato per il Clinical State.\n"
                "Apri ⚙ Configura LLM e configura un modello per il "
                "ruolo 'LLM per il Clinical State'.",
            )
            return

        uses_normalized = self._has_normalized_document_sources()

        if uses_normalized:
            action_label = "da TUTTI i testi clinici normalizzati"
        else:
            action_label = "da TUTTI gli eventi e le evidenze"

        reply = QMessageBox.question(
            self, "Ricostruzione Clinical State",
            "Eliminare il Clinical State attuale e ricostruirlo\n"
            f"{action_label}?\n\n"
            f"{'I documenti verranno rielaborati uno a uno tramite LLM per '
             'estrarre diagnosi, terapie, tossicità, sintomi, biomarcatori '
             'e altre entità cliniche.' if uses_normalized else ''}\n\n"
            "Usa questa funzione dopo aver cambiato modello LLM\n"
            "o rielaborato molti documenti.\n\n"
            "Procedere?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        # Delete existing Clinical State
        cs_repo = self._services.get("cs_repo")
        if cs_repo:
            cs_repo.delete(self._current_patient_id)

        if uses_normalized:
            self._build_from_normalized_texts(cs_manager)
        else:
            self._build_from_events(cs_manager)

    def _build_from_normalized_texts(self, cs_manager):
        """Build ClinicalState from normalized clinical texts with progress."""
        from .progress_dialog import ProgressDialog
        from .workers import ClinicalStateBuildWorker

        dialog = ProgressDialog(
            "Ricostruzione Clinical State", self.window()
        )
        dialog.set_progress(0, "Avvio ricostruzione...")
        dialog._cancel_btn.setEnabled(False)
        dialog.show()

        self._build_worker = ClinicalStateBuildWorker(
            cs_manager, self._current_patient_id, self
        )
        self._build_worker.progress.connect(
            lambda pct, msg: dialog.set_progress(pct, msg)
        )
        self._build_worker.finished.connect(
            lambda state: self._on_build_complete(dialog, state)
        )
        self._build_worker.error.connect(
            lambda err: self._on_build_error(dialog, err)
        )
        self._build_worker.start()

    def _build_from_events(self, cs_manager):
        """Legacy rebuild from structured events."""
        state = cs_manager.rebuild_from_events(self._current_patient_id)
        self._refresh()
        QMessageBox.information(
            self, "Ricostruzione completata",
            f"Clinical State ricostruito:\n"
            f"• {len(state.active_diagnoses)} diagnosi attive\n"
            f"• {len(state.active_treatments)} terapie attive\n"
            f"• {len(state.completed_treatments)} terapie concluse\n"
            f"• {len(state.toxicities)} tossicità\n"
            f"• {len(state.procedures)} procedure\n"
            f"• {len(state.observations)} osservazioni cliniche complessive",
        )

    def _on_build_complete(self, dialog, state):
        """Called when the ClinicalState build worker finishes."""
        dialog.accept()
        self._refresh()
        # Build a summary of what was extracted
        parts = []
        if state.active_diagnoses:
            parts.append(f"• {len(state.active_diagnoses)} diagnosi attive")
        if state.active_treatments:
            parts.append(f"• {len(state.active_treatments)} terapie in corso")
        if state.completed_treatments:
            parts.append(f"• {len(state.completed_treatments)} terapie concluse")
        if state.toxicities:
            parts.append(f"• {len(state.toxicities)} tossicità/eventi avversi")
        if state.symptoms:
            parts.append(f"• {len(state.symptoms)} sintomi rilevati")
        if state.biomarkers:
            parts.append(f"• {len(state.biomarkers)} biomarcatori")
        if state.procedures:
            parts.append(f"• {len(state.procedures)} procedure")
        if state.allergies:
            parts.append(f"• {len(state.allergies)} allergie")
        if state.hospitalizations:
            parts.append(f"• {len(state.hospitalizations)} ricoveri")
        if state.imaging_findings:
            parts.append(f"• {len(state.imaging_findings)} referti imaging")
        if state.performance_status:
            ps = state.performance_status
            ecog = ps.get("ecog_score")
            kps = ps.get("karnofsky_score")
            if ecog is not None:
                parts.append(f"• Performance status ECOG {ecog}")
            elif kps is not None:
                parts.append(f"• Karnofsky {kps}")
        QMessageBox.information(
            self, "Ricostruzione completata",
            "Clinical State ricostruito con successo dai testi clinici:\n\n"
            + "\n".join(parts),
        )

    def _on_build_error(self, dialog, error: str):
        """Called when the ClinicalState build worker fails."""
        dialog.accept()
        QMessageBox.critical(
            self, "Ricostruzione fallita",
            f"La ricostruzione del Clinical State non è stata completata.\n\n{error}",
        )
        self._refresh()

    # ------------------------------------------------------------------
    # Batch rebuild for all patients
    # ------------------------------------------------------------------

    def _on_rebuild_all(self):
        """Rebuild ClinicalState for all patients with normalized texts."""
        from .workers import BatchClinicalStateWorker

        cs_manager = self._services.get("cs_manager")
        if not cs_manager:
            QMessageBox.warning(self, "Servizio non disponibile",
                                "ClinicalStateManager non inizializzato.")
            return

        llm_client = self._services.get("clinical_state_llm_client")
        if not llm_client or not llm_client.is_available:
            QMessageBox.warning(
                self, "LLM non disponibile",
                "Per la ricostruzione è necessario un modello LLM configurato "
                "per il Clinical State.",
            )
            return

        patient_ids = self._get_rebuildable_patients()
        if not patient_ids:
            QMessageBox.information(
                self, "Nessun paziente",
                "Nessun paziente con testi clinici normalizzati trovato.\n"
                "Elabora prima i documenti di almeno un paziente.",
            )
            return

        reply = QMessageBox.question(
            self, "Ricostruzione batch",
            f"Ricostruire il Clinical State per TUTTI i {len(patient_ids)} "
            f"pazienti con testi clinici normalizzati?\n\n"
            f"Verranno elaborati fino a 2 pazienti in parallelo.\n"
            f"I Clinical State esistenti verranno sovrascritti.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        dialog = _BatchBuildProgressDialog(
            patient_ids, self.window()
        )
        worker = BatchClinicalStateWorker(
            cs_manager, patient_ids, max_concurrent=2
        )
        worker.progress.connect(dialog.on_progress)
        worker.patient_started.connect(dialog.on_patient_started)
        worker.patient_completed.connect(dialog.on_patient_completed)
        worker.patient_failed.connect(dialog.on_patient_failed)
        worker.all_completed.connect(
            lambda s, f: self._on_batch_complete(dialog, s, f)
        )
        worker.start()
        dialog.exec()

    def _on_batch_complete(self, dialog, success: int, failed: int):
        """Called when all patients in a batch build are done."""
        dialog.on_all_completed(success, failed)
        # Refresh the current patient's view if it was rebuilt.
        self._refresh()
        main_window = self.window()
        if hasattr(main_window, "patient_panel"):
            main_window.patient_panel.refresh()

    # ------------------------------------------------------------------
    # Excel export
    # ------------------------------------------------------------------

    def _on_export_excel(self):
        """Export the current patient's ClinicalState to a structured XLSX."""
        from PyQt6.QtWidgets import QFileDialog

        if not self._clinical_state:
            QMessageBox.information(
                self, "Nessun dato",
                "Nessun Clinical State da esportare.\n"
                "Ricostruisci prima il Clinical State per questo paziente.",
            )
            return

        pid = self._current_patient_id or "export"
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Salva Clinical State in Excel",
            f"clinical_state_{pid}.xlsx",
            "Excel (*.xlsx);;Tutti i file (*)",
        )
        if not file_path:
            return

        try:
            export_clinical_state_to_xlsx(self._clinical_state, file_path)
            QMessageBox.information(
                self, "Esportazione completata",
                f"Clinical State esportato con successo:\n{file_path}",
            )
        except Exception as e:
            QMessageBox.critical(
                self, "Errore esportazione",
                f"Impossibile esportare il Clinical State.\n\n{e}",
            )

    def _get_rebuildable_patients(self) -> list[str]:
        """Return patient IDs that have at least one document with an
        available normalized clinical text."""
        doc_repo = self._services.get("document_repo")
        patient_repo = self._services.get("patient_repo")
        if not doc_repo or not patient_repo:
            return []

        from ..config import WORKSPACES_DIR

        candidates = []
        for patient in patient_repo.list_all():
            pid = patient.id
            extraction_dir = WORKSPACES_DIR / pid / "extraction"
            if not extraction_dir.is_dir():
                continue
            # Look for active clinical text files (*.md that aren't raw/source)
            has_text = False
            for md_path in extraction_dir.glob("*.md"):
                stem = md_path.stem
                if stem.endswith(("_pages", "_words", "_tables",
                                  "_cleaned_source", "_source", "_raw")):
                    continue
                if (extraction_dir / f"{stem}_source.txt").exists():
                    has_text = True
                    break
            if has_text:
                candidates.append(pid)
        return candidates

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
        self._rebuild_btn.setEnabled(True)
        if uses_normalized_text:
            self._rebuild_btn.setToolTip(
                "Elimina il Clinical State e lo ricostruisce da tutti i "
                "testi clinici normalizzati tramite l'LLM.\n"
                "Estrae: diagnosi, terapie, tossicità, sintomi, "
                "biomarcatori, staging, procedure, allergie, comorbilità, "
                "ricoveri, performance status, imaging, follow-up."
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

        # Symptoms
        if cs.symptoms:
            root = QTreeWidgetItem([f"🫁 Sintomi ({len(cs.symptoms)})"])
            self._tree.addTopLevelItem(root)
            for s in cs.symptoms:
                parts = [
                    s.get("description", ""),
                    s.get("severity", ""),
                    s.get("status", ""),
                    s.get("attribution", ""),
                ]
                text = " | ".join(p for p in parts if p)
                item = QTreeWidgetItem([text])
                if s.get("date"):
                    item.setToolTip(0, f"Data: {s['date']}")
                root.addChild(item)
            root.setExpanded(True)

        # Performance status
        if cs.performance_status:
            ps = cs.performance_status
            parts = []
            ecog = ps.get("ecog_score")
            kps = ps.get("karnofsky_score")
            if ecog is not None:
                parts.append(f"ECOG {ecog}")
            if kps is not None:
                parts.append(f"Karnofsky {kps}")
            if ps.get("date"):
                parts.append(f"({ps['date']})")
            label = "📊 Performance Status — " + " ".join(parts)
            root = QTreeWidgetItem([label])
            self._tree.addTopLevelItem(root)

        # Procedures
        add_items("procedures", "🔧 Procedure", cs.procedures,
                  ["name", "date", "outcome"])

        # Hospitalizations
        if cs.hospitalizations:
            root = QTreeWidgetItem([f"🏨 Ricoveri ({len(cs.hospitalizations)})"])
            self._tree.addTopLevelItem(root)
            for h in cs.hospitalizations:
                parts = [
                    h.get("reason", ""),
                    h.get("department", ""),
                    h.get("admission_date", ""),
                ]
                text = " | ".join(p for p in parts if p)
                item = QTreeWidgetItem([text])
                if h.get("discharge_date"):
                    item.setToolTip(
                        0, f"Dimissione: {h['discharge_date']}"
                    )
                root.addChild(item)
            root.setExpanded(True)

        # Imaging findings
        if cs.imaging_findings:
            root = QTreeWidgetItem([
                f"🔬 Imaging ({len(cs.imaging_findings)})"
            ])
            self._tree.addTopLevelItem(root)
            for img in cs.imaging_findings:
                parts = [
                    img.get("exam_type", ""),
                    img.get("date", ""),
                    img.get("conclusion", ""),
                ]
                text = " | ".join(p for p in parts if p)
                item = QTreeWidgetItem([text])
                findings = img.get("findings", "")
                if findings:
                    item.setToolTip(0, findings)
                root.addChild(item)
            root.setExpanded(True)

        # Follow-up
        if cs.follow_up:
            root = QTreeWidgetItem([f"📅 Follow-up ({len(cs.follow_up)})"])
            self._tree.addTopLevelItem(root)
            for f in cs.follow_up:
                text = f.get("recommendation", f.get("date", str(f)))
                item = QTreeWidgetItem([text])
                if f.get("specialty"):
                    item.setToolTip(0, f"Specialità: {f['specialty']}")
                root.addChild(item)
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
                    child.setData(Qt.ItemDataRole.UserRole, observation)
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

        qwen_client = self._services.get("clinical_state_llm_client")
        event_repo = self._services.get("event_repo")

        if not qwen_client or not qwen_client.is_available:
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
            qwen_client, self._clinical_state.to_dict(), events, question
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

    def _on_query_all(self):
        """Run the current question against ALL patients."""
        from .workers import BatchQueryWorker

        question = self._query_text.toPlainText().strip()
        if not question:
            QMessageBox.warning(self, "Nessuna domanda",
                                "Scrivi una domanda prima di interrogarare tutti.")
            return

        cs_repo = self._services.get("cs_repo")
        event_repo = self._services.get("event_repo")
        qwen_client = self._services.get("clinical_state_llm_client")
        patient_repo = self._services.get("patient_repo")

        if not qwen_client or not qwen_client.is_available:
            QMessageBox.warning(self, "LLM non disponibile",
                                "Configura un modello LLM per il Clinical State.")
            return
        if not cs_repo or not patient_repo:
            return

        # Trova i pazienti che hanno un Clinical State salvato.
        patient_ids = []
        for p in patient_repo.list_all():
            if cs_repo.exists(p.id):
                patient_ids.append(p.id)

        if not patient_ids:
            QMessageBox.information(
                self, "Nessun dato",
                "Nessun paziente ha un Clinical State. "
                "Ricostruisci prima il Clinical State con "
                "'🔨 Ricostruisci TUTTI'.",
            )
            return

        dialog = _BatchQueryDialog(
            question, patient_ids, self.window()
        )
        worker = BatchQueryWorker(
            qwen_client, cs_repo, event_repo,
            patient_ids, question, max_concurrent=2,
        )
        worker.progress.connect(dialog.on_progress)
        worker.patient_result.connect(dialog.on_patient_result)
        worker.patient_skipped.connect(dialog.on_patient_skipped)
        worker.patient_error.connect(dialog.on_patient_error)
        worker.all_completed.connect(dialog.on_all_completed)
        worker.start()
        dialog.exec()

    def _local_query(self, question: str) -> str:
        """Local query when Qwen is not available — uses structured data."""
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


class _BatchBuildProgressDialog(QDialog):
    """Progress dialog for batch ClinicalState rebuild across patients."""

    def __init__(self, patient_ids: list[str], parent=None):
        super().__init__(parent)
        self.setWindowTitle("Ricostruzione Clinical State — Batch")
        self.setMinimumSize(600, 400)
        self.setModal(True)
        self._patient_ids = list(patient_ids)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        # Progress bar
        from PyQt6.QtWidgets import QProgressBar
        self._bar = QProgressBar()
        self._bar.setRange(0, 100)
        layout.addWidget(self._bar)

        # Status label
        self._status_label = QLabel("Inizializzazione...")
        layout.addWidget(self._status_label)

        # Patient status table
        self._table = QTableWidget()
        self._table.setColumnCount(4)
        self._table.setHorizontalHeaderLabels([
            "Paziente", "Stato", "Diagnosi", "Terapie"
        ])
        self._table.horizontalHeader().setStretchLastSection(True)
        self._table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self._table.setRowCount(len(self._patient_ids))
        for i, pid in enumerate(self._patient_ids):
            self._table.setItem(i, 0, QTableWidgetItem(pid))
            self._table.setItem(i, 1, QTableWidgetItem("⏳ In attesa"))
        layout.addWidget(self._table, stretch=1)

        # Close button
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        self._close_btn = QPushButton("Chiudi")
        self._close_btn.setEnabled(False)
        self._close_btn.clicked.connect(self.accept)
        btn_layout.addWidget(self._close_btn)
        layout.addLayout(btn_layout)

    # ------------------------------------------------------------------
    # Slots connected to BatchClinicalStateWorker signals
    # ------------------------------------------------------------------

    def on_progress(self, percent: int, message: str):
        self._bar.setValue(percent)
        self._status_label.setText(message)

    def on_patient_started(self, patient_id: str):
        self._update_status(patient_id, "🔄 In corso")

    def on_patient_completed(self, patient_id: str,
                              diagnosis_count: int, treatment_count: int):
        self._update_status(
            patient_id, "✅ Completato",
            str(diagnosis_count), str(treatment_count),
        )

    def on_patient_failed(self, patient_id: str, error: str):
        self._update_status(patient_id, f"❌ Fallito")
        # Store error in tooltip
        for i, pid in enumerate(self._patient_ids):
            if pid == patient_id:
                item = self._table.item(i, 1)
                if item:
                    item.setToolTip(error)
                break

    def on_all_completed(self, success: int, failed: int):
        self._bar.setValue(100)
        if failed == 0:
            self._status_label.setText(
                f"✅ {success} pazienti completati con successo."
            )
        else:
            self._status_label.setText(
                f"✅ {success} completati, ❌ {failed} falliti."
            )
        self._close_btn.setEnabled(True)
        self._close_btn.setText("Chiudi")

    def _update_status(self, patient_id: str, status: str,
                       diag: str = "", treat: str = ""):
        for i, pid in enumerate(self._patient_ids):
            if pid == patient_id:
                self._table.setItem(i, 1, QTableWidgetItem(status))
                if diag:
                    self._table.setItem(i, 2, QTableWidgetItem(diag))
                if treat:
                    self._table.setItem(i, 3, QTableWidgetItem(treat))
                break


class _BatchQueryDialog(QDialog):
    """Dialog showing batch query results across multiple patients."""

    def __init__(self, question: str, patient_ids: list[str], parent=None):
        super().__init__(parent)
        self.setWindowTitle("Batch Query — Clinical State")
        self.setMinimumSize(750, 450)
        self.setModal(True)
        self._question = question
        self._patient_ids = list(patient_ids)
        self._results = {}  # patient_id -> (status, answer_or_error)
        self._setup_ui()

    def _setup_ui(self):
        from PyQt6.QtWidgets import QProgressBar

        layout = QVBoxLayout(self)

        # Question display
        qlabel = QLabel(f"<b>Domanda:</b> {self._question}")
        qlabel.setWordWrap(True)
        layout.addWidget(qlabel)

        # Progress bar
        self._bar = QProgressBar()
        self._bar.setRange(0, 100)
        layout.addWidget(self._bar)

        # Status
        self._status_label = QLabel("Inizializzazione...")
        layout.addWidget(self._status_label)

        # Results table
        self._table = QTableWidget()
        self._table.setColumnCount(3)
        self._table.setHorizontalHeaderLabels([
            "Paziente", "Esito", "Risposta"
        ])
        self._table.horizontalHeader().setStretchLastSection(True)
        self._table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self._table.setRowCount(len(self._patient_ids))
        for i, pid in enumerate(self._patient_ids):
            self._table.setItem(i, 0, QTableWidgetItem(pid))
            self._table.setItem(i, 1, QTableWidgetItem("⏳ In attesa"))
        self._table.itemDoubleClicked.connect(self._show_full_answer)
        layout.addWidget(self._table, stretch=1)

        # Close button
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        self._close_btn = QPushButton("Chiudi")
        self._close_btn.setEnabled(False)
        self._close_btn.clicked.connect(self.accept)
        btn_layout.addWidget(self._close_btn)
        layout.addLayout(btn_layout)

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def on_progress(self, percent: int, message: str):
        self._bar.setValue(percent)
        self._status_label.setText(message)

    def on_patient_result(self, patient_id: str, answer: str):
        self._results[patient_id] = ("ok", answer)
        self._update_row(patient_id, "✅ OK",
                         self._truncate(answer, 120))

    def on_patient_skipped(self, patient_id: str, reason: str):
        self._results[patient_id] = ("skipped", reason)
        self._update_row(patient_id, "⏭️ Saltato", reason)

    def on_patient_error(self, patient_id: str, error: str):
        self._results[patient_id] = ("error", error)
        self._update_row(patient_id, "❌ Errore",
                         self._truncate(error, 120))

    def on_all_completed(self, total: int, success: int,
                         skipped: int, failed: int):
        self._bar.setValue(100)
        lines = [
            f"✅ {success} risposte",
        ]
        if skipped:
            lines.append(f"⏭️ {skipped} saltati")
        if failed:
            lines.append(f"❌ {failed} errori")
        self._status_label.setText(" | ".join(lines) + f"  (su {total})")
        self._close_btn.setEnabled(True)
        self._close_btn.setText("Chiudi")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _update_row(self, patient_id: str, status: str, summary: str):
        for i, pid in enumerate(self._patient_ids):
            if pid == patient_id:
                self._table.setItem(i, 1, QTableWidgetItem(status))
                item = QTableWidgetItem(summary)
                item.setToolTip(summary)
                self._table.setItem(i, 2, item)
                break

    def _show_full_answer(self, index):
        """Double-click a row to see the full answer in a popup."""
        row = index.row()
        if row < 0 or row >= len(self._patient_ids):
            return
        pid = self._patient_ids[row]
        result = self._results.get(pid)
        if not result:
            return
        _status, content = result
        dlg = QDialog(self)
        dlg.setWindowTitle(f"Risposta — {pid}")
        dlg.resize(600, 500)
        lay = QVBoxLayout(dlg)
        te = QTextEdit()
        te.setMarkdown(content)
        te.setReadOnly(True)
        lay.addWidget(te)
        cb = QPushButton("Chiudi")
        cb.clicked.connect(dlg.accept)
        lay.addWidget(cb)
        dlg.exec()

    @staticmethod
    def _truncate(text: str, max_len: int = 120) -> str:
        if len(text) <= max_len:
            return text
        return text[:max_len].rsplit(" ", 1)[0] + "…"


# ---------------------------------------------------------------------------
# Module-level export helpers
# ---------------------------------------------------------------------------

def export_clinical_state_to_xlsx(state, filepath: str) -> None:
    """Write a multi-sheet Excel workbook from a ClinicalState.

    Each entity type gets its own sheet with proper column headers,
    making the data immediately usable for analysis or reporting.
    """
    from openpyxl import Workbook
    from openpyxl.styles import Font, PatternFill

    HEADER_FILL = PatternFill(start_color="2c3e50", end_color="2c3e50",
                               fill_type="solid")
    HEADER_FONT = Font(color="ffffff", bold=True)

    def _style_header(ws, headers):
        for col_idx, h in enumerate(headers, start=1):
            cell = ws.cell(row=1, column=col_idx, value=h)
            cell.fill = HEADER_FILL
            cell.font = HEADER_FONT

    def _add_rows(ws, headers, rows):
        _style_header(ws, headers)
        for row_idx, row in enumerate(rows, start=2):
            for col_idx, value in enumerate(row, start=1):
                ws.cell(row=row_idx, column=col_idx, value=value)
        # Auto-adjust column widths (approximate)
        for col_idx in range(1, len(headers) + 1):
            max_len = len(str(headers[col_idx - 1]))
            for row in ws.iter_rows(min_col=col_idx, max_col=col_idx,
                                    min_row=2, max_row=ws.max_row):
                for cell in row:
                    if cell.value:
                        max_len = max(max_len, min(len(str(cell.value)), 50))
            ws.column_dimensions[
                ws.cell(row=1, column=col_idx).column_letter
            ].width = max_len + 3

    wb = Workbook()

    # --- Diagnosi attive ---
    if state.active_diagnoses:
        ws = wb.active
        ws.title = "Diagnosi Attive"
        headers = ["Nome", "Codice ICD", "Data", "Stato", "Documento", "Note"]
        rows = [
            [d.name, d.icd_code or "", d.date or "", d.status,
             d.source_document_id or "", d.notes or ""]
            for d in state.active_diagnoses
        ]
        _add_rows(ws, headers, rows)

    # --- Diagnosi pregresse ---
    if state.past_diagnoses:
        ws = wb.create_sheet("Diagnosi Pregresse")
        headers = ["Nome", "Codice ICD", "Data", "Stato", "Documento", "Note"]
        rows = [
            [d.name, d.icd_code or "", d.date or "", d.status,
             d.source_document_id or "", d.notes or ""]
            for d in state.past_diagnoses
        ]
        _add_rows(ws, headers, rows)

    # --- Terapie attive ---
    if state.active_treatments:
        ws = wb.create_sheet("Terapie Attive")
        headers = ["Nome", "Data Inizio", "Data Fine", "Stato",
                   "Linea", "Setting", "Documento", "Note"]
        rows = [
            [t.name, t.start_date or "", t.end_date or "", t.status,
             str(t.line or ""), t.setting or "",
             t.source_document_id or "", t.notes or ""]
            for t in state.active_treatments
        ]
        _add_rows(ws, headers, rows)

    # --- Terapie concluse ---
    if state.completed_treatments:
        ws = wb.create_sheet("Terapie Concluse")
        headers = ["Nome", "Data Inizio", "Data Fine", "Stato",
                   "Linea", "Setting", "Documento", "Note"]
        rows = [
            [t.name, t.start_date or "", t.end_date or "", t.status,
             str(t.line or ""), t.setting or "",
             t.source_document_id or "", t.notes or ""]
            for t in state.completed_treatments
        ]
        _add_rows(ws, headers, rows)

    # --- Tossicità ---
    if state.toxicities:
        ws = wb.create_sheet("Tossicità")
        headers = ["Nome", "Grado", "Data", "Stato",
                   "Correlato a", "Documento"]
        rows = [
            [t.name, t.grade if t.grade is not None else "",
             t.date or "", t.status, t.related_to or "",
             t.source_document_id or ""]
            for t in state.toxicities
        ]
        _add_rows(ws, headers, rows)

    # --- Sintomi ---
    if state.symptoms:
        ws = wb.create_sheet("Sintomi")
        headers = ["Descrizione", "Severità", "Data", "Stato",
                   "Attribuzione", "Documento"]
        rows = [
            [s.get("description", ""), s.get("severity", ""),
             s.get("date", ""), s.get("status", ""),
             s.get("attribution", ""), s.get("source_document_id", "")]
            for s in state.symptoms
        ]
        _add_rows(ws, headers, rows)

    # --- Biomarcatori ---
    if state.biomarkers:
        ws = wb.create_sheet("Biomarcatori")
        headers = ["Nome", "Valore", "Data", "Interpretazione"]
        rows = [
            [b.name, b.value or "", b.date or "", b.interpretation or ""]
            for b in state.biomarkers
        ]
        _add_rows(ws, headers, rows)

    # --- Procedure ---
    if state.procedures:
        ws = wb.create_sheet("Procedure")
        headers = ["Nome", "Data", "Esito", "Documento"]
        rows = [
            [p.name, p.date or "", p.outcome or "",
             p.source_document_id or ""]
            for p in state.procedures
        ]
        _add_rows(ws, headers, rows)

    # --- Allergie ---
    if state.allergies:
        ws = wb.create_sheet("Allergie")
        _add_rows(ws, ["Allergene"],
                  [[a] for a in state.allergies])

    # --- Ricoveri ---
    if state.hospitalizations:
        ws = wb.create_sheet("Ricoveri")
        headers = ["Motivo", "Data Ingresso", "Data Dimissione",
                   "Reparto", "Documento"]
        rows = [
            [h.get("reason", ""), h.get("admission_date", ""),
             h.get("discharge_date", ""), h.get("department", ""),
             h.get("source_document_id", "")]
            for h in state.hospitalizations
        ]
        _add_rows(ws, headers, rows)

    # --- Performance status ---
    if state.performance_status:
        ps = state.performance_status
        ws = wb.create_sheet("Performance Status")
        ecog = ps.get("ecog_score")
        kps = ps.get("karnofsky_score")
        _add_rows(ws, ["Parametro", "Valore", "Data"], [
            ["ECOG", str(ecog) if ecog is not None else "", ps.get("date", "")],
            ["Karnofsky", str(kps) if kps is not None else "", ps.get("date", "")],
        ])

    # --- Imaging ---
    if state.imaging_findings:
        ws = wb.create_sheet("Imaging")
        headers = ["Esame", "Data", "Referto", "Conclusione", "Documento"]
        rows = [
            [img.get("exam_type", ""), img.get("date", ""),
             img.get("findings", ""), img.get("conclusion", ""),
             img.get("source_document_id", "")]
            for img in state.imaging_findings
        ]
        _add_rows(ws, headers, rows)

    # --- Follow-up ---
    if state.follow_up:
        ws = wb.create_sheet("Follow-up")
        headers = ["Raccomandazione", "Data", "Specialità", "Documento"]
        rows = [
            [f.get("recommendation", ""), f.get("date", ""),
             f.get("specialty", ""), f.get("source_document_id", "")]
            for f in state.follow_up
        ]
        _add_rows(ws, headers, rows)

    # Remove the default empty sheet if we created any real ones.
    if wb.active and wb.active.title == "Sheet":
        wb.remove(wb.active)

    wb.save(filepath)

"""Clinical History tab — chronological timeline view with query capability."""

import json
from datetime import datetime

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTextEdit, QPushButton,
    QTreeWidget, QTreeWidgetItem, QComboBox, QLabel, QSplitter,
    QMessageBox, QProgressBar, QFileDialog, QMenu, QAction,
)
from PyQt5.QtCore import Qt

from ..models.clinical_timeline import CATEGORY_LABELS


# Predefined query templates for the clinical history
HISTORY_QUERIES = [
    ("— Prompt predefiniti —", ""),
    ("Qual e' la diagnosi principale?",
     "Qual e' la diagnosi oncologica principale del paziente e quando e' stata formulata?"),
    ("Quali terapie ha ricevuto il paziente?",
     "Elenca tutte le terapie oncologiche ricevute dal paziente, con date di inizio e fine, dosaggi e via di somministrazione."),
    ("Quali tossicita' sono state registrate?",
     "Elenca tutte le tossicita' e gli eventi avversi registrati, con data, grado CTCAE e correlazione con il trattamento."),
    ("Quali interventi chirurgici sono stati eseguiti?",
     "Elenca tutti gli interventi chirurgici e le procedure eseguite, con date ed esiti."),
    ("Quali esami di imaging rilevanti sono stati eseguiti?",
     "Elenca gli esami di imaging significativi e i loro risultati (progressione, risposta, stabilita')."),
    ("Qual e' lo stato attuale del paziente?",
     "Descrivi lo stato clinico attuale del paziente basandoti sulle ultime osservazioni registrate."),
    ("Mostra la storia completa",
     "Fornisci un riassunto cronologico completo della storia clinica del paziente."),
    ("Quali ricoveri ha avuto il paziente?",
     "Elenca tutti i ricoveri ospedalieri con date, motivo e reparto."),
]


class ClinicalHistoryTab(QWidget):
    """Chronological clinical history view with generation and query."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._services = {}
        self._current_patient_id = None
        self._timeline_entries = []
        self._clinical_profile = ""
        self._worker = None
        self._narrative_worker = None
        self._query_worker = None
        self._setup_ui()

    # ------------------------------------------------------------------
    # UI setup
    # ------------------------------------------------------------------

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        # ---- Top: Generation bar ---------------------------------------
        gen_layout = QHBoxLayout()

        self._gen_btn = QPushButton("🔄 Genera Registro Cronologico")
        self._gen_btn.setToolTip(
            "Elabora tutti i testi clinici normalizzati in ordine "
            "cronologico, estrae le informazioni cliniche e costruisce "
            "il registro temporale deduplicato."
        )
        self._gen_btn.clicked.connect(self._on_generate)
        gen_layout.addWidget(self._gen_btn)

        self._dedup_btn = QPushButton("🔍 Deduplica Registro")
        self._dedup_btn.setToolTip(
            "Analizza il registro cronologico esistente ed elimina le "
            "voci duplicate senza ri-estrarre i documenti."
        )
        self._dedup_btn.clicked.connect(self._on_dedup)
        self._dedup_btn.setEnabled(False)
        gen_layout.addWidget(self._dedup_btn)

        self._narrative_btn = QPushButton("📝 Genera Profilo Narrativo")
        self._narrative_btn.setToolTip(
            "Richiede al LLM di produrre una storia clinica narrativa "
            "basata sul registro cronologico. Da usare dopo aver generato "
            "il registro."
        )
        self._narrative_btn.clicked.connect(self._on_generate_narrative)
        self._narrative_btn.setEnabled(False)
        gen_layout.addWidget(self._narrative_btn)

        self._progress_bar = QProgressBar()
        self._progress_bar.setVisible(False)
        self._progress_bar.setMaximum(100)
        gen_layout.addWidget(self._progress_bar, stretch=1)

        self._status_label = QLabel("")
        gen_layout.addWidget(self._status_label)
        gen_layout.addStretch()

        layout.addLayout(gen_layout)

        # ---- Query bar --------------------------------------------------
        query_layout = QHBoxLayout()
        query_layout.addWidget(QLabel("Query:"))

        self._query_preset = QComboBox()
        for label, query in HISTORY_QUERIES:
            self._query_preset.addItem(label, query)
        self._query_preset.currentIndexChanged.connect(self._on_preset_changed)
        query_layout.addWidget(self._query_preset, stretch=1)

        self._query_text = QTextEdit()
        self._query_text.setMaximumHeight(50)
        self._query_text.setPlaceholderText(
            "Fai una domanda sulla storia clinica del paziente..."
        )
        query_layout.addWidget(self._query_text, stretch=2)

        self._query_btn = QPushButton("🔍 Interroga")
        self._query_btn.clicked.connect(self._run_query)
        self._query_btn.setEnabled(False)
        query_layout.addWidget(self._query_btn)

        layout.addLayout(query_layout)

        # ---- Main content: splitter timeline | profile + answer --------
        splitter = QSplitter(Qt.Horizontal)

        # Left: Flat chronological list with context menu
        self._tree = QTreeWidget()
        self._tree.setHeaderLabels(["Data", "Cat.", "Descrizione"])
        self._tree.setAlternatingRowColors(True)
        self._tree.setRootIsDecorated(False)
        self._tree.setContextMenuPolicy(Qt.CustomContextMenu)
        self._tree.customContextMenuRequested.connect(
            self._on_tree_context_menu
        )
        splitter.addWidget(self._tree)

        # Right: Profile + Answer
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)

        right_layout.addWidget(QLabel("<b>Profilo Clinico Narrativo</b>"))
        self._profile_text = QTextEdit()
        self._profile_text.setReadOnly(True)
        self._profile_text.setPlaceholderText(
            "Dopo aver generato il registro cronologico, clicca "
            "'Genera Profilo Narrativo' per produrre una sintesi "
            "narrativa tramite LLM."
        )
        right_layout.addWidget(self._profile_text, stretch=1)

        right_layout.addWidget(QLabel("<b>Risposta Query</b>"))
        self._answer_text = QTextEdit()
        self._answer_text.setReadOnly(True)
        self._answer_text.setPlaceholderText(
            "Seleziona un prompt predefinito o scrivi una domanda e "
            "clicca 'Interroga'."
        )
        right_layout.addWidget(self._answer_text, stretch=1)

        splitter.addWidget(right_widget)
        splitter.setSizes([500, 500])
        layout.addWidget(splitter, stretch=1)

        # ---- Bottom bar -------------------------------------------------
        bottom_layout = QHBoxLayout()

        self._refresh_btn = QPushButton("🔄 Aggiorna")
        self._refresh_btn.clicked.connect(
            lambda: self.load_patient(self._current_patient_id)
        )
        bottom_layout.addWidget(self._refresh_btn)

        self._export_btn = QPushButton("📥 Esporta")
        self._export_btn.clicked.connect(self._on_export)
        bottom_layout.addWidget(self._export_btn)

        bottom_layout.addStretch()

        self._clear_narrative_btn = QPushButton("🗑️ Elimina Profilo")
        self._clear_narrative_btn.setToolTip(
            "Cancella il profilo clinico narrativo (il registro "
            "cronologico rimane invariato)."
        )
        self._clear_narrative_btn.clicked.connect(self._on_clear_narrative)
        bottom_layout.addWidget(self._clear_narrative_btn)

        self._clear_registry_btn = QPushButton("🗑️ Elimina Registro")
        self._clear_registry_btn.setToolTip(
            "Cancella TUTTE le voci del registro cronologico e il profilo."
        )
        self._clear_registry_btn.clicked.connect(self._on_clear_registry)
        self._clear_registry_btn.setStyleSheet("color: #c0392b;")
        bottom_layout.addWidget(self._clear_registry_btn)

        self._count_label = QLabel("")
        bottom_layout.addWidget(self._count_label)

        layout.addLayout(bottom_layout)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_services(self, services: dict):
        self._services = services

    def load_patient(self, patient_id: str):
        """Refresh all data for the given patient."""
        self._current_patient_id = patient_id
        self._refresh()

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def _refresh(self):
        """Reload timeline entries and clinical profile from the database."""
        if not self._current_patient_id:
            self._tree.clear()
            self._profile_text.clear()
            self._answer_text.clear()
            return

        # Load timeline entries
        timeline_repo = self._services.get("timeline_repo")
        if timeline_repo:
            self._timeline_entries = timeline_repo.get_by_patient(
                self._current_patient_id
            )
        else:
            self._timeline_entries = []

        # Load clinical profile from ClinicalState
        cs_repo = self._services.get("cs_repo")
        if cs_repo:
            state = cs_repo.load(self._current_patient_id)
            self._clinical_profile = (
                state.clinical_profile if state else ""
            )
        else:
            self._clinical_profile = ""

        self._build_tree()
        self._profile_text.setMarkdown(
            self._clinical_profile
            or "*Nessun profilo narrativo generato. "
               "Clicca 'Genera Profilo Narrativo' per crearne uno.*"
        )
        has_entries = len(self._timeline_entries) > 0
        self._query_btn.setEnabled(has_entries)
        self._narrative_btn.setEnabled(has_entries)
        self._dedup_btn.setEnabled(has_entries)
        self._count_label.setText(
            f"{len(self._timeline_entries)} voci nel registro cronologico"
        )

    # ------------------------------------------------------------------
    # Tree building — flat chronological list
    # ------------------------------------------------------------------

    def _build_tree(self):
        self._tree.clear()
        if not self._timeline_entries:
            self._tree.addTopLevelItem(
                QTreeWidgetItem([
                    "Nessuna voce — clicca 'Genera Registro Cronologico'",
                    "", "",
                ])
            )
            return

        # Flat chronological list, most recent first
        sorted_entries = sorted(
            self._timeline_entries,
            key=lambda e: e.date_observed,
            reverse=True,
        )

        # Configure column widths
        self._tree.setColumnWidth(0, 110)   # Date
        self._tree.setColumnWidth(1, 100)   # Category
        self._tree.setColumnWidth(2, 500)   # Description

        for e in sorted_entries:
            cat_label = CATEGORY_LABELS.get(e.category, e.category)
            resolved = ""
            if e.date_resolved:
                resolved = f" → {e.date_resolved}"

            date_text = f"{e.date_observed}{resolved}"
            cat_icon = self._category_icon(e.category)

            child = QTreeWidgetItem([
                date_text,
                f"{cat_icon} {cat_label}",
                e.description,
            ])

            if e.status == "superseded":
                for col in range(3):
                    child.setForeground(col, Qt.gray)

            if e.source_texts:
                tooltip = "\n---\n".join(e.source_texts[:3])
                child.setToolTip(2, tooltip)

            # Color-code confidence
            if e.confidence < 0.6:
                child.setForeground(0, Qt.darkYellow)
                child.setForeground(1, Qt.darkYellow)
                child.setForeground(2, Qt.darkYellow)

            self._tree.addTopLevelItem(child)

    @staticmethod
    def _category_icon(category: str) -> str:
        icons = {
            "diagnosis": "🩺",
            "treatment": "💊",
            "procedure": "🩻",
            "surgery": "🔪",
            "toxicity": "⚠️",
            "adverse_event": "🚨",
            "imaging_finding": "🖼️",
            "laboratory": "🔬",
            "symptom": "🤒",
            "hospitalization": "🏥",
            "discharge": "📋",
            "follow_up": "📅",
            "other": "📌",
        }
        return icons.get(category, "📌")

    # ------------------------------------------------------------------
    # Context menu & deletion
    # ------------------------------------------------------------------

    def _on_tree_context_menu(self, pos):
        """Right-click menu to delete a single timeline entry."""
        item = self._tree.itemAt(pos)
        if not item:
            return

        entry_date = item.text(0)
        entry_desc = item.text(2)
        if not entry_desc:
            return

        menu = QMenu(self)
        delete_action = QAction("🗑️ Elimina questa voce", self)
        delete_action.triggered.connect(
            lambda: self._delete_single_entry(item)
        )
        menu.addAction(delete_action)
        menu.exec_(self._tree.viewport().mapToGlobal(pos))

    def _delete_single_entry(self, item):
        """Delete the timeline entry corresponding to the given tree item."""
        idx = self._tree.indexOfTopLevelItem(item)
        if idx < 0 or idx >= len(self._timeline_entries):
            return

        # Get the actual entry (tree is sorted reverse-chronological)
        sorted_entries = sorted(
            self._timeline_entries,
            key=lambda e: e.date_observed,
            reverse=True,
        )
        entry = sorted_entries[idx]

        reply = QMessageBox.question(
            self, "Conferma eliminazione",
            f"Eliminare questa voce dal registro?\n\n"
            f"{entry.date_observed} [{entry.category}]\n"
            f"{entry.description[:200]}",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return

        timeline_repo = self._services.get("timeline_repo")
        if timeline_repo:
            timeline_repo.delete_entry(entry.entry_id)
        self._refresh()

    def _on_clear_registry(self):
        """Delete the entire timeline registry and the narrative profile."""
        if not self._timeline_entries:
            return

        reply = QMessageBox.question(
            self, "Conferma eliminazione",
            f"Eliminare TUTTE le {len(self._timeline_entries)} voci del "
            f"registro cronologico e il profilo narrativo?\n\n"
            f"Questa operazione non e' reversibile.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return

        builder = self._services.get("clinical_history_builder")
        if builder:
            builder.clear_timeline(self._current_patient_id)
            builder.clear_narrative(self._current_patient_id)
        self._refresh()

    def _on_dedup(self):
        """Deduplicate the existing registry without re-extracting."""
        if not self._timeline_entries:
            return

        builder = self._services.get("clinical_history_builder")
        if not builder:
            return

        llm = self._services.get("clinical_state_llm_client")
        if not llm or not llm.is_available:
            QMessageBox.warning(
                self, "LLM non disponibile",
                "Il modello Clinical State non e' disponibile."
            )
            return

        from .workers import DedupWorker
        self._dedup_worker = DedupWorker(
            builder, self._current_patient_id
        )
        self._dedup_worker.finished.connect(self._on_dedup_finished)
        self._dedup_worker.error.connect(self._on_dedup_error)
        self._dedup_btn.setEnabled(False)
        self._dedup_btn.setText("⏳ Deduplica in corso...")
        self._dedup_worker.start()

    def _on_dedup_finished(self, removed: int):
        self._dedup_btn.setEnabled(True)
        self._dedup_btn.setText("🔍 Deduplica Registro")
        self._refresh()
        QMessageBox.information(
            self, "Deduplicazione completata",
            f"{removed} voci duplicate rimosse dal registro."
        )

    def _on_dedup_error(self, error: str):
        self._dedup_btn.setEnabled(True)
        self._dedup_btn.setText("🔍 Deduplica Registro")
        QMessageBox.critical(
            self, "Errore deduplicazione", error
        )

    def _on_clear_narrative(self):
        """Delete only the clinical profile narrative, keep the registry."""
        if not self._clinical_profile:
            QMessageBox.information(
                self, "Nessun profilo",
                "Nessun profilo narrativo da eliminare."
            )
            return

        reply = QMessageBox.question(
            self, "Conferma eliminazione",
            "Eliminare il profilo clinico narrativo?\n"
            "Il registro cronologico rimarra' invariato.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return

        builder = self._services.get("clinical_history_builder")
        if builder:
            builder.clear_narrative(self._current_patient_id)
        self._clinical_profile = ""
        self._profile_text.clear()
        self._profile_text.setPlaceholderText(
            "Profilo narrativo eliminato. "
            "Clicca 'Genera Profilo Narrativo' per ricrearlo."
        )

    # ------------------------------------------------------------------
    # Generation — chronological registry
    # ------------------------------------------------------------------

    def _on_generate(self):
        if not self._current_patient_id:
            QMessageBox.warning(
                self, "Nessun paziente",
                "Seleziona prima un paziente nel pannello di sinistra."
            )
            return

        llm = self._services.get("clinical_state_llm_client")
        if not llm or not llm.is_available:
            QMessageBox.warning(
                self, "LLM non disponibile",
                "Il modello LLM per il Clinical State non e' disponibile. "
                "Configuralo in Strumenti → Configura LLM."
            )
            return

        builder = self._services.get("clinical_history_builder")
        if not builder:
            QMessageBox.warning(
                self, "Servizio non disponibile",
                "Il ClinicalHistoryBuilder non e' inizializzato."
            )
            return

        from .workers import ClinicalHistoryWorker

        # Read parallel_workers from the persisted LLM config
        num_workers = 1
        llm_configs = self._services.get("llm_configs")
        if llm_configs:
            cs_config = llm_configs.get("clinical_state")
            if cs_config:
                num_workers = getattr(cs_config, "parallel_workers", 1)

        self._worker = ClinicalHistoryWorker(
            builder, self._current_patient_id,
            generate_narrative=False,
            num_workers=num_workers,
        )
        self._worker.progress.connect(self._on_generation_progress)
        self._worker.finished.connect(self._on_generation_finished)
        self._worker.error.connect(self._on_generation_error)
        self._gen_btn.setEnabled(False)
        self._narrative_btn.setEnabled(False)
        self._dedup_btn.setEnabled(False)
        self._progress_bar.setVisible(True)
        self._progress_bar.setValue(0)
        self._status_label.setText("Estrazione osservazioni cliniche...")
        self._worker.start()

    def _on_generation_progress(self, percent: int, message: str):
        self._progress_bar.setValue(percent)
        self._status_label.setText(message)

    def _on_generation_finished(self, result: dict):
        self._progress_bar.setVisible(False)
        self._gen_btn.setEnabled(True)
        self._narrative_btn.setEnabled(True)
        self._dedup_btn.setEnabled(True)

        total = result.get('total_entries', 0)
        dedup = result.get('deduplicated', 0)
        final = result.get('final_entries', 0)
        failed = result.get('documents_failed', 0)
        processed = result.get('documents_processed', 0)
        skipped = result.get('documents_skipped', 0)
        elapsed = result.get('elapsed_seconds')
        incremental = result.get('incremental', False)

        status_text = f"Completato: {final} voci (deduplicate: {dedup})"
        if elapsed is not None:
            status_text += f" in {elapsed:.0f}s"
        self._status_label.setText(status_text)

        if incremental and skipped:
            msg = (
                f"Registro aggiornato incrementalmente:\n\n"
                f"• {skipped} documenti già presenti (saltati)\n"
                f"• {processed} nuovi documenti analizzati\n"
                f"• {total} nuove osservazioni estratte\n"
                f"• {dedup} duplicati rimossi\n"
                f"• {final} voci totali nel registro"
            )
        else:
            msg = (
                f"Registro cronologico generato:\n\n"
                f"• {processed} documenti analizzati\n"
                f"• {total} osservazioni estratte\n"
                f"• {dedup} duplicati rimossi\n"
                f"• {final} voci finali nel registro"
            )
        if failed > 0:
            failed_ids = result.get('failed_doc_ids', [])
            msg += (
                f"\n\n⚠️ {failed} documenti non hanno prodotto voci: "
                f"{', '.join(failed_ids[:5])}"
                f"{'...' if len(failed_ids) > 5 else ''}"
                f"\n\nControlla il terminale per i dettagli sugli errori."
            )

        self._refresh()
        QMessageBox.information(self, "Registro Completato", msg)

    def _on_generation_error(self, error: str):
        self._progress_bar.setVisible(False)
        self._gen_btn.setEnabled(True)
        self._status_label.setText(f"Errore: {error}")
        QMessageBox.critical(
            self, "Errore Generazione",
            f"Errore durante la generazione del registro:\n\n{error}"
        )

    # ------------------------------------------------------------------
    # Narrative profile generation (optional, separate step)
    # ------------------------------------------------------------------

    def _on_generate_narrative(self):
        if not self._timeline_entries:
            QMessageBox.warning(
                self, "Nessun registro",
                "Genera prima il registro cronologico."
            )
            return

        llm = self._services.get("clinical_state_llm_client")
        if not llm or not llm.is_available:
            # Fallback: build simple markdown from entries
            lines = ["## Profilo Clinico Cronologico\n"]
            for e in sorted(
                self._timeline_entries, key=lambda x: x.date_observed
            ):
                resolved = ""
                if e.date_resolved:
                    resolved = f" → risolto {e.date_resolved}"
                cat_label = CATEGORY_LABELS.get(e.category, e.category)
                lines.append(
                    f"- **{e.date_observed}** [{cat_label}] "
                    f"{e.description}{resolved}"
                )
            self._clinical_profile = "\n".join(lines)
            self._profile_text.setMarkdown(self._clinical_profile)
            QMessageBox.information(
                self, "Profilo generato (locale)",
                "Profilo narrativo generato localmente (LLM non disponibile)."
            )
            return

        builder = self._services.get("clinical_history_builder")
        if not builder:
            return

        from .workers import NarrativeWorker
        self._narrative_worker = NarrativeWorker(
            builder, self._current_patient_id
        )
        self._narrative_worker.finished.connect(self._on_narrative_finished)
        self._narrative_worker.error.connect(self._on_narrative_error)
        self._narrative_btn.setEnabled(False)
        self._narrative_btn.setText("⏳ Generazione in corso...")
        self._narrative_worker.start()

    def _on_narrative_finished(self, narrative: str):
        self._clinical_profile = narrative
        self._profile_text.setMarkdown(narrative)
        self._narrative_btn.setEnabled(True)
        self._narrative_btn.setText("📝 Genera Profilo Narrativo")

    def _on_narrative_error(self, error: str):
        self._narrative_btn.setEnabled(True)
        self._narrative_btn.setText("📝 Genera Profilo Narrativo")
        QMessageBox.critical(
            self, "Errore",
            f"Errore durante la generazione del profilo narrativo:\n\n{error}"
        )

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def _on_preset_changed(self, index: int):
        query = self._query_preset.currentData()
        if query:
            self._query_text.setPlainText(query)

    def _run_query(self):
        question = self._query_text.toPlainText().strip()
        if not question:
            return

        if not self._timeline_entries:
            self._answer_text.setPlainText(
                "Nessuna storia clinica disponibile. "
                "Generala prima con il pulsante 'Genera Registro Cronologico'."
            )
            return

        llm = self._services.get("clinical_state_llm_client")
        if not llm or not llm.is_available:
            # Fallback to local keyword search
            self._answer_text.setMarkdown(self._local_search(question))
            return

        entries_data = [e.to_dict() for e in self._timeline_entries]

        from .workers import ClinicalHistoryQueryWorker
        self._query_worker = ClinicalHistoryQueryWorker(
            llm, entries_data, self._clinical_profile, question
        )
        self._query_worker.finished.connect(self._on_query_result)
        self._query_worker.error.connect(self._on_query_error)
        self._query_btn.setEnabled(False)
        self._query_btn.setText("⏳ Interrogazione in corso...")
        self._query_worker.start()

    def _on_query_result(self, answer: str):
        self._answer_text.setMarkdown(answer)
        self._query_btn.setEnabled(True)
        self._query_btn.setText("🔍 Interroga")

    def _on_query_error(self, error: str):
        self._answer_text.setPlainText(f"Errore: {error}")
        self._query_btn.setEnabled(True)
        self._query_btn.setText("🔍 Interroga")

    def _local_search(self, question: str) -> str:
        """Simple keyword-based search when LLM is unavailable."""
        q_lower = question.lower()
        results = []
        for e in self._timeline_entries:
            if (q_lower in e.description.lower()
                    or q_lower in e.category.lower()):
                resolved = ""
                if e.date_resolved:
                    resolved = f" (risolto: {e.date_resolved})"
                cat_label = CATEGORY_LABELS.get(e.category, e.category)
                results.append(
                    f"- **{e.date_observed}** [{cat_label}] "
                    f"{e.description}{resolved}"
                )
        if not results:
            return "Nessuna voce corrispondente trovata nel registro."
        return "### Risultati\n\n" + "\n".join(results[:20])

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def _on_export(self):
        if not self._timeline_entries:
            QMessageBox.information(
                self, "Nessun dato",
                "Nessun registro da esportare."
            )
            return

        path, _ = QFileDialog.getSaveFileName(
            self, "Esporta Registro Cronologico", "",
            "File Markdown (*.md);;File JSON (*.json);;File di testo (*.txt)"
        )
        if not path:
            return

        if path.endswith(".json"):
            data = {
                "patient_id": self._current_patient_id,
                "generated_at": datetime.now().isoformat(),
                "clinical_profile": self._clinical_profile,
                "entries": [e.to_dict() for e in self._timeline_entries],
            }
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        else:
            lines = [
                f"# Registro Cronologico — {self._current_patient_id}",
                "",
                f"*Generato il {datetime.now().strftime('%d/%m/%Y %H:%M')}*",
                "",
            ]
            if self._clinical_profile:
                lines += [
                    "## Profilo Clinico Narrativo",
                    "",
                    self._clinical_profile,
                    "",
                ]
            lines += [
                "## Elenco Cronologico",
                "",
            ]
            for e in sorted(
                self._timeline_entries, key=lambda x: x.date_observed
            ):
                resolved = ""
                if e.date_resolved:
                    resolved = f" → risolto: {e.date_resolved}"
                cat_label = CATEGORY_LABELS.get(e.category, e.category)
                lines.append(
                    f"- **{e.date_observed}** [{cat_label}] "
                    f"{e.description}{resolved}"
                )
            with open(path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines))

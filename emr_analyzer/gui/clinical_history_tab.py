"""Clinical History tab — chronological timeline view with query capability."""

from datetime import datetime

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTextEdit, QPushButton,
    QTreeWidget, QTreeWidgetItem, QComboBox, QLabel, QSplitter,
    QMessageBox, QProgressBar, QFileDialog, QMenu, QAction,
    QInputDialog, QCheckBox, QTextBrowser,
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor

from ..models.chat_message import ChatMessage
from ..models.clinical_timeline import CATEGORY_LABELS
from ..settings import (
    SETTINGS_PATH,
    load_chat_preferences,
    save_chat_preferences,
)
from ..utils.markdown_tables import render_markdown_to_html


# Predefined query templates for the clinical history
HISTORY_QUERIES = [
    ("— Prompt predefiniti —", ""),
    ("⚡ Analisi irAE (registro completo)",
     "__IRAE_ANALYSIS__"),
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

    def __init__(self, parent=None, settings_path=None):
        super().__init__(parent)
        self._services = {}
        self._current_patient_id = None
        self._timeline_entries = []
        self._clinical_profile = ""
        self._worker = None
        self._narrative_worker = None
        self._query_worker = None
        self._dedup_worker = None
        self._irae_worker = None
        # Per-patient chat trace: every prompt and response is persisted and
        # rendered as a timeline; switching patient replaces it entirely.
        self._chat_messages: list[ChatMessage] = []
        self._chat_transient_error = ""
        self._pending_query: dict | None = None
        self._chat_settings_path = settings_path or SETTINGS_PATH
        preferences = load_chat_preferences(self._chat_settings_path)
        self._use_conversation_context = bool(
            preferences.get("use_conversation_context", False)
        )
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

        self._irae_btn = QPushButton("⚡ Analisi irAE")
        self._irae_btn.setToolTip(
            "Analizza l'INTERO registro cronologico con il protocollo "
            "irAE (a chunk, senza limite delle ultime 100 voci) e mostra "
            "le tabelle degli eventi avversi immuno-correlati."
        )
        self._irae_btn.clicked.connect(self._on_irae_analysis)
        self._irae_btn.setEnabled(False)
        query_layout.addWidget(self._irae_btn)

        layout.addLayout(query_layout)

        # ---- Chat options --------------------------------------------------
        chat_options_layout = QHBoxLayout()
        self._use_context_check = QCheckBox(
            "Ricorda la conversazione (usa le risposte precedenti come "
            "contesto)"
        )
        self._use_context_check.setChecked(self._use_conversation_context)
        self._use_context_check.setToolTip(
            "Se attivo, ogni nuova risposta tiene conto delle domande e "
            "risposte precedenti per questo paziente. La preferenza è "
            "salvata globalmente."
        )
        self._use_context_check.stateChanged.connect(
            self._on_context_check_changed
        )
        chat_options_layout.addWidget(self._use_context_check)
        chat_options_layout.addStretch()
        layout.addLayout(chat_options_layout)

        # ---- Main content: splitter timeline | profile + answer --------
        splitter = QSplitter(Qt.Horizontal)

        # Left: Flat chronological list with context menu
        self._tree = QTreeWidget()
        self._tree.setHeaderLabels(["Data", "Cat.", "Descrizione"])
        self._tree.setAlternatingRowColors(True)
        self._tree.setRootIsDecorated(True)
        self._tree.itemExpanded.connect(self._on_event_expanded)
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

        chat_header = QHBoxLayout()
        chat_header.addWidget(QLabel("<b>Cronologia Domande & Risposte</b>"))
        chat_header.addStretch()
        self._clear_chat_btn = QPushButton("🗑️ Elimina cronologia")
        self._clear_chat_btn.setToolTip(
            "Cancella tutte le domande e risposte registrate per questo "
            "paziente. Il registro cronologico e il profilo restano "
            "invariati."
        )
        self._clear_chat_btn.clicked.connect(self._on_clear_chat)
        self._clear_chat_btn.setEnabled(False)
        chat_header.addWidget(self._clear_chat_btn)
        right_layout.addLayout(chat_header)

        self._chat_view = QTextBrowser()
        self._chat_view.setReadOnly(True)
        self._chat_view.setOpenExternalLinks(False)
        right_layout.addWidget(self._chat_view, stretch=1)
        self._render_chat()

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

        self._golden_export_btn = QPushButton("⭐ Esporta golden set")
        self._golden_export_btn.setToolTip(
            "Esporta le voci confermate dall'utente, le decisioni di "
            "validazione risolte e i valori di laboratorio validati come "
            "golden set per la valutazione dei prompt LLM."
        )
        self._golden_export_btn.clicked.connect(self._on_export_golden_set)
        bottom_layout.addWidget(self._golden_export_btn)

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
    # Background worker lifecycle
    # ------------------------------------------------------------------

    _WORKER_ATTRS = (
        "_worker", "_narrative_worker", "_dedup_worker", "_query_worker",
        "_irae_worker",
    )

    def _worker_running(self) -> bool:
        """True when any background worker of this tab is still running."""
        return any(
            getattr(self, attr, None) is not None
            and getattr(self, attr).isRunning()
            for attr in self._WORKER_ATTRS
        )

    def _running_operation_name(self) -> str | None:
        """Label of the operation currently running, if any."""
        running = (
            ("_irae_worker", "l'analisi irAE"),
            ("_query_worker", "l'interrogazione"),
            ("_worker", "la generazione del registro"),
            ("_dedup_worker", "la deduplicazione"),
            ("_narrative_worker", "la generazione del profilo narrativo"),
        )
        for attr, label in running:
            worker = getattr(self, attr, None)
            if worker is not None and worker.isRunning():
                return label
        return None

    def _guard_busy(self) -> bool:
        """Block an action while a worker is running (returns True)."""
        operation = self._running_operation_name()
        if operation is not None:
            QMessageBox.information(
                self, "Operazione in corso",
                f"Attendi il completamento di {operation} prima di "
                "avviarne un'altra.",
            )
            return True
        return False

    def _update_busy_ui(self) -> None:
        """Grey out the irAE entry points while a worker runs or when the
        patient has no registry (silent no-ops are never acceptable)."""
        busy = self._worker_running()
        has_entries = bool(self._timeline_entries)
        self._irae_btn.setEnabled(has_entries and not busy)
        index = self._query_preset.findData("__IRAE_ANALYSIS__")
        if index >= 0:
            self._query_preset.model().item(index).setEnabled(
                has_entries and not busy
            )

    def _release_worker(self, attr: str) -> None:
        """Drop a finished worker safely (thread already ended).

        QThread objects must never be garbage-collected while their run()
        is executing: Qt aborts the process ("Destroyed while thread is
        still running").  This slot is connected to the ``finished``
        signal, which fires only after the thread has stopped.
        """
        worker = getattr(self, attr, None)
        setattr(self, attr, None)
        if worker is not None:
            worker.deleteLater()
        self._update_busy_ui()

    def shutdown(self) -> None:
        """Wait for running workers (application quit path)."""
        for attr in self._WORKER_ATTRS:
            worker = getattr(self, attr, None)
            if worker is not None and worker.isRunning():
                worker.wait(5000)

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def _refresh(self):
        """Reload timeline entries and clinical profile from the database."""
        if not self._current_patient_id:
            self._tree.clear()
            self._profile_text.clear()
            self._chat_messages = []
            self._chat_transient_error = ""
            self._render_chat()
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
        self._update_busy_ui()
        self._count_label.setText(
            f"{len(self._timeline_entries)} voci nel registro cronologico"
        )

        # Load the chat trace of THIS patient only — the single reload
        # funnel guarantees that switching patients never shows another
        # patient's conversation.
        self._chat_messages = self._load_chat_messages()
        self._chat_transient_error = ""
        self._render_chat()

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
            child.setData(0, Qt.UserRole, e.entry_id)
            child.setData(0, Qt.UserRole + 1, e.is_golden)

            tooltip = ""
            if e.source_texts:
                tooltip = "\n---\n".join(e.source_texts[:3])
                child.setToolTip(2, tooltip)

            if e.is_golden:
                # User-confirmed entry — the gold highlight wins over the
                # other color codes below.
                child.setText(0, f"⭐ {date_text}")
                child.setToolTip(
                    2,
                    (f"{tooltip}\n\n" if tooltip else "")
                    + "Confermata dall'utente (golden set)",
                )
                gold = QColor(0x9A, 0x6A, 0x00)  # dark goldenrod
                for col in range(3):
                    child.setForeground(col, gold)
            elif e.status == "superseded":
                for col in range(3):
                    child.setForeground(col, Qt.gray)
            elif e.confidence < 0.6:
                # Color-code confidence
                child.setForeground(0, Qt.darkYellow)
                child.setForeground(1, Qt.darkYellow)
                child.setForeground(2, Qt.darkYellow)

            if str(e.entry_id).startswith("EVT_"):
                placeholder = QTreeWidgetItem(["", "", "Caricamento evidenze…"])
                placeholder.setData(0, Qt.UserRole + 2, "placeholder")
                child.addChild(placeholder)

            self._tree.addTopLevelItem(child)

    def _on_event_expanded(self, item):
        """Load every fused, excluded and conflicting source only on demand."""
        event_id = item.data(0, Qt.UserRole)
        if not str(event_id or "").startswith("EVT_"):
            return
        if item.childCount() != 1 or (
            item.child(0).data(0, Qt.UserRole + 2) != "placeholder"
        ):
            return
        item.takeChildren()
        registry_repo = self._services.get("registry_repo")
        detail = registry_repo.get_event_detail(event_id) if registry_repo else None
        if not detail:
            item.addChild(QTreeWidgetItem(["", "", "Dettaglio non disponibile"])); return

        event = detail["event"]
        item.addChild(QTreeWidgetItem([
            "Prima evidenza",
            event.get("date_precision") or "data n.d.",
            event.get("first_evidence_date") or "non determinabile",
        ]))
        evidence_root = QTreeWidgetItem([
            "", "Evidenze", f"{len(detail.get('evidence', []))} fonti collegate",
        ])
        for evidence in detail.get("evidence", []):
            relation = evidence.get("relation") or "supports"
            date = evidence.get("observed_date") or evidence.get("source_document_date") or "data n.d."
            source = evidence.get("source_text") or "(passaggio non disponibile)"
            citation = f"doc {evidence.get('document_id') or '?'}"
            if evidence.get("source_page"):
                citation += f", p. {evidence['source_page']}"
            included = "in sintesi" if evidence.get("included_in_summary") else "esclusa dalla sintesi"
            evidence_root.addChild(QTreeWidgetItem([
                str(date), relation,
                f"[{citation}; {included}] {source}",
            ]))
        item.addChild(evidence_root)

        updates = detail.get("updates", [])
        if updates:
            update_root = QTreeWidgetItem(["", "Aggiornamenti", str(len(updates))])
            for update in updates:
                update_root.addChild(QTreeWidgetItem([
                    update.get("update_date") or "data n.d.",
                    update.get("status_after") or "aggiornamento",
                    update.get("summary") or "",
                ]))
            item.addChild(update_root)

        reviews = detail.get("reviews", [])
        if reviews:
            review_root = QTreeWidgetItem(["", "Revisioni", str(len(reviews))])
            for review in reviews:
                review_root.addChild(QTreeWidgetItem([
                    str(review.get("created_at") or "")[:10],
                    review.get("decision") or "",
                    review.get("reason") or "Decisione del revisore",
                ]))
            item.addChild(review_root)

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
        """Right-click menu to confirm, edit or delete a timeline entry."""
        item = self._tree.itemAt(pos)
        if not item:
            return

        entry_id = item.data(0, Qt.UserRole)
        if not entry_id:
            return
        is_golden = bool(item.data(0, Qt.UserRole + 1))

        menu = QMenu(self)

        if is_golden:
            confirm_action = QAction("⭐ Rimuovi conferma golden", self)
        else:
            confirm_action = QAction("⭐ Conferma come golden", self)
        confirm_action.setToolTip(
            "Marca la voce come confermata dall'utente: entra nel golden "
            "set usato per valutare i prompt di estrazione."
        )
        confirm_action.triggered.connect(
            lambda: self._toggle_golden(item)
        )
        menu.addAction(confirm_action)

        edit_action = QAction("✏️ Modifica descrizione canonica", self)
        edit_action.setToolTip(
            "Modifica la descrizione e marca automaticamente la voce come "
            "confermata (golden set)."
        )
        edit_action.triggered.connect(
            lambda: self._edit_canonical(item)
        )
        menu.addAction(edit_action)

        menu.addSeparator()

        delete_action = QAction("🗑️ Elimina questa voce", self)
        delete_action.triggered.connect(
            lambda: self._delete_single_entry(item)
        )
        menu.addAction(delete_action)
        menu.exec_(self._tree.viewport().mapToGlobal(pos))

    def _toggle_golden(self, item):
        """Confirm a timeline entry as golden, or remove the confirmation."""
        entry_id = item.data(0, Qt.UserRole)
        if not entry_id:
            return
        new_state = not bool(item.data(0, Qt.UserRole + 1))
        review_repo = self._services.get("review_repo")
        if review_repo and str(entry_id).startswith("EVT_"):
            review_repo.decide_event(
                self._current_patient_id,
                entry_id,
                "accepted" if new_state else "deferred",
                reason=(
                    "Evento confermato dal clinico"
                    if new_state else "Conferma rimossa: richiede nuova revisione"
                ),
            )
        timeline_repo = self._services.get("timeline_repo")
        if timeline_repo:
            timeline_repo.set_golden(entry_id, new_state)
        self._refresh()

    def _edit_canonical(self, item):
        """Edit the canonical description (implies a golden confirmation)."""
        entry_id = item.data(0, Qt.UserRole)
        if not entry_id:
            return
        current = item.text(2)

        text, ok = QInputDialog.getText(
            self,
            "Modifica descrizione canonica",
            "Nuova descrizione (la voce sara' confermata come golden):",
            text=current,
        )
        if not ok or not text.strip():
            return

        timeline_repo = self._services.get("timeline_repo")
        review_repo = self._services.get("review_repo")
        if review_repo and str(entry_id).startswith("EVT_"):
            review_repo.decide_event(
                self._current_patient_id,
                entry_id,
                "corrected",
                corrected_value={"summary_short": text.strip()},
                reason="Descrizione canonica corretta manualmente",
            )
        if timeline_repo:
            timeline_repo.update_description(entry_id, text.strip())
        self._refresh()

    def _delete_single_entry(self, item):
        """Delete the timeline entry corresponding to the given tree item."""
        entry_id = item.data(0, Qt.UserRole)
        if not entry_id:
            return
        entry = next(
            (e for e in self._timeline_entries if e.entry_id == entry_id),
            None,
        )
        if entry is None:
            return

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
        review_repo = self._services.get("review_repo")
        if review_repo and str(entry_id).startswith("EVT_"):
            review_repo.decide_event(
                self._current_patient_id,
                entry_id,
                "rejected",
                reason="Evento escluso manualmente dal registro clinico",
            )
        if timeline_repo:
            timeline_repo.delete_entry(entry_id)
        self._refresh()

    def _on_clear_registry(self):
        """Delete the entire timeline registry and the narrative profile."""
        if not self._timeline_entries:
            return

        reply = QMessageBox.question(
            self, "Conferma eliminazione",
            f"Eliminare TUTTE le {len(self._timeline_entries)} voci del "
            f"registro cronologico e il profilo narrativo?\n\n"
            f"Le evidenze originali resteranno conservate e ogni esclusione "
            f"sarà registrata nella revisione clinica.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return

        registry_repo = self._services.get("registry_repo")
        review_repo = self._services.get("review_repo")
        if registry_repo and review_repo:
            for event in registry_repo.get_events(self._current_patient_id):
                review_repo.decide_event(
                    self._current_patient_id,
                    event.event_id,
                    "rejected",
                    reason="Esclusione richiesta con azzeramento del registro",
                )
        builder = self._services.get("clinical_history_builder")
        if builder:
            builder.clear_timeline(self._current_patient_id)
            builder.clear_narrative(self._current_patient_id)
        self._refresh()

    def _on_dedup(self):
        """Deduplicate the existing registry without re-extracting."""
        if self._guard_busy():
            return
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
        self._dedup_worker.finished.connect(
            lambda _result, attr="_dedup_worker": self._release_worker(attr)
        )
        self._dedup_btn.setEnabled(False)
        self._dedup_btn.setText("⏳ Deduplica in corso...")
        self._dedup_worker.start()
        self._update_busy_ui()

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
        if self._guard_busy():
            return
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
        self._worker.finished.connect(
            lambda _result, attr="_worker": self._release_worker(attr)
        )
        self._gen_btn.setEnabled(False)
        self._narrative_btn.setEnabled(False)
        self._dedup_btn.setEnabled(False)
        self._progress_bar.setVisible(True)
        self._progress_bar.setValue(0)
        self._status_label.setText("Estrazione osservazioni cliniche...")
        self._worker.start()
        self._update_busy_ui()

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
        newly_extracted = result.get('atomic_evidence_extracted', 0)
        reused_blocks = result.get('exact_blocks_reused', 0)
        reused_evidence = result.get('reused_evidence', 0)
        verified_reuse_blocks = result.get(
            'unique_reuse_blocks_verified', 0
        )
        targeted_reuse = result.get('targeted_reuse_verifications', 0)
        full_fallbacks = result.get('full_document_fallbacks', 0)
        llm_calls = result.get('llm_calls', 0)
        source_chunks = result.get('source_chunks', 0)
        output_retries = result.get('output_limit_retries', 0)

        status_text = f"Completato: {final} voci (deduplicate: {dedup})"
        if elapsed is not None:
            status_text += f" in {elapsed:.0f}s"
        self._status_label.setText(status_text)

        if incremental and skipped:
            msg = (
                f"Registro aggiornato incrementalmente:\n\n"
                f"• {skipped} documenti già presenti (saltati)\n"
                f"• {processed} nuovi documenti analizzati\n"
                f"• {newly_extracted} nuove evidenze atomiche\n"
                f"• {total} evidenze complessive disponibili\n"
                f"• {dedup} duplicati rimossi\n"
                f"• {final} voci totali nel registro"
            )
        else:
            msg = (
                f"Registro cronologico generato:\n\n"
                f"• {processed} documenti analizzati\n"
                f"• {newly_extracted} evidenze atomiche estratte\n"
                f"• {total} evidenze complessive disponibili\n"
                f"• {dedup} duplicati rimossi\n"
                f"• {final} voci finali nel registro"
            )
        if llm_calls or reused_blocks:
            msg += (
                f"\n\nOttimizzazione:\n"
                f"• {llm_calls} chiamate LLM\n"
                f"• {source_chunks} segmenti clinici elaborati\n"
                f"• {output_retries} risposte scartate per limite output\n"
                f"• {reused_blocks} blocchi identici riutilizzati"
                f" ({reused_evidence} evidenze replicate con nuova fonte)\n"
                f"• {verified_reuse_blocks} blocchi unici verificati una volta\n"
                f"• {targeted_reuse} verifiche mirate sul documento\n"
                f"• {full_fallbacks} fallback completi di sicurezza"
            )
        if failed > 0:
            failed_ids = result.get('failed_doc_ids', [])
            msg += (
                f"\n\n⚠️ {failed} documenti non completati: "
                f"{', '.join(failed_ids[:5])}"
                f"{'...' if len(failed_ids) > 5 else ''}"
                f"\n\nUna nuova esecuzione riprenderà i soli documenti "
                f"mancanti; le evidenze già salvate non verranno ricalcolate."
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
        if self._guard_busy():
            return
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
        self._narrative_worker.finished.connect(
            lambda _result, attr="_narrative_worker": (
                self._release_worker(attr)
            )
        )
        self._narrative_btn.setEnabled(False)
        self._narrative_btn.setText("⏳ Generazione in corso...")
        self._narrative_worker.start()
        self._update_busy_ui()

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
        if query == "__IRAE_ANALYSIS__":
            # Sentinel entry: launch the full-registry analysis instead of
            # dumping the 23K-char protocol into the query box.
            self._query_preset.blockSignals(True)
            self._query_preset.setCurrentIndex(0)
            self._query_preset.blockSignals(False)
            self._on_irae_analysis()
            return
        if query:
            self._query_text.setPlainText(query)

    def _run_query(self):
        if self._guard_busy():
            return
        question = self._query_text.toPlainText().strip()
        if not question:
            return

        if not self._timeline_entries:
            self._chat_transient_error = (
                "Nessuna storia clinica disponibile. Generala prima con il "
                "pulsante 'Genera Registro Cronologico'."
            )
            self._render_chat()
            return

        # The conversation slice must NOT include the question being asked
        # now: build it BEFORE appending the user message.
        use_context = (
            self._use_context_check.isChecked()
            and len(self._chat_messages) >= 2
        )
        conversation = (
            self._conversation_slice() if use_context else None
        )

        patient_id = self._current_patient_id
        llm = self._services.get("clinical_state_llm_client")
        model_used = getattr(llm, "model", "") if llm is not None else ""

        # Persist + render the user prompt (the trace survives restarts).
        user_message = self._make_message(
            "user", question, model_used=model_used, context_mode=0
        )
        self._persist_message(user_message)
        self._chat_messages.append(user_message)
        self._chat_transient_error = ""
        self._render_chat()

        if not llm or not llm.is_available:
            # Fallback to local keyword search — recorded like an answer.
            answer = self._local_search(question)
            assistant_message = self._make_message(
                "assistant", answer,
                model_used="local_search", context_mode=0,
            )
            self._persist_message(assistant_message)
            self._chat_messages.append(assistant_message)
            self._render_chat()
            return

        entries_data = [e.to_dict() for e in self._timeline_entries]

        # Capture the patient at query time: if the user switches patients
        # while the worker runs, the answer is persisted to the RIGHT
        # patient and never rendered on the wrong panel.
        self._pending_query = {
            "patient_id": patient_id,
            "model_used": model_used,
            "context_mode": int(use_context),
        }

        from .workers import ClinicalHistoryQueryWorker
        self._query_worker = ClinicalHistoryQueryWorker(
            llm, entries_data, self._clinical_profile, question,
            conversation=conversation,
            use_conversation_context=use_context,
            registry_repo=self._services.get("registry_repo"),
            patient_id=patient_id,
        )
        self._query_worker.finished.connect(self._on_query_result)
        self._query_worker.error.connect(self._on_query_error)
        self._query_worker.finished.connect(
            lambda _result, attr="_query_worker": self._release_worker(attr)
        )
        self._query_btn.setEnabled(False)
        self._query_btn.setText("⏳ Interrogazione in corso...")
        self._query_worker.start()
        self._update_busy_ui()

    def _on_query_result(self, answer: str):
        pending = self._pending_query
        self._pending_query = None
        self._query_btn.setEnabled(bool(self._timeline_entries))
        self._query_btn.setText("🔍 Interroga")

        if pending is None:
            return
        assistant_message = self._make_message(
            "assistant", answer,
            model_used=pending["model_used"],
            context_mode=pending["context_mode"],
            patient_id=pending["patient_id"],
        )
        self._persist_message(assistant_message)
        # Render only when the tab still shows the patient who asked.
        if self._current_patient_id == pending["patient_id"]:
            self._chat_messages.append(assistant_message)
            self._render_chat()

    def _on_query_error(self, error: str):
        pending = self._pending_query
        self._pending_query = None
        self._query_btn.setEnabled(bool(self._timeline_entries))
        self._query_btn.setText("🔍 Interroga")

        # Transient, never persisted; only for the patient who asked.
        if (
            pending is not None
            and self._current_patient_id == pending["patient_id"]
        ):
            self._chat_transient_error = f"Errore: {error}"
            self._render_chat()

    def _on_irae_analysis(self) -> None:
        """Run the irAE protocol over the WHOLE registry, chunk by chunk."""
        if self._guard_busy():
            return
        if not self._timeline_entries:
            QMessageBox.information(
                self, "Nessun registro",
                "Nessuna storia clinica disponibile per questo paziente. "
                "Genera prima il registro cronologico.",
            )
            return

        llm = self._services.get("clinical_state_llm_client")
        if not llm or not llm.is_available:
            QMessageBox.warning(
                self, "LLM non disponibile",
                "Il modello Clinical State non è disponibile.",
            )
            return

        from ..clinical import irae_analysis

        try:
            prompt_path = irae_analysis.ensure_prompt()
            protocol = irae_analysis.load_prompt(prompt_path)
            entries_data = [e.to_dict() for e in self._timeline_entries]
            prompts = irae_analysis.build_analysis_plan(
                entries_data, self._clinical_profile, protocol
            )
        except OSError as exc:
            QMessageBox.warning(
                self, "Protocollo non disponibile", str(exc)
            )
            return
        except Exception as exc:
            QMessageBox.critical(
                self, "Analisi irAE non riuscita",
                f"Errore durante la preparazione dell'analisi: {exc}",
            )
            return
        if not protocol:
            QMessageBox.warning(
                self, "Protocollo non disponibile",
                f"Il file del protocollo irAE è vuoto: {prompt_path}",
            )
            return

        from .workers import IraeAnalysisWorker

        self._irae_worker = IraeAnalysisWorker(llm, prompts)
        self._irae_worker.progress.connect(self._on_irae_progress)
        self._irae_worker.finished.connect(self._on_irae_finished)
        self._irae_worker.error.connect(self._on_irae_error)
        self._irae_worker.finished.connect(
            lambda _result, attr="_irae_worker": self._release_worker(attr)
        )
        self._irae_btn.setEnabled(False)
        self._progress_bar.setVisible(True)
        self._progress_bar.setMaximum(len(prompts))
        self._progress_bar.setValue(0)
        self._status_label.setText("Analisi irAE del registro completo...")
        self._irae_worker.start()
        self._update_busy_ui()

    def _on_irae_progress(self, chunk_index: int, chunk_total: int) -> None:
        self._progress_bar.setValue(chunk_index)
        self._status_label.setText(
            f"Analisi irAE: parte {chunk_index}/{chunk_total}..."
        )

    def _on_irae_finished(self, combined_markdown: str) -> None:
        self._progress_bar.setVisible(False)
        self._status_label.setText("")
        self._irae_btn.setEnabled(bool(self._timeline_entries))
        from .irae_result_dialog import IraeResultDialog

        dialog = IraeResultDialog(
            combined_markdown, patient_id=self._current_patient_id or "",
            parent=self,
        )
        dialog.exec_()

    def _on_irae_error(self, error: str) -> None:
        self._progress_bar.setVisible(False)
        self._status_label.setText("")
        self._irae_btn.setEnabled(bool(self._timeline_entries))
        QMessageBox.critical(self, "Analisi irAE non riuscita", error)

    def _on_context_check_changed(self, state: int) -> None:
        self._use_conversation_context = bool(state)
        try:
            save_chat_preferences(
                {"use_conversation_context": bool(state)},
                self._chat_settings_path,
            )
        except OSError:
            pass  # preference persistence is best-effort

    # ------------------------------------------------------------------
    # Chat trace helpers
    # ------------------------------------------------------------------

    def _load_chat_messages(self) -> list[ChatMessage]:
        """Messages of the current patient, newest last."""
        chat_repo = self._services.get("chat_repo")
        if not chat_repo:
            return []
        try:
            return chat_repo.get_by_patient(self._current_patient_id)
        except Exception:
            return []

    def _make_message(
        self, role: str, content: str, *, model_used: str, context_mode: int,
        patient_id: str | None = None,
    ) -> ChatMessage:
        from ..database.chat_repo import ChatRepository

        return ChatMessage(
            id=ChatRepository.new_id(),
            patient_id=patient_id or self._current_patient_id or "",
            role=role,
            content=content,
            model_used=model_used,
            context_mode=int(context_mode),
            created_at=datetime.now().isoformat(),
        )

    def _persist_message(self, message: ChatMessage) -> None:
        chat_repo = self._services.get("chat_repo")
        if not chat_repo or not message.patient_id:
            return
        try:
            chat_repo.add_message(message)
        except Exception:
            pass  # the trace is best-effort; the UI keeps working

    def _conversation_slice(self) -> list[dict]:
        """Prior Q&A of the current patient, newest last (capped upstream)."""
        return [
            {"role": m.role, "content": m.content}
            for m in self._chat_messages
        ]

    def _on_clear_chat(self) -> None:
        """Delete the whole chat history of the current patient.

        Works regardless of the conversational-context checkbox (the
        toggle only affects how answers are generated, never the stored
        trace).
        """
        if not self._chat_messages:
            return
        reply = QMessageBox.question(
            self, "Conferma eliminazione",
            "Eliminare tutta la cronologia di domande e risposte "
            "per questo paziente?\nIl registro cronologico e il profilo "
            "narrativo restano invariati.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return

        chat_repo = self._services.get("chat_repo")
        if chat_repo and self._current_patient_id:
            try:
                chat_repo.clear_for_patient(self._current_patient_id)
            except Exception:
                pass  # the in-memory panel clears regardless
        self._chat_messages = []
        self._chat_transient_error = ""
        self._render_chat()

    def _render_chat(self) -> None:
        """Render the per-patient prompt/response timeline."""
        self._clear_chat_btn.setEnabled(bool(self._chat_messages))
        if not self._chat_messages and not self._chat_transient_error:
            self._chat_view.setHtml(
                "<i>La cronologia delle domande e risposte per questo "
                "paziente apparirà qui.</i>"
            )
            return

        lines = []
        for message in self._chat_messages:
            try:
                when = datetime.fromisoformat(
                    message.created_at
                ).strftime("%d/%m/%Y %H:%M")
            except (TypeError, ValueError):
                when = message.created_at or ""
            if message.role == "user":
                rendered = render_markdown_to_html(message.content)
                lines.append(
                    f"**🗨️ Tu** · {when}"
                    f"<br>&gt; {rendered}"
                )
            else:
                meta = []
                if message.model_used:
                    meta.append(f"modello `{message.model_used}`")
                if message.context_mode:
                    meta.append("contesto conversazionale")
                suffix = f" · {' · '.join(meta)}" if meta else ""
                rendered = render_markdown_to_html(message.content)
                lines.append(
                    f"**🤖 Assistente** · {when}{suffix}"
                    f"<br>{rendered}"
                )
            lines.append("---")
        if self._chat_transient_error:
            lines.append(
                f"<span style='color:#c0392b;'>⚠️ "
                f"{self._chat_transient_error}</span>"
            )
        self._chat_view.setHtml("<br>".join(lines))
        scrollbar = self._chat_view.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def _local_search(self, question: str) -> str:
        """Structured/FTS local search when the generation model is offline."""
        registry_repo = self._services.get("registry_repo")
        if registry_repo and self._current_patient_id:
            from ..clinical.query_service import ClinicalQueryService
            details = ClinicalQueryService(registry_repo).retrieve(
                self._current_patient_id, question, limit=50
            )
            if details:
                results = []
                for detail in details:
                    event = detail["event"]
                    sources = detail.get("evidence", [])
                    citations = " ".join(
                        f"[#{event['event_id']}; {source['document_id']}:"
                        f"p.{source.get('source_page') or 'n.d.'}]"
                        for source in sources
                    ) or f"[#{event['event_id']}]"
                    results.append(
                        f"- **{event.get('first_evidence_date') or 'data n.d.'}** "
                        f"{event['summary_short']} {citations}"
                    )
                return "### Risultati\n\n" + "\n".join(results)
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

        path, selected_filter = QFileDialog.getSaveFileName(
            self, "Esporta Registro Cronologico", "",
            "Markdown completo (*.md);;JSON completo (*.json);;"
            "Excel multi-foglio (*.xlsx);;CSV eventi (*.csv);;"
            "Documento Word (*.docx);;Documento PDF (*.pdf);;"
            "File di testo (*.txt)"
        )
        if not path:
            return
        if not any(path.lower().endswith(ext) for ext in (
            ".md", ".json", ".xlsx", ".csv", ".docx", ".pdf", ".txt",
        )):
            ext_by_filter = {
                "Markdown": ".md", "JSON": ".json", "Excel": ".xlsx",
                "CSV": ".csv", "Documento Word": ".docx",
                "Documento PDF": ".pdf", "File di testo": ".txt",
            }
            path += next(
                (extension for label, extension in ext_by_filter.items()
                 if selected_filter.startswith(label)),
                ".md",
            )
        try:
            from ..export.registry_export import ClinicalRegistryExporter
            ClinicalRegistryExporter(self._services).export(
                self._current_patient_id, path, include_sources=True,
            )
        except Exception as exc:
            QMessageBox.critical(
                self, "Errore esportazione",
                f"Impossibile esportare il registro completo:\n\n{exc}",
            )
            return
        QMessageBox.information(
            self, "Esportazione completata",
            f"Registro clinico completo esportato in:\n{path}",
        )

    def _on_export_golden_set(self):
        """Export the user-confirmed golden set for prompt evaluation."""
        from ..export.golden_set import save_golden_set

        path, _ = QFileDialog.getSaveFileName(
            self, "Esporta Golden Set", "",
            "File JSON (*.json)",
        )
        if not path:
            return

        timeline_repo = self._services.get("timeline_repo")
        if not timeline_repo:
            QMessageBox.warning(
                self, "Servizio non disponibile",
                "Il repository del registro cronologico non e' disponibile."
            )
            return

        try:
            count = save_golden_set(
                timeline_repo.db, path, self._current_patient_id
            )
        except Exception as exc:
            QMessageBox.critical(
                self, "Errore esportazione",
                f"Errore durante l'esportazione del golden set:\n\n{exc}"
            )
            return

        scope = (
            "del paziente selezionato"
            if self._current_patient_id
            else "di tutti i pazienti"
        )
        QMessageBox.information(
            self, "Golden set esportato",
            f"Esportate {count} voci timeline confermate {scope}.\n\n"
            f"Il golden set include anche le decisioni di validazione "
            f"risolte e i valori di laboratorio validati.\n\n"
            f"File: {path}"
        )

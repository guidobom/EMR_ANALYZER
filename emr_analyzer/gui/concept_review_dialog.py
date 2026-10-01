"""Project-wide review of concept → SNOMED CT codes, most frequent first."""

from __future__ import annotations

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QSplitter, QLabel, QPushButton, QLineEdit, QComboBox,
    QTableWidget, QTableWidgetItem, QAbstractItemView, QHeaderView, QListWidget,
    QListWidgetItem, QMessageBox, QWidget, QProgressBar,
)

from .event_review_tab import TYPE_LABELS, CODING_LABELS
from ..clinical.grounded_sources import METHOD
from ..clinical.snomed_coding import FACT_TYPE_TAGS, concept_key

STATUS_FILTERS = (("Tutti", None), ("Senza codifica", "missing"), ("Proposti", "proposed"),
                  ("Da rivedere", "needs_review"), ("Confermati", "confirmed"))


def project_concepts(db) -> list[dict]:
    """Every extracted concept with occurrence and patient counts."""
    concepts: dict = {}
    rows = db.execute("SELECT patient_id, normalized_entity, fact_type, source_text "
                      "FROM clinical_evidence WHERE extraction_method=?", (METHOD,))
    for row in rows:
        label = (row["normalized_entity"] or "").strip()
        if not label:
            continue
        key = concept_key(label, row["fact_type"])
        concept = concepts.setdefault(key, {"key": key, "label": label, "fact_type": row["fact_type"],
                                            "count": 0, "patients": set(), "examples": []})
        concept["count"] += 1
        concept["patients"].add(row["patient_id"])
        quote = " ".join(str(row["source_text"] or "").split())[:300]
        if quote and quote not in concept["examples"] and len(concept["examples"]) < 3:
            concept["examples"].append(quote)
    return sorted(concepts.values(), key=lambda concept: (-concept["count"], concept["label"].casefold()))


class ConceptReviewDialog(QDialog):
    def __init__(self, services: dict, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Concetti SNOMED CT del progetto")
        self.resize(1200, 760)
        self._services = services
        self._pipeline = services.get("extraction_pipeline")
        shared = services.get("shared_lexicon_repo")
        catalog = getattr(shared, "snomed_catalog", None) if shared is not None else None
        self._catalog = catalog if (catalog is not None and catalog.available) else None
        self._concepts: list[dict] = []
        self._mappings: dict = {}
        self._worker = None
        self._build()
        self._reload()

    def _build(self) -> None:
        layout = QVBoxLayout(self)
        intro = QLabel("Un codice confermato qui vale per tutte le occorrenze del concetto (stessa etichetta "
                       "e tipo di evento) in tutti i pazienti. Le eccezioni si correggono nella scheda "
                       "Eventi SNOMED del paziente.")
        intro.setWordWrap(True)
        layout.addWidget(intro)
        bar = QHBoxLayout()
        self._status_filter = QComboBox()
        for label, value in STATUS_FILTERS:
            self._status_filter.addItem(label, value)
        self._status_filter.currentIndexChanged.connect(self._render)
        bar.addWidget(self._status_filter)
        self._search = QLineEdit()
        self._search.setPlaceholderText("Cerca concetto o codice…")
        self._search.textChanged.connect(self._render)
        bar.addWidget(self._search, stretch=1)
        self._code_all_btn = QPushButton("Codifica concetti mancanti (tutti i pazienti)")
        self._code_all_btn.clicked.connect(self._code_all)
        bar.addWidget(self._code_all_btn)
        layout.addLayout(bar)
        self._progress = QProgressBar()
        self._progress.setVisible(False)
        layout.addWidget(self._progress)
        self._summary = QLabel("")
        layout.addWidget(self._summary)

        splitter = QSplitter(Qt.Horizontal)
        self._table = QTableWidget(0, 6)
        self._table.setHorizontalHeaderLabels(["Concetto", "Tipo", "Occorrenze", "Pazienti",
                                               "SNOMED CT", "Codifica"])
        self._table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._table.setSelectionMode(QAbstractItemView.SingleSelection)
        self._table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._table.verticalHeader().setVisible(False)
        self._table.itemSelectionChanged.connect(self._on_selection)
        splitter.addWidget(self._table)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        self._detail = QLabel("Seleziona un concetto.")
        self._detail.setWordWrap(True)
        self._detail.setTextInteractionFlags(Qt.TextSelectableByMouse)
        right_layout.addWidget(self._detail)
        search = QHBoxLayout()
        self._query = QLineEdit()
        self._query.setPlaceholderText("Cerca nel catalogo (termine inglese o codice)…")
        self._query.returnPressed.connect(self._search_catalog)
        search.addWidget(self._query, stretch=1)
        search_btn = QPushButton("Cerca")
        search_btn.clicked.connect(self._search_catalog)
        search.addWidget(search_btn)
        right_layout.addLayout(search)
        self._candidates = QListWidget()
        right_layout.addWidget(self._candidates, stretch=1)
        buttons = QHBoxLayout()
        self._confirm_btn = QPushButton("Conferma il codice proposto")
        self._confirm_btn.clicked.connect(self._confirm_proposed)
        self._choose_btn = QPushButton("Usa il codice selezionato")
        self._choose_btn.clicked.connect(self._choose_selected)
        self._none_btn = QPushButton("Nessun codice adatto")
        self._none_btn.clicked.connect(lambda: self._decide(None))
        self._reset_btn = QPushButton("Ripristina codifica automatica")
        self._reset_btn.setToolTip("Dimentica la proposta automatica: il concetto sarà ricodificato.")
        self._reset_btn.clicked.connect(self._reset)
        for button in (self._confirm_btn, self._choose_btn, self._none_btn, self._reset_btn):
            buttons.addWidget(button)
        right_layout.addLayout(buttons)
        splitter.addWidget(right)
        splitter.setSizes([700, 500])
        layout.addWidget(splitter, stretch=1)

    # ----------------------------------------------------------------- data
    def _mapping_repo(self):
        return getattr(getattr(self._pipeline, "coder", None), "mappings", None)

    def _reload(self, keep=None) -> None:
        db = self._services.get("db")
        self._concepts = project_concepts(db) if db is not None else []
        repo = self._mapping_repo()
        self._mappings = repo.get_many([concept["key"] for concept in self._concepts]) if repo else {}
        missing = sum(concept["key"] not in self._mappings for concept in self._concepts)
        confirmed = sum(row["status"] == "confirmed" for row in self._mappings.values())
        self._summary.setText(f"{len(self._concepts)} concetti · {missing} senza codifica · "
                              f"{confirmed} confermati")
        self._render(keep=keep)

    def _visible(self):
        wanted = self._status_filter.currentData()
        needle = self._search.text().strip().casefold()
        for concept in self._concepts:
            mapping = self._mappings.get(concept["key"])
            status = mapping["status"] if mapping else "missing"
            if wanted and status != wanted:
                continue
            if needle and needle not in f"{concept['label']} {(mapping or {}).get('code') or ''}".casefold():
                continue
            yield concept, mapping

    def _render(self, *_, keep=None) -> None:
        rows = list(self._visible())
        self._table.setRowCount(len(rows))
        selected = None
        for row, (concept, mapping) in enumerate(rows):
            code = f"{mapping['code']} {mapping.get('fsn') or ''}" if mapping and mapping.get("code") else "—"
            status = CODING_LABELS.get(mapping["status"], mapping["status"]) if mapping else "senza codifica"
            values = (concept["label"], TYPE_LABELS.get(concept["fact_type"], concept["fact_type"]),
                      concept["count"], len(concept["patients"]), code, status)
            for column, value in enumerate(values):
                item = QTableWidgetItem(str(value))
                if column == 0:
                    item.setData(Qt.UserRole, concept)
                self._table.setItem(row, column, item)
            if concept["key"] == keep:
                selected = row
        self._table.resizeColumnsToContents()
        self._table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        if selected is not None:
            self._table.selectRow(selected)

    def _current(self):
        row = self._table.currentRow()
        item = self._table.item(row, 0) if row >= 0 else None
        return item.data(Qt.UserRole) if item is not None else None

    def _on_selection(self) -> None:
        concept = self._current()
        if concept is None:
            return
        mapping = self._mappings.get(concept["key"]) or {}
        lines = [f"<b>{concept['label']}</b> · {TYPE_LABELS.get(concept['fact_type'], concept['fact_type'])}"
                 f" · {concept['count']} occorrenze in {len(concept['patients'])} pazienti"]
        if mapping.get("code"):
            lines.append(f"Codice: <b>{mapping['code']}</b> {mapping.get('fsn') or ''} "
                         f"({CODING_LABELS.get(mapping['status'], mapping['status'])})")
        elif mapping:
            lines.append(f"Nessun codice ({CODING_LABELS.get(mapping['status'], mapping['status'])})"
                         + (f": {mapping['note']}" if mapping.get("note") else ""))
        if mapping.get("english"):
            lines.append(f"Ricerca inglese: {mapping['english']}")
        lines += [f"«{quote}»" for quote in concept["examples"]]
        self._detail.setText("<br>".join(lines))
        self._candidates.clear()
        for row in mapping.get("candidates") or []:
            self._add_candidate(row)
        self._confirm_btn.setEnabled(bool(mapping.get("code")) and mapping.get("status") != "confirmed")

    def _add_candidate(self, row) -> None:
        item = QListWidgetItem(f"{row['code']} — {row.get('fsn') or row.get('term') or ''}")
        item.setData(Qt.UserRole, row)
        self._candidates.addItem(item)

    # -------------------------------------------------------------- actions
    def _search_catalog(self) -> None:
        concept = self._current()
        query = self._query.text().strip()
        if self._catalog is None:
            QMessageBox.information(self, "SNOMED CT", "Importa prima il catalogo SNOMED CT dal Lessico condiviso.")
            return
        if concept is None or not query:
            return
        self._candidates.clear()
        if query.isdigit():
            found = self._catalog.lookup(query)
            results = [found] if found and found["active"] else []
        else:
            results = self._catalog.search(query, query, limit=25, tags=FACT_TYPE_TAGS.get(concept["fact_type"]))
        for row in results:
            self._add_candidate(row)
        if not results:
            self._candidates.addItem("Nessun concetto trovato in questa gerarchia.")

    def _decide(self, code) -> None:
        concept = self._current()
        if concept is None or self._pipeline is None:
            return
        try:
            self._pipeline.review_service().confirm_concept(concept["label"], concept["fact_type"], code)
        except (ValueError, RuntimeError) as exc:
            QMessageBox.warning(self, "Codifica", str(exc))
            return
        self._reload(keep=concept["key"])

    def _confirm_proposed(self) -> None:
        concept = self._current()
        mapping = self._mappings.get(concept["key"]) if concept else None
        if mapping and mapping.get("code"):
            self._decide(mapping["code"])

    def _choose_selected(self) -> None:
        item = self._candidates.currentItem()
        row = item.data(Qt.UserRole) if item is not None else None
        if not row:
            QMessageBox.information(self, "Codifica", "Seleziona un concetto SNOMED CT nell'elenco.")
            return
        self._decide(row["code"])

    def _reset(self) -> None:
        concept = self._current()
        repo = self._mapping_repo()
        if concept is None or repo is None:
            return
        mapping = self._mappings.get(concept["key"])
        if mapping and mapping["status"] == "confirmed":
            QMessageBox.information(self, "Codifica", "La codifica è confermata: scegli un altro codice "
                                    "oppure «Nessun codice adatto».")
            return
        repo.reset([concept["key"]])
        self._reload(keep=concept["key"])

    def _code_all(self) -> None:
        from .pipeline_llm import prepare_pipeline
        from .workers import ConceptCodingWorker
        if self._pipeline is None or (self._worker is not None and self._worker.isRunning()):
            return
        if not prepare_pipeline(self._services, "atomic", self):
            return
        llm = self._services.get("atomic_evidence_llm_client")
        if llm is None or not getattr(llm, "is_available", False):
            QMessageBox.warning(self, "Modello non disponibile", "Configura e avvia il modello di estrazione.")
            return
        patients = sorted({patient for concept in self._concepts for patient in concept["patients"]})
        self._worker = ConceptCodingWorker(self._pipeline, patients, self)
        self._worker.progress.connect(lambda percent, message: (self._progress.setValue(percent),
                                                                self._summary.setText(message)))
        self._worker.result_ready.connect(self._coding_done)
        self._worker.error.connect(lambda message: self._coding_done({}, error=message))
        self._worker.cancelled.connect(lambda: self._coding_done({}, error="interrotta"))
        self._code_all_btn.setEnabled(False)
        self._progress.setVisible(True)
        self._worker.start()

    def _coding_done(self, result, error=None) -> None:
        self._code_all_btn.setEnabled(True)
        self._progress.setVisible(False)
        self._reload()
        if error:
            QMessageBox.warning(self, "Codifica", f"Codifica non completata: {error}")
        else:
            QMessageBox.information(self, "Codifica", f"{result.get('coded', 0)} concetti codificati, "
                                    f"{result.get('abstained', 0) + result.get('no_candidates', 0)} senza "
                                    f"candidato adatto, {result.get('errors', 0)} errori.")

    def reject(self) -> None:
        if self._worker is not None and self._worker.isRunning():
            self._worker.cancel()
            self._worker.wait(3000)
        super().reject()

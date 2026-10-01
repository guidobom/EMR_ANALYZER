"""Patient events: review extracted events, their SNOMED CT codes and sources."""

from __future__ import annotations

from pathlib import Path

from PyQt5.QtCore import Qt, QAbstractTableModel, QModelIndex, QTimer
from PyQt5.QtGui import QColor, QTextCursor
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QSplitter, QLabel, QPushButton, QProgressBar,
    QTableView, QAbstractItemView, QHeaderView, QLineEdit, QComboBox, QPlainTextEdit,
    QTextEdit, QMessageBox, QDialog, QDialogButtonBox, QFormLayout,
    QListWidget, QListWidgetItem, QTabWidget, QScrollArea, QCheckBox,
)

from .pipeline_llm import prepare_pipeline
from .qt_utils import qt_offset, python_offset
from ..clinical.snomed_coding import FACT_TYPE_TAGS
from ..models.clinical_pipeline import ATOMIC_FACT_TYPES

TYPE_LABELS = {
    "diagnosis": "Diagnosi", "symptom": "Sintomo", "clinical_sign": "Segno",
    "medication": "Farmaco", "procedure": "Procedura", "laboratory_test": "Laboratorio",
    "vital_sign": "Parametro vitale", "radiology_finding": "Imaging",
    "instrumental_finding": "Strumentale", "histopathology": "Istologia",
    "biomarker": "Biomarcatore", "clinical_decision": "Decisione",
    "hospitalization": "Ricovero", "discharge": "Dimissione",
}
ASSERTIONS = {"present": "presente", "absent": "negato", "unknown": "incerto"}
CERTAINTIES = {"confirmed": "certo", "suspected": "sospetto", "possible": "possibile", "unknown": "non determinato"}
SUBJECTS = {"patient": "paziente", "family": "familiare", "other": "altra persona", "unknown": "non determinato"}
TEMPORALITIES = {"": "—", "current": "attuale", "historical": "pregresso",
                 "hypothetical": "ipotetico", "unknown": "non determinato"}
PRECISIONS = {"day": "giorno", "month": "mese", "year": "anno", "unknown": "non nota"}
REVIEW_LABELS = {"proposed": "da revisionare", "needs_review": "da rivedere", "confirmed": "confermato",
                 "corrected": "corretto", "added": "aggiunto", "rejected": "scartato"}
CODING_LABELS = {"proposed": "proposto", "confirmed": "confermato", "needs_review": "da rivedere"}
FILTERS = ("Tutti", "Da revisionare", "Da rivedere o ricontrollare", "Revisionati", "Scartati",
           "Senza codice SNOMED", "Con codice SNOMED")


def _code_text(event) -> str:
    code = event.coding.get("code")
    if not code:
        return "—"
    mark = " ✓" if event.coding.get("status") == "confirmed" else ""
    return f"{code} {event.coding.get('term') or event.coding.get('fsn') or ''}{mark}"


class EventTableModel(QAbstractTableModel):
    HEADERS = ("Data", "Evento", "Tipo", "Asserzione", "SNOMED CT", "Revisione", "Referti",
               "Documento")

    def __init__(self, parent=None):
        super().__init__(parent)
        self.rows: list = []
        self.families: dict = {}

    def set_events(self, events, families=None) -> None:
        self.beginResetModel()
        self.rows = list(events)
        # statement key -> every occurrence of that statement in the patient
        self.families = families or {}
        self.endResetModel()

    def _family(self, event):
        return self.families.get(event.statement_key) or []

    def copies_text(self, event) -> str:
        family = self._family(event)
        if len(family) < 2:
            return "—"
        carrier = next((item.document_id for item in family
                        if (item.machine is not None
                            and (item.machine.data.get("statement_reuse") or {}).get("role") == "origin")),
                       family[-1].document_id)
        if event.document_id == carrier:
            return f"{len(family)} referti · origine"
        return f"{len(family)} referti · copia di {carrier}"

    def rowCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self.rows)

    def columnCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self.HEADERS)

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role == Qt.DisplayRole and orientation == Qt.Horizontal:
            return self.HEADERS[section]
        return None

    def _values(self, event):
        review = "da ricontrollare" if event.stale else REVIEW_LABELS.get(event.status, event.status)
        return (event.observed_date or "n.d.", event.label, TYPE_LABELS.get(event.fact_type, event.fact_type),
                ASSERTIONS.get(event.assertion, event.assertion), _code_text(event), review,
                self.copies_text(event),
                f"{event.document_id} ({event.document_date or 'n.d.'})")

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid():
            return None
        event = self.rows[index.row()]
        if role == Qt.DisplayRole:
            return self._values(event)[index.column()]
        if role == Qt.ToolTipRole:
            if index.column() == 6:
                family = self._family(event)
                if len(family) > 1:
                    return "Stesso enunciato in: " + ", ".join(
                        f"{item.document_id} ({item.document_date or 'n.d.'})" for item in family)
            return event.quote
        if role == Qt.ForegroundRole:
            if event.status == "rejected" or event.assertion == "absent":
                return QColor("#8a8a8a")
            if event.stale or event.status == "needs_review":
                return QColor("#a66a00")
            if event.status in ("confirmed", "corrected", "added"):
                return QColor("#1e7d32")
        return None

    def sort(self, column, order=Qt.AscendingOrder):
        self.layoutAboutToBeChanged.emit()
        self.rows.sort(key=lambda event: str(self._values(event)[column]).casefold(),
                       reverse=order == Qt.DescendingOrder)
        self.layoutChanged.emit()


class SourceTextView(QPlainTextEdit):
    """Read-only active text; the reviewer may select a new fragment."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setReadOnly(True)
        self.setLineWrapMode(QPlainTextEdit.WidgetWidth)
        self.text = ""

    def show_text(self, text: str, start=None, end=None) -> None:
        if text != self.text:
            self.text = text
            self.setPlainText(text)
        selections = []
        if start is not None and end is not None and 0 <= start < end <= len(text):
            highlight = QTextEdit.ExtraSelection()
            highlight.format.setBackground(QColor("#fff176"))
            cursor = self.textCursor()
            cursor.setPosition(qt_offset(text, start))
            cursor.setPosition(qt_offset(text, end), QTextCursor.KeepAnchor)
            highlight.cursor = cursor
            selections.append(highlight)
            caret = self.textCursor()
            caret.setPosition(qt_offset(text, start))
            self.setTextCursor(caret)
            self.centerCursor()
        self.setExtraSelections(selections)

    def selection(self) -> tuple[int, int] | None:
        cursor = self.textCursor()
        if not cursor.hasSelection():
            return None
        return (python_offset(self.text, cursor.selectionStart()),
                python_offset(self.text, cursor.selectionEnd()))


def _combo(options: dict) -> QComboBox:
    combo = QComboBox()
    for value, label in options.items():
        combo.addItem(label, value)
    return combo


def _select(combo: QComboBox, value) -> None:
    index = combo.findData(value if value is not None else "")
    combo.setCurrentIndex(max(0, index))


class EventReviewTab(QWidget):
    """One row per event; selecting it shows its source and its editor."""

    _WORKER_ATTRS = ("_worker",)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._services: dict = {}
        self._patient_id: str | None = None
        self._events: list = []
        self._current = None
        self._worker = None
        self._export_timer = QTimer(self)
        self._export_timer.setSingleShot(True)
        self._export_timer.timeout.connect(self._auto_export)
        self._setup_ui()

    # ------------------------------------------------------------------ UI
    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        actions = QHBoxLayout()
        self._run_btn = QPushButton("▶ Elabora paziente")
        self._run_btn.setToolTip("Estrae gli eventi dai referti non ancora elaborati, codifica i "
                                 "concetti nuovi in SNOMED CT e scrive il file FHIR del paziente.")
        self._run_btn.clicked.connect(self._on_run)
        actions.addWidget(self._run_btn)
        self._cancel_btn = QPushButton("⏹ Interrompi")
        self._cancel_btn.clicked.connect(self._on_cancel)
        self._cancel_btn.setVisible(False)
        actions.addWidget(self._cancel_btn)
        self._code_btn = QPushButton("Codifica concetti mancanti")
        self._code_btn.setToolTip("Codifica in SNOMED CT i concetti del paziente ancora senza codifica.")
        self._code_btn.clicked.connect(self._on_code_missing)
        actions.addWidget(self._code_btn)
        self._export_btn = QPushButton("Esporta FHIR")
        self._export_btn.clicked.connect(self._export_fhir)
        actions.addWidget(self._export_btn)
        self._fhir_btn = QPushButton("Apri FHIR")
        self._fhir_btn.clicked.connect(self._show_fhir)
        actions.addWidget(self._fhir_btn)
        self._progress = QProgressBar()
        self._progress.setVisible(False)
        actions.addWidget(self._progress, stretch=1)
        actions.addStretch()
        layout.addLayout(actions)

        self._status = QLabel("")
        self._status.setWordWrap(True)
        layout.addWidget(self._status)

        filters = QHBoxLayout()
        self._filter = QComboBox()
        self._filter.addItems(FILTERS)
        self._filter.currentIndexChanged.connect(self._render)
        filters.addWidget(self._filter)
        self._type_filter = QComboBox()
        self._type_filter.addItem("Tutti i tipi", "")
        for value in ATOMIC_FACT_TYPES:
            self._type_filter.addItem(TYPE_LABELS.get(value, value), value)
        self._type_filter.currentIndexChanged.connect(self._render)
        filters.addWidget(self._type_filter)
        self._search = QLineEdit()
        self._search.setPlaceholderText("Cerca evento, codice o citazione…")
        self._search.textChanged.connect(self._render)
        filters.addWidget(self._search, stretch=1)
        self._group_toggle = QCheckBox("Una riga per enunciato")
        self._group_toggle.setToolTip(
            "Mostra una sola riga per ogni enunciato ripetuto, con il numero di referti "
            "che lo riportano; le decisioni possono valere per tutte le copie.")
        self._group_toggle.stateChanged.connect(self._render)
        filters.addWidget(self._group_toggle)
        layout.addLayout(filters)

        splitter = QSplitter(Qt.Horizontal)
        self._model = EventTableModel(self)
        self._table = QTableView()
        self._table.setModel(self._model)
        self._table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._table.setSelectionMode(QAbstractItemView.SingleSelection)
        self._table.setAlternatingRowColors(True)
        self._table.setSortingEnabled(True)
        self._table.verticalHeader().setVisible(False)
        self._table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self._table.selectionModel().currentRowChanged.connect(lambda *_: self._on_selection())
        splitter.addWidget(self._table)

        right = QSplitter(Qt.Vertical)
        source_box = QWidget()
        source_layout = QVBoxLayout(source_box)
        source_layout.setContentsMargins(0, 0, 0, 0)
        self._detail = QLabel("Seleziona un evento per vedere il frammento sorgente.")
        self._detail.setWordWrap(True)
        self._detail.setTextInteractionFlags(Qt.TextSelectableByMouse)
        source_layout.addWidget(self._detail)
        self._source = SourceTextView()
        source_layout.addWidget(self._source, stretch=1)
        source_actions = QHBoxLayout()
        self._fragment_btn = QPushButton("Usa la selezione come frammento")
        self._fragment_btn.clicked.connect(self._on_use_selection)
        source_actions.addWidget(self._fragment_btn)
        self._new_btn = QPushButton("Nuovo evento dalla selezione")
        self._new_btn.clicked.connect(self._on_new_event)
        source_actions.addWidget(self._new_btn)
        self._edit_text_btn = QPushButton("Modifica testo del referto…")
        self._edit_text_btn.clicked.connect(self._on_edit_text)
        source_actions.addWidget(self._edit_text_btn)
        self._original_btn = QPushButton("Referto originale")
        self._original_btn.clicked.connect(self._open_original)
        source_actions.addWidget(self._original_btn)
        source_layout.addLayout(source_actions)
        right.addWidget(source_box)
        right.addWidget(self._build_editor())
        right.setSizes([380, 420])
        splitter.addWidget(right)
        splitter.setSizes([600, 560])
        layout.addWidget(splitter, stretch=1)
        self._set_editor_enabled(False)

    def _build_editor(self) -> QWidget:
        tabs = QTabWidget()
        fields = QWidget()
        form = QFormLayout(fields)
        self._label = QLineEdit()
        self._type = _combo({value: TYPE_LABELS.get(value, value) for value in ATOMIC_FACT_TYPES})
        self._assertion = _combo(ASSERTIONS)
        self._certainty = _combo(CERTAINTIES)
        self._subject = _combo(SUBJECTS)
        self._temporality = _combo(TEMPORALITIES)
        self._state = QLineEdit()
        self._date = QLineEdit()
        self._date.setPlaceholderText("AAAA, AAAA-MM o AAAA-MM-GG")
        self._date_end = QLineEdit()
        self._precision = _combo(PRECISIONS)
        self._value = QLineEdit()
        self._unit = QLineEdit()
        self._value_text = QLineEdit()
        self._attributes = QLineEdit()
        self._attributes.setPlaceholderText("chiave=valore; chiave=valore")
        self._note = QLineEdit()
        for label, widget in (("Evento", self._label), ("Tipo", self._type), ("Asserzione", self._assertion),
                              ("Certezza", self._certainty), ("Soggetto", self._subject),
                              ("Temporalità", self._temporality), ("Stato", self._state),
                              ("Data", self._date), ("Data fine", self._date_end),
                              ("Precisione data", self._precision), ("Valore numerico", self._value),
                              ("Unità", self._unit), ("Valore testuale", self._value_text),
                              ("Attributi", self._attributes), ("Nota", self._note)):
            form.addRow(label, widget)
        buttons = QHBoxLayout()
        self._fanout = QCheckBox("Su tutte le copie")
        self._fanout.setToolTip(
            "Applica la decisione a ogni referto che ripete questo enunciato. "
            "Il frammento resta per singolo referto: per spostarlo usare la selezione.")
        buttons.addWidget(self._fanout)
        self._confirm_btn = QPushButton("✓ Conferma")
        self._confirm_btn.clicked.connect(self._on_confirm)
        self._save_btn = QPushButton("Salva correzioni")
        self._save_btn.clicked.connect(self._on_save)
        self._reject_btn = QPushButton("Scarta")
        self._reject_btn.clicked.connect(self._on_reject)
        self._restore_btn = QPushButton("Ripristina estrazione")
        self._restore_btn.clicked.connect(self._on_restore)
        for button in (self._confirm_btn, self._save_btn, self._reject_btn, self._restore_btn):
            buttons.addWidget(button)
        form.addRow(buttons)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(fields)
        tabs.addTab(scroll, "Evento")

        coding = QWidget()
        coding_layout = QVBoxLayout(coding)
        self._coding_info = QLabel("")
        self._coding_info.setWordWrap(True)
        self._coding_info.setTextInteractionFlags(Qt.TextSelectableByMouse)
        coding_layout.addWidget(self._coding_info)
        search = QHBoxLayout()
        self._snomed_query = QLineEdit()
        self._snomed_query.setPlaceholderText("Cerca nel catalogo (termine inglese o codice)…")
        self._snomed_query.returnPressed.connect(self._on_snomed_search)
        search.addWidget(self._snomed_query, stretch=1)
        search_btn = QPushButton("Cerca")
        search_btn.clicked.connect(self._on_snomed_search)
        search.addWidget(search_btn)
        coding_layout.addLayout(search)
        self._candidates = QListWidget()
        coding_layout.addWidget(self._candidates, stretch=1)
        coding_buttons = QHBoxLayout()
        self._code_occurrence_btn = QPushButton("Usa per questa occorrenza")
        self._code_occurrence_btn.clicked.connect(lambda: self._apply_code(concept=False))
        self._code_concept_btn = QPushButton("Usa per tutte le occorrenze")
        self._code_concept_btn.setToolTip("Conferma il codice per l'etichetta e il tipo di evento in tutti i pazienti.")
        self._code_concept_btn.clicked.connect(lambda: self._apply_code(concept=True))
        self._no_code_btn = QPushButton("Nessun codice adatto")
        self._no_code_btn.clicked.connect(self._on_no_code)
        self._confirm_code_btn = QPushButton("Conferma il codice proposto")
        self._confirm_code_btn.clicked.connect(self._on_confirm_code)
        for button in (self._confirm_code_btn, self._code_occurrence_btn, self._code_concept_btn,
                       self._no_code_btn):
            coding_buttons.addWidget(button)
        coding_layout.addLayout(coding_buttons)
        tabs.addTab(coding, "Codice SNOMED CT")
        self._editor_tabs = tabs
        return tabs

    def _set_editor_enabled(self, enabled: bool) -> None:
        self._editor_tabs.setEnabled(enabled)
        for button in (self._fragment_btn, self._new_btn, self._edit_text_btn, self._original_btn):
            button.setEnabled(enabled)

    # --------------------------------------------------------------- data
    def set_services(self, services: dict) -> None:
        self._services = services

    def load_patient(self, patient_id: str | None) -> None:
        self._patient_id = patient_id
        self._current = None
        self._refresh()

    def _pipeline(self):
        return self._services.get("extraction_pipeline")

    def _review(self):
        pipeline = self._pipeline()
        return pipeline.review_service() if pipeline is not None else None

    def _refresh(self, keep: str | None = None) -> None:
        review = self._review()
        self._events = review.events(self._patient_id) if (review and self._patient_id) else []
        active = [event for event in self._events if event.status != "rejected"]
        coded = sum(bool(event.coding.get("code")) for event in active)
        pending = sum(event.status in ("proposed", "needs_review") or event.stale for event in active)
        pipeline = self._pipeline()
        status = pipeline.patient_status(self._patient_id) if (pipeline and self._patient_id) else {}
        self._status.setText(
            f"{len(active)} eventi · {coded} con codice SNOMED · {pending} da revisionare"
            + (f" · documenti elaborati {status['completed']}/{status['with_text']}"
               f" (non completati {status['failed']})" if status else ""))
        self._render(keep=keep or (self._current.key if self._current else None))

    def _visible(self):
        mode = self._filter.currentText()
        kind = self._type_filter.currentData()
        needle = self._search.text().strip().casefold()
        for event in self._events:
            if kind and event.fact_type != kind:
                continue
            if mode == "Da revisionare" and event.status not in ("proposed", "needs_review"):
                continue
            if mode == "Da rivedere o ricontrollare" and not (event.status == "needs_review" or event.stale):
                continue
            if mode == "Revisionati" and event.status not in ("confirmed", "corrected", "added"):
                continue
            if mode == "Scartati" and event.status != "rejected":
                continue
            if mode not in ("Scartati", "Tutti") and event.status == "rejected":
                continue
            if mode == "Senza codice SNOMED" and event.coding.get("code"):
                continue
            if mode == "Con codice SNOMED" and not event.coding.get("code"):
                continue
            if needle and needle not in " ".join(str(value or "") for value in (
                    event.label, event.coding.get("code"), event.coding.get("fsn"), event.quote,
                    event.document_id)).casefold():
                continue
            yield event

    def _render(self, *_, keep: str | None = None) -> None:
        events = sorted(self._visible(), key=lambda event: (
            event.observed_date or event.document_date or "", event.document_id, event.start or 0),
            reverse=True)
        families = {}
        for event in self._events:
            if event.statement_key:
                families.setdefault(event.statement_key, []).append(event)
        rows = events
        if getattr(self, "_group_toggle", None) is not None and self._group_toggle.isChecked():
            seen, rows = set(), []
            for event in events:
                key = event.statement_key or event.key
                if key in seen:
                    continue
                seen.add(key)
                rows.append(event)
        self._model.set_events(rows, families)
        self._table.resizeColumnsToContents()
        self._table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        row = next((index for index, event in enumerate(events) if event.key == keep), None)
        if row is not None:
            self._table.selectRow(row)
            self._on_selection()
        else:
            self._current = None
            self._set_editor_enabled(False)

    def _on_selection(self) -> None:
        index = self._table.currentIndex()
        if not index.isValid():
            return
        event = self._model.rows[index.row()]
        self._current = event
        self._set_editor_enabled(True)
        review = self._review()
        text = review.text(event.patient_id, event.document_id) if review else ""
        lines = [f"<b>{event.label}</b> · {TYPE_LABELS.get(event.fact_type, event.fact_type)} · "
                 f"{REVIEW_LABELS.get(event.status, event.status)}"
                 + (" · <span style='color:#a66a00'>da ricontrollare</span>" if event.stale else "")]
        if event.note:
            lines.append(f"<i>{event.note}</i>")
        if event.observed_date:
            lines.append(f"Data evento {event.observed_date}" + (f"–{event.observed_date_end}"
                         if event.observed_date_end else "") + f" · referto {event.document_date or 'n.d.'}")
        provenance = ((event.machine.data if event.machine is not None else {}) or {}).get("date_provenance") or {}
        origin = provenance.get("projected_from_document")
        if origin:
            lines.append(f"<span style='color:#5f6368'>Enunciato ripetuto: data e contenuto ereditati "
                         f"dal referto {origin} ({provenance.get('projected_from_date') or 'n.d.'})</span>")
        self._detail.setText("<br>".join(lines))
        self._source.show_text(text, None if event.stale else event.start, None if event.stale else event.end)
        self._load_editor(event)

    def _load_editor(self, event) -> None:
        self._label.setText(event.label)
        _select(self._type, event.fact_type)
        _select(self._assertion, event.assertion)
        _select(self._certainty, event.certainty)
        _select(self._subject, event.subject)
        _select(self._temporality, event.temporality or "")
        self._state.setText(event.state or "")
        self._date.setText(event.observed_date or "")
        self._date_end.setText(event.observed_date_end or "")
        _select(self._precision, event.date_precision or "unknown")
        self._value.setText("" if event.numeric_value is None else f"{event.numeric_value:g}")
        self._unit.setText(event.unit or "")
        self._value_text.setText(event.value_text or "")
        self._attributes.setText("; ".join(f"{key}={value}" for key, value in (event.attributes or {}).items()))
        self._note.setText("")
        self._restore_btn.setEnabled(event.override is not None)
        coding = event.coding
        source = "questa occorrenza" if coding.get("source") == "occurrence" else "tutte le occorrenze del concetto"
        if coding.get("code"):
            self._coding_info.setText(
                f"<b>{coding['code']}</b> {coding.get('fsn') or coding.get('term') or ''}<br>"
                f"{CODING_LABELS.get(coding.get('status'), coding.get('status'))} · vale per {source}")
        elif coding.get("status") == "confirmed":
            self._coding_info.setText(f"Nessun codice SNOMED CT adatto (confermato, vale per {source}).")
        else:
            self._coding_info.setText("Non codificato" + (f": {coding['note']}" if coding.get("note") else
                                      " — usa «Codifica concetti mancanti» o scegli un candidato."))
        self._confirm_code_btn.setEnabled(bool(coding.get("code")) and coding.get("status") != "confirmed")
        self._candidates.clear()
        for row in coding.get("candidates") or []:
            self._add_candidate(row)

    def _add_candidate(self, row) -> None:
        item = QListWidgetItem(f"{row['code']} — {row.get('fsn') or row.get('term') or ''}")
        item.setData(Qt.UserRole, row)
        self._candidates.addItem(item)

    # ------------------------------------------------------------ actions
    def _act(self, action) -> None:
        review = self._review()
        if review is None or self._current is None:
            return
        key = self._current.key
        try:
            action(review)
        except (ValueError, RuntimeError) as exc:
            QMessageBox.warning(self, "Revisione", str(exc))
            return
        self._refresh(keep=key)
        self._export_timer.start(3000)

    def _changes(self) -> dict:
        def number(text):
            text = text.strip().replace(",", ".")
            if not text:
                return None
            try:
                return float(text)
            except ValueError:
                raise ValueError(f"Valore numerico non valido: {text}")
        attributes = {}
        for part in self._attributes.text().split(";"):
            if "=" in part:
                key, value = part.split("=", 1)
                if key.strip():
                    attributes[key.strip()] = value.strip()
        for field in (self._date, self._date_end):
            value = field.text().strip()
            if value and not _valid_date(value):
                raise ValueError(f"Data non valida: {value} (usa AAAA, AAAA-MM o AAAA-MM-GG)")
        return {
            "label": self._label.text().strip(), "fact_type": self._type.currentData(),
            "assertion": self._assertion.currentData(), "certainty": self._certainty.currentData(),
            "subject": self._subject.currentData(), "temporality": self._temporality.currentData() or None,
            "state": self._state.text().strip() or None,
            "observed_date": self._date.text().strip() or None,
            "observed_date_end": self._date_end.text().strip() or None,
            "date_precision": self._precision.currentData(), "numeric_value": number(self._value.text()),
            "unit": self._unit.text().strip() or None, "value_text": self._value_text.text().strip() or None,
            "attributes": attributes,
        }

    def _on_confirm(self) -> None:
        self._act(lambda review: review.confirm(self._current, note=self._note.text().strip() or None,
                                                fanout=self._fanout.isChecked()))

    def _on_save(self) -> None:
        try:
            changes = self._changes()
        except ValueError as exc:
            QMessageBox.warning(self, "Revisione", str(exc))
            return
        self._act(lambda review: review.correct(self._current, changes,
                                                note=self._note.text().strip() or None,
                                                fanout=self._fanout.isChecked()))

    def _on_reject(self) -> None:
        self._act(lambda review: review.reject(self._current, note=self._note.text().strip() or None,
                                               fanout=self._fanout.isChecked()))

    def _on_restore(self) -> None:
        self._act(lambda review: review.restore(self._current, fanout=self._fanout.isChecked()))

    def _on_use_selection(self) -> None:
        selection = self._source.selection()
        if selection is None:
            QMessageBox.information(self, "Frammento", "Seleziona nel testo il nuovo frammento.")
            return
        start, end = selection
        self._act(lambda review: review.correct(self._current, {"start": start, "end": end}))

    def _on_new_event(self) -> None:
        selection = self._source.selection()
        if self._current is None or selection is None:
            QMessageBox.information(self, "Nuovo evento", "Seleziona nel testo il frammento dell'evento.")
            return
        dialog = QDialog(self)
        dialog.setWindowTitle("Nuovo evento")
        form = QFormLayout(dialog)
        label = QLineEdit(self._source.text[selection[0]:selection[1]].strip()[:80])
        kind = _combo({value: TYPE_LABELS.get(value, value) for value in ATOMIC_FACT_TYPES})
        form.addRow("Evento", label)
        form.addRow("Tipo", kind)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        form.addRow(buttons)
        if dialog.exec_() != QDialog.Accepted:
            return
        review = self._review()
        try:
            event = review.add(self._patient_id, self._current.document_id, *selection,
                               label=label.text(), fact_type=kind.currentData(),
                               document_date=self._current.document_date)
        except ValueError as exc:
            QMessageBox.warning(self, "Nuovo evento", str(exc))
            return
        self._filter.setCurrentIndex(0)
        self._refresh(keep=event.key)
        self._export_timer.start(3000)

    def _on_edit_text(self) -> None:
        event = self._current
        review = self._review()
        if event is None or review is None:
            return
        dialog = QDialog(self)
        dialog.setWindowTitle(f"Testo del referto {event.document_id}")
        dialog.resize(900, 700)
        layout = QVBoxLayout(dialog)
        notice = QLabel("Correggi il testo anonimizzato (per esempio errori di OCR). Viene salvato come "
                        "nuova versione; l'originale resta invariato e il referto sarà rielaborato alla "
                        "prossima elaborazione del paziente.")
        notice.setWordWrap(True)
        layout.addWidget(notice)
        editor = QPlainTextEdit(review.text(event.patient_id, event.document_id))
        layout.addWidget(editor, stretch=1)
        buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        if dialog.exec_() != QDialog.Accepted:
            return
        from ..clinical.document_text import save_text_overlay
        overlay = self._services.get("overlay_repo")
        if overlay is None:
            return
        save_text_overlay(overlay, event.patient_id, event.document_id, editor.toPlainText(),
                          audit_repo=self._services.get("audit_repo"))
        self._refresh(keep=event.key)
        if QMessageBox.question(self, "Testo salvato",
                                "Rielaborare ora il paziente? Verrà riletto solo il referto modificato.",
                                QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes) == QMessageBox.Yes:
            self._on_run()

    # ------------------------------------------------------------- coding
    def _catalog(self):
        shared = self._services.get("shared_lexicon_repo")
        catalog = getattr(shared, "snomed_catalog", None) if shared is not None else None
        return catalog if (catalog is not None and catalog.available) else None

    def _on_snomed_search(self) -> None:
        catalog = self._catalog()
        query = self._snomed_query.text().strip()
        if catalog is None:
            QMessageBox.information(self, "SNOMED CT", "Importa prima il catalogo SNOMED CT dal Lessico condiviso.")
            return
        if self._current is None or not query:
            return
        self._candidates.clear()
        if query.isdigit():
            found = catalog.lookup(query)
            results = [found] if found and found["active"] else []
        else:
            results = catalog.search(query, query, limit=25, tags=FACT_TYPE_TAGS.get(self._current.fact_type))
        for row in results:
            self._add_candidate(row)
        if not results:
            self._candidates.addItem("Nessun concetto trovato in questa gerarchia.")

    def _selected_candidate(self):
        item = self._candidates.currentItem()
        row = item.data(Qt.UserRole) if item is not None else None
        if not row:
            QMessageBox.information(self, "SNOMED CT", "Seleziona un concetto nell'elenco.")
        return row

    def _apply_code(self, *, concept: bool) -> None:
        row = self._selected_candidate()
        if not row:
            return
        event = self._current
        if concept:
            self._act(lambda review: review.confirm_concept(event.label, event.fact_type, row["code"]))
            return
        catalog = self._catalog()
        meta = catalog.metadata() if catalog else {}
        code = {"code": row["code"], "fsn": row.get("fsn"), "term": row.get("term"),
                "release": meta.get("version_uri") or meta.get("release")}
        self._act(lambda review: review.correct(event, {"code": code}))

    def _on_no_code(self) -> None:
        event = self._current
        if event is None:
            return
        answer = QMessageBox.question(
            self, "Nessun codice", "Vale per tutte le occorrenze di questo concetto?\n"
            "Sì = concetto · No = solo questa occorrenza",
            QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel, QMessageBox.Yes)
        if answer == QMessageBox.Yes:
            self._act(lambda review: review.confirm_concept(event.label, event.fact_type, None))
        elif answer == QMessageBox.No:
            self._act(lambda review: review.correct(event, {"code": {"code": None}}))

    def _on_confirm_code(self) -> None:
        event = self._current
        code = event.coding.get("code") if event else None
        if code:
            self._act(lambda review: review.confirm_concept(event.label, event.fact_type, code))

    # ------------------------------------------------------------- export
    def _auto_export(self) -> None:
        pipeline = self._pipeline()
        if pipeline is None or not self._patient_id or self._worker_running():
            return
        try:
            pipeline.export_patient(self._patient_id, use_llm=False)
        except Exception as exc:
            self._status.setText(f"FHIR non aggiornato: {exc}")

    def _export_fhir(self) -> None:
        pipeline = self._pipeline()
        if pipeline is None or not self._patient_id or self._worker_running():
            return
        try:
            result = pipeline.export_patient(self._patient_id)
        except Exception as exc:
            QMessageBox.warning(self, "Esportazione FHIR", f"Esportazione non riuscita: {exc}")
            return
        self._status.setText(f"FHIR aggiornato: {result['events']} risorse cliniche, "
                             f"{result['uncoded']} senza codice · {result['path']}")

    def _show_fhir(self) -> None:
        pipeline = self._pipeline()
        path = Path(pipeline.patient_status(self._patient_id)["fhir_path"]) if (pipeline and self._patient_id) else None
        if not path or not path.is_file():
            QMessageBox.information(self, "FHIR", "Il file FHIR non è presente: usa «Esporta FHIR».")
            return
        dialog = QDialog(self)
        dialog.setWindowTitle(str(path))
        dialog.resize(900, 700)
        layout = QVBoxLayout(dialog)
        view = QPlainTextEdit()
        view.setReadOnly(True)
        view.setPlainText(path.read_text(encoding="utf-8"))
        layout.addWidget(view)
        dialog.exec_()

    def _open_original(self) -> None:
        event = self._current
        documents = self._services.get("document_repo")
        document = documents.get_by_id(event.document_id) if (event and documents) else None
        if document is None:
            QMessageBox.information(self, "Referto", "Documento non disponibile nel progetto.")
            return
        from .pdf_viewer import PDFViewerDialog
        PDFViewerDialog(document.to_dict(), self._services, self, highlight={
            "evidence_id": event.key, "page_number": event.page, "bbox": None,
            "source_text": event.quote, "normalized_entity": event.label}).exec_()

    # ---------------------------------------------------------- workers
    def _worker_running(self) -> bool:
        return self._worker is not None and self._worker.isRunning()

    def _start_worker(self, worker) -> None:
        worker.progress.connect(self._on_progress)
        worker.result_ready.connect(self._on_finished)
        worker.cancelled.connect(lambda: self._on_stopped("Elaborazione interrotta."))
        worker.error.connect(lambda message: self._on_stopped(f"Errore: {message}", error=True))
        worker.finished.connect(self._release_worker)
        self._worker = worker
        for button in (self._run_btn, self._code_btn, self._export_btn):
            button.setEnabled(False)
        self._cancel_btn.setVisible(True)
        self._progress.setValue(0)
        self._progress.setVisible(True)
        worker.start()

    def _on_progress(self, percent: int, message: str) -> None:
        self._progress.setValue(int(percent))
        self._status.setText(message)

    def _ready_llm(self) -> bool:
        if not prepare_pipeline(self._services, "atomic", self):
            return False
        llm = self._services.get("atomic_evidence_llm_client")
        if llm is None or not getattr(llm, "is_available", False):
            QMessageBox.warning(self, "Modello non disponibile",
                                "Configura e avvia il modello di estrazione prima di procedere.")
            return False
        return True

    def _on_run(self) -> None:
        pipeline = self._pipeline()
        if not self._patient_id or pipeline is None or self._worker_running() or not self._ready_llm():
            return
        from .workers import PatientExtractionWorker
        llm = self._services.get("atomic_evidence_llm_client")
        self._start_worker(PatientExtractionWorker(
            pipeline, self._patient_id, num_workers=max(1, int(getattr(llm, "parallel_workers", 1) or 1))))

    def _on_code_missing(self) -> None:
        pipeline = self._pipeline()
        if not self._patient_id or pipeline is None or self._worker_running() or not self._ready_llm():
            return
        from .workers import ConceptCodingWorker
        self._start_worker(ConceptCodingWorker(pipeline, [self._patient_id]))

    def _on_finished(self, result: dict) -> None:
        if "documents_processed" in result:
            message = (f"Completato: {result.get('documents_processed', 0)} documenti elaborati, "
                       f"{result.get('documents_skipped', 0)} invariati, "
                       f"{result.get('documents_failed', 0)} non completati · "
                       f"{result.get('fhir_events', 0)} risorse FHIR, {result.get('fhir_uncoded', 0)} senza codice.")
        else:
            message = (f"Codifica: {result.get('coded', 0)} concetti codificati, "
                       f"{result.get('abstained', 0) + result.get('no_candidates', 0)} senza candidato adatto, "
                       f"{result.get('errors', 0)} errori.")
            self._export_timer.start(500)
        self._on_stopped(message)

    def _on_stopped(self, message: str, *, error: bool = False) -> None:
        for button in (self._run_btn, self._code_btn, self._export_btn):
            button.setEnabled(True)
        self._cancel_btn.setVisible(False)
        self._cancel_btn.setEnabled(True)
        self._progress.setVisible(False)
        self._refresh()
        self._status.setText(message + "  " + self._status.text())
        if error:
            QMessageBox.warning(self, "Operazione non completata", message)

    def _release_worker(self) -> None:
        worker, self._worker = self._worker, None
        if worker is not None:
            worker.deleteLater()

    def _on_cancel(self) -> None:
        if self._worker_running():
            self._worker.cancel()
            self._cancel_btn.setEnabled(False)

    def request_shutdown(self) -> None:
        self._export_timer.stop()
        if self._worker_running():
            self._worker.cancel()
            self._worker.requestInterruption()

    def shutdown(self, wait_ms: int = 1500) -> int:
        self.request_shutdown()
        if self._worker_running():
            self._worker.wait(max(0, int(wait_ms)))
        return int(self._worker_running())


def _valid_date(value: str) -> bool:
    import re
    from datetime import datetime
    formats = {4: "%Y", 7: "%Y-%m", 10: "%Y-%m-%d"}
    if not re.fullmatch(r"\d{4}(?:-\d{2}(?:-\d{2})?)?", value):
        return False
    try:
        datetime.strptime(value, formats[len(value)])
        return True
    except ValueError:
        return False

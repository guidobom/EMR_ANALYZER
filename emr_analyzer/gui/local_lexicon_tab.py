"""Select source text, assign a short label, and curate workspace-local examples."""
from dataclasses import asdict
import sqlite3

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QColor, QPalette, QTextCursor
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QSplitter, QLabel, QComboBox, QPushButton,
    QPlainTextEdit, QLineEdit, QListWidget, QListWidgetItem, QTabWidget,
    QTextEdit, QMessageBox, QInputDialog, QCompleter, QDialog, QDoubleSpinBox, QProgressBar, QTableWidget, QTableWidgetItem,
)

from .pipeline_llm import prepare_pipeline
from ..database.local_lexicon_repo import LocalLexiconRepository, CONTEXTS, source_hash
from .pdf_viewer import normalized_text_path
from .qt_utils import qt_offset, python_offset


class SourceText(QPlainTextEdit):
    def mouseReleaseEvent(self, event):
        super().mouseReleaseEvent(event)
        if not self.textCursor().hasSelection():
            self.parent_tab.activate_at(self.textCursor().position())


class LocalLexiconTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._services = {}
        self.repo = None
        self._documents = []
        self._doc = None
        self._source = ''
        self._editing = None
        self._rows = []
        self._source_highlights = []
        self._term_keys = set()
        self._search_generation = 0
        self.search_workers = []
        self._llm_drafts = {}
        self._llm_selected = None
        self._workspace_epoch = 0
        layout = QVBoxLayout(self)
        instruction = QLabel('Seleziona un frammento, scegli o crea un termine e premi Invio. Termini ed esempi sono disponibili nel registro condiviso fra progetti.')
        instruction.setWordWrap(True)
        layout.addWidget(instruction)
        bar = QHBoxLayout()
        self.documents = QComboBox()
        self.documents.currentIndexChanged.connect(self._load_document)
        bar.addWidget(self.documents, 1)
        reload_btn = QPushButton('Ricarica testo')
        reload_btn.clicked.connect(self._load_document)
        bar.addWidget(reload_btn)
        undo = QPushButton('Annulla ultima modifica')
        undo.clicked.connect(lambda: self._mutate(self.repo.undo) if self.repo else None)
        bar.addWidget(undo)
        layout.addLayout(bar)
        analysis_bar = QHBoxLayout()
        self.analyze_button = QPushButton('Analizza documento con LLM')
        self.analyze_button.clicked.connect(self.analyze_document)
        analysis_bar.addWidget(self.analyze_button)
        self.stop_analysis_button = QPushButton('Interrompi analisi')
        self.stop_analysis_button.clicked.connect(self.stop_analysis)
        self.stop_analysis_button.setEnabled(False)
        analysis_bar.addWidget(self.stop_analysis_button)
        self.analysis_progress = QProgressBar()
        self.analysis_progress.hide()
        analysis_bar.addWidget(self.analysis_progress)
        layout.addLayout(analysis_bar)
        self.analysis_status = QLabel('Analisi LLM non avviata. Le proposte non vengono salvate automaticamente.')
        self.analysis_status.setWordWrap(True)
        layout.addWidget(self.analysis_status)
        split = QSplitter()
        left = QWidget()
        left_layout = QVBoxLayout(left)
        self.text = SourceText()
        self.text.parent_tab = self
        self.text.setReadOnly(True)
        # Keep the chosen source yellow even while the annotation list or
        # term editor has focus; Qt otherwise paints the selection gray.
        palette = self.text.palette()
        for group in (QPalette.Active, QPalette.Inactive, QPalette.Disabled):
            palette.setColor(group, QPalette.Highlight, QColor('#fff2b3'))
            palette.setColor(group, QPalette.HighlightedText, QColor('#202020'))
        self.text.setPalette(palette)
        left_layout.addWidget(self.text, 1)
        self.selection = QLabel('Seleziona il passaggio da annotare.')
        self.selection.setWordWrap(True)
        left_layout.addWidget(self.selection)
        self.selection_text = QPlainTextEdit()
        self.selection_text.setReadOnly(True)
        self.selection_text.setPlaceholderText('Il testo origine completo della selezione comparirà qui.')
        self.selection_text.setMaximumHeight(120)
        self.selection_text.setMinimumHeight(70)
        self.selection_text.setAccessibleName('Testo origine integrale dell’annotazione')
        left_layout.addWidget(self.selection_text)
        self.term = QLineEdit()
        self.term.setPlaceholderText('Termine sintetico: scegli un termine o scrivine uno nuovo…')
        self.term.setMaxLength(200)
        self.term.returnPressed.connect(self.save)
        self.term.textChanged.connect(self._update_save_label)
        left_layout.addWidget(self.term)
        controls = QHBoxLayout()
        self.context = QComboBox()
        self.context.addItems(CONTEXTS)
        controls.addWidget(QLabel('Contesto (facoltativo)'))
        controls.addWidget(self.context)
        controls.addStretch()
        left_layout.addLayout(controls)
        controls = QHBoxLayout()
        self.save_button = QPushButton('Salva · Invio')
        self.save_button.clicked.connect(self.save)
        controls.addWidget(self.save_button)
        self.new_button = QPushButton('Nuova annotazione')
        self.new_button.clicked.connect(self._new)
        controls.addWidget(self.new_button)
        self.delete_button = QPushButton('Elimina annotazione')
        self.delete_button.clicked.connect(self._delete)
        controls.addWidget(self.delete_button)
        left_layout.addLayout(controls)
        split.addWidget(left)
        right = QWidget()
        right_layout = QVBoxLayout(right)
        self.views = QTabWidget()
        self.annotations = QListWidget()
        self.annotations.itemClicked.connect(self._choose_annotation)
        self.views.addTab(self.annotations, 'Nel documento')
        proposals = QWidget()
        proposals_layout = QVBoxLayout(proposals)
        legend = QLabel('Blu: proposte LLM temporanee, da confermare prima di chiudere. Giallo: annotazioni confermate. Seleziona una proposta, correggi testo, termine e contesto, poi premi Salva.')
        legend.setWordWrap(True)
        proposals_layout.addWidget(legend)
        self.llm_proposals = QListWidget()
        self.llm_proposals.itemClicked.connect(self._choose_llm_proposal)
        proposals_layout.addWidget(self.llm_proposals)
        discard = QPushButton('Scarta proposta selezionata')
        discard.clicked.connect(self._discard_llm_proposal)
        proposals_layout.addWidget(discard)
        self._llm_tab_index = self.views.addTab(proposals, 'Proposte LLM')
        lexicon = QWidget()
        lex_layout = QVBoxLayout(lexicon)
        self.filter = QLineEdit()
        self.filter.setPlaceholderText('Cerca nel lessico…')
        self.filter.textChanged.connect(self._filter_terms)
        lex_layout.addWidget(self.filter)
        self.terms = QListWidget()
        self.terms.itemClicked.connect(self._choose_term)
        lex_layout.addWidget(self.terms)
        term_buttons = QHBoxLayout()
        for title, action in [('Rinomina', self.rename), ('Unisci termini', self.merge), ('Scheda evento', self.edit_event_card), ('Catalogo SNOMED CT…', self.open_snomed_catalog), ('Catalogo LOINC…', self.open_loinc_catalog)]:
            button = QPushButton(title)
            button.clicked.connect(action)
            term_buttons.addWidget(button)
        lex_layout.addLayout(term_buttons)
        self.examples = QListWidget()
        self.examples.setWordWrap(True)
        self.examples.setTextElideMode(Qt.ElideNone)
        self.examples.itemClicked.connect(self._show_example_text)
        lex_layout.addWidget(QLabel('Frasi di esempio del termine: clicca per leggere il testo completo'))
        lex_layout.addWidget(self.examples)
        role_button = QPushButton('Imposta esempio / controesempio')
        role_button.clicked.connect(self.edit_example_role)
        lex_layout.addWidget(role_button)
        self.add_example_button = QPushButton('Aggiungi esempio scritto…')
        self.add_example_button.clicked.connect(lambda: self.edit_written_example())
        lex_layout.addWidget(self.add_example_button)
        self.delete_example_button = QPushButton('Elimina esempio selezionato')
        self.delete_example_button.clicked.connect(self.delete_example)
        lex_layout.addWidget(self.delete_example_button)
        self.views.addTab(lexicon, 'Lessico condiviso')
        matches = QWidget()
        matches_layout = QVBoxLayout(matches)
        find = QPushButton('Trova altre occorrenze identiche')
        find.clicked.connect(self.find_occurrences)
        matches_layout.addWidget(find)
        similar = QPushButton('Trova espressioni simili')
        similar.clicked.connect(self.find_similar)
        matches_layout.addWidget(similar)
        options = QHBoxLayout()
        options.addWidget(QLabel('Soglia di somiglianza:'))
        self.similarity_threshold = QDoubleSpinBox()
        self.similarity_threshold.setRange(0.1, 1.0)
        self.similarity_threshold.setSingleStep(0.05)
        self.similarity_threshold.setValue(0.55)
        self.similarity_threshold.setToolTip('Un valore più basso propone più risultati. Il punteggio non è una probabilità clinica.')
        options.addWidget(self.similarity_threshold)
        stop = QPushButton('Interrompi ricerca')
        stop.clicked.connect(self.cancel_search)
        options.addWidget(stop)
        matches_layout.addLayout(options)
        notice = QLabel('Proposte non annotate: apri una fonte e premi Salva per confermare.')
        notice.setWordWrap(True)
        matches_layout.addWidget(notice)
        self.matches = QListWidget()
        self.matches.itemClicked.connect(self._open_match)
        matches_layout.addWidget(self.matches)
        ignore = QPushButton('Ignora proposta selezionata')
        ignore.clicked.connect(lambda: self.matches.takeItem(self.matches.currentRow()) if self.matches.currentRow() >= 0 else None)
        matches_layout.addWidget(ignore)
        self.views.addTab(matches, 'Altre occorrenze')
        right_layout.addWidget(self.views)
        split.addWidget(right)
        split.setSizes([850,450])
        layout.addWidget(split, 1)
        self.status = QLabel()
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self.text.selectionChanged.connect(self._selection_changed)
        self._selection_changed()

    def set_services(self, services):
        self.stop_analysis()
        self._workspace_epoch += 1
        self._llm_drafts.clear()
        self.analyze_button.setEnabled(True)
        self.stop_analysis_button.setEnabled(False)
        self.analysis_progress.hide()
        self.cancel_search()
        self._services = services
        db = services.get('db')
        if db is None and services.get('document_repo') is not None:
            db = services['document_repo'].db
        self.repo = services.get("shared_lexicon_repo")
        if self.repo is None:
            self.repo = LocalLexiconRepository(db) if db is not None else None
        self.term.clear()
        self.load_patient(None)

    def showEvent(self, event):
        super().showEvent(event)
        self._refresh()

    def load_patient(self, patient_id):
        self.cancel_search()
        self.matches.clear()
        repo = self._services.get('document_repo')
        self._documents = repo.list_by_patient(patient_id) if repo and patient_id else []
        self.documents.blockSignals(True)
        self.documents.clear()
        for doc in self._documents:
            self.documents.addItem(f'{doc.document_date or "Data n.d."} — {doc.filename}', doc.id)
        self.documents.blockSignals(False)
        self._load_document()

    def _read(self, doc):
        path = normalized_text_path(asdict(doc))
        if path is None:
            return ''
        value = path.read_text(encoding='utf-8')
        overlay = self._services.get('overlay_repo')
        if overlay:
            value = overlay.effective_text(doc.id, value)
        # QTextDocument normalizes paragraph separators; hash exactly the displayed text.
        return value.replace('\r\n','\n').replace('\r','\n').replace('\u2029','\n').replace('\u2028','\n')

    def _load_document(self, *_):
        self._doc = next((d for d in self._documents if d.id == self.documents.currentData()), None)
        self._source = ''
        self.status.clear()
        if self._doc:
            try:
                self._source = self._read(self._doc)
            except (OSError, UnicodeError) as exc:
                self.status.setText(f'Impossibile leggere il testo: {exc}')
        self.text.setPlainText(self._source)
        self._new()
        self._refresh()
        if not self._source:
            self.status.setText('Testo non disponibile: seleziona un documento con testo clinico estratto.')

    def _selection_changed(self):
        cursor = self.text.textCursor()
        # A chosen passage is the only highlighted event. Retain the cursor
        # selection itself because it supplies the exact range for editing.
        self.text.setExtraSelections(
            [] if cursor.hasSelection() else self._source_highlights
        )
        self.save_button.setEnabled(bool(self.repo and self._doc and cursor.hasSelection()))
        start = python_offset(self._source, cursor.selectionStart())
        end = python_offset(self._source, cursor.selectionEnd())
        # Preview the same exact source slice that save() persists. Plain text
        # preserves whitespace and cannot interpret clinical text as HTML.
        value = self._source[start:end]
        self.selection.setText(
            f'Testo origine completo · {len(value)} caratteri'
            + (' · Modifica annotazione' if self._editing else '')
            if value else 'Seleziona il passaggio da annotare.'
        )
        self.selection_text.setPlainText(value)
        self.delete_button.setEnabled(bool(self._editing))

    def _update_save_label(self):
        labels = self._term_keys
        from ..database.local_lexicon_repo import label_key
        new = self.term.text().strip() and label_key(self.term.text()) not in labels
        self.save_button.setText('Crea termine e salva · Invio' if new else 'Salva · Invio')

    def _new(self):
        self._llm_selected = None
        self._editing = None
        self.context.setCurrentIndex(0)
        self._selection_changed()

    def _mutate(self, action):
        try:
            action()
        except (ValueError, sqlite3.Error) as exc:
            QMessageBox.warning(self, 'Lessico condiviso', str(exc))
            return False
        self._new()
        self._refresh()
        return True

    def save(self):
        if not self.repo or not self._doc:
            return
        try:
            if self._read(self._doc) != self._source:
                self.status.setText('Il testo è cambiato: premi Ricarica testo prima di annotare.')
                return
        except (OSError, UnicodeError) as exc:
            self.status.setText(str(exc)); return
        cursor = self.text.textCursor()
        start = python_offset(self._source, cursor.selectionStart())
        end = python_offset(self._source, cursor.selectionEnd())
        proposal = self._llm_selected
        saved_ids = []
        if self._mutate(lambda: saved_ids.append(self.repo.save(self._doc.id,self._source,start,end,self.term.text(),self.context.currentText(),self._editing))):
            if proposal is not None:
                proposal['review'] = 'confirmed'
                proposal['annotation_id'] = saved_ids[0]
                self._refresh()
            self.status.setText('Annotazione salvata nel lessico condiviso.')

    def _delete(self):
        if self._editing:
            self._mutate(lambda: self.repo.delete(self._editing))

    def _refresh(self):
        self.annotations.clear()
        self._rows = self.repo.annotations(document_id=self._doc.id) if self.repo and self._doc else []
        highlights = []
        stale = 0
        for row in self._rows:
            valid = self._valid(row)
            stale += int(not valid)
            item = QListWidgetItem(f'{row["label"]} · {row["assertion"]}' + (' · TESTO CAMBIATO' if not valid else '') + '\n' + row['quote'][:120])
            item.setData(Qt.UserRole, row)
            item.setToolTip(row['context'])
            self.annotations.addItem(item)
            if valid:
                highlight = QTextEdit.ExtraSelection()
                highlight.cursor = self._cursor(row['start'],row['end'])
                highlight.format.setBackground(QColor('#fff2b3'))
                highlights.append(highlight)
        self._render_llm_proposals(highlights)
        self._source_highlights = highlights
        current_term = self.terms.currentItem().data(Qt.UserRole)['id'] if self.terms.currentItem() else None
        self.terms.clear()
        all_terms = self.repo.terms() if self.repo else []
        self._term_keys = {t['label_key'] for t in all_terms}
        for term in all_terms:
            category = self.repo.event_definition(term['id']).get('category', '')
            prefix = category + ' · ' if category else ''
            item = QListWidgetItem(f'{prefix}{term["label"]} ({term["examples"]})')
            item.setData(Qt.UserRole, term)
            self.terms.addItem(item)
            if term['id'] == current_term:
                self.terms.setCurrentItem(item)
        completer = QCompleter([t['label'] for t in all_terms], self.term)
        completer.setCaseSensitivity(Qt.CaseInsensitive)
        completer.setFilterMode(Qt.MatchContains)
        self.term.setCompleter(completer)
        self._filter_terms()
        self._show_examples()
        self._selection_changed()
        self._update_save_label()
        if stale:
            self.status.setText(f'{stale} annotazioni riferite a una versione precedente. Seleziona nuovamente il passaggio per riallinearle.')

    def _valid(self, row):
        return row['source_hash'] == source_hash(self._source) and self._source[row['start']:row['end']] == row['quote']

    def _cursor(self, start, end):
        cursor = self.text.textCursor()
        cursor.setPosition(qt_offset(self._source,start))
        cursor.setPosition(qt_offset(self._source,end), QTextCursor.KeepAnchor)
        return cursor

    def _choose_annotation(self, item):
        row = item.data(Qt.UserRole)
        self._llm_selected = None
        self._editing = row['id']
        self.term.setText(row['label'])
        self.context.setCurrentText(row['assertion'])
        if self._valid(row):
            self.text.setTextCursor(self._cursor(row['start'],row['end']))
            self.text.ensureCursorVisible()
        else:
            cursor = self.text.textCursor(); cursor.clearSelection(); self.text.setTextCursor(cursor)
            self.status.setText('Fonte modificata. Seleziona il nuovo passaggio e salva per riallineare questa annotazione.')
        self._selection_changed()

    def activate_at(self, position):
        offset = python_offset(self._source, position)
        matches = [self.annotations.item(i) for i,r in enumerate(self._rows) if self._valid(r) and r['start'] <= offset < r['end']]
        drafts = [self.llm_proposals.item(i) for i in range(self.llm_proposals.count())
                  if self.llm_proposals.item(i).data(Qt.UserRole)['start'] <= offset < self.llm_proposals.item(i).data(Qt.UserRole)['end']]
        if drafts and not matches:
            self.views.setCurrentIndex(self._llm_tab_index)
            if len(drafts)==1:
                self._choose_llm_proposal(drafts[0])
            else:
                self.status.setText('Più eventi sullo stesso passaggio: seleziona una voce in Proposte LLM.')
            return
        if len(matches)==1:
            self._choose_annotation(matches[0])
        elif matches:
            self.views.setCurrentIndex(0)
            self.status.setText('Più annotazioni sovrapposte: scegli quella da modificare nell’elenco.')

    def _filter_terms(self):
        for i in range(self.terms.count()):
            item = self.terms.item(i)
            item.setHidden(self.filter.text().casefold() not in item.text().casefold())

    def _choose_term(self, item):
        self.term.setText(item.data(Qt.UserRole)['label'])
        self._show_examples()

    def _show_examples(self):
        self.examples.clear()
        selected = self.terms.currentItem()
        if not selected or not self.repo:
            return
        for row in self.repo.examples(term_id=selected.data(Qt.UserRole)['id']):
            item = QListWidgetItem(f"{row['assertion']}\n{row['quote']}")
            item.setData(Qt.UserRole,row)
            role = self.repo.example_role(row['id'])
            item.setText(('Controesempio · ' if role == 'counterexample' else 'Esempio · ') + item.text())
            item.setToolTip(row['quote'])
            self.examples.addItem(item)

    def edit_written_example(self, row=None):
        selected = self.terms.currentItem()
        if not self.repo or (row is None and selected is None):
            self.status.setText('Seleziona prima una voce nel Lessico condiviso.')
            return
        term = row if row is not None else selected.data(Qt.UserRole)
        term_id = row['term_id'] if row is not None else term['id']
        dialog = QDialog(self)
        dialog.setWindowTitle(('Modifica esempio — ' if row else 'Nuovo esempio — ') + term['label'])
        dialog.resize(650, 360)
        layout = QVBoxLayout(dialog)
        instruction = QLabel('Scrivi una frase che aiuti il modello a riconoscere questo evento. Non occorre un documento.')
        instruction.setWordWrap(True)
        layout.addWidget(instruction)
        text = QPlainTextEdit(row['quote'] if row else '')
        text.setObjectName('written_example_text')
        layout.addWidget(text)
        assertion = QComboBox()
        assertion.addItems(CONTEXTS)
        assertion.setCurrentText(row['assertion'] if row else 'Presente')
        layout.addWidget(QLabel('Stato dell’evento nella frase:'))
        layout.addWidget(assertion)
        role = QComboBox()
        role.addItem('Esempio', 'example')
        role.addItem('Controesempio', 'counterexample')
        if row:
            role.setCurrentIndex(1 if self.repo.example_role(row['id']) == 'counterexample' else 0)
        layout.addWidget(role)
        buttons = QHBoxLayout()
        save = QPushButton('Salva esempio')
        save.setObjectName('save_written_example')
        def commit():
            if self._mutate(lambda: self.repo.save_example(term_id, text.toPlainText(),
                assertion.currentText(), role.currentData(), row['id'] if row else None)):
                self.status.setText('Esempio salvato nel Lessico condiviso, disponibile per tutte le workspace.')
                dialog.accept()
        save.clicked.connect(commit)
        cancel = QPushButton('Annulla')
        cancel.clicked.connect(dialog.reject)
        buttons.addWidget(save)
        buttons.addWidget(cancel)
        layout.addLayout(buttons)
        dialog.exec_()

    def _show_example_text(self, item):
        row = item.data(Qt.UserRole)
        if row.get('manual'):
            self.edit_written_example(row)
            return
        dialog = QDialog(self)
        dialog.setWindowTitle('Esempio — ' + row['label'])
        dialog.resize(650, 360)
        layout = QVBoxLayout(dialog)
        layout.addWidget(QLabel(row['assertion']))
        text = QPlainTextEdit(row['quote'])
        text.setReadOnly(True)
        layout.addWidget(text)
        close = QPushButton('Chiudi')
        close.clicked.connect(dialog.accept)
        layout.addWidget(close)
        dialog.exec_()

    def delete_example(self):
        item = self.examples.currentItem()
        if item is not None and self.repo is not None:
            ident = item.data(Qt.UserRole)['id']
            if self._mutate(lambda: self.repo.delete(ident)):
                self.status.setText('Esempio eliminato. Puoi usare Annulla ultima modifica per ripristinarlo.')

    def select_document(self, document_id):
        """Follow the Documents tab without resetting an already open annotation."""
        if self._doc is not None and self._doc.id == document_id:
            return
        self._open_document(document_id)

    def _open_document(self, document_id):
        repo = self._services.get('document_repo')
        doc = repo.get_by_id(document_id) if repo else None
        if doc is None:
            self.status.setText('Documento non più disponibile.'); return False
        if self.documents.findData(doc.id)<0:
            self._documents.append(doc)
            self.documents.addItem(f'{doc.patient_id} · {doc.document_date or "Data n.d."} — {doc.filename}',doc.id)
        self.documents.setCurrentIndex(self.documents.findData(doc.id))
        self._load_document()
        return True

    def _open_example(self, item):
        row = item.data(Qt.UserRole)
        if not row.get('current_workspace', True):
            self._preview_external_example(row)
            return
        if self._open_document(row['document_id']):
            self.views.setCurrentIndex(0)
            for i in range(self.annotations.count()):
                if self.annotations.item(i).data(Qt.UserRole)['id']==row['id']:
                    self._choose_annotation(self.annotations.item(i)); break
        elif 'workspace_name' in row:
            self._preview_external_example(row)

    def _preview_external_example(self, row):
        dialog = QDialog(self)
        dialog.setWindowTitle('Esempio del registro condiviso')
        dialog.resize(700, 400)
        layout = QVBoxLayout(dialog)
        origin = QLabel(f"{row.get('workspace_name', 'Progetto corrente')} · {row['patient_id']} · {row['filename']}\n"
                        f"{row['label']} — {row['assertion']}\nFonte: {row.get('workspace_path', '')}")
        origin.setWordWrap(True)
        layout.addWidget(origin)
        passage = QPlainTextEdit()
        passage.setReadOnly(True)
        full_text = self.repo.annotated_text(row['id']) if hasattr(self.repo, 'annotated_text') else None
        if full_text is not None:
            passage.setPlainText(full_text)
            cursor = passage.textCursor()
            cursor.setPosition(qt_offset(full_text,row['start']))
            cursor.setPosition(qt_offset(full_text,row['end']),QTextCursor.KeepAnchor)
            passage.setTextCursor(cursor)
            passage.ensureCursorVisible()
        else:
            passage.setPlainText(f"Frammento annotato:\n{row['quote']}\n\nContesto salvato:\n{row['context']}\n\nVersione completa non disponibile per questo esempio precedente.")
        layout.addWidget(passage)
        notice = QLabel('Questo è l’esempio salvato nel registro. Per modificare la selezione originale, apri il progetto di origine.')
        notice.setWordWrap(True)
        layout.addWidget(notice)
        use = QPushButton('Usa questo termine nel documento corrente')
        use.clicked.connect(lambda: (self.term.setText(row['label']), dialog.accept()))
        layout.addWidget(use)
        close = QPushButton('Chiudi')
        close.clicked.connect(dialog.reject)
        layout.addWidget(close)
        dialog.exec_()

    def open_loinc_catalog(self):
        catalog = getattr(self.repo, 'loinc_catalog', None)
        if catalog is None:
            self.status.setText('Serve il registro condiviso per importare LOINC.')
            return
        from .loinc_catalog_dialog import LoincCatalogDialog
        LoincCatalogDialog(catalog,self).exec_()

    def open_snomed_catalog(self):
        catalog = getattr(self.repo, 'snomed_catalog', None)
        if catalog is None:
            self.status.setText('Il catalogo SNOMED richiede il registro condiviso.')
            return
        from .snomed_catalog_dialog import SnomedCatalogDialog
        SnomedCatalogDialog(catalog, self).exec_()
        pipeline = self._services.get('extraction_pipeline')
        if pipeline:
            pipeline.reload_policy()

    def edit_event_card(self):
        item = self.terms.currentItem()
        if not item:
            self.status.setText('Seleziona un termine nel lessico per definirne la scheda evento.')
            return
        term = item.data(Qt.UserRole)
        card = self.repo.event_definition(term['id'])
        dialog = QDialog(self)
        dialog.setWindowTitle('Scheda evento — ' + term['label'])
        from ..clinical.lexicon_structure import CATEGORIES, VALUE_TYPES, FIELD_TYPES, DEDUP_RULES
        dialog.resize(820,680)
        layout = QVBoxLayout(dialog)
        def combo(label, options, current, name, editable=False):
            layout.addWidget(QLabel(label))
            box = QComboBox()
            box.setObjectName(name)
            box.setEditable(editable)
            for key, title in options.items():
                box.addItem(title, key)
            if editable:
                box.setCurrentText(current)
            else:
                box.setCurrentIndex(max(0, box.findData(current)))
            layout.addWidget(box)
            return box
        category = combo('Categoria', {c:c for c in ['']+CATEGORIES}, card.get('category',''), 'event_category', True)
        layout.addWidget(QLabel('Cosa riconoscere (definizione ed eventuali esclusioni):'))
        definition = QPlainTextEdit(card['definition'])
        definition.setMaximumHeight(100)
        layout.addWidget(definition)
        value_type = combo('Tipo di valore (il primo campo è il valore principale)', VALUE_TYPES, card.get('value_type','none'), 'event_value_type')
        dedup = combo('Deduplicazione', DEDUP_RULES, card.get('dedup_rule','auto'), 'event_dedup_rule')
        layout.addWidget(QLabel('Campi da estrarre — per valori composti, inserisci un campo per componente.'))
        fields = QTableWidget(0,4)
        fields.setObjectName('event_fields')
        fields.setHorizontalHeaderLabels(['Nome', 'Tipo', 'Unità', 'Opzioni (separate da ;)'])
        fields.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(fields)
        def add_field(field=None):
            field = field or {}
            row = fields.rowCount()
            fields.insertRow(row)
            for column, text in ((0,field.get('name','')), (2,field.get('unit','')), (3,'; '.join(field.get('choices',[])))):
                fields.setItem(row,column,QTableWidgetItem(text))
            kind = QComboBox()
            for key,title in FIELD_TYPES.items():
                kind.addItem(title,key)
            kind.setCurrentIndex(max(0,kind.findData(field.get('type','text'))))
            fields.setCellWidget(row,1,kind)
        for field in card.get('field_definitions', [dict(name=f) for f in card['fields']]):
            add_field(field)
        buttons = QHBoxLayout()
        add = QPushButton('Aggiungi campo')
        add.clicked.connect(lambda: add_field())
        remove = QPushButton('Rimuovi campo selezionato')
        remove.clicked.connect(lambda: fields.removeRow(fields.currentRow()) if fields.currentRow()>=0 else None)
        buttons.addWidget(add)
        buttons.addWidget(remove)
        layout.addLayout(buttons)
        notice = QLabel('Esempio: Pressione arteriosa → valore composto; sistolica e diastolica numeriche, mmHg. '
                        'Per le terapie aggiungi farmaco, dose, ambito e stato. I dati assenti restano non determinati.')
        notice.setWordWrap(True)
        layout.addWidget(notice)
        save = QPushButton('Salva scheda')
        save.setObjectName('save_event_card')
        def commit():
            descriptors = [dict(name=fields.item(r,0).text(), type=fields.cellWidget(r,1).currentData(),
                                unit=fields.item(r,2).text(), choices=fields.item(r,3).text().split(';'))
                           for r in range(fields.rowCount())]
            structure = dict(category=category.currentText(), value_type=value_type.currentData(),
                             dedup_rule=dedup.currentData(), field_definitions=descriptors)
            if self._mutate(lambda: self.repo.set_event_definition(term['id'], definition.toPlainText(),
                    [f['name'] for f in descriptors], structure=structure)):
                dialog.accept()
        save.clicked.connect(commit)
        layout.addWidget(save)
        cancel = QPushButton('Annulla')
        cancel.clicked.connect(dialog.reject)
        layout.addWidget(cancel)
        dialog.exec_()

    def edit_example_role(self):
        item = self.examples.currentItem()
        if not item:
            self.status.setText('Seleziona un esempio del termine.')
            return
        row = item.data(Qt.UserRole)
        options = ['Esempio: illustra il concetto nel contesto indicato',
                   'Controesempio: non attesta questo tipo di evento']
        value, ok = QInputDialog.getItem(self,'Tipo di esempio',row['quote'][:200],options,
                                       int(self.repo.example_role(row['id']) == 'counterexample'),False)
        if ok:
            self._mutate(lambda: self.repo.set_example_role(row['id'],'counterexample' if value==options[1] else 'example'))

    def rename(self):
        item = self.terms.currentItem()
        if item:
            term = item.data(Qt.UserRole)
            value, ok = QInputDialog.getText(self,'Rinomina termine','Nome sintetico:',text=term['label'])
            if ok:
                self._mutate(lambda: self.repo.rename(term['id'],value))

    def merge(self):
        item = self.terms.currentItem()
        if not item:
            return
        source = item.data(Qt.UserRole)
        choices = [t for t in self.repo.terms() if t['id']!=source['id']]
        if not choices:
            self.status.setText('Crea almeno due termini per poterli unire.'); return
        value, ok = QInputDialog.getItem(self,'Unisci termini',f'Trasferisci gli esempi di «{source["label"]}» a:',[t['label'] for t in choices],0,False)
        if ok:
            target = next(t for t in choices if t['label']==value)
            self._mutate(lambda: self.repo.merge(source['id'],target['id']))

    def _draft_key(self):
        return (self._doc.id,source_hash(self._source)) if self._doc else None

    def _render_llm_proposals(self, highlights):
        self.llm_proposals.clear()
        stored_ids = {r['id'] for r in self._rows}
        term_keys = {r['label_key'] for r in self.repo.terms()} if self.repo else set()
        for row in self._llm_drafts.get(self._draft_key(),[]):
            if row['review'] == 'confirmed' and row.get('annotation_id') not in stored_ids:
                row['review'] = 'pending'
            if row['review'] != 'pending':
                continue
            is_new = row['label'].casefold() not in term_keys
            item=QListWidgetItem(f"{row['label']} · {row['state']}" + (' · Nuovo termine proposto' if is_new else '') + '\n' + row['quote'][:160])
            item.setData(Qt.UserRole,row)
            details = row['details']
            subject = {'patient':'paziente','family':'familiare','other':'altro','unknown':'non determinato'}.get(details.get('experiencer'),'non determinato')
            certainty = {'confirmed':'affermato nel testo','suspected':'sospetto','possible':'possibile','unknown':'non determinata'}.get(details.get('certainty'),'non determinata')
            item.setToolTip(f"Soggetto: {subject}\nCertezza: {certainty}\nData dell’evento: {details.get('observation_date') or 'non determinata'}\n\n{row['quote']}")
            self.llm_proposals.addItem(item)
            highlight=QTextEdit.ExtraSelection()
            highlight.cursor=self._cursor(row['start'],row['end'])
            highlight.format.setBackground(QColor('#cde7ff'))
            highlights.append(highlight)

    def _choose_llm_proposal(self, item):
        ident=item.data(Qt.UserRole)['id']
        row=next(r for r in self._llm_drafts.get(self._draft_key(),[]) if r['id']==ident)
        self._new()
        self._llm_selected=row
        self.term.setText(row['label'])
        self.context.setCurrentText(row['state'])
        self.text.setTextCursor(self._cursor(row['start'],row['end']))
        self.text.ensureCursorVisible()
        self.status.setText('Proposta LLM: puoi correggere selezione, termine e contesto. Salva conferma soltanto questa annotazione.')

    def _discard_llm_proposal(self):
        item=self.llm_proposals.currentItem()
        if item is None:
            return
        ident=item.data(Qt.UserRole)['id']
        for row in self._llm_drafts.get(self._draft_key(),[]):
            if row['id']==ident:
                row['review']='discarded'
        self._new()
        self._refresh()

    def stop_analysis(self):
        for worker in self.search_workers:
            if getattr(worker,'is_annotation_analysis',False):
                worker.requestInterruption()
                self.analysis_status.setText('Interruzione richiesta: attendo la chiamata LLM in corso.')

    def analyze_document(self):
        if self._doc is None or not self._source.strip():
            self.analysis_status.setText('Seleziona un documento con testo estratto.'); return
        parent=self.parentWidget()
        while parent is not None:
            busy=getattr(parent,'llm_operation_running',None)
            if callable(busy) and busy():
                self.analysis_status.setText('È già in corso un’elaborazione. Attendi il termine prima di avviare l’analisi.'); return
            parent=parent.parentWidget()
        if any(w.isRunning() for w in self.search_workers):
            self.analysis_status.setText('Attendi il termine dell’elaborazione in corso.'); return
        try:
            if self._read(self._doc)!=self._source:
                self.analysis_status.setText('Il testo è cambiato: premi Ricarica testo prima di analizzare.'); return
            examples=self.repo.extraction_examples() if self.repo else []
        except (OSError,UnicodeError,sqlite3.Error) as exc:
            self.analysis_status.setText(str(exc)); return
        if not prepare_pipeline(self._services, 'lexicon', self):
            return
        llm=self._services.get('atomic_evidence_llm_client')
        if llm is None or not getattr(llm,'is_available',False):
            self.analysis_status.setText('Modello non disponibile: seleziona un modello installato per avviare l’analisi.'); return
        from .lexicon_analysis_worker import LexiconAnalysisWorker
        from ..settings import load_pipeline_policy
        epoch=self._workspace_epoch
        key=self._draft_key()
        worker=LexiconAnalysisWorker(llm,self._doc,self._source,examples,load_pipeline_policy(),self)
        self.search_workers.append(worker)
        self.analyze_button.setEnabled(False)
        self.stop_analysis_button.setEnabled(True)
        self.analysis_progress.setRange(0,0)
        self.analysis_progress.show()
        self.analysis_status.setText(f"Analisi LLM in corso — {getattr(llm,'model','modello locale')} — {self._doc.filename}. Ricerca anche nuovi eventi fuori lessico.")
        self.views.setCurrentIndex(self._llm_tab_index)
        def progress(n,total):
            if epoch==self._workspace_epoch:
                self.analysis_progress.setRange(0,total)
                self.analysis_progress.setValue(n)
        def completed(rows,warning):
            if epoch!=self._workspace_epoch:
                return
            # Preserve already-reviewed and pending drafts when repeating analysis.
            existing={r['id']:r for r in self._llm_drafts.get(key,[])}
            for row in rows:
                existing.setdefault(row['id'],row)
            self._llm_drafts[key]=list(existing.values())
            self._refresh()
            self.analysis_status.setText(warning or f'Analisi LLM completata per {worker.document.filename}: {len(rows)} proposte. Nessuna annotazione salvata automaticamente.')
            if key!=self._draft_key():
                self.analysis_status.setText(self.analysis_status.text()+' Le proposte appartengono al documento analizzato: riaprilo per rivederle.')
        worker.progress.connect(progress)
        worker.completed.connect(completed)
        worker.failed.connect(lambda message:self.analysis_status.setText(message) if epoch==self._workspace_epoch else None)
        def finished():
            if epoch==self._workspace_epoch:
                self.analyze_button.setEnabled(True)
                self.stop_analysis_button.setEnabled(False)
                self.analysis_progress.hide()
                if worker.isInterruptionRequested():
                    self.analysis_status.setText('Analisi LLM interrotta. Le annotazioni confermate sono conservate.')
            self.search_workers.remove(worker)
            worker.deleteLater()
        worker.finished.connect(finished)
        worker.start()

    def cancel_search(self):
        self._search_generation += 1
        for worker in self.search_workers:
            if not getattr(worker, 'is_annotation_analysis', False):
                worker.requestInterruption()
        if hasattr(self, 'status'):
            self.status.setText('Ricerca interrotta.')

    def find_similar(self):
        quote = self.text.textCursor().selectedText().replace('\u2029','\n')
        repo = self._services.get('document_repo')
        if not quote.strip() or repo is None or self.repo is None:
            self.status.setText('Seleziona il frammento da cercare.'); return
        if any(w.isRunning() for w in self.search_workers):
            self.cancel_search()
            self.status.setText('Interruzione in corso; ripremi Trova espressioni simili al termine.'); return
        from .similar_passages_worker import SimilarPassagesWorker
        self.cancel_search()
        generation = self._search_generation
        self.matches.clear()
        self.views.setCurrentIndex(3)
        known = {(r['document_id'],r['source_hash'],r['start'],r['end']) for r in self.repo.annotations()
                 if r.get('current_workspace',True) and r['label'].casefold()==self.term.text().strip().casefold()}
        if self._doc:
            c=self.text.textCursor()
            known.add((self._doc.id,source_hash(self._source),python_offset(self._source,c.selectionStart()),python_offset(self._source,c.selectionEnd())))
        documents = [(doc,normalized_text_path(asdict(doc))) for doc in repo.list_all()]
        worker = SimilarPassagesWorker(documents,quote,self.term.text(),known,self.similarity_threshold.value(),self._services.get('overlay_repo'),self)
        self.search_workers.append(worker)
        worker.completed.connect(lambda rows,mode,missing: self._similar_results(generation,rows,mode,missing))
        worker.failed.connect(lambda message: self.status.setText(message) if generation==self._search_generation else None)
        worker.progress.connect(lambda current,total: self.status.setText(f'Ricerca di espressioni simili: {current}/{total} documenti…') if generation==self._search_generation else None)
        def finished():
            self.search_workers.remove(worker)
            worker.deleteLater()
        worker.finished.connect(finished)
        self.status.setText('Caricamento del modello locale e ricerca di espressioni simili…')
        worker.start()

    def _similar_results(self, generation, rows, mode, missing):
        if generation!=self._search_generation:
            return
        self.matches.clear()
        for row in rows:
            item=QListWidgetItem(f"Somiglianza {row['score']:.2f} · {row['patient_id']} · {row['filename']}\n{row['quote']}")
            item.setForeground(QColor('#666666'))
            item.setToolTip(row['context'])
            item.setData(Qt.UserRole,row)
            self.matches.addItem(item)
        self.status.setText(f'{len(rows)} proposte (massimo 100), ricerca {mode}. {missing} testi non disponibili. Verifica negazioni e contesto prima di salvare.')

    def find_occurrences(self):
        quote = self.text.textCursor().selectedText().replace('\u2029','\n')
        repo = self._services.get('document_repo')
        if not quote.strip() or not repo:
            self.status.setText('Seleziona il frammento da cercare.'); return
        self.cancel_search()
        generation = self._search_generation
        docs = iter(repo.list_all())
        self.matches.clear()
        self.views.setCurrentIndex(3)
        label = self.term.text()
        known = {(r['document_id'],r['source_hash'],r['start'],r['end']) for r in self.repo.annotations() if r.get('current_workspace', True)}
        failures = []
        def step():
            if generation != self._search_generation:
                return
            doc = next(docs,None)
            if doc is None:
                self.status.setText(f'{self.matches.count()} occorrenze non annotate. {len(failures)} testi non leggibili.'); return
            try:
                text = self._read(doc)
                digest = source_hash(text)
                start = text.find(quote)
                while start>=0:
                    end = start+len(quote)
                    if (doc.id,digest,start,end) not in known:
                        item = QListWidgetItem(f'{doc.patient_id} · {doc.filename}\n{text[max(0,start-65):end+65]}')
                        item.setForeground(QColor('#666666'))
                        item.setData(Qt.UserRole,dict(document_id=doc.id,source_hash=digest,start=start,end=end,quote=quote,label=label))
                        self.matches.addItem(item)
                    start = text.find(quote,start+max(1,len(quote)))
            except (OSError,UnicodeError):
                failures.append(doc.id)
            self.status.setText(f'Ricerca nel workspace… {self.matches.count()} proposte. Una nuova ricerca sostituisce questa.')
            QTimer.singleShot(0, step)
        QTimer.singleShot(0, step)

    def _open_match(self, item):
        row = item.data(Qt.UserRole)
        if self._open_document(row['document_id']):
            self.term.setText(row['label'])
            if self._valid(row):
                self.text.setTextCursor(self._cursor(row['start'],row['end']))
                self.text.ensureCursorVisible()
                self.status.setText('Proposta non salvata: verifica il contesto, scegli il termine e premi Salva.')
            else:
                self.status.setText('Il testo è cambiato dopo la ricerca: ripeti la ricerca.')

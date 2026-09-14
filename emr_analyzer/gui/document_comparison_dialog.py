"""Read-only, side-by-side inspection of the original PDF and active Markdown."""
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFontDatabase, QTextCursor, QTextDocument
from PyQt5.QtWidgets import (
    QDialog, QHBoxLayout, QLabel, QLineEdit, QPlainTextEdit, QPushButton,
    QSplitter, QVBoxLayout, QWidget,
)

from .pdf_viewer import DocumentEvidencePreview, normalized_text_path


class DocumentComparisonDialog(QDialog):
    def __init__(self, doc_data, services, parent=None):
        super().__init__(parent)
        self._doc_data = dict(doc_data)
        self.setWindowTitle(
            f"Confronto PDF e Markdown — {doc_data.get('id', '')} — {doc_data.get('filename', '')}"
        )
        self.resize(1400, 850)
        layout = QVBoxLayout(self)
        self._splitter = QSplitter(Qt.Horizontal)
        self._splitter.setChildrenCollapsible(False)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.addWidget(QLabel('PDF originale'))
        self._pdf = DocumentEvidencePreview(services, self, show_normalized_toggle=False)
        left_layout.addWidget(self._pdf)
        self._splitter.addWidget(left)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.addWidget(QLabel('Markdown estratto — sola lettura'))
        search_bar = QHBoxLayout()
        self._search = QLineEdit()
        self._search.setPlaceholderText('Cerca nel Markdown…')
        self._search.returnPressed.connect(lambda: self._find(False))
        search_bar.addWidget(self._search)
        previous = QPushButton('Precedente')
        previous.clicked.connect(lambda: self._find(True))
        search_bar.addWidget(previous)
        following = QPushButton('Successivo')
        following.clicked.connect(lambda: self._find(False))
        search_bar.addWidget(following)
        right_layout.addLayout(search_bar)
        self._text = QPlainTextEdit()
        self._text.setReadOnly(True)
        self._text.setFont(QFontDatabase.systemFont(QFontDatabase.FixedFont))
        right_layout.addWidget(self._text)
        self._status = QLabel()
        self._status.setWordWrap(True)
        right_layout.addWidget(self._status)
        self._splitter.addWidget(right)
        self._splitter.setSizes([700, 700])
        layout.addWidget(self._splitter, 1)

        buttons = QHBoxLayout()
        reload_button = QPushButton('Ricarica Markdown')
        reload_button.clicked.connect(self._load_markdown)
        buttons.addWidget(reload_button)
        buttons.addStretch()
        close_button = QPushButton('Chiudi')
        close_button.clicked.connect(self.reject)
        buttons.addWidget(close_button)
        layout.addLayout(buttons)
        for button in self.findChildren(QPushButton):
            button.setAutoDefault(False)
        self.finished.connect(lambda _result: self._pdf.close_document())
        self._pdf.set_document(self._doc_data)
        self._load_markdown()

    def _load_markdown(self):
        path = normalized_text_path(self._doc_data)
        self._text.clear()
        if path is None:
            self._status.setText('Markdown non disponibile. Estrai prima il testo clinico del documento.')
            return
        try:
            text = path.read_text(encoding='utf-8')
        except (OSError, UnicodeError) as exc:
            self._status.setText(f'Impossibile leggere il Markdown: {exc}')
            return
        self._text.setPlainText(text)
        self._status.setText(f'{path.name} — {len(text):,} caratteri')

    def _find(self, backwards=False):
        term = self._search.text()
        if not term:
            return
        flags = QTextDocument.FindBackward if backwards else QTextDocument.FindFlags()
        if not self._text.find(term, flags):
            previous = self._text.textCursor()
            cursor = self._text.textCursor()
            cursor.movePosition(QTextCursor.End if backwards else QTextCursor.Start)
            self._text.setTextCursor(cursor)
            if not self._text.find(term, flags):
                self._text.setTextCursor(previous)
                self._status.setText('Nessun risultato nel Markdown.')
                return
        self._text.ensureCursorVisible()
        self._status.setText('Corrispondenza selezionata nel Markdown.')

    def closeEvent(self, event):
        self._pdf.close_document()
        super().closeEvent(event)

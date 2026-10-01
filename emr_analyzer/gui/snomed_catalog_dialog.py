"""Local RF2 International import, retrieval inspection and reviewed Italian aliases."""
from pathlib import Path
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFileDialog, QLineEdit, QListWidget, QListWidgetItem, QCheckBox, QMessageBox)


class CatalogWorker(QThread):
    progress = pyqtSignal(str)
    completed = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, catalog, operation, argument, parent=None):
        super().__init__(parent)
        self.catalog, self.operation, self.argument = catalog, operation, argument

    def run(self):
        try:
            if self.operation == 'search':
                self.completed.emit(self.catalog.search(self.argument, limit=20))
            else:
                function = self.catalog.import_file if self.operation=='import' else self.catalog.build_vectors
                self.completed.emit(function(self.argument, progress=self.progress.emit, cancelled=self.isInterruptionRequested))
        except Exception as exc:
            self.failed.emit(str(exc))
        finally:
            self.catalog.db.close()


class SnomedCatalogDialog(QDialog):
    def __init__(self, catalog, parent=None):
        super().__init__(parent)
        self.catalog = catalog
        self.worker = None
        self.setWindowTitle('Catalogo SNOMED CT condiviso')
        self.resize(850,600)
        layout = QVBoxLayout(self)
        notice = QLabel('SNOMED CT International Edition: concetti, sinonimi inglesi e relazioni inferite. '
            'L’estrazione riconosce anche eventi nuovi; codici e date incerti restano da rivedere.')
        notice.setWordWrap(True)
        layout.addWidget(notice)
        self.status = QLabel()
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self.enabled = QCheckBox('Catalogo SNOMED CT disponibile per il registro FHIR')
        self.enabled.setObjectName('snomed_enabled')
        layout.addWidget(self.enabled)
        bar = QHBoxLayout()
        self.import_button = QPushButton('Importa catalogo SNOMED CT…')
        self.import_button.clicked.connect(self.import_rf2)
        self.vector_button = QPushButton('Crea indice vettoriale da modello locale…')
        self.vector_button.clicked.connect(self.build_vectors)
        self.cancel_button = QPushButton('Interrompi operazione')
        self.cancel_button.clicked.connect(self.cancel)
        self.cancel_button.setEnabled(False)
        for button in (self.import_button,self.vector_button,self.cancel_button):
            bar.addWidget(button)
        layout.addLayout(bar)
        self.query = QLineEdit()
        self.query.setPlaceholderText('Espressione italiana oppure termine inglese…')
        self.query.returnPressed.connect(self.search)
        layout.addWidget(self.query)
        self.search_button = QPushButton('Cerca concetti')
        self.search_button.clicked.connect(self.search)
        layout.addWidget(self.search_button)
        self.results = QListWidget()
        self.results.setWordWrap(True)
        layout.addWidget(self.results)
        self.alias = QLineEdit()
        self.alias.setPlaceholderText('Espressione italiana da associare al concetto selezionato')
        layout.addWidget(self.alias)
        self.alias_button = QPushButton('Conferma associazione italiana')
        self.alias_button.clicked.connect(self.save_alias)
        layout.addWidget(self.alias_button)
        layout.addWidget(QLabel('Senza indice multilingue: cerca in inglese o usa associazioni italiane già confermate.'))
        close = QPushButton('Chiudi')
        close.clicked.connect(self.reject)
        layout.addWidget(close)
        self.refresh()

    def refresh(self):
        meta = self.catalog.metadata()
        self.enabled.blockSignals(True)
        self.enabled.setChecked(self.catalog.available)
        self.enabled.blockSignals(False)
        self.enabled.setEnabled(False)
        self.vector_button.setEnabled(self.catalog.available)
        self.status.setText((f"Release {meta.get('release')} · International RF2 · {int(meta.get('active',0)):,} attivi / {int(meta.get('total',0)):,} totali · "
            + ('ricerca testuale + vettoriale' if meta.get('embedding_model') else 'ricerca testuale')) if self.catalog.available else 'Nessun catalogo SNOMED CT importato (formato supportato: International RF2 Snapshot).')

    def import_rf2(self):
        source = QFileDialog.getExistingDirectory(self,'Seleziona la cartella SNOMED CT International RF2',str(Path.home()/'Desktop'))
        if source:
            self.start('import',source)

    def build_vectors(self):
        model = QFileDialog.getExistingDirectory(self,'Modello SentenceTransformer multilingue locale')
        if model:
            self.start('vectors',model)

    def start(self, operation, argument):
        if self.worker and self.worker.isRunning():
            return
        for control in (self.import_button,self.vector_button,self.enabled,self.search_button,self.alias_button):
            control.setEnabled(False)
        self.cancel_button.setEnabled(True)
        self.status.setText('Operazione in corso…')
        self.worker = CatalogWorker(self.catalog,operation,argument,self)
        self.worker.progress.connect(self.status.setText)
        if operation == 'search':
            self.worker.completed.connect(self.show_results)
        self.worker.failed.connect(lambda error: QMessageBox.warning(self,'Catalogo SNOMED',error))
        self.worker.finished.connect(self.finished_work)
        self.worker.start()

    def finished_work(self):
        for control in (self.import_button,self.search_button,self.alias_button):
            control.setEnabled(True)
        self.cancel_button.setEnabled(False)
        self.refresh()

    def search(self):
        if self.worker and self.worker.isRunning():
            return
        self.results.clear()
        self.start('search', self.query.text())

    def show_results(self, concepts):
        for concept in concepts:
            item = QListWidgetItem(f"{concept['code']} — {concept['fsn']}")
            item.setData(Qt.UserRole,concept)
            self.results.addItem(item)

    def save_alias(self):
        item = self.results.currentItem()
        if not item:
            return
        try:
            self.catalog.add_alias(self.alias.text(),item.data(Qt.UserRole)['code'])
            self.status.setText('Associazione italiana salvata nel catalogo condiviso.')
        except ValueError as exc:
            self.status.setText(str(exc))

    def cancel(self):
        if self.worker:
            self.worker.requestInterruption()
            self.status.setText('Interruzione richiesta…')

    def reject(self):
        if self.worker and self.worker.isRunning():
            self.cancel()
            return
        super().reject()

    def closeEvent(self,event):
        if self.worker and self.worker.isRunning():
            self.cancel()
            event.ignore()
        else:
            event.accept()

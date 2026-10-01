"""Official LOINC import and review of specimen/unit-specific laboratory mappings."""
from pathlib import Path
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QDialog,QVBoxLayout,QLabel,QPushButton,QFileDialog,QLineEdit,QListWidget,QListWidgetItem,QMessageBox
from .snomed_catalog_dialog import CatalogWorker


class LoincCatalogDialog(QDialog):
    def __init__(self,catalog,parent=None):
        super().__init__(parent)
        self.catalog=catalog;self.worker=None
        self.setWindowTitle('Catalogo LOINC condiviso');self.resize(850,600)
        layout=QVBoxLayout(self)
        self.status=QLabel();self.status.setWordWrap(True);layout.addWidget(self.status)
        self.import_button=QPushButton('Importa Loinc.csv oppure ZIP…');layout.addWidget(self.import_button)
        self.import_button.clicked.connect(self.import_catalog)
        self.query=QLineEdit();self.query.setPlaceholderText('Cerca un analita in italiano o inglese');layout.addWidget(self.query)
        search=QPushButton('Cerca');layout.addWidget(search);search.clicked.connect(self.search)
        self.results=QListWidget();self.results.setWordWrap(True);layout.addWidget(self.results)
        self.name=QLineEdit();self.name.setPlaceholderText('Nome normalizzato dell’analita nell’applicazione');layout.addWidget(self.name)
        self.specimen=QLineEdit();self.specimen.setPlaceholderText('Campione esatto riportato (es. siero)');layout.addWidget(self.specimen)
        self.unit=QLineEdit();self.unit.setPlaceholderText('Unità esatta riportata (es. mg/dL)');layout.addWidget(self.unit)
        notice=QLabel('Conferma solo dopo aver verificato proprietà, campione, tempo, scala e metodo. '
            'La corrispondenza verrà riutilizzata per questa combinazione di analita, campione e unità.');notice.setWordWrap(True);layout.addWidget(notice)
        confirm=QPushButton('Conferma associazione LOINC');layout.addWidget(confirm);confirm.clicked.connect(self.confirm)
        close=QPushButton('Chiudi');layout.addWidget(close);close.clicked.connect(self.reject)
        self.refresh()

    def refresh(self):
        meta=self.catalog.metadata()
        self.status.setText(f"LOINC {meta.get('release','non importato')} · {meta.get('count','0')} codici · {meta.get('italian','0')} descrizioni italiane")

    def import_catalog(self):
        if self.worker and self.worker.isRunning():return
        preferred=Path(__file__).resolve().parents[2]/'Loinc_2.83/LoincTable/Loinc.csv'
        path,_=QFileDialog.getOpenFileName(self,'Catalogo ufficiale LOINC',str(preferred), 'LOINC (*.csv *.zip)')
        if not path:return
        self.import_button.setEnabled(False)
        self.worker=CatalogWorker(self.catalog,'import',path,self)
        self.worker.progress.connect(self.status.setText)
        self.worker.failed.connect(lambda e:QMessageBox.warning(self,'LOINC',e))
        self.worker.finished.connect(self.finished_import)
        self.worker.start()

    def finished_import(self):
        self.import_button.setEnabled(True);self.refresh()

    def search(self):
        if self.worker and self.worker.isRunning():return
        self.results.clear()
        for concept in self.catalog.search(self.query.text()):
            axes=concept['axes']
            text=f"{concept['code']} — {concept['label']}\n"+' · '.join(f'{k}: {v}' for k,v in axes.items() if v)
            item=QListWidgetItem(text);item.setData(Qt.UserRole,concept);self.results.addItem(item)

    def confirm(self):
        selected=self.results.currentItem()
        if not selected:return
        try:
            self.catalog.confirm(self.name.text(),self.specimen.text(),self.unit.text(),selected.data(Qt.UserRole)['code'])
            self.status.setText('Associazione confermata. Rigenera il registro FHIR per applicarla.')
        except ValueError as exc:self.status.setText(str(exc))

    def reject(self):
        if self.worker and self.worker.isRunning():
            self.worker.requestInterruption();self.status.setText('Annullamento richiesto…');return
        super().reject()

    def closeEvent(self,event):
        if self.worker and self.worker.isRunning():
            self.worker.requestInterruption();event.ignore()
        else:event.accept()

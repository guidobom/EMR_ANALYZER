"""Patient panel — left sidebar for patient list and management."""

from datetime import datetime

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QListWidget, QListWidgetItem,
    QPushButton, QDialog, QFormLayout, QLineEdit, QComboBox,
    QTextEdit, QDialogButtonBox, QLabel, QMessageBox, QHBoxLayout,
    QInputDialog,
)
from PyQt5.QtCore import pyqtSignal, Qt

from ..models import Patient


# ------------------------------------------------------------------
# New Patient Dialog
# ------------------------------------------------------------------
class NewPatientDialog(QDialog):
    """Dialog for creating a new patient."""

    def __init__(self, patient_repo, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Nuovo Paziente")
        self.setMinimumWidth(450)
        self._patient_repo = patient_repo
        self._setup_ui()

    def _setup_ui(self):
        layout = QFormLayout(self)

        self._pseudonym = QLineEdit()
        self._pseudonym.setPlaceholderText("Codice pseudonimo (es. P001)")
        layout.addRow("Codice paziente:*", self._pseudonym)

        self._initials = QLineEdit()
        self._initials.setPlaceholderText("AB")
        self._initials.setMaxLength(4)
        layout.addRow("Iniziali:", self._initials)

        self._sex = QComboBox()
        self._sex.addItems(["", "M", "F", "Altro"])
        layout.addRow("Sesso:", self._sex)

        self._birth_year = QLineEdit()
        self._birth_year.setPlaceholderText("AAAA")
        layout.addRow("Anno di nascita:", self._birth_year)

        self._pathology = QLineEdit()
        self._pathology.setPlaceholderText("Patologia principale")
        layout.addRow("Patologia:", self._pathology)

        self._center = QLineEdit()
        self._center.setPlaceholderText("Centro di riferimento")
        layout.addRow("Centro:", self._center)

        self._notes = QTextEdit()
        self._notes.setMaximumHeight(80)
        self._notes.setPlaceholderText("Note generali")
        layout.addRow("Note:", self._notes)

        # Buttons
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._on_create)
        buttons.rejected.connect(self.reject)
        layout.addRow(buttons)

    def _on_create(self):
        pseudonym = self._pseudonym.text().strip()
        if not pseudonym:
            QMessageBox.warning(self, "Attenzione",
                                "Il codice paziente è obbligatorio.")
            return

        patient_id = self._patient_repo.get_next_id()
        birth_str = self._birth_year.text().strip()
        patient = Patient(
            id=patient_id,
            pseudonym=pseudonym,
            initials=self._initials.text().strip() or None,
            sex=self._sex.currentText() or None,
            birth_year=int(birth_str) if birth_str.isdigit() else None,
            main_pathology=self._pathology.text().strip() or None,
            center=self._center.text().strip() or None,
            notes=self._notes.toPlainText().strip() or None,
            created_at=datetime.now().isoformat(),
            updated_at=datetime.now().isoformat(),
        )
        try:
            self._patient_repo.insert(patient)
            self.accept()
        except Exception as e:
            QMessageBox.critical(self, "Errore",
                                 f"Impossibile creare il paziente: {e}")


# ------------------------------------------------------------------
# Patient Panel
# ------------------------------------------------------------------
class PatientPanel(QWidget):
    """Left sidebar listing patients with search and actions."""

    patient_selected = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumWidth(200)
        self.setMaximumWidth(320)
        self._patient_repo = None
        self._services = {}
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        # Title
        title = QLabel("PAZIENTI")
        title.setObjectName("subheading")
        layout.addWidget(title)

        # Search
        self._search = QLineEdit()
        self._search.setPlaceholderText("🔍 Cerca paziente...")
        self._search.textChanged.connect(self._on_search)
        layout.addWidget(self._search)

        # Patient list
        self._patient_list = QListWidget()
        self._patient_list.itemClicked.connect(self._on_item_clicked)
        layout.addWidget(self._patient_list, stretch=1)

        # Buttons — vertical layout so all are visible
        btn_layout = QVBoxLayout()
        btn_layout.setSpacing(4)

        self._new_btn = QPushButton("➕ Nuovo Paziente")
        self._new_btn.clicked.connect(self._on_new_patient)
        btn_layout.addWidget(self._new_btn)

        # Edit + Delete in a horizontal row
        row_layout = QHBoxLayout()
        self._edit_btn = QPushButton("✎ Modifica")
        self._edit_btn.clicked.connect(self._on_edit_patient)
        row_layout.addWidget(self._edit_btn)

        self._delete_btn = QPushButton("🗑 Elimina")
        self._delete_btn.setObjectName("dangerButton")
        self._delete_btn.clicked.connect(self._on_delete_patient)
        row_layout.addWidget(self._delete_btn)

        btn_layout.addLayout(row_layout)
        layout.addLayout(btn_layout)

    def set_services(self, services: dict):
        self._services = services
        self._patient_repo = services.get("patient_repo")
        self.refresh()

    def refresh(self, query: str = ""):
        """Reload the patient list."""
        if not self._patient_repo:
            return
        self._patient_list.clear()
        patients = (self._patient_repo.search(query) if query
                    else self._patient_repo.list_all())
        for p in patients:
            item = QListWidgetItem()
            # Build display: "▪ MR • M • 68 aa — P001"
            label_parts = ["▪"]
            if p.initials:
                label_parts.append(p.initials)
            if p.sex:
                label_parts.append(f"• {p.sex}")
            if p.birth_year:
                age = datetime.now().year - p.birth_year
                label_parts.append(f"• {age} aa")
            label_parts.append(f"— {p.id}")
            item.setText(" ".join(label_parts))

            info_parts = []
            if p.main_pathology:
                info_parts.append(p.main_pathology)
            if p.center:
                info_parts.append(p.center)
            item.setToolTip(
                f"Pseudonimo: {p.pseudonym}\n"
                f"Sesso: {p.sex or 'N/D'}\n"
                f"Anno nascita: {p.birth_year or 'N/D'}\n"
                f"Patologia: {p.main_pathology or 'N/D'}\n"
                f"Centro: {p.center or 'N/D'}\n"
                f"Creato: {p.created_at[:10]}"
            )
            item.setData(Qt.UserRole, p.id)
            item.setData(Qt.UserRole + 1, p.to_dict())
            self._patient_list.addItem(item)

    def _on_search(self, text: str):
        self.refresh(text)

    def _on_item_clicked(self, item: QListWidgetItem):
        patient_id = item.data(Qt.UserRole)
        self.patient_selected.emit(patient_id)

    def _on_new_patient(self):
        dialog = NewPatientDialog(self._patient_repo, self)
        if dialog.exec_():
            self.refresh()

    def _on_edit_patient(self):
        item = self._patient_list.currentItem()
        if not item:
            return
        patient_id = item.data(Qt.UserRole)
        patient = self._patient_repo.get_by_id(patient_id)
        if patient:
            # Simple edit via input dialogs
            new_pseudonym, ok = QInputDialog.getText(
                self, "Modifica Paziente", "Pseudonimo:",
                text=patient.pseudonym)
            if ok and new_pseudonym.strip():
                patient.pseudonym = new_pseudonym.strip()
                patient.updated_at = datetime.now().isoformat()
                self._patient_repo.update(patient)
                self.refresh()

    def _on_delete_patient(self):
        item = self._patient_list.currentItem()
        if not item:
            return
        patient_id = item.data(Qt.UserRole)
        reply = QMessageBox.question(
            self, "Conferma eliminazione",
            f"Eliminare definitivamente il workspace {patient_id} e TUTTI "
            f"i dati collegati?\n\n"
            f"Verranno cancellati:\n"
            f"• documenti originali, testo grezzo e testo normalizzato\n"
            f"• tabelle, coordinate, eventi ed esami di laboratorio\n"
            f"• valutazioni, validazioni e Clinical State con tutta la cronologia\n"
            f"• identità di routing, audit e copie temporanee riconducibili\n"
            f"• l'intera cartella workspace e la cache dedicata\n\n"
            f"Questa azione è IRREVERSIBILE.",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            deletion = self._services.get("patient_workspace_deletion")
            if deletion is None:
                QMessageBox.critical(
                    self, "Cancellazione non disponibile",
                    "Il servizio di cancellazione integrale non è inizializzato.",
                )
                return
            result = deletion.delete(patient_id)
            if not result.deleted:
                details = result.error or "Errore non specificato"
                if result.warnings:
                    details += "\n" + "\n".join(result.warnings)
                QMessageBox.critical(
                    self, "Cancellazione incompleta",
                    f"Il workspace {patient_id} non è stato eliminato "
                    f"completamente.\n\n{details}",
                )
                return
            self.patient_selected.emit("")
            self.refresh()
            QMessageBox.information(
                self, "Workspace eliminato",
                f"Il workspace {patient_id} e tutti i dati collegati sono "
                "stati eliminati definitivamente.",
            )

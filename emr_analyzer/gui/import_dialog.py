"""Import dialog — shows file checks before processing."""

import json
import os
import shutil
from pathlib import Path

from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QTableWidget, QTableWidgetItem,
    QPushButton, QLabel, QHeaderView, QMessageBox, QComboBox,
    QCheckBox, QDialogButtonBox,
)
from PyQt5.QtCore import Qt, QThread

from ..utils.file_utils import compute_file_hash, verify_pdf, get_file_info, is_supported_file
from ..models.document import DocumentType, DocumentRecord, ParsingStatus
from ..config import active_workspace
from .quick_look import QuickLook


def guess_document_type(filename: str) -> str:
    """Quick type guess based on filename."""
    fn = filename.lower()
    if any(k in fn for k in ["lab", "laboratorio", "esami", "analisi", "emocromo"]):
        return DocumentType.LABORATORIO.value
    if any(k in fn for k in ["tac", "rm", "rx", "eco", "radiologia", "mammo"]):
        return DocumentType.RADIOLOGIA.value
    if any(k in fn for k in ["visita", "ambulatori", "specialist", "onco"]):
        return DocumentType.VISITA_SPECIALISTICA.value
    if any(k in fn for k in ["dimissione", "lettera"]):
        return DocumentType.LETTERA_DIMISSIONE.value
    if any(k in fn for k in ["sdo"]):
        return DocumentType.SDO.value
    if any(k in fn for k in ["cartella", "ricovero", "clinica"]):
        return DocumentType.CARTELLA_CLINICA.value
    if any(k in fn for k in ["diario", "medico"]):
        return DocumentType.DIARIO_MEDICO.value
    if any(k in fn for k in ["terapi", "piano"]):
        return DocumentType.PIANO_TERAPEUTICO.value
    return DocumentType.NON_CLASSIFICATO.value


def run_file_checks(services: dict, file_paths: list[str],
                    file_metadata: dict[str, dict] | None = None) -> list[dict]:
    """Run all file checks and return per-file metadata dictionaries.

    Each entry mirrors what the import table shows: path, info, check (PDF
    verify result), hash, duplicate flag, guessed type, status.
    """
    doc_repo = services.get("document_repo")
    file_metadata = file_metadata or {}
    checks = []
    for fp in file_paths:
        staged_metadata = file_metadata.get(str(fp), {})
        info = get_file_info(fp)
        if staged_metadata.get("original_name"):
            info["filename"] = staged_metadata["original_name"]
        if not is_supported_file(fp):
            checks.append({"path": fp, "status": "Formato non supportato"})
            continue

        check = staged_metadata.get("check") or (
            verify_pdf(fp) if info["extension"] == ".pdf" else {
                "readable": True, "page_count": 1, "has_text": True,
                "is_protected": False,
            }
        )

        file_hash = staged_metadata.get("hash") or compute_file_hash(fp)

        is_duplicate = False
        if doc_repo:
            is_duplicate = doc_repo.get_by_hash_global(file_hash) is not None

        checks.append({
            "path": fp,
            "info": info,
            "check": check,
            "hash": file_hash,
            "is_duplicate": is_duplicate,
            "guessed_type": guess_document_type(info["filename"]),
            "original_name": info["filename"],
            "identity_evidence": staged_metadata.get("identity_evidence"),
            "status": "Pronto" if not check.get("error") and not is_duplicate
                      else "⚠️ " + check.get("error", "Duplicato"),
        })
    return checks


def import_checked_documents(services: dict, patient_id: str,
                             file_checks: list[dict], selected_indexes,
                             status_callback=None,
                             type_overrides: dict[int, str] | None = None
                             ) -> list[str]:
    """Copy the checked files into a patient workspace and create records.

    Returns the imported document ids.  Duplicate files and unreadable ones
    are skipped silently.  Raises on hard failures (workspace missing, copy
    error, database error) so the caller can surface a message.
    """
    doc_repo = services.get("document_repo")
    if not doc_repo:
        raise RuntimeError("Repository documenti non disponibile.")

    patient_repo = services.get("patient_repo")
    if not patient_repo or not patient_repo.get_by_id(patient_id):
        raise ValueError(
            f"Il paziente {patient_id} non esiste nel registro.\n"
            "Chiudi questa finestra, riapri il workspace e conferma il "
            "suo recupero prima di importare i documenti."
        )

    workspace_dir = active_workspace.path / patient_id / "documents" / "original"
    workspace_dir.mkdir(parents=True, exist_ok=True)

    type_overrides = type_overrides or {}
    selected = set(selected_indexes)
    imported: list[str] = []
    for i, check_data in enumerate(file_checks):
        if i not in selected or "hash" not in check_data:
            continue

        # Re-check here as well: an earlier row in the same batch may have
        # inserted an identical file after the preview was built.
        if doc_repo.get_by_hash_global(check_data["hash"]):
            continue

        doc_type = type_overrides.get(i) or check_data.get("guessed_type") \
            or DocumentType.NON_CLASSIFICATO.value

        src_path = check_data["path"]
        dst_name = check_data.get("original_name") or os.path.basename(src_path)
        dst_path = workspace_dir / dst_name
        needs_copy = True

        # Preserve both files when two different reports share a filename.
        if dst_path.exists():
            existing_hash = compute_file_hash(dst_path)
            if existing_hash == check_data["hash"]:
                needs_copy = False
            else:
                source_name = Path(dst_name)
                dst_name = (
                    f"{source_name.stem}__{check_data['hash'][:8]}"
                    f"{source_name.suffix}"
                )
                dst_path = workspace_dir / dst_name
                suffix = 2
                while dst_path.exists():
                    dst_name = (
                        f"{source_name.stem}__{check_data['hash'][:8]}_{suffix}"
                        f"{source_name.suffix}"
                    )
                    dst_path = workspace_dir / dst_name
                    suffix += 1

        copied_by_this_import = False
        if needs_copy:
            try:
                shutil.copy2(src_path, dst_path)
                copied_by_this_import = True
            except OSError as exc:
                if status_callback:
                    status_callback(i, f"❌ Copia fallita: {exc}")
                raise

        doc_id = doc_repo.get_next_id()
        identity_evidence = check_data.get("identity_evidence")
        identity_metadata = None
        if identity_evidence:
            identity_metadata = {
                "confidence": identity_evidence.confidence,
                "method": identity_evidence.extraction_method,
                "fields": sorted(identity_evidence.fields),
            }
        doc = DocumentRecord(
            id=doc_id,
            patient_id=patient_id,
            filename=dst_name,
            original_path=str(dst_path),
            file_hash=check_data["hash"],
            page_count=check_data["check"].get("page_count", 1),
            document_date=None,  # Will be extracted later
            document_type=doc_type,
            parsing_status=ParsingStatus.PENDING.value,
            metadata_json=(
                json.dumps({"identity_assignment": identity_metadata})
                if identity_metadata else None
            ),
        )
        try:
            doc_repo.insert(doc)
        except Exception as exc:
            # Roll back only a file created by this attempt. A pre-existing
            # identical file may belong to a recoverable older workspace.
            if copied_by_this_import:
                try:
                    dst_path.unlink(missing_ok=True)
                except OSError:
                    pass
            if status_callback:
                status_callback(i, f"❌ Database: {exc}")
            raise

        identity_repo = services.get("identity_repo")
        if identity_repo and identity_evidence:
            identity_repo.add_document_evidence(
                doc_id, patient_id, identity_evidence
            )
            if identity_evidence.is_strong and not identity_repo.has_identity(
                patient_id
            ):
                identity_repo.upsert(
                    patient_id,
                    identity_evidence,
                    source_document_id=doc_id,
                    status="import",
                )
        imported.append(doc_id)

        audit_repo = services.get("audit_repo")
        if audit_repo:
            audit_repo.log(patient_id, "import", "document", doc_id,
                           {"filename": dst_name, "hash": check_data["hash"]})

    return imported


class ImportDialog(QDialog):
    """Dialog showing import status for each file before processing."""

    # Signal emitted with list of doc_ids that were imported
    import_completed = None  # Will be set by parent

    def __init__(self, file_paths: list[str], services: dict,
                 patient_id: str, parent=None,
                 workspace_tabs=None, file_metadata: dict[str, dict] | None = None):
        super().__init__(parent)
        self.setWindowTitle("Importazione Documenti")
        self.setMinimumSize(900, 520)
        self._services = services
        self._patient_id = patient_id
        self._file_paths = file_paths
        self._file_checks = []
        self._workspace_tabs = workspace_tabs
        self._file_metadata = file_metadata or {}
        self._imported_doc_ids = []
        self._setup_ui()
        self._run_checks()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        # Header
        header = QLabel(f"Importazione documenti per paziente {self._patient_id}")
        header.setObjectName("heading")
        layout.addWidget(header)

        sub = QLabel(f"{len(self._file_paths)} file(s) selezionato(i)")
        layout.addWidget(sub)

        # Table
        self._table = QTableWidget()
        self._table.setColumnCount(7)
        self._table.setHorizontalHeaderLabels([
            "Importa", "File", "Pagine", "Testo nativo",
            "Duplicato", "Tipo presunto", "Stato"
        ])
        self._table.horizontalHeader().setStretchLastSection(True)
        self._table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self._table.setAlternatingRowColors(True)
        # Double-click opens the full viewer; spacebar opens an in-app Quick
        # Look preview of the current file (Space/Esc again closes it).
        self._table.doubleClicked.connect(self._on_double_click)
        self._quick_look = QuickLook(self._table, self._quick_look_path)
        layout.addWidget(self._table, stretch=1)

        # Bottom: type override + buttons
        bottom = QHBoxLayout()

        bottom.addWidget(QLabel("Tipo per selezionati:"))
        self._type_combo = QComboBox()
        self._type_combo.addItem("(auto)", None)
        for dt in DocumentType:
            self._type_combo.addItem(dt.value, dt.value)
        self._type_combo.currentIndexChanged.connect(self._on_type_override)
        bottom.addWidget(self._type_combo)

        # Auto-process checkbox
        self._auto_process_chk = QCheckBox("⚡ Elabora automaticamente dopo l'importazione")
        self._auto_process_chk.setChecked(True)
        self._auto_process_chk.setToolTip(
            "Esegue la stessa pipeline del pulsante 'Estrai testo clinico': "
            "pdfplumber, fallback PyMuPDF, eventuale OCR, laboratorio e "
            "normalizzazione LLM"
        )
        bottom.addWidget(self._auto_process_chk)

        bottom.addStretch()

        # Buttons
        buttons = QDialogButtonBox()
        self._import_btn = QPushButton("Importa selezionati")
        self._import_btn.setDefault(True)
        self._import_btn.clicked.connect(self._on_import)
        cancel_btn = QPushButton("Annulla")
        cancel_btn.clicked.connect(self.reject)
        buttons.addButton(self._import_btn, QDialogButtonBox.AcceptRole)
        buttons.addButton(cancel_btn, QDialogButtonBox.RejectRole)
        bottom.addWidget(buttons)

        layout.addLayout(bottom)

    def _run_checks(self):
        """Run all file checks and populate the table."""
        self._file_checks = run_file_checks(
            self._services, self._file_paths, self._file_metadata
        )
        self._table.setRowCount(len(self._file_checks))

        for i, check_data in enumerate(self._file_checks):
            if "check" not in check_data:
                self._fill_row(i, fp=check_data["path"], check={},
                               status="❌ Formato non supportato")
                continue
            check = check_data["check"]
            status = (
                "✓ Pronto" if not check.get("error")
                and not check_data["is_duplicate"]
                else ("⚠️ Duplicato" if check_data["is_duplicate"]
                      else f"⚠️ {check.get('error', '')}")
            )
            self._fill_row(
                i, check_data["path"], check, status,
                check_data["is_duplicate"], check_data["guessed_type"],
            )

    def _fill_row(self, i: int, fp: str, check: dict, status: str,
                  is_duplicate: bool = False, guessed_type: str = ""):
        """Fill a table row with file check results."""
        # Checkbox
        chk = QCheckBox()
        chk.setChecked(status.startswith("✓"))
        chk.setEnabled(status.startswith("✓"))
        self._table.setCellWidget(i, 0, chk)

        # Filename
        metadata = self._file_metadata.get(str(fp), {})
        shown_name = metadata.get("original_name") or os.path.basename(fp)
        self._table.setItem(i, 1, QTableWidgetItem(shown_name))

        # Pages
        pages = check.get("page_count", 1) if check.get("page_count", 1) > 0 else 1
        self._table.setItem(i, 2, QTableWidgetItem(str(pages)))

        # Has native text
        has_text = "Sì" if check.get("has_text") else "No"
        self._table.setItem(i, 3, QTableWidgetItem(has_text))

        # Duplicate
        self._table.setItem(i, 4, QTableWidgetItem("Sì" if is_duplicate else "No"))

        # Guessed type
        self._table.setItem(i, 5, QTableWidgetItem(guessed_type))

        # Status
        self._table.setItem(i, 6, QTableWidgetItem(status))

        # Store full path
        self._table.item(i, 1).setData(Qt.UserRole, fp)

    def _on_type_override(self, index):
        """Apply type override to all selected rows."""
        new_type = self._type_combo.currentData()
        if new_type is None:
            return
        for i in range(self._table.rowCount()):
            chk = self._table.cellWidget(i, 0)
            if chk and chk.isChecked():
                self._table.setItem(i, 5, QTableWidgetItem(new_type))

    def _on_double_click(self, index):
        """Open the full PDF viewer for the double-clicked file."""
        row = index.row()
        item = self._table.item(row, 1)
        if not item:
            return
        path = item.data(Qt.UserRole)
        if not path or not os.path.exists(str(path)):
            return
        self._quick_look.dismiss()
        from .pdf_viewer import PDFViewerDialog
        viewer = PDFViewerDialog(
            {"filename": os.path.basename(str(path)), "original_path": str(path)},
            self._services,
            self,
        )
        viewer.exec_()

    def _quick_look_path(self):
        """Resolve the current table row to a file path for the Quick Look."""
        row = self._table.currentRow()
        if row < 0:
            return None
        item = self._table.item(row, 1)
        if not item:
            return None
        path = item.data(Qt.UserRole)
        if not path or not os.path.exists(str(path)):
            return None
        return str(path), os.path.basename(str(path))

    def _on_import(self):
        """Copy files to workspace and create DocumentRecords."""
        selected = []
        for i in range(len(self._file_checks)):
            chk = self._table.cellWidget(i, 0)
            if chk and chk.isChecked():
                selected.append(i)
        type_overrides = {}
        for i in range(len(self._file_checks)):
            type_item = self._table.item(i, 5)
            if type_item:
                type_overrides[i] = type_item.text()

        try:
            self._imported_doc_ids = import_checked_documents(
                self._services, self._patient_id, self._file_checks,
                selected,
                status_callback=self._set_row_status,
                type_overrides=type_overrides,
            )
        except ValueError as exc:
            QMessageBox.critical(self, "Workspace non registrato", str(exc))
            return
        except OSError as exc:
            QMessageBox.critical(
                self, "Errore di copia",
                f"Impossibile copiare i file:\n{exc}",
            )
            return
        except Exception as exc:
            QMessageBox.critical(
                self, "Importazione non riuscita",
                f"Il documento non è stato registrato.\n\n{exc}",
            )
            return

        if self._imported_doc_ids:
            self.accept()
        else:
            QMessageBox.information(
                self, "Nessun import",
                "Nessun file selezionato per l'importazione.",
            )

    def _set_row_status(self, index: int, message: str):
        """Update the status cell of a row (used as import callback)."""
        item = self._table.item(index, 6)
        if item:
            item.setText(message)

    def should_auto_process(self) -> bool:
        """Whether the user wants to auto-process after import."""
        return self._auto_process_chk.isChecked()

    def get_imported_doc_ids(self) -> list[str]:
        """Return the list of imported document IDs."""
        return self._imported_doc_ids

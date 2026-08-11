"""Documents tab with drag-and-drop zone and document list."""

from __future__ import annotations

import os
import json
import traceback
from datetime import datetime

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QTableWidget,
    QTableWidgetItem, QHeaderView, QAbstractItemView, QLabel,
    QFileDialog, QMessageBox, QMenu, QAction,
)
from PyQt5.QtCore import pyqtSignal, Qt, QMimeData, QTimer
from PyQt5.QtGui import QDragEnterEvent, QDropEvent

from ..models.document import DocumentType, ParsingStatus, ExtractionStatus
from ..extraction.clinical_text_isolator import ClinicalTextIsolationError


class _NullProgressLogger:
    """No-op ``add_log`` for parallel LLM workers.

    ``_run_llm_extraction`` is invoked from worker threads with
    ``progress=None``; the shared ProgressDialog widget must only be touched
    from the GUI thread, so the per-document internal logs are discarded and
    the parallel loop reports each document's outcome on the main thread.
    """

    def add_log(self, message: str) -> None:
        pass


class DropZoneWidget(QWidget):
    """Widget that accepts drag-and-drop of PDF/image files."""

    files_dropped = pyqtSignal(list)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("dropZone")
        self.setAcceptDrops(True)
        self.setMinimumHeight(100)

        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignCenter)
        label = QLabel(
            "📂 Trascina qui i PDF o clicca per importare\n"
            "I pazienti vengono rilevati automaticamente"
        )
        label.setAlignment(Qt.AlignCenter)
        label.setStyleSheet("color: #7f8c8d; font-size: 13px; padding: 12px;")
        layout.addWidget(label)

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            self.setStyleSheet(
                "QWidget#dropZone { border: 3px dashed #3498db; "
                "background-color: #eaf2fd; border-radius: 8px; }"
            )

    def dragLeaveEvent(self, event):
        self.setStyleSheet(
            "QWidget#dropZone { border: 3px dashed #bdc3c7; "
            "background-color: #f8f9fa; border-radius: 8px; }"
        )

    def dropEvent(self, event: QDropEvent):
        # Finalize the drop immediately: the import below can show modal
        # dialogs and take a long time.  Running it synchronously here keeps
        # the X11 drag transaction open, and the file manager's drag "ghost"
        # pixmap stays stuck on screen (even after the app exits) because the
        # compositor never receives the drag-end.
        event.accept()
        self.setStyleSheet(
            "QWidget#dropZone { border: 3px dashed #bdc3c7; "
            "background-color: #f8f9fa; border-radius: 8px; }"
        )
        files = []
        for url in event.mimeData().urls():
            path = url.toLocalFile()
            if os.path.isfile(path):
                files.append(path)
            elif os.path.isdir(path):
                for root, _, filenames in os.walk(path):
                    for f in filenames:
                        fp = os.path.join(root, f)
                        if f.lower().endswith(('.pdf', '.jpg', '.jpeg', '.png')):
                            files.append(fp)
        if files:
            QTimer.singleShot(0, lambda: self.files_dropped.emit(files))

    def mousePressEvent(self, event):
        """Click to open file dialog."""
        files, _ = QFileDialog.getOpenFileNames(
            self, "Seleziona Documenti PDF",
            "", "Documenti (*.pdf *.jpg *.jpeg *.png);;Tutti i file (*)"
        )
        if files:
            self.files_dropped.emit(files)


class DocumentsTab(QWidget):
    """Tab showing document list with drop zone."""

    document_selected = pyqtSignal(str, dict)
    import_requested = pyqtSignal(list)
    processing_complete = pyqtSignal(str)        # patient_id

    def __init__(self, parent=None):
        super().__init__(parent)
        self._services = {}
        self._current_patient_id = None
        self._consecutive_llm_errors = 0
        self._batch_success_count = 0
        self._batch_error_count = 0
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        # Drop zone (hidden when documents exist)
        self._drop_zone = DropZoneWidget()
        self._drop_zone.files_dropped.connect(self.import_requested.emit)
        layout.addWidget(self._drop_zone)

        # Toolbar for document actions
        toolbar = QHBoxLayout()
        self._import_btn = QPushButton("📥 Importa")
        self._import_btn.clicked.connect(self._on_import_click)

        # The parser layers are deliberately transparent to the user: this
        # single action runs native extraction, OCR if needed, laboratory
        # parsing and LLM clinical-text normalization.
        self._extract_clinical_text_btn = QPushButton("🧠 Estrai testo clinico")
        self._extract_clinical_text_btn.setObjectName("successButton")
        self._extract_clinical_text_btn.clicked.connect(
            lambda: self.extract_clinical_text()
        )
        self._extract_clinical_text_btn.setEnabled(False)

        self._view_pdf_btn = QPushButton("📖 Apri PDF")
        self._view_pdf_btn.clicked.connect(self._on_view_pdf)
        self._view_pdf_btn.setEnabled(False)

        self._delete_btn = QPushButton("🗑 Elimina selezionati")
        self._delete_btn.clicked.connect(self._on_delete_selected)
        self._delete_btn.setEnabled(False)

        toolbar.addWidget(self._import_btn)
        toolbar.addWidget(self._extract_clinical_text_btn)
        toolbar.addWidget(self._view_pdf_btn)
        toolbar.addWidget(self._delete_btn)
        toolbar.addStretch()
        layout.addLayout(toolbar)

        # Document table
        self._table = QTableWidget()
        self._table.setColumnCount(10)
        self._table.setHorizontalHeaderLabels([
            "File", "Tipo", "Data Doc", "Pagine", "Dimensione",
            "Importato il", "Stato elaborazione", "Testo clinico",
            "Lab Values", "Validato"
        ])
        self._table.horizontalHeader().setStretchLastSection(True)
        self._table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self._table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self._table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._table.setAlternatingRowColors(True)
        self._table.setContextMenuPolicy(Qt.CustomContextMenu)
        self._table.customContextMenuRequested.connect(self._on_context_menu)
        self._table.itemSelectionChanged.connect(self._on_selection_changed)
        self._table.doubleClicked.connect(self._on_double_click)
        self._table.installEventFilter(self)
        layout.addWidget(self._table, stretch=1)

    def set_services(self, services: dict):
        self._services = services

    def load_patient(self, patient_id: str):
        """Load documents for the given patient."""
        self._current_patient_id = patient_id
        self._refresh_table()

    def _refresh_table(self):
        if not self._current_patient_id or not self._services.get("document_repo"):
            return

        doc_repo = self._services["document_repo"]
        docs = doc_repo.list_by_patient(self._current_patient_id)

        self._table.setRowCount(len(docs))
        # Always show the drop zone so users can add documents anytime
        self._drop_zone.setMinimumHeight(100)
        self._drop_zone.setVisible(True)

        pending_clinical_text = 0

        for i, doc in enumerate(docs):
            self._table.setItem(i, 0, QTableWidgetItem(doc.filename))
            self._table.setItem(i, 1, QTableWidgetItem(doc.document_type))
            self._table.setItem(i, 2, QTableWidgetItem(doc.document_date or ""))
            self._table.setItem(i, 3, QTableWidgetItem(str(doc.page_count)))
            # Size from original path
            size_mb = ""
            if os.path.exists(doc.original_path):
                size_mb = f"{os.path.getsize(doc.original_path) / (1024*1024):.1f} MB"
            self._table.setItem(i, 4, QTableWidgetItem(size_mb))
            self._table.setItem(i, 5, QTableWidgetItem(doc.import_date[:10]))
            if doc.extraction_status == ExtractionStatus.DONE.value:
                processing_status = "completato"
            elif (
                doc.parsing_status == ParsingStatus.PROCESSING.value
                or doc.extraction_status == ExtractionStatus.PROCESSING.value
            ):
                processing_status = "in elaborazione"
            elif (
                doc.parsing_status == ParsingStatus.ERROR.value
                or doc.extraction_status == ExtractionStatus.ERROR.value
            ):
                processing_status = "errore"
            else:
                processing_status = "da elaborare"
            self._table.setItem(i, 6, QTableWidgetItem(processing_status))
            clinical_text_status = (
                "✓" if doc.extraction_status == ExtractionStatus.DONE.value
                else "—"
            )
            self._table.setItem(i, 7, QTableWidgetItem(clinical_text_status))
            self._table.setItem(i, 8, QTableWidgetItem(str(doc.lab_value_count)))

            val_text = "✓" if doc.validation_status == "validated" else "—"
            self._table.setItem(i, 9, QTableWidgetItem(val_text))

            # Color rows by status
            if doc.parsing_status == ParsingStatus.ERROR.value:
                for col in range(10):
                    item = self._table.item(i, col)
                    if item:
                        item.setForeground(Qt.red)

            if processing_status == "da elaborare":
                pending_clinical_text += 1
                self._table.item(i, 6).setForeground(Qt.darkYellow)
            elif processing_status == "errore":
                pending_clinical_text += 1
                self._table.item(i, 6).setForeground(Qt.red)
            elif processing_status == "completato":
                self._table.item(i, 6).setForeground(Qt.darkGreen)

            # Store doc_id in the first column
            self._table.item(i, 0).setData(Qt.UserRole, doc.id)
            self._table.item(i, 0).setData(Qt.UserRole + 1, doc.to_dict())

        self._extract_clinical_text_btn.setEnabled(pending_clinical_text > 0)
        self._extract_clinical_text_btn.setToolTip(
            f"Avvia la pipeline completa per {pending_clinical_text} "
            "documento/i non elaborato/i"
            if pending_clinical_text else
            "Tutti i documenti hanno già un testo clinico normalizzato"
        )

    def _on_selection_changed(self):
        rows = set()
        for item in self._table.selectedItems():
            rows.add(item.row())
        has_selection = len(rows) > 0
        self._view_pdf_btn.setEnabled(has_selection)
        self._delete_btn.setEnabled(has_selection)

        if has_selection:
            row = min(rows)
            item = self._table.item(row, 0)
            if item:
                doc_id = item.data(Qt.UserRole)
                doc_data = item.data(Qt.UserRole + 1)
                self.document_selected.emit(doc_id, doc_data)
        else:
            self._delete_btn.setText("🗑 Elimina selezionati")
        if has_selection:
            self._delete_btn.setText(f"🗑 Elimina {len(rows)} selezionati")

    def _on_double_click(self, index):
        row = index.row()
        item = self._table.item(row, 0)
        if item:
            doc_data = item.data(Qt.UserRole + 1)
            self._open_pdf_viewer(doc_data)

    def _on_import_click(self):
        files, _ = QFileDialog.getOpenFileNames(
            self, "Seleziona Documenti PDF",
            "", "Documenti (*.pdf *.jpg *.jpeg *.png);;Tutti i file (*)"
        )
        if files:
            self.import_requested.emit(files)

    def extract_clinical_text(self, doc_ids=None):
        """Run the complete clinical-text pipeline for unprocessed documents.

        This is the shared entry point used both by the visible button and by
        "Elabora automaticamente dopo l'importazione".

        Two phases so the configured parallel workers are actually used for
        the GPU-bound LLM step:
          1. parse + classify + lab for every document that still needs it
             (CPU-bound, sequential, fast);
          2. LLM isolation over every parsed document, in parallel.
        """
        doc_repo = self._services.get("document_repo")
        if not doc_repo or not self._current_patient_id:
            return
        requested = set(doc_ids) if doc_ids is not None else None
        to_parse = []
        parsed = []
        for doc in doc_repo.list_by_patient(self._current_patient_id):
            if requested is not None and doc.id not in requested:
                continue
            if doc.extraction_status == ExtractionStatus.DONE.value:
                continue
            if (
                doc.parsing_status == ParsingStatus.PROCESSING.value
                or doc.extraction_status == ExtractionStatus.PROCESSING.value
            ):
                continue
            if doc.parsing_status in (
                ParsingStatus.COMPLETED.value,
                ParsingStatus.COMPLETED_WITH_WARNINGS.value,
            ):
                parsed.append(doc.id)
            else:
                to_parse.append(doc.id)

        if to_parse:
            # Phase 1 (CPU): parse + classify + lab, no LLM.
            self._process_documents(to_parse, parse_only=True)
        llm_ids = parsed + to_parse
        if llm_ids:
            # Phase 2 (GPU): parallel LLM isolation over every parsed document.
            self._process_documents(llm_ids, llm_only=True)

    def _get_selected_doc_ids(self) -> list[str]:
        rows = set()
        for item in self._table.selectedItems():
            rows.add(item.row())
        return [self._table.item(r, 0).data(Qt.UserRole)
                for r in rows if self._table.item(r, 0)]

    def _process_documents(self, doc_ids: list[str],
                           parse_only: bool = False,
                           llm_only: bool = False):
        """
        Run the processing pipeline.

        parse_only=True:  pdfplumber + classification + lab (NO LLM)
        llm_only=True:    normalized clinical text from already-parsed docs
        both False:       full pipeline (parse + LLM)
        """
        from .progress_dialog import ProgressDialog

        self._consecutive_llm_errors = 0
        self._batch_success_count = 0
        self._batch_error_count = 0

        if not parse_only:
            document_llm = self._services.get("document_llm_client")
            if not document_llm or not document_llm.is_available:
                QMessageBox.warning(
                    self,
                    "LLM documentale non disponibile",
                    "Configura e testa il modello per i documenti tramite "
                    "il pulsante 'Configura LLM' nella barra superiore.",
                )
                return

        # Determine parallel workers for the Document model
        num_workers = 1
        if not parse_only:
            llm_configs = self._services.get("llm_configs")
            if llm_configs:
                doc_config = llm_configs.get("document")
                if doc_config:
                    num_workers = getattr(doc_config, "parallel_workers", 1)

        # Use parallel processing when >1 worker and not parse_only
        if num_workers > 1 and not parse_only and len(doc_ids) > 1:
            print(f"[EMR Analyzer] Avvio estrazione parallela: "
                  f"{len(doc_ids)} doc con {num_workers} worker "
                  f"(llm_only={llm_only})")
            self._process_documents_parallel(
                doc_ids, num_workers,
                parse_only=parse_only, llm_only=llm_only,
            )
            return

        if llm_only:
            progress = ProgressDialog("Isolamento testo clinico", self.window())
            progress.set_progress(
                0, f"Normalizzazione LLM di {len(doc_ids)} documento/i..."
            )
            progress.show()
            self._process_next_document(doc_ids, 0, progress, None,
                                        parse_only=False, llm_only=True)
            return

        converter = self._services.get("converter")
        if not converter:
            QMessageBox.warning(self, "Modello non disponibile",
                                "Parser PDF non disponibile.")
            return

        title = (
            "Parsing Documenti" if parse_only else "Estrazione testo clinico"
        )
        progress = ProgressDialog(title, self.window())
        progress.set_progress(
            0, f"Estrazione del testo clinico da {len(doc_ids)} documento/i..."
        )
        progress.show()

        self._process_next_document(doc_ids, 0, progress, converter,
                                    parse_only=parse_only, llm_only=False)

    def _process_documents_parallel(self, doc_ids: list[str],
                                     num_workers: int,
                                     parse_only: bool = False,
                                     llm_only: bool = False):
        """Run LLM isolation on multiple already-parsed documents in parallel.

        For new documents (llm_only=False), falls back to the standard
        sequential pipeline which handles parsing + LLM per document.
        """
        import time
        from concurrent.futures import ThreadPoolExecutor, as_completed
        from .progress_dialog import ProgressDialog
        from PyQt5.QtWidgets import QApplication

        doc_repo = self._services.get("document_repo")
        total = len(doc_ids)

        if not llm_only:
            # New documents need parsing first. Run the standard sequential
            # pipeline — parsing is CPU-bound and tightly coupled to the
            # document record. After parsing completes the user can re-run
            # with llm_only=True to get parallel LLM isolation.
            converter = self._services.get("converter")
            progress = ProgressDialog("Estrazione testo clinico", self.window())
            progress.set_progress(0, f"Analisi sequenziale di {total} doc...")
            progress.show()
            self._process_next_document(
                doc_ids, 0, progress, converter,
                parse_only=False, llm_only=False,
            )
            return

        # ---- Parallel LLM isolation ------------------------------------
        t0 = time.monotonic()
        actual_workers = min(num_workers, total)
        progress = ProgressDialog(
            f"Estrazione parallela ({actual_workers} worker)", self.window()
        )
        progress.set_progress(
            50 if not llm_only else 0,
            f"LLM su {total} documenti con {actual_workers} worker...",
        )
        progress.show()
        QApplication.processEvents()

        completed = 0

        def _llm_isolate_one(doc_id: str):
            doc = doc_repo.get_by_id(doc_id)
            if doc is None:
                return doc_id, "documento non trovato"

            extraction_dir = self._get_extraction_dir()
            source_path = next(
                (p for p in [
                    extraction_dir / f"{doc_id}_source.txt",
                    extraction_dir / f"{doc_id}_cleaned_source.md",
                    extraction_dir / f"{doc_id}_raw.md",
                ] if p.exists()), None
            )
            if source_path is None:
                return doc_id, "testo sorgente non trovato"

            source_text = source_path.read_text(encoding="utf-8")
            try:
                doc_repo.update_extraction_status(doc_id, "processing")
                self._run_llm_extraction(doc, source_text, progress=None)
                doc.extraction_status = ExtractionStatus.DONE.value
                doc.event_count = 0
                doc.error_message = None
                doc_repo.update(doc)
                return doc_id, None
            except Exception as e:
                doc.extraction_status = ExtractionStatus.ERROR.value
                doc.error_message = str(e)
                doc_repo.update(doc)
                return doc_id, str(e)[:200]

        with ThreadPoolExecutor(max_workers=actual_workers) as executor:
            futures = {executor.submit(_llm_isolate_one, did): did
                       for did in doc_ids}
            for future in as_completed(futures):
                completed += 1
                doc_id, error = future.result()
                elapsed = time.monotonic() - t0
                pct = int(50 + (completed / total) * 50) if not llm_only else int((completed / total) * 100)

                if error:
                    progress.add_log(f"❌ {doc_id}: {error}")
                    self._batch_error_count += 1
                else:
                    progress.add_log(f"✓ {doc_id}: testo clinico normalizzato")
                    self._batch_success_count += 1

                progress.set_progress(
                    pct, f"Completati {completed}/{total} ({elapsed:.0f}s)",
                )
                QApplication.processEvents()

        progress.set_progress(
            100,
            f"✓ {total} documenti in {elapsed:.0f}s "
            f"({self._batch_success_count} ok, {self._batch_error_count} errori)",
        )
        self.processing_complete.emit(self._current_patient_id)
        self._refresh_table()

    def _process_next_document(self, doc_ids: list[str], index: int,
                                progress, converter,
                                parse_only: bool = False,
                                llm_only: bool = False):
        """Process one document, then chain to the next."""
        if index >= len(doc_ids):
            progress.set_progress(
                100,
                f"✓ Elaborazione completata: "
                f"{self._batch_success_count} riusciti, "
                f"{self._batch_error_count} falliti",
            )
            progress.add_log(
                f"RIEPILOGO: {self._batch_success_count} riusciti, "
                f"{self._batch_error_count} falliti"
            )
            progress._cancel_btn.setText("Chiudi")
            if hasattr(progress._cancel_btn, 'clicked'):
                try:
                    progress._cancel_btn.clicked.disconnect()
                except TypeError:
                    pass
                progress._cancel_btn.clicked.connect(progress.accept)
            self._refresh_table()
            self.processing_complete.emit(self._current_patient_id)
            return

        doc_id = doc_ids[index]
        doc_repo = self._services.get("document_repo")
        doc = doc_repo.get_by_id(doc_id) if doc_repo else None

        if not doc:
            progress.add_log(f"⚠️ Documento {doc_id} non trovato, skip")
            self._batch_error_count += 1
            self._process_next_document(doc_ids, index + 1, progress, converter,
                                        parse_only, llm_only)
            return

        file_path = doc.original_path
        base_msg = f"[{index + 1}/{len(doc_ids)}] {doc.filename}"

        # ============================================================
        # LLM-ONLY PATH
        # ============================================================
        if llm_only:
            doc_repo.update_extraction_status(doc_id, "processing")
            self._refresh_table()

            # Load the immutable parser source, never a previous LLM output.
            extraction_dir = self._get_extraction_dir()
            parsing_result = None
            extraction_json = extraction_dir / f"{doc_id}.json"
            if extraction_json.exists():
                try:
                    from ..pipeline.pdf_extractor import PdfExtractionResult
                    parsing_result = PdfExtractionResult.from_dict(
                        json.loads(extraction_json.read_text(encoding="utf-8"))
                    )
                except Exception:
                    parsing_result = None

            source_candidates = [
                extraction_dir / f"{doc_id}_source.txt",
                extraction_dir / f"{doc_id}_cleaned_source.md",
                extraction_dir / f"{doc_id}_raw.md",
            ]
            # Backward compatibility for reports parsed before pdfplumber.
            from ..config import active_workspace
            source_candidates.append(
                active_workspace.path / self._current_patient_id / "docling" /
                f"{doc_id}.md"
            )
            source_path = next(
                (path for path in source_candidates if path.exists()), None
            )
            if source_path is not None:
                source_text = source_path.read_text(encoding="utf-8")
            elif parsing_result is not None:
                source_text = parsing_result.plain_text
            else:
                progress.add_log(f"⚠️ {doc.filename}: testo non trovato, fai prima il parsing")
                doc.extraction_status = ExtractionStatus.ERROR.value
                doc.error_message = "Testo sorgente non trovato"
                doc_repo.update(doc)
                self._batch_error_count += 1
                self._process_next_document(doc_ids, index + 1, progress, converter,
                                            parse_only, llm_only)
                return

            # Run LLM extraction with granular progress
            pct_base = int((index / len(doc_ids)) * 100)
            progress.set_progress(pct_base, f"{base_msg} — estrazione LLM...")
            from PyQt5.QtWidgets import QApplication
            QApplication.processEvents()

            try:
                progress.set_progress(pct_base + 2, f"{base_msg} — chiamata Ollama...")
                QApplication.processEvents()

                self._run_llm_extraction(
                    doc, source_text, progress, parsing_result=parsing_result
                )
                progress.set_progress(
                    min(99, pct_base + 8),
                    f"{base_msg} — salvataggio testo clinico...",
                )
                QApplication.processEvents()

                doc.extraction_status = ExtractionStatus.DONE.value
                doc.event_count = 0
                doc.error_message = None
                doc_repo.update(doc)
                self._consecutive_llm_errors = 0
                self._batch_success_count += 1
                progress.add_log(f"✓ {doc.filename}: testo clinico normalizzato")
            except Exception as e:
                progress.add_log(f"⚠️ {doc.filename}: errore LLM — {e}")
                doc.extraction_status = ExtractionStatus.ERROR.value
                doc.error_message = str(e)
                doc_repo.update(doc)
                if (
                    isinstance(e, ClinicalTextIsolationError)
                    and getattr(e, "systemic", False)
                ):
                    self._consecutive_llm_errors += 1
                else:
                    self._consecutive_llm_errors = 0
                self._batch_error_count += 1

            if self._consecutive_llm_errors >= 3:
                self._abort_llm_batch(
                    progress, remaining=len(doc_ids) - index - 1
                )
                return

            self._refresh_table()
            from PyQt5.QtWidgets import QApplication
            QApplication.processEvents()
            self._process_next_document(doc_ids, index + 1, progress, converter,
                                        parse_only, llm_only)
            return

        # ============================================================
        # PARSE PATH (with or without LLM follow-up)
        # ============================================================
        doc_repo.update_parsing_status(doc_id, ParsingStatus.PROCESSING.value)
        self._refresh_table()

        # ---- Step 1: deterministic PDF extraction ----
        progress.set_progress(
            self._batch_stage_progress(
                index, len(doc_ids), phase_fraction=0.0
            ),
            f"{base_msg} — Estrazione testo nativo..."
        )

        try:
            active_parser = converter
            try:
                result = active_parser.convert(file_path)
                markdown_text = active_parser.export_markdown(result)
                plain_text = active_parser.export_text(result)
                tables = active_parser.export_tables(result)
                page_count = active_parser.get_page_count(result)
                if not plain_text.strip():
                    raise ValueError("Nessun testo recuperato dal parser primario")
            except Exception as primary_error:
                fallback = self._services.get("parser_fallback")
                if fallback is None:
                    raise
                progress.add_log(
                    f"⚠️ Estrazione standard non sufficiente ({primary_error}); "
                    "utilizzo fallback Docling standard"
                )
                active_parser = fallback
                result = active_parser.convert(file_path)
                markdown_text = active_parser.export_markdown(result)
                plain_text = active_parser.export_text(result)
                tables = active_parser.export_tables(result)
                page_count = active_parser.get_page_count(result)

            parser_name = getattr(result, "method", "docling_fallback")
            progress.add_log(
                f"✓ {doc.filename}: estrazione {parser_name} completata "
                f"({page_count} pagine, {len(tables)} tabelle)"
            )

            # Extract document date from header
            doc_date = self._extract_document_date(markdown_text)
            if doc_date:
                doc.document_date = doc_date
                progress.add_log(f"  📅 Data referto: {doc_date}")

            # Persist immutable parser layers separately from the active text.
            extraction_dir = self._get_extraction_dir()
            extraction_dir.mkdir(parents=True, exist_ok=True)

            raw_path = extraction_dir / f"{doc_id}_raw.md"
            raw_path.write_text(markdown_text, encoding="utf-8")
            # Exact plain-text input for the document LLM. This remains
            # immutable when the active .md is replaced by normalized text.
            (extraction_dir / f"{doc_id}_source.txt").write_text(
                plain_text, encoding="utf-8"
            )

            extraction_dict = active_parser.export_dict(result)
            json_path = extraction_dir / f"{doc_id}.json"
            json_path.write_text(
                json.dumps(extraction_dict, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            pages = extraction_dict.get("pages", []) if isinstance(extraction_dict, dict) else []
            page_records = []
            word_records = []
            table_records = []
            for page_record in pages:
                page_records.append({
                    key: value for key, value in page_record.items()
                    if key not in {"words", "tables"}
                })
                for word in page_record.get("words", []):
                    word_records.append({"page": page_record.get("page"), **word})
                table_records.extend(page_record.get("tables", []))
            (extraction_dir / f"{doc_id}_pages.jsonl").write_text(
                "\n".join(json.dumps(item, ensure_ascii=False) for item in page_records),
                encoding="utf-8",
            )
            (extraction_dir / f"{doc_id}_words.jsonl").write_text(
                "\n".join(json.dumps(item, ensure_ascii=False) for item in word_records),
                encoding="utf-8",
            )
            (extraction_dir / f"{doc_id}_tables.json").write_text(
                json.dumps(table_records, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            # Extract first-page administrative/clinical metadata before the
            # cleaner removes institutional headers and service descriptions.
            header_extractor = self._services.get("header_metadata_extractor")
            if header_extractor:
                header = header_extractor.extract(markdown_text)
                try:
                    metadata = json.loads(doc.metadata_json or "{}")
                except (TypeError, ValueError):
                    metadata = {}
                metadata["header"] = header.to_dict()
                doc.metadata_json = json.dumps(metadata, ensure_ascii=False)
                if header.department or header.provenance:
                    origin = header.department or header.provenance
                    progress.add_log(f"  🏥 Reparto/provenienza: {origin}")
                if header.services:
                    progress.add_log(
                        f"  📋 Prestazioni erogate: {len(header.services)}"
                    )
                if header.specialty:
                    progress.add_log(f"  🩺 Specialità: {header.specialty}")
            try:
                metadata = json.loads(doc.metadata_json or "{}")
            except (TypeError, ValueError):
                metadata = {}
            metrics = extraction_dict.get("metrics", {}) if isinstance(extraction_dict, dict) else {}
            metadata["parser"] = {
                "name": parser_name,
                "page_count": page_count,
                "table_count": len(tables),
                "metrics": metrics,
            }
            doc.metadata_json = json.dumps(metadata, ensure_ascii=False)

            # Clean the text (remove boilerplate)
            cleaner = self._services.get("cleaner")
            cleaned_text = cleaner.clean(markdown_text) if cleaner else markdown_text

            # Save CLEANED markdown
            (extraction_dir / f"{doc_id}_cleaned_source.md").write_text(
                cleaned_text, encoding="utf-8"
            )
            md_path = extraction_dir / f"{doc_id}.md"
            md_path.write_text(cleaned_text, encoding="utf-8")

        except Exception as e:
            traceback.print_exc()
            progress.add_log(f"❌ {doc.filename}: errore conversione — {e}")
            doc_repo.update_parsing_status(doc_id, ParsingStatus.ERROR.value, str(e))
            self._batch_error_count += 1
            self._process_next_document(
                doc_ids, index + 1, progress, converter,
                parse_only=parse_only, llm_only=llm_only,
            )
            return

        # Update doc metadata
        doc.parsing_status = ParsingStatus.COMPLETED.value
        doc.page_count = page_count
        doc_repo.update(doc)

        # ---- Step 2: Lab + Clinical extraction ----
        progress.set_progress(
            self._batch_stage_progress(
                index, len(doc_ids), phase_fraction=0.55
            ),
            f"{base_msg} — Estrazione..." + (" (solo lab)" if parse_only else "")
        )

        try:
            extraction_result = self._run_extraction(
                doc, plain_text if plain_text else markdown_text,
                result, tables, progress,
                skip_llm=parse_only
            )

            # Parsing-only must leave the document available to the later LLM
            # batch. Previously it was incorrectly marked as fully extracted.
            doc.extraction_status = (
                ExtractionStatus.PENDING.value
                if parse_only else ExtractionStatus.DONE.value
            )
            doc.event_count = len(extraction_result.get("events", []))
            doc.lab_value_count = len(extraction_result.get("lab_values", []))
            doc.error_message = None
            doc_repo.update(doc)
            self._consecutive_llm_errors = 0
            self._batch_success_count += 1

            progress.add_log(
                f"✓ {doc.filename}: testo clinico "
                f"{'creato' if not parse_only else 'in attesa'}, "
                f"{doc.lab_value_count} valori lab"
            )
        except Exception as e:
            progress.add_log(f"⚠️ {doc.filename}: errore estrazione — {e}")
            doc.extraction_status = ExtractionStatus.ERROR.value
            doc.error_message = str(e)
            doc_repo.update(doc)
            self._batch_error_count += 1
            if (
                not parse_only
                and isinstance(e, ClinicalTextIsolationError)
                and getattr(e, "systemic", False)
            ):
                self._consecutive_llm_errors += 1
            else:
                self._consecutive_llm_errors = 0

        if not parse_only and self._consecutive_llm_errors >= 3:
            self._abort_llm_batch(
                progress, remaining=len(doc_ids) - index - 1
            )
            return

        self._refresh_table()
        from PyQt5.QtWidgets import QApplication
        QApplication.processEvents()

        self._process_next_document(doc_ids, index + 1, progress, converter,
                                    parse_only=parse_only)

    @staticmethod
    def _batch_stage_progress(
        index: int, total: int, phase_fraction: float
    ) -> int:
        """Return monotonic batch progress for a phase of one document."""
        safe_total = max(1, total)
        phase = min(0.99, max(0.0, phase_fraction))
        return min(
            99,
            int(5 + ((index + phase) / safe_total) * 90),
        )

    def _abort_llm_batch(self, progress, remaining: int) -> None:
        """Circuit breaker: avoid repeating the same Ollama failure."""
        message = (
            "Elaborazione LLM interrotta dopo 3 errori consecutivi: "
            f"{remaining} documenti non elaborati"
        )
        progress.add_log(f"❌ {message}")
        progress.set_progress(100, message)
        progress._cancel_btn.setText("Chiudi")
        try:
            progress._cancel_btn.clicked.disconnect()
        except TypeError:
            pass
        progress._cancel_btn.clicked.connect(progress.accept)
        self._refresh_table()
        self.processing_complete.emit(self._current_patient_id)

    def _run_extraction(self, doc, text: str, parsing_result,
                        tables: list, progress,
                        skip_llm: bool = False) -> dict:
        """Run extraction: always lab (deterministic), optionally LLM."""
        patient_id = doc.patient_id
        doc_id = doc.id

        cleaner = self._services.get("cleaner")
        cleaned = cleaner.clean(text) if cleaner else text

        classifier = self._services.get("classifier")
        try:
            metadata = json.loads(doc.metadata_json or "{}")
        except (TypeError, ValueError):
            metadata = {}
        header_metadata = metadata.get("header") or {}
        # Classify from the *raw* text so the heading bonus can match
        # "LETTERA DI DIMISSIONE" and similar markers that the cleaner
        # would strip from the clinical body.
        doc_type = (
            classifier.classify(
                text,
                doc.filename,
                header_metadata=header_metadata,
            )
            if classifier else doc.document_type
        )
        if doc_type != doc.document_type and doc_type != "non_classificato":
            doc.document_type = doc_type

        # Only dedicated laboratory reports populate the structured laboratory
        # table. Values quoted in visits, discharge letters or other reports
        # remain available in the normalized clinical text, but must not become
        # longitudinal lab rows.
        lab_parser = self._services.get("lab_parser")
        lab_repo = self._services.get("lab_repo")
        lab_values = []
        if lab_repo:
            # Also clears values produced by older, more permissive versions
            # when a document is reprocessed under the current rule.
            lab_repo.delete_by_document(doc_id)

        if (
            doc.document_type == DocumentType.LABORATORIO.value
            and lab_parser
        ):
            progress.add_log("  ⚗️  Estrazione valori di laboratorio...")
            lab_values = lab_parser.parse(
                cleaned, tables=tables,
                patient_id=patient_id, document_id=doc_id,
                sample_date=doc.document_date,
                parsing_result=parsing_result,
            )
            if lab_repo and lab_values:
                lab_repo.insert_batch(lab_values)
                progress.add_log(f"  ✓ {len(lab_values)} valori lab")
        else:
            progress.add_log(
                "  ⚗️  Valori di laboratorio conservati solo nel testo "
                "clinico (documento non laboratoristico)"
            )

        evidence_repo = self._services.get("evidence_repo")
        if evidence_repo:
            evidence_repo.replace_document_method(
                doc_id,
                "deterministic_lab",
                self._lab_values_to_evidence(
                    doc, lab_values, parsing_result=parsing_result
                ),
            )

        events = []
        # Lab reports have no clinical narrative — only structured values.
        # The ClinicalTextIsolator would hallucinate numbers on a table of
        # parameters and fail validation.
        is_lab = doc.document_type == DocumentType.LABORATORIO.value
        if skip_llm or is_lab:
            reason = "parsing only" if skip_llm else "documento laboratoristico"
            progress.add_log(f"  ⏭️  LLM saltato ({reason})")
            if is_lab and not skip_llm:
                doc_repo = self._services.get("document_repo")
                if doc_repo:
                    doc.extraction_status = ExtractionStatus.DONE.value
                    doc.event_count = 0
                    doc.error_message = None
                    doc_repo.update(doc)
                    self._batch_success_count += 1
        else:
            events = self._run_llm_extraction(
                doc, cleaned, progress, parsing_result=parsing_result
            )

        return {"events": events, "lab_values": lab_values}

    def _run_llm_extraction(self, doc, text: str, progress,
                            parsing_result=None) -> list:
        """Replace the active parser text with normalized clinical prose."""
        doc_id = doc.id
        from PyQt5.QtWidgets import QApplication

        # Parallel LLM workers pass progress=None: the widget is not
        # thread-safe, so swallow the internal log lines there.
        if progress is None:
            progress = _NullProgressLogger()

        isolator = self._services.get("clinical_text_isolator")
        llm_client = self._services.get("document_llm_client")
        if not isolator or not llm_client or not llm_client.is_available:
            raise ClinicalTextIsolationError(
                "Ollama o il modello documentale non sono disponibili"
            )

        progress.add_log(
            f"  🧠 {llm_client.model}: isolamento del testo clinico..."
        )
        progress.add_log(
            "  🔒 Controllo deterministico dei dati sensibili..."
        )
        QApplication.processEvents()
        sensitive_identity = self._sensitive_identity_for_document(
            doc, text, parsing_result
        )
        result = isolator.isolate(
            text,
            document_date=doc.document_date,
            parsing_result=parsing_result,
            sensitive_identity=sensitive_identity,
        )
        for warning in result.warnings:
            progress.add_log(f"  ↻ {warning}")
        redaction_total = sum(result.redaction_counts.values())
        if redaction_total:
            progress.add_log(
                f"  🔒 {redaction_total} dato/i sensibile/i rimosso/i "
                "dal testo clinico"
            )
        self._save_normalized_clinical_text(doc, result)

        # Structured outputs generated by the previous document pipeline are
        # obsolete. Deterministic laboratory evidence remains untouched.
        try:
            self._clear_legacy_document_validation(doc_id)
        except Exception as exc:
            progress.add_log(
                f"  ⚠️ Impossibile ripulire la coda legacy: {exc}"
            )
        evidence_repo = self._services.get("evidence_repo")
        if evidence_repo:
            evidence_repo.replace_document_method(
                doc_id, "llm_document_projection", []
            )
        event_repo = self._services.get("event_repo")
        if event_repo:
            event_repo.delete_by_document(doc_id)
        projection_repo = self._services.get("projection_repo")
        if projection_repo:
            projection_repo.delete_by_document(doc_id)
        legacy_projection = (
            self._get_extraction_dir() /
            f"{doc_id}_clinical_evidence.json"
        )
        if legacy_projection.exists():
            legacy_projection.unlink()

        # A normalized document changes the source on which the longitudinal
        # reconstruction must operate. Invalidate the current state instead of
        # rebuilding it once per document (quadratic work on large dossiers).
        # The dedicated Clinical State phase will rebuild it from all active
        # normalized texts in one pass.
        cs_repo = self._services.get("cs_repo")
        if cs_repo:
            cs_repo.delete(doc.patient_id)

        audit_repo = self._services.get("audit_repo")
        if audit_repo:
            try:
                audit_repo.log(
                    doc.patient_id,
                    "normalize_clinical_text",
                    "document",
                    doc_id,
                    {
                        "prompt_version": result.prompt_version,
                        "chunk_count": result.chunk_count,
                        "character_count": len(result.text),
                        "output_format": "normalized_plain_text",
                        "deidentification_version": (
                            result.deidentification_version
                        ),
                        "redaction_counts": result.redaction_counts,
                    },
                    model_used=result.model_name,
                    model_version=result.prompt_version,
                )
            except Exception as exc:
                progress.add_log(f"  ⚠️ Audit non aggiornato: {exc}")

        progress.add_log(
            f"  ✓ Testo clinico normalizzato: {len(result.text)} caratteri, "
            f"{result.chunk_count} chunk"
        )
        return []

    def _sensitive_identity_for_document(
        self, doc, source_text: str, parsing_result=None
    ) -> dict[str, str]:
        """Read direct identifiers transiently; never persist their values."""
        values = {}
        identity_extractor = self._services.get("identity_extractor")
        if identity_extractor and doc.original_path:
            try:
                evidence = identity_extractor.extract(doc.original_path)
                for field_name in (
                    "name",
                    "fiscal_code",
                    "birth_date",
                    "hospital_patient_id",
                ):
                    field = getattr(evidence, field_name, None)
                    if field and getattr(field, "value", None):
                        values[field_name] = field.value
            except Exception:
                # Generic e-mail, phone, fiscal-code and address rules remain
                # active even if the PDF header cannot be read.
                pass

        if "name" not in values:
            cleaner = self._services.get("cleaner")
            find_name = getattr(cleaner, "_find_patient_name", None)
            if find_name:
                raw_text = (
                    parsing_result.plain_text
                    if parsing_result is not None
                    and getattr(parsing_result, "plain_text", None)
                    else source_text
                )
                try:
                    name = find_name(raw_text, allow_heuristic=False)
                    if name:
                        values["name"] = name
                except Exception:
                    pass
        return values

    def _clear_legacy_document_validation(self, document_id: str) -> None:
        """Remove review items created by the retired document JSON phase."""
        db = self._services.get("db")
        if not db:
            return
        db.execute(
            """DELETE FROM validation_queue
               WHERE (item_type='document_projection' AND item_id IN (
                   SELECT projection_id FROM document_clinical_projections
                   WHERE document_id=?
               ))
                  OR (item_type IN ('event', 'clinical_event') AND item_id IN (
                   SELECT event_id FROM clinical_events
                   WHERE source_document_id=?
               ))""",
            (document_id, document_id),
        )
        db.commit()

    def _save_normalized_clinical_text(self, doc, result) -> None:
        """Atomically replace the active text while preserving source layers."""
        output_dir = self._get_extraction_dir()
        output_dir.mkdir(parents=True, exist_ok=True)
        active_path = output_dir / f"{doc.id}.md"
        temporary_path = output_dir / f"{doc.id}.md.tmp"
        temporary_path.write_text(result.text, encoding="utf-8")
        temporary_path.replace(active_path)

        try:
            metadata = json.loads(doc.metadata_json or "{}")
        except (TypeError, ValueError):
            metadata = {}
        metadata["clinical_text"] = {
            "model": result.model_name,
            "prompt_version": result.prompt_version,
            "chunk_count": result.chunk_count,
            "character_count": len(result.text),
            "output_format": "normalized_plain_text",
            "deidentification_version": result.deidentification_version,
            "redaction_counts": result.redaction_counts,
            "redaction_total": sum(result.redaction_counts.values()),
            "created_at": datetime.now().isoformat(),
        }
        doc.metadata_json = json.dumps(metadata, ensure_ascii=False)

    @staticmethod
    def _lab_values_to_evidence(doc, lab_values: list,
                                parsing_result=None) -> list:
        from ..models.clinical_evidence import ClinicalEvidence

        evidence = []
        for lab in lab_values:
            page = lab.page
            bbox = None
            if parsing_result and hasattr(parsing_result, "locate_source"):
                located_page, bbox = parsing_result.locate_source(
                    lab.source_text, page
                )
                page = located_page or page
            if lab.is_abnormal:
                clinical_status = "abnormal"
            elif lab.reference_low is None and lab.reference_high is None:
                clinical_status = "not_assessable"
            else:
                clinical_status = "within_range"
            evidence.append(ClinicalEvidence(
                patient_id=doc.patient_id,
                document_id=doc.id,
                category="laboratory_finding",
                normalized_entity=lab.normalized_name,
                assertion="observed",
                temporality="current",
                clinical_status=clinical_status,
                observed_date=lab.sample_date or doc.document_date,
                value_text=lab.value_text or str(lab.value) if lab.value is not None else "",
                numeric_value=lab.value,
                unit=lab.unit,
                source_page=page,
                source_text=lab.source_text,
                bbox=bbox,
                confidence=lab.confidence,
                extraction_method="deterministic_lab",
                prompt_version=None,
                schema_version="1.0",
                status="auto",
                data={
                    "reference_low": lab.reference_low,
                    "reference_high": lab.reference_high,
                    "reference_text": lab.reference_text,
                    "flag": lab.flag,
                },
            ))
        return evidence

    def _extract_document_date(self, text: str) -> str | None:
        """
        Extract the document date from the report header.
        Italian medical reports typically have dates like:
        - "06.12.2024" or "29/10/2024" near "REFERTO" or "DATA/ORA ACCETTAZIONE"
        - Excludes patient birth dates (in DATI ANAGRAFICI section)
        """
        import re as re_m
        from ..utils.date_utils import parse_italian_date
        from datetime import datetime

        # Strategy 1: Date immediately after REFERTO heading (most reliable)
        match = re_m.search(
            r'(?:^|\n)##\s*REFERTO\s*\n\s*(\d{1,2}[./\-]\d{1,2}[./\-]\d{2,4})',
            text, re_m.IGNORECASE
        )
        if match:
            iso_date = parse_italian_date(match.group(1))
            if iso_date and self._is_plausible_doc_date(iso_date):
                return iso_date

        # Strategy 2: Date near DATA/ORA ACCETTAZIONE
        match = re_m.search(
            r'(?:DATA/ORA\s+ACCETTAZIONE|DATA\s+ACCETTAZIONE)\s*\n\s*(\d{1,2}[./\-]\d{1,2}[./\-]\d{2,4})',
            text, re_m.IGNORECASE
        )
        if match:
            iso_date = parse_italian_date(match.group(1))
            if iso_date and self._is_plausible_doc_date(iso_date):
                return iso_date

        # Strategy 3: "Referto del GG/MM/AAAA" or "Data referto: GG/MM/AAAA"
        match = re_m.search(
            r'(?i)(?:referto|data\s+referto|data\s+esame)\s+(?:del\s+|delle\s+ore\s+\d{1,2}:\d{2}\s+del\s+)?(\d{1,2}[./\-]\d{1,2}[./\-]\d{2,4})',
            text
        )
        if match:
            iso_date = parse_italian_date(match.group(1))
            if iso_date and self._is_plausible_doc_date(iso_date):
                return iso_date

        # Strategy 4: First plausible date in header (first 30 lines, excluding DATI ANAGRAFICI)
        lines = text.split('\n')
        in_demographics = False
        for i, line in enumerate(lines[:30]):
            if re_m.search(r'DATI\s+ANAGRAFICI', line, re_m.IGNORECASE):
                in_demographics = True
                continue
            if in_demographics and line.strip() and not re_m.match(r'^##\s', line):
                continue  # Skip demographics lines
            if re_m.match(r'^##\s', line) and 'DATI' not in line.upper():
                in_demographics = False

            date_match = re_m.search(
                r'\b(\d{1,2}[./\-]\d{1,2}[./\-]\d{2,4})\b', line
            )
            if date_match and not in_demographics:
                iso_date = parse_italian_date(date_match.group(1))
                if iso_date and self._is_plausible_doc_date(iso_date):
                    return iso_date

        # Strategy 5: Any plausible date anywhere in the document
        all_dates = re_m.findall(
            r'\b(\d{1,2}[./\-]\d{1,2}[./\-]\d{2,4})\b', text
        )
        for date_str in all_dates:
            iso_date = parse_italian_date(date_str)
            if iso_date and self._is_plausible_doc_date(iso_date):
                return iso_date

        return None

    @staticmethod
    def _is_plausible_doc_date(iso_date: str) -> bool:
        """Check if date is plausible as a document date (not a birth date)."""
        from datetime import datetime, timedelta
        try:
            dt = datetime.fromisoformat(iso_date)
            now = datetime.now()
            # Must be between 20 years ago and now
            return (now - timedelta(days=365*20)) <= dt <= now
        except (ValueError, TypeError):
            return False

    def _get_extraction_dir(self) -> 'Path':
        """Get deterministic extraction output directory for the patient."""
        from pathlib import Path
        from ..config import active_workspace
        return active_workspace.path / self._current_patient_id / "extraction"

    def _on_view_pdf(self):
        row = self._table.currentRow()
        if row < 0:
            return
        item = self._table.item(row, 0)
        if item:
            doc_data = item.data(Qt.UserRole + 1)
            self._open_pdf_viewer(doc_data)

    def _open_pdf_viewer(self, doc_data: dict):
        from .pdf_viewer import PDFViewerDialog
        viewer = PDFViewerDialog(doc_data, self._services, self)
        viewer.exec_()

    def _on_context_menu(self, pos):
        row = self._table.currentRow()
        if row < 0:
            return
        item = self._table.item(row, 0)
        if not item:
            return
        doc_id = item.data(Qt.UserRole)
        doc_data = item.data(Qt.UserRole + 1)

        menu = QMenu(self)
        extraction_done = (
            doc_data.get("extraction_status") == ExtractionStatus.DONE.value
        )
        extract_action = menu.addAction(
            "🧠 Riestrai testo clinico" if extraction_done
            else "🧠 Estrai testo clinico"
        )
        open_action = menu.addAction("📖 Apri PDF")
        view_text_action = menu.addAction("📝 Visualizza testo clinico")
        menu.addSeparator()
        edit_action = menu.addAction("✏️ Modifica tipo/metadati")
        menu.addSeparator()
        delete_action = menu.addAction("🗑 Elimina dal workspace")

        action = menu.exec_(self._table.viewport().mapToGlobal(pos))
        if action == extract_action:
            self._process_documents([doc_id])
        elif action == open_action:
            self._open_pdf_viewer(doc_data)
        elif action == view_text_action:
            self._view_extracted_text(doc_id, doc_data)
        elif action == edit_action:
            self._edit_document_metadata(doc_id, doc_data)
        elif action == delete_action:
            self._delete_documents([doc_id])

    def _view_extracted_text(self, doc_id: str, doc_data: dict):
        """Show the extracted markdown text in a dialog."""
        from PyQt5.QtWidgets import QDialog, QVBoxLayout, QTextEdit, QPushButton, QHBoxLayout

        extraction_dir = self._get_extraction_dir()
        md_path = extraction_dir / f"{doc_id}.md"
        if not md_path.exists():
            from ..config import active_workspace
            md_path = (
                active_workspace.path / self._current_patient_id / "docling" /
                f"{doc_id}.md"
            )

        if not md_path.exists():
            QMessageBox.information(self, "Non disponibile",
                                    "Il testo estratto non è ancora disponibile.\n"
                                    "Estrai prima il testo clinico del documento.")
            return

        text = md_path.read_text(encoding="utf-8")

        dialog = QDialog(self)
        dialog.setWindowTitle(
            f"Testo clinico attivo — {doc_data.get('filename', doc_id)}"
        )
        dialog.resize(800, 600)

        dlg_layout = QVBoxLayout(dialog)
        text_edit = QTextEdit()
        text_edit.setReadOnly(True)
        text_edit.setPlainText(text)
        text_edit.setStyleSheet("font-family: Menlo, Monaco, monospace; font-size: 12px;")
        dlg_layout.addWidget(text_edit)

        btn_layout = QHBoxLayout()
        close_btn = QPushButton("Chiudi")
        close_btn.clicked.connect(dialog.accept)
        btn_layout.addStretch()
        btn_layout.addWidget(close_btn)
        dlg_layout.addLayout(btn_layout)

        dialog.exec_()

    def _edit_document_metadata(self, doc_id: str, doc_data: dict):
        from PyQt5.QtWidgets import QInputDialog
        from ..models.document import DocumentType

        new_type, ok = QInputDialog.getItem(
            self, "Modifica tipo documento", "Tipo:",
            [t.value for t in DocumentType],
            editable=False
        )
        if ok:
            doc_repo = self._services["document_repo"]
            doc = doc_repo.get_by_id(doc_id)
            if doc:
                doc.document_type = new_type
                doc_repo.update(doc)
                self._refresh_table()

    def eventFilter(self, obj, event):
        """Handle key press events on the table."""
        from PyQt5.QtCore import QEvent
        from PyQt5.QtWidgets import QApplication
        if obj == self._table and event.type() == QEvent.KeyPress:
            if event.key() == Qt.Key_Delete or event.key() == Qt.Key_Backspace:
                doc_ids = self._get_selected_doc_ids()
                if doc_ids:
                    self._delete_documents(doc_ids)
                return True
        return super().eventFilter(obj, event)

    def _on_delete_selected(self):
        doc_ids = self._get_selected_doc_ids()
        if doc_ids:
            self._delete_documents(doc_ids)

    def _delete_document(self, doc_id: str):
        """Backward-compatible single-document entry point."""
        self._delete_documents([doc_id])

    def _delete_documents(self, doc_ids: list[str]):
        if not doc_ids:
            return
        count = len(doc_ids)
        reply = QMessageBox.question(
            self, "Conferma eliminazione",
            f"Eliminare {count} document{'o' if count == 1 else 'i'} dal workspace?\n"
            "Verranno cancellati PDF, testo estratto, eventi, valori di "
            "laboratorio e dati derivati. Il Clinical State verrà ricostruito.",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )
        if reply != QMessageBox.Yes:
            return

        deletion = self._services.get("document_deletion")
        if deletion is None:
            QMessageBox.critical(
                self, "Eliminazione non disponibile",
                "Il servizio di cancellazione non è inizializzato."
            )
            return

        deleted = 0
        errors = []
        warnings = []
        for doc_id in doc_ids:
            result = deletion.delete(doc_id)
            if result.deleted:
                deleted += 1
            if result.error:
                errors.append(f"{doc_id}: {result.error}")
            warnings.extend(result.warnings)

        self._current_document_id = None
        self._refresh_table()
        if self._current_patient_id:
            self.processing_complete.emit(self._current_patient_id)

        if errors:
            QMessageBox.critical(
                self, "Eliminazione incompleta",
                f"Eliminati {deleted}/{count} documenti.\n\n" + "\n".join(errors[:10])
            )
        elif warnings:
            QMessageBox.warning(
                self, "Eliminazione completata con avvisi",
                f"Eliminati {deleted} documenti.\n\n" + "\n".join(warnings[:10])
            )
        else:
            QMessageBox.information(
                self, "Eliminazione completata",
                f"Eliminati correttamente {deleted} documenti."
            )

    def reprocess_document(self, doc_id: str):
        """Re-run the complete clinical-text pipeline for one document."""
        self._process_documents([doc_id])

"""Run any clinical prompt over one patient or a selected cohort."""
from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import uuid

from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QListWidget, QListWidgetItem,
    QPlainTextEdit, QPushButton, QTextBrowser,
)

from ..clinical.dossier_query import (
    DossierQueryService, QueryCancelled, render_report, summarize_cohort,
)


def _write(path, text):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


class DossierQueryWorker(QThread):
    progress = pyqtSignal(str)
    result_ready = pyqtSignal(str)

    def __init__(self, service, patients, question, output_dir, parent=None):
        super().__init__(parent)
        self.service, self.patients, self.question = service, patients, question
        self.output_dir = Path(output_dir)

    def cancel(self):
        self.requestInterruption()

    def run(self):
        reports, errors = [], []
        cancelled = False
        try:
            self.output_dir.mkdir(parents=True, exist_ok=False)
            self._save_manifest(reports, errors, "running")
            for patient_id in self.patients:
                if self.isInterruptionRequested():
                    cancelled = True
                    break
                try:
                    report = self.service.query(
                        patient_id, self.question, cancel_check=self.isInterruptionRequested,
                        progress=self.progress.emit,
                    )
                    _write(self.output_dir / f"{patient_id}.json",
                           json.dumps(report, ensure_ascii=False, indent=2))
                    _write(self.output_dir / f"{patient_id}.md", render_report(report))
                    reports.append(report)
                except QueryCancelled:
                    cancelled = True
                    break
                except Exception as exc:
                    errors.append({"patient_id": patient_id, "error": str(exc)})
                self._save_manifest(reports, errors, "running")
            completed = sum(r["coverage_complete"] for r in reports)
            summary = "Sintesi di coorte non richiesta per analisi interrotta."
            if reports and not cancelled:
                self.progress.emit("Sintesi dei risultati della coorte...")
                try:
                    summary = summarize_cohort(reports, self.service.llm, self.question,
                                               check_cancel=self._check_cancel)
                except QueryCancelled:
                    cancelled = True
                    summary = "Sintesi interrotta; report individuali conservati."
                except Exception as exc:
                    summary = f"Sintesi non disponibile: {exc}. Consultare i report individuali."
            state = "cancelled" if cancelled else "finished"
            text = (
                f"# Report di coorte\n\nDomanda: {self.question}\n\n"
                f"Pazienti selezionati: {len(self.patients)}; report: {len(reports)}; "
                f"copertura completa: {completed}; errori: {len(errors)}. Stato: {state}.\n\n"
                "I conteggi descrivono l'elaborazione, non prevalenze cliniche. "
                "Ogni paziente è stato interrogato separatamente con lo stesso prompt.\n\n"
                + "## Sintesi di coorte\n\n" + summary + "\n\n"
                + "\n\n---\n\n".join(render_report(r) for r in reports)
            )
            if errors:
                text += "\n\nErrori:\n" + json.dumps(errors, ensure_ascii=False, indent=2)
            _write(self.output_dir / "cohort.md", text)
            self._save_manifest(reports, errors, state)
            self.result_ready.emit(text + f"\n\nReport salvati in: {self.output_dir}")
        except Exception as exc:
            self.result_ready.emit(f"Errore report: {exc}\nRisultati già salvati in: {self.output_dir}")
        finally:
            db = getattr(self.service.documents, "db", None)
            if db is not None:
                db.close()

    def _save_manifest(self, reports, errors, state):
        _write(self.output_dir / "manifest.json", json.dumps({
            "question": self.question, "selected_patient_ids": self.patients,
            "completed_patient_ids": [r["patient_id"] for r in reports],
            "errors": errors, "state": state,
        }, ensure_ascii=False, indent=2))

    def _check_cancel(self):
        if self.isInterruptionRequested():
            raise QueryCancelled("Sintesi interrotta")


class DossierQueryDialog(QDialog):
    def __init__(self, services, workspace_root, patient_id=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Interroga referti — paziente o coorte")
        self.resize(900, 750)
        self.services, self.root = services, Path(workspace_root)
        self._worker = None
        layout = QVBoxLayout(self)
        info = QLabel(
            "Applica una domanda clinica a tutti i Markdown clinici attivi (DOC_….md) dei pazienti selezionati. "
            "Ogni referto viene letto per blocchi; l'analisi di dossier grandi può "
            "richiedere tempo. I documenti privi di Markdown clinico sono segnalati; i file grezzi non vengono usati."
        )
        info.setWordWrap(True)
        layout.addWidget(info)
        self.patients = QListWidget()
        for patient in services["patient_repo"].list_all():
            item = QListWidgetItem(patient.id)
            item.setData(Qt.UserRole, patient.id)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked if patient.id == patient_id else Qt.Unchecked)
            self.patients.addItem(item)
        layout.addWidget(self.patients)
        selection = QHBoxLayout()
        for text, state in (("Seleziona tutti", Qt.Checked), ("Deseleziona tutti", Qt.Unchecked)):
            button = QPushButton(text)
            button.clicked.connect(lambda checked=False, value=state: self._select(value))
            selection.addWidget(button)
        layout.addLayout(selection)
        self.question = QPlainTextEdit()
        self.question.setPlaceholderText("Scrivi una domanda sulla storia clinica...")
        self.question.setMaximumHeight(120)
        layout.addWidget(self.question)
        actions = QHBoxLayout()
        self.start = QPushButton("Analizza e salva report")
        self.start.clicked.connect(self._start)
        self.stop = QPushButton("Interrompi dopo la richiesta in corso")
        self.stop.setEnabled(False)
        self.stop.clicked.connect(self._cancel)
        actions.addWidget(self.start)
        actions.addWidget(self.stop)
        layout.addLayout(actions)
        self.status = QLabel("")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self.results = QTextBrowser()
        layout.addWidget(self.results, 1)

    def _select(self, state):
        for index in range(self.patients.count()):
            self.patients.item(index).setCheckState(state)

    def _start(self):
        ids = [self.patients.item(i).data(Qt.UserRole) for i in range(self.patients.count())
               if self.patients.item(i).checkState() == Qt.Checked]
        question = self.question.toPlainText().strip()
        llm = self.services.get("clinical_state_llm_client")
        if not ids or not question or llm is None:
            self.status.setText("Seleziona almeno un paziente, scrivi una domanda e configura il modello di analisi.")
            return
        service = DossierQueryService(self.services["document_repo"], llm, self.root)
        output = self.root / "query_reports" / (datetime.now().strftime("%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:8])
        self._worker = DossierQueryWorker(service, ids, question, output, self)
        self._worker.progress.connect(self.status.setText)
        self._worker.result_ready.connect(self.results.setPlainText)
        self._worker.finished.connect(self._finished)
        self.start.setEnabled(False)
        self.stop.setEnabled(True)
        self.status.setText("Avvio analisi dei referti...")
        self._worker.start()

    def _cancel(self):
        if self._worker:
            self._worker.cancel()
            self.status.setText("Interruzione richiesta; attendo la richiesta LLM in corso.")

    def _finished(self):
        self.start.setEnabled(True)
        self.stop.setEnabled(False)
        self.status.setText("Elaborazione terminata. Consulta copertura ed eventuali errori nel report.")
        self._worker.deleteLater()
        self._worker = None

    def reject(self):
        if self._worker is not None and self._worker.isRunning():
            self._cancel()
            return
        super().reject()

    def closeEvent(self, event):
        if self._worker is not None and self._worker.isRunning():
            self._cancel()
            event.ignore()
        else:
            super().closeEvent(event)

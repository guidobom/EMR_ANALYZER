"""Result dialog for the full-registry irAE analysis."""

from __future__ import annotations

from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QTextBrowser, QPushButton,
    QFileDialog, QMessageBox, QLabel,
)

from ..clinical.irae_layers import render_irae_markdown
from ..config import active_workspace
from ..utils.markdown_tables import render_markdown_to_html
from .irae_patient_tab import IraePatientTab


class IraeResultDialog(QDialog):
    """Show the irAE analysis of a single patient.

    With a structured report dict and the services dict it embeds an
    ``IraePatientTab``, so every finding is selectable, its evidence is
    inspectable (quick view + PDF) and it can be corrected manually with
    persisted per-patient corrections.  Otherwise — a plain Markdown string
    or a report without the services — it falls back to the legacy read-only
    text view.
    """

    def __init__(self, report, patient_id: str = "", services=None,
                 parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Analisi irAE — {patient_id}")
        self.resize(1100, 760)
        self._patient_id = patient_id
        self._services = services
        self._markdown = ""
        self._structured_report = report if isinstance(report, dict) else None
        self._tab = None
        self._recon_worker = None

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(
            "<b>Eventi avversi immuno-correlati — registro completo</b>"
        ))

        if isinstance(report, dict) and services:
            self._tab = IraePatientTab(
                report, patient_id, services, self
            )
            self._tab.report_updated.connect(self._on_report_updated)
            self._markdown = self._tab.markdown()
            layout.addWidget(self._tab, stretch=1)
        else:
            self._markdown = (
                report
                if isinstance(report, str)
                else render_irae_markdown(report or {})
            )
            view = QTextBrowser()
            view.setOpenExternalLinks(False)
            view.setHtml(render_markdown_to_html(self._markdown))
            layout.addWidget(view, stretch=1)

        buttons = QHBoxLayout()
        self._recon_btn = QPushButton("🔄 Rianalizza consolidamento")
        self._recon_btn.setToolTip(
            "Riesegue il solo consolidamento finale (Layer 4) con un budget di "
            "token più alto. Usato per i pazienti in cui era fallito o per "
            "riapplicare il prompt corrente ai report già consolidati."
        )
        self._recon_btn.clicked.connect(self._on_reconsolidate)
        buttons.addWidget(self._recon_btn)
        save_btn = QPushButton("💾 Salva come Markdown")
        save_btn.clicked.connect(self._save)
        buttons.addWidget(save_btn)
        excel_btn = QPushButton("📊 Esporta Excel")
        excel_btn.clicked.connect(self._export_excel)
        buttons.addWidget(excel_btn)
        buttons.addStretch()
        close_btn = QPushButton("Chiudi")
        close_btn.clicked.connect(self.accept)
        buttons.addWidget(close_btn)
        layout.addLayout(buttons)
        self._recon_status = QLabel("")
        layout.addWidget(self._recon_status)

        self._recon_btn.setEnabled(self._can_reconsolidate())

    def _on_report_updated(self, report: dict) -> None:
        self._markdown = self._tab.markdown() if self._tab else self._markdown

    def _can_reconsolidate(self) -> bool:
        """Button enabled: per-organ results, an available clinical-state LLM
        and no run already in flight.  Re-runs are allowed also on reports
        whose consolidation succeeded (to re-apply the current Layer 4
        prompt); the confirm dialog warns before overwriting."""
        if self._recon_worker is not None and self._recon_worker.isRunning():
            return False
        report = self._structured_report
        if not isinstance(report, dict) or not report.get("organ_results"):
            return False
        llm = (self._services or {}).get("clinical_state_llm_client")
        return bool(llm is not None and getattr(llm, "is_available", False))

    def _on_reconsolidate(self) -> None:
        if not self._can_reconsolidate():
            return
        consolidation = (
            (self._structured_report or {}).get("consolidation") or {}
        )
        if consolidation.get("applied"):
            text = (
                "Il consolidamento di questo paziente è già riuscito. "
                "Verrà rieseguito con il prompt corrente del Layer 4, "
                "sovrascrivendo i risultati attuali. Continuare?"
            )
        else:
            text = (
                "Rieseguirà SOLO il consolidamento finale (Layer 4) di questo "
                "paziente con un budget di token più alto, sovrascrivendo i "
                "risultati parziali. Continuare?"
            )
        answer = QMessageBox.question(
            self,
            "Rianalizza consolidamento",
            text,
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if answer != QMessageBox.Yes:
            return
        llm = (self._services or {}).get("clinical_state_llm_client")
        from ..clinical.irae_evidence import load_registry_rows
        from ..clinical.irae_reconsolidate import (
            DEFAULT_RECONSOLIDATE_MAX_TOKENS,
        )
        from .workers import IraeReconsolidateWorker

        registry_rows = None
        db_path = active_workspace.path / "emr_registry.db"
        if db_path.exists():
            registry_rows = load_registry_rows(db_path, self._patient_id)

        worker = IraeReconsolidateWorker(
            llm,
            self._structured_report,
            patient_id=self._patient_id,
            max_tokens=DEFAULT_RECONSOLIDATE_MAX_TOKENS,
            registry_rows=registry_rows,
            parent=self,
        )
        worker.ready.connect(self._on_reconsolidate_ready)
        worker.error.connect(self._on_reconsolidate_error)
        worker.finished.connect(self._on_reconsolidate_finished)
        self._recon_worker = worker
        self._recon_btn.setEnabled(False)
        self._recon_status.setText("Rianalisi del consolidamento in corso...")
        worker.start()

    def _on_reconsolidate_ready(self, updated: dict) -> None:
        self._structured_report = updated
        if self._tab is not None:
            self._tab.set_report(updated)
        if updated.get("reconsolidation_error"):
            self._recon_status.setText("")
            QMessageBox.warning(
                self,
                "Riconsolidamento non applicato",
                "Il consolidamento non è andato a buon fine; conservati i "
                "risultati precedenti.\n"
                + str(updated["reconsolidation_error"]),
            )
        else:
            self._recon_status.setText(
                "Riconsolidamento completato (Layer 4 rieseguito con budget "
                "di token più alto)."
            )

    def _on_reconsolidate_error(self, message: str) -> None:
        self._recon_status.setText("")
        QMessageBox.critical(
            self, "Riconsolidamento non riuscito", message
        )

    def _on_reconsolidate_finished(self) -> None:
        self._recon_worker = None
        self._recon_btn.setEnabled(self._can_reconsolidate())

    def closeEvent(self, event) -> None:
        worker = self._recon_worker
        if worker is not None and worker.isRunning():
            worker.wait()
        super().closeEvent(event)

    def _current_markdown(self) -> str:
        if self._tab is not None:
            return self._tab.markdown()
        return self._markdown

    def _current_structured(self) -> dict | None:
        if self._tab is not None:
            return self._tab.current_report()
        return self._structured_report

    def _save(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Salva analisi irAE", "irae_analysis.md",
            "File Markdown (*.md);;File di testo (*.txt)",
        )
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(self._current_markdown())
        except OSError as exc:
            QMessageBox.warning(
                self, "Salvataggio non riuscito", str(exc)
            )

    def _export_excel(self):
        """Export the (corrected) findings to a single .xlsx workbook."""
        path, _ = QFileDialog.getSaveFileName(
            self, "Esporta risultati irAE",
            "irae_risultati.xlsx", "Excel (*.xlsx)",
        )
        if not path:
            return
        if not path.lower().endswith(".xlsx"):
            path += ".xlsx"

        from ..clinical.irae_export import write_irae_xlsx

        structured = self._current_structured()
        results = [{
            "patient_id": self._patient_id,
            "label": self._patient_id,
            "markdown": self._current_markdown(),
            "structured": structured,
            "error": None,
        }]
        try:
            written, skipped = write_irae_xlsx(results, path)
        except Exception as exc:
            QMessageBox.critical(
                self, "Esportazione fallita",
                f"Impossibile scrivere il file Excel:\n{exc}",
            )
            return
        message = f"Righe esportate: {written}\nFile: {path}"
        if skipped:
            message += (
                "\nPazienti senza dati strutturati (saltati): "
                + ", ".join(skipped)
            )
        QMessageBox.information(self, "Esportazione completata", message)

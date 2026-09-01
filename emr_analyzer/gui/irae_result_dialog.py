"""Result dialog for the full-registry irAE analysis."""

from __future__ import annotations

from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QTextBrowser, QPushButton,
    QFileDialog, QMessageBox, QLabel,
)

from ..clinical.irae_layers import render_irae_markdown
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
        self._markdown = ""
        self._structured_report = report if isinstance(report, dict) else None
        self._tab = None

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

    def _on_report_updated(self, report: dict) -> None:
        self._markdown = self._tab.markdown() if self._tab else self._markdown

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

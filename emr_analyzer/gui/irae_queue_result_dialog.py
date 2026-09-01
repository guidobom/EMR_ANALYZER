"""Summary dialog of a multi-patient irAE analysis queue."""

from __future__ import annotations

from pathlib import Path

from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QTextBrowser, QPushButton,
    QFileDialog, QMessageBox, QLabel, QTabWidget,
)

from ..clinical.irae_layers import render_irae_markdown
from ..utils.markdown_tables import render_markdown_to_html


class IraeQueueResultDialog(QDialog):
    """One tab per patient, with a bulk save of the Markdown reports.

    A result carrying a structured report dict (and with ``services``
    provided) gets an ``IraePatientTab``: its findings are selectable, their
    evidence is inspectable and each one can be corrected manually, with the
    persisted corrections reflected in the bulk save and the Excel export.
    Results without structured data keep the legacy read-only ``QTextBrowser``
    tab.
    """

    def __init__(self, results: list[dict], parent=None, *, services=None):
        """*results*: list of ``{patient_id, label, markdown, error,
        structured}``."""
        super().__init__(parent)
        self.setWindowTitle("Analisi irAE multi-paziente — risultati")
        self.resize(1100, 760)
        self._results = results
        self._services = services

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(
            f"<b>{len(results)} paziente/i analizzato/i</b> — una scheda "
            "per paziente."
        ))

        self._tabs = QTabWidget()
        for index, result in enumerate(results):
            view = self._build_tab(result, index)
            self._tabs.addTab(view, result.get("label") or result["patient_id"])
        layout.addWidget(self._tabs, stretch=1)

        buttons = QHBoxLayout()
        save_btn = QPushButton("💾 Salva tutto")
        save_btn.clicked.connect(self._save_all)
        buttons.addWidget(save_btn)
        excel_btn = QPushButton("📊 Esporta Excel")
        excel_btn.clicked.connect(self._export_excel)
        buttons.addWidget(excel_btn)
        buttons.addStretch()
        close_btn = QPushButton("Chiudi")
        close_btn.clicked.connect(self.accept)
        buttons.addWidget(close_btn)
        layout.addLayout(buttons)

    def _build_tab(self, result: dict, index: int):
        """The tab widget for one result: inspectable when structured."""
        structured = result.get("structured")
        if (
            isinstance(structured, dict)
            and not result.get("error")
            and self._services is not None
        ):
            from ..clinical.irae_corrections import (
                apply_irae_corrections,
                load_corrections,
            )
            from .irae_patient_tab import IraePatientTab

            patient_id = result["patient_id"]
            # The tab owns the raw report and re-applies the persisted
            # corrections; keep the result entry coherent so the bulk save
            # and the Excel export always read the corrected report.
            corrected, _ = apply_irae_corrections(
                structured, load_corrections(patient_id)
            )
            result["structured"] = corrected
            result["markdown"] = render_irae_markdown(corrected)
            tab = IraePatientTab(
                structured, patient_id, self._services, parent=self
            )
            tab.report_updated.connect(
                lambda report, idx=index, res=result: self._on_tab_updated(
                    idx, res, report
                )
            )
            return tab

        view = QTextBrowser()
        view.setOpenExternalLinks(False)
        if result.get("error"):
            view.setHtml(
                f"<span style='color:#c0392b;'>⚠️ "
                f"{result['error']}</span>"
            )
        else:
            view.setHtml(render_markdown_to_html(
                result.get("markdown") or ""
            ))
        return view

    def _on_tab_updated(self, index: int, result: dict, report: dict) -> None:
        """A finding was corrected inside the tab: refresh the result entry."""
        result["structured"] = report
        result["markdown"] = render_irae_markdown(report)

    def _export_excel(self):
        """Export the structured findings to a single .xlsx workbook."""
        path, _ = QFileDialog.getSaveFileName(
            self, "Esporta risultati irAE",
            "irae_risultati.xlsx", "Excel (*.xlsx)",
        )
        if not path:
            return
        if not path.lower().endswith(".xlsx"):
            path += ".xlsx"

        from ..clinical.irae_export import write_irae_xlsx

        try:
            written, skipped = write_irae_xlsx(self._results, path)
        except Exception as exc:
            QMessageBox.critical(
                self, "Esportazione fallita",
                f"Impossibile scrivere il file Excel:\n{exc}",
            )
            return
        message = (
            f"Righe esportate: {written}\n"
            f"File: {path}"
        )
        if skipped:
            message += (
                "\nPazienti senza dati strutturati (saltati): "
                + ", ".join(skipped)
            )
        QMessageBox.information(self, "Esportazione completata", message)

    def _save_all(self):
        directory = QFileDialog.getExistingDirectory(
            self, "Cartella per i report irAE"
        )
        if not directory:
            return
        written = []
        skipped = []
        for result in self._results:
            if result.get("error"):
                skipped.append(result["patient_id"])
                continue
            path = Path(directory) / f"{result['patient_id']}_irae.md"
            try:
                path.write_text(
                    result.get("markdown") or "", encoding="utf-8"
                )
                written.append(str(path))
            except OSError:
                skipped.append(result["patient_id"])
        message = f"Report scritti: {len(written)}"
        if skipped:
            message += f"\nSaltati (errore o scrittura fallita): {', '.join(skipped)}"
        QMessageBox.information(self, "Salvataggio completato", message)

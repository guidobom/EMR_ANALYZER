"""Summary dialog of a multi-patient irAE analysis queue."""

from __future__ import annotations

from pathlib import Path

from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QTextBrowser, QPushButton,
    QFileDialog, QMessageBox, QLabel, QTabWidget,
)

from ..utils.markdown_tables import render_markdown_to_html


class IraeQueueResultDialog(QDialog):
    """One tab per patient, with a bulk save of the Markdown reports."""

    def __init__(self, results: list[dict], parent=None):
        """*results*: list of ``{patient_id, label, markdown, error}``."""
        super().__init__(parent)
        self.setWindowTitle("Analisi irAE multi-paziente — risultati")
        self.resize(1100, 760)
        self._results = results

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(
            f"<b>{len(results)} paziente/i analizzato/i</b> — una scheda "
            "per paziente."
        ))

        self._tabs = QTabWidget()
        for result in results:
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
            self._tabs.addTab(view, result.get("label") or result["patient_id"])
        layout.addWidget(self._tabs, stretch=1)

        buttons = QHBoxLayout()
        save_btn = QPushButton("💾 Salva tutto")
        save_btn.clicked.connect(self._save_all)
        buttons.addWidget(save_btn)
        buttons.addStretch()
        close_btn = QPushButton("Chiudi")
        close_btn.clicked.connect(self.accept)
        buttons.addWidget(close_btn)
        layout.addLayout(buttons)

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

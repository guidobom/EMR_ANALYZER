"""Result dialog for the full-registry irAE analysis."""

from __future__ import annotations

from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QTextBrowser, QPushButton,
    QFileDialog, QMessageBox, QLabel,
)

from ..utils.markdown_tables import render_markdown_to_html


class IraeResultDialog(QDialog):
    """Show the combined Markdown tables of the irAE analysis.

    Tables render as HTML; the raw Markdown can be saved to a file.
    """

    def __init__(self, markdown_text: str, patient_id: str = "",
                 parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Analisi irAE — {patient_id}")
        self.resize(1100, 760)
        self._markdown = markdown_text

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(
            "<b>Eventi avversi immuno-correlati — registro completo</b>"
        ))

        self._view = QTextBrowser()
        self._view.setOpenExternalLinks(False)
        self._view.setHtml(render_markdown_to_html(markdown_text))
        layout.addWidget(self._view, stretch=1)

        buttons = QHBoxLayout()
        save_btn = QPushButton("💾 Salva come Markdown")
        save_btn.clicked.connect(self._save)
        buttons.addWidget(save_btn)
        buttons.addStretch()
        close_btn = QPushButton("Chiudi")
        close_btn.clicked.connect(self.accept)
        buttons.addWidget(close_btn)
        layout.addLayout(buttons)

    def _save(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Salva analisi irAE", "irae_analysis.md",
            "File Markdown (*.md);;File di testo (*.txt)",
        )
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(self._markdown)
        except OSError as exc:
            QMessageBox.warning(
                self, "Salvataggio non riuscito", str(exc)
            )

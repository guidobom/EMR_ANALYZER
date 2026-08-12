"""Progress dialog for long-running processing tasks."""

from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QProgressBar, QLabel, QPushButton,
    QTextEdit, QHBoxLayout,
)
from PyQt5.QtCore import Qt, pyqtSignal


class ProgressDialog(QDialog):
    """Modal dialog showing processing progress."""

    cancelled = pyqtSignal()

    def __init__(self, title: str = "Elaborazione in corso", parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setMinimumWidth(500)
        self.setModal(True)
        self._cancelled = False
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        # Overall progress
        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 100)
        layout.addWidget(self._progress_bar)

        # Status label
        self._status_label = QLabel("Inizializzazione...")
        layout.addWidget(self._status_label)

        # Detailed log
        self._log_text = QTextEdit()
        self._log_text.setReadOnly(True)
        self._log_text.setMaximumHeight(200)
        layout.addWidget(self._log_text)

        # Cancel button
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        self._cancel_btn = QPushButton("Annulla")
        self._cancel_btn.clicked.connect(self._on_cancel)
        btn_layout.addWidget(self._cancel_btn)
        layout.addLayout(btn_layout)

    def set_progress(self, percent: int, message: str):
        """Update progress bar and status message."""
        self._progress_bar.setValue(percent)
        self._status_label.setText(message)
        self._log_text.append(f"[{percent:3d}%] {message}")

    def add_log(self, message: str):
        """Add a line to the log."""
        self._log_text.append(message)

    def _on_cancel(self):
        self._cancelled = True
        self._cancel_btn.setEnabled(False)
        self._status_label.setText("Annullamento richiesto...")
        self.cancelled.emit()

    def reset_for_reuse(self):
        """Prepare the dialog for another phase/patient: re-enable Cancel."""
        self._cancelled = False
        self._cancel_btn.setText("Annulla")
        self._cancel_btn.setEnabled(True)
        try:
            self._cancel_btn.clicked.disconnect()
        except TypeError:
            pass
        self._cancel_btn.clicked.connect(self._on_cancel)

    def mark_done(self):
        """Turn the Cancel button into a Close button at the end."""
        self._cancel_btn.setText("Chiudi")
        try:
            self._cancel_btn.clicked.disconnect()
        except TypeError:
            pass
        self._cancel_btn.clicked.connect(self.accept)

    def is_cancelled(self) -> bool:
        return self._cancelled

    def closeEvent(self, event):
        if not self._cancelled:
            self._on_cancel()
        event.accept()

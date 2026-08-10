"""PDF Viewer dialog using PyMuPDF."""

import os

import fitz  # PyMuPDF

from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QScrollArea, QSpinBox, QWidget,
)
from PyQt5.QtCore import Qt, QRectF
from PyQt5.QtGui import QPixmap, QImage, QPainter, QColor, QPen


class PDFViewerDialog(QDialog):
    """Dialog for viewing a PDF document page by page."""

    def __init__(self, doc_data: dict, services: dict, parent=None,
                 highlight: dict = None):
        super().__init__(parent)
        self.setWindowTitle(f"PDF Viewer — {doc_data.get('filename', '')}")
        self.resize(900, 800)
        self._doc_data = doc_data
        self._services = services
        self._current_page = 0
        self._zoom = 1.5
        self._doc = None
        self._highlight = highlight
        self._setup_ui()
        self._load_pdf()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        # Toolbar
        toolbar = QHBoxLayout()

        self._prev_btn = QPushButton("◀")
        self._prev_btn.clicked.connect(self._prev_page)
        toolbar.addWidget(self._prev_btn)

        self._page_spin = QSpinBox()
        self._page_spin.setMinimum(1)
        self._page_spin.valueChanged.connect(self._on_page_changed)
        toolbar.addWidget(self._page_spin)

        self._page_count_label = QLabel("/ —")
        toolbar.addWidget(self._page_count_label)

        self._next_btn = QPushButton("▶")
        self._next_btn.clicked.connect(self._next_page)
        toolbar.addWidget(self._next_btn)

        toolbar.addStretch()

        zoom_out_btn = QPushButton("−")
        zoom_out_btn.clicked.connect(lambda: self._set_zoom(self._zoom - 0.25))
        toolbar.addWidget(zoom_out_btn)

        self._zoom_label = QLabel(f"{self._zoom:.0f}x")
        toolbar.addWidget(self._zoom_label)

        zoom_in_btn = QPushButton("+")
        zoom_in_btn.clicked.connect(lambda: self._set_zoom(self._zoom + 0.25))
        toolbar.addWidget(zoom_in_btn)

        layout.addLayout(toolbar)

        # Page display
        self._scroll_area = QScrollArea()
        self._scroll_area.setWidgetResizable(True)
        self._scroll_area.setAlignment(Qt.AlignCenter)

        self._page_label = QLabel()
        self._page_label.setAlignment(Qt.AlignCenter)
        self._scroll_area.setWidget(self._page_label)

        layout.addWidget(self._scroll_area, stretch=1)

    def _load_pdf(self):
        """Load the PDF document."""
        path = self._doc_data.get("original_path") or self._doc_data.get("stored_path", "")
        if path and not os.path.isabs(path):
            workspace_root = self._doc_data.get("workspace_root")
            if workspace_root:
                path = os.path.join(workspace_root, path)
        if not path:
            self._page_label.setText("❌ File non trovato")
            return

        try:
            if not os.path.exists(path):
                self._page_label.setText(f"❌ File non trovato: {path}")
                return

            self._doc = fitz.open(path)
            self._page_spin.setMaximum(max(1, self._doc.page_count))
            self._page_count_label.setText(f"/ {self._doc.page_count}")
            self._show_page(0)
        except Exception as e:
            self._page_label.setText(f"❌ Errore apertura PDF: {e}")

    def _show_page(self, page_num: int):
        """Render and display a specific page."""
        if not self._doc or page_num >= self._doc.page_count:
            return

        self._current_page = page_num
        self._page_spin.blockSignals(True)
        self._page_spin.setValue(page_num + 1)
        self._page_spin.blockSignals(False)

        page = self._doc[page_num]
        mat = fitz.Matrix(self._zoom, self._zoom)
        pix = page.get_pixmap(matrix=mat)

        # Convert to QImage and then QPixmap
        img = QImage(pix.samples, pix.width, pix.height,
                     pix.stride, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(img)

        # v2 coordinates use PDF points with a top-left origin, the same
        # coordinate system returned by PyMuPDF. Rendering only requires the
        # current zoom multiplier.
        if self._highlight and self._highlight.get("page_number") == page_num + 1:
            bbox = self._highlight.get("bbox") or []
            if len(bbox) == 4:
                x0, y0, x1, y1 = (float(value) * self._zoom for value in bbox)
                painter = QPainter(pixmap)
                painter.setPen(QPen(QColor(220, 40, 40), 3))
                painter.fillRect(QRectF(x0, y0, x1 - x0, y1 - y0), QColor(255, 235, 59, 90))
                painter.end()

        self._page_label.setPixmap(pixmap)
        self._page_label.setMinimumSize(pixmap.size())

        # Update navigation buttons
        self._prev_btn.setEnabled(page_num > 0)
        self._next_btn.setEnabled(page_num < self._doc.page_count - 1)

    def _prev_page(self):
        if self._current_page > 0:
            self._show_page(self._current_page - 1)

    def _next_page(self):
        if self._doc and self._current_page < self._doc.page_count - 1:
            self._show_page(self._current_page + 1)

    def _on_page_changed(self, page_num: int):
        if self._doc and 1 <= page_num <= self._doc.page_count:
            self._show_page(page_num - 1)

    def _set_zoom(self, zoom: float):
        self._zoom = max(0.5, min(4.0, zoom))
        self._zoom_label.setText(f"{self._zoom:.1f}x")
        if self._doc:
            self._show_page(self._current_page)

    def focus_evidence(self, page_number: int, bbox: list[float]):
        """Navigate to and highlight one evidence source."""
        self._highlight = {"page_number": page_number, "bbox": bbox}
        if self._doc and 1 <= page_number <= self._doc.page_count:
            self._show_page(page_number - 1)

    def closeEvent(self, event):
        if self._doc:
            self._doc.close()
        event.accept()

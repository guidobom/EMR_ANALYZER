"""macOS-style Quick Look preview for in-app PDF inspection.

A lightweight, non-modal preview that renders one PDF page at a time, whole
page fitted, in a frameless window centered on the monitor.  The hosting list
keeps keyboard focus, so Space toggles the preview on and off without ever
opening the file in an external viewer; Esc or a click closes it, the wheel
turns the pages, and moving to another row in the list switches the preview
to that document.  Double-click and the explicit "Apri PDF" actions keep
opening the full modal :class:`PDFViewerDialog`.
"""

from __future__ import annotations

import os

import fitz  # PyMuPDF

from PyQt5.QtCore import Qt, QEvent, QTimer
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QScrollArea, QPushButton,
    QApplication,
)


class QuickLook(QWidget):
    """Space-toggled, non-modal, screen-centered PDF preview.

    *anchor* is the file-list widget (``QTableWidget`` / ``QTreeWidget``)
    that owns the Space shortcut.  *resolve_path* is a zero-argument callable
    returning ``(path, filename)`` for the current selection, or ``None`` when
    the selection is not a file row.

    The preview is a frameless tool window centered on the monitor.  It never
    takes focus, so the list keeps receiving the keys: Space/Esc close it,
    arrows move the list selection, and the selection-change hook swaps the
    preview to the newly selected document.
    """

    def __init__(self, anchor, resolve_path, parent=None):
        super().__init__(
            parent or anchor.window(),
            Qt.FramelessWindowHint | Qt.Tool | Qt.WindowStaysOnTopHint,
        )
        self._anchor = anchor
        self._resolve_path = resolve_path
        self._doc = None
        self._path = None
        self._page = 0
        self._open = False  # tracks state independently of widget visibility

        # Never activate: the hosting list must keep keyboard focus while the
        # preview is on screen.
        self.setAttribute(Qt.WA_ShowWithoutActivating)
        self.setFocusPolicy(Qt.NoFocus)
        self._setup_ui()
        self.hide()

        # Moving to another row in the list swaps the preview to that file.
        # Tables hook ``currentCellChanged``: ``currentItemChanged`` only
        # fires when the current *item* changes, which misses moves across
        # empty cells (e.g. the checkbox column).  Trees have no cell-level
        # signal, so they fall back to ``currentItemChanged``.
        current_cell = getattr(self._anchor, "currentCellChanged", None)
        if current_cell is not None:
            current_cell.connect(self._on_anchor_current_changed)
        else:
            current_item = getattr(self._anchor, "currentItemChanged", None)
            if current_item is not None:
                current_item.connect(self._on_anchor_current_changed)

        # Vanish when the list is hidden (e.g. a tab switch); keep receiving
        # the Space key from the list.
        self._anchor.installEventFilter(self)

    # --- UI ------------------------------------------------------------

    def _setup_ui(self):
        self.setObjectName("quickLook")
        self.setStyleSheet(
            "QWidget#quickLook { background-color: rgba(30, 30, 36, 250); "
            "border: 1px solid #5a5a63; border-radius: 10px; }"
            "QLabel { background: transparent; color: #eee; }"
            "QPushButton { background: transparent; color: #ddd; "
            "border: none; font-size: 15px; padding: 2px 8px; }"
            "QPushButton:hover { color: #fff; background: #44444d; "
            "border-radius: 6px; }"
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 8, 14, 12)
        layout.setSpacing(10)

        header = QHBoxLayout()
        self._title_label = QLabel("")
        self._title_label.setStyleSheet("font-weight: bold; font-size: 13px;")
        header.addWidget(self._title_label, stretch=1)

        self._prev_btn = QPushButton("◀")
        self._prev_btn.setToolTip("Pagina precedente (rotellina in su)")
        self._prev_btn.clicked.connect(self._prev_page)
        header.addWidget(self._prev_btn)

        self._page_counter = QLabel("—")
        self._page_counter.setStyleSheet("color: #bbb; font-size: 12px;")
        header.addWidget(self._page_counter)

        self._next_btn = QPushButton("▶")
        self._next_btn.setToolTip("Pagina successiva (rotellina in giù)")
        self._next_btn.clicked.connect(self._next_page)
        header.addWidget(self._next_btn)

        close_btn = QPushButton("✕")
        close_btn.setToolTip("Chiudi anteprima (Spazio o Esc)")
        close_btn.clicked.connect(self.dismiss)
        header.addWidget(close_btn)
        layout.addLayout(header)

        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(False)
        self._scroll.setAlignment(Qt.AlignCenter)
        self._scroll.setFrameShape(QScrollArea.NoFrame)
        self._scroll.setStyleSheet("QScrollArea { background: #1e1e24; }")
        self._image = QLabel()
        self._image.setAlignment(Qt.AlignCenter)
        self._scroll.setWidget(self._image)
        layout.addWidget(self._scroll, stretch=1)

        footer = QLabel(
            "Spazio / Esc per chiudere · frecce per cambiare documento · "
            "rotellina per la pagina"
        )
        footer.setStyleSheet("color: #999; font-size: 11px;")
        footer.setAlignment(Qt.AlignCenter)
        layout.addWidget(footer)

    # --- geometry ------------------------------------------------------

    def _center_on_screen(self):
        parent_window = self.parentWidget() or self._anchor.window()
        screen = None
        if parent_window is not None:
            screen = parent_window.screen()
        if screen is None:
            screen = QApplication.primaryScreen()
        area = screen.availableGeometry() if screen is not None else None
        if area is None:
            self.resize(1000, 760)
            self.move(
                QApplication.primaryScreen().availableGeometry().center()
                - self.rect().center()
            )
            return
        width = max(360, min(1200, int(area.width() * 0.7)))
        height = max(280, min(900, int(area.height() * 0.85)))
        x = area.x() + (area.width() - width) // 2
        y = area.y() + (area.height() - height) // 2
        self.setGeometry(x, y, width, height)

    def _set_visible(self, visible: bool):
        self._open = visible
        if visible:
            self._center_on_screen()
            self.raise_()
            self.show()
            # Render once the new geometry has been applied.
            QTimer.singleShot(0, self._render_page)
        else:
            self.hide()
        self._anchor.setFocus(Qt.OtherFocusReason)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._open and self._doc:
            QTimer.singleShot(0, self._render_page)

    # --- toggle / lifecycle -------------------------------------------

    def toggle(self) -> bool:
        """Show the preview for the current row, or hide it if already open.

        Returns True when the Space key was consumed (the preview toggled).
        """
        if self._open:
            self.dismiss()
            return True
        target = self._resolve_path()
        if target is None:
            return False
        path, filename = target
        self._load(str(path), str(filename))
        self._set_visible(True)
        return True

    def dismiss(self):
        """Hide the overlay and release the open PDF document."""
        if self._doc:
            self._doc.close()
            self._doc = None
        self._path = None
        self._set_visible(False)

    # --- rendering -----------------------------------------------------

    def _load(self, path: str, filename: str):
        if self._doc:
            self._doc.close()
            self._doc = None
        self._title_label.setText(filename)
        self._path = path
        try:
            if not os.path.exists(path):
                self._show_error("File non trovato")
                return
            self._doc = fitz.open(path)
            self._page = 0
            self._render_page()
        except Exception as exc:
            self._show_error(f"Errore apertura PDF: {exc}")

    def _show_error(self, message: str):
        self._image.clear()
        self._image.setText(message)
        self._page_counter.setText("—")

    def _render_page(self):
        if not self._doc or self._doc.page_count == 0:
            return
        page = self._doc[min(self._page, self._doc.page_count - 1)]
        viewport = self._scroll.viewport()
        available_w = max(120, viewport.width() - 8)
        available_h = max(120, viewport.height() - 8)
        zoom = min(
            available_w / page.rect.width,
            available_h / page.rect.height,
            5.0,  # never blow a small page up to enormous pixel buffers
        )
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
        image = QImage(pix.samples, pix.width, pix.height, pix.stride,
                       QImage.Format_RGB888)
        self._image.setPixmap(QPixmap.fromImage(image))
        self._image.setMinimumSize(image.width(), image.height())
        self._page_counter.setText(f"{self._page + 1} / {self._doc.page_count}")
        self._scroll.verticalScrollBar().setValue(0)

    def _next_page(self):
        if self._doc and self._page < self._doc.page_count - 1:
            self._page += 1
            self._render_page()

    def _prev_page(self):
        if self._doc and self._page > 0:
            self._page -= 1
            self._render_page()

    def wheelEvent(self, event):
        """Turn the page; the whole page fits, so there is nothing to scroll."""
        delta = event.angleDelta().y()
        if delta < 0:
            self._next_page()
        elif delta > 0:
            self._prev_page()
        event.accept()

    def mousePressEvent(self, event):
        """Clicking the preview dismisses it, like macOS Quick Look."""
        self.dismiss()
        event.accept()

    # --- following the list selection ---------------------------------

    def _on_anchor_current_changed(self, *args):
        """While open, swap the preview to the newly selected document."""
        if not self._open:
            return
        target = self._resolve_path()
        if target is None:
            return  # a header/parent row: keep showing the current document
        path, filename = target
        if str(path) == self._path:
            return
        self._load(str(path), str(filename))

    # --- key handling on the hosting list -----------------------------

    def eventFilter(self, obj, event):
        if obj is self._anchor:
            if event.type() == QEvent.Hide:
                self.dismiss()
            elif event.type() == QEvent.KeyPress:
                key = event.key()
                if self._open:
                    # Space/Esc close the preview; every other key keeps
                    # working on the list, and the selection-change hook
                    # updates the preview accordingly.
                    if key in (Qt.Key_Space, Qt.Key_Escape):
                        self.dismiss()
                        return True
                    return False
                if key == Qt.Key_Space:
                    if self.toggle():
                        return True  # consumed: no checkbox toggle
                    return False
        return super().eventFilter(obj, event)

"""PDF and normalized-text evidence preview widgets."""

from __future__ import annotations

import html
import os
from pathlib import Path
import re

import fitz  # PyMuPDF

from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QScrollArea, QSpinBox, QWidget, QStackedWidget, QTextBrowser,
)
from PyQt5.QtCore import Qt, QRectF
from PyQt5.QtGui import QPixmap, QImage, QPainter, QColor, QPen


def normalized_text_path(doc_data: dict) -> Path | None:
    """Resolve the active normalized text belonging to a document."""
    from ..config import active_workspace

    patient_id = str(doc_data.get("patient_id") or "")
    document_id = str(doc_data.get("id") or doc_data.get("document_id") or "")
    if not patient_id or not document_id:
        return None
    candidates = (
        active_workspace.path / patient_id / "extraction" / f"{document_id}.md",
        active_workspace.path / patient_id / "docling" / f"{document_id}.md",
    )
    return next((path for path in candidates if path.exists()), None)


def highlighted_normalized_html(text: str, quotes: list[str]) -> str:
    """Render normalized text with every exact evidence quote highlighted."""
    source = str(text or "")
    spans: list[tuple[int, int]] = []
    for quote in quotes:
        words = str(quote or "").split()
        if not words:
            continue
        pattern = re.compile(
            r"\s+".join(re.escape(word) for word in words), re.IGNORECASE
        )
        for match in pattern.finditer(source):
            spans.append(match.span())
    merged: list[tuple[int, int]] = []
    for start, end in sorted(spans):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    pieces, cursor = [], 0
    for start, end in merged:
        pieces.append(html.escape(source[cursor:start]))
        pieces.append(
            "<mark style='background:#fff176;color:#111'>"
            + html.escape(source[start:end]) + "</mark>"
        )
        cursor = end
    pieces.append(html.escape(source[cursor:]))
    return "<pre style='white-space:pre-wrap'>" + "".join(pieces) + "</pre>"


class DocumentEvidencePreview(QWidget):
    """Embeddable original-document preview with a normalized-text toggle."""

    def __init__(self, services: dict, parent=None):
        super().__init__(parent)
        self._services = services
        self._doc_data: dict = {}
        self._current_page = 0
        self._zoom = 1.5
        self._doc = None
        self._highlights: list[dict] = []
        self._normalized_text = ""
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
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
        self._text_toggle = QPushButton("Testo clinico normalizzato")
        self._text_toggle.setCheckable(True)
        self._text_toggle.setToolTip(
            "Alterna il referto originale e il testo clinico normalizzato"
        )
        self._text_toggle.toggled.connect(self._toggle_text)
        toolbar.addWidget(self._text_toggle)
        zoom_out_btn = QPushButton("−")
        zoom_out_btn.clicked.connect(lambda: self._set_zoom(self._zoom - 0.25))
        toolbar.addWidget(zoom_out_btn)
        self._zoom_label = QLabel(f"{self._zoom:.1f}x")
        toolbar.addWidget(self._zoom_label)
        zoom_in_btn = QPushButton("+")
        zoom_in_btn.clicked.connect(lambda: self._set_zoom(self._zoom + 0.25))
        toolbar.addWidget(zoom_in_btn)
        layout.addLayout(toolbar)

        self._stack = QStackedWidget()
        self._scroll_area = QScrollArea()
        self._scroll_area.setWidgetResizable(True)
        self._scroll_area.setAlignment(Qt.AlignCenter)
        self._page_label = QLabel("Seleziona un'evidenza per aprire il referto")
        self._page_label.setAlignment(Qt.AlignCenter)
        self._scroll_area.setWidget(self._page_label)
        self._stack.addWidget(self._scroll_area)
        self._normalized_view = QTextBrowser()
        self._normalized_view.setReadOnly(True)
        self._stack.addWidget(self._normalized_view)
        layout.addWidget(self._stack, stretch=1)

    def set_document(
        self,
        doc_data: dict,
        *,
        highlights: list[dict] | None = None,
        selected_evidence_id: str | None = None,
    ) -> None:
        self.close_document()
        self._doc_data = dict(doc_data or {})
        self._highlights = [dict(item) for item in (highlights or [])]
        for item in self._highlights:
            item["selected"] = bool(
                selected_evidence_id
                and item.get("evidence_id") == selected_evidence_id
            )
        self._load_normalized_text()
        self._load_pdf()
        selected = next(
            (item for item in self._highlights if item.get("selected")),
            self._highlights[0] if self._highlights else None,
        )
        page = int((selected or {}).get("page_number") or 1)
        if self._doc and 1 <= page <= self._doc.page_count:
            self._show_page(page - 1)

    def _load_normalized_text(self) -> None:
        path = normalized_text_path(self._doc_data)
        try:
            self._normalized_text = path.read_text(encoding="utf-8") if path else ""
        except OSError:
            self._normalized_text = ""
        quotes = [str(item.get("source_text") or "") for item in self._highlights]
        if self._normalized_text:
            self._normalized_view.setHtml(
                highlighted_normalized_html(self._normalized_text, quotes)
            )
        else:
            self._normalized_view.setPlainText(
                "Testo clinico normalizzato non disponibile."
            )
        self._text_toggle.setEnabled(bool(self._normalized_text))
        if not self._normalized_text:
            self._text_toggle.setChecked(False)

    def _load_pdf(self) -> None:
        from ..utils.document_paths import resolve_document_path

        path = resolve_document_path(self._doc_data)
        if not path or not os.path.exists(path):
            self._page_label.setText("File originale non trovato")
            return
        try:
            self._doc = fitz.open(path)
            self._locate_selected_highlight()
            self._page_spin.setMaximum(max(1, self._doc.page_count))
            self._page_count_label.setText(f"/ {self._doc.page_count}")
            self._show_page(0)
        except Exception as exc:
            self._page_label.setText(f"Errore apertura documento: {exc}")

    @staticmethod
    def _search_text_candidates(item: dict) -> list[str]:
        """Short grounded fragments work better than a page-long quote."""
        quote_words = str(item.get("source_text") or "").split()
        entity = " ".join(str(item.get("normalized_entity") or "").split())
        candidates = [entity]
        if quote_words:
            candidates.extend((
                " ".join(quote_words[:32]),
                " ".join(quote_words[-24:]),
                " ".join(quote_words[:14]),
            ))
        return list(dict.fromkeys(
            candidate for candidate in candidates if len(candidate) >= 6
        ))

    def _locate_selected_highlight(self) -> None:
        """Recover page/geometry when legacy evidence has no saved bbox."""
        if not self._doc:
            return
        selected = next((
            item for item in self._highlights if item.get("selected")
        ), None)
        if selected is None:
            return
        page_number = int(selected.get("page_number") or 0)
        pages = (
            [self._doc[page_number - 1]]
            if 1 <= page_number <= self._doc.page_count
            else list(self._doc)
        )
        for page in pages:
            for candidate in self._search_text_candidates(selected):
                matches = page.search_for(candidate)
                if not matches:
                    continue
                match = matches[0]
                selected["page_number"] = page.number + 1
                selected["bbox"] = [match.x0, match.y0, match.x1, match.y1]
                return

    def _page_highlights(self, page, page_number: int) -> list[dict]:
        result = []
        for item in self._highlights:
            if int(item.get("page_number") or 0) != page_number:
                continue
            bbox = item.get("bbox") or []
            if len(bbox) == 4:
                result.append(item)
                continue
            for candidate in self._search_text_candidates(item):
                matches = page.search_for(candidate)
                if not matches:
                    continue
                for match in matches[:8]:
                    located = dict(item)
                    located["bbox"] = [match.x0, match.y0, match.x1, match.y1]
                    result.append(located)
                break
        return result

    def _show_page(self, page_num: int) -> None:
        if not self._doc or not (0 <= page_num < self._doc.page_count):
            return
        self._current_page = page_num
        self._page_spin.blockSignals(True)
        self._page_spin.setValue(page_num + 1)
        self._page_spin.blockSignals(False)
        page = self._doc[page_num]
        pix = page.get_pixmap(
            matrix=fitz.Matrix(self._zoom, self._zoom),
            colorspace=fitz.csRGB,
            alpha=False,
        )
        image = QImage(
            pix.samples, pix.width, pix.height, pix.stride,
            QImage.Format_RGB888,
        )
        pixmap = QPixmap.fromImage(image)
        painter = QPainter(pixmap)
        for highlight in self._page_highlights(page, page_num + 1):
            x0, y0, x1, y1 = (
                float(value) * self._zoom for value in highlight["bbox"]
            )
            selected = bool(highlight.get("selected"))
            color = QColor(255, 193, 7, 105) if selected else QColor(255, 235, 59, 70)
            pen = QColor(198, 40, 40) if selected else QColor(245, 124, 0)
            painter.setPen(QPen(pen, 3 if selected else 2))
            painter.fillRect(QRectF(x0, y0, x1 - x0, y1 - y0), color)
        painter.end()
        self._page_label.setPixmap(pixmap)
        self._page_label.setMinimumSize(pixmap.size())
        self._prev_btn.setEnabled(page_num > 0)
        self._next_btn.setEnabled(page_num < self._doc.page_count - 1)

    def _toggle_text(self, enabled: bool) -> None:
        self._stack.setCurrentIndex(1 if enabled else 0)

    def _prev_page(self) -> None:
        if self._current_page > 0:
            self._show_page(self._current_page - 1)

    def _next_page(self) -> None:
        if self._doc and self._current_page < self._doc.page_count - 1:
            self._show_page(self._current_page + 1)

    def _on_page_changed(self, page_num: int) -> None:
        if self._doc and 1 <= page_num <= self._doc.page_count:
            self._show_page(page_num - 1)

    def _set_zoom(self, zoom: float) -> None:
        self._zoom = max(0.5, min(4.0, zoom))
        self._zoom_label.setText(f"{self._zoom:.1f}x")
        if self._doc:
            self._show_page(self._current_page)

    def focus_evidence(self, page_number: int, bbox: list[float]) -> None:
        self._highlights = [{"page_number": page_number, "bbox": bbox, "selected": True}]
        if self._doc and 1 <= page_number <= self._doc.page_count:
            self._show_page(page_number - 1)

    def close_document(self) -> None:
        if self._doc:
            self._doc.close()
            self._doc = None


class PDFViewerDialog(QDialog):
    """Full document viewer retaining the original public interface."""

    def __init__(
        self,
        doc_data: dict,
        services: dict,
        parent=None,
        highlight: dict | None = None,
    ):
        super().__init__(parent)
        self.setWindowTitle(f"PDF Viewer — {doc_data.get('filename', '')}")
        self.resize(900, 800)
        layout = QVBoxLayout(self)
        self._preview = DocumentEvidencePreview(services, self)
        layout.addWidget(self._preview)
        self._preview.set_document(
            doc_data, highlights=[highlight] if highlight else []
        )

    def focus_evidence(self, page_number: int, bbox: list[float]) -> None:
        self._preview.focus_evidence(page_number, bbox)

    def closeEvent(self, event) -> None:
        self._preview.close_document()
        event.accept()

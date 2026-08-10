"""Deterministic extraction of electronic PDFs with pdfplumber."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re
import time
from typing import Any


@dataclass
class PdfTable:
    table_id: str
    page: int
    bbox: tuple[float, float, float, float] | None
    rows: list[list[str]]
    strategy: str

    def to_dict(self) -> dict:
        return {
            "table_id": self.table_id,
            "page": self.page,
            "bbox": list(self.bbox) if self.bbox else None,
            "rows": self.rows,
            "strategy": self.strategy,
        }


@dataclass
class PdfPage:
    page: int
    width: float
    height: float
    text: str
    words: list[dict]
    tables: list[PdfTable] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "page": self.page,
            "width": self.width,
            "height": self.height,
            "text": self.text,
            "words": self.words,
            "tables": [table.to_dict() for table in self.tables],
        }


@dataclass
class PdfExtractionResult:
    source_path: str
    pages: list[PdfPage]
    method: str = "pdfplumber"
    has_native_text: bool = True
    warnings: list[str] = field(default_factory=list)
    elapsed_seconds: float = 0.0

    @property
    def page_count(self) -> int:
        return len(self.pages)

    @property
    def tables(self) -> list[PdfTable]:
        return [table for page in self.pages for table in page.tables]

    @property
    def plain_text(self) -> str:
        return "\n\n".join(
            f"--- PAGINA {page.page} ---\n{page.text}" for page in self.pages
        ).strip()

    @property
    def markdown(self) -> str:
        sections = []
        for page in self.pages:
            content = [f"<!-- page:{page.page} -->", page.text.strip()]
            for table in page.tables:
                rendered = self._table_to_markdown(table.rows)
                if rendered:
                    content.extend([f"\n<!-- table:{table.table_id} -->", rendered])
            sections.append("\n".join(part for part in content if part))
        return "\n\n".join(sections).strip()

    @staticmethod
    def _table_to_markdown(rows: list[list[str]]) -> str:
        if not rows:
            return ""
        width = max(len(row) for row in rows)
        normalized = [row + [""] * (width - len(row)) for row in rows]
        escaped = [
            [str(cell or "").replace("|", "\\|").replace("\n", " ").strip()
             for cell in row]
            for row in normalized
        ]
        header = escaped[0]
        body = escaped[1:]
        lines = ["| " + " | ".join(header) + " |"]
        lines.append("| " + " | ".join("---" for _ in header) + " |")
        lines.extend("| " + " | ".join(row) + " |" for row in body)
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "source_path": self.source_path,
            "method": self.method,
            "has_native_text": self.has_native_text,
            "warnings": self.warnings,
            "elapsed_seconds": self.elapsed_seconds,
            "metrics": {
                "page_count": self.page_count,
                "character_count": sum(len(page.text) for page in self.pages),
                "word_count": sum(len(page.words) for page in self.pages),
                "table_count": len(self.tables),
            },
            "pages": [page.to_dict() for page in self.pages],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "PdfExtractionResult":
        pages = []
        for page_data in data.get("pages", []):
            tables = [
                PdfTable(
                    table_id=table["table_id"],
                    page=int(table.get("page") or page_data.get("page") or 1),
                    bbox=tuple(table["bbox"]) if table.get("bbox") else None,
                    rows=table.get("rows", []),
                    strategy=table.get("strategy", "unknown"),
                )
                for table in page_data.get("tables", [])
            ]
            pages.append(PdfPage(
                page=int(page_data.get("page") or len(pages) + 1),
                width=float(page_data.get("width") or 0),
                height=float(page_data.get("height") or 0),
                text=page_data.get("text", ""),
                words=page_data.get("words", []),
                tables=tables,
            ))
        return cls(
            source_path=data.get("source_path", ""),
            pages=pages,
            method=data.get("method", "pdfplumber"),
            has_native_text=bool(data.get("has_native_text", True)),
            warnings=data.get("warnings", []),
            elapsed_seconds=float(data.get("elapsed_seconds") or 0.0),
        )

    def locate_source(
        self, source_text: str, page_hint: int | None = None
    ) -> tuple[int | None, tuple[float, float, float, float] | None]:
        """Locate an LLM-quoted source span in deterministic word geometry."""

        wanted = self._tokens(source_text)
        if not wanted:
            return page_hint, None
        pages = self.pages
        if page_hint:
            pages = sorted(pages, key=lambda page: page.page != page_hint)
        for page in pages:
            pairs = [
                (self._normalize_token(word.get("text", "")), word)
                for word in page.words
            ]
            pairs = [(token, word) for token, word in pairs if token]
            tokens = [token for token, _ in pairs]
            limit = len(tokens) - len(wanted) + 1
            for start in range(max(0, limit)):
                if tokens[start:start + len(wanted)] != wanted:
                    continue
                selected = [word for _, word in pairs[start:start + len(wanted)]]
                return page.page, (
                    min(float(word["x0"]) for word in selected),
                    min(float(word["top"]) for word in selected),
                    max(float(word["x1"]) for word in selected),
                    max(float(word["bottom"]) for word in selected),
                )
        return page_hint, None

    @classmethod
    def _tokens(cls, value: str) -> list[str]:
        return [token for token in (cls._normalize_token(part) for part in value.split()) if token]

    @staticmethod
    def _normalize_token(value: str) -> str:
        return re.sub(r"[^A-ZÀ-Ü0-9]+", "", value.upper())


class PdfPlumberExtractor:
    """Extract text geometry and tables without a generative model."""

    def __init__(self):
        self._init_error: str | None = None

    @property
    def is_available(self) -> bool:
        try:
            import pdfplumber  # noqa: F401
            import pandas  # noqa: F401
            return True
        except ImportError as exc:
            self._init_error = str(exc)
            return False

    def convert(self, file_path: str | Path) -> PdfExtractionResult:
        path = Path(file_path)
        if path.suffix.lower() == ".pdf":
            return self._extract_pdf(path)
        return self._extract_image(path)

    def _extract_pdf(self, path: Path) -> PdfExtractionResult:
        import pdfplumber

        started = time.perf_counter()
        pages = []
        warnings = []
        with pdfplumber.open(path, unicode_norm="NFC") as pdf:
            for page_number, page in enumerate(pdf.pages, start=1):
                try:
                    words = page.extract_words(
                        x_tolerance=2,
                        y_tolerance=3,
                        keep_blank_chars=False,
                        use_text_flow=False,
                        expand_ligatures=True,
                    )
                    text = page.extract_text(
                        x_tolerance=2, y_tolerance=3, layout=True
                    ) or ""
                    tables = self._extract_tables(page, page_number)
                except Exception as exc:
                    warnings.append(f"Pagina {page_number}: {exc}")
                    words, text, tables = [], "", []
                pages.append(PdfPage(
                    page=page_number,
                    width=float(page.width),
                    height=float(page.height),
                    text=text,
                    words=[self._clean_word(word) for word in words],
                    tables=tables,
                ))

        sparse_page_numbers = {
            page.page for page in pages if len(page.text.strip()) < 20
        }
        method = "pdfplumber"

        # Some fully electronic PDFs (notably Adobe LiveCycle / XFA output)
        # expose no characters to pdfminer/pdfplumber even though a normal,
        # visible text layer is present. Try PyMuPDF native extraction before
        # treating any page as an image and invoking OCR.
        if sparse_page_numbers:
            pymupdf_result = self._extract_with_pymupdf(
                path, page_numbers=sparse_page_numbers
            )
            native_replacements = {
                page.page: page
                for page in (pymupdf_result.pages if pymupdf_result else [])
                if len(page.text.strip()) >= 20
            }
            if native_replacements:
                pages = self._replace_pages(pages, native_replacements)
                warnings.append(
                    "Testo nativo recuperato con PyMuPDF nelle pagine: "
                    + ", ".join(
                        str(number) for number in sorted(native_replacements)
                    )
                )
                method = (
                    "pymupdf_native"
                    if len(native_replacements) == len(pages)
                    else "pdfplumber+pymupdf_native"
                )
            if pymupdf_result:
                warnings.extend(pymupdf_result.warnings)

        # OCR is the final fallback and is restricted to pages for which both
        # native extractors returned too little text.
        remaining_sparse = {
            page.page for page in pages if len(page.text.strip()) < 20
        }
        native_character_count = sum(len(page.text.strip()) for page in pages)
        has_native_text = native_character_count >= max(20, len(pages) * 10)
        if remaining_sparse:
            ocr_result = self._extract_with_local_ocr(
                path, page_numbers=remaining_sparse
            )
            ocr_replacements = {
                page.page: page
                for page in (ocr_result.pages if ocr_result else [])
                if page.text.strip()
            }
            if ocr_replacements:
                pages = self._replace_pages(pages, ocr_replacements)
                warnings.append(
                    "OCR locale applicato solo dopo il fallimento dei due "
                    "estrattori nativi, pagine: "
                    + ", ".join(
                        str(number) for number in sorted(ocr_replacements)
                    )
                )
                method = (
                    "local_ocr" if not has_native_text
                    else f"{method}+local_ocr"
                )
            if ocr_result:
                warnings.extend(ocr_result.warnings)

        result = PdfExtractionResult(
            source_path=str(path),
            pages=pages,
            method=method,
            has_native_text=has_native_text,
            warnings=warnings,
            elapsed_seconds=time.perf_counter() - started,
        )
        return result

    def _extract_with_pymupdf(
        self, path: Path, page_numbers: set[int] | None = None
    ) -> PdfExtractionResult:
        """Extract the native text layer using PyMuPDF, never OCR."""
        import fitz

        if hasattr(fitz, "no_recommend_layout"):
            fitz.no_recommend_layout()

        pages = []
        warnings = []
        try:
            document = fitz.open(str(path))
        except Exception as exc:
            return PdfExtractionResult(
                source_path=str(path), pages=[], method="pymupdf_error",
                has_native_text=False, warnings=[str(exc)],
            )
        try:
            for page_number, page in enumerate(document, start=1):
                if page_numbers is not None and page_number not in page_numbers:
                    continue
                try:
                    raw_words = page.get_text("words", sort=True)
                    words = [self._fitz_word(word) for word in raw_words]
                    text = page.get_text("text", sort=True) or ""
                    tables = self._extract_pymupdf_tables(page, page_number)
                except Exception as exc:
                    warnings.append(f"PyMuPDF pagina {page_number}: {exc}")
                    words, text, tables = [], "", []
                pages.append(PdfPage(
                    page=page_number,
                    width=float(page.rect.width),
                    height=float(page.rect.height),
                    text=text,
                    words=words,
                    tables=tables,
                ))
        finally:
            document.close()
        character_count = sum(len(page.text.strip()) for page in pages)
        return PdfExtractionResult(
            source_path=str(path), pages=pages, method="pymupdf_native",
            has_native_text=character_count >= max(20, len(pages) * 10),
            warnings=warnings,
        )

    def _extract_image(self, path: Path) -> PdfExtractionResult:
        started = time.perf_counter()
        result = self._extract_with_local_ocr(path)
        if result is None:
            result = PdfExtractionResult(
                source_path=str(path), pages=[], method="ocr_unavailable",
                has_native_text=False,
                warnings=["OCR locale non disponibile per l'immagine"],
            )
        result.elapsed_seconds = time.perf_counter() - started
        return result

    def _extract_with_local_ocr(
        self, path: Path, page_numbers: set[int] | None = None
    ) -> PdfExtractionResult | None:
        import fitz

        pages = []
        warnings = []
        try:
            document = fitz.open(str(path))
        except Exception as exc:
            return PdfExtractionResult(
                source_path=str(path), pages=[], method="ocr_error",
                has_native_text=False, warnings=[str(exc)],
            )
        try:
            for page_number, page in enumerate(document, start=1):
                if page_numbers is not None and page_number not in page_numbers:
                    continue
                text_page = None
                for language in ("ita+eng", "eng"):
                    try:
                        text_page = page.get_textpage_ocr(
                            language=language, dpi=200, full=True
                        )
                        break
                    except Exception:
                        continue
                if text_page is None:
                    warnings.append(f"OCR non disponibile a pagina {page_number}")
                    pages.append(PdfPage(
                        page_number, float(page.rect.width), float(page.rect.height),
                        "", [], [],
                    ))
                    continue
                raw_words = page.get_text("words", textpage=text_page, sort=True)
                words = [self._fitz_word(word) for word in raw_words]
                text = page.get_text("text", textpage=text_page, sort=True)
                pages.append(PdfPage(
                    page_number, float(page.rect.width), float(page.rect.height),
                    text, words, [],
                ))
        finally:
            document.close()
        return PdfExtractionResult(
            source_path=str(path), pages=pages, method="local_ocr",
            has_native_text=False, warnings=warnings,
        )

    def _extract_pymupdf_tables(self, page, page_number: int) -> list[PdfTable]:
        """Best-effort native table extraction for pages missed by pdfplumber."""
        try:
            finder = page.find_tables()
            found = getattr(finder, "tables", []) or []
        except Exception:
            return []
        tables = []
        for table in found:
            try:
                raw_rows = table.extract() or []
                rows = [
                    [str(cell or "").replace("\x00", "").strip() for cell in row]
                    for row in raw_rows if row
                ]
                if len(rows) < 2 or max(
                    (len(row) for row in rows), default=0
                ) < 2:
                    continue
                bbox = tuple(float(value) for value in table.bbox)
                table_id = f"T_{page_number:03d}_{len(tables) + 1:02d}"
                tables.append(PdfTable(
                    table_id, page_number, bbox, rows, "pymupdf"
                ))
            except Exception:
                continue
        return tables

    @staticmethod
    def _fitz_word(word) -> dict:
        return {
            "text": str(word[4]),
            "x0": float(word[0]),
            "top": float(word[1]),
            "x1": float(word[2]),
            "bottom": float(word[3]),
            "doctop": float(word[1]),
        }

    @staticmethod
    def _replace_pages(
        pages: list[PdfPage], replacements: dict[int, PdfPage]
    ) -> list[PdfPage]:
        """Replace page text/geometry while retaining useful parsed tables."""
        merged = []
        for original in pages:
            replacement = replacements.get(original.page)
            if replacement is None:
                merged.append(original)
                continue
            tables = original.tables or replacement.tables
            merged.append(PdfPage(
                page=replacement.page,
                width=replacement.width,
                height=replacement.height,
                text=replacement.text,
                words=replacement.words,
                tables=tables,
            ))
        return merged

    def _extract_tables(self, page, page_number: int) -> list[PdfTable]:
        settings = [
            ("lines", {
                "vertical_strategy": "lines",
                "horizontal_strategy": "lines",
                "snap_tolerance": 3,
                "join_tolerance": 3,
                "intersection_tolerance": 3,
                "text_x_tolerance": 2,
                "text_y_tolerance": 3,
            }),
            ("text", {
                "vertical_strategy": "text",
                "horizontal_strategy": "text",
                "min_words_vertical": 2,
                "min_words_horizontal": 1,
                "intersection_tolerance": 4,
                "text_x_tolerance": 2,
                "text_y_tolerance": 3,
            }),
        ]
        tables = []
        seen_bboxes = []
        for strategy, table_settings in settings:
            try:
                found = page.find_tables(table_settings=table_settings)
            except Exception:
                continue
            for table in found:
                bbox = tuple(float(value) for value in table.bbox)
                if any(self._bbox_overlap(bbox, existing) >= 0.85 for existing in seen_bboxes):
                    continue
                raw_rows = table.extract(x_tolerance=2, y_tolerance=3) or []
                rows = [
                    [str(cell or "").replace("\x00", "").strip() for cell in row]
                    for row in raw_rows if row
                ]
                if len(rows) < 2 or max((len(row) for row in rows), default=0) < 2:
                    continue
                table_id = f"T_{page_number:03d}_{len(tables) + 1:02d}"
                tables.append(PdfTable(table_id, page_number, bbox, rows, strategy))
                seen_bboxes.append(bbox)
        return tables

    @staticmethod
    def _bbox_overlap(first, second) -> float:
        x0, top = max(first[0], second[0]), max(first[1], second[1])
        x1, bottom = min(first[2], second[2]), min(first[3], second[3])
        intersection = max(0.0, x1 - x0) * max(0.0, bottom - top)
        if intersection <= 0:
            return 0.0
        area_first = max(1.0, (first[2] - first[0]) * (first[3] - first[1]))
        area_second = max(1.0, (second[2] - second[0]) * (second[3] - second[1]))
        return intersection / min(area_first, area_second)

    @staticmethod
    def _clean_word(word: dict[str, Any]) -> dict:
        return {
            "text": str(word.get("text", "")),
            "x0": float(word.get("x0", 0.0)),
            "top": float(word.get("top", 0.0)),
            "x1": float(word.get("x1", 0.0)),
            "bottom": float(word.get("bottom", 0.0)),
            "doctop": float(word.get("doctop", word.get("top", 0.0))),
        }

    # Compatibility with the current processing orchestrator.
    def export_markdown(self, result: PdfExtractionResult) -> str:
        return result.markdown

    def export_text(self, result: PdfExtractionResult) -> str:
        return result.plain_text

    def export_dict(self, result: PdfExtractionResult) -> dict:
        return result.to_dict()

    def export_tables(self, result: PdfExtractionResult) -> list:
        import pandas as pd

        frames = []
        for table in result.tables:
            if not table.rows:
                continue
            width = max(len(row) for row in table.rows)
            rows = [row + [""] * (width - len(row)) for row in table.rows]
            header = self._unique_headers(rows[0])
            body = rows[1:]
            frames.append(pd.DataFrame(body, columns=header))
        return frames

    def get_page_count(self, result: PdfExtractionResult) -> int:
        return result.page_count

    @staticmethod
    def _unique_headers(header: list[str]) -> list[str]:
        seen = {}
        result = []
        for index, value in enumerate(header):
            base = str(value or "").strip() or f"column_{index + 1}"
            seen[base] = seen.get(base, 0) + 1
            result.append(base if seen[base] == 1 else f"{base}_{seen[base]}")
        return result

"""Docling standard converter."""

from __future__ import annotations

from pathlib import Path


def docling_source_markdown(document, *, page_no=None):
    """Retain visible source layers until clinical filtering has assessed them.

    A page header may contain an exam date. BODY alone is not a clinical-content
    classifier and must not silently discard such information at parser level.
    """
    from docling_core.types.doc import ContentLayer
    return document.export_to_markdown(
        page_no=page_no,
        included_content_layers={ContentLayer.BODY, ContentLayer.FURNITURE, ContentLayer.NOTES},
    )


class DoclingConverter:
    """Small lazy wrapper around Docling's standard DocumentConverter."""

    def __init__(self):
        self._converter = None
        self._initialized = False
        self._init_error = None

    @property
    def is_available(self) -> bool:
        """Check if the converter can be initialized."""
        if self._init_error:
            return False
        try:
            import docling
            return True
        except ImportError:
            self._init_error = "docling non installato"
            return False

    def _ensure_initialized(self):
        """Lazy initialization of the converter."""
        if self._initialized:
            return

        from docling.document_converter import DocumentConverter

        self._converter = DocumentConverter()
        self._initialized = True
        print("  ✓ Docling standard pipeline ready")

    def convert(self, file_path: str | Path):
        """Convert a PDF or image file to a DoclingDocument."""
        self._ensure_initialized()

        if not self._converter:
            raise RuntimeError(
                "Docling converter non inizializzato. "
                "Verifica che docling sia installato correttamente."
            )

        return self._converter.convert(str(file_path))

    def export_markdown(self, result) -> str:
        """Export conversion result to markdown."""
        if hasattr(result, 'document') and result.document:
            return docling_source_markdown(result.document)
        return ""

    def export_text(self, result) -> str:
        """Export conversion result to plain text."""
        if hasattr(result, 'document') and result.document:
            return docling_source_markdown(result.document)
        return ""

    def export_dict(self, result) -> dict:
        """Export conversion result to dictionary."""
        if hasattr(result, 'document') and result.document:
            return result.document.export_to_dict()
        return {}

    def export_tables(self, result) -> list:
        """Extract tables as pandas DataFrames from conversion result."""
        tables = []
        if hasattr(result, 'document') and result.document:
            for table in result.document.tables:
                try:
                    tables.append(table.export_to_dataframe())
                except Exception:
                    pass
        return tables

    def get_page_count(self, result) -> int:
        """Get the page count from the conversion result."""
        if hasattr(result, 'document') and result.document:
            try:
                return len(result.document.pages)
            except (AttributeError, TypeError):
                pass
        return 0

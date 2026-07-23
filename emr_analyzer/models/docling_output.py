"""Docling output data models."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DoclingElement:
    """A single element extracted by Docling from a document."""
    document_id: str
    element_id: str              # E001, E002, ...
    page: int
    type: str                    # heading, paragraph, table, list, ...
    text: str
    section: Optional[str] = None
    confidence: float = 1.0
    metadata_json: Optional[str] = None     # Additional JSON metadata

    def to_dict(self) -> dict:
        return {
            "document_id": self.document_id,
            "element_id": self.element_id,
            "page": self.page,
            "type": self.type,
            "text": self.text,
            "section": self.section,
            "confidence": self.confidence,
            "metadata_json": self.metadata_json,
        }


@dataclass
class DoclingPage:
    """Page-level information from Docling processing."""
    document_id: str
    page_number: int
    has_text: bool = True
    has_tables: bool = False
    image_path: Optional[str] = None       # Path to saved page image
    element_count: int = 0

    def to_dict(self) -> dict:
        return {
            "document_id": self.document_id,
            "page_number": self.page_number,
            "has_text": self.has_text,
            "has_tables": self.has_tables,
            "image_path": self.image_path,
            "element_count": self.element_count,
        }

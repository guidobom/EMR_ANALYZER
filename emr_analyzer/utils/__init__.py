"""Utility functions and constants for EMR Analyzer."""

from .file_utils import (
    compute_file_hash,
    is_supported_file,
    is_pdf,
    is_image,
    detect_file_type,
    verify_pdf,
    get_file_info,
)
from .text_utils import (
    normalize_italian_number,
    clean_clinical_text,
    fuzzy_find,
    split_into_sentences_ita,
    extract_section_text,
)
from .date_utils import parse_italian_date

__all__ = [
    "compute_file_hash",
    "is_supported_file",
    "is_pdf",
    "is_image",
    "detect_file_type",
    "verify_pdf",
    "get_file_info",
    "normalize_italian_number",
    "clean_clinical_text",
    "fuzzy_find",
    "split_into_sentences_ita",
    "extract_section_text",
    "parse_italian_date",
]

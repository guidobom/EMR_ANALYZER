"""Italian text normalization utilities."""

import re
from typing import Optional


def normalize_italian_number(text: str) -> Optional[float]:
    """
    Convert Italian number format to float.
    Italian: 1.234,56 = 1234.56 (dot=thousands, comma=decimal)
    """
    if not text:
        return None

    text = text.strip().replace(" ", "")

    # Pattern with both dot and comma: "1.234,56" -> "1234.56"
    if re.match(r'^\d{1,3}(?:\.\d{3})*,\d+$', text):
        text = text.replace(".", "").replace(",", ".")
    # Pattern with only comma: "1234,56" -> "1234.56"
    elif re.match(r'^\d+,\d+$', text):
        text = text.replace(",", ".")
    # Pattern with only dot as decimal: "1234.56"
    elif re.match(r'^\d+(?:\.\d+)?$', text):
        pass
    # Plain integer: "1234"
    elif text.isdigit():
        pass
    else:
        # Try cleaning up and converting
        cleaned = re.sub(r'[^\d.,]', '', text)
        if not cleaned:
            return None
        try:
            if '.' in cleaned and ',' in cleaned:
                cleaned = cleaned.replace(".", "").replace(",", ".")
            elif ',' in cleaned:
                cleaned = cleaned.replace(",", ".")
            return float(cleaned)
        except ValueError:
            return None

    try:
        return float(text)
    except ValueError:
        return None


def clean_clinical_text(text: str) -> str:
    """Clean clinical text from common artifacts."""
    # Remove page numbers
    text = re.sub(r'\n\s*\d{1,4}\s*\n', '\n', text)
    # Remove orphan dashes at line starts (word breaks)
    text = re.sub(r'\n\s*-\s*', '', text)
    # Collapse multiple newlines
    text = re.sub(r'\n{3,}', '\n\n', text)
    # Collapse multiple spaces
    text = re.sub(r' {2,}', ' ', text)
    # Remove common header/footer artifacts
    text = re.sub(r'(?i)pagina\s+\d+\s+di\s+\d+', '', text)
    text = re.sub(r'(?i)riservato\s+agli\s+operatori\s+sanitari', '', text)
    return text.strip()


def fuzzy_find(text: str, query: str, threshold: float = 0.8) -> bool:
    """
    Check if `query` appears in `text` with fuzzy matching.
    Used to verify that LLM-extracted source_text exists in the original.
    """
    # Normalize: lowercase, strip whitespace
    text_norm = re.sub(r'\s+', ' ', text.lower()).strip()
    query_norm = re.sub(r'\s+', ' ', query.lower()).strip()

    # Exact match (fast path)
    if query_norm in text_norm:
        return True

    # Fuzzy match using sequence matcher
    from difflib import SequenceMatcher
    # Try sliding window
    q_len = len(query_norm)
    if q_len > len(text_norm):
        return False

    for i in range(len(text_norm) - q_len + 1):
        window = text_norm[i:i + q_len]
        ratio = SequenceMatcher(None, window, query_norm).ratio()
        if ratio >= threshold:
            return True

    return False


def split_into_sentences_ita(text: str) -> list[str]:
    """Split Italian text into sentences."""
    # Simple sentence boundary detection for Italian
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-ZÀ-Ü])', text)
    return [s.strip() for s in sentences if s.strip()]


def extract_section_text(full_text: str, header_pattern: str,
                         next_patterns: list[str]) -> Optional[str]:
    """
    Extract text between a section header and the next section header.
    """
    match = re.search(header_pattern, full_text, re.IGNORECASE | re.MULTILINE)
    if not match:
        return None

    start = match.end()
    end = len(full_text)

    # Find the next section boundary
    for pattern in next_patterns:
        next_match = re.search(pattern, full_text[start:], re.IGNORECASE)
        if next_match:
            candidate = start + next_match.start()
            if candidate < end:
                end = candidate

    return full_text[start:end].strip()

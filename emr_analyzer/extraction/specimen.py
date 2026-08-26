"""Specimen (biological material) detection for laboratory reports.

The specimen is detected from ``Materiale:`` lines (document default) and
section headers (``[0] ESAME URINE COMPLETO``) that state the exam context.
Detection is deliberately conservative: a token is returned only when the
evidence is homogeneous and non-default; every uncertain case falls back
to ``None`` (blood/serum/plasma default, no suffix anywhere downstream).
"""

from __future__ import annotations

import re

from ..config import MATERIAL_RULES, SECTION_SPECIMEN_HINTS

# A section header is a short line without ':' that is not a numeric or
# recognized-textual RESULT line.  "URINOCOLTURA Negativa" is a header
# ("Negativa" is not in the textual-result vocabulary), while
# "Glucosio 98 mg/dL (70-110)" is a result line.
_TEXTUAL_VALUES = (
    r"NEGATIVO|POSITIVO|DEBOLE\s*POSITIVO|DEBOLMENTE\s*POSITIVO|"
    r"FORTEMENTE\s*POSITIVO|ASSENTE|PRESENTE|NON\s*RILEVABILE|RILEVABILE|"
    r"NELLA\s*NORMA|ALTERATO|SCARSI|NUMEROSI|RARI|DEBOLE|"
    r"NEGATIVITA|POSITIVITA"
)
# Compact mirror of lab_parser._NO_COLON_RESULT_RE: param + payload that
# starts with a numeric or recognized textual value.
_RESULT_LINE_RE = re.compile(
    r"^\s*[A-Za-zÀ-ÿ][^\n:]{1,100}?\s+"
    r"(?:[<>]=?\s*)?(?:"
    r"\d+(?:[.,]\d+)?"
    rf"|{_TEXTUAL_VALUES}"
    r").*$",
    re.IGNORECASE,
)
_SECTION_HEADER_RE = re.compile(r"^[A-Za-zÀ-ÿ][^:\n]{0,80}$")
_MATERIAL_LINE_RE = re.compile(
    r"(?im)^\s*Materiale\s*:?\s*(.{1,40})\s*$"
)
_HEADER_PREFIX_RE = re.compile(r"^\s*(?:\[\d+\])?\s*")


def _map_material(value: str) -> str | None:
    """Map a ``Materiale:`` value through MATERIAL_RULES (ordered)."""
    for pattern, token in MATERIAL_RULES:
        if re.search(pattern, value, re.IGNORECASE):
            return token
    return None


def detect_document_specimen(text: str) -> str | None:
    """Document-level default specimen from ``Materiale:`` lines.

    Conservative: only a HOMOGENEOUS non-default set of materials (one
    distinct token, no default-material line mixed in) yields a non-None
    default.  Mixed documents return None and rely on section hints.
    """
    tokens: set[str] = set()
    defaults_seen = False
    for line in _MATERIAL_LINE_RE.findall(text or ""):
        token = _map_material(line)
        if token is None:
            defaults_seen = True
        else:
            tokens.add(token)
    if len(tokens) == 1 and not defaults_seen:
        return tokens.pop()
    return None


def section_specimen(line: str) -> tuple[bool, str | None]:
    """(is_section_header, token_or_None) for one raw text line.

    A line is a section header only if it has no ':' or '|', is not a
    numeric/textual RESULT line, and matches a SECTION_SPECIMEN_HINTS rule.
    Token None from a hint means "reset to the standard blood/serum
    context" (EMOCROMO, SIEROLOGIA, ...).  Lines matching no hint are not
    headers at all (state unchanged).
    """
    stripped = _HEADER_PREFIX_RE.sub("", str(line or "")).strip()
    if not stripped:
        return False, None
    if ":" in stripped or "|" in stripped:
        return False, None
    if _RESULT_LINE_RE.match(stripped):
        return False, None
    if not _SECTION_HEADER_RE.match(stripped):
        return False, None
    for pattern, token in SECTION_SPECIMEN_HINTS:
        if re.search(pattern, stripped, re.IGNORECASE):
            return True, token
    return False, None

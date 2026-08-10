"""Regex patterns for Italian medical text extraction."""

import re

# --- Lab value extraction patterns ---

# Pattern 1: Table-like format: "Parametro    123    mg/dL    (70-110)"
LAB_TABULAR_PATTERN = re.compile(
    r'^(?P<param>[A-Za-zÀ-ÿ][\w\sÀ-ÿ/\-()%]*?[A-Za-zÀ-ÿ)])\s+'
    r'(?P<operator>[<>]?\s*[<>]?\s*)'
    r'(?P<value>[\d.,]+)\s*'
    r'(?P<unit>[a-zA-Zμ/³%²]+(?:\s*/\s*[a-zA-Zμ]+)?)\s*'
    r'(?:\(?\s*(?P<ref_range>[^)]+(?:\s*[-–]\s*[^)]+)?)\s*\)?)?',
    re.MULTILINE | re.IGNORECASE,
)

# Pattern 2: Inline format: "...la glicemia è 98 mg/dL (v.n. 70-110)..."
LAB_INLINE_PATTERN = re.compile(
    r'(?P<param>[A-Za-zÀ-ÿ][\w\sÀ-ÿ/\-]+)'
    r'(?:\s*(?:è|pari a|di|:|=)\s*)'
    r'(?P<operator>[<>]?\s*)'
    r'(?P<value>[\d.,]+)\s*'
    r'(?P<unit>[a-zA-Zμ/³%²]+)?\s*'
    r'(?:\(?\s*(?:(?:v\.?n\.?|riferimento|range|intervallo)\s*:?\s*)?'
    r'(?P<ref_range>[\d.,<>\s\-–]+\s*[-–]\s*[\d.,<>\s\-–]+)\s*\)?)?',
    re.IGNORECASE,
)

# Pattern 3: Known parameters lookup
def build_known_param_pattern(parameter: str) -> re.Pattern:
    """Build a regex for a specific known parameter name."""
    escaped = re.escape(parameter)
    return re.compile(
        rf'\b{escaped}\b[:\s]*'
        r'(?P<operator>[<>]?\s*)'
        r'(?P<value>[\d.,]+)\s*'
        r'(?P<unit>[a-zA-Zμ/³%²]+)?\s*'
        r'(?:\(?\s*(?P<ref_range>[\d.,<>\s\-–]+\s*[-–]\s*[\d.,<>\s\-–]+)\s*\)?)?',
        re.IGNORECASE,
    )


# Reference range sub-patterns
REF_RANGE_PATTERN = re.compile(
    r'(?P<low>[\d.,]+)\s*[-–]\s*(?P<high>[\d.,]+)'
)

REF_LESS_THAN_PATTERN = re.compile(
    r'[<]\s*(?P<high>[\d.,]+)'
)

REF_GREATER_THAN_PATTERN = re.compile(
    r'[>]\s*(?P<low>[\d.,]+)'
)


# --- Radiology section patterns ---

RADIOLOGY_SECTION_PATTERNS = {
    "technique": re.compile(
        r'(?im)^(?:TECNICA(?:\s+DI\s+ESAME)?|METODICA|PROCEDURA)\s*:?\s*\n'
        r'(?P<text>.+?)(?=\n\s*(?:REPERTO|REFERTO|DESCRIZIONE|CONCLUSIONI|IMPRESSIONI|GIUDIZIO|\Z))',
    ),
    "findings": re.compile(
        r'(?im)^(?:REPERTO|REFERTO|DESCRIZIONE|QUADRO\s+RADIOLOGICO|OSSERVAZIONI)\s*:?\s*\n'
        r'(?P<text>.+?)(?=\n\s*(?:CONCLUSIONI|IMPRESSIONI|GIUDIZIO|CONSIDERAZIONI|IN\s+CONCLUSIONE|\Z))',
    ),
    "conclusions": re.compile(
        r'(?im)^(?:CONCLUSIONI|IMPRESSIONI|GIUDIZIO|CONSIDERAZIONI|PARERE|IN\s+CONCLUSIONE)\s*:?\s*\n'
        r'(?P<text>.+?)(?=\n\s*(?:\Z))',
    ),
}


# --- Date extraction patterns ---

DATE_PATTERNS = [
    # ISO: 2026-04-12
    re.compile(r'\b(\d{4})[-/](\d{1,2})[-/](\d{1,2})\b'),
    # Italian: 12/04/2026
    re.compile(r'\b(\d{1,2})[/](\d{1,2})[/](\d{2,4})\b'),
    # Italian text: 12 aprile 2026
    re.compile(
        r'\b(\d{1,2})\s+(gennaio|febbraio|marzo|aprile|maggio|giugno|'
        r'luglio|agosto|settembre|ottobre|novembre|dicembre)\s+(\d{4})\b',
        re.IGNORECASE,
    ),
    # Short month: 12 apr 2026
    re.compile(
        r'\b(\d{1,2})\s+(gen|feb|mar|apr|mag|giu|lug|ago|set|ott|nov|dic)\w*\s+(\d{4})\b',
        re.IGNORECASE,
    ),
]


# --- Italian number patterns ---

ITALIAN_NUMBER = re.compile(
    r'-?(?:\d{1,3}(?:\.\d{3})*|\d+)(?:,\d+)?(?!\s*(?:gennaio|febbraio|marzo|aprile|maggio|giugno|luglio|agosto|settembre|ottobre|novembre|dicembre))'
)

# --- Therapy patterns ---

THERAPY_START_PATTERN = re.compile(
    r'(?i)(?:si\s+avvia|inizia|iniziato|avviato|impostato|prescritto|'
    r'somministrat[oa]|primo\s+ciclo|ciclo\s+1)\s+(?:trattamento\s+(?:con|a\s+base\s+di))?\s*'
    r'(?P<drug>[A-Za-zÀ-ÿ][\w\sÀ-ÿ/\-]+?)(?:\.|,|\s*$)'
)

# --- Diagnosis patterns ---

DIAGNOSIS_PATTERN = re.compile(
    r'(?i)(?:diagnosi(?:\s+(?:di|principale|secondaria|definitiva))?\s*(?::|=|è|di))\s*'
    r'(?P<diagnosis>[A-Za-zÀ-ÿ][\w\sÀ-ÿ/\-.,;()]+?)(?:\.|,|\n|$)'
)

"""Deterministic concept canonicalization for atomic-evidence identity.

The raw ``concept`` label stays untouched for display and provenance; these
mappings produce the *identity* form only, so that restatements of the same
clinical fact across documents (drug typos, exam-title variants, filler
attributes, English labels) merge into one atom while the original wording
remains visible on each fused source occurrence.
"""

from __future__ import annotations

from difflib import SequenceMatcher
import re
import unicodedata

# --- drug names -> active ingredient ---------------------------------------

_DRUG_CANONICALS = (
    "nivolumab", "ipilimumab", "pembrolizumab", "dabrafenib", "trametinib",
    "encorafenib", "binimetinib", "vemurafenib", "cobimetinib",
    "atezolizumab", "relatlimab", "temozolomide", "fotemustine",
    "dacarbazina", "interferone", "talimogene",
    "metformina", "clopidogrel", "acido acetilsalicilico",
    "deltacortene", "prednisone", "levotiroxina", "amlodipina",
    "ramipril", "bisoprololo", "pantoprazolo", "omeprazolo",
    "atorvastatina", "rosuvastatina", "allopurinolo", "dapagliflozin",
    "sitagliptin", "insulina",
)

_DRUG_MIN_LEN = 5
_DRUG_MIN_RATIO = 0.86


def _fuzzy_drug(name: str) -> str | None:
    value = _fold(name)
    if len(value) < _DRUG_MIN_LEN:
        return None
    best = None
    for candidate in _DRUG_CANONICALS:
        candidate_folded = _fold(candidate)
        if value == candidate_folded:
            return candidate_folded
        if abs(len(value) - len(candidate_folded)) > 3:
            continue
        ratio = SequenceMatcher(None, value, candidate_folded).ratio()
        if ratio >= _DRUG_MIN_RATIO:
            if best is None or ratio > best[0]:
                best = (ratio, candidate_folded)
    if best is None:
        return None
    # A real near-miss must share the first character; otherwise the fuzzy
    # step could map unrelated short names onto whitelist entries.
    if best[1][:1] != value[:1]:
        return None
    return best[1]


# --- imaging exam titles -> modality + region ------------------------------

_IMAGING_RULES: tuple[tuple[str, str], ...] = (
    # (regex on the folded concept, canonical form)
    (r"\btc\s+tb(?:\s+mdc)?\b.*", "tc total body"),
    (r"\btc\s+total\s+body(?:\s+mdc)?\b.*", "tc total body"),
    (r"\btac\s+total\s+body(?:\s+mdc)?\b.*", "tc total body"),
    (r"\btc\s+cranio\s+encefalo(?:\s+mdc)?\b.*", "tc encefalo"),
    (r"\btc\s+encefalo\b.*", "tc encefalo"),
    (r"\brm\s+encefalo\b.*", "rm encefalo"),
    (r"\bpet(?:[- ]tc)?\s*(?:con\s+)?(?:18f[- ]?)?fdg\b.*", "pet fdg"),
    (r"\bpet\s+fdg\b.*", "pet fdg"),
    (r"\bfdg[-\s]pet\b.*", "pet fdg"),
)
_IMAGING_PREFIX = ("imaging_finding", "radiology_finding",
                   "instrumental_finding")


def _canonical_imaging(name: str) -> str | None:
    folded = _fold(name)
    for pattern, canonical in _IMAGING_RULES:
        if re.fullmatch(pattern, folded):
            return canonical
    return None


# --- English labels --------------------------------------------------------

_EN_IT_GLOSSARY = (
    (r"\bmalignant\s+melanoma\b.*", "melanoma maligno"),
    (r"\bmetastatic\s+melanoma\b.*", "melanoma metastatico"),
    (r"\bmelanoma\s+metastatic\b.*", "melanoma metastatico"),
    (r"\bbasal\s+cell\s+carcinoma\b.*", "carcinoma basocellulare"),
    (r"\btumor\s+malignant\b.*", "tumore maligno"),
    (r"\bexcision\s+of\s+melanoma\b.*", "asportazione di melanoma"),
    (r"\bexcision\s+of\s+dorsal\s+melanoma\b.*", "asportazione di melanoma"),
    (r"\bsentinel\s+lymph\s+node\s+biopsy\b.*", "biopsia linfonodo sentinella"),
    (r"\bbiopsy\s+of\s+sentinel\s+lymph\s+node\b.*", "biopsia linfonodo sentinella"),
    (r"\bprogression\s+of\s+disease\b.*", "progressione di malattia"),
    (r"\blymph\s+node\s+enlargement\b.*", "linfoadenomegalia"),
    (r"\benlargement\s+of\s+lymph\s+node\b.*", "linfoadenomegalia"),
    (r"\blymph\s+node\s+involvement\b.*", "interessamento linfonodale"),
    (r"\blymph\s+node\s+group\b.*", "gruppo linfonodale"),
    (r"\blymph\s+node\s+sentinel\b.*", "linfonodo sentinella"),
    (r"\bbraf\s+mutation\b.*", "braf mutato"),
)


def _canonical_english(name: str) -> str | None:
    folded = _fold(name)
    for pattern, canonical in _EN_IT_GLOSSARY:
        if re.fullmatch(pattern, folded):
            return canonical
    return None


# --- severity fillers and EN/IT harmonization ------------------------------

_SEVERITY_FILLERS = {
    "none", "n.d.", "n.d", "n d", "nd", "n/a", "n a", "na",
    "non specificato", "non specificata",
    "non specificato nel testo", "non specificata nel testo",
    "unknown", "sconosciuto", "sconosciuta",
    "not specified", "non specified",
    "non documentato", "non documentata", "-", "0",
}

_SEVERITY_EN_IT = {
    "mild": "lieve",
    "moderate": "moderata",
    "severe": "grave",
    "high": "alto",
    "low": "basso",
    "grade 1": "g1",
    "grade 2": "g2",
    "grade 3": "g3",
    "grade 4": "g4",
    "stage i": "stadio i",
    "stage ii": "stadio ii",
    "stage iii": "stadio iii",
    "stage iv": "stadio iv",
}


def _fold(value: object) -> str:
    text = unicodedata.normalize("NFKD", str(value or "").casefold())
    without_marks = "".join(
        char for char in text if not unicodedata.combining(char)
    )
    return " ".join(
        "".join(char if char.isalnum() else " " for char in without_marks)
        .split()
    )


def canonical_concept(category: object, concept: object) -> str:
    """Identity form of a concept: display/provenance text is untouched."""
    folded = _fold(concept)
    # Categories carry underscores (``imaging_finding``): folding with the
    # alphanumeric-only normalizer would split them into words.
    category_text = str(category or "").strip().casefold()
    if category_text in {"medication", "terapia"}:
        drug = _fuzzy_drug(folded)
        if drug:
            return drug
        return folded
    if category_text in _IMAGING_PREFIX:
        imaging = _canonical_imaging(folded)
        if imaging:
            return imaging
    english = _canonical_english(folded)
    if english:
        return english
    folded = re.sub(r"\bpd[\s.-]?l[\s.-]?1\b", "pdl1", folded)
    return folded


def canonical_severity(value: object) -> str:
    """Identity form of a severity value; fillers collapse to empty."""
    folded = _fold(value)
    if folded in _SEVERITY_FILLERS:
        return ""
    harmonized = _SEVERITY_EN_IT.get(folded)
    if harmonized:
        return harmonized
    harmonized = _SEVERITY_EN_IT.get(folded.replace(" grade ", " "))
    if harmonized:
        return harmonized
    return folded

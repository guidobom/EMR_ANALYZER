"""Deterministic pre-filtering for immune-related adverse event (irAE) analysis.

Research prototype, three layers:

1. Temporal anchoring — the first immune checkpoint inhibitor (ICI) dose is
   located in the medication evidence (e.g. Nivolumab / Ipilimumab).
2. Organ toxicity lexicon — the atomic evidence registry is scanned for
   organ-specific toxicity tokens, each evidence dated relative to the anchor
   and bucketed into time bands.  No model is involved.
3. LLM hypothesis — a prompt fed ONLY the pre-filtered candidates (plus the
   full irAE protocol), producing the final per-organ assessment.  Driven by
   the caller once a Clinical State client is available (see
   ``tools/irae_headless.py``).

Layers 1-2 are deterministic and reproducible from the registry alone.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from typing import Any, Iterable

from .temporal import normalize_clinical_date

# Checkpoint inhibitors that mark immune-therapy exposure (Layer 1).
ICI_DRUGS = frozenset({
    "nivolumab",
    "ipilimumab",
    "pembrolizumab",
    "atezolizumab",
    "durvalumab",
    "avelumab",
    "tremelimumab",
    "cemiplimab",
    "relatlimab",
})

# Band boundaries (days from the first ICI dose). Late and post-stop toxicity
# is clinically real, so bands above the "classic" 0-112 day window are kept
# and flagged rather than dropped.
BAND_ORDER = ("pre_ici", "0_14d", "14_112d", "112_365d", "over_365d", "undated")
_DAYS_14 = 14
_DAYS_112 = 112
_DAYS_365 = 365

# Organ -> toxicity tokens, matched against the normalized entity and the
# atomic quote.  Full Italian laboratory names are included because the
# extractor stores "alanina_aminotransferasi" rather than "ALT".
PRIMARY_TOXICITY_LEXICON: dict[str, tuple[str, ...]] = {
    "Colite/Diarrea": (
        "diarrea",
        "diarrhe",
        "colite",
        "enterocolit",
        "mucose",
    ),
    "Epatite": (
        "epatit",
        "transaminas",
        "alanina_aminotransferasi",
        "aspartato_aminotransferasi",
        "bilirubina",
        r"\bgpt\b",
        r"\bgot\b",
        r"\balt\b",
        r"\bast\b",
    ),
    "Tiroidite": (
        "tiroidit",
        "ipotiroidismo",
        "ipertiroidismo",
        "ormone_tireostimolante",
        "tiroxina",
        "tireotropin",
        r"\btsh\b",
    ),
    "Polmonite": (
        "polmonit",
        "pneumonit",
        "interstiziopat",
        "alveolit",
        "tosse",
        "ipossia",
    ),
    "Dermatite": (
        "dermatit",
        "rash",
        "eruzione",
        "prurito",
        "vitiligine",
        "pemfigo",
        "eritema",
    ),
    "Miosite/CPK": (
        "miosit",
        "rabdomiolisi",
        "creatinchinasi",
        r"\bcpk\b",
    ),
    "Nefrite": (
        "nefrit",
        "insufficienza renale",
        "creatinina",
        r"\backi\b",
    ),
    "Artralgia": (
        "artralgi",
        "artrit",
        "dolore articolare",
    ),
    "Miocardite/Cardiotossicità": (
        "miocardit",
        "myocard",
        "troponin",
        "cardiotossic",
        "cardio_tossic",
        r"\bck-mb\b",
        r"\bck_mb\b",
        "frazione_di_eiezione",
        "frazione di eiezione",
        "aritm",
        "tachicardi",
        "fibrillazione",
        "insufficienza cardiaca",
        "insufficienza_cardiaca",
        "scompenso cardiaco",
        "scompenso_cardiaco",
    ),
    "Pancreatite": (
        "lipasi",
        "amilasi",
        "isoamilasi",
        "pancreat",
    ),
    "Oculari": (
        "uveit",
        "cheratit",
        "sclerit",
        "episclerit",
        "retinopatia",
        "coroidit",
    ),
    "Neurologiche": (
        "polineuropat",
        "neuropat",
        "encefalit",
        "mielit",
        "meningit",
        "miasten",
        "nevrite",
        "diplop",
        "atassia",
        "parkinson",
    ),
    "Ipofisite/Surrenale": (
        "ipofisit",
        "ipopituitarismo",
        r"\bacth\b",
        "cortisolo",
        "insufficienza surrenalica",
        "insufficienza_surrenalica",
        "sindrome_di_cushing",
    ),
    "Ematologiche": (
        "anemia_emolitica",
        "anemia emolitica",
        "emolisi",
        "neutropenia",
        "trombocitopenia",
        "pancitopenia",
        "agranulocitosi",
    ),
    "Cistite": (
        "cistit",
    ),
    "Mucosite/Sicca": (
        "mucosit",
        "stomatit",
        "xerostomia",
        "cheratocongiuntivite secca",
        "secchezza delle fauci",
    ),
    "Diabete/Iperglicemia": (
        "iperglicemia",
        "chetoacidosi",
        "diabete di tipo",
        "diabete_tipo",
    ),
}

# Non-organ-specific symptoms; kept separate and flagged ``generic`` so the
# LLM layer can weight them lower than organ-specific candidates.
GENERIC_SYMPTOM_TOKENS = (
    "astenia",
    "dolore",
    "nausea",
    "vomito",
    "inappetenza",
    "iporessia",
)


@dataclass(frozen=True)
class IciAnchor:
    """Layer 1 result: first (and last) dated checkpoint-inhibitor exposure."""

    first_drug: str
    first_date: date
    first_raw: str
    last_drug: str
    last_date: date
    last_raw: str
    occurrences: int


@dataclass(frozen=True)
class ToxicityCandidate:
    """One registry evidence flagged by the organ lexicon (Layer 2)."""

    organ: str
    evidence_id: str
    category: str
    entity: str
    observed_raw: str
    offset_days: int | None
    band: str
    quote: str
    generic: bool = False
    value: str = ""


def _parse_evidence_date(raw: str | None) -> date | None:
    """Best-effort day resolution of a registry ``observed_date``."""
    if not raw:
        return None
    normalized = normalize_clinical_date(raw)
    start = normalized.start
    if not start:
        return None
    try:
        parts = str(start).split("-")
        return date(int(parts[0]), int(parts[1]) if len(parts) > 1 else 1,
                    int(parts[2]) if len(parts) > 2 else 1)
    except (ValueError, IndexError):
        return None


def _normalized_text(row: dict[str, Any]) -> str:
    """The text the lexicon matches against: entity + the atomic quote."""
    entity = row.get("normalized_entity") or ""
    quote = ""
    data_json = row.get("data_json")
    if isinstance(data_json, dict):
        quote = data_json.get("quote_verified") or data_json.get(
            "wire_normalized"
        ) or ""
    return f"{entity} {quote}".lower()


def find_ici_anchor(rows: Iterable[dict[str, Any]]) -> IciAnchor | None:
    """Layer 1 — the first dated checkpoint-inhibitor medication.

    ``rows`` are registry evidence dicts with ``category``,
    ``normalized_entity`` and ``observed_date``.
    """
    hits: list[tuple[date, str, str]] = []
    for row in rows:
        entity = (row.get("normalized_entity") or "").lower()
        if row.get("category") != "medication":
            continue
        if not any(drug in entity for drug in ICI_DRUGS):
            continue
        parsed = _parse_evidence_date(row.get("observed_date"))
        if parsed:
            hits.append((parsed, row.get("normalized_entity") or "", row.get("observed_date") or ""))
    if not hits:
        return None
    hits.sort(key=lambda hit: hit[0])
    first = hits[0]
    last = hits[-1]
    return IciAnchor(
        first_drug=first[1],
        first_date=first[0],
        first_raw=first[2],
        last_drug=last[1],
        last_date=last[0],
        last_raw=last[2],
        occurrences=len(hits),
    )


def band_for_offset(offset: int | None) -> str:
    """Bucket a day offset from the ICI anchor into a clinical band."""
    if offset is None:
        return "undated"
    if offset < 0:
        return "pre_ici"
    if offset <= _DAYS_14:
        return "0_14d"
    if offset <= _DAYS_112:
        return "14_112d"
    if offset <= _DAYS_365:
        return "112_365d"
    return "over_365d"


def scan_for_irae(
    rows: Iterable[dict[str, Any]],
    anchor: IciAnchor | None,
) -> list[ToxicityCandidate]:
    """Layer 2 — organ lexicon scan over the registry.

    Every flagged evidence becomes a candidate dated relative to ``anchor``
    (``offset_days``) and bucketed into a time band.  Candidates are sorted by
    band (pre-ICI first, undated last) then by offset so the LLM layer reads
    them in clinical order.
    """
    candidates: list[ToxicityCandidate] = []
    for row in rows:
        text = _normalized_text(row)
        matched_organ: str | None = None
        for organ, tokens in PRIMARY_TOXICITY_LEXICON.items():
            if any(re.search(token, text) for token in tokens):
                matched_organ = organ
                break
        generic = bool(
            matched_organ is None
            and any(re.search(token, text) for token in GENERIC_SYMPTOM_TOKENS)
        )
        if matched_organ is None and not generic:
            continue
        parsed = _parse_evidence_date(row.get("observed_date"))
        offset = None
        if parsed is not None and anchor is not None:
            offset = (parsed - anchor.first_date).days
        candidates.append(
            ToxicityCandidate(
                organ=matched_organ or "Sintomi aspecifici",
                evidence_id=row.get("evidence_id") or "",
                category=row.get("category") or "",
                entity=row.get("normalized_entity") or "",
                observed_raw=row.get("observed_date") or "",
                offset_days=offset,
                band=band_for_offset(offset),
                quote=_quote_from_row(row),
                generic=generic,
                value=_value_from_row(row),
            )
        )
    band_rank = {name: index for index, name in enumerate(BAND_ORDER)}
    candidates.sort(key=lambda c: (band_rank[c.band], c.offset_days or 0))
    return candidates


def _quote_from_row(row: dict[str, Any]) -> str:
    data_json = row.get("data_json")
    if not isinstance(data_json, dict):
        return ""
    for key in ("quote_verified", "wire_normalized"):
        value = data_json.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _value_from_row(row: dict[str, Any]) -> str:
    """Compact lab value for the prompt: number, operator and unit.

    ``value_text`` wins (it is the human-readable rendering, e.g. ``>0.4``),
    then the numeric value with its unit (e.g. ``712 ng/L``).  Returns ""
    when the row is not a quantitative laboratory finding.
    """
    value_text = row.get("value_text")
    if isinstance(value_text, str) and value_text.strip():
        return value_text.strip()
    numeric_value = row.get("numeric_value")
    if numeric_value is None:
        return ""
    unit = row.get("unit")
    unit = unit.strip() if isinstance(unit, str) and unit.strip() else ""
    if isinstance(numeric_value, float) and numeric_value.is_integer():
        numeric_value = int(numeric_value)
    return f"{numeric_value} {unit}".strip()


def summarize_candidates(candidates: list[ToxicityCandidate]) -> list[dict[str, Any]]:
    """Compact per-organ counts, including band distribution (for reports)."""
    from collections import Counter

    summary: list[dict[str, Any]] = []
    by_organ: dict[str, list[ToxicityCandidate]] = {}
    for candidate in candidates:
        by_organ.setdefault(candidate.organ, []).append(candidate)
    for organ, group in by_organ.items():
        bands = Counter(candidate.band for candidate in group)
        summary.append(
            {
                "organ": organ,
                "count": len(group),
                "generic": all(c.generic for c in group),
                "bands": dict(bands),
            }
        )
    summary.sort(key=lambda item: -item["count"])
    return summary


def build_layer3_prompt(
    candidates: list[ToxicityCandidate],
    protocol_text: str,
    anchor: IciAnchor | None,
) -> str:
    """Layer 3 — the hypothesis prompt fed only the pre-filtered candidates.

    Only the candidate lines plus the anchor are included: the full registry is
    NOT sent.  Every candidate stays citable by ``[#id]`` and carries its
    latency band, so the protocol can judge new onset vs exacerbation without
    re-reading the whole timeline.
    """
    lines = []
    if anchor is not None:
        lines.append(
            f"ESPOSIZIONE ALL'IMMUNOTERAPIA: {anchor.first_drug} "
            f"dal {anchor.first_raw} ({anchor.first_date.isoformat()}); "
            f"ultima dose {anchor.last_drug} il {anchor.last_raw} "
            f"({anchor.last_date.isoformat()})."
        )
    else:
        lines.append("ESPOSIZIONE ALL'IMMUNOTERAPIA: non determinata dai farmaci datati.")
    lines.append("")
    lines.append(
        "CANDIDATI PRE-FILTRATI (screening deterministico per organo; ogni voce è "
        "citabile con il suo [#id]; offset in giorni dal primo checkpoint; "
        "le voci senza data non vanno scartate ma ancorate tramite il contesto):"
    )
    for candidate in candidates:
        offset = (
            f"d={candidate.offset_days:+d}g"
            if candidate.offset_days is not None
            else "data non risolta"
        )
        lines.append(
            f"[#{candidate.evidence_id}] [{candidate.observed_raw}] "
            f"[{candidate.band} {offset}] [{candidate.organ}] "
            f"[{candidate.category}] {candidate.entity}"
        )
        if candidate.quote:
            lines.append(f"    citazione: “{candidate.quote[:220]}”")
    lines.append("")
    lines.append(
        "Valuta SOLO i candidati elencati. Per ciascuno riporta: organo; finestra di "
        "latenza dall'inizio dell'immunoterapia; probabilità di origine immuno-correlata "
        "(CERTA_CONFERMATA | PROBABILE | POSSIBILE | IMPROBABILE | INDETERMINATA); grado "
        "CTCAE; insorgenza nuova vs riacutizzazione di condizione preesistente; "
        "cause alternative; confidenza. Distingui esplicitamente i reperti legati alla "
        "malattia tumorale (es. lesioni cutanee del melanoma) da una vera tossicità "
        "immuno-correlata. Riconciliare le voci ridondanti/duplicate con lo stesso [#id]."
    )
    lines.append("")
    lines.append("PROTOCOLLO DI ANALISI:")
    lines.append(protocol_text)
    return "\n".join(lines)

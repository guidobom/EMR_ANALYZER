"""Mapping from SNOMED semantics to atomic-evidence fact types.

SNOMED semantic tags are coarse (``(disorder)``, ``(finding)``,
``(observable entity)``...).  The atomic contract is finer, so ``(finding)``
and ``(observable entity)`` concepts are disambiguated by keyword hints on the
fully specified name.  This mapping is deliberately *permissive*: the extractor
never hard-fails on a mismatch, it only records an audit flag, and reclassifies
the fact type when a concept belongs to exactly one applicable bucket.
"""

from __future__ import annotations

from typing import Iterable

from .models import DEFAULT_LANG_ORDER, SnomedConcept


# Keyword -> fact type.  Order matters: the first fact type with a matching
# keyword wins, so radiology/histopathology/biomarker/lab/vital anchors must
# come before generic symptom/sign hints.
FSN_FACT_TYPE_HINTS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("radiology_finding", (
        "nodulo", "nodular", "opacit", "versamento", "effusion", "ispessimento",
        "thickening", "lesione", "lesion", "linfoadenopat", "adenopath",
        "consolidamento", "consolidation", "massa", "mass", "formazione",
        "formation", "metastasi", "metastasis",
    )),
    ("histopathology", (
        "melanoma", "carcinoma", "adenocarcinoma", "squamoso", "squamous",
        "breslow", "clark", "displasia", "displasia", "istotipo",
    )),
    ("biomarker", (
        "braf", "pd-l1", "cea", "mutazione", "mutation", "espressione",
        "expression", "antigene", "antigen", "immunoistochimic",
    )),
    ("vital_sign", (
        "pressione arteriosa", "blood pressure", "frequenza cardiaca",
        "heart rate", "temperatura corporea", "body temperature",
        "saturazione", "oxygen saturation", "peso corporeo", "body weight",
        "indice di massa corporea", "body mass index",
    )),
    ("laboratory_test", (
        "creatinina", "creatinine", "glicemia", "glucose", "emoglobina",
        "hemoglobin", "potassio", "potassium", "sodio", "sodium",
        "ormone tireostimolante", "thyrotropin", "velocità di filtrazione",
        "glomerular filtration", "eritrosedimentazione", "sedimentation rate",
        "proteina c reattiva", "c reactive protein", "reattiva",
    )),
    ("symptom", (
        "dolore", "pain", "tosse", "cough", "dispnea", "dyspnea", "astenia",
        "fatigue", "nausea", "vomito", "vomiting", "febbre", "fever",
        "cefalea", "headache", "vertigini", "vertigo", "prurito", "pruritus",
    )),
    ("clinical_sign", (
        "edema", "edema", "ittero", "jaundice", "soffio", "murmur",
        "linfonodo palpabile",
    )),
)

_FACT_TYPE_ORDER = (
    "medication", "laboratory_test", "radiology_finding",
    "instrumental_finding", "diagnosis", "symptom", "clinical_decision",
    "procedure", "clinical_sign", "vital_sign", "histopathology",
    "biomarker", "hospitalization", "discharge",
)

# Canonical semantic categories shared by English and Italian FSN tags.  A real
# release carries tags in the language of its descriptions, so both variants
# must resolve to the same decision.
_TAG_CANONICAL = {
    "disorder": "disorder", "disturbo": "disorder",
    "situation": "disorder", "situazione": "disorder",
    "event": "disorder", "evento": "disorder",
    "organism": "disorder", "organismo": "disorder",
    "finding": "finding", "reperto": "finding",
    "procedure": "procedure", "procedura": "procedure",
    "observable entity": "observable", "entità osservabile": "observable",
    "substance": "substance", "sostanza": "substance",
    "medicinal product": "medicinal", "prodotto medicinale": "medicinal",
    "medicinal product form": "medicinal",
    "forma di prodotto medicinale": "medicinal",
    "pharmaceutical/biologic product": "medicinal",
    "prodotto farmaceutico/prodotto biologico": "medicinal",
    "regime/therapy": "regime", "regime/terapia": "regime",
    "morphologic abnormality": "morphology",
    "struttura morfologicamente anomala": "morphology",
    "body structure": "body", "struttura corporea": "body",
}


def _ordered(fact_types: Iterable[str]) -> tuple[str, ...]:
    return tuple(sorted(set(fact_types), key=lambda ft: _FACT_TYPE_ORDER.index(
        ft
    ) if ft in _FACT_TYPE_ORDER else 99))


def fact_types_for_concept(
    concept: SnomedConcept,
    *,
    langs: Iterable[str] = DEFAULT_LANG_ORDER,
    hints: Iterable[tuple[str, tuple[str, ...]]] = FSN_FACT_TYPE_HINTS,
) -> tuple[str, ...]:
    """Return the atomic fact types a concept can represent.

    ``(disorder)`` and ``(procedure)`` concepts are unambiguous.  Findings and
    observable entities are resolved through the FSN keyword hints, defaulting
    to the broad permissive set when no hint matches.  Semantic tags are
    canonicalized across the requested languages (an Italian FSN carries
    ``(disturbo)`` where the English one carries ``(disorder)``).
    """
    fsn = concept.fsn(langs)
    lowered = (fsn or concept.display_label(langs)).casefold()
    tag_set: set[str] = set()
    for lang in langs:
        canonical = _TAG_CANONICAL.get(concept.semantic_tag((lang,)).casefold())
        if canonical:
            tag_set.add(canonical)

    if "disorder" in tag_set:
        return _ordered(("diagnosis",))
    if "procedure" in tag_set:
        return _ordered(("procedure", "clinical_decision"))
    if "medicinal" in tag_set or "substance" in tag_set or "regime" in tag_set:
        return _ordered(("medication", "clinical_decision"))
    if "morphology" in tag_set:
        return _ordered(("histopathology", "diagnosis"))
    if "body" in tag_set:
        return ()  # anatomical anchor, not an evidence bucket

    hinted = _hint_types(lowered, hints)
    if "finding" in tag_set or "observable" in tag_set:
        if hinted:
            return _ordered(hinted)
        if "observable" in tag_set:
            return _ordered((
                "vital_sign", "laboratory_test", "biomarker",
            ))
        return _ordered((
            "symptom", "clinical_sign", "radiology_finding",
            "histopathology", "biomarker", "diagnosis",
        ))

    # Unknown semantic tag: keep the union broad but never empty.
    return _ordered((
        "diagnosis", "symptom", "clinical_sign", "radiology_finding",
        "histopathology", "biomarker", "procedure", "medication",
    ))


def _hint_types(
    lowered_fsn: str,
    hints: Iterable[tuple[str, tuple[str, ...]]],
) -> tuple[str, ...]:
    result: list[str] = []
    for fact_type, keywords in hints:
        if any(keyword in lowered_fsn for keyword in keywords):
            if fact_type not in result:
                result.append(fact_type)
    return tuple(result)


def semantic_tag(concept: SnomedConcept) -> str:
    return concept.semantic_tag()

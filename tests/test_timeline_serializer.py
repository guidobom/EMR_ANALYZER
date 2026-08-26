"""Deterministic timeline: schema, ordering, dedup, filtering, serialization."""

from emr_analyzer.clinical.timeline_serializer import (
    build_timeline,
    filter_entries,
    format_compact,
)
from emr_analyzer.models.clinical_evidence import ClinicalEvidence


def _evidence(
    evidence_id: str,
    *,
    doc: str,
    entity: str,
    date: str | None,
    fact_type: str,
    category: str | None = None,
    value: float | None = None,
    unit: str | None = None,
    precision: str = "unknown",
    assertion: str = "present",
    polarity: str = "present",
    value_text: str | None = None,
    approximate: bool = False,
    flag: str | None = None,
) -> ClinicalEvidence:
    return ClinicalEvidence(
        evidence_id=evidence_id,
        patient_id="P1",
        document_id=doc,
        category=category or fact_type,
        fact_type=fact_type,
        normalized_entity=entity,
        concept_original=entity,
        canonical_label=entity,
        assertion=assertion,
        observed_date=date,
        document_date="2025-01-01",
        date_precision=precision,
        numeric_value=value,
        unit=unit,
        value_text=value_text,
        source_text=f"{entity} {value} {unit}",
        typed_payload={"flag": flag} if flag else {},
        data={
            "fact_type": fact_type,
            "polarity": polarity,
            "date_approximate": approximate,
        },
    )


def test_build_timeline_deduplicates_identical_facts_across_documents():
    items = [
        _evidence("E1", doc="DOC_A", entity="creatinina",
                  date="2023-03-14", fact_type="laboratory_test",
                  value=2.1, unit="mg/dL", precision="day"),
        _evidence("E2", doc="DOC_B", entity="creatinina",
                  date="2023-03-14", fact_type="laboratory_test",
                  value=2.1, unit="mg/dL", precision="day"),
        _evidence("E3", doc="DOC_A", entity="febbre",
                  date="2023-03-20", fact_type="symptom",
                  precision="day"),
    ]
    entries = build_timeline(items)
    assert [e.description for e in entries] == [
        "creatinina 2.1 mg/dL", "febbre",
    ]
    assert entries[0].document_id == "DOC_A"


def test_build_timeline_sorts_partial_dates_with_unknown_last():
    items = [
        _evidence("E1", doc="D", entity="anno", date="2022",
                  fact_type="diagnosis", precision="year"),
        _evidence("E2", doc="D", entity="mese", date="2023-03",
                  fact_type="diagnosis", precision="month"),
        _evidence("E3", doc="D", entity="giorno", date="2023-03-14",
                  fact_type="diagnosis", precision="day"),
        _evidence("E4", doc="D", entity="senza data", date=None,
                  fact_type="diagnosis"),
    ]
    entries = build_timeline(items, deduplicate=False)
    # Month precision counts as the start of the month; unknown dates last.
    assert [e.description for e in entries] == [
        "anno", "mese", "giorno", "senza data",
    ]


def test_filter_entries_by_type_and_date_range():
    entries = build_timeline([
        _evidence("E1", doc="D", entity="creatinina",
                  date="2023-03-14", fact_type="laboratory_test",
                  value=1.1, unit="mg/dL", precision="day"),
        _evidence("E2", doc="D", entity="ricovero",
                  date="2023-04-01", fact_type="hospitalization",
                  precision="day"),
        _evidence("E3", doc="D", entity="emoglobina",
                  date="2024-01-10", fact_type="laboratory_test",
                  value=9.0, unit="g/dL", precision="day"),
    ])
    labs = filter_entries(entries, types={"laboratory_test"})
    assert [e.description for e in labs] == [
        "creatinina 1.1 mg/dL", "emoglobina 9 g/dL",
    ]
    ranged = filter_entries(
        entries, types={"laboratory_test"},
        date_from="2023-06-01", date_to="2024-12-31",
    )
    assert [e.description for e in ranged] == ["emoglobina 9 g/dL"]


def test_format_compact_lines_include_type_value_and_reference():
    entries = build_timeline([
        _evidence("E1", doc="DOC_R", entity="creatinina",
                  date="2023-03-14", fact_type="laboratory_test",
                  value=2.1, unit="mg/dL", precision="day"),
        _evidence("E2", doc="DOC_R", entity="anemia",
                  date="2023-03", fact_type="diagnosis",
                  precision="month"),
        _evidence("E3", doc="DOC_R", entity="metastasi",
                  date="2023-06-01", fact_type="diagnosis",
                  precision="day", polarity="negated"),
    ])
    text = format_compact(entries)
    lines = text.splitlines()
    assert lines[0] == "2023-03 [mese] [diagnosi] anemia (rif: DOC_R)"
    assert lines[1] == "2023-03-14 [lab] creatinina 2.1 mg/dL (rif: DOC_R)"
    assert "non metastasi" in lines[2]


def test_format_compact_respects_limit_and_max_chars():
    entries = build_timeline([
        _evidence(f"E{i}", doc="D", entity=f"fatto{i}",
                  date="2023-01-01", fact_type="symptom",
                  precision="day")
        for i in range(10)
    ], deduplicate=False)
    assert len(format_compact(entries, limit=3).splitlines()) == 3
    short = format_compact(entries, max_chars=60)
    assert len(short) <= 60 and short


def test_approximate_and_abnormal_markers():
    entries = build_timeline([
        _evidence("E1", doc="D", entity="febbre", date="2023-03-14",
                  fact_type="symptom", precision="day",
                  approximate=True),
        _evidence("E2", doc="D", entity="pcr", date="2023-03-14",
                  fact_type="laboratory_test", value_text="positiva",
                  precision="day", flag="*"),
    ])
    text = format_compact(entries)
    assert "~2023-03-14" in text
    assert "positiva" in text

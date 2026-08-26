"""Deterministic repairs added in pipeline v12.

Covers medication polarity coherence, structured measurement recovery from
qualitative text, and date-precision audit markers.
"""

from types import SimpleNamespace

from emr_analyzer.clinical.atomic_evidence import (
    AtomicEvidenceExtractor,
    SentenceSpan,
    TextChunk,
    _flag_date_review,
    _measure_from_value_text,
)


def _extractor():
    """An extractor without LLM wiring: _to_evidence is deterministic."""
    extractor = object.__new__(AtomicEvidenceExtractor)
    extractor.llm = SimpleNamespace(model="test-model")
    return extractor


def _to_evidence(extractor, item, text, *, document_date="2025-01-01"):
    span = SentenceSpan(sentence_id=1, start=0, end=len(text), text=text)
    return extractor._to_evidence(
        item,
        patient_id="P1",
        document_id="D1",
        document_date=document_date,
        chunk=TextChunk(index=0, text=text, page_start=1),
        position=0,
        full_text=text,
        geometry=None,
        sentence_spans=[span],
    )


def _medication_item(quote, *, polarity="negated", status="ongoing"):
    return {
        "normalized_entity": "metformina",
        "source_refs": [1],
        "assertion": "present",
        "certainty": "confirmed",
        "polarity": polarity,
        "clinical_status": status,
        "category": "medication",
        "fact_type": "medication",
        "therapy": {"original_name": "metformina"},
    }


def test_ongoing_therapy_negated_by_model_is_repaired_to_present():
    text = "TD metformina 3 volte die, ramipril 5+25."
    evidence = _to_evidence(
        _extractor(), _medication_item(text), text
    )
    assert evidence is not None
    assert evidence.assertion == "present"
    assert evidence.data["polarity"] == "present"
    assert evidence.data["therapy"]["polarity_repaired"] == "negated→present"


def test_medication_with_explicit_negation_stays_negated():
    text = "Non assume ramipril."
    item = _medication_item(text)
    item["normalized_entity"] = "ramipril"
    item["therapy"] = {"original_name": "ramipril"}
    evidence = _to_evidence(_extractor(), item, text)
    assert evidence is not None
    assert evidence.data["polarity"] == "negated"
    assert "polarity_repaired" not in evidence.data.get("therapy", {})


def test_measure_buried_in_value_text_is_recovered():
    text = "Melanoma nodulare ulcerato."
    item = {
        "normalized_entity": "melanoma nodulare",
        "source_refs": [1],
        "assertion": "present",
        "certainty": "confirmed",
        "polarity": "present",
        "category": "diagnosis",
        "fact_type": "diagnosis",
        "value_text": "spessore di infiltrazione mm 23",
    }
    evidence = _to_evidence(_extractor(), item, text)
    assert evidence is not None
    assert evidence.numeric_value == 23.0
    assert evidence.unit == "mm"
    assert evidence.data["value_measurement_extracted"] == "value_text"


def test_measure_recovery_requires_a_single_candidate():
    assert _measure_from_value_text(
        "due aree con SUVmax 34.5 e una con SUVmax 28.8"
    ) == (None, None)
    assert _measure_from_value_text("145/80 mmHg") == (None, None)
    assert _measure_from_value_text("migliorata") == (None, None)


def test_measure_recovery_excluded_for_medication():
    text = "TD metformina 500 mg."
    item = _medication_item(text, polarity="present")
    item["value_text"] = "500 mg"
    item["normalized_entity"] = "metformina"
    evidence = _to_evidence(_extractor(), item, text)
    assert evidence is not None
    assert evidence.numeric_value is None
    assert "value_measurement_extracted" not in evidence.data


def test_date_review_marks_degraded_precision():
    data = {}
    _flag_date_review(
        data, "biopsia del 10/02/2016",
        SimpleNamespace(start="2016-02"),
    )
    assert "precisione degradata" in data["date_review"]


def test_date_review_marks_conflicting_date():
    data = {}
    _flag_date_review(
        data, "biopsia del 10/02/2016",
        SimpleNamespace(start="2016-03-10"),
    )
    assert "discordanza" in data["date_review"]


def test_date_review_silent_when_matching_or_ambiguous():
    data = {}
    _flag_date_review(
        data, "biopsia del 10/02/2016",
        SimpleNamespace(start="2016-02-10"),
    )
    assert "date_review" not in data
    data = {}
    _flag_date_review(
        data, "dal 10/02/2016 al 12/02/2016",
        SimpleNamespace(start="2016-02"),
    )
    assert "date_review" not in data
    data = {}
    _flag_date_review(
        data, "nessuna data completa qui",
        SimpleNamespace(start="2016-02"),
    )
    assert "date_review" not in data

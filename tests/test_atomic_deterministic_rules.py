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


def test_graph_voting_fails_loudly_when_model_produces_no_votes():
    from emr_analyzer.clinical.evidence_graph import EvidenceGraphBuilder
    from emr_analyzer.models.clinical_evidence import ClinicalEvidence

    class _DeadLlm:
        model = "dead"
        max_output_tokens = 512

        def generate_structured(self, prompt, system, schema, *, max_tokens=None):
            raise RuntimeError("server non raggiungibile")

    items = [
        ClinicalEvidence(
            evidence_id="E1", patient_id="P1", document_id="D1",
            category="symptom", normalized_entity="dispnea",
            source_text="dispnea", observed_date="2025-01-10",
        ),
        ClinicalEvidence(
            evidence_id="E2", patient_id="P1", document_id="D1",
            category="symptom", normalized_entity="astenia",
            source_text="astenia", observed_date="2025-01-10",
        ),
    ]
    builder = EvidenceGraphBuilder(_DeadLlm())
    try:
        builder.build("P1", items)
    except RuntimeError as exc:
        assert "Votazione relazioni fallita" in str(exc)
        assert "server non raggiungibile" in str(exc)
    else:
        raise AssertionError("il fallimento silenzioso non è stato bloccato")


def test_vote_batch_size_fits_small_contexts():
    from emr_analyzer.clinical.evidence_graph import _vote_batch_size

    assert _vote_batch_size(8192) == 13      # 4B voting model
    assert _vote_batch_size(16384) == 26
    assert _vote_batch_size(131072) == 32    # capped
    assert _vote_batch_size(2048) == 4       # floor


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


def _lab_item(quote, *, value, low=None, high=None, unit=None, flag=None,
              direction=None, operator=None, reference_text=None):
    return {
        "normalized_entity": "creatinina",
        "source_refs": [1],
        "assertion": "present",
        "certainty": "confirmed",
        "polarity": "present",
        "category": "laboratory_finding",
        "fact_type": "laboratory_test",
        "numeric_value": value,
        "unit": unit,
        "typed_payload": {
            "reference_low": low,
            "reference_high": high,
            "reference_text": reference_text,
            "flag": flag,
            "abnormal_direction": direction,
            "operator": operator,
        },
    }


def test_narrative_lab_value_in_range_without_flags_is_dropped():
    text = "Creatinina 1.1 mg/dL (0.6 - 1.2)."
    evidence = _to_evidence(
        _extractor(),
        _lab_item(
            text, value=1.1, low=0.6, high=1.2, unit="mg/dL",
            reference_text="0.6 - 1.2",
        ),
        text,
    )
    assert evidence is None


def test_narrative_lab_value_out_of_range_is_kept():
    text = "Creatinina 1.9 mg/dL (0.6 - 1.2)."
    evidence = _to_evidence(
        _extractor(),
        _lab_item(
            text, value=1.9, low=0.6, high=1.2, unit="mg/dL",
            reference_text="0.6 - 1.2",
        ),
        text,
    )
    assert evidence is not None
    assert evidence.numeric_value == 1.9


def test_narrative_lab_value_in_range_with_flag_is_kept():
    text = "Creatinina 1.1 mg/dL (0.6 - 1.2) H."
    evidence = _to_evidence(
        _extractor(),
        _lab_item(
            text, value=1.1, low=0.6, high=1.2, unit="mg/dL",
            reference_text="0.6 - 1.2", flag="H",
        ),
        text,
    )
    assert evidence is not None


def test_narrative_lab_value_with_unit_mismatch_is_kept():
    text = "Creatinina 1.1 mg/dL (0.6 - 1.2 g/L)."
    evidence = _to_evidence(
        _extractor(),
        _lab_item(
            text, value=1.1, low=0.6, high=1.2, unit="mg/dL",
            reference_text="0.6 - 1.2 g/L",
        ),
        text,
    )
    assert evidence is not None


def test_narrative_lab_one_sided_range_drops_normal_value():
    text = "TSH 0.4 mUI/L (< 0.5)."
    evidence = _to_evidence(
        _extractor(),
        _lab_item(
            text, value=0.4, high=0.5, unit="mUI/L",
            reference_text="< 0.5",
        ),
        text,
    )
    assert evidence is None


def test_narrative_lab_operator_bound_result_is_never_dropped():
    text = "TSH < 0.5 mUI/L."
    evidence = _to_evidence(
        _extractor(),
        _lab_item(
            text, value=0.5, high=0.5, unit="mUI/L", operator="<",
        ),
        text,
    )
    assert evidence is not None


def test_narrative_lab_abnormality_wording_overrides_range():
    text = "Creatinina aumentata 1.1 mg/dL (0.6 - 1.2)."
    evidence = _to_evidence(
        _extractor(),
        _lab_item(
            text, value=1.1, low=0.6, high=1.2, unit="mg/dL",
            reference_text="0.6 - 1.2",
        ),
        text,
    )
    assert evidence is not None


def test_narrative_lab_unreadable_range_is_kept():
    text = "Creatinina 1.1 mg/dL."
    evidence = _to_evidence(
        _extractor(),
        _lab_item(text, value=1.1, unit="mg/dL", low="n.d.", high="n.d."),
        text,
    )
    assert evidence is not None


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

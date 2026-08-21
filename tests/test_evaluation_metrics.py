from emr_analyzer.evaluation.metrics import (
    aggregate_patient_metrics,
    evaluate_patient_events,
)


def test_event_metrics_cover_temporality_certainty_and_citations():
    gold = [{
        "category": "symptom", "canonical_entity": "dispnea",
        "first_evidence_date": "2025-01", "certainty": "confirmed",
        "status": "active", "evidence_ids": ["E1", "E2"],
    }]
    predicted = [{
        "category": "symptom", "canonical_entity": "dispnea da sforzo",
        "first_evidence_date": "2025-01-12", "certainty": "confirmed",
        "status": "active", "evidence_ids": ["E1", "E2"],
    }]
    result = evaluate_patient_events(gold, predicted)
    assert result["f1"] == 1.0
    assert result["temporal_compatible_accuracy"] == 1.0
    assert result["citation_completeness"] == 1.0
    aggregate = aggregate_patient_metrics([result])
    assert aggregate["micro_f1"] == 1.0

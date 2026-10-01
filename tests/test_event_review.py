"""Reviewer decisions: persistence across extractions, text changes and FHIR."""

import json
from dataclasses import replace

import pytest

from emr_analyzer.models.clinical_registry import DocumentTextOverlay
from tests.helpers import FakeLlm, KeywordExtractor, Project

TEXT = "Riferisce tosse da tre giorni. Nega febbre. Prosegue nivolumab."


@pytest.fixture
def setup(workspace):
    project = Project(workspace)
    project.add_patient("P001")
    project.add_document("P001", "DOC_001", TEXT)
    extractor = KeywordExtractor()
    pipeline = project.pipeline(FakeLlm(), extractor)
    pipeline.extract_patient("P001")
    yield project, pipeline, extractor
    project.close()


def by_label(events):
    return {event.label: event for event in events}


def test_decisions_are_applied_and_survive_reextraction(setup):
    project, pipeline, extractor = setup
    review = pipeline.review_service()
    events = by_label(review.events("P001"))
    review.confirm(events["tosse"])
    review.correct(events["febbre"], {"label": "Febbre", "assertion": "absent"})
    review.reject(events["nivolumab"])

    pipeline.extract_patient("P001", incremental=False)      # same occurrences extracted again
    events = by_label(review.events("P001"))
    assert events["tosse"].status == "confirmed"
    assert (events["Febbre"].status, events["Febbre"].assertion) == ("corrected", "absent")
    assert events["nivolumab"].status == "rejected"
    assert "nivolumab" not in by_label(pipeline.events("P001"))      # excluded downstream

    review.restore(events["nivolumab"])
    assert by_label(review.events("P001"))["nivolumab"].status == "proposed"


def test_confirmation_lapses_when_the_occurrence_changes(setup):
    project, pipeline, extractor = setup
    review = pipeline.review_service()
    review.confirm(by_label(review.events("P001"))["tosse"])
    stored = project.evidence.get_by_document("DOC_001")
    changed = [replace(item, assertion="absent") if item.normalized_entity == "tosse" else item
               for item in stored]
    pipeline.replace_document_evidence("DOC_001", changed)
    event = by_label(review.events("P001"))["tosse"]
    assert event.status == "needs_review" and "assertion" in event.note


def test_corrected_event_is_kept_when_the_occurrence_disappears(setup):
    project, pipeline, extractor = setup
    review = pipeline.review_service()
    review.correct(by_label(review.events("P001"))["tosse"], {"label": "Tosse secca"})
    pipeline.replace_document_evidence("DOC_001", [])          # nothing extracted any more
    events = by_label(review.events("P001"))
    assert set(events) == {"Tosse secca"} and events["Tosse secca"].status == "corrected"


def test_fragment_moves_with_the_text_or_is_flagged(setup):
    project, pipeline, extractor = setup
    review = pipeline.review_service()
    events = by_label(review.events("P001"))
    review.correct(events["tosse"], {"label": "Tosse"})
    review.correct(events["febbre"], {"label": "Febbre"})
    project.overlays.save(DocumentTextOverlay(
        patient_id="P001", document_id="DOC_001", base_text_hash="x",
        corrected_text="Da tre giorni riferisce tosse. Nega rialzo termico. Prosegue nivolumab."))
    pipeline.replace_document_evidence("DOC_001", [])
    events = by_label(review.events("P001"))
    text = review.text("P001", "DOC_001")
    assert text[events["Tosse"].start:events["Tosse"].end] == "tosse" and not events["Tosse"].stale
    assert events["Febbre"].stale


def test_new_fragment_and_manual_event(setup):
    project, pipeline, extractor = setup
    review = pipeline.review_service()
    event = by_label(review.events("P001"))["tosse"]
    start = TEXT.index("tosse da tre giorni")
    review.correct(event, {"start": start, "end": start + len("tosse da tre giorni")})
    assert by_label(review.events("P001"))["tosse"].quote == "tosse da tre giorni"
    with pytest.raises(ValueError):
        review.correct(event, {"start": 5, "end": 2})

    manual = review.add("P001", "DOC_001", TEXT.index("tre giorni"), TEXT.index("tre giorni") + 10,
                        label="Durata della tosse", fact_type="symptom")
    assert manual.key.startswith("manual:")
    assert by_label(review.events("P001"))["Durata della tosse"].status == "added"
    review.reject(by_label(review.events("P001"))["Durata della tosse"])     # deletes a manual event
    assert "Durata della tosse" not in by_label(review.events("P001"))


def test_fhir_reflects_review_with_stable_ids(setup, workspace):
    project, pipeline, extractor = setup
    path = workspace / "P001" / "clinical_events.fhir.json"
    read = lambda: json.loads(path.read_text(encoding="utf-8"))["entry"]
    ids_before = {entry["resource"]["id"] for entry in read()}
    review = pipeline.review_service()
    events = by_label(review.events("P001"))
    review.confirm(events["tosse"])
    review.reject(events["nivolumab"])
    pipeline.extract_patient("P001")                           # nothing to extract: FHIR rewritten
    entries = read()
    resources = {entry["resource"]["id"]: entry["resource"] for entry in entries}
    assert not any(r["resourceType"] == "MedicationStatement" for r in resources.values())
    statuses = {json.loads(next(e["valueString"] for e in r["extension"]
                                if e["url"].endswith("extraction-context")))["state"] or r["id"]:
                next(e["valueString"] for e in r["extension"] if e["url"].endswith("review-status"))
                for r in resources.values() if any(e["url"].endswith("review-status") for e in r.get("extension", []))}
    assert "confirmed" in statuses.values()
    survivors = {entry["resource"]["id"] for entry in entries}
    assert len(survivors & ids_before) >= 2                    # tosse and febbre kept their ids

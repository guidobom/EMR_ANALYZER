"""The certified statement reaches its copies with the copy's own offsets."""

import pytest

from emr_analyzer.clinical.grounded_sources import METHOD
from emr_analyzer.clinical.statement_index import assign_roles, index_document
from emr_analyzer.clinical.statement_projection import (StatementProjectionService,
                                                        anchor_offsets, same_text)
from emr_analyzer.database.statement_repo import (StatementAnnotationRepository,
                                                 StatementIndexRepository)
from tests.helpers import FakeLlm, KeywordExtractor, Project

SHARED = "Anamnesi:\n\nRiferisce tosse da tre giorni.\n\n"
EARLY = SHARED + "Nega febbre."
LATE = SHARED + "Prosegue nivolumab."


@pytest.fixture
def project(workspace):
    project = Project(workspace)
    project.add_patient("P001")
    project.add_document("P001", "DOC_001", EARLY, date="2026-09-01")
    project.add_document("P001", "DOC_002", LATE, date="2026-10-01")
    yield project
    project.close()


def test_anchor_offsets_rejects_an_impossible_interval():
    assert anchor_offsets("Riferisce tosse da tre giorni.", 2, 2) == (0, 10, 15)
    assert anchor_offsets("Riferisce tosse.", 4, 4) is None
    assert anchor_offsets("Riferisce tosse.", "x", 1) is None
    assert same_text("a  b", "a b") and not same_text("a b", "a c")


def test_copies_receive_the_projected_event_with_their_own_offsets(project):
    extractor = KeywordExtractor()
    result = project.pipeline(FakeLlm(), extractor).extract_patient("P001")

    # Both the heading and the shared sentence repeat; only the sentence carries a fact.
    assert result["statement_copies"] == 2 and result["statement_projection_misses"] == 0
    stored = [row for row in project.evidence.get_by_patient("P001")
              if row.extraction_method == METHOD]
    copies = [row for row in stored
              if (row.data.get("statement_reuse") or {}).get("role") == "copy"]
    assert len(copies) == 1
    copy = copies[0]
    # The fragment belongs to the copy's document, at the copy's own offsets.
    assert copy.document_id == "DOC_002" and copy.document_date == "2026-10-01"
    span = copy.data["source_spans"][0]
    text = (project.root / "P001" / "extraction" / "DOC_002.md").read_text(encoding="utf-8")
    assert text[span["start"]:span["end"]] == copy.source_text == "tosse"
    assert copy.data["statement_reuse"]["carrier_document_id"] == "DOC_001"
    # No model call happened for the copy's sentence.
    assert sorted(extractor.calls) == ["DOC_001", "DOC_002"]
    assert {row.normalized_entity for row in stored} == {"tosse", "febbre", "nivolumab"}


def test_projection_is_idempotent(project):
    pipeline = project.pipeline(FakeLlm(), KeywordExtractor())
    pipeline.extract_patient("P001")
    first = {row.evidence_id for row in project.evidence.get_by_patient("P001")}
    pipeline.extract_patient("P001", incremental=False)
    assert {row.evidence_id for row in project.evidence.get_by_patient("P001")} == first


def test_a_changed_copy_sentence_is_not_projected(workspace):
    project = Project(workspace)
    project.add_patient("P002")
    project.add_document("P002", "DOC_001", EARLY, date="2026-09-01")
    # Same previous sentence and heading, but the words that carry the fact differ.
    project.add_document("P002", "DOC_002", "Anamnesi:\n\nRiferisce tosse da tre settimane.\n",
                         date="2026-10-01")
    try:
        pipeline = project.pipeline(FakeLlm(), KeywordExtractor())
        result = pipeline.extract_patient("P002")
        # The statements are different, so the second document is annotated in its own right.
        # Only the heading repeats; the two sentences are distinct statements.
        assert result["statement_copies"] == 1 and result["statement_projection_misses"] == 0
        assert {row.document_id for row in project.evidence.get_by_patient("P002")} == {"DOC_001", "DOC_002"}
    finally:
        project.close()


def test_a_missing_annotation_is_a_miss_not_a_lost_row(workspace):
    project = Project(workspace)
    project.add_patient("P003")
    project.add_document("P003", "DOC_001", EARLY, date="2026-09-01")
    project.add_document("P003", "DOC_002", LATE, date="2026-10-01")
    try:
        index = StatementIndexRepository(project.db)
        index.rebuild("P003", [("DOC_001", "2026-09-01", EARLY), ("DOC_002", "2026-10-01", LATE)])
        service = StatementProjectionService(index, StatementAnnotationRepository(project.db))
        rows, misses = service.project(patient_id="P003", document_id="DOC_002", text=LATE,
                                       document_date="2026-10-01", document_type="visita_oncologica",
                                       model_digest="m1", prompt_version="v9")
        assert rows == []
        assert misses and misses[0]["carrier_document_id"] == "DOC_001"
    finally:
        project.close()



def test_one_resource_with_one_provenance_per_document(project):
    import json

    pipeline = project.pipeline(FakeLlm(), KeywordExtractor())
    pipeline.extract_patient("P001")
    bundle = json.loads((project.root / "P001" / "clinical_events.fhir.json").read_text(encoding="utf-8"))
    resources = [entry["resource"] for entry in bundle["entry"]]
    observations = [item for item in resources
                    if item["resourceType"] == "Observation" and item["code"]["text"] == "tosse"]
    assert len(observations) == 1, "the repeated statement is one resource"
    target = "urn:uuid:" + observations[0]["id"]
    provenance = [item for item in resources if item["resourceType"] == "Provenance"
                  and target in provenance_target(item)]
    assert len(provenance) == 2, "one provenance per report that mentions it"
    # Both reports are cited as sources of the same resource.
    documents = {item["entity"][0]["what"]["reference"] for item in provenance}
    assert len(documents) == 2


def provenance_target(item):
    return [target.get("reference", "") for target in item.get("target", [])]


def test_a_correction_splits_the_resource(project):
    from emr_analyzer.database.event_override_repo import EventOverrideRepository
    from emr_analyzer.clinical.event_review import EventReviewService

    pipeline = project.pipeline(FakeLlm(), KeywordExtractor())
    pipeline.extract_patient("P001")
    review = EventReviewService(project.evidence, EventOverrideRepository(project.db), None,
                                overlay_repo=project.overlays, audit_repo=None, snomed=None)
    copy = next(event for event in review.events("P001")
                if event.document_id == "DOC_002" and event.label == "tosse")
    review.correct(copy, {"label": "Tosse risolta"})
    pipeline.write_fhir("P001", project.documents.list_by_patient("P001"), {}, use_llm=False)
    import json
    bundle = json.loads((project.root / "P001" / "clinical_events.fhir.json").read_text(encoding="utf-8"))
    resources = [entry["resource"] for entry in bundle["entry"]]
    labels = {item["code"]["text"] for item in resources
              if item["resourceType"] == "Observation"}
    assert labels == {"tosse", "Tosse risolta", "febbre"}, "the corrected copy is its own fact"


def test_a_decision_can_reach_every_copy(project):
    from emr_analyzer.clinical.event_review import EventReviewService
    from emr_analyzer.database.event_override_repo import EventOverrideRepository

    pipeline = project.pipeline(FakeLlm(), KeywordExtractor())
    pipeline.extract_patient("P001")
    review = EventReviewService(project.evidence, EventOverrideRepository(project.db), None,
                                overlay_repo=project.overlays, audit_repo=None, snomed=None)
    events = [event for event in review.events("P001") if event.label == "tosse"]
    assert len(events) == 2 and len({event.statement_key for event in events}) == 1
    # The two copies come from documents extracted concurrently: order is not promised.
    assert sorted(event.document_id for event in events) == ["DOC_001", "DOC_002"]

    review.correct(events[0], {"certainty": "suspected"}, fanout=True)

    after = [event for event in review.events("P001") if event.label == "tosse"]
    assert {event.certainty for event in after} == {"suspected"}
    assert {event.status for event in after} == {"corrected"}
    # Each copy keeps its own document and fragment.
    assert {event.document_id for event in after} == {"DOC_001", "DOC_002"}
    overrides = EventOverrideRepository(project.db).by_patient("P001")
    assert len(overrides) == 2

    review.restore(after[0], fanout=True)
    assert not EventOverrideRepository(project.db).by_patient("P001")


def test_a_fragment_change_never_propagates_to_the_copies(project):
    import pytest as _pytest

    from emr_analyzer.clinical.event_review import EventReviewService
    from emr_analyzer.database.event_override_repo import EventOverrideRepository

    pipeline = project.pipeline(FakeLlm(), KeywordExtractor())
    pipeline.extract_patient("P001")
    review = EventReviewService(project.evidence, EventOverrideRepository(project.db), None,
                                overlay_repo=project.overlays, audit_repo=None, snomed=None)
    event = next(item for item in review.events("P001") if item.label == "tosse")
    with _pytest.raises(ValueError):
        review.correct(event, {"start": 0, "end": 5}, fanout=True)

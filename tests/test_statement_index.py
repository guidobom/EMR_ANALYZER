"""Statement identity, roles and the two repositories built on them."""

import pytest

from emr_analyzer.clinical.statement_index import (assign_roles, document_digest,
                                                  index_document, is_heading)
from emr_analyzer.database.statement_repo import (StatementAnnotationRepository,
                                                 StatementIndexRepository)
from tests.helpers import Project


@pytest.fixture
def project(workspace):
    project = Project(workspace)
    yield project
    project.close()

TEXT = "Anamnesi:\n\nIl paziente riferisce diarrea da circa una settimana.\nNon assume farmaci.\n"


def occurrences(text, document_id="DOC_A", date="2026-10-01"):
    return index_document(document_id, date, text)


def test_identity_is_text_and_context_only():
    first = occurrences(TEXT)
    later = index_document("DOC_B", "2026-11-01", TEXT)
    assert [item.statement_key for item in first] == [item.statement_key for item in later]
    # Whitespace alone never changes a statement.
    spaced = index_document("DOC_C", "2026-11-01", TEXT.replace("  ", " ").replace("\n", " \n "))
    assert [item.statement_key for item in first] == [item.statement_key for item in spaced]


def test_changed_previous_sentence_or_heading_changes_the_key():
    base = index_document("DOC_A", "2026-10-01", "Non assume farmaci.\nIl paziente ha febbre.")
    other = index_document("DOC_B", "2026-11-01", "Assume anticoagulanti.\nIl paziente ha febbre.")
    assert base[-1].statement_key != other[-1].statement_key
    heading = index_document("DOC_C", "2026-11-01", "Terapia:\nIl paziente ha febbre.")
    assert heading[-1].statement_key != base[-1].statement_key


def test_header_is_metadata_not_a_source():
    text = ("<!-- emr-report-date:v1 -->\nData referto: 2026-10-01.\n"
            "<!-- /emr-report-date -->\n\nIl paziente ha febbre.")
    items = occurrences(text)
    assert [item.text for item in items] == ["Il paziente ha febbre."]


def test_roles_follow_clinical_order_and_allow_intra_document_copies():
    block = "Terapia:\n\nProsegue nivolumab.\n"
    early = index_document("DOC_B", "2026-01-01", block)
    late = index_document("DOC_C", "2026-02-01", block + "\nPoi.\n\n" + block)
    roles = assign_roles(early + late)
    by_document = {}
    for item in roles:
        by_document.setdefault(item.document_id, []).append(item)
    for items in by_document.values():
        items.sort(key=lambda item: item.ordinal)
    assert [item.role for item in by_document["DOC_B"]] == ["origin", "origin"]
    # The repeated block is a copy; "Poi." is new, and the second "Terapia:"
    # heading follows a different sentence, so it is a statement of its own.
    assert [item.role for item in by_document["DOC_C"]] == ["copy", "copy", "origin", "origin", "copy"]
    assert by_document["DOC_C"][1].carrier_document_id == "DOC_B"
    assert by_document["DOC_C"][3].carrier_document_id == "DOC_C"
    assert by_document["DOC_C"][4].carrier_document_id == "DOC_B"


def test_digest_changes_when_the_carrier_changes():
    carried = assign_roles(index_document("DOC_B", "2026-01-01", TEXT)
                           + index_document("DOC_C", "2026-02-01", TEXT))
    as_copy = document_digest([item for item in carried if item.document_id == "DOC_C"])
    alone = document_digest(assign_roles(index_document("DOC_C", "2026-02-01", TEXT)))
    assert as_copy != alone
    assert as_copy == document_digest(
        [item for item in carried if item.document_id == "DOC_C"])


def test_headings_are_recognised():
    assert is_heading("Anamnesi:")
    assert is_heading("12/03/2021")
    assert is_heading("ESAME OBIETTIVO")
    assert not is_heading("Il paziente ha febbre da tre giorni e non assume farmaci.")


def test_repositories_round_trip_and_cascade(project):
    project.add_patient("P1")
    project.add_document("P1", "DOC_A", TEXT, date="2026-01-01")
    project.add_document("P1", "DOC_B", TEXT, date="2026-02-01")
    index = StatementIndexRepository(project.db)
    grouped = index.rebuild("P1", [("DOC_A", "2026-01-01", TEXT), ("DOC_B", "2026-02-01", TEXT)])
    assert {item.role for item in grouped["DOC_B"]} == {"copy"}
    counts = index.counts("P1")
    assert counts["copies"] == 3 and counts["origins"] == 3
    assert counts["copy_chars"] > 0
    assert index.digest("DOC_B") == document_digest(grouped["DOC_B"])

    carrier = next(item for item in grouped["DOC_A"] if item.role == "origin")
    annotations = StatementAnnotationRepository(project.db)
    annotations.save(patient_id="P1", statement_key=carrier.statement_key,
                     carrier_occurrence_id=carrier.occurrence_id, carrier_document_id="DOC_A",
                     carrier_document_date="2026-01-01", carrier_sentence=carrier.text,
                     status="completed", payload=[{"label": "Diarrea"}], model_digest="m1",
                     prompt_version="v9")
    assert annotations.get("P1", carrier.statement_key, "m1", "v9")["payload"][0]["label"] == "Diarrea"
    assert annotations.get("P1", carrier.statement_key, "m2", "v9") is None
    assert len(annotations.by_carrier_document("DOC_A")) == 1

    # Deleting the carrier removes its certification; the index follows the documents.
    with project.db:
        project.db.execute("DELETE FROM documents WHERE id='DOC_A'")
    assert annotations.by_carrier_document("DOC_A") == []
    assert index.for_document("DOC_A") == []


def test_rebuild_replaces_previous_rows(project):
    project.add_patient("P2")
    project.add_document("P2", "DOC_A", TEXT, date="2026-01-01")
    index = StatementIndexRepository(project.db)
    index.rebuild("P2", [("DOC_A", "2026-01-01", TEXT)])
    assert len(index.for_document("DOC_A")) == 3
    index.rebuild("P2", [("DOC_A", "2026-01-01", "Il paziente ha febbre.")])
    assert len(index.for_document("DOC_A")) == 1


def test_planner_sends_only_origins_and_marks_copies():
    from emr_analyzer.clinical.sentence_groups import (plan_statement_groups,
                                                       statement_group_plan)

    roles = {1: "origin", 2: "origin", 3: "origin", 4: "origin", 5: "copy"}
    groups = plan_statement_groups(TEXT + "\nTerapia:\n\nProsegue nivolumab.\n", roles,
                                   lambda value: max(1, len(value.encode()) // 3), 1000)
    assert groups and all(5 not in group.target_ids for group in groups)
    assert any(5 in group.copy_ids for group in groups)
    prompt = groups[0].prompt("visita_oncologica", "2026-10-01")
    assert "COPIA [S5]" in prompt and "non estrarne nulla" in prompt
    # A document made only of copies is never sent to the model.
    planned, skipped = statement_group_plan(TEXT, {n: "copy" for n in range(1, 4)},
                                            lambda value: 10, 1000)
    assert planned == [] and skipped == 1


def test_split_group_keeps_a_copy_on_one_side_only():
    from emr_analyzer.clinical.sentence_groups import SentenceGroup, split_group
    from emr_analyzer.clinical.evidence_utils import SentenceSpan

    spans = [SentenceSpan(n, n * 10, n * 10 + 5, f"S{n}.") for n in range(1, 6)]
    group = SentenceGroup(targets=(spans[0], spans[1], spans[2], spans[3]),
                          context=(spans[4],), copies=(spans[0],))
    left, right = split_group(group)
    assert left.copies == (spans[0],) and right.copies == ()
    assert left.targets == (spans[0], spans[1]) and right.targets == (spans[2], spans[3])


def test_invalidation_forgets_what_a_document_contributed(project):
    from emr_analyzer.clinical.grounded_sources import METHOD
    from emr_analyzer.database.statement_repo import invalidate_documents
    from emr_analyzer.models.clinical_evidence import ClinicalEvidence

    project.add_patient("P004")
    project.add_document("P004", "DOC_A", TEXT, date="2026-01-01")
    project.add_document("P004", "DOC_B", TEXT, date="2026-02-01")
    index = StatementIndexRepository(project.db)
    index.rebuild("P004", [("DOC_A", "2026-01-01", TEXT), ("DOC_B", "2026-02-01", TEXT)])
    carrier = next(item for item in index.for_document("DOC_A") if item.role == "origin")
    annotations = StatementAnnotationRepository(project.db)
    annotations.save(patient_id="P004", statement_key=carrier.statement_key,
                     carrier_occurrence_id=carrier.occurrence_id, carrier_document_id="DOC_A",
                     carrier_document_date="2026-01-01", carrier_sentence=carrier.text,
                     status="completed", payload=[], model_digest="m1", prompt_version="v9")
    project.evidence.insert_batch([ClinicalEvidence(
        patient_id="P004", document_id="DOC_A", category="symptom",
        normalized_entity="tosse", source_text="tosse", evidence_id="EVD_move",
        extraction_method=METHOD, data={"fhir_pipeline": True})])

    invalidate_documents(project.db, ["DOC_A"])

    assert index.for_document("DOC_A") == []
    assert annotations.by_carrier_document("DOC_A") == []
    assert [row for row in project.evidence.get_by_document("DOC_A")
            if row.extraction_method == METHOD] == []
    # The other document of the patient is untouched.
    assert len(index.for_document("DOC_B")) == 3

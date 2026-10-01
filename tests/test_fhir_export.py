"""FHIR files follow the database: exports, NDJSON and invalidation."""

import json

import pytest

from emr_analyzer.models.lab_result import LabValue
from tests.helpers import FakeLlm, KeywordExtractor, Project


@pytest.fixture
def setup(workspace):
    project = Project(workspace)
    for patient in ("P001", "P002"):
        project.add_patient(patient)
    project.add_document("P001", "DOC_001", "Riferisce tosse. Nega febbre.")
    project.add_document("P001", "DOC_002", "Prosegue nivolumab.", date="2026-09-20")
    project.add_document("P002", "DOC_003", "Riferisce febbre.")
    project.labs.insert(LabValue(patient_id="P001", document_id="DOC_001", parameter_name="Creatinina",
                                 normalized_name="creatinina", value=1.9, unit="mg/dL"))
    pipeline = project.pipeline(FakeLlm(), KeywordExtractor())
    pipeline.extract_patients(["P001", "P002"])
    yield project, pipeline
    project.close()


def fhir_path(workspace, patient):
    return workspace / patient / "clinical_events.fhir.json"


def test_patient_export_reflects_the_review(setup, workspace):
    project, pipeline = setup
    review = pipeline.review_service()
    tosse = next(event for event in review.events("P001") if event.label == "tosse")
    review.reject(tosse)
    result = pipeline.export_patient("P001")
    bundle = json.loads(fhir_path(workspace, "P001").read_text(encoding="utf-8"))
    texts = [entry["resource"].get("code", {}).get("text") for entry in bundle["entry"]]
    assert "tosse" not in texts and "febbre" in texts
    assert result["events"] == 3            # febbre, nivolumab and the laboratory result


def test_project_ndjson_has_each_resource_once(setup, workspace):
    project, pipeline = setup
    result = pipeline.export_project()
    lines = [json.loads(line) for line in open(result["path"], encoding="utf-8")]
    identities = [(resource["resourceType"], resource["id"]) for resource in lines]
    assert len(identities) == len(set(identities)) == result["resources"]
    kinds = [resource["resourceType"] for resource in lines]
    assert kinds.count("Patient") == 2 and kinds.count("Device") == 1
    assert result["patients"] == 2 and result["events"] == 5
    assert fhir_path(workspace, "P002").is_file()


def test_document_deletion_invalidates_the_patient_file(setup, workspace):
    from emr_analyzer.clinical.document_deletion import DocumentDeletionService

    project, pipeline = setup
    assert fhir_path(workspace, "P001").is_file()
    result = DocumentDeletionService(project.db, project.documents, workspaces_dir=workspace).delete("DOC_002")
    assert result.deleted, result.error
    assert not fhir_path(workspace, "P001").exists()
    pipeline.export_patient("P001")
    bundle = json.loads(fhir_path(workspace, "P001").read_text(encoding="utf-8"))
    assert not any(entry["resource"]["resourceType"] == "MedicationStatement" for entry in bundle["entry"])


def test_reattribution_moves_review_and_status_with_the_document(setup, workspace):
    from emr_analyzer.clinical.document_reattribution import DocumentReattributionService
    from emr_analyzer.database.audit_repo import AuditRepository

    project, pipeline = setup
    review = pipeline.review_service()
    nivolumab = next(event for event in review.events("P001") if event.label == "nivolumab")
    review.confirm(nivolumab)
    service = DocumentReattributionService(project.db, project.documents, project.patients,
                                           AuditRepository(project.db), workspaces_dir=workspace)
    result = service.move_document("DOC_002", "P002")
    assert result.ok, result.error
    assert not fhir_path(workspace, "P001").exists() and not fhir_path(workspace, "P002").exists()
    moved = {event.label: event for event in review.events("P002")}
    assert moved["nivolumab"].status == "confirmed"
    assert "nivolumab" not in {event.label for event in review.events("P001")}
    # The report is read again in its new patient: the events it contributed
    # there depend on that patient's statements, not on the old ones.
    manifest = project.db.execute("SELECT patient_id FROM processing_manifest WHERE document_id='DOC_002'")
    assert {row[0] for row in manifest} == set()
    assert not [row for row in project.evidence.get_by_document("DOC_002")]

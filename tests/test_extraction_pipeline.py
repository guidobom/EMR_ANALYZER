"""Orchestration of one patient's extraction: persistence, resume and FHIR."""

import json

import pytest

from emr_analyzer.clinical.grounded_sources import METHOD
from tests.helpers import FakeLlm, KeywordExtractor, Project


@pytest.fixture
def project(workspace):
    project = Project(workspace)
    project.add_patient("P001")
    project.add_document("P001", "DOC_001", "Riferisce tosse da tre giorni. Nega febbre.")
    project.add_document("P001", "DOC_002", "Prosegue nivolumab.", date="2026-09-15")
    yield project
    project.close()


def test_extracts_persists_and_writes_fhir(project, workspace):
    extractor = KeywordExtractor()
    result = project.pipeline(FakeLlm(), extractor).extract_patient("P001")

    assert result["documents_processed"] == 2 and result["documents_failed"] == 0
    stored = [item for item in project.evidence.get_by_patient("P001") if item.extraction_method == METHOD]
    assert sorted(item.normalized_entity for item in stored) == ["febbre", "nivolumab", "tosse"]
    bundle = json.loads((workspace / "P001" / "clinical_events.fhir.json").read_text(encoding="utf-8"))
    kinds = sorted(entry["resource"]["resourceType"] for entry in bundle["entry"])
    assert kinds.count("Provenance") == 3 and "MedicationStatement" in kinds


def test_second_run_skips_current_documents(project):
    extractor = KeywordExtractor()
    pipeline = project.pipeline(FakeLlm(), extractor)
    pipeline.extract_patient("P001")
    result = pipeline.extract_patient("P001")

    assert result["documents_processed"] == 0 and result["documents_skipped"] == 2
    assert sorted(extractor.calls) == ["DOC_001", "DOC_002"]


def test_text_overlay_makes_only_that_document_stale(project):
    from emr_analyzer.models.clinical_registry import DocumentTextOverlay

    extractor = KeywordExtractor()
    pipeline = project.pipeline(FakeLlm(), extractor)
    pipeline.extract_patient("P001")
    project.overlays.save(DocumentTextOverlay(
        patient_id="P001", document_id="DOC_002",
        corrected_text="Sospende nivolumab per febbre.", base_text_hash="x"))
    extractor.calls.clear()
    result = pipeline.extract_patient("P001")

    assert extractor.calls == ["DOC_002"] and result["documents_skipped"] == 1


def test_missing_model_reports_failures_without_extracting(project):
    extractor = KeywordExtractor()

    class Offline(FakeLlm):
        is_available = False

    result = project.pipeline(Offline(), extractor).extract_patient("P001")
    assert result["documents_failed"] == 2 and extractor.calls == []

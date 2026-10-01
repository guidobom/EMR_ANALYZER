"""Regression tests for laboratory values, deletion and stable FHIR identities."""

from dataclasses import replace

import pytest

from emr_analyzer.clinical.fhir_registry import FhirRegistry, lab_occurrence_keys
from emr_analyzer.extraction.lab_parser import LabParser
from emr_analyzer.extraction.normalizer import LabNormalizer
from emr_analyzer.models.lab_result import LabValue
from emr_analyzer.utils.text_utils import normalize_italian_number
from tests.helpers import Project


@pytest.mark.parametrize("text,expected", [
    ("-3.2", -3.2), ("-3,2", -3.2), ("−4,5", -4.5), ("+3,2", 3.2),
    ("1.234,56", 1234.56), ("-0,5 mEq/L", -0.5), ("   ", None), ("abc", None),
])
def test_numbers_keep_their_sign(text, expected):
    assert normalize_italian_number(text) == expected


@pytest.mark.parametrize("value,low,high,operator,expected", [
    (100, 0, 50, "<", (False, None)),     # <100 could be 40 or 80: undecidable
    (5, 10, 50, "<", (True, "L")),        # every value below 5 is below 10
    (60, 0, 50, ">", (True, "H")),
    (10, 20, 100, ">", (False, None)),
    (60, 0, 50, None, (True, "H")),
    (50, 0, 50, None, (False, None)),
])
def test_censored_results_are_flagged_only_when_decidable(value, low, high, operator, expected):
    assert LabNormalizer().is_abnormal(value, low, high, operator) == expected


def test_parser_keeps_comparator_and_negative_ranges():
    parser = LabParser()
    censored, = parser.parse("Glucosio <100 mg/dL (0-50)", patient_id="P", document_id="D")
    assert (censored.value, censored.operator, censored.is_abnormal) == (100.0, "<", False)
    negative, = parser.parse("Base excess -3,2 mmol/L (-2 - 2)", patient_id="P", document_id="D")
    assert (negative.value, negative.flag) == (-3.2, "L")


@pytest.fixture
def project(workspace):
    project = Project(workspace)
    project.add_patient("P001")
    project.add_document("P001", "DOC_001", "Riferisce tosse.")
    yield project
    project.close()


def test_zero_confidence_survives_reload(project):
    project.labs.insert(LabValue(patient_id="P001", document_id="DOC_001", parameter_name="Glucosio",
                                 normalized_name="glucosio", value=90, confidence=0.0))
    assert project.labs.get_by_patient("P001")[0].confidence == 0.0


def test_document_with_source_refs_and_legacy_gold_rows_can_be_deleted(project, workspace):
    from emr_analyzer.clinical.document_deletion import DocumentDeletionService
    from emr_analyzer.models.clinical_evidence import ClinicalEvidence

    project.evidence.insert_batch([ClinicalEvidence(patient_id="P001", document_id="DOC_001",
                                                    category="symptom", normalized_entity="tosse",
                                                    source_text="tosse")])
    evidence_id = project.evidence.get_by_patient("P001")[0].evidence_id
    project.db.execute(
        """INSERT INTO evidence_source_refs (source_ref_id, evidence_id, document_id, passage, created_at)
           VALUES ('R1', ?, 'DOC_001', 'tosse', '2026-10-01')""", (evidence_id,))
    project.db.commit()

    result = DocumentDeletionService(project.db, project.documents, workspaces_dir=workspace).delete("DOC_001")
    assert result.deleted, result.error
    assert project.db.execute("SELECT COUNT(*) FROM clinical_evidence").fetchone()[0] == 0


def test_patient_deletion_rejects_paths_outside_the_project(project, workspace, tmp_path):
    from emr_analyzer.clinical.patient_deletion import PatientWorkspaceDeletionService
    from emr_analyzer.models import Patient

    (workspace / ".." / "outside").mkdir()
    project.patients.insert(Patient(id="../outside", pseudonym="X"))
    service = PatientWorkspaceDeletionService(project.db, project.patients,
                                              workspaces_dir=workspace, cache_dir=tmp_path / "cache")
    result = service.delete("../outside")
    assert not result.deleted and "non valido" in result.error
    assert (workspace / ".." / "outside").exists()


def test_lab_fhir_identity_ignores_review_and_other_documents():
    lab = LabValue(patient_id="P001", document_id="DOC_A", parameter_name="Creatinina",
                   normalized_name="creatinina", value=1.9, unit="mg/dL", sample_date="2026-09-10")
    other = LabValue(patient_id="P001", document_id="DOC_B", parameter_name="Emoglobina",
                     normalized_name="emoglobina", value=13.1, unit="g/dL", sample_date="2026-09-01")
    alone = lab_occurrence_keys([lab])[0]
    with_older_report = lab_occurrence_keys([other, lab])[1]
    reviewed = lab_occurrence_keys([replace(lab, validated_by_user=True, confidence=0.2)])[0]
    assert alone == with_older_report == reviewed

    first, second = FhirRegistry("P001"), FhirRegistry("P001")
    first.laboratory(lab, alone)
    second.laboratory(replace(lab, validated_by_user=True), reviewed)
    ids = [{r["id"] for r in registry.resources.values() if r["resourceType"] == "Observation"}
           for registry in (first, second)]
    assert ids[0] == ids[1]


def test_identical_rows_in_one_document_stay_distinct():
    lab = LabValue(patient_id="P001", document_id="DOC_A", parameter_name="Glucosio",
                   normalized_name="glucosio", value=90)
    keys = lab_occurrence_keys([lab, lab])
    assert keys[0] != keys[1]

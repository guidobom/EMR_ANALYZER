from types import SimpleNamespace

from emr_analyzer.clinical.lab_evidence import (
    LAB_EXTRACTION_METHOD,
    abnormal_lab_evidence,
    is_out_of_range,
)
from emr_analyzer.clinical.registry_builder import ClinicalRegistryBuilder
from emr_analyzer.models.lab_result import LabValue


def _lab(
    name: str,
    value: float,
    *,
    low: float | None = 4.0,
    high: float | None = 10.0,
    abnormal: bool = False,
    flag: str | None = None,
    source: str | None = None,
) -> LabValue:
    return LabValue(
        patient_id="P001",
        document_id="D_LAB",
        parameter_name=name,
        normalized_name=name.casefold(),
        value=value,
        unit="mg/dL",
        reference_low=low,
        reference_high=high,
        reference_text=f"{low}-{high}",
        is_abnormal=abnormal,
        flag=flag,
        sample_date="2025-04-12",
        page=2,
        source_text=source or f"{name} {value:g} mg/dL {low}-{high}",
    )


def test_out_of_range_uses_flag_or_numeric_reference_interval():
    assert not is_out_of_range(_lab("Normale", 7.0))
    assert is_out_of_range(_lab("Flag", 7.0, flag="H"))
    assert is_out_of_range(_lab("Alto", 12.0))
    assert is_out_of_range(_lab("Basso", 2.0))


def test_only_abnormal_labs_become_citable_atomic_evidence():
    class Geometry:
        @staticmethod
        def locate_source(text, page):
            return page, (10.0, 20.0, 100.0, 30.0)

    values = [
        _lab("Glucosio", 7.0),
        _lab("PCR", 12.0, flag="H"),
        _lab("Sodio", 2.0, flag="L"),
    ]
    evidence = abnormal_lab_evidence(
        patient_id="P001",
        document_id="D_LAB",
        document_date="2025-04-13",
        lab_values=values,
        geometry=Geometry(),
    )

    assert [item.normalized_entity for item in evidence] == ["pcr", "sodio"]
    first = evidence[0]
    assert first.extraction_method == LAB_EXTRACTION_METHOD
    assert first.assertion == "present"
    assert first.certainty == "confirmed"
    assert first.observed_date == "2025-04-12"
    assert first.document_date == "2025-04-13"
    assert first.data["fact_type"] == "laboratory_test"
    assert first.data["polarity"] == "present"
    assert first.data["source_reference"] == {
        "document_id": "D_LAB",
        "page": 2,
        "bbox": [10.0, 20.0, 100.0, 30.0],
        "passage": "PCR 12 mg/dL 4.0-10.0",
    }
    assert first.to_atomic_dict()["report_date"] == "2025-04-13"


def test_abnormal_lab_without_source_passage_is_not_promoted_to_evidence():
    value = _lab("PCR", 12.0, abnormal=True)
    value.source_text = ""
    assert abnormal_lab_evidence(
        patient_id="P001",
        document_id="D_LAB",
        document_date="2025-04-13",
        lab_values=[value],
    ) == []


def test_registry_rebuild_replaces_legacy_lab_atoms_with_abnormal_only():
    normal = _lab("Normale", 7.0)
    abnormal = _lab("PCR", 12.0, flag="H")

    class LabRepo:
        @staticmethod
        def get_by_patient(patient_id):
            assert patient_id == "P001"
            return [normal, abnormal]

    class EvidenceRepo:
        def __init__(self):
            self.calls = []

        def replace_document_method(self, document_id, method, evidence):
            self.calls.append((document_id, method, evidence))

    evidence_repo = EvidenceRepo()
    builder = object.__new__(ClinicalRegistryBuilder)
    builder.lab_repo = LabRepo()
    builder.evidence_repo = evidence_repo
    documents = [SimpleNamespace(
        id="D_LAB", document_type="laboratorio", document_date="2025-04-13"
    )]

    count = builder._sync_abnormal_lab_evidence("P001", documents)

    assert count == 1
    assert len(evidence_repo.calls) == 1
    document_id, method, evidence = evidence_repo.calls[0]
    assert document_id == "D_LAB"
    assert method == LAB_EXTRACTION_METHOD
    assert [item.normalized_entity for item in evidence] == ["pcr"]

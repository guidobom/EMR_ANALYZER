"""Fase 5 — the deterministic terminology resolver preserves SNOMED atoms.

A code assigned by the constrained SNOMED stage is final: the lab normalizer
must not rewrite the preferred-term label and the built-in LOINC mapping must
never overwrite the SNOMED code.  Non-coded laboratory atoms keep the existing
LOINC behavior.
"""

from __future__ import annotations

from emr_analyzer.clinical.terminology import DeterministicTerminologyResolver
from emr_analyzer.models.clinical_evidence import ClinicalEvidence


def _atom(**overrides) -> ClinicalEvidence:
    kwargs = dict(
        patient_id="P1",
        document_id="D1",
        category="diagnosis",
        normalized_entity="Diabete mellito",
        source_text="diabete mellito",
        fact_type="diagnosis",
    )
    kwargs.update(overrides)
    return ClinicalEvidence(**kwargs)


class TestSnomedPreservation:
    def test_diagnosis_atom_is_preserved_untouched(self):
        atom = _atom(
            terminology_system="SNOMED CT",
            terminology_code="44054006",
            canonical_label="Diabete mellito",
            mapping_status="resolved_llm",
            mapping_confidence=0.90,
        )
        resolved = DeterministicTerminologyResolver(None).resolve(atom)
        assert resolved is atom
        assert resolved.terminology_system == "SNOMED CT"
        assert resolved.terminology_code == "44054006"
        assert resolved.canonical_label == "Diabete mellito"
        assert resolved.mapping_status == "resolved_llm"
        assert resolved.mapping_confidence == 0.90
        assert resolved.typed_payload.get("mapping_source") is None

    def test_snomed_lab_atom_is_not_rewritten_by_loinc(self):
        # The regression the early-return protects: without it the lab block
        # normalizes the parameter and the built-in LOINC mapping would
        # overwrite the SNOMED code with ``2160-0``.
        atom = _atom(
            category="laboratory_finding",
            fact_type="laboratory_test",
            normalized_entity="Creatinina",
            source_text="creatinina 1.2 mg/dL",
            unit="mg/dL",
            terminology_system="SNOMED CT",
            terminology_code="709044004",
            canonical_label="Raised creatinine level",
            mapping_status="resolved_llm",
        )
        resolved = DeterministicTerminologyResolver(None).resolve(atom)
        assert resolved.terminology_system == "SNOMED CT"
        assert resolved.terminology_code == "709044004"
        assert resolved.mapping_status == "resolved_llm"
        assert resolved.canonical_label == "Raised creatinine level"
        assert resolved.unit == "mg/dL"

    def test_snomed_unmapped_status_is_promoted(self):
        atom = _atom(
            terminology_system="SNOMED CT",
            terminology_code="44054006",
            canonical_label="Diabete mellito",
            mapping_status="unmapped",
        )
        resolved = DeterministicTerminologyResolver(None).resolve(atom)
        assert resolved.mapping_status == "resolved_llm"

    def test_snomed_atom_without_code_flows_through(self):
        # ``terminology_system == SNOMED CT`` without a code is not final:
        # the guard requires the code, so the normal LOINC path still applies.
        atom = _atom(
            category="laboratory_finding",
            fact_type="laboratory_test",
            normalized_entity="Creatinina",
            source_text="creatinina",
            unit="mg/dL",
            terminology_system="SNOMED CT",
            terminology_code=None,
        )
        resolved = DeterministicTerminologyResolver(None).resolve(atom)
        assert resolved.terminology_system == "LOINC"
        assert resolved.terminology_code == "2160-0"
        assert resolved.mapping_status == "resolved_deterministic"


class TestLoincStillWorksForNonCodedLab:
    def test_unmapped_lab_atom_gets_loinc(self):
        atom = _atom(
            category="laboratory_finding",
            fact_type="laboratory_test",
            normalized_entity="Creatinina",
            source_text="creatinina",
            unit="mg/dL",
        )
        resolved = DeterministicTerminologyResolver(None).resolve(atom)
        assert resolved.terminology_system == "LOINC"
        assert resolved.terminology_code == "2160-0"
        assert resolved.mapping_status == "resolved_deterministic"
        assert resolved.unit == "mg/dL"

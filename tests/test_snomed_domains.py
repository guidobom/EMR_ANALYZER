"""Tests for the SNOMED semantic-tag -> atomic fact-type mapping."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from emr_analyzer.clinical.snomed import (
    ReleaseSnapshot,
    SnomedConcept,
    SnomedDescription,
    fact_types_for_concept,
    load_snapshot,
)
from emr_analyzer.clinical.snomed.models import FSN_TYPE, SYNONYM_TYPE

FIXTURE_DIR = (
    Path(__file__).resolve().parent / "fixtures" / "snomed_rf2" / "Snapshot_20260101"
)
MANIFEST = Path(__file__).resolve().parent / "fixtures" / "snomed_rf2" / "concepts.csv"


@pytest.fixture(scope="module")
def snapshot() -> ReleaseSnapshot:
    return load_snapshot(FIXTURE_DIR)


def _synthetic_concept(
    code: str,
    fsn_it: str,
    fsn_en: str,
    pt_it: str,
    pt_en: str,
) -> SnomedConcept:
    """Build an inline concept with Italian + English FSN/PT descriptions."""
    descriptions: list[SnomedDescription] = []
    for lang, fsn, pt in (
        ("it", fsn_it, pt_it), ("en", fsn_en, pt_en),
    ):
        descriptions.append(SnomedDescription(
            description_id=f"{code}-fsn-{lang}", concept_id=code, lang=lang,
            type_id=FSN_TYPE, term=fsn, acceptability="", active=True,
        ))
        descriptions.append(SnomedDescription(
            description_id=f"{code}-pt-{lang}", concept_id=code, lang=lang,
            type_id=SYNONYM_TYPE, term=pt, acceptability="preferred",
            active=True,
        ))
    return SnomedConcept(
        concept_id=code, active=True,
        definition_status_id="900000000000074008",
        descriptions=tuple(descriptions),
    )


class TestManifestConsistency:
    def test_every_manifest_concept_matches_expected(self, snapshot):
        with open(MANIFEST, newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        checked = 0
        for row in rows:
            expected = row["expected_fact_types"].strip()
            if not expected:
                # Anchor hierarchy nodes carry no evidence bucket; skip.
                continue
            concept = snapshot.concept(row["code"])
            assert concept is not None, row["code"]
            actual = fact_types_for_concept(concept)
            assert set(actual) == set(expected.split(";")), row["code"]
            checked += 1
        assert checked == len(rows) - 7


class TestSemanticTagMapping:
    def test_disorder_is_diagnosis(self, snapshot):
        assert fact_types_for_concept(snapshot.concept("44054006")) == (
            "diagnosis",
        )
        assert fact_types_for_concept(snapshot.concept("93655004")) == (
            "diagnosis",
        )

    def test_procedure_is_procedure_and_decision(self, snapshot):
        assert fact_types_for_concept(snapshot.concept("86273004")) == (
            "clinical_decision", "procedure",
        )

    def test_substance_is_medication_and_decision(self, snapshot):
        assert fact_types_for_concept(snapshot.concept("372250003")) == (
            "medication", "clinical_decision",
        )

    def test_morphology_is_histopathology_and_diagnosis(self, snapshot):
        assert fact_types_for_concept(snapshot.concept("104910003")) == (
            "diagnosis", "histopathology",
        )

    def test_body_structure_is_empty(self):
        body = _synthetic_concept(
            "1111", "Mano sinistra (struttura corporea)",
            "Left hand (body structure)", "Mano sinistra", "Left hand",
        )
        assert fact_types_for_concept(body) == ()


class TestFsnHintDisambiguation:
    def test_nodule_finding_is_radiology(self, snapshot):
        assert fact_types_for_concept(snapshot.concept("300928002")) == (
            "radiology_finding",
        )

    def test_jaundice_finding_is_clinical_sign(self, snapshot):
        assert fact_types_for_concept(snapshot.concept("18165001")) == (
            "clinical_sign",
        )

    def test_creatinine_observable_is_lab(self, snapshot):
        assert fact_types_for_concept(snapshot.concept("102299008")) == (
            "laboratory_test",
        )

    def test_braf_observable_is_biomarker(self, snapshot):
        assert fact_types_for_concept(snapshot.concept("447206003")) == (
            "biomarker",
        )

    def test_breslow_observable_is_histopathology(self, snapshot):
        assert fact_types_for_concept(snapshot.concept("81917007")) == (
            "histopathology",
        )

    def test_body_weight_observable_is_vital_sign(self, snapshot):
        assert fact_types_for_concept(snapshot.concept("27113001")) == (
            "vital_sign",
        )

    def test_fsn_hint_wins_over_broad_default(self):
        # A (finding) whose FSN contains a radiology keyword resolves to
        # radiology_finding instead of the broad finding union.
        finding = _synthetic_concept(
            "2222", "Nodulo del lobo superiore (reperto)",
            "Upper lobe nodule (finding)", "Nodulo del lobo superiore",
            "Upper lobe nodule",
        )
        assert fact_types_for_concept(finding) == ("radiology_finding",)


class TestObservableDefaults:
    def test_observable_without_hint_broad(self):
        observable = _synthetic_concept(
            "3333", "Misurazione strana (entità osservabile)",
            "Strange measurement (observable entity)",
            "Misurazione strana", "Strange measurement",
        )
        assert fact_types_for_concept(observable) == (
            "laboratory_test", "vital_sign", "biomarker",
        )

    def test_unknown_tag_is_broad_union(self):
        unknown = _synthetic_concept(
            "4444", "Strano reperto (entità bizzarra)",
            "Strange finding (bizarre entity)", "Strano reperto",
            "Strange finding",
        )
        assert fact_types_for_concept(unknown) == (
            "medication", "radiology_finding", "diagnosis", "symptom",
            "procedure", "clinical_sign", "histopathology", "biomarker",
        )

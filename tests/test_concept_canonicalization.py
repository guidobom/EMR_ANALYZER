"""Deterministic concept canonicalization for atomic-evidence identity.

Fixtures reproduce the variant pairs measured by the MELANOMA duplicate
audit (drug typos, exam-title variants, filler severities, English labels).
"""

import unittest

from emr_analyzer.clinical.atomic_evidence import (
    ClinicalEvidence,
    deduplicate_atomic_evidence,
)
from emr_analyzer.clinical.concept_canonicalization import (
    canonical_concept,
    canonical_severity,
)


def _atom(entity: str, *, category: str = "medication",
          document_id: str = "DOC_A", severity: str | None = None,
          observed_date: str | None = "2025-01-10",
          clinical_status: str | None = "ongoing",
          value_text: str | None = None) -> ClinicalEvidence:
    return ClinicalEvidence(
        patient_id="P001",
        document_id=document_id,
        category=category,
        normalized_entity=entity,
        source_text=f"text of {entity}",
        severity=severity,
        observed_date=observed_date,
        clinical_status=clinical_status,
        value_text=value_text,
    )


class CanonicalConceptTest(unittest.TestCase):
    def test_drug_typos_map_to_active_ingredient(self):
        for variant in ("niivolumab", "nicolumab", "nevolumab", "nivolumab"):
            self.assertEqual(
                canonical_concept("medication", variant), "nivolumab"
            )
        self.assertEqual(
            canonical_concept("medication", "trametenib"), "trametinib"
        )
        self.assertEqual(
            canonical_concept("medication", "enconafenib"), "encorafenib"
        )
        self.assertEqual(
            canonical_concept("medication", "binimetib"), "binimetinib"
        )
        self.assertEqual(
            canonical_concept("medication", "deltacorene"), "deltacortene"
        )

    def test_unknown_drug_names_stay_unchanged(self):
        # Not in the whitelist: the fuzzy step must not rewrite them.
        self.assertEqual(
            canonical_concept("medication", "paracetamolo"), "paracetamolo"
        )
        self.assertEqual(
            canonical_concept("medication", "metformina"), "metformina"
        )

    def test_imaging_exam_titles_canonicalize(self):
        self.assertEqual(
            canonical_concept("imaging_finding", "tc tb"), "tc total body"
        )
        self.assertEqual(
            canonical_concept("imaging_finding", "tc tb mdc"), "tc total body"
        )
        self.assertEqual(
            canonical_concept("imaging_finding", "tc total body con mdc"),
            "tc total body",
        )
        self.assertEqual(
            canonical_concept("imaging_finding", "tc cranio encefalo"),
            "tc encefalo",
        )
        self.assertEqual(
            canonical_concept("imaging_finding", "pet con fdg"), "pet fdg"
        )
        self.assertEqual(
            canonical_concept("imaging_finding", "pet 18f fdg"), "pet fdg"
        )
        self.assertEqual(
            canonical_concept("imaging_finding", "fdg pet"), "pet fdg"
        )

    def test_english_labels_map_to_italian(self):
        self.assertEqual(
            canonical_concept("diagnosis", "malignant melanoma"),
            "melanoma maligno",
        )
        self.assertEqual(
            canonical_concept("diagnosis", "basal cell carcinoma"),
            "carcinoma basocellulare",
        )
        self.assertEqual(
            canonical_concept("procedure", "sentinel lymph node biopsy"),
            "biopsia linfonodo sentinella",
        )
        self.assertEqual(
            canonical_concept("biomarker", "braf mutation"), "braf mutato"
        )
        self.assertEqual(
            canonical_concept("biomarker", "pd l1"), "pdl1"
        )

    def test_concept_outside_mapped_categories_is_folded_only(self):
        # No category rule applies: plain folding, wording untouched.
        self.assertEqual(
            canonical_concept("symptom", "dolore addominale"),
            "dolore addominale",
        )


class CanonicalSeverityTest(unittest.TestCase):
    def test_fillers_collapse_to_empty(self):
        for filler in ("none", "non specified", "not specified", "0", "n.d."):
            self.assertEqual(canonical_severity(filler), "")

    def test_english_severities_harmonize(self):
        self.assertEqual(canonical_severity("moderate"), "moderata")
        self.assertEqual(canonical_severity("mild"), "lieve")
        self.assertEqual(canonical_severity("severe"), "grave")

    def test_grades_are_preserved(self):
        self.assertEqual(canonical_severity("g1"), "g1")
        self.assertEqual(canonical_severity("grade 2"), "g2")


class DedupWithCanonicalKeyTest(unittest.TestCase):
    def test_drug_typo_variants_merge_into_one_atom(self):
        evidence = deduplicate_atomic_evidence([
            _atom("nivolumab", document_id="DOC_00001"),
            _atom("niivolumab", document_id="DOC_00002"),
        ])
        self.assertEqual(len(evidence), 1)
        self.assertEqual(evidence[0].normalized_entity, "nivolumab")
        self.assertEqual(
            evidence[0].data["atomic_duplicate_count"], 1
        )
        self.assertEqual(
            len(evidence[0].data["source_occurrences"]), 2
        )

    def test_imaging_exam_title_variants_merge(self):
        evidence = deduplicate_atomic_evidence([
            _atom("tc tb", category="imaging_finding",
                  document_id="DOC_00001", clinical_status=None),
            _atom("tc tb mdc", category="imaging_finding",
                  document_id="DOC_00002", clinical_status=None),
        ])
        self.assertEqual(len(evidence), 1)
        self.assertEqual(
            evidence[0].data["duplicate_source_evidence_ids"],
            [
                item["evidence_id"]
                for item in evidence[0].data["source_occurrences"][1:]
            ],
        )

    def test_severity_filler_variants_merge(self):
        evidence = deduplicate_atomic_evidence([
            _atom("nivolumab", document_id="DOC_00001", severity="none"),
            _atom("nivolumab", document_id="DOC_00002", severity="g1"),
        ])
        # The filler atom merges into the informative one (richest wins).
        self.assertEqual(len(evidence), 1)
        self.assertEqual(evidence[0].severity, "g1")

    def test_different_observed_dates_do_not_merge(self):
        evidence = deduplicate_atomic_evidence([
            _atom("nivolumab", document_id="DOC_00001",
                  observed_date="2025-01-10"),
            _atom("niivolumab", document_id="DOC_00002",
                  observed_date="2025-03-10"),
        ])
        self.assertEqual(len(evidence), 2)

    def test_english_and_italian_labels_merge(self):
        evidence = deduplicate_atomic_evidence([
            _atom("melanoma maligno", category="diagnosis",
                  document_id="DOC_00001", clinical_status=None),
            _atom("malignant melanoma", category="diagnosis",
                  document_id="DOC_00002", clinical_status=None),
        ])
        self.assertEqual(len(evidence), 1)


class DateWildcardDedupTest(unittest.TestCase):
    def test_undated_state_restatement_absorbs_into_dated_atom(self):
        evidence = deduplicate_atomic_evidence([
            _atom("nivolumab", document_id="DOC_00001",
                  observed_date="2024-08-12"),
            _atom("nivolumab", document_id="DOC_00002",
                  observed_date=None),
        ])
        self.assertEqual(len(evidence), 1)
        self.assertEqual(evidence[0].observed_date, "2024-08-12")
        self.assertEqual(evidence[0].data["atomic_duplicate_count"], 1)
        self.assertEqual(len(evidence[0].data["source_occurrences"]), 2)

    def test_two_distinct_dates_stay_split(self):
        evidence = deduplicate_atomic_evidence([
            _atom("nivolumab", document_id="DOC_00001",
                  observed_date="2024-08-12"),
            _atom("nivolumab", document_id="DOC_00002",
                  observed_date="2024-05-07"),
            _atom("nivolumab", document_id="DOC_00003",
                  observed_date=None),
        ])
        # Two dated anchors: ambiguous, nothing absorbs.
        self.assertEqual(len(evidence), 3)

    def test_measurements_never_participate_in_date_wildcard(self):
        atom_a = _atom("peso", category="vital_sign", document_id="DOC_00001",
                       observed_date="2024-01-10", clinical_status=None)
        atom_a.numeric_value = 72.5
        atom_a.unit = "kg"
        atom_b = _atom("peso", category="vital_sign", document_id="DOC_00002",
                       observed_date=None, clinical_status=None)
        atom_b.numeric_value = 72.5
        atom_b.unit = "kg"
        evidence = deduplicate_atomic_evidence([atom_a, atom_b])
        self.assertEqual(len(evidence), 2)

    def test_undated_diagnosis_absorbs_into_dated_one(self):
        evidence = deduplicate_atomic_evidence([
            _atom("melanoma nodulare", category="diagnosis",
                  document_id="DOC_00001", observed_date="2024-02-15",
                  clinical_status=None),
            _atom("melanoma nodulare", category="diagnosis",
                  document_id="DOC_00002", observed_date=None,
                  clinical_status=None),
        ])
        self.assertEqual(len(evidence), 1)
        self.assertEqual(evidence[0].observed_date, "2024-02-15")


if __name__ == "__main__":
    unittest.main(verbosity=2)

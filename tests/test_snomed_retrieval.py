"""Tests for the SNOMED candidate-set retriever."""

from __future__ import annotations

from pathlib import Path

import pytest

from emr_analyzer.clinical.snomed import (
    ReleaseSnapshot,
    SnomedCandidateRetriever,
    SnomedCandidateSet,
    SnomedConcept,
    SnomedIndex,
    candidate_digest,
    load_snapshot,
)

FIXTURE_DIR = (
    Path(__file__).resolve().parent / "fixtures" / "snomed_rf2" / "Snapshot_20260101"
)


@pytest.fixture(scope="module")
def snapshot() -> ReleaseSnapshot:
    return load_snapshot(FIXTURE_DIR)


@pytest.fixture(scope="module")
def retriever(snapshot: ReleaseSnapshot) -> SnomedCandidateRetriever:
    return SnomedCandidateRetriever(SnomedIndex(snapshot.concepts.values()))


def _code_set(candidates: SnomedCandidateSet) -> tuple[str, ...]:
    return candidates.codes


def _candidate_for(
    candidates: SnomedCandidateSet, code: str
):
    for candidate in candidates.candidates:
        if candidate.concept.concept_id == code:
            return candidate
    return None


class TestExactRetrieval:
    def test_single_concept_italian(self, retriever):
        candidates = retriever.candidates_for_text("diabete")
        assert candidates.codes == ("44054006",)
        candidate = candidates.candidates[0]
        assert candidate.concept.concept_id == "44054006"
        assert candidate.score == 1.0
        assert candidate.match_kind == "exact"
        assert candidate.matched_term == "diabete"

    def test_italian_synonyms(self, retriever):
        for text, code in (
            ("ipertensione", "38341003"),
            ("BPCO", "13645005"),
            ("VES", "31728002"),
            ("affanno", "267036007"),
            ("PCR", "40228008"),
        ):
            candidates = retriever.candidates_for_text(text)
            assert candidates.codes == (code,), text

    def test_abbreviations(self, retriever):
        for text, code in (
            ("SpO2", "431314004"),
            ("TC", "77477000"),
            ("PET", "105328002"),
            ("HbA1c", "407042007"),
            ("BRAF", "447206003"),
        ):
            candidates = retriever.candidates_for_text(text)
            assert candidates.codes == (code,), text

    def test_english_term(self, retriever):
        candidates = retriever.candidates_for_text("creatinine")
        assert candidates.codes == ("102299008",)

    def test_domain_fact_types_filled(self, retriever):
        candidates = retriever.candidates_for_text("diabete")
        assert candidates.domain_fact_types == {"44054006": ("diagnosis",)}


class TestFuzzyRetrieval:
    def test_typo_recovers_concept(self, retriever):
        candidates = retriever.candidates_for_text("melanom maligno")
        assert "93655004" in candidates.codes
        candidate = _candidate_for(candidates, "93655004")
        assert candidate is not None
        assert candidate.match_kind == "fuzzy"


class TestCapAndOrder:
    def test_cap_limits_by_score(self, retriever):
        # ``melanoma`` recalls two concepts; the top-1 by score is the
        # diagnosis whose synonym is an exact match.
        candidates = retriever.candidates_for_text("melanoma", cap=1)
        assert candidates.codes == ("93655004",)

    def test_deterministic_code_order(self, retriever):
        candidates = retriever.candidates_for_text("melanoma", cap=8)
        assert candidates.codes == ("104910003", "93655004")

    def test_min_score_filters_low(self, retriever):
        candidates = retriever.candidates_for_text("melanoma", min_score=0.9)
        assert candidates.codes == ("93655004",)


class TestCandidateDigest:
    def test_digest_deterministic(self, retriever):
        first = retriever.candidates_for_text("diabete")
        second = retriever.candidates_for_text("diabete")
        assert first.digest == second.digest
        assert first.digest == candidate_digest(["44054006"])

    def test_digest_changes_with_set(self, retriever):
        diabete = retriever.candidates_for_text("diabete")
        melanoma = retriever.candidates_for_text("melanoma")
        assert diabete.digest != melanoma.digest

    def test_digest_order_independent(self):
        assert candidate_digest(["b", "a"]) == candidate_digest(["a", "b"])
        assert candidate_digest(["a"]) != candidate_digest(["a", "b"])


class TestEmptyAndExplicit:
    def test_empty_text(self, retriever):
        candidates = retriever.candidates_for_text("zzzqwerty")
        assert candidates.empty
        assert candidates.codes == ()
        assert candidates.candidates == ()

    def test_candidates_for_concepts(self, retriever, snapshot):
        concepts = [
            snapshot.concept("44054006"),
            snapshot.concept("372250003"),
        ]
        assert all(c is not None for c in concepts)
        candidates = retriever.candidates_for_concepts(
            [c for c in concepts if c is not None]
        )
        assert candidates.codes == ("372250003", "44054006")
        assert candidates.domain_fact_types["372250003"] == (
            "medication", "clinical_decision",
        )

    def test_to_catalog(self, retriever):
        candidates = retriever.candidates_for_text("diabete")
        catalog = candidates.to_catalog(("it",))
        assert "44054006 | Diabete mellito [domini: diagnosis]" in catalog

    def test_empty_flag_false_when_hits(self, retriever):
        candidates = retriever.candidates_for_text("diabete")
        assert not candidates.empty


class TestRetrieverConstruction:
    def test_cap_clamped(self, snapshot):
        retriever = SnomedCandidateRetriever(
            SnomedIndex(snapshot.concepts.values()), cap=200
        )
        assert retriever.cap == 64

    def test_cap_minimum(self, snapshot):
        retriever = SnomedCandidateRetriever(
            SnomedIndex(snapshot.concepts.values()), cap=1
        )
        assert retriever.cap == 8

"""Tests for the SNOMED lexical index and bilingual term selection."""

from __future__ import annotations

from pathlib import Path

import pytest

from emr_analyzer.clinical.snomed import (
    ReleaseSnapshot,
    SnomedIndex,
    load_snapshot,
    normalize_tokens,
)

FIXTURE_DIR = (
    Path(__file__).resolve().parent / "fixtures" / "snomed_rf2" / "Snapshot_20260101"
)


@pytest.fixture(scope="module")
def snapshot() -> ReleaseSnapshot:
    return load_snapshot(FIXTURE_DIR)


@pytest.fixture(scope="module")
def index(snapshot: ReleaseSnapshot) -> SnomedIndex:
    return SnomedIndex(snapshot.concepts.values())


class TestNormalizeTokens:
    def test_casefold_and_strip_punctuation(self):
        assert normalize_tokens("HbA1c") == ["hba1c"]
        assert normalize_tokens("anti-PD-1") == ["anti", "pd", "1"]
        assert normalize_tokens("SpO2") == ["spo2"]

    def test_accent_nfkd(self):
        assert normalize_tokens("Opacità") == ["opacita"]

    def test_empty(self):
        assert normalize_tokens("") == []
        assert normalize_tokens(None) == []


class TestBilingualTermSelection:
    def test_preferred_term_it(self, snapshot):
        concept = snapshot.concept("44054006")
        assert concept.preferred_term(("it",)) == "Diabete mellito"
        assert concept.preferred_term(("en",)) == "Diabetes mellitus"

    def test_fsn_and_semantic_tag(self, snapshot):
        concept = snapshot.concept("44054006")
        assert concept.fsn(("it",)) == "Diabete mellito (disturbo)"
        assert concept.fsn(("en",)) == "Diabetes mellitus (disorder)"
        assert concept.semantic_tag(("it",)) == "disturbo"
        assert concept.semantic_tag(("en",)) == "disorder"

    def test_all_terms_includes_synonyms(self, snapshot):
        concept = snapshot.concept("44054006")
        italian = concept.all_terms(("it",))
        assert "diabete" in italian
        assert "Diabete mellito (disturbo)" in italian

    def test_display_label_falls_back_to_id(self):
        from emr_analyzer.clinical.snomed import SnomedConcept
        bare = SnomedConcept(
            concept_id="123", active=True, definition_status_id=""
        )
        assert bare.display_label(("it",)) == "123"

    def test_preferred_term_fallback_across_languages(self):
        # A concept with only English descriptions still resolves an Italian PT
        # through the cross-language fallback.
        from emr_analyzer.clinical.snomed import (
            SnomedConcept,
            SnomedDescription,
        )
        concept = SnomedConcept(
            concept_id="123",
            active=True,
            definition_status_id="900000000000074008",
            descriptions=(
                SnomedDescription(
                    description_id="1", concept_id="123", lang="en",
                    type_id="900000000000013009", term="Fallback term",
                    acceptability="preferred", active=True,
                ),
            ),
        )
        assert concept.preferred_term(("it",)) == "Fallback term"


class TestTokenIndex:
    def test_search_exact_italian_synonym(self, index):
        rows = index.search("diabete")
        assert any(
            concept.concept_id == "44054006" and score == 1.0 and kind == "exact"
            for concept, score, matched_term, kind in rows
        )

    def test_search_english_term(self, index):
        rows = index.search("hypertension")
        assert any(
            concept.concept_id == "38341003"
            for concept, *_ in rows
        )

    def test_search_multiword_token_coverage(self, index):
        rows = index.search("diabete mellito")
        assert any(
            concept.concept_id == "44054006" and score == 1.0
            for concept, score, *_ in rows
        )

    def test_search_no_match(self, index):
        assert index.search("zzzqwerty") == []

    def test_search_empty_query(self, index):
        assert index.search("") == []

    def test_abbreviation_italian(self, index):
        rows = index.search("BPCO")
        assert any(concept.concept_id == "13645005" for concept, *_ in rows)


class TestIsaNavigation:
    def test_ancestors_transitive(self, index):
        assert index.ancestors("44054006") == {
            "64572001", "404684003", "138875005",
        }

    def test_descendants(self, index):
        assert index.descendants("64572001") == {
            "44054006", "38341003", "42343007", "271737000", "93655004",
            "13645005",
        }

    def test_is_descendant_of(self, index):
        assert index.is_descendant_of("44054006", "404684003")
        assert index.is_descendant_of("44054006", "64572001")
        assert not index.is_descendant_of("404684003", "44054006")

    def test_root_has_no_parents(self, index):
        assert index.parents("138875005") == frozenset()
        assert "404684003" in index.children("138875005")

    def test_unknown_code_isolated(self, index):
        assert index.ancestors("99999999") == set()
        assert index.parents("99999999") == frozenset()


class TestIndexDigest:
    def test_index_concepts_are_immutable_copy(self, snapshot, index):
        concept = index.concept("44054006")
        assert concept is not None
        assert concept.preferred_term(("it",)) == "Diabete mellito"

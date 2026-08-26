"""Abbreviation fallback for the SNOMED candidate retriever.

A curated dictionary expands abbreviations the release does not carry as
literal tokens (e.g. ``ht``, ``dm``, ``so2``).  The fallback fires only when
the index has no literal token for the abbreviation, so a release that already
resolves ``spo2`` keeps the exact lexical path untouched.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from emr_analyzer.clinical.snomed import (
    ReleaseSnapshot,
    SnomedCandidateRetriever,
    SnomedIndex,
    candidate_digest,
    load_snapshot,
)
from emr_analyzer.clinical.snomed.retrieval import load_abbreviations

FIXTURE_DIR = (
    Path(__file__).resolve().parent / "fixtures" / "snomed_rf2" / "Snapshot_20260101"
)


@pytest.fixture(scope="module")
def snapshot() -> ReleaseSnapshot:
    return load_snapshot(FIXTURE_DIR)


def _retriever(snapshot, **kwargs) -> SnomedCandidateRetriever:
    return SnomedCandidateRetriever(SnomedIndex(snapshot.concepts.values()),
                                    **kwargs)


def _candidate(candidates, code):
    for candidate in candidates.candidates:
        if candidate.concept.concept_id == code:
            return candidate
    return None


class TestFallbackResolvesMissingAbbreviations:
    def test_ht_recalls_hypertension_via_fallback(self, snapshot):
        # ``ht`` is not a literal token anywhere in the fixture index.
        candidates = _retriever(snapshot).candidates_for_text("HT")
        assert candidates.codes == ("38341003",)
        candidate = candidates.candidates[0]
        assert candidate.match_kind == "abbreviation"
        assert candidate.matched_term == "ht"
        assert candidate.score == 1.0

    def test_so2_recalls_oxygen_saturation_via_fallback(self, snapshot):
        # ``so2`` is absent as a literal token; the expansion "oxygen
        # saturation" resolves 431314004.
        candidates = _retriever(snapshot).candidates_for_text("SO2")
        assert candidates.codes == ("431314004",)
        assert candidates.candidates[0].match_kind == "abbreviation"

    def test_custom_abbreviation_dict_is_injected(self, snapshot):
        retriever = _retriever(
            snapshot, abbreviations={"sats": ["oxygen saturation"]}
        )
        candidates = retriever.candidates_for_text("sats")
        assert candidates.codes == ("431314004",)
        assert candidates.candidates[0].match_kind == "abbreviation"

    def test_fallback_merges_with_primary_hits(self, snapshot):
        # ``diabete`` is a lexical exact hit; ``ht`` arrives via fallback.
        candidates = _retriever(snapshot).candidates_for_text(
            "Paziente con HT e diabete mellito"
        )
        assert set(candidates.codes) == {"38341003", "44054006"}
        assert _candidate(candidates, "38341003").match_kind == "abbreviation"
        # In the multi-word text "diabete mellito" is a full-coverage token
        # match at score 1.0 (``exact`` requires the query to be only the term).
        assert _candidate(candidates, "44054006").score == 1.0
        assert _candidate(candidates, "44054006").match_kind == "token"
        # Deterministic code-ascending order, like the base search.
        assert candidates.codes == ("38341003", "44054006")

    def test_digest_reflects_expanded_set(self, snapshot):
        candidates = _retriever(snapshot).candidates_for_text("HT")
        assert candidates.digest == candidate_digest(["38341003"])


class TestLiteralTokensStayAuthoritative:
    def test_spo2_is_not_expanded(self, snapshot):
        # The fixture carries ``spo2`` as a literal description token, so the
        # exact lexical path wins and the "oxygen saturation" expansion is not
        # merged in (which would otherwise recall extra concepts).
        candidates = _retriever(snapshot).candidates_for_text("SpO2")
        assert candidates.codes == ("431314004",)
        assert candidates.candidates[0].match_kind == "exact"
        assert candidates.candidates[0].matched_term == "spo2"

    def test_dm_resolves_literally_when_present(self, snapshot):
        candidates = _retriever(snapshot).candidates_for_text("DM")
        assert candidates.codes == ("44054006",)
        assert candidates.candidates[0].match_kind == "exact"

    def test_bpco_uses_existing_synonym(self, snapshot):
        candidates = _retriever(snapshot).candidates_for_text("BPCO")
        assert candidates.codes == ("13645005",)
        assert candidates.candidates[0].match_kind == "exact"


class TestDisabling:
    def test_empty_dict_disables_fallback(self, snapshot):
        retriever = _retriever(snapshot, abbreviations={})
        candidates = retriever.candidates_for_text("HT")
        assert candidates.empty
        assert candidates.codes == ()

    def test_unknown_token_unchanged(self, snapshot):
        candidates = _retriever(snapshot).candidates_for_text("zzzqwerty")
        assert candidates.empty


class TestResource:
    def test_packaged_resource_loads(self):
        abbreviations = load_abbreviations()
        assert isinstance(abbreviations, dict)
        assert abbreviations
        # Keys are normalized lower-case tokens; values are non-empty phrase
        # lists.
        for key, phrases in abbreviations.items():
            assert key == key.casefold()
            assert phrases

    def test_packaged_resource_covers_common_abbreviations(self):
        abbreviations = load_abbreviations()
        for abbr in ("ht", "dm", "so2", "spo2", "fa", "bpco", "hb"):
            assert abbr in abbreviations, abbr

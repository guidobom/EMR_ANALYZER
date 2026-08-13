"""Tests for the chronological registry deduplication redesign.

Covers: deterministic pre-filter, the new ``groups`` LLM contract
(canonical synthesis + ``merged_into_ids`` provenance), the legacy
contract, date_observed normalization and the end-to-end pipeline
(``_deduplicate_all``).
"""

from __future__ import annotations

import unittest

from emr_analyzer.clinical.clinical_history_builder import ClinicalHistoryBuilder
from emr_analyzer.models.clinical_timeline import ClinicalTimelineEntry


def make_entry(entry_id, date, category, description, confidence=0.8,
               doc_ids=("DOC_1",), texts=("source text",)):
    return ClinicalTimelineEntry(
        entry_id=entry_id,
        patient_id="P001",
        date_observed=date,
        category=category,
        description=description,
        source_document_ids=list(doc_ids),
        source_texts=list(texts),
        confidence=confidence,
    )


class FakeLlm:
    """Stub Clinical State LLM returning a canned dedup ``groups`` result."""

    def __init__(self, groups):
        self.groups = groups
        self.called_with = None

    def deduplicate_timeline(self, entries_dicts):
        self.called_with = entries_dicts
        return {"groups": self.groups}


def builder_with_llm(llm):
    return ClinicalHistoryBuilder(
        timeline_repo=None,
        document_repo=None,
        cs_repo=None,
        clinical_state_llm_client=llm,
        audit_repo=None,
    )


def apply_groups(entries, result):
    """Call the instance method ``_apply_dedup_groups`` without a real LLM."""
    return builder_with_llm(FakeLlm(groups=[]))._apply_dedup_groups(
        entries, result
    )


class TestNormalizeDateObserved(unittest.TestCase):
    """date_observed normalization + document-date fallback."""

    def test_iso_passthrough(self):
        self.assertEqual(
            ClinicalHistoryBuilder._normalize_date_observed(
                "2020-05-01", "2020-05-01"
            ),
            "2020-05-01",
        )

    def test_month_precision(self):
        self.assertEqual(
            ClinicalHistoryBuilder._normalize_date_observed("2020-05", ""),
            "2020-05",
        )

    def test_non_padded_iso(self):
        self.assertEqual(
            ClinicalHistoryBuilder._normalize_date_observed("2020-5-1", ""),
            "2020-05-01",
        )

    def test_european_day_first(self):
        self.assertEqual(
            ClinicalHistoryBuilder._normalize_date_observed("01/05/2020", ""),
            "2020-05-01",
        )

    def test_european_dash(self):
        self.assertEqual(
            ClinicalHistoryBuilder._normalize_date_observed("01-05-2020", ""),
            "2020-05-01",
        )

    def test_none_falls_back_to_document_date(self):
        self.assertEqual(
            ClinicalHistoryBuilder._normalize_date_observed(None, "2020-05-01"),
            "2020-05-01",
        )

    def test_empty_falls_back_to_document_date(self):
        self.assertEqual(
            ClinicalHistoryBuilder._normalize_date_observed("  ", "2020-05-01"),
            "2020-05-01",
        )

    def test_unparseable_falls_back_to_document_date(self):
        self.assertEqual(
            ClinicalHistoryBuilder._normalize_date_observed(
                "proseguimento terapia", "2020-05-01"
            ),
            "2020-05-01",
        )

    def test_no_document_date_and_empty_value(self):
        self.assertEqual(
            ClinicalHistoryBuilder._normalize_date_observed(None, ""), ""
        )

    def test_impossible_calendar_date_falls_back(self):
        self.assertEqual(
            ClinicalHistoryBuilder._normalize_date_observed(
                "2020-13-40", "2020-05-01"
            ),
            "2020-05-01",
        )


class TestDeterministicDedup(unittest.TestCase):
    """Near-verbatim same-date/same-category pre-filter."""

    def test_merges_verbatim_variants(self):
        entries = [
            make_entry("CTL_000001", "2020-05-01", "treatment",
                       "Inizio dabrafenib 150 mg", confidence=0.8,
                       doc_ids=("A",), texts=("t1",)),
            make_entry("CTL_000002", "2020-05-01", "treatment",
                       "inizia dabrafenib 150 mg", confidence=0.9,
                       doc_ids=("B",), texts=("t2",)),
        ]
        out = ClinicalHistoryBuilder._deterministic_dedup(entries)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].entry_id, "CTL_000002")  # higher confidence

    def test_survivor_excludes_own_id_from_merged_into_ids(self):
        entries = [
            make_entry("CTL_000001", "2020-05-01", "treatment",
                       "Inizio dabrafenib 150 mg", doc_ids=("A",)),
            make_entry("CTL_000002", "2020-05-01", "treatment",
                       "Inizio dabrafenib 150 mg", doc_ids=("B",)),
        ]
        out = ClinicalHistoryBuilder._deterministic_dedup(entries)
        self.assertEqual(len(out), 1)
        survivor = out[0]
        self.assertNotIn(survivor.entry_id, survivor.merged_into_ids)
        self.assertEqual(
            set(survivor.merged_into_ids),
            {"CTL_000001", "CTL_000002"} - {survivor.entry_id},
        )
        # Sources accumulated from both documents
        self.assertEqual(set(survivor.source_document_ids), {"A", "B"})

    def test_survivor_is_longest_description(self):
        entries = [
            make_entry("CTL_000001", "2020-05-01", "treatment",
                       "Inizio dabrafenib 150 mg", doc_ids=("A",)),
            make_entry("CTL_000002", "2020-05-01", "treatment",
                       "inizio dabrafenib 150 mg bid", doc_ids=("B",)),
        ]
        out = ClinicalHistoryBuilder._deterministic_dedup(entries)
        self.assertEqual(out[0].entry_id, "CTL_000002")
        self.assertEqual(out[0].merged_into_ids, ["CTL_000001"])

    def test_different_date_not_merged(self):
        entries = [
            make_entry("CTL_000001", "2020-05-01", "treatment",
                       "Inizio dabrafenib 150 mg"),
            make_entry("CTL_000002", "2020-06-01", "treatment",
                       "Inizio dabrafenib 150 mg"),
        ]
        out = ClinicalHistoryBuilder._deterministic_dedup(entries)
        self.assertEqual(len(out), 2)

    def test_different_category_not_merged(self):
        entries = [
            make_entry("CTL_000001", "2020-05-01", "treatment",
                       "Inizio dabrafenib 150 mg"),
            make_entry("CTL_000002", "2020-05-01", "diagnosis",
                       "Inizio dabrafenib 150 mg"),
        ]
        out = ClinicalHistoryBuilder._deterministic_dedup(entries)
        self.assertEqual(len(out), 2)

    def test_unrelated_entries_preserved(self):
        entries = [
            make_entry("CTL_000001", "2020-05-01", "treatment",
                       "Inizio dabrafenib 150 mg"),
            make_entry("CTL_000002", "2020-05-01", "treatment",
                       "Inizio dabrafenib 150 mg"),
            make_entry("CTL_000003", "2020-09-10", "toxicity",
                       "Ipotiroidismo 3° grado"),
        ]
        out = ClinicalHistoryBuilder._deterministic_dedup(entries)
        self.assertEqual(len(out), 2)

    def test_does_not_merge_semantically_distinct_text(self):
        entries = [
            make_entry("CTL_000001", "2020-05-01", "treatment",
                       "Inizio dabrafenib 150 mg"),
            make_entry("CTL_000002", "2020-05-01", "treatment",
                       "Sospensione dabrafenib per tossicita'"),
        ]
        out = ClinicalHistoryBuilder._deterministic_dedup(entries)
        self.assertEqual(len(out), 2)


class TestApplyDedupGroups(unittest.TestCase):
    """Consumption of the new ``groups`` contract."""

    def test_groups_contract_canonical_and_provenance(self):
        entries = [
            make_entry("CTL_000001", "2020-05-01", "treatment",
                       "Nivolumab", doc_ids=("A",), texts=("tA",)),
            make_entry("CTL_000002", "2020-05-05", "treatment",
                       "Inizio nivolumab adiuvante", doc_ids=("B",),
                       texts=("tB",)),
            make_entry("CTL_000003", "2020-06-01", "follow_up",
                       "Controllo trimestrale", doc_ids=("C",)),
        ]
        result = {
            "groups": [{
                "kept_id": "CTL_000002",
                "merged_into_ids": ["CTL_000001"],
                "canonical_description": "Inizio nivolumab con intento adiuvante",
                "date_observed": "2020-05-01",
                "category": "treatment",
                "status": "active",
            }],
        }
        out = apply_groups(entries, result)

        self.assertEqual(len(out), 2)
        kept = next(e for e in out if e.entry_id == "CTL_000002")
        self.assertEqual(
            kept.description,
            "Inizio nivolumab con intento adiuvante",
        )
        self.assertEqual(kept.merged_into_ids, ["CTL_000001"])
        self.assertEqual(set(kept.source_document_ids), {"A", "B"})
        # Group date (earliest/most precise) wins
        self.assertEqual(kept.date_observed, "2020-05-01")
        self.assertTrue(all(e.entry_id != "CTL_000001" for e in out))

    def test_groups_contract_date_fallback_to_earliest_merged(self):
        entries = [
            make_entry("CTL_000001", "2020-03-10", "treatment", "Nivolumab"),
            make_entry("CTL_000002", "", "treatment", "Inizio nivolumab"),
        ]
        result = {
            "groups": [{
                "kept_id": "CTL_000002",
                "merged_into_ids": ["CTL_000001"],
                "canonical_description": "Inizio nivolumab",
                "date_observed": "",
                "category": "treatment",
                "status": "active",
            }],
        }
        out = apply_groups(entries, result)
        kept = out[0]
        self.assertEqual(kept.date_observed, "2020-03-10")

    def test_legacy_contract(self):
        entries = [
            make_entry("CTL_000001", "2020-05-01", "treatment", "Nivolumab"),
            make_entry("CTL_000002", "2020-05-01", "treatment", "Dabrafenib"),
        ]
        result = {
            "removed_entry_ids": ["CTL_000001"],
            "enrichments": {"CTL_000002": "Terapia combinata"},
        }
        out = apply_groups(entries, result)
        self.assertEqual(len(out), 1)
        self.assertIn("Terapia combinata", out[0].description)

    def test_no_groups_returns_all(self):
        entries = [
            make_entry("CTL_000001", "2020-05-01", "treatment", "Nivolumab"),
        ]
        out = apply_groups(entries, {"groups": []})
        self.assertEqual(len(out), 1)

    def test_hallucinated_kept_id_keeps_merged_entries(self):
        """A kept_id not in the registry must NOT drop the fused entries."""
        entries = [
            make_entry("CTL_000001", "2020-05-01", "treatment", "Nivolumab"),
            make_entry("CTL_000002", "2020-05-05", "treatment",
                       "Nivolumab: terapia avviata"),
        ]
        result = {
            "groups": [{
                "kept_id": "CTL_999999",  # hallucinated — no such entry
                "merged_into_ids": ["CTL_000001"],
                "canonical_description": "Inizio nivolumab",
                "date_observed": "2020-05-01",
            }],
        }
        out = apply_groups(entries, result)
        # No entry is removed because there is no survivor to inherit it.
        self.assertEqual(len(out), 2)
        self.assertEqual(
            {e.entry_id for e in out}, {"CTL_000001", "CTL_000002"}
        )

    def test_hallucinated_kept_id_also_in_merged_is_still_safe(self):
        """kept_id inside merged_into_ids must not delete the survivor."""
        entries = [
            make_entry("CTL_000001", "2020-05-01", "treatment", "Nivolumab"),
        ]
        result = {
            "groups": [{
                "kept_id": "CTL_000001",
                "merged_into_ids": ["CTL_000001"],  # self-reference
                "canonical_description": "Inizio nivolumab",
            }],
        }
        out = apply_groups(entries, result)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].entry_id, "CTL_000001")

    def test_group_date_is_normalized_to_iso(self):
        entries = [
            make_entry("CTL_000001", "2020-05-01", "treatment", "Nivolumab"),
            make_entry("CTL_000002", "2020-05-05", "treatment",
                       "Nivolumab: terapia avviata"),
        ]
        result = {
            "groups": [{
                "kept_id": "CTL_000002",
                "merged_into_ids": ["CTL_000001"],
                "canonical_description": "Inizio nivolumab",
                "date_observed": "01/02/2024",  # European, not ISO
            }],
        }
        out = apply_groups(entries, result)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].date_observed, "2024-02-01")

    def test_group_date_unparseable_falls_back_to_survivor(self):
        entries = [
            make_entry("CTL_000001", "2020-05-01", "treatment", "Nivolumab"),
            make_entry("CTL_000002", "2020-05-05", "treatment",
                       "Nivolumab: terapia avviata"),
        ]
        result = {
            "groups": [{
                "kept_id": "CTL_000002",
                "merged_into_ids": ["CTL_000001"],
                "canonical_description": "Inizio nivolumab",
                "date_observed": "febbraio 2024",  # free-form
            }],
        }
        out = apply_groups(entries, result)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].date_observed, "2020-05-05")


class TestDeduplicateAll(unittest.TestCase):
    """End-to-end: deterministic pre-filter + LLM groups."""

    def test_full_pipeline_with_canonical_synthesis(self):
        llm = FakeLlm(groups=[{
            "kept_id": "CTL_000001",
            "merged_into_ids": ["CTL_000003"],
            "canonical_description": "Inizio nivolumab adiuvante",
            "date_observed": "2020-05-01",
            "category": "treatment",
            "status": "active",
        }])
        builder = builder_with_llm(llm)

        entries = [
            # Near-verbatim pair → collapsed by the deterministic pre-filter
            make_entry("CTL_000001", "2020-05-01", "treatment",
                       "Inizio nivolumab 480 mg", doc_ids=("A",)),
            make_entry("CTL_000002", "2020-05-01", "treatment",
                       "inizio nivolumab 480 mg", doc_ids=("B",)),
            # Semantic duplicate → merged by the LLM stage
            make_entry("CTL_000003", "2020-05-05", "treatment",
                       "Nivolumab: terapia avviata", doc_ids=("C",)),
            # Unrelated entry → untouched
            make_entry("CTL_000004", "2020-09-10", "toxicity",
                       "Ipotiroidismo 3° grado", doc_ids=("D",)),
        ]

        out = builder._deduplicate_all(entries)

        self.assertEqual(len(out), 2)
        nivolumab = next(e for e in out if e.entry_id == "CTL_000001")
        self.assertEqual(nivolumab.description, "Inizio nivolumab adiuvante")
        self.assertEqual(set(nivolumab.merged_into_ids),
                         {"CTL_000002", "CTL_000003"})
        self.assertEqual(set(nivolumab.source_document_ids),
                         {"A", "B", "C"})
        self.assertEqual(nivolumab.date_observed, "2020-05-01")
        self.assertTrue(all(e.entry_id != "CTL_000002" for e in out))
        self.assertTrue(all(e.entry_id != "CTL_000003" for e in out))

        # The LLM receives the post-deterministic survivors (not the originals)
        llm_ids = {e["entry_id"] for e in llm.called_with}
        self.assertEqual(llm_ids, {"CTL_000001", "CTL_000003", "CTL_000004"})

    def test_scalability_20_duplicates(self):
        llm = FakeLlm(groups=[])
        builder = builder_with_llm(llm)
        entries = [
            make_entry(
                f"CTL_{i:06d}", "2020-05-01", "treatment",
                "Inizio dabrafenib 150 mg" + ("." * (i % 3)),
                doc_ids=(f"D{i}",),
            )
            for i in range(1, 21)
        ]
        out = builder._deduplicate_all(entries)
        # All 20 near-verbatim duplicates collapse deterministically
        self.assertEqual(len(out), 1)
        self.assertEqual(len(out[0].merged_into_ids), 19)

    def test_deduplicate_existing_returns_removed_count(self):
        class FakeRepo:
            def __init__(self, entries):
                self._entries = entries
                self.saved = None

            def get_by_patient(self, patient_id):
                return list(self._entries)

            def replace_all_for_patient(self, patient_id, final):
                self.saved = final

        repo = FakeRepo([
            make_entry("CTL_000001", "2020-05-01", "treatment",
                       "Inizio dabrafenib 150 mg"),
            make_entry("CTL_000002", "2020-05-01", "treatment",
                       "inizio dabrafenib 150 mg"),
        ])
        builder = ClinicalHistoryBuilder(
            timeline_repo=repo,
            document_repo=None,
            cs_repo=None,
            clinical_state_llm_client=FakeLlm(groups=[]),
            audit_repo=None,
        )
        removed = builder.deduplicate_existing("P001")
        self.assertEqual(removed, 1)
        self.assertEqual(len(repo.saved), 1)


if __name__ == "__main__":
    unittest.main()

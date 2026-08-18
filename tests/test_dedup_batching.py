"""Tests for the context-sized, parallel dedup batching in LlmClient."""

from __future__ import annotations

import unittest
from unittest import mock

from emr_analyzer.extraction.llm_client import LlmClient
from emr_analyzer.settings import LLMRoleConfig


def make_entries(count: int) -> list[dict]:
    return [
        {
            "entry_id": f"e{i}",
            "date_observed": f"2020-01-{i + 1:02d}",
            "category": "symptom",
            "description": f"descrizione {i}",
        }
        for i in range(count)
    ]


class TestDedupBatchSize(unittest.TestCase):
    def test_default_context_reaches_cap(self):
        client = LlmClient(model="local-test")
        self.assertEqual(client._dedup_batch_size(), 120)

    def test_scales_with_context(self):
        small = LlmClient(config=LLMRoleConfig(
            model="m", context_length=8192
        ))
        large = LlmClient(config=LLMRoleConfig(
            model="m", context_length=131072
        ))
        self.assertGreater(large._dedup_batch_size(),
                           small._dedup_batch_size())

    def test_respects_bounds(self):
        huge = LlmClient(config=LLMRoleConfig(
            model="m", context_length=2_000_000
        ))
        tiny = LlmClient(config=LLMRoleConfig(
            model="m", context_length=512
        ))
        self.assertEqual(huge._dedup_batch_size(), 120)
        self.assertGreaterEqual(tiny._dedup_batch_size(), 20)


class TestDedupBatching(unittest.TestCase):
    def setUp(self):
        self.client = LlmClient(model="local-test")
        # Small batch for tests: 5 entries per call, 2-entry overlap.
        self.client._dedup_batch_size = lambda: 5
        self.client._DEDUP_OVERLAP = 2

    def test_single_call_when_registry_fits(self):
        entries = make_entries(4)
        with mock.patch.object(
            self.client, "_dedup_batch", return_value={"groups": []}
        ) as fake_batch:
            result = self.client.deduplicate_timeline(entries)
        fake_batch.assert_called_once_with(entries)
        self.assertEqual(result, {"groups": []})

    def test_slices_are_sorted_with_overlap(self):
        entries = make_entries(10)
        seen = []

        def fake_batch(slice_entries):
            seen.append([e["entry_id"] for e in slice_entries])
            return {"groups": []}

        with mock.patch.object(
            self.client, "_dedup_batch", side_effect=fake_batch
        ):
            self.client.deduplicate_timeline(entries)

        # 10 entries, batch 5, overlap 2 → slices [0:5], [3:8], [6:10].
        self.assertEqual(seen[0], ["e0", "e1", "e2", "e3", "e4"])
        self.assertEqual(seen[1], ["e3", "e4", "e5", "e6", "e7"])
        self.assertEqual(seen[2], ["e6", "e7", "e8", "e9"])
        # Entries are delivered to the LLM sorted by (date, category).
        for slice_ids in seen:
            self.assertEqual(slice_ids, sorted(slice_ids))

    def test_parallel_executor_used_with_multiple_workers(self):
        self.client.parallel_workers = 3
        entries = make_entries(10)
        real_pool = None
        import concurrent.futures
        real_pool = concurrent.futures.ThreadPoolExecutor

        with mock.patch(
            "concurrent.futures.ThreadPoolExecutor", wraps=real_pool
        ) as pool_cls, mock.patch.object(
            self.client, "_dedup_batch", return_value={"groups": []}
        ) as fake_batch:
            self.client.deduplicate_timeline(entries)

        self.assertEqual(fake_batch.call_count, 3)
        self.assertEqual(pool_cls.call_args.kwargs["max_workers"], 3)

    def test_conflicting_groups_across_batches_resolve_cleanly(self):
        """Two batches keep opposite ends of the same pair.

        The second group must absorb the first one, leaving exactly one
        group with no self-references in ``merged_into_ids``.
        """
        entries = make_entries(10)

        def fake_batch(slice_entries):
            ids = [e["entry_id"] for e in slice_entries]
            if "e3" not in ids or "e4" not in ids:
                return {"groups": []}
            if "e2" in ids:
                # First slice: keeps e3, merges e4.
                return {"groups": [{
                    "kept_id": "e3", "merged_into_ids": ["e4"],
                    "canonical_description": "descrizione 3",
                    "date_observed": "2020-01-04",
                    "category": "symptom", "status": "active",
                }]}
            # Overlap slice: keeps e4, merges e3 (the conflict).
            return {"groups": [{
                "kept_id": "e4", "merged_into_ids": ["e3"],
                "canonical_description": "descrizione 4",
                "date_observed": "2020-01-05",
                "category": "symptom", "status": "active",
            }]}

        with mock.patch.object(
            self.client, "_dedup_batch", side_effect=fake_batch
        ):
            result = self.client.deduplicate_timeline(entries)

        groups = result["groups"]
        self.assertEqual(len(groups), 1)
        kept = groups[0]["kept_id"]
        merged = groups[0]["merged_into_ids"]
        self.assertEqual(set([kept] + merged), {"e3", "e4"})
        self.assertNotIn(kept, merged, "nessuna auto-referenza nel merge")

    def test_empty_registry(self):
        self.assertEqual(self.client.deduplicate_timeline([]), {"groups": []})


if __name__ == "__main__":
    unittest.main()

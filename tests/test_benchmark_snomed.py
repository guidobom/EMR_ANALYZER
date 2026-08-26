"""Fase 7 — benchmark tooling: the recording-retriever proxy.

``_RecordingRetriever`` wraps ``SnomedCandidateRetriever`` so the benchmark can
compute per-case retrieval diagnostics (retrieved-code union, candidate-set
sizes, empty chunks) that ``tools/score_snomed_benchmark.py`` turns into
``retrieval_recall``.  The proxy is exercised against the synthetic RF2
fixture, exactly the machinery ``benchmark_atomic_round.py snomed`` uses —
without any LLM.
"""

from __future__ import annotations

from pathlib import Path
import sys

import pytest

from emr_analyzer.clinical.snomed import (
    SnomedCandidateRetriever,
    SnomedIndex,
    load_snapshot,
)

TOOLS_DIR = Path(__file__).resolve().parents[1] / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from benchmark_atomic_round import _RecordingRetriever  # noqa: E402

FIXTURE_DIR = (
    Path(__file__).resolve().parent / "fixtures" / "snomed_rf2" / "Snapshot_20260101"
)


@pytest.fixture(scope="module")
def snapshot():
    return load_snapshot(FIXTURE_DIR)


def _recording(snapshot, **kwargs):
    inner = SnomedCandidateRetriever(
        SnomedIndex(snapshot.concepts.values()),
        release_digest=snapshot.digest,
        **kwargs,
    )
    return _RecordingRetriever(inner)


def _retrieved_codes(retriever):
    return sorted({code for item in retriever.candidate_sets for code in item.codes})


class TestRecordingRetriever:
    def test_records_one_set_per_call(self, snapshot):
        retriever = _recording(snapshot)
        assert retriever.candidate_sets == []
        retriever.candidates_for_text("diabete mellito")
        retriever.candidates_for_text("HT")
        assert len(retriever.candidate_sets) == 2
        # Each recorded set is a real SnomedCandidateSet.
        assert [item.empty for item in retriever.candidate_sets] == [False, False]

    def test_release_digest_is_proxied(self, snapshot):
        retriever = _recording(snapshot)
        assert retriever.release_digest == snapshot.digest

    def test_retrieved_codes_union_and_digests(self, snapshot):
        # The union of codes across recorded sets is the recall ceiling for
        # constrained generation, exactly what the scorer compares against gold.
        retriever = _recording(snapshot)
        retriever.candidates_for_text("Paziente con HT e diabete mellito")
        retriever.candidates_for_text("BPCO")
        codes = _retrieved_codes(retriever)
        assert set(codes) == {"38341003", "44054006", "13645005"}
        # Deterministic ordering matches the payload written by ``run_snomed``.
        assert codes == ["13645005", "38341003", "44054006"]

    def test_sizes_and_empty_counts_feed_the_payload(self, snapshot):
        retriever = _recording(snapshot)
        retriever.candidates_for_text("diabete mellito")
        retriever.candidates_for_text("zzzqwerty")
        sizes = [len(item.codes) for item in retriever.candidate_sets]
        assert sizes[0] >= 1
        assert sizes[1] == 0
        assert sum(1 for item in retriever.candidate_sets if item.empty) == 1

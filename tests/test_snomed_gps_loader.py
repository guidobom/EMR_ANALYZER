"""Tests for the SNOMED Global Patient Set (GPS) freeset loader.

The freeset is a flat TSV (``ConceptID | Active | FSN | USPreferredTerm``)
with no IS-A relationships, no synonyms and only US-English terms.  ``load_gps``
builds a ``ReleaseSnapshot`` in the same shape as the RF2 loader so the index,
retrieval and constrained extraction run unchanged.
"""

from __future__ import annotations

from pathlib import Path
import zipfile

import pytest

from emr_analyzer.clinical.snomed import (
    SnomedIndex,
    is_event_like,
    load_gps,
    load_snapshot,
    snomed_manager,
)

# Official freeset rows: ConceptID, Active, FSN, USPreferredTerm.
_GPS_ROWS = [
    ("22298006", "1", "Myocardial infarction (disorder)", "Myocardial infarction"),
    ("386661006", "1", "Fever (finding)", "Fever"),
    ("425417004", "1", "Platelet count outside reference range (finding)",
     "Platelet count outside reference range"),
    ("73211009", "0", "Diabetes mellitus (disorder)", "Diabetes mellitus"),
]


def _write_gps_tsv(
    path: Path,
    rows: list[tuple[str, str, str, str]] | None = None,
    *,
    header: bool = True,
    delimiter: str = "\t",
) -> Path:
    rows = rows if rows is not None else _GPS_ROWS
    lines: list[str] = []
    if header:
        lines.append(delimiter.join(
            ["ConceptID", "Active", "FSN", "USPreferredTerm"]
        ))
    for concept_id, active, fsn, preferred in rows:
        lines.append(delimiter.join([concept_id, active, fsn, preferred]))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


class TestLoadGps:
    def test_official_header_tsv(self, tmp_path):
        source = _write_gps_tsv(tmp_path / "freeset.tsv")
        snapshot = load_gps(source)
        assert snapshot.edition == "GPS"
        assert snapshot.languages == ("en",)
        assert len(snapshot.concepts) == 3  # the active=0 row is dropped
        concept = snapshot.concept("22298006")
        assert concept is not None
        assert concept.preferred_term(("en",)) == "Myocardial infarction"
        assert concept.fsn(("en",)) == "Myocardial infarction (disorder)"
        assert concept.parents == frozenset()
        assert concept.children == frozenset()

    def test_fsn_and_preferred_pt_typed(self, tmp_path):
        source = _write_gps_tsv(tmp_path / "freeset.tsv")
        snapshot = load_gps(source)
        concept = snapshot.concept("22298006")
        descriptions = {d.type_id: d for d in concept.descriptions}
        from emr_analyzer.clinical.snomed.models import FSN_TYPE, SYNONYM_TYPE
        assert descriptions[FSN_TYPE].term == "Myocardial infarction (disorder)"
        preferred = descriptions[SYNONYM_TYPE]
        assert preferred.term == "Myocardial infarction"
        assert preferred.acceptability == "preferred"
        assert preferred.lang == "en"

    def test_headerless_positional_fallback(self, tmp_path):
        source = _write_gps_tsv(
            tmp_path / "freeset.tsv", header=False,
        )
        snapshot = load_gps(source)
        assert snapshot.concept("386661006") is not None
        assert snapshot.concept("386661006").preferred_term(("en",)) == "Fever"

    def test_pipe_delimited_variant(self, tmp_path):
        source = _write_gps_tsv(
            tmp_path / "freeset.txt", delimiter="|",
        )
        snapshot = load_gps(source)
        assert len(snapshot.concepts) == 3

    def test_directory_resolution(self, tmp_path):
        directory = tmp_path / "gps_dir"
        directory.mkdir()
        _write_gps_tsv(directory / "freeset.tsv")
        snapshot = load_gps(directory)
        assert snapshot.concept("22298006") is not None

    def test_zip_resolution(self, tmp_path):
        archive = tmp_path / "freeset.zip"
        with zipfile.ZipFile(archive, "w") as handle:
            handle.write(
                _write_gps_tsv(tmp_path / "freeset.tsv"),
                arcname="freeset.tsv",
            )
        snapshot = load_gps(archive)
        assert snapshot.concept("22298006") is not None

    def test_digest_deterministic(self, tmp_path):
        source = _write_gps_tsv(tmp_path / "freeset.tsv")
        first = load_gps(source)
        second = load_gps(source)
        assert first.digest == second.digest
        assert first.digest

    def test_missing_source_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_gps(tmp_path / "nope")

    def test_empty_freeset_raises(self, tmp_path):
        source = _write_gps_tsv(tmp_path / "empty.tsv", rows=[])
        with pytest.raises(ValueError):
            load_gps(source)


class TestGpsIndexIntegration:
    def test_search_finds_gps_terms_with_italian_langs(self, tmp_path):
        source = _write_gps_tsv(tmp_path / "freeset.tsv")
        snapshot = load_gps(source)
        index = SnomedIndex(snapshot.concepts.values())
        rows = index.search(
            "the patient has fever", langs=("en", "it"), cap=8, min_score=0.2,
        )
        codes = {concept.concept_id for concept, *_ in rows}
        assert "386661006" in codes

    def test_event_like_from_gps_fsn(self, tmp_path):
        source = _write_gps_tsv(tmp_path / "freeset.tsv")
        snapshot = load_gps(source)
        disorder = snapshot.concept("22298006")
        assert is_event_like(disorder, langs=("en",))


class TestGpsManagerWiring:
    def test_gps_kind_via_manager(self, tmp_path):
        source = _write_gps_tsv(tmp_path / "freeset.tsv")
        index = snomed_manager.index(str(source), kind="gps")
        assert index.concept("22298006") is not None

    def test_rf2_kind_still_requires_rf2_release(self, tmp_path):
        source = _write_gps_tsv(tmp_path / "freeset.tsv")
        # The RF2 path must stay untouched: a GPS freeset is not a valid RF2
        # release and must raise exactly as before.
        with pytest.raises(FileNotFoundError):
            snomed_manager.snapshot(str(source), kind="rf2")

    def test_unknown_kind_raises(self, tmp_path):
        source = _write_gps_tsv(tmp_path / "freeset.tsv")
        with pytest.raises(ValueError):
            snomed_manager.snapshot(str(source), kind="nope")

    def test_rf2_manager_untouched(self):
        # The licensed RF2 fixture still loads through the default kind.
        fixture = (
            Path(__file__).resolve().parent / "fixtures" / "snomed_rf2"
            / "Snapshot_20260101"
        )
        snapshot = load_snapshot(fixture)
        assert "44054006" in snapshot.concepts

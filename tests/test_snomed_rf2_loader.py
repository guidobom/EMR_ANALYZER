"""Tests for the SNOMED RF2 release loader (snapshot semantics)."""

from __future__ import annotations

from pathlib import Path

import pytest

from emr_analyzer.clinical.snomed import load_snapshot
from emr_analyzer.clinical.snomed.rf2_loader import (
    CONCEPT_HEADER,
    DESCRIPTION_HEADER,
    RELATIONSHIP_HEADER,
    _latest_active,
    _parse_rows,
)

FIXTURE_DIR = (
    Path(__file__).resolve().parent / "fixtures" / "snomed_rf2" / "Snapshot_20260101"
)


def _write_mini_release(
    root: Path,
    *,
    languages=("en", "it"),
    with_concept_headers: bool = True,
) -> Path:
    """Write a tiny valid RF2 release with a known concept."""
    root.mkdir(parents=True, exist_ok=True)
    # ``with_concept_headers=False`` exercises the reordered-header tolerance:
    # the header line is present but its columns are in a different order, and
    # the data rows follow that same column order.
    concept_header = (
        CONCEPT_HEADER if with_concept_headers
        else ("effectiveTime", "active", "moduleId", "definitionStatusId", "id")
    )
    concept_rows = [{
        "id": "99900001", "effectiveTime": "20260101", "active": "1",
        "moduleId": "900000000000207008",
        "definitionStatusId": "900000000000074008",
    }]
    (root / "sct2_Concept_Full_INT_20260101.txt").write_text(
        "|".join(concept_header) + "\n"
        + "".join("|".join(row[c] for c in concept_header) + "\n"
                  for row in concept_rows),
        encoding="utf-8",
    )
    for lang_index, lang in enumerate(languages):
        rows = [
            {"id": f"8000000{lang_index}", "effectiveTime": "20260101",
             "active": "1", "moduleId": "900000000000207008",
             "conceptId": "99900001", "languageCode": lang,
             "typeId": "900000000000003001",
             "term": "Test disorder (disorder)",
             "caseSignificanceId": "900000000000448009"},
            {"id": f"7000000{lang_index}", "effectiveTime": "20260101",
             "active": "1", "moduleId": "900000000000207008",
             "conceptId": "99900001", "languageCode": lang,
             "typeId": "900000000000013009",
             "term": "Test disorder", "caseSignificanceId": "900000000000448009"},
        ]
        (root / f"sct2_Description_Full-{lang}_INT_20260101.txt").write_text(
            "|".join(DESCRIPTION_HEADER) + "\n"
            + "".join("|".join(row[c] for c in DESCRIPTION_HEADER) + "\n"
                      for row in rows),
            encoding="utf-8",
        )
    rel_rows = [{
        "id": "50000001", "effectiveTime": "20260101", "active": "1",
        "moduleId": "900000000000207008", "sourceId": "99900001",
        "destinationId": "404684003", "relationshipGroup": "0",
        "typeId": "116680003",
        "characteristicTypeId": "900000000000011006",
        "modifierId": "900000000000451002",
    }]
    (root / "sct2_Relationship_Full_INT_20260101.txt").write_text(
        "|".join(RELATIONSHIP_HEADER) + "\n"
        + "".join("|".join(row[c] for c in RELATIONSHIP_HEADER) + "\n"
                  for row in rel_rows),
        encoding="utf-8",
    )
    return root


class TestLoadSnapshot:
    def test_loads_fixture_with_bilingual_languages(self):
        snapshot = load_snapshot(FIXTURE_DIR)
        assert len(snapshot.concepts) >= 50
        assert "44054006" in snapshot.concepts
        assert set(snapshot.languages) == {"en", "it"}

    def test_preferred_term_honours_refset(self):
        snapshot = load_snapshot(FIXTURE_DIR)
        concept = snapshot.concept("44054006")
        assert concept.preferred_term(("it",)) == "Diabete mellito"
        assert concept.preferred_term(("en",)) == "Diabetes mellitus"

    def test_digest_deterministic(self):
        first = load_snapshot(FIXTURE_DIR)
        second = load_snapshot(FIXTURE_DIR)
        assert first.digest == second.digest
        assert first.digest

    def test_mini_release_language_tolerance(self, tmp_path):
        root = _write_mini_release(tmp_path, languages=("en",))
        snapshot = load_snapshot(root, languages=("en", "it"))
        assert "99900001" in snapshot.concepts
        # Only English is present: Italian is tolerated as missing.
        assert snapshot.languages == ("en",)

    def test_national_namespace_descriptions_are_merged(self, tmp_path):
        # A national extension (e.g. the Italian edition) ships its descriptions
        # under a national namespace rather than ``_INT_``.  The loader must
        # still merge them so the full-language RAG is usable.
        root = _write_mini_release(tmp_path, languages=("en",))
        it_rows = [
            {"id": f"80000{i}", "effectiveTime": "20260101", "active": "1",
             "moduleId": "900000000000207008", "conceptId": "99900001",
             "languageCode": "it", "typeId": "900000000000013009",
             "term": "Disturbo di prova",
             "caseSignificanceId": "900000000000448009"}
            for i in range(2)
        ]
        (root / "sct2_Description_Full-it_IT_20260101.txt").write_text(
            "|".join(DESCRIPTION_HEADER) + "\n"
            + "".join("|".join(row[c] for c in DESCRIPTION_HEADER) + "\n"
                      for row in it_rows),
            encoding="utf-8",
        )
        snapshot = load_snapshot(root, languages=("en", "it"))
        assert set(snapshot.languages) == {"en", "it"}
        concept = snapshot.concept("99900001")
        assert any(
            d.lang == "it" and d.term == "Disturbo di prova"
            for d in concept.descriptions
        )

    def test_reordered_header_is_tolerated(self, tmp_path):
        root = _write_mini_release(tmp_path, with_concept_headers=False)
        snapshot = load_snapshot(root)
        assert "99900001" in snapshot.concepts

    def test_missing_directory_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_snapshot(tmp_path / "nope")

    def test_missing_concept_file_raises(self, tmp_path):
        root = _write_mini_release(tmp_path)
        (root / "sct2_Concept_Full_INT_20260101.txt").unlink()
        with pytest.raises(FileNotFoundError):
            load_snapshot(root)


class TestSnapshotSemantics:
    def test_latest_active_wins(self):
        rows = [
            {"id": "1", "effectiveTime": "20250101", "active": "1",
             "term": "old"},
            {"id": "1", "effectiveTime": "20260101", "active": "1",
             "term": "new"},
            {"id": "2", "effectiveTime": "20260101", "active": "0",
             "term": "inactive"},
            {"id": "3", "effectiveTime": "20260101", "active": "1",
             "term": "kept"},
        ]
        latest = _latest_active(rows)
        assert latest["1"]["term"] == "new"
        assert "2" not in latest
        assert latest["3"]["term"] == "kept"

    def test_parse_rows_skips_header(self):
        text = "id|term\n1|a\n2|b\n"
        rows = _parse_rows(text, ("id", "term"))
        assert len(rows) == 2
        assert rows[0]["term"] == "a"

    def test_parse_rows_without_header(self):
        text = "1|a\n2|b\n"
        rows = _parse_rows(text, ("id", "term"))
        assert len(rows) == 2
        assert rows[0] == {"id": "1", "term": "a"}

    def test_is_a_relationships_are_attached(self):
        snapshot = load_snapshot(FIXTURE_DIR)
        concept = snapshot.concept("44054006")
        assert "64572001" in concept.parents
        assert "64572001" in snapshot.concept("44054006").parents


class TestFixtureIntegrity:
    def test_fixture_generated_files_present(self):
        assert (FIXTURE_DIR / "Snapshot" / "Content"
                / "sct2_Concept_Snapshot_INT_20260101.txt").exists()
        assert (FIXTURE_DIR / "Snapshot" / "Content"
                / "sct2_Description_Snapshot-it_INT_20260101.txt").exists()
        assert (FIXTURE_DIR / "Snapshot" / "Refset" / "Language"
                / "der2_cRefset_LanguageSnapshot-it_INT_20260101.txt").exists()

    def test_manifest_matches_concept_count(self):
        import csv
        with open(
            Path(__file__).resolve().parent / "fixtures" / "snomed_rf2"
            / "concepts.csv",
            newline="", encoding="utf-8",
        ) as handle:
            rows = list(csv.DictReader(handle))
        snapshot = load_snapshot(FIXTURE_DIR)
        assert len(rows) == len(snapshot.concepts)

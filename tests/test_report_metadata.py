import pytest

from emr_analyzer.clinical.report_metadata import report_date, with_report_date


@pytest.mark.parametrize("value", [None, "", "2025-02-29", "Mario 2024-01-01", "2024", "01/02/2024"])
def test_unknown_or_invalid_date_never_inserts_unvalidated_text(value):
    assert report_date(value) is None
    assert "non disponibile" in with_report_date("Testo clinico", value)


def test_header_is_idempotent_and_preserves_body_exactly():
    body = "Terapia iniziata il 01/01/2020.\n\n  Dose 5 mg.\n"
    result = with_report_date(body, "2024-02-29")
    assert result.endswith(body)
    assert with_report_date(result, "2024-02-29") == result
    corrected = with_report_date(result, "2024-03-01")
    assert "2024-02-29" not in corrected
    assert corrected.endswith(body)
    assert corrected.count("Data del referto") == 1



def test_migration_backs_up_and_skips_documents_without_normalization(tmp_path, monkeypatch, capsys):
    import json
    import sqlite3
    import sys
    from tools.add_report_dates import main
    folder = tmp_path / "P001/extraction"
    folder.mkdir(parents=True)
    body = "Testo clinico invariato.\n"
    for name in ("DOC_1.md", "DOC_2.md"):
        (folder / name).write_text(body)
    with sqlite3.connect(tmp_path / "emr_registry.db") as con:
        con.execute("CREATE TABLE documents (id, patient_id, document_date, metadata_json)")
        con.executemany("INSERT INTO documents VALUES (?,?,?,?)", [
            ("DOC_1", "P001", "2024-02-29", json.dumps({"clinical_text": {"deidentification_version": "v1"}})),
            ("DOC_2", "P001", "2024-02-29", "{}"),
        ])
    monkeypatch.setattr(sys, "argv", ["add_report_dates", str(tmp_path)])
    main()
    assert (folder / "DOC_1.md").read_text() == body
    monkeypatch.setattr(sys, "argv", ["add_report_dates", str(tmp_path), "--apply"])
    main()
    assert (folder / "DOC_1.md").read_text() == with_report_date(body, "2024-02-29")
    assert (folder / "DOC_2.md").read_text() == body
    backups = list((tmp_path / "metadata_backups").iterdir())
    assert len(backups) == 1
    assert (backups[0] / "P001/extraction/DOC_1.md").read_text() == body
    main()
    assert len(list((tmp_path / "metadata_backups").iterdir())) == 1

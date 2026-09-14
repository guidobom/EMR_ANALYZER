import hashlib
import sqlite3

import pytest

from emr_analyzer.database.engine import DatabaseEngine
from emr_analyzer.database.migrations import init_database
from tools.reset_project_to_pdfs import plan, apply_reset


def fixture_project(root):
    pdf = root / 'P001/documents/original/original.pdf'
    pdf.parent.mkdir(parents=True)
    pdf.write_bytes(b'%PDF-1.4 synthetic fixture')
    derived = root / 'P001/extraction/DOC_1.md'
    derived.parent.mkdir()
    derived.write_text('Clinical content to delete')
    db = DatabaseEngine(root / 'emr_registry.db')
    init_database(db)
    db.execute("INSERT INTO patients(id,pseudonym,notes,created_at,updated_at) VALUES (?,?,?,?,?)",
               ('P001', 'P001', 'Clinical notes', '2026-01-01', '2026-01-01'))
    db.execute("INSERT INTO documents(id,patient_id,filename,original_path,file_hash,document_type,"
               "import_date,extraction_status,metadata_json) VALUES (?,?,?,?,?,?,?,?,?)",
               ('DOC_1', 'P001', pdf.name, str(pdf), hashlib.sha256(pdf.read_bytes()).hexdigest(),
                'visita', '2026-01-01', 'done', '{"clinical_text":{}}'))
    db.commit()
    db.close()
    (root / 'emr_registry.db.bak-old').write_text('old clinical backup')
    return pdf, derived


def test_reset_preserves_original_and_rebuilds_empty_clinical_schema(tmp_path):
    pdf, derived = fixture_project(tmp_path)
    before = pdf.read_bytes()
    data = plan(tmp_path)
    assert derived.exists()  # planning never modifies the project
    apply_reset(data)
    assert pdf.read_bytes() == before
    assert not derived.exists()
    assert not (tmp_path / 'emr_registry.db.bak-old').exists()
    con = sqlite3.connect(tmp_path / 'emr_registry.db')
    try:
        assert con.execute('SELECT notes FROM patients').fetchone()[0] is None
        assert con.execute('SELECT extraction_status,metadata_json FROM documents').fetchone() == ('pending', None)
        assert con.execute('SELECT count(*) FROM clinical_category_registry').fetchone()[0] == 0
        assert con.execute('PRAGMA integrity_check').fetchone()[0] == 'ok'
        assert not con.execute('PRAGMA foreign_key_check').fetchall()
        assert con.execute("SELECT count(*) FROM sqlite_master WHERE type='trigger'").fetchone()[0] > 0
    finally:
        con.close()
    assert set(p for p in tmp_path.rglob('*') if p.is_file()) == {pdf, tmp_path / 'emr_registry.db'}


def test_missing_original_aborts_before_deletion(tmp_path):
    pdf, derived = fixture_project(tmp_path)
    pdf.unlink()
    with pytest.raises(ValueError, match='originale non verificato'):
        plan(tmp_path)
    assert derived.exists()


def test_unclassified_pdf_aborts_before_deletion(tmp_path):
    pdf, derived = fixture_project(tmp_path)
    (tmp_path / 'unclassified.pdf').write_bytes(pdf.read_bytes())
    with pytest.raises(ValueError, match='PDF fuori'):
        plan(tmp_path)
    assert derived.exists()

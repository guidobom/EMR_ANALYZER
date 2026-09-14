"""Regression coverage for dossier completeness and transaction integrity."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from emr_analyzer.database.engine import DatabaseEngine
from emr_analyzer.database.patient_repo import PatientRepository
from emr_analyzer.database.document_repo import DocumentRepository
from emr_analyzer.clinical.query_context import context_budget, reconcile_partials, split_text
from emr_analyzer.clinical.query_service import ClinicalQueryService, _split_oversized_event
from emr_analyzer.clinical.irae_analysis import chunk_entries, format_entry
from emr_analyzer.pipeline.pdf_extractor import PdfPlumberExtractor
from emr_analyzer.gui.workers import ClinicalHistoryQueryWorker


def test_repository_commit_does_not_escape_outer_rollback(tmp_path):
    db = DatabaseEngine(tmp_path / 'db.sqlite')
    db.execute('CREATE TABLE sample (value INTEGER)')
    with pytest.raises(RuntimeError):
        with db:
            db.execute('INSERT INTO sample VALUES (1)')
            db.commit()  # Repository-style commit.
            with db:
                db.execute('INSERT INTO sample VALUES (2)')
                db.commit()
            raise RuntimeError('late failure')
    assert db.execute('SELECT count(*) FROM sample').fetchone()[0] == 0
    with db:
        db.execute('INSERT INTO sample VALUES (3)')
        with pytest.raises(ValueError):
            with db:
                db.execute('INSERT INTO sample VALUES (4)')
                db.commit()
                raise ValueError('inner failure')
    assert [r[0] for r in db.execute('SELECT value FROM sample')] == [3]
    db.close()


def test_ids_continue_past_display_padding(tmp_path):
    db = DatabaseEngine(tmp_path / 'db.sqlite')
    db.execute('CREATE TABLE patients (id TEXT)')
    db.execute('CREATE TABLE documents (id TEXT)')
    db.executemany('INSERT INTO patients VALUES (?)', [('P999',), ('P1000',), ('custom',)])
    db.executemany('INSERT INTO documents VALUES (?)', [('DOC_999999',), ('DOC_1000000',)])
    assert PatientRepository(db).get_next_id() == 'P1001'
    assert DocumentRepository(db).get_next_id() == 'DOC_1000001'
    db.close()


def test_default_query_does_not_drop_events_after_500():
    events = [SimpleNamespace(event_id=f'EVT_{i:x}', canonical_entity='dato',
                             summary_short='dato', category='diagnosis',
                             status='active', first_evidence_date='2020') for i in range(650)]
    repo = Mock()
    repo.get_events.return_value = events
    repo.get_event_detail.side_effect = lambda eid: {'event': {'event_id': eid}, 'evidence': []}
    assert len(ClinicalQueryService(repo).retrieve('P001', 'domanda generica')) == 650
    repo.search_events.assert_not_called()


def test_context_chunks_preserve_text_and_bounds():
    text = ('abc\n' * 1000) + 'tail'
    chunks = split_text(text, 513)
    assert ''.join(chunks) == text
    assert max(map(len, chunks)) <= 513
    chunks = _split_oversized_event('[#EVT_a] ' + 'summary' * 100 + '\n' + 'source' * 500, 150)
    assert max(map(len, chunks)) <= 150
    assert all('[#EVT_a]' in c for c in chunks)


def test_small_context_rejected_instead_of_forcing_12000_chars():
    with pytest.raises(ValueError, match='Contesto insufficiente'):
        context_budget(SimpleNamespace(context_length=4096, max_output_tokens=4096))


def test_reconciliation_never_forces_oversized_pairs():
    llm = Mock()
    result = reconcile_partials(llm, ['a' * 900, 'b' * 900], 'q', 's', max_chars=1000)
    llm.generate_text.assert_not_called()
    assert 'a' * 900 in result and 'b' * 900 in result


def test_timeline_fallback_has_no_import_error_and_reads_tail():
    from emr_analyzer.models.clinical_evidence import ClinicalEvidence
    evidence = []
    # Use the real timeline serializer with synthetic evidence.
    for i in range(200):
        evidence.append(ClinicalEvidence(
            evidence_id=f'E{i}', patient_id='P001', document_id='DOC_1',
            category='symptom', fact_type='symptom',
            normalized_entity=f'osservazione_{i} ' + 'x' * 100,
            source_text='test',
        ))
    llm = Mock(context_length=8192, max_output_tokens=1024)
    llm.generate_text.return_value = 'risposta'
    repo = Mock()
    repo.get_by_patient.return_value = evidence
    worker = ClinicalHistoryQueryWorker(llm, [], '', 'storia completa',
                                        patient_id='P001', evidence_repo=repo)
    worker._run_timeline_query('system')
    prompts = '\n'.join(c.args[0] for c in llm.generate_text.call_args_list)
    assert 'osservazione_0' in prompts and 'osservazione_199' in prompts


def test_zero_overlap_and_large_overlap_respect_budget():
    entries = [{'entry_id': str(i), 'description': 'x' * 80} for i in range(20)]
    for overlap in (0, 10):
        chunks = chunk_entries(entries, chunk_chars=350, overlap=overlap)
        assert all(sum(len(format_entry(e)) + 1 for e in c) <= 350 for c in chunks)
        assert {e['entry_id'] for c in chunks for e in c} == {str(i) for i in range(20)}
        if overlap == 0:
            assert sum(map(len, chunks)) == len(entries)


def test_windows1252_text_not_misdecoded_as_utf16(tmp_path):
    path = tmp_path / 'referto.txt'
    text = 'Tossicità lieve.'
    assert len(text.encode('cp1252')) % 2 == 0
    path.write_bytes(text.encode('cp1252'))
    assert PdfPlumberExtractor._read_text(path) == text


def test_citations_reject_invented_pages_and_unknown_ids():
    from emr_analyzer.clinical.query_service import answer_has_valid_citations
    ids, pairs, pages = {'EVT_abc'}, {('EVT_abc', 'DOC_1')}, {('EVT_abc', 'DOC_1', '2')}
    assert answer_has_valid_citations('[#EVT_abc; DOC_1:p.2]', ids, pairs, pages)
    assert not answer_has_valid_citations('[#EVT_abc; DOC_1:p.99]', ids, pairs, pages)
    assert not answer_has_valid_citations('[#EVT_abc; DOC_1:p.2] #EVT_inventato', ids, pairs, pages)


def test_empty_registry_without_evidence_never_calls_model():
    llm, repo = Mock(), Mock()
    repo.get_events.return_value = []
    worker = ClinicalHistoryQueryWorker(llm, [], 'profilo vecchio', 'domanda',
                                        registry_repo=repo, patient_id='P001')
    assert 'Nessuna evidenza' in worker._run_registry_query('system')
    llm.generate_text.assert_not_called()


def test_deletion_never_collects_external_original(tmp_path):
    from emr_analyzer.clinical.document_deletion import DocumentDeletionService
    root = tmp_path / 'workspace'
    root.mkdir()
    external = tmp_path / 'external.pdf'
    external.write_bytes(b'synthetic-original')
    doc = SimpleNamespace(id='DOC_1', patient_id='P001', original_path=str(external))
    service = DocumentDeletionService(Mock(), Mock(), workspaces_dir=root)
    with pytest.raises(ValueError, match='esterno'):
        service._document_artifacts(doc)
    assert external.read_bytes() == b'synthetic-original'

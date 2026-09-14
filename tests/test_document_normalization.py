from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from emr_analyzer.extraction.document_normalization import normalize_document
from emr_analyzer.extraction.clinical_text_isolator import ClinicalTextIsolationResult, ClinicalTextIsolationError
from emr_analyzer.pipeline.pdf_extractor import PdfExtractionResult, PdfPage
from emr_analyzer.gui.documents_tab import DocumentsTab
from emr_analyzer.models.document import DocumentRecord


def source():
    return PdfExtractionResult('original.pdf', [
        PdfPage(1, 100, 100, 'Nessuna febbre. Dose 5 mg.', []),
        PdfPage(2, 100, 100, 'Controllo il 01/03/2024.', []),
    ])


def test_page_provenance_is_added_after_normalization():
    isolator = Mock()
    isolator.isolate.side_effect = lambda text, **kw: ClinicalTextIsolationResult(text, 'fake', chunk_count=1)
    result = normalize_document(isolator, '', parsing_result=source())
    assert result.text == 'Nessuna febbre. Dose 5 mg.\nControllo il 01/03/2024.'
    assert [p['page'] for p in result.retention_audit['output_pages']] == [1, 2]
    assert result.chunk_count == 2
    assert all('<!-- page' not in call.args[0] for call in isolator.isolate.call_args_list)


def test_signature_on_last_page_redacts_author_columns_on_previous_pages():
    from emr_analyzer.clinical.dossier_query import _mapped_source_pages
    parsed = PdfExtractionResult('synthetic.pdf', [
        PdfPage(1, 100, 100,
            '19.01.2026 15:16  Scala dolore NRS 02 Dolore lieve    MARIO ROSSI\n'
            'Dispnea. Terapia 5 mg. Malattia di Parkinson.\n'
            '19.01.2026 15:17  PA 110/60 mmHg    ANNA BIANCHI', []),
        PdfPage(2, 100, 100,
            'Il Medico\n\nMARIO ROSSI\nIl CPSI triagista\n\nANNA BIANCHI\n'
            'Controllo il 20.01.2026.', []),
    ])
    result = normalize_document(None, '', parsing_result=parsed)
    assert 'MARIO ROSSI' not in result.text and 'ANNA BIANCHI' not in result.text
    assert 'Scala dolore NRS 02 Dolore lieve' in result.text
    assert 'PA 110/60 mmHg' in result.text and '19.01.2026 15:17' in result.text
    assert 'Terapia 5 mg. Malattia di Parkinson.' in result.text
    assert result.redaction_counts['staff_name'] == 4
    assert 'MARIO ROSSI' not in str(result.retention_audit)
    assert len(_mapped_source_pages(result.text, result.retention_audit)) == 2


def test_legacy_markdown_becomes_continuous_with_verified_page_map():
    from emr_analyzer.clinical.dossier_query import _mapped_source_pages
    stamp = '19.08.2021 08:06:12'
    text = (f'<!-- page:1 -->\n{stamp}\nTerapia con\n'
            f' <!-- page:2 -->\n{stamp}\nnivolumab. Visita il 19.08.2021.')
    result = normalize_document(None, text)
    assert result.text == 'Terapia con\nnivolumab. Visita il 19.08.2021.'
    assert _mapped_source_pages(result.text, result.retention_audit) == [
        (1, 'Terapia con'), (2, 'nivolumab. Visita il 19.08.2021.')]
    assert len(result.retention_audit['join_removals']) == 2


def test_legacy_markdown_preserves_prefix_and_distinct_timestamps():
    result = normalize_document(None,
        'Anamnesi: melanoma.\n<!-- page:1 -->\r\n19.08.2021 08:06:12\r\n'
        'Dose 5 mg.\r\n<!-- page:2 -->\r\n20.08.2021 09:07:13\r\nControllo.')
    assert 'Anamnesi: melanoma.' in result.text
    assert '19.08.2021 08:06:12' in result.text
    assert '20.08.2021 09:07:13' in result.text
    assert '<!-- page:' not in result.text


@pytest.mark.parametrize('numbers', [(1, 3), (1, 1), (2, 1)])
def test_invalid_legacy_page_numbers_do_not_create_false_citations(numbers):
    text = '\n'.join(f'<!-- page:{n} -->\nDose 5 mg.' for n in numbers)
    with pytest.raises(ClinicalTextIsolationError, match='Sequenza'):
        normalize_document(None, text)


def test_failed_page_never_returns_a_partial_document():
    isolator = Mock()
    isolator.isolate.side_effect = [ClinicalTextIsolationResult('Pagina valida', 'fake'), ValueError('Invalid output')]
    with pytest.raises(ValueError):
        normalize_document(isolator, '', parsing_result=source())


def test_cancel_stops_before_next_page():
    isolator = Mock()
    stopped = [False]
    def first(text, **kwargs):
        stopped[0] = True
        return ClinicalTextIsolationResult(text, 'fake')
    isolator.isolate.side_effect = first
    with pytest.raises(ClinicalTextIsolationError, match='interrotta'):
        normalize_document(isolator, '', parsing_result=source(), cancel_check=lambda: stopped[0])
    assert isolator.isolate.call_count == 1


def test_lab_deterministic_path_redacts_identity_preserving_values():
    result = normalize_document(None, 'Mario Rossi. Hb 12.3 g/dL. 01/03/2024.',
                                sensitive_identity={'name': 'Mario Rossi'})
    assert 'Mario Rossi' not in result.text
    assert '12.3 g/dL' in result.text
    assert '01/03/2024' in result.text
    assert result.model_name == 'deterministic'


def test_lab_preserves_repeated_specimen_context():
    text = 'Materiale: Urina\nValore 5 mg/dL\nMateriale: Siero\nValore 5 mg/dL'
    result = normalize_document(None, text)
    assert result.text == text


@pytest.mark.parametrize('fail', [False, True])
def test_gui_extraction_reads_original_once_and_writes_only_final_markdown(tmp_path, fail):
    tab = DocumentsTab()
    tab._current_patient_id = 'P001'
    tab._refresh_table = Mock()
    folder = tmp_path / 'extraction'
    tab._get_extraction_dir = lambda: folder
    doc = DocumentRecord(id='DOC_1', patient_id='P001', filename='original.pdf',
                         original_path=str(tmp_path / 'original.pdf'), file_hash='hash')
    parser = Mock()
    parsed = source()
    parser.convert.return_value = parsed
    parser.export_markdown.return_value = parsed.markdown
    parser.export_text.return_value = parsed.plain_text
    parser.export_tables.return_value = []
    parser.get_page_count.return_value = 2
    isolator = Mock()
    if fail:
        isolator.isolate.side_effect = ValueError('Output non valido')
    else:
        from emr_analyzer.extraction.clinical_text_filter import ClinicalTextFilter
        isolator = ClinicalTextFilter(Mock())
    repo = Mock()
    repo.get_by_id.return_value = doc
    tab._services = {'document_repo': repo, 'converter': parser,
                     'clinical_text_isolator': isolator,
                     'document_llm_client': SimpleNamespace(is_available=True, model='fake')}
    tab._sensitive_identity_for_document = Mock(return_value={})
    tab._verify_document_attribution = Mock()
    progress = Mock()
    progress.is_cancelled.return_value = False
    tab._process_next_document(['DOC_1'], 0, progress, parser)
    assert parser.convert.call_count == 1
    outputs = list(tmp_path.rglob('*.*'))
    if fail:
        assert not outputs
        assert doc.extraction_status == 'error'
    else:
        assert outputs == [folder / 'DOC_1.md']
        text = outputs[0].read_text()
        assert 'Data del referto' in text
        assert '<!-- page:' not in text
        assert 'Controllo il 01/03/2024.' in text
        assert doc.extraction_status == 'done'
    tab.close()


def test_parallel_pipeline_needs_no_persisted_parser_files(tmp_path):
    tab = DocumentsTab()
    tab._current_patient_id = 'P001'
    tab._refresh_table = Mock()
    tab._get_extraction_dir = lambda: tmp_path / 'extraction'
    docs = {f'DOC_{i}': DocumentRecord(id=f'DOC_{i}', patient_id='P001',
            filename=f'{i}.pdf', original_path=str(tmp_path / f'{i}.pdf'), file_hash=str(i))
            for i in range(3)}
    parser = Mock()
    parsed = source()
    parser.convert.return_value = parsed
    parser.export_markdown.return_value = parsed.markdown
    parser.export_text.return_value = parsed.plain_text
    parser.export_tables.return_value = []
    parser.get_page_count.return_value = 2
    isolator = Mock()
    isolator.isolate.side_effect = lambda text, **kw: ClinicalTextIsolationResult(text, 'fake', chunk_count=1)
    repo = Mock()
    repo.get_by_id.side_effect = docs.get
    tab._services = {'document_repo': repo, 'converter': parser,
                     'clinical_text_isolator': isolator,
                     'document_llm_client': SimpleNamespace(is_available=True, model='fake')}
    tab._sensitive_identity_for_document = Mock(return_value={})
    tab._verify_document_attribution = Mock()
    progress = Mock()
    progress.is_cancelled.return_value = False
    tab._process_documents_parallel(list(docs), 2, progress=progress)
    assert parser.convert.call_count == 3
    assert all(d.extraction_status == 'done' for d in docs.values())
    assert sorted(p.name for p in tmp_path.rglob('*.*')) == ['DOC_0.md', 'DOC_1.md', 'DOC_2.md']
    tab.close()


def test_large_queue_does_not_recurse():
    tab = DocumentsTab()
    tab._refresh_table = Mock()
    repo = Mock()
    repo.get_by_id.return_value = None
    tab._services = {'document_repo': repo}
    progress = Mock()
    progress.is_cancelled.return_value = False
    tab._process_next_document([f'DOC_{i}' for i in range(1500)], 0, progress, None)
    assert tab._batch_error_count == 1500
    tab.close()


def test_geometry_can_be_rebuilt_from_original_without_json(tmp_path):
    import fitz
    from emr_analyzer.clinical.atomic_evidence import AtomicEvidenceExtractor
    from emr_analyzer.clinical.registry_builder import _geometry_source
    pdf = tmp_path / 'original.pdf'
    document = fitz.open()
    document.new_page().insert_text((40, 40), 'Nessuna febbre. Dose 5 mg.')
    document.save(pdf)
    document.close()
    record = SimpleNamespace(id='DOC_1', patient_id='P001', original_path=str(pdf), filename=pdf.name)
    path = _geometry_source(record, tmp_path)
    geometry = AtomicEvidenceExtractor._load_geometry(path)
    assert geometry.page_count == 1
    assert 'Dose 5 mg' in geometry.plain_text
    assert list(tmp_path.iterdir()) == [pdf]


def test_docling_pages_keep_provenance_and_header_exam_date():
    pytest.importorskip('docling_core')
    document = SimpleNamespace(pages={1: object(), 2: object()})
    def export(**kwargs):
        assert {str(x.value) for x in kwargs['included_content_layers']} == {'body', 'furniture', 'notes'}
        return {1: 'Data esame: 01/03/2024\nNessuna febbre.', 2: 'Terapia: 5 mg al giorno.'}[kwargs['page_no']]
    document.export_to_markdown = export
    result = normalize_document(None, '', parsing_result=SimpleNamespace(document=document))
    assert '<!-- page:' not in result.text
    assert [p['page'] for p in result.retention_audit['output_pages']] == [1, 2]
    assert '01/03/2024' in result.text and '5 mg al giorno' in result.text

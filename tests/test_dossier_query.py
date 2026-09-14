from types import SimpleNamespace
from unittest.mock import Mock
import json
import re

import pytest

from emr_analyzer.clinical.dossier_query import DossierQueryService, QueryCancelled, render_report
from emr_analyzer.gui.dossier_query_dialog import DossierQueryWorker, DossierQueryDialog


class LocalModel:
    context_length = 4096
    max_output_tokens = 1024
    model = 'synthetic-test'

    def __init__(self):
        self.prompts = []

    def generate_structured(self, prompt, system, schema):
        self.prompts.append(prompt)
        fragment = prompt.split('REFERTO:\n', 1)[1]
        return {'findings': [{'statement': fragment.strip(), 'quote': fragment.strip()}]}

    def generate_text(self, prompt, system):
        return 'Sintesi: ' + ' '.join(dict.fromkeys(re.findall(r'\[SRC_[a-f0-9]+\]', prompt)))


def make_service(tmp_path, patients=('P001',), text=None):
    docs = []
    for patient in patients:
        directory = tmp_path / patient / 'extraction'
        directory.mkdir(parents=True)
        (directory / 'DOC_1.md').write_text(
            text or '<!-- page:1 -->\nReferto sintetico ' + patient, encoding='utf-8')
        docs.append(SimpleNamespace(id='DOC_1', patient_id=patient, page_count=1))
    repo = Mock()
    repo.list_by_patient.side_effect = lambda pid: [d for d in docs if d.patient_id == pid]
    model = LocalModel()
    return DossierQueryService(repo, model, tmp_path), model


def test_full_active_markdown_read_without_registry_and_quotes_verified(tmp_path):
    service, model = make_service(tmp_path, text='<!-- page:1 -->\n' + 'inizio ' * 1000 + 'REPERTO_FINALE')
    report = service.query('P001', 'qualsiasi domanda')
    assert report['coverage_complete']
    assert report['chunks_processed'] > 1
    assert any('REPERTO_FINALE' in f['quote'] for f in report['findings'])
    assert all(s['page'] == 1 for s in report['sources'].values())
    assert 'Copertura di elaborazione: completa' in render_report(report)


@pytest.mark.parametrize("suffix", ["_raw.md", "_source.txt", "_cleaned_source.md"])
def test_missing_active_markdown_never_falls_back_to_raw(tmp_path, suffix):
    service, model = make_service(tmp_path)
    path = tmp_path / 'P001/extraction/DOC_1.md'
    path.rename(path.with_name('DOC_1' + suffix))
    report = service.query('P001', 'domanda')
    assert report['missing_documents'] == ['DOC_1']
    assert not report['coverage_complete']
    assert not model.prompts


def test_raw_content_is_excluded_when_active_markdown_exists(tmp_path):
    service, model = make_service(tmp_path)
    directory = tmp_path / 'P001/extraction'
    for suffix in ('_raw.md', '_source.txt', '_cleaned_source.md'):
        (directory / ('DOC_1' + suffix)).write_text('IDENTITA_GREZZA_BOILERPLATE')
    report = service.query('P001', 'domanda')
    assert report['coverage_complete']
    assert all('IDENTITA_GREZZA_BOILERPLATE' not in p for p in model.prompts)
    assert all(s['parser_file'].endswith('/DOC_1.md') for s in report['sources'].values())
    assert report['source_policy'] == 'active_clinical_markdown_only'


def test_legacy_docling_active_markdown_is_used_before_extraction_raw(tmp_path):
    service, model = make_service(tmp_path)
    directory = tmp_path / 'P001/docling'
    directory.mkdir()
    path = tmp_path / 'P001/extraction/DOC_1.md'
    path.rename(directory / path.name)
    path.with_name('DOC_1_raw.md').write_text('IDENTITA_GREZZA_BOILERPLATE')
    report = service.query('P001', 'domanda')
    assert report['coverage_complete']
    assert all(s['parser_file'] == 'P001/docling/DOC_1.md' for s in report['sources'].values())


def test_hallucinated_quote_marks_incomplete_and_is_excluded(tmp_path):
    service, model = make_service(tmp_path)
    model.generate_structured = lambda *args: {'findings': [{'statement': 'inventato', 'quote': 'inesistente'}]}
    report = service.query('P001', 'domanda')
    assert not report['coverage_complete']
    assert not report['findings']
    assert report['errors']


def test_patient_data_never_shared_in_individual_calls(tmp_path):
    service, model = make_service(tmp_path, patients=('P001', 'P002'))
    first = service.query('P001', 'stesso prompt')
    assert all('P002' not in p for p in model.prompts)
    model.prompts.clear()
    second = service.query('P002', 'stesso prompt')
    assert all('P001' not in p for p in model.prompts)
    assert set(first['sources']).isdisjoint(second['sources'])


def test_cross_patient_repository_error_rejected(tmp_path):
    service, model = make_service(tmp_path)
    service.documents.list_by_patient.side_effect = None
    service.documents.list_by_patient.return_value = [SimpleNamespace(id='DOC_1', patient_id='P002')]
    with pytest.raises(ValueError, match='altro paziente'):
        service.query('P001', 'domanda')
    assert not model.prompts


def test_cancel_prevents_llm_calls(tmp_path):
    service, model = make_service(tmp_path)
    with pytest.raises(QueryCancelled):
        service.query('P001', 'domanda', cancel_check=lambda: True)
    assert not model.prompts


def test_page_missing_from_parser_is_not_complete(tmp_path):
    service, model = make_service(tmp_path)
    service.documents.list_by_patient('P001')[0].page_count = 2
    assert not service.query('P001', 'domanda')['coverage_complete']


def test_cohort_saves_individual_reports_and_manifest(tmp_path):
    service, model = make_service(tmp_path, patients=('P001', 'P002'))
    output = tmp_path / 'reports'
    worker = DossierQueryWorker(service, ['P001', 'P002'], 'domanda', output)
    worker.run()
    manifest = json.loads((output / 'manifest.json').read_text())
    assert manifest['completed_patient_ids'] == ['P001', 'P002']
    assert manifest['state'] == 'finished'
    for pid in manifest['completed_patient_ids']:
        assert json.loads((output / f'{pid}.json').read_text())['patient_id'] == pid
    assert 'Sintesi di coorte' in (output / 'cohort.md').read_text()


def test_dialog_selection_starts_with_current_patient(tmp_path):
    services = {'patient_repo': Mock()}
    services['patient_repo'].list_all.return_value = [SimpleNamespace(id='P001'), SimpleNamespace(id='P002')]
    dialog = DossierQueryDialog(services, tmp_path, patient_id='P002')
    from PyQt5.QtCore import Qt
    assert dialog.patients.item(0).checkState() == Qt.Unchecked
    assert dialog.patients.item(1).checkState() == Qt.Checked
    dialog.close()


def test_report_date_reaches_every_query_fragment_and_sources(tmp_path):
    service, model = make_service(tmp_path, text="<!-- page:1 -->\n" + "Reperto clinico. " * 1000)
    service.documents.list_by_patient("P001")[0].document_date = "2024-02-29"
    result = service.query("P001", "Domanda")
    assert len(model.prompts) > 1
    assert all("DATA DEL REFERTO (metadato document_date): 2024-02-29" in p for p in model.prompts)
    assert all(s["document_date"] == "2024-02-29" for s in result["sources"].values())


def test_generated_metadata_does_not_add_an_extra_llm_call(tmp_path):
    from emr_analyzer.clinical.report_metadata import with_report_date
    text = with_report_date('<!-- page:1 -->\nNessuna febbre.', '2024-02-29')
    service, model = make_service(tmp_path, text=text)
    service.documents.list_by_patient('P001')[0].document_date = '2024-02-29'
    report = service.query('P001', 'Febbre?')
    assert report['coverage_complete']
    assert report['chunks_processed'] == 1
    assert len(model.prompts) == 1
    assert 'DATA DEL REFERTO (metadato document_date): 2024-02-29' in model.prompts[0]

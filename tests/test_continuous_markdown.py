import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from emr_analyzer.extraction.continuous_markdown import merge_pages
from emr_analyzer.clinical.dossier_query import DossierQueryService, _mapped_source_pages
from emr_analyzer.clinical.report_metadata import with_report_date
from emr_analyzer.pipeline.staff_identity import redact_staff_names


def test_merge_removes_only_repeated_page_timestamps():
    stamp='28.02.2024 15:43:13'
    text,audit=merge_pages([(1,stamp+'\nPrima parte della storia'),(2,stamp+'\nche prosegue. Esame il 28.02.2024.'),(3,stamp+'\nNessuna febbre.')])
    assert stamp not in text
    assert 'Esame il 28.02.2024.' in text
    assert '<!--' not in text
    assert text == 'Prima parte della storia\nche prosegue. Esame il 28.02.2024.\nNessuna febbre.'
    assert _mapped_source_pages(text,audit) == [(1,'Prima parte della storia'),(2,'che prosegue. Esame il 28.02.2024.'),(3,'Nessuna febbre.')]
    assert len(audit['join_removals']) == 3
    assert _mapped_source_pages(text+'modificato',audit) is None


def test_nonrepeated_timestamp_and_narrative_dates_stay():
    text,_=merge_pages([(1,'28.02.2024 15:43:13\nVisita.'),(2,'01.03.2024 14:00:00\nControllo.')])
    assert '28.02.2024 15:43:13' in text and '01.03.2024 14:00:00' in text


def test_empty_administrative_page_preserves_source_coverage():
    text,audit=merge_pages([(1,'Nessuna febbre.'),(2,'[NESSUN CONTENUTO CLINICO]'),(3,'Dose 5 mg.')])
    assert '[NESSUN' not in text
    assert _mapped_source_pages(text,audit) == [(1,'Nessuna febbre.'),(2,''),(3,'Dose 5 mg.')]


def test_query_reads_database_page_map_and_skips_admin_only_page(tmp_path):
    text,audit=merge_pages([(1,'Nessuna febbre.'),(2,'[NESSUN CONTENUTO CLINICO]'),(3,'Dose 5 mg.')])
    directory=tmp_path/'P001/extraction';directory.mkdir(parents=True)
    path=directory/'DOC_1.md';path.write_text(with_report_date(text,'2024-02-28'))
    doc=SimpleNamespace(id='DOC_1',patient_id='P001',page_count=3,metadata_json=json.dumps({'clinical_text':{'retention_audit':audit}}))
    repo=Mock();repo.list_by_patient.return_value=[doc]
    llm=Mock(context_length=4096,max_output_tokens=1024,model='synthetic')
    llm.generate_structured.return_value={'findings':[]}
    report=DossierQueryService(repo,llm,tmp_path).query('P001','Febbre?')
    assert report['coverage_complete']
    assert {s['page'] for s in report['sources'].values()} == {1,3}
    assert llm.generate_structured.call_count == 2
    path.write_text(with_report_date(text+' Alterato.','2024-02-28'))
    report=DossierQueryService(repo,llm,tmp_path).query('P001','Febbre?')
    assert not report['coverage_complete']
    assert any('Mappa' in e['error'] for e in report['errors'])


@pytest.mark.parametrize('prefix',['Dott.ssa','Dott.','Dr.','Prof.','Medico:','TSRM:'])
def test_staff_names_redacted_preserving_clinical_statement(prefix):
    text=f'Valutazione con {prefix} Mario Rossi: non evidenza di metastasi.'
    out,count=redact_staff_names(text)
    assert out=='Valutazione con [MEDICO]: non evidenza di metastasi.'
    assert count==1


def test_staff_redaction_stops_before_clinical_column_or_new_sentence():
    for suffix in ('     CLAVULANICO','     ENOXAPARINA','. Terapia invariata.'):
        text='Dott. Mario Rossi'+suffix
        assert redact_staff_names(text)[0]=='[MEDICO]'+suffix
    assert redact_staff_names('Malattia di Parkinson.')[0]=='Malattia di Parkinson.'


def test_signature_identifies_full_name_without_redacting_unrelated_surname():
    source = ('ANNA DE LUCA - 19.01.2026 15:17:00\n'
              'Visita con De Luca.\nIl CPSI triagista\n\nANNA DE LUCA')
    text, count = redact_staff_names(source)
    assert count == 2
    assert 'ANNA DE LUCA' not in text
    assert 'Visita con De Luca.' in text
    assert '19.01.2026 15:17:00' in text


@pytest.mark.parametrize('source', [
    'Il Medico\nSODIO CLORURO\nDose 5 mg.',
    'Il Medico\nTERAPIA DOMICILIARE',
    'Il Medico\n\n<!-- page:2 -->\nMARIO ROSSI',
    'VALORI NORMALI\nACIDO ACETILSALICILICO',
])
def test_signature_detection_does_not_guess_names_without_valid_context(source):
    assert redact_staff_names(source) == (source, 0)

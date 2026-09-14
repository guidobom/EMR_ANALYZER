import json
from unittest.mock import Mock

from emr_analyzer.extraction.clinical_text_filter import ClinicalTextFilter
from emr_analyzer.extraction.administrative_templates import tidy_layout, layout_cells


def test_administrative_lines_removed_clinical_text_copied_verbatim():
    clinical = 'Anamnesi: nega febbre.\nTerapia: prednisone 5 mg per via orale.\n'
    source = 'DATI ANAGRAFICI\nNome e cognome: Mario Rossi\nOspedale Sant Anna\n' + clinical + 'Firmato digitalmente il 01/03/2024\nPag. 1/1\n'
    llm = Mock()
    result = ClinicalTextFilter(llm).isolate(source, sensitive_identity={'name': 'Mario Rossi'})
    assert result.text == clinical
    assert not llm.mock_calls
    assert 'Mario Rossi' not in json.dumps(result.retention_audit)


def test_mixed_line_never_deleted_silently():
    source = 'Firmato digitalmente con nota: terapia sospesa per febbre.\n'
    result = ClinicalTextFilter().isolate(source)
    assert result.text == source
    assert result.retention_audit['review_required']
    assert result.warnings


def test_repeated_clinical_lines_and_numeric_rows_are_not_deduplicated():
    source = 'Materiale: Urina\nValore 5 mg/dL\nMateriale: Siero\nValore 5 mg/dL\n'
    result = ClinicalTextFilter().isolate(source)
    assert result.text == source


def test_decisions_partition_source_and_explain_every_deletion():
    source = 'DATI ANAGRAFICI\n\nDiagnosi: melanoma.\nGDPR e privacy\n'
    result = ClinicalTextFilter().isolate(source)
    audit = result.retention_audit
    decisions = audit['decisions']
    assert decisions[0]['start'] == 0
    assert decisions[-1]['end'] == len(source)
    assert all(a['end'] == b['start'] for a, b in zip(decisions, decisions[1:]))
    assert tidy_layout(''.join(source[d['start']:d['end']] for d in decisions if d['action'] == 'keep')) == result.text
    assert audit['retained_characters'] + audit['removed_characters'] == len(source)


def test_administrative_only_document_has_explicit_empty_marker():
    result = ClinicalTextFilter().isolate('DATI ANAGRAFICI\nPag. 1/1\n')
    assert result.text == '[NESSUN CONTENUTO CLINICO]'


def test_mixed_administrative_and_clinical_fields_separated_by_semicolon():
    source = 'Indirizzo: Via Roma 12; nega febbre.\nTelefono: 3331234567; terapia 5 mg.\n'
    result = ClinicalTextFilter().isolate(source)
    assert result.text == 'nega febbre.\nterapia 5 mg.\n'


def test_hospital_templates_and_padding_removed_without_losing_clinical_columns():
    source = ('NOME E COGNOME     SESSO\n[PAZIENTE] F\nArcispedale S.Anna\n'
              'Dott. Mario Rossi     Si consiglia controllo tra 3 mesi.\n'
              '   \n    Materiale: Siero\n    Creatinina   0.80   mg/dL\n'
              'Risultati validati da:\nDott. Mario Rossi\n'
              'Questo documento è firmato digitalmente.\n')
    result = ClinicalTextFilter().isolate(source)
    assert 'Si consiglia controllo tra 3 mesi.' in result.text
    assert 'Creatinina   0.80   mg/dL' in result.text
    assert 'Materiale: Siero' in result.text
    for removed in ('COGNOME', 'PAZIENTE', 'Arcispedale', 'Dott.', 'validati', 'digitalmente'):
        assert removed not in result.text
    assert ''.join(layout_cells(source)) == source


def test_clinical_staff_mentions_and_specimen_references_remain():
    source = ('Dott. Mario Rossi consiglia biopsia.\n'
              'Materiale inviato: cute, cfr. 23F12345.\n'
              'Diagnosi Istologica\nNegativa per metastasi.\n')
    assert ClinicalTextFilter().isolate(source).text == source.replace('Dott. Mario Rossi', '[MEDICO]')


def test_contact_sidebar_does_not_delete_medication_table_rows():
    source = ('Tel. 0532 123456               [X] ENOXAPARINA   4000UI 0,4ML  1 fiala sottocute ore 20, per 10 giorni\n'
              'Tel. 0532 123457 [ ] AMOXICILLINA / ACIDO CLAVULANICO 875MG+125MG 1 compressa tre volte al dì\n'
              'CODICE FISCALE: [IDENTIFICATIVO RIMOSSO] ONCOLOGICA DI CONTROLLO\n')
    result = ClinicalTextFilter().isolate(source)
    assert 'ENOXAPARINA   4000UI 0,4ML  1 fiala sottocute ore 20, per 10 giorni' in result.text
    assert 'AMOXICILLINA / ACIDO CLAVULANICO 875MG+125MG 1 compressa tre volte al dì' in result.text
    assert 'ONCOLOGICA DI CONTROLLO' in result.text
    assert 'TELEFONO' not in result.text
    assert 'CODICE FISCALE' not in result.text


def test_patient_row_is_removed_as_a_whole_and_audit_stays_contiguous():
    source = '[PAZIENTE]                        F\nMateriale: Siero\n'
    result = ClinicalTextFilter().isolate(source)
    assert result.text == 'Materiale: Siero\n'
    assert ''.join(layout_cells(source)) == source


def test_signature_next_to_drug_name_preserves_drug_without_unit():
    source = 'Dott. Mario Rossi       CLAVULANICO\n'
    assert ClinicalTextFilter().isolate(source).text == 'CLAVULANICO\n'


def test_identifier_prefix_in_second_column_preserves_clinical_tail():
    source = 'LUOGO E DATA DI NASCITA     CODICE FISCALE: [IDENTIFICATIVO RIMOSSO]: Terapia ben tollerata.\n'
    result = ClinicalTextFilter().isolate(source)
    assert 'Terapia ben tollerata.' in result.text
    assert 'CODICE FISCALE' not in result.text
    assert ''.join(layout_cells(source)) == source

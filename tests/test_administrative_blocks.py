import pytest
from emr_analyzer.extraction.clinical_text_filter import ClinicalTextFilter
from emr_analyzer.extraction.administrative_templates import tidy_layout


@pytest.mark.parametrize('field', [
    'Acc Number: FSA1234567', 'Acc. Number: FSA1234567',
    'Accession Number: ABC-12345', 'Accession No.: ABC/123',
    'Provenienza: ESTERNO',
    'Provenienza: D.H. CHIR.PLASTICA PREOP (4          Acc Number: FSA1234567',
    'Provenienza: 1E2 DAY SERVICE ONCOLOGIA CLINICA 596 Acc. Number: FSA1234567',
])
def test_radiology_identifiers_removed_without_losing_exam(field):
    clinical = 'Data-Ora Esame: 13/11/2025 - 09:19\nReferto: non lesioni. Dose 5 mg.'
    result = ClinicalTextFilter().isolate(field + '\n' + clinical)
    assert result.text == clinical


@pytest.mark.parametrize('field', ['Provenienza: ESTERNO', 'Acc Number: FSA1234567'])
def test_radiology_form_and_clinical_field_on_same_line(field):
    source = field + ' Quesito Clinico: sospetta miocardite.\n'
    result = ClinicalTextFilter().isolate(source)
    assert result.text.strip() == 'Quesito Clinico: sospetta miocardite.'
    decisions = result.retention_audit['decisions']
    assert decisions[-1]['end'] == len(source)
    assert all(a['end'] == b['start'] for a,b in zip(decisions, decisions[1:]))


def test_radiology_signature_columns_removed_and_clinical_prose_kept():
    source = ('Referto: non lesioni.\n'
        'T.S.R.M       Medico Radiologo Specializzando DR. IL MEDICO RADIOLOGO\n'
        'Mario Rossi       Anna Bianchi\n'
        'Conclusioni: nessun versamento.\n')
    result = ClinicalTextFilter().isolate(source)
    assert [line for line in result.text.splitlines() if line.strip()] == [
        'Referto: non lesioni.', 'Conclusioni: nessun versamento.']
    assert 'Mario Rossi' not in str(result.retention_audit)


def test_radiology_staff_role_in_clinical_sentence_is_not_a_signature():
    text = 'Il medico radiologo consiglia controllo.\nLesione di incerta provenienza.'
    assert ClinicalTextFilter().isolate(text).text == text


@pytest.mark.parametrize('header', [
    'SERVIZIO DI CARDIOLOGIA', 'SERVIZIO DI MEDICINA NUCLEARE',
    'AMBULATORIO DI ENDOCRINOLOGIA GENERALE', 'AMBULATORIO ORTOPEDICO DIVISIONALE',
    'DH CHIR. PLAS PREOP 413P', 'DAY HOSPITAL ONCOLOGIA',
    '28.03.2022 12:15:00                 DEG. ONCOLOGIA CLINICA (1B3)',
    '06.07.2021 10:00:00        [PAZIENTE]/ORA ACCETTAZIONE      PROVENIENZA',
    '06.07.2021 10:00:00        DATA/ORA ACCETTAZIONE      PROVENIENZA',
    'S.C. di Radiologia Diagnostica ed Interventistica Interaziendale - Sede di Cona',
    'Ambulatorio n°18', 'U.O.A.R.O.', 'U.O.A.R.U.',
    'DH Oncologico [TELEFONO RIMOSSO]',
])
def test_other_specialties_administrative_headers(header):
    clinical = 'Nessun dolore. Dose 5 mg.\nControllo il 27.07.2024.'
    result = ClinicalTextFilter().isolate(header + '\n' + clinical)
    assert result.text == clinical


@pytest.mark.parametrize('clinical', [
    '17.01.2025 22:36:12          Scala NUMERICA: 00',
    '17.01.2025 22:36:12 VISITA DI PRONTO SOCCORSO',
    '17.01.2025 22:36:12 Scala dolore NRS: 02 Dolore lieve',
    '17.01.2025 22:36:12 Dispnea associata a tosse stizzosa.',
    '27.07.2024 08:41:04 DSA RADIOTERAPIA dose totale 30 Gy',
    'DAY HOSPITAL ONCOLOGIA: terapia sospesa.',
    'SERVIZIO DI CARDIOLOGIA: ECG normale.',
    'DSA: stenosi del 50%.',
    '27.07.2024 08:41:04\nGlucosio 100 mg/dL',
])
def test_clinical_timeline_and_mixed_department_notes_preserved(clinical):
    assert ClinicalTextFilter().isolate(clinical).text == clinical


@pytest.mark.parametrize('gap', [' ', '                  ', '\t'])
def test_radiotherapy_header_removes_timestamp_with_admission_label(gap):
    clinical = 'Radioterapia: 30 Gy in 10 frazioni.\nControllo il 27.07.2024.\n'
    header = ('SERVIZIO DI RADIOTERAPIA ONCOLOGICA\n'
              f'27.07.2024 08:41:04{gap}DSA RADIOTERAPIA\n')
    result = ClinicalTextFilter().isolate(header + clinical)
    assert result.text == clinical
    assert sum(d['action'] == 'remove' for d in result.retention_audit['decisions']) == 2


@pytest.mark.parametrize('text', [
    '27.07.2024 08:41:04\nRadioterapia: 30 Gy in 10 frazioni.',
    'Invio al servizio di radioterapia oncologica per valutazione.',
    '27.07.2024 08:41:04 DSA RADIOTERAPIA: trattamento sospeso.',
    'DSA: non evidenza di stenosi.',
    'SERVIZIO DI RADIOTERAPIA ONCOLOGICA: dose totale 30 Gy.',
])
def test_radiotherapy_clinical_mentions_and_independent_dates_stay(text):
    assert ClinicalTextFilter().isolate(text).text == text


def test_radiotherapy_header_alongside_clinical_column_preserves_clinical_text():
    result = ClinicalTextFilter().isolate(
        'SERVIZIO DI RADIOTERAPIA ONCOLOGICA     Dose totale 30 Gy.\n')
    assert result.text.strip() == 'Dose totale 30 Gy.'


@pytest.mark.parametrize('hospital', ['ASST Ospedali del Nord', 'IRCCS Istituto Esempio', 'Azienda Ospedaliera Universitaria Citta Nuova', 'Casa di cura Villa Esempio'])
def test_other_institutions_do_not_require_local_names(hospital):
    text = (hospital + '\nDirettore: Dott. Mario Rossi\n'
            'Data esame: 12/05/2025\nAnamnesi e quesito clinico\n'
            'Non febbre. Allergia alla penicillina.\n'
            'Materiale: Siero\nPotassio 4.2 mmol/L (3.5 - 5.1)\n'
            'Conclusioni\nNessuna lesione. Controllo il 01/06/2025.\n'
            'IL MEDICO RADIOLOGO\nROSSI MARIO\n'
            'Documento provvisto di firma digitale ai sensi della normativa vigente.\n')
    result = ClinicalTextFilter().isolate(text)
    assert hospital not in result.text
    assert 'ROSSI MARIO' not in result.text
    assert 'Direttore' not in result.text
    assert '12/05/2025' in result.text
    assert 'Allergia alla penicillina.' in result.text
    assert 'Potassio 4.2 mmol/L (3.5 - 5.1)' in result.text
    assert 'Controllo il 01/06/2025.' in result.text


def test_wrapped_notices_are_removed_but_clinical_disclaimers_and_instructions_stay():
    clinical = ('Limitatamente al potere risolutivo della metodica non si evidenziano lesioni.\n'
                'Al domicilio si consiglia riposo per 10 giorni.\n')
    notice = ('"Copia del referto informatico predisposto e conservato presso Azienda Esempio in\n'
              "conformità alle regole tecniche di cui all'art. 71 del D. Lgs 82/2005\"\n")
    result = ClinicalTextFilter().isolate(clinical + notice)
    assert result.text == clinical
    source = clinical + notice
    decisions = result.retention_audit['decisions']
    assert decisions[0]['start'] == 0 and decisions[-1]['end'] == len(source)
    assert all(a['end'] == b['start'] for a,b in zip(decisions, decisions[1:]))
    assert tidy_layout(''.join(source[d['start']:d['end']] for d in decisions if d['action']=='keep')) == result.text


def test_formulary_notice_does_not_remove_actual_prescription():
    prescription = 'ENOXAPARINA 4000 UI sottocute ogni 24 ore per 10 giorni.\n'
    notice = ("Si autorizza l'erogazione del medicinale/equivalente presente nel Prontuario dell'Azienda Ospedaliera a parità di principio\n"
              'attivo, dosaggio e forma farmaceutica.\n')
    result = ClinicalTextFilter().isolate(prescription + notice)
    assert result.text == prescription


def test_notice_cannot_swallow_clinical_section_or_cross_pages():
    text = ('"Copia del referto informatico predisposto e conservato presso Azienda Esempio\n'
            'Conclusioni\nNon evidenza di metastasi.\n82/2005"\n')
    assert 'Non evidenza di metastasi.' in ClinicalTextFilter().isolate(text).text


def test_staff_reference_in_narrative_remains():
    text = 'Il Dott. Mario Rossi consiglia rivalutazione per febbre.\n'
    assert ClinicalTextFilter().isolate(text).text == text.replace('Dott. Mario Rossi', '[MEDICO]')


def test_staff_heading_with_inline_clinical_finding_is_not_deleted():
    text = 'IL MEDICO RADIOLOGO Non si osserva versamento.\n'
    assert ClinicalTextFilter().isolate(text).text == text


def test_surgical_staff_roster_and_repeated_clinic_footer_removed():
    clinical = ('DIAGNOSI E TIPO INTERVENTO: exeresi bioptica e medicazione.\n'
                'DATA E ORA ESECUZIONE: odierna, 18:30\n'
                '-Antibiotico: Augmentin 1g, 1 cpr ogni 12 ore, per 6 giorni.\n')
    roster = 'NOME OPERATORI: [MEDICO], [MEDICO], Infermiere di sala: Bianchi, Verdi\n'
    footer = 'AMBULATORIO CHIRURGIA PLASTICA N°3, IN AREA 1\n\nMEDICI: [MEDICO]. ROSSI-[MEDICO]\n'
    result = ClinicalTextFilter().isolate(clinical + roster + footer + footer)
    assert result.text == clinical
    assert any(d['reason'] == 'staff_roster' for d in result.retention_audit['decisions'])


def test_roster_does_not_consume_clinical_column_or_semicolon_note():
    source = ('NOME OPERATORI: [MEDICO], [MEDICO]; Nega febbre.\n'
              'MEDICI: [MEDICO]      ENOXAPARINA 4000UI\n'
              'Controllo presso AMBULATORIO CHIRURGIA PLASTICA N°3, IN AREA 1\n')
    result = ClinicalTextFilter().isolate(source)
    assert 'Nega febbre.' in result.text
    assert 'ENOXAPARINA 4000UI' in result.text
    assert 'Controllo presso AMBULATORIO CHIRURGIA PLASTICA N°3, IN AREA 1' in result.text
    assert 'NOME OPERATORI' not in result.text


def test_unredacted_nursing_roster_is_removed_without_hospital_specific_names():
    result = ClinicalTextFilter().isolate('Infermiere di sala: Bianchi, Verdi\nMedicazione in ordine.\n')
    assert result.text == 'Medicazione in ordine.\n'

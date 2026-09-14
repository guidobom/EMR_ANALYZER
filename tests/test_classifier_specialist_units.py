"""Specialist-unit letterhead regression tests for the document classifier.

Fixtures reproduce the MELANOMA project layout: dermatology, surgery,
cardiology, radiotherapy and emergency units whose documents were typed
``visita_oncologica`` because the patient's history is full of oncology
terms or the trust's oncology department name appears in the letterhead
line ("DIP. ONCO MEDICO SPECIALISTICO UO DERMATOLOGIA").  The texts use
the de-identified placeholders of the real extracts.
"""

import unittest

from emr_analyzer.models.document import DocumentType
from emr_analyzer.pipeline.classifier import DocumentClassifier


def _header(**overrides) -> dict:
    base = {
        "department": None,
        "provenance": None,
        "services": [],
        "specialty": None,
        "document_type_hint": None,
        "confidence": 0.0,
    }
    base.update(overrides)
    return base


class SpecialistUnitClassificationTest(unittest.TestCase):
    def setUp(self):
        self.classifier = DocumentClassifier()

    def test_dermatology_ambulatory_stays_specialistica(self):
        """A dermatoscopy visit is not an oncological visit.

        Reproduces DOC_000347: the letterhead carries the oncology trust
        department, so the stored specialty was oncologia — the issuing
        unit line must win.
        """
        text = """Arcispedale S.Anna
AMB. VIDEODERMATOSCOPIA - CONA DIP. ONCO MEDICO SPECIALISTICO
UO DERMATOLOGIA

REFERTO
Paziente con melanoma nodulare del dorso in follow-up.
Controllo dei nevi: nessuna lesione sospetta di nuova insorgenza.
Si consiglia rivalutazione tra 6 mesi.
"""
        result = self.classifier.classify(
            text,
            "",
            header_metadata=_header(
                department="AMB. VIDEODERMATOSCOPIA - DIP. ONCO MEDICO "
                            "SPECIALISTICO UO DERMATOLOGIA",
                specialty="oncologia",
            ),
        )
        self.assertEqual(result, DocumentType.VISITA_SPECIALISTICA.value)

    def test_cardiologia_echo_stays_specialistica(self):
        """Reproduces the Eco(Color)Dopplergrafia Cardiaca reports."""
        text = """U.O. di Cardiologia, Azienda Ospedaliero Universitaria di Ferrara
Eco(Color)Dopplergrafia Cardiaca N°

Data Esame: 11/04/2023 Paziente: [PAZIENTE]
Provenienza: ALTRI REPARTI

Paziente con melanoma metastatico in terapia con nivolumab.
Ventricolo sinistro di normali dimensioni, funzione sistolica conservata.
Non versamento pericardico.
"""
        result = self.classifier.classify(
            text,
            "",
            header_metadata=_header(
                department="U.O. di Cardiologia",
                specialty="cardiologia",
            ),
        )
        self.assertEqual(result, DocumentType.VISITA_SPECIALISTICA.value)

    def test_surgical_ambulatory_stays_specialistica(self):
        """AMB. CHIRURGICO SENOLOGICO wound care, not oncological visit."""
        text = """Arcispedale S.Anna
CH-15 (37) AMB. CHIRURGICO SENOLOGICO - CHIRURGIA 2 -
DIPARTIMENTO CHIRURGICO UO CHIRURGIA 2

PRESTAZIONI EROGATE
MEDICAZIONE

REFERTO
Lesione secondaria sovraclaveare destra da melanoma.
Medicazione eseguita, ferita in via di guarigione.
"""
        result = self.classifier.classify(
            text,
            "",
            header_metadata=_header(
                department="AMB. CHIRURGICO SENOLOGICO - CHIRURGIA 2",
                specialty="chirurgia",
            ),
        )
        self.assertEqual(result, DocumentType.VISITA_SPECIALISTICA.value)

    def test_radiotherapy_session_note_is_radioterapia(self):
        """UO RADIOTERAPIA ONCOLOGICA session notes (DOC_000012 pattern)."""
        text = """Arcispedale S.Anna
UO RADIOTERAPIA ONCOLOGICA RX-19 (609) AMB. RADIOTERAPICO
DSA RADIOTERAPIA

Il Paziente è stato sottoposto oggi a seduta di radioterapia antalgica
su D2-D3 dove, mediante fotoni X 6 MV e tecnica VMAT, è stata erogata
la dose di 800 cGy. Buona la tolleranza immediata.

Paziente con melanoma nodulare stadio IV pluritrattato.
"""
        result = self.classifier.classify(
            text,
            "",
            header_metadata=_header(
                department="UO RADIOTERAPIA ONCOLOGICA AMB. RADIOTERAPICO",
                specialty="oncologia",
                document_type_hint=DocumentType.VISITA_ONCOLOGICA.value,
            ),
        )
        self.assertEqual(result, DocumentType.RADIOTERAPIA.value)

    def test_operating_block_sheet_is_verbale_operatorio(self):
        """Blocco Operatorio sheets (DOC_000141 pattern)."""
        text = """AZIENDA OSPEDALIERA UNIVERSITARIA DI FERRARA
ARCISPEDALE S. ANNA N°. Progressivo 425
Blocco Operatorio B0 - BLOCCO 9 Sala Operatoria 4 - SALA 4
Specialità Chirurgica

Diagnosi Operatoria: melanoma avanzato operato
Intervento: asportazione lesione, linfonodo sentinella.
"""
        result = self.classifier.classify(
            text,
            "",
            header_metadata=_header(),
        )
        self.assertEqual(result, DocumentType.VERBALE_OPERATORIO.value)

    def test_emergency_sheet_is_pronto_soccorso(self):
        """PS acceptance sheets (DIP. EMERGENZE pattern)."""
        text = """Arcispedale S.Anna
DIP. EMERGENZE UO MED. D'URGENZA-EMERGENZA
00000471 - PRONTO SOCCORSO GENERALE

DATI ACCETTAZIONE
TRIAGE: CODICE GIALLO

Paziente con melanoma metastatico, dolore addominale.
"""
        result = self.classifier.classify(
            text,
            "",
            header_metadata=_header(),
        )
        self.assertEqual(result, DocumentType.PRONTO_SOCCORSO.value)

    def test_oncology_visit_scheduling_radiotherapy_stays_oncologica(self):
        """A bare "si programma radioterapia" has no unit prefix."""
        text = """AMB. ONCOLOGIA TERAPIE 2 - UO ONCOLOGIA CLINICA

Paziente con melanoma metastatico, dolore al rachide dorsale.
Si programma radioterapia antalgica su D11.
Prosegue immunoterapia con nivolumab.
"""
        result = self.classifier.classify(
            text,
            "",
            header_metadata=_header(
                department="AMB. ONCOLOGIA TERAPIE 2 - UO ONCOLOGIA CLINICA",
                specialty="oncologia",
                document_type_hint=DocumentType.VISITA_ONCOLOGICA.value,
            ),
        )
        self.assertEqual(result, DocumentType.VISITA_ONCOLOGICA.value)

    def test_discharge_letter_from_surgery_ward_stays_discharge(self):
        """The guard keeps discharge letters with their own type even when
        the decorso mentions the operating room."""
        text = """LETTERA DI DIMISSIONE
Arcispedale S. Anna UO CHIRURGIA 1 CHIRURGIA TORACICA

Motivo del ricovero: intervento di asportazione di localizzazione
polmonare in paziente con melanoma.

Decorso clinico: il paziente è stato trasferito in sala operatoria,
intervento eseguito senza complicanze.

Paziente con melanoma nodulare stadio IV in terapia con nivolumab.
"""
        result = self.classifier.classify(
            text,
            "",
            header_metadata=_header(
                department="UO CHIRURGIA 1 CHIRURGIA TORACICA",
                specialty="chirurgia",
            ),
        )
        self.assertEqual(result, DocumentType.LETTERA_DIMISSIONE.value)

    def test_oncology_visit_quoting_histology_stays_oncologica(self):
        """A quoted histology report in the anamnesis does not block the
        letterhead rule for oncology visits (guard is letterhead-only)."""
        text = """AMB. ONCOLOGIA TERAPIE 2 - UO ONCOLOGIA CLINICA

Visita oncologica di controllo.
Esame istologico del 12/3/2025: melanoma nodulare, Breslow 2.1 mm.
Il paziente prosegue terapia con nivolumab, buona tolleranza.
"""
        result = self.classifier.classify(
            text,
            "",
            header_metadata=_header(
                department="AMB. ONCOLOGIA TERAPIE 2 - UO ONCOLOGIA CLINICA",
                specialty="oncologia",
            ),
        )
        self.assertEqual(result, DocumentType.VISITA_ONCOLOGICA.value)

    def test_interventional_radiology_with_fluoroscopy_is_radiologia(self):
        """Fluoroscopy assistance in the OR on a radiology letterhead is
        still radiology (DOC_000123 pattern)."""
        text = """Dipartimento Attività Integrata di Radiologia
U.O. Radiologia Ospedaliera

Paziente: [PAZIENTE]
Provenienza: 1E2 DAY SERVICE ONCOLOGIA C

Quesito Clinico: CONTROLLO PORT
Prestazioni eseguite: ATTIVITA' DI ASSISTENZA FLUOROSCOPICA IN SALA
OPERATORIA E ARCHIVIAZIONE IMMAGINI

Paziente con melanoma in trattamento immunoterapico.
"""
        result = self.classifier.classify(
            text,
            "",
            header_metadata=_header(
                department="Dipartimento Attività Integrata di Radiologia",
                provenance="1E2 DAY SERVICE ONCOLOGIA C",
                specialty="radiologia",
            ),
        )
        self.assertEqual(result, DocumentType.RADIOLOGIA.value)


if __name__ == "__main__":
    unittest.main(verbosity=2)

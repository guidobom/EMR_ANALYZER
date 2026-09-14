"""Imaging-vs-oncology regression tests for the document classifier.

Fixtures reproduce the MELANOMA project layout: radiology reports whose
letterhead says "Dipartimento ... di Radiologia" but whose requesting
provenance is "DAY SERVICE ONCOLOGIA".  Before these rules, the provenance
alone scored 10 points for ``visita_oncologica`` and every imaging report
of an oncology patient was misclassified.  The texts use the de-identified
placeholders of the real extracts (``[PAZIENTE]``, ``[TELEFONO RIMOSSO]``).
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


class ImagingClassificationTest(unittest.TestCase):
    def setUp(self):
        self.classifier = DocumentClassifier()

    def test_tc_encefalo_from_radiology_department_is_radiologia(self):
        """The radiology letterhead wins over the oncology provenance.

        Reproduces DOC_000154: the body carries no oncology term at all —
        the old ``visita_oncologica`` score of 10 came entirely from the
        provenance line "DAY SERVICE ONCOLOGIA CLINICA".
        """
        text = """Dipartimento di Radiologia - NEURORADIOLOGIA - Dir.FF Dr. A.Saletti
[INDIRIZZO RIMOSSO]

Data Esame: 25/02/2019 Id Dicom: FSA2005533221 Id Paz: [IDENTIFICATIVO RIMOSSO]
N. Pratica: 97/0022589

Paziente: [PAZIENTE] Sesso: M Data Nascita: [DATA ANAGRAFICA RIMOSSA]

Provenienza: 1E2 DAY SERVICE ONCOLOGIA CLINICA

TC ENCEFALO SENZA E CON CONTRASTO

Esame eseguito prima e dopo somministrazione endovenosa di m.d.c. organo-iodato.
Non alterazioni tomodensitometriche focali, né aree di accentuazione
contrastografica patologica a carico del parenchima cerebrale sovra e
sottotentoriale. Spazi liquorali e cavità ventricolari di ampiezza nei limiti.
"""
        result = self.classifier.classify(
            text,
            "",
            header_metadata=_header(
                department="Dipartimento di Radiologia - NEURORADIOLOGIA",
                provenance="1E2 DAY SERVICE ONCOLOGIA CLINICA",
                specialty="radiologia",
            ),
        )
        self.assertEqual(result, DocumentType.RADIOLOGIA.value)

    def test_tc_report_with_exam_title_after_letterhead_is_radiologia(self):
        """The exam title sits ~450 chars in, after a long letterhead.

        Reproduces DOC_000124: even with oncology terms in the request the
        report stays radiology — the title boost and the department
        short-circuit both fire.
        """
        letterhead = (
            "Dipartimento Attività Integrata di Radiologia "
            "U.O. Radiologia Universitaria Direttore: Prof. Melchiore Giganti "
            "[INDIRIZZO RIMOSSO] Segreteria: tel.: [TELEFONO RIMOSSO] "
            "Fax: [TELEFONO RIMOSSO] Paziente: [PAZIENTE] "
            "Data Esame: 13/05/2025 Data di Nascita: [DATA ANAGRAFICA RIMOSSA] "
            "Ora Esame: [INDIRIZZO RIMOSSO] "
            "Provenienza: 1E2 DAY SERVICE ONCOLOGIA C Acc Number: FSA8907092 "
        )
        body = (
            "Quesito Clinico: RIVALUTAZIONE "
            "Prestazioni eseguite e indicazione di dose secondo l'art.161 del "
            "D.Lgs 101/2020: TC TORACE E ADDOME COMPLETO SENZA E CON CONTRASTO "
            "Classe dose: 4 "
            "Confronto con precedente TC del 10/1/2025. "
            "In ambito toracico non comparsa di alterazioni polmonari "
            "significative; pervie le vie aeree centrali. "
            "Paziente con melanoma nodulare stadio IV in terapia con "
            "nivolumab, ristadiazione richiesta per sospetta metastasi."
        )
        result = self.classifier.classify(
            letterhead + body,
            "",
            header_metadata=_header(
                department="Dipartimento Attività Integrata di Radiologia",
                provenance="1E2 DAY SERVICE ONCOLOGIA C",
                specialty="radiologia",
            ),
        )
        self.assertEqual(result, DocumentType.RADIOLOGIA.value)

    def test_visita_oncologica_di_controllo_stays_oncologica(self):
        """A genuine oncological visit keeps its type.

        Reproduces DOC_000023: the header hint fires before any imaging rule.
        """
        text = """ONCOLOGIA CLINICA AMB. ONCOLOGIA 2

VISITA ONCOLOGICA DI CONTROLLO

Paziente con melanoma nodulare PT4b del braccio sinistro, operato.
Alla PET-FDG preoperatoria non localizzazioni secondarie.
In trattamento con nivolumab da 6 mesi, buona tolleranza.
Si programma TC torace-addome di ristadiazione tra 3 mesi.
"""
        result = self.classifier.classify(
            text,
            "",
            header_metadata=_header(
                department="ONCOLOGIA CLINICA AMB. ONCOLOGIA 2",
                services=["VISITA ONCOLOGICA DI CONTROLLO"],
                specialty="oncologia",
                document_type_hint=DocumentType.VISITA_ONCOLOGICA.value,
            ),
        )
        self.assertEqual(result, DocumentType.VISITA_ONCOLOGICA.value)

    def test_radiotherapy_note_is_radioterapia(self):
        """"RADIOTERAPIA ONCOLOGICA" is a radiotherapy unit, not radiology.

        Reproduces DOC_000011: the radiotherapy session note mentions TC and
        PET but must become ``radioterapia``, not ``radiologia``.
        """
        text = """RADIOTERAPIA ONCOLOGICA AMB. RADIOTERAPICO

inviamo notizie del Sig. [PAZIENTE], giunto alla nostra attenzione per
localizzazioni secondarie al rachide dorsale in paziente pluritrattato
per melanoma nodulare stadio IV.

FDG-PET (24/07/24): l'indagine odierna ha mostrato la presenza di focale
uptake del radiotracciante a carico di alcune stazioni linfonodali.

Previa TC di centratura, il paziente viene oggi sottoposto a radioterapia
su D11, mediante tecnica VMAT e fotoni X da 10 MV, raggiungendo la dose
complessiva di 800 cGy in singola frazione.
"""
        result = self.classifier.classify(
            text,
            "",
            header_metadata=_header(
                department="RADIOTERAPIA ONCOLOGICA AMB. RADIOTERAPICO",
                services=["SEDUTA DI RADIOTERAPIA"],
                specialty="oncologia",
                document_type_hint=DocumentType.VISITA_ONCOLOGICA.value,
            ),
        )
        self.assertEqual(result, DocumentType.RADIOTERAPIA.value)

    def test_visit_mentioning_upcoming_tc_without_report_phrases(self):
        """A visit that schedules an exam has no report-structural phrase.

        The title boost requires a modality *and* a phrase like
        "Esame eseguito" / "Prestazioni eseguite"; a bare "si programma TC"
        must not flip the document.
        """
        text = """ONCOLOGIA CLINICA

Paziente con melanoma metastatico in terapia di mantenimento con
nivolumab. Riferisce astenia di grado 1, non altri disturbi.

Si programma TC torace-addome di ristadiazione alla fine del ciclo,
la paziente eseguirà l'esame in regime ambulatoriale.
"""
        result = self.classifier.classify(
            text,
            "",
            header_metadata=_header(
                department="ONCOLOGIA CLINICA",
                services=["VISITA ONCOLOGICA"],
                specialty="oncologia",
                document_type_hint=DocumentType.VISITA_ONCOLOGICA.value,
            ),
        )
        self.assertEqual(result, DocumentType.VISITA_ONCOLOGICA.value)

    def test_pet_report_from_nuclear_medicine_unit(self):
        """A PET report issued by Medicina Nucleare is medicina_nucleare."""
        text = """U.O. MEDICINA NUCLEARE

Paziente: [PAZIENTE] Sesso: M Data Nascita: [DATA ANAGRAFICA RIMOSSA]
Provenienza: DAY SERVICE ONCOLOGIA CLINICA

ESAME PET/TC TOTAL BODY CON 18F-FDG

Indagine eseguita dopo somministrazione endovenosa di 18F-FDG.
Si documentano aree di ipercaptazione del radiotracciante a livello
epatico (SUVmax 17.8) e scheletrico (SUVmax 20.6), compatibili con
localizzazioni secondarie di melanoma.
"""
        result = self.classifier.classify(
            text,
            "",
            header_metadata=_header(
                department="U.O. MEDICINA NUCLEARE",
                provenance="DAY SERVICE ONCOLOGIA CLINICA",
                specialty="medicina_nucleare",
            ),
        )
        self.assertEqual(result, DocumentType.MEDICINA_NUCLEARE.value)

    def test_imaging_report_without_header_metadata_uses_title_boost(self):
        """No letterhead parsed: the exam title + report phrase still win."""
        text = """TC ENCEFALO SENZA E CON CONTRASTO

Esame eseguito prima e dopo somministrazione endovenosa di m.d.c.
organo-iodato. Non alterazioni tomodensitometriche focali.

Paziente affetto da melanoma nodulare stadio IV, in immunoterapia.
"""
        result = self.classifier.classify(
            text,
            "",
            header_metadata=_header(),
        )
        self.assertEqual(result, DocumentType.RADIOLOGIA.value)

    def test_radiology_letterhead_in_text_without_stored_metadata(self):
        """The letterhead line in the text itself is enough.

        Reproduces reports whose stored header metadata is missing the
        radiology department (extraction quirks): the document opens with
        the radiology letterhead, so the short-circuit still fires.
        """
        text = """Dipartimento Attività Integrata di Radiologia
U.O. Radiologia Universitaria
Direttore: Prof. Melchiore Giganti
[INDIRIZZO RIMOSSO]

Paziente: [PAZIENTE] Data Esame: 13/05/2025

Quesito Clinico: RIVALUTAZIONE
TC TORACE E ADDOME COMPLETO SENZA E CON CONTRASTO

Paziente con melanoma nodulare stadio IV in trattamento con nivolumab.
Non comparsa di alterazioni polmonari significative.
"""
        result = self.classifier.classify(
            text,
            "",
            header_metadata=_header(),
        )
        self.assertEqual(result, DocumentType.RADIOLOGIA.value)

    def test_oncology_visit_quoting_a_past_tc_stays_oncologica(self):
        """Reproduces DOC_008695: a chemo-plan visit that quotes a TC.

        The quoted exam ("TC total body co mdc: ...") carries the modality
        and the contrast abbreviation but none of the report-structural
        phrases, and the document's own letterhead is the oncology
        ambulatory.  As in the real record, no service hint was extracted —
        the visit must keep its type by scoring alone.
        """
        text = """Arcispedale S.Anna
AMB. ONCOLOGIA TERAPIE 2 - UO ONCOLOGIA CLINICA

PRESTAZIONI EROGATE
STESURA PIANO TRATTAM.CHEMIOT.ONCOL.

REFERTO
Anamnesi oncologica: 22/09/20 exeresi di neoformazione sottoscapolare
destra, E.I: metastasi ipodermica di melanoma maligno. BRAF assente.

3/11/20 TC total body co mdc: Il parenchima epatico appare sovvertito da
multiple lesioni nodulari, suggestive per lesioni replicative.

Si propone terapia oncologica di prima linea con nivolumab.
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

    def test_oncology_provenance_does_not_score_as_visit(self):
        """The provenance alone must not inflate visita_oncologica.

        Reproduces the DOC_000154 failure mode in isolation: a document
        with no oncology terms in department or body, only in provenance.
        """
        text = """U.O. DIAGNOSTICA PER IMMAGINI

TC CRANIO SENZA CONTRASTO

Esame eseguito su richiesta del reparto oncologico di provenienza.
Quadro nei limiti, nessuna alterazione focale rilevabile.
"""
        result = self.classifier.classify(
            text,
            "",
            header_metadata=_header(
                department="U.O. DIAGNOSTICA PER IMMAGINI",
                provenance="DAY SERVICE ONCOLOGIA CLINICA",
                specialty="radiologia",
            ),
        )
        self.assertEqual(result, DocumentType.RADIOLOGIA.value)


if __name__ == "__main__":
    unittest.main(verbosity=2)

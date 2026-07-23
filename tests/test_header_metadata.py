"""Regression tests for first-page provenance and service metadata."""

import unittest

from emr_analyzer.models.document import DocumentType
from emr_analyzer.pipeline.classifier import DocumentClassifier
from emr_analyzer.pipeline.cleaner import TextCleaner
from emr_analyzer.pipeline.header_metadata import HeaderMetadataExtractor


class HeaderMetadataExtractorTest(unittest.TestCase):
    def setUp(self):
        self.extractor = HeaderMetadataExtractor()
        self.classifier = DocumentClassifier()

    def test_value_before_department_label_and_services_are_recovered(self):
        raw = """## DATI ANAGRAFICI DEL PAZIENTE
Mario Rossi

DAY SERVICE CARDIOLOGIA
REPARTO/AMBULATORIO

## PRESTAZIONI EROGATE
89.7
VISITA CARDIOLOGICA DI CONTROLLO

## REFERTO
Paziente con carcinoma in immunoterapia. ECG in ritmo sinusale.
"""
        metadata = self.extractor.extract(raw)

        self.assertEqual(metadata.department, "DAY SERVICE CARDIOLOGIA")
        self.assertEqual(metadata.specialty, "cardiologia")
        self.assertEqual(metadata.services, ("VISITA CARDIOLOGICA DI CONTROLLO",))
        self.assertEqual(
            metadata.document_type_hint,
            DocumentType.VISITA_SPECIALISTICA.value,
        )

        # The body contains strong oncology terms, but the explicit performed
        # service remains the authoritative document classification.
        cleaned = TextCleaner().clean(raw)
        classified = self.classifier.classify(
            cleaned,
            "referto_generico.pdf",
            header_metadata=metadata.to_dict(),
        )
        self.assertEqual(classified, DocumentType.VISITA_SPECIALISTICA.value)

    def test_inline_provenance_identifies_neurology(self):
        raw = """PROVENIENZA: AMBULATORIO NEUROLOGIA
PRESTAZIONI EROGATE
PRIMA VISITA NEUROLOGICA
REFERTO
Parestesie agli arti inferiori.
"""
        metadata = self.extractor.extract(raw)
        self.assertEqual(metadata.provenance, "AMBULATORIO NEUROLOGIA")
        self.assertEqual(metadata.specialty, "neurologia")
        self.assertEqual(metadata.services, ("PRIMA VISITA NEUROLOGICA",))
        self.assertEqual(
            metadata.document_type_hint,
            DocumentType.VISITA_SPECIALISTICA.value,
        )

    def test_explicit_oncology_service_remains_oncology(self):
        raw = """U.O. ONCOLOGIA
PRESTAZIONI EROGATE
VISITA ONCOLOGICA
REFERTO
Controllo durante immunoterapia.
"""
        metadata = self.extractor.extract(raw)
        self.assertEqual(metadata.specialty, "oncologia")
        self.assertEqual(
            metadata.document_type_hint,
            DocumentType.VISITA_ONCOLOGICA.value,
        )
        self.assertEqual(
            self.classifier.classify(
                "Controllo durante immunoterapia.",
                "visita.pdf",
                header_metadata=metadata.to_dict(),
            ),
            DocumentType.VISITA_ONCOLOGICA.value,
        )

    def test_services_block_stops_before_report_body(self):
        raw = """PRESTAZIONI EROGATE
VISITA ENDOCRINOLOGICA
REFERTO
Controllo della terapia con levotiroxina.
"""
        metadata = self.extractor.extract(raw)
        self.assertEqual(metadata.services, ("VISITA ENDOCRINOLOGICA",))
        self.assertNotIn("levotiroxina", metadata.classification_text())


class PatientNameHeaderTest(unittest.TestCase):
    def setUp(self):
        self.cleaner = TextCleaner()

    def test_name_is_found_after_or_before_the_label(self):
        label_first = """DATI ANAGRAFICI
NOME E COGNOME
Rossi Maria Luisa
LUOGO E DATA DI NASCITA
"""
        value_first = """DATI ANAGRAFICI
ROSSI MARIA LUISA
NOME E COGNOME
LUOGO E DATA DI NASCITA
"""
        self.assertEqual(
            self.cleaner._find_patient_name(label_first, allow_heuristic=False),
            "Rossi Maria Luisa",
        )
        self.assertEqual(
            self.cleaner._find_patient_name(value_first, allow_heuristic=False),
            "ROSSI MARIA LUISA",
        )

    def test_surname_and_given_name_can_be_separate_table_lines(self):
        text = """NOME E COGNOME
D'AMICO
ANNA-MARIA
LUOGO E DATA DI NASCITA
"""
        self.assertEqual(
            self.cleaner._find_patient_name(text, allow_heuristic=False),
            "D'AMICO ANNA-MARIA",
        )

    def test_unlabelled_uppercase_department_is_not_used_for_routing(self):
        text = """AZIENDA OSPEDALIERA
UNITA OPERATIVA CARDIOLOGIA
DAY SERVICE ONCOLOGIA
"""
        self.assertIsNone(
            self.cleaner._find_patient_name(text, allow_heuristic=False)
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)

"""Tests for the deterministic lab value parser."""

import sys
import os
import unittest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from emr_analyzer.extraction.lab_parser import LabParser
from emr_analyzer.extraction.normalizer import LabNormalizer
from emr_analyzer.pipeline.classifier import DocumentClassifier


def test_normalize_italian_number():
    normalizer = LabNormalizer()
    assert normalizer.normalize_value("1.234,56") == 1234.56
    assert normalizer.normalize_value("98") == 98.0
    assert normalizer.normalize_value("0,9") == 0.9
    assert normalizer.normalize_value("14.5") == 14.5


def test_normalize_parameter():
    normalizer = LabNormalizer()
    assert normalizer.normalize_parameter("Hb") == "emoglobina"
    assert normalizer.normalize_parameter("GOT") == "ast"  # GOT -> AST abbreviation
    assert normalizer.normalize_parameter("Creatinina") == "creatinina"
    assert normalizer.normalize_parameter("Glicemia") == "glucosio"
    assert normalizer.normalize_parameter("PLT") == "piastrine"


def test_normalize_unit():
    normalizer = LabNormalizer()
    assert normalizer.normalize_unit("mg/dl") == "mg/dL"
    assert normalizer.normalize_unit("u/l") == "U/L"
    assert normalizer.normalize_unit("mg/100ml") == "mg/dL"


def test_parse_reference_range():
    normalizer = LabNormalizer()
    low, high = normalizer.parse_reference_range("70-110")
    assert low == 70.0
    assert high == 110.0

    low, high = normalizer.parse_reference_range("< 0.5")
    assert low is None
    assert high == 0.5

    low, high = normalizer.parse_reference_range("> 60")
    assert low == 60.0
    assert high is None


def test_lab_parser_tabular():
    parser = LabParser(LabNormalizer())

    text = """Glucosio           98 mg/dL        (70-110)
Creatinina         0.9 mg/dL       (0.7-1.2)
Emoglobina         14.5 g/dL       (12.0-16.0)
GB                 7.500 /μL       (4.000-10.000)
Piastrine          234.000 /μL     (150.000-450.000)"""

    results = parser.parse(text, patient_id="TEST", document_id="DOC_TEST")

    assert len(results) > 0
    glucose = [r for r in results if "glucosio" in r.normalized_name]
    assert len(glucose) > 0
    assert glucose[0].value == 98.0
    assert glucose[0].unit == "mg/dL"
    assert glucose[0].reference_low == 70.0
    assert glucose[0].reference_high == 110.0


def test_is_abnormal():
    normalizer = LabNormalizer()
    is_ab, flag = normalizer.is_abnormal(150.0, 70.0, 110.0)
    assert is_ab is True
    assert flag == "H"

    is_ab, flag = normalizer.is_abnormal(50.0, 70.0, 110.0)
    assert is_ab is True
    assert flag == "L"

    is_ab, flag = normalizer.is_abnormal(90.0, 70.0, 110.0)
    assert is_ab is False
    assert flag is None


class LabParserRegressionTest(unittest.TestCase):
    """Expose the original function-style checks to unittest discovery."""

    test_normalize_italian_number = staticmethod(test_normalize_italian_number)
    test_normalize_parameter = staticmethod(test_normalize_parameter)
    test_normalize_unit = staticmethod(test_normalize_unit)
    test_parse_reference_range = staticmethod(test_parse_reference_range)
    test_lab_parser_tabular = staticmethod(test_lab_parser_tabular)
    test_is_abnormal = staticmethod(test_is_abnormal)

    def test_real_laboratory_layout_preserves_flags_units_and_ranges(self):
        text = """EMOCROMO
GLOBULI BIANCHI : 6.55 x10^3/µl 4.00 - 11.00
GLOBULI ROSSI : 4.21 x10^6/µl 3.80 - 5.80
HGB : 12.2 g/dl 11.5 - 16.5
HCT : 37 * % 40 - 54
MCV : 87 fl 76 - 96
MCH : 29.0 pg 27.0 - 32.0
MCHC : 33.3 g/dl 30.0 - 35.0
PLT : 249 x10^3/µl 150 - 450
ERITROBLASTI : 0.00 %
NEUTROFILI : 4.51 x10^3/µl 2.00 - 7.50
LINFOCITI : 1.11 * x10^3/µl 1.50 - 5.00
MONOCITI : 0.83 x10^3/µl 0.20 - 1.00
EOSINOFILI: 0.06 x10^3/µl 0.04 - 0.40
BASOFILI : 0.04 x10^3/µl 0.01 - 0.10
Neutrofili : 68.90 %
Linfociti : 16.90 %
Monociti : 12.70 %
Eosinofili : 0.90 %
Basofili : 0.60 %
PT (INR): 1.02 INR Per pazienti in terapia con AVK
PT (Ratio): 1.02 Ratio 0.80 - 1.20
APTT: 0.98 Ratio 0.82 - 1.20
GLUCOSIO : 112 * mg/dl 70 - 110
UREA : 38 mg/dl 17 - 43
CREATININA : 0.78 mg/dl 0.50 - 1.20
Vel. Filtr. Glomerulare (eGFR) 76 ml/min Rapportato alla superficie standard
BILIRUBINA TOTALE : 1.25 * mg/dl < 1.20
BILIRUBINA DIRETTA : 0.23 mg/dl 0.00 - 0.30
SODIO : 133 * mmol/l 136 - 145
POTASSIO : 3.8 mmol/l 3.5 - 5.3
ALT : 77 * U/L < 35
CPK : 104 U/L < 145
LDH : 515 * U/L < 247
LIPASI: 22 U/L < 67
PCR : 2.41 * mg/dl < 0.50
TROPONINA I HS: 5 ng/L < 12
LoD: 2 ng/L
valido dal 16/10/2018
"""
        values = LabParser().parse(
            text,
            patient_id="P001",
            document_id="DOC_LAB",
            sample_date="2023-08-25",
        )
        by_name = {value.normalized_name: value for value in values}

        self.assertEqual(len(values), 36)
        self.assertNotIn("lod", by_name)
        self.assertEqual(by_name["globuli_bianchi"].unit, "×10³/μL")
        self.assertEqual(by_name["globuli_rossi"].unit, "×10⁶/μL")
        self.assertEqual(
            by_name["neutrofili_assoluti"].value, 4.51
        )
        self.assertEqual(
            by_name["neutrofili_percentuale"].value, 68.9
        )
        self.assertEqual(by_name["ematocrito"].flag, "L")
        self.assertEqual(by_name["linfociti_assoluti"].flag, "L")
        self.assertEqual(by_name["glucosio"].flag, "H")
        self.assertEqual(by_name["sodio"].flag, "L")
        self.assertEqual(
            by_name["proteina_c_reattiva"].reference_high, 0.5
        )
        self.assertEqual(by_name["proteina_c_reattiva"].flag, "H")
        self.assertEqual(by_name["egfr"].unit, "mL/min")
        self.assertEqual(
            by_name["tempo_protrombina_inr"].unit, "INR"
        )
        self.assertTrue(all(
            value.sample_date == "2023-08-25" for value in values
        ))

    def test_fragmented_layout_table_does_not_create_zero_value_garbage(self):
        import pandas as pd

        fragmented = pd.DataFrame(
            [
                ["E", "sam", "", ""],
                ["Mate", "rial", "", ""],
                ["Referto id. 20", "593", "", ""],
                ["Dott.ssa Le", "tizia", "", ""],
            ],
            columns=["column_1", "column_2", "column_3", "column_4"],
        )

        values = LabParser().parse(
            "", tables=[fragmented],
            patient_id="P001", document_id="DOC_LAYOUT",
        )

        self.assertEqual(values, [])

    def test_short_lab_sheet_from_oncology_is_still_laboratory(self):
        text = """DEG.ONCOLOGIA CLINICA
Esame Esito U.M. Intervalli Riferimento
Materiale: Siero
ISOAMILASI PANCREATICA : 20 U/L 13 - 53
LIPASI: 21 U/L < 67
Referto Completo
Risultati validati da:
"""

        self.assertEqual(
            DocumentClassifier().classify(text, "documento.pdf"),
            "laboratorio",
        )

    def test_lab_sheet_with_visit_header_hint_still_laboratory(self):
        # Regression B6: a lab result sheet whose first-page services mention
        # "VISITA DI CONTROLLO" gets a visita_specialistica header hint. The
        # structural lab detection must win over that hint, otherwise the lab
        # parser never runs on the values.
        text = """ESAME ESITO U.M. INTERVALLI RIFERIMENTO
Materiale: Siero
GLUCOSIO : 112 mg/dl 70 - 110
CREATININA: 0.9 mg/dl 0.6 - 1.2
Referto Completo
"""
        self.assertEqual(
            DocumentClassifier().classify(
                text,
                "documento.pdf",
                header_metadata={"document_type_hint": "visita_specialistica"},
            ),
            "laboratorio",
        )

    def test_invalid_numeric_cell_is_not_converted_to_zero(self):
        with self.assertRaises(ValueError):
            LabNormalizer().normalize_value("Siero")

    def test_textual_negative_result_is_parsed(self):
        """Determinazioni con valore 'NEGATIVO' vengono riconosciute."""
        text = """SIEROLOGIA
    HIV 1-2 Ab/Ag : NEGATIVO
    HCV Ab : NEGATIVO
    HBsAg : NEGATIVO
    HBsAb : POSITIVO
    HBcAb : NEGATIVO
    """
        values = LabParser().parse(
            text,
            patient_id="P001",
            document_id="DOC_SIER",
            sample_date="2026-01-15",
        )
        self.assertGreater(len(values), 0, "Nessun valore sierologico parsato")

        by_name = {v.parameter_name.strip().rstrip(':'): v for v in values}

        hiv = by_name.get("HIV 1-2 Ab/Ag")
        self.assertIsNotNone(hiv, "HIV non trovato")
        self.assertEqual(hiv.value_text, "NEGATIVO")
        self.assertIsNone(hiv.value)
        self.assertFalse(hiv.is_abnormal)

        hbsab = by_name.get("HBsAb")
        self.assertIsNotNone(hbsab, "HBsAb non trovato")
        self.assertEqual(hbsab.value_text, "POSITIVO")
        self.assertFalse(
            hbsab.is_abnormal,
            "HBsAb positivo non è di per sé patologico (possibile immunità)",
        )

    def test_textual_result_in_table_format(self):
        """Valori testuali in formato tabellare."""
        text = """| Esame | Esito | U.M. | Intervalli Riferimento |
    | HIV 1-2 Ab/Ag : | NEGATIVO | | |
    | HBsAg : | NEGATIVO | | |
    | HBsAb : | POSITIVO | | |
    """
        values = LabParser().parse(
            text,
            patient_id="P002",
            document_id="DOC_TAB",
        )
        self.assertGreater(len(values), 0, "Nessun valore tabellare parsato")

        hiv = [v for v in values if "hiv" in v.normalized_name]
        self.assertTrue(len(hiv) > 0, "HIV non trovato nella tabella")
        self.assertEqual(hiv[0].value_text, "NEGATIVO")

    def test_mixed_numeric_and_textual_results(self):
        """Referto con valori sia numerici che testuali."""
        text = """EMOCROMO
    GLOBULI BIANCHI : 6.55 x10^3/µl 4.00 - 11.00
    HGB : 12.2 g/dl 11.5 - 16.5
    PLT : 249 x10^3/µl 150 - 450
    SIEROLOGIA
    HIV : NEGATIVO
    HCV : NEGATIVO
    TPHA : NEGATIVO
    GLUCOSIO : 112 mg/dl 70 - 110
    """
        values = LabParser().parse(
            text,
            patient_id="P003",
            document_id="DOC_MIX",
        )
        self.assertGreater(len(values), 0)

        # Numeric values still work
        hgb = [v for v in values if "emoglobina" in v.normalized_name]
        self.assertTrue(len(hgb) > 0)
        self.assertEqual(hgb[0].value, 12.2)

        glucose = [v for v in values if "glucosio" in v.normalized_name]
        self.assertTrue(len(glucose) > 0)
        self.assertEqual(glucose[0].value, 112.0)

        # Textual values are captured
        hiv = [v for v in values if "hiv" in v.normalized_name]
        self.assertTrue(len(hiv) > 0, "HIV non trovato nel referto misto")
        self.assertEqual(hiv[0].value_text, "NEGATIVO")

    def test_debole_positivo_is_normalized(self):
        """'DEBOLE POSITIVO' viene normalizzato."""
        normalizer = LabNormalizer()
        self.assertTrue(normalizer.is_textual_result("DEBOLE POSITIVO"))
        self.assertEqual(
            normalizer.normalize_textual_value("DEBOLMENTE POSITIVO"),
            "DEBOLE POSITIVO",
        )

    # ------------------------------------------------------------------
    # Classifier — pre-acute / post-acute discharge letters
    # ------------------------------------------------------------------

    def test_pre_acute_ward_is_detected_in_text(self):
        """I reparti pre-acuti/post-acuti vengono riconosciuti."""
        self.assertTrue(
            DocumentClassifier.is_pre_acute_discharge(
                "REPARTO POST-ACUTI\nLETTERA DI DIMISSIONE"
            )
        )
        self.assertTrue(
            DocumentClassifier.is_pre_acute_discharge(
                "DEGENZA POST ACUTI - DIMISSIONE"
            )
        )
        self.assertTrue(
            DocumentClassifier.is_pre_acute_discharge(
                "LUNGODEGENZA - REPARTO RIABILITATIVO"
            )
        )
        self.assertTrue(
            DocumentClassifier.is_pre_acute_discharge(
                "CURE INTERMEDIE - DIMISSIONE PROTETTA"
            )
        )
        self.assertTrue(
            DocumentClassifier.is_pre_acute_discharge(
                "RIABILITAZIONE INTENSIVA NEUROLOGICA"
            )
        )
        self.assertTrue(
            DocumentClassifier.is_pre_acute_discharge(
                "UNITÀ SPINALE - LETTERA DI DIMISSIONE"
            )
        )
        self.assertFalse(
            DocumentClassifier.is_pre_acute_discharge(
                "ONCOLOGIA CLINICA - VISITA DI CONTROLLO"
            )
        )
        self.assertFalse(
            DocumentClassifier.is_pre_acute_discharge(
                "LABORATORIO ANALISI - EMOCROMO"
            )
        )

    def test_discharge_letter_from_pre_acute_is_classified(self):
        text = """DIPARTIMENTO POST-ACUTI
    REPARTO LUNGODEGENZA
    LETTERA DI DIMISSIONE
    Diagnosi di ingresso: esiti di stroke ischemico
    Decorso clinico: miglioramento progressivo
    Terapia: ASA 100 mg, atorvastatina 20 mg
    Outcome: dimissione a domicilio con follow-up
    """
        classifier = DocumentClassifier()
        result = classifier.classify(text, "lettera_dimissione.pdf")
        self.assertEqual(result, "lettera_dimissione")

    def test_discharge_letter_classification_weight(self):
        """I pattern pre-acuti danno peso extra alla classificazione."""
        classifier = DocumentClassifier()
        text = """REPARTO POST-ACUTI
    DEGENZA RIABILITATIVA
    LETTERA DI DIMISSIONE
    """
        candidates = classifier.get_all_candidates(text)
        # "lettera_dimissione" should be the top candidate
        self.assertTrue(len(candidates) > 0)
        self.assertEqual(candidates[0][0], "lettera_dimissione")
        # High confidence because of the pre-acute patterns
        self.assertGreaterEqual(candidates[0][1], 0.85)

    def test_discharge_letter_with_lab_results_not_classified_as_lab(self):
        """Una lettera di dimissione con esami di laboratorio NON è un referto."""
        text = """REPARTO LUNGODEGENZA
    LETTERA DI DIMISSIONE
    Diagnosi di ingresso: stroke ischemico
    Decorso clinico: miglioramento progressivo
    Terapia: ASA 100 mg/die

    Esami di laboratorio alla dimissione:
    Emoglobina 12.5 g/dL
    Creatinina 0.9 mg/dL
    Glucosio 98 mg/dL
    PCR 0.3 mg/dL

    Esame obiettivo: paziente vigile, orientato
    Outcome: dimissione a domicilio
    """
        classifier = DocumentClassifier()
        result = classifier.classify(text, "dimissione_lungodegenza.pdf")
        self.assertEqual(
            result, "lettera_dimissione",
            f"Una lettera di dimissione con esami NON deve essere "
            f"classificata come '{result}'"
        )

    def test_discharge_letter_beats_visit_classification(self):
        """Lettera di dimissione prevale su visita specialistica."""
        text = """LETTERA DI DIMISSIONE
    REPARTO POST-ACUTI
    Motivo del ricovero: riabilitazione post-ictus
    Visita di controllo: miglioramento della motilità
    Consulenza fisiatrica: proseguire fisioterapia
    Anamnesi: ipertensione, diabete tipo 2
    Esame obiettivo: paziente stabile
    Terapia alla dimissione: invariata
    Follow-up: controllo tra 30 giorni
    """
        classifier = DocumentClassifier()
        result = classifier.classify(text, "lettera.pdf")
        self.assertEqual(
            result, "lettera_dimissione",
            f"Lettera di dimissione classificata come '{result}' "
            f"invece di 'lettera_dimissione'"
        )

    def test_genuine_lab_sheet_still_recognized(self):
        """Un vero referto di laboratorio viene ancora riconosciuto."""
        text = """REFERTO COMPLETO
    ESAME ESITO U.M. INTERVALLI RIFERIMENTO
    Materiale: Siero
    LABORATORIO UNICO

    EMOCROMO
    GLOBULI BIANCHI : 6.55 x10^3/µl 4.00 - 11.00
    GLOBULI ROSSI : 4.21 x10^6/µl 3.80 - 5.80
    HGB : 12.2 g/dl 11.5 - 16.5
    HCT : 37 * % 40 - 54
    PLT : 249 x10^3/µl 150 - 450

    GLUCOSIO : 112 * mg/dl 70 - 110
    CREATININA : 0.78 mg/dl 0.50 - 1.20
    ALT : 77 * U/L < 35

    Risultati validati da: Dott. Rossi
    """
        classifier = DocumentClassifier()
        result = classifier.classify(text, "esami.pdf")
        self.assertEqual(result, "laboratorio")


if __name__ == "__main__":
    unittest.main(verbosity=2)

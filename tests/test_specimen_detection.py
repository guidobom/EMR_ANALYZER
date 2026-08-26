from emr_analyzer.extraction.normalizer import LabNormalizer
from emr_analyzer.extraction.specimen import (
    detect_document_specimen,
    section_specimen,
)


def test_normalize_parameter_specimen_suffixes_urine_hemoglobin():
    normalizer = LabNormalizer()
    assert normalizer.normalize_parameter(
        "Emoglobina", "mg/dL", specimen="urine"
    ) == "emoglobina_urine"
    assert normalizer.normalize_parameter(
        "Emoglobina", "mg/dL"
    ) == "emoglobina"


def test_normalize_parameter_suffix_unit_gate():
    normalizer = LabNormalizer()
    # Blood hemoglobin is g/dL: the gate must refuse the suffix.
    assert normalizer.normalize_parameter(
        "Emoglobina", "g/dL", specimen="urine"
    ) == "emoglobina"
    assert normalizer.normalize_parameter(
        "Proteine", "g/dL", specimen="urine"
    ) == "proteine"
    assert normalizer.normalize_parameter(
        "Proteine", "mg/dL", specimen="urine"
    ) == "proteine_urine"


def test_normalize_parameter_suffix_unknown_specimen_noop():
    normalizer = LabNormalizer()
    assert normalizer.normalize_parameter(
        "Emoglobina", "mg/dL", specimen="liquor"
    ) == "emoglobina"


def test_normalize_parameter_default_specimen_keeps_all_callers():
    normalizer = LabNormalizer()
    assert normalizer.normalize_parameter("Hb") == "emoglobina"
    assert normalizer.normalize_parameter(
        "Neutrofili", "%"
    ) == "neutrofili_percentuale"


def test_detect_document_specimen_serum_is_none():
    text = "Materiale: Siero\nMateriale: Siero\nMateriale: Siero\n"
    assert detect_document_specimen(text) is None


def test_detect_document_specimen_urine():
    assert detect_document_specimen("Materiale: Urina\n") == "urine"
    assert detect_document_specimen(
        "Materiale: Mitto Intermedio\n"
    ) == "urine"
    assert detect_document_specimen(
        "Materiale: Catetere vesc.permanente\n"
    ) == "urine"


def test_detect_document_specimen_liquor_and_feces():
    assert detect_document_specimen("Materiale: LIQUOR\n") == "liquor"
    assert detect_document_specimen("Materiale: Feci\n") == "feces"


def test_detect_document_specimen_mixed_materials_is_none():
    text = "Materiale: Siero\nMateriale: Urina\n"
    assert detect_document_specimen(text) is None


def test_section_specimen_urine_header():
    assert section_specimen("[0] ESAME URINE COMPLETO") == (True, "urine")
    assert section_specimen("URINOCOLTURA Negativa") == (True, "urine")


def test_section_specimen_reset_headers():
    assert section_specimen("EMOCROMO") == (True, None)
    assert section_specimen("[0] ELETTROFORESI PROTEINE") == (True, None)
    assert section_specimen("SIEROLOGIA") == (True, None)


def test_section_specimen_result_lines_are_not_headers():
    assert section_specimen("[0] PROTEINE: 6.0 g/dl") == (False, None)
    assert section_specimen("Glucosio 98 mg/dL (70-110)") == (False, None)
    assert section_specimen("Materiale: Siero") == (False, None)

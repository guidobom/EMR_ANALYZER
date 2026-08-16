"""Tests for name reconstruction in the coordinate-aware identity extractor.

Hospital headers frequently fragment the patient name into per-word rows.
PyMuPDF returns each word of a single-line header in its own baseline group
(``Paziente`` / ``CARAVITA`` / ``CRISTIANA``), and some layouts wrap the
surname and given names onto two lines.  In both cases the extractor must
rebuild the name instead of anchoring on a later label row such as
``Richiesto da:``, which is what pushed such documents into the "Da
assegnare" bucket.
"""

from __future__ import annotations

import unittest

from emr_analyzer.pipeline.patient_identity import (
    PatientIdentityExtractor,
    decode_birth_date_from_cf,
    decode_sex_from_cf,
)


def _row(x0: float, y0: float, text: str) -> dict:
    width = 6.0 * len(text)
    return {
        "key": (int(y0), int(y0 + 12)),
        "x0": float(x0),
        "y0": float(y0),
        "x1": float(x0 + width),
        "y1": float(y0 + 12),
        "words": text.split(),
        "text": text,
    }


class MergeNameValueTest(unittest.TestCase):
    """Pure tests of ``_merged_name_value`` / ``_extend_same_line``."""

    def setUp(self):
        self.extractor = PatientIdentityExtractor()

    def test_single_line_split_name_is_rebuilt(self):
        # Reproduces the Cardiologia header that pushed a real document into
        # "Da assegnare": "Paziente" alone, then each name word as its own
        # row on the same baseline, then a wrong anchor ("Richiesto da:").
        rows = [
            _row(19.6, 33.0, "Paziente"),
            _row(68.5, 33.0, "CARAVITA"),
            _row(164.7, 33.0, "CRISTIANA"),
            _row(582.5, 33.0, "Richiesto da:"),
            _row(663.4, 33.0, "CENTRO PREOPERATORIO"),
        ]
        value = self.extractor._merged_name_value(rows, rows[1])
        self.assertEqual(value, "CARAVITA CRISTIANA")

    def test_split_name_never_returns_nearby_label(self):
        # Even when the label is the only other same-line row, the rebuilt
        # name wins; "Richiesto da:" is rejected by the label vocabulary.
        rows = [
            _row(68.5, 33.0, "CARAVITA"),
            _row(164.7, 33.0, "CRISTIANA"),
            _row(582.5, 33.0, "Richiesto da:"),
        ]
        value = self.extractor._merged_name_value(rows, rows[0])
        self.assertEqual(value, "CARAVITA CRISTIANA")

    def test_single_word_fragment_without_completion_is_rejected(self):
        # A lone single-word row that cannot extend into a valid name must
        # fall through (None), never anchoring on the fragment itself nor on
        # a following label.
        rows = [
            _row(68.5, 33.0, "CARAVITA"),
            _row(582.5, 33.0, "Richiesto da:"),
        ]
        self.assertIsNone(
            self.extractor._merged_name_value(rows, rows[0])
        )

    def test_multi_word_row_unaffected(self):
        rows = [
            _row(68.5, 33.0, "MARIO ROSSI"),
            _row(582.5, 33.0, "Richiesto da:"),
        ]
        value = self.extractor._merged_name_value(rows, rows[0])
        self.assertEqual(value, "MARIO ROSSI")

    def test_wrapped_second_line_still_merges_vertically(self):
        # "CARAVITA" and "CRISTIANA" on two lines in the same column: the
        # vertical continuation path still merges them.
        rows = [
            _row(72.0, 195.0, "CARAVITA"),
            _row(72.0, 210.0, "CRISTIANA"),
            _row(72.0, 230.0, "Richiesto da:"),
        ]
        value = self.extractor._merged_name_value(rows, rows[0])
        self.assertEqual(value, "CARAVITA CRISTIANA")

    def test_rejected_row_does_not_consume_next_candidate(self):
        # The first candidate is a bare single word with no completion; the
        # fall-through must still let a later multi-word row be evaluated
        # (it is a *candidate*, not the label-anchored wrong value).
        rows = [
            _row(68.5, 33.0, "CARAVITA"),
            _row(400.0, 33.0, "RICHIESTA"),        # forbidden vocabulary
            _row(582.5, 33.0, "Richiesto da:"),    # forbidden vocabulary
        ]
        # _merged_name_value on the first candidate returns None; the caller
        # (extract) then moves to the next near-label candidate.
        self.assertIsNone(
            self.extractor._merged_name_value(rows, rows[0])
        )


class SexExtractionTest(unittest.TestCase):
    """Sex values spelled out ("Femmina"/"Maschio") map to F/M.

    Hospital headers often print the sex as a full word next to the SESSO
    label instead of the bare "M"/"F"; the old extractor only matched the
    single letter, so those workspaces were created with ``sex=None`` and
    their folder name lost the sex component.
    """

    def setUp(self):
        self.extractor = PatientIdentityExtractor()

    def test_femmina_on_same_line_maps_to_f(self):
        # CARAVITA-style header: "Sesso:" at x=21, "Femmina" to its right.
        rows = [
            _row(21.1, 96.6, "Sesso:"),
            _row(87.7, 96.6, "Femmina"),
        ]
        sex = self.extractor._extract_sex(rows, 842.0, "native_text", 0.97)
        self.assertIsNotNone(sex)
        self.assertEqual(sex.normalized, "F")

    def test_maschio_maps_to_m(self):
        rows = [
            _row(21.1, 96.6, "Sesso:"),
            _row(87.7, 96.6, "Maschio"),
        ]
        sex = self.extractor._extract_sex(rows, 842.0, "native_text", 0.97)
        self.assertEqual(sex.normalized, "M")

    def test_label_and_value_on_one_row(self):
        # "Sesso: M" inline: the label token makes the single letter valid.
        rows = [
            _row(21.1, 96.6, "Sesso: M"),
            _row(300.0, 96.6, "Data:"),
            _row(400.0, 96.6, "01 08 1985"),
        ]
        sex = self.extractor._extract_sex(rows, 842.0, "native_text", 0.97)
        self.assertEqual(sex.normalized, "M")

    def test_single_letter_value_below_label(self):
        rows = [
            _row(21.1, 96.6, "Sesso:"),
            _row(21.1, 110.0, "F"),
        ]
        sex = self.extractor._extract_sex(rows, 842.0, "native_text", 0.97)
        self.assertEqual(sex.normalized, "F")

    def test_lone_letter_inside_longer_row_is_not_sex(self):
        # A near row whose "F" is one of many tokens (a measure, not sex):
        # without an inline SESSO label the bare letter must not be taken.
        rows = [
            _row(21.1, 96.6, "Sesso:"),
            _row(21.1, 110.0, "EF FV 35 F 42"),
        ]
        sex = self.extractor._extract_sex(rows, 842.0, "native_text", 0.97)
        self.assertIsNone(sex)

    def test_missing_sex_label_returns_none(self):
        rows = [
            _row(21.1, 96.6, "Paziente"),
            _row(87.7, 96.6, "CARAVITA CRISTIANA"),
        ]
        self.assertIsNone(
            self.extractor._extract_sex(rows, 842.0, "native_text", 0.97)
        )


class NameContaminationTest(unittest.TestCase):
    """Name values must not absorb address/locality/type contamination."""

    def setUp(self):
        self.extractor = PatientIdentityExtractor()

    def test_ra_prefix_rejected_as_name(self):
        # "RA" (raccomandata) before the name: the value must be rejected.
        value = self.extractor._validated_name("RA VITALI REMO")
        self.assertIsNone(value)

    def test_tipo_documento_rejected_as_name(self):
        # A document-type header line is never a patient name.
        value = self.extractor._validated_name("TIPO DOCUMENTO")
        self.assertIsNone(value)

    def test_trailing_cf_token_rejected(self):
        # "ADIL GUENNANE CF" — the "CF" abbreviation is not a name token.
        value = self.extractor._validated_name("ADIL GUENNANE CF")
        self.assertIsNone(value)

    def test_vertical_continuation_stops_at_address(self):
        # "CAVALLINI ORESTINO" wrapped onto a second line "VIA ACQUEDOTTO":
        # the address line must not be absorbed into the surname.
        rows = [
            _row(72.0, 195.0, "CAVALLINI"),
            _row(72.0, 210.0, "ORESTINO"),
            _row(72.0, 225.0, "VIA ACQUEDOTTO"),
        ]
        value = self.extractor._merged_name_value(rows, rows[0])
        self.assertEqual(value, "CAVALLINI ORESTINO")

    def test_same_line_extension_stops_before_cf(self):
        # A same-baseline "CF" marker must not extend the name.
        rows = [
            _row(68.5, 33.0, "ADIL"),
            _row(164.7, 33.0, "GUENNANE"),
            _row(300.0, 33.0, "CF"),
        ]
        value = self.extractor._merged_name_value(rows, rows[0])
        self.assertEqual(value, "ADIL GUENNANE")


class FullExtractionTest(unittest.TestCase):
    """End-to-end extraction from a real PDF laid out with PyMuPDF."""

    def setUp(self):
        self.extractor = PatientIdentityExtractor()

    def _extract_name(self, page_lines):
        import fitz
        document = fitz.open()
        page = document.new_page(width=595, height=842)
        for x, y, text in page_lines:
            page.insert_text((x, y), text)
        path = "/tmp/identity_extractor_test.pdf"
        document.save(path)
        document.close()
        try:
            return self.extractor.extract(path).name
        finally:
            from pathlib import Path
            Path(path).unlink(missing_ok=True)

    def test_vertical_two_line_name_from_real_pdf(self):
        # Name wrapped onto two lines below the label, same column: the
        # vertical continuation path merges "CARAVITA" + "CRISTIANA".
        name = self._extract_name([
            (72, 180, "NOME E COGNOME"),
            (72, 195, "CARAVITA"),
            (72, 210, "CRISTIANA"),
            (72, 240, "DATA DI NASCITA"),
            (200, 240, "31 12 1950"),
        ])
        self.assertEqual(name.value, "CARAVITA CRISTIANA")

    def test_normal_single_line_name_still_extracts(self):
        name = self._extract_name([
            (72, 180, "NOME E COGNOME"),
            (200, 180, "MARIO ROSSI"),
            (72, 240, "DATA DI NASCITA"),
            (200, 240, "01 08 1985"),
        ])
        self.assertEqual(name.value, "MARIO ROSSI")


class CfDecodeTest(unittest.TestCase):
    """Deterministic decoding of birth date and sex from a fiscal code."""

    def test_birth_date_and_sex_from_known_codes(self):
        cases = [
            ("CRSMRA29M66D429M", "1929-08-26", "F"),
            ("MRAMRS44P45C980J", "1944-09-05", "F"),
            ("GRZDRO46P24G768B", "1946-09-24", "M"),
            ("PNNRME54D11G916N", "1954-04-11", "M"),
        ]
        for cf, birth, sex in cases:
            with self.subTest(cf=cf):
                self.assertEqual(decode_birth_date_from_cf(cf), birth)
                self.assertEqual(decode_sex_from_cf(cf), sex)

    def test_female_day_increment_and_december(self):
        # 50 → day 10 (female); T is the December month letter.
        self.assertEqual(decode_sex_from_cf("RSSMRA85T50H501W"), "F")
        self.assertEqual(
            decode_birth_date_from_cf("RSSMRA85T50H501W"), "1985-12-10"
        )
        self.assertEqual(decode_sex_from_cf("RSSMRA85T10H501U"), "M")
        self.assertEqual(
            decode_birth_date_from_cf("RSSMRA85T10H501U"), "1985-12-10"
        )

    def test_invalid_structure_returns_none(self):
        self.assertIsNone(decode_birth_date_from_cf("ABC"))
        self.assertIsNone(decode_birth_date_from_cf(""))
        self.assertIsNone(decode_sex_from_cf("XYZ"))


class BirthDateCfFallbackTest(unittest.TestCase):
    """_extract_birth_date decodes the birth date from the CF when no
    dedicated birth-date label is printed on the header."""

    def setUp(self):
        self.extractor = PatientIdentityExtractor()

    def test_decode_from_fiscal_code_when_no_birth_label(self):
        rows = [
            _row(20.0, 100.0, "CODICE FISCALE"),
            _row(200.0, 100.0, "CRSMRA29M66D429M"),
        ]
        field = self.extractor._extract_birth_date(
            rows, 842.0, "native_text", 0.97
        )
        self.assertIsNotNone(field)
        self.assertEqual(field.normalized, "1929-08-26")

    def test_none_when_no_birth_label_and_no_cf(self):
        rows = [
            _row(20.0, 100.0, "NOME E COGNOME"),
            _row(200.0, 100.0, "MARIO ROSSI"),
        ]
        field = self.extractor._extract_birth_date(
            rows, 842.0, "native_text", 0.97
        )
        self.assertIsNone(field)

    def test_labelled_birth_wins_over_cf_decode(self):
        rows = [
            _row(20.0, 100.0, "CODICE FISCALE"),
            _row(200.0, 100.0, "CRSMRA29M66D429M"),
            _row(20.0, 140.0, "DATA DI NASCITA"),
            _row(200.0, 140.0, "26.08.1929"),
        ]
        field = self.extractor._extract_birth_date(
            rows, 842.0, "native_text", 0.97
        )
        self.assertIsNotNone(field)
        self.assertEqual(field.value, "26.08.1929")
        self.assertEqual(field.normalized, "1929-08-26")


if __name__ == "__main__":
    unittest.main()

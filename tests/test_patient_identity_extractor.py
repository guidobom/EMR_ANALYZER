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

from emr_analyzer.pipeline.patient_identity import PatientIdentityExtractor


def _row(x0: float, y0: float, text: str) -> dict:
    width = 40.0 + 5.0 * len(text)
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


if __name__ == "__main__":
    unittest.main()

"""Regression tests for parse_italian_date.

Covers the P015 corruption: the DMY pattern used to match "29-08-26" inside
"1929-08-26", turning a 1929 birth date into "2026-08-29".
"""

from __future__ import annotations

import unittest

from emr_analyzer.utils.date_utils import parse_italian_date


class ParseItalianDateTest(unittest.TestCase):
    def test_ymd_four_digit_year_not_truncated(self):
        # Regression P015: "29" is the tail of "1929", not the day.
        self.assertEqual(parse_italian_date("1929-08-26"), "1929-08-26")

    def test_ymd_with_20xx_year_not_truncated(self):
        # Same corruption would have hit any YMD whose last two digits form
        # a plausible DMY, e.g. "2029-08-26" → "2026-08-29".
        self.assertEqual(parse_italian_date("2029-08-26"), "2029-08-26")

    def test_dmy_dot(self):
        self.assertEqual(parse_italian_date("26.08.1929"), "1929-08-26")

    def test_dmy_slash(self):
        self.assertEqual(parse_italian_date("29/08/1926"), "1926-08-29")

    def test_dmy_two_digit_year(self):
        # A standalone DMY with a 2-digit year must still be parsed as DMY.
        self.assertEqual(parse_italian_date("29-08-26"), "2026-08-29")

    def test_dmy_space_separated(self):
        # Anatomy-pathology reports print dates space-separated
        # ("firma referto (23 10 2014)"), invisible to the ./ - patterns.
        self.assertEqual(parse_italian_date("23 10 2014"), "2014-10-23")

    def test_dmy_space_separated_birth(self):
        # "Luogo di Nascita: 21 02 1964" — same space-separated form.
        self.assertEqual(parse_italian_date("21 02 1964"), "1964-02-21")

    def test_dmy_space_separated_two_digit_year(self):
        self.assertEqual(parse_italian_date("23 10 14"), "2014-10-23")

    def test_ymd_slash(self):
        self.assertEqual(parse_italian_date("2026/08/29"), "2026-08-29")

    def test_ymd_dot(self):
        self.assertEqual(parse_italian_date("2026.04.12"), "2026-04-12")

    def test_dmy_without_padding(self):
        self.assertEqual(parse_italian_date("1.1.2026"), "2026-01-01")

    def test_invalid_date_returns_none(self):
        self.assertIsNone(parse_italian_date("31.02.2026"))
        self.assertIsNone(parse_italian_date(""))

    def test_date_inside_longer_number_not_matched(self):
        # The guards must not let a DMY match inside a longer digit run.
        self.assertIsNone(parse_italian_date("0129-08-26"))


if __name__ == "__main__":
    unittest.main()

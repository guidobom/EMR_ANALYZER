"""Regression tests for the deterministic de-identification module.

Covers the v1.0 privacy blockers:
- P1: postal addresses without a civic number or CAP were left in clear
      ("Via Roma 12", "Via Travaglio, Migliarino").
- P2: names in narrative prose survived when no identity was extracted
      ("Il paziente Mario Rossi …", "Sig. Mario Rossi").
"""

from __future__ import annotations

import unittest

from emr_analyzer.pipeline.sensitive_data import SensitiveDataSanitizer


class PostalAddressTest(unittest.TestCase):
    def setUp(self):
        self.sanitizer = SensitiveDataSanitizer()

    def _text(self, raw: str) -> str:
        return self.sanitizer.sanitize(raw).text

    def test_civic_number_without_comma(self):
        # The reported P1 leak: no comma, no CAP.
        self.assertEqual(self._text("Via Roma 12"), "[INDIRIZZO RIMOSSO]")

    def test_street_comma_town_without_cap(self):
        # The reported P1 leak: comma + locality, no CAP, no civic number.
        self.assertEqual(
            self._text("Via Travaglio, Migliarino"), "[INDIRIZZO RIMOSSO]"
        )

    def test_civic_number_with_comma(self):
        self.assertEqual(
            self._text("Via delle Margherite 45, Firenze"),
            "[INDIRIZZO RIMOSSO]",
        )

    def test_cap_alone(self):
        self.assertEqual(
            self._text("Piazza Garibaldi 20100 Milano"), "[INDIRIZZO RIMOSSO]"
        )

    def test_via_suffix_civic(self):
        self.assertEqual(
            self._text("Via Garibaldi 12/b"), "[INDIRIZZO RIMOSSO]"
        )

    def test_labeled_address(self):
        self.assertEqual(
            self._text("Residenza: Via Cesare Battisti 3"),
            "[INDIRIZZO RIMOSSO]",
        )

    def test_drug_administration_routes_not_redacted(self):
        # The street keywords are also drug-administration routes in clinical
        # prose: these must survive verbatim.
        self.assertNotIn(
            "[INDIRIZZO RIMOSSO]",
            self._text("Via orale: 1 compressa al giorno"),
        )
        self.assertNotIn(
            "[INDIRIZZO RIMOSSO]",
            self._text("Via endovenosa in bolo 40 mg"),
        )
        self.assertNotIn(
            "[INDIRIZZO RIMOSSO]",
            self._text("assumere il farmaco per via orale ogni 8 ore"),
        )

    def test_short_street_name_without_anchor_not_redacted(self):
        # A bare street name with no number/CAP/comma is left alone: it is
        # indistinguishable from clinical prose ("Via Valsalva").
        self.assertEqual(self._text("Via Valsalva\nmanovra valida"), "Via Valsalva\nmanovra valida")

    def test_civic_number_followed_by_dose_unit_not_redacted(self):
        # A civic number immediately followed by a dose unit is a medication
        # line, not an address. The unit guard must survive back-tracking of
        # the "1-4 digits" civic anchor.
        self.assertEqual(
            self._text("Via delle Margherite 40 mg"),
            "Via delle Margherite 40 mg",
        )

    def test_particle_street_without_civic_not_redacted(self):
        # Particles alone (no civic number/CAP/comma) stay in clear.
        self.assertEqual(
            self._text("Via della Repubblica\npaziente in cura"),
            "Via della Repubblica\npaziente in cura",
        )


class ProseNameTest(unittest.TestCase):
    def setUp(self):
        self.sanitizer = SensitiveDataSanitizer()

    def _text(self, raw: str) -> str:
        return self.sanitizer.sanitize(raw).text

    def test_paziente_full_name_redacted_without_identity(self):
        # P2: no identity was extracted, but the full name must still go.
        self.assertEqual(
            self._text("Il paziente Mario Rossi è in terapia"),
            "Il [PAZIENTE] è in terapia",
        )

    def test_paziente_full_name_in_flowing_prose(self):
        self.assertEqual(
            self._text("paziente Mario Rossi riferisce dolore"),
            "[PAZIENTE] riferisce dolore",
        )

    def test_sig_variant_redacted(self):
        self.assertEqual(
            self._text("Il Sig. Mario Rossi è stato dimesso"),
            "Il [PAZIENTE] è stato dimesso",
        )

    def test_sigra_variant_redacted(self):
        self.assertEqual(
            self._text("La sig.ra Bianchi riferisce"), "La [PAZIENTE] riferisce"
        )

    def test_signora_variant_redacted(self):
        self.assertEqual(
            self._text("La Signora Maria Verdi presenta"), "La [PAZIENTE] presenta"
        )

    def test_single_capitalized_descriptor_not_redacted(self):
        # Precision guard: a single capitalized clinical descriptor after
        # "paziente" must not be mistaken for a name.
        self.assertEqual(
            self._text("paziente Diabetico in terapia"),
            "paziente Diabetico in terapia",
        )

    def test_lowercase_descriptor_not_redacted(self):
        self.assertEqual(
            self._text("il paziente oncologico è seguito"),
            "il paziente oncologico è seguito",
        )

    def test_no_double_redaction_when_identity_known(self):
        # A known identity already redacts the name; the prose fallback must
        # not re-process the [PAZIENTE] placeholder.
        identity = {"name": "Mario Rossi"}
        out = self.sanitizer.sanitize(
            "Il paziente Mario Rossi è in terapia", identity
        )
        self.assertNotIn("Rossi", out.text)
        self.assertIn("[PAZIENTE]", out.text)


class ExistingRedactionsTest(unittest.TestCase):
    def setUp(self):
        self.sanitizer = SensitiveDataSanitizer()

    def _text(self, raw: str, identity=None) -> str:
        return self.sanitizer.sanitize(raw, identity).text

    def test_email(self):
        self.assertEqual(
            self._text("contatto mario@example.com"),
            "contatto [EMAIL RIMOSSA]",
        )

    def test_fiscal_code(self):
        self.assertEqual(
            self._text("CF: RSSMRA80A01H501U"),
            "CF: [CODICE FISCALE RIMOSSO]",
        )

    def test_birth_date(self):
        self.assertEqual(
            self._text("Data di nascita: 01/02/1950"),
            "Data di nascita: [DATA ANAGRAFICA RIMOSSA]",
        )

    def test_phone(self):
        self.assertEqual(
            self._text("tel. 333 1234567"),
            "tel.: [TELEFONO RIMOSSO]",
        )

    def test_phone_without_separator_dot(self):
        # The separator must not be doubled into "cell::".
        self.assertEqual(
            self._text("cell: 333 1234567"),
            "cell: [TELEFONO RIMOSSO]",
        )

    def test_known_name_redacted(self):
        identity = {"name": "MARIO ROSSI"}
        out = self.sanitizer.sanitize(
            "Il paziente MARIO ROSSI è seguito", identity
        ).text
        self.assertNotIn("ROSSI", out)
        self.assertNotIn("MARIO", out)


if __name__ == "__main__":
    unittest.main()

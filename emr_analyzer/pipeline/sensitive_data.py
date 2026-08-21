"""Deterministic removal of direct identifiers from clinical prose.

Raw identity values are accepted only in memory.  They are never included in
the returned counters or persisted by this module.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import re
from typing import Mapping


DEIDENTIFICATION_VERSION = "clinical_text_deidentification_v1"

_EMAIL_RE = re.compile(
    r"(?<![\w.+-])[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}(?![\w-])",
    re.IGNORECASE,
)
_FISCAL_CODE_RE = re.compile(
    r"\b[A-Z]{6}\s*\d{2}\s*[A-Z]\s*\d{2}\s*[A-Z]\s*\d{3}\s*[A-Z]\b",
    re.IGNORECASE,
)
_LABELLED_PHONE_RE = re.compile(
    r"(?i)\b(?P<label>tel(?:efono)?|cell(?:ulare)?|mobile|fax)"
    r"\s*(?P<sep>[:.]?)\s*(?P<value>(?:\+?\d[\d\s()./-]{5,}\d))"
)
_ITALIAN_PHONE_RE = re.compile(
    r"(?<![\w])(?:\+?39[\s./-]*)?(?:0\d{1,3}|3\d{2})"
    r"(?:[\s./-]*\d){6,9}(?![\w])"
)
_LABELLED_ADDRESS_RE = re.compile(
    r"(?im)^[ \t]*(?:indirizzo|residente|residenza|domicilio|domiciliat[oa])"
    r"\s*:?\s*[^\n]+$"
)
_POSTAL_ADDRESS_RE = re.compile(
    r"(?m)^[ \t]*(?:Via|VIA|Viale|VIALE|Piazza|PIAZZA|Corso|CORSO|"
    r"Strada|STRADA|Vicolo|VICOLO|Largo|LARGO|Località|LOCALITÀ)"
    r"(?!\s+(?i:orale|endovenosa|intravenosa|intramuscolare|"
    r"sottocutanea|topica|inalatoria|rettale|sublinguale|transdermica)\b)"
    r"(?:\s+(?:de(?:i|l|lla|llo|lle|gli)|di|san|santa|santo|santi|"
    r"sant'|al|alla|allo|alle|ai|il|lo|la|le|gli|in|su|da)){0,2}"
    r"\s+[A-ZÀ-Ü][^\n;]{1,90}?"
    r"(?:"
    r",\s*\d{1,4}(?!\d)(?:[/A-Za-z])?"
    r"(?!\s*(?i:mg|mcg|µg|g|ml|ui|unità)\b)"                       # ,12
    r"|,\s*[A-ZÀ-Ü][A-Za-zÀ-ÖØ-öø-ÿ'’\-\. ]{0,30}"                 # ,Migliarino
    r"|\s+\d{1,4}(?!\d)(?:[/A-Za-z])?"
    r"(?!\s*(?i:mg|mcg|µg|g|ml|ui|unità)\b)"                       # 12 (no comma)
    r"|\b\d{5}\b)"                                                 # CAP
    r"[^\n]*$"
)
_LABELLED_IDENTIFIER_RE = re.compile(
    r"(?i)(?<!\[)\b(?P<label>codice\s+fiscale|codice\s+assistito|id\s+paziente|"
    r"numero\s+cartella|n[°.]?\s*cartella|nosologico)"
    r"\s*[:#]?\s*(?P<value>(?!RIMOSS[OA]\b)[A-Z0-9][A-Z0-9./_-]{3,})"
)
_LABELLED_BIRTH_DATE_RE = re.compile(
    r"(?i)\b(?P<label>data\s+di\s+nascita|nato\s+il|nata\s+il)"
    r"\s*:?\s*(?P<value>\d{1,2}[./-]\d{1,2}[./-]\d{2,4})"
)
# Names in prose that survive even when no identity was extracted from the
# document header (e.g. scanned PDFs without OCR). Deterministic fallback:
# a capitalized name (1–2 tokens) after an explicit courtesy title, or a full
# name (two consecutive capitalized tokens) after "paziente".  Only the
# prefixes are matched case-insensitively: the name tokens must keep their
# initial capital letter, otherwise any lowercase word ("riferisce", "in")
# would be consumed as a name and corrupt clinical prose.
_PROSE_SIG_NAME_RE = re.compile(
    r"\b(?i:signor(?:a|e)?|sig(?:\.|\.ra|\.na|ra|na)?)\s+"
    r"[A-ZÀ-Ü][A-Za-zÀ-ÖØ-öø-ÿ'’\-]+"
    r"(?:\s+[A-ZÀ-Ü][A-Za-zÀ-ÖØ-öø-ÿ'’\-]+)?"
    r"\b"
)
_PROSE_PAZIENTE_FULLNAME_RE = re.compile(
    r"\b(?i:paziente)\s+"
    r"[A-ZÀ-Ü][A-Za-zÀ-ÖØ-öø-ÿ'’\-]+"
    r"\s+[A-ZÀ-Ü][A-Za-zÀ-ÖØ-öø-ÿ'’\-]+"
    r"\b"
)


@dataclass
class SensitiveDataSanitizationResult:
    text: str
    counts: dict[str, int] = field(default_factory=dict)

    @property
    def total(self) -> int:
        return sum(self.counts.values())


class SensitiveDataSanitizer:
    """Remove known patient identity and common direct identifiers."""

    def sanitize(
        self,
        text: str,
        identity=None,
    ) -> SensitiveDataSanitizationResult:
        value = str(text or "")
        counts: Counter[str] = Counter()
        identity_values = self._identity_values(identity)

        patient_name = identity_values.get("name")
        if patient_name:
            value, count = self._redact_patient_name(value, patient_name)
            counts["patient_name"] += count

        for field_name, replacement in (
            ("fiscal_code", "[CODICE FISCALE RIMOSSO]"),
            ("birth_date", "[DATA ANAGRAFICA RIMOSSA]"),
            ("hospital_patient_id", "[IDENTIFICATIVO RIMOSSO]"),
        ):
            raw_identifier = identity_values.get(field_name)
            if not raw_identifier:
                continue
            pattern = self._flexible_literal_pattern(raw_identifier)
            value, count = pattern.subn(replacement, value)
            counts[field_name] += count

        value, count = _EMAIL_RE.subn("[EMAIL RIMOSSA]", value)
        counts["email"] += count

        value, count = _FISCAL_CODE_RE.subn(
            "[CODICE FISCALE RIMOSSO]", value
        )
        counts["fiscal_code"] += count

        value, count = _LABELLED_BIRTH_DATE_RE.subn(
            lambda match: (
                f"{match.group('label')}: [DATA ANAGRAFICA RIMOSSA]"
            ),
            value,
        )
        counts["birth_date"] += count

        value, count = _LABELLED_IDENTIFIER_RE.subn(
            lambda match: (
                f"{match.group('label')}: [IDENTIFICATIVO RIMOSSO]"
            ),
            value,
        )
        counts["administrative_identifier"] += count

        def _phone_replacement(match: re.Match) -> str:
            separator = match.group("sep")
            # Normalise the separator to a single colon: "tel." stays "tel.:",
            # while "tel:" and "tel 123" both become "tel: …".
            normalized = "" if separator in ("", ":") else separator
            return f"{match.group('label')}{normalized}: [TELEFONO RIMOSSO]"

        value, count = _LABELLED_PHONE_RE.subn(_phone_replacement, value)
        counts["phone"] += count

        value, count = self._replace_verified_phones(value)
        counts["phone"] += count

        value, count = _LABELLED_ADDRESS_RE.subn(
            "[INDIRIZZO RIMOSSO]", value
        )
        counts["address"] += count
        value, count = _POSTAL_ADDRESS_RE.subn(
            "[INDIRIZZO RIMOSSO]", value
        )
        counts["address"] += count

        # Deterministic prose fallback: these run even when no identity was
        # extracted, so a name introduced in narrative prose never survives.
        value, count = _PROSE_SIG_NAME_RE.subn("[PAZIENTE]", value)
        counts["patient_name"] += count
        value, count = _PROSE_PAZIENTE_FULLNAME_RE.subn("[PAZIENTE]", value)
        counts["patient_name"] += count

        value = self._clean_spacing(value)
        return SensitiveDataSanitizationResult(
            text=value,
            counts={
                category: amount
                for category, amount in counts.items()
                if amount
            },
        )

    def sanitize_payload(self, payload, identity=None):
        """Recursively de-identify parser JSON while preserving geometry.

        Word-level geometry stores names as separate tokens, where a normal
        full-name regex cannot match. Exact identity-name tokens are therefore
        removed in addition to ordinary string sanitization.
        """
        identity_values = self._identity_values(identity)
        name_tokens = {
            token.casefold()
            for token in re.findall(
                r"[A-Za-zÀ-ÖØ-öø-ÿ]+(?:['’\-][A-Za-zÀ-ÖØ-öø-ÿ]+)?",
                identity_values.get("name", ""),
            )
            if len(token) >= 3
        }
        direct_values = {
            re.sub(r"\W", "", value).casefold()
            for key, value in identity_values.items()
            if key != "name" and value
        }

        def walk(value):
            if isinstance(value, Mapping):
                return {str(key): walk(child) for key, child in value.items()}
            if isinstance(value, list):
                return [walk(child) for child in value]
            if isinstance(value, tuple):
                return [walk(child) for child in value]
            if isinstance(value, str):
                compact = re.sub(r"\W", "", value).casefold()
                if value.strip().casefold() in name_tokens or (
                    compact and compact in direct_values
                ):
                    return "[IDENTIFICATIVO RIMOSSO]"
                return self.sanitize(value, identity).text
            return value

        return walk(payload)

    @classmethod
    def _redact_patient_name(
        cls, text: str, patient_name: str
    ) -> tuple[str, int]:
        words = re.findall(
            r"[A-Za-zÀ-ÖØ-öø-ÿ]+(?:['’\-][A-Za-zÀ-ÖØ-öø-ÿ]+)?",
            patient_name,
        )
        if not words:
            return text, 0

        variants = []
        for candidate in (
            words,
            list(reversed(words)),
            words[-1:] + words[:-1],
            words[1:] + words[:1],
        ):
            if candidate not in variants:
                variants.append(candidate)

        result = text
        total = 0
        for variant in sorted(variants, key=len, reverse=True):
            if len(variant) < 2:
                continue
            pattern = re.compile(
                r"(?<!\w)"
                + r"(?:[\s,]+)".join(re.escape(word) for word in variant)
                + r"(?!\w)",
                re.IGNORECASE,
            )
            result, count = pattern.subn("[PAZIENTE]", result)
            total += count

        # A surname or given name may occur alone in narrative prose.  Redact
        # it when explicitly introduced as a person, or when it is printed in
        # uppercase.  Avoid case-insensitive global replacement because names
        # such as "Massimo" and "Rosa" can also be clinical words.
        for word in words:
            if len(word) < 4:
                continue
            contextual = re.compile(
                r"(?i)(?P<prefix>\b(?:sig(?:\.|nor[ae]?|\.?\s*(?:ra)?)|"
                r"paziente)\s+)"
                + re.escape(word)
                + r"\b"
            )
            result, count = contextual.subn(
                lambda match: f"{match.group('prefix')}[PAZIENTE]",
                result,
            )
            total += count

            uppercase = re.compile(
                r"(?<!\w)" + re.escape(word.upper()) + r"(?!\w)"
            )
            result, count = uppercase.subn("[PAZIENTE]", result)
            total += count

        return result, total

    @staticmethod
    def _identity_values(identity) -> dict[str, str]:
        if not identity:
            return {}
        if isinstance(identity, Mapping):
            source = identity
        else:
            source = {
                field_name: getattr(identity, field_name, None)
                for field_name in (
                    "name",
                    "fiscal_code",
                    "birth_date",
                    "hospital_patient_id",
                )
            }

        values = {}
        for field_name, field_value in source.items():
            if field_value is None:
                continue
            raw_value = getattr(field_value, "value", field_value)
            if raw_value:
                values[str(field_name)] = str(raw_value).strip()
        return values

    @staticmethod
    def _flexible_literal_pattern(raw_value: str) -> re.Pattern:
        parts = re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ0-9]+", raw_value)
        if not parts:
            return re.compile(r"(?!x)x")
        return re.compile(
            r"(?<!\w)" + r"[\s./_-]*".join(
                re.escape(part) for part in parts
            ) + r"(?!\w)",
            re.IGNORECASE,
        )

    @staticmethod
    def _replace_verified_phones(text: str) -> tuple[str, int]:
        count = 0

        def replacement(match: re.Match) -> str:
            nonlocal count
            digits = re.sub(r"\D", "", match.group(0))
            # Avoid confusing short clinical values and dates with contacts.
            if not 9 <= len(digits) <= 13:
                return match.group(0)
            count += 1
            return "[TELEFONO RIMOSSO]"

        return _ITALIAN_PHONE_RE.sub(replacement, text), count

    @staticmethod
    def _clean_spacing(text: str) -> str:
        value = re.sub(r"[ \t]+\n", "\n", text)
        value = re.sub(r"[ \t]{2,}", " ", value)
        value = re.sub(r"\n{3,}", "\n\n", value)
        return value.strip()

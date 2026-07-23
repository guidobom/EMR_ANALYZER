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
    r"\s*[:.]?\s*(?P<value>(?:\+?\d[\d\s()./-]{5,}\d))"
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
    r"sottocutanea|topica|inalatoria|rettale|sublinguale)\b)"
    r"\s+[A-ZÀ-Ü][^\n;]{1,90}?"
    r"(?:,\s*\d{1,4}(?:[/A-Za-z])?"
    r"(?!\s*(?i:mg|mcg|µg|g|ml|ui|unità)\b)|\b\d{5}\b)"
    r"[^\n]*$"
)
_LABELLED_IDENTIFIER_RE = re.compile(
    r"(?i)\b(?P<label>codice\s+fiscale|codice\s+assistito|id\s+paziente|"
    r"numero\s+cartella|n[°.]?\s*cartella|nosologico)"
    r"\s*[:#]?\s*(?P<value>[A-Z0-9][A-Z0-9./_-]{3,})"
)
_LABELLED_BIRTH_DATE_RE = re.compile(
    r"(?i)\b(?P<label>data\s+di\s+nascita|nato\s+il|nata\s+il)"
    r"\s*:?\s*(?P<value>\d{1,2}[./-]\d{1,2}[./-]\d{2,4})"
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

        value, count = _LABELLED_PHONE_RE.subn(
            lambda match: f"{match.group('label')}: [TELEFONO RIMOSSO]",
            value,
        )
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

        value = self._clean_spacing(value)
        return SensitiveDataSanitizationResult(
            text=value,
            counts={
                category: amount
                for category, amount in counts.items()
                if amount
            },
        )

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

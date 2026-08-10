"""Lab parameter name and unit normalization."""

import re

from ..config import LAB_SYNONYMS, UNIT_MAPPING


class LabNormalizer:
    """Normalizes lab parameter names and units to canonical forms."""

    def normalize_parameter(self, name: str, unit: str = "") -> str:
        """
        Normalize a lab parameter name:
        - Lowercase
        - Map synonyms (Hb -> emoglobina, GOT -> AST)
        - Remove extraneous whitespace and punctuation
        """
        if not name:
            return ""

        cleaned = name.strip().lower()
        # Remove trailing punctuation
        cleaned = cleaned.rstrip(".,;: ")

        # Direct synonym lookup
        if cleaned in LAB_SYNONYMS:
            cleaned = LAB_SYNONYMS[cleaned]

        else:
            # Partial matching for composite names
            for key, value in LAB_SYNONYMS.items():
                if cleaned.startswith(key) or key in cleaned:
                    # Only replace if key is a whole word
                    import re
                    pattern = re.compile(
                        rf'\b{re.escape(key)}\b', re.IGNORECASE
                    )
                    if pattern.search(cleaned):
                        cleaned = pattern.sub(value, cleaned)

        # Replace spaces with underscores for canonical form
        cleaned = cleaned.replace(" ", "_")
        # Remove duplicate underscores
        import re
        cleaned = re.sub(r'_+', '_', cleaned)

        normalized_unit = self.normalize_unit(unit) if unit else ""
        differential = {
            "neutrofili", "linfociti", "monociti", "eosinofili",
            "basofili", "eritroblasti",
        }
        if cleaned in differential:
            if normalized_unit == "%":
                cleaned = f"{cleaned}_percentuale"
            elif normalized_unit in {
                "/μL", "×10³/μL", "×10⁶/μL",
            }:
                cleaned = f"{cleaned}_assoluti"

        return cleaned or name.lower().replace(" ", "_")

    def normalize_unit(self, unit: str) -> str:
        """Normalize a unit to canonical form."""
        if not unit:
            return ""

        cleaned = unit.strip().lower()
        cleaned = cleaned.replace("µ", "μ").replace("×", "x")
        cleaned = cleaned.replace(" ", "")
        # Remove trailing/leading dots, spaces
        cleaned = cleaned.strip(".,;: ")

        import re
        count_match = re.fullmatch(
            r"x10(?:\^|\*\*)?([36])/([μu])l", cleaned
        )
        if count_match:
            exponent = "³" if count_match.group(1) == "3" else "⁶"
            return f"×10{exponent}/μL"

        # Direct mapping
        if cleaned in UNIT_MAPPING:
            return UNIT_MAPPING[cleaned]

        # Try with original casing variations
        for key, value in UNIT_MAPPING.items():
            if cleaned == key.lower():
                return value

        # Return original if no mapping found (preserving common formatting)
        return unit.strip()

    # Recognized textual (non-numeric) lab result values.
    _TEXTUAL_RESULT_PATTERN = re.compile(
        r"^\s*(NEGATIVO|POSITIVO|DEBOLE\s*POSITIVO|DEBOLMENTE\s*POSITIVO|"
        r"ASSENTE|PRESENTE|NON\s*RILEVABILE|RILEVABILE|"
        r"NELLA\s*NORMA|ALTERATO|SCARSI|NUMEROSI|RARI|"
        r"DEBOLE|FORTEMENTE\s*POSITIVO|NEGATIVITA|POSITIVITA)\s*$",
        re.IGNORECASE,
    )

    @classmethod
    def is_textual_result(cls, value_str: str) -> bool:
        """Check whether a value string is a recognized textual lab result."""
        if not value_str:
            return False
        return bool(cls._TEXTUAL_RESULT_PATTERN.match(value_str.strip()))

    def normalize_value(self, value_str: str, unit: str = "") -> float:
        """Convert Italian-formatted number to float.

        Raises ValueError for textual (non-numeric) results — callers should
        check ``is_textual_result()`` first and route those to ``value_text``.
        """
        from ..utils.text_utils import normalize_italian_number

        if not value_str:
            raise ValueError("Valore di laboratorio vuoto")

        result = normalize_italian_number(value_str)
        if result is None:
            raise ValueError(
                f"Valore di laboratorio non numerico: {value_str!r}"
            )
        normalized_unit = self.normalize_unit(unit) if unit else ""
        if (
            normalized_unit == "/μL"
            and re.fullmatch(r"\d{1,3}\.\d{3}", value_str.strip())
        ):
            return float(value_str.strip().replace(".", ""))
        return result

    @classmethod
    def normalize_textual_value(cls, value_str: str) -> str:
        """Normalize a textual lab result to its canonical form."""
        if not value_str:
            return ""
        cleaned = value_str.strip().upper()
        # Remove extra whitespace
        cleaned = re.sub(r"\s+", " ", cleaned)
        # Canonical forms
        mapping = {
            "DEBOLE POSITIVO": "DEBOLE POSITIVO",
            "DEBOLMENTE POSITIVO": "DEBOLE POSITIVO",
            "FORTEMENTE POSITIVO": "FORTEMENTE POSITIVO",
            "NON RILEVABILE": "NON RILEVABILE",
            "NELLA NORMA": "NELLA NORMA",
            "NEGATIVITA": "NEGATIVO",
            "POSITIVITA": "POSITIVO",
        }
        return mapping.get(cleaned, cleaned)

    def parse_reference_range(self, ref_text: str) -> tuple:
        """
        Parse a reference range string like "(70-110)", "<0.5", ">60", "0-10".
        Returns (low, high) tuple of floats or None.
        """
        from ..extraction.patterns import (
            REF_RANGE_PATTERN, REF_LESS_THAN_PATTERN, REF_GREATER_THAN_PATTERN,
        )

        if not ref_text:
            return (None, None)

        ref_text = ref_text.strip().strip("()[] ")

        # Case 1: Standard range "70-110"
        match = REF_RANGE_PATTERN.search(ref_text)
        if match:
            low = self.normalize_value(match.group("low"))
            high = self.normalize_value(match.group("high"))
            return (low, high)

        # Case 2: Less than or equal: "< 0.5", "<= 1.2"
        match = REF_LESS_THAN_PATTERN.search(ref_text)
        if match:
            high = self.normalize_value(match.group("high"))
            return (None, high)

        # Case 3: Greater than: "> 60"
        match = REF_GREATER_THAN_PATTERN.search(ref_text)
        if match:
            low = self.normalize_value(match.group("low"))
            return (low, None)

        # Case 4: Single number with operator
        import re
        cleaned = re.sub(r'[<>=≤≥]', '', ref_text).strip()
        try:
            val = self.normalize_value(cleaned)
            if any(op in ref_text for op in ['<', '≤']):
                return (None, val)
            elif any(op in ref_text for op in ['>', '≥']):
                return (val, None)
        except (ValueError, TypeError):
            pass

        return (None, None)

    def is_abnormal(self, value: float, ref_low, ref_high,
                    operator: str = None) -> tuple:
        """
        Determine if a value is abnormal relative to reference range.
        Returns (is_abnormal: bool, flag: str | None).
        """
        if ref_low is None and ref_high is None:
            return (False, None)

        # Flag based on operator
        if operator:
            op = operator.strip()
            if op == "<" and ref_high is not None:
                return (value >= ref_high, "H" if value >= ref_high else None)
            if op == ">" and ref_low is not None:
                return (value <= ref_low, "L" if value <= ref_low else None)

        # Standard comparison
        if ref_high is not None and value > ref_high:
            return (True, "H")
        if ref_low is not None and value < ref_low:
            return (True, "L")

        return (False, None)

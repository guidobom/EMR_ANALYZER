"""Deterministic lab value parser."""

from __future__ import annotations

import re
from typing import Optional

from .normalizer import LabNormalizer
from .patterns import (
    LAB_TABULAR_PATTERN, LAB_INLINE_PATTERN,
    build_known_param_pattern,
)
from ..models.lab_result import LabValue


_NUMBER = r"[+-]?(?:\d{1,3}(?:[.,]\d{3})+|\d+)(?:[.,]\d+)?"
_RESULT_UNIT = (
    r"(?:"
    r"[x×]\s*10\s*(?:\^|\*\*)?\s*[36]\s*/\s*[µμu]?[lL]"
    r"|(?:mg|g|ng|pg|µg|μg|ug|mmol|µmol|μmol|umol|mEq|U|UI|IU|"
    r"µU|μU|uU|mL|ml|L|fL|fl|pmol|nmol|kU)"
    r"\s*/\s*(?:dL|dl|mL|ml|L|l|µL|μL|uL|min|24h)"
    r"|/[\s]*(?:µL|μL|uL|mm3)"
    r"|%|fL|fl|pg|INR|Ratio|mm/h|mm3|m²"
    r")"
)
_REFERENCE = (
    rf"(?:{_NUMBER}\s*[-–]\s*{_NUMBER}|[<>]=?\s*{_NUMBER})"
)
_PAYLOAD_RE = re.compile(
    rf"^\s*(?P<operator>[<>]=?)?\s*"
    rf"(?P<value>{_NUMBER})\s*"
    rf"(?P<explicit_flag>\*{{1,3}}|!|H|L|↑|↓)?\s*"
    rf"(?P<unit>{_RESULT_UNIT})?\s*"
    rf"(?:\(?\s*(?P<ref_range>{_REFERENCE})\s*\)?)?"
    rf"(?:\s+.*)?$",
    re.IGNORECASE,
)
_NO_COLON_RESULT_RE = re.compile(
    rf"^\s*(?P<param>[A-Za-zÀ-ÿΑ-ωµμ][^\n:]{{1,100}}?)\s+"
    rf"(?P<payload>(?:[<>]=?\s*)?{_NUMBER}.*)$",
    re.IGNORECASE,
)
_PAGE_LINE_RE = re.compile(
    r"(?:---\s*)?(?:PAGINA|\[PAGINA)\s*:?\s*(\d+)",
    re.IGNORECASE,
)


class LabParser:
    """
    Deterministic parser for laboratory values.
    Uses regex patterns + normalization rules — no LLM.
    """

    def __init__(self, normalizer: Optional[LabNormalizer] = None):
        self.normalizer = normalizer or LabNormalizer()

    def parse(
        self,
        text: str,
        tables: list = None,
        patient_id: str = "",
        document_id: str = "",
        sample_date: str | None = None,
        parsing_result=None,
    ) -> list[LabValue]:
        """
        Extract lab values from text and tables.
        Two phases:
        1. Extract from parser tables (DataFrame)
        2. Extract from free text with regex multi-line patterns
        """
        results = []

        # Extract dates from the text
        detected_dates = self._extract_dates(text) if not sample_date else []
        effective_sample_date = (
            sample_date or (detected_dates[0] if detected_dates else None)
        )

        # Phase 1: Extract from structured tables
        if tables:
            results.extend(self._parse_tables(tables, patient_id, document_id))

        # Phase 2: Extract from free text
        text_results = self._parse_text(text, patient_id, document_id)

        # Assign the explicit document/sample date to every source layer.
        if effective_sample_date:
            for lv in results + text_results:
                if not lv.sample_date:
                    lv.sample_date = effective_sample_date

        results.extend(text_results)

        # Phase 3: Deduplicate by parameter+value within same document
        results = self._deduplicate(results)

        if parsing_result and hasattr(parsing_result, "locate_source"):
            for lab_value in results:
                if lab_value.page:
                    continue
                page, _ = parsing_result.locate_source(
                    lab_value.source_text, None
                )
                lab_value.page = page

        return results

    def _extract_dates(self, text: str) -> list[str]:
        """Extract dates from the document text (ISO: YYYY-MM-DD)."""
        from ..utils.date_utils import parse_italian_date
        import re as re_module

        dates = []
        # Look for date patterns in the text
        date_patterns = [
            r'(?:data|del|il)\s+(?:prelievo|esame|analisi|accettazione|referto)?\s*:?\s*(\d{1,2}[/\-.]\d{1,2}[/\-.]\d{2,4})',
            r'(\d{1,2}[/\-.]\d{1,2}[/\-.]\d{2,4})',
        ]
        for pattern in date_patterns:
            for match in re_module.finditer(pattern, text, re_module.IGNORECASE):
                date_str = match.group(1) if match.lastindex else match.group(0)
                iso_date = parse_italian_date(date_str)
                if iso_date and iso_date not in dates:
                    dates.append(iso_date)

        return dates

    def _parse_tables(self, tables: list, patient_id: str,
                      document_id: str) -> list[LabValue]:
        """Extract lab values from pandas DataFrames produced by the PDF parser."""
        results = []
        for df in tables:
            if df is None or df.empty or len(df.columns) < 2:
                continue

            # Try to identify columns by name patterns
            param_col = None
            value_col = None
            unit_col = None
            ref_col = None

            for col in df.columns:
                col_lower = str(col).lower()
                if any(k in col_lower for k in ['parametro', 'esame', 'analita', 'test', 'parameter']):
                    param_col = col
                elif any(k in col_lower for k in ['valore', 'risultato', 'value', 'result']):
                    value_col = col
                elif any(k in col_lower for k in ['unità', 'unita', 'unit', 'u.m.']):
                    unit_col = col
                elif any(k in col_lower for k in ['range', 'riferimento', 'intervallo', 'ref']):
                    ref_col = col

            # Headerless geometric tables are frequently page-layout grids
            # rather than clinical result tables.  The native line parser is
            # safer for those documents.
            if param_col is None or value_col is None:
                continue

            for _, row in df.iterrows():
                param_name = str(row[param_col]).strip()
                value_str = str(row[value_col]).strip()
                unit = str(row[unit_col]).strip() if unit_col and unit_col in df.columns else ""
                ref_text = str(row[ref_col]).strip() if ref_col and ref_col in df.columns else ""

                if (
                    not param_name
                    or not value_str
                    or param_name.lower() == "nan"
                    or value_str.lower() == "nan"
                ):
                    continue

                unit_norm = self.normalizer.normalize_unit(unit)
                if not self._is_plausible_parameter(
                    param_name, unit_norm, ref_text
                ):
                    continue
                try:
                    value = self.normalizer.normalize_value(
                        value_str, unit_norm
                    )
                except (ValueError, TypeError):
                    continue

                normalized_name = self.normalizer.normalize_parameter(
                    param_name, unit_norm
                )
                ref_low, ref_high = self.normalizer.parse_reference_range(ref_text)
                is_abnormal, flag = self._abnormal_status(
                    value, ref_low, ref_high, ref_text
                )

                results.append(LabValue(
                    patient_id=patient_id,
                    document_id=document_id,
                    parameter_name=param_name,
                    normalized_name=normalized_name,
                    value=value,
                    unit=unit_norm,
                    reference_low=ref_low,
                    reference_high=ref_high,
                    reference_text=ref_text,
                    is_abnormal=is_abnormal,
                    flag=flag,
                    source_text=f"{param_name} {value_str} {unit}".strip(),
                    confidence=0.9,  # Tables have higher confidence
                ))

        return results

    def _parse_text(self, text: str, patient_id: str,
                    document_id: str) -> list[LabValue]:
        """Extract lab values from free text and markdown tables."""
        results = []

        # Phase A: Parse markdown tables (pipe-separated)
        results.extend(self._parse_markdown_tables(text, patient_id, document_id))

        # Phase B: Parse one native result line at a time.  This preserves the
        # association value → flag → unit → reference interval and prevents a
        # regex from crossing into headers, signatures or adjacent analytes.
        current_page = None
        for raw_line in text.splitlines():
            page_match = _PAGE_LINE_RE.search(raw_line)
            if page_match:
                current_page = int(page_match.group(1))
                continue
            if "|" in raw_line:
                continue
            lab_value = self._parse_result_line(
                raw_line,
                patient_id,
                document_id,
                current_page,
            )
            if lab_value:
                results.append(lab_value)

        return results

    def _parse_result_line(
        self,
        raw_line: str,
        patient_id: str,
        document_id: str,
        page: int | None = None,
    ) -> LabValue | None:
        source_line = str(raw_line or "").strip()
        if not source_line:
            return None

        line = re.sub(r"^\[\d+\]\s*", "", source_line).strip()
        if ":" in line:
            param_name, payload = line.split(":", 1)
        else:
            match = _NO_COLON_RESULT_RE.match(line)
            if not match:
                return None
            param_name = match.group("param")
            payload = match.group("payload")

        param_name = re.sub(r"\s+", " ", param_name).strip(" -")
        payload_match = _PAYLOAD_RE.match(payload)
        if not payload_match:
            return None

        unit_raw = (payload_match.group("unit") or "").strip()
        unit = self.normalizer.normalize_unit(unit_raw) if unit_raw else ""
        reference_text = (
            payload_match.group("ref_range") or ""
        ).strip()
        if not self._is_plausible_parameter(
            param_name, unit, reference_text
        ):
            return None

        value_text = payload_match.group("value")
        try:
            value = self.normalizer.normalize_value(value_text, unit)
        except (TypeError, ValueError):
            return None

        reference_low, reference_high = (
            self.normalizer.parse_reference_range(reference_text)
        )
        explicit_flag = (
            payload_match.group("explicit_flag") or ""
        ).strip()
        is_abnormal, flag = self._abnormal_status(
            value,
            reference_low,
            reference_high,
            reference_text,
            explicit_flag,
        )

        return LabValue(
            patient_id=patient_id,
            document_id=document_id,
            parameter_name=param_name,
            normalized_name=self.normalizer.normalize_parameter(
                param_name, unit
            ),
            value=value,
            operator=(
                (payload_match.group("operator") or "").strip() or None
            ),
            unit=unit or None,
            reference_low=reference_low,
            reference_high=reference_high,
            reference_text=reference_text,
            is_abnormal=is_abnormal,
            flag=flag,
            page=page,
            source_text=source_line,
            confidence=0.98 if reference_text else 0.92,
        )

    @staticmethod
    def _is_plausible_parameter(
        parameter_name: str,
        unit: str = "",
        reference_text: str = "",
    ) -> bool:
        """Reject administrative numbers while accepting real analytes."""
        parameter = re.sub(
            r"^\[\d+\]\s*", "", str(parameter_name or "")
        ).strip()
        if not parameter or len(parameter) > 100:
            return False
        if not re.search(r"[A-Za-zÀ-ÿΑ-ω]", parameter):
            return False

        lowered = parameter.casefold()
        excluded = (
            "referto", "richiesta", "pagina", "pag.", "data nascita",
            "data di nascita", "età", "telefono", "cellulare", "fax",
            "codice fiscale", "materiale", "risultati validati",
            "dott.", "dott.ssa", "ai sensi", "ogni informazione",
            "centro prelievi", "laboratorio", "legenda", "valido dal",
            "lod", "limite di rilevabilità", "direttore",
        )
        if any(term in lowered for term in excluded):
            return False

        known_tokens = (
            "globuli", "leucocit", "eritrocit", "emoglobina", "hgb",
            "hct", "mcv", "mch", "mchc", "rdw", "plt", "piastrin",
            "eritroblast", "neutrofil", "linfocit", "monocit",
            "eosinofil", "basofil", "glucos", "glicem", "urea",
            "azotem", "creatinin", "egfr", "filtr", "bilirubin",
            "sodio", "potassio", "cloro", "calcio", "fosforo",
            "magnesio", "ferro", "ferritin", "transferrin",
            "proteine", "albumin", "globuline", "ast", "alt", "got",
            "gpt", "ggt", "ldh", "cpk", " ck", "amilasi", "lipasi",
            "isoamilasi", "pcr", "procalciton", "pct", "troponin",
            "pt ", "pt(", "inr", "aptt", "ptt", "fibrinogeno",
            "d-dimero", "tsh", "ft3", "ft4", "hba1c", "vitamina",
            "folati", "cortisolo", "acth", "prolattina", "estradiolo",
            "testosterone", "paratormone", "immunoglobulin", "iga",
            "igg", "igm", "ige", "hbs", "hbc", "hbe", "hcv", "hiv",
            "cea", "afp", "psa", "ca 125", "ca125", "ca 19",
        )
        has_known_name = any(token in lowered for token in known_tokens)

        normalized_unit = str(unit or "").replace("µ", "μ").casefold()
        recognized_units = {
            "mg/dl", "g/dl", "ng/ml", "ng/l", "pg/ml", "μg/ml",
            "μg/dl", "u/l", "mmol/l", "μmol/l", "meq/l", "fl",
            "pg", "%", "mm/h", "/μl", "×10³/μl", "×10⁶/μl",
            "ml/min", "μu/ml", "pmol/l", "nmol/l", "inr", "ratio",
        }
        has_recognized_unit = normalized_unit in recognized_units
        return bool(
            has_known_name
            or has_recognized_unit
            or (reference_text and has_known_name)
        )

    def _abnormal_status(
        self,
        value: float,
        reference_low,
        reference_high,
        reference_text: str = "",
        explicit_flag: str = "",
    ) -> tuple[bool, str | None]:
        """Combine calculated ranges with the report's explicit marker."""
        is_abnormal, flag = self.normalizer.is_abnormal(
            value, reference_low, reference_high
        )
        compact_reference = re.sub(r"\s+", "", reference_text or "")
        if compact_reference.startswith("<=") and reference_high is not None:
            is_abnormal = value > reference_high
            flag = "H" if is_abnormal else None
        elif compact_reference.startswith("<") and reference_high is not None:
            is_abnormal = value >= reference_high
            flag = "H" if is_abnormal else None
        elif compact_reference.startswith(">=") and reference_low is not None:
            is_abnormal = value < reference_low
            flag = "L" if is_abnormal else None
        elif compact_reference.startswith(">") and reference_low is not None:
            is_abnormal = value <= reference_low
            flag = "L" if is_abnormal else None

        marker = str(explicit_flag or "").strip().upper()
        if marker in {"H", "↑"}:
            return True, "H"
        if marker in {"L", "↓"}:
            return True, "L"
        if marker in {"*", "**", "***", "!"}:
            return True, flag or "*"
        return is_abnormal, flag

    def _parse_markdown_tables(self, text: str, patient_id: str,
                               document_id: str) -> list[LabValue]:
        """
        Parse markdown tables from parser output.
        Format:
        | Esame | Esito | (flag) | U.M. | Intervalli Riferimento |
        | GLOBULI BIANCHI : | 9.06 | | x10^3/µl | 4.00 - 11.00 |
        | GLOBULI ROSSI : | 4.27 | * | x10^6/µl | 4.50 - 6.50 |
        | EOSINOFILI: | | 0.04 | x10^3/µl | 0.04 - 0.40 |
        """
        import re as re_m
        results = []

        # Find all markdown table blocks
        table_blocks = re_m.findall(
            r'(\|.+\|(?:\s*\n\s*\|.+\|)+)',
            text
        )

        for block in table_blocks:
            lines = block.strip().split('\n')
            # Skip the header and separator lines
            data_lines = []
            for line in lines:
                stripped = line.strip()
                # Skip separator lines like |---|----|
                if re_m.match(r'^\|[\s\-:]+\|', stripped):
                    continue
                # Split into cells, KEEP empty cells (important for flag column)
                raw_cells = [c.strip() for c in stripped.split('|')]
                # Remove first and last (always empty from leading/trailing |)
                if raw_cells and raw_cells[0] == '':
                    raw_cells = raw_cells[1:]
                if raw_cells and raw_cells[-1] == '':
                    raw_cells = raw_cells[:-1]
                if not raw_cells:
                    continue
                # Skip pure header rows (no numeric content at all)
                has_numbers = any(
                    re_m.search(r'\d', c) for c in raw_cells
                )
                if not has_numbers and len(data_lines) == 0:
                    continue  # Skip header row
                data_lines.append(raw_cells)

            # Process each data line
            for cells in data_lines:
                lv = self._parse_table_row(cells, patient_id, document_id)
                if lv:
                    results.append(lv)

        return results

    def _parse_table_row(self, cells: list[str], patient_id: str,
                         document_id: str) -> LabValue | None:
        """
        Parse a single row from a markdown table.
        Columns (5): [Esame, Esito, Flag, U.M., Intervalli]
        Empty cells are preserved (important for flag column).
        """
        import re as re_m

        if len(cells) < 2:
            return None

        # Extract parameter name (first cell)
        param_cell = cells[0].strip()
        # Remove leading markers like "[0]", "[1]" etc.
        param_cell = re_m.sub(r'^\[\d+\]\s*', '', param_cell)
        # Remove trailing colon
        param_cell = param_cell.rstrip(':').strip()

        # Skip section header rows (like "[0] EMOCROMO" where rest is empty)
        if all(c.strip() == '' for c in cells[1:]) and not re_m.search(r'\d', param_cell):
            # It's a section header, not a data row
            # But section headers may have [0] prefix and section name
            if re_m.match(r'^\[\d+\]', cells[0]):
                return None
            # If param has no numeric content and other cells are empty, skip
            if not re_m.search(r'[\d.,]', param_cell):
                return None

        if not param_cell or len(param_cell) < 2:
            return None

        # Get cells with defaults
        cell1 = cells[1].strip() if len(cells) > 1 else ""  # Esito (value)
        cell2 = cells[2].strip() if len(cells) > 2 else ""  # Flag or alt value
        cell3 = cells[3].strip() if len(cells) > 3 else ""  # U.M.
        cell4 = cells[4].strip() if len(cells) > 4 else ""  # Intervalli

        value_str = None
        flag_str = None
        unit_str = None
        ref_str = None

        # Case 1: Flag (*, H, L) in cell2 → value in cell1
        if cell2 in ('*', '**', '***', '!', 'H', 'L', '↑', '↓'):
            flag_str = cell2
            value_str = cell1
            unit_str = cell3
            ref_str = cell4

        # Case 2: Flag at end of cell1 ("114 *")
        elif re_m.search(r'\*\s*$', cell1):
            flag_str = '*'
            value_str = re_m.sub(r'\s*\*\s*$', '', cell1)
            unit_str = cell3 if cell3 else cell2
            ref_str = cell4

        # Case 3: Cell1 empty → value is in cell2 (e.g., EOSINOFILI, PT(INR))
        elif not cell1:
            value_str = cell2
            unit_str = cell3
            ref_str = cell4

        # Case 4: Cell1 has number, cell3 empty → unit may be in cell2
        elif cell1 and not cell3 and cell2 and re_m.search(r'\d', cell1):
            value_str = cell1
            unit_str = cell2  # Unit shifted to flag column
            ref_str = cell4

        # Case 5: Normal case — value in cell1, unit in cell3, ref in cell4
        else:
            value_str = cell1
            unit_str = cell3 if cell3 else cell2
            ref_str = cell4

        if not value_str:
            return None

        # Clean up value: remove extra spaces, keep only number
        value_clean = re_m.sub(r'[^\d.,]', '', value_str)
        if not value_clean:
            return None

        unit_norm = self.normalizer.normalize_unit(unit_str) if unit_str else None
        if not self._is_plausible_parameter(
            param_cell, unit_norm or "", ref_str or ""
        ):
            return None
        try:
            value = self.normalizer.normalize_value(
                value_clean, unit_norm or ""
            )
        except (ValueError, TypeError):
            return None

        # Clean up parameter name (remove content in parentheses for display)
        param_display = param_cell

        normalized_name = self.normalizer.normalize_parameter(
            param_cell, unit_norm or ""
        )

        # Parse reference range
        ref_low, ref_high = self.normalizer.parse_reference_range(ref_str)

        # Check abnormal
        is_abnormal, flag = self._abnormal_status(
            value, ref_low, ref_high, ref_str or "", flag_str or ""
        )

        # Build source text
        source = f"{param_cell}: {value_str}"
        if unit_str:
            source += f" {unit_str}"
        if ref_str:
            source += f" ({ref_str})"

        return LabValue(
            patient_id=patient_id,
            document_id=document_id,
            parameter_name=param_display,
            normalized_name=normalized_name,
            value=value,
            unit=unit_norm,
            reference_low=ref_low,
            reference_high=ref_high,
            reference_text=ref_str[:200] if ref_str else "",
            is_abnormal=is_abnormal,
            flag=flag,
            source_text=source,
            confidence=0.85,
        )

    def _match_to_lab_value(self, match, patient_id: str,
                            document_id: str, base_confidence: float) -> Optional[LabValue]:
        """Convert a regex match to a LabValue — with strict filtering."""
        param_name = match.group("param").strip()
        value_str = match.group("value").strip()

        if not param_name or not value_str:
            return None

        # ---- FILTER 1: Reject multi-line parameter names (garbage from non-lab text) ----
        if '\n' in param_name:
            return None

        # ---- FILTER 2: Must look like a real lab parameter ----
        # Accept only if:
        # a) param contains a known lab keyword, OR
        # b) a recognized lab unit is immediately next to the value

        unit = match.groupdict().get("unit", "")
        if unit:
            unit = unit.strip()
        normalized_unit = self.normalizer.normalize_unit(unit) if unit else None

        # Recognized lab units (case-insensitive, normalized)
        RECOGNIZED_UNITS = {
            "mg/dL", "g/dL", "ng/mL", "pg/mL", "μg/mL", "U/L", "UI/L", "IU/L",
            "mEq/L", "mmol/L", "μmol/L", "fL", "pg", "%", "mm/h", "mg/24h",
            "g/24h", "/μL", "×10³/μL", "×10⁶/μL", "mL/min", "L/L", "μU/mL",
            "ng/dL", "μg/dL", "pmol/L", "nmol/L", "U/mL", "kU/L",
        }

        # Known lab parameter keywords (must appear as whole words)
        LAB_PARAM_KEYWORDS = {
            "emoglobina", "hb", "hgb", "ematocrito", "hct", "globuli", "wbc",
            "leucociti", "piastrine", "plt", "neutrofili", "linfociti",
            "monociti", "eosinofili", "basofili", "mcv", "mch", "mchc", "rdw",
            "glucosio", "glicemia", "creatinina", "azotemia", "urea",
            "colesterolo", "trigliceridi", "hdl", "ldl", "proteine",
            "albumina", "bilirubina", "transaminasi", "got", "gpt", "ast", "alt",
            "ggt", "fosfatasi", "amilasi", "lipasi", "ldh", "cpk", "ck",
            "sodio", "potassio", "cloro", "calcio", "magnesio", "ferro",
            "ferritina", "transferrina", "acido_urico", "uricemia",
            "pt", "ptt", "inr", "fibrinogeno", "d-dimero",
            "pcr", "ves", "tsh", "ft3", "ft4", "t3", "t4", "troponina",
            "pro_bnp", "nt-probnp", "ca125", "ca_125", "cea", "afp", "psa",
            "ca19-9", "ca_19.9", "pct", "procalcitonina",
            "iga", "igg", "igm", "ige", "immunoglobulina",
            "proteina_c_reattiva", "velocità", "eritrosedimentazione",
            "emoglobina_glicata", "hba1c", "microalbuminuria",
            "omocisteina", "vitamina", "folati", "b12",
            "cortisolo", "acth", "prolattina", "estradiolo", "progesterone",
            "testosterone", "paratormone", "pth", "calcitonina",
        }

        # Check if parameter name contains any known lab keyword
        param_lower = param_name.lower()
        has_lab_keyword = any(
            kw in param_lower.split() or kw == param_lower
            for kw in LAB_PARAM_KEYWORDS
        )

        # Check if unit is recognized
        has_recognized_unit = normalized_unit in RECOGNIZED_UNITS

        # Must have EITHER a lab keyword OR a recognized unit
        if not has_lab_keyword and not has_recognized_unit:
            return None

        # ---- FILTER 3: Skip common false positives ----
        skip_words = {
            "pagina", "data", "ora", "nome", "cognome", "codice", "cod",
            "reparto", "ospedale", "telefono", "fax", "email", "cell",
            "il", "lo", "la", "i", "gli", "le", "un", "una", "e", "ed",
            "a", "da", "di", "in", "con", "su", "per", "del", "della",
            "al", "alla", "presso", "via", "viale", "piazza", "nato",
            "nata", "sesso", "indirizzo", "città", "cap", "provincia",
            "cf", "tessera", "sanitaria", "esenzione", "medico", "dott",
            "paziente", "referto", "prestazioni", "erogate", "unità",
            "operativa", "accettazione", "programma", "mese", "mesi",
            "anno", "anni", "ore", "giorni", "settimane", "circa",
            "rivalutazione", "controllo", "prossimo", "visita",
            "informazioni", "cliniche", "materiale", "inviato",
            "descrizione", "esame", "eseguito", "data", "firma",
            "tampone", "biopsia", "punch", "pettorale", "cutaneo",
            "medicazione", "ustioni", "rinnovata", "domicilio",
            "attivazione", "percorso", "preso", "visione", "seguito",
            "programma", "restante", "limiti", "troppo", "troppi",
            "prendere", "fatto", "fatta", "viene", "grazie",
            "ulteriori", "proseguire",
        }
        # Filter: if the entire param_name (first word) is a skip word, reject
        first_word = param_lower.split()[0] if param_lower.split() else param_lower
        if first_word in skip_words:
            return None
        # Also reject if param is just a single common word
        if len(param_lower.split()) == 1 and param_lower in skip_words:
            return None

        # ---- FILTER 4: Parameter name should not contain obvious non-medical text ----
        non_medical_patterns = [
            r'\b[A-Z]{3,}\s+[A-Z]{3,}',  # ALL CAPS sequences (names/surnames)
            r'\bfiscale\b', r'\btelefono\b', r'\bindirizzo\b',
            r'\bvia\s+\w+', r'\bpiazza\s+\w+', r'\bn[°.]\s*\d',
            r'\b\d{5,}\b',  # long numbers (phone, codes)
            r'\b[A-Z]{4,}\d{2,}[A-Z]\d{3,}[A-Z]\b',  # Italian fiscal code
        ]
        import re as re_module
        for pattern in non_medical_patterns:
            if re_module.search(pattern, param_name):
                return None

        # ---- FILTER 5: Reject implausible values ----
        if len(param_name) < 2:
            return None

        try:
            value = self.normalizer.normalize_value(value_str)
        except (ValueError, TypeError):
            return None

        if abs(value) > 999999 or abs(value) < 0:
            return None

        # ---- FILTER 6: Value should be in a plausible lab range ----
        # (very permissive, just excludes absurd values)
        if has_recognized_unit and abs(value) > 500000:
            return None

        operator = match.groupdict().get("operator", "").strip() or None
        normalized_name = self.normalizer.normalize_parameter(param_name)

        unit = match.groupdict().get("unit", "")
        if unit:
            unit = unit.strip()
        unit_norm = self.normalizer.normalize_unit(unit) if unit else None

        ref_text = match.groupdict().get("ref_range", "")
        if ref_text:
            ref_text = ref_text.strip()
        ref_low, ref_high = self.normalizer.parse_reference_range(ref_text)
        is_abnormal, flag = self.normalizer.is_abnormal(value, ref_low, ref_high)

        # Confidence: higher if we have units and reference range
        confidence = base_confidence
        if unit_norm:
            confidence += 0.1
        if ref_low is not None or ref_high is not None:
            confidence += 0.05

        return LabValue(
            patient_id=patient_id,
            document_id=document_id,
            parameter_name=param_name,
            normalized_name=normalized_name,
            value=value,
            operator=operator,
            unit=unit_norm,
            reference_low=ref_low,
            reference_high=ref_high,
            reference_text=ref_text,
            is_abnormal=is_abnormal,
            flag=flag,
            source_text=match.group(),
            confidence=min(confidence, 1.0),
        )

    def _deduplicate(self, lab_values: list[LabValue]) -> list[LabValue]:
        """Keep the richest copy of the same result within one document."""
        unique_by_key = {}
        for lv in lab_values:
            key = (lv.normalized_name, lv.value, lv.unit)
            previous = unique_by_key.get(key)
            if previous is None or self._quality_score(lv) > (
                self._quality_score(previous)
            ):
                unique_by_key[key] = lv
        return list(unique_by_key.values())

    @staticmethod
    def _quality_score(lab_value: LabValue) -> float:
        return (
            float(lab_value.confidence)
            + (0.2 if lab_value.unit else 0.0)
            + (
                0.3
                if lab_value.reference_low is not None
                or lab_value.reference_high is not None
                else 0.0
            )
            + (0.1 if lab_value.page else 0.0)
        )

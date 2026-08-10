"""Fast, coordinate-aware patient identity extraction.

Only the first page is inspected.  Native PDF words are preferred; a local
PyMuPDF/Tesseract OCR fallback is attempted for scanned documents.  The
extractor deliberately ignores free clinical prose and accepts values only
when they are spatially anchored to an explicit demographic label.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from pathlib import Path
import re
import unicodedata

from ..models.patient_identity import IdentityField, PatientIdentityEvidence


NAME_LABELS = ("NOME E COGNOME", "COGNOME E NOME")
FISCAL_LABELS = ("CODICE FISCALE",)
BIRTH_LABELS = ("LUOGO E DATA DI NASCITA", "DATA DI NASCITA")
SEX_LABELS = ("SESSO",)


def normalize_text(value: str) -> str:
    """Normalize Italian administrative text for deterministic matching."""

    value = unicodedata.normalize("NFKD", value or "")
    value = "".join(ch for ch in value if not unicodedata.combining(ch))
    value = value.upper().replace("’", "'")
    return re.sub(r"[^A-Z0-9]+", " ", value).strip()


def normalize_name(value: str) -> str:
    return normalize_text(value)


def normalize_fiscal_code(value: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", normalize_text(value))


def fiscal_code_has_valid_checksum(value: str) -> bool:
    """Validate the structure and checksum of an Italian fiscal code."""

    value = normalize_fiscal_code(value)
    if not re.fullmatch(r"[A-Z]{6}\d{2}[A-Z]\d{2}[A-Z]\d{3}[A-Z]", value):
        return False
    odd = {
        "0": 1, "1": 0, "2": 5, "3": 7, "4": 9,
        "5": 13, "6": 15, "7": 17, "8": 19, "9": 21,
        "A": 1, "B": 0, "C": 5, "D": 7, "E": 9,
        "F": 13, "G": 15, "H": 17, "I": 19, "J": 21,
        "K": 2, "L": 4, "M": 18, "N": 20, "O": 11,
        "P": 3, "Q": 6, "R": 8, "S": 12, "T": 14,
        "U": 16, "V": 10, "W": 22, "X": 25, "Y": 24, "Z": 23,
    }
    even = {str(number): number for number in range(10)}
    even.update({chr(ord("A") + number): number for number in range(26)})
    total = sum(
        (odd if position % 2 else even)[character]
        for position, character in enumerate(value[:15], start=1)
    )
    return value[-1] == chr(ord("A") + total % 26)


class PatientIdentityExtractor:
    """Extract patient identifiers from labelled first-page header cells."""

    _NAME_WORD = re.compile(r"^[A-Za-zÀ-ÖØ-öø-ÿ'’\-]{2,}$")
    _DATE = re.compile(r"\b(\d{1,2})[./-](\d{1,2})[./-](\d{4})\b")
    _FORBIDDEN_NAME_WORDS = {
        "NOME", "COGNOME", "LUOGO", "DATA", "NASCITA", "SESSO",
        "CODICE", "FISCALE", "REPARTO", "AMBULATORIO", "UNITA",
        "UNITÀ", "OPERATIVA", "DAY", "SERVICE", "OSPEDALE", "AZIENDA",
        "SANITARIA", "DATI", "ANAGRAFICI", "PRESTAZIONI", "EROGATE",
        "REFERTO", "MEDICO", "DIRETTORE", "LABORATORIO",
    }

    def extract(self, file_path: str | Path) -> PatientIdentityEvidence:
        import fitz

        source_path = str(file_path)
        evidence = PatientIdentityEvidence(source_path=source_path)
        try:
            document = fitz.open(source_path)
        except Exception as exc:
            evidence.warnings.append(f"Documento non leggibile: {exc}")
            return evidence

        try:
            if document.page_count == 0:
                evidence.warnings.append("Documento senza pagine")
                return evidence
            page = document[0]
            words = page.get_text("words", sort=False)
            method = "native_text"
            if len(words) < 5:
                words = self._ocr_words(page)
                method = "ocr" if words else "unavailable"
            evidence.extraction_method = method
            if not words:
                evidence.warnings.append("Testo anagrafico non disponibile")
                return evidence
            rows = self._rows_from_words(words)
            confidence = 0.97 if method == "native_text" else 0.84
            evidence.name = self._extract_name(rows, page.rect.height, method, confidence)
            evidence.fiscal_code = self._extract_fiscal_code(
                rows, page.rect.height, method, confidence
            )
            evidence.birth_date = self._extract_birth_date(
                rows, page.rect.height, method, confidence
            )
            evidence.sex = self._extract_sex(rows, page.rect.height, method, confidence)
            if not evidence.name:
                evidence.warnings.append("Nome non trovato nel campo anagrafico")
            if not evidence.fiscal_code:
                evidence.warnings.append("Codice fiscale valido non trovato")
            if not evidence.birth_date:
                evidence.warnings.append("Data di nascita non trovata")
            return evidence
        finally:
            document.close()

    @staticmethod
    def _ocr_words(page) -> list[tuple]:
        """Attempt fully local OCR when PyMuPDF can access Tesseract."""

        try:
            text_page = page.get_textpage_ocr(
                language="ita+eng", dpi=200, full=True
            )
            return page.get_text("words", textpage=text_page, sort=False)
        except Exception:
            return []

    @staticmethod
    def _rows_from_words(words: list[tuple]) -> list[dict]:
        grouped: dict[tuple[int, int], list[tuple]] = defaultdict(list)
        for word in words:
            grouped[(int(word[5]), int(word[6]))].append(word)
        rows = []
        for key, items in grouped.items():
            items.sort(key=lambda item: item[0])
            rows.append({
                "key": key,
                "x0": min(item[0] for item in items),
                "y0": min(item[1] for item in items),
                "x1": max(item[2] for item in items),
                "y1": max(item[3] for item in items),
                "words": [str(item[4]).strip() for item in items if str(item[4]).strip()],
                "text": " ".join(str(item[4]).strip() for item in items if str(item[4]).strip()),
            })
        rows.sort(key=lambda row: (round(row["y0"], 1), row["x0"]))
        return rows

    @staticmethod
    def _bbox(row: dict) -> tuple[float, float, float, float]:
        return (row["x0"], row["y0"], row["x1"], row["y1"])

    @staticmethod
    def _label_row_indices(rows: list[dict], labels: tuple[str, ...]) -> list[int]:
        indices = []
        for index, row in enumerate(rows):
            row_text = normalize_text(row["text"])
            if any(label in row_text for label in labels):
                indices.append(index)
        return indices

    @staticmethod
    def _near_label_rows(
        rows: list[dict], label_index: int, page_height: float
    ) -> list[dict]:
        """Return candidate value rows ordered by geometric plausibility."""

        label = rows[label_index]
        max_vertical = max(35.0, page_height * 0.045)
        candidates = []
        for index, row in enumerate(rows):
            if index == label_index:
                continue
            vertical = row["y0"] - label["y0"]
            overlap = min(row["x1"], label["x1"]) - max(row["x0"], label["x0"])
            width = max(1.0, min(row["x1"] - row["x0"], label["x1"] - label["x0"]))
            overlap_ratio = overlap / width
            aligned_left = abs(row["x0"] - label["x0"]) <= 6.0
            if -max_vertical <= vertical < 0 and (overlap_ratio >= 0.35 or aligned_left):
                candidates.append((0, abs(vertical), abs(row["x0"] - label["x0"]), row))
            elif abs(vertical) <= 8.0 and row["x0"] >= label["x1"] - 3.0:
                candidates.append((1, abs(row["x0"] - label["x1"]), 0.0, row))
            elif 0 < vertical <= max_vertical and (overlap_ratio >= 0.35 or aligned_left):
                candidates.append((2, abs(vertical), abs(row["x0"] - label["x0"]), row))
        candidates.sort(key=lambda item: item[:3])
        return [item[3] for item in candidates]

    def _extract_name(
        self, rows: list[dict], page_height: float, method: str, confidence: float
    ) -> IdentityField | None:
        for label_index in self._label_row_indices(rows, NAME_LABELS):
            label_row = rows[label_index]
            label_text = label_row["text"]
            for label in NAME_LABELS:
                match = re.search(label.replace(" ", r"\s+"), label_text, re.IGNORECASE)
                if match:
                    for inline in (label_text[match.end():], label_text[:match.start()]):
                        value = self._validated_name(inline)
                        if value:
                            return IdentityField(
                                value=value, normalized=normalize_name(value),
                                bbox=self._bbox(label_row), method=method,
                                confidence=confidence - 0.02,
                            )
            for row in self._near_label_rows(rows, label_index, page_height):
                value = self._validated_name(row["text"])
                if value:
                    return IdentityField(
                        value=value, normalized=normalize_name(value),
                        bbox=self._bbox(row), method=method, confidence=confidence,
                    )
        return None

    def _validated_name(self, value: str) -> str | None:
        value = re.sub(r"\s+", " ", value).strip(" :-|")
        words = value.split()
        if not 2 <= len(words) <= 6:
            return None
        if not all(self._NAME_WORD.fullmatch(word) for word in words):
            return None
        if {normalize_text(word) for word in words} & self._FORBIDDEN_NAME_WORDS:
            return None
        return value

    def _extract_fiscal_code(
        self, rows: list[dict], page_height: float, method: str, confidence: float
    ) -> IdentityField | None:
        for label_index in self._label_row_indices(rows, FISCAL_LABELS):
            search_rows = [rows[label_index]] + self._near_label_rows(
                rows, label_index, page_height
            )
            for row in search_rows:
                for token in re.findall(r"[A-Za-z0-9]{16}", row["text"]):
                    normalized = normalize_fiscal_code(token)
                    if fiscal_code_has_valid_checksum(normalized):
                        return IdentityField(
                            value=token, normalized=normalized,
                            bbox=self._bbox(row), method=method, confidence=confidence,
                        )
        return None

    def _extract_birth_date(
        self, rows: list[dict], page_height: float, method: str, confidence: float
    ) -> IdentityField | None:
        for label_index in self._label_row_indices(rows, BIRTH_LABELS):
            search_rows = [rows[label_index]] + self._near_label_rows(
                rows, label_index, page_height
            )
            for row in search_rows:
                for match in self._DATE.finditer(row["text"]):
                    day, month, year = map(int, match.groups())
                    try:
                        parsed = datetime(year, month, day)
                    except ValueError:
                        continue
                    if not 1850 <= parsed.year <= datetime.now().year:
                        continue
                    return IdentityField(
                        value=match.group(0), normalized=parsed.date().isoformat(),
                        bbox=self._bbox(row), method=method, confidence=confidence,
                    )
        return None

    def _extract_sex(
        self, rows: list[dict], page_height: float, method: str, confidence: float
    ) -> IdentityField | None:
        for label_index in self._label_row_indices(rows, SEX_LABELS):
            search_rows = [rows[label_index]] + self._near_label_rows(
                rows, label_index, page_height
            )
            for row in search_rows:
                tokens = [normalize_text(word) for word in row["words"]]
                for token in tokens:
                    if token in {"M", "F"}:
                        return IdentityField(
                            value=token, normalized=token,
                            bbox=self._bbox(row), method=method,
                            confidence=confidence - 0.03,
                        )
        return None

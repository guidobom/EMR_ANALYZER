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


NAME_LABELS = ("NOME E COGNOME", "COGNOME E NOME", "PAZIENTE", "SIG")
FISCAL_LABELS = ("CODICE FISCALE",)
BIRTH_LABELS = ("LUOGO E DATA DI NASCITA", "DATA DI NASCITA", "DATA NASCITA")
SEX_LABELS = ("SESSO",)
# Ospedale "ID Paziente" header (e.g. ``FE204467``): a strong per-hospital
# identifier that the coordinate extractor must capture and the matcher use.
# The shorter label also covers the Radiologia variant ``Id Paz:``; the value
# may be purely numeric (``8100455504``).
HOSPITAL_ID_LABELS = ("ID PAZ",)

# Single common words that also appear in clinical prose: only header-like
# rows (the label alone, or label + a short inline value) are accepted.
# "SIG" covers the lab-report convention ``Sig. COGNOME NOME`` where no
# dedicated name label exists.
_COMMON_LABELS = frozenset({"PAZIENTE", "SIG"})

_HOSPITAL_ID_RE = re.compile(r"\b([A-Z]{1,4}\d{4,8})\b")
# Purely numeric hospital IDs (``8100455504``) from the "Id Paz:" header.
_NUMERIC_ID_RE = re.compile(r"\b(\d{6,12})\b")


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
    # Some reports print the birth date as "31 12 1950" (space-separated);
    # the separator class accepts both that and the common "./-" form.
    _DATE = re.compile(r"\b(\d{1,2})(?:[./-]| +)(\d{1,2})(?:[./-]| +)(\d{4})\b")
    _FORBIDDEN_NAME_WORDS = {
        "NOME", "COGNOME", "LUOGO", "DATA", "NASCITA", "SESSO",
        "CODICE", "FISCALE", "REPARTO", "AMBULATORIO", "UNITA",
        "UNITÀ", "OPERATIVA", "DAY", "SERVICE", "OSPEDALE", "AZIENDA",
        "SANITARIA", "DATI", "ANAGRAFICI", "PRESTAZIONI", "EROGATE",
        "REFERTO", "MEDICO", "DIRETTORE", "LABORATORIO",
        # Label vocabulary seen on Ferrara USL headers: a label row
        # ("Esame Numero:", "Id Paziente:", "Richiesto da:") must never
        # validate as a name.
        "PAZIENTE", "ESAME", "NUMERO", "DIAGNOSI", "ISTOPATOLOGICA",
        "ENTE", "INDIRIZZO", "PROVENIENZA", "RICHIESTA", "RICHIESTO",
        "QUESITO",
        "CLINICO", "VERSIONE", "PAGINA", "ACC", "NUMBER", "DICOM",
        "PRATICA", "CONSENSO", "DICHIARAZIONE", "RISERVATEZZA",
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
            evidence.hospital_patient_id = self._extract_hospital_id(
                rows, page.rect.height, method, confidence
            )
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
    def _label_row_indices(
        rows: list[dict],
        labels: tuple[str, ...],
        common: tuple[str, ...] = (),
    ) -> list[int]:
        """Row indices carrying a demographic label.

        Labels in *common* (single words that also occur in clinical prose)
        only match header-like rows: the label alone, or the label followed
        by a short inline value.
        """
        common = set(common)
        indices = []
        for index, row in enumerate(rows):
            row_text = normalize_text(row["text"])
            for label in labels:
                if label in common:
                    if not (
                        row_text == label or row_text.startswith(label + " ")
                    ):
                        continue
                    if len(row["words"]) > 8:
                        continue
                    indices.append(index)
                    break
                if label in row_text:
                    indices.append(index)
                    break
        return indices

    @staticmethod
    def _near_label_rows(
        rows: list[dict], label_index: int, page_height: float,
        prefer_same_line: bool = False,
        exclude_above: bool = False,
    ) -> list[dict]:
        """Return candidate value rows ordered by geometric plausibility.

        By default rows *above* the label sort first (some headers put the
        value above the label).  With ``prefer_same_line`` the same-row and
        below candidates are tried before the above ones, which matters when
        the value on the label's own baseline is the authoritative one (a
        birth date) and a higher row only carries another label/date.  With
        ``exclude_above`` rows above the label are skipped entirely: an
        identifier above the demographic labels (e.g. a letterhead phone
        number) is never the value sought.
        """

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
            if (not exclude_above and -max_vertical <= vertical < 0
                    and (overlap_ratio >= 0.35 or aligned_left)):
                candidates.append((0, abs(vertical), abs(row["x0"] - label["x0"]), row))
            elif abs(vertical) <= 8.0 and row["x0"] >= label["x1"] - 3.0:
                candidates.append((1, abs(row["x0"] - label["x1"]), 0.0, row))
            elif 0 < vertical <= max_vertical and (overlap_ratio >= 0.35 or aligned_left):
                candidates.append((2, abs(vertical), abs(row["x0"] - label["x0"]), row))
        if prefer_same_line:
            candidates.sort(key=lambda item: (item[0] == 0, item[1], item[2]))
        else:
            candidates.sort(key=lambda item: item[:3])
        return [item[3] for item in candidates]

    @classmethod
    def _demographic_label_y0s(cls, rows: list[dict]) -> set[float]:
        """Baselines of the birth-date / sex / hospital-ID labels.

        Used to confirm that a ``Sig.`` row is the patient's name inside the
        demographic block rather than an arbitrary title line in the prose.
        """
        y0s = set()
        for labels in (BIRTH_LABELS, SEX_LABELS, HOSPITAL_ID_LABELS):
            for index in cls._label_row_indices(rows, labels):
                y0s.add(rows[index]["y0"])
        return y0s

    def _extract_name(
        self, rows: list[dict], page_height: float, method: str, confidence: float
    ) -> IdentityField | None:
        demographic_y0s: set[float] | None = None
        for label_index in self._label_row_indices(
            rows, NAME_LABELS, common=_COMMON_LABELS
        ):
            label_row = rows[label_index]
            label_text = label_row["text"]
            for label in NAME_LABELS:
                match = re.search(label.replace(" ", r"\s+"), label_text, re.IGNORECASE)
                if not match:
                    continue
                if label == "SIG":
                    if demographic_y0s is None:
                        demographic_y0s = self._demographic_label_y0s(rows)
                    if not any(
                        abs(label_row["y0"] - y) <= max(35.0, page_height * 0.045)
                        for y in demographic_y0s
                    ):
                        continue
                for inline in (label_text[match.end():], label_text[:match.start()]):
                    value = self._validated_name(inline)
                    if value:
                        value = self._merge_continuation(rows, label_row, value)
                        return IdentityField(
                            value=value, normalized=normalize_name(value),
                            bbox=self._bbox(label_row), method=method,
                            confidence=confidence - 0.02,
                        )
            for row in self._near_label_rows(rows, label_index, page_height):
                value = self._merged_name_value(rows, row)
                if value:
                    return IdentityField(
                        value=value, normalized=normalize_name(value),
                        bbox=self._bbox(row), method=method, confidence=confidence,
                    )
        return None

    def _merge_continuation(
        self, rows: list[dict], start_row: dict, base_value: str
    ) -> str:
        """Extend a short name value with a wrapped continuation line.

        Some report headers wrap the patient name onto a second line
        (``COGNOME E NOME: GUERRA VILLIAM`` / ``SILVESTRO``).  A short
        value is extended with the same-column row that directly follows it
        when the combined text is still a valid name; otherwise the short
        name alone is kept.
        """
        if len(base_value.split()) >= 3:
            return base_value
        start_index = next(
            (i for i, row in enumerate(rows) if row is start_row), -1
        )
        if start_index < 0:
            return base_value
        for nxt in rows[start_index + 1:]:
            dy = nxt["y0"] - start_row["y0"]
            if dy > 22.0:
                break
            if abs(nxt["x0"] - start_row["x0"]) > 4.0:
                continue  # different column — another label/value, not a wrap
            merged = (base_value + " " + nxt["text"]).strip()
            merged_value = self._validated_name(merged)
            if merged_value:
                return merged_value
            break
        return base_value

    def _merged_name_value(self, rows: list[dict], start_row: dict) -> str | None:
        """Validate a candidate name row, merging a wrapped continuation line.

        Two layouts fragment the patient name into per-word rows, and either
        must be reconstructed before the extractor can anchor on the name:

        * wrapped onto a second line (``Paziente`` / ``CARAVITA`` on one
          line, ``CRISTIANA`` below);
        * printed on one line but grouped into separate rows by PyMuPDF
          (``Paziente`` / ``CARAVITA`` / ``CRISTIANA`` at the same y).

        A single-word row is only accepted when it extends into a valid
        name; otherwise the extractor falls through to a later row, which
        would otherwise be a nearby label like ``Richiesto da``.
        """
        value = self._validated_name(start_row["text"])
        if value:
            return self._merge_continuation(rows, start_row, value)
        raw = start_row["text"].strip()
        if raw and all(
            self._NAME_WORD.fullmatch(word.rstrip(",")) for word in raw.split()
        ):
            merged = self._merge_continuation(rows, start_row, raw)
            if self._validated_name(merged):
                return merged
            return self._extend_same_line(rows, start_row)
        return None

    def _extend_same_line(self, rows: list[dict], start_row: dict) -> str | None:
        """Rebuild a name from side-by-side words on the same baseline.

        When PyMuPDF returns each word of a single-line header in its own
        row (``CARAVITA`` / ``CRISTIANA`` next to the ``Paziente`` label),
        extend the fragment rightwards one word at a time and keep the first
        combination that validates as a name.  The label row itself is a
        neighbor too, so the merged value is validated (label vocabulary is
        rejected) before being accepted.
        """
        base = start_row["text"].strip()
        if not base:
            return None
        neighbors = [
            row for row in rows
            if row is not start_row
            and abs(row["y0"] - start_row["y0"]) <= 2.0
            and row["x0"] > start_row["x0"]
        ]
        neighbors.sort(key=lambda row: row["x0"])
        merged = base
        for row in neighbors:
            merged = (merged + " " + row["text"]).strip()
            value = self._validated_name(merged)
            if value:
                return value
        return None

    def _validated_name(self, value: str) -> str | None:
        # "." removes the abbreviation mark in "Sig. NOME COGNOME"; the
        # name regex still rejects any word with an internal period.
        value = re.sub(r"\s+", " ", value).strip(" :-|.")
        words = value.split()
        if not 2 <= len(words) <= 6:
            return None
        # "SURNAME, FIRSTNAME" headers carry a comma after the surname.
        if not all(self._NAME_WORD.fullmatch(word.rstrip(",")) for word in words):
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
                rows, label_index, page_height, prefer_same_line=True
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

    def _extract_hospital_id(
        self, rows: list[dict], page_height: float, method: str, confidence: float
    ) -> IdentityField | None:
        for label_index in self._label_row_indices(rows, HOSPITAL_ID_LABELS):
            search_rows = [rows[label_index]] + self._near_label_rows(
                rows, label_index, page_height, exclude_above=True
            )
            for row in search_rows:
                value = self._validated_hospital_id(row["text"])
                if value:
                    return IdentityField(
                        value=value, normalized=value,
                        bbox=self._bbox(row), method=method,
                        confidence=confidence,
                    )
        return None

    def _validated_hospital_id(self, text: str) -> str | None:
        """A hospital patient ID token such as ``FE204467`` or ``8100455504``.

        The rows searched are geometrically anchored to the ``ID PAZ``
        label, so a code like the accession number ``FSA9005070`` is only
        considered when the report actually places it next to the label.
        Numeric-only IDs are accepted too, since the Neuroradiologia system
        emits them without letters.
        """
        for token in _HOSPITAL_ID_RE.findall(text):
            normalized = normalize_text(token)
            if not (6 <= len(normalized) <= 12):
                continue
            if normalized in self._FORBIDDEN_NAME_WORDS:
                continue
            if not (any(c.isalpha() for c in normalized)
                    and any(c.isdigit() for c in normalized)):
                continue
            return normalized
        for token in _NUMERIC_ID_RE.findall(text):
            normalized = normalize_text(token)
            if not (6 <= len(normalized) <= 12):
                continue
            if normalized in self._FORBIDDEN_NAME_WORDS:
                continue
            return normalized
        return None

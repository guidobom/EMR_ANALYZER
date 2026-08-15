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


NAME_LABELS = (
    "NOME E COGNOME", "COGNOME E NOME",
    "NOME COGNOME", "COGNOME NOME",
    "PAZIENTE", "SIG",
)
FISCAL_LABELS = ("CODICE FISCALE",)
BIRTH_LABELS = ("LUOGO E DATA DI NASCITA", "DATA DI NASCITA", "DATA NASCITA")
SEX_LABELS = ("SESSO",)

# Table-style forms (pre-op anaesthesia, surgery logs) split the name across
# two cells: 'Cognome' and 'Nome', each followed by its value on the same
# baseline or in the column below.  The reassembled name must never absorb a
# neighbouring label/value that starts another column, so the same-line scan
# stops at these tokens.
_TABLE_LABEL_GAP = 20.0
_TABLE_VALUE_GAP = 40.0
_TABLE_STOP_WORDS = frozenset({
    "SESSO", "DATA", "NASCITA", "ETA", "BMI", "PESO", "ALTEZZA",
    "CODICE", "FISCALE", "LUOGO", "INDIRIZZO", "TELEFONO", "REPARTO",
    "DIAGNOSI", "INTERVENTO", "ANESTESISTA", "SPECIALITA",
    "NOME", "COGNOME", "E",
})

# Sex values accepted after the SESSO label.  Hospital headers spell the
# value out ("Femmina"/"Maschio") as often as they use the single letter;
# both are mapped to the canonical M/F used for workspace naming and routing.
_SEX_TOKEN_MAP = {
    "M": "M",
    "MASCHIO": "M",
    "MASCHILE": "M",
    "F": "F",
    "FEMMINA": "F",
    "FEMMINILE": "F",
}
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
        # Contaminants observed on real headers: the registered-mail marker
        # "RA", a document-type line ("TIPO DOCUMENTO"), and the "CF"
        # abbreviation printed next to the fiscal code.  None of these is
        # ever part of a patient name.
        "RA", "TIPO", "DOCUMENTO", "CF",
    }

    # Tokens that open an address/locality line.  A name continuation never
    # extends past them (``VIA X``, ``PIAZZA X``, ``STR Y``, a bare "CF"
    # next to the fiscal code), so the extractor stops the name there
    # instead of absorbing the address into the surname.
    _ADDRESS_STOP_WORDS = frozenset({
        "VIA", "PIAZZA", "P.ZA", "CORSO", "VIALE", "VLE", "STR", "LARGO",
        "BORGO", "CF",
    })

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
                # Per-word coordinates let the table-name extractor split a
                # baseline into cells ('Cognome' vs the value 'RIZZI').
                "word_data": [
                    (item[0], item[1], item[2], item[3], str(item[4]).strip())
                    for item in items
                ],
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
        # Table-style forms split the name across 'Cognome'/'Nome' column
        # cells.  Their values are anchored to the labels themselves, so this
        # path is tried before the prose-scanning full-name labels: a
        # reassembled table name can never fall back to a prose sentence.
        table_name = self._extract_table_name(rows, page_height)
        if table_name is not None:
            value, bbox = table_name
            return IdentityField(
                value=value, normalized=normalize_name(value),
                bbox=bbox, method=method, confidence=confidence - 0.02,
            )

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
            if self._starts_with_address_stop(nxt["text"]):
                break  # "VIA ..."/"CF": the name ends before the address
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

    @staticmethod
    def _starts_with_address_stop(text: str) -> bool:
        """True when a row opens with an address/locality marker.

        Used to stop name continuation at ``VIA ...`` / ``CF`` rows, which
        would otherwise be absorbed into the surname (``VITALI REMO
        COPPARO``) and make the name discordant across the same person's
        documents.
        """
        words = text.split()
        if not words:
            return False
        return normalize_text(words[0]) in PatientIdentityExtractor._ADDRESS_STOP_WORDS

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
            if self._starts_with_address_stop(row["text"]):
                break  # "VIA ..."/"CF": the name ends before the address
            merged = (merged + " " + row["text"]).strip()
            value = self._validated_name(merged)
            if value:
                return value
        return None

    def _validated_name(self, value: str) -> str | None:
        # "." removes the abbreviation mark in "Sig. NOME COGNOME"; the
        # name regex still rejects any word with an internal period.
        value = re.sub(r"\s+", " ", value).strip(" :-|.")
        # "MARIO - RIZZI" (value printed before the label, dash-separated)
        # carries a separator that is not part of the name.
        value = re.sub(r"\s+[-–]\s+", " ", value).strip(" :-|.")
        words = value.split()
        if not 2 <= len(words) <= 6:
            return None
        # "SURNAME, FIRSTNAME" headers carry a comma after the surname.
        if not all(self._NAME_WORD.fullmatch(word.rstrip(",")) for word in words):
            return None
        if {normalize_text(word) for word in words} & self._FORBIDDEN_NAME_WORDS:
            return None
        return value

    # --- table-style 'Cognome'/'Nome' column headers ----------------------

    def _extract_table_name(
        self, rows: list[dict], page_height: float
    ) -> tuple[str, tuple[float, float, float, float]] | None:
        """Reassemble a name from table-style ``Cognome``/``Nome`` cells.

        Some forms (pre-op anaesthesia, surgery logs) split the demographic
        header into two column cells, ``Cognome`` and ``Nome``, each with its
        value: ``Cognome ... RIZZI`` on one baseline and ``Nome ... MARIO``
        on the next (or below in the same column).  The full-name labels
        (``NOME E COGNOME``, ``COGNOME NOME``) never occur there, so the name
        must be rebuilt from the two column values.  Returns ``(name, bbox)``
        or ``None`` when the labels or a valid combination are absent.
        """
        surname_value = None
        surname_bbox = None
        firstname_value = None
        firstname_bbox = None
        for row in rows:
            surname_label, firstname_label = self._table_name_signal(
                row, rows
            )
            if surname_label and surname_value is None:
                value = self._name_value_for_label(
                    rows, surname_label, page_height
                )
                if value and self._plausible_name_value(value):
                    surname_value = value
                    surname_bbox = self._bbox(row)
            if firstname_label and firstname_value is None:
                value = self._name_value_for_label(
                    rows, firstname_label, page_height
                )
                if value and self._plausible_name_value(value):
                    firstname_value = value
                    firstname_bbox = self._bbox(row)
            if surname_value and firstname_value:
                break
        if not (surname_value and firstname_value):
            return None
        name = f"{surname_value} {firstname_value}".strip()
        if self._validated_name(name) is None:
            return None
        return name, surname_bbox or firstname_bbox

    def _table_name_signal(
        self, row: dict, rows: list[dict]
    ) -> tuple[tuple | None, tuple | None]:
        """Standalone ``COGNOME``/``NOME`` label words in a table row.

        The words must not belong to a full-name phrase (``COGNOME E NOME``,
        ``NOME COGNOME``) — those are handled by the standard name labels.
        PyMuPDF splits a printed line into several rows, so phrase membership
        is decided on the real baselines across all rows.
        """
        surname = None
        firstname = None
        for word in row.get("word_data", []):
            normalized = normalize_text(word[4])
            if normalized not in ("COGNOME", "NOME"):
                continue
            if self._in_name_phrase(word, rows):
                continue
            if normalized == "COGNOME" and surname is None:
                surname = word
            elif normalized == "NOME" and firstname is None:
                firstname = word
        return surname, firstname

    def _in_name_phrase(self, word, rows: list[dict]) -> bool:
        """True when ``word`` belongs to ``NOME E COGNOME``/``COGNOME NOME``.

        The neighbouring word on the same baseline decides: column headers
        ``Cognome``/``Nome`` are separated by a wide gap, while the phrase
        words sit a few points apart (``NOME E COGNOME``).
        """
        normalized = normalize_text(word[4])
        same_baseline = sorted(
            (
                candidate for candidate in self._all_words(rows)
                if candidate is not word
                and abs(candidate[1] - word[1]) <= 3.0
            ),
            key=lambda candidate: candidate[0],
        )
        before = [
            candidate for candidate in same_baseline
            if candidate[0] < word[2] - 2.0
        ]
        after = [
            candidate for candidate in same_baseline
            if candidate[0] > word[0] + 2.0
        ]
        previous = before[-1] if before else None
        following = after[0] if after else None
        previous_gap = (word[0] - previous[2]) if previous else float("inf")
        following_gap = (following[0] - word[2]) if following else float("inf")

        if normalized == "COGNOME":
            if (previous is not None
                    and previous_gap <= _TABLE_LABEL_GAP
                    and normalize_text(previous[4]) in ("NOME", "E")):
                return True
            if (following is not None
                    and following_gap <= _TABLE_LABEL_GAP
                    and normalize_text(following[4]) in ("NOME", "E")):
                return True
        else:  # NOME
            if (following is not None
                    and following_gap <= _TABLE_LABEL_GAP
                    and normalize_text(following[4]) in ("E", "COGNOME")):
                return True
            if (previous is not None
                    and previous_gap <= _TABLE_LABEL_GAP
                    and normalize_text(previous[4]) == "COGNOME"):
                return True
        return False

    def _name_value_for_label(
        self, rows: list[dict], label_word, page_height: float
    ) -> str | None:
        """The value paired with a table label: same baseline, else below."""
        same_line = self._same_row_value(rows, label_word)
        if same_line:
            return same_line
        return self._below_column_value(rows, label_word, page_height)

    @staticmethod
    def _all_words(rows: list[dict]) -> list[tuple]:
        return [
            word for row in rows for word in row.get("word_data", [])
        ]

    def _same_row_value(self, rows: list[dict], label_word) -> str | None:
        """Value words on the label's baseline, right of it, until a gap.

        PyMuPDF often splits a single printed line into several ``(block,
        line)`` rows, so the comparison uses the real word coordinates across
        every row: the words sharing the label's baseline are collected until
        a column gap or the next column label.
        """
        label_x1 = label_word[2]
        collected = []
        previous_end = label_x1
        for word in sorted(self._all_words(rows), key=lambda item: item[0]):
            if word[0] < label_x1 - 2.0:
                continue  # same column as the label or to its left
            if abs(word[1] - label_word[1]) > 3.0:
                continue  # different baseline
            if word[0] - previous_end > _TABLE_VALUE_GAP:
                break
            if normalize_text(word[4]) in _TABLE_STOP_WORDS:
                break
            collected.append(word[4])
            previous_end = word[2]
        if not collected:
            return None
        return " ".join(collected).strip()

    def _below_column_value(
        self, rows: list[dict], label_word, page_height: float
    ) -> str | None:
        """Word on a following line horizontally aligned with the label.

        Column-table forms print the value on the next line in the same
        column (``Nome`` header above ``MARIO``), so the label's x position
        selects the value word.
        """
        max_vertical = max(35.0, page_height * 0.045)
        candidates = []
        for word in self._all_words(rows):
            vertical = word[1] - label_word[1]
            if not 0 < vertical <= max_vertical:
                continue
            if abs(word[0] - label_word[0]) <= 6.0 and normalize_text(word[4]):
                candidates.append((word[1], word[0], word[4]))
        if not candidates:
            return None
        candidates.sort(key=lambda item: item[:2])
        return candidates[0][2]

    def _plausible_name_value(self, value: str) -> bool:
        """A value that may be part of a name (alphabetic, not a label)."""
        tokens = value.split()
        if not tokens:
            return False
        if any(normalize_text(token) in self._FORBIDDEN_NAME_WORDS for token in tokens):
            return False
        return all(self._NAME_WORD.fullmatch(token.rstrip(",")) for token in tokens)

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
                label_inline = any(
                    any(label in token for label in SEX_LABELS)
                    for token in tokens
                )
                for token in tokens:
                    mapped = _SEX_TOKEN_MAP.get(token)
                    if mapped is None:
                        continue
                    # "M"/"F" are a single letter that can appear inside a
                    # longer value (e.g. a measure "15 F"); accept them only
                    # when the row is the value alone or the SESSO label is
                    # inline.  The spelled-out forms are unambiguous anywhere.
                    if token in ("M", "F") and len(tokens) != 1 and not label_inline:
                        continue
                    return IdentityField(
                        value=mapped, normalized=mapped,
                        bbox=self._bbox(row), method=method,
                        confidence=confidence - 0.03,
                    )
        return None

    def _extract_hospital_id(
        self, rows: list[dict], page_height: float, method: str, confidence: float
    ) -> IdentityField | None:
        for label_index in self._label_row_indices(rows, HOSPITAL_ID_LABELS):
            label_row = rows[label_index]
            # The row may carry several identifiers ('Id. Dicom: FSA20024
            # Id.Paz FSA38432'): only the value that follows 'ID PAZ' itself
            # is the hospital patient ID.
            value = self._hospital_id_after_label(label_row)
            if value:
                return IdentityField(
                    value=value, normalized=value,
                    bbox=self._bbox(label_row), method=method,
                    confidence=confidence,
                )
            search_rows = [label_row] + self._near_label_rows(
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

    def _hospital_id_after_label(self, row: dict) -> str | None:
        """The identifier that directly follows the 'ID PAZ' token.

        Radiology footers print ``Id. Dicom``, ``Id.Paz`` and the accession
        number on one line; the value adjacent to ``ID PAZ``/``ID PAZIENTE``
        is the hospital patient ID and the others must not be captured.  The
        label may be one token (``Id.Paz``) or two (``Id Paz``).
        """
        words = sorted(row.get("word_data", []), key=lambda item: item[0])
        for index, word in enumerate(words):
            tokens = normalize_text(word[4]).split()
            if "PAZ" not in tokens and "PAZIENTE" not in tokens:
                continue
            for following in words[index + 1:]:
                if not normalize_text(following[4]):
                    continue  # punctuation like ':' is an empty token
                value = self._validated_hospital_id(following[4])
                if value:
                    return value
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

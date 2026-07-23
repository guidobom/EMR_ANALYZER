"""Text cleaner — removes boilerplate and artifacts from clinical text."""

from __future__ import annotations

import re
from collections import Counter
from typing import Optional


class TextCleaner:
    """
    Cleans clinical text by removing:
    - Repeated headers/footers (hospital name, patient demographics)
    - Boilerplate blocks (DATI ANAGRAFICI, PRESTAZIONI EROGATE, etc.)
    - Page numbers, timestamps, disclaimers, signatures
    - Empty lines and whitespace artifacts
    """

    # Boilerplate section headers — remove the entire block from header to next ## or REFERTO
    BOILERPLATE_SECTION_STARTS = [
        r'DATI\s+ANAGRAFICI\s+DEL\s+PAZIENTE',
        r'DATI\s+ANAGRAFICI',
        r'PRESTAZIONI\s+EROGATE',
        r'RICHIEDENTI',
    ]

    # Fields that identify boilerplate key-value pairs (value-then-label in docling)
    BOILERPLATE_LABELS = {
        'nome e cognome', 'cognome e nome', 'luogo e data di nascita',
        'data di nascita', 'indirizzo', 'telefono', 'sesso',
        'codice fiscale', 'codice u.o.', 'data/ora accettazione',
        'ente', 'ospedale', 'reparto/ambulatorio', 'medico',
        'unità operativa', 'unita operativa',
    }

    # Patient data patterns to redact
    PATIENT_DATA_PATTERNS = [
        # Italian fiscal code
        (re.compile(r'\b[A-Z]{6}\d{2}[A-Z]\d{2}[A-Z]\d{3}[A-Z]\b'), '[CF]'),
        # Phone numbers (9-11 digits)
        (re.compile(r'\b\d{9,11}\b'), '[TEL]'),
        # Addresses: "VIA GHINI PRIMO, 27/1 - 44011 ARGENTA ( FE )"
        (re.compile(r'(?i)via\s+\w[\w\s,]*?\d+.*?(?:\d{5}|\(?\s*[A-Z]{2}\s*\)?)?'), '[INDIRIZZO]'),
        # "CITTÀ , GG.MM.AAAA" date/place of birth
        (re.compile(r'\b[A-ZÀ-Ü]{3,}\s*,?\s*\d{1,2}\.\d{1,2}\.\d{4}\b'), '[NATO]'),
        # Standalone city names followed by comma + date
        (re.compile(r'\b[A-ZÀ-Ü]{4,}\s*(?=,\s*\d{1,2}\.\d{1,2}\.\d{4})'), '[CITTÀ]'),
    ]

    # Labels that are orphan (label without value after boilerplate removal)
    ORPHAN_LABELS = {
        'provenienza', 'provenienza:', 'richiedenti', 'ente', 'ospedale',
        'reparto/ambulatorio', 'reparto', 'medico', 'medico:',
        'unità operativa', 'unita operativa', 'day service oncologia',
        'nosologico', 'nosologico:', 'codice fiscale:', 'codice fiscale',
        'nr figlia:', 'nr figlia', 'mmg:', 'mmg', 'diagnosi di ingresso',
        'diagnosi di dimissione', 'data/ora accettazione',
        'copia informatica del referto', 'professore', 'dott.', 'dott.ssa',
        'day service oncologia', 'day service oncologia clinica',
        'amb. oncologia terapie', 'dh oncologico',
    }

    # Department/service headers (always boilerplate)
    DEPARTMENT_PATTERNS = [
        re.compile(r'(?i)^\s*UO\s+[\w\s./]+$'),
        re.compile(r'(?i)^\s*DIP\.\s+[\w\s./]+$'),
        re.compile(r'(?i)^\s*U\.O\.\s+[\w\s./]+$'),
        re.compile(r'(?i)^\s*UNITÀ\s+OPERATIVA\s+[\w\s./]+$'),
        re.compile(r'(?i)^\s*DAY\s+SERVICE\s+[\w\s./()-]+$'),
        re.compile(r'(?i)^\s*AMB\.\s+[\w\s./()-]+$'),
        re.compile(r'(?i)^\s*AMBULATORIO\s+[\w\s./()-]+$'),
    ]

    # Doctor name patterns
    DOCTOR_PATTERNS = [
        re.compile(r'(?i)(?:Prof\.|Prof\.ssa|Dott\.|Dott\.ssa|Dr\.|Dr\.ssa)\s+[A-ZÀ-Ü]{2,}(?:\s+[A-ZÀ-Ü]{2,}){0,3}'),
    ]

    # Patterns to remove completely
    REMOVE_PATTERNS = [
        # =================================================================
        # ANYTHING below matches boilerplate → replaced with empty string
        # =================================================================

        # --- Doctor name with title tag residual: "[MEDICO].ssa LUANA CALABRO'" ---
        (re.compile(r'\[MEDICO\]\s*\.?(?:ssa|Prof\.|Dott\.)?\s*[A-ZÀ-Ü]{2,}(?:\s+[A-ZÀ-Ü]{2,}){0,3}\s*\'?\s*'), ''),
        # --- Doctor names: "Prof. MASSIMO GUIDOBONI", "Dott.ssa Sara Saggini" ---
        (re.compile(r'(?i)\b(?:Prof\.|Prof\.ssa|Dott\.|Dott\.ssa|Dr\.|Dr\.ssa)\s+[A-ZÀ-Ü]{2,}(?:\s+[A-ZÀ-Ü]{2,}){0,3}'), ''),
        # --- Medical role labels (with or without ## prefix) ---
        (re.compile(r'(?i)^\s*(?:##\s*)?(?:Medico\s+in\s+Formazione|Dirigente\s+Medico|Medico\s+Specializzando|Medico\s+Curante|Medico\s+Responsabile|Specializzando|Medico\s+Radiologo|Medico\s+Oncologo)\s*$', re.MULTILINE), ''),
        # --- Birthplace prefix standalone: "PIEVE DI", "PIEVE DI CENTO" ---
        (re.compile(r'(?i)^\s*(?:PIEVE\s+DI\s*\w*|FRACTION\s+OF\s*\w*)\s*$', re.MULTILINE), ''),
        # --- Address residual after [INDIRIZZO] removal: " - 44121 FERRARA (FE)" ---
        (re.compile(r'^\s*[-–]\s*\d{5}\s+[A-ZÀ-Ü\s]+\s*\(?\s*[A-Z]{2}\s*\)?\s*$', re.MULTILINE), ''),
        # --- PROVENIENZA anything (more aggressive) ---
        (re.compile(r'(?i)^\s*PROVENIENZA\s*:?\s*.*$', re.MULTILINE), ''),
        # --- DAY SERVICE / AMB standalone ---
        (re.compile(r'(?i)^\s*(?:DAY\s+)?SERVICE\s+\w+\s*$', re.MULTILINE), ''),
        # --- Underscore signature lines (_____) ---
        (re.compile(r'_{3,}'), ''),
        # --- Name immediately followed by label on SAME line: "GIORGIO MARIA BABBINI NOME E COGNOME" ---
        (re.compile(r'^([A-ZÀ-Ü]{2,}(?:\s+[A-ZÀ-Ü]{2,}){1,4})\s+(?:NOME\s+E\s+COGNOME|COGNOME\s+E\s+NOME|LUOGO\s+E\s+DATA\s+DI\s+NASCITA|DATA\s+DI\s+NASCITA)\s*$', re.MULTILINE), ''),
        # --- Birthplace prefix: "PIEVE DI [NATO] LUOGO E DATA DI NASCITA" ---
        (re.compile(r'(?i)^[A-ZÀ-Ü\s]+\s+\[NATO\]\s+LUOGO\s+E\s+DATA\s+DI\s+NASCITA\s*$', re.MULTILINE), ''),
        # --- City with province: "BERTIANO (RO)", "ARGENTA (FE)" ---
        (re.compile(r'^[A-ZÀ-Ü]{3,}\s*\([A-Z]{2}\)\s*$', re.MULTILINE), ''),
        # --- GDPR / disclaimer fragments ---
        (re.compile(r'(?i)^\s*(?:Documento\s+provvisto\s+di|documento\s+informatico|documento\s+firmato\s+digitalmente|documento\s+riservato).*$', re.MULTILINE), ''),
        (re.compile(r'(?i)(?:ai\s+sensi\s+(?:dell|del)|codice\s+dell.amministrazione\s+digitale|firma\s+digitale|firmato\s+digitalmente).*$', re.MULTILINE), ''),
        # --- Standalone "REFERTO" (label, not content ---
        (re.compile(r'^\s*REFERTO\s*$', re.MULTILINE), ''),
        # --- Email addresses ---
        (re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b'), ''),
        # --- Address lines (various formats) ---
        (re.compile(r'(?i)^\s*(?:Via|Piazza|Viale|Corso|Strada|Località|Contrada)\s+[\w\s,()/-]+?\d+.*$', re.MULTILINE), ''),
        # --- Address with comma and province: "VIA TRAVAGLIO (, MIGLIARINO), - 44027 FISCAGLIA ( FE )" ---
        (re.compile(r'(?i)^\s*(?:Via|Piazza|Viale)\s+[\w\s,()/-]+?\d{5}\s+[\w\s()]+$', re.MULTILINE), ''),
        # --- Page count labels ---
        (re.compile(r'Pag\.:\s*\d+\s*/\s*\d+', re.IGNORECASE), ''),
        # --- Hospital names ---
        (re.compile(r'(?i)^\s*(?:Azienda\s+Ospedaliero[-\s]Universitaria|Azienda\s+Ospedaliera|Arcispedale|Ospedale|Policlinico|Istituto\s+Clinico)\s+[\w\s\'-]+$', re.MULTILINE), ''),
        # --- Regional health service headers ---
        (re.compile(r'(?i)^\s*##\s*SERVIZIO\s+SANITARIO\s+(?:REGIONALE|NAZIONALE)\s+[\w\s\'-]+\s*$', re.MULTILINE), ''),
        (re.compile(r'(?i)^\s*SERVIZIO\s+SANITARIO\s+(?:REGIONALE|NAZIONALE)\s+[\w\s\'-]+\s*$', re.MULTILINE), ''),
        # --- PROVENIENZA + DAY SERVICE combos ---
        (re.compile(r'(?i)^\s*(?:PROVENIENZA|PROVENIENZA:?)\s*(?:DAY\s+SERVICE|AMBULATORIO|REPARTO|DH)?\s*[\w\s./()-]*$', re.MULTILINE), ''),
        # --- Double bracket tag garbage: "[[PAZIENTE]]", "## DATI [[PAZIENTE]]" ---
        (re.compile(r'\[\[?\s*(?:PAZIENTE|NATO|MEDICO|INDIRIZZO|CF|TEL)\s*\]?\]'), ''),
        # --- "## DATI NOME E COGNOME" (residual after tag removal) ---
        (re.compile(r'^\s*##\s*DATI\s*(?:NOME\s+E\s+COGNOME|ANAGRAFICI)?\s*$', re.MULTILINE), ''),
        # --- "NOME E COGNOME" standalone (should have been caught by boilerplate_lines, but extra safety) ---
        (re.compile(r'^\s*(?:NOME\s+E\s+COGNOME|COGNOME\s+E\s+NOME|LUOGO\s+E\s+DATA\s+DI\s+NASCITA)\s*$', re.MULTILINE), ''),
        # --- Redacted tag + timestamp residual: "[NATO] 17:22:48" ---
        (re.compile(r'\[(?:NATO|INDIRIZZO|CF|TEL|PAZIENTE|MEDICO)\]\s*\d{1,2}:\d{2}(?::\d{2})?\s*'), ''),
        # --- Standalone redacted tags on their own line ---
        (re.compile(r'^\s*\[(?:NATO|INDIRIZZO|CF|TEL|PAZIENTE|MEDICO)\]\s*$', re.MULTILINE), ''),
        # --- "Referto del GG/MM/AAAA delle ore HH:MM:SS" (electronic signature) ---
        (re.compile(r'(?i)referto\s+del\s+\d{1,2}[./\-]\d{1,2}[./\-]\d{2,4}\s+delle\s+ore\s+\d{1,2}:\d{2}:\d{2}'), ''),
        # --- Prestazione label lines ---
        (re.compile(r'^\s*(?:VISITA\s+(?:ONCOLOGICA|OCULISTICA|SPECIALISTICA|DERMATOLOGICA|CARDIOLOGICA|ORTOPEDICA|NEUROLOGICA|DI\s+CONTROLLO)\s*(?:DI\s+CONTROLLO)?|MEDICAZIONE\s+USTIONI)\s*$', re.MULTILINE | re.IGNORECASE), ''),
        # --- Standalone "## REFERTO" ---
        (re.compile(r'^\s*##\s*REFERTO\s*$', re.MULTILINE | re.IGNORECASE), ''),
        # --- Any timestamp: "12.09.2023 08:30:02", "21.08.2020  08:36:27" ---
        (re.compile(r'\d{1,2}[./\-]\d{1,2}[./\-]\d{2,4}\s+\d{1,2}:\d{2}(?::\d{2})?'), ''),
        # --- "[INDIRIZZO]/1 - 44011 ARGENTA ( FE )" residual ---
        (re.compile(r'\[INDIRIZZO\]\s*/\d+\s*[-–]\s*\d{5}\s*[A-ZÀ-Ü\s]+\s*\(?\s*[A-Z]{2}\s*\)?\s*'), ''),
        # --- "[INDIRIZZO] - 44121 FERRARA (FE)" (without leading /N) ---
        (re.compile(r'\[INDIRIZZO\]\s*[-–]\s*\d{5}\s*[A-ZÀ-Ü\s]+\s*\(?\s*[A-Z]{2}\s*\)?\s*'), ''),
        # --- Italian fiscal code ---
        (re.compile(r'\b[A-Z]{6}\d{2}[A-Z]\d{2}[A-Z]\d{3}[A-Z]\b'), ''),
        # --- Phone numbers ---
        (re.compile(r'\b\d{9,11}\b'), ''),
        # --- Hospital codes ---
        (re.compile(r'\b[A-Z]{2}-\d{2}\s*\(?\d+\)?\b'), ''),
        (re.compile(r'\b\d{10}\b'), ''),
        # --- "Copia informatica..." footer ---
        (re.compile(r'(?i)copia\s+informatica\s+del\s+referto.*?(?:\n|$)'), ''),
        # --- HTML comment page markers ---
        (re.compile(r'<!--\s*image\s*-->'), ''),
        # --- Department headers ---
        (re.compile(r'(?i)^\s*(?:UO|U\.O\.|DIP\.|UNITÀ\s+OPERATIVA)\s+[\w\s./()-]+$', re.MULTILINE), ''),
        # --- "Arcispedale..." headers ---
        (re.compile(r'(?i)arcispedale\s+s\.?\s*anna.*?(?:\n|$)'), ''),
        # --- "Dipartimento..." headers ---
        (re.compile(r'(?i)dipartimento\s+ad\s+attività\s+integrata.*?(?:\n|$)'), ''),
        # --- Signature lines ---
        (re.compile(r'(?i)(?:firma|firmato)\s*(?:digitalmente|elettronicamente|il\s+medico)?.*?(?:\n|$)'), ''),
        # --- GDPR / privacy ---
        (re.compile(r'(?i)(?:documento\s+riservato|informazioni\s+riservate|riservato\s+agli\s+operatori).*?(?:\n|$)'), ''),
        (re.compile(r'(?i)(?:regolamento\s+(?:ue|europeo)\s+\d+/\d+|gdpr|trattamento\s+dei\s+dati).*?(?:\n|$)'), ''),
        # --- Standalone "##" ---
        (re.compile(r'^\s*##\s*$', re.MULTILINE), ''),
        # --- Empty "## DATI" line after double-bracket removal ---
        (re.compile(r'^\s*##\s*DATI\s*$', re.MULTILINE), ''),
    ]

    # Lines that contain ONLY these words are boilerplate
    SINGLE_LINE_BOILERPLATE = {
        'nome e cognome', 'cognome e nome', 'luogo e data di nascita',
        'data di nascita', 'indirizzo', 'telefono', 'sesso',
        'codice fiscale', 'codice u.o.', 'data/ora accettazione',
        'prestazioni erogate', 'dati anagrafici del paziente',
        'dati anagrafici', 'richiedenti', 'ente', 'ospedale',
        'reparto/ambulatorio', 'medico',
    }

    def clean(self, text: str) -> str:
        """
        Aggressively clean clinical text for better readability.
        Removes boilerplate, repeated headers, demographics, and artifacts.
        """
        if not text:
            return ""

        # Phase 1: Remove known patterns (timestamps, codes, etc.)
        cleaned = text
        for pattern, replacement in self.REMOVE_PATTERNS:
            cleaned = pattern.sub(replacement, cleaned)

        # Phase 2: Remove department headers, doctors, and admin boilerplate FIRST
        cleaned = self._remove_department_headers(cleaned)
        cleaned = self._remove_doctor_names(cleaned)

        # Phase 3: Find and redact patient name (while NOME E COGNOME label is still present)
        cleaned = self._remove_patient_name(cleaned)

        # Phase 4: Remove boilerplate section blocks
        cleaned = self._remove_boilerplate_sections(cleaned)

        # Phase 5: Remove repeated lines (headers/footers)
        cleaned = self._remove_repeated_lines(cleaned)

        # Phase 6: Remove standalone boilerplate labels + their preceding values
        cleaned = self._remove_boilerplate_lines(cleaned)

        # Phase 6b: Remove orphan labels and residual artifacts
        cleaned = self._remove_orphan_labels(cleaned)

        # Phase 7: Run doctor removal again (some may have survived)
        cleaned = self._remove_doctor_names(cleaned)

        # Phase 8: Redact patient-specific data (address, phone, etc.)
        cleaned = self._remove_patient_data(cleaned)

        # Phase 9: FINAL AGGRESSIVE CLEANUP — run after all other phases
        # Remove [NATO]/[INDIRIZZO] residuals (created by Phase 8 redaction)
        cleaned = re.sub(
            r'\[(?:NATO|INDIRIZZO|CF|TEL)\]\s*\d{1,2}:\d{2}(?::\d{2})?\s*',
            '', cleaned
        )
        cleaned = re.sub(
            r'\[INDIRIZZO\]\s*(?:/\d+\s*)?[-–]\s*\d{5}\s*[A-ZÀ-Ü\s]+\s*\(?\s*[A-Z]{2}\s*\)?\s*',
            '', cleaned
        )
        # Standalone orphan tags
        cleaned = re.sub(
            r'^\s*\[(?:NATO|INDIRIZZO|CF|TEL|PAZIENTE|MEDICO)\]\s*$',
            '', cleaned, flags=re.MULTILINE
        )
        # Address residual after tag removal: " - 44121 FERRARA (FE)"
        cleaned = re.sub(
            r'^\s*[-–]\s*\d{5}\s+[A-ZÀ-Ü\s]+\s*\(?\s*[A-Z]{2}\s*\)?\s*$',
            '', cleaned, flags=re.MULTILINE
        )
        # Birthplace residual: "PIEVE DI", "PIEVE DI CENTO"
        cleaned = re.sub(
            r'(?i)^\s*(?:PIEVE\s+DI\s*\w*)\s*$',
            '', cleaned, flags=re.MULTILINE
        )
        # PROVENIENZA anything
        cleaned = re.sub(
            r'(?i)^\s*PROVENIENZA\s*:?\s*.*$',
            '', cleaned, flags=re.MULTILINE
        )
        # Any "SERVICE ONCOLOGIA" / "DAY SERVICE" standalone
        cleaned = re.sub(
            r'(?i)^\s*(?:DAY\s+)?SERVICE\s+\w+\s*$',
            '', cleaned, flags=re.MULTILINE
        )
        # "## DATI NOME E COGNOME" residual
        cleaned = re.sub(
            r'^\s*##\s*DATI\s*(?:NOME\s+E\s+COGNOME|ANAGRAFICI)?\s*$',
            '', cleaned, flags=re.MULTILINE
        )
        # "## Medico ..." residual
        cleaned = re.sub(
            r'(?i)^\s*##\s*(?:Medico|Dirigente|Specializzando)\s+.*$',
            '', cleaned, flags=re.MULTILINE
        )

        # Phase 10: Whitespace cleanup
        cleaned = self._clean_whitespace(cleaned)

        return cleaned.strip()

    def _remove_boilerplate_sections(self, text: str) -> str:
        """
        Remove entire boilerplate blocks (demographics, richiedenti).
        Strategy: when we see a boilerplate section header (## DATI ANAGRAFICI...),
        remove everything until the next ## heading or a blank line followed by ##.
        """
        lines = text.split('\n')
        result = []
        skip_block = False

        for i, line in enumerate(lines):
            stripped = line.strip()
            stripped_lower = stripped.lower()

            # Detect ## BOILERPLATE HEADER
            if skip_block:
                # End block: blank line or a new ## heading
                if not stripped:
                    skip_block = False
                    result.append(line)
                    continue
                if re.match(r'^##\s', stripped):
                    skip_block = False
                    result.append(line)
                    continue
                # Still in block → skip this line
                continue

            # Check if this is a boilerplate section start
            is_boilerplate_header = any(
                re.search(pat, stripped_lower, re.IGNORECASE)
                for pat in self.BOILERPLATE_SECTION_STARTS
            )
            if is_boilerplate_header and re.match(r'^##\s', stripped):
                skip_block = True
                continue

            # Also detect lines that are just boilerplate labels
            if stripped_lower in self.BOILERPLATE_LABELS:
                # Remove this line AND the previous line (which is the value in docling)
                if result and result[-1].strip():
                    result.pop()
                continue

            result.append(line)

        return '\n'.join(result)

    def _remove_department_headers(self, text: str) -> str:
        """Remove department/service header lines."""
        lines = text.split('\n')
        result = []
        for line in lines:
            stripped = line.strip()
            if not stripped:
                result.append(line)
                continue
            # Skip department header lines
            if any(p.match(stripped) for p in self.DEPARTMENT_PATTERNS):
                continue
            # Skip lines starting with ")" (residual from corrupted text)
            if stripped.startswith(')'):
                continue
            # Skip lines that are just a standalone "##" heading
            if stripped == '##':
                continue
            result.append(line)
        return '\n'.join(result)

    def _remove_doctor_names(self, text: str) -> str:
        """Remove doctor names (Prof., Dott., etc.)."""
        result = text
        for pattern in self.DOCTOR_PATTERNS:
            result = pattern.sub('[MEDICO]', result)
        return result

    def _remove_orphan_labels(self, text: str) -> str:
        """Remove orphan labels left after boilerplate section removal."""
        import re as re_m
        lines = text.split('\n')
        result = []
        for line in lines:
            stripped = line.strip()
            stripped_lower = stripped.lower().rstrip(':')
            # Skip orphan labels
            if stripped_lower in self.ORPHAN_LABELS:
                continue
            # Skip lines that are just ")" or other single-char artifacts
            if stripped_lower in (')', '(>)', '>', 'm', 'f'):
                continue
            # Skip lines that are just hospital codes (e.g., "A0229123")
            if re_m.match(r'^[A-Z]\d{6,10}$', stripped, re_m.IGNORECASE):
                continue
            # Skip "Nr figlia:" and "MMG:" lines
            if re_m.match(r'^(?:nr\s+figlia|mmg|medico\s+curante)\b', stripped, re_m.IGNORECASE):
                continue
            result.append(line)
        return '\n'.join(result)

    def _remove_patient_name(self, text: str) -> str:
        """
        Detect the patient's full name from demographics patterns
        and remove ALL occurrences of that name from the document.
        """
        full_name = self._find_patient_name(text)
        if not full_name or len(full_name) < 5:
            return text

        # Remove ALL occurrences of this name (case-insensitive)
        escaped = re.escape(full_name)
        result = re.sub(escaped, '[PAZIENTE]', text, flags=re.IGNORECASE)

        # Also redact surname and first name individually (for robustness)
        parts = full_name.split()
        if len(parts) >= 2:
            for part in parts:
                if len(part) >= 4:
                    result = re.sub(
                        r'\b' + re.escape(part) + r'\b',
                        '[PAZIENTE]', result, flags=re.IGNORECASE
                    )

        return result

    def _find_patient_name(
        self, text: str, *, allow_heuristic: bool = True
    ) -> str | None:
        """Find the patient's name, preferring explicit demographic labels.

        ``allow_heuristic`` is disabled for patient routing: a missed name is
        safer than creating a workspace for a repeated uppercase heading.
        """
        text = text.replace("\xa0", " ").replace("\r\n", "\n")

        # Strategy 0: inspect lines around the explicit label in both orders.
        # Visually identical PDFs may expose table cells as value-before-label
        # or label-before-value depending on their internal object order.
        lines = [re.sub(r"\s+", " ", line).strip() for line in text.split("\n")]
        label_re = re.compile(
            r"\b(?:NOME\s+E\s+COGNOME|COGNOME\s+E\s+NOME)\b",
            re.IGNORECASE,
        )
        for index, line in enumerate(lines):
            label_match = label_re.search(line)
            if not label_match:
                continue

            inline_before = line[:label_match.start()].strip(" :-|")
            inline_after = line[label_match.end():].strip(" :-|")
            for value in (inline_after, inline_before):
                candidate = self._validated_patient_name(value)
                if candidate:
                    return candidate

            nearby = []
            for offset in (-1, 1, -2, 2, -3, 3):
                position = index + offset
                if 0 <= position < len(lines) and lines[position]:
                    nearby.append((position, lines[position]))
                    candidate = self._validated_patient_name(lines[position])
                    if candidate:
                        return candidate

            # Some table encodings place surname and given name in separate
            # adjacent cells/lines. Combine only consecutive one-word values.
            for first, second in ((index + 1, index + 2), (index - 2, index - 1)):
                if not (0 <= first < len(lines) and 0 <= second < len(lines)):
                    continue
                if len(lines[first].split()) == len(lines[second].split()) == 1:
                    candidate = self._validated_patient_name(
                        f"{lines[first]} {lines[second]}"
                    )
                    if candidate:
                        return candidate

        # Strategy D (NOW FIRST): Name in ## DATI ANAGRAFICI demographics block
        demog_match = re.search(
            r'##\s*DATI\s+ANAGRAFICI.*?\n',
            text, re.IGNORECASE
        )
        if demog_match:
            after_header = text[demog_match.end():]
            for line in after_header.split('\n'):
                stripped = line.strip()
                if not stripped:
                    continue
                words = stripped.split()
                if len(words) >= 2 and all(
                    len(w) >= 3 and w == w.upper() and re.match(r'^[A-ZÀ-Ü]+$', w)
                    for w in words
                ):
                    return stripped
                if len(stripped) < 3 or not re.match(r'^[A-ZÀ-Ü\s]+$', stripped):
                    break

        # Strategy A: "NAME\nNOME E COGNOME" (docling value-then-label)
        match = re.search(
            r'([A-ZÀ-Ü]{2,}(?:\s+[A-ZÀ-Ü]{2,}){1,4})\s*\n\s*(?:NOME\s+E\s+COGNOME|COGNOME\s+E\s+NOME)',
            text, re.IGNORECASE
        )
        if match:
            return match.group(1).strip()

        # Strategy C: "Cognome e nome: NAME" (inline)
        match = re.search(
            r'(?:Cognome\s+e\s+nome)\s*:?\s*([A-ZÀ-Ü]{2,}(?:\s+[A-ZÀ-Ü]{2,}){1,4})',
            text, re.IGNORECASE
        )
        if match:
            return match.group(1).strip()

        if not allow_heuristic:
            return None

        # Strategy E: Find probable person name that repeats 2+ times
        caps_sequences = re.findall(
            r'\b([A-ZÀ-Ü]{3,}(?:\s+[A-ZÀ-Ü]{3,}){1,3})\b',
            text
        )
        if caps_sequences:
            from collections import Counter
            counts = Counter(caps_sequences)
            # Words that indicate this is NOT a person name
            non_person_words = {
                'OSPEDALE', 'OSPEDALIERO', 'AZIENDA', 'UNITA', 'OPERATIVA',
                'COMPLESSA', 'PATOLOGIA', 'CLINICA', 'DAY', 'SERVICE',
                'AMBULATORIO', 'REPARTO', 'DIPARTIMENTO', 'DIREZIONE',
                'DIAGNOSTICA', 'LABORATORIO', 'ANALISI', 'TERAPIA',
                'ONCOLOGIA', 'CHIRURGIA', 'MEDICINA', 'RADIOLOGIA',
                'FERRARA', 'ARGENTA', 'BOLOGNA', 'MILANO', 'ROMA', 'TORINO',
                'REFERTO', 'DATI', 'ANAGRAFICI', 'RICHIEDENTI', 'ACCETTAZIONE',
                'ROUTINE', 'URGENTE', 'LEGENDA', 'ESEGUITE', 'GLOBULI',
                'EMOCROMO', 'BIANCHI', 'ROSSI', 'FRAZIONATA', 'DIRETTA',
                'TOTALE', 'SESSO', 'TELEFONO', 'INDIRIZZO', 'CODICE',
                'FISCALE', 'NOME', 'COGNOME', 'LUOGO', 'NASCITA', 'DATA',
            }
            for name, count in counts.most_common(20):
                if count < 2 or len(name) < 8:
                    continue
                words = set(name.upper().split())
                # Must be exactly 2 words (first + last name typical)
                if len(words) != 2:
                    continue
                # Neither word should be in the non-person list
                if words & non_person_words:
                    continue
                return name.strip()

        return None

    @staticmethod
    def _validated_patient_name(value: str) -> str | None:
        """Validate a label-adjacent name without accepting clinical headers."""

        value = re.sub(r"\s+", " ", value).strip(" :-|")
        if not value or any(character.isdigit() for character in value):
            return None
        words = value.split()
        if not 2 <= len(words) <= 6:
            return None
        word_pattern = re.compile(r"^[A-Za-zÀ-ÖØ-öø-ÿ'’\-]{2,}$")
        if not all(word_pattern.fullmatch(word) for word in words):
            return None
        forbidden = {
            "NOME", "COGNOME", "LUOGO", "DATA", "NASCITA", "SESSO",
            "CODICE", "FISCALE", "REPARTO", "AMBULATORIO", "UNITA",
            "UNITÀ", "OPERATIVA", "DAY", "SERVICE", "OSPEDALE",
            "AZIENDA", "SANITARIA", "DATI", "ANAGRAFICI", "PRESTAZIONI",
            "EROGATE", "REFERTO",
            "MEDICO", "DIRETTORE", "ONCOLOGIA", "CARDIOLOGIA",
            "NEUROLOGIA", "RADIOLOGIA", "LABORATORIO",
        }
        normalized_words = {word.upper().strip("'’-") for word in words}
        if normalized_words & forbidden:
            return None
        return value

    def _remove_patient_data(self, text: str) -> str:
        """Redact patient-specific data patterns (names, addresses, etc.)."""
        result = text
        for pattern, replacement in self.PATIENT_DATA_PATTERNS:
            result = pattern.sub(replacement, result)
        return result

    def _remove_repeated_lines(self, text: str) -> str:
        """
        Remove lines that repeat identically many times (headers/footers).
        A line appearing 3+ times in a document is likely boilerplate.
        """
        lines = text.split('\n')
        # Count non-empty lines
        non_empty = [l.strip() for l in lines if l.strip()]
        line_counts = Counter(non_empty)

        # Lines that appear 3+ times AND are not clinical content
        repeated = {
            line for line, count in line_counts.items()
            if count >= 3 and len(line) > 2
        }

        if not repeated:
            return text

        result = []
        for line in lines:
            stripped = line.strip()
            if stripped in repeated:
                continue
            result.append(line)

        return '\n'.join(result)

    def _remove_boilerplate_lines(self, text: str) -> str:
        """
        Remove boilerplate label lines AND the value lines that precede them.
        Docling outputs key-value pairs as:
          VALUE
          LABEL
        So when we find a boilerplate LABEL, we also remove the line above it.
        """
        lines = text.split('\n')
        # First pass: mark lines to remove
        remove_indices = set()
        for i, line in enumerate(lines):
            stripped = line.strip().lower()
            if stripped in self.BOILERPLATE_LABELS:
                remove_indices.add(i)
                # Also remove the previous line (the value)
                if i > 0 and lines[i - 1].strip():
                    remove_indices.add(i - 1)
            # Single-char sex field
            if stripped in ('m', 'f') and i > 0:
                prev = lines[i - 1].strip().lower()
                if prev == 'sesso':
                    remove_indices.add(i)
                    remove_indices.add(i - 1)

        result = [line for i, line in enumerate(lines) if i not in remove_indices]
        return '\n'.join(result)

    def _clean_whitespace(self, text: str) -> str:
        """Clean up whitespace artifacts."""
        # Remove lines that are just underscores (markdown horizontal rules from docling)
        text = re.sub(r'^_{3,}\s*$', '', text, flags=re.MULTILINE)
        # Remove lines that are just dashes
        text = re.sub(r'^\s*-{3,}\s*$', '', text, flags=re.MULTILINE)

        # Collapse 3+ newlines into 2
        text = re.sub(r'\n{3,}', '\n\n', text)

        # Remove empty lines at start and end
        text = text.strip()

        # Remove trailing whitespace on each line
        text = '\n'.join(line.rstrip() for line in text.split('\n'))

        return text

    def extract_clinical_only(self, text: str) -> str:
        """
        Extract only the clinical content, removing everything else.
        More aggressive than clean().
        """
        # First, full clean
        cleaned = self.clean(text)

        # Try to find the actual clinical content
        # Look for section headers that indicate clinical content
        clinical_start_patterns = [
            r'(?i)^##\s*(?:REFERTO|REPERTO|DESCRIZIONE|DIAGNOSI|CONCLUSIONI|ESAME\s+OBIETTIVO|ANAMNESI|MOTIVO|DECORSO)',
            r'(?i)^(?:REFERTO|REPERTO|DESCRIZIONE|CONCLUSIONI)\s*$',
        ]

        lines = cleaned.split('\n')
        start_idx = 0
        for i, line in enumerate(lines):
            for pat in clinical_start_patterns:
                if re.match(pat, line):
                    start_idx = i
                    break
            if start_idx > 0:
                break

        if start_idx > 0:
            # Keep from the clinical section onwards
            cleaned = '\n'.join(lines[start_idx:])

        return cleaned.strip()

    # ------------------------------------------------------------------
    # Legacy methods preserved for compatibility
    # ------------------------------------------------------------------
    def identify_artifacts(self, text: str) -> list[dict]:
        """Identify artifacts without removing them."""
        artifacts = []
        for pattern, artifact_type in self.REMOVE_PATTERNS:
            for match in pattern.finditer(text):
                artifacts.append({
                    "type": "boilerplate",
                    "text": match.group()[:100],
                    "start": match.start(),
                    "end": match.end(),
                })
        return artifacts

    def find_duplicate_paragraphs(self, text: str,
                                  threshold: float = 0.9) -> list[tuple[int, int]]:
        """Find near-duplicate paragraphs."""
        from difflib import SequenceMatcher
        paragraphs = [p.strip() for p in text.split('\n\n') if len(p.strip()) > 50]
        duplicates = []
        for i in range(len(paragraphs)):
            for j in range(i + 1, len(paragraphs)):
                ratio = SequenceMatcher(None, paragraphs[i], paragraphs[j]).ratio()
                if ratio >= threshold:
                    duplicates.append((i, j))
        return duplicates

    def extract_body_text(self, text: str) -> str:
        """Extract body text (delegates to clean)."""
        return self.clean(text)

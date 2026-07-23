"""Conservative metadata extraction from the first-page report header."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Optional

from ..models.document import DocumentType


@dataclass(frozen=True)
class DocumentHeaderMetadata:
    department: Optional[str] = None
    provenance: Optional[str] = None
    services: tuple[str, ...] = ()
    specialty: Optional[str] = None
    document_type_hint: Optional[str] = None
    confidence: float = 0.0

    def to_dict(self) -> dict:
        value = asdict(self)
        value["services"] = list(self.services)
        return value

    def classification_text(self) -> str:
        return "\n".join(
            value
            for value in (self.department, self.provenance, *self.services)
            if value
        )


class HeaderMetadataExtractor:
    """Extract trusted classification signals before destructive cleaning."""

    _label_patterns = {
        "department": re.compile(
            r"^(?:REPARTO(?:/AMBULATORIO)?|UNIT[ÀA]\s+OPERATIVA|U\.?O\.?)\s*:?\s*(.*)$",
            re.IGNORECASE,
        ),
        "provenance": re.compile(r"^PROVENIENZA\s*:?\s*(.*)$", re.IGNORECASE),
    }
    _unit_line = re.compile(
        r"^(?:U\.?O\.?|UNIT[ÀA]\s+OPERATIVA|DIPARTIMENTO|REPARTO|"
        r"DAY\s+SERVICE|DAY\s+HOSPITAL|DH|AMBULATORIO|AMB\.)\b",
        re.IGNORECASE,
    )
    _services_header = re.compile(r"^PRESTAZIONI\s+EROGATE\s*:?$", re.IGNORECASE)
    _stop_header = re.compile(
        r"^(?:DATI\s+ANAGRAFICI|RICHIEDENTI|REFERTO|PROVENIENZA|"
        r"REPARTO(?:/AMBULATORIO)?|MEDICO|DIAGNOSI|ANAMNESI|QUESITO|"
        r"FIRMA|CONCLUSIONI)\b",
        re.IGNORECASE,
    )
    _service_terms = re.compile(
        r"\b(?:VISITA|CONSULENZA|CONTROLLO|MEDICAZIONE|ECOGRAFIA|"
        r"ELETTROCARDIOGRAMMA|ECG|ECOCARDIOGRAMMA|HOLTER|SPIROMETRIA|"
        r"ENDOSCOPIA|GASTROSCOPIA|COLONSCOPIA|OCT|FLUORANGIOGRAFIA|"
        r"BIOPSIA|PRELIEVO|TERAPIA|TRATTAMENTO|PROCEDURA|INTERVENTO|"
        r"TC|TAC|RMN?|PET|SCINTIGRAFIA|RADIOGRAFIA|RX)\b",
        re.IGNORECASE,
    )
    _specialties = (
        ("oncologia", ("oncolog", "oncoematolog")),
        ("cardiologia", ("cardiolog", "cardiochir")),
        ("endocrinologia", ("endocrinolog", "diabetolog")),
        ("neurologia", ("neurolog", "neurochir")),
        ("dermatologia", ("dermatolog",)),
        ("pneumologia", ("pneumolog",)),
        ("gastroenterologia", ("gastroenterolog", "endoscopia digestiva")),
        ("nefrologia", ("nefrolog", "dialisi")),
        ("reumatologia", ("reumatolog",)),
        ("oculistica", ("oculist", "oftalmolog")),
        ("otorinolaringoiatria", ("otorin", "orl")),
        ("urologia", ("urolog",)),
        ("ginecologia", ("ginecolog", "ostetric")),
        ("ematologia", ("ematolog",)),
        ("malattie_infettive", ("infettiv",)),
        ("ortopedia", ("ortoped",)),
        ("fisiatria", ("fisiatr", "riabilitaz")),
        ("chirurgia", ("chirurg",)),
        ("medicina_nucleare", ("medicina nucleare",)),
        ("radiologia", ("radiolog", "diagnostica per immagini")),
        ("anatomia_patologica", ("anatomia patologica", "anatomopatolog")),
    )

    def __init__(self, *, max_header_lines: int = 160, max_header_chars: int = 12000):
        self.max_header_lines = max_header_lines
        self.max_header_chars = max_header_chars

    def extract(self, raw_markdown: str) -> DocumentHeaderMetadata:
        if not raw_markdown:
            return DocumentHeaderMetadata()
        lines = self._header_lines(raw_markdown)
        department = None
        provenance = None
        services: list[str] = []

        for index, line in enumerate(lines):
            for field, pattern in self._label_patterns.items():
                match = pattern.match(line)
                if not match:
                    continue
                inline = self._clean_value(match.group(1))
                value = inline or self._nearby_value(lines, index)
                if field == "department" and value:
                    department = value
                elif field == "provenance" and value:
                    provenance = value
            if department is None and self._unit_line.match(line):
                department = self._clean_value(line)
            if self._services_header.match(line):
                services.extend(self._services_after(lines, index))

        services = list(dict.fromkeys(value for value in services if value))
        evidence = " ".join(
            value for value in (department, provenance, *services) if value
        )
        specialty = self._detect_specialty(evidence)
        type_hint = self._document_type_hint(services, specialty)
        confidence = 0.0
        if department or provenance:
            confidence += 0.35
        if services:
            confidence += 0.45
        if specialty:
            confidence += 0.15
        if type_hint:
            confidence += 0.05

        return DocumentHeaderMetadata(
            department=department,
            provenance=provenance,
            services=tuple(services),
            specialty=specialty,
            document_type_hint=type_hint,
            confidence=min(confidence, 1.0),
        )

    def _header_lines(self, text: str) -> list[str]:
        result = []
        for raw_line in text[: self.max_header_chars].splitlines():
            if re.match(r"\s*<!--\s*page\s*[=:]?\s*2\b", raw_line, re.IGNORECASE):
                break
            value = self._clean_value(raw_line)
            if value:
                result.append(value)
            if len(result) >= self.max_header_lines:
                break
        return result

    @staticmethod
    def _clean_value(value: str) -> str:
        value = value.strip()
        value = re.sub(r"^#{1,6}\s*", "", value)
        value = value.strip(" |\t")
        value = re.sub(r"\s*\|\s*", " ", value)
        value = re.sub(r"\s+", " ", value).strip(" :-")
        if not value or re.fullmatch(r"[-:|\s]+", value):
            return ""
        return value[:300]

    def _nearby_value(self, lines: list[str], index: int) -> Optional[str]:
        # Several institutional Docling templates emit value before label.
        candidates = []
        if index > 0:
            candidates.append(lines[index - 1])
        if index + 1 < len(lines):
            candidates.append(lines[index + 1])
        for candidate in candidates:
            if self._is_metadata_value(candidate):
                return candidate
        return None

    def _is_metadata_value(self, value: str) -> bool:
        if any(pattern.match(value) for pattern in self._label_patterns.values()):
            return False
        if self._services_header.match(value) or self._stop_header.match(value):
            return False
        if len(value) < 3 or len(value) > 160:
            return False
        return bool(self._unit_line.match(value) or self._detect_specialty(value))

    def _services_after(self, lines: list[str], index: int) -> list[str]:
        services = []
        for value in lines[index + 1:index + 15]:
            if self._services_header.match(value):
                continue
            if self._stop_header.match(value):
                break
            if self._unit_line.match(value) and services:
                break
            if self._service_terms.search(value):
                services.append(value)
            elif services:
                # A non-service line marks the end of this compact header block.
                break
        return services

    def _detect_specialty(self, text: str) -> Optional[str]:
        normalized = text.casefold()
        for specialty, stems in self._specialties:
            if any(stem in normalized for stem in stems):
                return specialty
        return None

    @staticmethod
    def _document_type_hint(
        services: list[str], specialty: Optional[str]
    ) -> Optional[str]:
        text = " ".join(services).casefold()
        if not text:
            return None
        if "visita" in text or "controllo" in text or "consulenza" in text:
            if specialty == "oncologia":
                return DocumentType.VISITA_ONCOLOGICA.value
            return DocumentType.VISITA_SPECIALISTICA.value
        if any(term in text for term in ("tc ", "tac", "rm ", "rmn", "radiografia", " rx")):
            return DocumentType.RADIOLOGIA.value
        if any(term in text for term in ("pet", "scintigrafia")):
            return DocumentType.MEDICINA_NUCLEARE.value
        return None

"""Radiology report section extractor."""

import re
from typing import Optional

from ..models.radiology_report import RadiologyReport
from .patterns import RADIOLOGY_SECTION_PATTERNS


class RadiologyExtractor:
    """
    Extracts structured sections from Italian radiology reports.
    Uses section header patterns for technique, findings, conclusions.
    """

    def extract(self, text: str, document_id: str = "") -> RadiologyReport:
        """Extract radiology sections from clinical text."""
        report = RadiologyReport(
            full_text=text,
            document_id=document_id,
        )

        # Extract technique
        match = RADIOLOGY_SECTION_PATTERNS["technique"].search(text)
        if match:
            report.technique = match.group("text").strip()

        # Extract findings
        match = RADIOLOGY_SECTION_PATTERNS["findings"].search(text)
        if match:
            report.findings = match.group("text").strip()

        # Extract conclusions
        match = RADIOLOGY_SECTION_PATTERNS["conclusions"].search(text)
        if match:
            report.conclusions = match.group("text").strip()

        # Detect exam type
        report.exam_type = self._detect_exam_type(text)

        # Detect body region
        report.body_region = self._detect_body_region(text)

        # Detect contrast usage
        report.contrast_used = self._detect_contrast(text)

        return report

    def _detect_exam_type(self, text: str) -> Optional[str]:
        """Detect the type of radiological exam."""
        text_upper = text.upper()
        exam_patterns = [
            (r'\bTC\b|\bTAC\b|\bTOMOGRAFIA\s+COMPUTERIZZATA\b', 'TC'),
            (r'\bRM\b|\bRMN\b|\bRISONANZA\s+MAGNETICA\b', 'RM'),
            (r'\bRX\b|\bRADIOGRAFIA\b|\bRADIOGRAMMA\b', 'RX'),
            (r'\bECOGRAFIA\b|\bECO\b|\bECOTOMOGRAFIA\b', 'Ecografia'),
            (r'\bMAMMOGRAFIA\b', 'Mammografia'),
            (r'\bANGIOGRAFIA\b|\bANGIO\b', 'Angiografia'),
            (r'\bSCINTIGRAFIA\b', 'Scintigrafia'),
            (r'\bPET\b|\bPET-TC\b|\bPET/TC\b', 'PET'),
        ]
        for pattern, exam_type in exam_patterns:
            if re.search(pattern, text_upper):
                return exam_type
        return None

    def _detect_body_region(self, text: str) -> Optional[str]:
        """Detect the body region examined."""
        regions = [
            (r'\b(?:TORACE|TORACIC[AO]|POLMON[AEIO])\b', 'Torace'),
            (r'\b(?:ADDOME|ADDOMINAL[EI])\b', 'Addome'),
            (r'\b(?:ENCEFALO|CRANIC[AO]|CEREBRAL[EI]|TESTA|CAPO)\b', 'Encefalo'),
            (r'\b(?:RACHIDE|VERTEBRAL[EI]|COLONNA)\b', 'Rachide'),
            (r'\b(?:BACINO|PELVI|PELVIC[AO])\b', 'Bacino'),
            (r'\b(?:ARTO\s+(?:SUPERIORE|INFERIORE)|FEMORE|GINOCCHIO|SPALLA|ANCA|PIEDE|MANO|POLSO|CAVIGLIA|GOMITO)\b', 'Arti'),
            (r'\b(?:CARDIA[CO]|CUORE)\b', 'Cardiaco'),
            (r'\b(?:MAMMELL[AE]|MAMMAR[IO])\b', 'Mammella'),
            (r'\b(?:TIROIDE|TIROIDEO)\b', 'Tiroide'),
        ]
        for pattern, region in regions:
            if re.search(pattern, text, re.IGNORECASE):
                return region
        return None

    def _detect_contrast(self, text: str) -> bool:
        """Detect if contrast medium was used."""
        contrast_patterns = [
            r'(?i)(?:mezzo\s+di\s+contrasto|m\.?d\.?c\.?|contrastografico|'
            r'con\s+contrasto|somministrazione\s+di\s+contrasto|'
            r'contrast\s+enhanced)',
        ]
        return any(re.search(p, text) for p in contrast_patterns)

    def extract_summary(self, text: str) -> str:
        """
        Extract key sentences that summarize the radiology report.
        Focuses on conclusions/impressions section with fallback to findings.
        """
        report = self.extract(text)

        if report.conclusions:
            return report.conclusions
        if report.findings:
            # Return first 300 chars of findings as summary
            return report.findings[:300] + ("..." if len(report.findings) > 300 else "")
        if report.technique:
            return f"Esame: {report.technique[:200]}"
        return text[:300] + ("..." if len(text) > 300 else "")

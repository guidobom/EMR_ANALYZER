"""Clinical segmenter — splits document into semantic sections."""

import re
from typing import Optional

from ..config import CLINICAL_SECTIONS, ADMISSION_SECTIONS
from ..models.document import ClinicalSection


class ClinicalSegmenter:
    """
    Segments a clinical document into semantically coherent sections
    using heading detection patterns for Italian medical reports.
    """

    def segment(self, text: str,
                is_admission: bool = False) -> list[ClinicalSection]:
        """
        Segment text into clinical sections.
        Uses section header patterns from config.
        """
        section_patterns = ADMISSION_SECTIONS if is_admission else CLINICAL_SECTIONS

        # Build a combined regex that matches any section header
        all_patterns = {}
        for section_type, patterns in section_patterns.items():
            for pattern in patterns:
                all_patterns[section_type] = re.compile(
                    pattern, re.IGNORECASE | re.MULTILINE
                )

        # Find all section headers with their positions
        matches = []
        for section_type, compiled_re in all_patterns.items():
            for match in compiled_re.finditer(text):
                matches.append((match.start(), match.end(), section_type, match.group()))

        # Sort by position
        matches.sort(key=lambda m: m[0])

        if not matches:
            # No sections found — return entire text as single section
            return [ClinicalSection(
                section_type="full_text",
                header="",
                text=text.strip(),
                page_start=1,
                page_end=1,
                confidence=0.5,
            )]

        # Build sections
        sections = []
        for i, (start, end, section_type, header) in enumerate(matches):
            # Section text: from end of this header to start of next header (or end of text)
            next_start = matches[i + 1][0] if i + 1 < len(matches) else len(text)
            section_text = text[end:next_start].strip()

            if section_text:
                sections.append(ClinicalSection(
                    section_type=section_type,
                    header=header.strip(),
                    text=section_text,
                    page_start=1,   # Will be refined with Docling page info
                    page_end=1,
                    confidence=0.8,
                ))

        # Add text before first section as "preamble"
        if matches[0][0] > 0:
            preamble = text[:matches[0][0]].strip()
            if preamble:
                sections.insert(0, ClinicalSection(
                    section_type="preamble",
                    header="",
                    text=preamble,
                    page_start=1,
                    page_end=1,
                    confidence=0.6,
                ))

        return sections

    def segment_with_docling_elements(self, docling_elements: list,
                                      is_admission: bool = False) -> list[ClinicalSection]:
        """
        Segment using Docling element types as structural hints.
        Elements with type 'heading' or 'section-header' are treated as
        section boundaries with higher confidence.
        """
        section_patterns = ADMISSION_SECTIONS if is_admission else CLINICAL_SECTIONS
        sections = []

        # Group elements by heading boundaries
        current_section = None
        current_text = []

        for elem in docling_elements:
            if elem.type in ("heading", "section-header", "title"):
                # Save previous section
                if current_section and current_text:
                    current_section.text = "\n".join(current_text)
                    sections.append(current_section)

                # Determine section type
                section_type = self._classify_header(elem.text, section_patterns)
                current_section = ClinicalSection(
                    section_type=section_type,
                    header=elem.text,
                    text="",
                    page_start=elem.page,
                    page_end=elem.page,
                    confidence=0.9 if section_type != "unknown" else 0.6,
                )
                current_text = []
            elif current_section:
                current_text.append(elem.text)

        # Save last section
        if current_section and current_text:
            current_section.text = "\n".join(current_text)
            sections.append(current_section)

        return sections

    def _classify_header(self, header: str,
                         section_patterns: dict) -> str:
        """Classify a section header text into a section type."""
        header_lower = header.lower()
        for section_type, patterns in section_patterns.items():
            for pattern in patterns:
                if re.search(pattern, header_lower, re.IGNORECASE):
                    return section_type
        return "unknown"

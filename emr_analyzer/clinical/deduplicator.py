"""Semantic deduplication for clinical events and documents."""

import re
from typing import Optional


class Deduplicator:
    """
    Identifies duplicate clinical information across documents.
    Handles exact duplicates (hash-based) and near-duplicates (semantic).
    """

    def find_duplicate_events(self, events: list,
                              threshold: float = 0.85) -> list[tuple]:
        """
        Find near-duplicate events.
        Returns pairs of (event1_index, event2_index).
        """
        from difflib import SequenceMatcher

        duplicates = []
        for i in range(len(events)):
            for j in range(i + 1, len(events)):
                e1 = events[i]
                e2 = events[j]

                # Must be same type
                if e1.event_type != e2.event_type:
                    continue

                # Must be same entity (fuzzy)
                entity_ratio = SequenceMatcher(
                    None,
                    e1.entity.lower(),
                    e2.entity.lower()
                ).ratio()

                if entity_ratio < 0.8:
                    continue

                # Must be close in date (within 30 days unless exact)
                date_similar = self._dates_similar(e1.event_date, e2.event_date)

                if entity_ratio >= threshold and date_similar:
                    duplicates.append((i, j))
                elif entity_ratio >= 0.95:  # Very similar entity, allow date mismatch
                    duplicates.append((i, j))

        return duplicates

    def find_duplicate_paragraphs(self, text: str,
                                  threshold: float = 0.85) -> list[tuple]:
        """
        Find near-duplicate paragraphs within a text.
        Useful for detecting copied medical histories.
        """
        from difflib import SequenceMatcher

        paragraphs = [
            p.strip() for p in re.split(r'\n\s*\n', text)
            if len(p.strip()) > 50
        ]

        duplicates = []
        for i in range(len(paragraphs)):
            for j in range(i + 1, len(paragraphs)):
                ratio = SequenceMatcher(
                    None, paragraphs[i], paragraphs[j]
                ).ratio()
                if ratio >= threshold:
                    duplicates.append((i, j))

        return duplicates

    def deduplicate_texts(self, texts: list[str],
                          threshold: float = 0.9) -> list[int]:
        """
        Return indices of texts that are duplicates of earlier texts.
        For use when processing multiple documents for the same patient.
        """
        from difflib import SequenceMatcher
        duplicate_indices = set()

        for i in range(len(texts)):
            for j in range(i):
                ratio = SequenceMatcher(
                    None, texts[i][:1000], texts[j][:1000]
                ).ratio()
                if ratio >= threshold:
                    duplicate_indices.add(i)
                    break

        return sorted(duplicate_indices)

    @staticmethod
    def _dates_similar(date1: str, date2: str,
                       max_days: int = 30) -> bool:
        """Check if two dates are within max_days of each other."""
        if not date1 or not date2:
            return True  # Can't compare, assume similar

        try:
            from datetime import datetime, timedelta
            d1 = datetime.fromisoformat(date1)
            d2 = datetime.fromisoformat(date2)
            return abs((d1 - d2).days) <= max_days
        except (ValueError, TypeError):
            return True  # Can't parse, assume similar

    def find_repeated_anamnesis(self, documents: list[str]) -> list[dict]:
        """
        Identify repeated anamnesis (patient history) across documents.
        Returns groups of documents sharing the same anamnesis.
        """
        from difflib import SequenceMatcher

        # Extract anamnesis sections from each document
        anamnesis_texts = []
        for text in documents:
            anam_text = self._extract_anamnesis(text)
            anamnesis_texts.append(anam_text)

        # Group by similarity
        groups = []
        assigned = set()

        for i in range(len(anamnesis_texts)):
            if i in assigned:
                continue
            group = [i]
            for j in range(i + 1, len(anamnesis_texts)):
                if j in assigned:
                    continue
                ratio = SequenceMatcher(
                    None, anamnesis_texts[i], anamnesis_texts[j]
                ).ratio()
                if ratio >= 0.85:
                    group.append(j)
                    assigned.add(j)
            if len(group) > 1:
                groups.append({"source": i, "copies": group[1:]})

        return groups

    @staticmethod
    def _extract_anamnesis(text: str) -> str:
        """Extract the anamnesis section from a clinical document."""
        import re
        patterns = [
            r'(?i)ANAMNESI.*?\n(.*?)(?=\n\s*(?:ESAME|DIAGNOSI|TERAPIA|CONCLUSIONI|MOTIVO|\Z))',
            r'(?i)STORIA\s+CLINICA.*?\n(.*?)(?=\n\s*(?:ESAME|DIAGNOSI|TERAPIA|\Z))',
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.DOTALL)
            if match:
                return match.group(0)
        return text[:2000]  # Fallback: first 2000 chars
